from __future__ import annotations

import asyncio
import base64
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.alerts import (
    LiveSignalEmailer,
    LiveSignalSmsSender,
    NotificationThrottle,
    NotificationThrottlePolicy,
)
from ote_live.dashboard import fetch_confidence_history, fetch_recent_signals
from ote_live.ingestion.aggregator import MultiTimeframeBarAggregator
from ote_live.ingestion.gap_detector import GapDetector
from ote_live.ingestion.heartbeat import HeartbeatMonitor
from ote_live.ingestion.runtime import LiveCollectorConfig, LiveCollectorRuntime
from ote_live.ingestion.service import LiveBarIngestionService
from ote_live.ingestion.signals import LiveSignalProcessor, SignalRuntimeModelBinding
from ote_live.media import SignalChartCaptureService
from ote_live.models.ensemble import LoadedDirectionModels
from ote_live.models.loaders import LoadedRuntimeModel, load_live_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
from ote_live.contracts.market_data import MarketBar

SHORT_XGB_V1_MANIFEST_PATH = (
    ROOT / "ote_live" / "runtime_manifests" / "short_ote_xgb_v1_candidate" / "live_runtime_manifest.json"
)
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+nX8cAAAAASUVORK5CYII="
)


class _FakeClient:
    def __init__(self, *, seed_bars: list[MarketBar]) -> None:
        self.seed_bars = seed_bars

    async def fetch_historical_chart(self, **kwargs) -> list[MarketBar]:
        return self.seed_bars

    async def fetch_time_series(self, **kwargs) -> list[MarketBar]:
        return await self.fetch_historical_chart(**kwargs)

    async def aclose(self) -> None:
        return None


class _FakeStream:
    def __init__(self, polled_bars: list[MarketBar]) -> None:
        self.polled_bars = polled_bars
        self.last_emitted_timestamp = None
        self._called = False

    async def poll(self) -> list[MarketBar]:
        if self._called:
            return []
        self._called = True
        return self.polled_bars


class _FakeBackfill:
    async def backfill_bars(self, window) -> list[MarketBar]:
        return []


class _RecordingEmailTransport:
    def __init__(self) -> None:
        self.messages = []

    def send_email(self, message) -> str:
        self.messages.append(message)
        return f"email-{len(self.messages)}"


class _DummyChartRenderer:
    def render_chart(self, audit_trail) -> bytes:
        return PNG_1X1


class _RecordingSmsTransport:
    def __init__(self) -> None:
        self.messages = []

    def send_sms(self, message) -> str:
        self.messages.append(message)
        return f"sms-{len(self.messages)}"


def test_runtime_emits_signal_and_automatically_persists_notification_and_screenshot() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_signal_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    chart_root = tmp_root / "charts"

    seed_bars = [_minute_bar(offset) for offset in range(1_250)]
    live_cycle_bars = [_minute_bar(offset) for offset in range(1_250, 1_255)]

    store = SQLiteLiveDataStore(db_path)
    audit = LiveAuditRepository(store)
    email_transport = _RecordingEmailTransport()
    sms_transport = _RecordingSmsTransport()
    signal_processor = _build_signal_processor(
        store=store,
        audit=audit,
        email_transport=email_transport,
        sms_transport=sms_transport,
        chart_root=chart_root,
    )
    runtime = LiveCollectorRuntime(
        config=LiveCollectorConfig(
            asset="EURUSD",
            source_timeframe="1m",
            aggregation_timeframes=("5m",),
            startup_history_bars=len(seed_bars),
            poll_interval_seconds=0.0,
            max_cycles=1,
        ),
        client=_FakeClient(seed_bars=seed_bars),  # type: ignore[arg-type]
        stream=_FakeStream(live_cycle_bars),  # type: ignore[arg-type]
        backfill_connector=_FakeBackfill(),  # type: ignore[arg-type]
        store=store,
        service=LiveBarIngestionService(
            stream=_FakeStream([]),  # placeholder, runtime uses its own stream above
            backfill=_FakeBackfill(),
            store=store,
            gap_detector=GapDetector(asset="EURUSD", timeframe="1m"),
            aggregator=MultiTimeframeBarAggregator(target_timeframes=("5m",)),
            heartbeat_monitor=HeartbeatMonitor(source="runtime-signal-test"),
            audit_repository=audit,
        ),
        audit_repository=audit,
        signal_processor=signal_processor,
    )
    runtime.service.stream = runtime.stream
    runtime.service.backfill = runtime.backfill_connector

    summary = asyncio.run(runtime.run())

    assert summary.emitted_signals == 1
    assert summary.sent_notifications == 2
    assert summary.captured_media_artifacts == 1
    assert len(email_transport.messages) == 1
    assert len(sms_transport.messages) == 1

    with SQLiteLiveDataStore(db_path) as reopened_store:
        reopened_audit = LiveAuditRepository(reopened_store)
        signals = fetch_recent_signals(
            reopened_audit,
            decisions=("emit",),
            limit=10,
        )
        assert len(signals) == 1
        signal_decision_id = int(signals.iloc[0]["signal_decision_id"])
        reconstructed = reopened_audit.reconstruct_signal(signal_decision_id, include_bars=False)

    assert len(reconstructed.notifications) == 2
    assert reconstructed.notifications[0].status == "sent"
    assert reconstructed.notifications[1].status == "sent"
    assert len(reconstructed.media_artifacts) == 1
    assert reconstructed.media_artifacts[0].artifact_type == "chart_screenshot"
    assert Path(reconstructed.media_artifacts[0].file_path).exists()


def test_runtime_shadow_signal_still_persists_notification_and_screenshot() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_signal_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    chart_root = tmp_root / "charts"

    seed_bars = [_minute_bar(offset) for offset in range(1_250)]
    live_cycle_bars = [_minute_bar(offset) for offset in range(1_250, 1_255)]

    store = SQLiteLiveDataStore(db_path)
    audit = LiveAuditRepository(store)
    email_transport = _RecordingEmailTransport()
    sms_transport = _RecordingSmsTransport()
    signal_processor = _build_signal_processor(
        store=store,
        audit=audit,
        email_transport=email_transport,
        sms_transport=sms_transport,
        chart_root=chart_root,
        shadow_mode=True,
    )
    runtime = LiveCollectorRuntime(
        config=LiveCollectorConfig(
            asset="EURUSD",
            source_timeframe="1m",
            aggregation_timeframes=("5m",),
            startup_history_bars=len(seed_bars),
            poll_interval_seconds=0.0,
            max_cycles=1,
        ),
        client=_FakeClient(seed_bars=seed_bars),  # type: ignore[arg-type]
        stream=_FakeStream(live_cycle_bars),  # type: ignore[arg-type]
        backfill_connector=_FakeBackfill(),  # type: ignore[arg-type]
        store=store,
        service=LiveBarIngestionService(
            stream=_FakeStream([]),
            backfill=_FakeBackfill(),
            store=store,
            gap_detector=GapDetector(asset="EURUSD", timeframe="1m"),
            aggregator=MultiTimeframeBarAggregator(target_timeframes=("5m",)),
            heartbeat_monitor=HeartbeatMonitor(source="runtime-signal-test"),
            audit_repository=audit,
        ),
        audit_repository=audit,
        signal_processor=signal_processor,
    )
    runtime.service.stream = runtime.stream
    runtime.service.backfill = runtime.backfill_connector

    summary = asyncio.run(runtime.run())

    assert summary.evaluated_signals == 1
    assert summary.emitted_signals == 0
    assert summary.sent_notifications == 2
    assert summary.captured_media_artifacts == 1
    assert len(email_transport.messages) == 1
    assert len(sms_transport.messages) == 1

    with SQLiteLiveDataStore(db_path) as reopened_store:
        reopened_audit = LiveAuditRepository(reopened_store)
        signals = fetch_recent_signals(
            reopened_audit,
            decisions=("shadow",),
            limit=10,
        )
        assert len(signals) == 1
        signal_decision_id = int(signals.iloc[0]["signal_decision_id"])
        reconstructed = reopened_audit.reconstruct_signal(signal_decision_id, include_bars=False)

    assert reconstructed.signal.decision == "shadow"
    assert len(reconstructed.notifications) == 2
    assert reconstructed.notifications[0].status == "sent"
    assert reconstructed.notifications[1].status == "sent"
    assert len(reconstructed.media_artifacts) == 1
    assert reconstructed.media_artifacts[0].artifact_type == "chart_screenshot"
    assert Path(reconstructed.media_artifacts[0].file_path).exists()


def test_live_signal_processor_requires_the_configured_primary_model(monkeypatch) -> None:
    configured_primary_id = "long_ote_tcn_v2_candidate"
    fallback_model_id = "long_ote_tcn_v1_candidate"
    bundle = LoadedDirectionModels(
        direction_manifest=SimpleNamespace(
            recommendations=SimpleNamespace(recommended_primary_model_id=configured_primary_id),
            models=(
                SimpleNamespace(model_id=configured_primary_id),
                SimpleNamespace(model_id=fallback_model_id),
            ),
        ),
        loaded_models={
            fallback_model_id: SimpleNamespace(model_id=fallback_model_id),
        },
        unavailable_models={
            configured_primary_id: "configured primary backend is unavailable",
        },
    )
    monkeypatch.setattr("ote_live.ingestion.signals.load_direction_models", lambda *args, **kwargs: bundle)

    with pytest.raises(RuntimeError, match=configured_primary_id):
        LiveSignalProcessor.from_direction_manifest_paths(
            audit_repository=SimpleNamespace(),
            long_manifest_path="ote_live/runtime_manifests/live_runtime_manifest_long.json",
            short_manifest_path=None,
        )


def test_live_signal_processor_can_activate_all_loaded_models(monkeypatch) -> None:
    configured_primary_id = "long_breakout_tcn_champion"
    shadow_model_id = "long_reversal_tcn_champion"
    deprecated_model_id = "long_legacy_model"
    bundle = LoadedDirectionModels(
        direction_manifest=SimpleNamespace(
            recommendations=SimpleNamespace(recommended_primary_model_id=configured_primary_id),
            models=(
                SimpleNamespace(model_id=configured_primary_id, status="candidate"),
                SimpleNamespace(model_id=shadow_model_id, status="active"),
                SimpleNamespace(model_id=deprecated_model_id, status="deprecated"),
            ),
        ),
        loaded_models={
            configured_primary_id: SimpleNamespace(model_id=configured_primary_id),
            shadow_model_id: SimpleNamespace(model_id=shadow_model_id),
            deprecated_model_id: SimpleNamespace(model_id=deprecated_model_id),
        },
        unavailable_models={},
    )
    monkeypatch.setattr("ote_live.ingestion.signals.load_direction_models", lambda *args, **kwargs: bundle)

    processor = _InspectableSignalProcessor.from_direction_manifest_paths(
        audit_repository=SimpleNamespace(),
        long_manifest_path="ote_live/runtime_manifests/live_runtime_manifest_long.json",
        short_manifest_path=None,
        all_models_active=True,
    )

    assert processor is not None
    assert [binding.loaded_model.model_id for binding in processor.bindings] == [
        configured_primary_id,
        shadow_model_id,
    ]
    assert all(binding.shadow_mode is False for binding in processor.bindings)


def test_live_signal_processor_all_models_active_tolerates_missing_primary(monkeypatch) -> None:
    configured_primary_id = "long_breakout_tcn_champion"
    fallback_model_id = "long_reversal_tcn_champion"
    bundle = LoadedDirectionModels(
        direction_manifest=SimpleNamespace(
            recommendations=SimpleNamespace(recommended_primary_model_id=configured_primary_id),
            models=(
                SimpleNamespace(model_id=configured_primary_id, status="active"),
                SimpleNamespace(model_id=fallback_model_id, status="candidate"),
            ),
        ),
        loaded_models={
            fallback_model_id: SimpleNamespace(model_id=fallback_model_id),
        },
        unavailable_models={
            configured_primary_id: "configured primary backend is unavailable",
        },
    )
    monkeypatch.setattr("ote_live.ingestion.signals.load_direction_models", lambda *args, **kwargs: bundle)

    processor = _InspectableSignalProcessor.from_direction_manifest_paths(
        audit_repository=SimpleNamespace(),
        long_manifest_path="ote_live/runtime_manifests/live_runtime_manifest_long.json",
        short_manifest_path=None,
        all_models_active=True,
    )

    assert processor is not None
    assert [binding.loaded_model.model_id for binding in processor.bindings] == [fallback_model_id]
    assert processor.bindings[0].shadow_mode is False


def test_live_signal_processor_uses_manifest_status_for_shadow_mode(monkeypatch) -> None:
    configured_primary_id = "long_ote_tcn_v2_candidate"
    shadow_model_id = "long_breakout_tcn_champion"
    deprecated_model_id = "long_old_model"
    bundle = LoadedDirectionModels(
        direction_manifest=SimpleNamespace(
            recommendations=SimpleNamespace(recommended_primary_model_id=configured_primary_id),
            models=(
                SimpleNamespace(model_id=configured_primary_id, status="active"),
                SimpleNamespace(model_id=shadow_model_id, status="candidate"),
                SimpleNamespace(model_id=deprecated_model_id, status="deprecated"),
            ),
        ),
        loaded_models={
            configured_primary_id: SimpleNamespace(model_id=configured_primary_id),
            shadow_model_id: SimpleNamespace(model_id=shadow_model_id),
            deprecated_model_id: SimpleNamespace(model_id=deprecated_model_id),
        },
        unavailable_models={},
    )
    monkeypatch.setattr("ote_live.ingestion.signals.load_direction_models", lambda *args, **kwargs: bundle)

    processor = _InspectableSignalProcessor.from_direction_manifest_paths(
        audit_repository=SimpleNamespace(),
        long_manifest_path="ote_live/runtime_manifests/live_runtime_manifest_long.json",
        short_manifest_path=None,
    )

    assert processor is not None
    assert [binding.loaded_model.model_id for binding in processor.bindings] == [
        configured_primary_id,
        shadow_model_id,
    ]
    assert [binding.shadow_mode for binding in processor.bindings] == [False, True]


def test_runtime_persists_primary_hold_decisions_for_dashboard_confidence() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_signal_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    chart_root = tmp_root / "charts"

    seed_bars = [_minute_bar(offset) for offset in range(1_250)]
    live_cycle_bars = [_minute_bar(offset) for offset in range(1_250, 1_255)]

    store = SQLiteLiveDataStore(db_path)
    audit = LiveAuditRepository(store)
    signal_processor = _build_signal_processor(
        store=store,
        audit=audit,
        email_transport=_RecordingEmailTransport(),
        sms_transport=_RecordingSmsTransport(),
        chart_root=chart_root,
        global_threshold=0.99,
    )
    runtime = LiveCollectorRuntime(
        config=LiveCollectorConfig(
            asset="EURUSD",
            source_timeframe="1m",
            aggregation_timeframes=("5m",),
            startup_history_bars=len(seed_bars),
            poll_interval_seconds=0.0,
            max_cycles=1,
        ),
        client=_FakeClient(seed_bars=seed_bars),  # type: ignore[arg-type]
        stream=_FakeStream(live_cycle_bars),  # type: ignore[arg-type]
        backfill_connector=_FakeBackfill(),  # type: ignore[arg-type]
        store=store,
        service=LiveBarIngestionService(
            stream=_FakeStream([]),
            backfill=_FakeBackfill(),
            store=store,
            gap_detector=GapDetector(asset="EURUSD", timeframe="1m"),
            aggregator=MultiTimeframeBarAggregator(target_timeframes=("5m",)),
            heartbeat_monitor=HeartbeatMonitor(source="runtime-signal-test"),
            audit_repository=audit,
        ),
        audit_repository=audit,
        signal_processor=signal_processor,
    )
    runtime.service.stream = runtime.stream
    runtime.service.backfill = runtime.backfill_connector

    summary = asyncio.run(runtime.run())

    assert summary.evaluated_signals == 1
    assert summary.emitted_signals == 0

    with SQLiteLiveDataStore(db_path) as reopened_store:
        reopened_audit = LiveAuditRepository(reopened_store)
        confidence = fetch_confidence_history(
            reopened_audit,
            model_id="short_ote_xgb_v1_candidate",
            direction="short",
            limit=10,
        )
        signals = fetch_recent_signals(
            reopened_audit,
            model_ids=("short_ote_xgb_v1_candidate",),
            decisions=("hold",),
            limit=10,
        )

    assert len(confidence) == 1
    assert confidence.iloc[0]["decision"] == "hold"
    assert len(signals) == 1
    assert signals.iloc[0]["decision"] == "hold"


def test_build_policy_frame_batches_feature_columns_without_changing_overwrite_behavior() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_signal_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    chart_root = tmp_root / "charts"

    store = SQLiteLiveDataStore(db_path)
    audit = LiveAuditRepository(store)
    signal_processor = _build_signal_processor(
        store=store,
        audit=audit,
        email_transport=_RecordingEmailTransport(),
        sms_transport=_RecordingSmsTransport(),
        chart_root=chart_root,
    )
    signal_processor.feature_engine.state.extend([_minute_bar(offset) for offset in range(3)])

    feature_frame = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "custom_feature": [0.25, 0.50],
        }
    )

    policy_frame = signal_processor._build_policy_frame(feature_frame)

    assert list(policy_frame.columns) == [
        "asset",
        "timeframe",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid",
        "ask",
        "spread",
        "source",
        "custom_feature",
    ]
    assert policy_frame["open"].tolist() == [10.0, 11.0]
    assert policy_frame["custom_feature"].tolist() == [0.25, 0.50]
    assert policy_frame["timestamp"].tolist() == [_minute_bar(1).timestamp, _minute_bar(2).timestamp]


def _build_signal_processor(
    *,
    store: SQLiteLiveDataStore,
    audit: LiveAuditRepository,
    email_transport: _RecordingEmailTransport,
    sms_transport: _RecordingSmsTransport,
    chart_root: Path,
    global_threshold: float = 0.0,
    shadow_mode: bool = False,
) -> LiveSignalProcessor:
    base_manifest = load_live_runtime_manifest(SHORT_XGB_V1_MANIFEST_PATH)
    selected_feature_names = [
        "rsi_14",
        "dist_to_prior_high_20_atr",
        "atr_ratio_14_50_roll_mean_100",
    ]
    manifest = _subset_manifest(base_manifest, selected_feature_names)
    manifest = manifest.model_copy(
        update={
            "live_policy": manifest.live_policy.model_copy(
                update={
                    "thresholds": manifest.live_policy.thresholds.model_copy(
                        update={
                            "global_threshold": float(global_threshold),
                            "regime_thresholds": {},
                        }
                    ),
                    "abstain_policy": manifest.live_policy.abstain_policy.model_copy(
                        update={"enabled": False}
                    ),
                }
            )
        }
    )
    loaded_model = LoadedRuntimeModel(
        manifest=manifest,
        model=_ConstantBooster(probability=0.85),
        scaler=None,
        calibrator=None,
        training_summary={},
        scale_clip=0.0,
        batch_size=1,
        use_amp=False,
    )
    emailer = LiveSignalEmailer(
        audit_repository=audit,
        throttle=NotificationThrottle(store),
        transport=email_transport,
        recipients=("ops@example.com",),
        policy=NotificationThrottlePolicy(cooldown_seconds=0.0),
    )
    sms_sender = LiveSignalSmsSender(
        audit_repository=audit,
        throttle=NotificationThrottle(store),
        transport=sms_transport,
        recipients=("+15550001111",),
        policy=NotificationThrottlePolicy(cooldown_seconds=0.0),
    )
    chart_capture = SignalChartCaptureService(
        audit_repository=audit,
        output_root=chart_root,
        renderer=_DummyChartRenderer(),
    )
    return LiveSignalProcessor(
        bindings=[SignalRuntimeModelBinding(loaded_model=loaded_model, shadow_mode=bool(shadow_mode))],
        audit_repository=audit,
        emailer=emailer,
        sms_sender=sms_sender,
        chart_capture_service=chart_capture,
        dashboard_url="http://localhost:8050",
        rolling_window_bars=260,
    )


def _subset_manifest(base_manifest, selected_feature_names: list[str]):
    feature_manifest = base_manifest.feature_manifest.model_copy(
        update={
            "selected_feature_names": list(selected_feature_names),
            "selected_feature_count": len(selected_feature_names),
            "strategy_feature_count": 0,
        }
    )
    return base_manifest.model_copy(update={"feature_manifest": feature_manifest})


class _ConstantBooster:
    def __init__(self, *, probability: float) -> None:
        self.probability = float(probability)

    def predict(self, _dmatrix) -> np.ndarray:
        clipped = min(max(self.probability, 1e-6), 1.0 - 1e-6)
        logit = np.log(clipped / (1.0 - clipped))
        return np.asarray([logit], dtype=np.float32)


class _InspectableSignalProcessor(LiveSignalProcessor):
    def __init__(self, *, bindings, audit_repository, **kwargs) -> None:
        self.bindings = tuple(bindings)
        self.audit_repository = audit_repository


def _minute_bar(offset: int) -> MarketBar:
    timestamp = datetime(2024, 1, 2, 0, 0, tzinfo=UTC) + timedelta(minutes=offset)
    price = 1.1000 + (offset * 0.00001)
    return MarketBar(
        asset="EURUSD",
        timeframe="1m",
        timestamp=timestamp,
        open=price,
        high=price + 0.0002,
        low=price - 0.0002,
        close=price + 0.00005,
        volume=100.0 + offset,
        source="runtime-signal-test",
    )
