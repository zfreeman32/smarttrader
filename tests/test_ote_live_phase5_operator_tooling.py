from __future__ import annotations

import base64
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ote_live.alerts import (
    LiveSignalEmailer,
    LiveSignalSmsSender,
    NotificationThrottle,
    NotificationThrottlePolicy,
)
from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.dashboard import (
    build_health_summary,
    compute_signal_markouts,
    fetch_confidence_history,
    fetch_recent_bars,
    fetch_recent_signals,
    summarize_signal_markouts,
)
from ote_live.ingestion.base import HeartbeatStatus, IngestionGap
from ote_live.media import SignalChartCaptureService
from ote_live.models.loaders import load_live_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

ROOT = Path(__file__).resolve().parents[1]
LONG_MANIFEST_PATH = ROOT / "ote_live" / "runtime_manifests" / "long_ote_champion_v1" / "live_runtime_manifest.json"
SHORT_MANIFEST_PATH = ROOT / "ote_live" / "runtime_manifests" / "short_ote_candidate_tcn_v2" / "live_runtime_manifest.json"
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+nX8cAAAAASUVORK5CYII="
)


class DummyEmailTransport:
    def __init__(self) -> None:
        self.messages = []

    def send_email(self, message) -> str:
        self.messages.append(message)
        return f"email-{len(self.messages)}"


class DummyChartRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def render_chart(self, audit_trail) -> bytes:
        self.calls += 1
        return PNG_1X1


class DummyAnnotator:
    def __init__(self) -> None:
        self.calls = 0

    def annotate(self, image_bytes: bytes, *, audit_trail) -> bytes:
        self.calls += 1
        return image_bytes


class DummySmsTransport:
    def __init__(self) -> None:
        self.messages = []

    def send_sms(self, message) -> str:
        self.messages.append(message)
        return f"sms-{len(self.messages)}"


def test_dashboard_queries_surface_live_state_and_markouts() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_phase5_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        manifest, signal_decision_id = _seed_signal_audit(store, audit)

        bars = fetch_recent_bars(
            store,
            asset=manifest.asset,
            timeframe=manifest.timeframe,
            limit=20,
        )
        signals = fetch_recent_signals(audit, limit=10)
        confidence = fetch_confidence_history(audit, limit=10)
        markouts = compute_signal_markouts(
            store,
            audit,
            horizon_bars=2,
            limit=10,
        )
        performance = summarize_signal_markouts(markouts)
        health = build_health_summary(
            store,
            audit,
            asset=manifest.asset,
            timeframe=manifest.timeframe,
        )

    assert len(bars) == 12
    assert len(signals) == 1
    assert int(signals.iloc[0]["signal_decision_id"]) == signal_decision_id
    assert int(confidence.iloc[0]["signal_decision_id"]) == signal_decision_id
    assert len(markouts) == 1
    assert markouts.iloc[0]["status"] == "complete"
    assert health.latest_heartbeat_source == "collector-test"
    assert health.unresolved_gap_count == 1
    assert len(health.recent_health_events) == 1
    assert performance.completed_count == 1
    assert performance.signal_count == 1


def test_fetch_recent_signals_can_filter_by_model_id() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_phase5_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        long_manifest, long_signal_decision_id = _seed_signal_audit(store, audit)
        short_manifest = load_live_runtime_manifest(SHORT_MANIFEST_PATH)
        _seed_additional_signal_audit(
            store,
            audit,
            manifest=short_manifest,
            timestamp=datetime(2026, 4, 1, 13, 0, tzinfo=UTC),
            source_row_idx=7001,
            probability=0.79,
            threshold=0.75,
        )

        all_signals = fetch_recent_signals(audit, limit=10)
        long_signals = fetch_recent_signals(audit, model_ids=(long_manifest.model_id,), limit=10)

    assert len(all_signals) == 2
    assert len(long_signals) == 1
    assert int(long_signals.iloc[0]["signal_decision_id"]) == long_signal_decision_id
    assert str(long_signals.iloc[0]["model_id"]) == long_manifest.model_id


def test_notification_throttle_blocks_active_cooldowns() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_phase5_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    base_time = datetime(2026, 4, 2, 12, 0, tzinfo=UTC)

    with SQLiteLiveDataStore(db_path) as store:
        throttle = NotificationThrottle(store)
        policy = NotificationThrottlePolicy(cooldown_seconds=300.0)

        first = throttle.evaluate(
            channel="email",
            dedupe_key="email:first",
            cooldown_scope="long_ote_champion_v1:long:emit",
            policy=policy,
            now=base_time,
        )
        throttle.remember_sent(
            channel="email",
            dedupe_key="email:first",
            cooldown_scope="long_ote_champion_v1:long:emit",
            signal_decision_id=101,
            policy=policy,
            now=base_time,
        )
        second = throttle.evaluate(
            channel="email",
            dedupe_key="email:second",
            cooldown_scope="long_ote_champion_v1:long:emit",
            policy=policy,
            now=base_time + timedelta(seconds=90),
        )

    assert first.allowed is True
    assert second.allowed is False
    assert second.reason == "cooldown_active"
    assert second.cooldown_remaining_seconds is not None
    assert second.cooldown_remaining_seconds > 0


def test_email_and_chart_capture_persist_phase5_operator_artifacts() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_phase5_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        _, signal_decision_id = _seed_signal_audit(store, audit)

        email_transport = DummyEmailTransport()
        emailer = LiveSignalEmailer(
            audit_repository=audit,
            throttle=NotificationThrottle(store),
            transport=email_transport,
            recipients=("ops@example.com",),
            policy=NotificationThrottlePolicy(cooldown_seconds=300.0),
        )
        first_email = emailer.send_signal_alert(
            signal_decision_id=signal_decision_id,
            dashboard_url="http://localhost:8050",
        )
        duplicate_email = emailer.send_signal_alert(
            signal_decision_id=signal_decision_id,
            dashboard_url="http://localhost:8050",
        )

        renderer = DummyChartRenderer()
        annotator = DummyAnnotator()
        capture_service = SignalChartCaptureService(
            audit_repository=audit,
            output_root=tmp_root / "media",
            renderer=renderer,
            annotator=annotator,
            lookback_bars=8,
        )
        artifact = capture_service.capture_signal_chart(
            signal_decision_id=signal_decision_id,
            metadata={"source": "phase5-test"},
        )
        reconstructed = audit.reconstruct_signal(signal_decision_id, include_bars=False)
        signals = fetch_recent_signals(audit, limit=10)

    assert first_email.status == "sent"
    assert duplicate_email.status == "skipped"
    assert len(email_transport.messages) == 1
    assert renderer.calls == 1
    assert annotator.calls == 1
    assert artifact.file_path.exists()
    assert artifact.file_hash
    assert artifact.metadata["source"] == "phase5-test"
    assert len(reconstructed.notifications) == 2
    assert reconstructed.notifications[0].status == "sent"
    assert reconstructed.notifications[1].status == "skipped"
    assert reconstructed.notifications[0].payload["dashboard_url"].endswith(f"signal_decision_id={signal_decision_id}")
    assert len(reconstructed.media_artifacts) == 1
    assert reconstructed.media_artifacts[0].artifact_type == "chart_screenshot"
    assert int(signals.iloc[0]["notification_count"]) == 2
    assert int(signals.iloc[0]["media_artifact_count"]) == 1


def test_sms_sender_persists_sms_alerts_with_dedupe() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_phase5_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        _, signal_decision_id = _seed_signal_audit(store, audit)

        sms_transport = DummySmsTransport()
        sms_sender = LiveSignalSmsSender(
            audit_repository=audit,
            throttle=NotificationThrottle(store),
            transport=sms_transport,
            recipients=("+15550001111",),
            policy=NotificationThrottlePolicy(cooldown_seconds=300.0),
        )
        first_sms = sms_sender.send_signal_alert(signal_decision_id=signal_decision_id)
        duplicate_sms = sms_sender.send_signal_alert(signal_decision_id=signal_decision_id)
        reconstructed = audit.reconstruct_signal(signal_decision_id, include_bars=False)

    assert first_sms.status == "sent"
    assert duplicate_sms.status == "skipped"
    assert len(sms_transport.messages) == 1
    sms_notifications = [item for item in reconstructed.notifications if item.channel == "sms"]
    assert len(sms_notifications) == 2
    assert sms_notifications[0].status == "sent"
    assert sms_notifications[1].status == "skipped"


def _seed_signal_audit(
    store: SQLiteLiveDataStore,
    audit: LiveAuditRepository,
):
    manifest = load_live_runtime_manifest(LONG_MANIFEST_PATH)
    persisted_manifest = audit.record_runtime_manifest(
        manifest,
        manifest_path=LONG_MANIFEST_PATH,
        policy_path=LONG_MANIFEST_PATH.with_name("live_policy.json"),
    )
    base_timestamp = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    for offset in range(12):
        timestamp = base_timestamp + timedelta(minutes=5 * offset)
        store.upsert_bar(
            MarketBar(
                asset=manifest.asset,
                timeframe=manifest.timeframe,
                timestamp=timestamp,
                open=1.1000 + offset * 0.0002,
                high=1.1004 + offset * 0.0002,
                low=1.0997 + offset * 0.0002,
                close=1.1002 + offset * 0.0002,
                volume=50.0 + offset,
                source="phase5-test",
            )
        )

    signal_timestamp = base_timestamp + timedelta(minutes=25)
    snapshot = FeatureSnapshot(
        asset=manifest.asset,
        timeframe=manifest.timeframe,
        direction=manifest.direction,
        timestamp=signal_timestamp,
        source_row_idx=6001,
        feature_values={
            manifest.feature_manifest.selected_feature_names[0]: 0.12,
            manifest.feature_manifest.selected_feature_names[1]: -0.08,
            "quality__feed_is_stale": False,
        },
        valid_feature_count=3,
    )
    feature_snapshot_id = audit.record_feature_snapshot(
        snapshot,
        runtime_manifest_id=persisted_manifest.runtime_manifest_id,
        metadata={"family_counts": {"phase5_test": 3}},
    )
    prediction = ModelPrediction(
        model_id=manifest.model_id,
        direction=manifest.direction,
        backend=manifest.backend,
        timestamp=signal_timestamp,
        source_row_idx=snapshot.source_row_idx,
        regime="strong_up_medium",
        raw_score=0.71,
        calibrated_probability=0.82,
        threshold_applied=0.68,
        threshold_source="regime",
    )
    prediction_id = audit.record_prediction(
        prediction,
        feature_snapshot_id=feature_snapshot_id,
    )
    signal_decision_id = audit.record_signal_decision(
        SignalDecision(
            model_id=manifest.model_id,
            direction=manifest.direction,
            timestamp=signal_timestamp,
            source_row_idx=snapshot.source_row_idx,
            decision="emit",
            probability=prediction.calibrated_probability,
            threshold=prediction.threshold_applied,
            regime=prediction.regime,
            reasons=["threshold_passed", "regime_threshold"],
            cooldown_bars_remaining=None,
        ),
        prediction_id=prediction_id,
    )
    store.record_heartbeat(
        HeartbeatStatus(
            source="collector-test",
            observed_at_utc=signal_timestamp + timedelta(seconds=10),
            stale_after_seconds=90.0,
            lag_seconds=10.0,
            is_stale=False,
        ),
        metadata={"service": "phase5-test"},
    )
    store.record_gap(
        IngestionGap(
            asset=manifest.asset,
            timeframe="5m",
            expected_timestamp=base_timestamp + timedelta(minutes=10),
            observed_timestamp=base_timestamp + timedelta(minutes=20),
            missing_timestamps=[base_timestamp + timedelta(minutes=15)],
            gap_size=1,
        )
    )
    audit.record_health_event(
        component="dashboard",
        event_type="phase5_seed",
        severity="info",
        message="Seeded Phase 5 operator tooling test data.",
        payload={"signal_decision_id": signal_decision_id},
        event_timestamp_utc=signal_timestamp,
    )
    return manifest, signal_decision_id


def _seed_additional_signal_audit(
    store: SQLiteLiveDataStore,
    audit: LiveAuditRepository,
    *,
    manifest,
    timestamp: datetime,
    source_row_idx: int,
    probability: float,
    threshold: float,
) -> int:
    persisted_manifest = audit.record_runtime_manifest(
        manifest,
        manifest_path=SHORT_MANIFEST_PATH,
        policy_path=SHORT_MANIFEST_PATH.with_name("live_policy.json"),
    )
    snapshot = FeatureSnapshot(
        asset=manifest.asset,
        timeframe=manifest.timeframe,
        direction=manifest.direction,
        timestamp=timestamp,
        source_row_idx=source_row_idx,
        feature_values={
            manifest.feature_manifest.selected_feature_names[0]: 0.21,
            manifest.feature_manifest.selected_feature_names[1]: -0.03,
        },
        valid_feature_count=2,
    )
    feature_snapshot_id = audit.record_feature_snapshot(
        snapshot,
        runtime_manifest_id=persisted_manifest.runtime_manifest_id,
    )
    prediction = ModelPrediction(
        model_id=manifest.model_id,
        direction=manifest.direction,
        backend=manifest.backend,
        timestamp=timestamp,
        source_row_idx=source_row_idx,
        regime="strong_down_medium",
        raw_score=0.65,
        calibrated_probability=probability,
        threshold_applied=threshold,
        threshold_source="global",
    )
    prediction_id = audit.record_prediction(
        prediction,
        feature_snapshot_id=feature_snapshot_id,
    )
    return audit.record_signal_decision(
        SignalDecision(
            model_id=manifest.model_id,
            direction=manifest.direction,
            timestamp=timestamp,
            source_row_idx=source_row_idx,
            decision="emit",
            probability=probability,
            threshold=threshold,
            regime=prediction.regime,
            reasons=["threshold_passed"],
            cooldown_bars_remaining=None,
        ),
        prediction_id=prediction_id,
    )
