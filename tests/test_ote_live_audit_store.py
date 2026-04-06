from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.models.loaders import load_live_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

LONG_MANIFEST_PATH = ROOT / "ote_live" / "runtime_manifests" / "long_ote_champion_v1" / "live_runtime_manifest.json"


def test_live_audit_repository_reconstructs_signal_from_database_rows() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_audit_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    manifest = load_live_runtime_manifest(LONG_MANIFEST_PATH)
    base_timestamp = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        persisted_manifest = audit.record_runtime_manifest(
            manifest,
            manifest_path=LONG_MANIFEST_PATH,
            policy_path=LONG_MANIFEST_PATH.with_name("live_policy.json"),
        )

        for offset in range(3):
            store.upsert_bar(
                MarketBar(
                    asset=manifest.asset,
                    timeframe=manifest.timeframe,
                    timestamp=base_timestamp + timedelta(minutes=5 * offset),
                    open=1.1000 + offset * 0.0002,
                    high=1.1004 + offset * 0.0002,
                    low=1.0997 + offset * 0.0002,
                    close=1.1002 + offset * 0.0002,
                    volume=50.0 + offset,
                    source="audit-test",
                )
            )

        snapshot = FeatureSnapshot(
            asset=manifest.asset,
            timeframe=manifest.timeframe,
            direction=manifest.direction,
            timestamp=base_timestamp + timedelta(minutes=10),
            source_row_idx=4321,
            feature_values={
                manifest.feature_manifest.selected_feature_names[0]: 0.25,
                manifest.feature_manifest.selected_feature_names[1]: -0.10,
                "quality__feed_is_stale": False,
            },
            valid_feature_count=3,
        )
        feature_snapshot_id = audit.record_feature_snapshot(
            snapshot,
            runtime_manifest_id=persisted_manifest.runtime_manifest_id,
            metadata={"family_counts": {"manual_test": 3}},
        )

        prediction = ModelPrediction(
            model_id=manifest.model_id,
            direction=manifest.direction,
            backend=manifest.backend,
            timestamp=snapshot.timestamp,
            source_row_idx=snapshot.source_row_idx,
            regime="trend_up_normal",
            raw_score=0.681,
            calibrated_probability=0.732,
            threshold_applied=0.64,
            threshold_source="regime",
        )
        prediction_id = audit.record_prediction(
            prediction,
            feature_snapshot_id=feature_snapshot_id,
        )

        signal = SignalDecision(
            model_id=manifest.model_id,
            direction=manifest.direction,
            timestamp=snapshot.timestamp,
            source_row_idx=snapshot.source_row_idx,
            decision="emit",
            probability=prediction.calibrated_probability,
            threshold=prediction.threshold_applied,
            regime=prediction.regime,
            reasons=["threshold_passed", "regime_threshold"],
            cooldown_bars_remaining=None,
        )
        signal_decision_id = audit.record_signal_decision(
            signal,
            prediction_id=prediction_id,
        )

        audit.record_notification(
            signal_decision_id=signal_decision_id,
            channel="email",
            status="queued",
            dedupe_key="signal-4321",
            payload={"subject": "OTE signal", "probability": signal.probability},
        )
        audit.record_media_artifact(
            signal_decision_id=signal_decision_id,
            artifact_type="chart_screenshot",
            file_path=tmp_root / "signal_4321.png",
            file_hash="abc123",
            metadata={"width": 1280, "height": 720},
        )
        health_event_id = audit.record_health_event(
            component="decision_engine",
            event_type="signal_emitted",
            severity="info",
            message="Signal persisted for audit replay.",
            payload={"signal_decision_id": signal_decision_id},
            event_timestamp_utc=snapshot.timestamp,
        )

        reconstructed = audit.reconstruct_signal(signal_decision_id, lookback_bars=2)
        fetched_snapshot = audit.get_feature_snapshot(feature_snapshot_id)
        fetched_prediction = audit.get_prediction(prediction_id)
        fetched_signal = audit.get_signal_decision(signal_decision_id)
        health_event = audit.get_health_event(health_event_id)

    assert reconstructed.runtime_manifest is not None
    assert reconstructed.runtime_manifest.runtime_manifest_id == persisted_manifest.runtime_manifest_id
    assert reconstructed.runtime_manifest.manifest.model_id == manifest.model_id
    assert reconstructed.feature_snapshot == snapshot
    assert reconstructed.prediction == prediction
    assert reconstructed.signal == signal
    assert len(reconstructed.bars) == 2
    assert reconstructed.bars[-1].timestamp == snapshot.timestamp
    assert reconstructed.notifications[0].channel == "email"
    assert reconstructed.notifications[0].payload["probability"] == signal.probability
    assert reconstructed.media_artifacts[0].artifact_type == "chart_screenshot"
    assert fetched_snapshot == snapshot
    assert fetched_prediction == prediction
    assert fetched_signal == signal
    assert health_event is not None
    assert health_event.component == "decision_engine"
    assert health_event.payload["signal_decision_id"] == signal_decision_id
