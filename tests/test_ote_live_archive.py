from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("pyarrow")

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.models.loaders import load_live_runtime_manifest
from ote_live.storage import (
    LiveAuditRepository,
    SQLiteLiveDataStore,
    archive_calendar_month,
    build_retention_plan,
    purge_retained_data,
    restore_calendar_month,
)
from ote_live.storage.retention import RetentionPolicy

SHORT_MANIFEST_PATH = ROOT / "ote_live" / "runtime_manifests" / "short_ote_xgb_v1_candidate" / "live_runtime_manifest.json"


def test_monthly_archive_restore_preserves_signal_lineage() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_archive_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)

    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    restored_db_path = tmp_root / f"{uuid.uuid4().hex}.restored.sqlite"
    archive_root = tmp_root / f"archive_{uuid.uuid4().hex}"

    manifest = load_live_runtime_manifest(SHORT_MANIFEST_PATH)
    signal_timestamp = datetime(2026, 4, 1, 14, 0, tzinfo=UTC)

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        persisted_manifest = audit.record_runtime_manifest(manifest, manifest_path=SHORT_MANIFEST_PATH)
        store.upsert_bar(
            MarketBar(
                asset=manifest.asset,
                timeframe=manifest.timeframe,
                timestamp=signal_timestamp,
                open=1.0820,
                high=1.0826,
                low=1.0817,
                close=1.0822,
                volume=21.0,
                source="archive-test",
            )
        )
        feature_snapshot_id = audit.record_feature_snapshot(
            FeatureSnapshot(
                asset=manifest.asset,
                timeframe=manifest.timeframe,
                direction=manifest.direction,
                timestamp=signal_timestamp,
                source_row_idx=987,
                feature_values={
                    manifest.feature_manifest.selected_feature_names[0]: 0.42,
                    manifest.feature_manifest.selected_feature_names[1]: -0.05,
                },
                valid_feature_count=2,
            ),
            runtime_manifest_id=persisted_manifest.runtime_manifest_id,
        )
        prediction_id = audit.record_prediction(
            ModelPrediction(
                model_id=manifest.model_id,
                direction=manifest.direction,
                backend=manifest.backend,
                timestamp=signal_timestamp,
                source_row_idx=987,
                regime="range_low",
                raw_score=0.61,
                calibrated_probability=0.66,
                threshold_applied=0.63,
                threshold_source="global",
            ),
            feature_snapshot_id=feature_snapshot_id,
        )
        signal_decision_id = audit.record_signal_decision(
            SignalDecision(
                model_id=manifest.model_id,
                direction=manifest.direction,
                timestamp=signal_timestamp,
                source_row_idx=987,
                decision="emit",
                probability=0.66,
                threshold=0.63,
                regime="range_low",
                reasons=["threshold_passed"],
                cooldown_bars_remaining=None,
            ),
            prediction_id=prediction_id,
        )

        archive_result = archive_calendar_month(
            store,
            archive_root,
            year=2026,
            month=4,
            table_names=("canonical_bars", "runtime_manifests", "feature_snapshots", "model_predictions", "signal_decisions"),
        )
        retention_plan = build_retention_plan(
            store,
            policies=[
                RetentionPolicy("signal_decisions", max_age_days=1, require_archive=True),
                RetentionPolicy("model_predictions", max_age_days=1, require_archive=True),
            ],
            reference_time=datetime(2026, 4, 5, tzinfo=UTC),
            archive_root=archive_root,
        )
        purge_result = purge_retained_data(store, plan=retention_plan, dry_run=True)

    assert archive_result.total_rows == 5
    assert retention_plan.total_eligible_rows == 2
    assert all(candidate.archive_confirmed for candidate in retention_plan.candidates)
    assert purge_result.deleted_rows_by_table["signal_decisions"] == 1
    assert purge_result.deleted_rows_by_table["model_predictions"] == 1

    with SQLiteLiveDataStore(restored_db_path) as restored_store:
        restore_result = restore_calendar_month(
            restored_store,
            archive_root,
            year=2026,
            month=4,
            table_names=("canonical_bars", "runtime_manifests", "feature_snapshots", "model_predictions", "signal_decisions"),
        )
        restored_audit = LiveAuditRepository(restored_store)
        reconstructed = restored_audit.reconstruct_signal(signal_decision_id)

    assert restore_result.total_rows == 5
    assert reconstructed.signal_decision_id == signal_decision_id
    assert reconstructed.runtime_manifest is not None
    assert reconstructed.runtime_manifest.manifest.model_id == manifest.model_id
    assert reconstructed.feature_snapshot.source_row_idx == 987
    assert reconstructed.signal.decision == "emit"
