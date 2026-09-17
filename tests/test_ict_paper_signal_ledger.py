from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.models.loaders import load_live_runtime_manifest
from ote_live.policies.decision_engine import PersistedAuditRecord
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
from ote_live.storage.ict_paper_signal import IctPaperSignalLedgerRepository


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT
    / "ote_live"
    / "runtime_manifests"
    / "ict_es_paper_signal_20260813"
    / "ict_long_meta_xgb_v1"
    / "live_runtime_manifest.json"
)


def test_ict_paper_signal_event_is_idempotent_and_settles_on_exact_twentieth_bar() -> None:
    db_path = ROOT / "tmp" / "ict_paper_signal_ledger_tests" / f"{uuid.uuid4().hex}.sqlite"
    manifest = load_live_runtime_manifest(MANIFEST_PATH)
    entry_time = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        persisted = audit.record_runtime_manifest(manifest, manifest_path=MANIFEST_PATH)
        entry_bar = _bar(entry_time, 5000.0)
        store.upsert_bar(entry_bar)
        snapshot = FeatureSnapshot(
            asset="ES",
            timeframe="5m",
            direction="long",
            timestamp=entry_time,
            source_row_idx=100,
            feature_values={manifest.feature_manifest.selected_feature_names[0]: 1.0},
            valid_feature_count=1,
        )
        snapshot_id = audit.record_feature_snapshot(
            snapshot, runtime_manifest_id=persisted.runtime_manifest_id
        )
        prediction = ModelPrediction(
            model_id=manifest.model_id,
            direction="long",
            backend="xgboost",
            timestamp=entry_time,
            source_row_idx=100,
            regime="trend_up_normal",
            raw_score=0.8,
            calibrated_probability=0.75,
            threshold_applied=0.4,
            threshold_source="global",
        )
        prediction_id = audit.record_prediction(prediction, feature_snapshot_id=snapshot_id)
        signal = SignalDecision(
            model_id=manifest.model_id,
            direction="long",
            timestamp=entry_time,
            source_row_idx=100,
            decision="emit",
            probability=0.75,
            threshold=0.4,
            regime="trend_up_normal",
            reasons=["threshold_passed"],
        )
        signal_id = audit.record_signal_decision(signal, prediction_id=prediction_id)
        audit_record = PersistedAuditRecord(
            runtime_manifest_id=persisted.runtime_manifest_id,
            feature_snapshot_id=snapshot_id,
            prediction_id=prediction_id,
            signal_decision_id=signal_id,
        )
        ledger = IctPaperSignalLedgerRepository(audit)

        first = ledger.open_event(
            manifest=manifest,
            audit_record=audit_record,
            signal=signal,
            bar=entry_bar,
            session_regime="new_york",
        )
        duplicate = ledger.open_event(
            manifest=manifest,
            audit_record=audit_record,
            signal=signal,
            bar=entry_bar,
            session_regime="new_york",
        )
        assert duplicate.event_id == first.event_id
        assert len(ledger.fetch_events()) == 1
        assert first.total_cost_ticks == pytest.approx(2.65)

        future_bars = [
            _bar(entry_time + timedelta(minutes=5 * offset), 5000.0 + 0.25 * offset)
            for offset in range(1, 21)
        ]
        store.upsert_bars(future_bars)
        assert ledger.settle_completed_events(future_bars[18]) == ()
        settled = ledger.settle_completed_events(future_bars[19])

        assert len(settled) == 1
        assert settled[0].exit_timestamp == future_bars[19].timestamp
        assert settled[0].gross_pnl_ticks == pytest.approx(20.0)
        assert settled[0].net_pnl_ticks == pytest.approx(17.35)
        assert settled[0].outcome == "win"
        row = store.connection.execute("SELECT * FROM ict_paper_signal_events").fetchone()
        assert row["stop_price"] is None
        assert row["target_price"] is None
        assert row["stop_target_semantics"] == "not_applicable"
        assert ledger.settle_completed_events(future_bars[19]) == ()


def _bar(timestamp: datetime, close: float) -> MarketBar:
    return MarketBar(
        asset="ES",
        timeframe="5m",
        timestamp=timestamp,
        open=close,
        high=close + 0.25,
        low=close - 0.25,
        close=close,
        volume=100.0,
        source="ict-ledger-test",
    )
