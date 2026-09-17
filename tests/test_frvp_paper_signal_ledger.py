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
from ote_live.storage.frvp_paper_signal import (
    FRVP_PAPER_SIGNAL_BUNDLE_ID,
    FrvpPaperSignalLedgerRepository,
    supports_frvp_paper_signal_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER_SIGNAL_MANIFEST_ROOT = (
    ROOT / "ote_live" / "runtime_manifests" / "frvp_es_paper_signal_20260816"
)
CONTINUATION_MODEL_ID = "frvp_long_continuation_xgb_v1"
REVERSAL_MODEL_ID = "frvp_long_reversal_xgb_v1"


def test_frvp_reversal_events_overlap_and_settle_on_each_exact_120th_bar() -> None:
    db_path = ROOT / "tmp" / "frvp_paper_signal_ledger_tests" / f"{uuid.uuid4().hex}.sqlite"
    entry_time = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)
    bars = [
        _bar(entry_time + timedelta(minutes=5 * offset), 5000.0 + 0.25 * offset)
        for offset in range(122)
    ]

    with SQLiteLiveDataStore(db_path) as store:
        store.upsert_bars(bars)
        audit = LiveAuditRepository(store)
        ledger = FrvpPaperSignalLedgerRepository(audit)
        continuation_manifest = _contract_manifest(CONTINUATION_MODEL_ID)
        reversal_manifest = _contract_manifest(REVERSAL_MODEL_ID)
        assert not supports_frvp_paper_signal_manifest(continuation_manifest)
        assert supports_frvp_paper_signal_manifest(reversal_manifest)

        continuation_audit, continuation_signal = _persist_signal_chain(
            audit,
            manifest=continuation_manifest,
            bar=bars[0],
            source_row_idx=100,
            probability=0.75,
        )
        first_audit, first_signal = _persist_signal_chain(
            audit,
            manifest=reversal_manifest,
            bar=bars[0],
            source_row_idx=100,
            probability=0.72,
        )
        second_audit, second_signal = _persist_signal_chain(
            audit,
            manifest=reversal_manifest,
            bar=bars[1],
            source_row_idx=101,
            probability=0.74,
        )

        with pytest.raises(ValueError, match="not an active FRVP paper-signal contract"):
            ledger.open_event(
                manifest=continuation_manifest,
                audit_record=continuation_audit,
                signal=continuation_signal,
                bar=bars[0],
                session_regime="new_york",
            )

        first = ledger.open_event(
            manifest=reversal_manifest,
            audit_record=first_audit,
            signal=first_signal,
            bar=bars[0],
            session_regime="new_york",
        )
        duplicate = ledger.open_event(
            manifest=reversal_manifest,
            audit_record=first_audit,
            signal=first_signal,
            bar=bars[0],
            session_regime="new_york",
        )
        second = ledger.open_event(
            manifest=reversal_manifest,
            audit_record=second_audit,
            signal=second_signal,
            bar=bars[1],
            session_regime="asia",
        )

        assert duplicate.event_id == first.event_id
        assert second.event_id != first.event_id
        assert len(ledger.fetch_events(lifecycle_status="open")) == 2
        assert first.total_cost_ticks == pytest.approx(2.65)
        assert second.total_cost_ticks == pytest.approx(3.65)

        assert ledger.settle_completed_events(bars[119]) == ()
        first_settlement = ledger.settle_completed_events(bars[120])
        assert [event.model_id for event in first_settlement] == [REVERSAL_MODEL_ID]
        assert first_settlement[0].gross_pnl_ticks == pytest.approx(120.0)
        assert first_settlement[0].net_pnl_ticks == pytest.approx(117.35)
        assert first_settlement[0].outcome == "win"
        assert len(ledger.fetch_events(lifecycle_status="open")) == 1

        second_settlement = ledger.settle_completed_events(bars[121])
        assert [event.model_id for event in second_settlement] == [REVERSAL_MODEL_ID]
        assert second_settlement[0].gross_pnl_ticks == pytest.approx(120.0)
        assert second_settlement[0].net_pnl_ticks == pytest.approx(116.35)
        assert ledger.settle_completed_events(bars[121]) == ()

        rows = store.connection.execute(
            "SELECT * FROM frvp_paper_signal_events ORDER BY source_timestamp_utc"
        ).fetchall()
        assert {row["bundle_id"] for row in rows} == {FRVP_PAPER_SIGNAL_BUNDLE_ID}
        assert {row["model_id"] for row in rows} == {REVERSAL_MODEL_ID}
        assert all(row["manifest_hash"] for row in rows)
        assert all(row["stop_price"] is None and row["target_price"] is None for row in rows)
        assert {row["stop_target_semantics"] for row in rows} == {"not_applicable"}
        assert {row["holding_period_bars"] for row in rows} == {120}


def test_reconcile_recovers_only_eligible_missing_emit_and_is_idempotent() -> None:
    db_path = ROOT / "tmp" / "frvp_paper_signal_ledger_tests" / f"{uuid.uuid4().hex}.sqlite"
    entry_time = datetime(2026, 8, 4, 14, 0, tzinfo=UTC)
    bars = [
        _bar(entry_time + timedelta(minutes=5 * offset), 5100.0 + 0.25 * offset)
        for offset in range(3)
    ]

    with SQLiteLiveDataStore(db_path) as store:
        store.upsert_bars(bars)
        audit = LiveAuditRepository(store)
        ledger = FrvpPaperSignalLedgerRepository(audit)
        manifest = _contract_manifest(REVERSAL_MODEL_ID)

        legacy_audit, _ = _persist_signal_chain(
            audit,
            manifest=manifest,
            bar=bars[0],
            source_row_idx=200,
            probability=0.71,
            paper_signal_event_eligible=None,
        )
        seed_audit, _ = _persist_signal_chain(
            audit,
            manifest=manifest,
            bar=bars[1],
            source_row_idx=201,
            probability=0.72,
            paper_signal_event_eligible=False,
        )
        eligible_audit, _ = _persist_signal_chain(
            audit,
            manifest=manifest,
            bar=bars[2],
            source_row_idx=202,
            probability=0.73,
            paper_signal_event_eligible=True,
            session_regime="asia",
        )

        recovered = ledger.reconcile_missing_events()

        assert len(recovered) == 1
        assert recovered[0].source_row_idx == 202
        assert recovered[0].total_cost_ticks == pytest.approx(3.65)
        rows = store.connection.execute(
            "SELECT signal_decision_id FROM frvp_paper_signal_events ORDER BY id"
        ).fetchall()
        assert [int(row["signal_decision_id"]) for row in rows] == [
            eligible_audit.signal_decision_id
        ]
        assert legacy_audit.signal_decision_id not in {
            int(row["signal_decision_id"]) for row in rows
        }
        assert seed_audit.signal_decision_id not in {
            int(row["signal_decision_id"]) for row in rows
        }

        assert ledger.reconcile_missing_events() == ()
        assert len(ledger.fetch_events()) == 1


def test_frozen_manifest_hash_rejects_live_policy_mutation() -> None:
    db_path = ROOT / "tmp" / "frvp_paper_signal_ledger_tests" / f"{uuid.uuid4().hex}.sqlite"
    bar = _bar(datetime(2026, 8, 5, 14, 0, tzinfo=UTC), 5200.0)
    manifest = _contract_manifest(REVERSAL_MODEL_ID)
    mutated_thresholds = manifest.live_policy.thresholds.model_copy(
        update={"global_threshold": 0.61}
    )
    mutated_policy = manifest.live_policy.model_copy(update={"thresholds": mutated_thresholds})
    mutated_manifest = manifest.model_copy(update={"live_policy": mutated_policy})

    assert supports_frvp_paper_signal_manifest(manifest)
    assert not supports_frvp_paper_signal_manifest(mutated_manifest)

    with SQLiteLiveDataStore(db_path) as store:
        store.upsert_bars([bar])
        audit = LiveAuditRepository(store)
        ledger = FrvpPaperSignalLedgerRepository(audit)
        audit_record, signal = _persist_signal_chain(
            audit,
            manifest=mutated_manifest,
            bar=bar,
            source_row_idx=300,
            probability=0.72,
            paper_signal_event_eligible=True,
        )

        with pytest.raises(ValueError, match="not an active FRVP paper-signal contract"):
            ledger.open_event(
                manifest=mutated_manifest,
                audit_record=audit_record,
                signal=signal,
                bar=bar,
                session_regime="new_york",
            )


def _contract_manifest(model_id: str):
    manifest_path = PAPER_SIGNAL_MANIFEST_ROOT / model_id / "live_runtime_manifest.json"
    return load_live_runtime_manifest(manifest_path)


def _persist_signal_chain(
    audit: LiveAuditRepository,
    *,
    manifest,
    bar: MarketBar,
    source_row_idx: int,
    probability: float,
    paper_signal_event_eligible: bool | None = None,
    session_regime: str = "new_york",
) -> tuple[PersistedAuditRecord, SignalDecision]:
    persisted = audit.record_runtime_manifest(manifest)
    snapshot = FeatureSnapshot(
        asset="ES",
        timeframe="5m",
        direction="long",
        timestamp=bar.timestamp,
        source_row_idx=source_row_idx,
        feature_values={manifest.feature_manifest.selected_feature_names[0]: 1.0},
        valid_feature_count=1,
    )
    snapshot_id = audit.record_feature_snapshot(
        snapshot,
        runtime_manifest_id=persisted.runtime_manifest_id,
    )
    threshold = float(manifest.live_policy.thresholds.global_threshold)
    prediction = ModelPrediction(
        model_id=manifest.model_id,
        direction="long",
        backend=manifest.backend,
        timestamp=bar.timestamp,
        source_row_idx=source_row_idx,
        regime="trend_up_normal",
        raw_score=probability,
        calibrated_probability=probability,
        threshold_applied=threshold,
        threshold_source="global",
    )
    prediction_id = audit.record_prediction(prediction, feature_snapshot_id=snapshot_id)
    signal = SignalDecision(
        model_id=manifest.model_id,
        direction="long",
        timestamp=bar.timestamp,
        source_row_idx=source_row_idx,
        decision="emit",
        probability=probability,
        threshold=threshold,
        regime="trend_up_normal",
        reasons=["threshold_passed"],
    )
    metadata = {"policy_context": {"session_regime": session_regime}}
    if paper_signal_event_eligible is not None:
        metadata["paper_signal_event_eligible"] = paper_signal_event_eligible
    signal_id = audit.record_signal_decision(
        signal,
        prediction_id=prediction_id,
        metadata=metadata,
    )
    return (
        PersistedAuditRecord(
            runtime_manifest_id=persisted.runtime_manifest_id,
            feature_snapshot_id=snapshot_id,
            prediction_id=prediction_id,
            signal_decision_id=signal_id,
        ),
        signal,
    )


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
        source="frvp-ledger-test",
    )
