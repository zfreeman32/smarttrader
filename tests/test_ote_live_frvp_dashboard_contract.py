from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.dashboard.app import (
    _format_frvp_paper_markout_summary,
    _runtime_manifest_hash,
)
from ote_live.dashboard.queries import (
    fetch_confidence_history,
    fetch_frvp_paper_signal_markouts,
    fetch_recent_signals,
    summarize_frvp_paper_signal_markouts,
)
from ote_live.dashboard import view_registry
from ote_live.models.loaders import load_direction_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore


CONTROLLED_LONG_PATH = (
    ROOT
    / "ote_live"
    / "runtime_manifests"
    / "frvp_es_paper_signal_20260816"
    / "live_runtime_manifest_long.json"
)
CONTROLLED_SHORT_PATH = CONTROLLED_LONG_PATH.with_name(
    "live_runtime_manifest_short.json"
)
CONTROLLED_REGISTRY_PATH = (
    ROOT / "models" / "frvp_es_paper_signal_registry_20260816.json"
)
SHADOW_LONG_PATH = (
    ROOT
    / "ote_live"
    / "runtime_manifests"
    / "frvp_es_shadow_20260721"
    / "live_runtime_manifest_long.json"
)


def test_controlled_frvp_label_requires_every_frozen_file_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_path = tmp_path / "long.json"
    short_path = tmp_path / "short.json"
    registry_path = tmp_path / "registry.json"
    long_path.write_bytes(CONTROLLED_LONG_PATH.read_bytes())
    short_path.write_bytes(CONTROLLED_SHORT_PATH.read_bytes())
    registry_path.write_bytes(CONTROLLED_REGISTRY_PATH.read_bytes())
    monkeypatch.setattr(
        view_registry, "FRVP_PAPER_SIGNAL_LONG_RUNTIME_MANIFEST_PATH", long_path
    )
    monkeypatch.setattr(
        view_registry, "FRVP_PAPER_SIGNAL_SHORT_RUNTIME_MANIFEST_PATH", short_path
    )
    monkeypatch.setattr(
        view_registry, "FRVP_PAPER_SIGNAL_REGISTRY_PATH", registry_path
    )

    active_ids, label = view_registry.resolve_frvp_dashboard_presentation(
        long_path, short_path, registry_path
    )
    assert active_ids == ("frvp_long_reversal_xgb_v1",)
    assert label == "controlled paper-signal view"

    registry_path.write_text("{}", encoding="utf-8")
    active_ids, label = view_registry.resolve_frvp_dashboard_presentation(
        long_path, short_path, registry_path
    )
    assert active_ids == ()
    assert "INVALID" in label
    assert "controlled label refused" in label


def test_controlled_history_filters_same_model_id_by_exact_manifest_hash(
    tmp_path: Path,
) -> None:
    controlled = next(
        model
        for model in load_direction_runtime_manifest(CONTROLLED_LONG_PATH).models
        if model.model_id == "frvp_long_reversal_xgb_v1"
    )
    shadow = next(
        model
        for model in load_direction_runtime_manifest(SHADOW_LONG_PATH).models
        if model.model_id == controlled.model_id
    )
    db_path = tmp_path / "dashboard.sqlite"
    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        shadow_manifest = audit.record_runtime_manifest(shadow)
        controlled_manifest = audit.record_runtime_manifest(controlled)
        _seed_signal(
            audit,
            runtime_manifest_id=shadow_manifest.runtime_manifest_id,
            model_id=controlled.model_id,
            timestamp=datetime(2026, 7, 31, 14, 0, tzinfo=UTC),
            probability=0.99,
            decision="shadow",
        )
        controlled_signal_id = _seed_signal(
            audit,
            runtime_manifest_id=controlled_manifest.runtime_manifest_id,
            model_id=controlled.model_id,
            timestamp=datetime(2026, 8, 20, 14, 0, tzinfo=UTC),
            probability=0.71,
            decision="emit",
        )

        expected_hash = _runtime_manifest_hash(controlled)
        signals = fetch_recent_signals(
            audit,
            model_ids=(controlled.model_id,),
            runtime_manifest_hashes=(expected_hash,),
        )
        confidence = fetch_confidence_history(
            audit,
            model_id=controlled.model_id,
            runtime_manifest_hashes=(expected_hash,),
        )

    assert signals["signal_decision_id"].tolist() == [controlled_signal_id]
    assert signals["probability"].tolist() == [pytest.approx(0.71)]
    assert confidence["calibrated_probability"].tolist() == [pytest.approx(0.71)]
    assert set(signals["manifest_hash"]) == {expected_hash}
    assert set(confidence["manifest_hash"]) == {expected_hash}


def test_controlled_paper_markout_uses_120_bar_friction_adjusted_es_ticks(
    tmp_path: Path,
) -> None:
    controlled = next(
        model
        for model in load_direction_runtime_manifest(CONTROLLED_LONG_PATH).models
        if model.model_id == "frvp_long_reversal_xgb_v1"
    )
    db_path = tmp_path / "markouts.sqlite"
    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        persisted = audit.record_runtime_manifest(controlled)
        signal_id = _seed_signal(
            audit,
            runtime_manifest_id=persisted.runtime_manifest_id,
            model_id=controlled.model_id,
            timestamp=datetime(2026, 8, 20, 14, 0, tzinfo=UTC),
            probability=0.71,
            decision="emit",
        )
        manifest_hash = _runtime_manifest_hash(controlled)
        _insert_settled_ledger_event(
            store,
            runtime_manifest_id=persisted.runtime_manifest_id,
            signal_decision_id=signal_id,
            manifest_hash=manifest_hash,
        )

        markouts = fetch_frvp_paper_signal_markouts(
            store,
            bundle_id=view_registry.FRVP_PAPER_SIGNAL_BUNDLE_ID,
            runtime_manifest_hashes=(manifest_hash,),
        )
        summary = summarize_frvp_paper_signal_markouts(markouts)

    assert markouts["holding_period_bars"].tolist() == [120]
    assert markouts["net_markout_ticks"].tolist() == [pytest.approx(9.35)]
    assert markouts["total_cost_ticks"].tolist() == [pytest.approx(2.65)]
    assert summary.avg_net_ticks == pytest.approx(9.35)
    display = _format_frvp_paper_markout_summary(summary)
    assert "120-bar ES" in display
    assert "9.35 net ticks after friction" in display
    assert "pips" not in display


def _seed_signal(
    audit: LiveAuditRepository,
    *,
    runtime_manifest_id: int,
    model_id: str,
    timestamp: datetime,
    probability: float,
    decision: str,
) -> int:
    snapshot_id = audit.record_feature_snapshot(
        FeatureSnapshot(
            asset="ES",
            timeframe="5m",
            direction="long",
            timestamp=timestamp,
            source_row_idx=1,
            feature_values={"feature_a": 0.5},
            valid_feature_count=1,
        ),
        runtime_manifest_id=runtime_manifest_id,
    )
    prediction_id = audit.record_prediction(
        ModelPrediction(
            model_id=model_id,
            direction="long",
            backend="xgboost",
            timestamp=timestamp,
            source_row_idx=1,
            regime="ranging_low",
            raw_score=probability,
            calibrated_probability=probability,
            threshold_applied=0.6,
            threshold_source="global",
        ),
        feature_snapshot_id=snapshot_id,
        runtime_manifest_id=runtime_manifest_id,
    )
    return audit.record_signal_decision(
        SignalDecision(
            model_id=model_id,
            direction="long",
            timestamp=timestamp,
            source_row_idx=1,
            decision=decision,
            probability=probability,
            threshold=0.6,
            regime="ranging_low",
            reasons=["test_seed"],
            cooldown_bars_remaining=None,
        ),
        prediction_id=prediction_id,
        runtime_manifest_id=runtime_manifest_id,
    )


def _insert_settled_ledger_event(
    store: SQLiteLiveDataStore,
    *,
    runtime_manifest_id: int,
    signal_decision_id: int,
    manifest_hash: str,
) -> None:
    store.connection.execute(
        """
        INSERT INTO frvp_paper_signal_events (
            event_key, bundle_id, runtime_manifest_id, manifest_hash,
            signal_decision_id, model_id, direction, asset, timeframe,
            source_timestamp_utc, source_row_idx, probability, threshold,
            composite_regime, session_regime, entry_timing,
            entry_timestamp_utc, entry_source_row_idx, entry_price,
            holding_period_bars, exit_timing, stop_price, target_price,
            stop_target_semantics, tick_size, tick_value, spread_cost_mode,
            entry_spread_ticks, exit_spread_ticks, fixed_slippage_ticks,
            commission_ticks, total_cost_ticks, lifecycle_status,
            exit_timestamp_utc, exit_source_row_idx, exit_price,
            gross_pnl_ticks, net_pnl_ticks, gross_pnl_dollars,
            net_pnl_dollars, outcome, metadata_json, created_at_utc,
            updated_at_utc, settled_at_utc
        ) VALUES (
            'dashboard-test-event', ?, ?, ?, ?, 'frvp_long_reversal_xgb_v1',
            'long', 'ES', '5m', '2026-08-20T14:00:00+00:00', 1, 0.71,
            0.60, 'ranging_low', 'new_york', 'signal_close',
            '2026-08-20T14:00:00+00:00', 1, 6500.00, 120,
            'close_after_120_completed_bars', NULL, NULL, 'not_applicable',
            0.25, 12.5, 'session_schedule', 1.0, 1.0, 0.25, 0.40, 2.65,
            'settled', '2026-08-21T00:00:00+00:00', 121, 6503.00,
            12.0, 9.35, 150.0, 116.875, 'win', '{}',
            '2026-08-21T00:00:00+00:00', '2026-08-21T00:00:00+00:00',
            '2026-08-21T00:00:00+00:00'
        )
        """,
        (
            view_registry.FRVP_PAPER_SIGNAL_BUNDLE_ID,
            runtime_manifest_id,
            manifest_hash,
            signal_decision_id,
        ),
    )
    store.connection.commit()
