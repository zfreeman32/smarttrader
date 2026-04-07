from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.dashboard.app import (
    _confidence_title,
    _override_signal_selection_from_query,
    _resolve_dashboard_model_selection,
)
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore


def test_resolve_dashboard_model_selection_stays_pinned_to_configured_primary() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        _record_signal(
            audit,
            model_id="long_ote_tcn_v1_candidate",
            direction="long",
            timestamp=datetime(2026, 4, 7, 3, 50, tzinfo=UTC),
        )

        selection = _resolve_dashboard_model_selection(
            audit,
            configured_model_id="long_ote_tcn_v2_candidate",
            direction="long",
            manifest_model_ids=(
                "long_ote_tcn_v2_candidate",
                "long_ote_tcn_v1_candidate",
                "long_ote_xgb_v2_candidate",
            ),
        )

    assert selection.configured_model_id == "long_ote_tcn_v2_candidate"
    assert selection.resolved_model_id == "long_ote_tcn_v2_candidate"
    assert selection.used_fallback is False
    assert _confidence_title("Long Primary Confidence", selection) == "Long Primary Confidence | long_ote_tcn_v2_candidate"


def test_override_signal_selection_from_query_routes_by_signal_direction() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        long_signal_id = _record_signal(
            audit,
            model_id="long_ote_tcn_v1_candidate",
            direction="long",
            timestamp=datetime(2026, 4, 7, 3, 50, tzinfo=UTC),
        )

        resolved_long, resolved_short = _override_signal_selection_from_query(
            audit,
            query_signal_id=long_signal_id,
            long_signal_id=None,
            short_signal_id=None,
        )

    assert resolved_long == long_signal_id
    assert resolved_short is None


def _record_signal(
    audit: LiveAuditRepository,
    *,
    model_id: str,
    direction: str,
    timestamp: datetime,
) -> int:
    snapshot = FeatureSnapshot(
        asset="EURUSD",
        timeframe="5m",
        direction=direction,
        timestamp=timestamp,
        source_row_idx=1,
        feature_values={"feature_a": 0.42},
        valid_feature_count=1,
    )
    feature_snapshot_id = audit.record_feature_snapshot(snapshot)
    prediction = ModelPrediction(
        model_id=model_id,
        direction=direction,
        backend="xgboost",
        timestamp=timestamp,
        source_row_idx=1,
        regime="test_regime",
        raw_score=0.42,
        calibrated_probability=0.73,
        threshold_applied=0.55,
        threshold_source="global",
    )
    prediction_id = audit.record_prediction(
        prediction,
        feature_snapshot_id=feature_snapshot_id,
    )
    return audit.record_signal_decision(
        SignalDecision(
            model_id=model_id,
            direction=direction,
            timestamp=timestamp,
            source_row_idx=1,
            decision="shadow",
            probability=prediction.calibrated_probability,
            threshold=prediction.threshold_applied,
            regime=prediction.regime,
            reasons=["test_seed"],
            cooldown_bars_remaining=None,
        ),
        prediction_id=prediction_id,
    )
