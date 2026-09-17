from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ote_live.contracts.prediction import ModelPrediction
from ote_live.features.manifest import LivePolicy
from ote_live.policies.decision_engine import LiveDecisionEngine
from ote_live.policies.regime import resolve_latest_regime


def test_resolve_latest_regime_accepts_timestamp_only_policy_context() -> None:
    regime = resolve_latest_regime(
        {
            "timestamp": "2024-01-02T07:30:00+00:00",
            "close": 1.20,
            "ema_alignment": 0.25,
            "atr_14": 1.0,
            "range_shock_20": 2.5,
        }
    )

    assert regime is not None
    assert regime.trend_regime == "weak_up"
    assert regime.vol_regime == "high"
    assert regime.session_regime == "london"
    assert regime.stress_regime == "elevated"
    assert regime.composite_regime == "weak_up_high"


@pytest.mark.parametrize(
    ("timestamp", "range_shock", "abstain_high_stress", "abstain_off_hours", "reason"),
    [
        ("2024-01-02T07:30:00+00:00", 4.0, True, False, "high_stress"),
        ("2024-01-02T23:00:00+00:00", 1.0, False, True, "off_hours"),
    ],
)
def test_decision_engine_applies_abstains_from_generated_regime_context(
    timestamp: str,
    range_shock: float,
    abstain_high_stress: bool,
    abstain_off_hours: bool,
    reason: str,
) -> None:
    live_policy = LivePolicy.model_validate(
        {
            "model_id": "regime_context_test",
            "direction": "long",
            "backend": "xgboost",
            "calibration_method": "sigmoid",
            "policy_status": "complete",
            "thresholds": {"global_threshold": 0.5, "regime_thresholds": {}},
            "abstain_policy": {
                "enabled": True,
                "abstain_high_stress": abstain_high_stress,
                "abstain_off_hours": abstain_off_hours,
                "cooldown_bars": 0,
                "minimum_expected_move_to_spread": 0.0,
            },
            "cost_assumptions": {},
            "lineage": {"threshold_registry_path": "unit-test"},
        }
    )
    prediction = ModelPrediction(
        model_id=live_policy.model_id,
        direction=live_policy.direction,
        backend=live_policy.backend,
        timestamp=datetime.fromisoformat(timestamp),
        source_row_idx=100,
        regime=None,
        raw_score=0.75,
        calibrated_probability=0.75,
        threshold_applied=None,
        threshold_source="unknown",
    )

    path = LiveDecisionEngine().evaluate_prediction(
        prediction,
        live_policy=live_policy,
        policy_context={
            "timestamp": timestamp,
            "close": 1.20,
            "ema_alignment": 0.25,
            "atr_14": 1.0,
            "range_shock_20": range_shock,
        },
    )

    assert path.signal.decision == "abstain"
    assert path.signal.reasons == [reason]
