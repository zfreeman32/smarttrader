"""Production full-policy gates for the frozen corrected S3 experiment."""
import json
from pathlib import Path

import pytest

from scripts.backtest_b2_s3_corrected import full_policy_decision
from ote_live.features.manifest import LivePolicy
from ote_live.policies.abstain import LiveAbstainState


@pytest.fixture
def policy():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/b2_s3_corrected_backtest_20260927.json").read_text())
    return LivePolicy.model_validate(cfg["full_policy"])


def decide(policy, state, idx=100, **changes):
    row = dict(datetime="2025-01-02T15:00:00Z", session_regime="new_york",
               stress_regime="normal", expected_move_pips=3.)
    row.update(changes)
    return full_policy_decision(policy, row, probability=.7, threshold_passed=True,
                                source_row_idx=idx, ready=True, state=state)


def test_missing_expected_move_fails_closed_and_does_not_advance_emit(policy):
    state = LiveAbstainState()
    result = decide(policy, state, expected_move_pips=None)
    assert not result["policy_candidate_eligible"]
    assert result["policy_checks"]["expected_move_spread"]["status"] == "unverified"
    assert "policy_input_missing:expected_move_pips" in result["rejection_reasons"]
    assert state.last_emitted_source_row_idx is None


def test_cooldown_counts_source_bars_after_policy_emit(policy):
    state = LiveAbstainState()
    assert decide(policy, state)["policy_candidate_eligible"]
    assert not decide(policy, state, idx=104)["policy_candidate_eligible"]
    assert state.last_emitted_source_row_idx == 100
    assert decide(policy, state, idx=105)["policy_candidate_eligible"]


@pytest.mark.parametrize("changes,reason", [
    ({"stress_regime": "high"}, "high_stress"),
    ({"session_regime": "off_hours"}, "off_hours"),
    ({"expected_move_pips": 2.99}, "expected_move_below_spread"),
    ({"stress_regime": None}, "policy_input_missing:stress_regime"),
])
def test_exclusions_are_retained(policy, changes, reason):
    state = LiveAbstainState()
    result = decide(policy, state, **changes)
    assert not result["policy_candidate_eligible"]
    assert reason in result["rejection_reasons"]
    assert state.last_emitted_source_row_idx is None


def test_readiness_and_threshold_cannot_be_bypassed(policy):
    row = dict(datetime="2025-01-02T15:00:00Z", session_regime="new_york",
               stress_regime="normal", expected_move_pips=3.)
    for ready, passed in [(False, True), (True, False)]:
        state = LiveAbstainState()
        result = full_policy_decision(policy, row, probability=.7, threshold_passed=passed,
                                      source_row_idx=100, ready=ready, state=state)
        assert not result["policy_candidate_eligible"]
        assert state.last_emitted_source_row_idx is None
