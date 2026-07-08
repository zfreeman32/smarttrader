from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from model_testing.ote_abstain_policy import DEFAULT_SESSION_SPREAD_PIPS
from ote_live.features.manifest import LivePolicy


@dataclass
class LiveAbstainState:
    probability_history_limit: int = 512
    last_emitted_source_row_idx: int | None = None
    candidate_probabilities: deque[float] = field(default_factory=deque)

    def remember_candidate(self, probability: float) -> None:
        self.candidate_probabilities.append(float(probability))
        while len(self.candidate_probabilities) > self.probability_history_limit:
            self.candidate_probabilities.popleft()

    def record_emit(self, source_row_idx: int | None) -> None:
        if source_row_idx is not None:
            self.last_emitted_source_row_idx = int(source_row_idx)


@dataclass(frozen=True)
class LiveAbstainDecision:
    abstain: bool
    reason: str | None = None
    cooldown_bars_remaining: int | None = None


def evaluate_live_abstain(
    live_policy: LivePolicy,
    *,
    probability: float,
    source_row_idx: int | None,
    policy_context: pd.DataFrame | pd.Series | Mapping[str, Any] | None = None,
    state: LiveAbstainState | None = None,
) -> LiveAbstainDecision:
    abstain_policy = live_policy.abstain_policy
    if not abstain_policy.enabled:
        return LiveAbstainDecision(abstain=False)

    context_row = _coerce_policy_row(policy_context)
    session_regime = _optional_str(context_row.get("session_regime"))
    stress_regime = _optional_str(context_row.get("stress_regime"))
    composite_regime = _optional_str(context_row.get("composite_regime"))

    if abstain_policy.abstain_high_stress and stress_regime == "high":
        return LiveAbstainDecision(abstain=True, reason="high_stress")

    if abstain_policy.abstain_off_hours and session_regime == "off_hours":
        return LiveAbstainDecision(abstain=True, reason="off_hours")

    blocked_sessions = {str(value) for value in abstain_policy.abstain_session_regimes}
    if session_regime in blocked_sessions:
        return LiveAbstainDecision(abstain=True, reason="session_filter")

    blocked_composites = {str(value) for value in abstain_policy.abstain_composite_regimes}
    if composite_regime in blocked_composites:
        return LiveAbstainDecision(abstain=True, reason="composite_filter")

    blocked_composite_session_pairs = {
        (str(composite_value), str(session_value))
        for composite_value, session_value in abstain_policy.abstain_composite_session_pairs
    }
    if (
        composite_regime is not None
        and session_regime is not None
        and (composite_regime, session_regime) in blocked_composite_session_pairs
    ):
        return LiveAbstainDecision(abstain=True, reason="composite_session_filter")

    blocked_composite_stress_pairs = {
        (str(composite_value), str(stress_value))
        for composite_value, stress_value in abstain_policy.abstain_composite_stress_pairs
    }
    if (
        composite_regime is not None
        and stress_regime is not None
        and (composite_regime, stress_regime) in blocked_composite_stress_pairs
    ):
        return LiveAbstainDecision(abstain=True, reason="composite_stress_filter")

    minimum_quantile = abstain_policy.minimum_probability_quantile
    if minimum_quantile is not None and state is not None and state.candidate_probabilities:
        history = np.asarray([*state.candidate_probabilities, float(probability)], dtype=np.float32)
        cutoff = float(np.quantile(history, float(minimum_quantile)))
        if float(probability) <= cutoff:
            return LiveAbstainDecision(abstain=True, reason="probability_quantile_filter")

    expected_move_pips = _optional_float(context_row.get("expected_move_pips"))
    if expected_move_pips is not None and session_regime is not None:
        session_spread_pips = dict(DEFAULT_SESSION_SPREAD_PIPS)
        session_spread_pips.update(live_policy.cost_assumptions.session_spread_pips)
        spread = float(session_spread_pips.get(session_regime, session_spread_pips["off_hours"]))
        required_move = float(abstain_policy.minimum_expected_move_to_spread) * spread
        if expected_move_pips < required_move:
            return LiveAbstainDecision(abstain=True, reason="expected_move_below_spread")

    if (
        state is not None
        and source_row_idx is not None
        and state.last_emitted_source_row_idx is not None
        and int(source_row_idx) - int(state.last_emitted_source_row_idx) <= abstain_policy.cooldown_bars
    ):
        bars_since_emit = int(source_row_idx) - int(state.last_emitted_source_row_idx)
        cooldown_remaining = max(int(abstain_policy.cooldown_bars) - bars_since_emit + 1, 1)
        return LiveAbstainDecision(
            abstain=True,
            reason="cooldown",
            cooldown_bars_remaining=cooldown_remaining,
        )

    return LiveAbstainDecision(abstain=False)


def _coerce_policy_row(
    policy_context: pd.DataFrame | pd.Series | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if policy_context is None:
        return {}
    if isinstance(policy_context, pd.DataFrame):
        if policy_context.empty:
            return {}
        return policy_context.iloc[-1].to_dict()
    if isinstance(policy_context, pd.Series):
        return policy_context.to_dict()
    if isinstance(policy_context, Mapping):
        return dict(policy_context)
    raise TypeError("policy_context must be a pandas DataFrame, Series, or mapping.")


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
