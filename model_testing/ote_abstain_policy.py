from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import pandas as pd


DEFAULT_SESSION_SPREAD_PIPS = {
    "overlap": 1.0,
    "london": 1.5,
    "new_york": 1.5,
    "asia": 2.5,
    "off_hours": 3.0,
}


@dataclass(frozen=True)
class HardAbstainConfig:
    abstain_high_stress: bool = True
    abstain_off_hours: bool = True
    cooldown_bars: int = 4
    minimum_expected_move_to_spread: float = 2.0
    session_spread_pips: Mapping[str, float] = field(default_factory=lambda: DEFAULT_SESSION_SPREAD_PIPS)
    expected_move_by_regime: Optional[Mapping[str, float]] = None
    abstain_session_regimes: tuple[str, ...] = ()
    abstain_composite_regimes: tuple[str, ...] = ()
    abstain_composite_session_pairs: tuple[tuple[str, str], ...] = ()
    abstain_composite_stress_pairs: tuple[tuple[str, str], ...] = ()
    minimum_probability_quantile: float | None = None
    apply_to_base_policy_variants: bool = False
    signal_candidate_column: str = "policy_signal_candidate"
    probability_column: str = "policy_probability"
    composite_column: str = "composite_regime"
    session_column: str = "session_regime"
    stress_column: str = "stress_regime"
    position_column: str = "source_row_idx"


def apply_abstain_policy(
    frame: pd.DataFrame,
    *,
    config: HardAbstainConfig | None = None,
) -> pd.DataFrame:
    config = config or HardAbstainConfig()
    if config.signal_candidate_column not in frame.columns:
        raise ValueError(
            f"Signal candidate column {config.signal_candidate_column!r} not found in frame."
        )

    working = frame.copy()
    candidate_mask = working[config.signal_candidate_column].fillna(False).astype(bool)
    working["policy_emit_signal"] = False
    working["policy_abstain"] = False
    working["policy_abstain_reason"] = pd.Series(pd.NA, index=working.index, dtype="object")

    if config.abstain_high_stress and config.stress_column in working.columns:
        high_stress_mask = candidate_mask & (working[config.stress_column] == "high")
        _mark_abstain(working, high_stress_mask, "high_stress")

    if config.abstain_off_hours and config.session_column in working.columns:
        off_hours_mask = candidate_mask & working["policy_abstain_reason"].isna() & (
            working[config.session_column] == "off_hours"
        )
        _mark_abstain(working, off_hours_mask, "off_hours")

    explicit_session_mask = _build_explicit_session_filter_mask(working, config)
    if explicit_session_mask is not None:
        _mark_abstain(working, explicit_session_mask, "session_filter")

    explicit_composite_mask = _build_explicit_composite_filter_mask(working, config)
    if explicit_composite_mask is not None:
        _mark_abstain(working, explicit_composite_mask, "composite_filter")

    composite_session_pair_mask = _build_composite_session_pair_filter_mask(working, config)
    if composite_session_pair_mask is not None:
        _mark_abstain(working, composite_session_pair_mask, "composite_session_filter")

    composite_stress_pair_mask = _build_composite_stress_pair_filter_mask(working, config)
    if composite_stress_pair_mask is not None:
        _mark_abstain(working, composite_stress_pair_mask, "composite_stress_filter")

    expected_move_mask = _build_expected_move_filter_mask(working, config)
    if expected_move_mask is not None:
        _mark_abstain(working, expected_move_mask, "expected_move_below_spread")

    probability_quantile_mask = _build_probability_quantile_filter_mask(working, config)
    if probability_quantile_mask is not None:
        _mark_abstain(working, probability_quantile_mask, "probability_quantile_filter")

    active_candidates = candidate_mask & working["policy_abstain_reason"].isna()
    position_values = _resolve_positions(working, config.position_column)

    last_accepted_position: int | None = None
    for row_index in np.flatnonzero(active_candidates.to_numpy(copy=False)):
        current_position = int(position_values[row_index])
        if (
            last_accepted_position is not None
            and current_position - last_accepted_position <= config.cooldown_bars
        ):
            working.iat[row_index, working.columns.get_loc("policy_abstain")] = True
            working.iat[row_index, working.columns.get_loc("policy_abstain_reason")] = "cooldown"
            continue

        working.iat[row_index, working.columns.get_loc("policy_emit_signal")] = True
        last_accepted_position = current_position

    already_abstained = candidate_mask & working["policy_abstain_reason"].notna()
    working.loc[already_abstained, "policy_abstain"] = True
    return working


def estimate_session_spread_pips(
    session_regime: str,
    *,
    spread_table: Mapping[str, float] | None = None,
) -> float:
    spreads = spread_table or DEFAULT_SESSION_SPREAD_PIPS
    return float(spreads.get(session_regime, spreads["off_hours"]))


def _mark_abstain(frame: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    if not bool(mask.any()):
        return
    frame.loc[mask, "policy_abstain"] = True
    frame.loc[mask, "policy_abstain_reason"] = reason


def _build_expected_move_filter_mask(
    frame: pd.DataFrame,
    config: HardAbstainConfig,
) -> Optional[pd.Series]:
    if config.expected_move_by_regime is None:
        return None
    if config.composite_column not in frame.columns or config.session_column not in frame.columns:
        return None
    if config.probability_column not in frame.columns:
        return None

    probabilities = pd.to_numeric(frame[config.probability_column], errors="coerce").fillna(0.0)
    regime_expected_move = frame[config.composite_column].map(config.expected_move_by_regime).fillna(np.nan)
    session_spread = frame[config.session_column].map(
        lambda value: estimate_session_spread_pips(value, spread_table=config.session_spread_pips)
    )
    minimum_required_move = config.minimum_expected_move_to_spread * session_spread
    estimated_move = probabilities * regime_expected_move

    candidate_mask = frame[config.signal_candidate_column].fillna(False).astype(bool)
    not_already_abstained = frame["policy_abstain_reason"].isna()
    return candidate_mask & not_already_abstained & regime_expected_move.notna() & (
        estimated_move < minimum_required_move
    )


def _build_explicit_session_filter_mask(
    frame: pd.DataFrame,
    config: HardAbstainConfig,
) -> Optional[pd.Series]:
    if not config.abstain_session_regimes or config.session_column not in frame.columns:
        return None
    candidate_mask = frame[config.signal_candidate_column].fillna(False).astype(bool)
    not_already_abstained = frame["policy_abstain_reason"].isna()
    blocked_sessions = set(str(value) for value in config.abstain_session_regimes)
    return candidate_mask & not_already_abstained & frame[config.session_column].astype(str).isin(blocked_sessions)


def _build_explicit_composite_filter_mask(
    frame: pd.DataFrame,
    config: HardAbstainConfig,
) -> Optional[pd.Series]:
    if not config.abstain_composite_regimes or config.composite_column not in frame.columns:
        return None
    candidate_mask = frame[config.signal_candidate_column].fillna(False).astype(bool)
    not_already_abstained = frame["policy_abstain_reason"].isna()
    blocked_regimes = set(str(value) for value in config.abstain_composite_regimes)
    return candidate_mask & not_already_abstained & frame[config.composite_column].astype(str).isin(blocked_regimes)


def _build_composite_session_pair_filter_mask(
    frame: pd.DataFrame,
    config: HardAbstainConfig,
) -> Optional[pd.Series]:
    if (
        not config.abstain_composite_session_pairs
        or config.composite_column not in frame.columns
        or config.session_column not in frame.columns
    ):
        return None

    candidate_mask = frame[config.signal_candidate_column].fillna(False).astype(bool)
    not_already_abstained = frame["policy_abstain_reason"].isna()
    composite_values = frame[config.composite_column].astype(str)
    session_values = frame[config.session_column].astype(str)
    blocked_pairs = {
        (str(composite_regime), str(session_regime))
        for composite_regime, session_regime in config.abstain_composite_session_pairs
    }
    pair_matches = [
        (composite_regime, session_regime) in blocked_pairs
        for composite_regime, session_regime in zip(composite_values, session_values)
    ]
    return candidate_mask & not_already_abstained & pd.Series(pair_matches, index=frame.index, dtype=bool)


def _build_composite_stress_pair_filter_mask(
    frame: pd.DataFrame,
    config: HardAbstainConfig,
) -> Optional[pd.Series]:
    if (
        not config.abstain_composite_stress_pairs
        or config.composite_column not in frame.columns
        or config.stress_column not in frame.columns
    ):
        return None

    candidate_mask = frame[config.signal_candidate_column].fillna(False).astype(bool)
    not_already_abstained = frame["policy_abstain_reason"].isna()
    composite_values = frame[config.composite_column].astype(str)
    stress_values = frame[config.stress_column].astype(str)
    blocked_pairs = {
        (str(composite_regime), str(stress_regime))
        for composite_regime, stress_regime in config.abstain_composite_stress_pairs
    }
    pair_matches = [
        (composite_regime, stress_regime) in blocked_pairs
        for composite_regime, stress_regime in zip(composite_values, stress_values)
    ]
    return candidate_mask & not_already_abstained & pd.Series(pair_matches, index=frame.index, dtype=bool)


def _build_probability_quantile_filter_mask(
    frame: pd.DataFrame,
    config: HardAbstainConfig,
) -> Optional[pd.Series]:
    quantile = config.minimum_probability_quantile
    if quantile is None:
        return None
    quantile_value = float(quantile)
    if not 0.0 < quantile_value < 1.0:
        raise ValueError("minimum_probability_quantile must be between 0 and 1.")
    if config.probability_column not in frame.columns:
        return None

    candidate_mask = frame[config.signal_candidate_column].fillna(False).astype(bool)
    not_already_abstained = frame["policy_abstain_reason"].isna()
    candidate_probabilities = pd.to_numeric(
        frame.loc[candidate_mask & not_already_abstained, config.probability_column],
        errors="coerce",
    ).dropna()
    if candidate_probabilities.empty:
        return None

    cutoff = float(candidate_probabilities.quantile(quantile_value))
    probabilities = pd.to_numeric(frame[config.probability_column], errors="coerce")
    return candidate_mask & not_already_abstained & probabilities.le(cutoff)


def _resolve_positions(frame: pd.DataFrame, position_column: str) -> np.ndarray:
    if position_column not in frame.columns:
        return np.arange(len(frame), dtype=np.int64)
    return pd.to_numeric(frame[position_column], errors="coerce").fillna(-1).astype(np.int64).to_numpy(copy=True)
