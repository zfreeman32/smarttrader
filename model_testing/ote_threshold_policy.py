from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from model_testing.ote_abstain_policy import (
    DEFAULT_SESSION_SPREAD_PIPS,
    HardAbstainConfig,
    apply_abstain_policy,
    estimate_session_spread_pips,
)
from model_testing.ote_regime_slices import (
    DEFAULT_THRESHOLD_GRID,
    event_metrics_by_position,
    group_positive_zones_by_position,
    predictions_to_events_by_position,
)


LONG_THRESHOLD_HINTS = {
    "strong_up_low": 0.55,
    "strong_up_medium": 0.60,
    "strong_up_high": 0.65,
    "weak_up_low": 0.65,
    "weak_up_medium": 0.65,
    "weak_up_high": 0.65,
    "ranging_low": 0.70,
    "strong_down_medium": 0.75,
    "strong_down_high": None,
}

SHORT_THRESHOLD_HINTS = {
    "strong_up_low": 0.80,
    "strong_up_medium": 0.75,
    "strong_up_high": None,
    "weak_up_low": 0.70,
    "weak_up_medium": 0.70,
    "weak_up_high": 0.70,
    "ranging_low": 0.70,
    "strong_down_medium": 0.60,
    "strong_down_high": 0.65,
}


@dataclass(frozen=True)
class ThresholdSearchConfig:
    probability_column: str
    global_threshold: float
    composite_column: str = "composite_regime"
    session_column: str = "session_regime"
    target_column: str = "target"
    year_column: str = "year"
    datetime_column: str = "datetime"
    position_column: str = "source_row_idx"
    entry_price_column: str = "close"
    exit_price_column: str = "policy_exit_close"
    exit_datetime_column: str = "policy_exit_datetime"
    gross_pnl_column: str = "policy_gross_pnl_pips"
    total_cost_column: str = "policy_total_cost_pips"
    net_pnl_column: str = "policy_net_pnl_pips"
    threshold_grid: Tuple[float, ...] = field(default_factory=lambda: DEFAULT_THRESHOLD_GRID)
    event_tolerance_bars: int = 2
    event_cooldown_bars: int = 4
    min_positive_events: int = 50
    min_events_per_month: float = 3.0
    label_max_holding_bars: int = 120
    pip_size: float = 1e-4
    approx_spread_column: str = "approx_spread"
    slippage_spread_multiplier: float = 0.25
    fixed_slippage_pips_per_trade: float = 0.0
    commission_pips_per_trade: float = 0.0
    session_spread_pips: Mapping[str, float] = field(default_factory=lambda: DEFAULT_SESSION_SPREAD_PIPS)


def default_threshold_hints(direction: str) -> Mapping[str, Optional[float]]:
    if direction == "long":
        return LONG_THRESHOLD_HINTS
    if direction == "short":
        return SHORT_THRESHOLD_HINTS
    raise ValueError(f"Unsupported direction for threshold hints: {direction!r}")


def search_global_threshold(
    frame: pd.DataFrame,
    *,
    config: ThresholdSearchConfig,
) -> Dict[str, object]:
    prepared = prepare_policy_frame(frame, config)
    if prepared.empty:
        return {
            "threshold": float(config.global_threshold),
            "selection_score": 0.0,
            "event_f05": 0.0,
            "avg_events_per_month": 0.0,
            "used_leave_one_year_out": False,
        }
    return _search_best_threshold(prepared, config=config, fallback_threshold=config.global_threshold)


def search_thresholds_by_composite_regime(
    frame: pd.DataFrame,
    *,
    direction: str,
    config: ThresholdSearchConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    prepared = prepare_policy_frame(frame, config)
    global_result = search_global_threshold(prepared, config=config)
    hints = default_threshold_hints(direction)

    rows: List[Dict[str, object]] = []
    for regime_key, bucket in prepared.groupby(config.composite_column, sort=True):
        bucket = bucket.sort_values(config.position_column).reset_index(drop=True)
        positions = bucket[config.position_column].to_numpy(dtype=np.int64, copy=True)
        y_true = bucket[config.target_column].to_numpy(dtype=np.uint8, copy=True)
        true_events = len(group_positive_zones_by_position(y_true, positions))
        threshold_data_sufficient = true_events >= config.min_positive_events

        if threshold_data_sufficient:
            result = _search_best_threshold(
                bucket,
                config=config,
                fallback_threshold=config.global_threshold,
            )
            threshold_source = "regime"
        else:
            result = evaluate_threshold_candidate(
                bucket,
                threshold=global_result["threshold"],
                config=config,
            )
            result["threshold"] = float(global_result["threshold"])
            result["selection_score"] = float(global_result["selection_score"])
            result["used_leave_one_year_out"] = bool(global_result["used_leave_one_year_out"])
            threshold_source = "global_fallback"

        hint_value = hints.get(str(regime_key))
        rows.append(
            {
                "composite_regime": str(regime_key),
                "threshold": float(result["threshold"]),
                "threshold_source": threshold_source,
                "global_threshold": float(global_result["threshold"]),
                "plan_hint_threshold": np.nan if hint_value is None else float(hint_value),
                "plan_hint_abstain": bool(str(regime_key) in hints and hint_value is None),
                "true_events": int(true_events),
                "threshold_data_sufficient": bool(threshold_data_sufficient),
                "selection_score": float(result["selection_score"]),
                "normalized_expectancy": float(result.get("normalized_expectancy", 0.0)),
                "event_precision": float(result["event_precision"]),
                "event_recall": float(result["event_recall"]),
                "event_f05": float(result["event_f05"]),
                "predicted_events": int(result["predicted_events"]),
                "avg_events_per_month": float(result["avg_events_per_month"]),
                "trades_per_week": float(result.get("trades_per_week", 0.0)),
                "gross_expectancy_pips": float(result.get("gross_expectancy_pips", 0.0)),
                "gross_pnl_pips": float(result.get("gross_pnl_pips", 0.0)),
                "post_cost_expectancy_pips": float(result.get("post_cost_expectancy_pips", 0.0)),
                "net_pnl_pips": float(result.get("net_pnl_pips", 0.0)),
                "used_leave_one_year_out": bool(result["used_leave_one_year_out"]),
            }
        )

    return pd.DataFrame(rows), global_result


def apply_threshold_policy(
    frame: pd.DataFrame,
    *,
    config: ThresholdSearchConfig,
    policy_table: pd.DataFrame,
    policy_name: str,
    fallback_threshold: float,
) -> pd.DataFrame:
    if config.probability_column not in frame.columns:
        raise ValueError(f"Probability column {config.probability_column!r} not present in frame.")

    working = prepare_policy_frame(frame, config)
    threshold_lookup = {
        str(row["composite_regime"]): float(row["threshold"])
        for _, row in policy_table.iterrows()
        if pd.notna(row.get("threshold"))
    }
    threshold_source_lookup = {
        str(row["composite_regime"]): str(row["threshold_source"])
        for _, row in policy_table.iterrows()
    }

    probabilities = pd.to_numeric(working[config.probability_column], errors="coerce").fillna(0.0)
    working["policy_name"] = policy_name
    working["policy_probability"] = probabilities
    working["policy_threshold"] = (
        working[config.composite_column].astype(str).map(threshold_lookup).fillna(float(fallback_threshold))
    )
    working["policy_threshold_source"] = (
        working[config.composite_column].astype(str).map(threshold_source_lookup).fillna("global")
    )
    working["policy_signal_candidate"] = probabilities >= working["policy_threshold"]
    return working


def evaluate_policy_variants(
    *,
    oof_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    policy_table: pd.DataFrame,
    config: ThresholdSearchConfig,
    abstain_config: HardAbstainConfig | None = None,
) -> pd.DataFrame:
    abstain_config = abstain_config or HardAbstainConfig(cooldown_bars=config.event_cooldown_bars)

    variants = [
        ("global_threshold", pd.DataFrame(columns=policy_table.columns), float(config.global_threshold), False),
        ("global_threshold_plus_abstain", pd.DataFrame(columns=policy_table.columns), float(config.global_threshold), True),
        ("regime_threshold", policy_table, float(config.global_threshold), False),
        ("regime_threshold_plus_abstain", policy_table, float(config.global_threshold), True),
    ]

    rows: List[Dict[str, object]] = []
    for dataset_split, frame in (("oof", oof_frame), ("test", test_frame)):
        frame_config = _resolve_frame_config(frame, config)
        for policy_name, table, fallback_threshold, use_abstain in variants:
            thresholded = apply_threshold_policy(
                frame,
                config=frame_config,
                policy_table=table,
                policy_name=policy_name,
                fallback_threshold=fallback_threshold,
            )
            if use_abstain:
                decided = apply_abstain_policy(thresholded, config=abstain_config)
                emitted_positions = _resolve_positions(decided, config.position_column)[
                    decided["policy_emit_signal"].to_numpy(copy=False)
                ]
                abstain_count = int(decided["policy_abstain"].sum())
                candidate_count = int(decided["policy_signal_candidate"].sum())
            else:
                decided = thresholded.copy()
                decided["policy_emit_signal"] = decided["policy_signal_candidate"]
                decided["policy_abstain"] = False
                emitted_positions = _resolve_positions(decided, frame_config.position_column)[
                    decided["policy_emit_signal"].to_numpy(copy=False)
                ]
                abstain_count = 0
                candidate_count = int(decided["policy_signal_candidate"].sum())

            metrics = event_metrics_for_emitted_positions(
                y_true=decided[frame_config.target_column].to_numpy(dtype=np.uint8, copy=True),
                positions=_resolve_positions(decided, frame_config.position_column),
                emitted_positions=np.asarray(emitted_positions, dtype=np.int64),
                tolerance=frame_config.event_tolerance_bars,
            )
            trade_metrics = trade_metrics_for_emitted_positions(
                frame=decided,
                emitted_positions=np.asarray(emitted_positions, dtype=np.int64),
                config=frame_config,
            )
            rows.append(
                {
                    "dataset_split": dataset_split,
                    "policy_name": policy_name,
                    "rows_evaluated": int(len(decided)),
                    "candidate_signals": int(candidate_count),
                    "emitted_signals": int(len(emitted_positions)),
                    "coverage_rate": 0.0 if len(decided) == 0 else float(len(emitted_positions) / len(decided)),
                    "candidate_rate": 0.0 if len(decided) == 0 else float(candidate_count / len(decided)),
                    "abstain_count": int(abstain_count),
                    "abstain_rate": 0.0 if len(decided) == 0 else float(abstain_count / len(decided)),
                    "event_precision": float(metrics["event_precision"]),
                    "event_recall": float(metrics["event_recall"]),
                    "event_f05": float(metrics["event_f05"]),
                    "true_events": int(metrics["true_events"]),
                    "matched_events": int(metrics["matched_events"]),
                    "scored_trades": int(trade_metrics["scored_trades"]),
                    "trades_per_week": float(trade_metrics["trades_per_week"]),
                    "gross_expectancy_pips": float(trade_metrics["gross_expectancy_pips"]),
                    "gross_pnl_pips": float(trade_metrics["gross_pnl_pips"]),
                    "post_cost_expectancy_pips": float(trade_metrics["post_cost_expectancy_pips"]),
                    "net_pnl_pips": float(trade_metrics["net_pnl_pips"]),
                }
            )

    return pd.DataFrame(rows)


def attach_forward_trade_outcomes(
    frame: pd.DataFrame,
    *,
    market_frame: pd.DataFrame,
    direction: str,
    config: ThresholdSearchConfig,
) -> pd.DataFrame:
    if direction not in {"long", "short"}:
        raise ValueError(f"Unsupported direction for trade outcomes: {direction!r}")

    if config.position_column not in frame.columns:
        raise ValueError(
            f"Trade outcome attachment requires {config.position_column!r} in the frame."
        )
    if config.entry_price_column not in frame.columns:
        raise ValueError(
            f"Trade outcome attachment requires {config.entry_price_column!r} in the frame."
        )
    if "source_row_idx" not in market_frame.columns or "close" not in market_frame.columns:
        raise ValueError("Market frame must include 'source_row_idx' and 'close' columns.")

    working = frame.copy()
    source_lookup = market_frame.loc[
        :,
        [
            column
            for column in ("source_row_idx", "datetime", "close", config.approx_spread_column)
            if column in market_frame.columns
        ],
    ].copy()
    source_lookup["source_row_idx"] = pd.to_numeric(
        source_lookup["source_row_idx"],
        errors="coerce",
    ).fillna(-1).astype(np.int64)

    exit_lookup = source_lookup.rename(
        columns={
            "source_row_idx": "policy_exit_source_row_idx",
            "datetime": config.exit_datetime_column,
            "close": config.exit_price_column,
            config.approx_spread_column: "policy_exit_approx_spread",
        }
    )

    if config.approx_spread_column not in working.columns and config.approx_spread_column in source_lookup.columns:
        spread_lookup = source_lookup.loc[:, ["source_row_idx", config.approx_spread_column]].copy()
        working = working.merge(
            spread_lookup,
            on="source_row_idx",
            how="left",
            validate="many_to_one",
        )

    working["policy_exit_source_row_idx"] = (
        pd.to_numeric(working[config.position_column], errors="coerce").fillna(-1).astype(np.int64)
        + int(config.label_max_holding_bars)
    )
    working = working.merge(
        exit_lookup,
        on="policy_exit_source_row_idx",
        how="left",
        validate="many_to_one",
    )

    entry_price = pd.to_numeric(working[config.entry_price_column], errors="coerce")
    exit_price = pd.to_numeric(working[config.exit_price_column], errors="coerce")
    direction_sign = 1.0 if direction == "long" else -1.0
    working[config.gross_pnl_column] = direction_sign * (
        (exit_price - entry_price) / float(config.pip_size)
    )

    if config.approx_spread_column in working.columns:
        entry_spread_pips = pd.to_numeric(working[config.approx_spread_column], errors="coerce") / float(config.pip_size)
    else:
        entry_spread_pips = pd.Series(np.nan, index=working.index, dtype=float)
    if "policy_exit_approx_spread" in working.columns:
        exit_spread_pips = pd.to_numeric(working["policy_exit_approx_spread"], errors="coerce") / float(config.pip_size)
    else:
        exit_spread_pips = pd.Series(np.nan, index=working.index, dtype=float)

    if config.session_column in working.columns:
        session_spread_pips = working[config.session_column].map(
            lambda value: estimate_session_spread_pips(value, spread_table=config.session_spread_pips)
        )
        entry_spread_pips = entry_spread_pips.where(entry_spread_pips > 0.0, session_spread_pips)
        exit_spread_pips = exit_spread_pips.where(exit_spread_pips > 0.0, session_spread_pips)
    else:
        default_spread = float(config.session_spread_pips["off_hours"])
        entry_spread_pips = entry_spread_pips.fillna(default_spread)
        exit_spread_pips = exit_spread_pips.fillna(default_spread)

    slippage_pips = float(config.slippage_spread_multiplier) * (entry_spread_pips + exit_spread_pips)
    working[config.total_cost_column] = (
        entry_spread_pips
        + exit_spread_pips
        + slippage_pips
        + float(config.fixed_slippage_pips_per_trade)
        + float(config.commission_pips_per_trade)
    )
    working[config.net_pnl_column] = (
        pd.to_numeric(working[config.gross_pnl_column], errors="coerce")
        - pd.to_numeric(working[config.total_cost_column], errors="coerce")
    )
    return working


def event_metrics_for_emitted_positions(
    *,
    y_true: np.ndarray,
    positions: np.ndarray,
    emitted_positions: np.ndarray,
    tolerance: int,
    beta: float = 0.5,
) -> Dict[str, float]:
    true_zones = group_positive_zones_by_position(y_true, positions)

    matched_true = set()
    matched_events = 0
    for prediction in emitted_positions.tolist():
        match_index = None
        for zone_index, (start, end) in enumerate(true_zones):
            if zone_index in matched_true:
                continue
            if (start - tolerance) <= prediction <= (end + tolerance):
                match_index = zone_index
                break
        if match_index is not None:
            matched_true.add(match_index)
            matched_events += 1

    predicted_count = len(emitted_positions)
    true_count = len(true_zones)
    precision = 0.0 if predicted_count == 0 else matched_events / predicted_count
    recall = 0.0 if true_count == 0 else matched_events / true_count
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    event_f05 = 0.0 if denom <= 0.0 else (1.0 + beta_sq) * precision * recall / denom
    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f05": float(event_f05),
        "true_events": float(true_count),
        "matched_events": float(matched_events),
    }


def prepare_policy_frame(
    frame: pd.DataFrame,
    config: ThresholdSearchConfig,
) -> pd.DataFrame:
    if config.probability_column not in frame.columns:
        raise ValueError(f"Probability column {config.probability_column!r} not found in frame.")
    if config.target_column not in frame.columns:
        raise ValueError(f"Target column {config.target_column!r} not found in frame.")
    if config.composite_column not in frame.columns:
        raise ValueError(f"Composite regime column {config.composite_column!r} not found in frame.")

    working = frame.copy()
    working[config.target_column] = pd.to_numeric(working[config.target_column], errors="coerce")
    working = working.loc[working[config.target_column].isin([0, 1])].copy()
    working[config.probability_column] = pd.to_numeric(working[config.probability_column], errors="coerce")
    working = working.loc[working[config.probability_column].notna()].copy()

    if config.position_column not in working.columns:
        working[config.position_column] = np.arange(len(working), dtype=np.int64)
    else:
        working[config.position_column] = pd.to_numeric(
            working[config.position_column],
            errors="coerce",
        ).fillna(-1).astype(np.int64)

    if config.datetime_column in working.columns:
        working[config.datetime_column] = pd.to_datetime(working[config.datetime_column], errors="coerce")

    if config.year_column in working.columns:
        working[config.year_column] = pd.to_numeric(working[config.year_column], errors="coerce").astype("Int64")
    elif config.datetime_column in working.columns:
        working[config.year_column] = pd.to_datetime(
            working[config.datetime_column],
            errors="coerce",
        ).dt.year.astype("Int64")
    else:
        working[config.year_column] = pd.Series(pd.NA, index=working.index, dtype="Int64")

    return working.sort_values(config.position_column).reset_index(drop=True)


def evaluate_threshold_candidate(
    frame: pd.DataFrame,
    *,
    threshold: float,
    config: ThresholdSearchConfig,
) -> Dict[str, object]:
    prepared = prepare_policy_frame(frame, config)
    positions = _resolve_positions(prepared, config.position_column)
    emitted_positions = np.asarray(
        predictions_to_events_by_position(
            prepared[config.probability_column].to_numpy(dtype=np.float32, copy=True),
            positions,
            float(threshold),
            config.event_cooldown_bars,
        ),
        dtype=np.int64,
    )
    metrics = event_metrics_by_position(
        y_true=prepared[config.target_column].to_numpy(dtype=np.uint8, copy=True),
        y_score=prepared[config.probability_column].to_numpy(dtype=np.float32, copy=True),
        positions=positions,
        threshold=float(threshold),
        tolerance=config.event_tolerance_bars,
        cooldown=config.event_cooldown_bars,
    )
    avg_events_per_month = _average_events_per_month(
        prepared,
        predicted_events=int(metrics["predicted_events"]),
        datetime_column=config.datetime_column,
    )
    trade_metrics = trade_metrics_for_emitted_positions(
        frame=prepared,
        emitted_positions=emitted_positions,
        config=config,
    )
    return {
        "threshold": float(threshold),
        "event_precision": float(metrics["event_precision"]),
        "event_recall": float(metrics["event_recall"]),
        "event_f05": float(metrics["event_f05"]),
        "predicted_events": int(metrics["predicted_events"]),
        "true_events": int(metrics["true_events"]),
        "matched_events": int(metrics["matched_events"]),
        "avg_events_per_month": float(avg_events_per_month),
        "trades_per_week": float(trade_metrics["trades_per_week"]),
        "scored_trades": int(trade_metrics["scored_trades"]),
        "gross_expectancy_pips": float(trade_metrics["gross_expectancy_pips"]),
        "gross_pnl_pips": float(trade_metrics["gross_pnl_pips"]),
        "post_cost_expectancy_pips": float(trade_metrics["post_cost_expectancy_pips"]),
        "net_pnl_pips": float(trade_metrics["net_pnl_pips"]),
    }


def _search_best_threshold(
    frame: pd.DataFrame,
    *,
    config: ThresholdSearchConfig,
    fallback_threshold: float,
) -> Dict[str, object]:
    prepared = prepare_policy_frame(frame, config)
    years = sorted(int(value) for value in prepared[config.year_column].dropna().unique())
    use_leave_one_year_out = len(years) >= 2

    candidate_results: List[Dict[str, object]] = []
    seen_thresholds = set()
    for threshold in (float(fallback_threshold), *[float(value) for value in config.threshold_grid]):
        if threshold in seen_thresholds:
            continue
        seen_thresholds.add(threshold)
        candidate = evaluate_threshold_candidate(prepared, threshold=threshold, config=config)
        if candidate["avg_events_per_month"] < config.min_events_per_month:
            continue
        candidate_results.append(candidate)

    if not candidate_results:
        best_result = evaluate_threshold_candidate(prepared, threshold=fallback_threshold, config=config)
        best_result["selection_score"] = float(best_result["event_f05"])
        best_result["normalized_expectancy"] = 0.0
        best_result["used_leave_one_year_out"] = bool(use_leave_one_year_out)
        return best_result

    best_result: Dict[str, object] | None = None
    normalized_expectancies = _normalize_expectancy_scores(candidate_results)

    for candidate, normalized_expectancy in zip(candidate_results, normalized_expectancies):
        candidate["normalized_expectancy"] = float(normalized_expectancy)
        candidate["used_leave_one_year_out"] = bool(use_leave_one_year_out)
        if use_leave_one_year_out:
            yearly_scores = []
            for year in years:
                held_out = prepared.loc[prepared[config.year_column] == year].copy()
                if held_out.empty:
                    continue
                yearly_results = [
                    evaluate_threshold_candidate(
                        held_out,
                        threshold=float(result["threshold"]),
                        config=config,
                    )
                    for result in candidate_results
                ]
                yearly_expectancy = _normalize_expectancy_scores(yearly_results)
                yearly_by_threshold = {
                    float(result["threshold"]): float(score)
                    for result, score in zip(yearly_results, yearly_expectancy)
                }
                yearly_result = next(
                    result
                    for result in yearly_results
                    if float(result["threshold"]) == float(candidate["threshold"])
                )
                yearly_scores.append(
                    (0.60 * float(yearly_result["event_f05"]))
                    + (0.40 * yearly_by_threshold[float(candidate["threshold"])])
                )
            selection_score = (
                float(np.mean(yearly_scores))
                if yearly_scores
                else (0.60 * float(candidate["event_f05"])) + (0.40 * float(normalized_expectancy))
            )
        else:
            selection_score = (0.60 * float(candidate["event_f05"])) + (0.40 * float(normalized_expectancy))

        candidate["selection_score"] = float(selection_score)
        candidate_score = (
            float(selection_score),
            float(candidate["post_cost_expectancy_pips"]),
            float(candidate["event_precision"]),
            -abs(candidate["predicted_events"] - candidate["true_events"]),
        )
        if best_result is None:
            best_result = candidate
            continue
        best_score = (
            float(best_result["selection_score"]),
            float(best_result["post_cost_expectancy_pips"]),
            float(best_result["event_precision"]),
            -abs(best_result["predicted_events"] - best_result["true_events"]),
        )
        if candidate_score > best_score:
            best_result = candidate

    assert best_result is not None
    return best_result


def _average_events_per_month(
    frame: pd.DataFrame,
    *,
    predicted_events: int,
    datetime_column: str,
) -> float:
    if predicted_events <= 0:
        return 0.0
    if datetime_column not in frame.columns:
        return float(predicted_events)
    timestamps = pd.to_datetime(frame[datetime_column], errors="coerce").dropna()
    if timestamps.empty:
        return float(predicted_events)
    month_count = max(int(timestamps.dt.to_period("M").nunique()), 1)
    return float(predicted_events / month_count)


def _average_events_per_week(
    frame: pd.DataFrame,
    *,
    predicted_events: int,
    datetime_column: str,
) -> float:
    if predicted_events <= 0:
        return 0.0
    if datetime_column not in frame.columns:
        return float(predicted_events)
    timestamps = pd.to_datetime(frame[datetime_column], errors="coerce").dropna()
    if timestamps.empty:
        return float(predicted_events)
    week_count = max(int(timestamps.dt.to_period("W").nunique()), 1)
    return float(predicted_events / week_count)


def _resolve_positions(frame: pd.DataFrame, position_column: str) -> np.ndarray:
    if position_column not in frame.columns:
        return np.arange(len(frame), dtype=np.int64)
    return pd.to_numeric(frame[position_column], errors="coerce").fillna(-1).astype(np.int64).to_numpy(copy=True)


def _resolve_frame_config(
    frame: pd.DataFrame,
    config: ThresholdSearchConfig,
) -> ThresholdSearchConfig:
    if config.probability_column in frame.columns:
        return config

    for candidate in ("oof_calibrated_probability", "calibrated_probability", "oof_raw_probability", "raw_probability"):
        if candidate in frame.columns:
            return ThresholdSearchConfig(
                probability_column=candidate,
                global_threshold=config.global_threshold,
                composite_column=config.composite_column,
                session_column=config.session_column,
                target_column=config.target_column,
                year_column=config.year_column,
                datetime_column=config.datetime_column,
                position_column=config.position_column,
                entry_price_column=config.entry_price_column,
                exit_price_column=config.exit_price_column,
                exit_datetime_column=config.exit_datetime_column,
                gross_pnl_column=config.gross_pnl_column,
                total_cost_column=config.total_cost_column,
                net_pnl_column=config.net_pnl_column,
                threshold_grid=config.threshold_grid,
                event_tolerance_bars=config.event_tolerance_bars,
                event_cooldown_bars=config.event_cooldown_bars,
                min_positive_events=config.min_positive_events,
                min_events_per_month=config.min_events_per_month,
                label_max_holding_bars=config.label_max_holding_bars,
                pip_size=config.pip_size,
                approx_spread_column=config.approx_spread_column,
                slippage_spread_multiplier=config.slippage_spread_multiplier,
                fixed_slippage_pips_per_trade=config.fixed_slippage_pips_per_trade,
                commission_pips_per_trade=config.commission_pips_per_trade,
                session_spread_pips=config.session_spread_pips,
            )

    available = ", ".join(sorted(frame.columns))
    raise ValueError(
        f"Could not resolve a probability column for threshold evaluation from frame columns: {available}"
    )


def trade_metrics_for_emitted_positions(
    *,
    frame: pd.DataFrame,
    emitted_positions: np.ndarray,
    config: ThresholdSearchConfig,
) -> Dict[str, float]:
    positions = _resolve_positions(frame, config.position_column)
    if len(emitted_positions) == 0:
        return {
            "scored_trades": 0.0,
            "trades_per_week": 0.0,
            "gross_expectancy_pips": 0.0,
            "gross_pnl_pips": 0.0,
            "post_cost_expectancy_pips": 0.0,
            "net_pnl_pips": 0.0,
        }

    emitted_mask = np.isin(positions, emitted_positions)
    gross_pnl = (
        pd.to_numeric(frame.get(config.gross_pnl_column, pd.Series(np.nan, index=frame.index)), errors="coerce")
        .loc[emitted_mask]
        .dropna()
    )
    net_pnl = (
        pd.to_numeric(frame.get(config.net_pnl_column, pd.Series(np.nan, index=frame.index)), errors="coerce")
        .loc[emitted_mask]
        .dropna()
    )
    scored_trades = int(net_pnl.shape[0] if not net_pnl.empty else gross_pnl.shape[0])
    if scored_trades <= 0:
        return {
            "scored_trades": 0.0,
            "trades_per_week": _average_events_per_week(
                frame,
                predicted_events=int(len(emitted_positions)),
                datetime_column=config.datetime_column,
            ),
            "gross_expectancy_pips": 0.0,
            "gross_pnl_pips": 0.0,
            "post_cost_expectancy_pips": 0.0,
            "net_pnl_pips": 0.0,
        }

    return {
        "scored_trades": float(scored_trades),
        "trades_per_week": float(
            _average_events_per_week(
                frame,
                predicted_events=scored_trades,
                datetime_column=config.datetime_column,
            )
        ),
        "gross_expectancy_pips": float(gross_pnl.mean()) if not gross_pnl.empty else 0.0,
        "gross_pnl_pips": float(gross_pnl.sum()) if not gross_pnl.empty else 0.0,
        "post_cost_expectancy_pips": float(net_pnl.mean()) if not net_pnl.empty else 0.0,
        "net_pnl_pips": float(net_pnl.sum()) if not net_pnl.empty else 0.0,
    }


def _normalize_expectancy_scores(results: Sequence[Mapping[str, object]]) -> List[float]:
    if not results:
        return []
    expectancy = np.asarray(
        [float(result.get("post_cost_expectancy_pips", 0.0)) for result in results],
        dtype=np.float64,
    )
    max_abs = float(np.max(np.abs(expectancy))) if expectancy.size else 0.0
    if max_abs <= 0.0:
        return [0.5 for _ in results]
    scaled = np.clip(expectancy / max_abs, -1.0, 1.0)
    return (0.5 + (0.5 * scaled)).astype(float).tolist()
