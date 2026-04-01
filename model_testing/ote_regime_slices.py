from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


DEFAULT_SLICE_FAMILIES = (
    "composite_regime",
    "trend_regime",
    "vol_regime",
    "session_regime",
    "stress_regime",
    "year",
)
DEFAULT_THRESHOLD_GRID = tuple(np.round(np.arange(0.40, 0.901, 0.05), 2).tolist())


@dataclass(frozen=True)
class RegimeSliceConfig:
    dataset_split: str
    probability_column: str
    threshold: float
    event_tolerance_bars: int = 2
    event_cooldown_bars: int = 4
    min_positive_events_for_threshold: int = 50
    bootstrap_iterations: int = 200
    bootstrap_confidence_level: float = 0.95
    random_seed: int = 42
    target_column: str = "target"
    sample_weight_column: str = "sample_weight"
    position_column: str = "source_row_idx"
    threshold_grid: Tuple[float, ...] = field(default_factory=lambda: DEFAULT_THRESHOLD_GRID)
    slice_families: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_SLICE_FAMILIES)


def build_regime_slice_report(
    frame: pd.DataFrame,
    *,
    model_id: str,
    direction: str,
    config: RegimeSliceConfig,
    backend: str | None = None,
    role: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    scored = _prepare_scored_frame(frame, config)
    if scored.empty:
        return pd.DataFrame()

    report_rows: List[Dict[str, object]] = []
    for family in config.slice_families:
        if family not in scored.columns:
            continue

        family_frame = scored.loc[scored[family].notna()].copy()
        if family_frame.empty:
            continue

        for regime_key, bucket in family_frame.groupby(family, sort=True):
            bucket = bucket.sort_values(config.position_column).reset_index(drop=True)
            positions = bucket[config.position_column].to_numpy(dtype=np.int64, copy=True)
            y_true = bucket[config.target_column].to_numpy(dtype=np.uint8, copy=True)
            y_score = bucket[config.probability_column].to_numpy(dtype=np.float32, copy=True)
            sample_weight = (
                bucket[config.sample_weight_column].to_numpy(dtype=np.float32, copy=True)
                if config.sample_weight_column in bucket.columns
                else None
            )

            true_zones = group_positive_zones_by_position(y_true, positions)
            current_metrics = event_metrics_by_position(
                y_true=y_true,
                y_score=y_score,
                positions=positions,
                threshold=config.threshold,
                tolerance=config.event_tolerance_bars,
                cooldown=config.event_cooldown_bars,
            )
            threshold_data_sufficient = len(true_zones) >= config.min_positive_events_for_threshold
            if threshold_data_sufficient:
                optimal_threshold, optimal_metrics = select_optimal_threshold_by_position(
                    y_true=y_true,
                    y_score=y_score,
                    positions=positions,
                    tolerance=config.event_tolerance_bars,
                    cooldown=config.event_cooldown_bars,
                    thresholds=config.threshold_grid,
                    fallback_threshold=config.threshold,
                )
            else:
                optimal_threshold = float(config.threshold)
                optimal_metrics = dict(current_metrics)

            ap = safe_average_precision(y_true, y_score, sample_weight=sample_weight)
            ap_ci_lower, ap_ci_upper = bootstrap_average_precision_ci(
                y_true=y_true,
                y_score=y_score,
                sample_weight=sample_weight,
                iterations=config.bootstrap_iterations,
                confidence_level=config.bootstrap_confidence_level,
                seed=config.random_seed
                + _stable_bucket_seed(model_id, direction, config.dataset_split, family, str(regime_key)),
            )

            report_rows.append(
                {
                    "model_id": model_id,
                    "direction": direction,
                    "backend": backend,
                    "role": role,
                    "status": status,
                    "dataset_split": config.dataset_split,
                    "regime_family": family,
                    "regime_key": str(regime_key),
                    "rows_in_bucket": int(len(bucket)),
                    "n_positive": int(len(true_zones)),
                    "n_negative": int(np.sum(y_true == 0)),
                    "positive_row_count": int(np.sum(y_true == 1)),
                    "sample_weight_sum": float(np.sum(sample_weight))
                    if sample_weight is not None
                    else float(len(bucket)),
                    "threshold_used": float(config.threshold),
                    "ap": float(ap),
                    "ap_ci_lower": float(ap_ci_lower),
                    "ap_ci_upper": float(ap_ci_upper),
                    "event_precision": float(current_metrics["event_precision"]),
                    "event_recall": float(current_metrics["event_recall"]),
                    "event_f05": float(current_metrics["event_f05"]),
                    "predicted_events": int(current_metrics["predicted_events"]),
                    "true_events": int(current_metrics["true_events"]),
                    "matched_events": int(current_metrics["matched_events"]),
                    "optimal_threshold": float(optimal_threshold),
                    "optimal_event_precision": float(optimal_metrics["event_precision"]),
                    "optimal_event_recall": float(optimal_metrics["event_recall"]),
                    "optimal_event_f05": float(optimal_metrics["event_f05"]),
                    "optimal_predicted_events": int(optimal_metrics["predicted_events"]),
                    "optimal_matched_events": int(optimal_metrics["matched_events"]),
                    "threshold_data_sufficient": bool(threshold_data_sufficient),
                    "insufficient_data_for_regime_threshold": bool(not threshold_data_sufficient),
                }
            )

    return pd.DataFrame(report_rows)


def build_regime_winner_table(
    report: pd.DataFrame,
    *,
    regime_family: str = "composite_regime",
    metric_column: str = "ap",
) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame()

    required = {"dataset_split", "direction", "regime_family", "regime_key", "model_id", metric_column}
    missing = required - set(report.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"Winner table requires report columns: {missing_csv}")

    filtered = report.loc[report["regime_family"] == regime_family].copy()
    if filtered.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    group_columns = ["dataset_split", "direction", "regime_family", "regime_key"]
    for group_key, bucket in filtered.groupby(group_columns, sort=True):
        bucket = bucket.sort_values(
            [metric_column, "ap_ci_lower", "event_f05"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        winner = bucket.iloc[0]
        runner_up = bucket.iloc[1] if len(bucket) > 1 else None
        confidence_gap = (
            float(winner["ap_ci_lower"] - runner_up["ap_ci_upper"])
            if runner_up is not None
            else np.nan
        )
        rows.append(
            {
                "dataset_split": group_key[0],
                "direction": group_key[1],
                "regime_family": group_key[2],
                "regime_key": group_key[3],
                "models_compared": int(len(bucket)),
                "winner_model_id": winner["model_id"],
                "winner_backend": winner.get("backend"),
                "winner_metric": float(winner[metric_column]),
                "winner_ap_ci_lower": float(winner["ap_ci_lower"]),
                "winner_ap_ci_upper": float(winner["ap_ci_upper"]),
                "winner_event_f05": float(winner["event_f05"]),
                "runner_up_model_id": None if runner_up is None else runner_up["model_id"],
                "runner_up_metric": np.nan if runner_up is None else float(runner_up[metric_column]),
                "runner_up_ap_ci_lower": np.nan if runner_up is None else float(runner_up["ap_ci_lower"]),
                "runner_up_ap_ci_upper": np.nan if runner_up is None else float(runner_up["ap_ci_upper"]),
                "confidence_gap": confidence_gap,
                "winner_confident": bool(runner_up is not None and confidence_gap > 0.0),
            }
        )

    return pd.DataFrame(rows)


def resolve_probability_column(
    frame: pd.DataFrame,
    *,
    dataset_split: str | None = None,
) -> str:
    preferred: Sequence[str]
    if dataset_split == "oof":
        preferred = ("oof_calibrated_probability", "oof_raw_probability")
    elif dataset_split == "test":
        preferred = ("calibrated_probability", "raw_probability")
    else:
        preferred = (
            "oof_calibrated_probability",
            "calibrated_probability",
            "oof_raw_probability",
            "raw_probability",
        )

    for column in preferred:
        if column in frame.columns:
            return column

    available = ", ".join(sorted(frame.columns))
    raise ValueError(f"Could not resolve a probability column from frame columns: {available}")


def safe_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.0
    return float(average_precision_score(y_true, y_score, sample_weight=sample_weight))


def bootstrap_average_precision_ci(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> Tuple[float, float]:
    point_estimate = safe_average_precision(y_true, y_score, sample_weight=sample_weight)
    if iterations <= 0 or len(y_true) == 0:
        return point_estimate, point_estimate

    rng = np.random.default_rng(seed)
    bootstrap_scores = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.integers(0, len(y_true), size=len(y_true))
        sampled_weight = None if sample_weight is None else sample_weight[sampled]
        bootstrap_scores[index] = safe_average_precision(
            y_true[sampled],
            y_score[sampled],
            sample_weight=sampled_weight,
        )

    alpha = max(0.0, min(1.0, (1.0 - confidence_level) / 2.0))
    lower = float(np.quantile(bootstrap_scores, alpha))
    upper = float(np.quantile(bootstrap_scores, 1.0 - alpha))
    return lower, upper


def group_positive_zones_by_position(
    y_true: np.ndarray,
    positions: np.ndarray,
) -> List[Tuple[int, int]]:
    zones: List[Tuple[int, int]] = []
    zone_start: Optional[int] = None
    previous_position: Optional[int] = None

    for value, position in zip(y_true, positions):
        current_position = int(position)
        if value == 1:
            if zone_start is None:
                zone_start = current_position
            elif previous_position is not None and current_position != previous_position + 1:
                zones.append((zone_start, int(previous_position)))
                zone_start = current_position
        elif zone_start is not None and previous_position is not None:
            zones.append((zone_start, int(previous_position)))
            zone_start = None
        previous_position = current_position

    if zone_start is not None and previous_position is not None:
        zones.append((zone_start, int(previous_position)))

    return zones


def predictions_to_events_by_position(
    y_score: np.ndarray,
    positions: np.ndarray,
    threshold: float,
    cooldown: int,
) -> List[int]:
    accepted: List[int] = []
    candidate_positions = positions[y_score >= threshold]
    for position in candidate_positions:
        current_position = int(position)
        if not accepted or current_position - accepted[-1] > cooldown:
            accepted.append(current_position)
    return accepted


def event_metrics_by_position(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    positions: np.ndarray,
    threshold: float,
    tolerance: int,
    cooldown: int,
    beta: float = 0.5,
) -> Dict[str, float]:
    true_zones = group_positive_zones_by_position(y_true, positions)
    predicted_events = predictions_to_events_by_position(y_score, positions, threshold, cooldown)

    matched_true = set()
    matched_events = 0
    for prediction in predicted_events:
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

    predicted_count = len(predicted_events)
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
        "predicted_events": float(predicted_count),
        "true_events": float(true_count),
        "matched_events": float(matched_events),
    }


def select_optimal_threshold_by_position(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    positions: np.ndarray,
    tolerance: int,
    cooldown: int,
    thresholds: Iterable[float],
    fallback_threshold: float,
) -> Tuple[float, Dict[str, float]]:
    best_threshold = float(fallback_threshold)
    best_metrics = event_metrics_by_position(
        y_true=y_true,
        y_score=y_score,
        positions=positions,
        threshold=best_threshold,
        tolerance=tolerance,
        cooldown=cooldown,
    )
    best_score = (
        best_metrics["event_f05"],
        best_metrics["event_precision"],
        -abs(best_metrics["predicted_events"] - best_metrics["true_events"]),
    )

    for threshold in thresholds:
        candidate_metrics = event_metrics_by_position(
            y_true=y_true,
            y_score=y_score,
            positions=positions,
            threshold=float(threshold),
            tolerance=tolerance,
            cooldown=cooldown,
        )
        candidate_score = (
            candidate_metrics["event_f05"],
            candidate_metrics["event_precision"],
            -abs(candidate_metrics["predicted_events"] - candidate_metrics["true_events"]),
        )
        if candidate_score > best_score:
            best_threshold = float(threshold)
            best_metrics = candidate_metrics
            best_score = candidate_score

    return best_threshold, best_metrics


def _prepare_scored_frame(frame: pd.DataFrame, config: RegimeSliceConfig) -> pd.DataFrame:
    if config.probability_column not in frame.columns:
        raise ValueError(f"Probability column {config.probability_column!r} not found in frame.")
    if config.target_column not in frame.columns:
        raise ValueError(f"Target column {config.target_column!r} not found in frame.")

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
        ).astype(np.int64)

    if config.sample_weight_column in working.columns:
        working[config.sample_weight_column] = pd.to_numeric(
            working[config.sample_weight_column],
            errors="coerce",
        ).fillna(1.0)

    return working.sort_values(config.position_column).reset_index(drop=True)


def _stable_bucket_seed(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], byteorder="big", signed=False) % 100000
