"""
OTE model training pipeline for prepared EURUSD datasets.

This module is designed around the current repository constraints:
- Prepared datasets live under ``data/prepared/<dataset>/<target_name>``
- TensorFlow/PyTorch are not available in the local environment
- XGBoost and Optuna are available

The implementation therefore uses a research-aligned tabular time-series
pipeline with:
- causal sparse-window lag features
- robust scaling fit on each training fold only
- custom focal loss for extreme class imbalance
- purged walk-forward cross-validation
- progressive boosting phases
- probability calibration and event-level scoring

The default targets are ``long_ote`` and ``short_ote``.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler


LEAKAGE_PATTERNS = (
    "target_profit",
    "future",
    "lookahead",
    "forward_",
    "_forward",
    "pnl",
    "mfe_",
    "_mfe",
    "mae_",
    "_mae",
    "take_profit",
    "tp_hit",
    "sl_hit",
    "exit_signal",
)

EPS = 1e-9


@dataclass
class OTETrainingConfig:
    prepared_root: str = "data/prepared/eurusd_5min_ote_2000_v2"
    targets: List[str] = field(default_factory=lambda: ["long_ote", "short_ote"])
    output_root: str = "models/ote_xgboost"
    random_seed: int = 42
    n_trials: int = 20
    cv_splits: int = 3
    min_train_rows: int = 350
    min_val_rows: int = 160
    min_positive_per_fold: int = 3
    max_loaded_features: int = 160
    top_feature_min: int = 24
    top_feature_max: int = 96
    top_feature_step: int = 8
    window_min: int = 8
    window_max: int = 40
    lag_count_min: int = 4
    lag_count_max: int = 8
    delta_feature_cap: int = 16
    scale_quantile_low: float = 5.0
    scale_quantile_high: float = 95.0
    scale_clip: float = 8.0
    purge_additional_bars: int = 12
    threshold_grid_size: int = 31
    event_tolerance_bars: int = 2
    event_cooldown_bars: int = 4
    tuning_negative_ratio: int = 8
    use_balanced_tuning_sample: bool = True
    calibration_method: str = "platt"
    final_refit_on_dev: bool = True
    verbosity: int = 1


@dataclass
class PreparedTargetDataset:
    target_name: str
    prepared_dir: Path
    feature_names: List[str]
    ranked_features: List[str]
    report: Dict[str, object]
    X_train: np.ndarray
    y_train: np.ndarray
    w_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    w_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    w_test: np.ndarray

    @property
    def dev_X(self) -> np.ndarray:
        return np.concatenate([self.X_train, self.X_val], axis=0)

    @property
    def dev_y(self) -> np.ndarray:
        return np.concatenate([self.y_train, self.y_val], axis=0)

    @property
    def dev_w(self) -> np.ndarray:
        return np.concatenate([self.w_train, self.w_val], axis=0)


@dataclass
class TrialArtifacts:
    params: Dict[str, float]
    oof_pred: np.ndarray
    fold_metrics: List[Dict[str, float]]
    selected_feature_names: List[str]
    lag_steps: List[int]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def safe_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))


def blocked_for_leakage(feature_name: str) -> bool:
    lower = feature_name.lower()
    return any(pattern in lower for pattern in LEAKAGE_PATTERNS)


def load_ranked_features(
    prepared_dir: Path,
    max_loaded_features: int,
) -> List[str]:
    features_path = prepared_dir / "features.json"
    ranking_path = prepared_dir / "feature_importance.csv"

    with features_path.open("r", encoding="utf-8") as handle:
        raw_features = json.load(handle)["features"]

    raw_features = [feature for feature in raw_features if not blocked_for_leakage(feature)]

    if ranking_path.exists():
        ranking_df = pd.read_csv(ranking_path)
        ranking_df = ranking_df[~ranking_df["feature"].map(blocked_for_leakage)]
        ranked = [
            feature
            for feature in ranking_df.sort_values("composite_score", ascending=False)["feature"].tolist()
            if feature in raw_features
        ]
    else:
        ranked = list(raw_features)

    seen = set()
    deduped = []
    for feature in ranked:
        if feature in seen:
            continue
        seen.add(feature)
        deduped.append(feature)
        if len(deduped) >= max_loaded_features:
            break

    return deduped


def load_prepared_target_dataset(
    prepared_root: Path,
    target_name: str,
    config: OTETrainingConfig,
) -> PreparedTargetDataset:
    prepared_dir = prepared_root / target_name
    report_path = prepared_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    ranked_features = load_ranked_features(prepared_dir, config.max_loaded_features)
    usecols = ranked_features + ["target", "sample_weight"]

    def read_split(split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = pd.read_csv(prepared_dir / f"{split_name}.csv", usecols=usecols)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        feature_frame = frame[ranked_features].astype(np.float32).fillna(0.0)
        target = frame["target"].astype(np.uint8).to_numpy(copy=True)
        weight = frame["sample_weight"].astype(np.float32).fillna(1.0).to_numpy(copy=True)
        return (
            np.ascontiguousarray(feature_frame.to_numpy(copy=True), dtype=np.float32),
            np.ascontiguousarray(target, dtype=np.uint8),
            np.ascontiguousarray(weight, dtype=np.float32),
        )

    X_train, y_train, w_train = read_split("train")
    X_val, y_val, w_val = read_split("val")
    X_test, y_test, w_test = read_split("test")

    return PreparedTargetDataset(
        target_name=target_name,
        prepared_dir=prepared_dir,
        feature_names=ranked_features,
        ranked_features=ranked_features,
        report=report,
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val,
        w_val=w_val,
        X_test=X_test,
        y_test=y_test,
        w_test=w_test,
    )


def make_lag_steps(window_size: int, lag_count: int) -> List[int]:
    if window_size <= 1:
        return [0]
    if lag_count >= window_size:
        return list(range(window_size))
    if lag_count <= 1:
        return [0]

    raw = np.geomspace(1, window_size - 1, num=max(lag_count - 1, 1))
    lag_steps = sorted(set([0] + [int(round(value)) for value in raw] + [window_size - 1]))

    candidate = 1
    while len(lag_steps) < lag_count and candidate < window_size:
        lag_steps.append(candidate)
        lag_steps = sorted(set(lag_steps))
        candidate += 1

    while len(lag_steps) > lag_count:
        lag_steps.pop(-2)

    return lag_steps


def make_windowed_feature_names(
    selected_feature_names: Sequence[str],
    lag_steps: Sequence[int],
    delta_feature_count: int,
) -> List[str]:
    names: List[str] = []
    for lag in lag_steps:
        suffix = f"lag_{lag}"
        names.extend([f"{feature}__{suffix}" for feature in selected_feature_names])

    delta_feature_names = list(selected_feature_names[:delta_feature_count])
    current_suffix = lag_steps[0]
    for lag in lag_steps[1:]:
        names.extend(
            [f"{feature}__delta_lag_{current_suffix}_vs_{lag}" for feature in delta_feature_names]
        )
    return names


def build_sparse_lag_view(
    matrix: np.ndarray,
    lag_steps: Sequence[int],
    delta_feature_count: int = 0,
) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")

    max_lag = max(lag_steps)
    if matrix.shape[0] <= max_lag:
        raise ValueError(
            f"Need more than {max_lag} rows to build lagged features, got {matrix.shape[0]}."
        )

    rows = matrix.shape[0] - max_lag
    parts = [matrix[max_lag - lag : max_lag - lag + rows] for lag in lag_steps]

    if delta_feature_count > 0 and len(lag_steps) > 1:
        delta_count = min(delta_feature_count, matrix.shape[1])
        current = matrix[max_lag : max_lag + rows, :delta_count]
        for lag in lag_steps[1:]:
            lagged = matrix[max_lag - lag : max_lag - lag + rows, :delta_count]
            parts.append(current - lagged)

    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


def fit_scaler(train_matrix: np.ndarray, config: OTETrainingConfig) -> RobustScaler:
    scaler = RobustScaler(
        quantile_range=(config.scale_quantile_low, config.scale_quantile_high)
    )
    scaler.fit(train_matrix)
    return scaler


def transform_matrix(
    scaler: RobustScaler,
    matrix: np.ndarray,
    clip_value: float,
) -> np.ndarray:
    scaled = scaler.transform(matrix).astype(np.float32, copy=False)
    if clip_value > 0:
        np.clip(scaled, -clip_value, clip_value, out=scaled)
    return np.ascontiguousarray(scaled, dtype=np.float32)


def build_near_positive_mask(y: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.zeros_like(y, dtype=bool)
    kernel = np.ones((radius * 2) + 1, dtype=np.int16)
    touched = np.convolve(y.astype(np.int16), kernel, mode="same")
    return touched > 0


def apply_hard_negative_weights(
    y: np.ndarray,
    weights: np.ndarray,
    radius: int,
    multiplier: float,
) -> np.ndarray:
    adjusted = weights.astype(np.float32, copy=True)
    if radius <= 0 or multiplier <= 1.0:
        return adjusted
    mask = build_near_positive_mask(y, radius)
    hard_negative_mask = np.logical_and(mask, y == 0)
    adjusted[hard_negative_mask] *= multiplier
    return adjusted


def sample_balanced_training_indices(
    y: np.ndarray,
    hard_negative_mask: np.ndarray,
    negative_ratio: int,
    seed: int,
) -> np.ndarray:
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if len(positives) == 0 or len(negatives) == 0:
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    hard_negatives = negatives[hard_negative_mask[negatives]]
    easy_negatives = negatives[~hard_negative_mask[negatives]]

    target_negative_count = max(len(positives) * negative_ratio, len(hard_negatives))
    additional_needed = max(target_negative_count - len(hard_negatives), 0)
    if additional_needed > 0 and len(easy_negatives) > 0:
        sampled_easy = rng.choice(
            easy_negatives,
            size=min(additional_needed, len(easy_negatives)),
            replace=False,
        )
    else:
        sampled_easy = np.empty(0, dtype=np.int64)

    combined = np.concatenate([positives, hard_negatives, sampled_easy])
    return np.sort(np.unique(combined))


def group_positive_zones(y_true: np.ndarray) -> List[Tuple[int, int]]:
    zones: List[Tuple[int, int]] = []
    in_zone = False
    start = 0
    for index, value in enumerate(y_true):
        if value == 1 and not in_zone:
            start = index
            in_zone = True
        elif value == 0 and in_zone:
            zones.append((start, index - 1))
            in_zone = False
    if in_zone:
        zones.append((start, len(y_true) - 1))
    return zones


def enforce_cooldown(indices: Iterable[int], cooldown: int) -> List[int]:
    accepted: List[int] = []
    for index in sorted(indices):
        if not accepted or index - accepted[-1] > cooldown:
            accepted.append(index)
    return accepted


def predictions_to_events(
    y_score: np.ndarray,
    threshold: float,
    cooldown: int,
) -> List[int]:
    raw_indices = np.flatnonzero(y_score >= threshold).tolist()
    return enforce_cooldown(raw_indices, cooldown)


def precision_recall_fbeta(
    precision: float,
    recall: float,
    beta: float,
) -> float:
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


def event_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    tolerance: int,
    cooldown: int,
    beta: float = 0.5,
) -> Dict[str, float]:
    true_zones = group_positive_zones(y_true)
    predicted_events = predictions_to_events(y_score, threshold, cooldown)

    matched_true = set()
    true_positives = 0

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
            true_positives += 1

    predicted_count = len(predicted_events)
    true_count = len(true_zones)
    precision = 0.0 if predicted_count == 0 else true_positives / predicted_count
    recall = 0.0 if true_count == 0 else true_positives / true_count
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    fbeta = precision_recall_fbeta(precision, recall, beta=beta)

    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "event_fbeta_0_5": fbeta,
        "predicted_events": float(predicted_count),
        "true_events": float(true_count),
        "matched_events": float(true_positives),
    }


def select_operating_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tolerance: int,
    cooldown: int,
    grid_size: int,
) -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.05, 0.95, num=grid_size)
    best_threshold = 0.5
    best_metrics = {
        "event_precision": 0.0,
        "event_recall": 0.0,
        "event_f1": 0.0,
        "event_fbeta_0_5": 0.0,
        "predicted_events": 0.0,
        "true_events": 0.0,
        "matched_events": 0.0,
    }

    for threshold in thresholds:
        metrics = event_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=float(threshold),
            tolerance=tolerance,
            cooldown=cooldown,
            beta=0.5,
        )
        score = (
            metrics["event_fbeta_0_5"],
            metrics["event_precision"],
            -abs(metrics["predicted_events"] - metrics["true_events"]),
        )
        best_score = (
            best_metrics["event_fbeta_0_5"],
            best_metrics["event_precision"],
            -abs(best_metrics["predicted_events"] - best_metrics["true_events"]),
        )
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


class PurgedWalkForwardSplitter:
    def __init__(
        self,
        n_splits: int,
        min_train_rows: int,
        min_val_rows: int,
        purge_bars: int,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_rows = min_train_rows
        self.min_val_rows = min_val_rows
        self.purge_bars = purge_bars

    def split(self, n_rows: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        if n_rows < (self.min_train_rows + self.min_val_rows + self.purge_bars):
            raise ValueError(
                f"Not enough rows ({n_rows}) for min_train={self.min_train_rows}, "
                f"min_val={self.min_val_rows}, purge={self.purge_bars}."
            )

        usable_after_train = n_rows - self.min_train_rows - self.purge_bars
        val_rows = max(self.min_val_rows, usable_after_train // self.n_splits)

        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        train_end = self.min_train_rows
        while len(folds) < self.n_splits:
            val_start = train_end + self.purge_bars
            val_end = min(val_start + val_rows, n_rows)
            if (val_end - val_start) < self.min_val_rows:
                break
            train_idx = np.arange(0, train_end, dtype=np.int64)
            val_idx = np.arange(val_start, val_end, dtype=np.int64)
            folds.append((train_idx, val_idx))
            if val_end >= n_rows:
                break
            train_end = val_end

        if len(folds) < 2:
            raise ValueError(
                f"Unable to create enough temporal folds. Generated {len(folds)} folds for {n_rows} rows."
            )
        return folds


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output.astype(np.float64)


def focal_loss_binary(
    y_true: np.ndarray,
    logits: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    probs = np.clip(sigmoid(logits), EPS, 1.0 - EPS)
    y_true = y_true.astype(np.float64)
    pos_term = -alpha * np.power(1.0 - probs, gamma) * np.log(probs)
    neg_term = -(1.0 - alpha) * np.power(probs, gamma) * np.log(1.0 - probs)
    return np.where(y_true == 1.0, pos_term, neg_term)


def make_focal_objective(alpha: float, gamma: float):
    def focal_objective(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        labels = dtrain.get_label().astype(np.float64)
        weights = dtrain.get_weight()
        if weights.size == 0:
            weights = np.ones_like(labels, dtype=np.float64)
        else:
            weights = weights.astype(np.float64)

        probs = np.clip(sigmoid(predt), EPS, 1.0 - EPS)
        grad = np.empty_like(probs, dtype=np.float64)
        hess = np.empty_like(probs, dtype=np.float64)

        pos_mask = labels == 1.0
        neg_mask = ~pos_mask

        if np.any(pos_mask):
            p = probs[pos_mask]
            b = (gamma * p * np.log(p)) - (1.0 - p)
            db_dp = (gamma * np.log(p)) + gamma + 1.0
            grad_pos = alpha * np.power(1.0 - p, gamma) * b
            hess_pos = alpha * p * np.power(1.0 - p, gamma) * (
                (-gamma * b) + ((1.0 - p) * db_dp)
            )
            grad[pos_mask] = grad_pos
            hess[pos_mask] = hess_pos

        if np.any(neg_mask):
            p = probs[neg_mask]
            b = p - (gamma * (1.0 - p) * np.log(1.0 - p))
            db_dp = 1.0 + gamma + (gamma * np.log(1.0 - p))
            grad_neg = (1.0 - alpha) * np.power(p, gamma) * b
            hess_neg = (1.0 - alpha) * np.power(p, gamma) * (1.0 - p) * (
                (gamma * b) + (p * db_dp)
            )
            grad[neg_mask] = grad_neg
            hess[neg_mask] = hess_neg

        grad *= weights
        hess = np.maximum(hess * weights, 1e-6)
        return grad.astype(np.float32), hess.astype(np.float32)

    return focal_objective


def weighted_ap_metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    probs = sigmoid(predt)
    score = safe_average_precision(labels, probs, sample_weight=weights if weights.size else None)
    return "weighted_ap", score


def make_training_dmatrix(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    max_bin: int,
    ref: Optional[xgb.QuantileDMatrix] = None,
) -> xgb.QuantileDMatrix:
    if ref is None:
        return xgb.QuantileDMatrix(X, label=y, weight=weights, max_bin=max_bin)
    return xgb.QuantileDMatrix(X, label=y, weight=weights, ref=ref, max_bin=max_bin)


def make_prediction_dmatrix(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> xgb.DMatrix:
    kwargs = {}
    if y is not None:
        kwargs["label"] = y
    if weights is not None:
        kwargs["weight"] = weights
    return xgb.DMatrix(X, **kwargs)


def train_progressive_booster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    w_eval: np.ndarray,
    params: Mapping[str, float],
    verbosity: int,
) -> xgb.Booster:
    focal_alpha = float(params["focal_alpha"])
    focal_gamma = float(params["focal_gamma"])
    warmup_fraction = float(params["warmup_fraction"])
    warmup_rounds = int(params["warmup_rounds"])
    main_rounds = int(params["main_rounds"])
    fine_rounds = int(params["fine_rounds"])
    eta = float(params["learning_rate"])
    fine_lr_scale = float(params["fine_lr_scale"])

    if X_train.shape[0] < 32:
        raise ValueError("Not enough training rows after windowing for progressive training.")

    warmup_rows = max(int(len(X_train) * warmup_fraction), min(len(X_train), 64))
    warmup_X = X_train[:warmup_rows]
    warmup_y = y_train[:warmup_rows]
    warmup_w = w_train[:warmup_rows]

    max_bin = int(params["max_bin"])
    train_dm = make_training_dmatrix(X_train, y_train, w_train, max_bin=max_bin)
    eval_dm = make_training_dmatrix(X_eval, y_eval, w_eval, max_bin=max_bin, ref=train_dm)
    warmup_dm = make_training_dmatrix(warmup_X, warmup_y, warmup_w, max_bin=max_bin)

    base_params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "lambda": float(params["reg_lambda"]),
        "alpha": float(params["reg_alpha"]),
        "gamma": float(params["min_split_loss"]),
        "max_delta_step": float(params["max_delta_step"]),
        "max_bin": int(params["max_bin"]),
        "eta": eta,
        "base_score": float(np.clip(np.average(y_train, weights=w_train), 1e-3, 1.0 - 1e-3)),
        "disable_default_eval_metric": 1,
        "verbosity": 0,
    }

    booster = xgb.train(
        params=base_params,
        dtrain=warmup_dm,
        num_boost_round=warmup_rounds,
        obj=make_focal_objective(alpha=focal_alpha, gamma=focal_gamma),
        custom_metric=weighted_ap_metric,
        evals=[(warmup_dm, "warmup")],
        verbose_eval=False,
    )

    booster = xgb.train(
        params=base_params,
        dtrain=train_dm,
        num_boost_round=main_rounds,
        evals=[(train_dm, "train"), (eval_dm, "eval")],
        obj=make_focal_objective(alpha=focal_alpha, gamma=focal_gamma),
        custom_metric=weighted_ap_metric,
        xgb_model=booster,
        verbose_eval=False if verbosity < 2 else 25,
    )

    fine_params = dict(base_params)
    fine_params["eta"] = eta * fine_lr_scale
    booster = xgb.train(
        params=fine_params,
        dtrain=train_dm,
        num_boost_round=fine_rounds,
        evals=[(train_dm, "train"), (eval_dm, "eval")],
        obj=make_focal_objective(alpha=focal_alpha, gamma=focal_gamma),
        custom_metric=weighted_ap_metric,
        xgb_model=booster,
        verbose_eval=False if verbosity < 2 else 25,
    )

    return booster


def calibrate_probabilities(
    raw_probabilities: np.ndarray,
    y_true: np.ndarray,
    sample_weight: np.ndarray,
    method: str,
):
    if method == "none":
        return None

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probabilities, y_true, sample_weight=sample_weight)
        return calibrator

    if method == "platt":
        calibrator = LogisticRegression(
            random_state=0,
            solver="lbfgs",
            max_iter=500,
        )
        calibrator.fit(
            raw_probabilities.reshape(-1, 1),
            y_true,
            sample_weight=sample_weight,
        )
        return calibrator

    raise ValueError(f"Unsupported calibration method: {method}")


def apply_calibrator(calibrator, raw_probabilities: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return raw_probabilities.astype(np.float32, copy=False)

    if isinstance(calibrator, IsotonicRegression):
        return np.asarray(calibrator.predict(raw_probabilities), dtype=np.float32)

    calibrated = calibrator.predict_proba(raw_probabilities.reshape(-1, 1))[:, 1]
    return np.asarray(calibrated, dtype=np.float32)


def build_fold_datasets(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    w_dev: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    selected_feature_idx: np.ndarray,
    lag_steps: Sequence[int],
    delta_feature_count: int,
    config: OTETrainingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_lag = max(lag_steps)

    if not np.all(np.diff(train_idx) == 1) or not np.all(np.diff(val_idx) == 1):
        raise ValueError("This pipeline expects contiguous expanding-window folds.")

    train_end = int(train_idx[-1]) + 1
    val_start = int(val_idx[0])
    val_end = int(val_idx[-1]) + 1

    X_train_raw = X_dev[:train_end, :][:, selected_feature_idx]
    scaler = fit_scaler(X_train_raw, config)

    X_train_scaled = transform_matrix(scaler, X_train_raw, clip_value=config.scale_clip)
    X_train = build_sparse_lag_view(X_train_scaled, lag_steps, delta_feature_count=delta_feature_count)
    y_train = y_dev[max_lag:train_end].astype(np.uint8, copy=False)
    w_train = w_dev[max_lag:train_end].astype(np.float32, copy=True)

    val_context_start = val_start - max_lag
    if val_context_start < 0:
        raise ValueError("Validation fold does not have enough causal context.")

    X_val_context = X_dev[val_context_start:val_end, :][:, selected_feature_idx]
    X_val_scaled = transform_matrix(scaler, X_val_context, clip_value=config.scale_clip)
    X_val = build_sparse_lag_view(X_val_scaled, lag_steps, delta_feature_count=delta_feature_count)
    y_val = y_dev[val_start:val_end].astype(np.uint8, copy=False)
    w_val = w_dev[val_start:val_end].astype(np.float32, copy=True)

    return X_train, y_train, w_train, X_val, y_val, w_val


def train_and_score_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    params: Mapping[str, float],
    config: OTETrainingConfig,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    hard_radius = int(params["hard_negative_radius"])
    hard_multiplier = float(params["hard_negative_multiplier"])
    train_weights = apply_hard_negative_weights(y_train, w_train, hard_radius, hard_multiplier)

    if config.use_balanced_tuning_sample:
        hard_mask = build_near_positive_mask(y_train, hard_radius)
        sampled_idx = sample_balanced_training_indices(
            y=y_train,
            hard_negative_mask=hard_mask,
            negative_ratio=config.tuning_negative_ratio,
            seed=seed,
        )
        if len(sampled_idx) >= max(64, int(np.sum(y_train)) * 6):
            X_fit = X_train[sampled_idx]
            y_fit = y_train[sampled_idx]
            w_fit = train_weights[sampled_idx]
        else:
            X_fit = X_train
            y_fit = y_train
            w_fit = train_weights
    else:
        X_fit = X_train
        y_fit = y_train
        w_fit = train_weights

    booster = train_progressive_booster(
        X_train=X_fit,
        y_train=y_fit,
        w_train=w_fit,
        X_eval=X_val,
        y_eval=y_val,
        w_eval=w_val,
        params=params,
        verbosity=config.verbosity,
    )

    raw_pred = booster.predict(make_prediction_dmatrix(X_val))
    probabilities = sigmoid(raw_pred).astype(np.float32)

    threshold, threshold_metrics = select_operating_threshold(
        y_true=y_val,
        y_score=probabilities,
        tolerance=config.event_tolerance_bars,
        cooldown=config.event_cooldown_bars,
        grid_size=config.threshold_grid_size,
    )

    ap = safe_average_precision(y_val, probabilities, sample_weight=w_val)
    roc_auc = safe_roc_auc(y_val, probabilities, sample_weight=w_val)
    brier = float(brier_score_loss(y_val, probabilities, sample_weight=w_val))

    metrics = {
        "average_precision": ap,
        "roc_auc": roc_auc,
        "brier": brier,
        "threshold": threshold,
        **threshold_metrics,
    }
    return probabilities, metrics


def build_trial_params(trial: optuna.Trial, config: OTETrainingConfig) -> Dict[str, float]:
    max_top_features = min(config.top_feature_max, config.max_loaded_features)
    min_top_features = min(config.top_feature_min, max_top_features)
    return {
        "top_features": trial.suggest_int(
            "top_features",
            min_top_features,
            max_top_features,
            step=config.top_feature_step,
        ),
        "window_size": trial.suggest_int("window_size", config.window_min, config.window_max, step=4),
        "lag_count": trial.suggest_int("lag_count", config.lag_count_min, config.lag_count_max),
        "delta_feature_count": trial.suggest_int(
            "delta_feature_count", 0, min(config.delta_feature_cap, config.top_feature_max), step=4
        ),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
        "focal_alpha": trial.suggest_float("focal_alpha", 0.70, 0.95),
        "focal_gamma": trial.suggest_float("focal_gamma", 1.25, 3.75),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 14.0),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_split_loss": trial.suggest_float("min_split_loss", 0.0, 3.0),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 4.0),
        "max_bin": trial.suggest_categorical("max_bin", [256, 384, 512]),
        "warmup_fraction": trial.suggest_float("warmup_fraction", 0.15, 0.35),
        "warmup_rounds": trial.suggest_int("warmup_rounds", 20, 80, step=10),
        "main_rounds": trial.suggest_int("main_rounds", 120, 420, step=30),
        "fine_rounds": trial.suggest_int("fine_rounds", 20, 90, step=10),
        "fine_lr_scale": trial.suggest_float("fine_lr_scale", 0.15, 0.6),
        "hard_negative_radius": trial.suggest_int("hard_negative_radius", 1, 8),
        "hard_negative_multiplier": trial.suggest_float("hard_negative_multiplier", 1.0, 2.5),
    }


def cross_validate_trial(
    dataset: PreparedTargetDataset,
    params: Mapping[str, float],
    config: OTETrainingConfig,
    trial: Optional[optuna.Trial] = None,
) -> TrialArtifacts:
    top_features = min(int(params["top_features"]), len(dataset.ranked_features))
    window_size = int(params["window_size"])
    lag_count = int(params["lag_count"])
    delta_feature_count = min(int(params["delta_feature_count"]), top_features)
    lag_steps = make_lag_steps(window_size=window_size, lag_count=lag_count)
    purge_bars = max(
        window_size + config.purge_additional_bars,
        config.event_tolerance_bars + config.event_cooldown_bars,
    )

    selected_feature_names = dataset.ranked_features[:top_features]
    selected_feature_idx = np.arange(top_features, dtype=np.int64)

    splitter = PurgedWalkForwardSplitter(
        n_splits=config.cv_splits,
        min_train_rows=max(config.min_train_rows, window_size * 3),
        min_val_rows=max(config.min_val_rows, window_size * 2),
        purge_bars=purge_bars,
    )
    folds = splitter.split(len(dataset.dev_X))

    oof_predictions = np.full(len(dataset.dev_y), np.nan, dtype=np.float32)
    fold_metrics: List[Dict[str, float]] = []

    for fold_index, (train_idx, val_idx) in enumerate(folds, start=1):
        X_train, y_train, w_train, X_val, y_val, w_val = build_fold_datasets(
            X_dev=dataset.dev_X,
            y_dev=dataset.dev_y,
            w_dev=dataset.dev_w,
            train_idx=train_idx,
            val_idx=val_idx,
            selected_feature_idx=selected_feature_idx,
            lag_steps=lag_steps,
            delta_feature_count=delta_feature_count,
            config=config,
        )

        if int(np.sum(y_train)) < config.min_positive_per_fold or int(np.sum(y_val)) < config.min_positive_per_fold:
            raise optuna.TrialPruned(
                f"Insufficient positives in fold {fold_index}: train={int(np.sum(y_train))}, val={int(np.sum(y_val))}"
            )

        probabilities, metrics = train_and_score_fold(
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_val=X_val,
            y_val=y_val,
            w_val=w_val,
            params=params,
            config=config,
            seed=config.random_seed + fold_index,
        )
        oof_predictions[val_idx] = probabilities
        metrics["fold"] = float(fold_index)
        fold_metrics.append(metrics)

        if trial is not None:
            running_score = np.mean(
                [
                    (0.70 * fold["average_precision"]) + (0.30 * fold["event_fbeta_0_5"])
                    for fold in fold_metrics
                ]
            )
            trial.report(running_score, step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        gc.collect()

    return TrialArtifacts(
        params=dict(params),
        oof_pred=oof_predictions,
        fold_metrics=fold_metrics,
        selected_feature_names=selected_feature_names,
        lag_steps=lag_steps,
    )


def summarize_fold_metrics(fold_metrics: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = sorted({key for metrics in fold_metrics for key in metrics.keys() if key != "fold"})
    summary: Dict[str, float] = {}
    for key in keys:
        values = [float(metrics[key]) for metrics in fold_metrics if key in metrics]
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"std_{key}"] = float(np.std(values))
    return summary


def fit_final_model(
    dataset: PreparedTargetDataset,
    artifacts: TrialArtifacts,
    config: OTETrainingConfig,
) -> Dict[str, object]:
    params = dict(artifacts.params)
    top_features = min(int(params["top_features"]), len(dataset.ranked_features))
    delta_feature_count = min(int(params["delta_feature_count"]), top_features)
    selected_feature_names = artifacts.selected_feature_names
    selected_feature_idx = np.arange(top_features, dtype=np.int64)
    lag_steps = artifacts.lag_steps
    max_lag = max(lag_steps)

    X_dev_raw = dataset.dev_X[:, selected_feature_idx]
    scaler = fit_scaler(X_dev_raw, config)
    X_dev_scaled = transform_matrix(scaler, X_dev_raw, clip_value=config.scale_clip)
    X_dev_windowed = build_sparse_lag_view(
        X_dev_scaled,
        lag_steps=lag_steps,
        delta_feature_count=delta_feature_count,
    )
    y_dev = dataset.dev_y[max_lag:].astype(np.uint8, copy=False)
    w_dev = apply_hard_negative_weights(
        dataset.dev_y[max_lag:],
        dataset.dev_w[max_lag:],
        radius=int(params["hard_negative_radius"]),
        multiplier=float(params["hard_negative_multiplier"]),
    )

    eval_rows = max(max_lag + config.min_val_rows, int(len(X_dev_windowed) * 0.15))
    eval_rows = min(eval_rows, len(X_dev_windowed) - 32)
    if eval_rows < 32:
        eval_rows = min(64, max(32, len(X_dev_windowed) // 4))

    fit_end = len(X_dev_windowed) - eval_rows
    X_fit = X_dev_windowed[:fit_end]
    y_fit = y_dev[:fit_end]
    w_fit = w_dev[:fit_end]
    X_eval = X_dev_windowed[fit_end:]
    y_eval = y_dev[fit_end:]
    w_eval = w_dev[fit_end:]

    booster = train_progressive_booster(
        X_train=X_fit,
        y_train=y_fit,
        w_train=w_fit,
        X_eval=X_eval,
        y_eval=y_eval,
        w_eval=w_eval,
        params=params,
        verbosity=config.verbosity,
    )

    if config.final_refit_on_dev:
        booster = train_progressive_booster(
            X_train=X_dev_windowed,
            y_train=y_dev,
            w_train=w_dev,
            X_eval=X_eval,
            y_eval=y_eval,
            w_eval=w_eval,
            params=params,
            verbosity=config.verbosity,
        )

    oof_slice = artifacts.oof_pred[max_lag:]
    valid_oof_mask = ~np.isnan(oof_slice)
    if not np.any(valid_oof_mask):
        raise RuntimeError("Cross-validation did not produce any held-out predictions.")

    oof_for_calibration = oof_slice[valid_oof_mask]
    y_for_calibration = y_dev[valid_oof_mask]
    w_for_calibration = dataset.dev_w[max_lag:][valid_oof_mask]

    if len(np.unique(y_for_calibration)) < 2:
        calibrator = None
        calibrated_oof = oof_for_calibration.astype(np.float32, copy=False)
    else:
        calibrator = calibrate_probabilities(
            raw_probabilities=oof_for_calibration,
            y_true=y_for_calibration,
            sample_weight=w_for_calibration,
            method=config.calibration_method,
        )
        calibrated_oof = apply_calibrator(calibrator, oof_for_calibration)

    threshold, threshold_metrics = select_operating_threshold(
        y_true=y_for_calibration,
        y_score=calibrated_oof,
        tolerance=config.event_tolerance_bars,
        cooldown=config.event_cooldown_bars,
        grid_size=config.threshold_grid_size,
    )

    test_context = np.concatenate(
        [
            dataset.dev_X[-max_lag:, selected_feature_idx],
            dataset.X_test[:, selected_feature_idx],
        ],
        axis=0,
    )
    test_scaled = transform_matrix(scaler, test_context, clip_value=config.scale_clip)
    X_test_windowed = build_sparse_lag_view(
        test_scaled,
        lag_steps=lag_steps,
        delta_feature_count=delta_feature_count,
    )
    y_test = dataset.y_test.astype(np.uint8, copy=False)
    w_test = dataset.w_test.astype(np.float32, copy=False)

    raw_test = sigmoid(booster.predict(make_prediction_dmatrix(X_test_windowed))).astype(np.float32)
    calibrated_test = apply_calibrator(calibrator, raw_test)

    feature_names = make_windowed_feature_names(
        selected_feature_names=selected_feature_names,
        lag_steps=lag_steps,
        delta_feature_count=delta_feature_count,
    )

    return {
        "booster": booster,
        "scaler": scaler,
        "calibrator": calibrator,
        "threshold": threshold,
        "threshold_metrics": threshold_metrics,
        "test_metrics": {
            "average_precision": safe_average_precision(y_test, calibrated_test, sample_weight=w_test),
            "roc_auc": safe_roc_auc(y_test, calibrated_test, sample_weight=w_test),
            "brier": float(brier_score_loss(y_test, calibrated_test, sample_weight=w_test)),
            "threshold": threshold,
            **event_metrics(
                y_true=y_test,
                y_score=calibrated_test,
                threshold=threshold,
                tolerance=config.event_tolerance_bars,
                cooldown=config.event_cooldown_bars,
                beta=0.5,
            ),
        },
        "feature_names": feature_names,
        "selected_feature_names": selected_feature_names,
        "lag_steps": lag_steps,
        "delta_feature_count": delta_feature_count,
        "oof_calibrated": calibrated_oof,
        "test_raw": raw_test,
        "test_calibrated": calibrated_test,
    }


def save_training_outputs(
    output_dir: Path,
    dataset: PreparedTargetDataset,
    config: OTETrainingConfig,
    artifacts: TrialArtifacts,
    final_model: Dict[str, object],
    study: optuna.Study,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    final_model["booster"].save_model(output_dir / "model.json")
    joblib.dump(final_model["scaler"], output_dir / "scaler.joblib")
    if final_model["calibrator"] is not None:
        joblib.dump(final_model["calibrator"], output_dir / "calibrator.joblib")

    study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)

    importance = final_model["booster"].get_score(importance_type="gain")
    importance_rows = [
        {"feature": final_model["feature_names"][int(name[1:])], "gain": gain}
        for name, gain in importance.items()
        if name.startswith("f") and int(name[1:]) < len(final_model["feature_names"])
    ]
    importance_frame = pd.DataFrame(importance_rows, columns=["feature", "gain"])
    if not importance_frame.empty:
        importance_frame = importance_frame.sort_values("gain", ascending=False)
    importance_frame.to_csv(output_dir / "window_feature_importance.csv", index=False)

    pd.DataFrame(
        {
            "row_index": np.arange(len(dataset.y_test), dtype=np.int64),
            "target": dataset.y_test.astype(np.int64),
            "sample_weight": dataset.w_test.astype(np.float32),
            "raw_probability": final_model["test_raw"],
            "calibrated_probability": final_model["test_calibrated"],
            "predicted_label": (final_model["test_calibrated"] >= final_model["threshold"]).astype(np.int8),
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    metadata = {
        "config": asdict(config),
        "target": dataset.target_name,
        "prepared_dir": str(dataset.prepared_dir),
        "report": dataset.report,
        "best_params": artifacts.params,
        "selected_feature_names": final_model["selected_feature_names"],
        "lag_steps": final_model["lag_steps"],
        "delta_feature_count": final_model["delta_feature_count"],
        "threshold": final_model["threshold"],
        "cv_summary": summarize_fold_metrics(artifacts.fold_metrics),
        "cv_folds": artifacts.fold_metrics,
        "threshold_metrics": final_model["threshold_metrics"],
        "test_metrics": final_model["test_metrics"],
        "study_best_value": float(study.best_value),
        "study_best_trial": int(study.best_trial.number),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def train_target_pipeline(
    dataset: PreparedTargetDataset,
    config: OTETrainingConfig,
) -> Dict[str, object]:
    seed_everything(config.random_seed)

    def objective(trial: optuna.Trial) -> float:
        params = build_trial_params(trial, config)
        artifacts = cross_validate_trial(dataset=dataset, params=params, config=config, trial=trial)
        fold_score = np.mean(
            [
                (0.70 * fold["average_precision"]) + (0.30 * fold["event_fbeta_0_5"])
                for fold in artifacts.fold_metrics
            ]
        )
        return float(fold_score)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.random_seed),
        pruner=MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)

    best_artifacts = cross_validate_trial(
        dataset=dataset,
        params=study.best_params,
        config=config,
        trial=None,
    )
    final_model = fit_final_model(dataset=dataset, artifacts=best_artifacts, config=config)

    output_dir = Path(config.output_root) / dataset.target_name
    save_training_outputs(
        output_dir=output_dir,
        dataset=dataset,
        config=config,
        artifacts=best_artifacts,
        final_model=final_model,
        study=study,
    )

    return {
        "target": dataset.target_name,
        "study": study,
        "artifacts": best_artifacts,
        "final_model": final_model,
        "output_dir": output_dir,
    }


def parse_args() -> OTETrainingConfig:
    parser = argparse.ArgumentParser(description="Train OTE models on prepared target folders.")
    parser.add_argument("--prepared-root", default="data/prepared/eurusd_5min_ote_2000_v2")
    parser.add_argument("--output-root", default="models/ote_xgboost")
    parser.add_argument("--targets", nargs="+", default=["long_ote", "short_ote"])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--max-loaded-features", type=int, default=160)
    parser.add_argument("--top-feature-max", type=int, default=96)
    parser.add_argument("--top-feature-min", type=int, default=24)
    parser.add_argument("--window-max", type=int, default=40)
    parser.add_argument("--window-min", type=int, default=8)
    parser.add_argument("--event-tolerance-bars", type=int, default=2)
    parser.add_argument("--event-cooldown-bars", type=int, default=4)
    parser.add_argument("--calibration-method", choices=["platt", "isotonic", "none"], default="platt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return OTETrainingConfig(
        prepared_root=args.prepared_root,
        output_root=args.output_root,
        targets=args.targets,
        n_trials=args.trials,
        cv_splits=args.cv_splits,
        max_loaded_features=args.max_loaded_features,
        top_feature_max=args.top_feature_max,
        top_feature_min=args.top_feature_min,
        window_max=args.window_max,
        window_min=args.window_min,
        event_tolerance_bars=args.event_tolerance_bars,
        event_cooldown_bars=args.event_cooldown_bars,
        calibration_method=args.calibration_method,
        random_seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    prepared_root = Path(config.prepared_root)
    results = []
    for target_name in config.targets:
        dataset = load_prepared_target_dataset(prepared_root=prepared_root, target_name=target_name, config=config)
        result = train_target_pipeline(dataset=dataset, config=config)
        results.append(
            {
                "target": target_name,
                "output_dir": str(result["output_dir"]),
                "best_value": float(result["study"].best_value),
                "test_metrics": result["final_model"]["test_metrics"],
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
