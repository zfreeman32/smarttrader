"""
OTE model training pipeline for prepared EURUSD datasets.

This module keeps the prepared-data contract fixed while supporting
multiple training backends:

- XGBoost with causal sparse lag windows
- PyTorch sequence models such as TCN and LSTM

Shared pieces such as target loading, fold-safe scaling, event scoring,
calibration, thresholding, and artifact writing remain in this module.
When no explicit targets are supplied, the trainer auto-discovers prepared
target folders under ``--prepared-root``.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

from model_training.ote_training.feature_ranking import (
    ranking_file_candidates_for_backend,
    ranking_sort_column,
    resolve_ranking_path,
)

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
    prepared_root: str = "data/prepared/eurusd_5min_ote_full"
    targets: List[str] = field(default_factory=list)
    output_root: str = "models/ote_full_xgb_v2"
    backend: str = "xgboost"
    model_type: str = "tcn"
    random_seed: int = 42
    n_trials: int = 20
    cv_initial_train_rows: int = 250_000
    cv_val_rows: int = 100_000
    cv_step_rows: int = 100_000
    cv_max_train_rows: Optional[int] = None
    cv_min_folds: int = 2
    min_train_positive_rows: int = 1000
    min_val_positive_rows: int = 300
    min_val_true_events: int = 50
    final_eval_min_rows: int = 160
    max_loaded_features: int = 160
    top_feature_min: int = 24
    top_feature_max: int = 128
    top_feature_step: int = 8
    window_min: int = 8
    window_max: int = 40
    lag_count_min: int = 4
    lag_count_max: int = 8
    delta_feature_cap: int = 16
    scale_quantile_low: float = 5.0
    scale_quantile_high: float = 95.0
    scale_clip: float = 8.0
    label_max_holding_bars: int = 120
    label_exclusion_pre_bars: int = 10
    label_zone_pre_bars: int = 2
    purge_buffer_bars: int = 12
    threshold_grid_size: int = 31
    event_tolerance_bars: int = 2
    event_cooldown_bars: int = 4
    threshold_event_fbeta_weight: float = 0.55
    threshold_event_precision_weight: float = 0.45
    threshold_turnover_penalty_weight: float = 0.40
    threshold_turnover_target_ratio: float = 0.85
    tuning_negative_ratio: int = 8
    use_balanced_tuning_sample: bool = True
    calibration_method: str = "platt"
    final_refit_on_dev: bool = True
    objective_average_precision_weight: float = 0.45
    objective_threshold_score_weight: float = 0.45
    objective_brier_penalty_weight: float = 0.10
    batch_size: int = 256
    epochs: int = 18
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.20
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    use_amp: bool = True
    learning_rate: float = 1e-3
    window_size: int = 24
    focal_alpha_min: float = 0.70
    focal_alpha_max: float = 0.95
    focal_gamma_min: float = 1.25
    focal_gamma_max: float = 3.75
    hard_negative_radius_min: int = 1
    hard_negative_radius_max: int = 8
    hard_negative_multiplier_min: float = 1.0
    hard_negative_multiplier_max: float = 2.5
    torch_warmup_epochs: Optional[int] = None
    torch_main_epochs: Optional[int] = None
    torch_fine_epochs: Optional[int] = None
    torch_tail_epochs: int = 0
    torch_fine_lr_scale: float = 0.35
    torch_tail_lr_scale: float = 0.10
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
    source_row_idx_train: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    source_row_idx_val: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    source_row_idx_test: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    prepared_summary: Dict[str, object] = field(default_factory=dict)
    prepared_summary_path: str | None = None

    @property
    def dev_X(self) -> np.ndarray:
        return np.concatenate([self.X_train, self.X_val], axis=0)

    @property
    def dev_y(self) -> np.ndarray:
        return np.concatenate([self.y_train, self.y_val], axis=0)

    @property
    def dev_w(self) -> np.ndarray:
        return np.concatenate([self.w_train, self.w_val], axis=0)

    @property
    def dev_source_row_idx(self) -> np.ndarray:
        train = self._normalize_source_row_idx(self.source_row_idx_train, len(self.y_train))
        val = self._normalize_source_row_idx(self.source_row_idx_val, len(self.y_val))
        return np.concatenate([train, val], axis=0)

    @property
    def test_source_row_idx(self) -> np.ndarray:
        return self._normalize_source_row_idx(self.source_row_idx_test, len(self.y_test))

    @staticmethod
    def _normalize_source_row_idx(values: np.ndarray, expected_length: int) -> np.ndarray:
        if expected_length == 0:
            return np.empty(0, dtype=np.int64)
        if values.size != expected_length:
            return np.full(expected_length, -1, dtype=np.int64)
        return np.ascontiguousarray(values, dtype=np.int64)


@dataclass
class FoldPlan:
    fold: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    purge_bars: int


@dataclass
class TrialArtifacts:
    params: Dict[str, float]
    oof_pred: np.ndarray
    fold_metrics: List[Dict[str, float]]
    selected_feature_names: List[str]
    oof_fold_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int16))
    fold_plans: List[Dict[str, Any]] = field(default_factory=list)
    fold_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    lag_steps: List[int] = field(default_factory=list)
    backend: str = "xgboost"
    window_size: int = 1
    context_rows: int = 0
    resolved_purge_bars: int = 0


@dataclass
class ModelTrainingResult:
    model: object
    val_probabilities: np.ndarray
    training_history: List[Dict[str, Any]]
    checkpoint: Optional[Dict[str, Any]] = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_prepared_targets(prepared_root: Path, targets: Sequence[str]) -> List[str]:
    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared root not found: {prepared_root}")

    requested: List[str] = []
    for target in targets:
        name = str(target).strip()
        if name and name not in requested:
            requested.append(name)

    if requested:
        missing = [name for name in requested if not (prepared_root / name).exists()]
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(
                f"Prepared target directories not found under {prepared_root}: {missing_list}"
            )
        return requested

    discovered: List[str] = []
    for child in sorted(prepared_root.iterdir()):
        if not child.is_dir():
            continue
        required_files = (
            child / "train.csv",
            child / "val.csv",
            child / "test.csv",
            child / "features.json",
            child / "report.json",
        )
        if all(path.exists() for path in required_files):
            discovered.append(child.name)

    if not discovered:
        raise FileNotFoundError(
            f"No prepared target directories were found under {prepared_root}."
        )
    return discovered


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
    *,
    backend: str = "xgboost",
    model_type: str | None = None,
) -> List[str]:
    features_path = prepared_dir / "features.json"
    ranking_path = resolve_ranking_path(
        prepared_dir,
        candidate_filenames=ranking_file_candidates_for_backend(backend, model_type),
    )

    with features_path.open("r", encoding="utf-8") as handle:
        raw_features = json.load(handle)["features"]

    raw_features = [feature for feature in raw_features if not blocked_for_leakage(feature)]

    if ranking_path is not None and ranking_path.exists():
        ranking_df = pd.read_csv(ranking_path)
        ranking_df = ranking_df[~ranking_df["feature"].map(blocked_for_leakage)]
        sort_column = ranking_sort_column(ranking_df.columns)
        ranked = [
            feature
            for feature in (
                ranking_df.sort_values(sort_column, ascending=False)["feature"].tolist()
                if sort_column is not None
                else ranking_df["feature"].tolist()
            )
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
    prepared_summary_path = prepared_root / "summary.json"
    report_path = prepared_dir / "report.json"
    prepared_summary = (
        json.loads(prepared_summary_path.read_text(encoding="utf-8"))
        if prepared_summary_path.exists()
        else {}
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    ranked_features = load_ranked_features(
        prepared_dir,
        config.max_loaded_features,
        backend=config.backend,
        model_type=config.model_type,
    )
    usecols = ranked_features + ["target", "sample_weight"]

    def read_split(split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_path = prepared_dir / f"{split_name}.csv"
        header = pd.read_csv(split_path, nrows=0)
        split_usecols = list(usecols)
        has_source_row_idx = "source_row_idx" in header.columns
        if has_source_row_idx:
            split_usecols = ["source_row_idx"] + split_usecols

        frame = pd.read_csv(split_path, usecols=split_usecols)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        feature_frame = frame[ranked_features].astype(np.float32).fillna(0.0)
        target = frame["target"].astype(np.uint8).to_numpy(copy=True)
        weight = frame["sample_weight"].astype(np.float32).fillna(1.0).to_numpy(copy=True)
        if has_source_row_idx:
            source_row_idx = (
                pd.to_numeric(frame["source_row_idx"], errors="coerce")
                .fillna(-1)
                .astype(np.int64)
                .to_numpy(copy=True)
            )
        else:
            source_row_idx = np.full(len(frame), -1, dtype=np.int64)
        return (
            np.ascontiguousarray(feature_frame.to_numpy(copy=True), dtype=np.float32),
            np.ascontiguousarray(target, dtype=np.uint8),
            np.ascontiguousarray(weight, dtype=np.float32),
            np.ascontiguousarray(source_row_idx, dtype=np.int64),
        )

    X_train, y_train, w_train, source_row_idx_train = read_split("train")
    X_val, y_val, w_val, source_row_idx_val = read_split("val")
    X_test, y_test, w_test, source_row_idx_test = read_split("test")

    return PreparedTargetDataset(
        target_name=target_name,
        prepared_dir=prepared_dir,
        feature_names=ranked_features,
        ranked_features=ranked_features,
        report=report,
        prepared_summary=prepared_summary,
        prepared_summary_path=str(prepared_summary_path) if prepared_summary_path.exists() else None,
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val,
        w_val=w_val,
        X_test=X_test,
        y_test=y_test,
        w_test=w_test,
        source_row_idx_train=source_row_idx_train,
        source_row_idx_val=source_row_idx_val,
        source_row_idx_test=source_row_idx_test,
    )


def resolve_dataset_timezone_contract(dataset: PreparedTargetDataset) -> Dict[str, object]:
    report_contract = dataset.report.get("timezone_contract", {})
    if isinstance(report_contract, dict) and report_contract:
        return report_contract
    prepared_summary_contract = dataset.prepared_summary.get("timezone_contract", {})
    if isinstance(prepared_summary_contract, dict):
        return prepared_summary_contract
    return {}


def resolve_dataset_source_lineage(dataset: PreparedTargetDataset) -> Dict[str, object]:
    report_lineage = dataset.report.get("source_lineage", {})
    if isinstance(report_lineage, dict) and report_lineage:
        return report_lineage
    prepared_summary_lineage = dataset.prepared_summary.get("source_lineage", {})
    if isinstance(prepared_summary_lineage, dict):
        return prepared_summary_lineage
    return {}


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


def build_sequence_windows(
    matrix: np.ndarray,
    window_size: int,
) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")
    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}.")
    if matrix.shape[0] < window_size:
        raise ValueError(
            f"Need at least {window_size} rows to build sequence windows, got {matrix.shape[0]}."
        )

    windows = np.lib.stride_tricks.sliding_window_view(matrix, window_shape=window_size, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    return np.ascontiguousarray(windows, dtype=np.float32)


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


def validate_backend_config(config: OTETrainingConfig) -> None:
    scalar_ranges = {
        "focal_alpha": (config.focal_alpha_min, config.focal_alpha_max),
        "focal_gamma": (config.focal_gamma_min, config.focal_gamma_max),
        "hard_negative_radius": (config.hard_negative_radius_min, config.hard_negative_radius_max),
        "hard_negative_multiplier": (
            config.hard_negative_multiplier_min,
            config.hard_negative_multiplier_max,
        ),
    }
    for name, (lower, upper) in scalar_ranges.items():
        if lower > upper:
            raise ValueError(f"{name} min cannot exceed max: {lower} > {upper}.")

    non_negative_fields = {
        "threshold_event_fbeta_weight": config.threshold_event_fbeta_weight,
        "threshold_event_precision_weight": config.threshold_event_precision_weight,
        "threshold_turnover_penalty_weight": config.threshold_turnover_penalty_weight,
        "threshold_turnover_target_ratio": config.threshold_turnover_target_ratio,
        "objective_average_precision_weight": config.objective_average_precision_weight,
        "objective_threshold_score_weight": config.objective_threshold_score_weight,
        "objective_brier_penalty_weight": config.objective_brier_penalty_weight,
    }
    for name, value in non_negative_fields.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")

    if (
        config.threshold_event_fbeta_weight <= 0
        and config.threshold_event_precision_weight <= 0
    ):
        raise ValueError(
            "Threshold scoring needs at least one positive event metric weight."
        )
    if (
        config.objective_average_precision_weight <= 0
        and config.objective_threshold_score_weight <= 0
    ):
        raise ValueError(
            "Optuna scoring needs at least one positive model-quality weight."
        )

    if config.backend not in {"xgboost", "torch"}:
        raise ValueError(f"Unsupported backend: {config.backend}")
    if config.backend == "torch" and config.model_type not in {"tcn", "lstm"}:
        raise ValueError(f"Unsupported torch model type: {config.model_type}")
    if config.backend != "torch":
        return

    if config.torch_fine_lr_scale <= 0:
        raise ValueError("Torch fine LR scale must be positive.")
    if config.torch_tail_lr_scale <= 0:
        raise ValueError("Torch tail LR scale must be positive.")
    if config.torch_tail_epochs < 0:
        raise ValueError("Torch tail epochs cannot be negative.")

    phase_values = {
        "torch_warmup_epochs": config.torch_warmup_epochs,
        "torch_main_epochs": config.torch_main_epochs,
        "torch_fine_epochs": config.torch_fine_epochs,
    }
    explicit_schedule = any(value is not None for value in phase_values.values()) or config.torch_tail_epochs > 0
    if not explicit_schedule:
        return

    missing = [name for name, value in phase_values.items() if value is None]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(
            "Explicit torch phase scheduling requires warmup, main, and fine epoch counts. "
            f"Missing: {missing_csv}"
        )

    phase_sum = int(config.torch_tail_epochs)
    for name, value in phase_values.items():
        if int(value) < 0:
            raise ValueError(f"{name} cannot be negative.")
        phase_sum += int(value)
    if phase_sum != int(config.epochs):
        raise ValueError(
            "Explicit torch phase schedule must sum to total epochs. "
            f"Got warmup/main/fine/tail={config.torch_warmup_epochs}/"
            f"{config.torch_main_epochs}/{config.torch_fine_epochs}/{config.torch_tail_epochs} "
            f"for epochs={config.epochs}."
        )


def resolve_trial_window_size(
    params: Mapping[str, float],
    config: OTETrainingConfig,
) -> int:
    window_size = int(params.get("window_size", config.window_size))
    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}.")
    return window_size


def resolve_purge_bars(config: OTETrainingConfig, context_rows: int) -> int:
    base = max(
        config.label_max_holding_bars,
        config.label_exclusion_pre_bars + config.label_zone_pre_bars,
        context_rows,
        config.event_tolerance_bars + config.event_cooldown_bars,
    )
    return int(base + config.purge_buffer_bars)


def max_context_rows_for_config(config: OTETrainingConfig) -> int:
    max_window_size = max(int(config.window_size), int(config.window_max))
    return max(max_window_size - 1, 0)


def validate_cv_geometry_for_dataset(
    dataset: PreparedTargetDataset,
    config: OTETrainingConfig,
) -> None:
    max_context_rows = max_context_rows_for_config(config)
    resolved_purge_bars = resolve_purge_bars(config, context_rows=max_context_rows)
    required_rows = (
        int(config.cv_initial_train_rows)
        + resolved_purge_bars
        + int(config.cv_val_rows)
        + max(int(config.cv_min_folds) - 1, 0) * int(config.cv_step_rows)
    )
    available_rows = int(len(dataset.dev_X))

    if available_rows < required_rows:
        suggestion = ""
        prepared_dir = str(dataset.prepared_dir).replace("\\", "/").lower()
        if prepared_dir.endswith("eurusd_5min_ote_2000_v2"):
            suggestion = (
                " The current V2 geometry is intended for the full prepared dataset. "
                "Try --prepared-root data/prepared/eurusd_5min_ote_full."
            )

        raise ValueError(
            f"Dataset {dataset.target_name!r} at {dataset.prepared_dir} has {available_rows} development rows, "
            f"but the configured V2 walk-forward geometry needs at least {required_rows} rows "
            f"(initial_train={config.cv_initial_train_rows}, val_rows={config.cv_val_rows}, "
            f"step_rows={config.cv_step_rows}, min_folds={config.cv_min_folds}, "
            f"max_context_rows={max_context_rows}, resolved_purge_bars={resolved_purge_bars})."
            f"{suggestion}"
        )


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


def describe_event_count_balance(
    predicted_count: int,
    true_count: int,
) -> Dict[str, float]:
    baseline = max(float(true_count), 1.0)
    predicted_to_true_ratio = float(predicted_count) / baseline
    return {
        "predicted_to_true_event_ratio": predicted_to_true_ratio,
        "event_count_gap_ratio": abs(float(predicted_count) - float(true_count)) / baseline,
    }


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
        **describe_event_count_balance(predicted_count=predicted_count, true_count=true_count),
    }


def select_operating_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tolerance: int,
    cooldown: int,
    grid_size: int,
    config: OTETrainingConfig,
) -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.05, 0.95, num=grid_size)
    best_threshold = 0.5
    best_metrics: Optional[Dict[str, float]] = None

    for threshold in thresholds:
        metrics = event_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=float(threshold),
            tolerance=tolerance,
            cooldown=cooldown,
            beta=0.5,
        )
        turnover_excess_ratio = max(
            0.0,
            float(metrics["predicted_to_true_event_ratio"]) - float(config.threshold_turnover_target_ratio),
        )
        threshold_score = (
            (float(config.threshold_event_fbeta_weight) * float(metrics["event_fbeta_0_5"]))
            + (float(config.threshold_event_precision_weight) * float(metrics["event_precision"]))
            - (float(config.threshold_turnover_penalty_weight) * turnover_excess_ratio)
        )
        candidate_metrics = {
            **metrics,
            "turnover_excess_ratio": turnover_excess_ratio,
            "threshold_score": threshold_score,
        }
        candidate_score = (
            float(candidate_metrics["threshold_score"]),
            float(candidate_metrics["event_precision"]),
            float(candidate_metrics["event_fbeta_0_5"]),
            -float(candidate_metrics["predicted_events"]),
        )
        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = candidate_metrics
            continue

        best_score = (
            float(best_metrics["threshold_score"]),
            float(best_metrics["event_precision"]),
            float(best_metrics["event_fbeta_0_5"]),
            -float(best_metrics["predicted_events"]),
        )
        if candidate_score > best_score:
            best_threshold = float(threshold)
            best_metrics = candidate_metrics

    if best_metrics is None:
        raise RuntimeError("Threshold search did not evaluate any candidates.")
    return best_threshold, best_metrics


def score_fold_metrics(
    metrics: Mapping[str, float],
    config: OTETrainingConfig,
) -> float:
    return float(
        (float(config.objective_average_precision_weight) * float(metrics["average_precision"]))
        + (float(config.objective_threshold_score_weight) * float(metrics["threshold_score"]))
        - (float(config.objective_brier_penalty_weight) * float(metrics["brier"]))
    )


class PurgedWalkForwardSplitter:
    def __init__(
        self,
        initial_train_rows: int,
        val_rows: int,
        step_rows: int,
        purge_bars: int,
        max_train_rows: Optional[int] = None,
        min_folds: int = 2,
    ) -> None:
        if initial_train_rows <= 0:
            raise ValueError(f"Initial train rows must be positive, got {initial_train_rows}.")
        if val_rows <= 0:
            raise ValueError(f"Validation rows must be positive, got {val_rows}.")
        if step_rows <= 0:
            raise ValueError(f"Step rows must be positive, got {step_rows}.")
        if purge_bars < 0:
            raise ValueError(f"Purge bars must be non-negative, got {purge_bars}.")
        if max_train_rows is not None and max_train_rows <= 0:
            raise ValueError(f"Max train rows must be positive, got {max_train_rows}.")
        if min_folds <= 0:
            raise ValueError(f"Minimum folds must be positive, got {min_folds}.")

        self.initial_train_rows = initial_train_rows
        self.val_rows = val_rows
        self.step_rows = step_rows
        self.purge_bars = purge_bars
        self.max_train_rows = max_train_rows
        self.min_folds = min_folds

    def split(self, n_rows: int) -> List[FoldPlan]:
        if n_rows < (self.initial_train_rows + self.val_rows + self.purge_bars):
            raise ValueError(
                f"Not enough rows ({n_rows}) for initial_train={self.initial_train_rows}, "
                f"val_rows={self.val_rows}, purge={self.purge_bars}."
            )

        folds: List[FoldPlan] = []
        train_end = self.initial_train_rows

        while True:
            train_start = 0
            if self.max_train_rows is not None:
                train_start = max(0, train_end - self.max_train_rows)
            val_start = train_end + self.purge_bars
            val_end = val_start + self.val_rows
            if val_end > n_rows:
                break

            folds.append(
                FoldPlan(
                    fold=len(folds) + 1,
                    train_start=int(train_start),
                    train_end=int(train_end),
                    val_start=int(val_start),
                    val_end=int(val_end),
                    purge_bars=int(self.purge_bars),
                )
            )
            train_end += self.step_rows

        if len(folds) < self.min_folds:
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


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    w_eval: np.ndarray,
    params: Mapping[str, float],
    verbosity: int,
) -> ModelTrainingResult:
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

    raw_pred = booster.predict(make_prediction_dmatrix(X_eval))
    probabilities = sigmoid(raw_pred).astype(np.float32)
    training_history = [
        {
            "phase": "warmup",
            "epochs": float(warmup_rounds),
            "train_rows": float(len(warmup_X)),
            "eval_rows": float(len(X_eval)),
            "learning_rate": eta,
        },
        {
            "phase": "main",
            "epochs": float(main_rounds),
            "train_rows": float(len(X_train)),
            "eval_rows": float(len(X_eval)),
            "learning_rate": eta,
        },
        {
            "phase": "fine",
            "epochs": float(fine_rounds),
            "train_rows": float(len(X_train)),
            "eval_rows": float(len(X_eval)),
            "learning_rate": eta * fine_lr_scale,
        },
    ]
    return ModelTrainingResult(
        model=booster,
        val_probabilities=probabilities,
        training_history=training_history,
        checkpoint=None,
    )


def build_torch_training_config(
    input_size: int,
    params: Mapping[str, float],
    config: OTETrainingConfig,
) -> Dict[str, Any]:
    return {
        "model_type": config.model_type,
        "input_size": input_size,
        "hidden_size": int(config.hidden_size),
        "num_layers": int(params.get("num_layers", config.num_layers)),
        "dropout": float(config.dropout),
        "batch_size": int(config.batch_size),
        "epochs": int(config.epochs),
        "weight_decay": float(config.weight_decay),
        "gradient_clip": float(config.gradient_clip),
        "use_amp": bool(config.use_amp),
        "learning_rate": float(params.get("learning_rate", config.learning_rate)),
        "focal_alpha": float(params["focal_alpha"]),
        "focal_gamma": float(params["focal_gamma"]),
        "warmup_epochs": None if config.torch_warmup_epochs is None else int(config.torch_warmup_epochs),
        "main_epochs": None if config.torch_main_epochs is None else int(config.torch_main_epochs),
        "fine_epochs": None if config.torch_fine_epochs is None else int(config.torch_fine_epochs),
        "tail_epochs": int(config.torch_tail_epochs),
        "fine_lr_scale": float(config.torch_fine_lr_scale),
        "tail_lr_scale": float(config.torch_tail_lr_scale),
        "verbosity": int(config.verbosity),
        "seed": int(config.random_seed),
    }


def train_backend_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    w_eval: np.ndarray,
    params: Mapping[str, float],
    config: OTETrainingConfig,
    seed: int,
) -> ModelTrainingResult:
    if config.backend == "xgboost":
        return train_xgboost_model(
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_eval=X_eval,
            y_eval=y_eval,
            w_eval=w_eval,
            params=params,
            verbosity=config.verbosity,
        )

    if config.backend == "torch":
        try:
            from model_training.ote_training.torch_trainer import train_torch_model
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Torch backend requested, but PyTorch is not available in the active Python environment."
            ) from exc

        trainer_config = build_torch_training_config(
            input_size=X_train.shape[-1],
            params=params,
            config=config,
        )
        trainer_config["seed"] = seed
        return train_torch_model(
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_eval=X_eval,
            y_eval=y_eval,
            w_eval=w_eval,
            trainer_config=trainer_config,
        )

    raise ValueError(f"Unsupported backend: {config.backend}")


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


def resolve_calibration_method(
    config: OTETrainingConfig,
    target_name: str,
) -> str:
    requested_method = str(config.calibration_method).strip().lower()
    if requested_method == "none":
        return "none"

    if "continuation" in str(target_name).strip().lower():
        return "none"

    return requested_method


def build_xgboost_fold_datasets(
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


def build_sequence_fold_datasets(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    w_dev: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    selected_feature_idx: np.ndarray,
    window_size: int,
    config: OTETrainingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    context_rows = window_size - 1

    if not np.all(np.diff(train_idx) == 1) or not np.all(np.diff(val_idx) == 1):
        raise ValueError("This pipeline expects contiguous expanding-window folds.")

    train_end = int(train_idx[-1]) + 1
    val_start = int(val_idx[0])
    val_end = int(val_idx[-1]) + 1

    X_train_raw = X_dev[:train_end, :][:, selected_feature_idx]
    scaler = fit_scaler(X_train_raw, config)

    X_train_scaled = transform_matrix(scaler, X_train_raw, clip_value=config.scale_clip)
    X_train = build_sequence_windows(X_train_scaled, window_size=window_size)
    y_train = y_dev[context_rows:train_end].astype(np.uint8, copy=False)
    w_train = w_dev[context_rows:train_end].astype(np.float32, copy=True)

    val_context_start = val_start - context_rows
    if val_context_start < 0:
        raise ValueError("Validation fold does not have enough causal context.")

    X_val_context = X_dev[val_context_start:val_end, :][:, selected_feature_idx]
    X_val_scaled = transform_matrix(scaler, X_val_context, clip_value=config.scale_clip)
    X_val = build_sequence_windows(X_val_scaled, window_size=window_size)
    y_val = y_dev[val_start:val_end].astype(np.uint8, copy=False)
    w_val = w_dev[val_start:val_end].astype(np.float32, copy=True)

    return X_train, y_train, w_train, X_val, y_val, w_val


def build_fold_diagnostics(
    fold_plan: FoldPlan,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    window_size: int,
    context_rows: int,
) -> Dict[str, Any]:
    train_positive_rows = int(np.sum(y_train))
    val_positive_rows = int(np.sum(y_val))
    train_true_events = len(group_positive_zones(y_train))
    val_true_events = len(group_positive_zones(y_val))

    return {
        "fold": int(fold_plan.fold),
        "train_start_row": int(fold_plan.train_start),
        "train_end_row": int(fold_plan.train_end - 1),
        "val_start_row": int(fold_plan.val_start),
        "val_end_row": int(fold_plan.val_end - 1),
        "train_rows_raw": int(fold_plan.train_end - fold_plan.train_start),
        "val_rows_raw": int(fold_plan.val_end - fold_plan.val_start),
        "train_rows_model": int(len(X_train)),
        "val_rows_model": int(len(X_val)),
        "train_positive_rows": train_positive_rows,
        "val_positive_rows": val_positive_rows,
        "train_true_events": int(train_true_events),
        "val_true_events": int(val_true_events),
        "purge_bars": int(fold_plan.purge_bars),
        "window_size": int(window_size),
        "context_rows": int(context_rows),
    }


def apply_fold_training_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    params: Mapping[str, float],
    config: OTETrainingConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hard_radius = int(params["hard_negative_radius"])
    hard_multiplier = float(params["hard_negative_multiplier"])
    train_weights = apply_hard_negative_weights(y_train, w_train, hard_radius, hard_multiplier)

    if not config.use_balanced_tuning_sample:
        return X_train, y_train, train_weights

    hard_mask = build_near_positive_mask(y_train, hard_radius)
    sampled_idx = sample_balanced_training_indices(
        y=y_train,
        hard_negative_mask=hard_mask,
        negative_ratio=config.tuning_negative_ratio,
        seed=seed,
    )
    if len(sampled_idx) < max(64, int(np.sum(y_train)) * 6):
        return X_train, y_train, train_weights

    return X_train[sampled_idx], y_train[sampled_idx], train_weights[sampled_idx]


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
    X_fit, y_fit, w_fit = apply_fold_training_weights(
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        params=params,
        config=config,
        seed=seed,
    )

    training_result = train_backend_model(
        X_train=X_fit,
        y_train=y_fit,
        w_train=w_fit,
        X_eval=X_val,
        y_eval=y_val,
        w_eval=w_val,
        params=params,
        config=config,
        seed=seed,
    )
    probabilities = training_result.val_probabilities

    threshold, threshold_metrics = select_operating_threshold(
        y_true=y_val,
        y_score=probabilities,
        tolerance=config.event_tolerance_bars,
        cooldown=config.event_cooldown_bars,
        grid_size=config.threshold_grid_size,
        config=config,
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
    metrics["fold_objective_score"] = score_fold_metrics(metrics, config)
    return probabilities, metrics


def build_xgboost_trial_params(trial: optuna.Trial, config: OTETrainingConfig) -> Dict[str, float]:
    return {
        "window_size": trial.suggest_int("window_size", config.window_min, config.window_max, step=4),
        "lag_count": trial.suggest_int("lag_count", config.lag_count_min, config.lag_count_max),
        "delta_feature_count": trial.suggest_int(
            "delta_feature_count", 0, min(config.delta_feature_cap, config.max_loaded_features), step=4
        ),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
        "focal_alpha": trial.suggest_float("focal_alpha", config.focal_alpha_min, config.focal_alpha_max),
        "focal_gamma": trial.suggest_float("focal_gamma", config.focal_gamma_min, config.focal_gamma_max),
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
        "hard_negative_radius": trial.suggest_int(
            "hard_negative_radius",
            config.hard_negative_radius_min,
            config.hard_negative_radius_max,
        ),
        "hard_negative_multiplier": trial.suggest_float(
            "hard_negative_multiplier",
            config.hard_negative_multiplier_min,
            config.hard_negative_multiplier_max,
        ),
    }


def build_torch_trial_params(trial: optuna.Trial, config: OTETrainingConfig) -> Dict[str, float]:
    min_window = int(min(config.window_min, config.window_max))
    max_window = int(max(config.window_min, config.window_max))
    if min_window == max_window:
        window_size = min_window
    else:
        window_size = trial.suggest_int("window_size", min_window, max_window, step=4)

    if config.model_type == "tcn":
        min_num_layers = 2
        max_num_layers = max(4, int(config.num_layers))
    else:
        min_num_layers = 1
        max_num_layers = max(3, int(config.num_layers))

    lower_lr = max(config.learning_rate / 5.0, 1e-5)
    upper_lr = max(config.learning_rate * 5.0, lower_lr * 1.5)
    return {
        "window_size": int(window_size),
        "num_layers": int(trial.suggest_int("num_layers", min_num_layers, max_num_layers)),
        "learning_rate": trial.suggest_float("learning_rate", lower_lr, upper_lr, log=True),
        "focal_alpha": trial.suggest_float("focal_alpha", config.focal_alpha_min, config.focal_alpha_max),
        "focal_gamma": trial.suggest_float("focal_gamma", config.focal_gamma_min, config.focal_gamma_max),
        "hard_negative_radius": trial.suggest_int(
            "hard_negative_radius",
            config.hard_negative_radius_min,
            config.hard_negative_radius_max,
        ),
        "hard_negative_multiplier": trial.suggest_float(
            "hard_negative_multiplier",
            config.hard_negative_multiplier_min,
            config.hard_negative_multiplier_max,
        ),
    }


def build_trial_params(trial: optuna.Trial, config: OTETrainingConfig) -> Dict[str, float]:
    if config.backend == "xgboost":
        return build_xgboost_trial_params(trial, config)
    if config.backend == "torch":
        return build_torch_trial_params(trial, config)
    raise ValueError(f"Unsupported backend: {config.backend}")


def cross_validate_trial(
    dataset: PreparedTargetDataset,
    params: Mapping[str, float],
    config: OTETrainingConfig,
    trial: Optional[optuna.Trial] = None,
) -> TrialArtifacts:
    validate_backend_config(config)

    if not dataset.ranked_features:
        raise ValueError("Prepared dataset does not contain any ranked features for training.")

    window_size = resolve_trial_window_size(params, config)
    lag_steps: List[int] = []
    delta_feature_count = 0
    context_rows = window_size - 1
    selected_feature_names = list(dataset.ranked_features)
    selected_feature_idx = np.arange(len(selected_feature_names), dtype=np.int64)

    if config.backend == "xgboost":
        lag_count = int(params["lag_count"])
        delta_feature_count = min(int(params["delta_feature_count"]), len(selected_feature_names))
        lag_steps = make_lag_steps(window_size=window_size, lag_count=lag_count)
        context_rows = max(lag_steps)

    purge_bars = resolve_purge_bars(config=config, context_rows=context_rows)

    splitter = PurgedWalkForwardSplitter(
        initial_train_rows=max(config.cv_initial_train_rows, context_rows + 1),
        val_rows=config.cv_val_rows,
        step_rows=config.cv_step_rows,
        purge_bars=purge_bars,
        max_train_rows=config.cv_max_train_rows,
        min_folds=config.cv_min_folds,
    )
    folds = splitter.split(len(dataset.dev_X))

    oof_predictions = np.full(len(dataset.dev_y), np.nan, dtype=np.float32)
    oof_fold_id = np.full(len(dataset.dev_y), -1, dtype=np.int16)
    fold_metrics: List[Dict[str, float]] = []
    fold_plans = [asdict(fold) for fold in folds]
    fold_diagnostics: List[Dict[str, Any]] = []

    for fold_plan in folds:
        train_idx = np.arange(fold_plan.train_start, fold_plan.train_end, dtype=np.int64)
        val_idx = np.arange(fold_plan.val_start, fold_plan.val_end, dtype=np.int64)
        if config.backend == "xgboost":
            X_train, y_train, w_train, X_val, y_val, w_val = build_xgboost_fold_datasets(
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
        else:
            X_train, y_train, w_train, X_val, y_val, w_val = build_sequence_fold_datasets(
                X_dev=dataset.dev_X,
                y_dev=dataset.dev_y,
                w_dev=dataset.dev_w,
                train_idx=train_idx,
                val_idx=val_idx,
                selected_feature_idx=selected_feature_idx,
                window_size=window_size,
                config=config,
            )

        diagnostics = build_fold_diagnostics(
            fold_plan=fold_plan,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            window_size=window_size,
            context_rows=context_rows,
        )
        if diagnostics["train_positive_rows"] < config.min_train_positive_rows:
            raise optuna.TrialPruned(
                f"Insufficient train positives in fold {fold_plan.fold}: "
                f"{diagnostics['train_positive_rows']} < {config.min_train_positive_rows}."
            )
        if diagnostics["val_positive_rows"] < config.min_val_positive_rows:
            raise optuna.TrialPruned(
                f"Insufficient validation positives in fold {fold_plan.fold}: "
                f"{diagnostics['val_positive_rows']} < {config.min_val_positive_rows}."
            )
        if diagnostics["val_true_events"] < config.min_val_true_events:
            raise optuna.TrialPruned(
                f"Insufficient validation true events in fold {fold_plan.fold}: "
                f"{diagnostics['val_true_events']} < {config.min_val_true_events}."
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
            seed=config.random_seed + fold_plan.fold,
        )
        oof_predictions[val_idx] = probabilities
        oof_fold_id[val_idx] = int(fold_plan.fold)
        metrics["fold"] = int(fold_plan.fold)
        fold_metrics.append(metrics)
        fold_diagnostics.append(diagnostics)

        if trial is not None:
            running_score = np.mean([fold["fold_objective_score"] for fold in fold_metrics])
            trial.report(running_score, step=fold_plan.fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        gc.collect()

    return TrialArtifacts(
        params=dict(params),
        oof_pred=oof_predictions,
        fold_metrics=fold_metrics,
        selected_feature_names=selected_feature_names,
        oof_fold_id=oof_fold_id,
        fold_plans=fold_plans,
        fold_diagnostics=fold_diagnostics,
        lag_steps=lag_steps,
        backend=config.backend,
        window_size=window_size,
        context_rows=context_rows,
        resolved_purge_bars=purge_bars,
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
    validate_backend_config(config)

    params = dict(artifacts.params)
    if not artifacts.selected_feature_names:
        raise ValueError("Final model fit requires at least one selected feature.")

    selected_feature_names = artifacts.selected_feature_names
    selected_feature_count = len(selected_feature_names)
    delta_feature_count = min(int(params.get("delta_feature_count", 0)), selected_feature_count)
    selected_feature_idx = np.arange(selected_feature_count, dtype=np.int64)
    lag_steps = artifacts.lag_steps
    window_size = artifacts.window_size
    context_rows = artifacts.context_rows

    X_dev_raw = dataset.dev_X[:, selected_feature_idx]
    scaler = fit_scaler(X_dev_raw, config)
    X_dev_scaled = transform_matrix(scaler, X_dev_raw, clip_value=config.scale_clip)

    if config.backend == "xgboost":
        X_dev_windowed = build_sparse_lag_view(
            X_dev_scaled,
            lag_steps=lag_steps,
            delta_feature_count=delta_feature_count,
        )
        feature_names = make_windowed_feature_names(
            selected_feature_names=selected_feature_names,
            lag_steps=lag_steps,
            delta_feature_count=delta_feature_count,
        )
    else:
        X_dev_windowed = build_sequence_windows(
            X_dev_scaled,
            window_size=window_size,
        )
        feature_names = list(selected_feature_names)

    y_dev = dataset.dev_y[context_rows:].astype(np.uint8, copy=False)
    w_dev = apply_hard_negative_weights(
        dataset.dev_y[context_rows:],
        dataset.dev_w[context_rows:],
        radius=int(params["hard_negative_radius"]),
        multiplier=float(params["hard_negative_multiplier"]),
    )

    if len(X_dev_windowed) < 64:
        raise ValueError("Not enough development rows after windowing to fit the final model.")

    eval_rows = max(context_rows + config.final_eval_min_rows, int(len(X_dev_windowed) * 0.15))
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

    training_result = train_backend_model(
        X_train=X_fit,
        y_train=y_fit,
        w_train=w_fit,
        X_eval=X_eval,
        y_eval=y_eval,
        w_eval=w_eval,
        params=params,
        config=config,
        seed=config.random_seed,
    )

    if config.final_refit_on_dev:
        training_result = train_backend_model(
            X_train=X_dev_windowed,
            y_train=y_dev,
            w_train=w_dev,
            X_eval=X_eval,
            y_eval=y_eval,
            w_eval=w_eval,
            params=params,
            config=config,
            seed=config.random_seed,
        )

    final_model_object = training_result.model
    checkpoint = training_result.checkpoint

    if config.backend == "torch":
        from model_training.ote_training.torch_trainer import load_torch_model_from_checkpoint, predict_torch_model

        if checkpoint is None:
            raise RuntimeError("Torch training did not produce a checkpoint.")
        final_model_object = load_torch_model_from_checkpoint(checkpoint)
    else:
        predict_torch_model = None

    valid_oof_mask = ~np.isnan(artifacts.oof_pred)
    if not np.any(valid_oof_mask):
        raise RuntimeError("Cross-validation did not produce any held-out predictions.")

    resolved_calibration_method = resolve_calibration_method(config, dataset.target_name)
    oof_for_calibration = artifacts.oof_pred[valid_oof_mask]
    y_for_calibration = dataset.dev_y[valid_oof_mask]
    w_for_calibration = dataset.dev_w[valid_oof_mask]

    if resolved_calibration_method == "none" or len(np.unique(y_for_calibration)) < 2:
        calibrator = None
        calibrated_oof_valid = oof_for_calibration.astype(np.float32, copy=False)
    else:
        calibrator = calibrate_probabilities(
            raw_probabilities=oof_for_calibration,
            y_true=y_for_calibration,
            sample_weight=w_for_calibration,
            method=resolved_calibration_method,
        )
        calibrated_oof_valid = apply_calibrator(calibrator, oof_for_calibration)

    calibrated_oof = np.full(len(dataset.dev_y), np.nan, dtype=np.float32)
    calibrated_oof[valid_oof_mask] = calibrated_oof_valid

    threshold, threshold_metrics = select_operating_threshold(
        y_true=y_for_calibration,
        y_score=calibrated_oof_valid,
        tolerance=config.event_tolerance_bars,
        cooldown=config.event_cooldown_bars,
        grid_size=config.threshold_grid_size,
        config=config,
    )

    if context_rows > 0:
        test_context = np.concatenate(
            [
                dataset.dev_X[-context_rows:, selected_feature_idx],
                dataset.X_test[:, selected_feature_idx],
            ],
            axis=0,
        )
    else:
        test_context = dataset.X_test[:, selected_feature_idx]

    test_scaled = transform_matrix(scaler, test_context, clip_value=config.scale_clip)

    if config.backend == "xgboost":
        X_test_windowed = build_sparse_lag_view(
            test_scaled,
            lag_steps=lag_steps,
            delta_feature_count=delta_feature_count,
        )
        raw_test = sigmoid(final_model_object.predict(make_prediction_dmatrix(X_test_windowed))).astype(np.float32)
    else:
        X_test_windowed = build_sequence_windows(
            test_scaled,
            window_size=window_size,
        )
        raw_test = predict_torch_model(
            model=final_model_object,
            X=X_test_windowed,
            batch_size=config.batch_size,
        )

    y_test = dataset.y_test.astype(np.uint8, copy=False)
    w_test = dataset.w_test.astype(np.float32, copy=False)
    calibrated_test = apply_calibrator(calibrator, raw_test)

    return {
        "backend": config.backend,
        "model": final_model_object,
        "checkpoint": checkpoint,
        "scaler": scaler,
        "calibrator": calibrator,
        "requested_calibration_method": str(config.calibration_method).strip().lower(),
        "resolved_calibration_method": resolved_calibration_method,
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
        "window_size": window_size,
        "context_rows": context_rows,
        "delta_feature_count": delta_feature_count,
        "oof_calibrated": calibrated_oof,
        "test_raw": raw_test,
        "test_calibrated": calibrated_test,
        "training_history": training_result.training_history,
        "model_config": {
            "backend": config.backend,
            "model_type": config.model_type if config.backend == "torch" else "xgboost",
            "window_size": window_size,
            "context_rows": context_rows,
            "feature_count": selected_feature_count,
            "selected_feature_names": selected_feature_names,
            "lag_steps": lag_steps,
            "delta_feature_count": delta_feature_count,
            "params": params,
            "calibration": {
                "requested_method": str(config.calibration_method).strip().lower(),
                "resolved_method": resolved_calibration_method,
            },
            "trainer": {
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "hidden_size": config.hidden_size,
                "num_layers": int(params.get("num_layers", config.num_layers)),
                "dropout": config.dropout,
                "weight_decay": config.weight_decay,
                "gradient_clip": config.gradient_clip,
                "use_amp": config.use_amp,
                "learning_rate": float(params.get("learning_rate", config.learning_rate)),
                "warmup_epochs": config.torch_warmup_epochs,
                "main_epochs": config.torch_main_epochs,
                "fine_epochs": config.torch_fine_epochs,
                "tail_epochs": config.torch_tail_epochs,
                "fine_lr_scale": config.torch_fine_lr_scale,
                "tail_lr_scale": config.torch_tail_lr_scale,
            },
        },
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
    timezone_contract = resolve_dataset_timezone_contract(dataset)
    source_lineage = resolve_dataset_source_lineage(dataset)

    if final_model["backend"] == "xgboost":
        final_model["model"].save_model(output_dir / "model.json")
    else:
        import torch

        if final_model["checkpoint"] is None:
            raise RuntimeError("Torch final model is missing a checkpoint.")
        torch.save(final_model["checkpoint"], output_dir / "model.pt")
        torch.save(final_model["checkpoint"], output_dir / "best_checkpoint.pt")

    joblib.dump(final_model["scaler"], output_dir / "scaler.joblib")
    if final_model["calibrator"] is not None:
        joblib.dump(final_model["calibrator"], output_dir / "calibrator.joblib")

    study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)

    if final_model["backend"] == "xgboost":
        importance = final_model["model"].get_score(importance_type="gain")
        importance_rows = [
            {"feature": final_model["feature_names"][int(name[1:])], "gain": gain}
            for name, gain in importance.items()
            if name.startswith("f") and int(name[1:]) < len(final_model["feature_names"])
        ]
        importance_frame = pd.DataFrame(importance_rows, columns=["feature", "gain"])
        if not importance_frame.empty:
            importance_frame = importance_frame.sort_values("gain", ascending=False)
    else:
        importance_frame = pd.DataFrame(columns=["feature", "gain"])
    importance_frame.to_csv(output_dir / "window_feature_importance.csv", index=False)

    history_frame = pd.DataFrame(final_model.get("training_history", []))
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "training_history.json").write_text(
        json.dumps(final_model.get("training_history", []), indent=2),
        encoding="utf-8",
    )
    model_config_payload = dict(final_model["model_config"])
    if timezone_contract:
        model_config_payload["timezone_contract"] = timezone_contract
    if source_lineage:
        model_config_payload["source_lineage"] = source_lineage
    if dataset.prepared_summary_path:
        model_config_payload["prepared_summary_file"] = dataset.prepared_summary_path
    (output_dir / "model_config.json").write_text(
        json.dumps(model_config_payload, indent=2),
        encoding="utf-8",
    )

    fold_diagnostics_frame = pd.DataFrame(artifacts.fold_diagnostics)
    fold_metrics_frame = pd.DataFrame(artifacts.fold_metrics)
    if not fold_diagnostics_frame.empty:
        fold_diagnostics_frame["fold"] = fold_diagnostics_frame["fold"].astype(np.int64)
    if not fold_metrics_frame.empty:
        fold_metrics_frame["fold"] = fold_metrics_frame["fold"].astype(np.int64)

    cv_fold_manifest = fold_diagnostics_frame.merge(
        fold_metrics_frame,
        on="fold",
        how="left",
        sort=True,
    )
    cv_fold_manifest.to_csv(output_dir / "cv_fold_manifest.csv", index=False)
    cv_fold_manifest_records = json.loads(cv_fold_manifest.to_json(orient="records"))
    (output_dir / "cv_fold_manifest.json").write_text(
        json.dumps(cv_fold_manifest_records, indent=2),
        encoding="utf-8",
    )

    oof_predictions_frame = pd.DataFrame(
        {
            "source_row_idx": dataset.dev_source_row_idx.astype(np.int64, copy=False),
            "row_index_in_dev": np.arange(len(dataset.dev_y), dtype=np.int64),
            "fold": artifacts.oof_fold_id.astype(np.int16, copy=False),
            "target": dataset.dev_y.astype(np.int64, copy=False),
            "sample_weight": dataset.dev_w.astype(np.float32, copy=False),
            "oof_raw_probability": artifacts.oof_pred.astype(np.float32, copy=False),
            "oof_calibrated_probability": final_model["oof_calibrated"].astype(np.float32, copy=False),
        }
    )
    oof_predictions_frame.to_csv(output_dir / "oof_predictions.csv", index=False)

    pd.DataFrame(
        {
            "source_row_idx": dataset.test_source_row_idx.astype(np.int64, copy=False),
            "row_index": np.arange(len(dataset.y_test), dtype=np.int64),
            "target": dataset.y_test.astype(np.int64),
            "sample_weight": dataset.w_test.astype(np.float32),
            "raw_probability": final_model["test_raw"],
            "calibrated_probability": final_model["test_calibrated"],
            "predicted_label": (final_model["test_calibrated"] >= final_model["threshold"]).astype(np.int8),
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    valid_oof_mask = ~np.isnan(artifacts.oof_pred)
    oof_summary = {
        "dev_rows": int(len(dataset.dev_y)),
        "oof_rows": int(np.sum(valid_oof_mask)),
        "missing_oof_rows": int(np.sum(~valid_oof_mask)),
        "covered_folds": sorted(int(value) for value in np.unique(artifacts.oof_fold_id[artifacts.oof_fold_id >= 0])),
    }
    if np.any(valid_oof_mask):
        oof_summary.update(
            {
                "raw_average_precision": safe_average_precision(
                    dataset.dev_y[valid_oof_mask],
                    artifacts.oof_pred[valid_oof_mask],
                    sample_weight=dataset.dev_w[valid_oof_mask],
                ),
                "calibrated_average_precision": safe_average_precision(
                    dataset.dev_y[valid_oof_mask],
                    final_model["oof_calibrated"][valid_oof_mask],
                    sample_weight=dataset.dev_w[valid_oof_mask],
                ),
            }
        )

    metadata = {
        "config": asdict(config),
        "target": dataset.target_name,
        "backend": final_model["backend"],
        "prepared_dir": str(dataset.prepared_dir),
        "prepared_summary_file": dataset.prepared_summary_path,
        "prepared_summary": {
            "input_file": dataset.prepared_summary.get("input_file"),
            "metadata_file": dataset.prepared_summary.get("metadata_file"),
            "timezone_contract": dataset.prepared_summary.get("timezone_contract"),
            "source_lineage": dataset.prepared_summary.get("source_lineage"),
        }
        if dataset.prepared_summary
        else {},
        "report": dataset.report,
        "timezone_contract": timezone_contract,
        "source_lineage": source_lineage,
        "best_params": artifacts.params,
        "selected_feature_names": final_model["selected_feature_names"],
        "lag_steps": final_model["lag_steps"],
        "window_size": final_model["window_size"],
        "context_rows": final_model["context_rows"],
        "delta_feature_count": final_model["delta_feature_count"],
        "threshold": final_model["threshold"],
        "model_config": model_config_payload,
        "calibration": {
            "requested_method": final_model["requested_calibration_method"],
            "resolved_method": final_model["resolved_calibration_method"],
            "disabled_for_continuation_target": (
                final_model["requested_calibration_method"] != "none"
                and final_model["resolved_calibration_method"] == "none"
                and "continuation" in dataset.target_name.lower()
            ),
        },
        "cv_splitter": {
            "type": "purged_walk_forward_explicit_geometry",
            "initial_train_rows": config.cv_initial_train_rows,
            "val_rows": config.cv_val_rows,
            "step_rows": config.cv_step_rows,
            "max_train_rows": config.cv_max_train_rows,
            "min_folds": config.cv_min_folds,
            "fold_plans": artifacts.fold_plans,
        },
        "resolved_purge_bars": artifacts.resolved_purge_bars,
        "label_horizon_assumptions": {
            "label_max_holding_bars": config.label_max_holding_bars,
            "label_exclusion_pre_bars": config.label_exclusion_pre_bars,
            "label_zone_pre_bars": config.label_zone_pre_bars,
            "purge_buffer_bars": config.purge_buffer_bars,
            "event_tolerance_bars": config.event_tolerance_bars,
            "event_cooldown_bars": config.event_cooldown_bars,
        },
        "cv_summary": summarize_fold_metrics(artifacts.fold_metrics),
        "cv_folds": artifacts.fold_metrics,
        "cv_fold_manifest": cv_fold_manifest_records,
        "row_identity": {
            "source_row_idx_column": "source_row_idx",
            "source_row_idx_available": bool(
                np.all(dataset.dev_source_row_idx >= 0) and np.all(dataset.test_source_row_idx >= 0)
            ),
        },
        "oof_summary": oof_summary,
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
    validate_backend_config(config)
    validate_cv_geometry_for_dataset(dataset, config)
    seed_everything(config.random_seed)

    def objective(trial: optuna.Trial) -> float:
        params = build_trial_params(trial, config)
        artifacts = cross_validate_trial(dataset=dataset, params=params, config=config, trial=trial)
        fold_score = np.mean([fold["fold_objective_score"] for fold in artifacts.fold_metrics])
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
    parser.add_argument("--prepared-root", default="data/prepared/eurusd_5min_ote_full")
    parser.add_argument("--output-root", default="models/ote_full_xgb_v2")
    parser.add_argument("--backend", choices=["xgboost", "torch"], default="xgboost")
    parser.add_argument("--model-type", choices=["tcn", "lstm"], default="tcn")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Prepared target folder name(s) to train. Omit to auto-discover all prepared targets.",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--cv-initial-train-rows", type=int, default=250_000)
    parser.add_argument("--cv-val-rows", type=int, default=100_000)
    parser.add_argument("--cv-step-rows", type=int, default=100_000)
    parser.add_argument("--cv-max-train-rows", type=int, default=None)
    parser.add_argument("--cv-min-folds", type=int, default=2)
    parser.add_argument("--cv-splits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--min-train-positive-rows", type=int, default=1000)
    parser.add_argument("--min-val-positive-rows", type=int, default=300)
    parser.add_argument("--min-val-true-events", type=int, default=50)
    parser.add_argument("--label-max-holding-bars", type=int, default=120)
    parser.add_argument("--label-exclusion-pre-bars", type=int, default=10)
    parser.add_argument("--label-zone-pre-bars", type=int, default=2)
    parser.add_argument("--purge-buffer-bars", type=int, default=12)
    parser.add_argument("--final-eval-min-rows", type=int, default=160)
    parser.add_argument("--max-loaded-features", type=int, default=160)
    parser.add_argument("--top-feature-max", type=int, default=96, help=argparse.SUPPRESS)
    parser.add_argument("--top-feature-min", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--window-max", type=int, default=40)
    parser.add_argument("--window-min", type=int, default=8)
    parser.add_argument("--event-tolerance-bars", type=int, default=2)
    parser.add_argument("--event-cooldown-bars", type=int, default=4)
    parser.add_argument("--threshold-event-fbeta-weight", type=float, default=0.55)
    parser.add_argument("--threshold-event-precision-weight", type=float, default=0.45)
    parser.add_argument("--threshold-turnover-penalty-weight", type=float, default=0.40)
    parser.add_argument("--threshold-turnover-target-ratio", type=float, default=0.85)
    parser.add_argument("--calibration-method", choices=["platt", "isotonic", "none"], default="platt")
    parser.add_argument("--objective-average-precision-weight", type=float, default=0.45)
    parser.add_argument("--objective-threshold-score-weight", type=float, default=0.45)
    parser.add_argument("--objective-brier-penalty-weight", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--focal-alpha-min", type=float, default=0.70)
    parser.add_argument("--focal-alpha-max", type=float, default=0.95)
    parser.add_argument("--focal-gamma-min", type=float, default=1.25)
    parser.add_argument("--focal-gamma-max", type=float, default=3.75)
    parser.add_argument("--hard-negative-radius-min", type=int, default=1)
    parser.add_argument("--hard-negative-radius-max", type=int, default=8)
    parser.add_argument("--hard-negative-multiplier-min", type=float, default=1.0)
    parser.add_argument("--hard-negative-multiplier-max", type=float, default=2.5)
    parser.add_argument("--torch-warmup-epochs", type=int, default=None)
    parser.add_argument("--torch-main-epochs", type=int, default=None)
    parser.add_argument("--torch-fine-epochs", type=int, default=None)
    parser.add_argument("--torch-tail-epochs", type=int, default=0)
    parser.add_argument("--torch-fine-lr-scale", type=float, default=0.35)
    parser.add_argument("--torch-tail-lr-scale", type=float, default=0.10)
    parser.add_argument("--use-amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(use_amp=True)
    args = parser.parse_args()

    if args.cv_splits is not None:
        parser.error(
            "--cv-splits is deprecated in V2. Use --cv-initial-train-rows, --cv-val-rows, "
            "and --cv-step-rows instead."
        )

    return OTETrainingConfig(
        prepared_root=args.prepared_root,
        output_root=args.output_root,
        backend=args.backend,
        model_type=args.model_type,
        targets=args.targets or [],
        n_trials=args.trials,
        cv_initial_train_rows=args.cv_initial_train_rows,
        cv_val_rows=args.cv_val_rows,
        cv_step_rows=args.cv_step_rows,
        cv_max_train_rows=args.cv_max_train_rows,
        cv_min_folds=args.cv_min_folds,
        min_train_positive_rows=args.min_train_positive_rows,
        min_val_positive_rows=args.min_val_positive_rows,
        min_val_true_events=args.min_val_true_events,
        final_eval_min_rows=args.final_eval_min_rows,
        max_loaded_features=args.max_loaded_features,
        top_feature_max=args.top_feature_max,
        top_feature_min=args.top_feature_min,
        window_max=args.window_max,
        window_min=args.window_min,
        label_max_holding_bars=args.label_max_holding_bars,
        label_exclusion_pre_bars=args.label_exclusion_pre_bars,
        label_zone_pre_bars=args.label_zone_pre_bars,
        purge_buffer_bars=args.purge_buffer_bars,
        event_tolerance_bars=args.event_tolerance_bars,
        event_cooldown_bars=args.event_cooldown_bars,
        threshold_event_fbeta_weight=args.threshold_event_fbeta_weight,
        threshold_event_precision_weight=args.threshold_event_precision_weight,
        threshold_turnover_penalty_weight=args.threshold_turnover_penalty_weight,
        threshold_turnover_target_ratio=args.threshold_turnover_target_ratio,
        calibration_method=args.calibration_method,
        objective_average_precision_weight=args.objective_average_precision_weight,
        objective_threshold_score_weight=args.objective_threshold_score_weight,
        objective_brier_penalty_weight=args.objective_brier_penalty_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        use_amp=args.use_amp,
        learning_rate=args.learning_rate,
        window_size=args.window_size,
        focal_alpha_min=args.focal_alpha_min,
        focal_alpha_max=args.focal_alpha_max,
        focal_gamma_min=args.focal_gamma_min,
        focal_gamma_max=args.focal_gamma_max,
        hard_negative_radius_min=args.hard_negative_radius_min,
        hard_negative_radius_max=args.hard_negative_radius_max,
        hard_negative_multiplier_min=args.hard_negative_multiplier_min,
        hard_negative_multiplier_max=args.hard_negative_multiplier_max,
        torch_warmup_epochs=args.torch_warmup_epochs,
        torch_main_epochs=args.torch_main_epochs,
        torch_fine_epochs=args.torch_fine_epochs,
        torch_tail_epochs=args.torch_tail_epochs,
        torch_fine_lr_scale=args.torch_fine_lr_scale,
        torch_tail_lr_scale=args.torch_tail_lr_scale,
        random_seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    prepared_root = Path(config.prepared_root)
    resolved_targets = resolve_prepared_targets(prepared_root, config.targets)
    results = []
    for target_name in resolved_targets:
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
