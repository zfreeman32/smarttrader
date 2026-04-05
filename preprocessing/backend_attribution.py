from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from model_training.ote_training.feature_ranking import (
    filter_features_by_attribution_gates,
    load_feature_allowlist,
    merge_feature_rankings,
    ranking_sort_column,
)


SUPPORTED_BACKENDS: tuple[str, ...] = ("xgboost", "tcn", "lstm")


@dataclass
class BackendAttributionConfig:
    prepared_root: str = "data/prepared/eurusd_5min_ote_full"
    targets: List[str] = field(default_factory=lambda: ["long_ote", "short_ote"])
    backends: List[str] = field(default_factory=lambda: list(SUPPORTED_BACKENDS))
    max_features: int = 160
    attribution_max_rows: int = 25_000
    top_n_features: int = 25

    base_weight: float = 0.20
    shap_weight: float = 0.55
    shap_positive_weight: float = 0.25
    attribution_floor_fraction: float = 0.15
    attribution_cumulative_importance: float = 0.90

    random_seed: int = 42
    nthread: int = 0

    xgb_num_boost_round: int = 300
    xgb_early_stopping_rounds: int = 40
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6
    xgb_min_child_weight: float = 5.0
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_reg_lambda: float = 1.0
    xgb_reg_alpha: float = 0.0

    window_size: int = 24
    batch_size: int = 256
    attribution_batch_size: int = 128
    torch_epochs: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.20
    torch_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    scale_quantile_low: float = 5.0
    scale_quantile_high: float = 95.0
    scale_clip: float = 8.0
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    use_amp: bool = False
    ig_steps: int = 16


@dataclass
class PreparedAttributionDataset:
    target_name: str
    prepared_dir: Path
    feature_names: List[str]
    allowed_features: List[str]
    base_ranking: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    w_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    w_val: np.ndarray


def safe_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.0
    return float(average_precision_score(y_true, y_score, sample_weight=sample_weight))


def safe_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))


def normalize_backends(backends: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for backend in backends:
        normalized = str(backend).strip().lower()
        if normalized not in SUPPORTED_BACKENDS:
            supported = ", ".join(SUPPORTED_BACKENDS)
            raise ValueError(f"Unsupported backend {backend!r}. Supported values: {supported}")
        if normalized not in resolved:
            resolved.append(normalized)
    return resolved


def resolve_targets(prepared_root: Path, targets: Sequence[str]) -> List[str]:
    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared root not found: {prepared_root}")

    if targets:
        resolved = []
        for target in targets:
            name = str(target).strip()
            target_dir = prepared_root / name
            if not target_dir.exists():
                raise FileNotFoundError(f"Prepared target directory not found: {target_dir}")
            if name not in resolved:
                resolved.append(name)
        return resolved

    discovered: List[str] = []
    for child in sorted(prepared_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "train.csv").exists() and (child / "feature_importance.csv").exists():
            discovered.append(child.name)
    if not discovered:
        raise FileNotFoundError(
            f"No prepared target directories were found under {prepared_root}."
        )
    return discovered


def build_empty_stats(n_features: int) -> Dict[str, np.ndarray | int]:
    return {
        "count_all": 0,
        "count_positive": 0,
        "count_negative": 0,
        "abs_sum_all": np.zeros(n_features, dtype=np.float64),
        "signed_sum_all": np.zeros(n_features, dtype=np.float64),
        "value_sum_all": np.zeros(n_features, dtype=np.float64),
        "abs_sum_positive": np.zeros(n_features, dtype=np.float64),
        "signed_sum_positive": np.zeros(n_features, dtype=np.float64),
        "value_sum_positive": np.zeros(n_features, dtype=np.float64),
        "abs_sum_negative": np.zeros(n_features, dtype=np.float64),
        "signed_sum_negative": np.zeros(n_features, dtype=np.float64),
        "value_sum_negative": np.zeros(n_features, dtype=np.float64),
    }


def update_attribution_stats(
    stats: Dict[str, np.ndarray | int],
    contributions: np.ndarray,
    feature_values: np.ndarray,
    labels: np.ndarray,
) -> None:
    stats["count_all"] += int(len(labels))
    stats["abs_sum_all"] += np.abs(contributions).sum(axis=0)
    stats["signed_sum_all"] += contributions.sum(axis=0)
    stats["value_sum_all"] += feature_values.sum(axis=0)

    positive_mask = labels == 1
    negative_mask = ~positive_mask

    if np.any(positive_mask):
        stats["count_positive"] += int(np.sum(positive_mask))
        stats["abs_sum_positive"] += np.abs(contributions[positive_mask]).sum(axis=0)
        stats["signed_sum_positive"] += contributions[positive_mask].sum(axis=0)
        stats["value_sum_positive"] += feature_values[positive_mask].sum(axis=0)

    if np.any(negative_mask):
        stats["count_negative"] += int(np.sum(negative_mask))
        stats["abs_sum_negative"] += np.abs(contributions[negative_mask]).sum(axis=0)
        stats["signed_sum_negative"] += contributions[negative_mask].sum(axis=0)
        stats["value_sum_negative"] += feature_values[negative_mask].sum(axis=0)


def finalize_feature_stats(
    feature_names: Sequence[str],
    stats: Dict[str, np.ndarray | int],
) -> pd.DataFrame:
    count_all = max(int(stats["count_all"]), 1)
    count_positive = max(int(stats["count_positive"]), 1)
    count_negative = max(int(stats["count_negative"]), 1)

    frame = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap_all": stats["abs_sum_all"] / count_all,
            "mean_signed_shap_all": stats["signed_sum_all"] / count_all,
            "mean_feature_value_all": stats["value_sum_all"] / count_all,
            "mean_abs_shap_positive": stats["abs_sum_positive"] / count_positive,
            "mean_signed_shap_positive": stats["signed_sum_positive"] / count_positive,
            "mean_feature_value_positive": stats["value_sum_positive"] / count_positive,
            "mean_abs_shap_negative": stats["abs_sum_negative"] / count_negative,
            "mean_signed_shap_negative": stats["signed_sum_negative"] / count_negative,
            "mean_feature_value_negative": stats["value_sum_negative"] / count_negative,
        }
    )
    return frame.sort_values("mean_abs_shap_all", ascending=False).reset_index(drop=True)


def top_features(frame: pd.DataFrame, limit: int) -> List[Dict[str, float | str]]:
    subset = frame.sort_values("mean_abs_shap_all", ascending=False).head(limit)
    rows: List[Dict[str, float | str]] = []
    for row in subset.itertuples(index=False):
        rows.append(
            {
                "feature": str(row.feature),
                "mean_abs_shap_all": float(row.mean_abs_shap_all),
                "mean_signed_shap_all": float(row.mean_signed_shap_all),
                "mean_abs_shap_positive": float(row.mean_abs_shap_positive),
                "mean_signed_shap_positive": float(row.mean_signed_shap_positive),
            }
        )
    return rows


def trim_recent_rows(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(y) <= max_rows:
        return X, y, w
    return X[-max_rows:], y[-max_rows:], w[-max_rows:]


def best_iteration_range(booster: xgb.Booster) -> tuple[int, int] | None:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None or best_iteration < 0:
        return None
    return (0, int(best_iteration) + 1)


def predict_xgboost_probabilities(
    booster: xgb.Booster,
    X: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    dmatrix = xgb.DMatrix(X, feature_names=list(feature_names))
    kwargs: Dict[str, Any] = {}
    iteration_range = best_iteration_range(booster)
    if iteration_range is not None:
        kwargs["iteration_range"] = iteration_range
    return np.asarray(booster.predict(dmatrix, **kwargs), dtype=np.float32)


def load_base_ranked_feature_list(
    prepared_dir: Path,
    *,
    max_features: int,
) -> tuple[List[str], List[str], pd.DataFrame]:
    allowed_features = load_feature_allowlist(prepared_dir)
    ranking_path = prepared_dir / "feature_importance.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Base feature ranking not found: {ranking_path}")

    base_ranking = pd.read_csv(ranking_path)
    if "feature" not in base_ranking.columns:
        raise ValueError(f"Base ranking at {ranking_path} is missing a 'feature' column.")

    base_ranking = base_ranking[base_ranking["feature"].isin(allowed_features)].copy()
    sort_column = ranking_sort_column(base_ranking.columns)
    if sort_column is not None:
        base_ranking = base_ranking.sort_values(sort_column, ascending=False, kind="stable")

    ranked_features = base_ranking["feature"].tolist()
    ranked_features.extend(feature for feature in allowed_features if feature not in set(ranked_features))

    deduped: List[str] = []
    seen: set[str] = set()
    for feature in ranked_features:
        if feature in seen:
            continue
        seen.add(feature)
        deduped.append(feature)
        if max_features > 0 and len(deduped) >= max_features:
            break

    if not deduped:
        raise ValueError(f"No usable ranked features were found in {prepared_dir}.")

    return deduped, allowed_features, base_ranking.reset_index(drop=True)


def read_prepared_split(
    prepared_dir: Path,
    split_name: str,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_path = prepared_dir / f"{split_name}.csv"
    header = pd.read_csv(split_path, nrows=0)
    available_features = [feature for feature in feature_names if feature in header.columns]
    missing_features = [feature for feature in feature_names if feature not in header.columns]
    if missing_features:
        missing_csv = ", ".join(missing_features[:10])
        raise ValueError(
            f"Prepared split {split_path} is missing ranked features: {missing_csv}"
        )

    frame = pd.read_csv(
        split_path,
        usecols=["target", "sample_weight"] + available_features,
    )
    frame = frame.replace([np.inf, -np.inf], np.nan)
    X = np.ascontiguousarray(
        frame.loc[:, available_features].astype(np.float32).fillna(0.0).to_numpy(copy=True),
        dtype=np.float32,
    )
    y = np.ascontiguousarray(frame["target"].astype(np.uint8).to_numpy(copy=True), dtype=np.uint8)
    w = np.ascontiguousarray(
        frame["sample_weight"].astype(np.float32).fillna(1.0).to_numpy(copy=True),
        dtype=np.float32,
    )
    return X, y, w


def load_prepared_dataset(
    prepared_root: Path,
    target_name: str,
    *,
    max_features: int,
) -> PreparedAttributionDataset:
    prepared_dir = prepared_root / target_name
    feature_names, allowed_features, base_ranking = load_base_ranked_feature_list(
        prepared_dir,
        max_features=max_features,
    )

    X_train, y_train, w_train = read_prepared_split(prepared_dir, "train", feature_names)
    X_val, y_val, w_val = read_prepared_split(prepared_dir, "val", feature_names)

    if int(np.sum(y_train == 1)) == 0:
        raise ValueError(f"{target_name}: training split has no positive rows.")
    if int(np.sum(y_val == 1)) == 0:
        raise ValueError(f"{target_name}: validation split has no positive rows.")

    return PreparedAttributionDataset(
        target_name=target_name,
        prepared_dir=prepared_dir,
        feature_names=feature_names,
        allowed_features=allowed_features,
        base_ranking=base_ranking,
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val,
        w_val=w_val,
    )


def train_xgboost_proxy(
    dataset: PreparedAttributionDataset,
    config: BackendAttributionConfig,
) -> xgb.Booster:
    positive_weight = float(np.sum(dataset.w_train[dataset.y_train == 1]))
    negative_weight = float(np.sum(dataset.w_train[dataset.y_train == 0]))
    scale_pos_weight = negative_weight / max(positive_weight, 1e-6)

    dtrain = xgb.DMatrix(
        dataset.X_train,
        label=dataset.y_train,
        weight=dataset.w_train,
        feature_names=list(dataset.feature_names),
    )
    dval = xgb.DMatrix(
        dataset.X_val,
        label=dataset.y_val,
        weight=dataset.w_val,
        feature_names=list(dataset.feature_names),
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["aucpr", "logloss"],
        "tree_method": "hist",
        "max_depth": int(config.xgb_max_depth),
        "eta": float(config.xgb_learning_rate),
        "min_child_weight": float(config.xgb_min_child_weight),
        "subsample": float(config.xgb_subsample),
        "colsample_bytree": float(config.xgb_colsample_bytree),
        "lambda": float(config.xgb_reg_lambda),
        "alpha": float(config.xgb_reg_alpha),
        "scale_pos_weight": float(scale_pos_weight),
        "seed": int(config.random_seed),
        "verbosity": 0,
    }
    if config.nthread > 0:
        params["nthread"] = int(config.nthread)

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(config.xgb_num_boost_round),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=int(config.xgb_early_stopping_rounds),
        verbose_eval=False,
    )


def compute_tree_shap_stats(
    booster: xgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    dmatrix = xgb.DMatrix(X, feature_names=list(feature_names))
    kwargs: Dict[str, Any] = {"pred_contribs": True}
    iteration_range = best_iteration_range(booster)
    if iteration_range is not None:
        kwargs["iteration_range"] = iteration_range
    contributions = np.asarray(booster.predict(dmatrix, **kwargs), dtype=np.float32)[:, :-1]

    stats = build_empty_stats(len(feature_names))
    update_attribution_stats(stats, contributions=contributions, feature_values=X, labels=y)
    return finalize_feature_stats(feature_names, stats)


def fit_scaler(
    train_matrix: np.ndarray,
    config: BackendAttributionConfig,
) -> RobustScaler:
    scaler = RobustScaler(
        quantile_range=(float(config.scale_quantile_low), float(config.scale_quantile_high))
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


def build_sequence_windows(
    matrix: np.ndarray,
    window_size: int,
) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix for sequence windows.")
    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}.")
    if matrix.shape[0] < window_size:
        raise ValueError(
            f"Need at least {window_size} rows to build sequence windows, got {matrix.shape[0]}."
        )

    windows = np.lib.stride_tricks.sliding_window_view(matrix, window_shape=window_size, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    return np.ascontiguousarray(windows, dtype=np.float32)


def build_eval_sequence_windows(
    train_scaled: np.ndarray,
    eval_scaled: np.ndarray,
    window_size: int,
) -> np.ndarray:
    context_rows = max(window_size - 1, 0)
    if context_rows > 0:
        context = train_scaled[-context_rows:] if len(train_scaled) >= context_rows else train_scaled
        combined = np.concatenate([context, eval_scaled], axis=0)
    else:
        combined = eval_scaled

    windows = build_sequence_windows(combined, window_size)
    if len(windows) > len(eval_scaled):
        windows = windows[-len(eval_scaled):]
    return np.ascontiguousarray(windows, dtype=np.float32)


def trim_eval_sequence_rows(
    X_eval_seq: np.ndarray,
    y_eval_seq: np.ndarray,
    w_eval_seq: np.ndarray,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(y_eval_seq) <= max_rows:
        return X_eval_seq, y_eval_seq, w_eval_seq
    return X_eval_seq[-max_rows:], y_eval_seq[-max_rows:], w_eval_seq[-max_rows:]


def train_torch_proxy(
    dataset: PreparedAttributionDataset,
    model_type: str,
    config: BackendAttributionConfig,
) -> Dict[str, Any]:
    try:
        from model_training.ote_training.torch_trainer import train_torch_model
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Torch backend attribution was requested, but PyTorch is not available."
        ) from exc

    scaler = fit_scaler(dataset.X_train, config)
    train_scaled = transform_matrix(scaler, dataset.X_train, clip_value=float(config.scale_clip))
    eval_scaled = transform_matrix(scaler, dataset.X_val, clip_value=float(config.scale_clip))

    X_train_seq = build_sequence_windows(train_scaled, int(config.window_size))
    y_train_seq = np.ascontiguousarray(dataset.y_train[int(config.window_size) - 1 :], dtype=np.uint8)
    w_train_seq = np.ascontiguousarray(dataset.w_train[int(config.window_size) - 1 :], dtype=np.float32)

    X_eval_seq = build_eval_sequence_windows(train_scaled, eval_scaled, int(config.window_size))
    y_eval_seq = np.ascontiguousarray(dataset.y_val[-len(X_eval_seq) :], dtype=np.uint8)
    w_eval_seq = np.ascontiguousarray(dataset.w_val[-len(X_eval_seq) :], dtype=np.float32)

    trainer_config = {
        "model_type": model_type,
        "input_size": int(X_train_seq.shape[-1]),
        "hidden_size": int(config.hidden_size),
        "num_layers": int(config.num_layers),
        "dropout": float(config.dropout),
        "batch_size": int(config.batch_size),
        "epochs": int(config.torch_epochs),
        "weight_decay": float(config.weight_decay),
        "gradient_clip": float(config.gradient_clip),
        "use_amp": bool(config.use_amp),
        "learning_rate": float(config.torch_learning_rate),
        "focal_alpha": float(config.focal_alpha),
        "focal_gamma": float(config.focal_gamma),
        "warmup_epochs": None,
        "main_epochs": None,
        "fine_epochs": None,
        "tail_epochs": 0,
        "fine_lr_scale": 0.35,
        "tail_lr_scale": 0.10,
        "verbosity": 0,
        "seed": int(config.random_seed),
    }

    result = train_torch_model(
        X_train=X_train_seq,
        y_train=y_train_seq,
        w_train=w_train_seq,
        X_eval=X_eval_seq,
        y_eval=y_eval_seq,
        w_eval=w_eval_seq,
        trainer_config=trainer_config,
    )

    return {
        "model": result.model,
        "scaler": scaler,
        "training_history": result.training_history,
        "val_probabilities": np.asarray(result.val_probabilities, dtype=np.float32),
        "train_sequence_rows": int(len(X_train_seq)),
        "X_eval_seq": X_eval_seq,
        "y_eval_seq": y_eval_seq,
        "w_eval_seq": w_eval_seq,
    }


def compute_integrated_gradients(
    model: Any,
    X: np.ndarray,
    *,
    steps: int,
    batch_size: int,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if steps <= 0:
        raise ValueError(f"Integrated-gradient steps must be positive, got {steps}.")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    was_training = bool(getattr(model, "training", False))
    model.eval()
    pin_memory = device.type == "cuda"
    disable_cudnn = should_disable_cudnn_for_integrated_gradients(model, device)
    features = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(features),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )

    alpha_values = torch.linspace(0.0, 1.0, int(steps) + 1, device=device)[1:]
    outputs: List[np.ndarray] = []

    cudnn_context = (
        torch.backends.cudnn.flags(enabled=False)
        if disable_cudnn
        else nullcontext()
    )
    try:
        with cudnn_context:
            for (batch_features,) in loader:
                batch_features = batch_features.to(device, non_blocking=pin_memory)
                baseline = torch.zeros_like(batch_features)
                feature_delta = batch_features - baseline
                integrated_gradients = torch.zeros_like(batch_features)

                for alpha in alpha_values:
                    interpolated = (baseline + (alpha * feature_delta)).detach().requires_grad_(True)
                    model.zero_grad(set_to_none=True)
                    # cuDNN RNN kernels do not support the input-gradient backward
                    # path used by integrated gradients while the model is in eval mode.
                    logits = model(interpolated)
                    gradients = torch.autograd.grad(logits.sum(), interpolated, retain_graph=False)[0]
                    integrated_gradients = integrated_gradients + gradients

                attributions = feature_delta * (integrated_gradients / float(len(alpha_values)))
                outputs.append(attributions.detach().cpu().numpy().astype(np.float32, copy=False))
    finally:
        model.train(was_training)

    if not outputs:
        return np.empty((0,) + X.shape[1:], dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def should_disable_cudnn_for_integrated_gradients(model: Any, device: Any) -> bool:
    import torch

    return getattr(device, "type", None) == "cuda" and any(
        isinstance(module, torch.nn.RNNBase) for module in model.modules()
    )


def compute_sequence_attribution_stats(
    attributions: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    collapsed = np.sum(attributions, axis=1)
    latest_values = inputs[:, -1, :]
    stats = build_empty_stats(len(feature_names))
    update_attribution_stats(
        stats,
        contributions=np.asarray(collapsed, dtype=np.float32),
        feature_values=np.asarray(latest_values, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.uint8),
    )
    return finalize_feature_stats(feature_names, stats)


def save_backend_outputs(
    dataset: PreparedAttributionDataset,
    backend: str,
    stats_frame: pd.DataFrame,
    merged_frame: pd.DataFrame,
) -> Dict[str, str]:
    stats_path = dataset.prepared_dir / f"shap_feature_stats_{backend}.csv"
    merged_path = dataset.prepared_dir / f"feature_importance_merged_{backend}.csv"
    summary_path = dataset.prepared_dir / f"backend_attribution_summary_{backend}.json"

    stats_frame.to_csv(stats_path, index=False)
    merged_frame.to_csv(merged_path, index=False)

    return {
        "stats_csv": str(stats_path),
        "merged_csv": str(merged_path),
        "summary_json": str(summary_path),
    }


def apply_attribution_feature_filter(
    merged_frame: pd.DataFrame,
    config: BackendAttributionConfig,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    return filter_features_by_attribution_gates(
        merged_frame,
        floor_fraction=float(config.attribution_floor_fraction),
        cumulative_importance=float(config.attribution_cumulative_importance),
    )


def analyze_xgboost_backend(
    dataset: PreparedAttributionDataset,
    config: BackendAttributionConfig,
) -> Dict[str, Any]:
    booster = train_xgboost_proxy(dataset, config)
    val_probabilities = predict_xgboost_probabilities(booster, dataset.X_val, dataset.feature_names)
    X_attr, y_attr, _w_attr = trim_recent_rows(
        dataset.X_val,
        dataset.y_val,
        dataset.w_val,
        int(config.attribution_max_rows),
    )
    stats_frame = compute_tree_shap_stats(
        booster=booster,
        X=X_attr,
        y=y_attr,
        feature_names=dataset.feature_names,
    )

    merged_frame = merge_feature_rankings(
        base_ranking=dataset.base_ranking,
        shap_ranking=stats_frame,
        allowed_features=dataset.allowed_features,
        base_weight=float(config.base_weight),
        shap_weight=float(config.shap_weight),
        shap_positive_weight=float(config.shap_positive_weight),
    )
    filtered_merged_frame, filter_summary = apply_attribution_feature_filter(merged_frame, config)

    summary = {
        "target": dataset.target_name,
        "backend": "xgboost",
        "attribution_method": "tree_shap",
        "prepared_dir": str(dataset.prepared_dir),
        "candidate_feature_count": int(len(dataset.feature_names)),
        "selected_feature_count": int(len(filtered_merged_frame)),
        "feature_count": int(len(filtered_merged_frame)),
        "merge_weights": {
            "base_weight": float(config.base_weight),
            "shap_weight": float(config.shap_weight),
            "shap_positive_weight": float(config.shap_positive_weight),
        },
        "feature_filter": filter_summary,
        "model_metrics": {
            "average_precision": safe_average_precision(dataset.y_val, val_probabilities, dataset.w_val),
            "roc_auc": safe_roc_auc(dataset.y_val, val_probabilities, dataset.w_val),
            "train_rows": int(len(dataset.y_train)),
            "val_rows": int(len(dataset.y_val)),
            "val_positive_count": int(np.sum(dataset.y_val == 1)),
            "best_iteration": int(getattr(booster, "best_iteration", -1)),
        },
        "attribution_window": {
            "rows": int(len(y_attr)),
            "positive_count": int(np.sum(y_attr == 1)),
            "negative_count": int(np.sum(y_attr == 0)),
        },
        "top_features_overall": top_features(stats_frame, int(config.top_n_features)),
        "top_features_merged": filtered_merged_frame.head(int(config.top_n_features)).to_dict(orient="records"),
    }
    summary["artifacts"] = save_backend_outputs(dataset, "xgboost", stats_frame, filtered_merged_frame)
    Path(summary["artifacts"]["summary_json"]).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def analyze_sequence_backend(
    dataset: PreparedAttributionDataset,
    model_type: str,
    config: BackendAttributionConfig,
) -> Dict[str, Any]:
    proxy = train_torch_proxy(dataset, model_type=model_type, config=config)
    X_attr, y_attr, _w_attr = trim_eval_sequence_rows(
        proxy["X_eval_seq"],
        proxy["y_eval_seq"],
        proxy["w_eval_seq"],
        int(config.attribution_max_rows),
    )
    attributions = compute_integrated_gradients(
        model=proxy["model"],
        X=X_attr,
        steps=int(config.ig_steps),
        batch_size=int(config.attribution_batch_size),
    )
    stats_frame = compute_sequence_attribution_stats(
        attributions=attributions,
        inputs=X_attr,
        labels=y_attr,
        feature_names=dataset.feature_names,
    )

    merged_frame = merge_feature_rankings(
        base_ranking=dataset.base_ranking,
        shap_ranking=stats_frame,
        allowed_features=dataset.allowed_features,
        base_weight=float(config.base_weight),
        shap_weight=float(config.shap_weight),
        shap_positive_weight=float(config.shap_positive_weight),
    )
    filtered_merged_frame, filter_summary = apply_attribution_feature_filter(merged_frame, config)

    summary = {
        "target": dataset.target_name,
        "backend": model_type,
        "attribution_method": "integrated_gradients",
        "prepared_dir": str(dataset.prepared_dir),
        "candidate_feature_count": int(len(dataset.feature_names)),
        "selected_feature_count": int(len(filtered_merged_frame)),
        "feature_count": int(len(filtered_merged_frame)),
        "window_size": int(config.window_size),
        "merge_weights": {
            "base_weight": float(config.base_weight),
            "shap_weight": float(config.shap_weight),
            "shap_positive_weight": float(config.shap_positive_weight),
        },
        "feature_filter": filter_summary,
        "model_metrics": {
            "average_precision": safe_average_precision(
                proxy["y_eval_seq"],
                proxy["val_probabilities"],
                proxy["w_eval_seq"],
            ),
            "roc_auc": safe_roc_auc(
                proxy["y_eval_seq"],
                proxy["val_probabilities"],
                proxy["w_eval_seq"],
            ),
            "train_sequence_rows": int(proxy["train_sequence_rows"]),
            "val_sequence_rows": int(len(proxy["y_eval_seq"])),
            "val_positive_count": int(np.sum(proxy["y_eval_seq"] == 1)),
        },
        "attribution_window": {
            "rows": int(len(y_attr)),
            "positive_count": int(np.sum(y_attr == 1)),
            "negative_count": int(np.sum(y_attr == 0)),
            "ig_steps": int(config.ig_steps),
        },
        "top_features_overall": top_features(stats_frame, int(config.top_n_features)),
        "top_features_merged": filtered_merged_frame.head(int(config.top_n_features)).to_dict(orient="records"),
    }
    summary["artifacts"] = save_backend_outputs(dataset, model_type, stats_frame, filtered_merged_frame)
    Path(summary["artifacts"]["summary_json"]).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def analyze_backend(
    dataset: PreparedAttributionDataset,
    backend: str,
    config: BackendAttributionConfig,
) -> Dict[str, Any]:
    if backend == "xgboost":
        return analyze_xgboost_backend(dataset, config)
    if backend in {"tcn", "lstm"}:
        return analyze_sequence_backend(dataset, model_type=backend, config=config)
    raise ValueError(f"Unsupported backend: {backend}")


def run_backend_attribution(
    config: BackendAttributionConfig,
) -> Dict[str, Any]:
    prepared_root = Path(config.prepared_root)
    backends = normalize_backends(config.backends)
    targets = resolve_targets(prepared_root, config.targets)

    summary: Dict[str, Any] = {
        "prepared_root": str(prepared_root),
        "config": asdict(config),
        "targets": {},
    }

    for target_name in targets:
        dataset = load_prepared_dataset(
            prepared_root=prepared_root,
            target_name=target_name,
            max_features=int(config.max_features),
        )
        target_payload: Dict[str, Any] = {}
        for backend in backends:
            target_payload[backend] = analyze_backend(dataset, backend, config)
        summary["targets"][target_name] = target_payload

    summary_path = prepared_root / "backend_attribution_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "BackendAttributionConfig",
    "PreparedAttributionDataset",
    "SUPPORTED_BACKENDS",
    "run_backend_attribution",
]
