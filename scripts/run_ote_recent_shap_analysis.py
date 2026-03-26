from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_INPUT_FILE = Path("data/features/eurusd_5min_ote_full.csv")
DEFAULT_PREPARED_ROOT = Path("data/prepared/eurusd_5min_ote_full")
DEFAULT_OUTPUT_DIR = Path("features/shap_outputs/ote_recent_300k")


@dataclass(frozen=True)
class TargetSpec:
    name: str
    target_column: str
    direction: str
    sample_weight_column: str
    safe_negative_column: str
    exclude_column: str


TARGET_SPECS: Dict[str, TargetSpec] = {
    "long_ote": TargetSpec(
        name="long_ote",
        target_column="label_long_ote",
        direction="long",
        sample_weight_column="sample_weight_long",
        safe_negative_column="neg_ok_long",
        exclude_column="exclude_long",
    ),
    "short_ote": TargetSpec(
        name="short_ote",
        target_column="label_short_ote",
        direction="short",
        sample_weight_column="sample_weight_short",
        safe_negative_column="neg_ok_short",
        exclude_column="exclude_short",
    ),
}


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


def print_step(message: str) -> None:
    print(f"[ote-shap] {message}")


def load_ranked_features(target_dir: Path, max_features: int) -> List[str]:
    features_path = target_dir / "features.json"
    ranking_path = target_dir / "feature_importance.csv"

    payload = json.loads(features_path.read_text(encoding="utf-8"))
    raw_features = list(payload.get("features", []))

    ranked_features = list(raw_features)
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        if "feature" in ranking.columns:
            sort_column = "composite_score" if "composite_score" in ranking.columns else None
            if sort_column is not None:
                ranking = ranking.sort_values(sort_column, ascending=False)
            ranked_features = [
                feature
                for feature in ranking["feature"].tolist()
                if feature in raw_features
            ]

    seen = set()
    deduped: List[str] = []
    for feature in ranked_features:
        if feature in seen:
            continue
        seen.add(feature)
        deduped.append(feature)
        if max_features > 0 and len(deduped) >= max_features:
            break

    if max_features <= 0:
        return deduped
    return deduped[:max_features]


def validate_targets(targets: Sequence[str]) -> List[str]:
    invalid = [target for target in targets if target not in TARGET_SPECS]
    if invalid:
        supported = ", ".join(sorted(TARGET_SPECS))
        raise ValueError(f"Unsupported target(s): {invalid}. Supported values: {supported}")
    return list(dict.fromkeys(targets))


def available_columns(input_file: Path) -> List[str]:
    return pd.read_csv(input_file, nrows=0).columns.tolist()


def build_usecols(
    all_columns: Sequence[str],
    feature_columns: Sequence[str],
    spec: TargetSpec,
    time_column: str,
) -> List[str]:
    required = [
        time_column,
        "warmup_mask",
        spec.target_column,
        spec.sample_weight_column,
        spec.safe_negative_column,
        spec.exclude_column,
    ]
    missing_required = [column for column in required if column not in all_columns]
    if missing_required:
        raise ValueError(
            f"Input file is missing required columns for {spec.name}: {missing_required}"
        )

    present_features = [column for column in feature_columns if column in all_columns]
    if not present_features:
        raise ValueError(
            f"No ranked features for {spec.name} were found in the input dataset."
        )

    return required + present_features


def usable_mask_for_chunk(chunk: pd.DataFrame, spec: TargetSpec) -> pd.Series:
    target = pd.to_numeric(chunk[spec.target_column], errors="coerce").fillna(0.0)
    positive_mask = target > 0.0
    warmup_mask = chunk["warmup_mask"].fillna(False).astype(bool)
    safe_negative_mask = chunk[spec.safe_negative_column].fillna(False).astype(bool)
    exclude_mask = chunk[spec.exclude_column].fillna(False).astype(bool)
    negative_mask = (~positive_mask) & safe_negative_mask & (~exclude_mask)
    return (~warmup_mask) & (positive_mask | negative_mask)


def trim_to_recent_rows(frames: List[pd.DataFrame], recent_rows: int) -> List[pd.DataFrame]:
    total_rows = sum(len(frame) for frame in frames)
    while frames and total_rows > recent_rows:
        overflow = total_rows - recent_rows
        first = frames[0]
        if len(first) <= overflow:
            total_rows -= len(first)
            frames.pop(0)
            continue

        frames[0] = first.iloc[overflow:].reset_index(drop=True)
        total_rows -= overflow
    return frames


def load_recent_usable_rows(
    input_file: Path,
    feature_columns: Sequence[str],
    spec: TargetSpec,
    *,
    time_column: str,
    recent_rows: int,
    chunksize: int,
) -> pd.DataFrame:
    all_columns = available_columns(input_file)
    usecols = build_usecols(all_columns, feature_columns, spec, time_column)
    resolved_features = [column for column in feature_columns if column in usecols]
    kept_frames: List[pd.DataFrame] = []

    for chunk in pd.read_csv(input_file, usecols=usecols, chunksize=chunksize):
        usable_mask = usable_mask_for_chunk(chunk, spec)
        filtered = chunk.loc[usable_mask].copy()
        if filtered.empty:
            continue

        filtered[resolved_features] = (
            filtered[resolved_features]
            .apply(pd.to_numeric, errors="coerce")
            .astype(np.float32)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        filtered[spec.target_column] = (
            pd.to_numeric(filtered[spec.target_column], errors="coerce")
            .fillna(0.0)
            .astype(np.uint8)
        )
        filtered[spec.sample_weight_column] = (
            pd.to_numeric(filtered[spec.sample_weight_column], errors="coerce")
            .fillna(1.0)
            .clip(lower=0.0)
            .replace(0.0, 1.0)
            .astype(np.float32)
        )
        kept_frames.append(filtered[[time_column, spec.target_column, spec.sample_weight_column] + resolved_features])
        trim_to_recent_rows(kept_frames, recent_rows)

    if not kept_frames:
        raise ValueError(f"No usable rows were found for {spec.name} in {input_file}.")

    recent = pd.concat(kept_frames, ignore_index=True)
    recent[time_column] = pd.to_datetime(recent[time_column], errors="coerce")
    recent = recent.dropna(subset=[time_column]).sort_values(time_column, kind="stable").tail(recent_rows)
    recent = recent.reset_index(drop=True)

    if recent.empty:
        raise ValueError(f"No timestamped usable rows remained for {spec.name}.")

    return recent


def prune_constant_features(frame: pd.DataFrame, feature_columns: Sequence[str]) -> List[str]:
    nunique = frame.loc[:, feature_columns].nunique(dropna=False)
    kept = [column for column in feature_columns if int(nunique.get(column, 0)) > 1]
    if not kept:
        raise ValueError("All requested features are constant in the selected recent window.")
    return kept


def split_recent_window(
    frame: pd.DataFrame,
    spec: TargetSpec,
    feature_columns: Sequence[str],
    train_fraction: float,
) -> Dict[str, np.ndarray]:
    n_rows = len(frame)
    split_at = int(n_rows * train_fraction)
    split_at = max(1000, split_at)
    split_at = min(split_at, n_rows - 1)
    if split_at <= 0 or split_at >= n_rows:
        raise ValueError(
            f"Could not create a train/eval split for {spec.name} from {n_rows} recent rows."
        )

    train_frame = frame.iloc[:split_at]
    eval_frame = frame.iloc[split_at:]

    X_train = np.ascontiguousarray(train_frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True))
    y_train = np.ascontiguousarray(train_frame[spec.target_column].to_numpy(dtype=np.uint8, copy=True))
    w_train = np.ascontiguousarray(train_frame[spec.sample_weight_column].to_numpy(dtype=np.float32, copy=True))

    X_eval = np.ascontiguousarray(eval_frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True))
    y_eval = np.ascontiguousarray(eval_frame[spec.target_column].to_numpy(dtype=np.uint8, copy=True))
    w_eval = np.ascontiguousarray(eval_frame[spec.sample_weight_column].to_numpy(dtype=np.float32, copy=True))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "w_train": w_train,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "w_eval": w_eval,
    }


def train_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    w_eval: np.ndarray,
    feature_names: Sequence[str],
    *,
    num_boost_round: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    min_child_weight: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    reg_alpha: float,
    random_seed: int,
    nthread: int,
) -> xgb.Booster:
    positive_weight = float(np.sum(w_train[y_train == 1]))
    negative_weight = float(np.sum(w_train[y_train == 0]))
    scale_pos_weight = negative_weight / max(positive_weight, 1e-6)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, feature_names=list(feature_names))
    deval = xgb.DMatrix(X_eval, label=y_eval, weight=w_eval, feature_names=list(feature_names))

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["aucpr", "logloss"],
        "tree_method": "hist",
        "max_depth": int(max_depth),
        "eta": float(learning_rate),
        "min_child_weight": float(min_child_weight),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "lambda": float(reg_lambda),
        "alpha": float(reg_alpha),
        "scale_pos_weight": float(scale_pos_weight),
        "seed": int(random_seed),
        "verbosity": 0,
    }
    if nthread > 0:
        params["nthread"] = int(nthread)

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtrain, "train"), (deval, "eval")],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )


def best_iteration_range(booster: xgb.Booster) -> tuple[int, int] | None:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None or best_iteration < 0:
        return None
    return (0, int(best_iteration) + 1)


def predict_probabilities(
    booster: xgb.Booster,
    X: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    dmatrix = xgb.DMatrix(X, feature_names=list(feature_names))
    kwargs = {}
    iteration_range = best_iteration_range(booster)
    if iteration_range is not None:
        kwargs["iteration_range"] = iteration_range
    return booster.predict(dmatrix, **kwargs).astype(np.float32, copy=False)


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


def update_shap_stats(
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


def compute_shap_feature_stats(
    booster: xgb.Booster,
    frame: pd.DataFrame,
    spec: TargetSpec,
    feature_columns: Sequence[str],
    chunk_size: int,
) -> pd.DataFrame:
    stats = build_empty_stats(len(feature_columns))

    iteration_range = best_iteration_range(booster)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        chunk = frame.iloc[start:stop]
        X_chunk = np.ascontiguousarray(chunk.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True))
        y_chunk = np.ascontiguousarray(chunk[spec.target_column].to_numpy(dtype=np.uint8, copy=True))
        dmatrix = xgb.DMatrix(X_chunk, feature_names=list(feature_columns))
        predict_kwargs = {"pred_contribs": True}
        if iteration_range is not None:
            predict_kwargs["iteration_range"] = iteration_range
        contributions = booster.predict(dmatrix, **predict_kwargs)
        contributions = np.asarray(contributions[:, :-1], dtype=np.float32)
        update_shap_stats(stats, contributions=contributions, feature_values=X_chunk, labels=y_chunk)

    return finalize_feature_stats(feature_columns, stats)


def top_features(payload: pd.DataFrame, sort_column: str, limit: int) -> List[Dict[str, float | str]]:
    subset = payload.sort_values(sort_column, ascending=False).head(limit)
    records: List[Dict[str, float | str]] = []
    for row in subset.itertuples(index=False):
        records.append(
            {
                "feature": row.feature,
                "mean_abs_shap_all": float(row.mean_abs_shap_all),
                "mean_signed_shap_all": float(row.mean_signed_shap_all),
                "mean_abs_shap_positive": float(row.mean_abs_shap_positive),
                "mean_signed_shap_positive": float(row.mean_signed_shap_positive),
                "mean_abs_shap_negative": float(row.mean_abs_shap_negative),
                "mean_signed_shap_negative": float(row.mean_signed_shap_negative),
            }
        )
    return records


def save_bar_chart(
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    output_path: Path,
    top_n: int,
    color: str,
) -> None:
    top = frame.sort_values(value_column, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(top["feature"], top[value_column], color=color)
    ax.set_title(title)
    ax.set_xlabel(value_column)
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def analyze_target(
    *,
    input_file: Path,
    prepared_root: Path,
    output_dir: Path,
    spec: TargetSpec,
    recent_rows: int,
    max_features: int,
    chunksize: int,
    shap_chunk_size: int,
    time_column: str,
    train_fraction: float,
    num_boost_round: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    min_child_weight: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    reg_alpha: float,
    top_n: int,
    random_seed: int,
    nthread: int,
) -> Dict[str, object]:
    target_dir = prepared_root / spec.name
    if not target_dir.exists():
        raise FileNotFoundError(f"Prepared target directory not found: {target_dir}")

    ranked_features = load_ranked_features(target_dir, max_features=max_features)
    print_step(
        f"{spec.name}: loading last {recent_rows:,} usable rows with {len(ranked_features)} ranked features"
    )
    recent = load_recent_usable_rows(
        input_file=input_file,
        feature_columns=ranked_features,
        spec=spec,
        time_column=time_column,
        recent_rows=recent_rows,
        chunksize=chunksize,
    )
    feature_columns = prune_constant_features(recent, [column for column in ranked_features if column in recent.columns])

    split_payload = split_recent_window(
        frame=recent,
        spec=spec,
        feature_columns=feature_columns,
        train_fraction=train_fraction,
    )

    train_positives = int(np.sum(split_payload["y_train"] == 1))
    eval_positives = int(np.sum(split_payload["y_eval"] == 1))
    if train_positives == 0 or eval_positives == 0:
        raise ValueError(
            f"{spec.name}: not enough positives in the recent window "
            f"(train={train_positives}, eval={eval_positives})."
        )

    print_step(
        f"{spec.name}: training XGBoost on {len(split_payload['y_train']):,} train rows "
        f"and {len(split_payload['y_eval']):,} eval rows"
    )
    booster = train_xgboost_classifier(
        X_train=split_payload["X_train"],
        y_train=split_payload["y_train"],
        w_train=split_payload["w_train"],
        X_eval=split_payload["X_eval"],
        y_eval=split_payload["y_eval"],
        w_eval=split_payload["w_eval"],
        feature_names=feature_columns,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        random_seed=random_seed,
        nthread=nthread,
    )

    eval_probabilities = predict_probabilities(
        booster=booster,
        X=split_payload["X_eval"],
        feature_names=feature_columns,
    )
    metrics = {
        "average_precision": safe_average_precision(
            split_payload["y_eval"],
            eval_probabilities,
            sample_weight=split_payload["w_eval"],
        ),
        "roc_auc": safe_roc_auc(
            split_payload["y_eval"],
            eval_probabilities,
            sample_weight=split_payload["w_eval"],
        ),
        "train_rows": int(len(split_payload["y_train"])),
        "eval_rows": int(len(split_payload["y_eval"])),
        "train_positive_count": train_positives,
        "eval_positive_count": eval_positives,
        "best_iteration": int(getattr(booster, "best_iteration", -1)),
    }

    print_step(f"{spec.name}: computing TreeSHAP feature contributions")
    shap_stats = compute_shap_feature_stats(
        booster=booster,
        frame=recent,
        spec=spec,
        feature_columns=feature_columns,
        chunk_size=shap_chunk_size,
    )

    target_output_dir = output_dir / spec.name
    target_output_dir.mkdir(parents=True, exist_ok=True)

    shap_stats_path = target_output_dir / "shap_feature_stats.csv"
    shap_stats.to_csv(shap_stats_path, index=False)

    save_bar_chart(
        frame=shap_stats,
        value_column="mean_abs_shap_all",
        title=f"{spec.name} | mean_abs_shap_all | recent {len(recent):,} usable rows",
        output_path=target_output_dir / "top_mean_abs_shap_all.png",
        top_n=top_n,
        color="#0b6e4f",
    )
    save_bar_chart(
        frame=shap_stats,
        value_column="mean_abs_shap_positive",
        title=f"{spec.name} | mean_abs_shap_positive | recent positive rows",
        output_path=target_output_dir / "top_mean_abs_shap_positive.png",
        top_n=top_n,
        color="#b85c38",
    )

    report = {
        "target": spec.name,
        "target_column": spec.target_column,
        "input_file": str(input_file),
        "prepared_root": str(prepared_root),
        "output_dir": str(target_output_dir),
        "recent_window": {
            "rows": int(len(recent)),
            "start": recent[time_column].iloc[0].isoformat(),
            "end": recent[time_column].iloc[-1].isoformat(),
            "positive_count": int(recent[spec.target_column].sum()),
            "negative_count": int((recent[spec.target_column] == 0).sum()),
            "positive_rate": float(recent[spec.target_column].mean()),
        },
        "model_metrics": metrics,
        "feature_config": {
            "requested_ranked_features": int(len(ranked_features)),
            "used_features_after_recent_window_pruning": int(len(feature_columns)),
            "recent_rows_requested": int(recent_rows),
            "train_fraction": float(train_fraction),
        },
        "artifacts": {
            "shap_feature_stats_csv": str(shap_stats_path),
            "top_mean_abs_shap_all_png": str(target_output_dir / "top_mean_abs_shap_all.png"),
            "top_mean_abs_shap_positive_png": str(target_output_dir / "top_mean_abs_shap_positive.png"),
        },
        "top_features_overall": top_features(shap_stats, "mean_abs_shap_all", top_n),
        "top_features_positive_rows": top_features(shap_stats, "mean_abs_shap_positive", top_n),
    }

    report_path = target_output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run XGBoost TreeSHAP analysis on the most recent usable OTE rows for "
            "long_ote and short_ote separately."
        )
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", nargs="+", default=["long_ote", "short_ote"])
    parser.add_argument("--time-column", default="datetime")
    parser.add_argument("--recent-rows", type=int, default=300_000)
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Top ranked prepared features to use per target. Use 0 for all prepared features.",
    )
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--shap-chunk-size", type=int, default=20_000)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--num-boost-round", type=int, default=400)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--nthread", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = validate_targets(args.targets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_summary = {
        "input_file": str(args.input_file),
        "prepared_root": str(args.prepared_root),
        "output_dir": str(args.output_dir),
        "targets": {},
        "config": {
            "recent_rows": int(args.recent_rows),
            "max_features": int(args.max_features),
            "train_fraction": float(args.train_fraction),
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "read_chunk_size": int(args.read_chunk_size),
            "shap_chunk_size": int(args.shap_chunk_size),
            "random_seed": int(args.random_seed),
        },
    }

    for target_name in targets:
        report = analyze_target(
            input_file=args.input_file,
            prepared_root=args.prepared_root,
            output_dir=args.output_dir,
            spec=TARGET_SPECS[target_name],
            recent_rows=args.recent_rows,
            max_features=args.max_features,
            chunksize=args.read_chunk_size,
            shap_chunk_size=args.shap_chunk_size,
            time_column=args.time_column,
            train_fraction=args.train_fraction,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            top_n=args.top_n,
            random_seed=args.random_seed,
            nthread=args.nthread,
        )
        combined_summary["targets"][target_name] = report

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")
    print_step(f"Saved combined summary to {summary_path}")


if __name__ == "__main__":
    main()
