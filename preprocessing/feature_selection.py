from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from .config import PreprocessingConfig, TargetDatasetSpec
from .feature_importance import association_scores


def downcast_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return series
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_integer_dtype(series):
        return pd.to_numeric(series, downcast="integer")
    if pd.api.types.is_float_dtype(series):
        return pd.to_numeric(series, downcast="float")
    return series


def optimize_loaded_frame(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        df[column] = downcast_numeric_series(df[column])
    return df


def attach_source_row_idx(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "source_row_idx" not in frame.columns:
        frame.insert(0, "source_row_idx", np.arange(len(frame), dtype=np.int64))
        return frame

    source_row_idx = pd.to_numeric(frame["source_row_idx"], errors="coerce")
    if source_row_idx.isna().any():
        source_row_idx = pd.Series(np.arange(len(frame), dtype=np.int64), index=frame.index)
    frame["source_row_idx"] = source_row_idx.astype(np.int64, copy=False)
    return frame


def fallback_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded_prefixes = (
        "label_",
        "sample_weight",
        "entry_quality_",
        "label_quality_",
        "exclude_",
        "neg_ok_",
        "htf_confluence_",
        "bars_since_1h_swing_",
        "bars_since_30m_swing_",
        "concurrency_",
    )
    excluded_columns = {
        "datetime",
        "date",
        "time",
        "timestamp",
        "source_row_idx",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "warmup_mask",
        "is_anomaly",
        "atr",
        "structural_atr",
    }
    return [
        column
        for column in df.columns
        if column not in excluded_columns and not column.startswith(excluded_prefixes)
    ]


def resolve_feature_columns(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    config: PreprocessingConfig,
) -> List[str]:
    metadata_features = [
        column
        for column in metadata.get("feature_columns", [])
        if column in df.columns
    ]
    features = metadata_features or fallback_feature_columns(df)

    if config.include_base_price_columns:
        for column in ("open", "high", "low", "close", "volume"):
            if column in df.columns and column not in features:
                features.append(column)

    return list(dict.fromkeys(features))


def encode_candidate_features(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    encoded_parts: List[pd.Series] = []
    encoders: Dict[str, Dict[str, int]] = {}
    encoded_columns: List[str] = []

    for column in feature_columns:
        if column not in df.columns:
            continue

        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            encoded_parts.append(series.fillna(False).astype(np.int8).rename(column))
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            encoded_parts.append(downcast_numeric_series(numeric).rename(column))
            continue

        cleaned = series.fillna("__missing__").astype(str).str.strip()
        if cleaned.nunique(dropna=False) <= 1:
            encoded_parts.append(pd.Series(0, index=df.index, dtype=np.int8, name=column))
            continue

        encoder = LabelEncoder()
        encoded = pd.Series(encoder.fit_transform(cleaned), index=df.index, name=column)
        encoded_parts.append(downcast_numeric_series(encoded))
        encoders[column] = {
            value: int(code)
            for code, value in enumerate(encoder.classes_.tolist())
        }
        encoded_columns.append(column)

    encoded = pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=df.index)
    return encoded, {
        "encoders": encoders,
        "encoded_columns": encoded_columns,
    }


def analysis_probe_positions(n_rows: int, max_analysis_rows: int) -> np.ndarray:
    if n_rows <= 0:
        return np.array([], dtype=int)

    probe_rows = min(n_rows, max(1, int(max_analysis_rows)))
    if probe_rows == n_rows:
        return np.arange(n_rows, dtype=int)

    return np.unique(np.linspace(0, n_rows - 1, num=probe_rows, dtype=int))


def _column_signature(series: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(series, index=False).values.tobytes()
    return hashlib.blake2b(hashed, digest_size=16).hexdigest()


def remove_exact_duplicate_features(
    df: pd.DataFrame,
    max_analysis_rows: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty or df.shape[1] < 2:
        return df, {
            "probe_rows": int(len(df)),
            "removed_count": 0,
            "removed_columns": [],
        }

    probe_positions = analysis_probe_positions(len(df), max_analysis_rows)
    probe_frame = df.iloc[probe_positions]

    signatures: Dict[str, List[str]] = {}
    keep: List[str] = []
    removed: List[Dict[str, str]] = []

    for column in df.columns:
        signature = _column_signature(probe_frame[column])
        candidates = signatures.setdefault(signature, [])
        original = next((candidate for candidate in candidates if df[column].equals(df[candidate])), None)
        if original is not None:
            removed.append({"dropped": column, "kept": original})
            continue

        candidates.append(column)
        keep.append(column)

    return df.loc[:, keep], {
        "probe_rows": int(len(probe_positions)),
        "removed_count": len(removed),
        "removed_columns": removed,
    }


def remove_global_constant_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    nunique = df.nunique(dropna=False)
    constant_columns = nunique[nunique <= 1].index.tolist()
    kept = [column for column in df.columns if column not in constant_columns]
    return df.loc[:, kept], {
        "removed_count": len(constant_columns),
        "removed_columns": constant_columns,
    }


def build_target_context(
    df: pd.DataFrame,
    target_specs: List[TargetDatasetSpec],
    time_column: str,
) -> pd.DataFrame:
    required_columns: List[str] = []

    for column in (time_column, "warmup_mask"):
        if column in df.columns:
            required_columns.append(column)

    if "source_row_idx" in df.columns:
        required_columns.append("source_row_idx")

    for spec in target_specs:
        for column in (
            spec.target_column,
            spec.sample_weight_column,
            spec.quality_column,
            spec.exclude_column,
            spec.safe_negative_column,
        ):
            if column and column in df.columns:
                required_columns.append(column)

    return df.loc[:, list(dict.fromkeys(required_columns))].copy()


def discover_targets(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> List[TargetDatasetSpec]:
    specs: List[TargetDatasetSpec] = []
    for target_column in config.target_columns:
        if target_column not in df.columns:
            continue

        direction = "long" if "_long_" in target_column else "short"
        label_kind = "entry" if target_column.endswith("_entry") else "ote"

        sample_weight_column = f"sample_weight_{direction}"
        quality_column = f"label_quality_{direction}"
        if label_kind == "entry":
            sample_weight_column = f"sample_weight_entry_{direction}"
            quality_column = f"entry_quality_{direction}"

        specs.append(
            TargetDatasetSpec(
                name=f"{direction}_{label_kind}",
                target_column=target_column,
                direction=direction,
                label_kind=label_kind,
                sample_weight_column=sample_weight_column if sample_weight_column in df.columns else None,
                quality_column=quality_column if quality_column in df.columns else None,
                exclude_column=f"exclude_{direction}" if f"exclude_{direction}" in df.columns else None,
                safe_negative_column=f"neg_ok_{direction}" if f"neg_ok_{direction}" in df.columns else None,
            )
        )

    return specs


def detect_binary_target(target: pd.Series) -> bool:
    unique_values = set(pd.Series(target).dropna().unique().tolist())
    return unique_values.issubset({0, 1, 0.0, 1.0})


def resolve_sample_weight(
    sample_weight_source: Optional[pd.Series],
    positive_mask: pd.Series,
) -> pd.Series:
    if sample_weight_source is not None:
        weights = pd.to_numeric(sample_weight_source, errors="coerce").fillna(1.0)
    else:
        weights = pd.Series(1.0, index=positive_mask.index)

    weights = weights.clip(lower=0.0)
    weights = weights.where(weights > 0, 1.0)
    weights.loc[~positive_mask] = weights.loc[~positive_mask].replace(0.0, 1.0)
    return weights.astype(np.float32)


def ordered_usable_positions(
    df: pd.DataFrame,
    usable_mask: pd.Series,
    time_column: str,
) -> Tuple[np.ndarray, Optional[pd.Series]]:
    usable_positions = np.flatnonzero(usable_mask.to_numpy())
    if time_column not in df.columns or len(usable_positions) == 0:
        return usable_positions, None

    time_values = pd.to_datetime(
        df[time_column].iloc[usable_positions],
        errors="coerce",
    ).reset_index(drop=True)
    if time_values.is_monotonic_increasing:
        return usable_positions, time_values
    sort_order = time_values.sort_values(kind="stable").index.to_numpy(dtype=int, copy=False)
    ordered_positions = usable_positions[sort_order]
    ordered_time_values = time_values.iloc[sort_order].reset_index(drop=True)
    return ordered_positions, ordered_time_values


def build_split_indices(
    n_rows: int,
    config: PreprocessingConfig,
) -> Dict[str, np.ndarray]:
    if n_rows == 0:
        empty = np.array([], dtype=int)
        return {"train": empty, "val": empty, "test": empty}

    train_end = int(n_rows * config.train_size)
    val_end = int(n_rows * (config.train_size + config.val_size))

    if n_rows >= 3:
        train_end = min(max(train_end, 1), n_rows - 2)
        val_end = min(max(val_end, train_end + 1), n_rows - 1)
    else:
        train_end = n_rows
        val_end = n_rows

    if not config.use_time_based_split:
        order = np.random.default_rng(42).permutation(n_rows)
    else:
        order = np.arange(n_rows)

    return {
        "train": order[:train_end],
        "val": order[train_end:val_end],
        "test": order[val_end:n_rows],
    }


def build_fill_values(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if df.empty:
        return {}, {"columns_filled": 0, "strategies": {}}

    clean = df.replace([np.inf, -np.inf], np.nan)
    missing_mask = clean.isna()
    has_missing = missing_mask.any(axis=0)
    all_missing = missing_mask.all(axis=0)

    means = clean.mean(axis=0)
    medians = clean.median(axis=0)
    valid_counts = clean.notna().sum(axis=0)
    skews = clean.skew(axis=0).where(valid_counts > 2, other=0.0).fillna(0.0)

    fill_series = means.astype(np.float64, copy=True)
    fill_series.loc[all_missing] = 0.0

    use_median = has_missing & ~all_missing & (skews.abs() > 1.0)
    fill_series.loc[use_median] = medians.loc[use_median]
    fill_series = fill_series.fillna(0.0).astype(np.float32)

    strategies: Dict[str, str] = {}
    for column in clean.columns[has_missing]:
        if bool(all_missing.loc[column]):
            strategies[column] = "zero_all_missing"
        elif bool(use_median.loc[column]):
            strategies[column] = "median"
        else:
            strategies[column] = "mean"

    return (
        {column: float(value) for column, value in fill_series.items()},
        {
            "columns_filled": int(has_missing.sum()),
            "strategies": strategies,
        },
    )


def apply_fill_values(
    df: pd.DataFrame,
    fill_values: Dict[str, float],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    clean = df.replace([np.inf, -np.inf], np.nan)
    fill_series = pd.Series(fill_values, dtype=np.float32).reindex(clean.columns).fillna(np.float32(0.0))
    return clean.fillna(fill_series).astype(np.float32, copy=False)


def remove_low_variance_columns(
    df: pd.DataFrame,
    variance_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    variances = df.var()
    low_variance = variances[variances <= variance_threshold].index.tolist()
    kept = [column for column in df.columns if column not in low_variance]
    return df.loc[:, kept], {
        "removed_count": len(low_variance),
        "removed_columns": low_variance,
    }


def prune_collinear_features(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: PreprocessingConfig,
) -> Dict[str, Any]:
    if X_train.empty or X_train.shape[1] < 2:
        return {
            "dropped_columns": [],
            "dropped_pairs": [],
            "similarity_pairs": [],
            "max_remaining_correlation": 0.0,
        }

    analysis_rows = min(len(X_train), max(1, int(config.max_analysis_rows)))
    X_analysis = X_train.iloc[-analysis_rows:].copy()
    y_analysis = y_train.iloc[-analysis_rows:].copy()

    association = association_scores(X_analysis, y_analysis)
    ranking_scores = association.fillna(0.0).to_dict()

    corr = X_analysis.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs: List[Tuple[str, str, float]] = []
    similarity_pairs: List[Dict[str, Any]] = []
    for column in upper.columns:
        series = upper[column].dropna()
        hits = series[series >= config.correlation_threshold]
        for other, corr_value in hits.items():
            value = float(corr_value)
            pairs.append((other, column, value))
            if value >= config.similarity_threshold:
                similarity_pairs.append(
                    {
                        "feature_a": other,
                        "feature_b": column,
                        "correlation": value,
                    }
                )
    pairs.sort(key=lambda item: item[2], reverse=True)

    dropped: set[str] = set()
    drop_records: List[Dict[str, Any]] = []
    for left, right, corr_value in pairs:
        if left in dropped or right in dropped:
            continue

        left_score = float(ranking_scores.get(left, 0.0))
        right_score = float(ranking_scores.get(right, 0.0))

        if left_score > right_score:
            drop_feature, keep_feature = right, left
        elif right_score > left_score:
            drop_feature, keep_feature = left, right
        else:
            drop_feature, keep_feature = sorted([left, right])[-1], sorted([left, right])[0]

        dropped.add(drop_feature)
        drop_records.append(
            {
                "dropped": drop_feature,
                "kept": keep_feature,
                "correlation": float(corr_value),
                "dropped_score": float(ranking_scores.get(drop_feature, 0.0)),
                "kept_score": float(ranking_scores.get(keep_feature, 0.0)),
            }
        )

    remaining = [column for column in X_train.columns if column not in dropped]
    if len(remaining) >= 2:
        remaining_corr = X_analysis[remaining].corr().abs()
        remaining_upper = remaining_corr.where(np.triu(np.ones(remaining_corr.shape), k=1).astype(bool))
        max_remaining = float(remaining_upper.max().max()) if not remaining_upper.empty else 0.0
        if np.isnan(max_remaining):
            max_remaining = 0.0
    else:
        max_remaining = 0.0

    return {
        "analysis_rows": int(analysis_rows),
        "dropped_columns": sorted(dropped),
        "dropped_pairs": drop_records[:200],
        "similarity_pairs": similarity_pairs[:100],
        "max_remaining_correlation": max_remaining,
    }


def maybe_scale(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = None
    resolved_scaler = scaler_type.lower()

    if resolved_scaler == "robust":
        scaler = RobustScaler()
    elif resolved_scaler == "standard":
        scaler = StandardScaler()
    else:
        return None, X_train, X_val, X_test

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


__all__ = [
    "analysis_probe_positions",
    "apply_fill_values",
    "attach_source_row_idx",
    "build_fill_values",
    "build_split_indices",
    "build_target_context",
    "detect_binary_target",
    "discover_targets",
    "downcast_numeric_series",
    "encode_candidate_features",
    "maybe_scale",
    "optimize_loaded_frame",
    "ordered_usable_positions",
    "prune_collinear_features",
    "remove_exact_duplicate_features",
    "remove_global_constant_features",
    "remove_low_variance_columns",
    "resolve_feature_columns",
    "resolve_sample_weight",
]
