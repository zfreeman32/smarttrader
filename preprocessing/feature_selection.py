from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from .config import PreprocessingConfig, TargetDatasetSpec
from .feature_importance import association_scores


TARGET_HELPER_FAMILY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "": ("",),
    "ote": ("", "ote"),
    "reversal": ("reversal", ""),
    "continuation": ("continuation",),
    "continuation_pullback": ("continuation", "continuation_pullback"),
    "breakout": ("breakout",),
}

SYNTHETIC_TARGET_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "label_long_frvp_meta": {
        "name": "long_frvp_meta",
        "direction": "long",
        "label_kind": "frvp_meta",
        "component_target_columns": [
            "label_long_frvp_reversal",
            "label_long_frvp_continuation",
        ],
    },
    "label_short_frvp_meta": {
        "name": "short_frvp_meta",
        "direction": "short",
        "label_kind": "frvp_meta",
        "component_target_columns": [
            "label_short_frvp_reversal",
            "label_short_frvp_continuation",
        ],
    },
    "label_long_ote": {
        "name": "long_ote",
        "direction": "long",
        "label_kind": "ote",
        "component_target_columns": [
            "label_long_reversal",
            "label_long_continuation_pullback",
            "label_long_breakout",
            "label_long_frvp_reversal",
            "label_long_frvp_continuation",
        ],
    },
    "label_short_ote": {
        "name": "short_ote",
        "direction": "short",
        "label_kind": "ote",
        "component_target_columns": [
            "label_short_reversal",
            "label_short_continuation_pullback",
            "label_short_breakout",
            "label_short_frvp_reversal",
            "label_short_frvp_continuation",
        ],
    },
}


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


def resolve_source_row_idx(df: pd.DataFrame) -> pd.Series:
    if "source_row_idx" not in df.columns:
        return pd.Series(np.arange(len(df), dtype=np.int64), index=df.index, name="source_row_idx")

    source_series = df["source_row_idx"]
    if pd.api.types.is_integer_dtype(source_series) and not bool(source_series.isna().any()):
        resolved = source_series if source_series.dtype == np.int64 else source_series.astype(np.int64, copy=False)
        return resolved.rename("source_row_idx")

    source_row_idx = pd.to_numeric(source_series, errors="coerce")
    if source_row_idx.isna().any():
        source_row_idx = pd.Series(np.arange(len(df), dtype=np.int64), index=df.index, name="source_row_idx")
    return source_row_idx.astype(np.int64, copy=False).rename("source_row_idx")


def attach_source_row_idx(df: pd.DataFrame) -> pd.DataFrame:
    df["source_row_idx"] = resolve_source_row_idx(df).to_numpy(copy=False)
    return df


def fallback_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded_prefixes = (
        "label_",
        "sample_weight",
        "entry_quality_",
        "label_quality_",
        "exclude_",
        "neg_ok_",
        "bars_since_1h_swing_",
        "bars_since_30m_swing_",
        "concurrency_",
    )
    excluded_columns = {
        "datetime",
        "date",
        "time",
        "timestamp",
        "ts_event",
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
        "reversal_atr",
        "reversal_structural_atr",
        "continuation_atr",
        "continuation_structural_atr",
        "breakout_atr",
        "breakout_structural_atr",
        "frvp_atr",
        "frvp_structural_atr",
        "symbol",
        "instrument_id",
        "contract_id",
        "contract_symbol",
        "contract_expiration",
        "is_roll_boundary",
        "bars_since_roll",
        "market_day_close",
        "market_day_index",
        "in_roll_bracket",
    }
    return [
        column
        for column in df.columns
        if column not in excluded_columns and not column.startswith(excluded_prefixes)
    ]


def _parse_target_column(target_column: str) -> Optional[Tuple[str, str, bool]]:
    body = target_column.removeprefix("label_")
    if body == target_column:
        return None

    direction = None
    remainder = ""
    for candidate in ("long", "short"):
        prefix = f"{candidate}_"
        if body.startswith(prefix):
            direction = candidate
            remainder = body[len(prefix) :]
            break
    if direction is None or not remainder:
        return None

    is_entry = remainder == "entry" or remainder.endswith("_entry")
    family = remainder[: -len("_entry")].rstrip("_") if is_entry else remainder
    if not family and not is_entry:
        return None

    return direction, family, is_entry


def _target_context_feature_candidates(target_column: str) -> List[str]:
    parsed = _parse_target_column(target_column)
    if parsed is None:
        return []

    direction, family, _is_entry = parsed
    family_aliases = TARGET_HELPER_FAMILY_ALIASES.get(family, (family,))
    candidates: List[str] = []

    for alias in family_aliases:
        if alias:
            candidates.append(f"htf_confluence_{direction}_{alias}")
            candidates.append(f"htf_confluence_{direction}_{alias}_1hr")
        else:
            candidates.append(f"htf_confluence_{direction}")
            candidates.append(f"htf_confluence_{direction}_1hr")

    return list(dict.fromkeys(candidates))


def resolve_target_context_feature_columns(
    df: pd.DataFrame,
    target_specs: Iterable[TargetDatasetSpec],
) -> List[str]:
    columns: List[str] = []

    for spec in target_specs:
        target_columns = (
            spec.component_target_columns
            if spec.is_synthetic
            else [spec.target_column]
        )
        for target_column in target_columns:
            for candidate in _target_context_feature_candidates(str(target_column)):
                if candidate in df.columns:
                    columns.append(candidate)

    return list(dict.fromkeys(columns))


def resolve_feature_columns(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    config: PreprocessingConfig,
    *,
    target_specs: Optional[Iterable[TargetDatasetSpec]] = None,
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

    if target_specs is not None:
        for column in resolve_target_context_feature_columns(df, target_specs):
            if column not in features:
                features.append(column)

    return list(dict.fromkeys(features))


def encode_candidate_features(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    available_columns = [column for column in feature_columns if column in df.columns]
    encoded = df.loc[:, available_columns].copy(deep=False)
    encoders: Dict[str, Dict[str, int]] = {}
    encoded_columns: List[str] = []

    for column in available_columns:
        series = encoded[column]
        if pd.api.types.is_bool_dtype(series):
            continue

        if pd.api.types.is_numeric_dtype(series):
            optimized = downcast_numeric_series(pd.to_numeric(series, errors="coerce"))
            if optimized.dtype != series.dtype:
                encoded[column] = optimized.rename(column)
            continue

        cleaned = series.fillna("__missing__").astype(str).str.strip()
        if cleaned.nunique(dropna=False) <= 1:
            encoded[column] = pd.Series(0, index=df.index, dtype=np.int8, name=column)
            continue

        encoder = LabelEncoder()
        encoded_series = pd.Series(encoder.fit_transform(cleaned), index=df.index, name=column)
        encoded[column] = downcast_numeric_series(encoded_series)
        encoders[column] = {
            value: int(code)
            for code, value in enumerate(encoder.classes_.tolist())
        }
        encoded_columns.append(column)

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
    constant_columns: List[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_float_dtype(series):
            values = series.to_numpy(copy=False)
            if np.isinf(values).any():
                series = series.replace([np.inf, -np.inf], np.nan)
        if int(series.nunique(dropna=False)) <= 1:
            constant_columns.append(column)
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
        for column in spec.required_columns():
            if column and column in df.columns:
                required_columns.append(column)

    return df.loc[:, list(dict.fromkeys(required_columns))].copy()


def _build_direct_target_spec(
    df: pd.DataFrame,
    target_column: str,
) -> Optional[TargetDatasetSpec]:
    if target_column not in df.columns:
        return None

    parsed = _parse_target_column(target_column)
    if parsed is None:
        return None

    direction, family, is_entry = parsed
    body = target_column.removeprefix("label_")
    label_kind = f"{family}_entry" if is_entry and family else ("entry" if is_entry else family)

    family_aliases = TARGET_HELPER_FAMILY_ALIASES.get(family, (family,))

    def first_existing(candidates: List[str]) -> Optional[str]:
        for column in candidates:
            if column in df.columns:
                return column
        return None

    def helper_candidates(prefix: str) -> List[str]:
        candidates: List[str] = []
        for alias in family_aliases:
            if alias:
                candidates.append(f"{prefix}_{direction}_{alias}")
            else:
                candidates.append(f"{prefix}_{direction}")
        return list(dict.fromkeys(candidates))

    sample_weight_prefix = "sample_weight_entry" if is_entry else "sample_weight"
    quality_prefix = "entry_quality" if is_entry else "label_quality"
    sample_weight_column = first_existing(helper_candidates(sample_weight_prefix))
    quality_column = first_existing(helper_candidates(quality_prefix))
    exclude_column = first_existing(helper_candidates("exclude"))
    safe_negative_column = first_existing(helper_candidates("neg_ok"))

    return TargetDatasetSpec(
        name=body,
        target_column=target_column,
        direction=direction,
        label_kind=label_kind,
        sample_weight_column=sample_weight_column,
        quality_column=quality_column,
        exclude_column=exclude_column,
        safe_negative_column=safe_negative_column,
    )


def _build_synthetic_target_spec(
    synthetic_target_column: str,
    component_specs: List[TargetDatasetSpec],
) -> Optional[TargetDatasetSpec]:
    synthetic_config = SYNTHETIC_TARGET_COMPONENTS.get(synthetic_target_column)
    if synthetic_config is None:
        return None

    required_helper_attrs = (
        "sample_weight_column",
        "exclude_column",
        "safe_negative_column",
    )
    for attr_name in required_helper_attrs:
        if any(getattr(spec, attr_name) is None for spec in component_specs):
            return None

    quality_columns = [
        spec.quality_column
        for spec in component_specs
        if spec.quality_column is not None
    ]

    return TargetDatasetSpec(
        name=str(synthetic_config["name"]),
        target_column=synthetic_target_column,
        direction=str(synthetic_config["direction"]),
        label_kind=str(synthetic_config["label_kind"]),
        component_target_columns=[spec.target_column for spec in component_specs],
        component_sample_weight_columns=[
            str(spec.sample_weight_column)
            for spec in component_specs
            if spec.sample_weight_column is not None
        ],
        component_quality_columns=[str(column) for column in quality_columns],
        component_exclude_columns=[
            str(spec.exclude_column)
            for spec in component_specs
            if spec.exclude_column is not None
        ],
        component_safe_negative_columns=[
            str(spec.safe_negative_column)
            for spec in component_specs
            if spec.safe_negative_column is not None
        ],
    )


def discover_targets(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> List[TargetDatasetSpec]:
    specs: List[TargetDatasetSpec] = []
    seen_names: set[str] = set()
    direct_specs: Dict[str, Optional[TargetDatasetSpec]] = {}

    def get_direct_spec(target_column: str) -> Optional[TargetDatasetSpec]:
        if target_column not in direct_specs:
            direct_specs[target_column] = _build_direct_target_spec(df, target_column)
        return direct_specs[target_column]

    for target_column in config.target_columns:
        spec = get_direct_spec(target_column)
        if spec is None and target_column in SYNTHETIC_TARGET_COMPONENTS:
            component_target_columns = SYNTHETIC_TARGET_COMPONENTS[target_column]["component_target_columns"]
            component_specs = [
                component_spec
                for component_spec in (get_direct_spec(str(column)) for column in component_target_columns)
                if component_spec is not None
            ]
            if component_specs:
                spec = _build_synthetic_target_spec(
                    target_column,
                    component_specs,
                )

        if spec is None or spec.name in seen_names:
            continue
        seen_names.add(spec.name)
        specs.append(spec)

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

    strategies: Dict[str, str] = {}
    fill_values: Dict[str, float] = {}
    columns_filled = 0

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_float_dtype(series):
            values = series.to_numpy(copy=False)
            if np.isinf(values).any():
                series = series.replace([np.inf, -np.inf], np.nan)

        if not bool(series.isna().any()):
            continue

        columns_filled += 1
        valid = series.dropna()
        if valid.empty:
            fill_values[column] = 0.0
            strategies[column] = "zero_all_missing"
            continue

        skew = float(valid.skew()) if len(valid) > 2 else 0.0
        if np.isnan(skew):
            skew = 0.0

        if abs(skew) > 1.0:
            fill_values[column] = float(valid.median())
            strategies[column] = "median"
        else:
            fill_values[column] = float(valid.mean())
            strategies[column] = "mean"

    return (
        fill_values,
        {
            "columns_filled": int(columns_filled),
            "strategies": strategies,
        },
    )


def apply_fill_values(
    df: pd.DataFrame,
    fill_values: Dict[str, float],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    filled = df.copy(deep=False)

    for column in df.columns:
        series = df[column]
        clean = series
        if pd.api.types.is_float_dtype(series):
            values = series.to_numpy(copy=False)
            if np.isinf(values).any():
                clean = series.replace([np.inf, -np.inf], np.nan)

        if not bool(clean.isna().any()):
            continue

        fill_value = fill_values.get(column, 0.0)
        repaired = clean.fillna(fill_value)
        filled[column] = downcast_numeric_series(pd.to_numeric(repaired, errors="coerce")).rename(column)

    return filled


def remove_low_variance_columns(
    df: pd.DataFrame,
    variance_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    low_variance: List[str] = []
    for column in df.columns:
        variance = float(df[column].var())
        if np.isnan(variance):
            variance = 0.0
        if variance <= variance_threshold:
            low_variance.append(column)
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
    "resolve_source_row_idx",
    "resolve_feature_columns",
    "resolve_target_context_feature_columns",
    "resolve_sample_weight",
]
