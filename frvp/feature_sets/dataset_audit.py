from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.config import FRVP_DIRECT_TARGET_COLUMNS, PreprocessingConfig
from preprocessing.feature_importance import compute_feature_importance
from preprocessing.feature_selection import discover_targets, resolve_sample_weight


def summarize_frvp_feature_dataset(
    df: pd.DataFrame,
    *,
    correlation_threshold: float = 0.98,
    warmup_column: str = "warmup_mask",
    top_n_mi: int = 20,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize FRVP feature coverage and high-correlation pairs for Phase 2 audits."""

    frvp_columns = [column for column in df.columns if column.startswith("frvp_")]
    carry_through_columns = [column for column in df.columns if column.startswith("htf_confluence_")]
    numeric_columns = [
        column
        for column in frvp_columns + carry_through_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    coverage = {
        column: {
            "non_null_fraction": float(pd.to_numeric(df[column], errors="coerce").notna().mean()),
            "null_count": int(pd.to_numeric(df[column], errors="coerce").isna().sum()),
        }
        for column in frvp_columns + carry_through_columns
    }

    correlation_pairs: list[dict[str, Any]] = []
    max_abs_correlation = 0.0
    for left, right in combinations(numeric_columns, 2):
        pair = df.loc[:, [left, right]].apply(pd.to_numeric, errors="coerce")
        pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 3:
            continue
        corr = float(pair[left].corr(pair[right]))
        if not np.isfinite(corr):
            continue
        abs_corr = abs(corr)
        max_abs_correlation = max(max_abs_correlation, abs_corr)
        if abs_corr >= float(correlation_threshold):
            correlation_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "correlation": corr,
                    "abs_correlation": abs_corr,
                    "rows_used": int(len(pair)),
                }
            )

    correlation_pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    setup_type_distribution: dict[str, int] = {}
    if "frvp_setup_type" in df.columns:
        counts = pd.Series(df["frvp_setup_type"]).dropna().astype(int).value_counts().sort_index()
        setup_type_distribution = {str(int(index)): int(value) for index, value in counts.items()}

    summary = {
        "rows": int(len(df)),
        "datetime_start": _iso_or_none(df["datetime"].min()) if "datetime" in df.columns else None,
        "datetime_end": _iso_or_none(df["datetime"].max()) if "datetime" in df.columns else None,
        "frvp_column_count": len(frvp_columns),
        "frvp_numeric_column_count": len(numeric_columns),
        "carry_through_column_count": len(carry_through_columns),
        "correlation_threshold": float(correlation_threshold),
        "max_abs_correlation": float(max_abs_correlation),
        "high_correlation_pairs": correlation_pairs,
        "coverage": coverage,
        "setup_type_distribution": setup_type_distribution,
        "warmup_rows": int(df[warmup_column].fillna(False).astype(bool).sum()) if warmup_column in df.columns else 0,
        "duplicate_time_columns": [
            column for column in ("ts_event", "timestamp") if column in df.columns
        ],
        "roll_lineage_columns_present": [
            column
            for column in (
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
            )
            if column in df.columns
        ],
        "mi_rankings": _mi_rankings(df, frvp_columns + carry_through_columns, top_n=top_n_mi),
        "key_feature_mi": _key_feature_mi(df, frvp_columns + carry_through_columns),
    }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_make_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    return str(value)


def _mi_rankings(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    feature_columns = [column for column in feature_columns if column in df.columns]
    if not feature_columns:
        return {}

    config = PreprocessingConfig(
        target_columns=list(FRVP_DIRECT_TARGET_COLUMNS),
        max_analysis_rows=100_000,
        min_positive_samples=1,
        min_train_rows=1,
        min_usable_rows=1,
    )
    targets = discover_targets(df, config)
    payload: dict[str, list[dict[str, Any]]] = {}
    for spec in targets:
        if spec.target_column not in df.columns:
            continue
        target = pd.to_numeric(df[spec.target_column], errors="coerce").fillna(0).astype(int)
        positive_mask = target.eq(1)
        negative_mask = target.eq(0)
        if spec.safe_negative_column and spec.safe_negative_column in df.columns:
            negative_mask &= df[spec.safe_negative_column].fillna(False).astype(bool)
        if spec.exclude_column and spec.exclude_column in df.columns:
            negative_mask &= ~df[spec.exclude_column].fillna(False).astype(bool)
        if "warmup_mask" in df.columns:
            warmup_mask = df["warmup_mask"].fillna(False).astype(bool)
        else:
            warmup_mask = pd.Series(False, index=df.index, dtype=bool)
        usable_mask = ~warmup_mask & (positive_mask | negative_mask)
        X = df.loc[usable_mask, feature_columns].copy()
        if X.empty or target.loc[usable_mask].nunique(dropna=True) < 2:
            payload[spec.name] = []
            continue
        numeric = X.apply(pd.to_numeric, errors="coerce")
        sample_weight_source = (
            df[spec.sample_weight_column]
            if spec.sample_weight_column and spec.sample_weight_column in df.columns
            else None
        )
        weights = resolve_sample_weight(sample_weight_source, positive_mask).loc[usable_mask]
        importance_df, _summary = compute_feature_importance(
            X_train=numeric,
            y_train=target.loc[usable_mask].reset_index(drop=True),
            sample_weight=weights.reset_index(drop=True),
            is_binary=True,
            config=config,
        )
        payload[spec.name] = importance_df.head(int(top_n)).to_dict(orient="records")
    return payload


def _key_feature_mi(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, dict[str, float | None]]:
    feature_columns = [column for column in feature_columns if column in df.columns]
    if not feature_columns:
        return {}

    config = PreprocessingConfig(
        target_columns=list(FRVP_DIRECT_TARGET_COLUMNS),
        max_analysis_rows=100_000,
        min_positive_samples=1,
        min_train_rows=1,
        min_usable_rows=1,
    )
    targets = discover_targets(df, config)
    payload: dict[str, dict[str, float | None]] = {}
    for spec in targets:
        if spec.target_column not in df.columns:
            continue
        target = pd.to_numeric(df[spec.target_column], errors="coerce").fillna(0).astype(int)
        positive_mask = target.eq(1)
        negative_mask = target.eq(0)
        if spec.safe_negative_column and spec.safe_negative_column in df.columns:
            negative_mask &= df[spec.safe_negative_column].fillna(False).astype(bool)
        if spec.exclude_column and spec.exclude_column in df.columns:
            negative_mask &= ~df[spec.exclude_column].fillna(False).astype(bool)
        if "warmup_mask" in df.columns:
            warmup_mask = df["warmup_mask"].fillna(False).astype(bool)
        else:
            warmup_mask = pd.Series(False, index=df.index, dtype=bool)
        usable_mask = ~warmup_mask & (positive_mask | negative_mask)
        X = df.loc[usable_mask, feature_columns].copy()
        if X.empty or target.loc[usable_mask].nunique(dropna=True) < 2:
            payload[spec.name] = {
                "frvp_distance_max_mi": None,
                "frvp_open_type_mi": None,
                "htf_confluence_max_mi": None,
            }
            continue
        numeric = X.apply(pd.to_numeric, errors="coerce")
        sample_weight_source = (
            df[spec.sample_weight_column]
            if spec.sample_weight_column and spec.sample_weight_column in df.columns
            else None
        )
        weights = resolve_sample_weight(sample_weight_source, positive_mask).loc[usable_mask]
        importance_df, _summary = compute_feature_importance(
            X_train=numeric,
            y_train=target.loc[usable_mask].reset_index(drop=True),
            sample_weight=weights.reset_index(drop=True),
            is_binary=True,
            config=config,
        )
        importance_df = importance_df.copy()
        if "feature" not in importance_df.columns or "mutual_information" not in importance_df.columns:
            payload[spec.name] = {
                "frvp_distance_max_mi": None,
                "frvp_open_type_mi": None,
                "htf_confluence_max_mi": None,
            }
            continue
        mi_series = pd.to_numeric(importance_df["mutual_information"], errors="coerce").fillna(0.0)
        payload[spec.name] = {
            "frvp_distance_max_mi": float(
                mi_series.loc[importance_df["feature"].astype(str).str.startswith("frvp_dist_")].max()
            )
            if importance_df["feature"].astype(str).str.startswith("frvp_dist_").any()
            else None,
            "frvp_open_type_mi": float(
                mi_series.loc[importance_df["feature"].astype(str).eq("frvp_open_type")].max()
            )
            if importance_df["feature"].astype(str).eq("frvp_open_type").any()
            else None,
            "htf_confluence_max_mi": float(
                mi_series.loc[importance_df["feature"].astype(str).str.startswith("htf_confluence_")].max()
            )
            if importance_df["feature"].astype(str).str.startswith("htf_confluence_").any()
            else None,
        }
    return payload


def _iso_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()
