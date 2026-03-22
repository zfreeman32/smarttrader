from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..strategy_registry import (
    STRATEGY_REGISTRY,
    prepare_strategy_input,
    slugify_strategy_name,
)


@register_feature_set(
    name="strategy_signals",
    category="strategy",
    description="Standalone strategy-module outputs encoded as dataset features",
    required_columns=("open", "high", "low", "close"),
)
def build_strategy_signals(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    if not config.strategy_ids:
        return pd.DataFrame(index=df.index)

    strategy_input = prepare_strategy_input(df)
    input_columns = set(strategy_input.columns)
    built_frames = []
    requested_strategy_ids = list(dict.fromkeys(config.strategy_ids))
    built_strategy_ids: list[str] = []
    skipped_strategies: list[dict[str, str]] = []

    for strategy_name in requested_strategy_ids:
        resolved_name: str | None = None
        try:
            resolved_name = STRATEGY_REGISTRY.resolve(strategy_name)
            raw_output = STRATEGY_REGISTRY.build(resolved_name, strategy_input, copy_input=True)
            normalized_output = _normalize_strategy_output(
                strategy_id=resolved_name,
                output=raw_output,
                source_index=strategy_input.index,
                target_index=df.index,
                input_columns=input_columns,
            )
        except Exception as exc:
            if not config.skip_failed_strategies:
                raise
            skipped_strategies.append(
                _serialize_strategy_failure(
                    requested_name=strategy_name,
                    resolved_name=resolved_name,
                    exc=exc,
                )
            )
            continue

        built_strategy_ids.append(resolved_name)
        if not normalized_output.empty:
            built_frames.append(normalized_output)

    if built_frames:
        combined = pd.concat(built_frames, axis=1)
        result = combined.loc[:, ~combined.columns.duplicated()]
    else:
        result = pd.DataFrame(index=df.index)

    result.attrs["feature_build_report"] = {
        "requested": len(requested_strategy_ids),
        "built": len(built_strategy_ids),
        "skipped": len(skipped_strategies),
        "output_columns": int(result.shape[1]),
        "built_strategy_ids": built_strategy_ids,
        "skipped_strategies": skipped_strategies,
    }
    return result


def _normalize_strategy_output(
    *,
    strategy_id: str,
    output: pd.DataFrame,
    source_index: pd.Index,
    target_index: pd.Index,
    input_columns: set[str],
) -> pd.DataFrame:
    output = output.copy()
    output = output.loc[:, [column for column in output.columns if column not in input_columns]]
    if output.empty:
        return pd.DataFrame(index=target_index)

    if len(output.index) == len(source_index) and not output.index.equals(source_index):
        output.index = source_index
    else:
        output = output.reindex(source_index)

    output.index = target_index

    encoded_frames = []
    strategy_prefix = f"strategy__{slugify_strategy_name(strategy_id)}"

    for column in output.columns:
        series = output[column]
        column_slug = slugify_strategy_name(str(column))
        prefixed_name = f"{strategy_prefix}__{column_slug}"

        if is_bool_dtype(series):
            encoded_frames.append(series.fillna(False).astype(int).rename(prefixed_name).to_frame())
            continue

        if is_numeric_dtype(series):
            encoded_frames.append(pd.to_numeric(series, errors="coerce").rename(prefixed_name).to_frame())
            continue

        categorical = series.astype("string").fillna("missing")
        dummies = pd.get_dummies(
            categorical,
            prefix=prefixed_name,
            prefix_sep="__",
            dtype=int,
        )
        encoded_frames.append(dummies)

    if not encoded_frames:
        return pd.DataFrame(index=target_index)

    normalized = pd.concat(encoded_frames, axis=1)
    return normalized.loc[:, ~normalized.columns.duplicated()]


def _serialize_strategy_failure(
    *,
    requested_name: str,
    resolved_name: str | None,
    exc: Exception,
) -> dict[str, str]:
    message = " ".join(str(exc).split())
    return {
        "requested_strategy": requested_name,
        "resolved_strategy": resolved_name or "",
        "error_type": type(exc).__name__,
        "error_message": message[:500],
    }
