from __future__ import annotations

import gc
import json
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import feature_sets as _builtin_feature_sets  # noqa: F401
from .config import FeatureBuilderConfig
from .io import load_market_data, save_dataset, standardize_market_frame, validate_ohlcv
from .progress import ProgressCallback, progress_context, report_progress
from .registry import FEATURE_REGISTRY
from .transforms import (
    add_atr_normalized_features,
    add_interaction_features,
    add_lag_features,
    add_rolling_percentile_rank_features,
    add_rolling_statistics,
    add_rolling_winsorized_features,
    add_rolling_zscores,
    add_sigma_normalized_features,
)

REFERENCE_BAR_MINUTES = 5.0


class FeatureDatasetBuilder:
    """Build feature datasets from registered feature families and recipes."""

    def __init__(
        self,
        config: FeatureBuilderConfig | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config or FeatureBuilderConfig()
        self.progress_callback = progress_callback

    def build_from_csv(
        self,
        input_path: str | Path,
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        raw_df = load_market_data(input_path)
        return self.build(raw_df, source_path=input_path)

    def build(
        self,
        df: pd.DataFrame,
        *,
        source_path: str | Path | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        with progress_context(self.progress_callback):
            build_started = perf_counter()
            working = standardize_market_frame(
                df.copy(),
                source_timezone=self.config.source_timezone,
                canonical_timezone=self.config.canonical_timezone,
            )
            working = validate_ohlcv(working, drop_invalid=self.config.drop_invalid_ohlc)
            invalid_rows_removed = int(working.attrs.get("invalid_rows_removed", 0))
            active_config, tuning_info = self._retune_config_for_bar_interval(working)

            report_progress(
                stage="build",
                action="start",
                name="feature_dataset",
                source_path=str(source_path) if source_path is not None else "",
                input_rows=int(len(working)),
                input_columns=int(len(working.columns)),
                feature_sets=len(active_config.feature_sets),
                transform_jobs=self._count_enabled_transform_jobs(config=active_config),
                strategy_jobs=len(active_config.strategy_ids),
                transform_workers=int(active_config.transform_workers),
            )

            generated_columns: List[str] = []
            feature_counts: Dict[str, int] = {}
            transform_counts: Dict[str, int] = {}
            feature_set_reports: Dict[str, object] = {}
            step_timings: Dict[str, float] = {}

            total_feature_sets = len(active_config.feature_sets)
            for feature_index, feature_set_name in enumerate(active_config.feature_sets, start=1):
                report_progress(
                    stage="feature_set",
                    action="start",
                    name=feature_set_name,
                    index=feature_index,
                    total=total_feature_sets,
                )
                step_started = perf_counter()
                built = FEATURE_REGISTRY.build(feature_set_name, working, active_config)
                feature_report = built.attrs.get("feature_build_report")
                if feature_report is not None:
                    feature_set_reports[feature_set_name] = feature_report
                working, added = self._append_features(working, built)
                del built
                elapsed = perf_counter() - step_started
                generated_columns.extend(added)
                feature_counts[feature_set_name] = len(added)
                step_timings[f"feature_set:{feature_set_name}"] = elapsed
                report_progress(
                    stage="feature_set",
                    action="complete",
                    name=feature_set_name,
                    index=feature_index,
                    total=total_feature_sets,
                    elapsed_seconds=elapsed,
                    added_columns=len(added),
                )
            gc.collect()

            transform_blocks = self._build_transform_blocks(working, step_timings, config=active_config)
            if transform_blocks:
                existing_columns = set(working.columns)
                prepared_blocks: List[pd.DataFrame] = []
                total_transforms = len(transform_blocks)
                for transform_index, (transform_name, block) in enumerate(transform_blocks, start=1):
                    prepared, added = self._prepare_feature_block(existing_columns, block)
                    transform_counts[transform_name] = len(added)
                    report_progress(
                        stage="transform",
                        action="complete",
                        name=transform_name,
                        index=transform_index,
                        total=total_transforms,
                        elapsed_seconds=step_timings.get(f"transform:{transform_name}"),
                        added_columns=len(added),
                    )
                    if not added:
                        continue
                    prepared_blocks.append(prepared)
                    existing_columns.update(added)
                    generated_columns.extend(added)

                if prepared_blocks:
                    working = pd.concat([working, *prepared_blocks], axis=1)
                del prepared_blocks
                del transform_blocks
                gc.collect()

            self._replace_infinite_values_inplace(working)

            if active_config.drop_warmup_rows and active_config.warmup_rows > 0:
                working = working.iloc[active_config.warmup_rows :].reset_index(drop=True)

            if active_config.fillna_numeric:
                self._fill_numeric_columns_inplace(working)

            generated_columns = list(dict.fromkeys(generated_columns))
            memory_before_dtype_optimization = int(working.memory_usage(index=True, deep=False).sum())
            if active_config.optimize_feature_dtypes:
                working = self._optimize_feature_dtypes(working, generated_columns)
                gc.collect()
            step_timings["total"] = perf_counter() - build_started
            memory_usage_bytes = int(working.memory_usage(index=True, deep=False).sum())
            feature_memory_usage_bytes = (
                int(working.loc[:, generated_columns].memory_usage(index=False, deep=False).sum())
                if generated_columns
                else 0
            )
            source_metadata = self._load_source_metadata(source_path)
            source_metadata_path = self._resolve_source_metadata_path(source_path)
            metadata = {
                "source_path": str(source_path) if source_path is not None else None,
                "source_metadata_file": str(source_metadata_path) if source_metadata_path is not None else None,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows": int(len(working)),
                "columns": int(len(working.columns)),
                "feature_sets": list(active_config.feature_sets),
                "feature_counts": feature_counts,
                "transform_counts": transform_counts,
                "invalid_rows_removed": invalid_rows_removed,
                "feature_columns": generated_columns,
                "feature_column_count": len(generated_columns),
                "input_bar_minutes": tuning_info["input_bar_minutes"],
                "input_timeframe": tuning_info["input_timeframe"],
                "runtime_bar_interval_tuning": tuning_info,
                "build_timings_seconds": {key: round(value, 6) for key, value in step_timings.items()},
                "memory_usage_bytes": memory_usage_bytes,
                "feature_memory_usage_bytes": feature_memory_usage_bytes,
                "timezone_contract": {
                    "source_timezone": active_config.source_timezone,
                    "canonical_timezone": active_config.canonical_timezone,
                    "feature_clock_timezone": active_config.feature_clock_timezone,
                    "london_timezone": active_config.london_timezone,
                    "new_york_timezone": active_config.new_york_timezone,
                    "market_close_timezone": active_config.market_close_timezone,
                    "market_close_time": (
                        f"{int(active_config.market_close_hour):02d}:{int(active_config.market_close_minute):02d}"
                    ),
                },
                "config": active_config.to_dict(),
            }
            if source_metadata:
                metadata["upstream_source_path"] = source_metadata.get("source_path")
                metadata["upstream_metadata_file"] = str(source_metadata_path) if source_metadata_path is not None else None
                metadata["upstream_timezone_contract"] = source_metadata.get("timezone_contract")
                metadata["upstream_bar_timestamp_semantics"] = source_metadata.get("bar_timestamp_semantics")
            if active_config.optimize_feature_dtypes:
                metadata["memory_usage_bytes_before_dtype_optimization"] = memory_before_dtype_optimization
            if feature_set_reports:
                metadata["feature_set_reports"] = feature_set_reports

            report_progress(
                stage="build",
                action="complete",
                name="feature_dataset",
                elapsed_seconds=step_timings["total"],
                output_rows=int(len(working)),
                output_columns=int(len(working.columns)),
                feature_columns=len(generated_columns),
                invalid_rows_removed=invalid_rows_removed,
            )
            return working, metadata

    def build_and_save(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> Tuple[pd.DataFrame, Dict[str, object], Path, Path]:
        dataset, metadata = self.build_from_csv(input_path)
        saved_csv, saved_metadata = save_dataset(dataset, metadata, output_path)
        return dataset, metadata, saved_csv, saved_metadata

    @staticmethod
    def _infer_bar_minutes(working: pd.DataFrame) -> float:
        if "datetime" not in working.columns or len(working) < 2:
            return REFERENCE_BAR_MINUTES

        timestamps = pd.to_datetime(working["datetime"], errors="coerce").dropna()
        if len(timestamps) < 2:
            return REFERENCE_BAR_MINUTES

        diffs = timestamps.diff().dropna()
        minute_diffs = pd.Series(diffs.dt.total_seconds() / 60.0)
        minute_diffs = minute_diffs[minute_diffs > 0]
        if minute_diffs.empty:
            return REFERENCE_BAR_MINUTES

        dominant_minutes = float(minute_diffs.mode().iloc[0])
        return max(dominant_minutes, 1e-6)

    @staticmethod
    def _format_timeframe(bar_minutes: float) -> str:
        rounded = round(bar_minutes)
        if abs(bar_minutes - rounded) < 1e-9:
            if rounded % 60 == 0:
                hours = rounded // 60
                return "1h" if hours == 1 else f"{hours}h"
            return f"{rounded}m"
        return f"{bar_minutes:g}m"

    @staticmethod
    def _scale_window(
        count_at_reference: int,
        *,
        base_bar_minutes: float,
        reference_bar_minutes: float,
        minimum: int = 1,
    ) -> int:
        if count_at_reference <= 0:
            return 0
        scale = reference_bar_minutes / max(base_bar_minutes, 1e-6)
        return max(minimum, int(round(count_at_reference * scale)))

    @classmethod
    def _scale_window_list(
        cls,
        values_at_reference: List[int],
        *,
        base_bar_minutes: float,
        reference_bar_minutes: float,
    ) -> List[int]:
        scaled = [
            cls._scale_window(
                int(value),
                base_bar_minutes=base_bar_minutes,
                reference_bar_minutes=reference_bar_minutes,
                minimum=1,
            )
            for value in values_at_reference
        ]
        return list(dict.fromkeys(sorted(scaled)))

    def _retune_config_for_bar_interval(
        self,
        working: pd.DataFrame,
    ) -> Tuple[FeatureBuilderConfig, Dict[str, object]]:
        input_bar_minutes = self._infer_bar_minutes(working)
        input_timeframe = self._format_timeframe(input_bar_minutes)
        reference_minutes = float(getattr(self.config, "bar_interval_reference_minutes", REFERENCE_BAR_MINUTES))
        info: Dict[str, object] = {
            "input_bar_minutes": input_bar_minutes,
            "input_timeframe": input_timeframe,
            "reference_bar_minutes": reference_minutes,
            "applied": False,
            "updated_fields": [],
        }

        if not self.config.auto_retune_for_bar_interval:
            return self.config, info
        if input_bar_minutes <= 0 or abs(input_bar_minutes - reference_minutes) < 1e-9:
            return replace(self.config, auto_retune_for_bar_interval=False), info

        updates = {
            "warmup_rows": self._scale_window(
                int(self.config.warmup_rows),
                base_bar_minutes=input_bar_minutes,
                reference_bar_minutes=reference_minutes,
            ),
            "lag_periods": self._scale_window_list(
                list(self.config.lag_periods),
                base_bar_minutes=input_bar_minutes,
                reference_bar_minutes=reference_minutes,
            ),
            "rolling_windows": self._scale_window_list(
                list(self.config.rolling_windows),
                base_bar_minutes=input_bar_minutes,
                reference_bar_minutes=reference_minutes,
            ),
            "zscore_window": self._scale_window(
                int(self.config.zscore_window),
                base_bar_minutes=input_bar_minutes,
                reference_bar_minutes=reference_minutes,
            ),
            "winsorization_window": self._scale_window(
                int(self.config.winsorization_window),
                base_bar_minutes=input_bar_minutes,
                reference_bar_minutes=reference_minutes,
            ),
            "percentile_rank_window": self._scale_window(
                int(self.config.percentile_rank_window),
                base_bar_minutes=input_bar_minutes,
                reference_bar_minutes=reference_minutes,
            ),
            "sigma_normalization_window": self._scale_window(
                int(self.config.sigma_normalization_window),
                base_bar_minutes=input_bar_minutes,
                reference_bar_minutes=reference_minutes,
            ),
            "swing_window": self._scale_window(
                int(self.config.swing_window),
                base_bar_minutes=input_bar_minutes,
                reference_bar_minutes=reference_minutes,
            ),
            "ict_zone_max_age": self._scale_window(
                int(self.config.ict_zone_max_age),
                base_bar_minutes=input_bar_minutes,
                reference_bar_minutes=reference_minutes,
            ),
            "fractional_diff_max_size": self._scale_window(
                int(self.config.fractional_diff_max_size),
                base_bar_minutes=input_bar_minutes,
                reference_bar_minutes=reference_minutes,
            ),
        }

        active_config = replace(
            self.config,
            **updates,
            auto_retune_for_bar_interval=False,
        )
        info["applied"] = True
        info["updated_fields"] = sorted(updates.keys())
        return active_config, info

    @staticmethod
    def _append_features(
        working: pd.DataFrame,
        additions: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        prepared, new_columns = FeatureDatasetBuilder._prepare_feature_block(set(working.columns), additions)
        if not new_columns:
            return working, []
        merged = pd.concat([working, prepared], axis=1)
        return merged, new_columns

    def _build_transform_blocks(
        self,
        working: pd.DataFrame,
        step_timings: Dict[str, float],
        *,
        config: FeatureBuilderConfig,
    ) -> List[Tuple[str, pd.DataFrame]]:
        jobs: List[Tuple[str, Callable[[], pd.DataFrame]]] = []
        ordered_names: List[str] = []
        total_transform_jobs = self._count_enabled_transform_jobs(config=config)

        def register(name: str, enabled: bool, builder: Callable[[], pd.DataFrame]) -> None:
            if not enabled:
                return
            ordered_names.append(name)
            jobs.append((name, builder))

        register(
            "winsorization",
            config.enable_winsorization,
            lambda: add_rolling_winsorized_features(
                working,
                config.winsorize_columns,
                config.winsorization_window,
                config.winsorization_lower_quantile,
                config.winsorization_upper_quantile,
            ),
        )
        register(
            "percentile_ranks",
            config.enable_percentile_ranks,
            lambda: add_rolling_percentile_rank_features(
                working,
                config.percentile_rank_columns,
                config.percentile_rank_window,
            ),
        )
        register(
            "atr_normalization",
            config.enable_atr_normalization,
            lambda: add_atr_normalized_features(
                working,
                config.atr_normalization_columns,
                atr_column=config.atr_normalization_source,
            ),
        )
        register(
            "sigma_normalization",
            config.enable_sigma_normalization,
            lambda: add_sigma_normalized_features(
                working,
                config.sigma_normalization_columns,
                config.sigma_normalization_window,
            ),
        )
        register(
            "lags",
            config.enable_lags,
            lambda: add_lag_features(
                working,
                config.lag_columns,
                config.lag_periods,
            ),
        )
        register(
            "rolling_stats",
            config.enable_rolling_stats,
            lambda: add_rolling_statistics(
                working,
                config.rolling_stat_columns,
                config.rolling_windows,
            ),
        )
        register(
            "zscores",
            config.enable_zscores,
            lambda: add_rolling_zscores(
                working,
                config.zscore_columns,
                config.zscore_window,
            ),
        )

        blocks, timings = self._execute_transform_jobs(jobs, config=config)
        step_timings.update(timings)

        if config.enable_interactions:
            interaction_started = perf_counter()
            report_progress(
                stage="transform",
                action="start",
                name="interactions",
                index=len(ordered_names) + 1,
                total=total_transform_jobs,
            )
            interaction_source = working
            percentile_block = blocks.get("percentile_ranks")
            if percentile_block is not None and not percentile_block.empty:
                interaction_source = pd.concat([working, percentile_block], axis=1)
            blocks["interactions"] = add_interaction_features(interaction_source)
            ordered_names.append("interactions")
            step_timings["transform:interactions"] = perf_counter() - interaction_started

        return [
            (name, blocks[name])
            for name in ordered_names
            if name in blocks
        ]

    def _execute_transform_jobs(
        self,
        jobs: List[Tuple[str, Callable[[], pd.DataFrame]]],
        *,
        config: FeatureBuilderConfig,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
        if not jobs:
            return {}, {}

        if config.transform_workers <= 1 or len(jobs) == 1:
            blocks: Dict[str, pd.DataFrame] = {}
            timings: Dict[str, float] = {}
            total_jobs = self._count_enabled_transform_jobs(config=config)
            for index, (name, builder) in enumerate(jobs, start=1):
                block_name, frame, elapsed = self._run_transform_job(
                    name,
                    builder,
                    index=index,
                    total=total_jobs,
                )
                blocks[block_name] = frame
                timings[f"transform:{name}"] = elapsed
            return blocks, timings

        max_workers = min(max(1, config.transform_workers), len(jobs))
        blocks = {}
        timings = {}
        total_jobs = self._count_enabled_transform_jobs(config=config)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_transform_job,
                    name,
                    builder,
                    index=index,
                    total=total_jobs,
                ): name
                for index, (name, builder) in enumerate(jobs, start=1)
            }
            for future, name in futures.items():
                block_name, frame, elapsed = future.result()
                blocks[block_name] = frame
                timings[f"transform:{name}"] = elapsed
        return blocks, timings

    @staticmethod
    def _run_transform_job(
        name: str,
        builder: Callable[[], pd.DataFrame],
        *,
        index: int,
        total: int,
    ) -> Tuple[str, pd.DataFrame, float]:
        started = perf_counter()
        report_progress(
            stage="transform",
            action="start",
            name=name,
            index=index,
            total=total,
        )
        return name, builder(), perf_counter() - started

    def _count_enabled_transform_jobs(self, *, config: FeatureBuilderConfig | None = None) -> int:
        config = config or self.config
        count = 0
        if config.enable_winsorization:
            count += 1
        if config.enable_percentile_ranks:
            count += 1
        if config.enable_atr_normalization:
            count += 1
        if config.enable_sigma_normalization:
            count += 1
        if config.enable_lags:
            count += 1
        if config.enable_rolling_stats:
            count += 1
        if config.enable_zscores:
            count += 1
        if config.enable_interactions:
            count += 1
        return count

    @staticmethod
    def _prepare_feature_block(
        existing_columns: set[str],
        additions: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        additions = additions.loc[:, ~additions.columns.duplicated()]
        new_columns = [column for column in additions.columns if column not in existing_columns]
        if not new_columns:
            return additions.iloc[:, 0:0], []
        return additions.loc[:, new_columns], new_columns

    @staticmethod
    def _replace_infinite_values_inplace(frame: pd.DataFrame) -> None:
        for column in frame.columns:
            series = frame[column]
            if not pd.api.types.is_float_dtype(series):
                continue
            values = series.to_numpy(copy=False)
            if values.dtype.kind != "f":
                continue
            inf_mask = np.isinf(values)
            if not inf_mask.any():
                continue
            cleaned = values.copy()
            cleaned[inf_mask] = np.nan
            frame[column] = cleaned

    @staticmethod
    def _fill_numeric_columns_inplace(frame: pd.DataFrame) -> None:
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            series = frame[column]
            frame[column] = series.ffill().fillna(0.0)

    @staticmethod
    def _optimize_feature_dtypes(
        working: pd.DataFrame,
        feature_columns: List[str],
    ) -> pd.DataFrame:
        for column in feature_columns:
            if column not in working.columns:
                continue
            series = working[column]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            if pd.api.types.is_float_dtype(series):
                working[column] = pd.to_numeric(series, downcast="float")
            elif pd.api.types.is_integer_dtype(series):
                working[column] = pd.to_numeric(series, downcast="integer")
        return working

    @staticmethod
    def _resolve_source_metadata_path(source_path: str | Path | None) -> Path | None:
        if source_path is None:
            return None
        candidate = Path(source_path).with_suffix(".metadata.json")
        return candidate if candidate.exists() else None

    @classmethod
    def _load_source_metadata(cls, source_path: str | Path | None) -> Dict[str, object]:
        metadata_path = cls._resolve_source_metadata_path(source_path)
        if metadata_path is None:
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
