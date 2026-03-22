from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import feature_sets as _builtin_feature_sets  # noqa: F401
from .config import FeatureBuilderConfig
from .io import load_market_data, save_dataset, standardize_market_frame, validate_ohlcv
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


class FeatureDatasetBuilder:
    """Build feature datasets from registered feature families and recipes."""

    def __init__(self, config: FeatureBuilderConfig | None = None) -> None:
        self.config = config or FeatureBuilderConfig()

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
        working = standardize_market_frame(df.copy())
        working = validate_ohlcv(working, drop_invalid=self.config.drop_invalid_ohlc)
        invalid_rows_removed = int(working.attrs.get("invalid_rows_removed", 0))

        generated_columns: List[str] = []
        feature_counts: Dict[str, int] = {}
        transform_counts: Dict[str, int] = {}
        feature_set_reports: Dict[str, object] = {}

        for feature_set_name in self.config.feature_sets:
            built = FEATURE_REGISTRY.build(feature_set_name, working, self.config)
            feature_report = built.attrs.get("feature_build_report")
            if feature_report is not None:
                feature_set_reports[feature_set_name] = feature_report
            working, added = self._append_features(working, built)
            generated_columns.extend(added)
            feature_counts[feature_set_name] = len(added)

        if self.config.enable_winsorization:
            winsorized = add_rolling_winsorized_features(
                working,
                self.config.winsorize_columns,
                self.config.winsorization_window,
                self.config.winsorization_lower_quantile,
                self.config.winsorization_upper_quantile,
            )
            working, added = self._append_features(working, winsorized)
            generated_columns.extend(added)
            transform_counts["winsorization"] = len(added)

        if self.config.enable_percentile_ranks:
            percentile_ranks = add_rolling_percentile_rank_features(
                working,
                self.config.percentile_rank_columns,
                self.config.percentile_rank_window,
            )
            working, added = self._append_features(working, percentile_ranks)
            generated_columns.extend(added)
            transform_counts["percentile_ranks"] = len(added)

        if self.config.enable_atr_normalization:
            atr_normalized = add_atr_normalized_features(
                working,
                self.config.atr_normalization_columns,
                atr_column=self.config.atr_normalization_source,
            )
            working, added = self._append_features(working, atr_normalized)
            generated_columns.extend(added)
            transform_counts["atr_normalization"] = len(added)

        if self.config.enable_sigma_normalization:
            sigma_normalized = add_sigma_normalized_features(
                working,
                self.config.sigma_normalization_columns,
                self.config.sigma_normalization_window,
            )
            working, added = self._append_features(working, sigma_normalized)
            generated_columns.extend(added)
            transform_counts["sigma_normalization"] = len(added)

        if self.config.enable_lags:
            lagged = add_lag_features(
                working,
                self.config.lag_columns,
                self.config.lag_periods,
            )
            working, added = self._append_features(working, lagged)
            generated_columns.extend(added)
            transform_counts["lags"] = len(added)

        if self.config.enable_rolling_stats:
            rolling = add_rolling_statistics(
                working,
                self.config.rolling_stat_columns,
                self.config.rolling_windows,
            )
            working, added = self._append_features(working, rolling)
            generated_columns.extend(added)
            transform_counts["rolling_stats"] = len(added)

        if self.config.enable_zscores:
            zscores = add_rolling_zscores(
                working,
                self.config.zscore_columns,
                self.config.zscore_window,
            )
            working, added = self._append_features(working, zscores)
            generated_columns.extend(added)
            transform_counts["zscores"] = len(added)

        if self.config.enable_interactions:
            interactions = add_interaction_features(working)
            working, added = self._append_features(working, interactions)
            generated_columns.extend(added)
            transform_counts["interactions"] = len(added)

        working = working.replace([np.inf, -np.inf], np.nan)

        if self.config.drop_warmup_rows and self.config.warmup_rows > 0:
            working = working.iloc[self.config.warmup_rows :].reset_index(drop=True)

        if self.config.fillna_numeric:
            numeric_columns = working.select_dtypes(include=[np.number]).columns
            working.loc[:, numeric_columns] = working.loc[:, numeric_columns].ffill().fillna(0.0)

        generated_columns = list(dict.fromkeys(generated_columns))
        metadata = {
            "source_path": str(source_path) if source_path is not None else None,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(working)),
            "columns": int(len(working.columns)),
            "feature_sets": list(self.config.feature_sets),
            "feature_counts": feature_counts,
            "transform_counts": transform_counts,
            "invalid_rows_removed": invalid_rows_removed,
            "feature_columns": generated_columns,
            "feature_column_count": len(generated_columns),
            "config": self.config.to_dict(),
        }
        if feature_set_reports:
            metadata["feature_set_reports"] = feature_set_reports

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
    def _append_features(
        working: pd.DataFrame,
        additions: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        additions = additions.loc[:, ~additions.columns.duplicated()]
        new_columns = [column for column in additions.columns if column not in working.columns]
        if not new_columns:
            return working, []
        merged = pd.concat([working, additions[new_columns]], axis=1)
        return merged, new_columns
