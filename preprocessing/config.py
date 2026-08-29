from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from frvp.target_lanes import (
    FRVP_DIRECT_TARGET_COLUMNS,
    FRVP_META_TARGET_COLUMNS,
    FRVP_POOLED_DIRECT_TARGET_COLUMNS,
    FRVP_SETUP_TARGET_COLUMNS,
    FRVP_TARGET_COLUMNS,
)


REFERENCE_BAR_MINUTES = 5.0

LEGACY_OTE_TARGET_COLUMNS = (
    "label_long_reversal",
    "label_short_reversal",
    "label_long_reversal_entry",
    "label_short_reversal_entry",
    "label_long_continuation_pullback",
    "label_short_continuation_pullback",
    "label_long_continuation_entry",
    "label_short_continuation_entry",
    "label_long_breakout",
    "label_short_breakout",
    "label_long_breakout_entry",
    "label_short_breakout_entry",
)

ICT_TARGET_COLUMNS = (
    "label_long_ict_reversal",
    "label_short_ict_reversal",
    "label_long_ict_continuation",
    "label_short_ict_continuation",
    "label_long_ict_meta",
    "label_short_ict_meta",
)

AGGREGATE_TARGET_COLUMNS = (
    "label_long_ote",
    "label_short_ote",
    "label_long_entry",
    "label_short_entry",
)


def _default_target_columns() -> List[str]:
    return [
        *LEGACY_OTE_TARGET_COLUMNS,
        *FRVP_TARGET_COLUMNS,
        *ICT_TARGET_COLUMNS,
        *AGGREGATE_TARGET_COLUMNS,
    ]


@dataclass
class PreprocessingConfig:
    """Configuration for target-aware feature preprocessing."""

    target_columns: List[str] = field(default_factory=_default_target_columns)
    time_column: str = "datetime"
    load_time_column: bool = True
    auto_retune_for_bar_interval: bool = True
    bar_interval_reference_minutes: float = REFERENCE_BAR_MINUTES
    max_analysis_rows: int = 100_000
    additional_skip_rows: int = 0
    respect_upstream_warmup: bool = True

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    use_time_based_split: bool = True
    split_embargo_bars: int = 0
    target_split_embargo_bars: Dict[str, int] = field(default_factory=dict)

    similarity_threshold: float = 0.995
    correlation_threshold: float = 0.98
    variance_threshold: float = 1e-9

    min_usable_rows: int = 250
    min_train_rows: int = 100
    min_positive_samples: int = 25
    top_n_features: int = 50

    scaler_type: str = "none"  # one of: none, robust, standard

    rf_n_estimators: int = 250
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 8
    mutual_info_neighbors: int = 5

    include_base_price_columns: bool = False
    save_scaler: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetDatasetSpec:
    """Metadata about one label target and its helper columns."""

    name: str
    target_column: str
    direction: str
    label_kind: str
    sample_weight_column: Optional[str] = None
    quality_column: Optional[str] = None
    exclude_column: Optional[str] = None
    safe_negative_column: Optional[str] = None
    component_target_columns: List[str] = field(default_factory=list)
    component_sample_weight_columns: List[str] = field(default_factory=list)
    component_quality_columns: List[str] = field(default_factory=list)
    component_exclude_columns: List[str] = field(default_factory=list)
    component_safe_negative_columns: List[str] = field(default_factory=list)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.component_target_columns)

    def required_columns(self) -> List[str]:
        columns: List[str] = []
        for column in (
            self.target_column,
            self.sample_weight_column,
            self.quality_column,
            self.exclude_column,
            self.safe_negative_column,
        ):
            if column:
                columns.append(column)

        for component_columns in (
            self.component_target_columns,
            self.component_sample_weight_columns,
            self.component_quality_columns,
            self.component_exclude_columns,
            self.component_safe_negative_columns,
        ):
            for column in component_columns:
                if column:
                    columns.append(column)

        return list(dict.fromkeys(columns))


__all__ = [
    "AGGREGATE_TARGET_COLUMNS",
    "FRVP_DIRECT_TARGET_COLUMNS",
    "FRVP_META_TARGET_COLUMNS",
    "FRVP_POOLED_DIRECT_TARGET_COLUMNS",
    "FRVP_SETUP_TARGET_COLUMNS",
    "FRVP_TARGET_COLUMNS",
    "ICT_TARGET_COLUMNS",
    "LEGACY_OTE_TARGET_COLUMNS",
    "PreprocessingConfig",
    "TargetDatasetSpec",
    "_default_target_columns",
]
