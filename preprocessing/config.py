from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _default_target_columns() -> List[str]:
    return [
        "label_long_entry",
        "label_short_entry",
        "label_long_ote",
        "label_short_ote",
    ]


@dataclass
class PreprocessingConfig:
    """Configuration for target-aware feature preprocessing."""

    target_columns: List[str] = field(default_factory=_default_target_columns)
    time_column: str = "datetime"
    max_analysis_rows: int = 10_000
    additional_skip_rows: int = 0
    respect_upstream_warmup: bool = True

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    use_time_based_split: bool = True

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


__all__ = [
    "PreprocessingConfig",
    "TargetDatasetSpec",
    "_default_target_columns",
]
