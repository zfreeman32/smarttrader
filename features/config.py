from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List


def _default_feature_sets() -> List[str]:
    return [
        "price_action",
        "volatility",
        "trend",
        "momentum",
        "volume",
        "structure",
        "htf_context",
        "continuation_pullback",
        "ict_context",
        "exhaustion",
        "microstructure",
        "session",
        "temporal_context",
        "fractional_diff",
        "quality",
    ]


def _default_preserve_columns() -> List[str]:
    return [
        "datetime",
        "date",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


@dataclass
class FeatureBuilderConfig:
    """Configuration for a feature-dataset build."""

    feature_sets: List[str] = field(default_factory=_default_feature_sets)
    preserve_columns: List[str] = field(default_factory=_default_preserve_columns)
    strategy_ids: List[str] = field(default_factory=list)
    include_all_strategies: bool = False
    skip_failed_strategies: bool = False

    warmup_rows: int = 200
    drop_warmup_rows: bool = True
    drop_invalid_ohlc: bool = True
    fillna_numeric: bool = True
    transform_workers: int = 1
    strategy_timeout_seconds: float | None = 300.0
    optimize_feature_dtypes: bool = False

    source_timezone: str = "UTC"
    canonical_timezone: str = "UTC"
    feature_clock_timezone: str = "America/New_York"
    london_timezone: str = "Europe/London"
    new_york_timezone: str = "America/New_York"
    market_close_timezone: str = "America/New_York"
    market_close_hour: int = 17
    market_close_minute: int = 0
    asia_session_reference_timezone: str = "America/New_York"
    asia_session_start_hour: int = 19
    asia_session_end_hour: int = 2
    london_session_start_hour: int = 7
    london_session_end_hour: int = 16
    new_york_session_start_hour: int = 8
    new_york_session_end_hour: int = 17
    london_killzone_reference_timezone: str = "America/New_York"
    london_killzone_start_hour: int = 2
    london_killzone_end_hour: int = 5
    new_york_killzone_reference_timezone: str = "America/New_York"
    new_york_killzone_start_hour: int = 8
    new_york_killzone_end_hour: int = 11

    enable_lags: bool = True
    lag_columns: List[str] = field(
        default_factory=lambda: [
            "close_return_1",
            "range_atr_14",
            "atr_ratio_14_50",
            "rsi_14",
            "roc_5",
            "volume_relative_20",
            "price_position_50",
            "dist_to_prior_high_20_atr",
            "dist_to_prior_low_20_atr",
            "continuation_bull_pullback_depth_atr",
            "continuation_bear_pullback_depth_atr",
            "continuation_bull_retest_quality",
            "continuation_bear_retest_quality",
        ]
    )
    lag_periods: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    enable_rolling_stats: bool = True
    rolling_stat_columns: List[str] = field(
        default_factory=lambda: [
            "close_return_1",
            "range_atr_14",
            "atr_ratio_14_50",
            "rsi_14",
            "roc_5",
            "volume_relative_20",
            "price_position_50",
            "continuation_bull_pullback_depth_atr",
            "continuation_bear_pullback_depth_atr",
            "continuation_bull_retest_quality",
            "continuation_bear_retest_quality",
        ]
    )
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    enable_zscores: bool = True
    zscore_columns: List[str] = field(
        default_factory=lambda: [
            "close_return_1",
            "range_atr_14",
            "atr_ratio_14_50",
            "roc_5",
            "volume_relative_20",
            "rsi_14",
            "price_position_50",
            "continuation_bull_pullback_depth_atr",
            "continuation_bear_pullback_depth_atr",
            "continuation_bull_pullback_channel_slope_8_atr",
            "continuation_bear_pullback_channel_slope_8_atr",
        ]
    )
    zscore_window: int = 60

    enable_winsorization: bool = True
    winsorize_columns: List[str] = field(
        default_factory=lambda: [
            "close_return_1",
            "close_return_3",
            "close_return_5",
            "close_return_10",
            "log_return_1",
            "roc_5",
            "roc_10",
            "price_acceleration_3",
            "return_slope_3",
            "return_slope_8",
            "return_slope_decay_3_8",
            "return_slope_decay_8_21",
            "fracdiff_close_return_1",
        ]
    )
    winsorization_window: int = 100
    winsorization_lower_quantile: float = 0.01
    winsorization_upper_quantile: float = 0.99

    enable_percentile_ranks: bool = True
    percentile_rank_columns: List[str] = field(
        default_factory=lambda: [
            "log_volume",
            "volume_relative_20",
            "volume_imbalance_10",
            "rolling_vol_20",
            "rolling_vol_60",
            "atr_ratio_14_50",
            "range_atr_14",
            "volatility_expansion_20",
            "range_compression_10_50",
            "volatility_compression_10_50",
        ]
    )
    percentile_rank_window: int = 60

    enable_atr_normalization: bool = True
    atr_normalization_source: str = "atr_14"
    atr_normalization_columns: List[str] = field(
        default_factory=lambda: [
            "candle_body",
            "candle_range",
            "upper_wick",
            "lower_wick",
            "true_range",
            "approx_spread",
            "fracdiff_close",
        ]
    )

    enable_sigma_normalization: bool = True
    sigma_normalization_columns: List[str] = field(
        default_factory=lambda: [
            "return_slope_3",
            "return_slope_8",
            "return_slope_decay_3_8",
            "return_slope_decay_8_21",
            "rsi_slope_3",
            "rsi_slope_8",
            "rsi_slope_decay_3_8",
            "rsi_slope_decay_8_21",
            "bullish_divergence_strength",
            "bearish_divergence_strength",
            "bull_volume_exhaustion",
            "bear_volume_exhaustion",
            "ict_total_confluence_1atr",
            "ict_zone_balance_1atr",
            "major_event_cluster_10",
            "major_event_cluster_25",
            "fracdiff_close_return_1",
        ]
    )
    sigma_normalization_window: int = 60

    enable_interactions: bool = True

    swing_window: int = 3
    htf_swing_window: int = 3
    ict_zone_max_age: int = 120
    ict_fvg_min_gap_atr: float = 0.15
    ict_order_block_range_atr: float = 1.25
    ict_liquidity_tolerance_atr: float = 0.2
    ict_break_buffer_atr: float = 0.05

    fractional_diff_d: float = 0.4
    fractional_diff_threshold: float = 1e-3
    fractional_diff_max_size: int = 200

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "FeatureBuilderConfig":
        valid_fields = {item.name for item in fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_recipe(cls, recipe_path: str | Path | None) -> "FeatureBuilderConfig":
        if recipe_path is None:
            return cls()
        recipe_path = Path(recipe_path)
        with recipe_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
