from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .instruments import normalize_ict_instrument


DEFAULT_ICT_SETUP_TYPES = (
    "sweep_reclaim",
    "sweep_displacement_fvg",
    "ob_retest_after_mss",
    "ifvg_reversal",
    "premium_discount_continuation",
    "session_open_manipulation_pre_ib",
    "session_open_manipulation_post_ib",
    "displacement_continuation_after_raid",
)


@dataclass(frozen=True)
class ICTSetupDetectorConfig:
    """Config for the causal ICT detector layer and later setup engine."""

    instrument: str = "es"
    enabled_setup_types: tuple[str, ...] = field(default_factory=lambda: DEFAULT_ICT_SETUP_TYPES)
    sweep_lookback_bars: int = 20
    sweep_buffer_atr: float = 0.05
    sweep_buffer_ticks: float = 1.0
    sweep_close_back_bars: int = 1
    reclaim_confirmation_bars: int = 2
    confluence_window_bars: int = 3
    cooldown_bars: int = 6
    invalidation_buffer_atr: float = 0.05
    fvg_max_age: int = 120
    fvg_min_gap_atr: float = 0.15
    order_block_max_age: int = 120
    order_block_use_wicks: bool = False
    order_block_range_atr: float = 1.25
    displacement_range_atr: float = 1.5
    displacement_body_to_range: float = 0.65
    displacement_close_location: float = 0.75
    displacement_volume_zscore: float = 0.5
    liquidity_tolerance_atr: float = 0.2
    structure_break_buffer_atr: float = 0.05
    ote_lower: float = 0.62
    ote_upper: float = 0.79
    ote_mid: float = 0.705
    max_same_side_fires_per_session: int = 3
    target_daily_trade_count_min: int = 1
    target_daily_trade_count_max: int = 3

    def normalized_instrument(self) -> str:
        return normalize_ict_instrument(self.instrument)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
