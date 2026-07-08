"""FRVP package scaffold with Phase 0 continuity implemented."""

from .continuity import (
    AbsolutePriceLevel,
    BackAdjustedPathBars,
    ContinuousContractResult,
    CoordinateMismatchError,
    ProfileSlice,
    RawProfileBars,
    RollBoundaryError,
    RollCalendarResult,
    build_continuous_contract,
    build_continuous_contract_from_tagged_series,
    build_volume_roll_calendar,
)

__all__ = [
    "AbsolutePriceLevel",
    "BackAdjustedPathBars",
    "ContinuousContractResult",
    "CoordinateMismatchError",
    "ProfileSlice",
    "RawProfileBars",
    "RollBoundaryError",
    "RollCalendarResult",
    "build_continuous_contract",
    "build_continuous_contract_from_tagged_series",
    "build_volume_roll_calendar",
]
