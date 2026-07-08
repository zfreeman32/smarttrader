"""Phase 0 continuity and roll handling per design Section 4."""

from .continuous_contract import (
    BackAdjustedPathBars,
    ContinuousContractResult,
    RawProfileBars,
    build_continuous_contract,
    build_continuous_contract_from_tagged_series,
)
from .roll_calendar import build_volume_roll_calendar
from .types import (
    AbsolutePriceLevel,
    CoordinateMismatchError,
    ProfileSlice,
    RollBoundaryError,
    RollCalendarResult,
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
