"""Compatibility shims for the design-paper futures paths."""

from .continuous_contract import build_continuous_contract
from .roll_calendar import build_volume_roll_calendar

__all__ = ["build_continuous_contract", "build_volume_roll_calendar"]
