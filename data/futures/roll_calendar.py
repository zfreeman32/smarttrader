"""Compatibility shim for the design-paper path.

Real implementation lives in frvp.continuity.roll_calendar.
"""

from frvp.continuity.roll_calendar import build_volume_roll_calendar
from frvp.continuity.types import RollCalendarResult

__all__ = ["RollCalendarResult", "build_volume_roll_calendar"]
