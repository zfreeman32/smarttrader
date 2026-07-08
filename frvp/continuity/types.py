from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ContinuityError(ValueError):
    """Base error for continuity-layer violations."""


class RollBoundaryError(ContinuityError):
    """Raised when a raw profile slice spans more than one contract."""


class CoordinateMismatchError(ContinuityError):
    """Raised when absolute levels are compared across contract coordinates."""


@dataclass(frozen=True)
class AbsolutePriceLevel:
    """An absolute price level that is valid only in one contract coordinate."""

    price: float
    contract_id: str
    source_time: pd.Timestamp


@dataclass(frozen=True)
class ProfileSlice:
    """A raw-coordinate slice that is safe for profile construction."""

    contract_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    bars: pd.DataFrame


@dataclass(frozen=True)
class RollCalendarResult:
    """Causal lead-contract schedule and roll audit tables for Phase 0."""

    contract_bars: pd.DataFrame
    session_metrics: pd.DataFrame
    lead_schedule: pd.DataFrame
    rolls: pd.DataFrame
