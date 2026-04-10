from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ote_live.contracts.market_data import MarketBar

CanonicalTimeframe = Literal["1m", "5m", "30m", "1h"]

_TIMEFRAME_TO_MINUTES: dict[CanonicalTimeframe, int] = {
    "1m": 1,
    "5m": 5,
    "30m": 30,
    "1h": 60,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def timeframe_to_minutes(timeframe: CanonicalTimeframe) -> int:
    return _TIMEFRAME_TO_MINUTES[timeframe]


def timeframe_to_timedelta(timeframe: CanonicalTimeframe) -> timedelta:
    return timedelta(minutes=timeframe_to_minutes(timeframe))


def floor_timestamp_to_timeframe_start(value: datetime, timeframe: CanonicalTimeframe) -> datetime:
    resolved = ensure_utc(value)
    step_seconds = int(timeframe_to_timedelta(timeframe).total_seconds())
    epoch_seconds = int(resolved.timestamp())
    floored_epoch = epoch_seconds - (epoch_seconds % step_seconds)
    return datetime.fromtimestamp(floored_epoch, tz=timezone.utc)


def latest_finalized_bar_start(
    timeframe: CanonicalTimeframe,
    *,
    now: datetime | None = None,
    grace_period_seconds: float = 0.0,
) -> datetime:
    resolved_now = ensure_utc(now or utc_now()) - timedelta(seconds=max(0.0, float(grace_period_seconds)))
    latest_complete_reference = resolved_now - timeframe_to_timedelta(timeframe)
    return floor_timestamp_to_timeframe_start(latest_complete_reference, timeframe)


def is_bar_finalized(
    bar: MarketBar,
    *,
    timeframe: CanonicalTimeframe | None = None,
    now: datetime | None = None,
    grace_period_seconds: float = 0.0,
) -> bool:
    resolved_timeframe = timeframe or bar.timeframe  # type: ignore[assignment]
    return ensure_utc(bar.timestamp) <= latest_finalized_bar_start(
        resolved_timeframe,  # type: ignore[arg-type]
        now=now,
        grace_period_seconds=grace_period_seconds,
    )


def filter_finalized_bars(
    bars: list[MarketBar],
    *,
    timeframe: CanonicalTimeframe | None = None,
    now: datetime | None = None,
    grace_period_seconds: float = 0.0,
) -> list[MarketBar]:
    return [
        bar
        for bar in bars
        if is_bar_finalized(
            bar,
            timeframe=timeframe,
            now=now,
            grace_period_seconds=grace_period_seconds,
        )
    ]


def canonical_asset_symbol(symbol: str) -> str:
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


class BackfillWindow(BaseModel):
    """Inclusive timestamp window for fetching missing bars."""

    model_config = ConfigDict(extra="forbid")

    asset: str = Field(default="EURUSD")
    timeframe: CanonicalTimeframe = Field(default="1m")
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def _normalize_datetimes(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "BackfillWindow":
        if self.end < self.start:
            raise ValueError("BackfillWindow.end must be greater than or equal to start.")
        self.asset = canonical_asset_symbol(self.asset)
        return self


class IngestionGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: str
    timeframe: CanonicalTimeframe
    expected_timestamp: datetime
    observed_timestamp: datetime
    missing_timestamps: list[datetime]
    gap_size: int
    detected_at_utc: datetime = Field(default_factory=utc_now)

    @field_validator("expected_timestamp", "observed_timestamp", "detected_at_utc")
    @classmethod
    def _normalize_datetimes(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @field_validator("missing_timestamps")
    @classmethod
    def _normalize_missing_timestamps(cls, value: list[datetime]) -> list[datetime]:
        return [ensure_utc(item) for item in value]

    @model_validator(mode="after")
    def _validate_gap(self) -> "IngestionGap":
        self.asset = canonical_asset_symbol(self.asset)
        if self.gap_size != len(self.missing_timestamps):
            raise ValueError("gap_size must match the length of missing_timestamps.")
        if self.gap_size <= 0:
            raise ValueError("gap_size must be positive.")
        return self

    def to_backfill_window(self) -> BackfillWindow:
        return BackfillWindow(
            asset=self.asset,
            timeframe=self.timeframe,
            start=self.missing_timestamps[0],
            end=self.missing_timestamps[-1],
        )


class GapCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_timestamp: datetime
    duplicate: bool = False
    out_of_order: bool = False
    gap: IngestionGap | None = None

    @field_validator("observed_timestamp")
    @classmethod
    def _normalize_observed_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @property
    def has_issue(self) -> bool:
        return self.duplicate or self.out_of_order or self.gap is not None


class HeartbeatStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    observed_at_utc: datetime
    stale_after_seconds: float
    lag_seconds: float
    is_stale: bool

    @field_validator("observed_at_utc")
    @classmethod
    def _normalize_observed_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class AbstractBarStream(ABC):
    @abstractmethod
    async def poll(self) -> list[MarketBar]:
        """Return any newly available bars since the previous poll."""


class AbstractBackfillConnector(ABC):
    @abstractmethod
    async def backfill_bars(self, window: BackfillWindow) -> list[MarketBar]:
        """Fetch bars for an inclusive timestamp window."""
