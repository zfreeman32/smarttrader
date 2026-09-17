from __future__ import annotations

from typing import Any
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MarketBar(BaseModel):
    """Canonical live bar contract expected by the OTE runtime."""

    model_config = ConfigDict(extra="forbid")

    asset: str = Field(default="EURUSD")
    timeframe: str = Field(default="5m")
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    source: str | None = None
    symbol: str | None = None
    contract_symbol: str | None = None
    instrument_id: int | None = None
    feature_context: dict[str, Any] = Field(default_factory=dict)
    # Unknown provenance is retained as diagnostic data, never assumed live.
    source_timestamp: datetime | None = None
    bar_version: str | None = None
    is_complete: bool | None = None
    feed_type: str | None = None
    first_observed_at: datetime | None = None
    last_observed_at: datetime | None = None
    observation_kind: str = "unknown"

    @field_validator("timestamp", "source_timestamp", "first_observed_at", "last_observed_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _validate_ohlc(self) -> "MarketBar":
        if self.high < max(self.open, self.close):
            raise ValueError("high must be greater than or equal to open and close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be less than or equal to open and close")
        if self.low > self.high:
            raise ValueError("low must be less than or equal to high")
        return self
