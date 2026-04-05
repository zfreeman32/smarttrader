from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="after")
    def _validate_ohlc(self) -> "MarketBar":
        if self.high < max(self.open, self.close):
            raise ValueError("high must be greater than or equal to open and close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be less than or equal to open and close")
        if self.low > self.high:
            raise ValueError("low must be less than or equal to high")
        return self

