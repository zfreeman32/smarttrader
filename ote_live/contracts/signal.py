from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SignalDecision(BaseModel):
    """Final live runtime decision emitted for a model stream."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    direction: Literal["long", "short"]
    timestamp: datetime
    source_row_idx: int | None = None
    decision: Literal["emit", "abstain", "hold", "shadow"]
    probability: float
    threshold: float | None = None
    regime: str | None = None
    reasons: list[str] = Field(default_factory=list)
    cooldown_bars_remaining: int | None = None

