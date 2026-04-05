from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelPrediction(BaseModel):
    """Runtime model prediction before the final policy decision."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    direction: Literal["long", "short"]
    backend: Literal["tcn", "lstm", "xgboost"]
    timestamp: datetime
    source_row_idx: int | None = None
    regime: str | None = None
    raw_score: float | None = None
    calibrated_probability: float
    threshold_applied: float | None = None
    threshold_source: Literal["global", "regime", "shadow", "unknown"] = "unknown"

