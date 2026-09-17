from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureSnapshot(BaseModel):
    """Auditable feature snapshot for one finalized bar."""

    model_config = ConfigDict(extra="forbid")

    asset: str = Field(default="EURUSD")
    timeframe: str = Field(default="5m")
    direction: Literal["long", "short"] | None = None
    timestamp: datetime
    collection_version: str = "legacy-unversioned"
    observation_metadata: dict = Field(default_factory=dict)
    snapshot_id: str | None = None
    generation_id: int | None = None
    source_row_idx: int | None = None
    feature_values: dict[str, float | int | bool | None]
    valid_feature_count: int | None = None
