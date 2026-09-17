"""Draft execution assumptions for A7 diagnostic replays, pending B1/B3.

No economic defaults or scored/live activation are supplied by this contract.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ote_live.ingestion.base import ensure_utc, timeframe_to_timedelta


class ResearchContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    version: Literal["es-research-execution-v1"] = "es-research-execution-v1"
    mode: Literal["diagnostic"] = "diagnostic"
    collection_version: str = Field(min_length=1)
    outcome_contract_ref: str = Field(min_length=1)
    comparison_contract_ref: str = Field(min_length=1)
    asset: Literal["ES"] = "ES"
    timeframe: str
    model_ids: tuple[str, ...]
    entry_mode: Literal["next_bar_open"] = "next_bar_open"
    entry_latency_seconds: float = Field(ge=0)
    entry_wait_seconds: float = Field(gt=0)
    max_bar_delay_seconds: float = Field(ge=0)
    tick_size: float = Field(gt=0)
    tick_value: float = Field(gt=0)
    stop_ticks: float = Field(gt=0)
    target_ticks: float = Field(gt=0)
    timeout_bars: int = Field(gt=0, strict=True)
    ambiguity: Literal["stop_first", "censor"]
    missing_bars: Literal["censor"] = "censor"
    quantity: int = Field(gt=0, strict=True)
    max_positions: int = Field(gt=0, strict=True)
    max_contracts: int = Field(gt=0, strict=True)
    concurrency_scope: Literal["per_model", "all_models"]
    allow_opposite_positions: bool
    # Costs are round-trip ticks per contract, charged once on resolution.
    spread_ticks: float = Field(ge=0)
    slippage_ticks_per_side: float = Field(ge=0)
    commission_ticks_round_trip: float = Field(ge=0)
    cost_multipliers: tuple[float, ...]

    @model_validator(mode="after")
    def validate_contract(self):
        if timeframe_to_timedelta(self.timeframe).total_seconds() <= 0:
            raise ValueError("timeframe must be positive")
        if not self.model_ids or len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model_ids must be nonempty and unique")
        if any(not model.startswith(("frvp_", "ict_")) for model in self.model_ids):
            raise ValueError("Only FRVP/ICT comparisons are supported")
        if (not self.cost_multipliers or len(set(self.cost_multipliers)) != len(self.cost_multipliers)
                or any(value < 1 for value in self.cost_multipliers) or 1.0 not in self.cost_multipliers):
            raise ValueError("Cost cases must be unique, include 1, and be at least 1")
        if self.quantity > self.max_contracts:
            raise ValueError("quantity exceeds max_contracts")
        return self


class ResearchOpportunity(BaseModel):
    """Explicit replay input. Production linkage uses submit_audit instead."""
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    opportunity_key: str = Field(min_length=1)
    collection_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    setup_event_id: int = Field(gt=0)
    prediction_id: int = Field(gt=0)
    signal_decision_id: int = Field(gt=0)
    setup_type: str = Field(min_length=1)
    setup_family: str = Field(min_length=1)
    detector_version: str = Field(min_length=1)
    policy_sha256: str = Field(min_length=1)
    source_bar_version: str = Field(min_length=1)
    source_timestamp: datetime
    setup_observed_at: datetime
    prediction_recorded_at: datetime
    side: Literal[-1, 1]
    setup_matched: bool
    threshold_crossed: bool
    full_policy_passed: bool
    common_rejection_reasons: tuple[str, ...] = ()
    policy_rejection_reasons: tuple[str, ...] = ()
    source_evidence: dict = Field(default_factory=dict)

    @field_validator("source_timestamp", "setup_observed_at", "prediction_recorded_at")
    @classmethod
    def utc(cls, value):
        if value.tzinfo is None:
            raise ValueError("Research times must have an explicit timezone")
        return ensure_utc(value)

    @model_validator(mode="after")
    def causal(self):
        if self.setup_observed_at < self.source_timestamp or self.prediction_recorded_at < self.setup_observed_at:
            raise ValueError("Setup must be known by prediction recording time")
        return self
