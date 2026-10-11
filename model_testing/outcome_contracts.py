"""B1 contracts. Training labels and archived evaluators are never rewritten here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ROOT = Path(__file__).resolve().parents[1]
# Explicit active release; archived catalogues remain selectable for replay.
CATALOG = ROOT / "configs" / "outcome_contracts" / "b1_v3.json"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @property
    def sha256(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class CostContract(FrozenModel):
    cost_model_id: str
    version: int = Field(ge=1)
    instrument: Literal["ES", "6E", "EURUSD"]
    unit: Literal["ticks", "pips"]
    price_increment: float = Field(gt=0)
    cash_per_unit: float | None = Field(default=None, gt=0)
    spread_mode: Literal["session_schedule"] = "session_schedule"
    # Existing Phase 6/7 code charges ONE FULL configured spread at EACH leg.
    session_spreads: tuple[tuple[str, float], ...]
    spread_interpretation: Literal["full_configured_spread_per_side"] = "full_configured_spread_per_side"
    fixed_slippage_round_trip: float = Field(ge=0)
    commission_round_trip: float = Field(ge=0)
    spread_slippage_multiplier: float = Field(default=0, ge=0)
    session_clock: Literal["london_07_16_ny_08_17_asia_ny_19_02_v1"] = "london_07_16_ny_08_17_asia_ny_19_02_v1"
    adjustments: Literal["none"] = "none"

    @model_validator(mode="after")
    def check_spreads(self):
        table = dict(self.session_spreads)
        if set(table) != {"overlap", "london", "new_york", "asia", "off_hours"} or len(table) != len(self.session_spreads):
            raise ValueError("All five sessions must occur exactly once")
        if any(v < 0 for v in table.values()):
            raise ValueError("Negative spread")
        return self


class RiskContract(FrozenModel):
    rule: Literal["atr", "structural", "UNRESOLVED"]
    stop_atr: float | None = Field(default=None, gt=0)
    target_atr: float | None = Field(default=None, gt=0)
    min_stop_ticks: float = Field(default=1, gt=0)
    min_target_rr: float = Field(default=1.25, gt=0)
    timeout_bars: int | None = Field(default=None, gt=0)
    atr_definition: str
    source: str
    unresolved_reason: str = ""
    recommendation: str = ""

    @model_validator(mode="after")
    def complete(self):
        if self.rule != "UNRESOLVED" and None in (self.stop_atr, self.target_atr, self.timeout_bars):
            raise ValueError("Executable risk requires stop, target and timeout")
        if self.rule == "UNRESOLVED" and not self.unresolved_reason:
            raise ValueError("Unresolved contracts require an explanation")
        return self


class OutcomeContract(FrozenModel):
    contract_id: str
    contract_version: int = Field(ge=1)
    instrument: Literal["ES", "6E", "EURUSD"]
    timeframe_minutes: Literal[1, 5]
    strategy: Literal["ote", "frvp", "ict"]
    setup_family: str
    model_families: tuple[str, ...]
    side: Literal["long_or_short", "long", "short"] = "long_or_short"
    anchor_definition: str
    known_at_definition: str
    entry_rule: Literal["first_scheduled_open_strictly_after_known_at"] = "first_scheduled_open_strictly_after_known_at"
    max_entry_delay_bars: Literal[1] = 1
    entry_price: Literal["unadjusted_open"] = "unadjusted_open"
    barrier_price_grid: Literal["futures_outward_tick_fx_unrounded"] = "futures_outward_tick_fx_unrounded"
    risk: RiskContract
    paper_management: Literal["fixed_barriers"] = "fixed_barriers"
    concurrency: Literal["independent_every_signal"] = "independent_every_signal"
    barrier_resolution: Literal["complete_causal_1m_else_adverse_first"] = "complete_causal_1m_else_adverse_first"
    exit_price: Literal["barrier_or_adverse_stop_gap_open_timeout_close"] = "barrier_or_adverse_stop_gap_open_timeout_close"
    missing_entry: Literal["DATA_GAP_no_search"] = "DATA_GAP_no_search"
    missing_holding: Literal["DATA_GAP"] = "DATA_GAP"
    overnight: Literal["hold", "exit_before_session_boundary"]
    roll_rule: Literal["censor_coordinate_change"] = "censor_coordinate_change"
    forward_horizons: tuple[int, ...] = (1, 3, 6, 12, 24, 48)
    costs: CostContract

    @model_validator(mode="after")
    def identity(self):
        if not self.contract_id.endswith(f"_v{self.contract_version}"):
            raise ValueError("Contract ID must include version")
        if self.instrument != self.costs.instrument:
            raise ValueError("Cost instrument mismatch")
        if self.forward_horizons != (1, 3, 6, 12, 24, 48):
            raise ValueError("B1 requires all six standardized horizons")
        return self


class PositionContract(FrozenModel):
    """A deliberately separate fixed-barrier simulator. No production policy defaults."""
    simulator_id: str
    version: int = Field(ge=1)
    management: Literal["fixed_barriers"] = "fixed_barriers"
    policy_id: str
    policy_hash: str = Field(min_length=64, max_length=64)
    max_positions: int = Field(gt=0)
    max_quantity: float = Field(gt=0)
    concurrency_scope: Literal["instrument", "model"]
    allow_opposite: bool
    cooldown_minutes: int = Field(ge=0)
    starting_equity: float = Field(gt=0)
    accounting_unit: Literal["ticks", "pips", "cash"]

    @model_validator(mode="after")
    def identity(self):
        if not self.simulator_id.endswith(f"_v{self.version}"):
            raise ValueError("Simulator ID must include version")
        return self


def load_contracts(path: str | Path = CATALOG) -> dict[str, OutcomeContract]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    pins = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    contracts = {}
    for value in payload["contracts"]:
        contract = OutcomeContract.model_validate(value)
        if contract.contract_id in contracts:
            raise ValueError("Duplicate contract ID")
        if pins.get(contract.contract_id) != contract.sha256:
            raise ValueError(f"Frozen contract changed: {contract.contract_id}; publish a new version")
        contracts[contract.contract_id] = contract
    if set(pins) != set(contracts):
        raise ValueError("Catalogue and lock IDs differ")
    return contracts


def resolve_contract(*, instrument: str, strategy: str, model_family: str,
                     setup_family: str, timeframe_minutes: int = 5,
                     side: Literal["long", "short"] | None = None,
                     contracts: dict[str, OutcomeContract] | None = None) -> OutcomeContract:
    if side not in {None, "long", "short"}:
        raise ValueError("Contract side must be long or short")
    candidates = [c for c in (load_contracts() if contracts is None else contracts).values()
                  if (c.instrument, c.strategy, c.setup_family, c.timeframe_minutes)
                  == (instrument, strategy, setup_family, timeframe_minutes)
                  and model_family in c.model_families
                  and (c.side == "long_or_short" or c.side == side)]
    if len(candidates) != 1:
        raise ValueError(f"No unique frozen contract for {instrument}/{strategy}/{model_family}/{setup_family}/{side}/{timeframe_minutes}m; side is required for side-specific contracts")
    return candidates[0]
