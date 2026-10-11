"""Causal signal outcomes, market response, and separate position accounting (B1)."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import math
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from model_testing.outcome_contracts import FrozenModel, OutcomeContract, PositionContract, content_hash
from model_testing.outcome_scope import BaselineScope, load_scope
from model_testing.outcome_atr import StructuralATREvidence, Structural30mATREvidence, STRUCTURAL_ATR_MINUTES
from model_testing.outcome_diagnostics import ModelSetupContext


class PaperOutcome(StrEnum):
    NOT_EVALUATED = "NOT_EVALUATED"
    TARGET = "TARGET"
    STOP = "STOP"
    TIMEOUT_WIN = "TIMEOUT_WIN"
    TIMEOUT_LOSS = "TIMEOUT_LOSS"
    TIMEOUT_FLAT = "TIMEOUT_FLAT"
    NO_ENTRY = "NO_ENTRY"
    DATA_GAP = "DATA_GAP"
    INVALID = "INVALID"
    AMBIGUOUS = "AMBIGUOUS"


# b1-v5: opt-in mandatory bar completion, open-position marks, shared cost cases.
# b1-v4 is reproducible only from the engine_source archived with each b1-v4 run.
ENGINE_VERSION = "b1-v5"
COST_CASES = {"base": 1.0, "double_all_costs": 2.0}
RESOLVED = {"TARGET", "STOP", "TIMEOUT_WIN", "TIMEOUT_LOSS", "TIMEOUT_FLAT"}
TIME_FIELDS = ("event_anchor_ts", "known_at_ts", "signal_bar_close_ts", "confirmation_ts",
               "features_available_at", "feature_max_source_close_ts", "regime_available_at",
               "prediction_available_at", "calibration_available_at", "policy_available_at",
               "geometry_available_at", "atr_available_at")


class SignalEvidence(FrozenModel):
    event_id: str
    model_id: str
    model_version: str
    model_family: str
    strategy: Literal["ote", "frvp", "ict"]
    setup_family: str
    instrument: Literal["ES", "6E", "EURUSD"]
    timeframe_minutes: Literal[1, 5] = 5
    side: Literal["long", "short"]
    training_label_outcome: int | float | str | None = None
    training_label_version: str = "UNKNOWN_LEGACY"
    training_label_hash: str | None = None
    training_label_provenance: Literal["verified", "unknown_legacy"] = "unknown_legacy"
    event_anchor_ts: datetime
    known_at_ts: datetime
    signal_bar_close_ts: datetime
    confirmation_ts: datetime
    features_available_at: datetime
    feature_max_source_close_ts: datetime
    regime_available_at: datetime
    prediction_available_at: datetime
    calibration_available_at: datetime
    policy_available_at: datetime
    geometry_available_at: datetime
    atr_available_at: datetime
    availability_status: Literal["verified", "unverified"]
    availability_evidence_refs: tuple[str, ...]
    atr: float | None = Field(default=None, gt=0)
    atr_definition: str
    atr_provenance: StructuralATREvidence | Structural30mATREvidence | None = None
    audit_context: ModelSetupContext | None = None
    stop_reference: float | None = None
    target_reference: float | None = None
    coordinate_id: str | None = None
    source_structure_formed_ts: datetime | None = None
    structure_activation_ts: datetime | None = None
    setup_retest_ts: datetime | None = None
    # Policy and sizing are used exclusively by the position simulator.
    policy_eligible: bool = False
    policy_rejection_reason: str = "not_qualified"
    quantity: float = Field(default=1.0, gt=0)

    @field_validator(*TIME_FIELDS, "source_structure_formed_ts", "structure_activation_ts", "setup_retest_ts")
    @classmethod
    def explicit_timezone(cls, value):
        if value is None:
            return value
        if value.tzinfo is None:
            raise ValueError("Availability timestamps require an explicit timezone")
        return value.astimezone(timezone.utc)


def utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Bar and schedule timestamps require explicit timezones")
    return timestamp.tz_convert("UTC")


class EvaluationTape:
    """Unfilled OHLC plus an independently supplied, versioned expected-open calendar.

    Schedule columns: open_ts, session_id. Missing rows in the OHLC tape do not
    change the schedule. Holidays/maintenance are omitted ONLY by this calendar.
    Bar timestamps are opens; closes occur timeframe_minutes later.
    require_completion makes a bar without is_complete == True unusable, including
    when the column was never supplied.
    """

    def __init__(self, bars: pd.DataFrame, schedule: pd.DataFrame, *, timeframe_minutes: int,
                 calendar_id: str, calendar_source: str, coverage_start, coverage_end,
                 finer_bars: pd.DataFrame | None = None, require_completion: bool = False):
        if not calendar_id or not calendar_source:
            raise ValueError("Calendar identity and evidence source are required")
        self.delta = pd.Timedelta(minutes=timeframe_minutes)
        self.calendar_id = calendar_id
        self.calendar_source = calendar_source
        self.coverage_start, self.coverage_end = utc(coverage_start), utc(coverage_end)
        self.schedule = self._index(schedule)
        if "session_id" not in self.schedule or self.schedule["session_id"].isna().any():
            raise ValueError("Expected calendar needs session IDs")
        if self.schedule.empty or self.coverage_end <= self.coverage_start:
            raise ValueError("Empty calendar coverage")
        if self.schedule.index[0] < self.coverage_start or self.schedule.index[-1] + self.delta > self.coverage_end:
            raise ValueError("Schedule outside declared coverage")
        if len(self.schedule) > 1 and (self.schedule.index.to_series().diff().dropna() < self.delta).any():
            raise ValueError("Overlapping scheduled bars")
        self.bars = self._index(bars)
        if not set(self.bars.index).issubset(set(self.schedule.index)):
            raise ValueError("OHLC contains bars outside the expected calendar")
        self.finer = None if finer_bars is None else self._index(finer_bars)
        self.require_completion = bool(require_completion)
        for frame in (self.bars, self.finer):
            if self.require_completion and frame is not None and "is_complete" not in frame:
                frame["is_complete"] = np.nan
        self.schedule_hash = content_hash({"calendar_id": calendar_id, "source": calendar_source,
            "coverage_start": self.coverage_start.isoformat(), "coverage_end": self.coverage_end.isoformat(),
            "opens": [(t.isoformat(), str(row.session_id)) for t, row in self.schedule.iterrows()]})

    @staticmethod
    def _index(frame):
        result = frame.copy(deep=True)
        if "open_ts" not in result:
            raise ValueError("Explicit open_ts column required; do not guess timestamp convention")
        result.index = pd.DatetimeIndex([utc(t) for t in result.pop("open_ts")])
        if result.index.has_duplicates or not result.index.is_monotonic_increasing:
            raise ValueError("Bar timestamps must be sorted and unique")
        return result

    def row(self, timestamp):
        return None if timestamp not in self.bars.index else self.bars.loc[timestamp]


def valid_ohlc(row) -> bool:
    if row is None:
        return False
    try:
        o, h, low, c = (float(row[k]) for k in ("open", "high", "low", "close"))
        return bool(np.isfinite([o, h, low, c]).all() and low <= min(o, c) <= max(o, c) <= h)
    except (KeyError, ValueError, TypeError):
        return False


def causal_error(signal: SignalEvidence, contract: OutcomeContract) -> str | None:
    if (signal.instrument, signal.strategy, signal.setup_family, signal.timeframe_minutes) != (
            contract.instrument, contract.strategy, contract.setup_family, contract.timeframe_minutes):
        return "contract_family_mismatch"
    if signal.model_family not in contract.model_families:
        return "model_family_not_bound"
    if contract.side not in {"long_or_short", signal.side}:
        return "contract_side_mismatch"
    if signal.availability_status != "verified" or not signal.availability_evidence_refs:
        return "availability_unverified"
    known = signal.known_at_ts
    if signal.event_anchor_ts > known:
        return "anchor_after_known_at"
    if signal.confirmation_ts < signal.event_anchor_ts:
        return "confirmation_before_anchor"
    if any(getattr(signal, field) > known for field in TIME_FIELDS if field not in {"known_at_ts", "event_anchor_ts"}):
        return "input_unavailable_at_decision"
    if signal.feature_max_source_close_ts > signal.features_available_at:
        return "feature_source_after_feature_availability"
    if not (signal.features_available_at <= signal.prediction_available_at <= signal.calibration_available_at <= signal.policy_available_at):
        return "prediction_pipeline_out_of_order"
    if signal.regime_available_at > signal.policy_available_at:
        return "regime_unavailable_to_policy"
    if signal.instrument in {"ES", "6E"} and not signal.coordinate_id:
        return "missing_contract_coordinate"
    if signal.setup_family == "classic_breaker":
        formed, activated, retested = signal.source_structure_formed_ts, signal.structure_activation_ts, signal.setup_retest_ts
        if None in (formed, activated, retested):
            return "missing_breaker_causal_lineage"
        if not formed < activated < retested <= signal.signal_bar_close_ts <= known:
            return "invalid_breaker_causal_lineage"
    if signal.training_label_provenance == "verified" and (
            not signal.training_label_hash or signal.training_label_version == "UNKNOWN_LEGACY"):
        return "invalid_training_label_provenance"
    return None


def _path_error(row, signal):
    if not valid_ohlc(row):
        return "missing_or_corrupt_bar"
    if "is_complete" in row and (pd.isna(row["is_complete"]) or row["is_complete"] != True):
        return "provisional_or_unknown_completion_bar"
    if signal.instrument in {"ES", "6E"}:
        if str(row.get("coordinate_id", "")) != signal.coordinate_id:
            return "roll_coordinate_change_or_unknown"
    if bool(row.get("invalid_roll", False)):
        return "invalid_roll_boundary"
    return None


def _barriers(signal, contract, entry):
    risk, sign = contract.risk, 1 if signal.side == "long" else -1
    if signal.atr is None or signal.atr_definition != risk.atr_definition:
        raise ValueError("missing_or_wrong_causal_atr")
    if risk.atr_definition in STRUCTURAL_ATR_MINUTES and structural_atr_error(signal):
        raise ValueError(structural_atr_error(signal))
    distance = risk.stop_atr * signal.atr
    if risk.rule == "structural":
        distance = max(distance, risk.min_stop_ticks * contract.costs.price_increment)
        if signal.stop_reference is not None:
            distance = max(distance, sign * (entry - signal.stop_reference))
        target_distance = max(risk.target_atr * signal.atr, distance * risk.min_target_rr)
        if signal.target_reference is not None:
            target_distance = max(target_distance, sign * (signal.target_reference - entry))
    else:
        target_distance = risk.target_atr * signal.atr
    stop, target = entry - sign * distance, entry + sign * target_distance
    if signal.instrument in {"ES", "6E"}:
        tick = contract.costs.price_increment
        # No nontradable fractional futures tick fills. Targets round away
        # (harder to reach); stops round away (no optimistic reduction in loss).
        down = lambda price: math.floor(price / tick + 1e-10) * tick
        up = lambda price: math.ceil(price / tick - 1e-10) * tick
        stop, target = (down(stop), up(target)) if sign == 1 else (up(stop), down(target))
    return stop, target


def structural_atr_error(signal) -> str | None:
    proof = signal.atr_provenance
    if proof is None:
        return "structural_atr_provenance_missing"
    if (proof.definition != signal.atr_definition or proof.coordinate_id != signal.coordinate_id
            or (signal.instrument in {"ES", "6E"} and not proof.coordinate_id)):
        return "structural_atr_identity_mismatch"
    if (proof.source_close_ts > proof.available_at_ts or proof.available_at_ts > signal.atr_available_at
            or signal.atr_available_at > signal.known_at_ts or proof.history_start_ts >= proof.source_close_ts):
        return "structural_atr_not_available"
    minutes = proof.timeframe_minutes
    if (utc(proof.source_close_ts) != utc(proof.source_close_ts).floor(f"{minutes}min")
            or proof.source_close_ts - proof.history_start_ts < pd.Timedelta(minutes=14 * minutes)):
        return "structural_atr_invalid_hour_or_warmup"
    if signal.atr is None or not np.isclose(proof.value, signal.atr, rtol=1e-12, atol=0):
        return "structural_atr_value_mismatch"
    return None


def _touch(row, side, stop, target):
    sign = 1 if side == "long" else -1
    # A stop gap pays the adverse open; target gaps receive no favorable improvement.
    if sign * (float(row["open"]) - stop) <= 0:
        return "STOP", float(row["open"]), "adverse_gap_open", False
    if sign * (float(row["open"]) - target) >= 0:
        return "TARGET", target, "target_at_open", False
    stop_hit = row["low"] <= stop if sign == 1 else row["high"] >= stop
    target_hit = row["high"] >= target if sign == 1 else row["low"] <= target
    if stop_hit:
        return "STOP", stop, "adverse_first" if target_hit else "5m_single_touch", bool(target_hit)
    if target_hit:
        return "TARGET", target, "5m_single_touch", False
    return None


def _resolve_touch(tape, ts, row, signal, stop, target):
    hit = _touch(row, signal.side, stop, target)
    if hit is None:
        return None
    outcome, price, method, collision = hit
    if method == "5m_single_touch" and tape.delta == pd.Timedelta(minutes=1):
        method = "1m_single_touch"
    exit_ts = ts if method.endswith("open") else ts + tape.delta
    if collision and tape.finer is not None and tape.delta > pd.Timedelta(minutes=1):
        index = pd.date_range(ts, ts + tape.delta, freq="1min", inclusive="left")
        # Require complete coverage AND agreement with the coarse bar. Never search
        # retrospectively in the signal bar or accept a partial favorable subpath.
        if index.isin(tape.finer.index).all() and (index > utc(signal.known_at_ts)).all():
            fine = tape.finer.loc[index]
            consistent = all(_path_error(r, signal) is None for _, r in fine.iterrows())
            consistent = consistent and np.allclose(
                [fine.iloc[0]["open"], fine.high.max(), fine.low.min(), fine.iloc[-1]["close"]],
                [row["open"], row["high"], row["low"], row["close"]], rtol=0, atol=1e-9)
            if consistent:
                for minute, fine_row in fine.iterrows():
                    fine_hit = _touch(fine_row, signal.side, stop, target)
                    if fine_hit is not None:
                        outcome, price, fine_method, fine_collision = fine_hit
                        method = "1m_adverse_first" if fine_collision else "1m_first_touch"
                        exit_ts = minute if fine_method.endswith("open") else minute + pd.Timedelta(minutes=1)
                        break
    return outcome, price, method, collision, exit_ts


def evaluate_signal(signal: SignalEvidence, tape: EvaluationTape, contract: OutcomeContract,
                    *, scope: BaselineScope | None = None) -> dict:
    """Evaluate B and C without reading training outcome, quantity, or policy eligibility."""
    scope = scope or load_scope()
    scope_reason = scope.signal_reason(signal)
    result = {field: getattr(signal, field) for field in (
        "event_id", "model_id", "model_version", "model_family", "strategy", "setup_family", "instrument", "side",
        "event_anchor_ts", "known_at_ts", "training_label_outcome", "training_label_version", "training_label_hash",
        "training_label_provenance")}
    result.update(evaluation_contract_id=contract.contract_id, evaluation_contract_version=contract.contract_version,
                  baseline_scope_id=scope.scope_id, baseline_scope_hash=scope.sha256,
                  baseline_scope="diagnostics_only" if scope_reason else "included",
                  baseline_scope_reason=scope_reason or "",
                  evaluation_contract_hash=contract.sha256, calendar_id=tape.calendar_id, calendar_hash=tape.schedule_hash,
                  contract_status="UNRESOLVED" if contract.risk.rule == "UNRESOLVED" else "FROZEN",
                  entry_ts=None, entry_price=None, stop_price=None, target_price=None, timeout_ts=None,
                  paper_contract_outcome="INVALID", paper_exit_ts=None, paper_exit_price=None,
                  paper_signed_price_movement=None, paper_mfe=None, paper_mae=None, paper_resolved=False,
                  barrier_resolution_method="not_applicable", ambiguous_bar_resolution_required=False,
                  data_quality_flags=[], paper_outcome_reason="", forward_mfe=None, forward_mae=None,
                  paper_mark_ts=None, paper_mark_price=None, paper_unrealized_price_movement=None)
    result["paper_excursion_convention"] = "whole_exit_bar_bounds"
    atr_error = (structural_atr_error(signal) if contract.risk.atr_definition in STRUCTURAL_ATR_MINUTES
                 else None)
    atr_valid = bool(signal.atr and signal.atr_definition == contract.risk.atr_definition and not atr_error)
    result.update(atr_price=signal.atr, atr_definition=signal.atr_definition,
                  atr_status="verified_definition" if atr_valid else (atr_error or "missing_or_wrong_causal_atr"),
                  atr_source_close_ts=signal.atr_provenance.source_close_ts if signal.atr_provenance else None,
                  atr_source_hash=signal.atr_provenance.source_hash if signal.atr_provenance else None)
    for horizon in contract.forward_horizons:
        for suffix in ("", "_units", "_pct", "_log", "_atr", "_mfe", "_mae"):
            result[f"forward_return_{horizon}{suffix}"] = None
        result[f"forward_return_{horizon}_status"] = "NO_ENTRY"
    error = causal_error(signal, contract)
    if error or tape.delta != pd.Timedelta(minutes=contract.timeframe_minutes):
        result["paper_outcome_reason"] = error or "timeframe_mismatch"
        return result
    known = utc(signal.known_at_ts)
    if not tape.coverage_start <= known < tape.coverage_end:
        result.update(paper_contract_outcome="NO_ENTRY", paper_outcome_reason="calendar_out_of_coverage")
        return result
    entry_index = int(tape.schedule.index.searchsorted(known, side="right"))
    if entry_index >= len(tape.schedule):
        result.update(paper_contract_outcome="NO_ENTRY", paper_outcome_reason="end_of_calendar")
        return result
    entry_ts = tape.schedule.index[entry_index]
    entry_row = tape.row(entry_ts)
    error = _path_error(entry_row, signal)
    if error:
        result.update(paper_contract_outcome="DATA_GAP", paper_outcome_reason="entry_" + error,
                      data_quality_flags=["entry_" + error])
        return result
    assert entry_ts > known >= utc(signal.event_anchor_ts)
    entry = float(entry_row["open"])
    result.update(entry_ts=entry_ts, entry_price=entry)
    sign = 1 if signal.side == "long" else -1
    # Diagnostics continue past a paper stop/target and never consult its barriers.
    max_favorable, max_adverse = 0.0, 0.0
    for offset in range(max(contract.forward_horizons)):
        i = entry_index + offset
        error = "incomplete_horizon" if i >= len(tape.schedule) else _path_error(tape.row(tape.schedule.index[i]), signal)
        if error:
            for horizon in contract.forward_horizons:
                if horizon > offset:
                    result[f"forward_return_{horizon}_status"] = error
            break
        row = tape.row(tape.schedule.index[i])
        favorable = sign * (float(row["high" if sign == 1 else "low"]) - entry)
        adverse = sign * (float(row["low" if sign == 1 else "high"]) - entry)
        max_favorable, max_adverse = max(max_favorable, favorable), min(max_adverse, adverse)
        horizon = offset + 1
        if horizon in contract.forward_horizons:
            move = sign * (float(row["close"]) - entry)
            result.update({f"forward_return_{horizon}": move,
                f"forward_return_{horizon}_units": move / contract.costs.price_increment,
                f"forward_return_{horizon}_pct": move / entry if entry else None,
                f"forward_return_{horizon}_log": sign * np.log(float(row["close"]) / entry) if entry > 0 and row["close"] > 0 else None,
                f"forward_return_{horizon}_atr": move / signal.atr if atr_valid else None,
                f"forward_return_{horizon}_mfe": max_favorable, f"forward_return_{horizon}_mae": max_adverse,
                f"forward_return_{horizon}_status": "OK"})
    result["forward_mfe"] = result["forward_return_48_mfe"]
    result["forward_mae"] = result["forward_return_48_mae"]
    if scope_reason:
        result.update(paper_contract_outcome="NOT_EVALUATED", paper_outcome_reason=scope_reason)
        return result
    if contract.risk.rule == "UNRESOLVED":
        result["paper_outcome_reason"] = "UNRESOLVED: " + contract.risk.unresolved_reason
        return result
    try:
        stop, target = _barriers(signal, contract, entry)
    except ValueError as exc:
        result["paper_outcome_reason"] = str(exc)
        return result
    result.update(stop_price=stop, target_price=target)
    last_index = entry_index + contract.risk.timeout_bars - 1
    if last_index < len(tape.schedule):
        result["timeout_ts"] = tape.schedule.index[last_index] + tape.delta
    session = tape.schedule.iloc[entry_index].session_id
    mfe, mae, mark = 0.0, 0.0, None
    for i in range(entry_index, last_index + 1):
        if i >= len(tape.schedule):
            result.update(paper_contract_outcome="DATA_GAP", paper_outcome_reason="incomplete_holding_horizon",
                          data_quality_flags=["incomplete_holding_horizon"])
            break
        ts = tape.schedule.index[i]
        row = tape.row(ts)
        error = _path_error(row, signal)
        if error:
            result.update(paper_contract_outcome="DATA_GAP", paper_outcome_reason=error, data_quality_flags=[error])
            break
        mark = (ts + tape.delta, float(row["close"]))
        hit = _resolve_touch(tape, ts, row, signal, stop, target)
        mfe = max(mfe, sign * (float(row["high" if sign == 1 else "low"]) - entry))
        mae = min(mae, sign * (float(row["low" if sign == 1 else "high"]) - entry))
        session_end = (contract.overnight == "exit_before_session_boundary" and i + 1 < len(tape.schedule)
                       and tape.schedule.iloc[i + 1].session_id != session)
        if hit is not None:
            outcome, price, method, collision, exit_ts = hit
            result.update(paper_contract_outcome=outcome, paper_exit_price=price, paper_exit_ts=exit_ts,
                          barrier_resolution_method=method, ambiguous_bar_resolution_required=collision)
        elif i == last_index or session_end:
            price = float(row["close"])
            move = sign * (price - entry)
            outcome = "TIMEOUT_WIN" if move > 0 else "TIMEOUT_LOSS" if move < 0 else "TIMEOUT_FLAT"
            result.update(paper_contract_outcome=outcome, paper_exit_price=price, paper_exit_ts=ts + tape.delta,
                          barrier_resolution_method="session_boundary_close" if session_end else "timeout_close")
        else:
            continue
        result.update(paper_resolved=True, paper_signed_price_movement=sign * (result["paper_exit_price"] - entry))
        break
    if not result["paper_resolved"] and mark is not None:
        # Still open when the usable path ended: last completed close, never an exit.
        result.update(paper_mark_ts=mark[0], paper_mark_price=mark[1],
                      paper_unrealized_price_movement=sign * (mark[1] - entry))
    # OHLC extrema in the exit bar are upper bounds, not an asserted tick path.
    result.update(paper_mfe=mfe, paper_mae=mae, paper_excursion_convention="whole_exit_bar_bounds")
    return result


def session_at(timestamp) -> str:
    ts = utc(timestamp)
    london = ts.tz_convert("Europe/London")
    ny = ts.tz_convert("America/New_York")
    in_london, in_ny = 7 <= london.hour < 16, 8 <= ny.hour < 17
    if in_london and in_ny:
        return "overlap"
    if in_london:
        return "london"
    if in_ny:
        return "new_york"
    return "asia" if ny.hour >= 19 or ny.hour < 2 else "off_hours"


def position_costs(*, entry_ts, exit_ts, cost, quantity: float, accounting_unit: str) -> dict:
    if accounting_unit != cost.unit and accounting_unit != "cash":
        raise ValueError("Cannot combine tick and pip ledgers")
    if accounting_unit == "cash" and cost.cash_per_unit is None:
        raise ValueError("Cash conversion is undefined for this instrument/quantity contract")
    multiplier = quantity * (cost.cash_per_unit if accounting_unit == "cash" else 1.0)
    spreads = dict(cost.session_spreads)
    spread = spreads[session_at(entry_ts)] + spreads[session_at(exit_ts)]
    slippage = cost.fixed_slippage_round_trip + cost.spread_slippage_multiplier * spread
    values = dict(spread_cost=spread * multiplier, slippage_cost=slippage * multiplier,
                  commission_cost=cost.commission_round_trip * multiplier)
    values["total_cost"] = sum(values.values())
    return values


def simulate_positions(signals: list[SignalEvidence], outcomes: list[dict],
                       contracts: dict[str, OutcomeContract], simulator: PositionContract,
                       *, scope: BaselineScope | None = None,
                       priority: dict[str, int] | None = None) -> list[dict]:
    """Chronological admission. Unknown exits retain capacity; no hindsight rejection.

    priority (model_id -> frozen selection rank) is the instrument-capacity ordering:
    decision time, then selection rank, then event ID; an exit at a bar's open then
    does not free capacity for an entry at that same open.
    """
    evidence = {s.event_id: s for s in signals}
    scope = scope or load_scope()
    if len(evidence) != len(signals) or len(outcomes) != len(signals):
        raise ValueError("Each signal needs one unique outcome")
    positions, active, last_entry = [], [], {}
    order = ((lambda r: (utc(r["entry_ts"] or r["known_at_ts"]), r["event_id"])) if priority is None else
             (lambda r: (utc(r["known_at_ts"]), priority[r["model_id"]], r["event_id"])))
    for paper in sorted(outcomes, key=order):
        s = evidence[paper["event_id"]]
        contract = contracts[paper["evaluation_contract_id"]]
        row = {"event_id": s.event_id, "position_taken": False, "no_trade_reason": "",
               "simulator_id": simulator.simulator_id, "simulator_hash": simulator.sha256,
               "policy_id": simulator.policy_id, "policy_hash": simulator.policy_hash,
               "quantity": s.quantity, "accounting_unit": simulator.accounting_unit,
               "gross_pnl": None, "spread_cost": None, "slippage_cost": None, "commission_cost": None,
               "total_cost": None, "net_pnl": None, "position_status": "NOT_TAKEN", "cost_case": "base",
               "mark_ts": None, "mark_price": None, "unrealized_gross_pnl": None}
        row["position_exit_ts"] = None
        reason = None
        scope_reason = scope.signal_reason(s)
        if scope_reason:
            reason = scope_reason
        elif paper.get("baseline_scope") == "diagnostics_only":
            reason = paper["baseline_scope_reason"]
        elif not s.policy_eligible:
            reason = s.policy_rejection_reason or "policy_rejected"
        elif paper["entry_ts"] is None:
            reason = paper["paper_outcome_reason"] or "no_entry"
        elif paper["paper_contract_outcome"] == "INVALID":
            reason = paper["paper_outcome_reason"]
        if reason:
            row["no_trade_reason"] = reason
            positions.append(row)
            continue
        ts = utc(paper["entry_ts"])
        active = [a for a in active if a["exit"] is None or a["exit"] > ts
                  or (priority is not None and a["at_open"] and a["exit"] == ts)]
        position_scope = s.instrument if simulator.concurrency_scope == "instrument" else (s.instrument, s.model_id)
        scoped = [a for a in active if a["scope"] == position_scope]
        if len(scoped) >= simulator.max_positions:
            reason = "concurrent_position_limit"
        elif sum(a["quantity"] for a in scoped) + s.quantity > simulator.max_quantity:
            reason = "quantity_limit"
        elif not simulator.allow_opposite and any(a["side"] != s.side for a in scoped):
            reason = "opposite_position"
        elif position_scope in last_entry and ts < last_entry[position_scope] + pd.Timedelta(minutes=simulator.cooldown_minutes):
            reason = "cooldown"
        if reason:
            row["no_trade_reason"] = reason
        else:
            row.update(position_taken=True, position_status="CLOSED" if paper["paper_resolved"] else "UNRESOLVED")
            active.append(dict(scope=position_scope, quantity=s.quantity, side=s.side, exit=paper["paper_exit_ts"],
                               at_open=str(paper["barrier_resolution_method"]).endswith("open")))
            last_entry[position_scope] = ts
            if not paper["paper_resolved"] and paper.get("paper_unrealized_price_movement") is not None:
                position_costs(entry_ts=paper["entry_ts"], exit_ts=paper["paper_mark_ts"], cost=contract.costs,
                               quantity=s.quantity, accounting_unit=simulator.accounting_unit)  # unit guard only
                multiplier = s.quantity * (contract.costs.cash_per_unit if simulator.accounting_unit == "cash" else 1)
                row.update(mark_ts=paper["paper_mark_ts"], mark_price=paper["paper_mark_price"],
                           unrealized_gross_pnl=paper["paper_unrealized_price_movement"] / contract.costs.price_increment * multiplier)
            if paper["paper_resolved"]:
                costs = position_costs(entry_ts=paper["entry_ts"], exit_ts=paper["paper_exit_ts"], cost=contract.costs,
                                       quantity=s.quantity, accounting_unit=simulator.accounting_unit)
                multiplier = s.quantity * (contract.costs.cash_per_unit if simulator.accounting_unit == "cash" else 1)
                gross = (1 if s.side == "long" else -1) * (paper["paper_exit_price"] - paper["entry_price"]) / contract.costs.price_increment * multiplier
                row.update(costs, gross_pnl=gross, net_pnl=gross - costs["total_cost"], position_exit_ts=paper["paper_exit_ts"])
                assert np.isclose(row["net_pnl"], row["gross_pnl"] - row["total_cost"], rtol=0, atol=1e-10)
        positions.append(row)
    return positions


def reprice_positions(positions: list[dict], cost_case: str) -> list[dict]:
    """Same admitted trades and fills at another cost case; admission is never re-run."""
    factor = COST_CASES[cost_case]
    repriced = []
    for position in positions:
        if position.get("cost_case") != "base":
            raise ValueError("Cost cases are derived from the base ledger only")
        row = dict(position, cost_case=cost_case)
        if row["net_pnl"] is not None:
            for key in ("spread_cost", "slippage_cost", "commission_cost"):
                row[key] = position[key] * factor
            row["total_cost"] = row["spread_cost"] + row["slippage_cost"] + row["commission_cost"]
            row["net_pnl"] = row["gross_pnl"] - row["total_cost"]
        repriced.append(row)
    return repriced
