"""The vectorized research evaluator must equal the B1 engine (evaluate_signal + simulate_positions)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_testing import b1_fast_outcomes as fast
from model_testing.outcome_atr import completed_structural_atr, structural_atr_at
from model_testing.outcome_contracts import PositionContract, content_hash, load_contracts
from model_testing.outcome_engine import TIME_FIELDS, EvaluationTape, SignalEvidence, evaluate_signal, simulate_positions
from model_testing.outcome_scope import load_research_scope

CASES = {"EURUSD": ("eurusd_ote_continuation_eval_v2", "continuation", "ote", 0.0001),
         "ES": ("es_frvp_setup2_eval_v2", "continuation_setup2", "frvp", 0.25)}


def synthetic(instrument: str, seed: int):
    rng = np.random.default_rng(seed)
    opens = pd.date_range("2024-01-08 00:00", periods=2600, freq="5min", tz="UTC")
    schedule = pd.DataFrame({"open_ts": opens, "session_id": (opens + pd.Timedelta(hours=3)).strftime("%Y-%m-%d")})
    tick = CASES[instrument][3]
    close = (1.1 if instrument == "EURUSD" else 4800.0) + np.cumsum(rng.normal(0, 4 * tick, len(opens)))
    if instrument == "ES":
        close = np.round(close / tick) * tick
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, tick, len(opens)) * (rng.random(len(opens)) < 0.05) * 20
    if instrument == "ES":
        open_ = np.round(open_ / tick) * tick
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 3 * tick, len(opens)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 3 * tick, len(opens)))
    bars = pd.DataFrame(dict(open_ts=opens, open=open_, high=high, low=low, close=close, is_complete=True))
    bars = bars.loc[rng.random(len(bars)) > 0.01].reset_index(drop=True)  # missing bars
    bars.loc[bars.sample(5, random_state=seed).index, "high"] = bars.low - tick  # corrupt bars
    if instrument == "ES":
        bars["coordinate_id"] = np.where(bars.open_ts < opens[1800], "ES:1", "ES:2")  # roll
    return bars, schedule


@pytest.mark.parametrize("instrument", ["EURUSD", "ES"])
def test_fast_evaluator_equals_engine(instrument):
    contract_id, family, strategy, _ = CASES[instrument]
    contract = load_contracts()[contract_id]
    bars, schedule = synthetic(instrument, 7 if instrument == "ES" else 3)
    start, end = schedule.open_ts.iloc[0], schedule.open_ts.iloc[-1] + pd.Timedelta(minutes=5)
    tape = EvaluationTape(bars.copy(), schedule, timeframe_minutes=5, calendar_id="t", calendar_source="t",
                          coverage_start=start, coverage_end=end, require_completion=True)
    minutes = 60 if contract.risk.atr_definition == "completed_1h_structural_atr" else 30
    hourly = completed_structural_atr(tape, timeframe_minutes=minutes, instrument=instrument)
    quick = fast.FastTape(bars, schedule, instrument=instrument, coverage_start=start, coverage_end=end)
    table = fast.structural_atr(quick, minutes)
    rng = np.random.default_rng(1)
    pos = np.sort(rng.choice(np.arange(150, len(schedule) - 10), 250, replace=False))
    side = np.where(rng.random(len(pos)) < 0.5, 1, -1)
    mine = fast.evaluate(quick, {minutes: table}, pos, side, np.zeros(len(pos), int), [contract])
    signals, outcomes, ids = [], [], []
    scope = load_research_scope()
    for i, (p, s) in enumerate(zip(pos, side)):
        ts = schedule.open_ts.iloc[p]
        if ts not in tape.bars.index:
            assert mine.outcome.iloc[i] == "NO_SIGNAL"
            continue
        close = ts + pd.Timedelta(minutes=5)
        known = close + pd.Timedelta(seconds=1)
        coordinate = str(tape.bars.coordinate_id[ts]) if instrument == "ES" else None
        proof = structural_atr_at(hourly, known, coordinate_id=coordinate)
        clocks = {f: known.to_pydatetime() for f in TIME_FIELDS}
        clocks.update(event_anchor_ts=ts.to_pydatetime(), signal_bar_close_ts=close.to_pydatetime(), confirmation_ts=close.to_pydatetime(),
                      feature_max_source_close_ts=close.to_pydatetime(),
                      atr_available_at=known.to_pydatetime() if proof is None else proof.available_at_ts)
        signal = SignalEvidence(event_id=f"e{i:04d}", model_id="t", model_version="t", model_family=family, strategy=strategy,
                                setup_family="setup2" if instrument == "ES" else "continuation", instrument=instrument,
                                side="long" if s > 0 else "short", **clocks, availability_status="verified", availability_evidence_refs=("t",),
                                atr=None if proof is None else proof.value, atr_definition=contract.risk.atr_definition,
                                atr_provenance=proof, coordinate_id=coordinate, policy_eligible=True, policy_rejection_reason="")
        o = evaluate_signal(signal, tape, contract, scope=scope)
        expected = "DATA_GAP_ENTRY" if o["paper_contract_outcome"] == "DATA_GAP" and o["entry_ts"] is None else o["paper_contract_outcome"]
        assert mine.outcome.iloc[i] == expected, i
        if o["paper_resolved"]:
            assert pd.Timestamp(o["paper_exit_ts"]).value == mine.exit_ns.iloc[i]
            assert o["paper_exit_price"] == pytest.approx(mine.exit_price.iloc[i], abs=1e-12)
        signals.append(signal)
        outcomes.append(o)
        ids.append(i)
    assert len(set(mine.outcome)) >= 4  # the sample exercises several outcome kinds
    unit = "pips" if instrument == "EURUSD" else "ticks"
    simulator = PositionContract(simulator_id="t_v1", version=1, policy_id="t", policy_hash=content_hash({"t": 1}), max_positions=1,
                                 max_quantity=1.0, concurrency_scope="instrument", allow_opposite=False, cooldown_minutes=0,
                                 starting_equity=1.0, accounting_unit=unit)
    positions = {p["event_id"]: p for p in simulate_positions(signals, outcomes, {contract.contract_id: contract}, simulator, scope=scope)}
    frame = mine.assign(event_id=[f"e{i:04d}" for i in range(len(mine))])
    taken = fast.ledger(frame, restart=None)
    for i in ids:
        p = positions[f"e{i:04d}"]
        assert bool(p["position_taken"]) == bool(taken[i]), i
        if p["position_taken"] and p["net_pnl"] is not None:
            assert p["net_pnl"] == pytest.approx(frame.net.iloc[i], abs=1e-9)
