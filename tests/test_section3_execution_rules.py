"""Section 3 R3: one shared execution implementation, checked per rule and per stream.

Synthetic fixtures only. Contracts are the frozen B1 v3 catalogue entries, unmodified.
ES (ticks) and EURUSD (pips) are parametrized, never pooled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_testing.outcome_contracts import load_contracts
from model_testing.outcome_engine import (
    COST_CASES, ENGINE_VERSION, EvaluationTape, evaluate_signal, reprice_positions, simulate_positions,
)
from test_outcome_contracts import signal, simulator
from test_outcome_ote_contracts import ote_contract, ote_signal

STREAMS = ["ES", "EURUSD"]
PX = {"ES": 100.0, "EURUSD": 1.1}
ATR = {"ES": 1.0, "EURUSD": 0.002}
UNIT = {"ES": "ticks", "EURUSD": "pips"}
START = "2026-09-14T14:05:00Z"

pytestmark = pytest.mark.parametrize("stream", STREAMS)


def frozen_contract(stream):
    return load_contracts()["es_frvp_setup1_eval_v1"] if stream == "ES" else ote_contract()


def make_signal(stream, **changes):
    return signal(**changes) if stream == "ES" else ote_signal(**changes)


def bars(stream, n=80, *, times=None, complete=True):
    times = pd.date_range(START, periods=n, freq="5min") if times is None else pd.DatetimeIndex(times)
    px, atr = PX[stream], ATR[stream]
    # Distinct opens, so an earlier or later price can never be mistaken for the fill.
    opens = [px + i * 0.001 * atr for i in range(len(times))]
    frame = pd.DataFrame(dict(open_ts=times, open=opens, high=[o + 0.1 * atr for o in opens],
                              low=[o - 0.1 * atr for o in opens], close=opens))
    if stream == "ES":
        frame["coordinate_id"] = "ESU6"
    if complete is not None:
        frame["is_complete"] = complete
    return frame


def make_tape(stream, frame, *, schedule_times=None, sessions=None, require_completion=True):
    times = pd.DatetimeIndex(frame["open_ts"] if schedule_times is None else schedule_times)
    schedule = pd.DataFrame(dict(open_ts=times, session_id=sessions or ["s1"] * len(times)))
    return EvaluationTape(frame, schedule, timeframe_minutes=5, calendar_id=f"synthetic_{stream}_v1",
                          calendar_source="synthetic fixture", coverage_start=times[0] - pd.Timedelta(days=1),
                          coverage_end=times[-1] + pd.Timedelta(minutes=5), require_completion=require_completion)


def set_bar(frame, i, *, open=None, high=None, low=None, close=None):
    for key, value in dict(open=open, high=high, low=low, close=close).items():
        if value is not None:
            frame.loc[i, key] = value


def position(stream, s, outcome):
    c = frozen_contract(stream)
    row, = simulate_positions([s], [outcome], {c.contract_id: c}, simulator(accounting_unit=UNIT[stream]))
    return row


def barriers(stream, side="long"):
    result = evaluate_signal(make_signal(stream, side=side), make_tape(stream, bars(stream)), frozen_contract(stream))
    return result["stop_price"], result["target_price"]


def test_engine_version_is_bumped(stream):
    assert ENGINE_VERSION == "b1-v5" and COST_CASES == {"base": 1.0, "double_all_costs": 2.0}


@pytest.mark.parametrize("known,expected_entry", [
    ("2026-09-14T14:22:00Z", "2026-09-14T14:25:00Z"),  # late decision inside a bar
    ("2026-09-14T14:25:00Z", "2026-09-14T14:30:00Z"),  # backfilled exactly on an open: strictly after
])
def test_late_or_backfilled_decision_gets_next_open_after_actual_decision(stream, known, expected_entry):
    frame = bars(stream)
    s = make_signal(stream, known_at_ts=known, policy_available_at=known)
    result = evaluate_signal(s, make_tape(stream, frame), frozen_contract(stream))
    entry_ts = pd.Timestamp(expected_entry)
    assert result["entry_ts"] == entry_ts and result["entry_ts"] > pd.Timestamp(known)
    assert result["entry_price"] == frame.loc[frame.open_ts.eq(entry_ts), "open"].item()
    earlier = frame.loc[frame.open_ts.le(pd.Timestamp(known)), ["open", "close"]].to_numpy().ravel()
    assert result["entry_price"] not in set(earlier)
    assert position(stream, s, result)["position_taken"]


@pytest.mark.parametrize("defect", ["incomplete", "unknown_completion", "stale"])
def test_stale_or_incomplete_bar_is_a_recorded_rejection_and_no_trade(stream, defect):
    frame = bars(stream, complete=None if defect == "unknown_completion" else True)
    schedule_times = frame["open_ts"].copy()
    if defect == "incomplete":
        frame.loc[0, "is_complete"] = False
    if defect == "stale":  # newest available bar is older than the expected entry open
        frame = frame.drop(index=0).reset_index(drop=True)
    s = make_signal(stream)
    result = evaluate_signal(s, make_tape(stream, frame, schedule_times=schedule_times), frozen_contract(stream))
    reason = "entry_missing_or_corrupt_bar" if defect == "stale" else "entry_provisional_or_unknown_completion_bar"
    assert result["paper_contract_outcome"] == "DATA_GAP" and result["paper_outcome_reason"] == reason
    assert result["entry_ts"] is None and result["entry_price"] is None and result["paper_mark_price"] is None
    row = position(stream, s, result)
    assert not row["position_taken"] and row["no_trade_reason"] == reason
    assert row["gross_pnl"] is None and row["net_pnl"] is None


def test_completion_is_optional_only_when_not_required(stream):
    tape = make_tape(stream, bars(stream, complete=None), require_completion=False)
    assert evaluate_signal(make_signal(stream), tape, frozen_contract(stream))["entry_ts"] is not None


@pytest.mark.parametrize("side", ["long", "short"])
def test_barriers_are_fixed_at_entry_and_ignore_later_atr(stream, side):
    c, s, sign = frozen_contract(stream), make_signal(stream, side=side), 1 if side == "long" else -1
    calm = evaluate_signal(s, make_tape(stream, bars(stream)), c)
    frame = bars(stream)
    stop = calm["stop_price"]
    for i in range(3, 8):  # later ranges ten times wider than the decision-time ATR
        set_bar(frame, i, high=frame.loc[i, "open"] + (0.2 if side == "long" else 10) * ATR[stream],
                low=frame.loc[i, "open"] - (10 if side == "long" else 0.2) * ATR[stream])
    wild = evaluate_signal(s, make_tape(stream, frame), c)
    assert (wild["stop_price"], wild["target_price"]) == (calm["stop_price"], calm["target_price"])
    assert sign * (calm["entry_price"] - stop) == pytest.approx(c.risk.stop_atr * ATR[stream], abs=c.costs.price_increment)
    assert wild["paper_contract_outcome"] == "STOP" and wild["paper_exit_price"] == stop
    assert wild["paper_exit_ts"] == frame.loc[3, "open_ts"] + pd.Timedelta(minutes=5)


@pytest.mark.parametrize("side", ["long", "short"])
def test_same_bar_stop_target_collision_resolves_adverse_first(stream, side):
    stop, target = barriers(stream, side)
    frame = bars(stream)
    set_bar(frame, 2, high=max(stop, target) + ATR[stream], low=min(stop, target) - ATR[stream])
    s = make_signal(stream, side=side)
    result = evaluate_signal(s, make_tape(stream, frame), frozen_contract(stream))
    assert result["paper_contract_outcome"] == "STOP" and result["paper_exit_price"] == stop
    assert result["barrier_resolution_method"] == "adverse_first" and result["ambiguous_bar_resolution_required"]
    assert position(stream, s, result)["gross_pnl"] < 0


@pytest.mark.parametrize("side", ["long", "short"])
def test_gap_through_stop_fills_at_gap_open_not_at_barrier(stream, side):
    stop, _ = barriers(stream, side)
    sign = 1 if side == "long" else -1
    gap = stop - sign * 2 * ATR[stream]
    frame = bars(stream)
    set_bar(frame, 2, open=gap, high=gap + 0.1 * ATR[stream], low=gap - 0.1 * ATR[stream], close=gap)
    result = evaluate_signal(make_signal(stream, side=side), make_tape(stream, frame), frozen_contract(stream))
    assert result["paper_contract_outcome"] == "STOP" and result["barrier_resolution_method"] == "adverse_gap_open"
    assert result["paper_exit_price"] == gap != stop
    assert result["paper_exit_ts"] == frame.loc[2, "open_ts"]


@pytest.mark.parametrize("side", ["long", "short"])
def test_gap_through_target_gets_no_favorable_improvement(stream, side):
    # Frozen exit_price term: barrier_or_adverse_stop_gap_open_timeout_close.
    assert frozen_contract(stream).exit_price == "barrier_or_adverse_stop_gap_open_timeout_close"
    _, target = barriers(stream, side)
    gap = target + (1 if side == "long" else -1) * 2 * ATR[stream]
    frame = bars(stream)
    set_bar(frame, 2, open=gap, high=gap + 0.1 * ATR[stream], low=gap - 0.1 * ATR[stream], close=gap)
    result = evaluate_signal(make_signal(stream, side=side), make_tape(stream, frame), frozen_contract(stream))
    assert result["paper_contract_outcome"] == "TARGET" and result["paper_exit_price"] == target != gap


def weekend_times():
    friday = pd.date_range("2026-09-18T20:30:00Z", periods=6, freq="5min")  # last open 20:55
    sunday = pd.date_range("2026-09-20T22:00:00Z", periods=80, freq="5min")
    return friday.append(sunday), ["fri"] * 6 + ["sun"] * 80


def weekend_signal(stream, known):
    early = "2026-09-18T20:00:00Z"
    changes = {k: early for k in ("signal_bar_close_ts", "confirmation_ts", "features_available_at",
               "feature_max_source_close_ts", "regime_available_at", "prediction_available_at",
               "calibration_available_at", "policy_available_at", "geometry_available_at", "atr_available_at")}
    s = make_signal(stream, event_anchor_ts="2026-09-18T19:55:00Z", known_at_ts=known, **changes)
    proof = s.atr_provenance.model_copy(update=dict(source_close_ts=pd.Timestamp(early).to_pydatetime(),
        available_at_ts=pd.Timestamp(early).to_pydatetime(),
        history_start_ts=pd.Timestamp("2026-09-18T00:00:00Z").to_pydatetime()))
    return s.model_copy(update=dict(atr_provenance=proof))


def test_planned_closure_is_skipped_by_the_calendar_and_held_per_contract(stream):
    c = frozen_contract(stream)
    assert c.overnight == "hold"
    times, sessions = weekend_times()
    frame = bars(stream, times=times)
    tape = make_tape(stream, frame, sessions=sessions)
    held = evaluate_signal(weekend_signal(stream, "2026-09-18T20:50:00Z"), tape, c)
    assert held["entry_ts"] == pd.Timestamp("2026-09-18T20:55:00Z")
    assert held["paper_resolved"] and held["barrier_resolution_method"] == "timeout_close"
    # Timeout counts scheduled bars only; the closure is neither a gap nor elapsed holding time.
    assert held["paper_exit_ts"] == times[5 + c.risk.timeout_bars - 1] + pd.Timedelta(minutes=5)
    after_close = evaluate_signal(weekend_signal(stream, "2026-09-18T20:57:00Z"), tape, c)
    assert after_close["entry_ts"] == pd.Timestamp("2026-09-20T22:00:00Z")
    assert after_close["entry_price"] == frame.loc[6, "open"]


def test_unplanned_missing_bar_is_a_gap_not_a_closure(stream):
    times, sessions = weekend_times()
    frame = bars(stream, times=times).drop(index=8).reset_index(drop=True)
    s = weekend_signal(stream, "2026-09-18T20:50:00Z")
    result = evaluate_signal(s, make_tape(stream, frame, schedule_times=times, sessions=sessions), frozen_contract(stream))
    assert result["paper_contract_outcome"] == "DATA_GAP" and not result["paper_resolved"]
    assert result["paper_outcome_reason"] == "missing_or_corrupt_bar"


def test_roll_is_handled_per_contract(stream):
    c, frame = frozen_contract(stream), bars(stream)
    assert c.roll_rule == "censor_coordinate_change"
    if stream == "EURUSD":  # spot: no contract coordinate and no roll
        assert make_signal(stream).coordinate_id is None and "coordinate_id" not in frame
        assert evaluate_signal(make_signal(stream), make_tape(stream, frame), c)["paper_resolved"]
        return
    frame.loc[4:, "coordinate_id"] = "ESZ6"
    s = make_signal(stream)
    result = evaluate_signal(s, make_tape(stream, frame), c)
    assert result["paper_contract_outcome"] == "DATA_GAP"
    assert result["paper_outcome_reason"] == "roll_coordinate_change_or_unknown"
    row = position(stream, s, result)
    # Censored at the last bar of the decision-time contract; never continued on the new one.
    assert row["position_status"] == "UNRESOLVED" and row["mark_price"] == frame.loc[3, "close"]
    assert row["net_pnl"] is None


def test_position_open_at_window_end_is_marked_unresolved_and_not_zero(stream):
    c, frame = frozen_contract(stream), bars(stream, n=6)
    assert c.risk.timeout_bars > 6
    s = make_signal(stream)
    result = evaluate_signal(s, make_tape(stream, frame), c)
    assert result["paper_contract_outcome"] == "DATA_GAP" and not result["paper_resolved"]
    assert result["paper_outcome_reason"] == "incomplete_holding_horizon"
    assert result["paper_exit_ts"] is None and result["paper_exit_price"] is None
    row = position(stream, s, result)
    assert row["position_taken"] and row["position_status"] == "UNRESOLVED"
    assert row["mark_price"] == frame.loc[5, "close"]
    assert row["mark_ts"] == frame.loc[5, "open_ts"] + pd.Timedelta(minutes=5)
    assert row["unrealized_gross_pnl"] == pytest.approx(
        (frame.loc[5, "close"] - frame.loc[0, "open"]) / c.costs.price_increment)
    assert row["gross_pnl"] is None and row["net_pnl"] is None and row["total_cost"] is None
    assert row["position_exit_ts"] is None
    closed = [r for r in [row] if r["net_pnl"] is not None]
    assert closed == []
    stressed, = reprice_positions([row], "double_all_costs")
    assert stressed["net_pnl"] is None and stressed["position_status"] == "UNRESOLVED"


def test_open_position_keeps_instrument_capacity(stream):
    c, frame = frozen_contract(stream), bars(stream, n=6)
    first = make_signal(stream, event_id="a")
    second = make_signal(stream, event_id="b", known_at_ts="2026-09-14T14:12:00Z")
    tape = make_tape(stream, frame)
    outcomes = [evaluate_signal(s, tape, c) for s in (first, second)]
    rows = simulate_positions([first, second], outcomes, {c.contract_id: c}, simulator(accounting_unit=UNIT[stream]))
    assert rows[0]["position_status"] == "UNRESOLVED"
    assert not rows[1]["position_taken"] and rows[1]["no_trade_reason"] == "concurrent_position_limit"


def test_cost_stress_doubles_every_component_on_identical_trades(stream):
    stop, _ = barriers(stream)
    frame = bars(stream)
    set_bar(frame, 2, low=stop - 0.1 * ATR[stream])
    s = make_signal(stream)
    base = position(stream, s, evaluate_signal(s, make_tape(stream, frame), frozen_contract(stream)))
    assert base["cost_case"] == "base" and base["accounting_unit"] == UNIT[stream]
    stressed, = reprice_positions([base], "double_all_costs")
    for key in ("spread_cost", "slippage_cost", "commission_cost", "total_cost"):
        assert stressed[key] == pytest.approx(2 * base[key]) and base[key] > 0
    same = [k for k in base if k not in {"cost_case", "spread_cost", "slippage_cost", "commission_cost",
                                         "total_cost", "net_pnl"}]
    assert {k: stressed[k] for k in same} == {k: base[k] for k in same}
    assert stressed["net_pnl"] == pytest.approx(base["gross_pnl"] - 2 * base["total_cost"])
    with pytest.raises(ValueError, match="base ledger only"):
        reprice_positions([stressed], "double_all_costs")
    with pytest.raises(KeyError):
        reprice_positions([base], "half_costs")


def test_ledger_unit_cannot_be_the_other_streams_unit(stream):
    c, s = frozen_contract(stream), make_signal(stream)
    outcome = evaluate_signal(s, make_tape(stream, bars(stream)), c)
    other = UNIT["EURUSD" if stream == "ES" else "ES"]
    with pytest.raises(ValueError, match="Cannot combine tick and pip ledgers"):
        simulate_positions([s], [outcome], {c.contract_id: c}, simulator(accounting_unit=other))


def test_label_close_can_never_become_a_fill(stream):
    c, tape = frozen_contract(stream), make_tape(stream, bars(stream))
    a = evaluate_signal(make_signal(stream, training_label_outcome=1), tape, c)
    b = evaluate_signal(make_signal(stream, training_label_outcome=0, training_label_version="other_label_v9",
                                    training_label_hash="e" * 64), tape, c)
    fills = ("entry_ts", "entry_price", "stop_price", "target_price", "paper_exit_ts", "paper_exit_price",
             "paper_contract_outcome")
    assert {k: a[k] for k in fills} == {k: b[k] for k in fills}


def test_best_historical_entry_can_never_become_a_fill(stream):
    c, frame = frozen_contract(stream), bars(stream)
    s = make_signal(stream, known_at_ts="2026-09-14T14:22:00Z")
    plain = evaluate_signal(s, make_tape(stream, frame), c)
    better = frame.copy()
    best = PX[stream] - 5 * ATR[stream]  # a far better long price printed before the decision
    set_bar(better, 1, open=best, high=best + 0.1 * ATR[stream], low=best - 0.1 * ATR[stream], close=best)
    result = evaluate_signal(s, make_tape(stream, better), c)
    assert result["entry_price"] == plain["entry_price"] != best
    assert result["entry_ts"] == plain["entry_ts"] > s.known_at_ts
    # A missing entry bar is a gap; no better or later open is searched for.
    gone = frame.drop(index=4).reset_index(drop=True)
    missing = evaluate_signal(s, make_tape(stream, gone, schedule_times=frame["open_ts"]), c)
    assert missing["entry_ts"] is None and missing["paper_contract_outcome"] == "DATA_GAP"


def test_target_hit_price_can_never_become_a_fill(stream):
    c = frozen_contract(stream)
    _, target = barriers(stream)
    s = make_signal(stream, known_at_ts="2026-09-14T14:22:00Z")
    frame = bars(stream)
    set_bar(frame, 1, high=target + 5 * ATR[stream])  # target level traded before the decision
    result = evaluate_signal(s, make_tape(stream, frame), c)
    assert result["paper_contract_outcome"] != "TARGET"
    assert result["entry_price"] == frame.loc[4, "open"]
    hinted = evaluate_signal(s.model_copy(update=dict(target_reference=PX[stream] + 0.01 * ATR[stream],
                                                      stop_reference=PX[stream] - 50 * ATR[stream])),
                             make_tape(stream, frame), c)
    assert c.risk.rule == "atr"
    assert (hinted["stop_price"], hinted["target_price"], hinted["paper_exit_price"]) == (
        result["stop_price"], result["target_price"], result["paper_exit_price"])


@pytest.mark.parametrize("field", ["confirmation_ts", "signal_bar_close_ts", "geometry_available_at"])
def test_later_setup_can_never_become_a_fill(stream, field):
    s = make_signal(stream, **{field: "2026-09-14T14:30:00Z"})  # setup evidence dated after the decision
    result = evaluate_signal(s, make_tape(stream, bars(stream)), frozen_contract(stream))
    assert result["paper_contract_outcome"] == "INVALID" and result["entry_ts"] is None
    row = position(stream, s, result)
    assert not row["position_taken"] and row["no_trade_reason"] == "input_unavailable_at_decision"
