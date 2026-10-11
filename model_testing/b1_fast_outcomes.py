"""Vectorized B1 outcomes for research censuses. The b1_v3 contracts and the B1 engine are unchanged.

Reproduces ``outcome_engine.evaluate_signal`` + ``simulate_positions`` (one position, ``fixed_barriers``) for the
configuration of the 2026-10-07 threshold sweep, so hundreds of thousands of candidate decision bars can be priced:

* 5m tape, no 1m bars (collisions resolve adverse-first), ``atr`` risk rule, completed structural ATR (30m/1h,
  Wilder 14) as of ``known_at``; ES barriers rounded outward to the tick; ES path censored on a coordinate change.
* ``known_at`` = decision bar close + 1 s (the smoke/sweep offline timing), so the entry is the first scheduled open
  strictly after it: the open of the second bar after the decision bar.
* Costs: one full session spread per leg + fixed slippage + commission (``position_costs``), base cost case.

``scripts/research_b1_outcome_labels.py verify`` checks field-for-field equality with the engine on real samples;
``tests/test_b1_fast_outcomes.py`` does the same on synthetic tapes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model_testing.outcome_contracts import OutcomeContract

FIVE_MIN = np.int64(300_000_000_000)
ONE_SEC = np.int64(1_000_000_000)
NAT = np.iinfo(np.int64).min
OUTCOMES = ("NO_SIGNAL", "NO_ENTRY", "DATA_GAP_ENTRY", "INVALID", "DATA_GAP", "TARGET", "STOP",
            "TIMEOUT_WIN", "TIMEOUT_LOSS", "TIMEOUT_FLAT")
CODE = {name: i for i, name in enumerate(OUTCOMES)}
RESOLVED_CODES = (CODE["TARGET"], CODE["STOP"], CODE["TIMEOUT_WIN"], CODE["TIMEOUT_LOSS"], CODE["TIMEOUT_FLAT"])
SESSIONS = ("overlap", "london", "new_york", "asia", "off_hours")


def session_codes(ns: np.ndarray) -> np.ndarray:
    """``outcome_engine.session_at`` for an int64 UTC ns array (index into SESSIONS)."""
    idx = pd.DatetimeIndex(ns.astype("datetime64[ns]")).tz_localize("UTC")
    london, ny = idx.tz_convert("Europe/London").hour, idx.tz_convert("America/New_York").hour
    in_london, in_ny = (london >= 7) & (london < 16), (ny >= 8) & (ny < 17)
    out = np.where((ny >= 19) | (ny < 2), 3, 4)
    out = np.where(in_ny, 2, out)
    out = np.where(in_london, 1, out)
    return np.where(in_london & in_ny, 0, out).astype(np.int8)


class FastTape:
    """Array view of an ``EvaluationTape`` input: expected-open schedule plus unfilled OHLC on it."""

    def __init__(self, bars: pd.DataFrame, schedule: pd.DataFrame, *, instrument: str, coverage_start, coverage_end):
        self.instrument = instrument
        sched = pd.DatetimeIndex(pd.to_datetime(schedule.open_ts, utc=True))
        if sched.has_duplicates or not sched.is_monotonic_increasing:
            raise ValueError("Schedule must be sorted and unique")
        self.sched = sched.as_unit("ns").asi8
        self.session_id = schedule.session_id.to_numpy()
        self.coverage_start = pd.Timestamp(coverage_start).as_unit("ns").value
        self.coverage_end = pd.Timestamp(coverage_end).as_unit("ns").value
        opens = pd.DatetimeIndex(pd.to_datetime(bars.open_ts, utc=True)).as_unit("ns").asi8
        pos = np.searchsorted(self.sched, opens)
        if (pos >= len(self.sched)).any() or (self.sched[np.minimum(pos, len(self.sched) - 1)] != opens).any():
            raise ValueError("OHLC contains bars outside the expected calendar")
        n = len(self.sched)
        self.present = np.zeros(n, bool)
        self.present[pos] = True
        self.o, self.h, self.l, self.c = (np.full(n, np.nan) for _ in range(4))
        for name, arr in (("open", self.o), ("high", self.h), ("low", self.l), ("close", self.c)):
            arr[pos] = pd.to_numeric(bars[name], errors="coerce").to_numpy(float)
        finite = np.isfinite(self.o) & np.isfinite(self.h) & np.isfinite(self.l) & np.isfinite(self.c)
        with np.errstate(invalid="ignore"):
            shape = (self.l <= np.minimum(self.o, self.c)) & (np.minimum(self.o, self.c) <= np.maximum(self.o, self.c)) \
                & (np.maximum(self.o, self.c) <= self.h)
        complete = np.zeros(n, bool)
        complete[pos] = bars["is_complete"].eq(True).to_numpy() if "is_complete" in bars else False
        self.ok = self.present & finite & shape & complete
        self.coord = np.full(n, -1, np.int64)
        self.has_coord = "coordinate_id" in bars
        if self.has_coord:
            codes, self.coord_names = pd.factorize(bars.coordinate_id.astype(object).where(bars.coordinate_id.notna()))
            self.coord[pos] = codes
        if instrument in ("ES", "6E") and not self.has_coord:
            raise ValueError("Futures tapes need coordinate IDs")

    def position(self, open_ts) -> np.ndarray:
        ns = pd.DatetimeIndex(pd.to_datetime(open_ts, utc=True)).as_unit("ns").asi8
        pos = np.searchsorted(self.sched, ns)
        bad = (pos >= len(self.sched)) | (self.sched[np.minimum(pos, len(self.sched) - 1)] != ns)
        return np.where(bad, -1, pos)


def structural_atr(tape: FastTape, minutes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``outcome_atr.completed_structural_atr`` values: (source_close_ns, value or NaN, coordinate code)."""
    period = np.int64(minutes * 60 * 1_000_000_000)
    slots = minutes // 5
    starts = tape.sched - tape.sched % period
    unique, first, counts = np.unique(starts, return_index=True, return_counts=True)
    closes, values, coords = unique + period, np.full(len(unique), np.nan), np.full(len(unique), -1, np.int64)
    ok_cum = np.r_[0, np.cumsum(tape.ok)]
    state = prev_close = coordinate = None
    history = 0
    for k, (a, n) in enumerate(zip(first, counts)):
        b = a + n
        good = n == slots and tape.sched[a] == unique[k] and ok_cum[b] - ok_cum[a] == n
        if good and tape.instrument != "EURUSD" or (good and tape.has_coord):
            cc = tape.coord[a:b]
            good = bool((cc >= 0).all() and (cc == cc[0]).all())
        if not good:
            state = prev_close = coordinate = None
            history = 0
            continue
        current = int(tape.coord[a]) if tape.has_coord else -1
        if current != coordinate:
            state = prev_close = None
            history = 0
            coordinate = current
        high, low, close = float(tape.h[a:b].max()), float(tape.l[a:b].min()), float(tape.c[b - 1])
        tr = high - low if prev_close is None else max(high - low, abs(high - prev_close), abs(low - prev_close))
        state = tr if state is None else state + (tr - state) / 14
        prev_close = close
        history += 1
        if history >= 14 and np.isfinite(state) and state > 0:
            values[k], coords[k] = state, current
    return closes, values, coords


def atr_at(table, known_ns: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """``structural_atr_at``: the latest published period at or before known_at, same coordinate, else NaN."""
    closes, values, coords = table
    idx = np.searchsorted(closes, known_ns, side="right") - 1
    safe = np.maximum(idx, 0)
    value = np.where(idx >= 0, values[safe], np.nan)
    return np.where(coords[safe] == coordinate, value, np.nan)


def contract_arrays(contracts: list[OutcomeContract]) -> dict:
    risk = [c.risk for c in contracts]
    if any(r.rule != "atr" for r in risk):
        raise ValueError("Only the frozen atr risk rule is reproduced")
    if any(c.overnight != "hold" for c in contracts):
        raise ValueError("Session-boundary exits are not reproduced")
    return dict(stop_atr=np.array([r.stop_atr for r in risk], float), target_atr=np.array([r.target_atr for r in risk], float),
                timeout=np.array([r.timeout_bars for r in risk], np.int64),
                minutes=np.array([60 if r.atr_definition == "completed_1h_structural_atr" else 30 for r in risk], np.int64))


def evaluate(tape: FastTape, atr_tables: dict[int, tuple], decision_pos: np.ndarray, side: np.ndarray,
             contract_index: np.ndarray, contracts: list[OutcomeContract]) -> pd.DataFrame:
    """One B1 paper outcome per candidate. side: +1 long / -1 short. decision_pos: schedule position (-1 = absent)."""
    n, S = len(decision_pos), len(tape.sched)
    p = contract_arrays(contracts)
    costs = contracts[0].costs
    if any(c.costs != costs for c in contracts):
        raise ValueError("One cost contract per evaluation call")
    inc = costs.price_increment
    stop_atr, target_atr = p["stop_atr"][contract_index], p["target_atr"][contract_index]
    timeout, minutes = p["timeout"][contract_index], p["minutes"][contract_index]
    code = np.full(n, CODE["NO_SIGNAL"], np.int8)
    d = np.asarray(decision_pos, np.int64)
    signal = (d >= 0) & tape.present[np.maximum(d, 0)]
    coord = np.where(signal, tape.coord[np.maximum(d, 0)], -1)
    known = np.where(signal, tape.sched[np.maximum(d, 0)] + FIVE_MIN + ONE_SEC, NAT)
    covered = signal & (known >= tape.coverage_start) & (known < tape.coverage_end)
    e = np.searchsorted(tape.sched, known, side="right")
    has_entry = covered & (e < S)
    code[signal] = CODE["NO_ENTRY"]
    es = np.minimum(e, S - 1)
    coord_ok = (tape.coord[es] == coord) if tape.instrument in ("ES", "6E") else np.ones(n, bool)
    entry_ok = has_entry & tape.ok[es] & coord_ok
    code[has_entry & ~entry_ok] = CODE["DATA_GAP_ENTRY"]
    atr = np.full(n, np.nan)
    for m in np.unique(minutes):
        sel = entry_ok & (minutes == m)
        atr[sel] = atr_at(atr_tables[int(m)], known[sel], coord[sel] if tape.instrument in ("ES", "6E") else np.full(sel.sum(), -1))
    valid = entry_ok & np.isfinite(atr) & (atr > 0)
    code[entry_ok & ~valid] = CODE["INVALID"]
    entry = np.where(entry_ok, tape.o[es], np.nan)
    sign = side.astype(float)
    dist, tdist = stop_atr * atr, target_atr * atr
    stop, target = entry - sign * dist, entry + sign * tdist
    if tape.instrument in ("ES", "6E"):
        down = lambda x: np.floor(x / inc + 1e-10) * inc
        up = lambda x: np.ceil(x / inc - 1e-10) * inc
        stop, target = np.where(sign > 0, down(stop), up(stop)), np.where(sign > 0, up(target), down(target))
    exit_price, exit_ns = np.full(n, np.nan), np.full(n, NAT, np.int64)
    mfe, mae = np.zeros(n), np.zeros(n)
    bars_held = np.zeros(n, np.int64)
    active = valid.copy()
    for j in range(int(timeout[valid].max()) if valid.any() else 0):
        idx = np.flatnonzero(active & (j < timeout))
        if not len(idx):
            break
        i = e[idx] + j
        beyond = i >= S
        ii = np.minimum(i, S - 1)
        err = beyond | ~tape.ok[ii]
        if tape.instrument in ("ES", "6E"):
            err |= tape.coord[ii] != coord[idx]
        gap = idx[err]
        code[gap] = CODE["DATA_GAP"]
        active[gap] = False
        idx, ii = idx[~err], ii[~err]
        sg, o, h, l, c = sign[idx], tape.o[ii], tape.h[ii], tape.l[ii], tape.c[ii]
        st, tg, en = stop[idx], target[idx], entry[idx]
        ts = tape.sched[ii]
        mfe[idx] = np.maximum(mfe[idx], sg * (np.where(sg > 0, h, l) - en))
        mae[idx] = np.minimum(mae[idx], sg * (np.where(sg > 0, l, h) - en))
        stop_gap = sg * (o - st) <= 0
        target_gap = ~stop_gap & (sg * (o - tg) >= 0)
        stop_hit = ~stop_gap & ~target_gap & np.where(sg > 0, l <= st, h >= st)
        target_hit = ~stop_gap & ~target_gap & ~stop_hit & np.where(sg > 0, h >= tg, l <= tg)
        last = ~(stop_gap | target_gap | stop_hit | target_hit) & (j == timeout[idx] - 1)
        for mask, name, price, at_open in ((stop_gap, "STOP", o, True), (target_gap, "TARGET", tg, True),
                                            (stop_hit, "STOP", st, False), (target_hit, "TARGET", tg, False)):
            k = idx[mask]
            code[k] = CODE[name]
            exit_price[k] = price[mask]
            exit_ns[k] = ts[mask] if at_open else ts[mask] + FIVE_MIN
        k = idx[last]
        move = sg[last] * (c[last] - en[last])
        code[k] = np.where(move > 0, CODE["TIMEOUT_WIN"], np.where(move < 0, CODE["TIMEOUT_LOSS"], CODE["TIMEOUT_FLAT"]))
        exit_price[k], exit_ns[k] = c[last], ts[last] + FIVE_MIN
        done = idx[stop_gap | target_gap | stop_hit | target_hit | last]
        bars_held[done] = j + 1
        active[done] = False
    resolved = np.isin(code, RESOLVED_CODES)
    entry_ns = np.where(entry_ok, tape.sched[es], NAT)
    spreads = np.array([dict(costs.session_spreads)[s] for s in SESSIONS])
    gross, total = np.full(n, np.nan), np.full(n, np.nan)
    if resolved.any():
        spread = spreads[session_codes(entry_ns[resolved])] + spreads[session_codes(exit_ns[resolved])]
        slippage = costs.fixed_slippage_round_trip + costs.spread_slippage_multiplier * spread
        total[resolved] = spread + slippage + costs.commission_round_trip
        gross[resolved] = sign[resolved] * (exit_price[resolved] - entry[resolved]) / inc * 1.0
    # Session cost known at decision time (entry leg) and the minimum round trip: causal cost features.
    entry_spread = np.full(n, np.nan)
    entry_spread[entry_ok] = spreads[session_codes(entry_ns[entry_ok])]
    return pd.DataFrame(dict(outcome=np.array(OUTCOMES)[code], known_ns=known, entry_ns=entry_ns, exit_ns=exit_ns,
        entry_price=entry, stop_price=np.where(valid, stop, np.nan), target_price=np.where(valid, target, np.nan),
        exit_price=exit_price, atr=np.where(entry_ok, atr, np.nan), stop_units=np.where(valid, dist / inc, np.nan),
        target_units=np.where(valid, tdist / inc, np.nan), gross=gross, cost=total, net=gross - total,
        bars_held=bars_held, mfe_units=np.where(valid, mfe / inc, np.nan), mae_units=np.where(valid, mae / inc, np.nan),
        entry_spread=entry_spread, resolved=resolved,
        executable=valid))


def ledger(frame: pd.DataFrame, *, order_key: str = "event_id", restart: str | None = "week") -> np.ndarray:
    """One position at a time (B1 ``simulate_positions``, max_positions=1). Returns a taken mask aligned to frame.

    Candidates without an executable entry (no signal, no entry, entry gap, INVALID) never take capacity; a DATA_GAP
    position stays open (exit unknown) until the ledger restarts. restart='week' mirrors the 2026-10-07 sweep
    (ISO week of known_at); None runs one ledger over the whole frame.
    """
    taken = np.zeros(len(frame), bool)
    executable = frame.executable.to_numpy()
    if not executable.any():
        return taken
    entry = frame.entry_ns.to_numpy()
    exit_ = np.where(frame.resolved.to_numpy(), frame.exit_ns.to_numpy(), np.iinfo(np.int64).max)
    sort_ts = np.where(entry != NAT, entry, frame.known_ns.to_numpy())
    week = (pd.DatetimeIndex(frame.known_ns.to_numpy().astype("datetime64[ns]")).tz_localize("UTC").strftime("%G-%V").to_numpy()
            if restart == "week" else np.zeros(len(frame), int))
    order = np.lexsort((frame[order_key].to_numpy(), sort_ts))
    busy, current = {}, None
    for i in order:
        if not executable[i]:
            continue
        key = week[i]
        if busy.get(key, NAT) <= entry[i]:
            taken[i] = True
            busy[key] = exit_[i]
    return taken
