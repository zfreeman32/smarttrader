"""B1-outcome label research (option A, 2026-10-08). Offline research only: nothing is frozen, installed or activated.

Stages (cached under ``artifacts/b1_outcome_research_20261008``; each reuses earlier caches):

* ``inputs``   full ES / EURUSD B1 tapes (same sources, calendars and bar filtering as the 2026-10-07 sweep), completed
               structural ATR tables, and the setup census: every detector-fired setup with its declared side and its
               setup-specific b1_v3 contract (ES FRVP setup1-6 from the v2 feature tape; EURUSD OTE base events).
* ``verify``   the vectorized evaluator (``model_testing.b1_fast_outcomes``) against ``evaluate_signal`` +
               ``simulate_positions`` on real samples (both streams, every family, offsets, ledger).
* ``census``   candidate decision bars at offsets -10..+5 around each setup, each priced from its OWN decision time
               (known_at = bar close + 1 s, entry at the next permitted open). Negative offsets are diagnostics: the setup
               was not knowable there. Positive offsets are eligible only while the setup stays valid (rule below).
* ``everybar`` every scheduled bar, long and short, priced with template contracts (diagnostic for a primary model),
               plus the existing every-bar OTE models (21 long, 25 short) scored against those outcomes.
* ``report``   setup-only baseline, label audit, timing study, every-bar study and the opportunity ceiling.

Post-setup validity at bar s+k (k = 1..5), all known by that bar's close: same session as the setup bar; decision bar
present; no opposite-side setup and no newer same-side setup in (s, s+k]; no bar missing in (s, s+k]; price has not
reached the setup's own stop or target distance measured from the setup bar close with the setup's ATR.

Periods: development = before the earliest 2026-10-07 test-split start of the stream's roster models (ES 2024-10-01
11:15Z, EURUSD 2025-02-20 15:05Z). After that: "inspected" (the sweep already looked at it; never called a fresh
holdout). A fresh holdout needs forward bars (see the report).

Usage: python scripts/research_b1_outcome_labels.py {inputs,verify,census,everybar,report}
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp/b2_s3_calendar_dependencies"))

import numpy as np
import pandas as pd

from model_testing import b1_fast_outcomes as fast
from model_testing.outcome_contracts import file_hash, load_contracts, resolve_contract

OUT = ROOT / "artifacts/b1_outcome_research_20261008"
CACHE = OUT / "cache"
REPORT = ROOT / "docs/audits/b1_outcome_research_20261008.json"
ES_DATASET = ROOT / "artifacts/b2_frvp_july_prepared_v2_20261006/dataset.parquet"
OTE_LABELS = ROOT / "artifacts/b2_ote_labels_20260920_v2/labels.csv.gz"
OTE_EVENTS = ROOT / "artifacts/b2_ote_labels_20260920_v2/events.csv.gz"
FRVP_LABEL_EVENTS = ROOT / "artifacts/b2_frvp_labels_20260920_v1/frvp_es_primary_refresh_20260701/events.csv.gz"
DEV_END = {"ES": pd.Timestamp("2024-10-01 11:15", tz="UTC"), "EURUSD": pd.Timestamp("2025-02-20 15:05", tz="UTC")}
UNITS = {"ES": "ticks", "EURUSD": "pips"}
OFFSETS = tuple(range(-10, 6))
POST = 5
OTE_FAMILY = {"reversal": "reversal", "continuation_pullback": "continuation", "breakout": "breakout"}
FRVP_BROAD = {1: "reversal", 2: "continuation", 3: "continuation", 4: "reversal", 5: "continuation", 6: "reversal"}
TEMPLATES = {"ES": ("es_frvp_setup2_eval_v2", "es_frvp_setup1_eval_v1"),
             "EURUSD": ("eurusd_ote_continuation_eval_v2", "eurusd_ote_reversal_eval_v2")}
ROUTES = {2: ("ES", "long", ("setup2", "setup3", "setup5")), 3: ("ES", "long", tuple(f"setup{i}" for i in range(1, 7))),
          5: ("ES", "long", ("setup2",)), 10: ("ES", "long", ("setup1", "setup4", "setup6")),
          21: ("EURUSD", "long", ("reversal", "continuation", "breakout")), 23: ("EURUSD", "long", ("reversal", "continuation", "breakout")),
          25: ("EURUSD", "short", ("reversal", "continuation", "breakout")), 22: ("EURUSD", "long", ("reversal",)),
          26: ("EURUSD", "short", ("reversal",))}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def save(name: str, value) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / name).open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(name: str):
    with (CACHE / name).open("rb") as handle:
        return pickle.load(handle)


# ----------------------------------------------------------------------------------------------------------- inputs
def es_inputs():
    from scripts import run_section3_smoke_replay as smoke
    bars = smoke.es_source()
    start, end = pd.Timestamp("2017-01-01", tz="UTC"), min(bars.open_ts.max() + pd.Timedelta(minutes=5), pd.Timestamp("2026-12-31", tz="UTC"))
    schedule = smoke.es_schedule(start, end)
    keep = bars.open_ts.ge(start) & bars.open_ts.lt(end) & bars.open_ts.isin(schedule.open_ts)
    notes = dict(bars_off_calendar_dropped=int((bars.open_ts.ge(start) & bars.open_ts.lt(end)).sum() - keep.sum()))
    return bars.loc[keep].reset_index(drop=True), schedule, start, end, notes


def fx_inputs():
    from scripts import run_section3_smoke_replay as smoke
    labels = pd.read_csv(OTE_LABELS, usecols=["timestamp", "open", "high", "low", "close"])
    labels["open_ts"] = pd.to_datetime(labels.pop("timestamp"), utc=True)
    labels["is_complete"] = True
    start, end = pd.Timestamp("2018-12-01", tz="UTC"), min(labels.open_ts.max() + pd.Timedelta(minutes=5), pd.Timestamp("2026-12-31", tz="UTC"))
    schedule = smoke.fx_schedule(start, end)
    keep = labels.open_ts.isin(schedule.open_ts)
    notes = dict(bars_off_calendar_dropped=int((~keep).sum()), calendar_sha256=file_hash(smoke.FX_CALENDAR))
    return labels.loc[keep].reset_index(drop=True), schedule, start, end, notes, labels


def es_setups(tape: fast.FastTape) -> pd.DataFrame:
    d = pd.read_parquet(ES_DATASET, columns=["source_row_idx", "datetime", "frvp_setup_type", "frvp_setup_side"])
    d["datetime"] = pd.to_datetime(d.datetime, utc=True)
    s = d.loc[d.frvp_setup_type.gt(0) & d.frvp_setup_side.ne(0)].copy()
    s["setup_type"] = s.frvp_setup_type.astype(int)
    s["family"] = "setup" + s.setup_type.astype(str)
    s["side"] = np.where(s.frvp_setup_side > 0, 1, -1)
    contracts = {}
    for (t, side), _ in s.groupby(["setup_type", "side"]):
        contracts[(t, side)] = resolve_contract(instrument="ES", strategy="frvp", model_family=f"{FRVP_BROAD[t]}_setup{t}",
                                                setup_family=f"setup{t}", side="long" if side > 0 else "short").contract_id
    s["contract_id"] = [contracts[(t, sd)] for t, sd in zip(s.setup_type, s.side)]
    s["decision_open"] = s.datetime
    s["setup_id"] = "ES-" + s.source_row_idx.astype(str)
    # All bars' setup sides (for validity: opposite / newer same-side setups) on schedule positions.
    pos_all = tape.position(d.datetime)
    side_all = np.zeros(len(tape.sched), np.int8)
    ok = pos_all >= 0
    side_all[pos_all[ok]] = np.sign(d.frvp_setup_side.to_numpy()[ok] * (d.frvp_setup_type.to_numpy()[ok] > 0)).astype(np.int8)
    return s[["setup_id", "source_row_idx", "decision_open", "family", "side", "contract_id"]].reset_index(drop=True), \
        dict(long=side_all == 1, short=side_all == -1)


def fx_setups(tape: fast.FastTape, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    e = pd.read_csv(OTE_EVENTS, usecols=["label_family", "swing_type", "swing_index", "confirm_time", "confirm_index", "tb_outcome",
                                         "label_quality"])
    e["family"] = e.label_family.map(OTE_FAMILY)
    e["side"] = np.where(e.swing_type.eq("low"), 1, -1)
    e["decision_open"] = labels.open_ts.iloc[e.confirm_index].to_numpy()
    if (pd.to_datetime(e.confirm_time, utc=True) != e.decision_open + pd.Timedelta(minutes=5)).any():
        raise ValueError("OTE confirmation is not the close of its decision bar")
    e["contract_id"] = [resolve_contract(instrument="EURUSD", strategy="ote", model_family=f, setup_family=f,
                                         side="long" if sd > 0 else "short").contract_id for f, sd in zip(e.family, e.side)]
    e["setup_id"] = "FX-" + e.confirm_index.astype(str) + "-" + e.family + "-" + np.where(e.side > 0, "L", "S")
    e["source_row_idx"] = e.confirm_index
    pos = tape.position(e.decision_open)
    flags = dict(long=np.zeros(len(tape.sched), bool), short=np.zeros(len(tape.sched), bool))
    for side, key in ((1, "long"), (-1, "short")):
        p = pos[(e.side == side).to_numpy() & (pos >= 0)]
        flags[key][p] = True
    keep = ["setup_id", "source_row_idx", "decision_open", "family", "side", "contract_id", "label_family", "swing_index",
            "confirm_index", "tb_outcome", "label_quality"]
    return e[keep].reset_index(drop=True), flags


def stage_inputs() -> None:
    contracts = load_contracts()
    for stream in ("ES", "EURUSD"):
        if (CACHE / f"inputs_{stream}.pkl").exists():
            print(f"{stream}: inputs cached", flush=True)
            continue
        if stream == "ES":
            bars, schedule, start, end, notes = es_inputs()
        else:
            bars, schedule, start, end, notes, labels = fx_inputs()
        tape = fast.FastTape(bars, schedule, instrument=stream, coverage_start=start, coverage_end=end)
        atr = {m: fast.structural_atr(tape, m) for m in ((60,) if stream == "ES" else (60, 30))}
        setups, flags = es_setups(tape) if stream == "ES" else fx_setups(tape, labels)
        save(f"inputs_{stream}.pkl", dict(tape=tape, atr=atr, setups=setups, flags=flags, notes=notes,
                                          coverage=(start, end), contracts={k: contracts[k] for k in set(setups.contract_id) | set(TEMPLATES[stream])}))
        print(f"{stream}: {len(tape.sched)} scheduled bars, {tape.present.sum()} present, {len(setups)} setups; notes {notes}", flush=True)


# ----------------------------------------------------------------------------------------------------------- census
def price(stream: str, inputs: dict, pos: np.ndarray, side: np.ndarray, contract_ids: np.ndarray) -> pd.DataFrame:
    names = sorted(set(contract_ids))
    index = {n: i for i, n in enumerate(names)}
    return fast.evaluate(inputs["tape"], inputs["atr"], pos, side, np.array([index[c] for c in contract_ids]),
                         [inputs["contracts"][n] for n in names])


def post_validity(inputs: dict, setups: pd.DataFrame, base: pd.DataFrame) -> dict[int, np.ndarray]:
    """Causal validity of the post-setup bar s+k (see module docstring), from the setup's own k=0 references."""
    tape, flags = inputs["tape"], inputs["flags"]
    pos = tape.position(setups.decision_open)
    side = setups.side.to_numpy()
    contracts = inputs["contracts"]
    stop_atr = np.array([contracts[c].risk.stop_atr for c in setups.contract_id])
    target_atr = np.array([contracts[c].risk.target_atr for c in setups.contract_id])
    ref, atr = tape.c[np.maximum(pos, 0)], base.atr.to_numpy()
    lo_stop, hi_target = ref - side * stop_atr * atr, ref + side * target_atr * atr
    S = len(tape.sched)
    same = np.where(side > 0, 1, 0)
    valid = np.ones(len(setups), bool) & (pos >= 0) & np.isfinite(atr)
    out = {}
    for k in range(1, POST + 1):
        q = pos + k
        inside = (q < S) & (pos >= 0)
        qq = np.minimum(q, S - 1)
        valid &= inside & tape.ok[qq] & (tape.session_id[qq] == tape.session_id[np.maximum(pos, 0)])
        opposite = np.where(side > 0, flags["short"][qq], flags["long"][qq])
        newer = np.where(same == 1, flags["long"][qq], flags["short"][qq])
        valid &= ~opposite & ~newer
        h, l = tape.h[qq], tape.l[qq]
        stop_touched = np.where(side > 0, l <= lo_stop, h >= lo_stop)
        target_touched = np.where(side > 0, h >= hi_target, l <= hi_target)
        valid &= ~(stop_touched | target_touched)
        out[k] = valid.copy()
    return out


def stage_census() -> None:
    for stream in ("ES", "EURUSD"):
        path = CACHE / f"census_{stream}.parquet"
        if path.exists():
            print(f"{stream}: census cached", flush=True)
            continue
        inputs = load(f"inputs_{stream}.pkl")
        tape, setups = inputs["tape"], inputs["setups"]
        pos = tape.position(setups.decision_open)
        frames = []
        base = None
        for k in (0, *[k for k in OFFSETS if k != 0]):
            p = np.where(pos >= 0, pos + k, -1)
            p = np.where((p >= 0) & (p < len(tape.sched)), p, -1)
            priced = price(stream, inputs, p, setups.side.to_numpy(), setups.contract_id.to_numpy())
            if k == 0:
                base = priced
                validity = post_validity(inputs, setups, base)
            frame = pd.concat([setups[["setup_id", "family", "side", "contract_id", "decision_open"]].reset_index(drop=True), priced], axis=1)
            frame["offset"] = k
            frame["knowable"] = k >= 0
            frame["valid"] = (k == 0) | (validity[k] if k > 0 else False)
            frame["candidate_open_ns"] = np.where(p >= 0, tape.sched[np.maximum(p, 0)], fast.NAT)
            frames.append(frame)
            print(f"{stream} offset {k:+d}: {priced.executable.sum()} executable / {len(priced)}", flush=True)
        census = pd.concat(frames, ignore_index=True)
        census["event_id"] = census.setup_id + "@" + census.offset.astype(str)
        census["period"] = np.where(pd.to_datetime(census.decision_open, utc=True) < DEV_END[stream], "dev", "inspected")
        census.to_parquet(path, index=False)


# ---------------------------------------------------------------------------------------------------------- everybar
def stage_everybar() -> None:
    for stream in ("ES", "EURUSD"):
        path = CACHE / f"everybar_{stream}.parquet"
        if path.exists():
            print(f"{stream}: every-bar cached", flush=True)
            continue
        inputs = load(f"inputs_{stream}.pkl")
        tape = inputs["tape"]
        pos = np.flatnonzero(tape.present)
        frames = []
        for template in TEMPLATES[stream]:
            for side in (1, -1):
                priced = price(stream, inputs, pos, np.full(len(pos), side), np.full(len(pos), template))
                keep = ["outcome", "entry_ns", "exit_ns", "known_ns", "gross", "cost", "net", "resolved", "executable", "entry_spread",
                        "stop_units", "target_units", "bars_held"]
                frame = priced[keep].copy()
                frame["pos"], frame["side"], frame["template"] = pos, side, template
                frames.append(frame)
                print(f"{stream} every-bar {template} side {side}: {priced.executable.sum()} executable", flush=True)
        out = pd.concat(frames, ignore_index=True)
        out["open_ns"] = tape.sched[out.pos.to_numpy()]
        out["session_id"] = tape.session_id[out.pos.to_numpy()]
        out.to_parquet(path, index=False)


# ------------------------------------------------------------------------------------------------------------ verify
def stage_verify(samples: int = 600) -> None:
    """Field-for-field equality with the B1 engine on real windows (fresh tapes built by the engine's own classes)."""
    from model_testing.outcome_atr import completed_structural_atr, structural_atr_at
    from model_testing.outcome_contracts import PositionContract, content_hash
    from model_testing.outcome_engine import TIME_FIELDS, EvaluationTape, SignalEvidence, evaluate_signal, simulate_positions
    from model_testing.outcome_scope import load_research_scope
    from scripts import run_section3_smoke_replay as smoke
    scope = load_research_scope()
    rng = np.random.default_rng(20261008)
    results = {}
    windows = {"ES": ("2023-02-01", "2023-05-01"), "EURUSD": ("2024-01-01", "2024-03-15")}
    for stream, (a, b) in windows.items():
        inputs = load(f"inputs_{stream}.pkl")
        start, end = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        if stream == "ES":
            bars, schedule = smoke.es_source(), smoke.es_schedule(start, end)
        else:
            bars, schedule = fx_inputs()[0], smoke.fx_schedule(start, end)
        bars = bars.loc[bars.open_ts.ge(start) & bars.open_ts.lt(end) & bars.open_ts.isin(schedule.open_ts)].reset_index(drop=True)
        calendar = smoke.ES_CALENDAR_ID if stream == "ES" else smoke.fx_calendar()["calendar_id"]
        real = EvaluationTape(bars.copy(), schedule, timeframe_minutes=5, calendar_id=calendar, calendar_source="verify",
                              coverage_start=start, coverage_end=end, require_completion=True)
        atr = {m: completed_structural_atr(real, timeframe_minutes=m, instrument=stream) for m in ((60,) if stream == "ES" else (60, 30))}
        quick = fast.FastTape(bars, schedule, instrument=stream, coverage_start=start, coverage_end=end)
        quick_atr = {m: fast.structural_atr(quick, m) for m in atr}
        # ATR equality on every period.
        for m in atr:
            ref = atr[m]
            values = np.array([e.value if e is not None else np.nan for e in ref.evidence])
            closes = pd.DatetimeIndex(ref.source_close_ts).as_unit("ns").asi8
            got = dict(zip(quick_atr[m][0], quick_atr[m][1]))
            mine = np.array([got.get(c, np.nan) for c in closes])
            if not np.array_equal(np.isnan(values), np.isnan(mine)) or not np.allclose(values[~np.isnan(values)], mine[~np.isnan(mine)], rtol=0, atol=1e-12):
                raise AssertionError(f"{stream} {m}m ATR differs")
        setups = inputs["setups"]
        window = setups.loc[setups.decision_open.ge(start) & setups.decision_open.lt(end - pd.Timedelta(days=3))]
        pick = window.sample(min(samples, len(window)), random_state=1).reset_index(drop=True)
        offsets = rng.integers(-3, 4, len(pick))
        decision = pick.decision_open + pd.to_timedelta(offsets * 5, unit="min")
        pos = quick.position(decision)
        mine = price(stream, dict(inputs, tape=quick, atr=quick_atr), pos, pick.side.to_numpy(), pick.contract_id.to_numpy())
        signals, outcomes = [], []
        for i, row in pick.iterrows():
            ts = decision.iloc[i]
            if pos[i] < 0 or ts not in real.bars.index:
                assert mine.outcome.iloc[i] == "NO_SIGNAL"
                continue
            contract = inputs["contracts"][row.contract_id]
            close = ts + pd.Timedelta(minutes=5)
            known = close + pd.Timedelta(seconds=1)
            coordinate = str(real.bars.coordinate_id[ts]) if stream == "ES" else None
            minutes = 60 if contract.risk.atr_definition == "completed_1h_structural_atr" else 30
            proof = structural_atr_at(atr[minutes], known, coordinate_id=coordinate)
            clocks = {f: known.to_pydatetime() for f in TIME_FIELDS}
            clocks.update(event_anchor_ts=ts.to_pydatetime(), signal_bar_close_ts=close.to_pydatetime(), confirmation_ts=close.to_pydatetime(),
                          feature_max_source_close_ts=close.to_pydatetime(),
                          atr_available_at=known.to_pydatetime() if proof is None else proof.available_at_ts)
            family = row.family
            model_family = f"{FRVP_BROAD[int(family[5:])]}_{family}" if stream == "ES" else family
            s = SignalEvidence(event_id=f"v-{i:05d}", model_id="b1_outcome_research_verify", model_version="v1",
                               model_family=model_family, strategy="frvp" if stream == "ES" else "ote", setup_family=family,
                               instrument=stream, side="long" if row.side > 0 else "short", **clocks, availability_status="verified",
                               availability_evidence_refs=("verify",), atr=None if proof is None else proof.value,
                               atr_definition=contract.risk.atr_definition, atr_provenance=proof, coordinate_id=coordinate,
                               policy_eligible=True, policy_rejection_reason="", quantity=1.0)
            o = evaluate_signal(s, real, contract, scope=scope)
            signals.append(s)
            outcomes.append(o)
            m = mine.iloc[i]
            expected = o["paper_contract_outcome"]
            if expected == "DATA_GAP" and o["entry_ts"] is None:
                expected = "DATA_GAP_ENTRY"
            if expected != m.outcome:
                raise AssertionError(f"{stream} sample {i}: engine {expected} vs fast {m.outcome}")
            if o["paper_resolved"]:
                assert pd.Timestamp(o["paper_exit_ts"]).value == m.exit_ns and np.isclose(o["paper_exit_price"], m.exit_price, rtol=0, atol=1e-12)
                assert np.isclose(o["stop_price"], m.stop_price, rtol=0, atol=1e-12) and np.isclose(o["target_price"], m.target_price, rtol=0, atol=1e-12)
        contracts = {c.contract_id: c for c in inputs["contracts"].values()}
        simulator = PositionContract(simulator_id="b1_outcome_research_verify_v1", version=1, policy_id="verify",
                                     policy_hash=content_hash({"verify": 1}), max_positions=1, max_quantity=1.0, concurrency_scope="instrument",
                                     allow_opposite=False, cooldown_minutes=0, starting_equity=100000.0, accounting_unit=UNITS[stream])
        positions = simulate_positions(signals, outcomes, contracts, simulator, scope=scope)
        by_id = {p["event_id"]: p for p in positions}
        frame = mine.copy()
        frame["event_id"] = [f"v-{i:05d}" for i in range(len(frame))]
        taken = fast.ledger(frame, restart=None)
        mismatched = 0
        for i, flag in enumerate(taken):
            p = by_id.get(f"v-{i:05d}")
            if p is None:
                continue
            if bool(p["position_taken"]) != bool(flag):
                mismatched += 1
            elif p["position_taken"] and p["net_pnl"] is not None:
                assert np.isclose(p["net_pnl"], frame.net.iloc[i], rtol=0, atol=1e-9), (stream, i)
                assert np.isclose(p["total_cost"], frame.cost.iloc[i], rtol=0, atol=1e-9)
        if mismatched:
            raise AssertionError(f"{stream}: {mismatched} ledger admissions differ")
        results[stream] = dict(window=[a, b], samples=len(pick), engine_signals=len(signals),
                               outcomes=pd.Series([o["paper_contract_outcome"] for o in outcomes]).value_counts().to_dict(),
                               positions_taken=int(sum(p["position_taken"] for p in positions)), atr_periods_equal=True,
                               outcomes_equal=True, ledger_equal=True)
        print(f"{stream} VERIFY PASS: {results[stream]}", flush=True)
    dump(OUT / "verify.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("inputs", "verify", "census", "everybar", "report"))
    args = parser.parse_args()
    if args.stage == "report":
        from scripts.report_b1_outcome_research import main as report_main
        report_main()
    else:
        dict(inputs=stage_inputs, verify=stage_verify, census=stage_census, everybar=stage_everybar)[args.stage]()
