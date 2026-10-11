"""Operating-threshold sweep for the feature contract v2 retrained models (B3 research; nothing is frozen or installed).

Goal (user, 2026-10-07): per model, the threshold with the MOST trades per week that keeps returns positive, with a
floor of 6 trades a week.

* Opportunities: only bars where the detector fires a same-side setup on the model's declared route (route bindings
  v2), exactly as in the app and the section 3 smoke replay (same matching code paths). The model score gates them.
* Returns: the B1 engine (``evaluate_signal`` on the b1_v3 contracts, full base costs, causal entry, data-gap
  censoring) under the 2026-09-29 research scope (pooled models trade only the detector-selected same-side setup), then
  B1 ``simulate_positions`` with one position at a time per model. Each outcome is computed once; every threshold
  reruns the B1 ledger on the signals that pass. Ledgers restart each calendar week, so an unresolved (data-gap) trade
  holds capacity for the rest of its week instead of the rest of the sample.
* Selection never sees the evaluation rows: thresholds are picked on out-of-fold development scores and evaluated once
  on the held-out test split. IDs 22/26: their training rows exclude the reversal confirmation bars, so they have no
  out-of-fold scores on setups; the final model scores the test-period confirmation bars through the production
  runtime, the first half (by time) selects and the second half evaluates.
* Feasible threshold: trades/week >= 6, net P&L > 0, profit factor > 1, and net P&L positive in at least half of the
  calendar years (quarters for the 22/26 half-split) of the selection period. Pick: most trades/week, then net P&L.

Usage: python scripts/sweep_b2_v2_thresholds.py [--ids 2 3 ...] [--output-root artifacts/b2_v2_threshold_sweep_20261007]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp/b2_s3_calendar_dependencies"))

import numpy as np
import pandas as pd

from model_testing.outcome_atr import completed_structural_atr
from model_testing.outcome_contracts import PositionContract, content_hash, file_hash
from model_testing.outcome_engine import EvaluationTape, SignalEvidence, evaluate_signal, simulate_positions
from model_testing.outcome_routes import load_route_bindings, resolve_model_setup_route
from model_testing.outcome_scope import load_research_scope
from scripts import run_section3_smoke_replay as smoke
from scripts.review_b2_v2_models import runtime

ROSTER = ROOT / "docs/audits/b2_v2_model_data_roster_20261006.csv"
MODELS = ROOT / "models/b2_v2_retrained_20261006"
HISTORY = ROOT / "artifacts/b2_v2_source_history_20261006"
OTE_EVENTS = ROOT / "artifacts/b2_ote_labels_20260920_v2/events.csv.gz"
DEFAULT_OUTPUT = ROOT / "artifacts/b2_v2_threshold_sweep_20261007"
IDS = (2, 3, 5, 10, 21, 23, 25, 22, 26)
HALF_SPLIT_IDS = (22, 26)
MIN_TRADES_PER_WEEK = 6.0
UNITS = {"ES": "ticks", "EURUSD": "pips"}
TAPE_NOTES: dict = {}
smoke.COLLECTION_ID = "b2_v2_threshold_sweep_20261007"  # audit label carried by the reused signal builders


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def model_entry(model_id: int, bindings) -> dict:
    row = next(r for r in csv.DictReader(ROSTER.open(encoding="utf-8")) if int(r["model_id"]) == model_id)
    binding = next(b for b in bindings.bindings if b.queue_id == model_id)
    model_dir = Path(glob.glob(str(MODELS / f"{model_id:02d}_*" / row["target"]))[0])
    review = json.loads((ROOT / f"docs/audits/b2_v2_review_20261007/{model_id}.json").read_text())
    if review["status"] != "v2_artifact_review_passed":
        raise ValueError(f"Model {model_id} has not passed review")
    prepared = HISTORY / str(model_id) / "prepared" / row["target"]
    return dict(queue_id=model_id, model_id=binding.model_id, stream=binding.instrument, binding=binding, target=row["target"],
                threshold=0.0, model_dir=model_dir, prepared=prepared, version_hash=review["training_manifest_sha256"],
                freeze=dict(artifacts=dict(sha256=dict(test_predictions=file_hash(model_dir / "test_predictions.csv")))),
                summary=json.loads((model_dir / "training_summary.json").read_text()))


def history_times(model) -> pd.DataFrame:
    history = pd.read_parquet(model["prepared"] / "source_history.parquet", columns=["source_row_idx", "datetime"])
    history["datetime"] = pd.to_datetime(history.datetime, utc=True)
    return history


def es_rows(model, endpoints: np.ndarray) -> pd.DataFrame:
    manifest = json.loads((HISTORY / str(model["queue_id"]) / "manifest.json").read_text())
    columns = ["source_row_idx", "datetime", "atr_14", "frvp_setup_type", "frvp_setup_side"]
    rows = pd.read_parquet(ROOT / manifest["source_dataset"], columns=columns).set_index("source_row_idx", drop=False).loc[endpoints]
    rows = rows.reset_index(drop=True)
    rows["datetime"] = pd.to_datetime(rows.datetime, utc=True)
    side = 1 if model["binding"].side == "long" else -1
    matched = rows.frvp_setup_side.eq(side) & rows.frvp_setup_type.gt(0)
    rows["setup_family"] = ("setup" + rows.frvp_setup_type.astype("Int64").astype(str)).where(matched)
    declared = rows.setup_family.isin(list(model["binding"].setup_bindings))
    rows["match_state"] = np.where(rows.setup_family.isna(), "no_same_side_setup",
                                   np.where(declared, "selected_same_side_setup", "undeclared_setup_family"))
    rows["detector_version"] = "b2_prepared_frvp_detector_columns"
    rows["stop_reference"] = np.nan
    rows["target_reference"] = np.nan
    return rows


def ote_events(model, history: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_csv(OTE_EVENTS, usecols=["label_family", "swing_type", "confirm_time", "confirm_index"])
    events["family"] = events.label_family.map(smoke.OTE_EVENT_FAMILIES)
    events["side"] = np.where(events.swing_type.eq("low"), "long", "short")
    if events.family.isna().any() or events.duplicated(["confirm_index", "side", "family"]).any():
        raise ValueError("Unknown or duplicated OTE base event")
    closes = history.datetime + pd.Timedelta(minutes=5)
    if (pd.to_datetime(events.confirm_time, utc=True).to_numpy() != closes.iloc[events.confirm_index].to_numpy()).any():
        raise ValueError("OTE event confirmation is not the close of its decision bar (source-row alignment)")
    return events


def fx_rows(model, endpoints: np.ndarray, history: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    side = model["binding"].side
    grouped = events.sort_values("family").groupby(["confirm_index", "side"]).family.agg("|".join).unstack("side")
    rows = history.set_index("source_row_idx", drop=False).loc[endpoints].reset_index(drop=True)
    other = "short" if side == "long" else "long"
    rows["same_side_families"] = grouped[side].reindex(rows.source_row_idx).to_numpy()
    rows["other_side_families"] = grouped[other].reindex(rows.source_row_idx).to_numpy()
    single = rows.same_side_families.notna() & ~rows.same_side_families.fillna("").str.contains("|", regex=False)
    rows["setup_family"] = rows.same_side_families.where(single)
    declared = rows.setup_family.isin(list(model["binding"].setup_bindings))
    rows["match_state"] = np.where(rows.same_side_families.isna(), "no_same_side_setup",
        np.where(~single, "ambiguous_multiple_same_side_setups",
                 np.where(declared, "selected_same_side_setup", "undeclared_setup_family")))
    rows["detector_version"] = "b2_corrected_ote_base_events"
    rows["saved_prediction"] = True
    for column in ("same_side_families", "other_side_families", "setup_family"):
        rows[column] = rows[column].astype(object).where(rows[column].notna(), None)
    return rows


def keep_setups(rows: pd.DataFrame, stream: str) -> pd.DataFrame:
    if stream == "ES":
        return rows.loc[rows.match_state.eq("selected_same_side_setup")].reset_index(drop=True)
    return rows.loc[rows.same_side_families.notna()].reset_index(drop=True)  # ambiguous rows stay for the census


def attach_scores(rows: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = rows.merge(scores, on="source_row_idx", how="inner", validate="one_to_one")
    rows["threshold"] = 0.0
    rows["threshold_passed"] = True
    return rows


def saved_scores(model, name: str, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(model["model_dir"] / name)
    raw = frame[f"{prefix}raw_probability"]
    calibrated = frame[f"{prefix}calibrated_probability"] if f"{prefix}calibrated_probability" in frame else raw
    out = pd.DataFrame(dict(source_row_idx=frame.source_row_idx.astype(np.int64), raw_score=raw.astype(float),
                            calibrated_probability=calibrated.astype(float)))
    return out.loc[out.calibrated_probability.notna()].reset_index(drop=True)


def runtime_scores(model, endpoints: np.ndarray) -> pd.DataFrame:
    """Final model through the production loader/runner on raw source history; checked on saved test bars first."""
    summary = model["summary"]
    runner = runtime(model["model_dir"], summary, model["stream"])
    columns = ["source_row_idx", "datetime", *summary["selected_feature_names"]]
    history = pd.read_parquet(model["prepared"] / "source_history.parquet", columns=columns)
    context = summary["context_rows"]
    saved = pd.read_csv(model["model_dir"] / "test_predictions.csv")
    probe = saved.iloc[np.linspace(0, len(saved) - 1, 50).astype(int)]
    check = [runner.predict_latest(history.iloc[int(e) - context:int(e) + 1]) for e in probe.source_row_idx]
    np.testing.assert_allclose([p.calibrated_probability for p in check], probe.calibrated_probability, rtol=1e-5, atol=1e-6)
    predictions = [runner.predict_latest(history.iloc[int(e) - context:int(e) + 1]) for e in endpoints]
    return pd.DataFrame(dict(source_row_idx=endpoints.astype(np.int64), raw_score=[float(p.raw_score) for p in predictions],
                             calibrated_probability=[float(p.calibrated_probability) for p in predictions]))


def build_tape(stream: str, start: pd.Timestamp, end: pd.Timestamp, model=None):
    if stream == "ES":
        bars = smoke.es_source()
        end = min(bars.open_ts.max() + pd.Timedelta(minutes=5), end)
        schedule = smoke.es_schedule(start, end)
        window = bars.open_ts.ge(start) & bars.open_ts.lt(end)
        # As for EURUSD: bars off the pinned calendar are dropped (a setup on one becomes decision_bar_missing_from_tape).
        TAPE_NOTES["ES_bars_off_calendar_dropped"] = int((window & ~bars.open_ts.isin(schedule.open_ts)).sum())
        tape = EvaluationTape(bars.loc[window & bars.open_ts.isin(schedule.open_ts)].copy(), schedule,
            timeframe_minutes=5, calendar_id=smoke.ES_CALENDAR_ID,
            calendar_source="pandas_market_calendars 5.1.3 CME Globex Equity; B1 review frozen calendar",
            coverage_start=start, coverage_end=end, require_completion=True)
        return tape, completed_structural_atr(tape, instrument="ES")
    labels = pd.read_csv(ROOT / "artifacts/b2_ote_labels_20260920_v2/labels.csv.gz", usecols=["timestamp", "open", "high", "low", "close"])
    labels["open_ts"] = pd.to_datetime(labels.pop("timestamp"), utc=True)
    labels["is_complete"] = True
    end = min(labels.open_ts.max() + pd.Timedelta(minutes=5), end)
    schedule = smoke.fx_schedule(start, end)
    window = labels.open_ts.ge(start) & labels.open_ts.lt(end)
    TAPE_NOTES["EURUSD_bars_off_calendar_dropped"] = int((window & ~labels.open_ts.isin(schedule.open_ts)).sum())
    bars = labels.loc[labels.open_ts.ge(start) & labels.open_ts.lt(end) & labels.open_ts.isin(schedule.open_ts)].copy()
    tape = EvaluationTape(bars, schedule, timeframe_minutes=5, calendar_id=smoke.fx_calendar()["calendar_id"],
        calendar_source=f"configs/{smoke.FX_CALENDAR.name} sha256 {file_hash(smoke.FX_CALENDAR)}",
        coverage_start=start, coverage_end=end, require_completion=True)
    return tape, {minutes: completed_structural_atr(tape, timeframe_minutes=minutes, instrument="EURUSD") for minutes in (60, 30)}


def periods(model, history: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Scored rows per period with setup context. Selection and evaluation rows never overlap."""
    stream = model["stream"]
    if model["queue_id"] in HALF_SPLIT_IDS:
        events = ote_events(model, history)
        test_ids = pd.read_csv(model["model_dir"] / "test_predictions.csv", usecols=["source_row_idx"]).source_row_idx
        own = events.loc[events.side.eq(model["binding"].side) & events.family.isin(list(model["binding"].setup_bindings))
                         & events.confirm_index.between(test_ids.min(), test_ids.max())].confirm_index
        endpoints = np.array(sorted(set(own)), dtype=np.int64)
        rows = keep_setups(fx_rows(model, endpoints, history, events), stream)
        rows = attach_scores(rows, runtime_scores(model, rows.source_row_idx.to_numpy()))
        middle = rows.datetime.min() + (rows.datetime.max() - rows.datetime.min()) / 2
        return dict(select=rows.loc[rows.datetime < middle].reset_index(drop=True),
                    evaluate=rows.loc[rows.datetime >= middle].reset_index(drop=True))
    out = {}
    for period, name, prefix in (("select", "oof_predictions.csv", "oof_"), ("evaluate", "test_predictions.csv", "")):
        scores = saved_scores(model, name, prefix)
        endpoints = scores.source_row_idx.to_numpy()
        if stream == "ES":
            rows = keep_setups(es_rows(model, endpoints), stream)
        else:
            rows = keep_setups(fx_rows(model, endpoints, history, ote_events(model, history)), stream)
        out[period] = attach_scores(rows, scores)
    return out


def outcomes_for(model, rows: pd.DataFrame, tape, atr, scope, bindings) -> tuple[list[SignalEvidence], list[dict]]:
    signals, _ = smoke.signals_of(model, rows, tape, atr, content_hash({"sweep": smoke.COLLECTION_ID}))
    outcomes = []
    for signal in signals:
        contract = resolve_model_setup_route(model_id=model["model_id"], setup_family=signal.setup_family,
                                             instrument=signal.instrument, side=signal.side, manifest=bindings)
        outcomes.append(evaluate_signal(signal, tape, contract, scope=scope))
    return signals, outcomes


def ledger(signals, outcomes, contracts, simulator, scope) -> list[dict]:
    """B1 position simulation, restarted each calendar week (an unresolved trade blocks only its own week)."""
    weeks: dict[str, list[int]] = {}
    for index, signal in enumerate(signals):
        weeks.setdefault(pd.Timestamp(signal.known_at_ts).strftime("%G-%V"), []).append(index)
    positions = []
    for indices in weeks.values():
        positions.extend(simulate_positions([signals[i] for i in indices], [outcomes[i] for i in indices],
                                            contracts, simulator, scope=scope))
    return positions


def metrics(positions, signals_by_id, weeks: float, bucket: str) -> dict:
    taken = [p for p in positions if p["position_taken"]]
    closed = sorted((p for p in taken if p["position_status"] == "CLOSED"), key=lambda p: pd.Timestamp(signals_by_id[p["event_id"]].known_at_ts))
    pnl = np.array([p["net_pnl"] for p in closed], dtype=float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    curve = np.r_[0.0, np.cumsum(pnl)]
    by_bucket: dict[str, float] = {}
    for p, value in zip(closed, pnl):
        stamp = pd.Timestamp(signals_by_id[p["event_id"]].known_at_ts)
        key = str(stamp.year) if bucket == "year" else f"{stamp.year}Q{stamp.quarter}"
        by_bucket[key] = by_bucket.get(key, 0.0) + float(value)
    return dict(trades=len(taken), closed=len(closed), unresolved=len(taken) - len(closed),
        trades_per_week=len(taken) / weeks if weeks else 0.0, net_pnl=float(pnl.sum()),
        expectancy=float(pnl.mean()) if len(pnl) else 0.0, profit_factor=float(wins / losses) if losses else None,
        no_losing_trades=bool(len(pnl) and not losses and wins > 0),
        win_rate=float((pnl > 0).mean()) if len(pnl) else None, max_drawdown=float(np.max(np.maximum.accumulate(curve) - curve)),
        gross_pnl=float(sum(p["gross_pnl"] for p in closed)), total_cost=float(sum(p["total_cost"] for p in closed)),
        buckets=by_bucket, positive_bucket_share=(float(np.mean([v > 0 for v in by_bucket.values()])) if by_bucket else 0.0))


def feasible(m: dict) -> bool:
    pf = m["profit_factor"]
    return (m["trades_per_week"] >= MIN_TRADES_PER_WEEK and m["net_pnl"] > 0
            and ((pf is not None and pf > 1) or m["no_losing_trades"]) and m["positive_bucket_share"] >= 0.5)


def span_weeks(rows: pd.DataFrame) -> float:
    return max((rows.datetime.max() - rows.datetime.min()).total_seconds() / (7 * 86400), 1.0)


def run_model(model_id: int, out: Path, tapes: dict, scope, bindings) -> dict:
    model = model_entry(model_id, bindings)
    stream = model["stream"]
    cache = out / str(model_id) / "outcomes.pkl"
    if cache.exists():
        with cache.open("rb") as handle:
            data = pickle.load(handle)
    else:
        history = history_times(model) if stream == "EURUSD" else None
        data = {}
        for period, rows in periods(model, history).items():
            if rows.empty:
                data[period] = dict(rows=rows, signals=[], outcomes=[])
                continue
            tape, atr = tapes[stream]
            signals, outcomes = outcomes_for(model, rows, tape, atr, scope, bindings)
            data[period] = dict(rows=rows, signals=[s.model_dump_json() for s in signals], outcomes=outcomes)
            print(f"ID {model_id} {period}: {len(rows)} setup rows -> {len(signals)} signals", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as handle:
            pickle.dump(data, handle)
    contracts = {c.contract_id: c for c in (resolve_model_setup_route(model_id=model["model_id"], setup_family=f,
                 instrument=stream, side=model["binding"].side, manifest=bindings) for f in model["binding"].setup_bindings)}
    simulator = PositionContract(simulator_id=f"b2_v2_sweep_{model_id}_v1", version=1, policy_id="b2_v2_threshold_sweep_20261007",
        policy_hash=content_hash({"sweep": smoke.COLLECTION_ID, "model": model_id}), max_positions=1, max_quantity=1.0,
        concurrency_scope="instrument", allow_opposite=False, cooldown_minutes=0, starting_equity=100000.0,
        accounting_unit=UNITS[stream])
    bucket = "quarter" if model_id in HALF_SPLIT_IDS else "year"
    result = dict(model_id=model_id, target=model["target"], stream=stream, unit=UNITS[stream],
                  selection_source="first half of test-period confirmation bars (runtime scores)" if model_id in HALF_SPLIT_IDS
                  else "out-of-fold development scores", evaluation_source="second half of test period" if model_id in HALF_SPLIT_IDS
                  else "held-out test split", min_trades_per_week=MIN_TRADES_PER_WEEK)
    loaded = {}
    for period in ("select", "evaluate"):
        part = data[period]
        signals = [SignalEvidence.model_validate_json(s) for s in part["signals"]]
        loaded[period] = (part["rows"], signals, part["outcomes"], {s.event_id: s for s in signals})
        rows = part["rows"]
        result[f"{period}_period"] = dict(start=str(rows.datetime.min()), end=str(rows.datetime.max()), weeks=span_weeks(rows),
            setup_rows=len(rows), signals=len(signals),
            scope_admissible=sum(1 for o in part["outcomes"] if o["baseline_scope"] == "included"))

    def at(period: str, threshold: float) -> dict:
        rows, signals, outcomes, by_id = loaded[period]
        if not signals:
            return metrics([], by_id, span_weeks(rows), bucket)
        keep = [i for i, s in enumerate(signals) if s.audit_context.calibrated_probability >= threshold]
        positions = ledger([signals[i] for i in keep], [outcomes[i] for i in keep], contracts, simulator, scope) if keep else []
        return metrics(positions, by_id, span_weeks(rows), bucket)

    select_scores = np.array([s.audit_context.calibrated_probability for s in loaded["select"][1]])
    grid = sorted({0.0, *np.round(np.quantile(select_scores, np.linspace(0, 0.99, 100)), 6)}) if len(select_scores) else [0.0]
    sweep = []
    for threshold in grid:
        m = at("select", float(threshold))
        sweep.append(dict(threshold=float(threshold), **{k: v for k, v in m.items() if k != "buckets"}, feasible=feasible(m)))
    pd.DataFrame(sweep).to_csv(out / str(model_id) / "sweep_select.csv", index=False)
    candidates = [r for r in sweep if r["feasible"]]
    result["setup_only_select"] = {k: v for k, v in at("select", -1.0).items()}
    result["setup_only_evaluate"] = {k: v for k, v in at("evaluate", -1.0).items()}
    result["max_possible_trades_per_week_select"] = result["setup_only_select"]["trades_per_week"]
    if candidates:
        best = max(candidates, key=lambda r: (r["trades_per_week"], r["net_pnl"]))
        result.update(status="threshold_selected", threshold=best["threshold"], selected=best,
                      evaluation=at("evaluate", best["threshold"]))
        ev = result["evaluation"]
        result["evaluation_passes"] = feasible(ev)
    else:
        positive = [r for r in sweep if r["net_pnl"] > 0 and ((r["profit_factor"] or 0) > 1 or r["no_losing_trades"])]
        result.update(status="no_feasible_threshold",
            reason=("fewer than 6 detector setups a week on the declared routes even with no threshold"
                    if result["max_possible_trades_per_week_select"] < MIN_TRADES_PER_WEEK else
                    "no threshold keeps >= 6 trades/week with positive returns in the selection period"),
            most_trades_with_positive_return=max(positive, key=lambda r: (r["trades_per_week"], r["net_pnl"])) if positive else None,
            best_net_pnl=max(sweep, key=lambda r: r["net_pnl"]))
        if result["most_trades_with_positive_return"]:
            alt = result["most_trades_with_positive_return"]["threshold"]
            result["evaluation_of_most_trades_positive"] = at("evaluate", alt)
    dump(out / str(model_id) / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", type=int, default=list(IDS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a root under artifacts")
    out.mkdir(parents=True, exist_ok=True)
    scope, bindings = load_research_scope(), load_route_bindings()
    tapes = {}
    for stream, start, end in (("ES", "2017-06-01", "2026-12-31"), ("EURUSD", "2018-12-01", "2026-12-31")):
        if any((stream == "ES") == (i in (2, 3, 5, 10)) for i in args.ids) and not all(
                (out / str(i) / "outcomes.pkl").exists() for i in args.ids if (stream == "ES") == (i in (2, 3, 5, 10))):
            print(f"Building {stream} B1 tape", flush=True)
            tapes[stream] = build_tape(stream, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
    summary = {}
    for model_id in args.ids:
        try:
            result = run_model(model_id, out, tapes, scope, bindings)
        except Exception as exc:  # one model never stops the sweep
            summary[str(model_id)] = dict(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
            print(f"ID {model_id}: ERROR {exc}", flush=True)
            continue
        brief = {k: result.get(k) for k in ("status", "threshold", "reason", "evaluation_passes")}
        sel, ev = result.get("selected") or {}, result.get("evaluation") or {}
        brief.update(select_trades_per_week=sel.get("trades_per_week"), select_net=sel.get("net_pnl"), select_pf=sel.get("profit_factor"),
                     eval_trades_per_week=ev.get("trades_per_week"), eval_net=ev.get("net_pnl"), eval_pf=ev.get("profit_factor"),
                     max_possible_trades_per_week=result["max_possible_trades_per_week_select"])
        summary[str(model_id)] = brief
        print(f"ID {model_id}: {json.dumps(brief, default=str)}", flush=True)
    dump(out / "summary.json", dict(models=summary, tape_notes=TAPE_NOTES))
    print("SWEEP COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
