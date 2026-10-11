"""Per-model floor of 1 trade/week, portfolio check of >= 6 trades/week (user, 2026-10-08). Research only.

Each candidate model gets its own threshold, picked on development scores only: the most trades/week among thresholds
with >= 1 trade/week, net P&L > 0, profit factor > 1 and net positive in at least half of the selection buckets (years;
quarters for 22/26, whose selection set is the first half of their test-period confirmations). The selected models
are then combined. A trade two models would both take (same instrument, side, setup family and entry bar) counts once.
Two capacity rules are reported: one position per model (concurrency_scope="model") and one position per instrument
(the B1 research ledger). Development rates come from each model's own selection window; the combined ledger runs on
the inspected period (already seen by the 2026-10-07 sweep: reported, never a fresh holdout).

Candidates: the 9 v2 roster models of the sweep (cached B1 outcomes and scores) and the 12 B1-outcome research models
(scores regenerated with the same code, seeds and folds as ``scripts/train_b1_outcome_models.py``).

Usage: python scripts/portfolio_b1_floor_research.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from model_testing import b1_fast_outcomes as fast
from model_testing.outcome_contracts import load_contracts
from model_testing.outcome_engine import position_costs
from scripts import train_b1_outcome_models as tr
from scripts.research_b1_outcome_labels import DEV_END, dump

OUT = ROOT / "artifacts/b1_floor_portfolio_20261008"
REPORT = ROOT / "docs/audits/b1_floor_portfolio_20261008.json"
SWEEP = ROOT / "artifacts/b2_v2_threshold_sweep_20261007"
ROSTER = {2: "ES", 3: "ES", 5: "ES", 10: "ES", 21: "EURUSD", 22: "EURUSD", 23: "EURUSD", 25: "EURUSD", 26: "EURUSD"}
HALF_SPLIT = (22, 26)
FLOOR, PORTFOLIO_FLOOR = 1.0, 6.0
GRID = np.linspace(0, 0.995, 200)
COLS = ["event_id", "known_ns", "entry_ns", "exit_ns", "resolved", "executable", "gross", "cost", "net", "side", "family", "score"]


def span(frame: pd.DataFrame) -> float:
    return max((frame.known_ns.max() - frame.known_ns.min()) / tr.fast.FIVE_MIN / (12 * 24 * 7), 1.0)


def metrics(frame: pd.DataFrame, weeks: float, bucket: str) -> dict:
    if frame.empty:
        return dict(trades=0, trades_per_week=0.0, net_total=0.0, feasible=False)
    frame = frame.reset_index(drop=True)
    taken = fast.ledger(frame)
    t = frame.loc[taken]
    closed = t.loc[t.resolved]
    net = closed.net.to_numpy()
    stamps = pd.DatetimeIndex(closed.known_ns.to_numpy().astype("datetime64[ns]"))
    keys = stamps.year if bucket == "year" else stamps.year * 10 + stamps.quarter
    buckets = pd.Series(net).groupby(np.asarray(keys)).sum() if len(net) else pd.Series(dtype=float)
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    pf = float(wins / losses) if losses else None
    tpw = taken.sum() / weeks
    return dict(trades=int(taken.sum()), trades_per_week=float(tpw), closed=int(len(closed)), net_total=float(net.sum()),
                net_per_trade=float(net.mean()) if len(net) else None, gross_per_trade=float(closed.gross.mean()) if len(closed) else None,
                win_rate=float((net > 0).mean()) if len(net) else None, profit_factor=pf,
                positive_bucket_share=float((buckets > 0).mean()) if len(buckets) else 0.0,
                feasible=bool(tpw >= FLOOR and net.sum() > 0 and ((pf or 0) > 1 or (len(net) and not losses))
                              and len(buckets) and (buckets > 0).mean() >= 0.5))


def choose(dev: pd.DataFrame, weeks: float, bucket: str) -> tuple[dict | None, list[dict]]:
    scored = dev.loc[dev.score.notna()]
    grid = sorted({-np.inf, *np.unique(np.quantile(scored.score, GRID))}) if len(scored) else []
    table = [dict(threshold=float(t) if np.isfinite(t) else None, **metrics(scored.loc[scored.score >= t], weeks, bucket)) for t in grid]
    feasible = [r for r in table if r["feasible"]]
    return (max(feasible, key=lambda r: (r["trades_per_week"], r["net_total"])) if feasible else None), table


# ---------------------------------------------------------------------------------------------------- roster models
def roster_frames(model_id: int) -> dict[str, pd.DataFrame]:
    """The sweep's cached B1 outcomes as ledger frames (scope-excluded and non-executable rows never take capacity)."""
    from model_testing.outcome_engine import SignalEvidence
    contracts = load_contracts()
    with (SWEEP / str(model_id) / "outcomes.pkl").open("rb") as handle:
        data = pickle.load(handle)
    out = {}
    for period, name in (("select", "dev"), ("evaluate", "inspected")):
        signals = [SignalEvidence.model_validate_json(s) for s in data[period]["signals"]]
        rows = []
        for s, o in zip(signals, data[period]["outcomes"]):
            executable = (o["baseline_scope"] == "included" and o["entry_ts"] is not None and o["paper_contract_outcome"] != "INVALID")
            gross = cost = None
            if executable and o["paper_resolved"]:
                c = contracts[o["evaluation_contract_id"]].costs
                cost = position_costs(entry_ts=o["entry_ts"], exit_ts=o["paper_exit_ts"], cost=c, quantity=1.0, accounting_unit=c.unit)["total_cost"]
                gross = (1 if s.side == "long" else -1) * (o["paper_exit_price"] - o["entry_price"]) / c.price_increment
            rows.append(dict(event_id=s.event_id, known_ns=pd.Timestamp(s.known_at_ts).value,
                             entry_ns=pd.Timestamp(o["entry_ts"]).value if o["entry_ts"] is not None else fast.NAT,
                             exit_ns=pd.Timestamp(o["paper_exit_ts"]).value if o["paper_exit_ts"] is not None else fast.NAT,
                             resolved=bool(executable and o["paper_resolved"]), executable=bool(executable), gross=gross, cost=cost,
                             net=None if gross is None else gross - cost, side=1 if s.side == "long" else -1, family=s.setup_family,
                             score=s.audit_context.calibrated_probability))
        frame = pd.DataFrame(rows)
        for c in ("gross", "cost", "net"):
            frame[c] = frame[c].astype(float)
        out[name] = frame
    return out


# -------------------------------------------------------------------------------------------------- research models
def research_frames(stream: str, K: int, label: str, feats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    path = OUT / "scores" / f"{stream}_K{K}_{label}.parquet"
    if path.exists():
        both = pd.read_parquet(path)
    else:
        rows = tr.load_rows(stream, K, feats)
        cols = tr.predictors(rows)
        dev = rows.loc[rows.period.eq("dev")].copy()
        ins = rows.loc[rows.period.eq("inspected")].copy()
        dev["score"], _ = tr.oof(dev, cols, label)
        final = tr.fit(dev.loc[dev.group_end_ns < DEV_END[stream].value - tr.EMBARGO], cols, label)
        ins["score"] = tr.score(final, ins, cols, label)
        both = pd.concat([dev.assign(split="dev"), ins.assign(split="inspected")])[COLS + ["split"]]
        path.parent.mkdir(parents=True, exist_ok=True)
        both.to_parquet(path, index=False)
    return {s: both.loc[both.split.eq(s)].drop(columns="split").reset_index(drop=True) for s in ("dev", "inspected")}


# --------------------------------------------------------------------------------------------------------- portfolio
def portfolio(selected: dict[str, dict], frames: dict[str, dict], period: str) -> dict:
    """Combined trades of the selected models on one period; duplicates (same trade) counted once."""
    out = {}
    for stream in ("ES", "EURUSD"):
        parts = []
        for name, choice in selected.items():
            if choice["stream"] != stream:
                continue
            f = frames[name][period]
            f = f.loc[f.score >= choice["threshold"]].copy()
            f["model"], f["priority"] = name, choice["dev_net_per_trade_rank"]
            parts.append(f)
        if not parts:
            continue
        union = pd.concat(parts, ignore_index=True)
        weeks = span(union)
        # one position per model, then de-duplicate identical trades across models
        per_model = pd.concat([p.loc[fast.ledger(p.reset_index(drop=True))] for p in (q.reset_index(drop=True) for q in parts)], ignore_index=True)
        unique = per_model.sort_values("priority").drop_duplicates(["side", "family", "entry_ns"])
        closed = unique.loc[unique.resolved]
        model_scope = dict(trades=len(unique), trades_per_week=len(unique) / weeks, net_total=float(closed.net.sum()),
                           net_per_trade=float(closed.net.mean()) if len(closed) else None,
                           gross_per_trade=float(closed.gross.mean()) if len(closed) else None)
        # one position per instrument (B1 research ledger), ties by dev net/trade rank
        dedup = union.sort_values("priority").drop_duplicates(["side", "family", "entry_ns"])
        dedup["event_id"] = dedup.priority.astype(str).str.zfill(3) + "-" + dedup.event_id
        out[stream] = dict(models=sorted({p.model.iloc[0] for p in parts}), weeks=weeks, one_position_per_model=model_scope,
                           one_position_per_instrument=metrics(dedup, weeks, "year"))
    total = sum(v["one_position_per_model"]["trades_per_week"] for v in out.values())
    out["combined_trades_per_week_per_model_scope"] = total
    out["combined_trades_per_week_per_instrument_scope"] = sum(v["one_position_per_instrument"]["trades_per_week"] for k, v in out.items() if isinstance(v, dict))
    out["combined_net_per_model_scope"] = sum(v["one_position_per_model"]["net_total"] for k, v in out.items() if isinstance(v, dict) and "one_position_per_model" in v)
    return out


def main() -> None:
    frames, results = {}, {}
    for model_id, stream in ROSTER.items():
        name = f"roster_{model_id}"
        frames[name] = roster_frames(model_id)
        bucket = "quarter" if model_id in HALF_SPLIT else "year"
        choice, table = choose(frames[name]["dev"], span(frames[name]["dev"]), bucket)
        results[name] = dict(stream=stream, bucket=bucket, dev_weeks=span(frames[name]["dev"]), choice=choice,
                             positive_thresholds=sum(1 for r in table if r["net_total"] > 0 and ((r.get("profit_factor") or 0) > 1)),
                             no_threshold=table[0] if table else None)
        print(f"{name}: {choice and {k: round(choice[k], 3) for k in ('threshold', 'trades_per_week', 'net_total')}}", flush=True)
    for stream in ("ES", "EURUSD"):
        feats = tr.features(stream)
        for K in tr.WINDOWS:
            for label in tr.LABELS:
                name = f"research_{stream}_K{K}_{label}"
                frames[name] = research_frames(stream, K, label, feats)
                dev = frames[name]["dev"]
                choice, table = choose(dev, span(dev.loc[dev.score.notna()]), "year")
                results[name] = dict(stream=stream, bucket="year", dev_weeks=span(dev.loc[dev.score.notna()]), choice=choice,
                                     positive_thresholds=sum(1 for r in table if r["net_total"] > 0 and ((r.get("profit_factor") or 0) > 1)),
                                     no_threshold=table[0] if table else None)
                print(f"{name}: {choice and {k: round(choice[k], 3) for k in ('threshold', 'trades_per_week', 'net_total')}}", flush=True)
        del feats
    picked = {n: r for n, r in results.items() if r["choice"]}
    ranks = {n: i for i, (n, _) in enumerate(sorted(picked.items(), key=lambda kv: -(kv[1]["choice"]["net_per_trade"] or 0)))}
    selected = {n: dict(stream=r["stream"], threshold=r["choice"]["threshold"], dev_net_per_trade_rank=ranks[n]) for n, r in picked.items()}
    for name in selected:
        f = frames[name]["inspected"]
        results[name]["inspected_at_choice"] = metrics(f.loc[f.score >= selected[name]["threshold"]], span(f), results[name]["bucket"])
    dev_rate = sum(r["choice"]["trades_per_week"] for r in picked.values())
    report = dict(status="research_only_nothing_frozen_or_activated", per_model_floor=FLOOR, portfolio_floor=PORTFOLIO_FLOOR,
                  rule="most trades/week among thresholds with >= 1/week, net > 0, PF > 1, positive in >= half the buckets (dev only)",
                  models=results, selected=selected, models_meeting_floor=len(selected),
                  dev_sum_of_selected_rates=dev_rate, dev_portfolio_meets_6=dev_rate >= PORTFOLIO_FLOOR,
                  inspected_portfolio=portfolio(selected, frames, "inspected") if selected else None,
                  note="research variants of one stream share candidates; identical trades are counted once in the portfolio")
    dump(REPORT, report)
    print(json.dumps(dict(selected=selected, dev_sum=dev_rate, inspected=report["inspected_portfolio"]), indent=1, default=str), flush=True)
    print("PORTFOLIO COMPLETE", flush=True)


if __name__ == "__main__":
    main()
