"""Retraining path on B1 outcomes (option A, 2026-10-08). Research only: models are written under a research root,
never frozen, installed or activated, and a label metric alone never promotes anything.

Rows: setup-gated candidates from ``scripts/research_b1_outcome_labels.py census`` (setup bar k=0 and, for post window
K>0, the valid post-setup bars k=1..K, each priced from its own decision time). Executable rows only; unresolved
(DATA_GAP) rows have no label and are kept out of fitting but stay in every ledger (they hold capacity).
Labels: ``binary`` = B1 net P&L > 0; ``net`` = B1 net P&L in ticks/pips (regression; expected net return).
Predictors (decision-time only): the raw v2 source-history features of the decision bar (ES: union of the selected
features of models 2/3/5/10; EURUSD: the union tape of 21-26; rankings were fit on a frozen prefix inside development),
plus candidate context known at decision time: side, setup family, offset k, structural ATR (stop/target distance in
units), entry-session spread and the cost-to-stop ratio, hour and weekday.

Validation: 5 expanding walk-forward folds over the later development period; a training row must come from a setup
that fired before the fold and whose every candidate's outcome window ended at least one day before the fold
(purge + embargo; candidates of one setup stay together). K, label and threshold are chosen on out-of-fold scores
only, with the same feasibility rule as the sweep (>= 6 trades/week, net > 0, PF > 1, positive in >= half the years)
under the B1 one-position ledger. The final model is refit on all development rows (purged at the boundary) and scored
once on the inspected period, which the 2026-10-07 sweep already looked at: reported, never called a fresh holdout.

Usage: python scripts/train_b1_outcome_models.py [--streams ES EURUSD]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from model_testing import b1_fast_outcomes as fast
from scripts.report_b1_outcome_research import MIN_TPW, trade_stats, weeks
from scripts.research_b1_outcome_labels import CACHE, DEV_END, ROUTES, UNITS, dump

OUT = ROOT / "models/b1_outcome_research_20261008"
REPORT = ROOT / "docs/audits/b1_outcome_models_20261008.json"
HISTORY = ROOT / "artifacts/b2_v2_source_history_20261006"
ES_FEATURE_MODELS = {2: "long_frvp_continuation", 3: "long_frvp_meta", 5: "long_frvp_continuation_setup2", 10: "long_frvp_reversal"}
WINDOWS = (0, 1, 3)
LABELS = ("binary", "net")
FOLDS = 5
EMBARGO = pd.Timedelta(days=1).value
PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=20,
              reg_lambda=5.0, tree_method="hist", n_jobs=8, random_state=20261008)
QUANTILES = (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)


def features(stream: str) -> pd.DataFrame:
    if stream == "EURUSD":
        frame = pd.read_parquet(HISTORY / "source_history_union.parquet").drop(columns=["source_row_idx"])
    else:
        frame = None
        for model_id, target in ES_FEATURE_MODELS.items():
            part = pd.read_parquet(HISTORY / str(model_id) / "prepared" / target / "source_history.parquet").drop(columns=["source_row_idx"])
            frame = part if frame is None else frame.merge(part[["datetime", *[c for c in part if c not in frame]]], on="datetime", how="outer")
    frame["open_ns"] = pd.to_datetime(frame.pop("datetime"), utc=True).dt.as_unit("ns").astype("int64")
    return frame


def candidate_context(rows: pd.DataFrame, stream: str) -> pd.DataFrame:
    ny = pd.DatetimeIndex(rows.candidate_open_ns.to_numpy().astype("datetime64[ns]")).tz_localize("UTC").tz_convert("America/New_York")
    ctx = pd.DataFrame(index=rows.index)
    ctx["ctx_side"] = rows.side.to_numpy()
    ctx["ctx_offset"] = rows.offset.to_numpy()
    ctx["ctx_stop_units"] = rows.stop_units.to_numpy()
    ctx["ctx_target_units"] = rows.target_units.to_numpy()
    ctx["ctx_entry_spread"] = rows.entry_spread.to_numpy()
    ctx["ctx_cost_to_stop"] = (2 * rows.entry_spread / rows.stop_units).to_numpy()
    ctx["ctx_hour_ny"] = ny.hour + ny.minute / 60
    ctx["ctx_weekday"] = ny.weekday
    for family in sorted(rows.family.unique()):
        ctx[f"ctx_family_{family}"] = rows.family.eq(family).astype(np.int8).to_numpy()
    return ctx


def load_rows(stream: str, K: int, feats: pd.DataFrame) -> pd.DataFrame:
    census = pd.read_parquet(CACHE / f"census_{stream}.parquet")
    rows = census.loc[census.offset.between(0, K) & census.valid & census.executable].reset_index(drop=True)
    rows = rows.merge(feats, left_on="candidate_open_ns", right_on="open_ns", how="left", validate="many_to_one")
    rows = pd.concat([rows, candidate_context(rows, stream)], axis=1)
    rows["setup_ns"] = pd.to_datetime(rows.decision_open, utc=True).dt.as_unit("ns").astype("int64")
    # Outcome window end used for purging; an unresolved row is open to its last usable bar (bound: entry + timeout).
    end = np.where(rows.resolved, rows.exit_ns, rows.entry_ns + 120 * fast.FIVE_MIN)
    rows["window_end_ns"] = end
    rows["group_end_ns"] = rows.groupby("setup_id").window_end_ns.transform("max")
    return rows


def predictors(rows: pd.DataFrame) -> list[str]:
    skip = set(pd.read_parquet(CACHE / f"census_ES.parquet", columns=None).columns) | {"open_ns", "setup_ns", "window_end_ns", "group_end_ns"}
    return [c for c in rows.columns if c not in skip and rows[c].dtype.kind in "fiub"]


def fit(rows: pd.DataFrame, cols: list[str], label: str):
    train = rows.loc[rows.resolved]
    if label == "binary":
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **PARAMS)
        model.fit(train[cols], (train.net > 0).astype(int))
    else:
        model = xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=20.0, **PARAMS)
        model.fit(train[cols], train.net.clip(train.net.quantile(0.01), train.net.quantile(0.99)))
    return model


def score(model, rows: pd.DataFrame, cols: list[str], label: str) -> np.ndarray:
    return model.predict_proba(rows[cols])[:, 1] if label == "binary" else model.predict(rows[cols])


def folds(dev: pd.DataFrame) -> list[tuple[int, int]]:
    start, end = int(dev.setup_ns.min()), int(dev.setup_ns.max()) + 1
    first = start + int(0.4 * (end - start))
    edges = np.linspace(first, end, FOLDS + 1).astype(np.int64)
    return list(zip(edges[:-1], edges[1:]))


def oof(dev: pd.DataFrame, cols: list[str], label: str) -> tuple[pd.Series, list[dict]]:
    out = pd.Series(np.nan, index=dev.index)
    plan = []
    for a, b in folds(dev):
        train = dev.loc[(dev.group_end_ns < a - EMBARGO) & (dev.setup_ns < a)]
        val = dev.loc[(dev.setup_ns >= a) & (dev.setup_ns < b)]
        model = fit(train, cols, label)
        out.loc[val.index] = score(model, val, cols, label)
        plan.append(dict(val_start=str(pd.Timestamp(a, tz="UTC")), val_end=str(pd.Timestamp(b, tz="UTC")), train_rows=int(len(train)),
                         train_last_window_end=str(pd.Timestamp(int(train.window_end_ns.max()), tz="UTC")), val_rows=int(len(val)),
                         purged_groups_shared_with_val=int(len(set(train.setup_id) & set(val.setup_id)))))
    return out, plan


def sweep(rows: pd.DataFrame, scores: pd.Series, span: float) -> list[dict]:
    valid = scores.notna()
    grid = sorted(set(np.round(np.quantile(scores[valid], QUANTILES), 8)))
    out = []
    for t in grid:
        s = trade_stats(rows.loc[valid & (scores >= t)], span)
        out.append(dict(threshold=float(t), **{k: s.get(k) for k in ("trades_per_week", "gross_per_trade", "net_per_trade", "net_total",
                                                                     "win_rate", "profit_factor", "positive_year_share", "feasible")}))
    return out


def pick(table: list[dict]) -> dict | None:
    feasible = [r for r in table if r["feasible"]]
    return max(feasible, key=lambda r: (r["trades_per_week"], r["net_total"])) if feasible else None


def run_stream(stream: str) -> dict:
    feats = features(stream)
    result = dict(stream=stream, unit=UNITS[stream], dev_end=str(DEV_END[stream]), candidates={})
    for K in WINDOWS:
        rows = load_rows(stream, K, feats)
        cols = predictors(rows)
        dev = rows.loc[rows.period.eq("dev")].copy()
        ins = rows.loc[rows.period.eq("inspected")].copy()
        oof_span = None
        for label in LABELS:
            key = f"K{K}_{label}"
            scores, plan = oof(dev, cols, label)
            scored = dev.loc[scores.notna()]
            oof_span = weeks(int(scored.known_ns.min()), int(scored.known_ns.max()))
            table = sweep(dev, scores, oof_span)
            chosen = pick(table)
            unscored = trade_stats(scored, oof_span)
            routes = {}
            for model_id, (s, side, families) in ROUTES.items():
                if s != stream:
                    continue
                mask = dev.side.eq(1 if side == "long" else -1) & dev.family.isin(families)
                rt = sweep(dev.loc[mask], scores.loc[mask], oof_span)
                routes[str(model_id)] = dict(oof_no_threshold=trade_stats(dev.loc[mask & scores.notna()], oof_span), chosen=pick(rt),
                                             best_net=max(rt, key=lambda r: r["net_total"]) if rt else None)
            # Final refit on all development rows (purged at the boundary), scored once on the inspected period.
            dev_end = DEV_END[stream].value
            final = fit(dev.loc[dev.group_end_ns < dev_end - EMBARGO], cols, label)
            ins_scores = pd.Series(score(final, ins, cols, label), index=ins.index)
            ins_span = weeks(dev_end, int(ins.known_ns.max()))
            inspected = dict(no_threshold=trade_stats(ins, ins_span),
                             at_oof_threshold=trade_stats(ins.loc[ins_scores >= chosen["threshold"]], ins_span) if chosen else None,
                             sweep_for_reference_only=sweep(ins, ins_scores, ins_span))
            # Score quality on OOF: does the score rank B1 net? (Spearman, and net by score decile.)
            r = scored.loc[scored.resolved]
            sc = scores.loc[r.index]
            deciles = pd.qcut(sc.rank(method="first"), 10, labels=False)
            by_decile = r.groupby(deciles.to_numpy()).net.agg(["mean", "count"]).reset_index().to_dict("records")
            importance = pd.Series(final.feature_importances_, index=cols).sort_values(ascending=False).head(15)
            model_dir = OUT / stream / key
            model_dir.mkdir(parents=True, exist_ok=True)
            final.save_model(model_dir / "model.json")
            dump(model_dir / "manifest.json", dict(status="research_only_not_frozen_not_installed", stream=stream, post_window=K, label=label,
                                                   predictors=cols, params=PARAMS, fit_rows=int(dev.resolved.sum()), dev_end=str(DEV_END[stream])))
            result["candidates"][key] = dict(post_window=K, label=label, rows_dev=int(len(dev)), rows_inspected=int(len(ins)),
                predictors=len(cols), fold_plan=plan, oof_weeks=oof_span, oof_spearman_score_vs_net=float(sc.corr(r.net, method="spearman")),
                oof_net_by_score_decile=by_decile, oof_no_threshold=unscored, oof_sweep=table, chosen=chosen, routes=routes,
                inspected=inspected, top_importance=importance.round(4).to_dict())
            print(f"{stream} {key}: rows {len(dev)}; spearman {result['candidates'][key]['oof_spearman_score_vs_net']:.3f}; "
                  f"chosen {chosen and {k: chosen[k] for k in ('threshold', 'trades_per_week', 'net_per_trade')}}", flush=True)
    feasible = {k: v["chosen"] for k, v in result["candidates"].items() if v["chosen"]}
    result["selected"] = (max(feasible.items(), key=lambda kv: (kv[1]["trades_per_week"], kv[1]["net_total"]))[0] if feasible else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--streams", nargs="*", default=["ES", "EURUSD"])
    args = parser.parse_args()
    report = dict(status="research_only_nothing_frozen_or_activated", min_trades_per_week=MIN_TPW, windows=WINDOWS, labels=LABELS,
                  params=PARAMS, streams={})
    if REPORT.exists():
        report = json.loads(REPORT.read_text())
    for stream in args.streams:
        report["streams"][stream] = run_stream(stream)
        dump(REPORT, report)
    print(f"TRAINING COMPLETE: {REPORT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
