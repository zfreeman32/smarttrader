"""Analysis for ``scripts/research_b1_outcome_labels.py`` (stage ``report``). Reads only the research caches, the
label artifacts and the saved v2 model predictions; writes ``docs/audits/b1_outcome_research_20261008.json``.

Selection-relevant numbers come from the development period only; the inspected period is reported alongside and
labelled ``inspected`` (already looked at by the 2026-10-07 sweep, so never a fresh holdout).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from model_testing import b1_fast_outcomes as fast
from scripts.research_b1_outcome_labels import (CACHE, DEV_END, FRVP_LABEL_EVENTS, OUT, REPORT, ROUTES, TEMPLATES, UNITS,
                                                dump, load)

MIN_TPW = 6.0
WEEK_NS = 7 * 86400 * 10**9


def weeks(start_ns: int, end_ns: int) -> float:
    return max((end_ns - start_ns) / WEEK_NS, 1.0)


def period_bounds(stream: str, census: pd.DataFrame) -> dict[str, tuple[int, int]]:
    known = census.loc[census.offset.eq(0) & census.known_ns.ne(fast.NAT), "known_ns"]
    dev_end = DEV_END[stream].value
    return dict(dev=(int(known.min()), dev_end), inspected=(dev_end, int(known.max())))


def trade_stats(frame: pd.DataFrame, span_weeks: float, restart="week") -> dict:
    """B1 ledger (one position) on the frame's candidates; per-trade and per-week metrics on closed trades."""
    if frame.empty:
        return dict(candidates=0, candidates_per_week=0.0, trades=0, trades_per_week=0.0)
    frame = frame.reset_index(drop=True)
    taken = fast.ledger(frame, restart=restart)
    t = frame.loc[taken]
    closed = t.loc[t.resolved]
    net = closed.net.to_numpy()
    years = pd.DatetimeIndex(closed.known_ns.to_numpy().astype("datetime64[ns]")).year
    by_year = pd.Series(net).groupby(years).sum() if len(net) else pd.Series(dtype=float)
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    return dict(candidates=int(frame.executable.sum()), candidates_per_week=float(frame.executable.sum() / span_weeks),
                trades=int(taken.sum()), unresolved=int(taken.sum() - len(closed)), trades_per_week=float(taken.sum() / span_weeks),
                gross_per_trade=float(closed.gross.mean()) if len(closed) else None, cost_per_trade=float(closed.cost.mean()) if len(closed) else None,
                net_per_trade=float(net.mean()) if len(net) else None, net_total=float(net.sum()),
                win_rate=float((net > 0).mean()) if len(net) else None, profit_factor=float(wins / losses) if losses else None,
                positive_year_share=float((by_year > 0).mean()) if len(by_year) else None,
                feasible=bool(taken.sum() / span_weeks >= MIN_TPW and net.sum() > 0 and losses and wins / losses > 1
                              and len(by_year) and (by_year > 0).mean() >= 0.5))


def payoff(frame: pd.DataFrame) -> dict:
    """Per-candidate (no ledger) outcome profile of resolved trades."""
    r = frame.loc[frame.resolved]
    if r.empty:
        return dict(resolved=0)
    net = r.net
    win, loss = net[net > 0], net[net <= 0]
    aw, al = (float(win.mean()) if len(win) else 0.0), (float(-loss.mean()) if len(loss) else 0.0)
    return dict(resolved=int(len(r)), gross_mean=float(r.gross.mean()), cost_mean=float(r.cost.mean()), net_mean=float(net.mean()),
                net_median=float(net.median()), win_rate_net=float((net > 0).mean()), win_rate_gross=float((r.gross > 0).mean()),
                avg_win=aw, avg_loss=al, breakeven_win_rate=float(al / (aw + al)) if aw + al else None,
                target_share=float(r.outcome.eq("TARGET").mean()), stop_share=float(r.outcome.eq("STOP").mean()),
                timeout_share=float(r.outcome.str.startswith("TIMEOUT").mean()),
                stop_units_mean=float(r.stop_units.mean()), target_units_mean=float(r.target_units.mean()),
                cost_over_target=float((r.cost / r.target_units).mean()), bars_held_mean=float(r.bars_held.mean()),
                net_std=float(net.std()))


def outcome_counts(frame: pd.DataFrame) -> dict:
    return {k: int(v) for k, v in frame.outcome.value_counts().items()}


# ---------------------------------------------------------------------------------------------------------- baseline
def baseline(stream: str, census: pd.DataFrame, bounds) -> dict:
    k0 = census.loc[census.offset.eq(0)]
    out = {}
    for period, (a, b) in bounds.items():
        rows = k0.loc[k0.period.eq(period)]
        w = weeks(a, b)
        fam = {}
        for (family, side), g in rows.groupby(["family", "side"]):
            fam[f"{family}_{'long' if side > 0 else 'short'}"] = dict(setups=len(g), setups_per_week=len(g) / w, outcomes=outcome_counts(g),
                                                                      payoff=payoff(g), ledger=trade_stats(g, w))
        routes = {}
        for model_id, (s, side, families) in ROUTES.items():
            if s != stream:
                continue
            g = rows.loc[rows.side.eq(1 if side == "long" else -1) & rows.family.isin(families)]
            routes[str(model_id)] = dict(families=families, side=side, setups_per_week=len(g) / w, payoff=payoff(g), ledger=trade_stats(g, w))
        pooled = {name: dict(payoff=payoff(g), ledger=trade_stats(g, w), setups_per_week=len(g) / w, outcomes=outcome_counts(g))
                  for name, g in (("all_setups_both_sides", rows), ("all_long", rows.loc[rows.side.eq(1)]), ("all_short", rows.loc[rows.side.eq(-1)]))}
        out[period] = dict(weeks=w, by_family_side=fam, roster_routes=routes, pooled=pooled)
    return out


# ------------------------------------------------------------------------------------------------------- label audit
def label_audit_frvp(census: pd.DataFrame) -> dict:
    e = pd.read_csv(FRVP_LABEL_EVENTS, usecols=["setup_type", "setup_side", "swing_time", "tb_outcome", "excluded", "label_quality"])
    e["decision_open"] = pd.to_datetime(e.swing_time, utc=True)
    e["family"] = "setup" + e.setup_type.astype(str)
    e["side"] = np.where(e.setup_side > 0, 1, -1)
    k0 = census.loc[census.offset.eq(0)].copy()
    k0["decision_open"] = pd.to_datetime(k0.decision_open, utc=True)
    m = k0.merge(e, on=["decision_open", "family", "side"], how="left", validate="one_to_one")
    matched = m.tb_outcome.notna()
    m["label_positive"] = m.tb_outcome.eq("tp") & ~m.excluded.fillna(True).astype(bool)
    r = m.loc[matched & m.resolved & m.period.eq("dev")]
    usable = r.loc[~r.excluded.astype(bool)]
    out = dict(v2_setups=len(k0), matched_to_label_events=int(matched.sum()), dev_resolved_matched=len(r),
               label_excluded_share=float(r.excluded.astype(bool).mean()),
               b1_net_positive_given_label_tp=float((usable.loc[usable.tb_outcome.eq("tp"), "net"] > 0).mean()),
               b1_net_positive_given_label_not_tp=float((usable.loc[~usable.tb_outcome.eq("tp"), "net"] > 0).mean()),
               b1_target_given_label_tp=float(usable.loc[usable.tb_outcome.eq("tp"), "outcome"].eq("TARGET").mean()),
               b1_net_mean_label_tp=float(usable.loc[usable.tb_outcome.eq("tp"), "net"].mean()),
               b1_net_mean_label_not_tp=float(usable.loc[~usable.tb_outcome.eq("tp"), "net"].mean()),
               label_tp_rate=float(usable.tb_outcome.eq("tp").mean()), b1_net_positive_rate=float((usable.net > 0).mean()),
               excluded_rows_b1_net_mean=float(r.loc[r.excluded.astype(bool), "net"].mean()),
               note="label tb_outcome is the same-bar FRVP triple barrier (target checked before stop, no costs, timeout = negative)")
    return out


def label_audit_ote(census: pd.DataFrame, setups: pd.DataFrame) -> dict:
    k0 = census.loc[census.offset.eq(0)].merge(setups[["setup_id", "label_family", "swing_index", "confirm_index", "tb_outcome"]],
                                               on="setup_id", validate="one_to_one")
    r = k0.loc[k0.resolved & k0.period.eq("dev")]
    lag = (k0.confirm_index - k0.swing_index)
    zone = {"reversal": (2, 0), "continuation_pullback": (1, 1), "breakout": (0, 2)}
    out = {}
    for family, g in r.groupby("label_family"):
        pre, post = zone[family]
        lg = (g.confirm_index - g.swing_index)
        # Zone label bars are swing_index-pre .. swing_index+post; a bar is knowable only at or after confirm_index.
        before = np.mean([(np.arange(-pre, post + 1) + 0 < l).mean() for l in lg]) if len(lg) else None
        out[family] = dict(dev_resolved=len(g), confirm_lag_bars_median=float(lg.median()), confirm_lag_bars_p90=float(lg.quantile(0.9)),
                           share_of_label_zone_bars_before_confirmation=float(before),
                           label_tp_rate=float(g.tb_outcome.eq("tp").mean()),
                           b1_net_positive_given_label_tp=float((g.loc[g.tb_outcome.eq("tp"), "net"] > 0).mean()),
                           b1_net_positive_given_label_not_tp=float((g.loc[~g.tb_outcome.eq("tp"), "net"] > 0).mean()),
                           b1_net_mean_label_tp=float(g.loc[g.tb_outcome.eq("tp"), "net"].mean()),
                           b1_net_mean_label_not_tp=float(g.loc[~g.tb_outcome.eq("tp"), "net"].mean()))
    out["all_confirm_lag_bars_median"] = float(lag.median())
    return out


# ------------------------------------------------------------------------------------------------------------ timing
def timing(stream: str, census: pd.DataFrame, bounds) -> dict:
    out = {}
    for period, (a, b) in bounds.items():
        rows = census.loc[census.period.eq(period)]
        w = weeks(a, b)
        by_offset = {}
        for k, g in rows.groupby("offset"):
            eligible = g.loc[g.valid] if k > 0 else g
            entry = dict(knowable=bool(k >= 0), use="diagnostic_not_knowable" if k < 0 else "eligible_if_valid",
                         valid_share=float(g.valid.mean()) if k > 0 else None, eligible=payoff(eligible))
            if k > 0:
                entry["all_ignoring_validity"] = payoff(g)
            by_offset[str(k)] = entry
        # Family/side split of the same offset profile (net mean, win rate) for k in -10..5.
        split = {}
        for (family, side), g in rows.groupby(["family", "side"]):
            key = f"{family}_{'long' if side > 0 else 'short'}"
            prof = {}
            for k, h in g.groupby("offset"):
                e = h.loc[h.valid] if k > 0 else h
                r = e.loc[e.resolved]
                prof[str(k)] = dict(n=len(r), net_mean=float(r.net.mean()) if len(r) else None, gross_mean=float(r.gross.mean()) if len(r) else None,
                                    win_rate=float((r.net > 0).mean()) if len(r) else None)
            split[key] = prof
        # Hindsight best entry per setup (diagnostic upper bounds, never a policy).
        r = rows.loc[rows.resolved]
        best_all = r.groupby("setup_id").net.max()
        best_post = r.loc[(r.offset >= 0) & r.valid].groupby("setup_id").net.max()
        k0 = r.loc[r.offset.eq(0)].set_index("setup_id").net
        hindsight = dict(best_of_minus10_to_plus5_net_mean=float(best_all.mean()), best_valid_0_to_5_net_mean=float(best_post.mean()),
                         setup_bar_net_mean=float(k0.mean()),
                         share_setups_where_a_valid_later_bar_beats_setup_bar=float((best_post.reindex(k0.index) > k0).mean()),
                         use="diagnostic_upper_bound_selected_with_hindsight")
        # Fixed-delay policies: always enter D bars after the setup if it is still valid (declared, no hindsight).
        delays = {}
        for d in range(0, 6):
            g = rows.loc[rows.offset.eq(d) & rows.valid]
            delays[str(d)] = dict(ledger=trade_stats(g, w), payoff=payoff(g))
        # Candidate pool per post window K (what a model could choose from), and pre-window oracle values.
        pools = {}
        for K in (0, 1, 2, 3, 5):
            g = rows.loc[rows.offset.between(0, K) & rows.valid & rows.executable]
            pools[str(K)] = dict(eligible_candidates_per_week=float(len(g) / w), setups_with_candidates_per_week=float(g.setup_id.nunique() / w),
                                 payoff=payoff(g))
        pre = {}
        for P in (2, 5, 10):
            g = r.loc[r.offset.between(-P, 0)]
            pre[str(P)] = dict(use="diagnostic_not_knowable", best_of_window_net_mean=float(g.groupby("setup_id").net.max().mean()),
                               mean_net_pre_bars=float(g.loc[g.offset < 0, "net"].mean()))
        out[period] = dict(weeks=w, by_offset=by_offset, by_family_side=split, hindsight=hindsight, fixed_delay=delays,
                           post_window_pools=pools, pre_window_oracle=pre)
    return out


# ---------------------------------------------------------------------------------------------------------- everybar
def day_windows(stream: str, open_ns: np.ndarray) -> dict[str, np.ndarray]:
    idx = pd.DatetimeIndex(open_ns.astype("datetime64[ns]")).tz_localize("UTC")
    ny = idx.tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    if stream == "ES":
        return dict(full_globex_session=np.ones(len(idx), bool), rth_0930_1600_et=(minutes >= 570) & (minutes < 960))
    london = idx.tz_convert("Europe/London")
    lmin = london.hour * 60 + london.minute
    return dict(full_fx_day_17et_to_17et=np.ones(len(idx), bool), london_open_to_ny_close_07lon_17et=(lmin >= 420) & (minutes < 1020),
                london_ny_overlap_08_12et=(minutes >= 480) & (minutes < 720) & (lmin < 960))


def oof_scores(model_id: int) -> dict[str, pd.DataFrame]:
    model_dir = Path(glob.glob(str(ROOT / f"models/b2_v2_retrained_20261006/{model_id:02d}_*/*/oof_predictions.csv"))[0]).parent
    history = pd.read_parquet(ROOT / "artifacts/b2_v2_source_history_20261006/source_history_union.parquet", columns=["source_row_idx", "datetime"])
    history["open_ns"] = pd.to_datetime(history.datetime, utc=True).dt.as_unit("ns").astype("int64")
    out = {}
    for period, name, prefix in (("dev", "oof_predictions.csv", "oof_"), ("inspected", "test_predictions.csv", "")):
        f = pd.read_csv(model_dir / name)
        col = f"{prefix}calibrated_probability" if f"{prefix}calibrated_probability" in f else f"{prefix}raw_probability"
        f = f.loc[f[col].notna(), ["source_row_idx", col]].rename(columns={col: "score"})
        out[period] = f.merge(history[["source_row_idx", "open_ns"]], on="source_row_idx")
    return out


def everybar(stream: str, bounds) -> dict:
    eb = pd.read_parquet(CACHE / f"everybar_{stream}.parquet")
    eb["event_id"] = eb.pos.astype(str).str.zfill(7) + np.where(eb.side > 0, "L", "S")
    eb["period"] = np.where(eb.open_ns < DEV_END[stream].value, "dev", "inspected")
    windows = day_windows(stream, eb.open_ns.to_numpy())
    out = {}
    for period, (a, b) in bounds.items():
        w = weeks(a, b)
        res = {}
        for template in TEMPLATES[stream]:
            for name, mask in windows.items():
                for side in (1, -1):
                    g = eb.loc[mask & eb.period.eq(period).to_numpy() & eb.template.eq(template).to_numpy() & eb.side.eq(side).to_numpy()]
                    key = f"{template}|{name}|{'long' if side > 0 else 'short'}"
                    res[key] = dict(bars_per_week=float(len(g) / w), payoff=payoff(g), take_every_bar_ledger=trade_stats(g, w))
        out[period] = res
    if stream == "EURUSD":
        out["existing_every_bar_ote_models"] = existing_ote_models(eb, bounds)
    return out


def existing_ote_models(eb: pd.DataFrame, bounds) -> dict:
    """Models 21 (long_ote) and 25 (short_ote) were trained on every bar; score their saved predictions against the
    every-bar B1 outcomes of their side. Threshold picked on dev OOF; inspected reported, not a fresh holdout."""
    out = {}
    for model_id, side in ((21, 1), (25, -1)):
        scores = oof_scores(model_id)
        res = {}
        for template in TEMPLATES["EURUSD"]:
            sub = eb.loc[eb.template.eq(template) & eb.side.eq(side)]
            joined = {p: sub.merge(scores[p], on="open_ns", how="inner") for p in ("dev", "inspected")}
            dev = joined["dev"]
            w = {p: weeks(*bounds[p]) for p in bounds}
            # OOF covers only the CV validation folds; weeks from the scored span.
            w["dev"] = weeks(int(dev.open_ns.min()), int(dev.open_ns.max())) if len(dev) else 1.0
            grid = np.unique(np.quantile(dev.score, [0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999])) if len(dev) else []
            sweep = []
            for t in grid:
                s = trade_stats(dev.loc[dev.score >= t], w["dev"])
                sweep.append(dict(threshold=float(t), **{k: s[k] for k in ("trades_per_week", "net_per_trade", "gross_per_trade", "win_rate", "net_total", "feasible")}))
            feasible = [r for r in sweep if r["feasible"]]
            best = max(feasible, key=lambda r: (r["trades_per_week"], r["net_total"])) if feasible else None
            top = max(sweep, key=lambda r: r["gross_per_trade"] if r["gross_per_trade"] is not None else -1e9) if sweep else None
            res[template] = dict(dev_scored_bars=len(dev), dev_sweep=sweep, feasible_threshold=best,
                                 inspected_at_feasible=trade_stats(joined["inspected"].loc[joined["inspected"].score >= best["threshold"]], w["inspected"]) if best else None,
                                 best_gross_threshold_dev=top)
        out[str(model_id)] = res
    return out


# -------------------------------------------------------------------------------------------------------------- main
def main() -> None:
    report = dict(status="research_only_nothing_frozen_or_activated", min_trades_per_week=MIN_TPW, units=UNITS,
                  verify=json.loads((OUT / "verify.json").read_text()), periods={}, streams={})
    for stream in ("ES", "EURUSD"):
        census = pd.read_parquet(CACHE / f"census_{stream}.parquet")
        inputs = load(f"inputs_{stream}.pkl")
        bounds = period_bounds(stream, census)
        report["periods"][stream] = {k: [str(pd.Timestamp(a, tz='UTC')), str(pd.Timestamp(b, tz='UTC'))] for k, (a, b) in bounds.items()}
        print(f"{stream}: baseline", flush=True)
        s = dict(notes=inputs["notes"], baseline=baseline(stream, census, bounds))
        print(f"{stream}: label audit", flush=True)
        s["label_audit"] = label_audit_frvp(census) if stream == "ES" else label_audit_ote(census, inputs["setups"])
        print(f"{stream}: timing", flush=True)
        s["timing"] = timing(stream, census, bounds)
        print(f"{stream}: every bar", flush=True)
        s["everybar"] = everybar(stream, bounds)
        report["streams"][stream] = s
    dump(REPORT, report)
    print(f"REPORT written: {REPORT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
