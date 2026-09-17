"""Counterfactual Abstain Attribution (CAA) over immutable saved research artifacts.

Run from the repository root with the repo virtualenv:

    ote_venv\\Scripts\\python.exe research\\papers\\anti_filter_effect\\scripts\\analyze_attribution.py

The script only READS saved model_testing report artifacts and WRITES to
research/papers/anti_filter_effect/artifacts/. It does not train, does not touch
canonical labels/policies, and does not call any execution path.

Four analyses are produced:

1. trade_level_attribution   Partition every saved emitted-trade tape into accepted (A)
                             and rejected (R) populations under several abstention rules
                             and measure the economics of both populations.
2. walk_forward_pair_census  For every saved walk-forward backtest that evaluated the
                             four canonical policy variants per fold, compare each base
                             policy against its "+abstain" counterpart on the TEST split.
                             The abstain set is nested inside the base set, so the rejected
                             population's economics are exact differences.
3. static_pair_census        Same comparison for the static (single held-out test split)
                             threshold-policy searches.
4. curated_run_pairs         Baseline-vs-filtered comparisons between separately saved
                             walk-forward runs of the same frozen model, matched on the exact
                             (entry_datetime, source_row_idx) trade key.

Quantile cutoffs in analysis 1 come in three flavours. `retro_fold_quantile` reproduces the
2026-08-30 retrospective screen (cutoff computed on the same test-fold trades it filters, so
it is NOT causal). `causal_train_quantile` takes the cutoff from that fold's saved train-window
trades, which is knowable at fold start. `rolling_prior_quantile` takes the cutoff from the
previous emitted test trades only, mimicking the live rolling-window semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "artifacts"
REPORTS = ROOT / "model_testing" / "reports"

# --------------------------------------------------------------------------------------
# Analysis 1 sources: saved emitted-trade tapes (test split, walk-forward, selected policy)
# --------------------------------------------------------------------------------------
TAPES = {
    "frvp_short_meta_xgb_v1": "frvp_backtests/frvp_es_primary_refresh_20260701",
    "frvp_long_meta_xgb_v1": "frvp_backtests/frvp_es_primary_refresh_20260701",
    "ict_long_continuation_xgb_v1": "ict_backtests/ict_es_primary_bootstrap_20260726_full",
    "ict_short_continuation_xgb_v1": "ict_backtests/ict_es_primary_bootstrap_20260726_full",
    "frvp_long_continuation_setup2_xgb_v1": "frvp_backtests/frvp_long_setup_fullspan_20260829",
    "long_ote_tcn_v2_candidate": "ote_policy_backtests/v1_v2_tcn_focus",
}
QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
ABSOLUTE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
ROLLING_WINDOW = 100
ROLLING_MIN_HISTORY = 20

# --------------------------------------------------------------------------------------
# Analysis 2 sources: walk-forward roots whose per-model policy_evaluation.csv holds the
# four policy variants per fold.
# --------------------------------------------------------------------------------------
WALK_FORWARD_ROOTS = {
    "frvp_backtests/frvp_es_primary_refresh_20260701": "FRVP",
    "frvp_backtests/frvp_es_primary_current": "FRVP",
    "frvp_backtests/frvp_long_setup_fullspan_20260829": "FRVP",
    "ict_backtests/ict_es_primary_bootstrap_20260726_full": "ICT",
    "ict_backtests/ict_short_setup_family_20260811_audit": "ICT",
    "ote_policy_backtests/v1_v2_tcn_focus": "OTE",
    "ote_policy_backtests/multifamily_live_v1": "OTE",
    "ote_policy_backtests/champion_models_20260508_v2_min5": "OTE",
}
STATIC_ROOTS = {
    "frvp_threshold_policies/frvp_es_primary_refresh_20260701": "FRVP",
    "ict_threshold_policies/ict_es_primary_bootstrap_20260726_full": "ICT",
    "ote_threshold_policies/v1_v2_tcn_focus": "OTE",
    "ote_threshold_policies/multifamily_live_v1": "OTE",
    "ote_threshold_policies/experiment_a_breakout_sustained_pilot_20260621": "OTE",
}
PAIRS = [("global_threshold", "global_threshold_plus_abstain"), ("regime_threshold", "regime_threshold_plus_abstain")]

# --------------------------------------------------------------------------------------
# Analysis 4 sources: (label, family, baseline run dir, filtered run dir, description)
# --------------------------------------------------------------------------------------
RG = "frvp_regime_gated_deployment/frvp_regime_gated_deployment_20260721/backtest"
PQ = "probability_quantile_sweeps/frvp_ict_weak_discrimination_20260830/frvp_short_meta_xgb_v1"
CS = "ict_backtests/ict_probability_quantile_candidate_sweep_20260830"
CURATED = [
    ("frvp_long_cont_refresh_to_v3_prune", "FRVP", "frvp_backtests/frvp_es_primary_refresh_20260701/frvp_long_continuation_xgb_v1", "frvp_backtests/frvp_long_continuation_gatefix_v3_20260715_accountdd/frvp_long_continuation_xgb_v1", "Regime/session pair prune v3 (E02)"),
    ("frvp_long_meta_refresh_to_v3_prune", "FRVP", "frvp_backtests/frvp_es_primary_refresh_20260701/frvp_long_meta_xgb_v1", "frvp_backtests/frvp_long_meta_gatefix_v3_20260715_accountdd/frvp_long_meta_xgb_v1", "Regime/session pair prune v3 (E03)"),
    ("frvp_long_rev_refresh_to_v2_prune", "FRVP", "frvp_backtests/frvp_es_primary_refresh_20260701/frvp_long_reversal_xgb_v1", "frvp_backtests/frvp_long_reversal_gatefix_v2_20260703/frvp_long_reversal_xgb_v1", "Regime/session pair prune v2 (E04)"),
    ("frvp_long_rev_recent2y_sdh_overlap_prune", "FRVP", f"{RG}/long_reversal_recent2y_baseline/frvp_long_reversal_xgb_v1", f"{RG}/long_reversal_recent2y_sdh_overlap_prune_v1/frvp_long_reversal_xgb_v1", "Sparse-pocket prune that produced the accepted contract (E11)"),
    ("frvp_long_rev_recent2y_q10_floor", "FRVP", f"{RG}/long_reversal_recent2y_baseline/frvp_long_reversal_xgb_v1", f"{RG}/long_reversal_recent2y_q10_v1/frvp_long_reversal_xgb_v1", "10th-percentile probability floor (E11)"),
    ("frvp_long_rev_fullspan_sdm_asia_prune", "FRVP", f"{RG}/long_reversal_fullspan_baseline/frvp_long_reversal_xgb_v1", f"{RG}/long_reversal_fullspan_sdm_asia_prune_v1/frvp_long_reversal_xgb_v1", "Full-span strong_down_medium/asia prune (E11)"),
    ("frvp_setup2_medium_session_prune", "FRVP", "frvp_backtests/frvp_long_setup_fullspan_20260829/frvp_long_continuation_setup2_xgb_v1", "frvp_backtests/frvp_long_setup2_medium_session_prune_20260829/frvp_long_continuation_setup2_xgb_v1", "Setup-2 medium/session prune (E15)"),
    ("frvp_short_meta_wf_q10", "FRVP", f"{PQ}/baseline/frvp_short_meta_xgb_v1", f"{PQ}/q10/frvp_short_meta_xgb_v1", "Walk-forward probability quantile q10 (2026-08-30)"),
    ("frvp_short_meta_wf_q20", "FRVP", f"{PQ}/baseline/frvp_short_meta_xgb_v1", f"{PQ}/q20/frvp_short_meta_xgb_v1", "Walk-forward probability quantile q20 (2026-08-30)"),
    ("frvp_short_meta_wf_q30", "FRVP", f"{PQ}/baseline/frvp_short_meta_xgb_v1", f"{PQ}/q30/frvp_short_meta_xgb_v1", "Walk-forward probability quantile q30 (2026-08-30)"),
    ("ict_long_cont_wf_q40", "ICT", "ict_backtests/ict_es_primary_bootstrap_20260726_full/ict_long_continuation_xgb_v1", f"{CS}/ict_long_continuation_q40/ict_long_continuation_xgb_v1", "Walk-forward probability quantile q40 (2026-08-30)"),
    ("ict_long_cont_wf_q50", "ICT", "ict_backtests/ict_es_primary_bootstrap_20260726_full/ict_long_continuation_xgb_v1", f"{CS}/ict_long_continuation_q50/ict_long_continuation_xgb_v1", "Walk-forward probability quantile q50 (2026-08-30)"),
    ("ict_long_cont_wf_q60", "ICT", "ict_backtests/ict_es_primary_bootstrap_20260726_full/ict_long_continuation_xgb_v1", f"{CS}/ict_long_continuation_q60/ict_long_continuation_xgb_v1", "Walk-forward probability quantile q60 (2026-08-30)"),
    ("ict_short_cont_wf_q50", "ICT", "ict_backtests/ict_es_primary_bootstrap_20260726_full/ict_short_continuation_xgb_v1", f"{CS}/ict_short_continuation_q50/ict_short_continuation_xgb_v1", "Walk-forward probability quantile q50 (2026-08-30)"),
    ("ict_short_cont_wf_q60", "ICT", "ict_backtests/ict_es_primary_bootstrap_20260726_full/ict_short_continuation_xgb_v1", f"{CS}/ict_short_continuation_q60/ict_short_continuation_xgb_v1", "Walk-forward probability quantile q60 (2026-08-30)"),
    ("ict_long_rev_min3_to_min2_cadence", "ICT", "ict_backtests/ict_es_primary_20260718T002437Z/ict_long_reversal_xgb_v1", "ict_backtests/ict_es_primary_20260719_min2/ict_long_reversal_xgb_v1", "Looser cadence floor let +abstain variant be selected in 147 trades (2026-07-19 audit)"),
    ("ote_short_meta_regime_prune_v1", "OTE", "ote_policy_backtests/short_ote_meta_tcn_repair_20260511_v2/short_ote_meta_tcn_champion", "ote_policy_backtests/short_ote_meta_tcn_repair_20260511_v2_regime_prune_v1/short_ote_meta_tcn_champion", "Regime prune v1 (accepted live champion policy)"),
    ("ote_short_meta_prune_v1_plus_q20", "OTE", "ote_policy_backtests/short_ote_meta_tcn_repair_20260511_v2_regime_prune_v1/short_ote_meta_tcn_champion", "ote_policy_backtests/short_ote_meta_tcn_repair_20260511_v2_regime_prune_q20_v1/short_ote_meta_tcn_champion", "Probability quantile q20 stacked on accepted prune"),
    ("ote_long_breakout_v2_q20", "OTE", "ote_policy_backtests/long_breakout_tcn_repair_20260513_v2/long_breakout_tcn_champion", "ote_policy_backtests/long_breakout_tcn_repair_20260513_v2_regime_prune_v1_q20/long_breakout_tcn_champion", "Regime prune v1 plus probability quantile q20"),
    ("ote_long_breakout_v2_prune_v3", "OTE", "ote_policy_backtests/long_breakout_tcn_repair_20260513_v2/long_breakout_tcn_champion", "ote_policy_backtests/long_breakout_tcn_repair_20260513_v2_regime_prune_v3/long_breakout_tcn_champion", "Regime prune v3"),
]

MANIFEST: dict[str, dict] = {}
KEY = ["entry_datetime", "source_row_idx"]


def record(path: Path) -> None:
    rel = path.relative_to(ROOT).as_posix()
    if rel not in MANIFEST:
        MANIFEST[rel] = {"path": rel, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(path: Path) -> dict:
    record(path)
    return json.loads(path.read_text(encoding="utf-8"))


def unit_label(run_dir: Path) -> str:
    summary = read_json(run_dir / "summary.json")
    return str(summary.get("performance_unit_label") or "pips")


def read_tape(path: Path) -> pd.DataFrame:
    record(path)
    d = pd.read_csv(path)
    for col in ["gross_pnl", "total_cost", "net_pnl"]:
        if f"{col}_units" not in d.columns:
            d[f"{col}_units"] = d[f"{col}_pips"]
    for c in ["policy_probability", "target", "gross_pnl_units", "total_cost_units", "net_pnl_units", "source_row_idx"]:
        d[c] = pd.to_numeric(d[c], errors="raise")
    d["entry_datetime"] = pd.to_datetime(d["entry_datetime"], utc=True, errors="raise")
    assert not d[["policy_probability", "target", "net_pnl_units"]].isna().any().any(), path
    assert d["target"].isin([0, 1]).all(), path
    np.testing.assert_allclose(d["gross_pnl_units"] - d["total_cost_units"], d["net_pnl_units"], atol=1e-6)
    d["calendar_year"] = d["entry_datetime"].dt.year
    d["calendar_quarter"] = d["entry_datetime"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    d["volatility_state"] = d["composite_regime"].astype(str).str.rsplit("_", n=1).str[-1]
    d["absolute_confidence_band"] = pd.cut(d["policy_probability"], [0, .5, .6, .7, .8, .9, 1.0], include_lowest=True).astype(str)
    d["label_outcome"] = d["target"].map({0: "label_negative", 1: "label_positive"})
    d["net_outcome"] = np.where(d["net_pnl_units"] > 0, "win", np.where(d["net_pnl_units"] < 0, "loss", "flat"))
    return d.sort_values(["entry_datetime", "source_row_idx"], kind="stable").reset_index(drop=True)


def stats(d: pd.DataFrame, months: pd.PeriodIndex) -> dict:
    n = int(len(d))
    p = d["net_pnl_units"].to_numpy(dtype=float)
    net = float(p.sum())
    equity = np.r_[0.0, p.cumsum()]
    dd = float((np.maximum.accumulate(equity) - equity).max())
    monthly = d.groupby(d["entry_datetime"].dt.tz_localize(None).dt.to_period("M"))["net_pnl_units"].sum().reindex(months, fill_value=0.0)
    sd = float(monthly.std(ddof=1)) if len(months) > 1 else 0.0
    winners = np.sort(p[p > 0])[::-1]
    gross_wins = float(winners.sum())
    abs_total = float(np.abs(p).sum())
    out = {
        "n": n,
        "positive_labels": int(d["target"].sum()),
        "gross_pnl": float(d["gross_pnl_units"].sum()),
        "cost": float(d["total_cost_units"].sum()),
        "net_pnl": net,
        "expectancy": net / n if n else None,
        "label_precision": float(d["target"].mean()) if n else None,
        "win_rate": float((p > 0).mean()) if n else None,
        "profit_factor": float(gross_wins / -p[p < 0].sum()) if n and (p < 0).any() else None,
        "max_drawdown_units": dd,
        "monthly_sharpe_proxy": float(math.sqrt(12) * monthly.mean() / sd) if sd > 0 else None,
        "largest_winner_share_net": float(winners[0] / net) if len(winners) and net > 0 else None,
        "pnl_abs_hhi": float(np.square(p / abs_total).sum()) if abs_total else None,
        "folds_with_trades": int(d["fold_id"].nunique()) if n else 0,
        "first_entry": str(d["entry_datetime"].min()) if n else None,
        "last_entry": str(d["entry_datetime"].max()) if n else None,
    }
    for label, count in [("top1", 1), ("top5", 5), ("top10pct", max(1, math.ceil(n * 0.1)))]:
        total = float(winners[:count].sum())
        out[f"{label}_winner_pnl"] = total
        out[f"{label}_share_net"] = total / net if net > 0 else None
        out[f"{label}_share_gross_wins"] = total / gross_wins if gross_wins > 0 else None
    return out


def classify(base: dict, acc: dict) -> dict:
    """Anti-filter classification against the baseline population."""
    if not acc["n"] or acc["label_precision"] is None:
        return {"precision_up": None, "net_down": None, "expectancy_down": None, "anti_filter_total": None, "anti_filter_expectancy": None}
    precision_up = acc["label_precision"] > base["label_precision"] + 1e-12
    net_down = acc["net_pnl"] < base["net_pnl"] - 1e-8
    expectancy_down = acc["expectancy"] < base["expectancy"] - 1e-12
    return {"precision_up": bool(precision_up), "net_down": bool(net_down), "expectancy_down": bool(expectancy_down),
            "anti_filter_total": bool(precision_up and net_down), "anti_filter_expectancy": bool(precision_up and expectancy_down)}


# --------------------------------------------------------------------------------------
# Analysis 1
# --------------------------------------------------------------------------------------
def cutoff_retro(b: pd.DataFrame, q: float) -> pd.Series:
    return b.groupby("fold_id")["policy_probability"].transform(lambda s: s.quantile(q))


def cutoff_causal_train(b: pd.DataFrame, train: pd.DataFrame, q: float) -> pd.Series:
    per_fold = train.groupby("fold_id")["policy_probability"].quantile(q)
    return b["fold_id"].map(per_fold)


def cutoff_rolling(b: pd.DataFrame, q: float) -> pd.Series:
    probs = b["policy_probability"].to_numpy(dtype=float)
    out = np.full(len(b), -np.inf)
    for i in range(len(b)):
        history = probs[max(0, i - ROLLING_WINDOW):i]
        if len(history) >= ROLLING_MIN_HISTORY:
            out[i] = np.quantile(history, q)
    return pd.Series(out, index=b.index)


def trade_level_attribution() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, groups, membership, top, integrity = [], [], [], [], []
    for model, run in TAPES.items():
        run_dir = REPORTS / run / model
        unit = unit_label(run_dir)
        b = read_tape(run_dir / "selected_test_trades.csv")
        train = read_tape(run_dir / "selected_train_trades.csv")
        read_json(run_dir.parent / "run_summary.json")
        months = pd.period_range(b["entry_datetime"].min().tz_localize(None).to_period("M"), b["entry_datetime"].max().tz_localize(None).to_period("M"), freq="M")
        baseline = stats(b, months)
        policies: list[tuple[str, str, float, pd.Series]] = []
        for q in QUANTILES:
            policies.append((f"q{int(q*100):02d}", "retro_fold_quantile", q, cutoff_retro(b, q)))
            policies.append((f"q{int(q*100):02d}", "causal_train_quantile", q, cutoff_causal_train(b, train, q)))
            policies.append((f"q{int(q*100):02d}", "rolling_prior_quantile", q, cutoff_rolling(b, q)))
        for t in ABSOLUTE_THRESHOLDS:
            policies.append((f"p{int(round(t*100)):02d}", "absolute_probability", t, pd.Series(t, index=b.index)))
        for policy, kind, threshold, cutoff in policies:
            mask = b["policy_probability"] > cutoff
            a, r = b.loc[mask], b.loc[~mask]
            assert len(b) == len(a) + len(r)
            ma, mr = stats(a, months), stats(r, months)
            for metric in ["net_pnl", "gross_pnl", "cost", "positive_labels"]:
                assert math.isclose(baseline[metric], ma[metric] + mr[metric], abs_tol=1e-6), (model, policy, kind, metric)
            row = {"model_id": model, "family": model.split("_")[0].upper() if not model.endswith("candidate") else "OTE",
                   "policy": policy, "policy_kind": kind, "threshold": threshold, "coverage": len(a) / len(b), "unit": unit,
                   "source_path": (run_dir / "selected_test_trades.csv").relative_to(ROOT).as_posix()}
            for suffix, value in [("baseline", baseline), ("accepted", ma), ("rejected", mr)]:
                row.update({f"{k}_{suffix}": v for k, v in value.items()})
            for metric in ["net_pnl", "expectancy", "label_precision", "win_rate", "max_drawdown_units", "monthly_sharpe_proxy", "profit_factor"]:
                row[f"delta_{metric}"] = (ma[metric] - baseline[metric]) if (ma[metric] is not None and baseline[metric] is not None) else None
            row.update(classify(baseline, ma))
            row["rejected_expectancy_minus_accepted"] = (mr["expectancy"] - ma["expectancy"]) if (mr["expectancy"] is not None and ma["expectancy"] is not None) else None
            rows.append(row)
            if kind in {"retro_fold_quantile", "causal_train_quantile"} or (kind == "absolute_probability" and policy in {"p60", "p70", "p80"}):
                for population, d in [("accepted", a), ("rejected", r)]:
                    for dim in ["direction", "session_regime", "composite_regime", "stress_regime", "volatility_state", "calendar_year", "fold_id", "absolute_confidence_band", "label_outcome", "net_outcome"]:
                        for bucket, g in d.groupby(dim, observed=True, dropna=False):
                            groups.append({"model_id": model, "policy": policy, "policy_kind": kind, "population": population, "dimension": dim, "bucket": str(bucket), **stats(g, months)})
                    for rank, (_, trade) in enumerate(d.nlargest(5, "net_pnl_units").iterrows(), 1):
                        top.append({"model_id": model, "policy": policy, "policy_kind": kind, "population": population, "rank": rank,
                                    **trade[["entry_datetime", "source_row_idx", "policy_probability", "target", "net_pnl_units", "session_regime", "composite_regime"]].to_dict()})
                m = b[["fold_id", "entry_datetime", "source_row_idx", "target", "policy_probability", "net_pnl_units", "session_regime", "composite_regime"]].copy()
                m.insert(0, "policy_kind", kind)
                m.insert(0, "policy", policy)
                m.insert(0, "model_id", model)
                m["population"] = np.where(mask, "accepted", "rejected")
                m["cutoff"] = cutoff.to_numpy()
                membership.append(m)
        saved = read_json(run_dir / "summary.json")["overall_test_metrics"]
        integrity.append({"model_id": model, "saved_trade_count": saved["trade_count"], "observed_trade_count": len(b),
                          "saved_net": saved["total_net_pnl_pips"], "observed_net": baseline["net_pnl"],
                          "saved_period_start": saved["period_start"], "saved_period_end": saved["period_end"],
                          "duplicate_entry_keys": int(b.duplicated(KEY).sum()), "train_tape_rows": len(train)})
        np.testing.assert_allclose(saved["total_net_pnl_pips"], baseline["net_pnl"], atol=1e-6)
        assert saved["trade_count"] == len(b)
    return pd.DataFrame(rows), pd.DataFrame(groups), pd.concat(membership, ignore_index=True), pd.DataFrame(top), pd.DataFrame(integrity)


def verify_against_historical_screen(summary: pd.DataFrame) -> int:
    path = REPORTS / "probability_quantile_sweeps/frvp_ict_weak_discrimination_20260830/trade_level_quantile_screen.csv"
    record(path)
    old = pd.read_csv(path)
    screen = summary[(summary["policy_kind"] == "retro_fold_quantile") & summary["model_id"].isin(old["model_id"].unique())]
    checked = 0
    for _, row in screen.iterrows():
        oldrow = old[(old["model_id"] == row["model_id"]) & (old["label"] == row["policy"])].iloc[0]
        np.testing.assert_allclose(row["n_accepted"], oldrow["trades"])
        np.testing.assert_allclose(row["net_pnl_accepted"], oldrow["net_ticks"], atol=1e-6)
        np.testing.assert_allclose(row["expectancy_accepted"], oldrow["expectancy_ticks"], atol=1e-9)
        checked += 1
    return checked


# --------------------------------------------------------------------------------------
# Analysis 2 and 3: nested policy-pair censuses
# --------------------------------------------------------------------------------------
def pair_rows(d: pd.DataFrame, base: str, filt: str, context: dict) -> dict | None:
    b, f = d[d["policy_name"] == base], d[d["policy_name"] == filt]
    if b.empty or f.empty or len(b) != len(f):
        return None
    bn, fn = int(b["emitted_signals"].sum()), int(f["emitted_signals"].sum())
    if bn == 0 or fn == 0:
        return None
    bm, fm = int(b["matched_events"].sum()), int(f["matched_events"].sum())
    bnet, fnet = float(b["net_pnl_pips"].sum()), float(f["net_pnl_pips"].sum())
    bgross, fgross = float(b["gross_pnl_pips"].sum()), float(f["gross_pnl_pips"].sum())
    bp, fp = bm / bn, fm / fn
    rn, rnet = bn - fn, bnet - fnet
    row = {**context, "baseline_policy": base, "filtered_policy": filt, "n_baseline": bn, "n_filtered": fn, "n_rejected": rn, "coverage": fn / bn,
           "event_precision_baseline": bp, "event_precision_filtered": fp, "delta_precision": fp - bp,
           "gross_baseline": bgross, "gross_filtered": fgross, "net_baseline": bnet, "net_filtered": fnet, "net_rejected": rnet, "delta_net": fnet - bnet,
           "expectancy_baseline": bnet / bn, "expectancy_filtered": fnet / fn, "expectancy_rejected": (rnet / rn) if rn else None,
           "delta_expectancy": fnet / fn - bnet / bn,
           "precision_up": bool(fp > bp), "net_down": bool(fnet < bnet), "expectancy_down": bool(fnet / fn < bnet / bn),
           "anti_filter_total": bool(fp > bp and fnet < bnet), "anti_filter_expectancy": bool(fp > bp and fnet / fn < bnet / bn),
           "rejected_expectancy_exceeds_accepted": bool(rn and rnet / rn > fnet / fn)}
    if "fold_id" in d.columns:
        merged = b.set_index("fold_id")[["emitted_signals", "matched_events", "net_pnl_pips"]].join(f.set_index("fold_id")[["emitted_signals", "matched_events", "net_pnl_pips"]], lsuffix="_b", rsuffix="_f")
        merged = merged[(merged["emitted_signals_b"] > 0) & (merged["emitted_signals_f"] > 0)]
        pb, pf = merged["matched_events_b"] / merged["emitted_signals_b"], merged["matched_events_f"] / merged["emitted_signals_f"]
        row.update({"folds_compared": int(len(merged)), "folds_precision_up": int((pf > pb).sum()), "folds_net_down": int((merged["net_pnl_pips_f"] < merged["net_pnl_pips_b"]).sum()),
                    "folds_anti_filter_total": int(((pf > pb) & (merged["net_pnl_pips_f"] < merged["net_pnl_pips_b"])).sum()),
                    "folds_precision_up_net_up": int(((pf > pb) & (merged["net_pnl_pips_f"] > merged["net_pnl_pips_b"])).sum())})
    return row


def walk_forward_pair_census() -> pd.DataFrame:
    rows = []
    for root, family in WALK_FORWARD_ROOTS.items():
        for path in sorted((REPORTS / root).glob("*/policy_evaluation.csv")):
            record(path)
            d = pd.read_csv(path)
            if "fold_id" not in d.columns:
                continue
            d = d[d["dataset_split"] == "test"]
            unit = unit_label(path.parent)
            saved = read_json(path.parent / "summary.json")
            for base, filt in PAIRS:
                row = pair_rows(d, base, filt, {"family": family, "run_root": root, "model_id": path.parent.name, "unit": unit,
                                                "fold_count": int(d["fold_id"].nunique()), "evaluation": "walk_forward_per_fold_test",
                                                "selected_policy_counts": json.dumps(saved.get("selected_policy_counts", {})),
                                                "accepted_gate": saved.get("paper_trading_gate", {}).get("accepted"),
                                                "source_path": path.relative_to(ROOT).as_posix()})
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def static_pair_census() -> pd.DataFrame:
    rows = []
    for root, family in STATIC_ROOTS.items():
        path = REPORTS / root / "policy_evaluation.csv"
        record(path)
        d = pd.read_csv(path)
        d = d[d["dataset_split"] == "test"]
        for model, dm in d.groupby("model_id"):
            unit = str(dm["unit_label"].iloc[0]) if "unit_label" in dm.columns else "pips"
            for base, filt in PAIRS:
                row = pair_rows(dm, base, filt, {"family": family, "run_root": root, "model_id": model, "unit": unit, "fold_count": 1,
                                                 "evaluation": "static_held_out_test_split", "source_path": path.relative_to(ROOT).as_posix()})
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Analysis 4: curated run pairs with exact trade-key matching
# --------------------------------------------------------------------------------------
def curated_run_pairs() -> pd.DataFrame:
    rows = []
    for label, family, base_rel, filt_rel, note in CURATED:
        bdir, fdir = REPORTS / base_rel, REPORTS / filt_rel
        b, f = read_tape(bdir / "selected_test_trades.csv"), read_tape(fdir / "selected_test_trades.csv")
        bs, fs = read_json(bdir / "summary.json"), read_json(fdir / "summary.json")
        unit = unit_label(bdir)
        assert bs["overall_test_metrics"]["trade_count"] == len(b) and fs["overall_test_metrics"]["trade_count"] == len(f)
        months = pd.period_range(min(b["entry_datetime"].min(), f["entry_datetime"].min()).tz_localize(None).to_period("M"),
                                 max(b["entry_datetime"].max(), f["entry_datetime"].max()).tz_localize(None).to_period("M"), freq="M")
        bkeys, fkeys = set(map(tuple, b[KEY].to_numpy())), set(map(tuple, f[KEY].to_numpy()))
        shared = bkeys & fkeys
        acc = b[b[KEY].apply(tuple, axis=1).isin(shared)]
        rej = b[~b[KEY].apply(tuple, axis=1).isin(shared)]
        added = f[~f[KEY].apply(tuple, axis=1).isin(shared)]
        sb, sf, sa, sr, sadd = stats(b, months), stats(f, months), stats(acc, months), stats(rej, months), stats(added, months)
        bm, fm = bs["overall_test_metrics"], fs["overall_test_metrics"]
        row = {"label": label, "family": family, "note": note, "unit": unit, "baseline_run": base_rel, "filtered_run": filt_rel,
               "n_baseline": len(b), "n_filtered": len(f), "n_shared": len(shared), "n_rejected": len(rej), "n_added_by_filter": len(added),
               "nested": bool(len(added) == 0), "coverage": len(f) / len(b),
               "net_baseline": sb["net_pnl"], "net_filtered": sf["net_pnl"], "delta_net": sf["net_pnl"] - sb["net_pnl"],
               "net_rejected": sr["net_pnl"], "net_added": sadd["net_pnl"],
               "expectancy_baseline": sb["expectancy"], "expectancy_filtered": sf["expectancy"], "expectancy_rejected": sr["expectancy"],
               "label_precision_baseline": sb["label_precision"], "label_precision_filtered": sf["label_precision"], "label_precision_rejected": sr["label_precision"],
               "win_rate_baseline": sb["win_rate"], "win_rate_filtered": sf["win_rate"],
               "saved_sharpe_baseline": bm.get("monthly_sharpe"), "saved_sharpe_filtered": fm.get("monthly_sharpe"),
               "saved_dsr_baseline": bm.get("approx_deflated_sharpe"), "saved_dsr_filtered": fm.get("approx_deflated_sharpe"),
               "saved_max_dd_pct_baseline": bm.get("max_drawdown_pct"), "saved_max_dd_pct_filtered": fm.get("max_drawdown_pct"),
               "saved_wfe_baseline": bs["walk_forward_efficiency"]["overall_wfe"], "saved_wfe_filtered": fs["walk_forward_efficiency"]["overall_wfe"],
               "saved_largest_trade_share_baseline": bm.get("largest_single_trade_share_of_total_pnl"), "saved_largest_trade_share_filtered": fm.get("largest_single_trade_share_of_total_pnl"),
               "saved_profitable_quarter_share_baseline": bm.get("profitable_quarter_share"), "saved_profitable_quarter_share_filtered": fm.get("profitable_quarter_share"),
               "saved_gate_baseline": bs.get("paper_trading_gate", {}).get("accepted"), "saved_gate_filtered": fs.get("paper_trading_gate", {}).get("accepted"),
               "top5_share_net_baseline": sb["top5_share_net"], "top5_share_net_filtered": sf["top5_share_net"],
               "max_drawdown_units_baseline": sb["max_drawdown_units"], "max_drawdown_units_filtered": sf["max_drawdown_units"]}
        row.update(classify(sb, sf))
        row["sharpe_up"] = (fm.get("monthly_sharpe") is not None and bm.get("monthly_sharpe") is not None and fm["monthly_sharpe"] > bm["monthly_sharpe"])
        row["additive_net"] = bool(sf["net_pnl"] > sb["net_pnl"])
        rows.append(row)
    return pd.DataFrame(rows)


def parity_evidence() -> dict:
    frvp = read_json(ROOT / "ote_live/runtime_manifests/frvp_es_paper_signal_20260816/paper_signal_validation_summary.json")
    frvp_bundle = read_json(REPORTS / "frvp_paper_signal_bundles/frvp_es_paper_signal_20260816/run_summary.json")
    ict = read_json(ROOT / "ote_live/runtime_manifests/ict_es_paper_signal_20260813/paper_signal_validation_summary.json")
    cont = next(m for m in frvp_bundle["model_outputs"] if m["model_id"] == "frvp_long_continuation_xgb_v1")
    rev = next(m for m in frvp_bundle["model_outputs"] if m["model_id"] == "frvp_long_reversal_xgb_v1")
    return {
        "frvp_feature_parity": frvp["historical_feature_parity"],
        "frvp_prediction_parity": frvp["artifact_prediction_parity"],
        "frvp_active_policy_parity": frvp["active_policy_parity"],
        "frvp_continuation_adaptive_walk_forward": {k: cont["overall_test_metrics"].get(k) for k in ["trade_count", "total_net_pnl_pips", "monthly_sharpe", "approx_deflated_sharpe", "max_drawdown_pct", "profitable_quarter_share", "largest_single_trade_share_of_total_pnl"]} | {"overall_wfe": cont["walk_forward_efficiency"]["overall_wfe"], "all_gates": cont.get("paper_trading_gate")},
        "frvp_continuation_fixed_live_policy": cont["fixed_live_policy_sensitivity"],
        "frvp_reversal_adaptive_walk_forward": {k: rev["overall_test_metrics"].get(k) for k in ["trade_count", "total_net_pnl_pips", "monthly_sharpe", "approx_deflated_sharpe", "max_drawdown_pct", "profitable_quarter_share", "largest_single_trade_share_of_total_pnl"]} | {"overall_wfe": rev["walk_forward_efficiency"]["overall_wfe"], "all_gates": rev.get("paper_trading_gate")},
        "frvp_reversal_fixed_live_policy": rev.get("fixed_live_policy_sensitivity"),
        "ict_prediction_parity": ict["artifact_prediction_parity"],
        "ict_feature_parity": {k: ict["historical_feature_parity"][k] for k in ["status", "evaluation_bars", "compared_cells", "matched_cells", "mismatched_cells"]},
        "ict_shadow_reversal_coordinate_risk": ict["shadow_reversal_coordinate_risk"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, groups, membership, top, integrity = trade_level_attribution()
    summary.to_csv(OUT / "attribution_summary.csv", index=False)
    groups.to_csv(OUT / "attribution_breakdowns.csv", index=False)
    membership.to_csv(OUT / "trade_membership.csv", index=False)
    top.to_csv(OUT / "top_contributors.csv", index=False)
    integrity.to_csv(OUT / "source_integrity_checks.csv", index=False)
    screen_checks = verify_against_historical_screen(summary)

    wf = walk_forward_pair_census()
    wf.to_csv(OUT / "walk_forward_pair_census.csv", index=False)
    st = static_pair_census()
    st.to_csv(OUT / "static_pair_census.csv", index=False)
    cur = curated_run_pairs()
    cur.to_csv(OUT / "curated_run_pairs.csv", index=False)
    parity = parity_evidence()
    (OUT / "parity_evidence.json").write_text(json.dumps(parity, indent=2, default=str) + "\n", encoding="utf-8")

    def census(frame: pd.DataFrame) -> dict:
        return {"comparisons": int(len(frame)), "precision_up": int(frame["precision_up"].sum()), "net_down": int(frame["net_down"].sum()),
                "precision_up_net_down": int(frame["anti_filter_total"].sum()), "precision_up_expectancy_down": int(frame["anti_filter_expectancy"].sum()),
                "precision_up_net_up": int((frame["precision_up"] & ~frame["net_down"]).sum())}

    retro = summary[summary["policy_kind"] == "retro_fold_quantile"]
    causal = summary[summary["policy_kind"] == "causal_train_quantile"]
    rolling = summary[summary["policy_kind"] == "rolling_prior_quantile"]
    absolute = summary[summary["policy_kind"] == "absolute_probability"]
    checks = {
        "historical_screen_rows_reproduced": screen_checks,
        "all_partition_identities_pass": True,
        "trade_level": {"retro_fold_quantile": census(retro.dropna(subset=["precision_up"])), "causal_train_quantile": census(causal.dropna(subset=["precision_up"])),
                        "rolling_prior_quantile": census(rolling.dropna(subset=["precision_up"])), "absolute_probability": census(absolute.dropna(subset=["precision_up"]))},
        "walk_forward_pairs": census(wf), "walk_forward_pairs_by_family": {fam: census(g) for fam, g in wf.groupby("family")},
        "static_pairs": census(st), "curated_run_pairs": {"comparisons": int(len(cur)), "nested": int(cur["nested"].sum()),
                                                          "precision_up_net_down": int(cur["anti_filter_total"].sum()), "net_up": int(cur["additive_net"].sum()),
                                                          "net_down_but_sharpe_up": int((~cur["additive_net"] & cur["sharpe_up"]).sum())},
        "interpretation": "Descriptive retrospective census over shared saved artifacts; comparisons share trades and folds and are not independent trials.",
    }
    (OUT / "analysis_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(MANIFEST.values()).sort_values("path").to_csv(OUT / "attribution_source_manifest.csv", index=False)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
