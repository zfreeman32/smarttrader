"""Build the figures and LaTeX table fragments used by paper.tex from the CAA artifacts.

Run from the repository root AFTER analyze_attribution.py:

    ote_venv\\Scripts\\python.exe research\\papers\\anti_filter_effect\\scripts\\build_paper_assets.py

Writes research/papers/anti_filter_effect/artifacts/figures/*.pdf and
research/papers/anti_filter_effect/artifacts/tables/*.tex. Read-only elsewhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
ART = PAPER / "artifacts"
FIG = ART / "figures"
TAB = ART / "tables"
ROOT = Path(__file__).resolve().parents[4]
REPORTS = ROOT / "model_testing" / "reports"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150, "pdf.fonttype": 42})
FAMILY_COLOR = {"OTE": "#4C6EF5", "FRVP": "#E8590C", "ICT": "#2F9E44"}
SHORT = {
    "frvp_short_meta_xgb_v1": "FRVP short meta (XGB)",
    "frvp_long_meta_xgb_v1": "FRVP long meta (XGB)",
    "ict_long_continuation_xgb_v1": "ICT long continuation (XGB)",
    "ict_short_continuation_xgb_v1": "ICT short continuation (XGB)",
    "frvp_long_continuation_setup2_xgb_v1": "FRVP Setup-2 long continuation (XGB)",
    "long_ote_tcn_v2_candidate": "OTE long TCN v2",
}
KIND_STYLE = {"retro_fold_quantile": ("Retrospective fold quantile", "o", "-"),
              "causal_train_quantile": ("Causal train-window quantile", "s", "--"),
              "rolling_prior_quantile": ("Rolling prior-trade quantile", "^", ":"),
              "absolute_probability": ("Absolute probability", "D", "-.")}


def fmt(x, nd=1, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if pct:
        return f"{100 * x:.{nd}f}\\%"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def fig_coverage_profit(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3), sharey=False)
    for ax, model in zip(axes.ravel(), SHORT):
        d = summary[summary["model_id"] == model]
        base = float(d["net_pnl_baseline"].iloc[0])
        scale = abs(base) if abs(base) > 1e-9 else 1.0
        for kind, (label, marker, ls) in KIND_STYLE.items():
            x = d[d["policy_kind"] == kind].sort_values("coverage")
            x = x[x["n_accepted"] > 0]
            if x.empty:
                continue
            ax.plot(x["coverage"], x["net_pnl_accepted"] / scale, marker=marker, ls=ls, ms=3, lw=1, label=label)
        ax.axhline(base / scale, color="k", lw=0.8)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.set_title(f"{SHORT[model]}\nbaseline net = {base:,.0f} {d['unit'].iloc[0]}, N = {int(d['n_baseline'].iloc[0])}", fontsize=7.5)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("coverage (accepted / baseline trades)")
    axes[0, 0].set_ylabel("accepted net PnL / |baseline net PnL|")
    axes[1, 0].set_ylabel("accepted net PnL / |baseline net PnL|")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "coverage_profit_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_pair_scatter(wf: pd.DataFrame, st: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for frame, marker, tag in [(wf, "o", "walk-forward, per-fold test"), (st, "^", "static held-out test")]:
        for fam, g in frame.groupby("family"):
            rel = g["delta_net"] / g["net_baseline"].abs().clip(lower=1e-9)
            ax.scatter(g["delta_precision"], rel.clip(-2.5, 2.5), s=16, marker=marker, alpha=0.75,
                       color=FAMILY_COLOR[fam], edgecolor="k", linewidth=0.3, label=f"{fam} ({tag})")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0, color="k", lw=0.7)
    ax.axvspan(0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1, ymin=0, ymax=0.5, color="#E03131", alpha=0.06)
    ax.set_xlabel("change in event precision (filtered minus baseline)")
    ax.set_ylabel("change in net PnL / |baseline net PnL|\n(clipped to $\\pm$2.5)")
    ax.text(0.98, 0.04, "Anti-Filter region:\nprecision up, net PnL down", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#C92A2A")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "policy_pair_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_quintiles() -> None:
    sources = [
        ("OTE long TCN v2 (walk-forward)", REPORTS / "ote_policy_backtests/v1_v2_tcn_focus/long_ote_tcn_v2_candidate/breakdown_by_confidence_quintile.csv", "OTE"),
        ("ICT long reversal", REPORTS / "ict_backtests/ict_es_primary_bootstrap_20260726_full/ict_long_reversal_xgb_v1/breakdown_by_confidence_quintile.csv", "ICT"),
        ("ICT short reversal", REPORTS / "ict_backtests/ict_es_primary_bootstrap_20260726_full/ict_short_reversal_xgb_v1/breakdown_by_confidence_quintile.csv", "ICT"),
        ("FRVP long continuation", REPORTS / "frvp_backtests/frvp_es_primary_refresh_20260701/frvp_long_continuation_xgb_v1/breakdown_by_confidence_quintile.csv", "FRVP"),
        ("FRVP long meta", REPORTS / "frvp_backtests/frvp_es_primary_refresh_20260701/frvp_long_meta_xgb_v1/breakdown_by_confidence_quintile.csv", "FRVP"),
        ("ICT short continuation", REPORTS / "ict_backtests/ict_es_primary_bootstrap_20260726_full/ict_short_continuation_xgb_v1/breakdown_by_confidence_quintile.csv", "ICT"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6))
    for ax, (title, path, fam) in zip(axes.ravel(), sources):
        d = pd.read_csv(path)
        ax.bar(d["confidence_quintile"], d["expectancy_pips"], color=FAMILY_COLOR[fam], alpha=0.85)
        ax.axhline(0, color="k", lw=0.6)
        unit = "pips" if fam == "OTE" else "ticks"
        ax.set_title(f"{title} (n = {int(d['trade_count'].sum())})", fontsize=7.5)
        ax.set_ylabel(f"expectancy ({unit})", fontsize=7)
    fig.suptitle("Post-cost expectancy by within-tape confidence quintile (Q1 lowest probability, Q5 highest)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "confidence_quintiles.pdf", bbox_inches="tight")
    plt.close(fig)


def table_headline(summary: pd.DataFrame, st: pd.DataFrame, wf: pd.DataFrame, cur: pd.DataFrame) -> None:
    rows = []

    def st_row(model, base, label):
        r = st[(st["model_id"] == model) & (st["baseline_policy"] == base)].iloc[0]
        rows.append([label, "hard abstain, static test", r["unit"], r["n_baseline"], r["n_filtered"], r["coverage"],
                     r["event_precision_baseline"], r["event_precision_filtered"], r["expectancy_baseline"], r["expectancy_filtered"], r["expectancy_rejected"],
                     r["net_baseline"], r["net_filtered"], r["net_rejected"], "AF" if r["anti_filter_total"] else ("PF" if r["delta_net"] > 0 else "ND")])

    def cur_row(label_id, label, policy):
        r = cur[cur["label"] == label_id].iloc[0]
        if r["additive_net"]:
            cls = "PF"
        elif r["anti_filter_total"]:
            cls = "AF+S" if r["sharpe_up"] else "AF"
        else:
            cls = "ND+S" if r["sharpe_up"] else "ND"
        rows.append([label, policy, r["unit"], r["n_baseline"], r["n_filtered"], r["coverage"],
                     r["label_precision_baseline"], r["label_precision_filtered"], r["expectancy_baseline"], r["expectancy_filtered"], r["expectancy_rejected"],
                     r["net_baseline"], r["net_filtered"], r["net_rejected"], cls])

    st_row("long_ote_tcn_v2_candidate", "global_threshold", "OTE long TCN v2")
    st_row("short_ote_tcn_v2_candidate", "global_threshold", "OTE short TCN v2")
    st_row("long_ote_meta_tcn_champion", "global_threshold", "OTE long meta TCN")
    st_row("short_reversal_xgb_v2_20260525", "global_threshold", "OTE short reversal XGB v2")
    cur_row("ote_short_meta_prune_v1_plus_q20", "OTE short meta TCN", "quantile q20 on accepted prune, WF")
    cur_row("frvp_short_meta_wf_q10", "FRVP short meta XGB", "quantile q10, WF")
    cur_row("frvp_short_meta_wf_q20", "FRVP short meta XGB", "quantile q20, WF")
    cur_row("frvp_short_meta_wf_q30", "FRVP short meta XGB", "quantile q30, WF")
    cur_row("ict_short_cont_wf_q50", "ICT short continuation XGB", "quantile q50, WF")
    cur_row("ict_short_cont_wf_q60", "ICT short continuation XGB", "quantile q60, WF")
    cur_row("ict_long_cont_wf_q40", "ICT long continuation XGB", "quantile q40, WF")
    cur_row("ict_long_rev_min3_to_min2_cadence", "ICT long reversal XGB", "+abstain admitted by cadence floor")
    cur_row("frvp_long_cont_refresh_to_v3_prune", "FRVP long continuation XGB", "regime/session prune v3")
    cur_row("frvp_long_meta_refresh_to_v3_prune", "FRVP long meta XGB", "regime/session prune v3")
    cur_row("frvp_long_rev_recent2y_sdh_overlap_prune", "FRVP long reversal XGB (recent-2y)", "sparse-pocket prune (accepted)")
    cur_row("frvp_setup2_medium_session_prune", "FRVP Setup-2 continuation XGB", "medium/session prune")

    lines = ["\\begin{tabular}{@{}llrrrrrrrrrrrl@{}}", "\\toprule",
             "Model & Abstention policy & $N_B$ & $N_A$ & cov. & $\\pi_B$ & $\\pi_A$ & $e_B$ & $e_A$ & $e_R$ & $V_B$ & $V_A$ & $V_R$ & class \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join([tex_escape(r[0]), tex_escape(r[1]), f"{int(r[3])}", f"{int(r[4])}", fmt(r[5], 2), fmt(r[6], 3), fmt(r[7], 3),
                                 fmt(r[8], 1), fmt(r[9], 1), fmt(r[10], 1), fmt(r[11], 0), fmt(r[12], 0), fmt(r[13], 0), r[14]]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "headline_attribution.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_census(checks: dict, wf: pd.DataFrame, st: pd.DataFrame) -> None:
    lines = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
             "Census & comparisons & precision $\\uparrow$ & precision $\\uparrow$, net $\\downarrow$ & precision $\\uparrow$, net $\\uparrow$ & $e_R > e_A$ \\\\", "\\midrule"]
    for label, frame in [("Walk-forward nested pairs, OTE (EUR/USD)", wf[wf["family"] == "OTE"]), ("Walk-forward nested pairs, FRVP (ES)", wf[wf["family"] == "FRVP"]),
                         ("Walk-forward nested pairs, ICT (ES)", wf[wf["family"] == "ICT"]), ("Static held-out nested pairs, all families", st)]:
        lines.append(f"{label} & {len(frame)} & {int(frame['precision_up'].sum())} & {int(frame['anti_filter_total'].sum())} & {int((frame['precision_up'] & ~frame['net_down']).sum())} & {int(frame['rejected_expectancy_exceeds_accepted'].sum())} \\\\")
    lines.append("\\midrule")
    for kind, label in [("retro_fold_quantile", "Trade-level, retrospective fold quantile"), ("causal_train_quantile", "Trade-level, causal train-window quantile"),
                        ("rolling_prior_quantile", "Trade-level, rolling prior-trade quantile"), ("absolute_probability", "Trade-level, absolute probability")]:
        c = checks["trade_level"][kind]
        lines.append(f"{label} & {c['comparisons']} & {c['precision_up']} & {c['precision_up_net_down']} & {c['precision_up_net_up']} & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "census.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_cutoff_sensitivity(summary: pd.DataFrame) -> None:
    models = ["frvp_short_meta_xgb_v1", "ict_short_continuation_xgb_v1", "ict_long_continuation_xgb_v1", "frvp_long_continuation_setup2_xgb_v1"]
    kinds = ["retro_fold_quantile", "causal_train_quantile", "rolling_prior_quantile"]
    lines = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule", "Model (baseline net) & Cutoff rule & q10 & q20 & q30 & q40 & q50 & q60 \\\\", "\\midrule"]
    for model in models:
        d = summary[summary["model_id"] == model]
        base = float(d["net_pnl_baseline"].iloc[0])
        first = True
        for kind in kinds:
            x = d[d["policy_kind"] == kind].set_index("policy")
            cells = [fmt(x.loc[f"q{q}", "net_pnl_accepted"], 0) for q in ["10", "20", "30", "40", "50", "60"]]
            head = f"{tex_escape(SHORT[model])} ({base:,.0f})" if first else ""
            lines.append(f"{head} & {KIND_STYLE[kind][0]} & " + " & ".join(cells) + " \\\\")
            first = False
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "cutoff_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_parity(parity: dict) -> None:
    c_wf, c_fx = parity["frvp_continuation_adaptive_walk_forward"], parity["frvp_continuation_fixed_live_policy"]
    r_wf, r_fx = parity["frvp_reversal_adaptive_walk_forward"], parity["frvp_reversal_fixed_live_policy"]
    lines = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule", "FRVP branch / evaluation path & trades & net (ticks) & Sharpe & DSR & largest-trade share & gates failed \\\\", "\\midrule",
             f"Long continuation, adaptive walk-forward (research) & {c_wf['trade_count']} & {c_wf['total_net_pnl_pips']:,.0f} & {c_wf['monthly_sharpe']:.3f} & {c_wf['approx_deflated_sharpe']:.3f} & {c_wf['largest_single_trade_share_of_total_pnl']:.3f} & 0 of 8 \\\\",
             f"Long continuation, fixed live contract (global 0.70) & {c_fx['trade_count']} & {c_fx['net_pnl_ticks']:,.0f} & {c_fx['monthly_sharpe']:.3f} & {c_fx['approx_deflated_sharpe']:.3f} & {c_fx['largest_single_trade_share_of_total_pnl']:.3f} & {len(c_fx['failed_promotion_gates'])} of 8 \\\\",
             f"Long reversal (recent-2y), adaptive walk-forward & {r_wf['trade_count']} & {r_wf['total_net_pnl_pips']:,.0f} & {r_wf['monthly_sharpe']:.3f} & {r_wf['approx_deflated_sharpe']:.3f} & {r_wf['largest_single_trade_share_of_total_pnl']:.3f} & 0 of 8 \\\\",
             f"Long reversal (recent-2y), fixed live contract (global 0.60) & {r_fx['trade_count']} & {r_fx['net_pnl_ticks']:,.0f} & {r_fx['monthly_sharpe']:.3f} & {r_fx['approx_deflated_sharpe']:.3f} & {r_fx['largest_single_trade_share_of_total_pnl']:.3f} & {len(r_fx['failed_promotion_gates'])} of 8 \\\\",
             "\\bottomrule", "\\end{tabular}"]
    (TAB / "parity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ART / "attribution_summary.csv")
    wf = pd.read_csv(ART / "walk_forward_pair_census.csv")
    st = pd.read_csv(ART / "static_pair_census.csv")
    cur = pd.read_csv(ART / "curated_run_pairs.csv")
    checks = json.loads((ART / "analysis_checks.json").read_text(encoding="utf-8"))
    parity = json.loads((ART / "parity_evidence.json").read_text(encoding="utf-8"))
    fig_coverage_profit(summary)
    fig_pair_scatter(wf, st)
    fig_quintiles()
    table_headline(summary, st, wf, cur)
    table_census(checks, wf, st)
    table_cutoff_sensitivity(summary)
    table_parity(parity)
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")), sorted(p.name for p in TAB.glob("*.tex")))


if __name__ == "__main__":
    main()
