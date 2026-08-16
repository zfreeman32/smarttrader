from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation_contracts import PROMOTION_QUALITY_MODE
from scripts.ote_targeted_filter_presets import TARGETED_FILTER_PRESETS, resolve_targeted_filters
from scripts.run_ote_policy_backtest import run_policy_backtest
from scripts.run_ote_threshold_policy_search import run_threshold_policy_search


@dataclass(frozen=True)
class StudyRunSpec:
    run_id: str
    label: str
    model_id: str
    registry_path: Path
    regime_report_root: Path
    targeted_filter_preset: str
    control_group: str
    notes: str
    min_train_years: int = 2
    max_train_years: float | None = None
    min_scheduled_test_start: str | None = None
    dynamic_filters: dict[str, object] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the saved FRVP controls, summarize concentration, and rerun reversal policy candidates "
            "with explicit regime/session abstain filters."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "model_testing"
        / "reports"
        / "frvp_regime_gated_deployment"
        / f"frvp_regime_gated_deployment_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
    )
    parser.add_argument(
        "--evaluation-contract-mode",
        choices=(PROMOTION_QUALITY_MODE, "research"),
        default=PROMOTION_QUALITY_MODE,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_specs = build_default_run_specs()
    register_dynamic_presets(run_specs)

    run_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    composite_rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    train_year_pair_rows: list[dict[str, Any]] = []

    for spec in run_specs:
        threshold_root = output_root / "threshold" / spec.run_id
        backtest_root = output_root / "backtest" / spec.run_id
        threshold_summary = run_threshold_policy_search(
            regime_report_root=spec.regime_report_root,
            output_root=threshold_root,
            registry_path=spec.registry_path,
            model_ids=[spec.model_id],
            statuses=("candidate",),
            min_positive_events=50,
            min_events_per_month=3.0,
            min_trades_per_week=3.0,
            instrument="es",
            spread_cost_mode="session_schedule",
            targeted_filter_preset=spec.targeted_filter_preset,
            write_policy_decisions=True,
            evaluation_contract_mode=args.evaluation_contract_mode,
        )
        backtest_summary = run_policy_backtest(
            regime_report_root=spec.regime_report_root,
            output_root=backtest_root,
            registry_path=spec.registry_path,
            model_ids=[spec.model_id],
            statuses=("candidate",),
            min_train_years=spec.min_train_years,
            max_train_years=spec.max_train_years,
            test_window_months=3,
            rolling_step_months=3,
            min_scheduled_test_start=spec.min_scheduled_test_start,
            min_folds=8,
            min_positive_events=50,
            min_events_per_month=3.0,
            min_trades_per_week=3.0,
            instrument="es",
            spread_cost_mode="session_schedule",
            minimum_sharpe=0.80,
            maximum_drawdown_pct=12.0,
            drawdown_starting_balance_units=10000.0,
            minimum_dsr=0.30,
            targeted_filter_preset=spec.targeted_filter_preset,
            evaluation_contract_mode=args.evaluation_contract_mode,
        )

        threshold_model = _single_model_output(threshold_summary)
        backtest_model = _single_model_output(backtest_summary)
        trade_root = Path(backtest_model["output_dir"])
        test_trades = pd.read_csv(trade_root / "selected_test_trades.csv")
        train_trades = pd.read_csv(trade_root / "selected_train_trades.csv")

        concentration = summarize_concentration(test_trades)
        overall = backtest_model["overall_test_metrics"]
        acceptance = backtest_model["acceptance"]
        gate = backtest_model["paper_trading_gate"]

        run_rows.append(
            {
                "run_id": spec.run_id,
                "control_group": spec.control_group,
                "label": spec.label,
                "model_id": spec.model_id,
                "targeted_filter_preset": spec.targeted_filter_preset,
                "trade_count": int(overall["trade_count"]),
                "net_pnl_units": float(overall["total_net_pnl_units"]),
                "expectancy_units": float(overall["expectancy_units"]),
                "sharpe": _optional_float(overall.get("monthly_sharpe")),
                "dsr": _optional_float(overall.get("approx_deflated_sharpe")),
                "max_drawdown_pct": _optional_float(overall.get("max_drawdown_pct")),
                "overall_wfe": _optional_float(backtest_model["walk_forward_efficiency"].get("overall_wfe")),
                "profitable_quarter_share": _optional_float(overall.get("profitable_quarter_share")),
                "positive_composite_expectancy_share": _optional_float(
                    backtest_model.get("positive_composite_expectancy_share")
                ),
                "largest_single_trade_share_of_total_pnl": concentration["largest_single_trade_share_of_total_pnl"],
                "largest_single_trade_share_of_abs_pnl": concentration["largest_single_trade_share_of_abs_pnl"],
                "top_5_trade_share_of_total_pnl": concentration["top_5_trade_share_of_total_pnl"],
                "top_10_trade_share_of_total_pnl": concentration["top_10_trade_share_of_total_pnl"],
                "pnl_herfindahl_index": concentration["pnl_herfindahl_index"],
                "top_trade_net_pnl_units": concentration["top_trade_net_pnl_units"],
                "selected_policy_name": str(threshold_model.get("selected_policy_name") or ""),
                "selected_policy_reason": str(threshold_model.get("selected_policy_reason") or ""),
                "accepted_for_paper_trading_gate": bool(gate["accepted"]),
                "policy_profitable_after_costs": bool(acceptance["policy_profitable_after_costs"]),
                "annualized_sharpe_above_threshold": bool(acceptance["annualized_sharpe_above_threshold"]),
                "wfe_above_threshold": bool(acceptance["wfe_above_threshold"]),
                "dsr_above_threshold": bool(acceptance["dsr_above_threshold"]),
                "profitable_quarter_share_above_threshold": bool(
                    acceptance["profitable_quarter_share_above_threshold"]
                ),
                "positive_composite_expectancy_share_above_threshold": bool(
                    acceptance["positive_composite_expectancy_share_above_threshold"]
                ),
                "largest_single_trade_share_below_limit": bool(
                    acceptance["largest_single_trade_share_below_limit"]
                ),
                "max_drawdown_pct_below_threshold": bool(acceptance["max_drawdown_pct_below_threshold"]),
                "notes": spec.notes,
                "threshold_root": _repo_relative_str(threshold_root),
                "backtest_root": _repo_relative_str(backtest_root),
            }
        )
        pair_rows.extend(build_group_rows(test_trades, spec, ("composite_regime", "session_regime"), "pair"))
        session_rows.extend(build_group_rows(test_trades, spec, ("session_regime",), "session"))
        composite_rows.extend(build_group_rows(test_trades, spec, ("composite_regime",), "composite"))
        quarter_rows.extend(build_group_rows(test_trades, spec, ("calendar_quarter",), "quarter"))
        year_rows.extend(build_group_rows(test_trades, spec, ("calendar_year",), "year"))
        train_year_pair_rows.extend(
            build_group_rows(
                train_trades,
                spec,
                ("calendar_year", "composite_regime", "session_regime"),
                "train_year_pair",
            )
        )

    run_frame = pd.DataFrame(run_rows).sort_values(["control_group", "run_id"]).reset_index(drop=True)
    pair_frame = pd.DataFrame(pair_rows)
    session_frame = pd.DataFrame(session_rows)
    composite_frame = pd.DataFrame(composite_rows)
    quarter_frame = pd.DataFrame(quarter_rows)
    year_frame = pd.DataFrame(year_rows)
    train_year_pair_frame = pd.DataFrame(train_year_pair_rows)

    run_frame.to_csv(output_root / "run_comparison.csv", index=False)
    pair_frame.to_csv(output_root / "pair_breakdown_all.csv", index=False)
    session_frame.to_csv(output_root / "session_breakdown_all.csv", index=False)
    composite_frame.to_csv(output_root / "composite_breakdown_all.csv", index=False)
    quarter_frame.to_csv(output_root / "quarter_breakdown_all.csv", index=False)
    year_frame.to_csv(output_root / "year_breakdown_all.csv", index=False)
    train_year_pair_frame.to_csv(output_root / "train_year_pair_breakdown_all.csv", index=False)

    control_specs_path = output_root / "frozen_controls.json"
    control_specs_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "spread_cost_mode": "session_schedule",
                "controls": [describe_run_spec(spec) for spec in run_specs if spec.control_group.endswith("baseline")],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": _repo_relative_str(output_root),
        "evaluation_contract_mode": args.evaluation_contract_mode,
        "run_count": int(len(run_frame)),
        "run_comparison_path": _repo_relative_str(output_root / "run_comparison.csv"),
        "pair_breakdown_path": _repo_relative_str(output_root / "pair_breakdown_all.csv"),
        "quarter_breakdown_path": _repo_relative_str(output_root / "quarter_breakdown_all.csv"),
        "train_year_pair_breakdown_path": _repo_relative_str(output_root / "train_year_pair_breakdown_all.csv"),
        "frozen_controls_path": _repo_relative_str(control_specs_path),
    }
    (output_root / "study_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (output_root / "STUDY_SUMMARY.md").write_text(
        build_markdown_summary(run_frame, pair_frame, quarter_frame, train_year_pair_frame),
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, indent=2))
    return 0


def build_default_run_specs() -> list[StudyRunSpec]:
    continuation_registry = REPO_ROOT / "models" / "frvp_es_primary_model_registry_refresh_20260701.json"
    continuation_regime_root = (
        REPO_ROOT / "model_testing" / "reports" / "frvp_regime_slices" / "frvp_es_primary_refresh_20260701"
    )
    reversal_registry = (
        REPO_ROOT / "models" / "frvp_es_primary_model_registry_long_reversal_recency_trial1_v3_20260704.json"
    )
    fullspan_regime_root = (
        REPO_ROOT / "model_testing" / "reports" / "frvp_regime_slices" / "frvp_long_reversal_q11_control_20260719"
    )
    recent2y_regime_root = (
        REPO_ROOT / "model_testing" / "reports" / "frvp_regime_slices" / "frvp_long_reversal_q11_recent2y_20260719"
    )

    return [
        StudyRunSpec(
            run_id="long_continuation_v3_baseline",
            label="Long continuation v3 frozen control",
            model_id="frvp_long_continuation_xgb_v1",
            registry_path=continuation_registry,
            regime_report_root=continuation_regime_root,
            targeted_filter_preset="frvp_long_continuation_xgb_overlap_composite_prune_v3",
            control_group="long_continuation_baseline",
            notes="Promotion-near continuation control used only as the frozen reference branch.",
        ),
        StudyRunSpec(
            run_id="long_reversal_fullspan_baseline",
            label="Long reversal full-span frozen control",
            model_id="frvp_long_reversal_xgb_v1",
            registry_path=reversal_registry,
            regime_report_root=fullspan_regime_root,
            targeted_filter_preset="frvp_long_reversal_xgb_composite_prune_v3",
            control_group="long_reversal_fullspan_baseline",
            notes="Best saved full-span reversal checkpoint from the July 19, 2026 Q11 readout.",
        ),
        StudyRunSpec(
            run_id="long_reversal_fullspan_sdm_asia_prune_v1",
            label="Long reversal full-span plus strong_down_medium/asia prune",
            model_id="frvp_long_reversal_xgb_v1",
            registry_path=reversal_registry,
            regime_report_root=fullspan_regime_root,
            targeted_filter_preset="frvp_long_reversal_xgb_fullspan_concentration_sdm_asia_v1",
            control_group="long_reversal_fullspan_candidates",
            notes="Tests the only negative selected-test pair plus a recurring negative train pocket.",
            dynamic_filters={
                "base_preset": "frvp_long_reversal_xgb_composite_prune_v3",
                "abstain_composite_session_pairs": (("strong_down_medium", "asia"),),
                "apply_to_base_policy_variants": True,
            },
        ),
        StudyRunSpec(
            run_id="long_reversal_fullspan_watch_prune_v1",
            label="Long reversal full-span plus sparse watch-pair prune",
            model_id="frvp_long_reversal_xgb_v1",
            registry_path=reversal_registry,
            regime_report_root=fullspan_regime_root,
            targeted_filter_preset="frvp_long_reversal_xgb_fullspan_concentration_watch_v1",
            control_group="long_reversal_fullspan_candidates",
            notes="Adds sparse high-concentration watch pockets to the baseline full-span prune set.",
            dynamic_filters={
                "base_preset": "frvp_long_reversal_xgb_composite_prune_v3",
                "abstain_composite_session_pairs": (
                    ("strong_down_medium", "asia"),
                    ("strong_down_high", "overlap"),
                    ("strong_up_high", "overlap"),
                    ("strong_up_high", "london"),
                ),
                "apply_to_base_policy_variants": True,
            },
        ),
        StudyRunSpec(
            run_id="long_reversal_recent2y_baseline",
            label="Long reversal recent-2y frozen control",
            model_id="frvp_long_reversal_xgb_v1",
            registry_path=reversal_registry,
            regime_report_root=recent2y_regime_root,
            targeted_filter_preset="frvp_long_reversal_xgb_recent_regime_prune_v1",
            control_group="long_reversal_recent2y_baseline",
            notes="Best saved recent-regime reversal checkpoint from the July 19, 2026 Q11 readout.",
            max_train_years=2.0,
            min_scheduled_test_start="2024-01-01",
        ),
        StudyRunSpec(
            run_id="long_reversal_recent2y_sdh_overlap_prune_v1",
            label="Long reversal recent-2y plus strong_down_high/overlap prune",
            model_id="frvp_long_reversal_xgb_v1",
            registry_path=reversal_registry,
            regime_report_root=recent2y_regime_root,
            targeted_filter_preset="frvp_long_reversal_xgb_recent2y_concentration_sdh_overlap_v1",
            control_group="long_reversal_recent2y_candidates",
            notes="Tests whether removing the sparse overlap pocket with the largest recent winner/loss reduces concentration cleanly.",
            max_train_years=2.0,
            min_scheduled_test_start="2024-01-01",
            dynamic_filters={
                "base_preset": "frvp_long_reversal_xgb_recent_regime_prune_v1",
                "abstain_composite_session_pairs": (("strong_down_high", "overlap"),),
                "apply_to_base_policy_variants": True,
            },
        ),
        StudyRunSpec(
            run_id="long_reversal_recent2y_q10_v1",
            label="Long reversal recent-2y plus 10th-percentile probability floor",
            model_id="frvp_long_reversal_xgb_v1",
            registry_path=reversal_registry,
            regime_report_root=recent2y_regime_root,
            targeted_filter_preset="frvp_long_reversal_xgb_recent2y_concentration_q10_v1",
            control_group="long_reversal_recent2y_candidates",
            notes="Tests whether a light probability floor lowers quarter fragility without blocking whole pockets.",
            max_train_years=2.0,
            min_scheduled_test_start="2024-01-01",
            dynamic_filters={
                "base_preset": "frvp_long_reversal_xgb_recent_regime_prune_v1",
                "minimum_probability_quantile": 0.10,
                "apply_to_base_policy_variants": True,
            },
        ),
        StudyRunSpec(
            run_id="long_reversal_recent2y_sdh_overlap_q10_v1",
            label="Long reversal recent-2y plus strong_down_high/overlap prune and q10 floor",
            model_id="frvp_long_reversal_xgb_v1",
            registry_path=reversal_registry,
            regime_report_root=recent2y_regime_root,
            targeted_filter_preset="frvp_long_reversal_xgb_recent2y_concentration_sdh_overlap_q10_v1",
            control_group="long_reversal_recent2y_candidates",
            notes="Combines the sparse overlap-pocket prune with a light probability floor.",
            max_train_years=2.0,
            min_scheduled_test_start="2024-01-01",
            dynamic_filters={
                "base_preset": "frvp_long_reversal_xgb_recent_regime_prune_v1",
                "abstain_composite_session_pairs": (("strong_down_high", "overlap"),),
                "minimum_probability_quantile": 0.10,
                "apply_to_base_policy_variants": True,
            },
        ),
    ]


def register_dynamic_presets(run_specs: Iterable[StudyRunSpec]) -> None:
    for spec in run_specs:
        if not spec.dynamic_filters:
            continue
        preset_name = spec.targeted_filter_preset
        base_preset = str(spec.dynamic_filters.get("base_preset") or "")
        base_filters = resolve_targeted_filters(spec.model_id, base_preset) if base_preset else {}
        merged = merge_filter_dicts(base_filters, spec.dynamic_filters)
        merged.pop("base_preset", None)
        TARGETED_FILTER_PRESETS[preset_name] = {spec.model_id: merged}


def merge_filter_dicts(base_filters: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    merged = copy.deepcopy(base_filters)
    tuple_keys = {
        "abstain_session_regimes",
        "abstain_composite_regimes",
        "abstain_composite_session_pairs",
        "abstain_composite_stress_pairs",
    }
    for key, value in overrides.items():
        if key == "base_preset":
            continue
        if key in tuple_keys and value:
            existing = list(merged.get(key, ()))
            for item in value:
                if item not in existing:
                    existing.append(item)
            merged[key] = tuple(existing)
            continue
        merged[key] = value
    return merged


def summarize_concentration(frame: pd.DataFrame) -> dict[str, float]:
    total_net = float(frame["net_pnl_units"].sum())
    abs_total = float(frame["net_pnl_units"].abs().sum())
    top_trade = float(frame["net_pnl_units"].max()) if not frame.empty else 0.0
    top5 = float(frame["net_pnl_units"].nlargest(5).sum()) if not frame.empty else 0.0
    top10 = float(frame["net_pnl_units"].nlargest(10).sum()) if not frame.empty else 0.0
    abs_shares = (
        frame["net_pnl_units"].abs() / abs_total
        if abs_total > 0.0
        else pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
    )
    return {
        "largest_single_trade_share_of_total_pnl": _safe_ratio(top_trade, total_net),
        "largest_single_trade_share_of_abs_pnl": float(abs_shares.max()) if not frame.empty else 0.0,
        "top_5_trade_share_of_total_pnl": _safe_ratio(top5, total_net),
        "top_10_trade_share_of_total_pnl": _safe_ratio(top10, total_net),
        "pnl_herfindahl_index": float((abs_shares.pow(2).sum())) if not frame.empty else 0.0,
        "top_trade_net_pnl_units": top_trade,
    }


def build_group_rows(
    frame: pd.DataFrame,
    spec: StudyRunSpec,
    group_columns: tuple[str, ...],
    group_kind: str,
) -> list[dict[str, object]]:
    grouped = (
        frame.groupby(list(group_columns), dropna=False)["net_pnl_units"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "net_pnl_units", "count": "trade_count", "mean": "expectancy_units"})
    )
    rows: list[dict[str, object]] = []
    for _, row in grouped.iterrows():
        payload = {
            "run_id": spec.run_id,
            "control_group": spec.control_group,
            "label": spec.label,
            "group_kind": group_kind,
            "net_pnl_units": float(row["net_pnl_units"]),
            "trade_count": int(row["trade_count"]),
            "expectancy_units": float(row["expectancy_units"]),
        }
        for column in group_columns:
            payload[column] = row[column]
        rows.append(payload)
    return rows


def build_markdown_summary(
    run_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    quarter_frame: pd.DataFrame,
    train_year_pair_frame: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# FRVP Regime-Gated Deployment / Concentration Study")
    lines.append("")
    lines.append(f"Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}.")
    lines.append("")
    lines.append("## Frozen controls and candidate runs")
    lines.append("")
    for _, row in run_frame.iterrows():
        lines.append(
            "- "
            f"`{row['run_id']}` ({row['control_group']}): "
            f"net `{row['net_pnl_units']:.2f}`, Sharpe `{row['sharpe']:.3f}`, "
            f"WFE `{row['overall_wfe']:.3f}`, profitable-quarter share `{row['profitable_quarter_share']:.3f}`, "
            f"largest-trade share `{row['largest_single_trade_share_of_total_pnl']:.4f}`, "
            f"accepted `{bool(row['accepted_for_paper_trading_gate'])}`."
        )
    lines.append("")
    lines.append("## Quick reads")
    lines.append("")
    for control_group in ("long_reversal_fullspan_baseline", "long_reversal_recent2y_baseline"):
        subset = run_frame[run_frame["control_group"] == control_group]
        if subset.empty:
            continue
        baseline_row = subset.iloc[0]
        lines.append(f"### {baseline_row['label']}")
        worst_pairs = pair_frame[pair_frame["run_id"] == baseline_row["run_id"]].sort_values("net_pnl_units").head(5)
        worst_quarters = (
            quarter_frame[quarter_frame["run_id"] == baseline_row["run_id"]]
            .sort_values("net_pnl_units")
            .head(5)
        )
        worst_train = (
            train_year_pair_frame[train_year_pair_frame["run_id"] == baseline_row["run_id"]]
            .sort_values("net_pnl_units")
            .head(5)
        )
        lines.append("")
        lines.append("Worst selected-test pairs:")
        for _, row in worst_pairs.iterrows():
            lines.append(
                "- "
                f"`{row.get('composite_regime','')}/{row.get('session_regime','')}`: "
                f"net `{row['net_pnl_units']:.2f}` over `{int(row['trade_count'])}` trades."
            )
        lines.append("Worst selected-test quarters:")
        for _, row in worst_quarters.iterrows():
            lines.append(
                "- "
                f"`{row.get('calendar_quarter','')}`: net `{row['net_pnl_units']:.2f}` over `{int(row['trade_count'])}` trades."
            )
        lines.append("Worst selected-train year/pair pockets:")
        for _, row in worst_train.iterrows():
            lines.append(
                "- "
                f"`{row.get('calendar_year','')}/{row.get('composite_regime','')}/{row.get('session_regime','')}`: "
                f"net `{row['net_pnl_units']:.2f}` over `{int(row['trade_count'])}` trades."
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def describe_run_spec(spec: StudyRunSpec) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "label": spec.label,
        "model_id": spec.model_id,
        "registry_path": _repo_relative_str(spec.registry_path),
        "regime_report_root": _repo_relative_str(spec.regime_report_root),
        "targeted_filter_preset": spec.targeted_filter_preset,
        "control_group": spec.control_group,
        "min_train_years": spec.min_train_years,
        "max_train_years": spec.max_train_years,
        "min_scheduled_test_start": spec.min_scheduled_test_start,
        "notes": spec.notes,
    }


def _single_model_output(summary: dict[str, Any]) -> dict[str, Any]:
    outputs = list(summary.get("model_outputs", []))
    if len(outputs) != 1:
        raise ValueError(f"Expected exactly one model output, found {len(outputs)}.")
    return outputs[0]


def _repo_relative_str(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    raise SystemExit(main())
