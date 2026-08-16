from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation_contracts import PROMOTION_QUALITY_MODE
from scripts.run_ote_policy_backtest import run_policy_backtest
from scripts.run_ote_threshold_policy_search import run_threshold_policy_search

DEFAULT_ARM_ORDER = ("session_schedule", "feature_proxy")


@dataclass(frozen=True)
class FrictionStudySpec:
    run_id: str
    label: str
    model_id: str
    registry_path: Path
    regime_report_root: Path
    targeted_filter_preset: str | None = None
    min_train_years: int = 2
    max_train_years: float | None = None
    min_scheduled_test_start: str | None = None
    notes: str = ""


def build_parser() -> argparse.ArgumentParser:
    specs = build_default_study_specs()
    parser = argparse.ArgumentParser(
        description=(
            "Run the FRVP friction A/B study by holding saved models/regime predictions fixed "
            "and varying only the spread-cost mode."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "model_testing"
        / "reports"
        / "frvp_friction_studies"
        / f"frvp_friction_ab_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
    )
    parser.add_argument(
        "--arm",
        action="append",
        dest="arms",
        choices=DEFAULT_ARM_ORDER,
        default=None,
        help="Repeat to limit the study to one or more spread-cost modes.",
    )
    parser.add_argument(
        "--spec",
        action="append",
        dest="spec_ids",
        choices=sorted(spec.run_id for spec in specs),
        default=None,
        help="Repeat to limit the study to one or more saved FRVP control branches.",
    )
    parser.add_argument(
        "--evaluation-contract-mode",
        choices=(PROMOTION_QUALITY_MODE, "research"),
        default=PROMOTION_QUALITY_MODE,
        help="Mark the saved threshold/backtest outputs as promotion-quality or research-only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    specs = build_default_study_specs()
    if args.spec_ids:
        allowed_spec_ids = set(args.spec_ids)
        specs = [spec for spec in specs if spec.run_id in allowed_spec_ids]
    arms = list(args.arms or DEFAULT_ARM_ORDER)

    run_rows: list[dict[str, Any]] = []
    threshold_summaries: list[dict[str, Any]] = []
    backtest_summaries: list[dict[str, Any]] = []

    for arm in arms:
        for spec in specs:
            threshold_root = output_root / "threshold" / arm / spec.run_id
            backtest_root = output_root / "backtest" / arm / spec.run_id

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
                spread_cost_mode=arm,
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
                spread_cost_mode=arm,
                minimum_sharpe=0.80,
                maximum_drawdown_pct=12.0,
                drawdown_starting_balance_units=10000.0,
                minimum_dsr=0.30,
                targeted_filter_preset=spec.targeted_filter_preset,
                evaluation_contract_mode=args.evaluation_contract_mode,
            )

            threshold_model_output = _get_model_output(threshold_summary, spec.model_id)
            backtest_model_output = _get_model_output(backtest_summary, spec.model_id)
            cost_snapshot = _summarize_trade_costs(backtest_model_output)

            run_rows.append(
                {
                    "run_id": spec.run_id,
                    "label": spec.label,
                    "model_id": spec.model_id,
                    "spread_cost_mode": arm,
                    "registry_path": _repo_relative_str(spec.registry_path),
                    "regime_report_root": _repo_relative_str(spec.regime_report_root),
                    "targeted_filter_preset": spec.targeted_filter_preset or "",
                    "selected_policy_name": str(threshold_model_output.get("selected_policy_name") or ""),
                    "selected_policy_reason": str(threshold_model_output.get("selected_policy_reason") or ""),
                    "qualified_policy_names": "|".join(
                        str(value) for value in threshold_model_output.get("qualified_policy_names", [])
                    ),
                    "threshold_output_root": _repo_relative_str(threshold_root),
                    "backtest_output_root": _repo_relative_str(backtest_root),
                    "trade_count": int(backtest_model_output["overall_test_metrics"]["trade_count"]),
                    "selected_test_net_pnl_units": float(
                        backtest_model_output["overall_test_metrics"]["total_net_pnl_units"]
                    ),
                    "selected_test_expectancy_units": float(
                        backtest_model_output["overall_test_metrics"]["expectancy_units"]
                    ),
                    "selected_test_sharpe": _coerce_optional_float(
                        backtest_model_output["overall_test_metrics"].get("monthly_sharpe")
                    ),
                    "selected_test_dsr": _coerce_optional_float(
                        backtest_model_output["overall_test_metrics"].get("approx_deflated_sharpe")
                    ),
                    "selected_test_max_drawdown_pct": _coerce_optional_float(
                        backtest_model_output["overall_test_metrics"].get("max_drawdown_pct")
                    ),
                    "overall_wfe": _coerce_optional_float(
                        backtest_model_output["walk_forward_efficiency"].get("overall_wfe")
                    ),
                    "profitable_quarter_share": _coerce_optional_float(
                        backtest_model_output["overall_test_metrics"].get("profitable_quarter_share")
                    ),
                    "positive_composite_expectancy_share": _coerce_optional_float(
                        backtest_model_output.get("positive_composite_expectancy_share")
                    ),
                    "paper_trading_gate_accepted": bool(
                        backtest_model_output["paper_trading_gate"]["accepted"]
                    ),
                    "mean_total_cost_units": cost_snapshot["mean_total_cost_units"],
                    "median_total_cost_units": cost_snapshot["median_total_cost_units"],
                    "mean_gross_pnl_units": cost_snapshot["mean_gross_pnl_units"],
                    "mean_net_pnl_units": cost_snapshot["mean_net_pnl_units"],
                    "notes": spec.notes,
                }
            )
            threshold_summaries.append(
                {
                    "run_id": spec.run_id,
                    "spread_cost_mode": arm,
                    "summary_path": _repo_relative_str(threshold_root / "run_summary.json"),
                }
            )
            backtest_summaries.append(
                {
                    "run_id": spec.run_id,
                    "spread_cost_mode": arm,
                    "summary_path": _repo_relative_str(backtest_root / "run_summary.json"),
                }
            )

    run_frame = pd.DataFrame(run_rows).sort_values(["run_id", "spread_cost_mode"]).reset_index(drop=True)
    comparison_frame = build_comparison_frame(run_frame)

    run_frame_path = output_root / "friction_ab_run_rows.csv"
    comparison_frame_path = output_root / "friction_ab_comparison.csv"
    run_frame.to_csv(run_frame_path, index=False)
    comparison_frame.to_csv(comparison_frame_path, index=False)

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": _repo_relative_str(output_root),
        "arms": arms,
        "spec_ids": [spec.run_id for spec in specs],
        "run_row_count": int(len(run_frame)),
        "comparison_row_count": int(len(comparison_frame)),
        "run_rows_path": _repo_relative_str(run_frame_path),
        "comparison_path": _repo_relative_str(comparison_frame_path),
        "threshold_summaries": threshold_summaries,
        "backtest_summaries": backtest_summaries,
    }
    (output_root / "study_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (output_root / "FRICTION_AB_SUMMARY.md").write_text(
        build_markdown_summary(run_frame, comparison_frame),
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, indent=2))
    return 0


def build_default_study_specs() -> list[FrictionStudySpec]:
    return [
        FrictionStudySpec(
            run_id="long_continuation_v3_baseline",
            label="Long continuation v3 baseline",
            model_id="frvp_long_continuation_xgb_v1",
            registry_path=REPO_ROOT / "models" / "frvp_es_primary_model_registry_refresh_20260701.json",
            regime_report_root=REPO_ROOT / "model_testing" / "reports" / "frvp_regime_slices" / "frvp_es_primary_refresh_20260701",
            targeted_filter_preset="frvp_long_continuation_xgb_overlap_composite_prune_v3",
            notes="Strongest saved long continuation checkpoint under the current FRVP promotion stack.",
        ),
        FrictionStudySpec(
            run_id="long_reversal_fullspan_control",
            label="Long reversal full-span control",
            model_id="frvp_long_reversal_xgb_v1",
            registry_path=REPO_ROOT / "models" / "frvp_es_primary_model_registry_long_reversal_recency_trial1_v3_20260704.json",
            regime_report_root=REPO_ROOT / "model_testing" / "reports" / "frvp_regime_slices" / "frvp_long_reversal_q11_control_20260719",
            targeted_filter_preset="frvp_long_reversal_xgb_composite_prune_v3",
            notes="Saved full-span long reversal control from the July 19, 2026 Q11 readout.",
        ),
        FrictionStudySpec(
            run_id="long_reversal_recent2y_control",
            label="Long reversal recent-2y control",
            model_id="frvp_long_reversal_xgb_v1",
            registry_path=REPO_ROOT / "models" / "frvp_es_primary_model_registry_long_reversal_recency_trial1_v3_20260704.json",
            regime_report_root=REPO_ROOT / "model_testing" / "reports" / "frvp_regime_slices" / "frvp_long_reversal_q11_recent2y_20260719",
            targeted_filter_preset="frvp_long_reversal_xgb_recent_regime_prune_v1",
            max_train_years=2.0,
            min_scheduled_test_start="2024-01-01",
            notes="Saved recent-regime long reversal control from the July 19, 2026 Q11 readout.",
        ),
        FrictionStudySpec(
            run_id="short_reversal_control",
            label="Short reversal control",
            model_id="frvp_short_reversal_xgb_v1",
            registry_path=REPO_ROOT / "models" / "frvp_es_primary_model_registry_refresh_20260701.json",
            regime_report_root=REPO_ROOT / "model_testing" / "reports" / "frvp_regime_slices" / "frvp_es_primary_refresh_20260701",
            notes="Direct short reversal control to test whether costs are the main failure driver.",
        ),
        FrictionStudySpec(
            run_id="short_meta_sentinel",
            label="Short meta sentinel",
            model_id="frvp_short_meta_xgb_v1",
            registry_path=REPO_ROOT / "models" / "frvp_es_primary_model_registry_refresh_20260701.json",
            regime_report_root=REPO_ROOT / "model_testing" / "reports" / "frvp_regime_slices" / "frvp_es_primary_refresh_20260701",
            notes="Current short-side FRVP sentinel from the shadow bundle menu.",
        ),
    ]


def build_comparison_frame(run_frame: pd.DataFrame) -> pd.DataFrame:
    if run_frame.empty:
        return pd.DataFrame()

    metric_columns = [
        "selected_policy_name",
        "trade_count",
        "selected_test_net_pnl_units",
        "selected_test_expectancy_units",
        "selected_test_sharpe",
        "selected_test_dsr",
        "selected_test_max_drawdown_pct",
        "overall_wfe",
        "profitable_quarter_share",
        "positive_composite_expectancy_share",
        "paper_trading_gate_accepted",
        "mean_total_cost_units",
        "median_total_cost_units",
        "mean_gross_pnl_units",
        "mean_net_pnl_units",
    ]
    index_columns = ["run_id", "label", "model_id", "targeted_filter_preset", "notes"]

    pivot = run_frame.set_index(index_columns + ["spread_cost_mode"])[metric_columns].unstack("spread_cost_mode")
    if pivot.empty:
        return pd.DataFrame()

    comparison = pivot.reset_index()
    comparison.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in comparison.columns
    ]

    feature_net = pd.to_numeric(
        comparison.get("selected_test_net_pnl_units_feature_proxy"),
        errors="coerce",
    )
    session_net = pd.to_numeric(
        comparison.get("selected_test_net_pnl_units_session_schedule"),
        errors="coerce",
    )
    feature_cost = pd.to_numeric(
        comparison.get("mean_total_cost_units_feature_proxy"),
        errors="coerce",
    )
    session_cost = pd.to_numeric(
        comparison.get("mean_total_cost_units_session_schedule"),
        errors="coerce",
    )
    comparison["delta_net_pnl_units_feature_minus_session"] = feature_net - session_net
    comparison["delta_mean_total_cost_units_feature_minus_session"] = feature_cost - session_cost
    comparison["preferred_spread_cost_mode_by_net_pnl"] = comparison.apply(
        lambda row: _preferred_arm(
            row.get("selected_test_net_pnl_units_session_schedule"),
            row.get("selected_test_net_pnl_units_feature_proxy"),
        ),
        axis=1,
    )
    return comparison.sort_values("run_id").reset_index(drop=True)


def build_markdown_summary(run_frame: pd.DataFrame, comparison_frame: pd.DataFrame) -> str:
    lines: list[str] = []
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines.append("# FRVP Friction A/B Study")
    lines.append("")
    lines.append(f"Generated at {generated_at}.")
    lines.append("")
    lines.append(
        "This study holds the saved FRVP model artifacts and saved regime-labeled prediction roots fixed, "
        "then reruns only the threshold/backtest economics layer under two spread-cost modes:"
    )
    lines.append("")
    lines.append("- `session_schedule`: use the ES session spread schedule as the spread cost source.")
    lines.append("- `feature_proxy`: use the saved `approx_spread` feature when available, with session spreads as fallback.")
    lines.append("")

    if comparison_frame.empty:
        lines.append("No comparison rows were produced.")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Branch Readout")
    lines.append("")
    for row in comparison_frame.to_dict(orient="records"):
        lines.append(f"### {row['label']}")
        lines.append("")
        lines.append(f"- Model: `{row['model_id']}`")
        if row.get("targeted_filter_preset"):
            lines.append(f"- Targeted filter preset: `{row['targeted_filter_preset']}`")
        lines.append(
            "- Session schedule: "
            f"net `{_format_metric(row.get('selected_test_net_pnl_units_session_schedule'))}`, "
            f"Sharpe `{_format_metric(row.get('selected_test_sharpe_session_schedule'))}`, "
            f"mean cost `{_format_metric(row.get('mean_total_cost_units_session_schedule'))}`."
        )
        lines.append(
            "- Feature proxy: "
            f"net `{_format_metric(row.get('selected_test_net_pnl_units_feature_proxy'))}`, "
            f"Sharpe `{_format_metric(row.get('selected_test_sharpe_feature_proxy'))}`, "
            f"mean cost `{_format_metric(row.get('mean_total_cost_units_feature_proxy'))}`."
        )
        lines.append(
            "- Delta (`feature - session`): "
            f"net `{_format_metric(row.get('delta_net_pnl_units_feature_minus_session'))}`, "
            f"mean cost `{_format_metric(row.get('delta_mean_total_cost_units_feature_minus_session'))}`, "
            f"preferred arm `{row.get('preferred_spread_cost_mode_by_net_pnl')}`."
        )
        if row.get("notes"):
            lines.append(f"- Notes: {row['notes']}")
        lines.append("")

    return "\n".join(lines)


def _get_model_output(summary_payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    for item in summary_payload.get("model_outputs", []):
        if item.get("model_id") == model_id:
            return item
    raise KeyError(f"Could not find model_id={model_id!r} in summary payload.")


def _summarize_trade_costs(model_output: dict[str, Any]) -> dict[str, float | None]:
    paths = dict(model_output.get("paths") or {})
    selected_test_trades_path = paths.get("selected_test_trades")
    if not selected_test_trades_path:
        return {
            "mean_total_cost_units": None,
            "median_total_cost_units": None,
            "mean_gross_pnl_units": None,
            "mean_net_pnl_units": None,
        }

    trade_frame = pd.read_csv(selected_test_trades_path)
    if trade_frame.empty:
        return {
            "mean_total_cost_units": None,
            "median_total_cost_units": None,
            "mean_gross_pnl_units": None,
            "mean_net_pnl_units": None,
        }
    return {
        "mean_total_cost_units": _series_stat(trade_frame, "total_cost_units", "mean"),
        "median_total_cost_units": _series_stat(trade_frame, "total_cost_units", "median"),
        "mean_gross_pnl_units": _series_stat(trade_frame, "gross_pnl_units", "mean"),
        "mean_net_pnl_units": _series_stat(trade_frame, "net_pnl_units", "mean"),
    }


def _series_stat(frame: pd.DataFrame, column: str, stat: str) -> float | None:
    if column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    if stat == "mean":
        return float(series.mean())
    if stat == "median":
        return float(series.median())
    raise ValueError(f"Unsupported stat={stat!r}.")


def _preferred_arm(session_value: object, feature_value: object) -> str:
    session = _coerce_optional_float(session_value)
    feature = _coerce_optional_float(feature_value)
    if session is None and feature is None:
        return "unknown"
    if session is None:
        return "feature_proxy"
    if feature is None:
        return "session_schedule"
    if feature > session:
        return "feature_proxy"
    if session > feature:
        return "session_schedule"
    return "tie"


def _format_metric(value: object) -> str:
    numeric = _coerce_optional_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.3f}"


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _repo_relative_str(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return path_obj.as_posix()
    return path_obj.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
