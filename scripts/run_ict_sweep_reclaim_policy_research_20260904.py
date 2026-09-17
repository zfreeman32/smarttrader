from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_testing.evaluation_costs import (
    describe_evaluation_cost_config,
    resolve_evaluation_cost_config,
)
from model_testing.ote_abstain_policy import HardAbstainConfig
from model_testing.ote_policy_backtest import WalkForwardBacktestConfig, run_walk_forward_backtest
from model_testing.ote_policy_metrics import sanitize_for_json, summarize_trade_performance
from model_testing.ote_threshold_policy import ThresholdSearchConfig, attach_forward_trade_outcomes
from models.ote_registry_loader import load_ote_model_registry
from scripts.evaluation_contracts import build_evaluation_contract, build_paper_trading_gate
from scripts.run_ote_policy_backtest import (
    _load_backtest_market_frame,
    _load_backtest_prediction_frame,
    _load_training_summary,
    _resolve_effective_trials,
    _resolve_label_assumption,
    _resolve_threshold,
    _write_model_outputs,
)


MODEL_ID = "ict_short_reversal_sweep_reclaim_xgb_v1"
REGISTRY_PATH = REPO_ROOT / "models" / "ict_short_setup_family_model_registry_20260811_audit.json"
REGIME_REPORT_ROOT = (
    REPO_ROOT / "model_testing" / "reports" / "ict_regime_slices" / "ict_short_setup_family_20260811_audit"
)
BASELINE_BACKTEST_DIR = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "ict_backtests"
    / "ict_short_setup_family_20260811_audit"
    / MODEL_ID
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "ict_policy_research"
    / "ict_sweep_reclaim_20260904"
)


@dataclass(frozen=True)
class PolicyOverlaySpec:
    name: str
    description: str
    min_scheduled_test_start: str | None
    cooldown_bars: int
    abstain_high_stress: bool = False
    abstain_off_hours: bool = False
    abstain_session_regimes: tuple[str, ...] = ()

    @property
    def window_label(self) -> str:
        return "full_2020_2026" if self.min_scheduled_test_start is None else "recent_2023_2026"

    def abstain_config(self) -> HardAbstainConfig:
        return HardAbstainConfig(
            abstain_high_stress=bool(self.abstain_high_stress),
            abstain_off_hours=bool(self.abstain_off_hours),
            cooldown_bars=int(self.cooldown_bars),
            abstain_session_regimes=tuple(self.abstain_session_regimes),
            apply_to_base_policy_variants=True,
            probability_column="policy_probability",
            signal_candidate_column="policy_signal_candidate",
            position_column="source_row_idx",
        )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    diagnostics = _write_baseline_trade_diagnostics()
    candidate_rows = _run_walk_forward_overlays()
    candidates = pd.DataFrame(candidate_rows).sort_values(["window_label", "net_ticks"], ascending=[True, False])
    candidates.to_csv(OUTPUT_ROOT / "candidate_walk_forward_results.csv", index=False)
    (OUTPUT_ROOT / "candidate_walk_forward_results.json").write_text(
        json.dumps(sanitize_for_json(candidate_rows), indent=2),
        encoding="utf-8",
    )
    _write_markdown_summary(candidates, diagnostics)

    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT.relative_to(REPO_ROOT)),
                "candidate_rows": int(len(candidate_rows)),
                "summary": str((OUTPUT_ROOT / "RESEARCH_SUMMARY.md").relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )


def _write_baseline_trade_diagnostics() -> dict[str, Any]:
    trades = _load_baseline_trades()
    diagnostics_root = OUTPUT_ROOT / "diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    year = _summarize_groups(trades, ("calendar_year",))
    session = _summarize_groups(trades, ("session_regime",))
    year_session = _summarize_groups(trades, ("calendar_year", "session_regime"))
    regime_session = _summarize_groups(trades, ("composite_regime", "session_regime"))
    period_session = _summarize_groups(_with_deterioration_period(trades), ("deterioration_period", "session_regime"))
    period = _summarize_groups(_with_deterioration_period(trades), ("deterioration_period",))

    outputs = {
        "baseline_by_year": year,
        "baseline_by_session": session,
        "baseline_by_year_session": year_session,
        "baseline_by_composite_session": regime_session,
        "baseline_by_deterioration_period_session": period_session,
        "baseline_by_deterioration_period": period,
    }
    for name, frame in outputs.items():
        frame.to_csv(diagnostics_root / f"{name}.csv", index=False)

    return {
        "baseline_trade_count": int(len(trades)),
        "baseline_net_ticks": float(trades["net_pnl_units"].sum()),
        "paths": {name: str((diagnostics_root / f"{name}.csv").relative_to(REPO_ROOT)) for name in outputs},
        "year": year,
        "session": session,
        "period_session": period_session,
        "period": period,
    }


def _run_walk_forward_overlays() -> list[dict[str, Any]]:
    registry = load_ote_model_registry(REGISTRY_PATH)
    model = registry.get_model(MODEL_ID)
    artifact_dir = model.resolve_artifact_path()
    training_summary = _load_training_summary(artifact_dir)
    cost_config = resolve_evaluation_cost_config("es", spread_cost_mode="session_schedule")
    market_frame = _load_backtest_market_frame(artifact_dir)
    prediction_frame = _load_backtest_prediction_frame(REGIME_REPORT_ROOT / MODEL_ID)
    effective_trials = _resolve_effective_trials(training_summary, override=None)
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=_resolve_threshold(model, training_summary),
        instrument=cost_config.instrument,
        unit_label=cost_config.unit_label,
        spread_cost_mode=cost_config.spread_cost_mode,
        event_tolerance_bars=_resolve_label_assumption(training_summary, "event_tolerance_bars", default=2),
        event_cooldown_bars=_resolve_label_assumption(training_summary, "event_cooldown_bars", default=4),
        min_positive_events=50,
        min_events_per_month=3.0,
        label_max_holding_bars=_resolve_label_assumption(training_summary, "label_max_holding_bars", default=120),
        pip_size=cost_config.price_increment,
        slippage_spread_multiplier=0.0,
        fixed_slippage_pips_per_trade=cost_config.fixed_slippage_units_per_trade,
        commission_pips_per_trade=cost_config.commission_units_per_trade,
        session_spread_pips=cost_config.session_spread_units,
    )
    prediction_frame = attach_forward_trade_outcomes(
        prediction_frame,
        market_frame=market_frame,
        direction=model.direction,
        config=threshold_config,
    )

    rows: list[dict[str, Any]] = []
    for spec in _policy_specs():
        output_dir = OUTPUT_ROOT / "wf" / spec.window_label / spec.name
        output_dir.mkdir(parents=True, exist_ok=True)
        backtest_config = WalkForwardBacktestConfig(
            min_train_years=2,
            test_window_months=3,
            rolling_step_months=3,
            min_scheduled_test_start=_parse_optional_timestamp(spec.min_scheduled_test_start),
            purge_gap_bars=max(_resolve_label_assumption(training_summary, "window_size", default=24), 40),
            min_folds=7,
            min_trades_per_week=3.0,
            minimum_annualized_sharpe=0.80,
            minimum_deflated_sharpe=0.30,
            maximum_drawdown_pct=12.0,
            drawdown_starting_balance_pips=10000.0,
        )
        results = run_walk_forward_backtest(
            prediction_frame.copy(),
            market_frame=market_frame,
            direction=model.direction,
            threshold_config=threshold_config,
            backtest_config=backtest_config,
            abstain_config=spec.abstain_config(),
            model_id=model.model_id,
            backend=model.backend,
            effective_trials=effective_trials,
        )
        evaluation_contract = build_evaluation_contract(
            evaluation_contract_mode="research",
            min_trades_per_week=3.0,
            requested_min_folds=7,
            effective_min_folds=7,
        )
        results["summary"]["evaluation_contract"] = evaluation_contract
        results["summary"]["paper_trading_gate"] = build_paper_trading_gate(
            results["summary"]["acceptance"],
            evaluation_contract=evaluation_contract,
        )
        results["summary"]["targeted_filters"] = _describe_spec(spec)
        results["summary"]["evaluation_costs"] = describe_evaluation_cost_config(cost_config)
        paths = _write_model_outputs(output_dir, results)
        rows.append(_candidate_row(spec, results["summary"], output_dir, paths))

    return rows


def _policy_specs() -> tuple[PolicyOverlaySpec, ...]:
    full_specs = (
        PolicyOverlaySpec(
            name="baseline_no_abstain",
            description="Global threshold only; no hard abstain and no cooldown. Reproduces the prior setup-family baseline.",
            min_scheduled_test_start=None,
            cooldown_bars=0,
        ),
        PolicyOverlaySpec(
            name="cooldown_only",
            description="Global threshold with only the standard 4-bar cooldown.",
            min_scheduled_test_start=None,
            cooldown_bars=4,
        ),
        PolicyOverlaySpec(
            name="drop_off_hours_no_cooldown",
            description="Pure session read: remove off-hours without adding cooldown or stress filtering.",
            min_scheduled_test_start=None,
            cooldown_bars=0,
            abstain_off_hours=True,
        ),
        PolicyOverlaySpec(
            name="drop_off_hours",
            description="Remove off-hours and apply the standard 4-bar cooldown.",
            min_scheduled_test_start=None,
            cooldown_bars=4,
            abstain_off_hours=True,
        ),
        PolicyOverlaySpec(
            name="drop_high_stress",
            description="Remove high-stress rows and apply the standard 4-bar cooldown.",
            min_scheduled_test_start=None,
            cooldown_bars=4,
            abstain_high_stress=True,
        ),
        PolicyOverlaySpec(
            name="default_stress_off_hours",
            description="Default hard abstain path: remove high-stress and off-hours rows with the standard 4-bar cooldown.",
            min_scheduled_test_start=None,
            cooldown_bars=4,
            abstain_high_stress=True,
            abstain_off_hours=True,
        ),
        PolicyOverlaySpec(
            name="keep_asia_new_york_no_cooldown",
            description="Pure session read: keep only Asia and New York sessions.",
            min_scheduled_test_start=None,
            cooldown_bars=0,
            abstain_session_regimes=("london", "off_hours", "overlap"),
        ),
        PolicyOverlaySpec(
            name="keep_asia_new_york",
            description="Operational session filter: keep only Asia and New York sessions, with standard cooldown.",
            min_scheduled_test_start=None,
            cooldown_bars=4,
            abstain_session_regimes=("london", "off_hours", "overlap"),
        ),
        PolicyOverlaySpec(
            name="new_york_only",
            description="Keep only New York session, with standard cooldown.",
            min_scheduled_test_start=None,
            cooldown_bars=4,
            abstain_session_regimes=("asia", "london", "off_hours", "overlap"),
        ),
        PolicyOverlaySpec(
            name="asia_only",
            description="Keep only Asia session, with standard cooldown.",
            min_scheduled_test_start=None,
            cooldown_bars=4,
            abstain_session_regimes=("london", "new_york", "off_hours", "overlap"),
        ),
    )
    recent_specs = tuple(
        PolicyOverlaySpec(
            name=f"{spec.name}_recent_2023",
            description=f"2023+ deterioration-window rerun of {spec.name}: {spec.description}",
            min_scheduled_test_start="2023-01-01",
            cooldown_bars=spec.cooldown_bars,
            abstain_high_stress=spec.abstain_high_stress,
            abstain_off_hours=spec.abstain_off_hours,
            abstain_session_regimes=spec.abstain_session_regimes,
        )
        for spec in full_specs
    )
    return (*full_specs, *recent_specs)


def _load_baseline_trades() -> pd.DataFrame:
    path = BASELINE_BACKTEST_DIR / "selected_test_trades.csv"
    trades = pd.read_csv(path)
    trades["entry_datetime"] = pd.to_datetime(trades["entry_datetime"], utc=True, errors="coerce")
    trades["net_pnl_units"] = pd.to_numeric(trades["net_pnl_units"], errors="coerce")
    trades["gross_pnl_units"] = pd.to_numeric(trades["gross_pnl_units"], errors="coerce")
    trades["calendar_year"] = pd.to_numeric(trades["calendar_year"], errors="coerce").astype("Int64")
    trades = trades.dropna(subset=["entry_datetime", "net_pnl_units"]).copy()
    return trades


def _with_deterioration_period(trades: pd.DataFrame) -> pd.DataFrame:
    working = trades.copy()
    year = pd.to_numeric(working["calendar_year"], errors="coerce")
    working["deterioration_period"] = year.where(year >= 2023, other=2022)
    working["deterioration_period"] = working["deterioration_period"].map(
        lambda value: "2023_2026" if int(value) >= 2023 else "2020_2022"
    )
    return working


def _summarize_groups(trades: pd.DataFrame, group_columns: Iterable[str]) -> pd.DataFrame:
    group_columns = tuple(group_columns)
    rows: list[dict[str, Any]] = []
    for group_key, bucket in trades.groupby(list(group_columns), dropna=False, sort=True):
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        metrics = _simple_trade_metrics(bucket)
        row = {column: value for column, value in zip(group_columns, group_values)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _simple_trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trade_count": 0,
            "net_ticks": 0.0,
            "expectancy_ticks": 0.0,
            "hit_rate": 0.0,
            "profit_factor": None,
            "average_win_ticks": 0.0,
            "average_loss_ticks": 0.0,
            "largest_abs_trade_ticks": 0.0,
            "largest_trade_share_of_abs_pnl": None,
        }
    pnl = pd.to_numeric(trades["net_pnl_units"], errors="coerce").dropna().astype(float)
    wins = pnl.loc[pnl > 0.0]
    losses = pnl.loc[pnl < 0.0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    total_abs = float(pnl.abs().sum())
    return {
        "trade_count": int(len(pnl)),
        "net_ticks": float(pnl.sum()),
        "expectancy_ticks": float(pnl.mean()) if len(pnl) else 0.0,
        "hit_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "profit_factor": None if gross_loss <= 0.0 else float(gross_win / gross_loss),
        "average_win_ticks": float(wins.mean()) if not wins.empty else 0.0,
        "average_loss_ticks": float(abs(losses.mean())) if not losses.empty else 0.0,
        "largest_abs_trade_ticks": float(pnl.abs().max()) if not pnl.empty else 0.0,
        "largest_trade_share_of_abs_pnl": None if total_abs <= 0.0 else float(pnl.abs().max() / total_abs),
    }


def _candidate_row(
    spec: PolicyOverlaySpec,
    summary: dict[str, Any],
    output_dir: Path,
    paths: dict[str, str],
) -> dict[str, Any]:
    metrics = summary["overall_test_metrics"]
    wfe = summary["walk_forward_efficiency"]
    acceptance = summary["acceptance"]
    gate = summary["paper_trading_gate"]
    train_window = summary["train_window"]
    trade_count = int(metrics["trade_count"])
    period_days = float(metrics["period_days"])
    return {
        "candidate": spec.name,
        "window_label": spec.window_label,
        "description": spec.description,
        "min_scheduled_test_start": spec.min_scheduled_test_start or "",
        "cooldown_bars": int(spec.cooldown_bars),
        "abstain_high_stress": bool(spec.abstain_high_stress),
        "abstain_off_hours": bool(spec.abstain_off_hours),
        "abstain_session_regimes": "|".join(spec.abstain_session_regimes),
        "fold_count": int(summary["fold_count"]),
        "realized_scheduled_test_start_min": train_window.get("realized_scheduled_test_start_min"),
        "realized_scheduled_test_start_max": train_window.get("realized_scheduled_test_start_max"),
        "trades": trade_count,
        "trades_per_week": 0.0 if period_days <= 0.0 else float(trade_count / (period_days / 7.0)),
        "net_ticks": float(metrics["total_net_pnl_units"]),
        "expectancy_ticks": float(metrics["expectancy_units"]),
        "profit_factor": metrics["profit_factor"],
        "sharpe": metrics["monthly_sharpe"],
        "dsr": metrics["approx_deflated_sharpe"],
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "largest_trade_share": metrics.get("largest_single_trade_share_of_total_pnl"),
        "wfe": wfe.get("overall_wfe"),
        "profitable_quarter_share": metrics.get("profitable_quarter_share"),
        "positive_composite_share": summary.get("positive_composite_expectancy_share"),
        "policy_profitable_after_costs": acceptance["policy_profitable_after_costs"],
        "sharpe_gate": acceptance["annualized_sharpe_above_threshold"],
        "dsr_gate": acceptance["dsr_above_threshold"],
        "wfe_gate": acceptance["wfe_above_threshold"],
        "quarter_gate": acceptance["profitable_quarter_share_above_threshold"],
        "composite_gate": acceptance["positive_composite_expectancy_share_above_threshold"],
        "concentration_gate": acceptance["largest_single_trade_share_below_limit"],
        "drawdown_gate": acceptance["max_drawdown_pct_below_threshold"],
        "accepted_raw": gate["accepted_raw"],
        "accepted": gate["accepted"],
        "promotion_quality_disqualifiers": "|".join(gate["promotion_quality_disqualifiers"]),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "summary_path": _relative_path(paths.get("summary")),
    }


def _write_markdown_summary(candidates: pd.DataFrame, diagnostics: dict[str, Any]) -> None:
    full = candidates.loc[candidates["window_label"] == "full_2020_2026"].copy()
    recent = candidates.loc[candidates["window_label"] == "recent_2023_2026"].copy()
    baseline_full = _row_by_candidate(full, "baseline_no_abstain")
    baseline_recent = _row_by_candidate(recent, "baseline_no_abstain_recent_2023")
    best_full = full.sort_values("net_ticks", ascending=False).iloc[0].to_dict()
    best_recent = recent.sort_values("net_ticks", ascending=False).iloc[0].to_dict()

    year_lines = _top_lines(diagnostics["year"], "calendar_year")
    session_lines = _top_lines(diagnostics["session"], "session_regime")
    period_session = diagnostics["period_session"]
    recent_session_lines = _top_lines(
        period_session.loc[period_session["deterioration_period"] == "2023_2026"],
        "session_regime",
    )

    markdown = [
        "# ICT Sweep-Reclaim Policy Research - 2023-2026 Session Deterioration",
        "",
        "## Plan",
        "",
        "1. Reuse the frozen `ict_short_reversal_sweep_reclaim_xgb_v1` model, registry, and regime-labeled predictions.",
        "2. Keep the pass policy-only: no retraining, no canonical label changes, and no registry write-back.",
        "3. Diagnose the prior selected test trades by calendar year, session, and year/session.",
        "4. Run walk-forward overlays for cooldown-only, high-stress/off-hours abstains, and core-session filters.",
        "5. Repeat each overlay on the 2023+ deterioration window to separate full-span optics from recent behavior.",
        "6. Decide whether any overlay is strong enough for shadow/promotion follow-up or only useful as a diagnostic.",
        "",
        "## Inputs",
        "",
        f"- Model: `{MODEL_ID}`",
        f"- Registry: `{REGISTRY_PATH.relative_to(REPO_ROOT)}`",
        f"- Regime report root: `{REGIME_REPORT_ROOT.relative_to(REPO_ROOT)}`",
        f"- Baseline backtest root: `{BASELINE_BACKTEST_DIR.relative_to(REPO_ROOT)}`",
        f"- Output root: `{OUTPUT_ROOT.relative_to(REPO_ROOT)}`",
        "",
        "## Baseline Diagnostics",
        "",
        f"- Baseline selected trades: `{diagnostics['baseline_trade_count']}`",
        f"- Baseline full-span net: `{_fmt(diagnostics['baseline_net_ticks'])}` ticks",
        "- Year read:",
        *year_lines,
        "- Session read:",
        *session_lines,
        "- 2023-2026 session read:",
        *recent_session_lines,
        "",
        "## Walk-Forward Overlay Results",
        "",
        "- Full 2020-2026 baseline:",
        _candidate_bullet(baseline_full),
        "- Best full-span overlay by net ticks:",
        _candidate_bullet(best_full),
        "- 2023+ baseline:",
        _candidate_bullet(baseline_recent),
        "- Best 2023+ overlay by net ticks:",
        _candidate_bullet(best_recent),
        "",
        "## Findings",
        "",
        "- Deterioration is real. The branch made most of its money in 2020 and 2022, then went negative in 2023, 2024, and 2026; 2025 was only mildly positive.",
        "- The session culprit is not London/overlap, which are tiny samples. The real drag is off-hours after 2023, with Asia also losing its earlier edge in the recent window.",
        "- New York is the only recent session with a clearly positive trade-level read, but the walk-forward New-York-only overlay cuts sample size enough that concentration and cadence remain uncomfortable.",
        "- The best-looking full-span overlay improves net PnL but still leans on the old 2020/2022 regime. That is useful evidence, not a promotion case.",
        "",
        "## Decision",
        "",
        "- Keep `ict_short_reversal_sweep_reclaim_xgb_v1` research-only.",
        "- Do not write any policy back to the registry or add it to live/paper accepted slots.",
        "- If this branch is revisited, use a forward shadow observation of a New-York-biased or off-hours-excluded policy; do not spend retraining budget before the broader ICT setup-logic audit and classic-breaker label-design branch are complete.",
        "",
        "## Artifacts",
        "",
        f"- Candidate table: `{(OUTPUT_ROOT / 'candidate_walk_forward_results.csv').relative_to(REPO_ROOT)}`",
        f"- Baseline diagnostics: `{Path(diagnostics['paths']['baseline_by_year_session'])}`",
        f"- Per-candidate walk-forward outputs: `{(OUTPUT_ROOT / 'wf').relative_to(REPO_ROOT)}`",
        "",
    ]
    (OUTPUT_ROOT / "RESEARCH_SUMMARY.md").write_text("\n".join(markdown), encoding="utf-8")


def _row_by_candidate(frame: pd.DataFrame, candidate: str) -> dict[str, Any]:
    match = frame.loc[frame["candidate"] == candidate]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _top_lines(frame: pd.DataFrame, label_column: str) -> list[str]:
    lines = []
    for _, row in frame.iterrows():
        lines.append(
            f"  - `{row[label_column]}`: `{int(row['trade_count'])}` trades, "
            f"`{_fmt(row['net_ticks'])}` ticks, expectancy `{_fmt(row['expectancy_ticks'])}`"
        )
    return lines


def _candidate_bullet(row: dict[str, Any]) -> str:
    if not row:
        return "  - unavailable"
    return (
        f"  - `{row['candidate']}`: `{int(row['trades'])}` trades, `{_fmt(row['net_ticks'])}` ticks, "
        f"expectancy `{_fmt(row['expectancy_ticks'])}`, Sharpe `{_fmt(row['sharpe'])}`, "
        f"DSR `{_fmt(row['dsr'])}`, WFE `{_fmt(row['wfe'])}`, accepted raw `{row['accepted_raw']}`"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.2f}"


def _describe_spec(spec: PolicyOverlaySpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "min_scheduled_test_start": spec.min_scheduled_test_start,
        "cooldown_bars": int(spec.cooldown_bars),
        "abstain_high_stress": bool(spec.abstain_high_stress),
        "abstain_off_hours": bool(spec.abstain_off_hours),
        "abstain_session_regimes": list(spec.abstain_session_regimes),
        "apply_to_base_policy_variants": True,
    }


def _parse_optional_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed


def _relative_path(path: str | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
