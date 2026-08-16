from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_testing.evaluation_costs import (
    DEFAULT_SPREAD_COST_MODE,
    SUPPORTED_SPREAD_COST_MODES,
    describe_evaluation_cost_config,
    resolve_evaluation_cost_config,
)
from model_testing.ote_breakout_event_metadata import BREAKOUT_EVENT_METADATA_COLUMNS
from model_testing.ote_abstain_policy import HardAbstainConfig
from model_testing.ote_policy_backtest import WalkForwardBacktestConfig, run_walk_forward_backtest
from model_testing.ote_policy_metrics import sanitize_for_json
from model_testing.ote_prediction_joiner import load_source_frame
from model_testing.ote_threshold_policy import ThresholdSearchConfig
from models.ote_registry_loader import OTEModelRecord, load_ote_model_registry
from scripts.evaluation_contracts import (
    DEFAULT_PROMOTION_MIN_TRADES_PER_WEEK_FLOOR,
    PROMOTION_QUALITY_MODE,
    build_evaluation_contract,
    build_paper_trading_gate,
)
from scripts.ote_targeted_filter_presets import (
    TARGETED_FILTER_PRESETS,
    build_targeted_abstain_config,
    describe_abstain_config,
)


def run_policy_backtest(
    *,
    regime_report_root: str | Path,
    output_root: str | Path,
    registry_path: str | Path,
    model_ids: Sequence[str] | None = None,
    statuses: Sequence[str] = ("active", "candidate"),
    include_roles: Sequence[str] | None = None,
    threshold_grid: Sequence[float] | None = None,
    min_train_years: int = 2,
    max_train_years: float | None = None,
    test_window_months: int = 3,
    rolling_step_months: int = 3,
    min_scheduled_test_start: str | None = None,
    min_folds: int = 8,
    max_folds: int | None = None,
    min_positive_events: int = 50,
    min_events_per_month: float = 3.0,
    min_trades_per_week: float = 3.0,
    purge_gap_bars: int | None = None,
    instrument: str = "fx",
    spread_cost_mode: str = DEFAULT_SPREAD_COST_MODE,
    minimum_sharpe: float = 0.80,
    maximum_drawdown_pct: float = 12.0,
    drawdown_starting_balance_units: float = 10000.0,
    fixed_slippage_units_per_trade: float | None = None,
    commission_units_per_trade: float | None = None,
    dsr_effective_trials: int | None = None,
    minimum_dsr: float = 0.30,
    targeted_filter_preset: str | None = None,
    evaluation_contract_mode: str = PROMOTION_QUALITY_MODE,
    promotion_min_trades_per_week_floor: float = DEFAULT_PROMOTION_MIN_TRADES_PER_WEEK_FLOOR,
    requested_min_folds: int | None = None,
    available_min_folds: int | None = None,
) -> Dict[str, Any]:
    if max_train_years is not None and float(max_train_years) < float(min_train_years):
        raise ValueError(
            f"max_train_years={max_train_years} must be greater than or equal to min_train_years={min_train_years}."
        )
    regime_report_root = Path(regime_report_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cost_config = resolve_evaluation_cost_config(
        instrument,
        spread_cost_mode=spread_cost_mode,
        fixed_slippage_units_per_trade=fixed_slippage_units_per_trade,
        commission_units_per_trade=commission_units_per_trade,
    )
    evaluation_contract = build_evaluation_contract(
        evaluation_contract_mode=evaluation_contract_mode,
        min_trades_per_week=min_trades_per_week,
        promotion_min_trades_per_week_floor=promotion_min_trades_per_week_floor,
        requested_min_folds=requested_min_folds,
        available_min_folds=available_min_folds,
        effective_min_folds=min_folds,
    )

    registry = load_ote_model_registry(registry_path)
    models = _resolve_models(
        registry,
        model_ids=model_ids,
        statuses=statuses,
        include_roles=include_roles,
    )

    model_outputs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model in models:
        model_report_dir = regime_report_root / model.model_id
        prediction_frame = _load_backtest_prediction_frame(model_report_dir)
        training_summary = _load_training_summary(model.resolve_artifact_path())
        market_frame = _load_backtest_market_frame(model.resolve_artifact_path())
        effective_trials = _resolve_effective_trials(
            training_summary,
            override=dsr_effective_trials,
        )

        effective_purge_gap = int(
            purge_gap_bars
            if purge_gap_bars is not None
            else max(
                int(_resolve_label_assumption(training_summary, "window_size", default=24)),
                40,
            )
        )
        default_threshold = _resolve_threshold(model, training_summary)
        threshold_config = ThresholdSearchConfig(
            probability_column="model_probability",
            global_threshold=default_threshold,
            instrument=cost_config.instrument,
            unit_label=cost_config.unit_label,
            spread_cost_mode=cost_config.spread_cost_mode,
            threshold_grid=tuple(threshold_grid) if threshold_grid is not None else ThresholdSearchConfig(
                probability_column="model_probability",
                global_threshold=default_threshold,
            ).threshold_grid,
            event_tolerance_bars=_resolve_label_assumption(training_summary, "event_tolerance_bars", default=2),
            event_cooldown_bars=_resolve_label_assumption(training_summary, "event_cooldown_bars", default=4),
            min_positive_events=min_positive_events,
            min_events_per_month=min_events_per_month,
            label_max_holding_bars=_resolve_label_assumption(training_summary, "label_max_holding_bars", default=120),
            pip_size=cost_config.price_increment,
            slippage_spread_multiplier=0.0,
            fixed_slippage_pips_per_trade=cost_config.fixed_slippage_units_per_trade,
            commission_pips_per_trade=cost_config.commission_units_per_trade,
            session_spread_pips=cost_config.session_spread_units,
        )
        backtest_config = WalkForwardBacktestConfig(
            min_train_years=min_train_years,
            max_train_years=max_train_years,
            test_window_months=test_window_months,
            rolling_step_months=rolling_step_months,
            min_scheduled_test_start=_parse_optional_utc_timestamp(min_scheduled_test_start),
            purge_gap_bars=effective_purge_gap,
            min_folds=min_folds,
            max_folds=max_folds,
            min_trades_per_week=min_trades_per_week,
            minimum_annualized_sharpe=minimum_sharpe,
            minimum_deflated_sharpe=minimum_dsr,
            maximum_drawdown_pct=maximum_drawdown_pct,
            drawdown_starting_balance_pips=drawdown_starting_balance_units,
        )
        abstain_config = _build_abstain_config(
            model,
            threshold_config,
            targeted_filter_preset=targeted_filter_preset,
        )
        results = run_walk_forward_backtest(
            prediction_frame,
            market_frame=market_frame,
            direction=model.direction,
            threshold_config=threshold_config,
            backtest_config=backtest_config,
            abstain_config=abstain_config,
            model_id=model.model_id,
            backend=model.backend,
            effective_trials=effective_trials,
        )

        model_output_dir = output_root / model.model_id
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_summary = dict(results["summary"])
        model_summary["paper_trading_gate"] = build_paper_trading_gate(
            model_summary["acceptance"],
            evaluation_contract=evaluation_contract,
        )
        model_summary["evaluation_contract"] = dict(evaluation_contract)
        results["summary"] = model_summary
        output_paths = _write_model_outputs(model_output_dir, results)
        model_summary = dict(results["summary"])
        model_summary["model_id"] = model.model_id
        model_summary["output_dir"] = str(model_output_dir)
        model_summary["paths"] = output_paths
        model_summary["targeted_filters"] = describe_abstain_config(abstain_config)
        model_summary["evaluation_costs"] = describe_evaluation_cost_config(cost_config)
        model_outputs.append(sanitize_for_json(model_summary))

        overall_test = model_summary["overall_test_metrics"]
        wfe = model_summary["walk_forward_efficiency"]
        train_window = model_summary.get("train_window", {})
        summary_rows.append(
            {
                "model_id": model.model_id,
                "direction": model.direction,
                "backend": model.backend,
                "fold_count": model_summary["fold_count"],
                "selected_test_trades": overall_test["trade_count"],
                "selected_test_net_pnl_pips": overall_test["total_net_pnl_pips"],
                "selected_test_net_pnl_units": overall_test.get("total_net_pnl_units"),
                "selected_test_expectancy_pips": overall_test["expectancy_pips"],
                "selected_test_expectancy_units": overall_test.get("expectancy_units"),
                "selected_test_profit_factor": overall_test["profit_factor"],
                "selected_test_sharpe": overall_test["monthly_sharpe"],
                "selected_test_sortino": overall_test["monthly_sortino"],
                "selected_test_approx_deflated_sharpe": overall_test["approx_deflated_sharpe"],
                "selected_test_max_drawdown_pct": overall_test.get("max_drawdown_pct"),
                "selected_test_max_profit_retracement_pct": overall_test.get("max_profit_retracement_pct"),
                "performance_unit_label": model_summary["performance_unit_label"],
                "effective_trials": model_summary["effective_trials"],
                "configured_min_train_years": train_window.get("min_train_years"),
                "configured_max_train_years": train_window.get("max_train_years"),
                "configured_min_scheduled_test_start": train_window.get("min_scheduled_test_start"),
                "realized_train_span_years_min": train_window.get("realized_train_span_years_min"),
                "realized_train_span_years_median": train_window.get("realized_train_span_years_median"),
                "realized_train_span_years_max": train_window.get("realized_train_span_years_max"),
                "realized_scheduled_test_start_min": train_window.get("realized_scheduled_test_start_min"),
                "realized_scheduled_test_start_max": train_window.get("realized_scheduled_test_start_max"),
                "overall_wfe": wfe["overall_wfe"],
                "profitable_quarter_share": overall_test["profitable_quarter_share"],
                "positive_composite_expectancy_share": model_summary["positive_composite_expectancy_share"],
                "drawdown_gate_passed": model_summary["paper_trading_gate"]["drawdown_gate_passed"],
                "accepted_for_paper_trading_gate": model_summary["paper_trading_gate"]["accepted"],
                "raw_accepted_for_paper_trading_gate": model_summary["paper_trading_gate"]["accepted_raw"],
                "promotion_quality_gate_eligible": model_summary["paper_trading_gate"][
                    "promotion_quality_gate_eligible"
                ],
                "promotion_quality_disqualifiers": ",".join(
                    model_summary["paper_trading_gate"]["promotion_quality_disqualifiers"]
                ),
                "output_dir": str(model_output_dir),
            }
        )

    model_summary_frame = pd.DataFrame(summary_rows).sort_values("model_id").reset_index(drop=True)
    model_summary_path = output_root / "model_summary.csv"
    model_summary_frame.to_csv(model_summary_path, index=False)

    run_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "regime_report_root": str(regime_report_root),
        "output_root": str(output_root),
        "registry_path": str(Path(registry_path)),
        "model_ids": [model.model_id for model in models],
        "statuses": list(statuses),
        "include_roles": [] if include_roles is None else list(include_roles),
        "min_train_years": int(min_train_years),
        "max_train_years": None if max_train_years is None else float(max_train_years),
        "test_window_months": int(test_window_months),
        "rolling_step_months": int(rolling_step_months),
        "min_scheduled_test_start": min_scheduled_test_start,
        "min_folds": int(min_folds),
        "max_folds": None if max_folds is None else int(max_folds),
        "min_positive_events": int(min_positive_events),
        "min_events_per_month": float(min_events_per_month),
        "min_trades_per_week": float(min_trades_per_week),
        "spread_cost_mode": cost_config.spread_cost_mode,
        "requested_min_folds": None if requested_min_folds is None else int(requested_min_folds),
        "available_min_folds": None if available_min_folds is None else int(available_min_folds),
        "minimum_sharpe": float(minimum_sharpe),
        "maximum_drawdown_pct": float(maximum_drawdown_pct),
        "drawdown_starting_balance_units": float(drawdown_starting_balance_units),
        "minimum_dsr": float(minimum_dsr),
        "dsr_effective_trials_override": None if dsr_effective_trials is None else int(dsr_effective_trials),
        "evaluation_contract": dict(evaluation_contract),
        "evaluation_costs": describe_evaluation_cost_config(cost_config),
        "targeted_filter_preset": targeted_filter_preset,
        "model_summary_path": str(model_summary_path),
        "model_outputs": model_outputs,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(sanitize_for_json(run_summary), indent=2),
        encoding="utf-8",
    )
    return sanitize_for_json(run_summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a walk-forward OTE threshold-policy backtest with realistic frictions."
    )
    parser.add_argument(
        "--regime-report-root",
        type=Path,
        required=True,
        help="Directory produced by run_ote_regime_slice_report.py containing per-model labeled predictions.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to model_testing/reports/ote_policy_backtests/<utc timestamp>.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=REPO_ROOT / "models" / "ote_model_registry_live_multifamily.json",
    )
    parser.add_argument(
        "--status",
        action="append",
        dest="statuses",
        default=None,
        help="Repeat to filter registry models by status. Defaults to active and candidate.",
    )
    parser.add_argument("--model-id", action="append", dest="model_ids", default=None)
    parser.add_argument(
        "--include-role",
        action="append",
        dest="include_roles",
        default=None,
        help="Repeat to restrict registry models by role. Defaults to all roles.",
    )
    parser.add_argument("--threshold", action="append", dest="threshold_grid", type=float, default=None)
    parser.add_argument("--min-train-years", type=int, default=2)
    parser.add_argument(
        "--max-train-years",
        type=float,
        default=None,
        help="Optional rolling train-window cap. For example 3 keeps only the most recent 3 years before each test fold.",
    )
    parser.add_argument("--test-window-months", type=int, default=3)
    parser.add_argument("--rolling-step-months", type=int, default=3)
    parser.add_argument(
        "--min-scheduled-test-start",
        type=str,
        default=None,
        help="Optional earliest scheduled walk-forward test start date, for example 2024-01-01.",
    )
    parser.add_argument("--min-folds", type=int, default=8)
    parser.add_argument(
        "--requested-min-folds",
        type=int,
        default=None,
        help="Original requested fold floor before any wrapper-level relaxation, for audit visibility.",
    )
    parser.add_argument(
        "--available-min-folds",
        type=int,
        default=None,
        help="Minimum fold count discovered by the pre-backtest fold audit, for audit visibility.",
    )
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--min-positive-events", type=int, default=50)
    parser.add_argument("--min-events-per-month", type=float, default=3.0)
    parser.add_argument("--min-trades-per-week", type=float, default=3.0)
    parser.add_argument(
        "--evaluation-contract-mode",
        choices=(PROMOTION_QUALITY_MODE, "research"),
        default=PROMOTION_QUALITY_MODE,
        help="Mark this run as promotion-quality or research-only for gate reporting.",
    )
    parser.add_argument(
        "--promotion-min-trades-per-week-floor",
        type=float,
        default=DEFAULT_PROMOTION_MIN_TRADES_PER_WEEK_FLOOR,
        help="Minimum trades/week floor required for a run to remain promotion-quality eligible.",
    )
    parser.add_argument("--purge-gap-bars", type=int, default=None)
    parser.add_argument(
        "--instrument",
        type=str,
        default="fx",
        help="Evaluation economics profile: fx, es, or 6e.",
    )
    parser.add_argument(
        "--spread-cost-mode",
        choices=sorted(SUPPORTED_SPREAD_COST_MODES),
        default=DEFAULT_SPREAD_COST_MODE,
        help="How to price spread costs: auto, session_schedule, or feature_proxy.",
    )
    parser.add_argument(
        "--minimum-sharpe",
        type=float,
        default=0.80,
        help="Minimum annualized Sharpe ratio required by the acceptance gate.",
    )
    parser.add_argument(
        "--maximum-drawdown-pct",
        type=float,
        default=12.0,
        help="Maximum account-equity drawdown percentage allowed by the advisory drawdown gate.",
    )
    parser.add_argument(
        "--drawdown-starting-balance-units",
        type=float,
        default=10000.0,
        help="Starting balance, in the evaluation unit (ticks or pips), used when computing account-equity drawdown percent.",
    )
    parser.add_argument(
        "--fixed-slippage-units-per-trade",
        "--fixed-slippage-pips-per-trade",
        dest="fixed_slippage_units_per_trade",
        type=float,
        default=None,
        help="Override fixed per-trade slippage in the evaluation unit for the chosen instrument.",
    )
    parser.add_argument(
        "--commission-units-per-trade",
        "--commission-pips-per-trade",
        dest="commission_units_per_trade",
        type=float,
        default=None,
        help="Override per-trade commission in the evaluation unit for the chosen instrument.",
    )
    parser.add_argument(
        "--dsr-effective-trials",
        type=int,
        default=None,
        help="Optional override for the effective number of tested model variants used in the DSR penalty.",
    )
    parser.add_argument(
        "--minimum-dsr",
        type=float,
        default=0.30,
        help="Minimum approximate deflated Sharpe ratio required by the acceptance gate.",
    )
    parser.add_argument(
        "--targeted-filter-preset",
        choices=sorted(TARGETED_FILTER_PRESETS),
        default=None,
        help="Apply named per-model abstain/confidence filters for targeted policy follow-up passes.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = REPO_ROOT / "model_testing" / "reports" / "ote_policy_backtests" / timestamp

    summary = run_policy_backtest(
        regime_report_root=args.regime_report_root,
        output_root=output_root,
        registry_path=args.registry_path,
        model_ids=args.model_ids,
        statuses=args.statuses or ("active", "candidate"),
        include_roles=args.include_roles,
        threshold_grid=args.threshold_grid,
        min_train_years=args.min_train_years,
        max_train_years=args.max_train_years,
        test_window_months=args.test_window_months,
        rolling_step_months=args.rolling_step_months,
        min_scheduled_test_start=args.min_scheduled_test_start,
        min_folds=args.min_folds,
        requested_min_folds=args.requested_min_folds,
        available_min_folds=args.available_min_folds,
        max_folds=args.max_folds,
        min_positive_events=args.min_positive_events,
        min_events_per_month=args.min_events_per_month,
        min_trades_per_week=args.min_trades_per_week,
        evaluation_contract_mode=args.evaluation_contract_mode,
        promotion_min_trades_per_week_floor=args.promotion_min_trades_per_week_floor,
        purge_gap_bars=args.purge_gap_bars,
        instrument=args.instrument,
        spread_cost_mode=args.spread_cost_mode,
        minimum_sharpe=args.minimum_sharpe,
        maximum_drawdown_pct=args.maximum_drawdown_pct,
        drawdown_starting_balance_units=args.drawdown_starting_balance_units,
        fixed_slippage_units_per_trade=args.fixed_slippage_units_per_trade,
        commission_units_per_trade=args.commission_units_per_trade,
        dsr_effective_trials=args.dsr_effective_trials,
        minimum_dsr=args.minimum_dsr,
        targeted_filter_preset=args.targeted_filter_preset,
    )
    print(json.dumps(summary, indent=2))


def _resolve_models(
    registry,
    *,
    model_ids: Sequence[str] | None,
    statuses: Sequence[str],
    include_roles: Sequence[str] | None,
) -> List[OTEModelRecord]:
    if model_ids:
        return [registry.get_model(model_id) for model_id in model_ids]

    allowed_statuses = set(statuses)
    allowed_roles = set(include_roles) if include_roles is not None else None
    return [
        model
        for model in registry.models
        if model.status in allowed_statuses
        and (allowed_roles is None or model.role in allowed_roles)
    ]


def _load_training_summary(artifact_dir: Path) -> Dict[str, object]:
    summary_path = artifact_dir / "training_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_threshold(model: OTEModelRecord, training_summary: Dict[str, object]) -> float:
    if model.global_threshold is not None:
        return float(model.global_threshold)
    if "threshold" in training_summary:
        return float(training_summary["threshold"])
    return 0.5


def _resolve_label_assumption(
    training_summary: Dict[str, object],
    key: str,
    *,
    default: int,
) -> int:
    assumptions = training_summary.get("label_horizon_assumptions", {})
    if isinstance(assumptions, dict) and key in assumptions:
        return int(assumptions[key])

    config = training_summary.get("config", {})
    if isinstance(config, dict) and key in config:
        return int(config[key])

    return default


def _resolve_effective_trials(
    training_summary: Dict[str, object],
    *,
    override: int | None,
) -> int:
    if override is not None:
        return max(int(override), 1)

    config = training_summary.get("config", {})
    if isinstance(config, dict) and "n_trials" in config:
        return max(int(config["n_trials"]), 1)

    if "n_trials" in training_summary:
        return max(int(training_summary["n_trials"]), 1)

    return 1


def _parse_optional_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed


def _load_backtest_prediction_frame(model_report_dir: Path) -> pd.DataFrame:
    oof_path = model_report_dir / "oof_regime_labeled_predictions.csv"
    test_path = model_report_dir / "test_regime_labeled_predictions.csv"
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected labeled OOF/test predictions under {model_report_dir}, but one or both files are missing."
        )

    shared_columns = [
        "source_row_idx",
        "datetime",
        "close",
        "target",
        "year",
        "composite_regime",
        "session_regime",
        "stress_regime",
    ]
    oof_columns = _resolve_prediction_usecols(
        oof_path,
        shared_columns=shared_columns,
        probability_column="oof_calibrated_probability",
    )
    oof_frame = pd.read_csv(oof_path, usecols=oof_columns)
    oof_frame = oof_frame.rename(columns={"oof_calibrated_probability": "model_probability"})
    oof_frame["prediction_source"] = "oof"

    test_columns = _resolve_prediction_usecols(
        test_path,
        shared_columns=shared_columns,
        probability_column="calibrated_probability",
    )
    test_frame = pd.read_csv(test_path, usecols=test_columns)
    test_frame = test_frame.rename(columns={"calibrated_probability": "model_probability"})
    test_frame["prediction_source"] = "test"

    combined = pd.concat([oof_frame, test_frame], ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
    combined["source_row_idx"] = pd.to_numeric(combined["source_row_idx"], errors="coerce").fillna(-1).astype("int64")
    combined = combined.loc[combined["model_probability"].notna()].copy()
    combined = combined.sort_values("datetime").reset_index(drop=True)
    return combined


def _resolve_prediction_usecols(
    path: Path,
    *,
    shared_columns: Sequence[str],
    probability_column: str,
) -> list[str]:
    available_columns = set(pd.read_csv(path, nrows=0).columns)
    return [
        column
        for column in [*shared_columns, probability_column, *BREAKOUT_EVENT_METADATA_COLUMNS]
        if column in available_columns
    ]


def _load_backtest_market_frame(artifact_dir: Path) -> pd.DataFrame:
    prediction_path = _resolve_prediction_path_for_source_lookup(artifact_dir)
    return load_source_frame(
        prediction_path=prediction_path,
        source_columns=("datetime", "close", "approx_spread"),
    )


def _resolve_prediction_path_for_source_lookup(artifact_dir: Path) -> Path:
    for filename in ("test_predictions.csv", "oof_predictions.csv"):
        candidate = artifact_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve a prediction path under {artifact_dir} for source lookup."
    )


def _build_abstain_config(
    model: OTEModelRecord,
    threshold_config: ThresholdSearchConfig,
    *,
    targeted_filter_preset: str | None = None,
) -> HardAbstainConfig:
    return build_targeted_abstain_config(
        model,
        threshold_config,
        targeted_filter_preset=targeted_filter_preset,
    )


def _write_model_outputs(model_output_dir: Path, results: Dict[str, Any]) -> Dict[str, str]:
    output_paths: dict[str, str] = {}
    for key, value in results.items():
        if key == "summary":
            path = model_output_dir / "summary.json"
            path.write_text(json.dumps(sanitize_for_json(value), indent=2), encoding="utf-8")
            output_paths[key] = str(path)
            continue
        if not isinstance(value, pd.DataFrame):
            continue
        path = model_output_dir / f"{key}.csv"
        value.to_csv(path, index=False)
        output_paths[key] = str(path)
    return output_paths


if __name__ == "__main__":
    main()
