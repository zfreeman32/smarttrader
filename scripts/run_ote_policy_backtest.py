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

from model_testing.ote_abstain_policy import DEFAULT_SESSION_SPREAD_PIPS, HardAbstainConfig
from model_testing.ote_policy_backtest import WalkForwardBacktestConfig, run_walk_forward_backtest
from model_testing.ote_policy_metrics import sanitize_for_json
from model_testing.ote_prediction_joiner import load_source_frame
from model_testing.ote_threshold_policy import ThresholdSearchConfig
from models.ote_registry_loader import OTEModelRecord, load_ote_model_registry


TARGETED_FILTER_PRESETS: dict[str, dict[str, dict[str, object]]] = {
    "full_run_v2": {
        "long_ote_champion_v1": {
            "abstain_composite_regimes": ("strong_up_high",),
            "minimum_probability_quantile": 0.20,
        },
        "short_ote_candidate_tcn_v2": {},
    },
    "long_breakout_regime_prune_v1": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
        }
    },
    "long_breakout_regime_prune_v2": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
                "strong_up_medium",
            ),
        }
    },
    "long_breakout_regime_prune_v1_q20": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
            "minimum_probability_quantile": 0.20,
        }
    },
    "long_breakout_regime_prune_v1_q20_asia": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
            "abstain_session_regimes": ("asia",),
            "minimum_probability_quantile": 0.20,
        }
    },
    "long_breakout_regime_prune_v1_q25": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
            "minimum_probability_quantile": 0.25,
        }
    },
    "long_breakout_regime_prune_v1_q30": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
            "minimum_probability_quantile": 0.30,
        }
    },
    "long_breakout_regime_prune_v3": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
            "abstain_composite_session_pairs": (("strong_up_medium", "asia"),),
            "abstain_composite_stress_pairs": (("strong_up_medium", "elevated"),),
            "minimum_probability_quantile": 0.20,
        }
    },
    "short_meta_regime_prune_v1": {
        "short_ote_meta_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_medium",
            ),
        }
    },
    "short_meta_regime_prune_q20_v1": {
        "short_ote_meta_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_medium",
            ),
            "minimum_probability_quantile": 0.20,
        }
    },
}


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
    test_window_months: int = 3,
    rolling_step_months: int = 3,
    min_folds: int = 8,
    max_folds: int | None = None,
    min_positive_events: int = 50,
    min_events_per_month: float = 3.0,
    min_trades_per_week: float = 3.0,
    purge_gap_bars: int | None = None,
    fixed_slippage_pips_per_trade: float = 0.3,
    commission_pips_per_trade: float = 0.35,
    targeted_filter_preset: str | None = None,
) -> Dict[str, Any]:
    regime_report_root = Path(regime_report_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

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

        effective_purge_gap = int(
            purge_gap_bars
            if purge_gap_bars is not None
            else max(
                int(_resolve_label_assumption(training_summary, "window_size", default=24)),
                40,
            )
        )
        threshold_config = ThresholdSearchConfig(
            probability_column="model_probability",
            global_threshold=_resolve_threshold(model, training_summary),
            threshold_grid=tuple(threshold_grid) if threshold_grid is not None else ThresholdSearchConfig(
                probability_column="model_probability",
                global_threshold=_resolve_threshold(model, training_summary),
            ).threshold_grid,
            event_tolerance_bars=_resolve_label_assumption(training_summary, "event_tolerance_bars", default=2),
            event_cooldown_bars=_resolve_label_assumption(training_summary, "event_cooldown_bars", default=4),
            min_positive_events=min_positive_events,
            min_events_per_month=min_events_per_month,
            label_max_holding_bars=_resolve_label_assumption(training_summary, "label_max_holding_bars", default=120),
            slippage_spread_multiplier=0.0,
            fixed_slippage_pips_per_trade=fixed_slippage_pips_per_trade,
            commission_pips_per_trade=commission_pips_per_trade,
        )
        backtest_config = WalkForwardBacktestConfig(
            min_train_years=min_train_years,
            test_window_months=test_window_months,
            rolling_step_months=rolling_step_months,
            purge_gap_bars=effective_purge_gap,
            min_folds=min_folds,
            max_folds=max_folds,
            min_trades_per_week=min_trades_per_week,
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
        )

        model_output_dir = output_root / model.model_id
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = _write_model_outputs(model_output_dir, results)
        model_summary = dict(results["summary"])
        model_summary["model_id"] = model.model_id
        model_summary["output_dir"] = str(model_output_dir)
        model_summary["paths"] = output_paths
        model_summary["targeted_filters"] = _describe_abstain_config(abstain_config)
        model_outputs.append(sanitize_for_json(model_summary))

        overall_test = model_summary["overall_test_metrics"]
        wfe = model_summary["walk_forward_efficiency"]
        summary_rows.append(
            {
                "model_id": model.model_id,
                "direction": model.direction,
                "backend": model.backend,
                "fold_count": model_summary["fold_count"],
                "selected_test_trades": overall_test["trade_count"],
                "selected_test_net_pnl_pips": overall_test["total_net_pnl_pips"],
                "selected_test_expectancy_pips": overall_test["expectancy_pips"],
                "selected_test_profit_factor": overall_test["profit_factor"],
                "selected_test_sharpe": overall_test["monthly_sharpe"],
                "selected_test_sortino": overall_test["monthly_sortino"],
                "overall_wfe": wfe["overall_wfe"],
                "profitable_quarter_share": overall_test["profitable_quarter_share"],
                "positive_composite_expectancy_share": model_summary["positive_composite_expectancy_share"],
                "accepted_for_paper_trading_gate": all(bool(value) for value in model_summary["acceptance"].values()),
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
        "test_window_months": int(test_window_months),
        "rolling_step_months": int(rolling_step_months),
        "min_folds": int(min_folds),
        "max_folds": None if max_folds is None else int(max_folds),
        "min_positive_events": int(min_positive_events),
        "min_events_per_month": float(min_events_per_month),
        "min_trades_per_week": float(min_trades_per_week),
        "fixed_slippage_pips_per_trade": float(fixed_slippage_pips_per_trade),
        "commission_pips_per_trade": float(commission_pips_per_trade),
        "targeted_filter_preset": targeted_filter_preset,
        "session_spread_pips": {key: float(value) for key, value in DEFAULT_SESSION_SPREAD_PIPS.items()},
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
        default=REPO_ROOT / "models" / "ote_model_registry_multifamily_candidates.json",
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
    parser.add_argument("--test-window-months", type=int, default=3)
    parser.add_argument("--rolling-step-months", type=int, default=3)
    parser.add_argument("--min-folds", type=int, default=8)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--min-positive-events", type=int, default=50)
    parser.add_argument("--min-events-per-month", type=float, default=3.0)
    parser.add_argument("--min-trades-per-week", type=float, default=3.0)
    parser.add_argument("--purge-gap-bars", type=int, default=None)
    parser.add_argument("--fixed-slippage-pips-per-trade", type=float, default=0.3)
    parser.add_argument("--commission-pips-per-trade", type=float, default=0.35)
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
        test_window_months=args.test_window_months,
        rolling_step_months=args.rolling_step_months,
        min_folds=args.min_folds,
        max_folds=args.max_folds,
        min_positive_events=args.min_positive_events,
        min_events_per_month=args.min_events_per_month,
        min_trades_per_week=args.min_trades_per_week,
        purge_gap_bars=args.purge_gap_bars,
        fixed_slippage_pips_per_trade=args.fixed_slippage_pips_per_trade,
        commission_pips_per_trade=args.commission_pips_per_trade,
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
    oof_frame = pd.read_csv(oof_path, usecols=shared_columns + ["oof_calibrated_probability"])
    oof_frame = oof_frame.rename(columns={"oof_calibrated_probability": "model_probability"})
    oof_frame["prediction_source"] = "oof"

    test_frame = pd.read_csv(test_path, usecols=shared_columns + ["calibrated_probability"])
    test_frame = test_frame.rename(columns={"calibrated_probability": "model_probability"})
    test_frame["prediction_source"] = "test"

    combined = pd.concat([oof_frame, test_frame], ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
    combined["source_row_idx"] = pd.to_numeric(combined["source_row_idx"], errors="coerce").fillna(-1).astype("int64")
    combined = combined.loc[combined["model_probability"].notna()].copy()
    combined = combined.sort_values("datetime").reset_index(drop=True)
    return combined


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
    metadata = model.abstain_policy or {}
    targeted_filters = _resolve_targeted_filters(model.model_id, targeted_filter_preset)
    session_spread_pips = metadata.get("session_spread_pips")
    expected_move_by_regime = metadata.get("expected_move_by_regime")
    return HardAbstainConfig(
        abstain_high_stress=bool(metadata.get("abstain_high_stress", True)),
        abstain_off_hours=bool(metadata.get("abstain_off_hours", True)),
        cooldown_bars=int(metadata.get("cooldown_bars", threshold_config.event_cooldown_bars)),
        minimum_expected_move_to_spread=float(metadata.get("minimum_expected_move_to_spread", 2.0)),
        session_spread_pips=(
            {str(key): float(value) for key, value in dict(session_spread_pips).items()}
            if isinstance(session_spread_pips, dict)
            else threshold_config.session_spread_pips
        ),
        expected_move_by_regime=(
            {str(key): float(value) for key, value in dict(expected_move_by_regime).items()}
            if isinstance(expected_move_by_regime, dict)
            else None
        ),
        abstain_session_regimes=tuple(
            str(value)
            for value in (
                targeted_filters.get("abstain_session_regimes")
                or metadata.get("abstain_session_regimes")
                or ()
            )
        ),
        abstain_composite_regimes=tuple(
            str(value)
            for value in (
                targeted_filters.get("abstain_composite_regimes")
                or metadata.get("abstain_composite_regimes")
                or ()
            )
        ),
        abstain_composite_session_pairs=_coerce_pair_filters(
            targeted_filters.get("abstain_composite_session_pairs")
            or metadata.get("abstain_composite_session_pairs")
            or ()
        ),
        abstain_composite_stress_pairs=_coerce_pair_filters(
            targeted_filters.get("abstain_composite_stress_pairs")
            or metadata.get("abstain_composite_stress_pairs")
            or ()
        ),
        minimum_probability_quantile=(
            float(targeted_filters["minimum_probability_quantile"])
            if "minimum_probability_quantile" in targeted_filters
            else (
                float(metadata["minimum_probability_quantile"])
                if "minimum_probability_quantile" in metadata and metadata["minimum_probability_quantile"] is not None
                else None
            )
        ),
        probability_column="policy_probability",
        signal_candidate_column="policy_signal_candidate",
        position_column=threshold_config.position_column,
    )


def _resolve_targeted_filters(
    model_id: str,
    preset_name: str | None,
) -> Dict[str, object]:
    if preset_name is None:
        return {}
    preset = TARGETED_FILTER_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(f"Unknown targeted_filter_preset: {preset_name!r}")
    return dict(preset.get(model_id, {}))


def _describe_abstain_config(config: HardAbstainConfig) -> Dict[str, object]:
    return {
        "abstain_high_stress": bool(config.abstain_high_stress),
        "abstain_off_hours": bool(config.abstain_off_hours),
        "cooldown_bars": int(config.cooldown_bars),
        "minimum_expected_move_to_spread": float(config.minimum_expected_move_to_spread),
        "abstain_session_regimes": [str(value) for value in config.abstain_session_regimes],
        "abstain_composite_regimes": [str(value) for value in config.abstain_composite_regimes],
        "abstain_composite_session_pairs": [
            [str(composite_regime), str(session_regime)]
            for composite_regime, session_regime in config.abstain_composite_session_pairs
        ],
        "abstain_composite_stress_pairs": [
            [str(composite_regime), str(stress_regime)]
            for composite_regime, stress_regime in config.abstain_composite_stress_pairs
        ],
        "minimum_probability_quantile": None
        if config.minimum_probability_quantile is None
        else float(config.minimum_probability_quantile),
    }


def _coerce_pair_filters(values: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, (list, tuple)):
        return ()

    normalized_pairs: list[tuple[str, str]] = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        normalized_pairs.append((str(value[0]), str(value[1])))
    return tuple(normalized_pairs)


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
