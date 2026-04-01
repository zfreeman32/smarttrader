from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_testing.ote_abstain_policy import HardAbstainConfig
from model_testing.ote_prediction_joiner import load_source_frame
from model_testing.ote_regime_slices import resolve_probability_column
from model_testing.ote_threshold_policy import (
    ThresholdSearchConfig,
    attach_forward_trade_outcomes,
    evaluate_policy_variants,
    search_thresholds_by_composite_regime,
)
from models.ote_registry_loader import OTEModelRecord, load_ote_model_registry


def run_threshold_policy_search(
    *,
    regime_report_root: str | Path,
    output_root: str | Path,
    registry_path: str | Path,
    model_ids: Sequence[str] | None = None,
    include_roles: Sequence[str] = ("champion",),
    threshold_grid: Sequence[float] | None = None,
    min_positive_events: int = 50,
    min_events_per_month: float = 3.0,
    min_trades_per_week: float = 3.0,
    write_policy_decisions: bool = False,
    write_registry_policies: bool = False,
) -> Dict[str, object]:
    regime_report_root = Path(regime_report_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    registry = load_ote_model_registry(registry_path)
    models = _resolve_models(registry, model_ids=model_ids, include_roles=include_roles)

    all_policy_tables: List[pd.DataFrame] = []
    all_policy_evaluations: List[pd.DataFrame] = []
    model_outputs: List[Dict[str, object]] = []
    registry_updates: Dict[str, Dict[str, object]] = {}

    for model in models:
        model_report_dir = regime_report_root / model.model_id
        oof_path = model_report_dir / "oof_regime_labeled_predictions.csv"
        test_path = model_report_dir / "test_regime_labeled_predictions.csv"
        if not oof_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Expected labeled OOF/test predictions under {model_report_dir}, but one or both files are missing."
            )

        oof_frame = pd.read_csv(oof_path)
        test_frame = pd.read_csv(test_path)
        training_summary = _load_training_summary(model.resolve_artifact_path())
        probability_column = resolve_probability_column(oof_frame, dataset_split="oof")
        config = ThresholdSearchConfig(
            probability_column=probability_column,
            global_threshold=_resolve_threshold(model, training_summary),
            threshold_grid=tuple(threshold_grid) if threshold_grid is not None else ThresholdSearchConfig(
                probability_column=probability_column,
                global_threshold=_resolve_threshold(model, training_summary),
            ).threshold_grid,
            event_tolerance_bars=_resolve_label_assumption(training_summary, "event_tolerance_bars", default=2),
            event_cooldown_bars=_resolve_label_assumption(training_summary, "event_cooldown_bars", default=4),
            min_positive_events=min_positive_events,
            min_events_per_month=min_events_per_month,
            label_max_holding_bars=_resolve_label_assumption(
                training_summary,
                "label_max_holding_bars",
                default=120,
            ),
        )
        market_frame = _load_policy_market_frame(model.resolve_artifact_path())
        oof_frame = attach_forward_trade_outcomes(
            oof_frame,
            market_frame=market_frame,
            direction=model.direction,
            config=config,
        )
        test_frame = attach_forward_trade_outcomes(
            test_frame,
            market_frame=market_frame,
            direction=model.direction,
            config=config,
        )

        policy_table, global_result = search_thresholds_by_composite_regime(
            oof_frame,
            direction=model.direction,
            config=config,
        )
        policy_table.insert(0, "model_id", model.model_id)
        policy_table.insert(1, "direction", model.direction)
        policy_table.insert(2, "backend", model.backend)

        abstain_config = HardAbstainConfig(
            cooldown_bars=config.event_cooldown_bars,
            probability_column="policy_probability",
            signal_candidate_column="policy_signal_candidate",
            position_column=config.position_column,
        )
        evaluation = evaluate_policy_variants(
            oof_frame=oof_frame,
            test_frame=test_frame,
            policy_table=policy_table,
            config=config,
            abstain_config=abstain_config,
        )
        evaluation.insert(0, "model_id", model.model_id)
        evaluation.insert(1, "direction", model.direction)
        evaluation.insert(2, "backend", model.backend)
        policy_selection = _select_qualified_policy(
            evaluation,
            min_trades_per_week=min_trades_per_week,
        )

        model_output_dir = output_root / model.model_id
        model_output_dir.mkdir(parents=True, exist_ok=True)
        policy_table_path = model_output_dir / "policy_table.csv"
        evaluation_path = model_output_dir / "policy_evaluation.csv"
        policy_table.to_csv(policy_table_path, index=False)
        evaluation.to_csv(evaluation_path, index=False)

        decision_outputs: Dict[str, str] = {}
        if write_policy_decisions:
            oof_decisions = _build_policy_decisions_frame(
                frame=oof_frame,
                policy_table=policy_table,
                config=config,
                abstain_config=abstain_config,
            )
            test_decisions = _build_policy_decisions_frame(
                frame=test_frame,
                policy_table=policy_table,
                config=ThresholdSearchConfig(
                    probability_column=resolve_probability_column(test_frame, dataset_split="test"),
                    global_threshold=config.global_threshold,
                    composite_column=config.composite_column,
                    target_column=config.target_column,
                    year_column=config.year_column,
                    datetime_column=config.datetime_column,
                    position_column=config.position_column,
                    threshold_grid=config.threshold_grid,
                    event_tolerance_bars=config.event_tolerance_bars,
                    event_cooldown_bars=config.event_cooldown_bars,
                    min_positive_events=config.min_positive_events,
                    min_events_per_month=config.min_events_per_month,
                ),
                abstain_config=abstain_config,
            )
            oof_decisions_path = model_output_dir / "oof_policy_decisions.csv"
            test_decisions_path = model_output_dir / "test_policy_decisions.csv"
            oof_decisions.to_csv(oof_decisions_path, index=False)
            test_decisions.to_csv(test_decisions_path, index=False)
            decision_outputs = {
                "oof_policy_decisions": str(oof_decisions_path),
                "test_policy_decisions": str(test_decisions_path),
            }

        all_policy_tables.append(policy_table)
        all_policy_evaluations.append(evaluation)
        regime_threshold_map = {
            str(row["composite_regime"]): float(row["threshold"])
            for _, row in policy_table.iterrows()
            if pd.notna(row.get("threshold"))
        }
        abstain_policy_metadata = _build_abstain_policy_metadata(
            policy_selection=policy_selection,
            abstain_config=abstain_config,
        )
        registry_updates[model.model_id] = {
            "global_threshold": float(global_result["threshold"]),
            "regime_thresholds": regime_threshold_map,
            "abstain_policy": abstain_policy_metadata,
        }
        model_outputs.append(
            {
                "model_id": model.model_id,
                "policy_table_path": str(policy_table_path),
                "policy_evaluation_path": str(evaluation_path),
                "global_threshold": float(global_result["threshold"]),
                "qualified_policy_names": list(policy_selection["qualified_policy_names"]),
                "selected_policy_name": policy_selection["selected_policy_name"],
                "selected_policy_metrics": policy_selection["selected_policy_metrics"],
                "decision_outputs": decision_outputs,
            }
        )

    consolidated_policy_table = pd.concat(all_policy_tables, ignore_index=True) if all_policy_tables else pd.DataFrame()
    consolidated_evaluation = pd.concat(all_policy_evaluations, ignore_index=True) if all_policy_evaluations else pd.DataFrame()
    policy_table_path = output_root / "policy_table.csv"
    evaluation_path = output_root / "policy_evaluation.csv"
    consolidated_policy_table.to_csv(policy_table_path, index=False)
    consolidated_evaluation.to_csv(evaluation_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "regime_report_root": str(regime_report_root),
        "output_root": str(output_root),
        "registry_path": str(Path(registry_path)),
        "model_ids": [model.model_id for model in models],
        "include_roles": list(include_roles),
        "min_positive_events": int(min_positive_events),
        "min_events_per_month": float(min_events_per_month),
        "min_trades_per_week": float(min_trades_per_week),
        "policy_table_path": str(policy_table_path),
        "policy_evaluation_path": str(evaluation_path),
        "model_outputs": model_outputs,
    }
    if write_registry_policies:
        _write_registry_policy_updates(
            registry_path=Path(registry_path),
            updates=registry_updates,
        )
        summary["registry_updates_written"] = True
        summary["registry_updated_model_ids"] = sorted(registry_updates)
    else:
        summary["registry_updates_written"] = False
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search composite-regime thresholds and evaluate abstain-aware policy variants."
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
        help="Defaults to model_testing/reports/ote_threshold_policies/<utc timestamp>.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=REPO_ROOT / "models" / "ote_model_registry.json",
    )
    parser.add_argument("--model-id", action="append", dest="model_ids", default=None)
    parser.add_argument(
        "--include-role",
        action="append",
        dest="include_roles",
        default=None,
        help="Repeat to include challenger/benchmark entries. Defaults to champion only.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        dest="threshold_grid",
        type=float,
        default=None,
        help="Repeat to override the default threshold grid.",
    )
    parser.add_argument("--min-positive-events", type=int, default=50)
    parser.add_argument("--min-events-per-month", type=float, default=3.0)
    parser.add_argument("--min-trades-per-week", type=float, default=3.0)
    parser.add_argument("--write-policy-decisions", action="store_true")
    parser.add_argument(
        "--write-registry-policies",
        action="store_true",
        help="Write regime_thresholds and qualifying abstain-policy metadata back into the registry.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = REPO_ROOT / "model_testing" / "reports" / "ote_threshold_policies" / timestamp

    summary = run_threshold_policy_search(
        regime_report_root=args.regime_report_root,
        output_root=output_root,
        registry_path=args.registry_path,
        model_ids=args.model_ids,
        include_roles=args.include_roles or ("champion",),
        threshold_grid=args.threshold_grid,
        min_positive_events=args.min_positive_events,
        min_events_per_month=args.min_events_per_month,
        min_trades_per_week=args.min_trades_per_week,
        write_policy_decisions=args.write_policy_decisions,
        write_registry_policies=args.write_registry_policies,
    )
    print(json.dumps(summary, indent=2))


def _resolve_models(
    registry,
    *,
    model_ids: Sequence[str] | None,
    include_roles: Sequence[str],
) -> List[OTEModelRecord]:
    if model_ids:
        return [registry.get_model(model_id) for model_id in model_ids]

    allowed_roles = set(include_roles)
    return [
        model
        for model in registry.models
        if model.status == "active" and model.role in allowed_roles
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


def _build_policy_decisions_frame(
    *,
    frame: pd.DataFrame,
    policy_table: pd.DataFrame,
    config: ThresholdSearchConfig,
    abstain_config: HardAbstainConfig,
) -> pd.DataFrame:
    from model_testing.ote_threshold_policy import apply_threshold_policy

    thresholded = apply_threshold_policy(
        frame,
        config=config,
        policy_table=policy_table,
        policy_name="regime_threshold_plus_abstain",
        fallback_threshold=config.global_threshold,
    )
    return apply_abstain_policy(thresholded, config=abstain_config)


def _load_policy_market_frame(artifact_dir: Path) -> pd.DataFrame:
    prediction_path = _resolve_prediction_path_for_source_lookup(artifact_dir)
    return load_source_frame(
        prediction_path=prediction_path,
        source_columns=("close", "approx_spread"),
    )


def _resolve_prediction_path_for_source_lookup(artifact_dir: Path) -> Path:
    for filename in ("test_predictions.csv", "oof_predictions.csv"):
        candidate = artifact_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve a prediction path under {artifact_dir} for source lookup."
    )


def _select_qualified_policy(
    evaluation: pd.DataFrame,
    *,
    min_trades_per_week: float,
) -> Dict[str, object]:
    test_rows = evaluation.loc[evaluation["dataset_split"] == "test"].copy()
    if test_rows.empty:
        return {
            "qualified_policy_names": [],
            "selected_policy_name": None,
            "selected_policy_metrics": {},
        }

    baseline = test_rows.loc[test_rows["policy_name"] == "global_threshold"]
    if baseline.empty:
        raise ValueError("Policy evaluation is missing the global_threshold test row.")
    baseline_row = baseline.iloc[0]

    qualified_rows = test_rows.loc[
        (test_rows["policy_name"] != "global_threshold")
        & (test_rows["event_f05"] > float(baseline_row["event_f05"]))
        & (
            test_rows["post_cost_expectancy_pips"]
            > float(baseline_row["post_cost_expectancy_pips"])
        )
        & (test_rows["trades_per_week"] >= float(min_trades_per_week))
    ].copy()

    selected_policy_name = None
    selected_policy_metrics: Dict[str, object] = {}
    if not qualified_rows.empty:
        qualified_rows = qualified_rows.sort_values(
            ["post_cost_expectancy_pips", "event_f05", "net_pnl_pips"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        selected = qualified_rows.iloc[0]
        selected_policy_name = str(selected["policy_name"])
        selected_policy_metrics = {
            "event_f05": float(selected["event_f05"]),
            "post_cost_expectancy_pips": float(selected["post_cost_expectancy_pips"]),
            "net_pnl_pips": float(selected["net_pnl_pips"]),
            "trades_per_week": float(selected["trades_per_week"]),
        }

    return {
        "qualified_policy_names": qualified_rows["policy_name"].astype(str).tolist(),
        "selected_policy_name": selected_policy_name,
        "selected_policy_metrics": selected_policy_metrics,
    }


def _build_abstain_policy_metadata(
    *,
    policy_selection: Dict[str, object],
    abstain_config: HardAbstainConfig,
) -> Dict[str, object] | None:
    selected_policy_name = policy_selection.get("selected_policy_name")
    if selected_policy_name not in {"global_threshold_plus_abstain", "regime_threshold_plus_abstain"}:
        return None

    return {
        "policy_name": str(selected_policy_name),
        "cooldown_bars": int(abstain_config.cooldown_bars),
        "abstain_high_stress": bool(abstain_config.abstain_high_stress),
        "abstain_off_hours": bool(abstain_config.abstain_off_hours),
        "minimum_expected_move_to_spread": float(abstain_config.minimum_expected_move_to_spread),
        "session_spread_pips": {
            str(key): float(value)
            for key, value in dict(abstain_config.session_spread_pips).items()
        },
        "abstain_session_regimes": [str(value) for value in abstain_config.abstain_session_regimes],
        "abstain_composite_regimes": [str(value) for value in abstain_config.abstain_composite_regimes],
        "minimum_probability_quantile": abstain_config.minimum_probability_quantile,
        "expected_move_by_regime": None
        if abstain_config.expected_move_by_regime is None
        else {
            str(key): float(value)
            for key, value in dict(abstain_config.expected_move_by_regime).items()
        },
        "selection_rule": "beats global_threshold on test event_f05 and post_cost_expectancy_pips while maintaining min_trades_per_week",
        "test_metrics": dict(policy_selection.get("selected_policy_metrics", {})),
    }


def _write_registry_policy_updates(
    *,
    registry_path: Path,
    updates: Dict[str, Dict[str, object]],
) -> None:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    for record in payload.get("models", []):
        model_id = record.get("model_id")
        if model_id not in updates:
            continue
        record["global_threshold"] = updates[model_id]["global_threshold"]
        record["regime_thresholds"] = updates[model_id]["regime_thresholds"]
        record["abstain_policy"] = updates[model_id]["abstain_policy"]
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
