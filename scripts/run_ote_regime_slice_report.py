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

from model_testing.ote_prediction_joiner import join_prediction_file, load_source_columns
from model_testing.ote_regime_labeler import RegimeLabelConfig, label_regimes
from model_testing.ote_regime_slices import (
    DEFAULT_SLICE_FAMILIES,
    RegimeSliceConfig,
    build_regime_slice_report,
    build_regime_winner_table,
    resolve_probability_column,
)
from models.ote_registry_loader import OTEModelRecord, load_ote_model_registry

DEFAULT_SOURCE_COLUMNS = (
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adx_14",
    "ema_alignment",
    "ema_50",
    "close_vs_ema_50_atr",
    "atr_14",
    "range_shock_20",
    "hour",
    "hour_sin",
    "hour_cos",
    "in_asian_session",
    "in_london_session",
    "in_newyork_session",
    "in_london_ny_overlap",
)


def run_regime_slice_report(
    *,
    registry_path: str | Path,
    output_root: str | Path,
    model_ids: Sequence[str] | None = None,
    statuses: Sequence[str] = ("active", "candidate"),
    source_columns: Sequence[str] | None = None,
    bootstrap_iterations: int = 200,
    min_positive_events_for_threshold: int = 50,
    slice_families: Sequence[str] | None = None,
    save_labeled_predictions: bool = True,
) -> Dict[str, object]:
    registry = load_ote_model_registry(registry_path)
    models = _resolve_models(registry, model_ids=model_ids, statuses=statuses)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    requested_source_columns = tuple(dict.fromkeys((source_columns or DEFAULT_SOURCE_COLUMNS)))
    configured_slice_families = tuple(slice_families) if slice_families is not None else None
    label_config = RegimeLabelConfig()
    all_reports: List[pd.DataFrame] = []
    model_outputs: List[Dict[str, object]] = []

    for model in models:
        artifact_dir = model.resolve_artifact_path()
        training_summary = _load_training_summary(artifact_dir)
        model_output_dir = output_root / model.model_id
        model_output_dir.mkdir(parents=True, exist_ok=True)
        source_columns_for_model = _resolve_available_source_columns(
            artifact_dir=artifact_dir,
            requested_source_columns=requested_source_columns,
        )

        split_reports: List[pd.DataFrame] = []
        split_outputs: Dict[str, object] = {}
        for dataset_split, filename in (("oof", "oof_predictions.csv"), ("test", "test_predictions.csv")):
            prediction_path = artifact_dir / filename
            if not prediction_path.exists():
                continue

            joined = join_prediction_file(
                prediction_path,
                source_columns=source_columns_for_model,
            )
            labeled = label_regimes(joined, config=label_config)
            labeled_path = None
            if save_labeled_predictions:
                labeled_path = model_output_dir / f"{dataset_split}_regime_labeled_predictions.csv"
                labeled.to_csv(labeled_path, index=False)

            probability_column = resolve_probability_column(labeled, dataset_split=dataset_split)
            slice_config = RegimeSliceConfig(
                dataset_split=dataset_split,
                probability_column=probability_column,
                threshold=_resolve_threshold(model, training_summary),
                event_tolerance_bars=_resolve_label_assumption(training_summary, "event_tolerance_bars", default=2),
                event_cooldown_bars=_resolve_label_assumption(training_summary, "event_cooldown_bars", default=4),
                min_positive_events_for_threshold=min_positive_events_for_threshold,
                bootstrap_iterations=bootstrap_iterations,
                slice_families=configured_slice_families
                if configured_slice_families is not None
                else DEFAULT_SLICE_FAMILIES,
            )

            report = build_regime_slice_report(
                labeled,
                model_id=model.model_id,
                direction=model.direction,
                backend=model.backend,
                role=model.role,
                status=model.status,
                config=slice_config,
            )
            if not report.empty:
                report.to_csv(model_output_dir / f"{dataset_split}_slice_report.csv", index=False)
                split_reports.append(report)
                all_reports.append(report)

            split_outputs[dataset_split] = {
                "prediction_path": str(prediction_path),
                "labeled_prediction_path": None if labeled_path is None else str(labeled_path),
                "probability_column": probability_column,
                "rows_joined": int(len(joined)),
                "rows_labeled": int(len(labeled)),
                "report_rows": int(len(report)),
            }

        model_report = pd.concat(split_reports, ignore_index=True) if split_reports else pd.DataFrame()
        if not model_report.empty:
            model_report.to_csv(model_output_dir / "slice_report_all_splits.csv", index=False)

        model_outputs.append(
            {
                "model_id": model.model_id,
                "artifact_path": model.artifact_path,
                "output_dir": str(model_output_dir),
                "splits": split_outputs,
            }
        )

    consolidated_report = pd.concat(all_reports, ignore_index=True) if all_reports else pd.DataFrame()
    consolidated_report_path = output_root / "slice_report.csv"
    consolidated_report.to_csv(consolidated_report_path, index=False)

    composite_winners = build_regime_winner_table(consolidated_report, regime_family="composite_regime")
    composite_winners_path = output_root / "composite_bucket_winners.csv"
    composite_winners.to_csv(composite_winners_path, index=False)

    run_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(Path(registry_path)),
        "output_root": str(output_root),
        "model_ids": [model.model_id for model in models],
        "statuses": list(statuses),
        "source_columns": list(requested_source_columns),
        "bootstrap_iterations": int(bootstrap_iterations),
        "min_positive_events_for_threshold": int(min_positive_events_for_threshold),
        "slice_families": list(configured_slice_families or DEFAULT_SLICE_FAMILIES),
        "slice_report_path": str(consolidated_report_path),
        "composite_winners_path": str(composite_winners_path),
        "model_outputs": model_outputs,
    }
    (output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build regime-labeled OTE prediction tables and a regime-sliced performance report."
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=REPO_ROOT / "models" / "ote_model_registry.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to model_testing/reports/ote_regime_slices/<utc timestamp>.",
    )
    parser.add_argument("--model-id", action="append", dest="model_ids", default=None)
    parser.add_argument(
        "--status",
        action="append",
        dest="statuses",
        default=None,
        help="Repeat to filter registry models by status. Defaults to active and candidate.",
    )
    parser.add_argument(
        "--source-column",
        action="append",
        dest="source_columns",
        default=None,
        help="Repeat to override the compact set of source columns included in labeled outputs.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--min-positive-events", type=int, default=50)
    parser.add_argument(
        "--slice-family",
        action="append",
        dest="slice_families",
        default=None,
        help="Repeat to restrict the report to specific regime families.",
    )
    parser.add_argument(
        "--skip-labeled-predictions",
        action="store_true",
        help="Generate reports without writing the joined+labeled prediction tables.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = REPO_ROOT / "model_testing" / "reports" / "ote_regime_slices" / timestamp

    summary = run_regime_slice_report(
        registry_path=args.registry_path,
        output_root=output_root,
        model_ids=args.model_ids,
        statuses=args.statuses or ("active", "candidate"),
        source_columns=args.source_columns,
        bootstrap_iterations=args.bootstrap_iterations,
        min_positive_events_for_threshold=args.min_positive_events,
        slice_families=args.slice_families,
        save_labeled_predictions=not args.skip_labeled_predictions,
    )
    print(json.dumps(summary, indent=2))


def _resolve_models(
    registry,
    *,
    model_ids: Sequence[str] | None,
    statuses: Sequence[str],
) -> List[OTEModelRecord]:
    if model_ids:
        return [registry.get_model(model_id) for model_id in model_ids]

    allowed_statuses = set(statuses)
    return [model for model in registry.models if model.status in allowed_statuses]


def _load_training_summary(artifact_dir: Path) -> Dict[str, object]:
    summary_path = artifact_dir / "training_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_available_source_columns(
    *,
    artifact_dir: Path,
    requested_source_columns: Sequence[str],
) -> List[str]:
    for candidate_name in ("oof_predictions.csv", "test_predictions.csv"):
        prediction_path = artifact_dir / candidate_name
        if prediction_path.exists():
            available = set(load_source_columns(prediction_path=prediction_path))
            return [column for column in requested_source_columns if column in available]
    return list(requested_source_columns)


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


if __name__ == "__main__":
    main()
