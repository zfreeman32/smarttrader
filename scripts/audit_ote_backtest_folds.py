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

from model_testing.ote_policy_backtest import WalkForwardBacktestConfig, build_walk_forward_folds
from models.ote_registry_loader import OTEModelRecord, load_ote_model_registry
from scripts.run_ote_policy_backtest import (
    _load_backtest_prediction_frame,
    _load_training_summary,
    _parse_optional_utc_timestamp,
    _resolve_label_assumption,
)


def audit_backtest_folds(
    *,
    regime_report_root: str | Path,
    registry_path: str | Path,
    model_ids: Sequence[str] | None = None,
    statuses: Sequence[str] = ("active", "candidate"),
    include_roles: Sequence[str] | None = None,
    min_train_years: int = 2,
    max_train_years: float | None = None,
    test_window_months: int = 3,
    rolling_step_months: int = 3,
    min_scheduled_test_start: str | None = None,
    max_folds: int | None = None,
    purge_gap_bars: int | None = None,
    requested_min_folds: int | None = None,
) -> Dict[str, Any]:
    regime_report_root = Path(regime_report_root)
    registry = load_ote_model_registry(registry_path)
    models = _resolve_models(
        registry,
        model_ids=model_ids,
        statuses=statuses,
        include_roles=include_roles,
    )

    rows: list[dict[str, Any]] = []
    insufficient_models: list[str] = []
    available_fold_counts: list[int] = []
    for model in models:
        model_report_dir = regime_report_root / model.model_id
        prediction_frame = _load_backtest_prediction_frame(model_report_dir)
        training_summary = _load_training_summary(model.resolve_artifact_path())
        effective_purge_gap = int(
            purge_gap_bars
            if purge_gap_bars is not None
            else max(
                int(_resolve_label_assumption(training_summary, "window_size", default=24)),
                40,
            )
        )
        backtest_config = WalkForwardBacktestConfig(
            min_train_years=min_train_years,
            max_train_years=max_train_years,
            test_window_months=test_window_months,
            rolling_step_months=rolling_step_months,
            min_scheduled_test_start=_parse_optional_utc_timestamp(min_scheduled_test_start),
            purge_gap_bars=effective_purge_gap,
            min_folds=1,
            max_folds=max_folds,
        )
        folds = build_walk_forward_folds(
            prediction_frame,
            config=backtest_config,
            datetime_column="datetime",
            position_column="source_row_idx",
        )
        fold_count = int(len(folds))
        available_fold_counts.append(fold_count)
        if requested_min_folds is not None and fold_count < int(requested_min_folds):
            insufficient_models.append(model.model_id)

        rows.append(
            {
                "model_id": model.model_id,
                "direction": model.direction,
                "backend": model.backend,
                "available_fold_count": fold_count,
                "prediction_rows": int(len(prediction_frame)),
                "prediction_start": _timestamp_or_none(prediction_frame["datetime"].min()),
                "prediction_end": _timestamp_or_none(prediction_frame["datetime"].max()),
                "effective_purge_gap_bars": effective_purge_gap,
            }
        )

    min_available_fold_count = min(available_fold_counts) if available_fold_counts else None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "regime_report_root": str(regime_report_root),
        "registry_path": str(Path(registry_path)),
        "model_ids": [model.model_id for model in models],
        "requested_min_folds": None if requested_min_folds is None else int(requested_min_folds),
        "min_available_fold_count": min_available_fold_count,
        "insufficient_models": insufficient_models,
        "model_summaries": rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit how many walk-forward folds are available per model before policy backtesting."
    )
    parser.add_argument("--regime-report-root", type=Path, required=True)
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=REPO_ROOT / "models" / "ote_model_registry_live_multifamily.json",
    )
    parser.add_argument("--status", action="append", dest="statuses", default=None)
    parser.add_argument("--model-id", action="append", dest="model_ids", default=None)
    parser.add_argument("--include-role", action="append", dest="include_roles", default=None)
    parser.add_argument("--min-train-years", type=int, default=2)
    parser.add_argument("--max-train-years", type=float, default=None)
    parser.add_argument("--test-window-months", type=int, default=3)
    parser.add_argument("--rolling-step-months", type=int, default=3)
    parser.add_argument("--min-scheduled-test-start", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--purge-gap-bars", type=int, default=None)
    parser.add_argument("--requested-min-folds", type=int, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = audit_backtest_folds(
        regime_report_root=args.regime_report_root,
        registry_path=args.registry_path,
        model_ids=args.model_ids,
        statuses=args.statuses or ("active", "candidate"),
        include_roles=args.include_roles,
        min_train_years=args.min_train_years,
        max_train_years=args.max_train_years,
        test_window_months=args.test_window_months,
        rolling_step_months=args.rolling_step_months,
        min_scheduled_test_start=args.min_scheduled_test_start,
        max_folds=args.max_folds,
        purge_gap_bars=args.purge_gap_bars,
        requested_min_folds=args.requested_min_folds,
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


def _timestamp_or_none(value: object) -> str | None:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


if __name__ == "__main__":
    main()
