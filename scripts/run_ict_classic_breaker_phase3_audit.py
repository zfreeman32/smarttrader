from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ict.pipelines.layout import build_ict_artifact_layout
from ict.reports.classic_breaker_phase3_audit import (
    build_ict_classic_breaker_phase3_audit,
    write_ict_classic_breaker_phase3_audit,
)


DEFAULT_ARTIFACT_RUN_ID = "ict_es_classic_breaker_research_20260904"
DEFAULT_REPORT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ict_classic_breaker_phase3_audits"
DEFAULT_CANONICAL_PHASE03 = (
    REPO_ROOT
    / "artifacts"
    / "ict_es_primary_refresh_20260724_spacing_refit_final_confirm"
    / "phase03_labeling"
)
DEFAULT_REGIME_SOURCE = (
    REPO_ROOT
    / "artifacts"
    / "ict_es_primary_refresh_20260724_spacing_refit_final_confirm"
    / "phase04_prepared"
    / "ict_es_phase06_merged_dataset.csv"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate-audit the isolated ICT classic breaker Phase 3 labels.")
    parser.add_argument("--artifact-run-id", default=DEFAULT_ARTIFACT_RUN_ID)
    parser.add_argument("--artifact-base-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--phase03-dir", type=Path, default=None)
    parser.add_argument("--canonical-phase03-dir", type=Path, default=DEFAULT_CANONICAL_PHASE03)
    parser.add_argument("--regime-source-csv", type=Path, default=DEFAULT_REGIME_SOURCE)
    parser.add_argument("--prepared-root", type=Path, default=None)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--report-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    layout = build_ict_artifact_layout(
        args.artifact_run_id,
        base_dir=args.artifact_base_dir,
        ensure_directories=False,
    )
    phase03_dir = args.phase03_dir or layout.phase03_labeling
    labels_path = phase03_dir / "ict_es_labels.csv"
    events_path = phase03_dir / "ict_es_events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Classic breaker events CSV does not exist: {events_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Classic breaker labels CSV does not exist: {labels_path}")

    canonical_events = None
    canonical_labels = None
    if args.canonical_phase03_dir is not None and args.canonical_phase03_dir.exists():
        canonical_events_path = args.canonical_phase03_dir / "ict_es_events.csv"
        canonical_labels_path = args.canonical_phase03_dir / "ict_es_labels.csv"
        if canonical_events_path.exists():
            canonical_events = pd.read_csv(canonical_events_path)
        if canonical_labels_path.exists():
            canonical_labels = _read_label_columns(canonical_labels_path)

    regime_source = None
    if args.regime_source_csv is not None and args.regime_source_csv.exists():
        regime_source = _read_regime_source(args.regime_source_csv)

    events = pd.read_csv(events_path)
    labels = _read_label_columns(labels_path)
    report_id = args.report_id or f"{args.artifact_run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    report_dir = args.report_root / report_id
    summary, review_rows = build_ict_classic_breaker_phase3_audit(
        events,
        labels=labels,
        canonical_events=canonical_events,
        canonical_labels=canonical_labels,
        regime_source=regime_source,
        prepared_root=args.prepared_root,
    )
    summary["report_id"] = report_id
    summary["artifact_run_id"] = args.artifact_run_id
    summary["source_paths"] = {
        "phase03_dir": str(phase03_dir),
        "events_csv": str(events_path),
        "labels_csv": str(labels_path),
        "canonical_phase03_dir": str(args.canonical_phase03_dir) if args.canonical_phase03_dir is not None else None,
        "regime_source_csv": str(args.regime_source_csv) if args.regime_source_csv is not None else None,
        "prepared_root": str(args.prepared_root) if args.prepared_root is not None else None,
    }
    written = write_ict_classic_breaker_phase3_audit(
        output_dir=report_dir,
        summary=summary,
        review_rows=review_rows,
    )
    manifest = {
        "report_id": report_id,
        "report_dir": str(report_dir),
        "summary_paths": written,
        "source_paths": summary["source_paths"],
    }
    (report_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    headline = summary["sample"]["headline"]
    verdict = summary["verdict"]
    print(f"[ICT classic breaker audit] report_dir={report_dir}")
    print(
        "[ICT classic breaker audit] "
        f"status={verdict['status']} "
        f"events={headline['total_events']} "
        f"usable={headline['usable_events']} "
        f"base_rate={headline['base_rate_pct']:.2f}%"
    )
    print(f"[ICT classic breaker audit] summary_json={written['summary_json']}")
    return 0


def _read_label_columns(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda column: str(column).startswith("label_"))


def _read_regime_source(path: Path) -> pd.DataFrame:
    keep = {
        "datetime",
        "timestamp",
        "ts_event",
        "ema_alignment",
        "adx_14",
        "atr_14",
        "range_shock_20",
        "ema_50",
        "close",
        "close_vs_ema_50_atr",
        "hour",
        "hour_sin",
        "hour_cos",
        "in_asian_session",
        "in_london_session",
        "in_newyork_session",
        "frvp_open_type",
        "frvp_day_type",
    }
    return pd.read_csv(path, usecols=lambda column: str(column).strip().lower() in keep)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
