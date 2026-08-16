from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ict.pipelines.layout import build_ict_artifact_layout
from ict.reports.event_sample_audit import load_ict_phase03_outputs
from ict.reports.leakage_control import (
    build_ict_leakage_control_audit,
    write_ict_leakage_control_audit,
)

DEFAULT_REPORT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ict_leakage_control_audits"
DEFAULT_ARTIFACT_RUN_ID = "ict_es_primary"
DEFAULT_PHASE02_METADATA = REPO_ROOT / "artifacts" / "ict_es_primary" / "phase02_features" / "ict_es_features.metadata.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit ICT event-window leakage control, embargo geometry, and prepared split boundaries."
    )
    parser.add_argument("--artifact-run-id", default=DEFAULT_ARTIFACT_RUN_ID)
    parser.add_argument("--artifact-base-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--phase03-dir", type=Path, default=None)
    parser.add_argument("--prepared-root", type=Path, default=None)
    parser.add_argument("--phase02-metadata", type=Path, default=None)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--report-id", default=None)
    parser.add_argument("--bootstrap-sample-size", type=int, default=256)
    parser.add_argument("--bootstrap-max-events", type=int, default=3000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    layout = build_ict_artifact_layout(
        args.artifact_run_id,
        base_dir=args.artifact_base_dir,
        ensure_directories=True,
    )
    phase03_dir = args.phase03_dir or layout.phase03_labeling
    events, diagnostics, phase03_paths = load_ict_phase03_outputs(phase03_dir)

    prepared_root = args.prepared_root
    if prepared_root is None:
        candidate = layout.phase04_prepared / "prepared"
        prepared_root = candidate if candidate.exists() else None

    phase02_metadata_path = args.phase02_metadata
    if phase02_metadata_path is None:
        candidate = layout.phase02_features / "ict_es_features.metadata.json"
        if candidate.exists():
            phase02_metadata_path = candidate
        elif DEFAULT_PHASE02_METADATA.exists():
            phase02_metadata_path = DEFAULT_PHASE02_METADATA

    phase02_metadata = {}
    if phase02_metadata_path is not None and phase02_metadata_path.exists():
        phase02_metadata = json.loads(phase02_metadata_path.read_text(encoding="utf-8"))

    report_id = args.report_id or f"{args.artifact_run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    report_dir = args.report_root / report_id

    summary = build_ict_leakage_control_audit(
        events,
        phase02_metadata=phase02_metadata,
        prepared_root=prepared_root,
        bootstrap_sample_size=args.bootstrap_sample_size,
        bootstrap_max_events=args.bootstrap_max_events,
    )
    summary["report_id"] = report_id
    summary["artifact_run_id"] = args.artifact_run_id
    summary["source_paths"] = {
        **phase03_paths,
        "prepared_root": str(prepared_root) if prepared_root is not None else None,
        "phase02_metadata": str(phase02_metadata_path) if phase02_metadata_path is not None else None,
    }
    summary["source_diagnostics"] = diagnostics

    written = write_ict_leakage_control_audit(
        output_dir=report_dir,
        summary=summary,
    )
    manifest = {
        "report_id": report_id,
        "report_dir": str(report_dir),
        "summary_paths": written,
        "source_paths": summary["source_paths"],
    }
    (report_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    headline = summary["headline"]
    print(f"[ICT leakage] report_dir={report_dir}")
    print(
        "[ICT leakage] "
        f"targets={headline['targets_covered']} "
        f"prepared_targets={headline['prepared_targets_audited']} "
        f"failures={headline['targets_with_boundary_leakage_or_embargo_failure']} "
        f"embargo={headline['overall_recommended_embargo_bars']}"
    )
    return 0


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
