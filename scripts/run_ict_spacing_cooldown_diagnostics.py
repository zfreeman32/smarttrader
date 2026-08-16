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
from ict.reports.spacing_cooldown_diagnostics import (
    build_ict_spacing_cooldown_diagnostics,
    write_ict_spacing_cooldown_diagnostics,
)

DEFAULT_REPORT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ict_spacing_cooldown_diagnostics"
DEFAULT_ARTIFACT_RUN_ID = "ict_es_primary"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize realized ICT barrier timing and clustering to refit setup spacing."
    )
    parser.add_argument("--artifact-run-id", default=DEFAULT_ARTIFACT_RUN_ID)
    parser.add_argument("--artifact-base-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--phase03-dir", type=Path, default=None)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--report-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.phase03_dir is None:
        layout = build_ict_artifact_layout(
            args.artifact_run_id,
            base_dir=args.artifact_base_dir,
            ensure_directories=True,
        )
        phase03_dir = layout.phase03_labeling
    else:
        phase03_dir = args.phase03_dir

    events, diagnostics, phase03_paths = load_ict_phase03_outputs(phase03_dir)
    report_id = args.report_id or f"{args.artifact_run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    report_dir = args.report_root / report_id

    summary, setup_frame = build_ict_spacing_cooldown_diagnostics(
        events,
        diagnostics=diagnostics,
    )
    summary["report_id"] = report_id
    summary["artifact_run_id"] = args.artifact_run_id
    summary["source_paths"] = phase03_paths

    written = write_ict_spacing_cooldown_diagnostics(
        output_dir=report_dir,
        summary=summary,
        setup_frame=setup_frame,
    )
    manifest = {
        "report_id": report_id,
        "report_dir": str(report_dir),
        "summary_paths": written,
        "source_paths": summary["source_paths"],
    }
    (report_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    headline = summary["headline"]
    print(f"[ICT spacing] report_dir={report_dir}")
    print(
        "[ICT spacing] "
        f"usable={headline['usable_events']} "
        f"setup_types={headline['setup_types_covered']} "
        f"changed={headline['changed_setup_count']}"
    )
    changed = summary["changed_setups"]
    if not changed:
        print("[ICT spacing] no spacing changes recommended")
        return 0
    for row in changed:
        print(
            "[ICT spacing] "
            f"{row['setup_type']}: "
            f"{row['current_spacing_bars']} -> {row['recommended_spacing_bars']} "
            f"({row['recommendation_status']})"
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
