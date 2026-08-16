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

from ict.labeling.ict_labeling_engine import ICTLabelingConfig
from ict.pipelines.layout import build_ict_artifact_layout
from ict.reports.event_sample_audit import (
    build_ict_event_sample_audit,
    load_ict_phase03_outputs,
    refresh_ict_phase03_labeling_artifacts,
    write_ict_event_sample_audit,
)

DEFAULT_MARKET_5M = REPO_ROOT / "data" / "futures_data" / "ES-5m-tagged.csv"
DEFAULT_MARKET_1M = REPO_ROOT / "data" / "futures_data" / "ES-1m.csv"
DEFAULT_REPORT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ict_event_sample_audits"
DEFAULT_ARTIFACT_RUN_ID = "ict_es_primary"
DEFAULT_PHASE02_FEATURES_NAME = "ict_es_features.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh and audit the ICT Phase 3 event sample for context and barrier-geometry sanity."
    )
    parser.add_argument("--artifact-run-id", default=DEFAULT_ARTIFACT_RUN_ID)
    parser.add_argument("--artifact-base-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--market-5m", type=Path, default=DEFAULT_MARKET_5M)
    parser.add_argument("--market-1m", type=Path, default=DEFAULT_MARKET_1M)
    parser.add_argument(
        "--setup-feature-csv",
        type=Path,
        default=None,
        help="Optional phase02 ICT feature CSV used to cheaply regenerate setup_output before labeling.",
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--report-id", default=None)
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Reuse the existing phase03_labeling artifacts instead of regenerating them from the current labeler.",
    )
    parser.add_argument("--instrument", default="es")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    layout = build_ict_artifact_layout(
        args.artifact_run_id,
        base_dir=args.artifact_base_dir,
        ensure_directories=True,
    )

    if args.skip_refresh:
        events, diagnostics, phase03_paths = load_ict_phase03_outputs(layout.phase03_labeling)
        refresh_summary = None
    else:
        setup_feature_csv = args.setup_feature_csv
        if setup_feature_csv is not None and not setup_feature_csv.exists():
            raise FileNotFoundError(f"Explicit setup-feature CSV does not exist: {setup_feature_csv}")
        if setup_feature_csv is None:
            candidate = layout.phase02_features / DEFAULT_PHASE02_FEATURES_NAME
            setup_feature_csv = candidate if candidate.exists() else None
        refresh_summary = refresh_ict_phase03_labeling_artifacts(
            phase03_dir=layout.phase03_labeling,
            market_5m_path=args.market_5m,
            market_1m_path=args.market_1m if args.market_1m.exists() else None,
            setup_feature_path=setup_feature_csv if setup_feature_csv is not None and setup_feature_csv.exists() else None,
            config=ICTLabelingConfig(instrument=args.instrument),
        )
        events, diagnostics, phase03_paths = load_ict_phase03_outputs(layout.phase03_labeling)

    report_id = args.report_id or f"{args.artifact_run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    report_dir = args.report_root / report_id
    summary, review_rows = build_ict_event_sample_audit(
        events,
        diagnostics=diagnostics,
    )
    summary["report_id"] = report_id
    summary["artifact_run_id"] = args.artifact_run_id
    summary["source_paths"] = {
        "market_5m": str(args.market_5m),
        "market_1m": str(args.market_1m) if args.market_1m.exists() else None,
        "setup_feature_csv": str(args.setup_feature_csv) if args.setup_feature_csv is not None else None,
        **phase03_paths,
    }
    if refresh_summary is not None:
        summary["refresh_summary"] = refresh_summary

    written = write_ict_event_sample_audit(
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

    headline = summary["headline"]
    reference = summary["reference_geometry"]["overall"]
    continuation = summary["continuation_management"]
    print(f"[ICT audit] report_dir={report_dir}")
    print(
        "[ICT audit] "
        f"usable={headline['usable_events']} "
        f"excluded={headline['excluded_events']} "
        f"base_rate={headline['base_rate_pct']:.2f}% "
        f"raw_reference_valid={reference['raw_reference_valid_pct']:.2f}% "
        f"final_geometry_valid={reference['final_geometry_valid_pct']:.2f}%"
    )
    print(
        "[ICT audit] "
        f"continuation_target_activations={continuation['target_activation_events']} "
        f"review_rows={len(review_rows)}"
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
