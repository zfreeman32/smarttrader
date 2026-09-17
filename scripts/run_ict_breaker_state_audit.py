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

from ict.reports.breaker_state_audit import (  # noqa: E402
    build_ict_breaker_state_audit,
    load_ict_breaker_feature_surface,
    write_ict_breaker_state_audit,
)

DEFAULT_FEATURE_CSV = REPO_ROOT / "artifacts" / "ict_es_primary" / "phase02_features" / "ict_es_features.csv"
DEFAULT_REPORT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ict_breaker_state_audits"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit causal ICT failed-order-block breaker state breadth.")
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--report-id", default=None)
    parser.add_argument("--min-total-retests", type=int, default=100)
    parser.add_argument("--min-side-retests", type=int, default=25)
    parser.add_argument("--min-retest-years", type=int, default=3)
    parser.add_argument("--top-n-review-rows", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    features = load_ict_breaker_feature_surface(args.feature_csv)
    summary, review_rows = build_ict_breaker_state_audit(
        features,
        min_total_retests=args.min_total_retests,
        min_side_retests=args.min_side_retests,
        min_retest_years=args.min_retest_years,
        top_n_review_rows=args.top_n_review_rows,
    )
    report_id = args.report_id or f"ict_breaker_state_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    report_dir = args.report_root / report_id
    summary["report_id"] = report_id
    summary["source_paths"] = {"feature_csv": str(args.feature_csv)}
    written = write_ict_breaker_state_audit(
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
    readiness = summary["readiness"]
    print(f"[ICT breaker audit] report_dir={report_dir}")
    print(
        "[ICT breaker audit] "
        f"created={headline['total_created_event_bars']} "
        f"retests={headline['total_retest_event_bars']} "
        f"bull_retests={headline['bull_retest_event_bars']} "
        f"bear_retests={headline['bear_retest_event_bars']} "
        f"decision={readiness['decision']}"
    )
    if readiness["reasons"]:
        print(f"[ICT breaker audit] reasons={','.join(readiness['reasons'])}")
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
