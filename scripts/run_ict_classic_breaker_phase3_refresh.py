from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ict.config.setups import ICTSetupDetectorConfig
from ict.labeling.ict_labeling_engine import ICTLabelingConfig
from ict.pipelines.layout import build_ict_artifact_layout
from ict.reports.event_sample_audit import refresh_ict_phase03_labeling_artifacts


DEFAULT_ARTIFACT_RUN_ID = "ict_es_classic_breaker_research_20260904"
DEFAULT_MARKET_5M = REPO_ROOT / "data" / "futures_data" / "ES-5m-tagged.csv"
DEFAULT_MARKET_1M = REPO_ROOT / "data" / "futures_data" / "ES-1m.csv"
DEFAULT_SETUP_FEATURE_CSV = (
    REPO_ROOT
    / "artifacts"
    / "ict_es_primary_breaker_phase02_audit_20260831"
    / "phase02_features"
    / "ict_es_features.csv"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh isolated Phase 3 labels/events for the ICT classic breaker research lane."
    )
    parser.add_argument("--artifact-run-id", default=DEFAULT_ARTIFACT_RUN_ID)
    parser.add_argument("--artifact-base-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--market-5m", type=Path, default=DEFAULT_MARKET_5M)
    parser.add_argument("--market-1m", type=Path, default=DEFAULT_MARKET_1M)
    parser.add_argument("--setup-feature-csv", type=Path, default=DEFAULT_SETUP_FEATURE_CSV)
    parser.add_argument("--instrument", default="es")
    parser.add_argument("--classic-breaker-max-age", type=int, default=120)
    parser.add_argument(
        "--classic-breaker-allow-later-retests",
        action="store_true",
        help="Allow classic-breaker retests beyond the first retest for expanded research samples.",
    )
    parser.add_argument(
        "--classic-breaker-allow-non-rejection-close",
        action="store_true",
        help="Allow breaker-zone retests without the default side-correct rejection close.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.market_5m.exists():
        raise FileNotFoundError(f"5m market CSV does not exist: {args.market_5m}")
    if args.market_1m is not None and not args.market_1m.exists():
        raise FileNotFoundError(f"1m market CSV does not exist: {args.market_1m}")
    if args.setup_feature_csv is not None and not args.setup_feature_csv.exists():
        raise FileNotFoundError(f"Setup feature CSV does not exist: {args.setup_feature_csv}")

    layout = build_ict_artifact_layout(
        args.artifact_run_id,
        base_dir=args.artifact_base_dir,
        ensure_directories=True,
    )
    summary = refresh_ict_phase03_labeling_artifacts(
        phase03_dir=layout.phase03_labeling,
        market_5m_path=args.market_5m,
        market_1m_path=args.market_1m,
        setup_feature_path=args.setup_feature_csv,
        config=ICTLabelingConfig(
            instrument=args.instrument,
            classic_breaker_enabled=True,
        ),
        setup_detector_config=ICTSetupDetectorConfig(
            instrument=args.instrument,
            enabled_setup_types=("classic_breaker",),
            classic_breaker_max_age=args.classic_breaker_max_age,
            classic_breaker_first_retest_only=not bool(args.classic_breaker_allow_later_retests),
            classic_breaker_require_rejection_close=not bool(args.classic_breaker_allow_non_rejection_close),
        ),
    )
    summary["artifact_run_id"] = args.artifact_run_id
    summary["artifact_root"] = str(layout.root)
    summary_path = layout.phase03_labeling / "ict_classic_breaker_refresh_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    diagnostics = summary.get("diagnostics", {})
    print(f"[ICT classic breaker refresh] artifact_root={layout.root}")
    print(f"[ICT classic breaker refresh] phase03_dir={layout.phase03_labeling}")
    print(
        "[ICT classic breaker refresh] "
        f"events={diagnostics.get('total_events_sampled', 0)} "
        f"usable={diagnostics.get('usable_events', 0)} "
        f"classic={diagnostics.get('events_ict_classic_breaker', 0)} "
        f"base_rate={float(diagnostics.get('base_rate_pct', 0.0) or 0.0):.2f}%"
    )
    print(f"[ICT classic breaker refresh] summary_json={summary_path}")
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
