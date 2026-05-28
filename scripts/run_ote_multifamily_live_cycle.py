from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ote_policy_backtest import run_policy_backtest
from scripts.run_ote_regime_slice_report import run_regime_slice_report
from scripts.run_ote_threshold_policy_search import run_threshold_policy_search
DEFAULT_REGISTRY_PATH = REPO_ROOT / "models" / "ote_model_registry_live_multifamily.json"
DEFAULT_REGIME_OUTPUT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ote_regime_slices" / "multifamily_live_v1"
DEFAULT_THRESHOLD_OUTPUT_ROOT = (
    REPO_ROOT / "model_testing" / "reports" / "ote_threshold_policies" / "multifamily_live_v1"
)
DEFAULT_BACKTEST_OUTPUT_ROOT = (
    REPO_ROOT / "model_testing" / "reports" / "ote_policy_backtests" / "multifamily_live_v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the multifamily OTE reevaluation cycle into stable live-facing report roots."
    )
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--regime-output-root", type=Path, default=DEFAULT_REGIME_OUTPUT_ROOT)
    parser.add_argument("--threshold-output-root", type=Path, default=DEFAULT_THRESHOLD_OUTPUT_ROOT)
    parser.add_argument("--backtest-output-root", type=Path, default=DEFAULT_BACKTEST_OUTPUT_ROOT)
    parser.add_argument("--skip-regime", action="store_true")
    parser.add_argument("--skip-threshold", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--min-folds", type=int, default=5)
    parser.add_argument("--min-trades-per-week", type=float, default=3.0)
    parser.add_argument("--write-policy-decisions", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    summaries: dict[str, object] = {}
    if not args.skip_regime:
        summaries["regime"] = run_regime_slice_report(
            registry_path=args.registry_path,
            output_root=args.regime_output_root,
            statuses=("active", "candidate"),
        )

    if not args.skip_threshold:
        summaries["threshold"] = run_threshold_policy_search(
            regime_report_root=args.regime_output_root,
            output_root=args.threshold_output_root,
            registry_path=args.registry_path,
            statuses=("active", "candidate"),
            write_policy_decisions=bool(args.write_policy_decisions),
        )

    if not args.skip_backtest:
        summaries["backtest"] = run_policy_backtest(
            regime_report_root=args.regime_output_root,
            output_root=args.backtest_output_root,
            registry_path=args.registry_path,
            statuses=("active", "candidate"),
            min_folds=args.min_folds,
            min_trades_per_week=args.min_trades_per_week,
        )

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
