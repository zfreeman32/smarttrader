from __future__ import annotations

import argparse
from pathlib import Path

from ote_live.env import load_repo_env
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.features.parity_replay import replay_feature_parity


def build_parser() -> argparse.ArgumentParser:
    load_repo_env()
    parser = argparse.ArgumentParser(
        description="Replay historical bars through the live subset feature engine and compare selected features to the canonical offline feature dataset.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        dest="manifest_paths",
        required=True,
        help="Path to a single-model live_runtime_manifest.json file. Repeatable.",
    )
    parser.add_argument("--evaluation-bars", type=int, default=25)
    parser.add_argument("--rolling-window-bars", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--max-mismatch-examples", type=int, default=10)
    return parser


def main() -> int:
    load_repo_env()
    parser = build_parser()
    args = parser.parse_args()

    manifests = [
        LiveRuntimeManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
        for path in args.manifest_paths
    ]
    report = replay_feature_parity(
        manifests,
        evaluation_bars=args.evaluation_bars,
        rolling_window_bars=args.rolling_window_bars,
        atol=args.atol,
        rtol=args.rtol,
        max_mismatch_examples=args.max_mismatch_examples,
    )

    print(f"asset={report.asset} timeframe={report.timeframe}")
    print(f"models={', '.join(report.model_ids)}")
    print(
        f"rows_evaluated={report.rows_evaluated} "
        f"cells={report.compared_cells} "
        f"matched={report.matched_cells} "
        f"mismatched={report.mismatched_cells} "
        f"ratio={report.matched_cell_ratio:.6f}"
    )
    if report.max_abs_diff is not None:
        print(f"max_abs_diff={report.max_abs_diff:.12g}")
    print(f"missing_reference_rows={report.missing_reference_rows}")

    if report.sample_mismatches:
        print("sample_mismatches:")
        for mismatch in report.sample_mismatches:
            print(
                f"  {mismatch.timestamp.isoformat()} | {mismatch.feature_name} | "
                f"expected={mismatch.expected_value!r} actual={mismatch.actual_value!r} "
                f"abs_diff={mismatch.abs_diff!r}"
            )

    return 1 if report.mismatched_cells or report.missing_reference_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
