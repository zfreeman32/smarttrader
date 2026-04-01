from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Lock
from time import perf_counter

from preprocessing.cli import add_prepare_arguments, run_prepare_command

from .builder import FeatureDatasetBuilder
from .config import FeatureBuilderConfig
from .progress import FeatureBuildProgressEvent
from .registry import FEATURE_REGISTRY
from .strategy_registry import STRATEGY_REGISTRY


def _recipes_dir() -> Path:
    return Path(__file__).resolve().parent / "recipes"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature dataset builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List feature sets and recipes")
    list_parser.add_argument(
        "--strategies",
        action="store_true",
        help="Also print discovered standalone strategy entries.",
    )
    list_parser.set_defaults(handler=_handle_list)

    build_parser = subparsers.add_parser("build", help="Build a feature dataset")
    build_parser.add_argument("input", help="Input CSV with OHLCV data")
    build_parser.add_argument("--output", required=True, help="Output CSV path")
    build_parser.add_argument(
        "--recipe",
        default=str(_recipes_dir() / "ote_base.json"),
        help="Recipe JSON path",
    )
    build_parser.add_argument(
        "--source-timezone",
        default=None,
        help="Timezone for naive input timestamps, for example UTC, America/New_York, or GMT-6.",
    )
    build_parser.add_argument(
        "--canonical-timezone",
        default=None,
        help="Canonical storage timezone for normalized datetimes. Defaults to UTC.",
    )
    build_parser.add_argument(
        "--feature-clock-timezone",
        default=None,
        help="Timezone used for cyclical hour/day features and strategy clock fields.",
    )
    build_parser.add_argument(
        "--market-close-timezone",
        default=None,
        help="Timezone that defines FX market-day and market-week close boundaries.",
    )
    build_parser.add_argument(
        "--feature-set",
        action="append",
        default=None,
        help="Override recipe feature-set order. Repeatable.",
    )
    build_parser.add_argument("--no-lags", action="store_true", help="Disable lag features")
    build_parser.add_argument("--no-rolling", action="store_true", help="Disable rolling stats")
    build_parser.add_argument("--no-zscores", action="store_true", help="Disable rolling z-scores")
    build_parser.add_argument("--no-interactions", action="store_true", help="Disable interaction features")
    build_parser.add_argument(
        "--transform-workers",
        type=int,
        default=None,
        help="Thread count for independent post-feature transforms and strategy execution (default: recipe/config value).",
    )
    build_parser.add_argument(
        "--strategy-timeout-seconds",
        type=float,
        default=None,
        help="Skip a standalone strategy if it does not finish within this many seconds. Use 0 to disable.",
    )
    build_parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Downcast generated numeric feature columns to smaller dtypes before saving.",
    )
    build_parser.add_argument(
        "--strategy",
        action="append",
        default=None,
        help="Add a standalone strategy id to generate strategy-derived feature columns. Repeatable.",
    )
    build_parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Include every discovered standalone strategy id as strategy-derived features.",
    )
    build_parser.add_argument(
        "--skip-strategy-errors",
        action="store_true",
        help="Skip standalone strategies that fail to import or execute and record them in metadata.",
    )
    build_parser.add_argument(
        "--progress",
        action="store_true",
        help="Print live progress for feature sets, transforms, and standalone strategies.",
    )
    build_parser.set_defaults(handler=_handle_build)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Prepare generated features for model training",
    )
    add_prepare_arguments(preprocess_parser)
    preprocess_parser.set_defaults(handler=_handle_preprocess)

    return parser


class _CliProgressReporter:
    def __init__(self) -> None:
        self._started = perf_counter()
        self._lock = Lock()

    def __call__(self, event: FeatureBuildProgressEvent) -> None:
        line = self._format(event)
        if not line:
            return
        with self._lock:
            print(line, file=sys.stdout, flush=True)

    def _format(self, event: FeatureBuildProgressEvent) -> str:
        prefix = f"[{self._format_elapsed(perf_counter() - self._started)}]"

        if event.stage == "build" and event.action == "start":
            return (
                f"{prefix} Build start | rows={event.metrics.get('input_rows', 0):,} "
                f"cols={event.metrics.get('input_columns', 0):,} "
                f"feature_sets={event.metrics.get('feature_sets', 0)} "
                f"transforms={event.metrics.get('transform_jobs', 0)} "
                f"strategies={event.metrics.get('strategy_jobs', 0):,} "
                f"workers={event.metrics.get('transform_workers', 0)}"
            )
        if event.stage == "build" and event.action == "complete":
            return (
                f"{prefix} Build complete | rows={event.metrics.get('output_rows', 0):,} "
                f"cols={event.metrics.get('output_columns', 0):,} "
                f"features={event.metrics.get('feature_columns', 0):,} "
                f"elapsed={self._format_duration(event.elapsed_seconds)}"
            )
        if event.stage == "feature_set" and event.action == "start":
            return f"{prefix} Feature set {event.index}/{event.total} start | {event.name}"
        if event.stage == "feature_set" and event.action == "complete":
            return (
                f"{prefix} Feature set {event.index}/{event.total} done | {event.name} "
                f"| +{event.metrics.get('added_columns', 0)} cols "
                f"| {self._format_duration(event.elapsed_seconds)}"
            )
        if event.stage == "transform" and event.action == "start":
            return f"{prefix} Transform {event.index}/{event.total} start | {event.name}"
        if event.stage == "transform" and event.action == "complete":
            return (
                f"{prefix} Transform {event.index}/{event.total} done | {event.name} "
                f"| +{event.metrics.get('added_columns', 0)} cols "
                f"| {self._format_duration(event.elapsed_seconds)}"
            )
        if event.stage == "strategy_batch" and event.action == "start":
            return (
                f"{prefix} Strategy batch start | requested={event.metrics.get('requested', 0):,} "
                f"workers={event.metrics.get('workers', 0)} "
                f"rows={event.metrics.get('input_rows', 0):,} "
                f"timeout={self._format_timeout(event.metrics.get('timeout_seconds'))}"
            )
        if event.stage == "strategy_batch" and event.action == "complete":
            return (
                f"{prefix} Strategy batch done | built={event.metrics.get('built', 0):,} "
                f"skipped={event.metrics.get('skipped', 0):,} "
                f"cols={event.metrics.get('output_columns', 0):,} "
                f"| {self._format_duration(event.elapsed_seconds)}"
            )
        if event.stage == "strategy" and event.action == "start":
            return f"{prefix} Strategy queued {event.index}/{event.total} | {event.name}"
        if event.stage == "strategy" and event.action == "complete":
            return (
                f"{prefix} Strategy done {event.index}/{event.total} | {event.name} "
                f"| built={event.metrics.get('built', 0):,} skipped={event.metrics.get('skipped', 0):,} "
                f"| +{event.metrics.get('output_columns', 0)} cols "
                f"| {self._format_duration(event.elapsed_seconds)}"
            )
        if event.stage == "strategy" and event.action == "skipped":
            error_message = event.message or "unknown error"
            return (
                f"{prefix} Strategy skipped {event.index}/{event.total} | {event.name} "
                f"| built={event.metrics.get('built', 0):,} skipped={event.metrics.get('skipped', 0):,} "
                f"| {self._format_duration(event.elapsed_seconds)} "
                f"| {error_message}"
            )
        return ""

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        if seconds < 60:
            return f"{seconds:.1f}s"
        total_seconds = int(round(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        return f"{minutes}m {secs}s"

    @staticmethod
    def _format_timeout(seconds: object) -> str:
        if seconds is None:
            return "off"
        try:
            numeric = float(seconds)
        except (TypeError, ValueError):
            return "off"
        if numeric <= 0:
            return "off"
        if numeric < 60:
            return f"{numeric:.0f}s"
        total_seconds = int(round(numeric))
        minutes, secs = divmod(total_seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {secs}s"


def _handle_list(args: argparse.Namespace) -> int:
    print("Available feature sets:")
    for spec in FEATURE_REGISTRY.describe():
        print(f"  - {spec.name:<16} [{spec.category}] {spec.description}")

    strategy_count = len(STRATEGY_REGISTRY.names())
    print(f"\nStandalone strategy entries discovered: {strategy_count}")
    if args.strategies:
        for spec in STRATEGY_REGISTRY.describe():
            detail = spec.module_path.name
            if spec.function_count_in_module > 1:
                detail = f"{detail}::{spec.function_name}"
            print(f"  - {spec.strategy_id:<64} {detail}")
    else:
        print("  Use `python -m features.cli list --strategies` to print them.")
        print("  Use `python -m features.strategy_similarity` to compare them against all_strategies.py.")

    print("\nAvailable recipes:")
    for recipe_path in sorted(_recipes_dir().glob("*.json")):
        print(f"  - {recipe_path}")
    return 0


def _handle_build(args: argparse.Namespace) -> int:
    config = FeatureBuilderConfig.from_recipe(args.recipe)

    if args.feature_set:
        config.feature_sets = args.feature_set
    if args.no_lags:
        config.enable_lags = False
    if args.no_rolling:
        config.enable_rolling_stats = False
    if args.no_zscores:
        config.enable_zscores = False
    if args.no_interactions:
        config.enable_interactions = False
    if args.transform_workers is not None:
        config.transform_workers = max(1, args.transform_workers)
    if args.strategy_timeout_seconds is not None:
        config.strategy_timeout_seconds = None if args.strategy_timeout_seconds <= 0 else args.strategy_timeout_seconds
    if args.optimize_memory:
        config.optimize_feature_dtypes = True
    if args.source_timezone is not None:
        config.source_timezone = args.source_timezone
    if args.canonical_timezone is not None:
        config.canonical_timezone = args.canonical_timezone
    if args.feature_clock_timezone is not None:
        config.feature_clock_timezone = args.feature_clock_timezone
    if args.market_close_timezone is not None:
        config.market_close_timezone = args.market_close_timezone

    selected_strategies = list(config.strategy_ids)
    if args.strategy:
        selected_strategies.extend(args.strategy)
    if args.all_strategies:
        config.include_all_strategies = True
        selected_strategies.extend(STRATEGY_REGISTRY.names())
        config.skip_failed_strategies = True
    if args.skip_strategy_errors:
        config.skip_failed_strategies = True
    if selected_strategies:
        config.strategy_ids = list(dict.fromkeys(selected_strategies))
        if "strategy_signals" not in config.feature_sets:
            config.feature_sets.append("strategy_signals")

    progress_callback = _CliProgressReporter() if args.progress else None
    builder = FeatureDatasetBuilder(config, progress_callback=progress_callback)
    _, metadata, saved_csv, saved_metadata = builder.build_and_save(args.input, args.output)

    print(f"Saved dataset:   {saved_csv}")
    print(f"Saved metadata:  {saved_metadata}")
    print(f"Rows:            {metadata['rows']:,}")
    print(f"Generated feats: {metadata['feature_column_count']:,}")
    print(f"Columns total:   {metadata['columns']:,}")
    strategy_report = metadata.get("feature_set_reports", {}).get("strategy_signals")
    if strategy_report:
        print(f"Strategies req:  {strategy_report['requested']:,}")
        print(f"Strategies built:{strategy_report['built']:,}")
        print(f"Strategies skip: {strategy_report['skipped']:,}")
    return 0


def _handle_preprocess(args: argparse.Namespace) -> int:
    return run_prepare_command(args)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
