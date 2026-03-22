from __future__ import annotations

import argparse
from pathlib import Path

from .builder import FeatureDatasetBuilder
from .config import FeatureBuilderConfig
from .preprocessing import FeaturePreprocessingPipeline, PreprocessingConfig
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
    build_parser.set_defaults(handler=_handle_build)

    preprocess_parser = subparsers.add_parser("preprocess", help="Prepare generated features for model training")
    preprocess_parser.add_argument("input", help="Input feature CSV")
    preprocess_parser.add_argument("--output-dir", required=True, help="Directory for prepared datasets and reports")
    preprocess_parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata sidecar path. Defaults to INPUT with .metadata.json",
    )
    preprocess_parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Specific target column(s) to prepare. Repeatable.",
    )
    preprocess_parser.add_argument(
        "--scaler",
        default="none",
        choices=["none", "robust", "standard"],
        help="Optional feature scaler to fit on the training split",
    )
    preprocess_parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.98,
        help="Absolute correlation threshold for collinearity pruning",
    )
    preprocess_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.995,
        help="Absolute correlation threshold for near-duplicate similarity reporting",
    )
    preprocess_parser.add_argument(
        "--max-analysis-rows",
        type=int,
        default=10_000,
        help="Cap for expensive analysis steps like correlation and importance scans",
    )
    preprocess_parser.set_defaults(handler=_handle_preprocess)

    return parser


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

    builder = FeatureDatasetBuilder(config)
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
    config = PreprocessingConfig(
        target_columns=args.target or PreprocessingConfig().target_columns,
        scaler_type=args.scaler,
        correlation_threshold=args.corr_threshold,
        similarity_threshold=args.similarity_threshold,
        max_analysis_rows=args.max_analysis_rows,
    )
    pipeline = FeaturePreprocessingPipeline(config)
    summary = pipeline.run(
        args.input,
        args.output_dir,
        metadata_path=args.metadata,
    )

    print(f"Prepared outputs: {args.output_dir}")
    print(f"Feature pool:     {summary['feature_pool']['encoded_feature_count']:,} encoded features")
    print("Targets:")
    for target_name, result in summary["targets"].items():
        print(
            f"  - {target_name:<12} rows={result['usable_rows']:,} "
            f"features={result['selected_features']:,} "
            f"readiness={result['readiness_score']:>5.1f} ({result['readiness_grade']})"
        )
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
