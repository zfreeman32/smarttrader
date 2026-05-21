from __future__ import annotations

import argparse
from typing import Sequence

from .backend_attribution import (
    BackendAttributionConfig,
    SUPPORTED_BACKENDS,
    run_backend_attribution,
)
from .config import PreprocessingConfig
from .pipeline import FeaturePreprocessingPipeline


def add_prepare_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("input", help="Input feature CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared datasets and reports")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata sidecar path. Defaults to INPUT with .metadata.json",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Specific target column(s) to prepare. Repeatable.",
    )
    parser.add_argument(
        "--scaler",
        default="none",
        choices=["none", "robust", "standard"],
        help="Optional feature scaler to fit on the training split",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.98,
        help="Absolute correlation threshold for collinearity pruning",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.995,
        help="Absolute correlation threshold for near-duplicate similarity reporting",
    )
    parser.add_argument(
        "--max-analysis-rows",
        type=int,
        default=100_000,
        help="Cap for expensive analysis steps like correlation and importance scans",
    )
    return parser


def build_config_from_args(args: argparse.Namespace) -> PreprocessingConfig:
    defaults = PreprocessingConfig()
    return PreprocessingConfig(
        target_columns=args.target or defaults.target_columns,
        scaler_type=args.scaler,
        correlation_threshold=args.corr_threshold,
        similarity_threshold=args.similarity_threshold,
        max_analysis_rows=args.max_analysis_rows,
    )


def run_prepare_command(args: argparse.Namespace) -> int:
    config = build_config_from_args(args)
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


def add_backend_attribution_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--prepared-root",
        required=True,
        help="Prepared dataset root containing one directory per target",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Specific prepared target directory to analyze. Repeatable.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=list(SUPPORTED_BACKENDS),
        default=None,
        help="Backend attribution variant to build. Repeatable.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=160,
        help="Top base-ranked prepared features to include in each backend proxy model",
    )
    parser.add_argument(
        "--attribution-max-rows",
        type=int,
        default=100_000,
        help="Cap the recent validation rows used for backend attribution. Use 0 for all rows.",
    )
    parser.add_argument(
        "--base-weight",
        type=float,
        default=0.20,
        help="Weight for the base preprocessing ranking in the merged score",
    )
    parser.add_argument(
        "--shap-weight",
        type=float,
        default=0.55,
        help="Weight for the backend attribution overall-importance component",
    )
    parser.add_argument(
        "--shap-positive-weight",
        type=float,
        default=0.25,
        help="Weight for the backend attribution positive-class importance component",
    )
    parser.add_argument(
        "--attribution-floor-fraction",
        type=float,
        default=0.15,
        help="Drop features below this fraction of the max mean_abs_shap_all value (0.15 = 15%%)",
    )
    parser.add_argument(
        "--attribution-cumulative-importance",
        type=float,
        default=0.90,
        help="Keep features until this cumulative mean_abs_shap_all share is reached after floor filtering",
    )
    parser.add_argument("--xgb-num-boost-round", type=int, default=300)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=40)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-min-child-weight", type=float, default=5.0)
    parser.add_argument("--xgb-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.85)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--attribution-batch-size", type=int, default=128)
    parser.add_argument("--torch-epochs", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--torch-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--scale-quantile-low", type=float, default=5.0)
    parser.add_argument("--scale-quantile-high", type=float, default=95.0)
    parser.add_argument("--scale-clip", type=float, default=8.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--top-n-features", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--nthread", type=int, default=0)
    parser.add_argument("--use-amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=False)
    return parser


def build_backend_attribution_config_from_args(args: argparse.Namespace) -> BackendAttributionConfig:
    defaults = BackendAttributionConfig()
    return BackendAttributionConfig(
        prepared_root=args.prepared_root,
        targets=args.target or defaults.targets,
        backends=args.backend or defaults.backends,
        max_features=args.max_features,
        attribution_max_rows=args.attribution_max_rows,
        top_n_features=args.top_n_features,
        base_weight=args.base_weight,
        shap_weight=args.shap_weight,
        shap_positive_weight=args.shap_positive_weight,
        attribution_floor_fraction=args.attribution_floor_fraction,
        attribution_cumulative_importance=args.attribution_cumulative_importance,
        random_seed=args.random_seed,
        nthread=args.nthread,
        xgb_num_boost_round=args.xgb_num_boost_round,
        xgb_early_stopping_rounds=args.xgb_early_stopping_rounds,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_max_depth=args.xgb_max_depth,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_reg_lambda=args.xgb_reg_lambda,
        xgb_reg_alpha=args.xgb_reg_alpha,
        window_size=args.window_size,
        batch_size=args.batch_size,
        attribution_batch_size=args.attribution_batch_size,
        torch_epochs=args.torch_epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        torch_learning_rate=args.torch_learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        scale_quantile_low=args.scale_quantile_low,
        scale_quantile_high=args.scale_quantile_high,
        scale_clip=args.scale_clip,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        use_amp=args.use_amp,
        ig_steps=args.ig_steps,
    )


def run_backend_attribution_command(args: argparse.Namespace) -> int:
    config = build_backend_attribution_config_from_args(args)
    summary = run_backend_attribution(config)

    print(f"Backend attribution summary: {summary['summary_path']}")
    for target_name, backend_payload in summary["targets"].items():
        print(f"Target: {target_name}")
        for backend_name, report in backend_payload.items():
            metrics = report["model_metrics"]
            selected_count = int(report.get("selected_feature_count", report["feature_count"]))
            candidate_count = int(report.get("candidate_feature_count", selected_count))
            print(
                f"  - {backend_name:<8} val_ap={metrics['average_precision']:.4f} "
                f"features={selected_count:,}/{candidate_count:,} "
                f"merged={report['artifacts']['merged_csv']}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature preprocessing toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare generated features for model training",
    )
    add_prepare_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=run_prepare_command)

    backend_parser = subparsers.add_parser(
        "backend-attribution",
        help="Build backend-aware attribution rankings for xgboost, TCN, and LSTM training flows",
    )
    add_backend_attribution_arguments(backend_parser)
    backend_parser.set_defaults(handler=run_backend_attribution_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


__all__ = [
    "add_backend_attribution_arguments",
    "add_prepare_arguments",
    "build_backend_attribution_config_from_args",
    "build_config_from_args",
    "build_parser",
    "main",
    "run_backend_attribution_command",
    "run_prepare_command",
]
