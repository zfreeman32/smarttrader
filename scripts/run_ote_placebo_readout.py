from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.ote_training.ote_xgboost_pipeline import (
    OTETrainingConfig,
    PreparedTargetDataset,
    load_prepared_target_dataset,
    safe_average_precision,
    train_target_pipeline,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run shuffled-label placebo retrains against a saved OTE/FRVP artifact."
    )
    parser.add_argument(
        "--real-artifact-dir",
        type=Path,
        required=True,
        help="Path to the saved real-model artifact directory containing training_summary.json.",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=None,
        help="Prepared root override. Defaults to the saved training config value.",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default=None,
        help="Prepared target override. Defaults to the saved training summary target.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where placebo shuffle artifacts and summary files will be written.",
    )
    parser.add_argument(
        "--num-shuffles",
        type=int,
        default=10,
        help="Number of shuffled-label retrains to run.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=20260717,
        help="Base RNG seed for label shuffling.",
    )
    parser.add_argument(
        "--trials-override",
        type=int,
        default=None,
        help="Optional Optuna trial count override. Defaults to the saved training config.",
    )
    parser.add_argument(
        "--training-seed-override",
        type=int,
        default=None,
        help="Optional model-training random seed override. Defaults to the saved training config.",
    )
    return parser


def load_real_training_summary(artifact_dir: Path) -> Dict[str, Any]:
    summary_path = artifact_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Real artifact training summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_config_from_training_summary(
    training_summary: Dict[str, Any],
    *,
    prepared_root: Path | None,
    output_root: Path,
    trials_override: int | None,
    training_seed_override: int | None,
) -> OTETrainingConfig:
    raw_config = training_summary.get("config", {})
    if not isinstance(raw_config, dict):
        raise ValueError("Saved training summary is missing a valid config payload.")

    allowed_fields = {field.name for field in fields(OTETrainingConfig)}
    config_kwargs = {key: value for key, value in raw_config.items() if key in allowed_fields}

    if prepared_root is not None:
        config_kwargs["prepared_root"] = str(prepared_root)
    if trials_override is not None:
        config_kwargs["n_trials"] = int(trials_override)
    if training_seed_override is not None:
        config_kwargs["random_seed"] = int(training_seed_override)
    config_kwargs["output_root"] = str(output_root)
    config_kwargs["targets"] = []

    return OTETrainingConfig(**config_kwargs)


def clone_dataset_with_targets(
    dataset: PreparedTargetDataset,
    *,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> PreparedTargetDataset:
    return PreparedTargetDataset(
        target_name=dataset.target_name,
        prepared_dir=dataset.prepared_dir,
        feature_names=list(dataset.feature_names),
        ranked_features=list(dataset.ranked_features),
        report=dict(dataset.report),
        prepared_summary=dict(dataset.prepared_summary),
        prepared_summary_path=dataset.prepared_summary_path,
        X_train=dataset.X_train,
        y_train=np.ascontiguousarray(y_train, dtype=np.uint8),
        w_train=dataset.w_train,
        X_val=dataset.X_val,
        y_val=np.ascontiguousarray(y_val, dtype=np.uint8),
        w_val=dataset.w_val,
        X_test=dataset.X_test,
        y_test=np.ascontiguousarray(y_test, dtype=np.uint8),
        w_test=dataset.w_test,
        source_row_idx_train=dataset.source_row_idx_train,
        source_row_idx_val=dataset.source_row_idx_val,
        source_row_idx_test=dataset.source_row_idx_test,
    )


def build_shuffled_dataset(
    dataset: PreparedTargetDataset,
    *,
    rng: np.random.Generator,
) -> PreparedTargetDataset:
    dev_y = dataset.dev_y.copy()
    rng.shuffle(dev_y)
    shuffled_train = dev_y[: len(dataset.y_train)]
    shuffled_val = dev_y[len(dataset.y_train) :]

    test_y = dataset.y_test.copy()
    rng.shuffle(test_y)

    return clone_dataset_with_targets(
        dataset,
        y_train=shuffled_train,
        y_val=shuffled_val,
        y_test=test_y,
    )


def run_placebo_readout(
    *,
    real_artifact_dir: Path,
    prepared_root: Path | None,
    target_name: str | None,
    output_root: Path,
    num_shuffles: int,
    shuffle_seed: int,
    trials_override: int | None,
    training_seed_override: int | None,
) -> Dict[str, Any]:
    if num_shuffles <= 0:
        raise ValueError("--num-shuffles must be positive.")

    real_summary = load_real_training_summary(real_artifact_dir)
    config = build_config_from_training_summary(
        real_summary,
        prepared_root=prepared_root,
        output_root=output_root,
        trials_override=trials_override,
        training_seed_override=training_seed_override,
    )
    resolved_prepared_root = Path(config.prepared_root)
    resolved_target_name = target_name or str(real_summary.get("target") or "").strip()
    if not resolved_target_name:
        raise ValueError("Could not resolve target name from arguments or training_summary.json.")

    output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_prepared_target_dataset(
        prepared_root=resolved_prepared_root,
        target_name=resolved_target_name,
        config=config,
    )

    real_oof_ap = float(real_summary.get("oof_summary", {}).get("calibrated_average_precision", 0.0))
    real_test_ap = float(real_summary.get("test_metrics", {}).get("average_precision", 0.0))
    real_positive_rate = float(np.mean(dataset.dev_y)) if len(dataset.dev_y) else 0.0

    shuffle_rows: List[Dict[str, Any]] = []
    shuffle_oof_values: List[float] = []

    for shuffle_index in range(1, num_shuffles + 1):
        shuffle_dir = output_root / f"shuffle_{shuffle_index:02d}"
        shuffle_config = OTETrainingConfig(**{
            **{field.name: getattr(config, field.name) for field in fields(OTETrainingConfig)},
            "output_root": str(shuffle_dir),
        })
        rng = np.random.default_rng(shuffle_seed + shuffle_index)
        shuffled_dataset = build_shuffled_dataset(dataset, rng=rng)
        result = train_target_pipeline(dataset=shuffled_dataset, config=shuffle_config)

        oof_calibrated = np.asarray(result["final_model"]["oof_calibrated"], dtype=np.float32)
        valid_mask = ~np.isnan(oof_calibrated)
        shuffle_oof_ap = safe_average_precision(
            shuffled_dataset.dev_y[valid_mask],
            oof_calibrated[valid_mask],
            sample_weight=shuffled_dataset.dev_w[valid_mask],
        )
        shuffle_test_ap = float(result["final_model"]["test_metrics"]["average_precision"])
        shuffle_best_value = float(result["study"].best_value)
        shuffle_positive_rate = float(np.mean(shuffled_dataset.dev_y)) if len(shuffled_dataset.dev_y) else 0.0

        shuffle_oof_values.append(float(shuffle_oof_ap))
        shuffle_rows.append(
            {
                "shuffle_index": int(shuffle_index),
                "shuffle_seed": int(shuffle_seed + shuffle_index),
                "output_dir": str(result["output_dir"]),
                "oof_average_precision": float(shuffle_oof_ap),
                "test_average_precision": float(shuffle_test_ap),
                "study_best_value": float(shuffle_best_value),
                "dev_positive_rate": float(shuffle_positive_rate),
            }
        )

    shuffled_mean = float(np.mean(shuffle_oof_values))
    shuffled_std = float(np.std(shuffle_oof_values))
    shuffled_min = float(np.min(shuffle_oof_values))
    shuffled_max = float(np.max(shuffle_oof_values))
    placebo_gap = float(real_oof_ap - shuffled_mean)
    placebo_gap_std = float(placebo_gap / shuffled_std) if shuffled_std > 0 else None
    gate_pass = bool(placebo_gap > 0.03 and num_shuffles >= 10)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_artifact_dir": str(real_artifact_dir),
        "prepared_root": str(resolved_prepared_root),
        "target_name": resolved_target_name,
        "output_root": str(output_root),
        "num_shuffles": int(num_shuffles),
        "shuffle_seed": int(shuffle_seed),
        "training_config_reference": {
            "n_trials": int(config.n_trials),
            "random_seed": int(config.random_seed),
            "backend": config.backend,
            "model_type": config.model_type,
            "cv_initial_train_rows": int(config.cv_initial_train_rows),
            "cv_val_rows": int(config.cv_val_rows),
            "cv_step_rows": int(config.cv_step_rows),
            "cv_min_folds": int(config.cv_min_folds),
        },
        "real_model": {
            "oof_average_precision": float(real_oof_ap),
            "test_average_precision": float(real_test_ap),
            "dev_positive_rate": float(real_positive_rate),
        },
        "placebo_distribution": {
            "mean_oof_average_precision": shuffled_mean,
            "std_oof_average_precision": shuffled_std,
            "min_oof_average_precision": shuffled_min,
            "max_oof_average_precision": shuffled_max,
        },
        "gate_check": {
            "rule": "real_oof_ap_minus_shuffled_mean > 0.03 across at least 10 shuffles",
            "placebo_gap": placebo_gap,
            "placebo_gap_std_units": placebo_gap_std,
            "passed": gate_pass,
        },
        "shuffle_runs": shuffle_rows,
    }

    (output_root / "placebo_readout_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_placebo_readout(
        real_artifact_dir=args.real_artifact_dir,
        prepared_root=args.prepared_root,
        target_name=args.target_name,
        output_root=args.output_root,
        num_shuffles=args.num_shuffles,
        shuffle_seed=args.shuffle_seed,
        trials_override=args.trials_override,
        training_seed_override=args.training_seed_override,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
