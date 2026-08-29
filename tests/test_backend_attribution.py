from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.backend_attribution import (
    BackendAttributionConfig,
    compute_integrated_gradients,
    run_backend_attribution,
    should_disable_cudnn_for_integrated_gradients,
)
from preprocessing.pipeline import FeaturePreprocessingPipeline
from preprocessing.config import PreprocessingConfig

if torch is not None:
    from model_training.ote_training.torch_models import LSTMClassifier, TCNClassifier


def _write_frvp_attribution_dataset(base_dir: Path) -> Path:
    rows = 120
    timestamps = pd.date_range("2024-04-01 00:00:00", periods=rows, freq="5min")
    positive_idx = {8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112}
    target = [1 if index in positive_idx else 0 for index in range(rows)]

    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "open": 5100.0 + np.arange(rows, dtype=float) * 0.25,
            "high": 5100.4 + np.arange(rows, dtype=float) * 0.25,
            "low": 5099.6 + np.arange(rows, dtype=float) * 0.25,
            "close": 5100.1 + np.arange(rows, dtype=float) * 0.25,
            "volume": 1500 + np.arange(rows, dtype=int) * 3,
            "frvp_dist_poc_session_atr": [1.5 if value else ((index % 9) - 4) / 4.0 for index, value in enumerate(target)],
            "frvp_dist_poc_day_atr": [0.9 if value else np.cos(index / 5.0) for index, value in enumerate(target)],
            "frvp_open_type": [1 if value else (-1 if index % 2 else 0) for index, value in enumerate(target)],
            "frvp_day_type": [3 if value else index % 4 for index, value in enumerate(target)],
            "frvp_poc_plus_ict_fvg_confluence": [1 if value else int(index % 6 == 0) for index, value in enumerate(target)],
            "label_long_frvp_reversal": target,
            "sample_weight_long_frvp_reversal": [2.5 if value else 1.0 for value in target],
            "label_quality_long_frvp_reversal": [0.82 if value else 0.0 for value in target],
            "exclude_long_frvp_reversal": [False] * rows,
            "neg_ok_long_frvp_reversal": [not value for value in target],
            "htf_confluence_long_frvp_reversal": [1 if value or index in {7, 9, 55, 57, 87, 89} else 0 for index, value in enumerate(target)],
            "warmup_mask": [index < 4 for index in range(rows)],
        }
    )

    dataset_path = base_dir / "frvp_attr_features.csv"
    metadata_path = base_dir / "frvp_attr_features.metadata.json"
    df.to_csv(dataset_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": [
                    "frvp_dist_poc_session_atr",
                    "frvp_dist_poc_day_atr",
                    "frvp_open_type",
                    "frvp_day_type",
                    "frvp_poc_plus_ict_fvg_confluence",
                ],
                "timezone_contract": {
                    "source_timezone": "UTC",
                    "canonical_timezone": "UTC",
                    "feature_clock_timezone": "America/New_York",
                    "market_close_timezone": "America/New_York",
                },
                "config": {
                    "drop_warmup_rows": False,
                    "warmup_rows": 0,
                    "fillna_numeric": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_path


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed in this environment")
def test_should_disable_cudnn_for_integrated_gradients_only_for_cuda_recurrent_models() -> None:
    lstm = LSTMClassifier(input_size=3, hidden_size=4, num_layers=2, dropout=0.2)
    tcn = TCNClassifier(input_size=3, hidden_size=4, num_layers=2, dropout=0.2)

    assert should_disable_cudnn_for_integrated_gradients(lstm, torch.device("cuda")) is True
    assert should_disable_cudnn_for_integrated_gradients(lstm, torch.device("cpu")) is False
    assert should_disable_cudnn_for_integrated_gradients(tcn, torch.device("cuda")) is False


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed in this environment")
def test_compute_integrated_gradients_restores_training_mode_and_shape() -> None:
    model = LSTMClassifier(input_size=3, hidden_size=4, num_layers=2, dropout=0.2)
    model.train()

    features = np.linspace(0.0, 1.0, num=5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    attributions = compute_integrated_gradients(model, features, steps=4, batch_size=2)

    assert attributions.shape == features.shape
    assert np.isfinite(attributions).all()
    assert model.training is True


def test_backend_attribution_runs_for_prepared_frvp_target(tmp_path: Path) -> None:
    dataset_path = _write_frvp_attribution_dataset(tmp_path)
    prepared_root = tmp_path / "prepared_frvp_attr"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_frvp_reversal"],
            min_usable_rows=30,
            min_train_rows=15,
            min_positive_samples=4,
            top_n_features=10,
        )
    )
    pipeline.run(dataset_path, prepared_root)

    summary = run_backend_attribution(
        BackendAttributionConfig(
            prepared_root=str(prepared_root),
            targets=["long_frvp_reversal"],
            backends=["xgboost"],
            max_features=8,
            attribution_max_rows=40,
            top_n_features=5,
            xgb_num_boost_round=25,
            xgb_early_stopping_rounds=5,
            xgb_learning_rate=0.10,
            xgb_max_depth=3,
            xgb_min_child_weight=1.0,
        )
    )

    target_summary = summary["targets"]["long_frvp_reversal"]["xgboost"]
    merged_csv = Path(target_summary["artifacts"]["merged_csv"])
    stats_csv = Path(target_summary["artifacts"]["stats_csv"])
    stats_features = pd.read_csv(stats_csv)["feature"].tolist()
    target_features = {
        row["feature"]
        for row in target_summary["top_features_overall"]
    }

    assert merged_csv.exists()
    assert stats_csv.exists()
    assert target_summary["selected_feature_count"] > 0
    assert "htf_confluence_long_frvp_reversal" in stats_features
    assert {"frvp_dist_poc_session_atr", "htf_confluence_long_frvp_reversal"} & target_features


def test_backend_attribution_soft_promotes_frvp_audit_features_beyond_requested_cap(tmp_path: Path) -> None:
    dataset_path = _write_frvp_attribution_dataset(tmp_path)
    prepared_root = tmp_path / "prepared_frvp_attr_soft_cap"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_frvp_reversal"],
            min_usable_rows=30,
            min_train_rows=15,
            min_positive_samples=4,
            top_n_features=10,
        )
    )
    pipeline.run(dataset_path, prepared_root)

    summary = run_backend_attribution(
        BackendAttributionConfig(
            prepared_root=str(prepared_root),
            targets=["long_frvp_reversal"],
            backends=["xgboost"],
            max_features=2,
            attribution_max_rows=40,
            top_n_features=5,
            xgb_num_boost_round=20,
            xgb_early_stopping_rounds=5,
            xgb_learning_rate=0.10,
            xgb_max_depth=3,
            xgb_min_child_weight=1.0,
        )
    )

    target_summary = summary["targets"]["long_frvp_reversal"]["xgboost"]
    stats_csv = Path(target_summary["artifacts"]["stats_csv"])
    stats_features = pd.read_csv(stats_csv)["feature"].tolist()

    assert target_summary["requested_feature_cap"] == 2
    assert target_summary["candidate_feature_count"] > 2
    assert len(target_summary["audit_promoted_features"]) > 0
    assert "htf_confluence_long_frvp_reversal" in stats_features
