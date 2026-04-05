import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_training.ote_training import ote_xgboost_pipeline as pipeline
from model_training.ote_training.ote_xgboost_pipeline import (
    OTETrainingConfig,
    PreparedTargetDataset,
    PurgedWalkForwardSplitter,
    build_sequence_windows,
    build_sparse_lag_view,
    cross_validate_trial,
    event_metrics,
    focal_loss_binary,
    load_prepared_target_dataset,
    load_ranked_features,
    make_focal_objective,
    make_lag_steps,
    resolve_purge_bars,
    save_training_outputs,
)
from model_training.ote_training.feature_ranking import merge_feature_rankings
from model_training.ote_training.feature_ranking import filter_features_by_attribution_gates


def make_synthetic_dataset(
    dev_rows: int = 120,
    test_rows: int = 24,
    feature_count: int = 6,
) -> PreparedTargetDataset:
    total_rows = dev_rows + test_rows
    feature_names = [f"feature_{index}" for index in range(feature_count)]
    X = np.arange(total_rows * feature_count, dtype=np.float32).reshape(total_rows, feature_count)
    y = np.zeros(total_rows, dtype=np.uint8)
    for start in range(4, total_rows, 10):
        y[start:start + 2] = 1
    w = np.ones(total_rows, dtype=np.float32)

    train_rows = dev_rows // 2
    val_rows = dev_rows - train_rows

    return PreparedTargetDataset(
        target_name="synthetic_ote",
        prepared_dir=ROOT,
        feature_names=feature_names,
        ranked_features=feature_names,
        report={},
        X_train=X[:train_rows],
        y_train=y[:train_rows],
        w_train=w[:train_rows],
        X_val=X[train_rows:dev_rows],
        y_val=y[train_rows:dev_rows],
        w_val=w[train_rows:dev_rows],
        X_test=X[dev_rows:],
        y_test=y[dev_rows:],
        w_test=w[dev_rows:],
    )


def test_make_lag_steps_uses_current_and_tail():
    lag_steps = make_lag_steps(window_size=20, lag_count=6)
    assert lag_steps[0] == 0
    assert lag_steps[-1] == 19
    assert lag_steps == sorted(lag_steps)
    assert len(lag_steps) == 6


def test_merge_feature_rankings_promotes_shap_signal():
    base = pd.DataFrame(
        {
            "feature": ["feature_a", "feature_b", "feature_c"],
            "composite_score": [0.90, 0.80, 0.70],
            "mutual_information": [0.05, 0.04, 0.03],
            "rf_importance": [0.07, 0.06, 0.05],
        }
    )
    shap = pd.DataFrame(
        {
            "feature": ["feature_a", "feature_b", "feature_c"],
            "mean_abs_shap_all": [0.10, 0.90, 0.20],
            "mean_abs_shap_positive": [0.05, 0.80, 0.10],
        }
    )

    merged = merge_feature_rankings(
        base_ranking=base,
        shap_ranking=shap,
        allowed_features=["feature_a", "feature_b", "feature_c"],
    )

    assert list(merged["feature"])[:2] == ["feature_b", "feature_a"]
    assert float(merged.loc[0, "merged_score"]) > float(merged.loc[1, "merged_score"])


def test_filter_features_by_attribution_gates_applies_floor_and_cumulative_thresholds():
    merged = pd.DataFrame(
        {
            "feature": ["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"],
            "merged_score": [0.99, 0.90, 0.82, 0.78, 0.70],
            "mean_abs_shap_all": [1.00, 0.50, 0.20, 0.14, 0.05],
        }
    )

    filtered, summary = filter_features_by_attribution_gates(
        merged,
        floor_fraction=0.15,
        cumulative_importance=0.90,
    )

    assert list(filtered["feature"]) == ["feature_a", "feature_b", "feature_c"]
    assert summary["after_floor_count"] == 3
    assert summary["selected_feature_count"] == 3
    assert summary["selected_cumulative_importance_share"] >= 0.90


def test_load_ranked_features_prefers_merged_ranking_file():
    test_root = ROOT / "tmp" / f"test_load_ranked_features_{uuid4().hex}"
    target_dir = test_root / "long_ote"
    target_dir.mkdir(parents=True)
    try:
        (target_dir / "features.json").write_text(
            '{"features": ["feature_a", "feature_b", "feature_c"], "n_features": 3}',
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "feature": ["feature_a", "feature_b", "feature_c"],
                "composite_score": [0.90, 0.80, 0.70],
            }
        ).to_csv(target_dir / "feature_importance.csv", index=False)
        pd.DataFrame(
            {
                "feature": ["feature_c", "feature_b", "feature_a"],
                "merged_score": [0.95, 0.85, 0.75],
            }
        ).to_csv(target_dir / "feature_importance_merged.csv", index=False)

        ranked = load_ranked_features(target_dir, max_loaded_features=2)

        assert ranked == ["feature_c", "feature_b"]
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def test_build_sparse_lag_view_alignment():
    matrix = np.arange(20, dtype=np.float32).reshape(10, 2)
    lag_steps = [0, 2, 4]
    transformed = build_sparse_lag_view(matrix, lag_steps=lag_steps, delta_feature_count=1)

    assert transformed.shape == (6, 8)
    first_row = transformed[0]
    expected = np.array(
        [
            8, 9,    # lag 0
            4, 5,    # lag 2
            0, 1,    # lag 4
            8 - 4,   # delta current vs lag 2 on first feature only
            8 - 0,   # delta current vs lag 4 on first feature only
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(first_row, expected)


def test_build_sequence_windows_alignment():
    matrix = np.arange(20, dtype=np.float32).reshape(10, 2)
    transformed = build_sequence_windows(matrix, window_size=4)

    assert transformed.shape == (7, 4, 2)
    expected = np.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(transformed[0], expected)


def test_event_metrics_respects_zone_matching():
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
    y_score = np.array([0.1, 0.2, 0.7, 0.6, 0.4, 0.1, 0.55, 0.9, 0.4, 0.2], dtype=np.float32)
    metrics = event_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        tolerance=1,
        cooldown=1,
    )

    assert metrics["true_events"] == 2
    assert metrics["matched_events"] == 2
    assert metrics["event_recall"] == 1.0


def test_focal_objective_gradient_matches_numeric_estimate():
    alpha = 0.8
    gamma = 2.0
    objective = make_focal_objective(alpha=alpha, gamma=gamma)
    labels = np.array([1.0, 0.0], dtype=np.float32)
    dtrain = xgb.DMatrix(np.zeros((2, 1), dtype=np.float32), label=labels)
    logits = np.array([0.35, -0.6], dtype=np.float32)
    grad, _ = objective(logits, dtrain)

    eps = 1e-5
    for index in range(len(labels)):
        plus = logits.copy()
        minus = logits.copy()
        plus[index] += eps
        minus[index] -= eps
        loss_plus = focal_loss_binary(labels[index:index + 1], plus[index:index + 1], alpha, gamma)[0]
        loss_minus = focal_loss_binary(labels[index:index + 1], minus[index:index + 1], alpha, gamma)[0]
        numeric_grad = (loss_plus - loss_minus) / (2 * eps)
        assert np.isfinite(grad[index])
        assert abs(grad[index] - numeric_grad) < 1e-3


def test_purged_walk_forward_splitter_uses_explicit_geometry():
    splitter = PurgedWalkForwardSplitter(
        initial_train_rows=40,
        val_rows=20,
        step_rows=20,
        purge_bars=5,
        min_folds=2,
    )

    folds = splitter.split(100)

    assert [(fold.train_start, fold.train_end, fold.val_start, fold.val_end) for fold in folds] == [
        (0, 40, 45, 65),
        (0, 60, 65, 85),
    ]


def test_purged_walk_forward_splitter_respects_max_train_rows():
    splitter = PurgedWalkForwardSplitter(
        initial_train_rows=40,
        val_rows=20,
        step_rows=20,
        purge_bars=5,
        max_train_rows=50,
        min_folds=2,
    )

    folds = splitter.split(140)

    assert folds[0].train_start == 0
    assert folds[1].train_start == 10
    assert max((fold.train_end - fold.train_start) for fold in folds) == 50


def test_resolve_purge_bars_prefers_label_horizon():
    config = OTETrainingConfig(
        label_max_holding_bars=30,
        label_exclusion_pre_bars=2,
        label_zone_pre_bars=1,
        purge_buffer_bars=4,
        event_tolerance_bars=2,
        event_cooldown_bars=4,
    )

    assert resolve_purge_bars(config=config, context_rows=12) == 34


def test_cross_validate_trial_records_fold_diagnostics(monkeypatch: pytest.MonkeyPatch):
    dataset = make_synthetic_dataset()
    config = OTETrainingConfig(
        cv_initial_train_rows=40,
        cv_val_rows=20,
        cv_step_rows=20,
        cv_min_folds=3,
        min_train_positive_rows=3,
        min_val_positive_rows=2,
        min_val_true_events=2,
        max_loaded_features=6,
        window_min=4,
        window_max=4,
        label_max_holding_bars=4,
        label_exclusion_pre_bars=1,
        label_zone_pre_bars=1,
        purge_buffer_bars=0,
        calibration_method="none",
        verbosity=0,
    )
    params = {
        "window_size": 4,
        "lag_count": 3,
        "delta_feature_count": 2,
        "hard_negative_radius": 1,
        "hard_negative_multiplier": 1.0,
    }

    def fake_train_and_score_fold(**kwargs):
        y_val = kwargs["y_val"]
        probabilities = np.linspace(0.2, 0.8, num=len(y_val), dtype=np.float32)
        return probabilities, {
            "average_precision": 0.4,
            "roc_auc": 0.7,
            "brier": 0.2,
            "threshold": 0.5,
            "event_precision": 0.75,
            "event_recall": 0.5,
            "event_f1": 0.6,
            "event_fbeta_0_5": 0.68,
            "predicted_events": 3.0,
            "true_events": 2.0,
            "matched_events": 2.0,
        }

    monkeypatch.setattr(pipeline, "train_and_score_fold", fake_train_and_score_fold)

    artifacts = cross_validate_trial(dataset=dataset, params=params, config=config)

    assert len(artifacts.fold_diagnostics) == 3
    assert len(artifacts.fold_plans) == 3
    assert artifacts.resolved_purge_bars == 6
    assert artifacts.selected_feature_names == dataset.ranked_features
    first_fold = artifacts.fold_diagnostics[0]
    assert first_fold["train_rows_raw"] == 40
    assert first_fold["val_rows_raw"] == 20
    assert first_fold["train_rows_model"] == 37
    assert first_fold["val_rows_model"] == 20
    assert first_fold["val_true_events"] == 2


def test_cross_validate_trial_records_oof_fold_ids(monkeypatch: pytest.MonkeyPatch):
    dataset = make_synthetic_dataset()
    config = OTETrainingConfig(
        cv_initial_train_rows=40,
        cv_val_rows=20,
        cv_step_rows=20,
        cv_min_folds=3,
        min_train_positive_rows=3,
        min_val_positive_rows=2,
        min_val_true_events=2,
        max_loaded_features=6,
        window_min=4,
        window_max=4,
        label_max_holding_bars=4,
        label_exclusion_pre_bars=1,
        label_zone_pre_bars=1,
        purge_buffer_bars=0,
        calibration_method="none",
        verbosity=0,
    )
    params = {
        "window_size": 4,
        "lag_count": 3,
        "delta_feature_count": 2,
        "hard_negative_radius": 1,
        "hard_negative_multiplier": 1.0,
    }

    def fake_train_and_score_fold(**kwargs):
        y_val = kwargs["y_val"]
        probabilities = np.linspace(0.15, 0.85, num=len(y_val), dtype=np.float32)
        return probabilities, {
            "average_precision": 0.45,
            "roc_auc": 0.72,
            "brier": 0.18,
            "threshold": 0.5,
            "event_precision": 0.8,
            "event_recall": 0.55,
            "event_f1": 0.65,
            "event_fbeta_0_5": 0.7,
            "predicted_events": 3.0,
            "true_events": 2.0,
            "matched_events": 2.0,
        }

    monkeypatch.setattr(pipeline, "train_and_score_fold", fake_train_and_score_fold)

    artifacts = cross_validate_trial(dataset=dataset, params=params, config=config)

    assert set(np.unique(artifacts.oof_fold_id[artifacts.oof_fold_id >= 0])) == {1, 2, 3}
    assert np.all(artifacts.oof_fold_id[46:66] == 1)
    assert np.all(artifacts.oof_fold_id[66:86] == 2)
    assert np.all(artifacts.oof_fold_id[86:106] == 3)
    assert np.all(artifacts.oof_fold_id[:46] == -1)


def test_load_prepared_target_dataset_reads_source_row_idx():
    prepared_root = ROOT / "tmp" / f"test_load_prepared_target_dataset_reads_source_row_idx_{uuid4().hex}"
    target_dir = prepared_root / "long_ote"
    target_dir.mkdir(parents=True)

    try:
        (target_dir / "features.json").write_text(
            json.dumps({"features": ["feature_a", "feature_b"], "n_features": 2}),
            encoding="utf-8",
        )
        (target_dir / "report.json").write_text(
            json.dumps({"target_name": "long_ote", "row_identity": {"source_row_idx_column": "source_row_idx"}}),
            encoding="utf-8",
        )

        pd.DataFrame(
            {
                "source_row_idx": [10, 11, 12],
                "feature_a": [0.1, 0.2, 0.3],
                "feature_b": [1.0, 1.1, 1.2],
                "target": [0, 1, 0],
                "sample_weight": [1.0, 2.0, 1.0],
            }
        ).to_csv(target_dir / "train.csv", index=False)
        pd.DataFrame(
            {
                "source_row_idx": [20, 21],
                "feature_a": [0.4, 0.5],
                "feature_b": [1.3, 1.4],
                "target": [1, 0],
                "sample_weight": [1.5, 1.0],
            }
        ).to_csv(target_dir / "val.csv", index=False)
        pd.DataFrame(
            {
                "source_row_idx": [30, 31],
                "feature_a": [0.6, 0.7],
                "feature_b": [1.5, 1.6],
                "target": [0, 1],
                "sample_weight": [1.0, 2.5],
            }
        ).to_csv(target_dir / "test.csv", index=False)

        dataset = load_prepared_target_dataset(
            prepared_root=prepared_root,
            target_name="long_ote",
            config=OTETrainingConfig(max_loaded_features=2),
        )

        np.testing.assert_array_equal(dataset.dev_source_row_idx, np.array([10, 11, 12, 20, 21], dtype=np.int64))
        np.testing.assert_array_equal(dataset.test_source_row_idx, np.array([30, 31], dtype=np.int64))
    finally:
        shutil.rmtree(prepared_root, ignore_errors=True)


def test_save_training_outputs_includes_source_row_idx():
    output_dir = ROOT / "tmp" / f"test_save_training_outputs_{uuid4().hex}"
    output_dir.mkdir(parents=True)

    try:
        dataset = make_synthetic_dataset(dev_rows=12, test_rows=4, feature_count=4)
        dataset.source_row_idx_train = np.array([100, 101, 102, 103, 104, 105], dtype=np.int64)
        dataset.source_row_idx_val = np.array([200, 201, 202, 203, 204, 205], dtype=np.int64)
        dataset.source_row_idx_test = np.array([300, 301, 302, 303], dtype=np.int64)
        dataset.report = {
            "timezone_contract": {
                "source_timezone": "UTC",
                "canonical_timezone": "UTC",
                "feature_clock_timezone": "America/New_York",
                "market_close_timezone": "America/New_York",
            },
            "source_lineage": {
                "feature_csv": "data/features/eurusd_5min_ote_full.csv",
                "feature_metadata_file": "data/features/eurusd_5min_ote_full.metadata.json",
                "feature_builder_source_path": "data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv",
                "feature_builder_source_metadata_file": "data/labeling/labeled_data/eurusd_5min_ote_labels_full.metadata.json",
                "upstream_source_path": "data/currency_data/eurusd-5m.csv",
                "upstream_bar_timestamp_semantics": "bar_open",
                "upstream_timezone_contract": {
                    "source_timezone": "GMT-6",
                    "canonical_timezone": "UTC",
                },
            },
        }
        dataset.prepared_summary = {
            "input_file": "data/features/eurusd_5min_ote_full.csv",
            "metadata_file": "data/features/eurusd_5min_ote_full.metadata.json",
            "timezone_contract": dataset.report["timezone_contract"],
            "source_lineage": dataset.report["source_lineage"],
        }
        dataset.prepared_summary_path = "data/prepared/eurusd_5min_ote_full/summary.json"

        artifacts = pipeline.TrialArtifacts(
            params={},
            oof_pred=np.linspace(0.1, 0.9, num=len(dataset.dev_y), dtype=np.float32),
            fold_metrics=[{"fold": 1, "average_precision": 0.5, "event_fbeta_0_5": 0.6}],
            selected_feature_names=dataset.feature_names,
            oof_fold_id=np.full(len(dataset.dev_y), 1, dtype=np.int16),
            fold_plans=[{"fold": 1}],
            fold_diagnostics=[{"fold": 1}],
        )

        scaler = RobustScaler().fit(dataset.dev_X)

        class DummyModel:
            def save_model(self, path):
                Path(path).write_text("dummy-model", encoding="utf-8")

            def get_score(self, importance_type="gain"):
                return {}

        final_model = {
            "backend": "xgboost",
            "model": DummyModel(),
            "checkpoint": None,
            "scaler": scaler,
            "calibrator": None,
            "threshold": 0.5,
            "threshold_metrics": {"event_fbeta_0_5": 0.6},
            "test_metrics": {"average_precision": 0.7, "event_fbeta_0_5": 0.65},
            "feature_names": dataset.feature_names,
            "selected_feature_names": dataset.feature_names,
            "lag_steps": [0],
            "window_size": 1,
            "context_rows": 0,
            "delta_feature_count": 0,
            "oof_calibrated": np.linspace(0.1, 0.9, num=len(dataset.dev_y), dtype=np.float32),
            "test_raw": np.linspace(0.2, 0.8, num=len(dataset.y_test), dtype=np.float32),
            "test_calibrated": np.linspace(0.25, 0.85, num=len(dataset.y_test), dtype=np.float32),
            "training_history": [],
            "model_config": {"backend": "xgboost", "model_type": "xgboost"},
        }

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        study.tell(trial, 0.42)

        save_training_outputs(
            output_dir=output_dir,
            dataset=dataset,
            config=OTETrainingConfig(verbosity=0),
            artifacts=artifacts,
            final_model=final_model,
            study=study,
        )

        oof_predictions = pd.read_csv(output_dir / "oof_predictions.csv")
        test_predictions = pd.read_csv(output_dir / "test_predictions.csv")
        training_summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
        model_config = json.loads((output_dir / "model_config.json").read_text(encoding="utf-8"))

        assert "source_row_idx" in oof_predictions.columns
        assert "source_row_idx" in test_predictions.columns
        np.testing.assert_array_equal(
            oof_predictions["source_row_idx"].to_numpy(dtype=np.int64),
            np.array([100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            test_predictions["source_row_idx"].to_numpy(dtype=np.int64),
            np.array([300, 301, 302, 303], dtype=np.int64),
        )
        assert training_summary["row_identity"]["source_row_idx_column"] == "source_row_idx"
        assert training_summary["row_identity"]["source_row_idx_available"] is True
        assert training_summary["timezone_contract"]["canonical_timezone"] == "UTC"
        assert training_summary["source_lineage"]["feature_builder_source_path"] == (
            "data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv"
        )
        assert training_summary["prepared_summary_file"] == "data/prepared/eurusd_5min_ote_full/summary.json"
        assert model_config["timezone_contract"]["feature_clock_timezone"] == "America/New_York"
        assert model_config["source_lineage"]["upstream_source_path"] == "data/currency_data/eurusd-5m.csv"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_cross_validate_trial_smoke():
    config = OTETrainingConfig(
        prepared_root="data/prepared/eurusd_5min_ote_2000_v2",
        n_trials=1,
        cv_initial_train_rows=180,
        cv_val_rows=80,
        cv_step_rows=80,
        cv_min_folds=2,
        min_train_positive_rows=0,
        min_val_positive_rows=0,
        min_val_true_events=0,
        max_loaded_features=48,
        label_max_holding_bars=8,
        label_exclusion_pre_bars=2,
        label_zone_pre_bars=1,
        purge_buffer_bars=2,
        calibration_method="none",
        verbosity=0,
    )
    if not (Path(config.prepared_root) / "long_ote" / "report.json").exists():
        pytest.skip("Prepared smoke-test dataset is not available in this workspace.")
    dataset = load_prepared_target_dataset(
        prepared_root=Path(config.prepared_root),
        target_name="long_ote",
        config=config,
    )
    params = {
        "window_size": 8,
        "lag_count": 4,
        "delta_feature_count": 8,
        "learning_rate": 0.05,
        "focal_alpha": 0.85,
        "focal_gamma": 2.0,
        "max_depth": 3,
        "min_child_weight": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "min_split_loss": 0.0,
        "max_delta_step": 0.0,
        "max_bin": 256,
        "warmup_fraction": 0.2,
        "warmup_rounds": 5,
        "main_rounds": 10,
        "fine_rounds": 5,
        "fine_lr_scale": 0.4,
        "hard_negative_radius": 2,
        "hard_negative_multiplier": 1.2,
    }

    artifacts = cross_validate_trial(dataset=dataset, params=params, config=config)

    assert len(artifacts.fold_metrics) >= 2
    assert np.nanmax(artifacts.oof_pred) <= 1.0
    assert np.nanmin(artifacts.oof_pred) >= 0.0


def test_tcn_forward_pass():
    torch = pytest.importorskip("torch")
    from model_training.ote_training.torch_models import TCNClassifier

    model = TCNClassifier(input_size=5, hidden_size=8, num_layers=4, dropout=0.1)
    inputs = torch.randn(3, 12, 5)
    outputs = model(inputs)

    assert outputs.shape == (3,)
    assert torch.isfinite(outputs).all()


def test_tcn_forward_uses_latest_timestep_representation():
    torch = pytest.importorskip("torch")
    from model_training.ote_training.torch_models import TCNClassifier

    model = TCNClassifier(input_size=2, hidden_size=2, num_layers=1, dropout=0.0)
    model.blocks = torch.nn.ModuleList([torch.nn.Identity()])
    model.dropout = torch.nn.Identity()
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
        model.head.bias.zero_()

    inputs = torch.tensor(
        [[[1.0, 0.0], [3.0, 0.0], [10.0, 0.0]]],
        dtype=torch.float32,
    )
    outputs = model(inputs)

    assert outputs.shape == (1,)
    assert torch.allclose(outputs, torch.tensor([10.0], dtype=torch.float32))


def test_lstm_forward_pass():
    torch = pytest.importorskip("torch")
    from model_training.ote_training.torch_models import LSTMClassifier

    model = LSTMClassifier(input_size=5, hidden_size=8, num_layers=2, dropout=0.1)
    inputs = torch.randn(3, 12, 5)
    outputs = model(inputs)

    assert outputs.shape == (3,)
    assert torch.isfinite(outputs).all()


def test_build_torch_trial_params_samples_window_and_depth():
    config = OTETrainingConfig(
        backend="torch",
        model_type="tcn",
        max_loaded_features=48,
        window_min=16,
        window_max=32,
        num_layers=3,
        learning_rate=1e-3,
    )
    trial = optuna.trial.FixedTrial(
        {
            "window_size": 20,
            "num_layers": 4,
            "learning_rate": 2e-3,
            "focal_alpha": 0.8,
            "focal_gamma": 2.1,
            "hard_negative_radius": 3,
            "hard_negative_multiplier": 1.4,
        }
    )

    params = pipeline.build_torch_trial_params(trial, config)

    assert params["window_size"] == 20
    assert params["num_layers"] == 4
    assert "top_features" not in params


def test_build_torch_training_config_uses_tuned_num_layers():
    config = OTETrainingConfig(
        backend="torch",
        model_type="tcn",
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        batch_size=128,
        epochs=4,
        learning_rate=1e-3,
        verbosity=0,
    )

    trainer_config = pipeline.build_torch_training_config(
        input_size=12,
        params={
            "learning_rate": 2e-3,
            "focal_alpha": 0.82,
            "focal_gamma": 2.2,
            "num_layers": 4,
        },
        config=config,
    )

    assert trainer_config["num_layers"] == 4
    assert trainer_config["hidden_size"] == 64
    assert trainer_config["learning_rate"] == pytest.approx(2e-3)


def test_build_torch_training_config_includes_explicit_phase_schedule():
    config = OTETrainingConfig(
        backend="torch",
        model_type="tcn",
        epochs=30,
        torch_warmup_epochs=4,
        torch_main_epochs=16,
        torch_fine_epochs=6,
        torch_tail_epochs=4,
        torch_fine_lr_scale=0.2,
        torch_tail_lr_scale=0.08,
    )

    trainer_config = pipeline.build_torch_training_config(
        input_size=8,
        params={
            "learning_rate": 1.5e-3,
            "focal_alpha": 0.8,
            "focal_gamma": 2.0,
            "num_layers": 3,
        },
        config=config,
    )

    assert trainer_config["warmup_epochs"] == 4
    assert trainer_config["main_epochs"] == 16
    assert trainer_config["fine_epochs"] == 6
    assert trainer_config["tail_epochs"] == 4
    assert trainer_config["fine_lr_scale"] == pytest.approx(0.2)
    assert trainer_config["tail_lr_scale"] == pytest.approx(0.08)


def test_validate_backend_config_rejects_partial_explicit_torch_schedule():
    config = OTETrainingConfig(
        backend="torch",
        model_type="tcn",
        epochs=30,
        torch_warmup_epochs=4,
        torch_main_epochs=16,
    )

    with pytest.raises(ValueError, match="requires warmup, main, and fine epoch counts"):
        pipeline.validate_backend_config(config)


def test_build_phase_schedule_accepts_explicit_tail_schedule():
    pytest.importorskip("torch")
    from model_training.ote_training.torch_trainer import build_phase_schedule

    schedule = build_phase_schedule(
        30,
        warmup_epochs=4,
        main_epochs=16,
        fine_epochs=6,
        tail_epochs=4,
        fine_lr_scale=0.2,
        tail_lr_scale=0.08,
    )

    assert [phase["name"] for phase in schedule] == ["warmup", "main", "fine", "tail"]
    assert [phase["epochs"] for phase in schedule] == [4.0, 16.0, 6.0, 4.0]
    assert schedule[2]["lr_scale"] == pytest.approx(0.2)
    assert schedule[3]["lr_scale"] == pytest.approx(0.08)


def test_cross_validate_trial_torch_smoke():
    pytest.importorskip("torch")

    config = OTETrainingConfig(
        prepared_root="data/prepared/eurusd_5min_ote_2000_v2",
        backend="torch",
        model_type="lstm",
        n_trials=1,
        cv_initial_train_rows=180,
        cv_val_rows=80,
        cv_step_rows=80,
        cv_min_folds=2,
        min_train_positive_rows=0,
        min_val_positive_rows=0,
        min_val_true_events=0,
        max_loaded_features=32,
        label_max_holding_bars=8,
        label_exclusion_pre_bars=2,
        label_zone_pre_bars=1,
        purge_buffer_bars=2,
        calibration_method="none",
        batch_size=128,
        epochs=2,
        hidden_size=16,
        num_layers=1,
        dropout=0.1,
        use_amp=False,
        learning_rate=1e-3,
        window_size=8,
        verbosity=0,
    )
    if not (Path(config.prepared_root) / "long_ote" / "report.json").exists():
        pytest.skip("Prepared smoke-test dataset is not available in this workspace.")
    dataset = load_prepared_target_dataset(
        prepared_root=Path(config.prepared_root),
        target_name="long_ote",
        config=config,
    )
    params = {
        "window_size": 8,
        "learning_rate": 1e-3,
        "focal_alpha": 0.85,
        "focal_gamma": 2.0,
        "hard_negative_radius": 2,
        "hard_negative_multiplier": 1.2,
    }

    artifacts = cross_validate_trial(dataset=dataset, params=params, config=config)

    assert len(artifacts.fold_metrics) >= 2
    assert np.nanmax(artifacts.oof_pred) <= 1.0
    assert np.nanmin(artifacts.oof_pred) >= 0.0
