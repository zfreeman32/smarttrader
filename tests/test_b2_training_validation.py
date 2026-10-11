"""Regression coverage for the two B2 readiness validation failures."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_training.ote_training import ote_xgboost_pipeline as t
from preprocessing.config import PreprocessingConfig
from preprocessing.pipeline import FeaturePreprocessingPipeline


def test_frozen_preparation_ignores_later_labels_features_and_categories(tmp_path):
    rng = np.random.default_rng(18)
    rows, boundary = 360, 100
    signal = rng.normal(size=rows)
    frame = pd.DataFrame(dict(datetime=pd.date_range("2024-01-01", periods=rows, freq="5min"),
                              source_row_idx=np.arange(rows),
                              label_long_entry=(signal > 0).astype(int),
                              feature_signal=signal, feature_correlated=signal + rng.normal(0, .005, rows),
                              feature_noise=rng.normal(size=rows),
                              feature_category=np.where(np.arange(rows) % 2, "b", "c"),
                              feature_late_constant=np.ones(rows)))
    frame.loc[::7, "feature_noise"] = np.nan
    frame.loc[boundary:, "feature_late_constant"] = rng.normal(size=rows-boundary)
    features = [c for c in frame if c.startswith("feature_")]
    config = PreprocessingConfig(target_columns=["label_long_entry"],
        target_selection_fit_rows={"long_entry": boundary}, scaler_type="robust",
        min_usable_rows=50, min_train_rows=20, rf_n_estimators=20, rf_min_samples_leaf=2)
    for version in ("before", "after"):
        path = tmp_path / f"{version}.csv"
        frame.to_csv(path, index=False)
        path.with_suffix(".metadata.json").write_text(json.dumps({"feature_columns": features}))
        FeaturePreprocessingPipeline(config).run(path, tmp_path / version)
        # Reverse later labels, move distributions, add an earlier-sorting unseen category,
        # and make a prefix-constant column appear useful. None may affect fitting.
        frame.loc[boundary:, "label_long_entry"] = 1 - frame.loc[boundary:, "label_long_entry"]
        frame.loc[boundary:, ["feature_signal", "feature_noise"]] *= 1000
        frame.loc[boundary:, "feature_category"] = "a"
        frame.loc[boundary:, "feature_late_constant"] = frame.loc[boundary:, "label_long_entry"]
    first, second = [tmp_path / v / "long_entry" for v in ("before", "after")]
    assert (first / "features.json").read_bytes() == (second / "features.json").read_bytes()
    pd.testing.assert_frame_equal(pd.read_csv(first / "feature_importance.csv"), pd.read_csv(second / "feature_importance.csv"))
    pd.testing.assert_frame_equal(pd.read_csv(first / "train.csv").iloc[:boundary], pd.read_csv(second / "train.csv").iloc[:boundary])
    a, b = [json.loads((p / "report.json").read_text())["selection_contract"] for p in (first, second)]
    assert a == b
    assert "feature_late_constant" not in json.loads((first / "features.json").read_text())["features"]
    assert a["preprocessing"]["encoding"]["encoders"]["feature_category"] == {"b": 0, "c": 1}


def dataset_and_config(backend="xgboost", refit=True):
    rows, train = 1200, 900
    matrix = np.arange(rows + 40, dtype=np.float32)[:, None]
    labels = (np.arange(rows + 40) % 7 == 0).astype(np.uint8)
    weights = np.ones(rows + 40, dtype=np.float32)
    source = np.arange(rows + 40, dtype=np.int64) * 5
    report = {"split_geometry": {"applied_embargo_bars": 1071},
              "selection_contract": {"mode": "frozen_development_prefix", "fit_rows": 300,
                                     "fit_end_source_row_idx": int(source[299]), "required_gap_bars": 1071}}
    data = t.PreparedTargetDataset("long_frvp_continuation", Path("unused"), ["row"], ["row"], report,
        matrix[:train], labels[:train], weights[:train], matrix[train:rows], labels[train:rows], weights[train:rows],
        matrix[rows:], labels[rows:], weights[rows:], source[:train], source[train:rows], source[rows:])
    config = t.OTETrainingConfig(backend=backend, model_type="tcn" if backend == "torch" else "xgboost",
        final_refit_on_dev=refit, final_eval_min_rows=64, window_size=8, window_min=8, window_max=8,
        label_max_holding_bars=10, label_exclusion_pre_bars=0, label_zone_pre_bars=0,
        event_tolerance_bars=0, event_cooldown_bars=0, purge_buffer_bars=2,
        calibration_method="none", sequential_bootstrap_final_refit=False)
    return data, config


@pytest.mark.parametrize("backend", ["torch", "xgboost"])
@pytest.mark.parametrize("refit", [False, True])
def test_final_fit_purges_holdout_scaler_and_refits_without_validation(monkeypatch, backend, refit):
    data, config = dataset_and_config(backend, refit)
    calls, scaler_rows = [], []
    original_scaler = t.fit_scaler
    def scaler_spy(matrix, config):
        scaler_rows.append(len(matrix))
        return original_scaler(matrix, config)
    monkeypatch.setattr(t, "fit_scaler", scaler_spy)
    monkeypatch.setattr(t, "transform_matrix", lambda scaler, matrix, clip_value: matrix)
    history = [dict(epoch=i, phase="main", train_fraction=1.0, learning_rate=.001) for i in (1, 2, 3)]
    class Model:
        def predict(self, matrix):
            return np.zeros(matrix.num_row())
    def recorder(**kwargs):
        calls.append(kwargs)
        return t.ModelTrainingResult(Model(), np.zeros(len(kwargs["y_eval"])), history,
                                     {"best_epoch": 2, "amp_enabled": False})
    monkeypatch.setattr(t, "train_backend_model", recorder)
    if backend == "torch":
        from model_training.ote_training import torch_trainer
        monkeypatch.setattr(torch_trainer, "load_torch_model_from_checkpoint", lambda checkpoint: Model())
        monkeypatch.setattr(torch_trainer, "predict_torch_model", lambda **kw: np.full(len(kw["X"]), .2))
    artifacts = t.TrialArtifacts(params={"delta_feature_count": 0, "hard_negative_radius": 0, "hard_negative_multiplier": 1.0}, oof_pred=np.full(1200, .2),
        fold_metrics=[], selected_feature_names=["row"], backend=backend, lag_steps=[0, 7], window_size=8, context_rows=7)
    result = t.fit_final_model(data, artifacts, config)
    plan = result["final_holdout"]
    assert plan["source_gap"] >= 1071
    assert scaler_rows == ([plan["fit_end"], 1200] if refit else [plan["fit_end"]])
    assert len(calls[0]["y_train"]) == plan["fit_end"] - 7
    assert len(calls[0]["y_eval"]) == 1200 - plan["val_start"]
    def label_rows(X):
        return X.matrix[X.window_size-1:, 0] if isinstance(X, t.SequenceWindowSource) else X[:, 0]
    fit_ids, eval_ids = label_rows(calls[0]["X_train"]), label_rows(calls[0]["X_eval"])
    assert not np.intersect1d(fit_ids, eval_ids).size
    assert (eval_ids.min() - fit_ids.max()) * 5 - 1 >= 1071
    if refit:
        assert len(calls) == 2
        assert len(calls[1]["y_train"]) == 1200 - 7
        assert not len(calls[1]["X_eval"]) and not len(calls[1]["y_eval"])
        assert calls[1]["fixed_epoch_schedule"] == (history[:2] if backend == "torch" else None)
    else:
        assert len(calls) == 1


def test_invalid_selection_and_final_gap_fail_before_fitting():
    data, config = dataset_and_config()
    with pytest.raises(ValueError, match="overlaps"):
        t.validate_frozen_selection(data, [t.FoldPlan(1, 0, 299, 500, 600, 10)])
    data.report["split_geometry"]["applied_embargo_bars"] = 100000
    with pytest.raises(ValueError, match="Not enough final-fit"):
        t.plan_final_holdout(data, config, 7)
    data.source_row_idx_train[:] = -1
    with pytest.raises(ValueError, match="source-row identity"):
        data._dev_source_row_idx_cache = None
        t.plan_final_holdout(data, config, 7)


def test_fixed_torch_refit_never_evaluates_or_stops_early(monkeypatch):
    import torch
    from model_training.ote_training import torch_trainer as trainer
    monkeypatch.setattr(trainer, "choose_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(trainer, "build_torch_model", lambda **kw: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 1), torch.nn.Flatten(0)))
    def forbidden(**kwargs):
        raise AssertionError("Fixed refit must not evaluate validation data")
    monkeypatch.setattr(trainer, "evaluate_torch_model", forbidden)
    _, config = dataset_and_config("torch")
    cfg = t.build_torch_training_config(1, {"focal_alpha": .5, "focal_gamma": 2.0}, config)
    cfg.update(batch_size=8, use_amp=False, preload_to_device=False, num_workers=0, eval_num_workers=0)
    cfg["fixed_epoch_schedule"] = [dict(phase="main", train_fraction=1.0, learning_rate=.001 / (i+1)) for i in range(6)]
    X = np.arange(64, dtype=np.float32).reshape(32, 2, 1) / 64
    result = trainer.train_torch_model(X, np.arange(32) % 2, np.ones(32), X[:0], np.empty(0), np.empty(0), cfg)
    assert result.checkpoint["best_epoch"] == 6
    assert result.checkpoint["selection_mode"] == "fixed_duration_refit"
    assert result.checkpoint["best_val_auprc"] is None
    assert [h["learning_rate"] for h in result.training_history] == [h["learning_rate"] for h in cfg["fixed_epoch_schedule"]]
    with pytest.raises(ValueError, match="no evaluation rows"):
        trainer.train_torch_model(X, np.arange(32) % 2, np.ones(32), X[:1], np.ones(1), np.ones(1), cfg)


@pytest.mark.parametrize("backend", ["xgboost", "torch"])
def test_real_backend_final_refit_smoke(monkeypatch, backend):
    """Exercise both backend APIs, including empty evals and saved TCN checkpoints."""
    import optuna
    data, config = dataset_and_config(backend)
    config.verbosity = 0
    config.epochs = 2
    config.hidden_size = 4
    config.num_layers = 1
    config.batch_size = 64
    config.use_amp = False
    config.torch_num_workers = config.torch_eval_num_workers = 0
    if backend == "torch":
        import torch
        from model_training.ote_training import torch_trainer
        monkeypatch.setattr(torch_trainer, "choose_device", lambda: torch.device("cpu"))
    study = optuna.create_study()
    params = t.build_trial_params(study.ask(), config)
    params.update(window_size=8, lag_count=2, delta_feature_count=0,
                  warmup_rounds=2, main_rounds=2, fine_rounds=2, num_layers=1)
    artifacts = t.TrialArtifacts(params=params, oof_pred=np.full(1200, .2),
        fold_metrics=[], selected_feature_names=["row"], backend=backend, lag_steps=[0, 7], window_size=8, context_rows=7)
    result = t.fit_final_model(data, artifacts, config)
    assert len(result["test_raw"]) == len(data.y_test)
    assert np.isfinite(result["test_raw"]).all()
    if backend == "torch":
        assert result["checkpoint"]["selection_mode"] == "fixed_duration_refit"
        assert result["checkpoint"]["best_epoch"] == len(result["refit_fixed_epoch_schedule"])
    else:
        assert all(h["eval_rows"] == 0 for h in result["training_history"])
