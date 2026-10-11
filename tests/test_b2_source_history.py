"""Source-bar window causality, scaler isolation, and saved runtime parity."""
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest
import xgboost as xgb

from model_training.ote_training import ote_xgboost_pipeline as t
from model_training.source_history import MODE, FrozenHistoryScaler, impute_matrix, indexed_lag_view
from ote_live.models.loaders import validate_source_history_runtime
from scripts.run_b2_source_history import runtime_model


def contract(names):
    return dict(mode=MODE, bar_minutes=5, feature_names=names,
                fill_values={n: -0.75 for n in names}, missing_value_rule="nonfinite_to_frozen_fill")


def source_dataset():
    ids = 30 + np.arange(640) * 10
    raw = np.column_stack([np.sin(np.arange(ids[-1]+1)/17), np.cos(np.arange(ids[-1]+1)/13)]).astype(np.float32)
    raw[::11, 1] = np.nan
    c = contract(["a", "b"])
    history = impute_matrix(raw, c)
    X, y, w = history[ids], (np.arange(len(ids)) % 3 == 0).astype(np.uint8), np.ones(len(ids), np.float32)
    report = dict(split_geometry=dict(applied_embargo_bars=20), selection_contract=dict(
        mode="frozen_development_prefix", fit_rows=100, fit_end_source_row_idx=int(ids[99]), required_gap_bars=20))
    data = t.PreparedTargetDataset("long_frvp_continuation_setup3", Path("unused"), ["a", "b"], ["a", "b"], report,
        X[:450], y[:450], w[:450], X[450:600], y[450:600], w[450:600], X[600:], y[600:], w[600:],
        ids[:450], ids[450:600], ids[600:], source_history_X=history, source_history_contract=c)
    cfg = t.OTETrainingConfig(backend="xgboost", window_min=8, window_max=8, window_size=8,
        cv_initial_train_rows=200, cv_val_rows=100, cv_step_rows=100, cv_min_folds=2,
        min_train_positive_rows=10, min_val_positive_rows=10, min_val_true_events=3,
        final_eval_min_rows=64, label_max_holding_bars=10, label_exclusion_pre_bars=0,
        label_zone_pre_bars=0, event_tolerance_bars=0, event_cooldown_bars=0, purge_buffer_bars=2,
        calibration_method="none", sequential_bootstrap_final_refit=False, use_balanced_tuning_sample=False,
        verbosity=0)
    return data, cfg, raw


def test_sparse_endpoints_use_unlabelled_bar_history_and_no_future():
    history = np.arange(100, dtype=np.float32)[:, None]
    identity = SimpleNamespace(transform=lambda x: x)
    result = indexed_lag_view(history, [10, 40], [0], identity, [0, 1, 3], 1, 0)
    np.testing.assert_array_equal(result, [[10, 9, 7, 1, 3], [40, 39, 37, 1, 3]])
    # A future outlier and adding/removing other supervised endpoints cannot
    # alter an earlier example's inputs.
    history[41:] = 99999
    np.testing.assert_array_equal(indexed_lag_view(history, [40], [0], identity, [0, 1, 3], 1, 0), result[1:])
    with pytest.raises(ValueError, match="Missing causal"):
        indexed_lag_view(history, [2], [0], identity, [0, 3], 0, 0)


def test_fold_scaler_sees_only_fit_endpoints_and_keeps_first_label(monkeypatch):
    data, cfg, _ = source_dataset()
    captured = []
    fit = t.fit_scaler
    def spy(matrix, config):
        captured.append(matrix.copy())
        return fit(matrix, config)
    monkeypatch.setattr(t, "fit_scaler", spy)
    train, val = np.arange(200), np.arange(230, 330)
    result = t.build_xgboost_fold_datasets(data.dev_X, data.dev_y, data.dev_w, train, val,
        np.array([0, 1]), [0, 7], 1, cfg, source_history=data.source_history_X,
        source_row_idx=data.dev_source_row_idx)
    assert len(result[0]) == len(result[1]) == 200  # No retained-row context drop.
    np.testing.assert_array_equal(captured[0], data.dev_X[train])
    expected = t.transform_matrix(fit(data.dev_X[train], cfg), data.source_history_X[[data.dev_source_row_idx[0]-7]], cfg.scale_clip)
    np.testing.assert_array_equal(result[0][0, 2:4], expected[0])


def test_real_cv_refit_save_and_raw_runtime_prediction_parity(tmp_path, monkeypatch):
    data, cfg, raw = source_dataset()
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = t.build_trial_params(trial, cfg)
    params.update(window_size=8, lag_count=2, delta_feature_count=1,
                  warmup_rounds=1, main_rounds=1, fine_rounds=1)
    artifacts = t.cross_validate_trial(data, params, cfg, trial=None)
    assert all(d["train_rows_model"] == d["train_rows_raw"] for d in artifacts.fold_diagnostics)
    calls = []
    train = t.train_backend_model
    def record_fit(**kwargs):
        calls.append((len(kwargs["y_train"]), len(kwargs["y_eval"])))
        return train(**kwargs)
    monkeypatch.setattr(t, "train_backend_model", record_fit)
    result = t.fit_final_model(data, artifacts, cfg)
    assert len(result["test_raw"]) == 40
    assert calls == [(result["final_holdout"]["fit_end"], 600-result["final_holdout"]["val_start"]), (600, 0)]
    assert all(h["eval_rows"] == 0 for h in result["training_history"])
    study.tell(trial, .5)
    t.save_training_outputs(tmp_path, data, cfg, artifacts, result, study)
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    scaler = joblib.load(tmp_path / "scaler.joblib")
    assert isinstance(scaler, FrozenHistoryScaler)
    booster = xgb.Booster(); booster.load_model(tmp_path / "model.json")
    runtime = runtime_model(booster, scaler, summary)
    probabilities = []
    for endpoint in data.test_source_row_idx:
        frame = pd.DataFrame(raw[endpoint-7:endpoint+1], columns=data.feature_names,
                             index=np.arange(endpoint-7, endpoint+1))
        probabilities.append(runtime.predict_latest(frame, timestamp=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime()).raw_score)
    np.testing.assert_allclose(probabilities, result["test_raw"], atol=1e-7, rtol=1e-6)
    saved = pd.read_csv(tmp_path / "test_predictions.csv")
    np.testing.assert_array_equal(np.array(probabilities) >= result["threshold"], saved.predicted_label)
    # The new runtime must refuse filtered source histories.
    bad = pd.DataFrame(raw[[1, 3, 5, 7, 9, 11, 13, 15]], columns=data.feature_names,
                       index=[1, 3, 5, 7, 9, 11, 13, 15])
    with pytest.raises(ValueError, match="consecutive"):
        runtime._coerce_ordered_feature_frame(bad)
    with pytest.raises(ValueError, match="mismatched"):
        validate_source_history_runtime(runtime.loaded_model.manifest, summary, scaler.scaler)


def test_runtime_fill_is_frozen_for_nan_and_infinities():
    c = contract(["a", "b"])
    s = FrozenHistoryScaler(SimpleNamespace(transform=lambda x: x), c)
    result = s.transform([[np.nan, np.inf], [-np.inf, 3.]])
    np.testing.assert_array_equal(result, [[-.75, -.75], [-.75, 3.]])
