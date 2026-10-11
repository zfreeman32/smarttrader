"""Causal TCN endpoint mapping, frozen fills and saved runtime replay."""
import json
from types import SimpleNamespace

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest

from model_training.ote_training import ote_xgboost_pipeline as t
from model_training.source_history import FrozenHistoryScaler, selected_contract
from ote_live.models.loaders import validate_source_history_runtime
from scripts.audit_b2_candidate_history import runtime
from test_b2_source_history import source_dataset


def materialize(source):
    indices = source.sampled_indices if source.sampled_indices is not None else np.arange(len(source))
    return np.stack([source.matrix[i:i+source.window_size] for i in indices])


def test_sequence_causality_endpoint_sampling_and_no_future():
    history = np.arange(100, dtype=np.float32)[:, None]
    identity = SimpleNamespace(transform=lambda x: x)
    source = t.indexed_sequence_view(history, [10, 40, 80], [0], identity, 4, 0)
    expected = np.array([[7, 8, 9, 10], [37, 38, 39, 40], [77, 78, 79, 80]])[:, :, None]
    np.testing.assert_array_equal(materialize(source), expected)
    sampled = source.with_sampled_indices(np.array([2, 0, 2])).with_sampled_indices(np.array([1, 2]))
    np.testing.assert_array_equal(materialize(sampled), expected[[0, 2]])
    history[41:] = 999999
    np.testing.assert_array_equal(materialize(t.indexed_sequence_view(history, [40], [0], identity, 4, 0)), expected[1:2])
    with pytest.raises(ValueError, match="Missing causal"):
        t.indexed_sequence_view(history, [2], [0], identity, 4, 0)
    with pytest.raises(ValueError, match="outside"):
        source.with_sampled_indices(np.array([-1]))


def test_tcn_fold_scaler_sees_only_training_endpoints(monkeypatch):
    data, cfg, _ = source_dataset()
    calls = []
    original = t.fit_scaler
    def spy(matrix, config):
        calls.append(matrix.copy())
        return original(matrix, config)
    monkeypatch.setattr(t, "fit_scaler", spy)
    train, val = np.arange(80, 200), np.arange(230, 330)
    result = t.build_sequence_fold_datasets(data.dev_X, data.dev_y, data.dev_w, train, val,
        np.array([1]), 8, cfg, source_history=data.source_history_X, source_row_idx=data.dev_source_row_idx)
    assert len(result[0]) == len(train) and len(result[3]) == len(val)
    np.testing.assert_array_equal(calls[0], data.dev_X[train, 1:2])
    for position, rows in ((0, train), (3, val)):
        expected = np.stack([t.transform_matrix(original(calls[0], cfg),
            data.source_history_X[end-7:end+1, 1:2], cfg.scale_clip) for end in data.dev_source_row_idx[rows]])
        np.testing.assert_array_equal(materialize(result[position]), expected)
        np.testing.assert_array_equal(result[position+1], data.dev_y[rows])
        np.testing.assert_array_equal(result[position+2], data.dev_w[rows])


def test_tcn_cv_holdout_refit_save_load_and_raw_missing_runtime(tmp_path, monkeypatch):
    import torch
    from model_training.ote_training import torch_trainer
    monkeypatch.setattr(torch_trainer, "choose_device", lambda: torch.device("cpu"))
    torch.set_num_threads(1)
    data, cfg, raw = source_dataset()
    cfg.backend, cfg.model_type = "torch", "tcn"
    cfg.epochs, cfg.hidden_size, cfg.num_layers, cfg.batch_size = 1, 4, 1, 64
    cfg.use_amp, cfg.use_balanced_tuning_sample = False, True
    cfg.num_workers = 0
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = t.build_trial_params(trial, cfg)
    artifacts = t.cross_validate_trial(data, params, cfg, trial=None)
    assert all(d["train_rows_model"] == d["train_rows_raw"] for d in artifacts.fold_diagnostics)
    calls = []
    train = t.train_backend_model
    def spy(**kw):
        calls.append((len(kw["X_train"]), len(kw["y_train"]), len(kw["y_eval"])))
        return train(**kw)
    monkeypatch.setattr(t, "train_backend_model", spy)
    result = t.fit_final_model(data, artifacts, cfg)
    holdout = result["final_holdout"]
    assert calls == [(holdout["fit_end"], holdout["fit_end"], 600-holdout["val_start"]), (600, 600, 0)]
    assert result["checkpoint"]["selection_mode"] == "fixed_duration_refit"
    assert result["checkpoint"]["best_val_auprc"] is None
    study.tell(trial, .5)
    t.save_training_outputs(tmp_path, data, cfg, artifacts, result, study)
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    scaler = joblib.load(tmp_path / "scaler.joblib")
    assert isinstance(scaler, FrozenHistoryScaler)
    runner = runtime(tmp_path, summary)
    predictions = []
    for end in data.test_source_row_idx:
        frame = pd.DataFrame(raw[end-7:end+1], columns=data.feature_names)
        frame["source_row_idx"] = np.arange(end-7, end+1)
        frame["datetime"] = pd.date_range("2025-01-01", periods=8, freq="5min", tz="UTC")
        # Column order in an incoming frame must not change the model tensor.
        predictions.append(runner.predict_latest(frame[frame.columns[::-1]]).raw_score)
    np.testing.assert_allclose(predictions, result["test_raw"], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.array(predictions) >= result["threshold"], result["test_raw"] >= result["threshold"])
    frame.loc[3, "source_row_idx"] -= 1
    with pytest.raises(ValueError, match="consecutive"):
        runner.predict_latest(frame)
    with pytest.raises(ValueError, match="mismatched"):
        validate_source_history_runtime(runner.loaded_model.manifest, summary, scaler.scaler)


def test_catchup_roster_admits_extension_ids_only_into_new_roots():
    from scripts import run_b2_candidate_history as es
    from scripts import run_b2_ote_candidate_history as ote
    assert es.CATCHUP_IDS == (5, 14, 18)
    assert set(es.CANDIDATE_IDS) == {2, 1, 11, 17, 7, 3, 10, 12, 13, 15, 16, 19, 5, 14, 18}
    name = "b2_candidate_history_catchup_es_20261001_v1"
    audit = es.ROOT / "docs/audits" / f"{name[:-3]}.json"
    for model_id in es.CATCHUP_IDS:
        es.require_catchup_roots(model_id, es.ROOT / "artifacts" / name, es.ROOT / "models" / name, audit)
        with pytest.raises(ValueError, match="catch-up roots"):
            es.require_catchup_roots(model_id, es.PREPARED, es.ROOT / "models" / name, audit)
        with pytest.raises(ValueError, match="catch-up roots"):
            es.require_catchup_roots(model_id, es.ROOT / "artifacts" / name,
                es.ROOT / "models/b2_candidate_history_minimum_core_es_20260929_v1", audit)
        with pytest.raises(ValueError, match="audit report"):
            es.require_catchup_roots(model_id, es.ROOT / "artifacts" / name, es.ROOT / "models" / name, es.AUDIT)
    # Earlier IDs keep their existing roots.
    es.require_catchup_roots(19, es.PREPARED, es.OUTPUT, es.AUDIT)
    reserve = es.ROOT / "artifacts/b2_candidate_history_promoted_reserve_ote_20261001_v1"
    ote.require_phase(23, reserve, expand_to_12=False, promote_reserve=True)
    assert ote.resolve_group(reserve, 23) == (23,)
    with pytest.raises(ValueError, match="reserve candidates"):
        ote.require_phase(23, reserve, expand_to_12=False, promote_reserve=False)
    with pytest.raises(ValueError, match="promoted reserve root"):
        ote.require_phase(23, ote.DEFAULT_PREPARED, expand_to_12=False, promote_reserve=True)


def test_selected_subset_contract_orders_fills():
    data, _, _ = source_dataset()
    c = selected_contract(data.source_history_contract, ["b"])
    assert c["feature_names"] == ["b"] and c["fill_values"] == {"b": -.75}
    scaler = FrozenHistoryScaler(SimpleNamespace(transform=lambda x: x), c)
    np.testing.assert_array_equal(scaler.transform([[np.nan], [np.inf]]), [[-.75], [-.75]])
