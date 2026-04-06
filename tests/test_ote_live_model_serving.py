from __future__ import annotations

from datetime import UTC, datetime, timedelta
import importlib.util
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ote_live.contracts.prediction import ModelPrediction
from ote_live.models.ensemble import load_direction_models
from ote_live.models.loaders import load_live_runtime_manifest, load_runtime_model
from ote_live.models.runners import RuntimeModelRunner
from ote_live.policies.decision_engine import LiveDecisionEngine
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

REPO_ROOT = Path(__file__).resolve().parents[1]
LONG_BUNDLE_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_long.json"
SHORT_XGB_V2_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "short_ote_xgb_v2_candidate" / "live_runtime_manifest.json"
)
LONG_TCN_V2_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "long_ote_tcn_v2_candidate" / "live_runtime_manifest.json"
)
LONG_LSTM_V2_MANIFEST_PATH = (
    REPO_ROOT / "ote_live" / "runtime_manifests" / "long_ote_lstm_v2_candidate" / "live_runtime_manifest.json"
)
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _synthetic_feature_frame(feature_names: list[str], rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    values = rng.normal(loc=0.0, scale=1.0, size=(rows, len(feature_names))).astype(np.float32)
    frame = pd.DataFrame(values, columns=feature_names)
    frame["timestamp"] = [datetime(2026, 4, 1, tzinfo=UTC) + timedelta(minutes=5 * index) for index in range(rows)]
    frame["source_row_idx"] = np.arange(10_000, 10_000 + rows, dtype=np.int64)
    return frame


def test_runtime_model_loader_supports_xgboost_artifacts() -> None:
    loaded = load_runtime_model(SHORT_XGB_V2_MANIFEST_PATH)

    assert loaded.model_id == "short_ote_xgb_v2_candidate"
    assert loaded.backend == "xgboost"
    assert loaded.scaler is not None
    assert loaded.calibrator is not None
    assert loaded.scale_clip > 0


def test_runtime_model_runner_produces_prediction_from_feature_frame() -> None:
    loaded = load_runtime_model(SHORT_XGB_V2_MANIFEST_PATH)
    runner = RuntimeModelRunner(loaded)
    frame = _synthetic_feature_frame(
        loaded.manifest.feature_manifest.selected_feature_names,
        rows=loaded.context_rows + 5,
    )

    prediction = runner.predict_latest(frame)

    assert prediction.model_id == loaded.model_id
    assert prediction.backend == "xgboost"
    assert 0.0 <= prediction.raw_score <= 1.0
    assert 0.0 <= prediction.calibrated_probability <= 1.0
    assert prediction.timestamp == frame["timestamp"].iloc[-1].to_pydatetime()
    assert prediction.source_row_idx == int(frame["source_row_idx"].iloc[-1])


def test_direction_model_loader_can_skip_unavailable_torch_backends() -> None:
    bundle = load_direction_models(
        LONG_BUNDLE_PATH,
        skip_unavailable_backends=True,
    )

    if TORCH_AVAILABLE:
        assert "long_ote_champion_v1" in bundle.loaded_models
        assert "long_ote_benchmark_lstm_v1" in bundle.loaded_models
        assert bundle.unavailable_models == {}
        assert bundle.primary_model is not None
        assert bundle.primary_model.backend == "tcn"
        assert bundle.primary_model.model_id == "long_ote_champion_v1"
    else:
        assert "long_ote_champion_v1" in bundle.unavailable_models
        assert "long_ote_benchmark_lstm_v1" in bundle.unavailable_models
        assert bundle.loaded_models == {}
        assert bundle.primary_model is None


def test_runtime_model_loader_handles_torch_dependency() -> None:
    if TORCH_AVAILABLE:
        tcn_loaded = load_runtime_model(LONG_TCN_V2_MANIFEST_PATH)
        lstm_loaded = load_runtime_model(LONG_LSTM_V2_MANIFEST_PATH)
        assert tcn_loaded.backend == "tcn"
        assert lstm_loaded.backend == "lstm"
    else:
        with pytest.raises(ImportError, match="torch"):
            load_runtime_model(LONG_TCN_V2_MANIFEST_PATH)


def test_runtime_model_runner_supports_torch_backends_when_available() -> None:
    if not TORCH_AVAILABLE:
        pytest.skip("torch is not available in this environment.")

    for manifest_path, backend in (
        (LONG_TCN_V2_MANIFEST_PATH, "tcn"),
        (LONG_LSTM_V2_MANIFEST_PATH, "lstm"),
    ):
        loaded = load_runtime_model(manifest_path)
        runner = RuntimeModelRunner(loaded)
        frame = _synthetic_feature_frame(
            loaded.manifest.feature_manifest.selected_feature_names,
            rows=loaded.context_rows + 5,
        )

        prediction = runner.predict_latest(frame)

        assert prediction.backend == backend
        assert 0.0 <= prediction.raw_score <= 1.0
        assert 0.0 <= prediction.calibrated_probability <= 1.0


def test_decision_engine_applies_global_threshold_and_cooldown() -> None:
    manifest = load_live_runtime_manifest(SHORT_XGB_V2_MANIFEST_PATH)
    engine = LiveDecisionEngine()
    threshold = float(manifest.live_policy.thresholds.global_threshold)

    loaded = load_runtime_model(SHORT_XGB_V2_MANIFEST_PATH)
    timestamp = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    base_prediction = RuntimeModelRunner(loaded).predict_latest(
        _synthetic_feature_frame(
            loaded.manifest.feature_manifest.selected_feature_names,
            rows=loaded.context_rows + 2,
        )
    ).model_copy(
        update={
            "timestamp": timestamp,
            "source_row_idx": 200,
            "calibrated_probability": threshold + 0.10,
            "raw_score": threshold + 0.08,
        }
    )

    first = engine.evaluate_prediction(
        base_prediction,
        live_policy=manifest.live_policy,
        policy_context={"session_regime": "new_york", "stress_regime": "normal"},
    )
    assert first.signal.decision == "emit"
    assert first.signal.reasons == ["threshold_passed"]

    second_prediction = base_prediction.model_copy(
        update={
            "timestamp": timestamp + timedelta(minutes=5),
            "source_row_idx": 201,
        }
    )
    second = engine.evaluate_prediction(
        second_prediction,
        live_policy=manifest.live_policy,
        policy_context={"session_regime": "new_york", "stress_regime": "normal"},
    )

    assert second.signal.decision == "abstain"
    assert second.signal.reasons[0] == "cooldown"
    assert second.signal.cooldown_bars_remaining is not None


def test_decision_engine_uses_regime_threshold_when_available() -> None:
    manifest = load_live_runtime_manifest(LONG_TCN_V2_MANIFEST_PATH)
    live_policy = manifest.live_policy.model_copy(
        update={
            "thresholds": manifest.live_policy.thresholds.model_copy(
                update={
                    "global_threshold": 0.90,
                    "regime_thresholds": {"strong_down_high": 0.70},
                }
            )
        }
    )
    engine = LiveDecisionEngine()

    prediction = ModelPrediction(
        model_id=manifest.model_id,
        direction=manifest.direction,
        backend=manifest.backend,
        timestamp=datetime(2026, 4, 1, 15, 0, tzinfo=UTC),
        source_row_idx=500,
        regime=None,
        raw_score=0.72,
        calibrated_probability=0.75,
        threshold_applied=None,
        threshold_source="unknown",
    )

    path = engine.evaluate_prediction(
        prediction,
        live_policy=live_policy,
        policy_context={"composite_regime": "strong_down_high", "session_regime": "new_york", "stress_regime": "normal"},
    )

    assert path.prediction.threshold_source == "regime"
    assert path.prediction.threshold_applied == pytest.approx(0.70)
    assert path.signal.decision == "emit"


def test_decision_engine_run_runner_persists_emitted_signals_when_audit_repo_is_configured(monkeypatch) -> None:
    loaded = load_runtime_model(SHORT_XGB_V2_MANIFEST_PATH)
    runner = RuntimeModelRunner(loaded)
    feature_frame = _synthetic_feature_frame(
        list(loaded.manifest.feature_manifest.selected_feature_names),
        rows=loaded.context_rows + 3,
    )

    def _predict_latest(_frame: pd.DataFrame, **_: object) -> ModelPrediction:
        return ModelPrediction(
            model_id=loaded.model_id,
            direction=loaded.manifest.direction,
            backend=loaded.backend,
            timestamp=feature_frame["timestamp"].iloc[-1].to_pydatetime(),
            source_row_idx=int(feature_frame["source_row_idx"].iloc[-1]),
            regime=None,
            raw_score=0.72,
            calibrated_probability=0.80,
            threshold_applied=None,
            threshold_source="unknown",
        )

    monkeypatch.setattr(runner, "predict_latest", _predict_latest)

    tmp_root = REPO_ROOT / "tmp" / "ote_live_model_serving_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        engine = LiveDecisionEngine(audit_repository=audit)

        path = engine.run_runner(
            runner,
            feature_frame,
            policy_context={"session_regime": "new_york", "stress_regime": "normal"},
        )
        assert path.audit_record is not None
        reconstructed = audit.reconstruct_signal(path.audit_record.signal_decision_id)

    assert path.signal.decision == "emit"
    assert reconstructed.signal == path.signal
    assert reconstructed.prediction == path.prediction
    assert reconstructed.runtime_manifest is not None
    assert reconstructed.runtime_manifest.manifest.model_id == loaded.model_id
    assert reconstructed.feature_snapshot.direction == loaded.manifest.direction
    assert reconstructed.feature_snapshot.source_row_idx == int(feature_frame["source_row_idx"].iloc[-1])
