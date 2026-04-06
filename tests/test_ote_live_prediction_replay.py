from __future__ import annotations

import importlib.util
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ote_live.models.replay import replay_prediction_parity

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw_part in version.split("."):
        digits = "".join(character for character in raw_part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _supports_exact_xgboost_replay() -> bool:
    try:
        sklearn = importlib.import_module("sklearn")
        xgboost = importlib.import_module("xgboost")
    except ImportError:
        return False
    return _version_tuple(sklearn.__version__) >= (1, 8) and _version_tuple(xgboost.__version__) >= (3, 2)


def test_replay_prediction_parity_matches_xgboost_reference_head_rows() -> None:
    if not _supports_exact_xgboost_replay():
        pytest.skip("Exact XGBoost artifact replay expects the sklearn/xgboost toolchain from ote_venv.")

    report = replay_prediction_parity(
        SHORT_XGB_V2_MANIFEST_PATH,
        evaluation_rows=4,
    )

    assert report.rows_evaluated == 4
    assert report.raw_probability_mismatches == 0
    assert report.calibrated_probability_mismatches == 0
    assert report.predicted_label_compared_rows == 4
    assert report.predicted_label_mismatches == 0
    assert report.max_raw_probability_abs_diff is not None
    assert report.max_raw_probability_abs_diff <= report.raw_probability_atol
    assert report.max_calibrated_probability_abs_diff is not None
    assert report.max_calibrated_probability_abs_diff <= report.calibrated_probability_atol
    assert report.sample_mismatches == ()


def test_replay_prediction_parity_matches_tcn_reference_head_rows_when_torch_available() -> None:
    if not TORCH_AVAILABLE:
        pytest.skip("torch is not available in this environment.")

    report = replay_prediction_parity(
        LONG_TCN_V2_MANIFEST_PATH,
        evaluation_rows=4,
        raw_probability_atol=5e-5,
        calibrated_probability_atol=1e-6,
    )

    assert report.rows_evaluated == 4
    assert report.raw_probability_mismatches == 0
    assert report.calibrated_probability_mismatches == 0
    assert report.predicted_label_compared_rows == 4
    assert report.predicted_label_mismatches == 0
    assert report.max_raw_probability_abs_diff is not None
    assert report.max_raw_probability_abs_diff <= report.raw_probability_atol
    assert report.max_calibrated_probability_abs_diff is not None
    assert report.max_calibrated_probability_abs_diff <= report.calibrated_probability_atol


def test_replay_prediction_parity_matches_lstm_reference_head_row_when_torch_available() -> None:
    if not TORCH_AVAILABLE:
        pytest.skip("torch is not available in this environment.")

    report = replay_prediction_parity(
        LONG_LSTM_V2_MANIFEST_PATH,
        evaluation_rows=1,
        raw_probability_atol=2e-4,
        calibrated_probability_atol=1e-6,
    )

    assert report.rows_evaluated == 1
    assert report.raw_probability_mismatches == 0
    assert report.calibrated_probability_mismatches == 0
    assert report.predicted_label_compared_rows == 1
    assert report.predicted_label_mismatches == 0
