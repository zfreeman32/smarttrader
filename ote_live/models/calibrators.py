from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


def load_probability_calibrator(path: str | Path | None) -> Any | None:
    if path is None:
        return None

    path_obj = Path(path)
    if not path_obj.exists():
        return None
    return joblib.load(path_obj)


def apply_probability_calibrator(
    calibrator: Any | None,
    raw_probabilities: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    probabilities = np.asarray(raw_probabilities, dtype=np.float32).reshape(-1)
    probabilities = np.clip(probabilities, 0.0, 1.0)

    if calibrator is None:
        return probabilities.astype(np.float32, copy=False)

    if hasattr(calibrator, "predict_proba"):
        try:
            calibrated = calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
            return np.asarray(calibrated, dtype=np.float32)
        except AttributeError:
            manual = _manual_binary_logistic_probability(calibrator, probabilities)
            if manual is not None:
                return manual
            raise

    if hasattr(calibrator, "predict"):
        calibrated = calibrator.predict(probabilities)
        return np.asarray(calibrated, dtype=np.float32)

    raise TypeError(
        "Unsupported probability calibrator object. Expected an object with "
        "'predict_proba' or 'predict'."
    )


def _manual_binary_logistic_probability(
    calibrator: Any,
    probabilities: np.ndarray,
) -> np.ndarray | None:
    if not hasattr(calibrator, "coef_") or not hasattr(calibrator, "intercept_"):
        return None

    coef = np.asarray(calibrator.coef_, dtype=np.float32)
    intercept = np.asarray(calibrator.intercept_, dtype=np.float32)
    if coef.ndim != 2 or coef.shape[0] != 1 or coef.shape[1] != 1:
        return None

    logits = (probabilities.reshape(-1, 1) @ coef.T).reshape(-1) + float(intercept.reshape(-1)[0])
    clipped = np.clip(logits, -40.0, 40.0)
    calibrated = 1.0 / (1.0 + np.exp(-clipped))
    return np.asarray(calibrated, dtype=np.float32)
