from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_testing.ote_regime_labeler import RegimeLabelConfig, label_regimes


def test_label_regimes_assigns_expected_primary_buckets() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-01-02 01:00:00",
                    "2024-01-02 07:30:00",
                    "2024-01-02 14:00:00",
                    "2024-01-02 21:00:00",
                    "2024-01-02 23:00:00",
                ]
            ),
            "close": [1.10, 1.20, 1.30, 1.15, 1.05],
            "ema_50": [1.00, 1.10, 1.30, 1.20, 1.10],
            "adx_14": [35.0, 25.0, 10.0, 22.0, 40.0],
            "ema_alignment": [0.9, 0.2, 0.0, -0.2, -0.9],
            "atr_14": [3.0, 1.0, 2.0, 4.0, 0.5],
            "range_shock_20": [1.5, 2.5, 3.5, 1.0, 2.0],
            "frvp_open_type": [1, 0, -1, 1, 0],
            "frvp_day_type": [3, 1, 5, 4, 2],
        }
    )

    labeled = label_regimes(frame, config=RegimeLabelConfig(atr_percentile_window=5))

    assert labeled["trend_regime"].tolist() == [
        "strong_up",
        "weak_up",
        "ranging",
        "weak_down",
        "strong_down",
    ]
    assert labeled["session_regime"].tolist() == [
        "asia",
        "london",
        "overlap",
        "new_york",
        "off_hours",
    ]
    assert labeled["stress_regime"].tolist() == [
        "normal",
        "elevated",
        "high",
        "normal",
        "elevated",
    ]
    assert labeled["vol_regime"].tolist() == [
        "high",
        "medium",
        "medium",
        "high",
        "low",
    ]
    assert labeled["composite_regime"].tolist() == [
        "strong_up_high",
        "weak_up_medium",
        "ranging_medium",
        "weak_down_high",
        "strong_down_low",
    ]
    assert labeled["frvp_open_type_label"].tolist() == [
        "above_value",
        "inside_value",
        "below_value",
        "above_value",
        "inside_value",
    ]
    assert labeled["frvp_day_type_label"].tolist() == [
        "trend",
        "normal",
        "neutral",
        "double_distribution_trend",
        "normal_variation",
    ]
    assert labeled["year"].astype(int).tolist() == [2024, 2024, 2024, 2024, 2024]


def test_label_regimes_uses_hour_sine_cosine_and_close_bias_fallbacks() -> None:
    hours = np.array([3.0, 20.0], dtype=np.float64)
    angle = (hours / 24.0) * (2.0 * np.pi)
    frame = pd.DataFrame(
        {
            "adx_14": [24.0, 24.0],
            "ema_alignment": [0.1, -0.1],
            "close_vs_ema_50_atr": [0.3, -0.2],
            "atr_14": [1.0, 0.5],
            "range_shock_20": [1.0, 4.0],
            "hour_sin": np.sin(angle),
            "hour_cos": np.cos(angle),
        }
    )

    labeled = label_regimes(frame, config=RegimeLabelConfig(atr_percentile_window=2))

    assert labeled["trend_regime"].tolist() == ["weak_up", "weak_down"]
    assert labeled["session_regime"].tolist() == ["london", "asia"]
    assert labeled["stress_regime"].tolist() == ["normal", "high"]
    assert labeled["session_hour_utc"].astype(int).tolist() == [3, 20]


def test_label_regimes_falls_back_to_ema_alignment_when_adx_is_missing() -> None:
    frame = pd.DataFrame(
        {
            "close_vs_ema_50_atr": [0.8, 0.3, 0.0, -0.3, -0.8],
            "ema_alignment": [0.9, 0.25, 0.0, -0.25, -0.9],
            "atr_14": [1.0, 1.1, 1.2, 1.3, 1.4],
            "range_shock_20": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    labeled = label_regimes(frame, config=RegimeLabelConfig(atr_percentile_window=5))

    assert labeled["trend_regime"].tolist() == [
        "strong_up",
        "weak_up",
        "ranging",
        "weak_down",
        "strong_down",
    ]


def test_label_regimes_splits_open_type_status_between_preopen_and_after_open_gaps() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-01-02 13:00:00+00:00",
                    "2024-01-02 15:00:00+00:00",
                    "2024-01-02 19:00:00+00:00",
                    "2024-01-02 23:30:00+00:00",
                ]
            ),
            "close": [1.10, 1.20, 1.25, 1.15],
            "ema_50": [1.00, 1.10, 1.20, 1.10],
            "adx_14": [20.0, 20.0, 20.0, 20.0],
            "ema_alignment": [0.2, 0.2, 0.2, 0.2],
            "atr_14": [1.0, 1.0, 1.0, 1.0],
            "range_shock_20": [1.0, 1.0, 1.0, 1.0],
            "frvp_open_type": [np.nan, np.nan, 1.0, np.nan],
            "frvp_day_type": [1, 1, 1, 1],
        }
    )

    labeled = label_regimes(frame, config=RegimeLabelConfig(atr_percentile_window=4))

    assert labeled["frvp_open_type_status"].tolist() == [
        "not_yet_known",
        "missing_after_open",
        "classified",
        "not_yet_known",
    ]
    assert labeled["frvp_open_type_label"].fillna("missing").tolist() == [
        "missing",
        "missing",
        "above_value",
        "missing",
    ]
