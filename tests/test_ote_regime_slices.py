from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_testing.ote_regime_slices import (
    RegimeSliceConfig,
    build_regime_slice_report,
    build_regime_winner_table,
    group_positive_zones_by_position,
)


def test_group_positive_zones_by_position_respects_gaps() -> None:
    zones = group_positive_zones_by_position(
        y_true=np.array([1, 1, 1], dtype=np.uint8),
        positions=np.array([0, 2, 3], dtype=np.int64),
    )

    assert zones == [(0, 0), (2, 3)]


def test_build_regime_slice_report_uses_event_counts_and_threshold_search() -> None:
    frame = pd.DataFrame(
        {
            "source_row_idx": [0, 1, 2, 3, 4, 5, 10, 12, 13, 14],
            "target": [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
            "sample_weight": [1.0] * 10,
            "calibrated_probability": [0.05, 0.90, 0.70, 0.20, 0.85, 0.10, 0.80, 0.75, 0.20, 0.10],
            "composite_regime": [
                "strong_up_high",
                "strong_up_high",
                "strong_up_high",
                "strong_up_high",
                "strong_up_high",
                "strong_up_high",
                "ranging_low",
                "ranging_low",
                "ranging_low",
                "ranging_low",
            ],
            "year": [2024] * 10,
        }
    )

    config = RegimeSliceConfig(
        dataset_split="test",
        probability_column="calibrated_probability",
        threshold=0.60,
        event_tolerance_bars=0,
        event_cooldown_bars=0,
        min_positive_events_for_threshold=2,
        bootstrap_iterations=20,
        random_seed=7,
        slice_families=("composite_regime",),
    )
    report = build_regime_slice_report(
        frame,
        model_id="demo_model",
        direction="long",
        backend="xgboost",
        role="candidate",
        status="active",
        config=config,
    )

    strong_up_row = report.loc[report["regime_key"] == "strong_up_high"].iloc[0]
    ranging_row = report.loc[report["regime_key"] == "ranging_low"].iloc[0]

    assert int(strong_up_row["n_positive"]) == 2
    assert int(strong_up_row["positive_row_count"]) == 3
    assert bool(strong_up_row["threshold_data_sufficient"]) is True
    assert float(strong_up_row["optimal_threshold"]) >= 0.40
    assert float(strong_up_row["optimal_event_f05"]) >= float(strong_up_row["event_f05"])

    assert int(ranging_row["n_positive"]) == 2
    assert bool(ranging_row["threshold_data_sufficient"]) is True
    assert float(ranging_row["ap_ci_upper"]) >= float(ranging_row["ap_ci_lower"])


def test_build_regime_winner_table_flags_confident_bucket_winner() -> None:
    report = pd.DataFrame(
        [
            {
                "dataset_split": "test",
                "direction": "short",
                "regime_family": "composite_regime",
                "regime_key": "strong_down_high",
                "model_id": "short_ote_candidate_tcn_v1",
                "backend": "tcn",
                "ap": 0.72,
                "ap_ci_lower": 0.68,
                "ap_ci_upper": 0.76,
                "event_f05": 0.70,
            },
            {
                "dataset_split": "test",
                "direction": "short",
                "regime_family": "composite_regime",
                "regime_key": "strong_down_high",
                "model_id": "short_ote_candidate_xgb_v1",
                "backend": "xgboost",
                "ap": 0.59,
                "ap_ci_lower": 0.52,
                "ap_ci_upper": 0.61,
                "event_f05": 0.64,
            },
        ]
    )

    winners = build_regime_winner_table(report)
    row = winners.iloc[0]

    assert row["winner_model_id"] == "short_ote_candidate_tcn_v1"
    assert bool(row["winner_confident"]) is True
    assert float(row["confidence_gap"]) > 0.0


def test_build_regime_slice_report_supports_frvp_open_and_day_type_labels() -> None:
    frame = pd.DataFrame(
        {
            "source_row_idx": [0, 1, 2, 3],
            "target": [1, 0, 1, 0],
            "calibrated_probability": [0.9, 0.2, 0.8, 0.1],
            "frvp_open_type_label": [
                "above_value",
                "above_value",
                "inside_value",
                "inside_value",
            ],
            "frvp_open_type_status": [
                "classified",
                "classified",
                "not_yet_known",
                "not_yet_known",
            ],
            "frvp_day_type_label": [
                "trend",
                "trend",
                "neutral",
                "neutral",
            ],
        }
    )

    config = RegimeSliceConfig(
        dataset_split="test",
        probability_column="calibrated_probability",
        threshold=0.60,
        event_tolerance_bars=0,
        event_cooldown_bars=0,
        min_positive_events_for_threshold=1,
        bootstrap_iterations=10,
        random_seed=11,
        slice_families=("frvp_open_type_status", "frvp_open_type_label", "frvp_day_type_label"),
    )

    report = build_regime_slice_report(
        frame,
        model_id="demo_model",
        direction="long",
        config=config,
    )

    assert set(report["regime_family"]) == {"frvp_open_type_status", "frvp_open_type_label", "frvp_day_type_label"}
    assert set(report["regime_key"]) == {
        "classified",
        "not_yet_known",
        "above_value",
        "inside_value",
        "trend",
        "neutral",
    }
