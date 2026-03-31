from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_testing.ote_abstain_policy import HardAbstainConfig
from model_testing.ote_threshold_policy import (
    ThresholdSearchConfig,
    attach_forward_trade_outcomes,
    evaluate_policy_variants,
    search_thresholds_by_composite_regime,
)


def test_search_thresholds_by_composite_regime_uses_regime_and_global_fallback() -> None:
    frame = pd.DataFrame(
        {
            "source_row_idx": [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:05:00",
                    "2024-01-01 00:10:00",
                    "2024-01-01 00:15:00",
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:05:00",
                    "2024-02-01 00:00:00",
                    "2024-02-01 00:05:00",
                    "2025-02-01 00:00:00",
                    "2025-02-01 00:05:00",
                ]
            ),
            "year": [2024, 2024, 2024, 2024, 2025, 2025, 2024, 2024, 2025, 2025],
            "target": [0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
            "oof_calibrated_probability": [0.1, 0.9, 0.8, 0.2, 0.45, 0.1, 0.2, 0.7, 0.1, 0.1],
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
            "session_regime": ["london"] * 10,
            "stress_regime": ["normal"] * 10,
            "policy_gross_pnl_pips": [-3.0, 18.0, 16.0, -2.0, 7.0, -1.0, -2.0, 8.0, -1.0, -1.0],
            "policy_net_pnl_pips": [-5.0, 14.0, 12.0, -4.0, 4.0, -3.0, -4.0, 5.0, -3.0, -3.0],
        }
    )

    config = ThresholdSearchConfig(
        probability_column="oof_calibrated_probability",
        global_threshold=0.6,
        threshold_grid=(0.4, 0.6, 0.8),
        event_tolerance_bars=0,
        event_cooldown_bars=0,
        min_positive_events=2,
        min_events_per_month=0.0,
    )

    policy_table, global_result = search_thresholds_by_composite_regime(
        frame,
        direction="short",
        config=config,
    )

    strong_up = policy_table.loc[policy_table["composite_regime"] == "strong_up_high"].iloc[0]
    ranging_low = policy_table.loc[policy_table["composite_regime"] == "ranging_low"].iloc[0]

    assert float(global_result["threshold"]) in {0.4, 0.6, 0.8}
    assert strong_up["threshold_source"] == "regime"
    assert float(strong_up["threshold"]) == 0.4
    assert bool(strong_up["threshold_data_sufficient"]) is True
    assert ranging_low["threshold_source"] == "global_fallback"
    assert bool(ranging_low["threshold_data_sufficient"]) is False


def test_evaluate_policy_variants_reports_global_regime_and_abstain_versions() -> None:
    frame = pd.DataFrame(
        {
            "source_row_idx": [0, 1, 2, 3, 4, 5],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:05:00",
                    "2024-01-01 00:10:00",
                    "2024-01-01 00:15:00",
                    "2024-01-01 00:20:00",
                    "2024-01-01 00:25:00",
                ]
            ),
            "year": [2024] * 6,
            "target": [0, 1, 1, 0, 1, 0],
            "oof_calibrated_probability": [0.1, 0.9, 0.8, 0.2, 0.75, 0.1],
            "calibrated_probability": [0.1, 0.9, 0.8, 0.2, 0.75, 0.1],
            "composite_regime": ["strong_up_high"] * 6,
            "session_regime": ["london", "london", "off_hours", "london", "london", "london"],
            "stress_regime": ["normal", "normal", "normal", "high", "normal", "normal"],
            "policy_gross_pnl_pips": [-4.0, 20.0, 16.0, -3.0, 12.0, -2.0],
            "policy_net_pnl_pips": [-6.0, 15.0, 10.0, -5.0, 8.0, -4.0],
        }
    )
    policy_table = pd.DataFrame(
        {
            "composite_regime": ["strong_up_high"],
            "threshold": [0.4],
            "threshold_source": ["regime"],
        }
    )
    config = ThresholdSearchConfig(
        probability_column="oof_calibrated_probability",
        global_threshold=0.6,
        event_tolerance_bars=0,
        event_cooldown_bars=1,
        min_positive_events=1,
        min_events_per_month=0.0,
    )

    evaluation = evaluate_policy_variants(
        oof_frame=frame,
        test_frame=frame.rename(columns={"oof_calibrated_probability": "ignore"}),
        policy_table=policy_table,
        config=config,
        abstain_config=HardAbstainConfig(
            cooldown_bars=1,
            probability_column="policy_probability",
            signal_candidate_column="policy_signal_candidate",
        ),
    )

    assert set(evaluation["policy_name"]) == {
        "global_threshold",
        "global_threshold_plus_abstain",
        "regime_threshold",
        "regime_threshold_plus_abstain",
    }
    assert "post_cost_expectancy_pips" in evaluation.columns
    assert "net_pnl_pips" in evaluation.columns
    assert "trades_per_week" in evaluation.columns
    abstain_row = evaluation.loc[
        (evaluation["dataset_split"] == "oof")
        & (evaluation["policy_name"] == "regime_threshold_plus_abstain")
    ].iloc[0]
    assert int(abstain_row["abstain_count"]) >= 1


def test_attach_forward_trade_outcomes_adds_gross_and_net_pnl_columns() -> None:
    frame = pd.DataFrame(
        {
            "source_row_idx": [10, 11],
            "close": [1.1000, 1.1010],
            "session_regime": ["london", "asia"],
        }
    )
    market_frame = pd.DataFrame(
        {
            "source_row_idx": [10, 11, 12, 13],
            "close": [1.1000, 1.1010, 1.1020, 1.1005],
            "approx_spread": [0.0002, 0.0003, 0.0002, 0.0004],
        }
    )
    config = ThresholdSearchConfig(
        probability_column="calibrated_probability",
        global_threshold=0.6,
        label_max_holding_bars=2,
    )

    enriched = attach_forward_trade_outcomes(
        frame,
        market_frame=market_frame,
        direction="long",
        config=config,
    )

    assert float(enriched.loc[0, "policy_exit_close"]) == 1.1020
    assert float(enriched.loc[0, "policy_gross_pnl_pips"]) == pytest.approx(20.0)
    assert float(enriched.loc[0, "policy_total_cost_pips"]) > 0.0
    assert float(enriched.loc[0, "policy_net_pnl_pips"]) < 20.0
