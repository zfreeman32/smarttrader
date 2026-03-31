from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_testing.ote_abstain_policy import HardAbstainConfig
from model_testing.ote_policy_backtest import (
    WalkForwardBacktestConfig,
    build_walk_forward_folds,
    run_walk_forward_backtest,
)
from model_testing.ote_threshold_policy import ThresholdSearchConfig


def test_build_walk_forward_folds_uses_quarterly_schedule_after_two_year_minimum() -> None:
    frame, _ = _build_synthetic_walk_forward_data()

    folds = build_walk_forward_folds(
        frame,
        config=WalkForwardBacktestConfig(
            min_train_years=2,
            test_window_months=3,
            rolling_step_months=3,
            purge_gap_bars=1,
            min_folds=1,
        ),
    )

    assert len(folds) == 8
    assert folds[0].scheduled_test_start == pd.Timestamp("2022-04-01 00:00:00")
    assert folds[-1].scheduled_test_start == pd.Timestamp("2024-01-01 00:00:00")


def test_run_walk_forward_backtest_returns_fold_summary_and_selected_trades() -> None:
    frame, market_frame = _build_synthetic_walk_forward_data()
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.6,
        threshold_grid=(0.4, 0.6, 0.8),
        event_tolerance_bars=0,
        event_cooldown_bars=0,
        min_positive_events=1,
        min_events_per_month=0.0,
        label_max_holding_bars=1,
        slippage_spread_multiplier=0.0,
        fixed_slippage_pips_per_trade=0.3,
        commission_pips_per_trade=0.35,
    )
    backtest_config = WalkForwardBacktestConfig(
        min_train_years=2,
        test_window_months=3,
        rolling_step_months=3,
        purge_gap_bars=1,
        min_folds=2,
        min_trades_per_week=0.0,
    )

    results = run_walk_forward_backtest(
        frame,
        market_frame=market_frame,
        direction="long",
        threshold_config=threshold_config,
        backtest_config=backtest_config,
        abstain_config=HardAbstainConfig(
            cooldown_bars=0,
            probability_column="policy_probability",
            signal_candidate_column="policy_signal_candidate",
            position_column="source_row_idx",
        ),
        model_id="synthetic_long",
        backend="tcn",
    )

    fold_summary = results["fold_summary"]
    selected_test_trades = results["selected_test_trades"]
    summary = results["summary"]

    assert not fold_summary.empty
    assert set(fold_summary["selected_policy_name"]) == {"global_threshold"}
    assert not selected_test_trades.empty
    assert summary["fold_count"] == len(fold_summary)
    assert summary["overall_test_metrics"]["trade_count"] == len(selected_test_trades)


def _build_synthetic_walk_forward_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01 00:00:00", periods=49, freq="MS")
    positions = np.arange(len(dates), dtype=np.int64)
    targets = np.asarray([(index % 3) == 0 for index in range(len(dates))], dtype=np.uint8)

    close_values = [1.1000]
    for index in range(len(dates) - 1):
        step = 0.0050 if bool(targets[index]) else -0.0010
        close_values.append(close_values[-1] + step)

    frame = pd.DataFrame(
        {
            "source_row_idx": positions,
            "datetime": dates,
            "close": close_values,
            "target": targets,
            "year": dates.year,
            "model_probability": np.where(targets == 1, 0.9, 0.2),
            "composite_regime": np.where(targets == 1, "strong_up_medium", "ranging_low"),
            "session_regime": ["london"] * len(dates),
            "stress_regime": ["normal"] * len(dates),
        }
    )
    market_frame = pd.DataFrame(
        {
            "source_row_idx": positions,
            "datetime": dates,
            "close": close_values,
            "approx_spread": [0.0001] * len(dates),
        }
    )
    return frame, market_frame
