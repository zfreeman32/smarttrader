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
    build_policy_variant_decisions,
    run_walk_forward_backtest,
)
from model_testing.ote_threshold_policy import ThresholdSearchConfig
from models.ote_registry_loader import OTEModelRecord
from scripts.run_ote_policy_backtest import _build_abstain_config


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


def test_build_walk_forward_folds_supports_timezone_aware_datetimes() -> None:
    frame, _ = _build_synthetic_walk_forward_data()
    frame = frame.copy()
    frame["datetime"] = frame["datetime"].dt.tz_localize("UTC")

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
    assert folds[0].scheduled_test_start == pd.Timestamp("2022-04-01 00:00:00+00:00")
    assert folds[-1].scheduled_test_start == pd.Timestamp("2024-01-01 00:00:00+00:00")


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


def test_build_abstain_config_supports_pairwise_targeted_presets() -> None:
    model = OTEModelRecord(
        model_id="long_breakout_tcn_champion",
        direction="long",
        role="candidate",
        backend="tcn",
        artifact_path="models/champion_models/breakout_models/long_breakout_tcn_repair_20260513_v2/long_breakout",
        cv_mean_ap=0.0,
        cv_mean_event_f05=0.0,
        test_ap=0.0,
        test_event_f05=0.0,
        global_threshold=0.92,
        regime_thresholds=None,
        abstain_policy=None,
        calibration_method="platt",
        promotion_date="2026-05-13",
        promotion_reason="test",
        status="candidate",
    )
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.92,
    )

    asia_config = _build_abstain_config(
        model,
        threshold_config,
        targeted_filter_preset="long_breakout_regime_prune_v1_q20_asia",
    )
    v3_config = _build_abstain_config(
        model,
        threshold_config,
        targeted_filter_preset="long_breakout_regime_prune_v3",
    )

    assert asia_config.abstain_session_regimes == ("asia",)
    assert asia_config.minimum_probability_quantile == 0.20
    assert v3_config.abstain_composite_session_pairs == (("strong_up_medium", "asia"),)
    assert v3_config.abstain_composite_stress_pairs == (("strong_up_medium", "elevated"),)
    assert v3_config.minimum_probability_quantile == 0.20


def test_build_abstain_config_supports_short_reversal_ranging_medium_london_prune() -> None:
    model = OTEModelRecord(
        model_id="short_reversal_xgb_v1",
        direction="short",
        role="candidate",
        backend="xgboost",
        artifact_path="models/champion_models/reversal_models/short_reversal_xgb_v1",
        cv_mean_ap=0.0,
        cv_mean_event_f05=0.0,
        test_ap=0.0,
        test_event_f05=0.0,
        global_threshold=0.9,
        regime_thresholds=None,
        abstain_policy=None,
        calibration_method="platt",
        promotion_date="2026-05-25",
        promotion_reason="test",
        status="candidate",
    )
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.9,
    )

    config = _build_abstain_config(
        model,
        threshold_config,
        targeted_filter_preset="short_reversal_xgb_ranging_medium_london_prune_v1",
    )

    assert config.abstain_composite_session_pairs == (("ranging_medium", "london"),)


def test_build_abstain_config_supports_short_reversal_ranging_medium_london_hard_prune() -> None:
    model = OTEModelRecord(
        model_id="short_reversal_xgb_v1",
        direction="short",
        role="candidate",
        backend="xgboost",
        artifact_path="models/champion_models/reversal_models/short_reversal_xgb_v1",
        cv_mean_ap=0.0,
        cv_mean_event_f05=0.0,
        test_ap=0.0,
        test_event_f05=0.0,
        global_threshold=0.9,
        regime_thresholds=None,
        abstain_policy=None,
        calibration_method="platt",
        promotion_date="2026-05-25",
        promotion_reason="test",
        status="candidate",
    )
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.9,
    )

    config = _build_abstain_config(
        model,
        threshold_config,
        targeted_filter_preset="short_reversal_xgb_ranging_medium_london_hard_prune_v1",
    )

    assert config.abstain_composite_session_pairs == (("ranging_medium", "london"),)
    assert config.apply_to_base_policy_variants is True


def test_build_policy_variant_decisions_applies_hard_prune_to_regime_threshold() -> None:
    frame = pd.DataFrame(
        {
            "source_row_idx": [0, 1, 2],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:05:00",
                    "2024-01-01 00:10:00",
                ]
            ),
            "target": [1, 1, 1],
            "model_probability": [0.95, 0.94, 0.93],
            "composite_regime": ["ranging_medium", "strong_up_medium", "ranging_medium"],
            "session_regime": ["london", "london", "asia"],
            "stress_regime": ["normal", "normal", "normal"],
        }
    )
    policy_table = pd.DataFrame(
        {
            "composite_regime": ["ranging_medium", "strong_up_medium"],
            "threshold": [0.5, 0.5],
            "threshold_source": ["regime", "regime"],
        }
    )
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.9,
        event_cooldown_bars=0,
    )

    decisions = build_policy_variant_decisions(
        frame,
        policy_name="regime_threshold",
        policy_table=policy_table,
        threshold_config=threshold_config,
        abstain_config=HardAbstainConfig(
            abstain_high_stress=False,
            abstain_off_hours=False,
            cooldown_bars=0,
            abstain_composite_session_pairs=(("ranging_medium", "london"),),
            apply_to_base_policy_variants=True,
            probability_column="policy_probability",
            signal_candidate_column="policy_signal_candidate",
            position_column="source_row_idx",
        ),
    )

    emitted = decisions.loc[decisions["policy_emit_signal"].fillna(False)].copy()
    assert len(emitted) == 2
    assert set(emitted["source_row_idx"].tolist()) == {1, 2}
    blocked = decisions.loc[decisions["source_row_idx"] == 0].iloc[0]
    assert blocked["policy_abstain_reason"] == "composite_session_filter"


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
