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
    select_walk_forward_policy,
)
from model_testing.promotion_gates import accepted_for_paper_trading, drawdown_acceptance_passed
from scripts.evaluation_contracts import build_evaluation_contract, resolve_saved_paper_trading_gate
from model_testing.ote_threshold_policy import ThresholdSearchConfig
from models.ote_registry_loader import OTEModelRecord
from scripts.run_ote_policy_backtest import build_arg_parser
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
    assert "test_total_net_pnl_units" in fold_summary.columns
    assert not selected_test_trades.empty
    assert "net_pnl_units" in selected_test_trades.columns
    assert summary["fold_count"] == len(fold_summary)
    assert summary["overall_test_metrics"]["trade_count"] == len(selected_test_trades)
    assert summary["performance_unit_label"] == "pips"
    assert "total_net_pnl_units" in summary["overall_test_metrics"]
    assert "annualized_sharpe_above_threshold" in summary["acceptance"]
    assert "dsr_above_threshold" in summary["acceptance"]
    assert "max_drawdown_pct_below_threshold" in summary["acceptance"]
    assert summary["overall_test_metrics"]["max_drawdown_pct"] is not None
    assert summary["overall_test_metrics"]["max_account_drawdown_pct"] is not None
    assert summary["overall_test_metrics"]["max_profit_retracement_pct"] is not None
    assert summary["paper_trading_gate"]["accepted"] == accepted_for_paper_trading(summary["acceptance"])
    assert summary["paper_trading_gate"]["drawdown_gate_passed"] == drawdown_acceptance_passed(summary["acceptance"])
    assert summary["paper_trading_gate"]["drawdown_gate_is_advisory"] is True
    assert summary["risk_gate_config"]["maximum_drawdown_pct"] == 12.0
    assert summary["risk_gate_config"]["drawdown_starting_balance_pips"] == 10000.0
    assert summary["overall_test_metrics"]["approx_deflated_sharpe"] is not None


def test_build_abstain_config_supports_pairwise_targeted_presets() -> None:
    model = OTEModelRecord(
        model_id="long_breakout_tcn_champion",
        direction="long",
        role="candidate",
        backend="tcn",
        artifact_path="models/live/long_breakout_tcn",
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


def test_build_abstain_config_supports_frvp_long_continuation_london_drawdown_prune() -> None:
    model = OTEModelRecord(
        model_id="frvp_long_continuation_xgb_v1",
        direction="long",
        role="candidate",
        backend="xgboost",
        artifact_path="models/frvp_es_primary_xgb_refresh_20260701/long_frvp_continuation",
        cv_mean_ap=0.0,
        cv_mean_event_f05=0.0,
        test_ap=0.0,
        test_event_f05=0.0,
        global_threshold=0.7,
        regime_thresholds=None,
        abstain_policy=None,
        calibration_method="none",
        promotion_date="2026-07-02",
        promotion_reason="test",
        status="candidate",
    )
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.7,
    )

    config = _build_abstain_config(
        model,
        threshold_config,
        targeted_filter_preset="frvp_long_continuation_xgb_london_drawdown_prune_v1",
    )

    assert config.apply_to_base_policy_variants is True
    assert config.abstain_composite_session_pairs == (
        ("strong_up_medium", "london"),
        ("strong_up_high", "london"),
        ("strong_down_medium", "london"),
        ("ranging_low", "london"),
        ("ranging_medium", "london"),
    )


def test_build_abstain_config_supports_frvp_long_continuation_overlap_composite_prune() -> None:
    model = OTEModelRecord(
        model_id="frvp_long_continuation_xgb_v1",
        direction="long",
        role="candidate",
        backend="xgboost",
        artifact_path="models/frvp_es_primary_xgb_refresh_20260701/long_frvp_continuation",
        cv_mean_ap=0.0,
        cv_mean_event_f05=0.0,
        test_ap=0.0,
        test_event_f05=0.0,
        global_threshold=0.7,
        regime_thresholds=None,
        abstain_policy=None,
        calibration_method="none",
        promotion_date="2026-07-03",
        promotion_reason="test",
        status="candidate",
    )
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.7,
    )

    config = _build_abstain_config(
        model,
        threshold_config,
        targeted_filter_preset="frvp_long_continuation_xgb_overlap_composite_prune_v2",
    )

    assert config.apply_to_base_policy_variants is True
    assert config.abstain_session_regimes == ("overlap",)
    assert config.abstain_composite_regimes == (
        "strong_down_medium",
        "strong_up_high",
    )
    assert config.abstain_composite_session_pairs == (
        ("strong_up_medium", "london"),
        ("strong_up_high", "london"),
        ("strong_down_medium", "london"),
        ("ranging_low", "london"),
        ("ranging_medium", "london"),
    )


def test_build_abstain_config_supports_frvp_long_continuation_overlap_composite_prune_v3() -> None:
    model = OTEModelRecord(
        model_id="frvp_long_continuation_xgb_v1",
        direction="long",
        role="candidate",
        backend="xgboost",
        artifact_path="models/frvp_es_primary_xgb_refresh_20260701/long_frvp_continuation",
        cv_mean_ap=0.0,
        cv_mean_event_f05=0.0,
        test_ap=0.0,
        test_event_f05=0.0,
        global_threshold=0.7,
        regime_thresholds=None,
        abstain_policy=None,
        calibration_method="none",
        promotion_date="2026-07-03",
        promotion_reason="test",
        status="candidate",
    )
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.7,
    )

    config = _build_abstain_config(
        model,
        threshold_config,
        targeted_filter_preset="frvp_long_continuation_xgb_overlap_composite_prune_v3",
    )

    assert config.apply_to_base_policy_variants is True
    assert config.abstain_session_regimes == ("overlap",)
    assert config.abstain_composite_regimes == (
        "strong_down_medium",
        "strong_up_high",
    )
    assert config.abstain_composite_session_pairs == (
        ("strong_up_medium", "london"),
        ("strong_up_high", "london"),
        ("strong_down_medium", "london"),
        ("ranging_low", "london"),
        ("ranging_medium", "london"),
        ("ranging_medium", "new_york"),
        ("strong_up_low", "asia"),
    )


def test_build_abstain_config_supports_frvp_long_continuation_setup2_medium_session_prune() -> None:
    model = OTEModelRecord(
        model_id="frvp_long_continuation_setup2_xgb_v1",
        direction="long",
        role="candidate",
        backend="xgboost",
        artifact_path="models/frvp_long_continuation_setup2_xgb_20260829/long_frvp_continuation_setup2",
        cv_mean_ap=0.0,
        cv_mean_event_f05=0.0,
        test_ap=0.0,
        test_event_f05=0.0,
        global_threshold=0.4,
        regime_thresholds=None,
        abstain_policy=None,
        calibration_method="none",
        promotion_date="2026-08-29",
        promotion_reason="test",
        status="candidate",
    )
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.4,
    )

    config = _build_abstain_config(
        model,
        threshold_config,
        targeted_filter_preset="frvp_long_continuation_setup2_xgb_medium_session_prune_v1",
    )

    assert config.apply_to_base_policy_variants is True
    assert config.abstain_composite_session_pairs == (
        ("strong_down_medium", "london"),
        ("strong_down_medium", "overlap"),
        ("strong_up_medium", "london"),
        ("strong_up_medium", "overlap"),
    )


def test_build_abstain_config_supports_frvp_long_continuation_setup5_repeated_drawdown_prune() -> None:
    model = OTEModelRecord(
        model_id="frvp_long_continuation_setup5_xgb_v1",
        direction="long",
        role="candidate",
        backend="xgboost",
        artifact_path="models/frvp_long_continuation_setup5_xgb_20260829/long_frvp_continuation_setup5",
        cv_mean_ap=0.0,
        cv_mean_event_f05=0.0,
        test_ap=0.0,
        test_event_f05=0.0,
        global_threshold=0.4,
        regime_thresholds=None,
        abstain_policy=None,
        calibration_method="none",
        promotion_date="2026-08-29",
        promotion_reason="test",
        status="candidate",
    )
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.4,
    )

    config = _build_abstain_config(
        model,
        threshold_config,
        targeted_filter_preset="frvp_long_continuation_setup5_xgb_repeated_drawdown_prune_v1",
    )

    assert config.apply_to_base_policy_variants is True
    assert config.abstain_composite_session_pairs == (
        ("strong_down_high", "london"),
        ("strong_up_high", "london"),
        ("strong_up_medium", "asia"),
        ("ranging_high", "new_york"),
        ("ranging_medium", "asia"),
        ("strong_down_low", "london"),
        ("strong_down_medium", "overlap"),
        ("strong_up_low", "asia"),
        ("strong_down_high", "new_york"),
        ("strong_up_medium", "new_york"),
    )


def test_build_arg_parser_accepts_frvp_long_continuation_overlap_composite_prune_preset() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--targeted-filter-preset",
            "frvp_long_continuation_xgb_overlap_composite_prune_v2",
        ]
    )

    assert args.targeted_filter_preset == "frvp_long_continuation_xgb_overlap_composite_prune_v2"


def test_build_arg_parser_accepts_frvp_long_continuation_overlap_composite_prune_v3_preset() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--targeted-filter-preset",
            "frvp_long_continuation_xgb_overlap_composite_prune_v3",
        ]
    )

    assert args.targeted_filter_preset == "frvp_long_continuation_xgb_overlap_composite_prune_v3"


def test_build_arg_parser_accepts_minimum_sharpe() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--minimum-sharpe",
            "0.95",
        ]
    )

    assert args.minimum_sharpe == 0.95


def test_build_arg_parser_accepts_maximum_drawdown_pct() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--maximum-drawdown-pct",
            "9.5",
        ]
    )

    assert args.maximum_drawdown_pct == 9.5


def test_build_arg_parser_accepts_drawdown_starting_balance_units() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--drawdown-starting-balance-units",
            "25000",
        ]
    )

    assert args.drawdown_starting_balance_units == 25000.0


def test_build_arg_parser_accepts_spread_cost_mode() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--spread-cost-mode",
            "feature_proxy",
        ]
    )

    assert args.spread_cost_mode == "feature_proxy"


def test_build_arg_parser_accepts_evaluation_contract_audit_fields() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--evaluation-contract-mode",
            "research",
            "--promotion-min-trades-per-week-floor",
            "3.0",
            "--requested-min-folds",
            "9",
            "--available-min-folds",
            "7",
        ]
    )

    assert args.evaluation_contract_mode == "research"
    assert args.promotion_min_trades_per_week_floor == 3.0
    assert args.requested_min_folds == 9
    assert args.available_min_folds == 7


def test_select_walk_forward_policy_prefers_expectancy_with_event_f05_tolerance() -> None:
    evaluation = pd.DataFrame(
        [
            {
                "dataset_split": "train",
                "policy_name": "global_threshold",
                "event_f05": 0.52,
                "post_cost_expectancy_pips": 1.00,
                "net_pnl_pips": 50.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "train",
                "policy_name": "global_threshold_plus_abstain",
                "event_f05": 0.50,
                "post_cost_expectancy_pips": 1.35,
                "net_pnl_pips": 54.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "train",
                "policy_name": "regime_threshold_plus_abstain",
                "event_f05": 0.56,
                "post_cost_expectancy_pips": 1.25,
                "net_pnl_pips": 52.0,
                "trades_per_week": 4.0,
            },
        ]
    )

    selected = select_walk_forward_policy(
        evaluation,
        split_name="train",
        min_trades_per_week=3.0,
    )

    assert selected["qualified_policy_names"] == [
        "global_threshold_plus_abstain",
        "regime_threshold_plus_abstain",
    ]
    assert selected["selected_policy_name"] == "global_threshold_plus_abstain"
    assert selected["selection_reason"] == "best_train_policy_by_post_cost_expectancy_and_robustness"


def test_select_walk_forward_policy_retains_hard_pruned_base_when_threshold_mode_does_not_add_expectancy() -> None:
    evaluation = pd.DataFrame(
        [
            {
                "dataset_split": "train",
                "policy_name": "global_threshold",
                "event_f05": 0.48,
                "post_cost_expectancy_pips": 1.10,
                "net_pnl_pips": 55.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "train",
                "policy_name": "regime_threshold",
                "event_f05": 0.53,
                "post_cost_expectancy_pips": 1.10,
                "net_pnl_pips": 55.0,
                "trades_per_week": 4.0,
            },
        ]
    )

    selected = select_walk_forward_policy(
        evaluation,
        split_name="train",
        min_trades_per_week=3.0,
        apply_to_base_policy_variants=True,
    )

    assert selected["qualified_policy_names"] == []
    assert selected["selected_policy_name"] == "global_threshold"
    assert (
        selected["selection_reason"]
        == "hard_pruned_base_policy_retained_no_non_global_variant_met_train_expectancy_frequency_and_robustness_requirements"
    )


def test_accepted_for_paper_trading_treats_drawdown_gate_as_advisory() -> None:
    acceptance = {
        "policy_profitable_after_costs": True,
        "annualized_sharpe_above_threshold": True,
        "wfe_above_threshold": True,
        "dsr_above_threshold": True,
        "profitable_quarter_share_above_threshold": True,
        "positive_composite_expectancy_share_above_threshold": True,
        "largest_single_trade_share_below_limit": True,
        "max_drawdown_pct_below_threshold": False,
    }

    assert drawdown_acceptance_passed(acceptance) is False
    assert accepted_for_paper_trading(acceptance) is True


def test_build_evaluation_contract_marks_research_low_frequency_and_auto_relaxed_folds() -> None:
    contract = build_evaluation_contract(
        evaluation_contract_mode="research",
        min_trades_per_week=0.5,
        requested_min_folds=9,
        available_min_folds=7,
        effective_min_folds=7,
    )

    assert contract["promotion_quality_run"] is False
    assert contract["used_low_frequency_override"] is True
    assert contract["used_auto_relaxed_min_folds"] is True
    assert contract["promotion_quality_gate_eligible"] is False
    assert contract["promotion_quality_disqualifiers"] == [
        "research_mode",
        "min_trades_per_week_below_promotion_floor",
        "auto_relaxed_min_folds",
    ]


def test_resolve_saved_paper_trading_gate_marks_legacy_low_frequency_runs_as_not_promotion_quality() -> None:
    acceptance = {
        "policy_profitable_after_costs": True,
        "annualized_sharpe_above_threshold": True,
        "wfe_above_threshold": True,
        "dsr_above_threshold": True,
        "profitable_quarter_share_above_threshold": True,
        "positive_composite_expectancy_share_above_threshold": True,
        "largest_single_trade_share_below_limit": True,
        "max_drawdown_pct_below_threshold": False,
    }

    gate = resolve_saved_paper_trading_gate(
        {"acceptance": acceptance},
        run_summary={
            "min_trades_per_week": 0.5,
            "min_folds": 7,
        },
    )

    assert gate["accepted_raw"] is True
    assert gate["accepted"] is False
    assert gate["promotion_quality_gate_eligible"] is False
    assert gate["promotion_quality_disqualifiers"] == [
        "research_mode",
        "min_trades_per_week_below_promotion_floor",
    ]


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


def test_build_abstain_config_ignores_legacy_fx_spreads_when_running_es_economics() -> None:
    model = OTEModelRecord(
        model_id="frvp_short_reversal_xgb_v1",
        direction="short",
        role="candidate",
        backend="xgboost",
        artifact_path="models/frvp_es_primary_xgb_v1/short_frvp_reversal",
        cv_mean_ap=0.0,
        cv_mean_event_f05=0.0,
        test_ap=0.0,
        test_event_f05=0.0,
        global_threshold=0.62,
        regime_thresholds=None,
        abstain_policy={
            "policy_name": "global_threshold_plus_abstain",
            "session_spread_pips": {
                "overlap": 1.0,
                "london": 1.5,
                "new_york": 1.5,
                "asia": 2.5,
                "off_hours": 3.0,
            },
        },
        calibration_method="platt",
        promotion_date="2026-06-30",
        promotion_reason="test",
        status="candidate",
    )
    threshold_config = ThresholdSearchConfig(
        probability_column="model_probability",
        global_threshold=0.62,
        instrument="es",
        unit_label="ticks",
        session_spread_pips={
            "overlap": 1.0,
            "london": 1.0,
            "new_york": 1.0,
            "asia": 1.5,
            "off_hours": 2.0,
        },
    )

    config = _build_abstain_config(model, threshold_config)

    assert config.session_spread_pips == threshold_config.session_spread_pips


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
