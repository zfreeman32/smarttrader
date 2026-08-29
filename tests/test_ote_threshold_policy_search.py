from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_testing.ote_abstain_policy import HardAbstainConfig
from scripts.run_ote_threshold_policy_search import (
    _build_registry_writeback_candidate,
    _describe_selected_policy_contract,
    _select_qualified_policy,
    _write_registry_policy_updates,
    build_arg_parser,
)


def test_write_registry_policy_updates_accepts_utf8_bom(tmp_path: Path) -> None:
    payload = {
        "promotion_rules": {
            "min_cv_splits": 3,
            "min_test_event_f05": 0.65,
            "require_regime_robustness": True,
            "require_post_cost_profitability": True,
            "require_paper_trading_confirmation": True,
        },
        "models": [
            {
                "model_id": "demo_model",
                "direction": "long",
                "role": "candidate",
                "backend": "tcn",
                "artifact_path": "models/ote_full_tcn_v2/long_ote",
                "cv_mean_ap": 0.6,
                "cv_mean_event_f05": 0.7,
                "test_ap": 0.7,
                "test_event_f05": 0.8,
                "global_threshold": 0.5,
                "regime_thresholds": None,
                "abstain_policy": None,
                "calibration_method": "platt",
                "promotion_date": "2026-05-13",
                "promotion_reason": "test",
                "status": "candidate",
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8-sig")

    _write_registry_policy_updates(
        registry_path=registry_path,
        updates={
            "demo_model": {
                "global_threshold": 0.81,
                "regime_thresholds": {"strong_up_high": 0.77},
                "abstain_policy": {"policy_name": "global_threshold_plus_abstain"},
            }
        },
    )

    updated = json.loads(registry_path.read_text(encoding="utf-8"))
    record = updated["models"][0]

    assert record["global_threshold"] == 0.81
    assert record["regime_thresholds"] == {"strong_up_high": 0.77}
    assert record["abstain_policy"] == {"policy_name": "global_threshold_plus_abstain"}


def test_build_arg_parser_accepts_targeted_filter_preset() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--targeted-filter-preset",
            "short_reversal_xgb_ranging_medium_london_prune_v1",
        ]
    )

    assert args.targeted_filter_preset == "short_reversal_xgb_ranging_medium_london_prune_v1"


def test_build_arg_parser_accepts_short_reversal_hard_prune_preset() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--targeted-filter-preset",
            "short_reversal_xgb_ranging_medium_london_hard_prune_v1",
        ]
    )

    assert args.targeted_filter_preset == "short_reversal_xgb_ranging_medium_london_hard_prune_v1"


def test_build_arg_parser_accepts_frvp_long_continuation_drawdown_prune_preset() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--targeted-filter-preset",
            "frvp_long_continuation_xgb_london_drawdown_prune_v1",
        ]
    )

    assert args.targeted_filter_preset == "frvp_long_continuation_xgb_london_drawdown_prune_v1"


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


def test_build_arg_parser_accepts_evaluation_contract_metadata() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--evaluation-contract-mode",
            "research",
            "--promotion-min-trades-per-week-floor",
            "3.0",
        ]
    )

    assert args.evaluation_contract_mode == "research"
    assert args.promotion_min_trades_per_week_floor == 3.0


def test_select_qualified_policy_falls_back_to_global_threshold_metrics_when_no_variant_qualifies() -> None:
    evaluation = pd.DataFrame(
        [
            {
                "dataset_split": "test",
                "policy_name": "global_threshold",
                "event_f05": 0.14,
                "post_cost_expectancy_pips": 6.62,
                "net_pnl_pips": 191.9,
                "trades_per_week": 0.35,
            },
            {
                "dataset_split": "test",
                "policy_name": "regime_threshold",
                "event_f05": 0.34,
                "post_cost_expectancy_pips": -1.63,
                "net_pnl_pips": -289.45,
                "trades_per_week": 2.17,
            },
        ]
    )

    selected = _select_qualified_policy(evaluation, min_trades_per_week=3.0)

    assert selected["qualified_policy_names"] == []
    assert selected["selected_policy_name"] == "global_threshold"
    assert (
        selected["selection_reason"]
        == "no_non_global_policy_met_test_expectancy_frequency_and_robustness_requirements"
    )
    assert selected["selected_policy_metrics"]["post_cost_expectancy_pips"] == 6.62
    assert selected["selected_policy_metrics"]["net_pnl_pips"] == 191.9


def test_select_qualified_policy_prefers_expectancy_with_split_robustness() -> None:
    evaluation = pd.DataFrame(
        [
            {
                "dataset_split": "oof",
                "policy_name": "global_threshold",
                "event_f05": 0.42,
                "post_cost_expectancy_pips": 0.90,
                "net_pnl_pips": 45.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "oof",
                "policy_name": "global_threshold_plus_abstain",
                "event_f05": 0.43,
                "post_cost_expectancy_pips": 1.00,
                "net_pnl_pips": 50.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "oof",
                "policy_name": "regime_threshold_plus_abstain",
                "event_f05": 0.40,
                "post_cost_expectancy_pips": 1.20,
                "net_pnl_pips": 60.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "test",
                "policy_name": "global_threshold",
                "event_f05": 0.44,
                "post_cost_expectancy_pips": 1.00,
                "net_pnl_pips": 40.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "test",
                "policy_name": "global_threshold_plus_abstain",
                "event_f05": 0.47,
                "post_cost_expectancy_pips": 1.20,
                "net_pnl_pips": 48.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "test",
                "policy_name": "regime_threshold_plus_abstain",
                "event_f05": 0.42,
                "post_cost_expectancy_pips": 1.45,
                "net_pnl_pips": 58.0,
                "trades_per_week": 4.0,
            },
        ]
    )

    selected = _select_qualified_policy(evaluation, min_trades_per_week=3.0)

    assert selected["qualified_policy_names"] == [
        "regime_threshold_plus_abstain",
        "global_threshold_plus_abstain",
    ]
    assert selected["selected_policy_name"] == "regime_threshold_plus_abstain"
    assert selected["selection_reason"] == "best_test_policy_by_post_cost_expectancy_and_split_robustness"
    assert selected["selected_policy_metrics"]["post_cost_expectancy_pips"] == 1.45


def test_select_qualified_policy_retains_hard_pruned_base_when_threshold_lift_is_not_robust() -> None:
    evaluation = pd.DataFrame(
        [
            {
                "dataset_split": "oof",
                "policy_name": "global_threshold",
                "event_f05": 0.50,
                "post_cost_expectancy_pips": 1.10,
                "net_pnl_pips": 55.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "oof",
                "policy_name": "regime_threshold",
                "event_f05": 0.49,
                "post_cost_expectancy_pips": 0.85,
                "net_pnl_pips": 43.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "test",
                "policy_name": "global_threshold",
                "event_f05": 0.51,
                "post_cost_expectancy_pips": 1.00,
                "net_pnl_pips": 50.0,
                "trades_per_week": 4.0,
            },
            {
                "dataset_split": "test",
                "policy_name": "regime_threshold",
                "event_f05": 0.50,
                "post_cost_expectancy_pips": 1.20,
                "net_pnl_pips": 60.0,
                "trades_per_week": 4.0,
            },
        ]
    )

    selected = _select_qualified_policy(
        evaluation,
        min_trades_per_week=3.0,
        apply_to_base_policy_variants=True,
    )

    assert selected["qualified_policy_names"] == []
    assert selected["selected_policy_name"] == "global_threshold"
    assert (
        selected["selection_reason"]
        == "hard_pruned_base_policy_retained_no_non_global_variant_met_test_expectancy_frequency_and_robustness_requirements"
    )


def test_build_registry_writeback_candidate_drops_regime_thresholds_when_global_policy_is_selected() -> None:
    candidate = _build_registry_writeback_candidate(
        policy_selection={
            "selected_policy_name": "global_threshold",
            "selection_reason": "no_non_global_policy_met_test_expectancy_frequency_and_robustness_requirements",
        },
        global_threshold=0.81,
        regime_threshold_map={"strong_up_high": 0.77},
        abstain_policy_metadata={"policy_name": "global_threshold"},
    )

    assert candidate["global_threshold"] == 0.81
    assert candidate["regime_thresholds"] is None
    assert candidate["abstain_policy"] == {"policy_name": "global_threshold"}


def test_describe_selected_policy_contract_marks_hard_pruned_base_policy() -> None:
    contract = _describe_selected_policy_contract(
        policy_selection={"selected_policy_name": "global_threshold"},
        abstain_config=HardAbstainConfig(
            apply_to_base_policy_variants=True,
            abstain_composite_session_pairs=(("ranging_medium", "london"),),
        ),
    )

    assert contract["policy_name"] == "global_threshold"
    assert contract["threshold_mode"] == "global_threshold"
    assert contract["base_policy_is_hard_pruned"] is True
    assert contract["uses_targeted_filters"] is True


def test_build_arg_parser_accepts_instrument_and_unit_cost_overrides() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--instrument",
            "es",
            "--fixed-slippage-units-per-trade",
            "0.25",
            "--commission-units-per-trade",
            "0.40",
        ]
    )

    assert args.instrument == "es"
    assert args.fixed_slippage_units_per_trade == 0.25
    assert args.commission_units_per_trade == 0.40


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
