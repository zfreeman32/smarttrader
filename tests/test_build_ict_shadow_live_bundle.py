from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scripts.build_ict_shadow_live_bundle import (
    BACKTEST_SUMMARY_PATH,
    OUTPUT_BUNDLE_ID,
    PREPARED_ROOT,
    SHADOW_BUNDLE_MODELS,
    SOURCE_REGISTRY_PATH,
    THRESHOLD_SUMMARY_PATH,
    _build_abstain_policy_payload,
    _build_selection_summary_payload,
)


def test_shadow_roster_covers_each_direction_and_trade_type_without_one_minute_inputs() -> None:
    roster = {
        (spec.direction, spec.trade_type)
        for spec in SHADOW_BUNDLE_MODELS
    }

    assert roster == {
        (direction, trade_type)
        for direction in ("long", "short")
        for trade_type in ("continuation", "reversal", "meta")
    }
    assert len(SHADOW_BUNDLE_MODELS) == len(roster) == 6

    source_paths = (
        SOURCE_REGISTRY_PATH,
        THRESHOLD_SUMMARY_PATH,
        BACKTEST_SUMMARY_PATH,
        PREPARED_ROOT,
    )
    assert all("1min" not in str(path).lower() for path in source_paths)
    assert all("1_min" not in str(path).lower() for path in source_paths)


def test_build_selection_summary_payload_exposes_trade_type_taxonomy() -> None:
    payload = _build_selection_summary_payload(
        generated_at_utc=datetime(2026, 7, 21, tzinfo=timezone.utc),
        registry_path=Path("models/ict_registry.json"),
        policy_dir=Path("ote_live/policy_artifacts/ict_bundle"),
        manifest_dir=Path("ote_live/runtime_manifests/ict_bundle"),
        run_summary_path=Path("model_testing/reports/ict_backtests/example/run_summary.json"),
        selection_rows=[
            {
                "trade_type": "meta",
                "label_family": "ict_meta",
                "setup_families": ["reversal", "continuation"],
                "setup_family_scope": "reversal,continuation",
                "family": "meta",
                "direction": "short",
                "model_id": "ict_short_meta_xgb_v1",
                "source_backtest_run": "ict_example",
                "selected_test_net_pnl_units": 12.5,
                "selected_test_sharpe": 1.1,
                "overall_wfe": 1.2,
                "dashboard_priority": 2,
                "selected_for_dashboard": True,
                "selection_reason": "Best short pooled ICT filter branch.",
            },
            {
                "trade_type": "reversal",
                "label_family": "ict_reversal",
                "setup_families": ["reversal"],
                "setup_family_scope": "reversal",
                "family": "reversal",
                "direction": "long",
                "model_id": "ict_long_reversal_xgb_v1",
                "source_backtest_run": "ict_example",
                "selected_test_net_pnl_units": 18.0,
                "selected_test_sharpe": 1.5,
                "overall_wfe": 1.8,
                "dashboard_priority": 1,
                "selected_for_dashboard": True,
                "selection_reason": "Preferred long ICT reversal sentinel.",
            },
        ],
    )

    assert payload["bundle_id"] == OUTPUT_BUNDLE_ID
    assert payload["taxonomy"]["trade_types"] == ["reversal", "continuation", "meta"]
    assert payload["trade_type_leaders"] == payload["family_leaders"]
    assert payload["trade_type_leaders"][0]["model_id"] == "ict_long_reversal_xgb_v1"

    recommended = payload["recommended_shadow_dashboard_models"]
    assert recommended[0]["trade_type"] == "reversal"
    assert recommended[0]["label_family"] == "ict_reversal"
    assert recommended[1]["trade_type"] == "meta"
    assert recommended[1]["setup_families"] == ["reversal", "continuation"]
    assert any("legacy alias" in note for note in payload["notes"])


def test_plain_global_policy_does_not_enable_abstention_filters() -> None:
    payload = _build_abstain_policy_payload(
        {
            "selected_policy_name": "global_threshold",
            "targeted_filters": {
                "abstain_high_stress": True,
                "abstain_off_hours": True,
                "cooldown_bars": 4,
                "minimum_expected_move_to_spread": 2.0,
                "apply_to_base_policy_variants": False,
            },
        }
    )

    assert payload is None


def test_explicit_abstain_policy_enables_packaged_filters() -> None:
    payload = _build_abstain_policy_payload(
        {
            "selected_policy_name": "regime_threshold_plus_abstain",
            "targeted_filters": {
                "abstain_high_stress": True,
                "abstain_off_hours": True,
                "cooldown_bars": 4,
                "minimum_expected_move_to_spread": 2.0,
                "apply_to_base_policy_variants": False,
            },
        }
    )

    assert payload is not None
    assert payload["enabled"] is True
    assert payload["abstain_high_stress"] is True


def test_base_policy_filter_override_enables_packaged_filters() -> None:
    payload = _build_abstain_policy_payload(
        {
            "selected_policy_name": "global_threshold",
            "targeted_filters": {
                "abstain_high_stress": True,
                "apply_to_base_policy_variants": True,
            },
        }
    )

    assert payload is not None
    assert payload["enabled"] is True
