from __future__ import annotations

import json

import pytest

from ote_live.features.manifest import LivePolicy
from scripts.build_frvp_paper_signal_bundle import (
    ACTIVE_MODEL_IDS,
    ACTIVE_THRESHOLDS,
    CONTINUATION_MODEL_ID,
    CONTROLLED_POLICY_MODEL_IDS,
    EXPECTED_MODEL_STATUSES,
    OUTPUT_BUNDLE_ID,
    OUTPUT_REGISTRY_PATH,
    REVERSAL_MODEL_ID,
    SOURCE_BACKTEST_SUMMARY_PATH,
    SOURCE_POLICY_DIR,
    SOURCE_REGISTRY_PATH,
    _accepted_targeted_filters,
    _calculate_overlap_evidence,
    _read_json,
    _write_immutable_text,
    build_paper_decision_summary,
    build_paper_live_policy,
    build_paper_policy_selection,
    build_paper_registry_payload,
    build_paper_run_summary,
)


def test_registry_activates_only_reversal_and_holds_continuation_candidate() -> None:
    source_payload = _read_json(SOURCE_REGISTRY_PATH)
    paper_payload = build_paper_registry_payload(source_payload)
    records = {record["model_id"]: record for record in paper_payload["models"]}

    assert {model_id: record["status"] for model_id, record in records.items()} == (
        EXPECTED_MODEL_STATUSES
    )
    assert ACTIVE_MODEL_IDS == (REVERSAL_MODEL_ID,)
    assert {
        model_id for model_id, record in records.items() if record["status"] == "active"
    } == {REVERSAL_MODEL_ID}
    assert records[CONTINUATION_MODEL_ID]["status"] == "candidate"
    assert records["frvp_long_meta_xgb_v1"]["status"] == "candidate"
    assert records["frvp_short_continuation_tcn_v1"]["status"] == "candidate"
    assert records["frvp_short_meta_xgb_v1"]["status"] == "candidate"
    assert records["frvp_short_reversal_xgb_v1"]["status"] == "deprecated"

    for model_id in CONTROLLED_POLICY_MODEL_IDS:
        assert records[model_id]["global_threshold"] == pytest.approx(
            ACTIVE_THRESHOLDS[model_id]
        )
        assert records[model_id]["regime_thresholds"] is None
        assert records[model_id]["promotion_date"] == "2026-08-16"
        assert records[model_id]["abstain_policy"]["apply_to_base_policy_variants"] is True


def test_active_policies_use_exact_accepted_backtest_filters() -> None:
    for model_id in CONTROLLED_POLICY_MODEL_IDS:
        source_policy = LivePolicy.model_validate_json(
            (SOURCE_POLICY_DIR / model_id / "live_policy.json").read_text(encoding="utf-8")
        )
        paper_policy = build_paper_live_policy(source_policy)
        accepted = _accepted_targeted_filters(model_id)

        assert paper_policy.thresholds.global_threshold == pytest.approx(
            ACTIVE_THRESHOLDS[model_id]
        )
        assert paper_policy.thresholds.regime_thresholds is None
        assert paper_policy.abstain_policy.enabled is True
        assert paper_policy.abstain_policy.model_dump(
            mode="json", exclude={"enabled"}
        ) == accepted
        assert paper_policy.lineage.threshold_registry_path == OUTPUT_REGISTRY_PATH.relative_to(
            OUTPUT_REGISTRY_PATH.parents[1]
        ).as_posix()
        expected_source_type = (
            "frvp_paper_signal_accepted_backtest_contract"
            if model_id == REVERSAL_MODEL_ID
            else "frvp_paper_signal_fixed_policy_shadow_contract"
        )
        assert paper_policy.lineage.policy_source_type == expected_source_type
        assert paper_policy.lineage.selected_policy_name == "global_threshold"


def test_reversal_policy_restores_strong_down_high_overlap_from_accepted_backtest() -> None:
    source_policy = LivePolicy.model_validate_json(
        (SOURCE_POLICY_DIR / REVERSAL_MODEL_ID / "live_policy.json").read_text(
            encoding="utf-8"
        )
    )
    source_pairs = source_policy.abstain_policy.abstain_composite_session_pairs
    paper_policy = build_paper_live_policy(source_policy)
    paper_pairs = paper_policy.abstain_policy.abstain_composite_session_pairs

    assert ("strong_down_high", "overlap") not in source_pairs
    assert ("strong_down_high", "overlap") in paper_pairs
    assert len(source_pairs) == 10
    assert len(paper_pairs) == 11
    assert any("repairs the stale" in note for note in paper_policy.lineage.notes)


def test_policy_selection_keeps_runtime_and_canonical_continuation_evidence_separate() -> None:
    source_selection = _read_json(
        SOURCE_POLICY_DIR / CONTINUATION_MODEL_ID / "policy_selection.json"
    )
    source_policy = LivePolicy.model_validate_json(
        (SOURCE_POLICY_DIR / CONTINUATION_MODEL_ID / "live_policy.json").read_text(
            encoding="utf-8"
        )
    )
    payload = build_paper_policy_selection(
        source_selection,
        paper_policy=build_paper_live_policy(source_policy),
    )

    assert payload["deployment_decision"]["status"] == "candidate"
    assert payload["deployment_decision"]["broker_order_submission_authorized"] is False
    assert payload["accepted_backtest_contract"]["walk_forward_snapshot"][
        "selected_test_trades"
    ] == 512
    assert payload["canonical_promotion_snapshot"]["selected_test_trades"] == 524
    assert "512-trade" in payload["runtime_evidence_precedence"]
    assert "524-trade" in payload["runtime_evidence_precedence"]
    assert payload["fixed_live_policy_sensitivity"]["trade_count"] == 180
    assert payload["fixed_live_policy_sensitivity"]["monthly_sharpe"] == pytest.approx(
        0.7384
    )
    assert payload["fixed_live_policy_sensitivity"]["failed_promotion_gates"] == [
        "annualized_sharpe_above_threshold",
        "profitable_quarter_share_above_threshold",
        "largest_single_trade_share_below_limit",
    ]


def test_decision_records_zero_overlap_120_bar_markouts_and_scoped_signoff_waiver() -> None:
    registry = build_paper_registry_payload(_read_json(SOURCE_REGISTRY_PATH))
    run_summary = build_paper_run_summary(_read_json(SOURCE_BACKTEST_SUMMARY_PATH))
    decision = build_paper_decision_summary(registry)

    assert run_summary["paper_signal_bundle"]["bundle_id"] == OUTPUT_BUNDLE_ID
    assert run_summary["paper_signal_bundle"]["active_model_ids"] == list(ACTIVE_MODEL_IDS)
    run_statuses = {
        output["model_id"]: output["paper_signal_status"]
        for output in run_summary["model_outputs"]
    }
    assert run_statuses == EXPECTED_MODEL_STATUSES

    assert decision["active_model_ids"] == list(ACTIVE_MODEL_IDS)
    assert decision["active_model_ids"] == [REVERSAL_MODEL_ID]
    assert decision["recommended_primary_model_id"] == REVERSAL_MODEL_ID
    assert decision["broker_order_submission_authorized"] is False
    assert decision["trial_lifecycle"]["state"] == (
        "authorized_pending_readiness_not_started"
    )
    assert decision["trial_lifecycle"]["decision_authorized"] is True
    assert decision["trial_lifecycle"]["start_authorized_after_readiness"] is True
    assert decision["trial_lifecycle"]["start_authorized_now"] is False
    assert decision["trial_lifecycle"]["minimum_calendar_days"] == 28
    assert decision["portfolio_contract"]["continuation_shadow_markouts_allowed"] is False
    assert decision["portfolio_contract"]["independent_active_signal_markouts"] is True
    assert decision["trial_lifecycle"]["start_requires_readiness_pass"] is True
    assert decision["trial_lifecycle"]["confirmation_start_utc"] is None
    assert decision["confirmation_markout_contract"]["horizon_completed_bars"] == 120
    assert decision["confirmation_markout_contract"]["entry_price_source"] == (
        "signal_bar_close"
    )
    assert decision["confirmation_markout_contract"]["overlapping_events_allowed"] is True
    assert decision["confirmation_markout_contract"]["broker_orders_or_positions"] is False
    assert decision["overlap_evidence"]["shared_trade_count"] == 0
    assert decision["overlap_evidence"]["continuation_unique_trade_count"] == 512
    assert decision["overlap_evidence"]["reversal_unique_trade_count"] == 63
    sensitivity = decision["fixed_live_policy_sensitivity_audit"]
    assert sensitivity["classification"] == (
        "fixed_live_policy_sensitivity_not_promotion_quality_wfo"
    )
    assert sensitivity["materialized_source_report_path"] is None
    assert sensitivity["method"]["prediction_source"] == (
        "frozen_labeled_oof_and_test_predictions"
    )
    assert sensitivity["method"]["horizon_completed_bars"] == 120
    assert sensitivity["model_results"][REVERSAL_MODEL_ID]["trade_count"] == 50
    assert sensitivity["model_results"][REVERSAL_MODEL_ID]["net_pnl_ticks"] == pytest.approx(
        2567.5
    )
    assert sensitivity["model_results"][REVERSAL_MODEL_ID][
        "failed_promotion_gates"
    ] == ["largest_single_trade_share_below_limit"]
    assert decision["trial_objective"]["primary_model_id"] == REVERSAL_MODEL_ID
    assert "concentration" in decision["trial_objective"]["objective"]
    assert decision["human_same_contract_signoff"] == {
        "status": "deferred_for_controlled_paper_signal",
        "required_for_controlled_paper_signal": False,
        "waiver_scope": "controlled_paper_signal_only",
        "required_for_full_promotion": True,
        "full_promotion_authorized": False,
        "note": (
            "The deferred TradingView same-contract human signoff is waived only for "
            "this no-order controlled paper-signal trial; it is not waived for full promotion."
        ),
    }


def test_exact_entry_overlap_is_computed_from_frozen_trade_ledgers() -> None:
    overlap = _calculate_overlap_evidence()

    assert overlap["match_key"] == ["entry_datetime", "source_row_idx"]
    assert overlap["shared_trade_count"] == 0
    assert overlap["shared_continuation_net_pnl_ticks"] == 0.0
    assert overlap["shared_reversal_net_pnl_ticks"] == 0.0
    assert overlap["continuation_unique_net_pnl_ticks"] == pytest.approx(7001.20)
    assert overlap["reversal_unique_net_pnl_ticks"] == pytest.approx(3677.05)


def test_immutable_writer_allows_identical_rebuild_and_rejects_contract_mutation(
    tmp_path,
) -> None:
    path = tmp_path / "immutable.json"

    assert _write_immutable_text(path, json.dumps({"status": "active"})) is True
    assert _write_immutable_text(path, json.dumps({"status": "active"})) is False
    with pytest.raises(FileExistsError, match="new dated bundle id"):
        _write_immutable_text(path, json.dumps({"status": "candidate"}))
