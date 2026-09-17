from __future__ import annotations

import json

import pytest

from ote_live.features.manifest import LivePolicy
from scripts.build_ict_paper_signal_bundle import (
    EXPECTED_MODEL_IDS,
    META_MODEL_ID,
    META_PROMOTION_REASON,
    OUTPUT_BUNDLE_ID,
    OUTPUT_REGISTRY_PATH,
    PROMOTION_DATE,
    REVERSAL_HOLD_REASON,
    REVERSAL_MODEL_ID,
    SOURCE_BACKTEST_SUMMARY_PATH,
    SOURCE_POLICY_DIR,
    SOURCE_REGISTRY_PATH,
    SOURCE_WALK_FORWARD_SUMMARY_PATH,
    _get_model_output,
    _read_json,
    _write_immutable_text,
    build_paper_decision_summary,
    build_paper_live_policy,
    build_paper_policy_selection,
    build_paper_registry_payload,
    build_paper_run_summary,
)


def test_registry_promotes_only_long_meta_and_records_reversal_overlap_hold() -> None:
    source_payload = _read_json(SOURCE_REGISTRY_PATH)
    paper_payload = build_paper_registry_payload(source_payload)
    source_records = {record["model_id"]: record for record in source_payload["models"]}
    records = {record["model_id"]: record for record in paper_payload["models"]}

    assert set(records) == EXPECTED_MODEL_IDS
    assert {model_id for model_id, record in records.items() if record["status"] == "active"} == {
        META_MODEL_ID
    }
    assert records[META_MODEL_ID]["global_threshold"] == pytest.approx(0.40)
    assert records[META_MODEL_ID]["regime_thresholds"] is None
    assert records[META_MODEL_ID]["abstain_policy"] is None
    assert records[META_MODEL_ID]["promotion_date"] == PROMOTION_DATE
    assert records[META_MODEL_ID]["promotion_reason"] == META_PROMOTION_REASON

    reversal = records[REVERSAL_MODEL_ID]
    assert reversal["status"] == "candidate"
    assert reversal["global_threshold"] == pytest.approx(0.40)
    assert reversal["regime_thresholds"] is None
    assert reversal["abstain_policy"] is None
    assert reversal["promotion_date"] == PROMOTION_DATE
    assert reversal["promotion_reason"] == REVERSAL_HOLD_REASON
    assert "612 exact shared" in reversal["promotion_reason"]
    assert "105.2%" in reversal["promotion_reason"]
    assert "663 unique" in reversal["promotion_reason"]
    assert "286.95" in reversal["promotion_reason"]

    untouched_id = "ict_short_meta_xgb_v1"
    assert records[untouched_id]["status"] == "candidate"
    assert records[untouched_id]["promotion_reason"] == source_records[untouched_id]["promotion_reason"]
    assert records[untouched_id]["promotion_date"] == source_records[untouched_id]["promotion_date"]


def test_long_meta_policy_is_exact_global_point_four_without_abstention() -> None:
    source_policy = LivePolicy.model_validate_json(
        (SOURCE_POLICY_DIR / META_MODEL_ID / "live_policy.json").read_text(encoding="utf-8")
    )
    paper_policy = build_paper_live_policy(source_policy)

    assert paper_policy.policy_status == "complete"
    assert paper_policy.thresholds.global_threshold == pytest.approx(0.40)
    assert paper_policy.thresholds.regime_thresholds is None
    assert paper_policy.abstain_policy.enabled is False
    assert paper_policy.abstain_policy.abstain_high_stress is False
    assert paper_policy.abstain_policy.abstain_off_hours is False
    assert paper_policy.abstain_policy.cooldown_bars == 0
    assert paper_policy.lineage.threshold_registry_path == OUTPUT_REGISTRY_PATH.relative_to(
        OUTPUT_REGISTRY_PATH.parents[1]
    ).as_posix()
    assert paper_policy.lineage.policy_source_type == "ict_paper_signal_walk_forward_contract"
    assert paper_policy.lineage.selected_policy_name == "global_threshold"
    assert paper_policy.lineage.qualified_policy_names == ["global_threshold"]
    assert paper_policy.lineage.policy_table_path.endswith(
        f"/{META_MODEL_ID}/policy_table.csv"
    )
    assert all("keep this model shadow-only" not in note for note in paper_policy.lineage.notes)


def test_long_reversal_policy_remains_non_additive_shadow_contract() -> None:
    source_policy = LivePolicy.model_validate_json(
        (SOURCE_POLICY_DIR / REVERSAL_MODEL_ID / "live_policy.json").read_text(encoding="utf-8")
    )
    paper_policy = build_paper_live_policy(source_policy)

    assert paper_policy.thresholds.global_threshold == pytest.approx(0.40)
    assert paper_policy.thresholds.regime_thresholds is None
    assert paper_policy.abstain_policy.enabled is False
    assert paper_policy.lineage.selected_policy_name == "global_threshold"
    assert any("612 shared trades" in note for note in paper_policy.lineage.notes)
    assert any("Do not stack" in note for note in paper_policy.lineage.notes)


def test_policy_selection_replaces_meta_static_mismatch_with_walk_forward_contract() -> None:
    source_selection = _read_json(SOURCE_POLICY_DIR / META_MODEL_ID / "policy_selection.json")
    source_policy = LivePolicy.model_validate_json(
        (SOURCE_POLICY_DIR / META_MODEL_ID / "live_policy.json").read_text(encoding="utf-8")
    )
    walk_forward_summary = _read_json(SOURCE_WALK_FORWARD_SUMMARY_PATH)
    payload = build_paper_policy_selection(
        source_selection,
        walk_forward_model_output=_get_model_output(walk_forward_summary, META_MODEL_ID),
        paper_policy=build_paper_live_policy(source_policy),
    )

    assert payload["selected_policy_name"] == "global_threshold"
    assert payload["static_vs_walk_forward_dominant_mismatch"] is False
    assert payload["thresholds"] == {"global_threshold": 0.40, "regime_thresholds": None}
    assert payload["deployment_decision"]["status"] == "active"
    assert payload["deployment_decision"]["broker_order_submission_authorized"] is False
    assert payload["source_static_policy_selection"]["selected_policy_name"] == (
        "regime_threshold_plus_abstain"
    )
    assert payload["selected_policy_metrics"]["trade_count"] == 1555
    assert payload["selected_policy_metrics"]["monthly_sharpe"] == pytest.approx(
        1.5864849028923118
    )


def test_run_and_decision_summaries_record_one_active_model_and_no_stacking() -> None:
    source_registry = _read_json(SOURCE_REGISTRY_PATH)
    paper_registry = build_paper_registry_payload(source_registry)
    run_summary = build_paper_run_summary(_read_json(SOURCE_BACKTEST_SUMMARY_PATH))
    decision_summary = build_paper_decision_summary(paper_registry)

    assert run_summary["statuses"] == ["active", "candidate"]
    assert run_summary["paper_signal_bundle"]["bundle_id"] == OUTPUT_BUNDLE_ID
    assert run_summary["paper_signal_bundle"]["active_model_ids"] == [META_MODEL_ID]
    run_statuses = {
        output["model_id"]: output["paper_signal_status"]
        for output in run_summary["model_outputs"]
    }
    assert {model_id for model_id, status in run_statuses.items() if status == "active"} == {
        META_MODEL_ID
    }

    assert decision_summary["active_model_ids"] == [META_MODEL_ID]
    assert decision_summary["recommended_primary_model_id"] == META_MODEL_ID
    assert decision_summary["broker_order_submission_authorized"] is False
    assert decision_summary["trial_lifecycle"] == {
        "state": "authorized_pending_paper_feed_not_started",
        "start_authorized": True,
        "confirmation_start_utc": None,
        "minimum_calendar_days": 28,
        "readiness_prerequisites": [
            {
                "prerequisite": "immutable_bundle_contract_validated",
                "required": True,
                "expected": True,
            },
            {
                "prerequisite": "active_meta_prediction_and_policy_parity",
                "required": True,
                "expected": "pass_or_documented_dead_inputs",
            },
            {
                "prerequisite": "es_collector_healthy",
                "required": True,
                "expected": True,
            },
            {
                "prerequisite": "ibkr_account_mode",
                "required": True,
                "expected": "paper",
            },
            {
                "prerequisite": "paper_signal_ledger_ready",
                "required": True,
                "expected": True,
            },
        ],
    }
    assert decision_summary["portfolio_contract"]["overlapping_event_markouts_allowed"] is True
    assert decision_summary["portfolio_contract"]["broker_position_tracking"] is False
    assert decision_summary["portfolio_contract"]["stack_meta_and_reversal"] is False
    assert decision_summary["confirmation_markout_contract"] == {
        "entry_price_source": "signal_bar_close",
        "exit_price_source": "horizon_bar_close",
        "horizon_completed_bars": 20,
        "stop_target_semantics": "not_applicable",
        "overlapping_events_allowed": True,
        "fixed_slippage_ticks": 0.25,
        "commission_ticks": 0.40,
        "spread_cost_source": "accepted_scheduled_round_trip_spread",
        "broker_orders_or_positions": False,
    }
    assert decision_summary["overlap_evidence"]["shared_trade_count"] == 612
    assert decision_summary["overlap_evidence"]["reversal_unique_net_pnl_ticks"] == pytest.approx(
        -286.95
    )


def test_immutable_writer_allows_identical_rebuild_and_rejects_contract_mutation(
    tmp_path,
) -> None:
    path = tmp_path / "immutable.json"

    assert _write_immutable_text(path, json.dumps({"status": "active"})) is True
    assert _write_immutable_text(path, json.dumps({"status": "active"})) is False
    with pytest.raises(FileExistsError, match="new dated bundle id"):
        _write_immutable_text(path, json.dumps({"status": "candidate"}))
