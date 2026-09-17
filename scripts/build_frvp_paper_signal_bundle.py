from __future__ import annotations

import csv
import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ote_registry_loader import load_ote_model_registry
from ote_live.features.manifest import (
    AbstainPolicy,
    DirectionRuntimeManifest,
    LivePolicy,
    ThresholdConfig,
)
from ote_live.models.registry import build_direction_runtime_manifests


SOURCE_BUNDLE_ID = "frvp_es_shadow_20260721"
SOURCE_REGISTRY_PATH = REPO_ROOT / "models" / "frvp_es_shadow_live_registry_20260721.json"
SOURCE_POLICY_DIR = REPO_ROOT / "ote_live" / "policy_artifacts" / SOURCE_BUNDLE_ID
SOURCE_BACKTEST_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_backtests"
    / "frvp_es_shadow_live_bundle_20260721"
    / "run_summary.json"
)
SOURCE_REGIME_GATED_ROOT = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_regime_gated_deployment"
    / "frvp_regime_gated_deployment_20260721"
)
SOURCE_PROMOTION_SNAPSHOT_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_promotion_packages"
    / "frvp_long_continuation_xgb_v1_20260717"
    / "promotion_package_summary.json"
)

OUTPUT_BUNDLE_ID = "frvp_es_paper_signal_20260816"
OUTPUT_REGISTRY_PATH = REPO_ROOT / "models" / "frvp_es_paper_signal_registry_20260816.json"
OUTPUT_POLICY_DIR = REPO_ROOT / "ote_live" / "policy_artifacts" / OUTPUT_BUNDLE_ID
OUTPUT_MANIFEST_DIR = REPO_ROOT / "ote_live" / "runtime_manifests" / OUTPUT_BUNDLE_ID
OUTPUT_REPORT_DIR = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_paper_signal_bundles"
    / OUTPUT_BUNDLE_ID
)
OUTPUT_RUN_SUMMARY_PATH = OUTPUT_REPORT_DIR / "run_summary.json"
OUTPUT_DECISION_SUMMARY_PATH = OUTPUT_MANIFEST_DIR / "paper_signal_decision_summary.json"

PREPARED_SUMMARY_PATH = (
    REPO_ROOT
    / "artifacts"
    / "frvp_es_primary_refresh_20260701"
    / "phase04"
    / "prepared"
    / "summary.json"
)
FEATURE_METADATA_PATH = (
    REPO_ROOT
    / "artifacts"
    / "frvp_es_primary_refresh_20260701"
    / "phase02"
    / "es_primary_frvp_phase04_dataset.csv.metadata.json"
)
LONG_FEATURE_PATH = (
    REPO_ROOT
    / "artifacts"
    / "frvp_es_primary_refresh_20260701"
    / "phase04"
    / "prepared"
    / "long_frvp_continuation"
    / "features.json"
)
SHORT_FEATURE_PATH = (
    REPO_ROOT
    / "artifacts"
    / "frvp_es_primary_refresh_20260701"
    / "phase04"
    / "prepared"
    / "short_frvp_continuation"
    / "features.json"
)

DECISION_DATE = "2026-08-16"
DECISION_EFFECTIVE_AT_UTC = datetime(2026, 8, 16, tzinfo=timezone.utc)
CONTINUATION_MODEL_ID = "frvp_long_continuation_xgb_v1"
REVERSAL_MODEL_ID = "frvp_long_reversal_xgb_v1"
ACTIVE_MODEL_IDS = (REVERSAL_MODEL_ID,)
ACTIVE_ARTIFACT_HASH_REFERENCE_KEYS = (
    "model_file",
    "scaler_file",
    "calibrator_file",
    "model_config_file",
    "training_summary_file",
)
CONTROLLED_POLICY_MODEL_IDS = (CONTINUATION_MODEL_ID, REVERSAL_MODEL_ID)
EXPECTED_MODEL_STATUSES = {
    CONTINUATION_MODEL_ID: "candidate",
    "frvp_long_meta_xgb_v1": "candidate",
    REVERSAL_MODEL_ID: "active",
    "frvp_short_continuation_tcn_v1": "candidate",
    "frvp_short_meta_xgb_v1": "candidate",
    "frvp_short_reversal_xgb_v1": "deprecated",
}
EXPECTED_MODEL_IDS = frozenset(EXPECTED_MODEL_STATUSES)
ACTIVE_THRESHOLDS = {
    CONTINUATION_MODEL_ID: 0.70,
    REVERSAL_MODEL_ID: 0.60,
}
ACTIVE_BACKTEST_RUN_NAMES = {
    CONTINUATION_MODEL_ID: "long_continuation_v3_baseline",
    REVERSAL_MODEL_ID: "long_reversal_recent2y_sdh_overlap_prune_v1",
}
ACTIVE_TARGETED_FILTER_PRESETS = {
    CONTINUATION_MODEL_ID: "frvp_long_continuation_xgb_overlap_composite_prune_v3",
    REVERSAL_MODEL_ID: "frvp_long_reversal_xgb_recent2y_concentration_sdh_overlap_v1",
}

CONTINUATION_PROMOTION_REASON = (
    "Held as a candidate/shadow branch on 2026-08-16. Its favorable July 21 WFO promotion "
    "evidence (512 trades, +7,001.20 ticks, Sharpe 1.187) mixed fold-selected global and "
    "regime policies, while the exact fixed live policy (global 0.70 plus accepted filters) "
    "replayed to 180 trades, +2,473.00 ticks, Sharpe 0.7384, DSR 0.7040, 7.64% drawdown, "
    "0.5833 profitable-quarter share, and 0.1453 largest-trade share. It fails the exact-policy "
    "Sharpe, quarter-share, and concentration gates and is not authorized for the trial."
)
REVERSAL_PROMOTION_REASON = (
    "Activated on 2026-08-16 for the controlled FRVP paper-signal trial under the exact "
    "accepted recent-regime contract: global threshold 0.60 plus all accepted targeted filters, "
    "including strong_down_high/overlap. The exact fixed-policy replay produced 50 trades, "
    "+2,567.50 ticks, Sharpe 1.3984, DSR 1.1477, 5.39% drawdown, and 0.60 profitable-quarter "
    "share; its 0.1287 largest-trade share still fails concentration. The trial is authorized "
    "to validate that concentration uncertainty only; it is not a full promotion or order authorization."
)
DECISION_REASONS = {
    CONTINUATION_MODEL_ID: CONTINUATION_PROMOTION_REASON,
    REVERSAL_MODEL_ID: REVERSAL_PROMOTION_REASON,
}

EXACT_FIXED_POLICY_SENSITIVITY = {
    CONTINUATION_MODEL_ID: {
        "policy_contract": "global_threshold_0.70_plus_accepted_targeted_filters",
        "trade_count": 180,
        "net_pnl_ticks": 2473.0,
        "monthly_sharpe": 0.7384,
        "approx_deflated_sharpe": 0.7040,
        "max_drawdown_pct": 7.64,
        "profitable_quarter_share": 0.5833,
        "largest_single_trade_share_of_total_pnl": 0.1453,
        "failed_promotion_gates": [
            "annualized_sharpe_above_threshold",
            "profitable_quarter_share_above_threshold",
            "largest_single_trade_share_below_limit",
        ],
        "decision": "candidate_shadow_hold",
    },
    REVERSAL_MODEL_ID: {
        "policy_contract": "global_threshold_0.60_plus_accepted_targeted_filters",
        "trade_count": 50,
        "net_pnl_ticks": 2567.5,
        "monthly_sharpe": 1.3984,
        "approx_deflated_sharpe": 1.1477,
        "max_drawdown_pct": 5.39,
        "profitable_quarter_share": 0.60,
        "largest_single_trade_share_of_total_pnl": 0.1287,
        "failed_promotion_gates": ["largest_single_trade_share_below_limit"],
        "decision": "controlled_paper_signal_concentration_validation",
    },
}

def main() -> int:
    _validate_frozen_source_bundle()

    source_registry_payload = _read_json(SOURCE_REGISTRY_PATH)
    source_run_summary = _read_json(SOURCE_BACKTEST_SUMMARY_PATH)
    registry_payload = build_paper_registry_payload(source_registry_payload)
    _write_immutable_json(OUTPUT_REGISTRY_PATH, registry_payload)
    load_ote_model_registry(OUTPUT_REGISTRY_PATH)

    paper_policies: dict[str, LivePolicy] = {}
    for model_id in sorted(EXPECTED_MODEL_IDS):
        source_policy = LivePolicy.model_validate_json(
            (SOURCE_POLICY_DIR / model_id / "live_policy.json").read_text(encoding="utf-8")
        )
        paper_policy = build_paper_live_policy(source_policy)
        paper_policies[model_id] = paper_policy
        _write_immutable_text(
            OUTPUT_POLICY_DIR / model_id / "live_policy.json",
            paper_policy.model_dump_json(indent=2) + "\n",
        )

        source_selection = _read_json(SOURCE_POLICY_DIR / model_id / "policy_selection.json")
        _write_immutable_json(
            OUTPUT_POLICY_DIR / model_id / "policy_selection.json",
            build_paper_policy_selection(
                source_selection,
                paper_policy=paper_policy,
            ),
        )

    run_summary_payload = build_paper_run_summary(source_run_summary)
    _write_immutable_json(OUTPUT_RUN_SUMMARY_PATH, run_summary_payload)

    direction_manifests = build_direction_runtime_manifests(
        registry_path=OUTPUT_REGISTRY_PATH,
        prepared_summary_path=PREPARED_SUMMARY_PATH,
        feature_metadata_path=FEATURE_METADATA_PATH,
        long_feature_path=LONG_FEATURE_PATH,
        short_feature_path=SHORT_FEATURE_PATH,
        policy_backtest_summary_path=OUTPUT_RUN_SUMMARY_PATH,
        packaged_policy_dir=OUTPUT_POLICY_DIR,
        preferred_primary_model_ids={
            "long": REVERSAL_MODEL_ID,
            "short": "frvp_short_meta_xgb_v1",
        },
    )
    direction_manifests = {
        direction: _with_deterministic_manifest_timestamp(manifest)
        for direction, manifest in direction_manifests.items()
    }
    direction_manifests = _with_active_artifact_content_hashes(direction_manifests)
    _validate_materialized_contract(
        registry_payload=registry_payload,
        paper_policies=paper_policies,
        direction_manifests=direction_manifests,
    )
    _write_direction_runtime_manifests_immutable(direction_manifests)

    decision_summary = build_paper_decision_summary(registry_payload)
    _write_immutable_json(OUTPUT_DECISION_SUMMARY_PATH, decision_summary)

    print(
        json.dumps(
            {
                "bundle_id": OUTPUT_BUNDLE_ID,
                "registry_path": _repo_relative_str(OUTPUT_REGISTRY_PATH),
                "policy_dir": _repo_relative_str(OUTPUT_POLICY_DIR),
                "manifest_dir": _repo_relative_str(OUTPUT_MANIFEST_DIR),
                "run_summary_path": _repo_relative_str(OUTPUT_RUN_SUMMARY_PATH),
                "decision_summary_path": _repo_relative_str(OUTPUT_DECISION_SUMMARY_PATH),
                "active_model_ids": list(ACTIVE_MODEL_IDS),
                "recommended_primary_model_id": REVERSAL_MODEL_ID,
                "trial_state": "authorized_pending_readiness_not_started",
                "broker_order_submission_authorized": False,
            },
            indent=2,
        )
    )
    return 0


def build_paper_registry_payload(source_payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(source_payload)
    records = {str(record["model_id"]): record for record in payload.get("models", [])}
    _validate_roster(records)

    for model_id, expected_status in EXPECTED_MODEL_STATUSES.items():
        record = records[model_id]
        record["status"] = expected_status
        if model_id not in CONTROLLED_POLICY_MODEL_IDS:
            continue

        targeted_filters = _accepted_targeted_filters(model_id)
        record.update(
            {
                "global_threshold": ACTIVE_THRESHOLDS[model_id],
                "regime_thresholds": None,
                "abstain_policy": _build_registry_abstain_payload(
                    record.get("abstain_policy"),
                    targeted_filters=targeted_filters,
                    targeted_filter_preset=ACTIVE_TARGETED_FILTER_PRESETS[model_id],
                ),
                "promotion_date": DECISION_DATE,
                "promotion_reason": DECISION_REASONS[model_id],
            }
        )

    payload["models"] = sorted(
        records.values(), key=lambda record: (str(record["direction"]), str(record["model_id"]))
    )
    return payload


def build_paper_live_policy(source_policy: LivePolicy) -> LivePolicy:
    model_id = source_policy.model_id
    if model_id not in EXPECTED_MODEL_IDS:
        raise ValueError(f"Unexpected FRVP model in frozen policy bundle: {model_id!r}.")

    status = EXPECTED_MODEL_STATUSES[model_id]
    lineage_updates: dict[str, Any] = {
        "threshold_registry_path": _repo_relative_str(OUTPUT_REGISTRY_PATH),
        "policy_backtest_summary_path": _repo_relative_str(OUTPUT_RUN_SUMMARY_PATH),
        "active_registry_path": _repo_relative_str(SOURCE_REGISTRY_PATH),
        "source_model_id": model_id,
        "source_match_type": "artifact_path",
    }
    policy_updates: dict[str, Any] = {}

    if model_id in CONTROLLED_POLICY_MODEL_IDS:
        source_summary_path = _active_backtest_summary_path(model_id)
        source_model_dir = source_summary_path.parent / model_id
        targeted_filters = _accepted_targeted_filters(model_id)
        notes = [
            (
                f"Controlled FRVP paper-signal branch decision recorded on {DECISION_DATE}; "
                f"status is {status}."
            ),
            f"Exact accepted source backtest: {_repo_relative_str(source_summary_path)}.",
            f"Exact policy contract is global threshold {ACTIVE_THRESHOLDS[model_id]:.2f} with the accepted targeted-filter payload.",
            "Broker order submission and broker position tracking remain unauthorized.",
        ]
        if model_id == CONTINUATION_MODEL_ID:
            notes.extend(
                [
                    "This continuation branch is candidate/shadow, not active, because its fixed-live-policy sensitivity fails Sharpe, quarter-share, and concentration gates.",
                    "The July 21 runtime source contains 512 selected walk-forward trades and is authoritative for this runtime bundle.",
                    "The canonical July 17 promotion snapshot with 524 trades remains cited separately and is not silently substituted for the runtime evidence.",
                ]
            )
        else:
            notes.append(
                "The accepted backtest payload contains strong_down_high/overlap; it repairs the stale July 21 source live-policy omission."
            )
        lineage_updates.update(
            {
                "policy_source_type": (
                    "frvp_paper_signal_accepted_backtest_contract"
                    if status == "active"
                    else "frvp_paper_signal_fixed_policy_shadow_contract"
                ),
                "policy_table_path": _repo_relative_str(source_model_dir / "policy_table.csv"),
                "policy_evaluation_path": _repo_relative_str(
                    source_model_dir / "policy_evaluation.csv"
                ),
                "selected_policy_name": "global_threshold",
                "qualified_policy_names": [],
                "notes": notes,
            }
        )
        policy_updates.update(
            {
                "thresholds": ThresholdConfig(
                    global_threshold=ACTIVE_THRESHOLDS[model_id],
                    regime_thresholds=None,
                ),
                "abstain_policy": _coerce_targeted_filters(targeted_filters),
                "cost_assumptions": source_policy.cost_assumptions.model_copy(
                    update={
                        "targeted_filter_preset": ACTIVE_TARGETED_FILTER_PRESETS[model_id]
                    }
                ),
            }
        )
    else:
        lineage_updates.update(
            {
                "policy_source_type": "frvp_paper_signal_shadow_challenger",
                "notes": [
                    *source_policy.lineage.notes,
                    f"This branch remains {status} in the {DECISION_DATE} controlled paper-signal bundle.",
                    "Broker order submission is not authorized.",
                ],
            }
        )

    policy_updates["lineage"] = source_policy.lineage.model_copy(update=lineage_updates)
    return source_policy.model_copy(deep=True, update=policy_updates)


def build_paper_policy_selection(
    source_payload: dict[str, Any],
    *,
    paper_policy: LivePolicy,
) -> dict[str, Any]:
    payload = deepcopy(source_payload)
    model_id = str(payload["model_id"])
    if model_id not in EXPECTED_MODEL_IDS:
        raise ValueError(f"Unexpected FRVP policy selection model: {model_id!r}.")

    status = EXPECTED_MODEL_STATUSES[model_id]
    reason = (
        DECISION_REASONS[model_id]
        if model_id in CONTROLLED_POLICY_MODEL_IDS
        else str(payload.get("selection_reason") or f"Retained as {status}.")
    )
    payload.update(
        {
            "selected_policy_name": paper_policy.lineage.selected_policy_name,
            "qualified_policy_names": paper_policy.lineage.qualified_policy_names,
            "selection_reason": reason,
            "thresholds": paper_policy.thresholds.model_dump(),
            "deployment_decision": {
                "effective_date": DECISION_DATE,
                "status": status,
                "mode": "paper_signal" if status == "active" else status,
                "reason": reason,
                "broker_order_submission_authorized": False,
            },
        }
    )

    if model_id in CONTROLLED_POLICY_MODEL_IDS:
        source_summary = _active_backtest_summary(model_id)
        source_model_output = _get_model_output(source_summary, model_id)
        payload["accepted_backtest_contract"] = {
            "source_backtest_summary_path": _repo_relative_str(
                _active_backtest_summary_path(model_id)
            ),
            "targeted_filter_preset": source_summary.get("targeted_filter_preset"),
            "targeted_filters": _normalize_targeted_filters(
                source_model_output.get("targeted_filters") or {}
            ),
            "walk_forward_snapshot": _walk_forward_snapshot(source_model_output),
        }
        payload["fixed_live_policy_sensitivity"] = deepcopy(
            EXACT_FIXED_POLICY_SENSITIVITY[model_id]
        )
        if model_id == CONTINUATION_MODEL_ID:
            payload["canonical_promotion_snapshot"] = _canonical_continuation_snapshot()
            payload["runtime_evidence_precedence"] = (
                "The July 21 512-trade rerun is the runtime source; the July 17 "
                "524-trade package is retained as a separate canonical promotion snapshot."
            )
    return payload


def build_paper_run_summary(source_payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(source_payload)
    payload.update(
        {
            "generated_at_utc": DECISION_EFFECTIVE_AT_UTC.isoformat(),
            "output_root": _repo_relative_str(OUTPUT_REPORT_DIR),
            "registry_path": _repo_relative_str(OUTPUT_REGISTRY_PATH),
            "statuses": ["active", "candidate", "deprecated"],
            "source_bundle_id": SOURCE_BUNDLE_ID,
            "source_backtest_summary_path": _repo_relative_str(
                SOURCE_BACKTEST_SUMMARY_PATH
            ),
            "paper_signal_bundle": _paper_signal_contract_payload(),
            "fixed_live_policy_sensitivity_audit": _fixed_live_policy_sensitivity_audit(),
        }
    )
    seen_ids: set[str] = set()
    for model_output in payload.get("model_outputs", []):
        model_id = str(model_output["model_id"])
        if model_id not in EXPECTED_MODEL_IDS:
            raise ValueError(f"Unexpected model output in frozen FRVP run summary: {model_id!r}.")
        seen_ids.add(model_id)
        model_output["paper_signal_status"] = EXPECTED_MODEL_STATUSES[model_id]
        if model_id in CONTROLLED_POLICY_MODEL_IDS:
            accepted_output = _get_model_output(_active_backtest_summary(model_id), model_id)
            model_output["selection_reason"] = DECISION_REASONS[model_id]
            model_output["source_backtest_summary_path"] = _repo_relative_str(
                _active_backtest_summary_path(model_id)
            )
            model_output["live_registry_thresholds"] = {
                "global_threshold": ACTIVE_THRESHOLDS[model_id],
                "regime_thresholds": None,
            }
            model_output["targeted_filters"] = deepcopy(
                accepted_output.get("targeted_filters") or {}
            )
            model_output["fixed_live_policy_sensitivity"] = deepcopy(
                EXACT_FIXED_POLICY_SENSITIVITY[model_id]
            )
    if seen_ids != EXPECTED_MODEL_IDS:
        raise ValueError(
            f"Frozen FRVP run-summary roster mismatch: expected {sorted(EXPECTED_MODEL_IDS)}, "
            f"got {sorted(seen_ids)}."
        )
    return payload


def build_paper_decision_summary(registry_payload: dict[str, Any]) -> dict[str, Any]:
    records = {str(record["model_id"]): record for record in registry_payload["models"]}
    _validate_roster(records)
    contract = _paper_signal_contract_payload()
    return {
        "bundle_id": OUTPUT_BUNDLE_ID,
        "source_bundle_id": SOURCE_BUNDLE_ID,
        "decision_effective_at_utc": DECISION_EFFECTIVE_AT_UTC.isoformat(),
        "registry_path": _repo_relative_str(OUTPUT_REGISTRY_PATH),
        "policy_dir": _repo_relative_str(OUTPUT_POLICY_DIR),
        "runtime_manifest_dir": _repo_relative_str(OUTPUT_MANIFEST_DIR),
        "backtest_summary_path": _repo_relative_str(OUTPUT_RUN_SUMMARY_PATH),
        "recommended_primary_model_id": REVERSAL_MODEL_ID,
        "active_model_ids": list(ACTIVE_MODEL_IDS),
        "broker_order_submission_authorized": False,
        "trial_lifecycle": contract["trial_lifecycle"],
        "portfolio_contract": contract["portfolio_contract"],
        "confirmation_markout_contract": contract["confirmation_markout_contract"],
        "human_same_contract_signoff": contract["human_same_contract_signoff"],
        "trial_objective": contract["trial_objective"],
        "overlap_evidence": _calculate_overlap_evidence(),
        "evidence_snapshots": {
            "runtime_continuation": {
                "source_path": _repo_relative_str(
                    _active_backtest_summary_path(CONTINUATION_MODEL_ID)
                ),
                **_walk_forward_snapshot(
                    _get_model_output(
                        _active_backtest_summary(CONTINUATION_MODEL_ID),
                        CONTINUATION_MODEL_ID,
                    )
                ),
            },
            "canonical_continuation_promotion": _canonical_continuation_snapshot(),
            "runtime_reversal": {
                "source_path": _repo_relative_str(
                    _active_backtest_summary_path(REVERSAL_MODEL_ID)
                ),
                **_walk_forward_snapshot(
                    _get_model_output(
                        _active_backtest_summary(REVERSAL_MODEL_ID),
                        REVERSAL_MODEL_ID,
                    )
                ),
            },
            "continuation_evidence_precedence": (
                "Use the July 21 512-trade rerun for runtime policy and parity; retain "
                "the July 17 524-trade package as the separately cited canonical snapshot."
            ),
        },
        "fixed_live_policy_sensitivity_audit": _fixed_live_policy_sensitivity_audit(),
        "policy_correction_audit": {
            "model_id": REVERSAL_MODEL_ID,
            "stale_source_bundle_pair_count": 10,
            "accepted_backtest_pair_count": 11,
            "restored_pair": ["strong_down_high", "overlap"],
            "authoritative_source": _repo_relative_str(
                _active_backtest_summary_path(REVERSAL_MODEL_ID)
            ),
        },
        "model_decisions": [
            {
                "model_id": model_id,
                "direction": records[model_id]["direction"],
                "status": records[model_id]["status"],
                "promotion_date": records[model_id]["promotion_date"],
                "promotion_reason": records[model_id]["promotion_reason"],
            }
            for model_id in sorted(records)
        ],
    }


def _paper_signal_contract_payload() -> dict[str, Any]:
    return {
        "bundle_id": OUTPUT_BUNDLE_ID,
        "decision_effective_at_utc": DECISION_EFFECTIVE_AT_UTC.isoformat(),
        "active_model_ids": list(ACTIVE_MODEL_IDS),
        "recommended_primary_model_id": REVERSAL_MODEL_ID,
        "broker_order_submission_authorized": False,
        "trial_lifecycle": {
            "state": "authorized_pending_readiness_not_started",
            "decision_authorized": True,
            "start_authorized_after_readiness": True,
            "start_authorized_now": False,
            "start_requires_readiness_pass": True,
            "confirmation_start_utc": None,
            "minimum_calendar_days": 28,
            "readiness_prerequisites": [
                {
                    "prerequisite": "immutable_bundle_contract_validated",
                    "required": True,
                    "expected": True,
                },
                {
                    "prerequisite": "active_model_prediction_and_policy_parity",
                    "required": True,
                    "expected": True,
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
        },
        "portfolio_contract": {
            "overlapping_event_markouts_allowed": True,
            "broker_position_tracking": False,
            "active_signal_model_id": REVERSAL_MODEL_ID,
            "continuation_shadow_markouts_allowed": False,
            "independent_active_signal_markouts": True,
            "historical_exact_entry_overlap_count": 0,
        },
        "confirmation_markout_contract": {
            "entry_price_source": "signal_bar_close",
            "exit_price_source": "horizon_bar_close",
            "horizon_completed_bars": 120,
            "timeframe": "5m",
            "stop_target_semantics": "not_applicable",
            "overlapping_events_allowed": True,
            "fixed_slippage_ticks": 0.25,
            "commission_ticks": 0.40,
            "spread_cost_source": "accepted_scheduled_round_trip_spread",
            "broker_orders_or_positions": False,
        },
        "trial_objective": {
            "primary_model_id": REVERSAL_MODEL_ID,
            "objective": (
                "Forward-validate the reversal contract's remaining trade-concentration "
                "uncertainty under exact runtime policy and cost semantics."
            ),
            "historical_largest_single_trade_share_of_total_pnl": 0.1287,
            "historical_failed_gate": "largest_single_trade_share_below_limit",
            "full_promotion_authorized": False,
        },
        "human_same_contract_signoff": {
            "status": "deferred_for_controlled_paper_signal",
            "required_for_controlled_paper_signal": False,
            "waiver_scope": "controlled_paper_signal_only",
            "required_for_full_promotion": True,
            "full_promotion_authorized": False,
            "note": (
                "The deferred TradingView same-contract human signoff is waived only for "
                "this no-order controlled paper-signal trial; it is not waived for full promotion."
            ),
        },
    }


def _validate_frozen_source_bundle() -> None:
    required_paths = [
        SOURCE_REGISTRY_PATH,
        SOURCE_BACKTEST_SUMMARY_PATH,
        SOURCE_PROMOTION_SNAPSHOT_PATH,
        PREPARED_SUMMARY_PATH,
        FEATURE_METADATA_PATH,
        LONG_FEATURE_PATH,
        SHORT_FEATURE_PATH,
    ]
    required_paths.extend(
        SOURCE_POLICY_DIR / model_id / filename
        for model_id in EXPECTED_MODEL_IDS
        for filename in ("live_policy.json", "policy_selection.json")
    )
    required_paths.extend(
        _active_backtest_summary_path(model_id) for model_id in CONTROLLED_POLICY_MODEL_IDS
    )
    required_paths.extend(
        _active_trade_path(model_id) for model_id in CONTROLLED_POLICY_MODEL_IDS
    )
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Frozen FRVP decision inputs are incomplete: "
            + ", ".join(_repo_relative_str(path) for path in missing)
        )

    registry = load_ote_model_registry(SOURCE_REGISTRY_PATH)
    records = {record.model_id: record for record in registry.models}
    _validate_roster(records)
    expected_source_statuses = {
        model_id: "deprecated" if model_id == "frvp_short_reversal_xgb_v1" else "candidate"
        for model_id in EXPECTED_MODEL_IDS
    }
    actual_source_statuses = {model_id: record.status for model_id, record in records.items()}
    if actual_source_statuses != expected_source_statuses:
        raise ValueError(
            "The frozen July 21 FRVP registry status contract changed: "
            f"{actual_source_statuses}."
        )

    expected_trade_counts = {CONTINUATION_MODEL_ID: 512, REVERSAL_MODEL_ID: 63}
    for model_id, expected_trade_count in expected_trade_counts.items():
        summary = _active_backtest_summary(model_id)
        model_output = _get_model_output(summary, model_id)
        trade_count = int(model_output["overall_test_metrics"]["trade_count"])
        if trade_count != expected_trade_count:
            raise ValueError(
                f"Frozen {model_id} runtime evidence changed: expected "
                f"{expected_trade_count} trades, got {trade_count}."
            )
        acceptance = dict(model_output.get("acceptance") or {})
        if len(acceptance) != 8 or not all(bool(value) for value in acceptance.values()):
            raise ValueError(f"Frozen {model_id} evidence no longer passes all 8 gates.")

    reversal_pairs = _accepted_targeted_filters(REVERSAL_MODEL_ID)[
        "abstain_composite_session_pairs"
    ]
    if ["strong_down_high", "overlap"] not in reversal_pairs:
        raise ValueError("Accepted reversal backtest lost strong_down_high/overlap.")

    overlap = _calculate_overlap_evidence()
    if overlap["shared_trade_count"] != 0:
        raise ValueError(f"FRVP active branches no longer have zero exact-entry overlap: {overlap}.")


def _validate_materialized_contract(
    *,
    registry_payload: dict[str, Any],
    paper_policies: dict[str, LivePolicy],
    direction_manifests: dict[str, DirectionRuntimeManifest],
) -> None:
    records = {str(record["model_id"]): record for record in registry_payload["models"]}
    statuses = {model_id: str(record["status"]) for model_id, record in records.items()}
    if statuses != EXPECTED_MODEL_STATUSES:
        raise ValueError(f"FRVP paper-signal status contract mismatch: {statuses}.")

    for model_id in CONTROLLED_POLICY_MODEL_IDS:
        policy = paper_policies[model_id]
        if (
            policy.thresholds.global_threshold != ACTIVE_THRESHOLDS[model_id]
            or policy.thresholds.regime_thresholds is not None
            or not policy.abstain_policy.enabled
        ):
            raise ValueError(f"Active FRVP policy contract mismatch for {model_id}.")
        if _normalize_targeted_filters(policy.abstain_policy.model_dump()) != (
            _accepted_targeted_filters(model_id)
        ):
            raise ValueError(f"Active FRVP targeted-filter mismatch for {model_id}.")

    long_manifest = direction_manifests.get("long")
    short_manifest = direction_manifests.get("short")
    if long_manifest is None or short_manifest is None:
        raise ValueError("FRVP paper-signal bundle must contain long and short manifests.")
    if long_manifest.recommendations.recommended_primary_model_id != REVERSAL_MODEL_ID:
        raise ValueError("Long reversal must be the recommended FRVP trial model.")
    manifest_statuses = {
        model.model_id: model.status
        for manifest in direction_manifests.values()
        for model in manifest.models
    }
    expected_manifest_statuses = {
        model_id: status
        for model_id, status in EXPECTED_MODEL_STATUSES.items()
        if status != "deprecated"
    }
    if manifest_statuses != expected_manifest_statuses:
        raise ValueError("Runtime-manifest statuses do not match the FRVP registry.")
    for manifest in direction_manifests.values():
        if manifest.registry_path != _repo_relative_str(OUTPUT_REGISTRY_PATH):
            raise ValueError("Runtime manifest does not point to the immutable FRVP registry.")
    active_manifest = next(
        model
        for manifest in direction_manifests.values()
        for model in manifest.models
        if model.model_id == REVERSAL_MODEL_ID
    )
    expected_artifact_hashes = _artifact_content_hashes(active_manifest)
    if active_manifest.artifact_references.content_sha256 != expected_artifact_hashes:
        raise ValueError("Active FRVP artifact SHA-256 pins are missing or stale.")


def _with_deterministic_manifest_timestamp(
    manifest: DirectionRuntimeManifest,
) -> DirectionRuntimeManifest:
    models = [
        model.model_copy(update={"generated_at_utc": DECISION_EFFECTIVE_AT_UTC})
        for model in manifest.models
    ]
    return manifest.model_copy(
        update={"generated_at_utc": DECISION_EFFECTIVE_AT_UTC, "models": models}
    )


def _with_active_artifact_content_hashes(
    manifests: dict[str, DirectionRuntimeManifest],
) -> dict[str, DirectionRuntimeManifest]:
    pinned: dict[str, DirectionRuntimeManifest] = {}
    for direction, manifest in manifests.items():
        models = []
        for model in manifest.models:
            if model.model_id == REVERSAL_MODEL_ID:
                artifact_references = model.artifact_references.model_copy(
                    update={"content_sha256": _artifact_content_hashes(model)}
                )
                model = model.model_copy(
                    update={"artifact_references": artifact_references}
                )
            models.append(model)
        pinned[direction] = manifest.model_copy(update={"models": models})
    return pinned


def _artifact_content_hashes(manifest: Any) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for reference_key in ACTIVE_ARTIFACT_HASH_REFERENCE_KEYS:
        reference = getattr(manifest.artifact_references, reference_key)
        if not reference:
            raise ValueError(
                f"Active FRVP model {manifest.model_id} is missing {reference_key}."
            )
        path = Path(str(reference))
        resolved = path if path.is_absolute() else REPO_ROOT / path
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Active FRVP artifact does not exist: {_repo_relative_str(resolved)}."
            )
        hashes[reference_key] = _file_sha256(resolved)
    return hashes


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_direction_runtime_manifests_immutable(
    manifests: dict[str, DirectionRuntimeManifest],
) -> None:
    for direction, manifest in manifests.items():
        _write_immutable_text(
            OUTPUT_MANIFEST_DIR / f"live_runtime_manifest_{direction}.json",
            manifest.model_dump_json(indent=2) + "\n",
        )
        for model_manifest in manifest.models:
            model_dir = OUTPUT_MANIFEST_DIR / model_manifest.model_id
            _write_immutable_text(
                model_dir / "live_runtime_manifest.json",
                model_manifest.model_dump_json(indent=2) + "\n",
            )
            _write_immutable_text(
                model_dir / "live_policy.json",
                model_manifest.live_policy.model_dump_json(indent=2) + "\n",
            )


def _build_registry_abstain_payload(
    source_payload: dict[str, Any] | None,
    *,
    targeted_filters: dict[str, Any],
    targeted_filter_preset: str,
) -> dict[str, Any]:
    payload = deepcopy(source_payload or {})
    payload.update(deepcopy(targeted_filters))
    payload.update(
        {
            "policy_name": "global_threshold",
            "targeted_filter_preset": targeted_filter_preset,
            "apply_to_base_policy_variants": True,
        }
    )
    return payload


def _coerce_targeted_filters(payload: dict[str, Any]) -> AbstainPolicy:
    normalized = _normalize_targeted_filters(payload)
    return AbstainPolicy(
        enabled=True,
        abstain_high_stress=bool(normalized["abstain_high_stress"]),
        abstain_off_hours=bool(normalized["abstain_off_hours"]),
        cooldown_bars=int(normalized["cooldown_bars"]),
        minimum_expected_move_to_spread=float(
            normalized["minimum_expected_move_to_spread"]
        ),
        abstain_session_regimes=list(normalized["abstain_session_regimes"]),
        abstain_composite_regimes=list(normalized["abstain_composite_regimes"]),
        abstain_composite_session_pairs=[
            tuple(pair) for pair in normalized["abstain_composite_session_pairs"]
        ],
        abstain_composite_stress_pairs=[
            tuple(pair) for pair in normalized["abstain_composite_stress_pairs"]
        ],
        minimum_probability_quantile=normalized["minimum_probability_quantile"],
    )


def _normalize_targeted_filters(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "abstain_high_stress": bool(payload.get("abstain_high_stress", True)),
        "abstain_off_hours": bool(payload.get("abstain_off_hours", True)),
        "cooldown_bars": int(payload.get("cooldown_bars", 4)),
        "minimum_expected_move_to_spread": float(
            payload.get("minimum_expected_move_to_spread", 2.0)
        ),
        "abstain_session_regimes": [
            str(value) for value in payload.get("abstain_session_regimes", [])
        ],
        "abstain_composite_regimes": [
            str(value) for value in payload.get("abstain_composite_regimes", [])
        ],
        "abstain_composite_session_pairs": [
            [str(pair[0]), str(pair[1])]
            for pair in payload.get("abstain_composite_session_pairs", [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ],
        "abstain_composite_stress_pairs": [
            [str(pair[0]), str(pair[1])]
            for pair in payload.get("abstain_composite_stress_pairs", [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ],
        "minimum_probability_quantile": (
            None
            if payload.get("minimum_probability_quantile") is None
            else float(payload["minimum_probability_quantile"])
        ),
    }


def _accepted_targeted_filters(model_id: str) -> dict[str, Any]:
    model_output = _get_model_output(_active_backtest_summary(model_id), model_id)
    payload = model_output.get("targeted_filters")
    if not isinstance(payload, dict):
        raise ValueError(f"Accepted FRVP backtest lacks targeted filters for {model_id}.")
    return _normalize_targeted_filters(payload)


def _walk_forward_snapshot(model_output: dict[str, Any]) -> dict[str, Any]:
    overall = dict(model_output.get("overall_test_metrics") or {})
    walk_forward = dict(model_output.get("walk_forward_efficiency") or {})
    acceptance = dict(model_output.get("acceptance") or {})
    return {
        "selected_test_trades": overall.get("trade_count"),
        "selected_test_net_pnl_ticks": overall.get("total_net_pnl_units"),
        "selected_test_expectancy_ticks": overall.get("expectancy_units"),
        "selected_test_sharpe": overall.get("monthly_sharpe"),
        "selected_test_approx_deflated_sharpe": overall.get(
            "approx_deflated_sharpe"
        ),
        "selected_test_max_drawdown_pct": overall.get("max_drawdown_pct"),
        "overall_wfe": walk_forward.get("overall_wfe"),
        "profitable_quarter_share": overall.get("profitable_quarter_share"),
        "all_eight_promotion_quality_gates_passed": (
            len(acceptance) == 8 and all(bool(value) for value in acceptance.values())
        ),
    }


def _canonical_continuation_snapshot() -> dict[str, Any]:
    payload = _read_json(SOURCE_PROMOTION_SNAPSHOT_PATH)
    readout = dict(payload.get("promotion_readout") or {})
    return {
        "source_path": _repo_relative_str(SOURCE_PROMOTION_SNAPSHOT_PATH),
        "package_status": payload.get("package_status"),
        "promotion_decision": payload.get("promotion_decision"),
        "selected_test_trades": readout.get("selected_test_trade_count"),
        "selected_test_net_pnl_ticks": readout.get("selected_test_net_pnl_units"),
        "selected_test_sharpe": readout.get("selected_test_sharpe"),
        "selected_test_approx_deflated_sharpe": readout.get(
            "selected_test_deflated_sharpe"
        ),
        "overall_wfe": readout.get("overall_wfe"),
        "paper_trading_gate_passed": readout.get("paper_trading_gate_passed"),
    }


def _fixed_live_policy_sensitivity_audit() -> dict[str, Any]:
    source_registry = _read_json(SOURCE_REGISTRY_PATH)
    source_records = {
        str(record["model_id"]): record for record in source_registry.get("models", [])
    }
    _validate_roster(source_records)
    model_results: dict[str, Any] = {}
    for model_id in CONTROLLED_POLICY_MODEL_IDS:
        source_summary = _active_backtest_summary(model_id)
        model_results[model_id] = {
            "source_regime_report_root": _repo_relative_str(
                Path(str(source_summary["regime_report_root"]))
            ),
            "source_artifact_path": _repo_relative_str(
                Path(str(source_records[model_id]["artifact_path"]))
            ),
            **deepcopy(EXACT_FIXED_POLICY_SENSITIVITY[model_id]),
        }
    return {
        "audit_id": "frvp_fixed_live_policy_sensitivity_20260816",
        "audit_date": DECISION_DATE,
        "classification": "fixed_live_policy_sensitivity_not_promotion_quality_wfo",
        "embedded_snapshot": True,
        "materialized_source_report_path": None,
        "method": {
            "prediction_source": "frozen_labeled_oof_and_test_predictions",
            "fold_contract": "original_chronological_folds",
            "threshold_contract": "final_static_global_threshold",
            "hard_filter_contract": "accepted_targeted_filters",
            "cost_contract": "es_session_schedule_plus_fixed_slippage_and_commission",
            "entry_price_source": "signal_bar_close",
            "exit_price_source": "horizon_bar_close",
            "horizon_completed_bars": 120,
            "overlapping_markouts_allowed": True,
        },
        "model_results": model_results,
        "interpretation": (
            "Promotion-quality WFO snapshots remain archived separately. This sensitivity "
            "audit applies each final fixed live policy to the frozen predictions and governs "
            "the reversal-only controlled-trial roster."
        ),
    }


def _calculate_overlap_evidence() -> dict[str, Any]:
    rows_by_model = {
        model_id: _read_trade_rows(_active_trade_path(model_id))
        for model_id in CONTROLLED_POLICY_MODEL_IDS
    }
    keyed_rows: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    for model_id, rows in rows_by_model.items():
        keyed = {
            (str(row["entry_datetime"]), int(row["source_row_idx"])): row
            for row in rows
        }
        if len(keyed) != len(rows):
            raise ValueError(f"Duplicate exact-entry trade keys found for {model_id}.")
        keyed_rows[model_id] = keyed

    continuation_rows = keyed_rows[CONTINUATION_MODEL_ID]
    reversal_rows = keyed_rows[REVERSAL_MODEL_ID]
    shared_keys = set(continuation_rows) & set(reversal_rows)
    continuation_unique_keys = set(continuation_rows) - shared_keys
    reversal_unique_keys = set(reversal_rows) - shared_keys
    return {
        "match_key": ["entry_datetime", "source_row_idx"],
        "shared_trade_count": len(shared_keys),
        "shared_continuation_net_pnl_ticks": round(
            sum(float(continuation_rows[key]["net_pnl_units"]) for key in shared_keys), 8
        ),
        "shared_reversal_net_pnl_ticks": round(
            sum(float(reversal_rows[key]["net_pnl_units"]) for key in shared_keys), 8
        ),
        "continuation_unique_trade_count": len(continuation_unique_keys),
        "continuation_unique_net_pnl_ticks": round(
            sum(
                float(continuation_rows[key]["net_pnl_units"])
                for key in continuation_unique_keys
            ),
            8,
        ),
        "reversal_unique_trade_count": len(reversal_unique_keys),
        "reversal_unique_net_pnl_ticks": round(
            sum(float(reversal_rows[key]["net_pnl_units"]) for key in reversal_unique_keys),
            8,
        ),
        "source_continuation_trades": _repo_relative_str(
            _active_trade_path(CONTINUATION_MODEL_ID)
        ),
        "source_reversal_trades": _repo_relative_str(
            _active_trade_path(REVERSAL_MODEL_ID)
        ),
    }


def _active_backtest_summary_path(model_id: str) -> Path:
    try:
        run_name = ACTIVE_BACKTEST_RUN_NAMES[model_id]
    except KeyError as exc:
        raise ValueError(f"No accepted FRVP backtest is defined for {model_id!r}.") from exc
    return SOURCE_REGIME_GATED_ROOT / "backtest" / run_name / "run_summary.json"


def _active_backtest_summary(model_id: str) -> dict[str, Any]:
    return _read_json(_active_backtest_summary_path(model_id))


def _active_trade_path(model_id: str) -> Path:
    return _active_backtest_summary_path(model_id).parent / model_id / "selected_test_trades.csv"


def _read_trade_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_roster(records: dict[str, Any]) -> None:
    actual = frozenset(records)
    if actual != EXPECTED_MODEL_IDS:
        raise ValueError(
            "Frozen FRVP roster mismatch; expected "
            f"{sorted(EXPECTED_MODEL_IDS)}, got {sorted(actual)}."
        )


def _get_model_output(summary_payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    for model_output in summary_payload.get("model_outputs", []):
        if model_output.get("model_id") == model_id:
            return model_output
    raise KeyError(f"Could not find {model_id!r} in the FRVP backtest summary.")


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> bool:
    return _write_immutable_text(path, json.dumps(payload, indent=2) + "\n")


def _write_immutable_text(path: Path, text: str) -> bool:
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return False
        raise FileExistsError(
            "Refusing to mutate immutable FRVP paper-signal artifact "
            f"{_repo_relative_str(path)}. Use a new dated bundle id for a changed contract."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _repo_relative_str(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return path_obj.as_posix()
    resolved = path_obj.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
