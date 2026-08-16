from __future__ import annotations

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
from ote_live.features.manifest import AbstainPolicy, DirectionRuntimeManifest, LivePolicy, ThresholdConfig
from ote_live.models.registry import build_direction_runtime_manifests


SOURCE_BUNDLE_ID = "ict_es_shadow_20260730"
SOURCE_REGISTRY_PATH = REPO_ROOT / "models" / "ict_es_shadow_live_registry_20260730.json"
SOURCE_POLICY_DIR = REPO_ROOT / "ote_live" / "policy_artifacts" / SOURCE_BUNDLE_ID
SOURCE_BACKTEST_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "ict_backtests"
    / SOURCE_BUNDLE_ID
    / "run_summary.json"
)
SOURCE_WALK_FORWARD_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "ict_backtests"
    / "ict_es_primary_bootstrap_20260726_full"
    / "run_summary.json"
)

OUTPUT_BUNDLE_ID = "ict_es_paper_signal_20260813"
OUTPUT_REGISTRY_PATH = (
    REPO_ROOT / "models" / "ict_es_paper_signal_registry_20260813.json"
)
OUTPUT_POLICY_DIR = REPO_ROOT / "ote_live" / "policy_artifacts" / OUTPUT_BUNDLE_ID
OUTPUT_MANIFEST_DIR = REPO_ROOT / "ote_live" / "runtime_manifests" / OUTPUT_BUNDLE_ID
OUTPUT_REPORT_DIR = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "ict_paper_signal_bundles"
    / OUTPUT_BUNDLE_ID
)
OUTPUT_RUN_SUMMARY_PATH = OUTPUT_REPORT_DIR / "run_summary.json"
OUTPUT_DECISION_SUMMARY_PATH = OUTPUT_MANIFEST_DIR / "paper_signal_decision_summary.json"

PREPARED_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "ict_es_primary_refresh_20260724_spacing_refit_final_confirm"
    / "phase04_prepared"
)
PREPARED_SUMMARY_PATH = PREPARED_ROOT / "prepared" / "summary.json"
FEATURE_METADATA_PATH = PREPARED_ROOT / "ict_es_phase06_merged_dataset.metadata.json"
LONG_FEATURE_PATH = PREPARED_ROOT / "prepared" / "long_ict_continuation" / "features.json"
SHORT_FEATURE_PATH = PREPARED_ROOT / "prepared" / "short_ict_continuation" / "features.json"

PROMOTION_DATE = "2026-08-13"
DECISION_EFFECTIVE_AT_UTC = datetime(2026, 8, 13, tzinfo=timezone.utc)
META_MODEL_ID = "ict_long_meta_xgb_v1"
REVERSAL_MODEL_ID = "ict_long_reversal_xgb_v1"
EXPECTED_MODEL_IDS = frozenset(
    {
        "ict_long_continuation_xgb_v1",
        META_MODEL_ID,
        REVERSAL_MODEL_ID,
        "ict_short_continuation_xgb_v1",
        "ict_short_meta_xgb_v1",
        "ict_short_reversal_xgb_v1",
    }
)

META_PROMOTION_REASON = (
    "Promoted on 2026-08-13 to the controlled ICT paper-signal trial using the exact "
    "walk-forward contract: global threshold 0.40 with no regime thresholds or abstention "
    "filters. The July 30, 2026 leakage-safe backtest passed all 8 promotion-quality gates "
    "with Sharpe 1.586, approximate deflated Sharpe 1.068, and max drawdown 6.40%. Active "
    "status authorizes signal and alert evaluation only; it does not authorize broker orders."
)
REVERSAL_HOLD_REASON = (
    "Held as a candidate/shadow challenger on 2026-08-13 despite passing all 8 individual "
    "promotion-quality gates because it is not additive to ict_long_meta_xgb_v1: 612 exact "
    "shared walk-forward trades produced +5,758.20 ticks, equal to 105.2% of reversal PnL, "
    "while its 663 unique trades lost 286.95 ticks. Do not stack it with or allocate a second "
    "position alongside the meta paper-signal trial."
)


def main() -> int:
    _validate_frozen_source_bundle()

    source_registry_payload = _read_json(SOURCE_REGISTRY_PATH)
    source_run_summary = _read_json(SOURCE_BACKTEST_SUMMARY_PATH)
    walk_forward_summary = _read_json(SOURCE_WALK_FORWARD_SUMMARY_PATH)

    registry_payload = build_paper_registry_payload(source_registry_payload)
    _write_immutable_json(OUTPUT_REGISTRY_PATH, registry_payload)
    # Validate the materialized registry before using it to construct runtime manifests.
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
        selection_payload = build_paper_policy_selection(
            source_selection,
            walk_forward_model_output=_get_model_output(walk_forward_summary, model_id),
            paper_policy=paper_policy,
        )
        _write_immutable_json(
            OUTPUT_POLICY_DIR / model_id / "policy_selection.json",
            selection_payload,
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
            "long": META_MODEL_ID,
            "short": "ict_short_reversal_xgb_v1",
        },
    )
    direction_manifests = {
        direction: _with_deterministic_manifest_timestamp(manifest)
        for direction, manifest in direction_manifests.items()
    }
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
                "active_model_ids": [META_MODEL_ID],
                "recommended_primary_model_id": META_MODEL_ID,
            },
            indent=2,
        )
    )
    return 0


def build_paper_registry_payload(source_payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(source_payload)
    records = {str(record["model_id"]): record for record in payload.get("models", [])}
    _validate_roster(records)

    for model_id, record in records.items():
        record["status"] = "candidate"
        if model_id == META_MODEL_ID:
            record.update(
                {
                    "global_threshold": 0.40,
                    "regime_thresholds": None,
                    "abstain_policy": None,
                    "promotion_date": PROMOTION_DATE,
                    "promotion_reason": META_PROMOTION_REASON,
                    "status": "active",
                }
            )
        elif model_id == REVERSAL_MODEL_ID:
            record.update(
                {
                    "global_threshold": 0.40,
                    "regime_thresholds": None,
                    "abstain_policy": None,
                    "promotion_date": PROMOTION_DATE,
                    "promotion_reason": REVERSAL_HOLD_REASON,
                }
            )

    payload["models"] = sorted(
        records.values(), key=lambda record: (str(record["direction"]), str(record["model_id"]))
    )
    return payload


def build_paper_live_policy(source_policy: LivePolicy) -> LivePolicy:
    model_id = source_policy.model_id
    if model_id not in EXPECTED_MODEL_IDS:
        raise ValueError(f"Unexpected ICT model in frozen policy bundle: {model_id!r}.")

    lineage_updates: dict[str, Any] = {
        "threshold_registry_path": _repo_relative_str(OUTPUT_REGISTRY_PATH),
        "policy_backtest_summary_path": _repo_relative_str(OUTPUT_RUN_SUMMARY_PATH),
        "active_registry_path": _repo_relative_str(SOURCE_REGISTRY_PATH),
        "source_model_id": model_id,
        "source_match_type": "artifact_path",
    }
    policy_updates: dict[str, Any] = {}

    if model_id == META_MODEL_ID:
        lineage_updates.update(
            {
                "policy_source_type": "ict_paper_signal_walk_forward_contract",
                "policy_search_summary_path": None,
                "policy_table_path": _walk_forward_model_artifact(model_id, "policy_table.csv"),
                "policy_evaluation_path": _walk_forward_model_artifact(
                    model_id, "policy_evaluation.csv"
                ),
                "selected_policy_name": "global_threshold",
                "qualified_policy_names": ["global_threshold"],
                "notes": [
                    "Controlled ICT paper-signal leader; active status enables the emit/notification path but no broker order-routing path is authorized.",
                    "Exact policy contract is global threshold 0.40 with no regime thresholds and no abstention filters.",
                    "The leakage-safe walk-forward backtest selected global_threshold for all 1,555 out-of-sample trades and passed all 8 promotion-quality gates.",
                    "The frozen static threshold search selected regime_threshold_plus_abstain; that mismatched static package is intentionally not carried into this trial.",
                ],
            }
        )
        policy_updates.update(
            {
                "thresholds": ThresholdConfig(global_threshold=0.40, regime_thresholds=None),
                "abstain_policy": _disabled_abstain_policy(),
            }
        )
    elif model_id == REVERSAL_MODEL_ID:
        lineage_updates.update(
            {
                "policy_source_type": "ict_paper_signal_shadow_challenger",
                "policy_search_summary_path": None,
                "policy_table_path": _walk_forward_model_artifact(model_id, "policy_table.csv"),
                "policy_evaluation_path": _walk_forward_model_artifact(
                    model_id, "policy_evaluation.csv"
                ),
                "selected_policy_name": "global_threshold",
                "qualified_policy_names": ["global_threshold"],
                "notes": [
                    "Candidate/shadow challenger under the exact global threshold 0.40 policy with no abstention filters.",
                    "Held from a second paper allocation because 612 shared trades produced 105.2% of reversal PnL while 663 reversal-only trades lost 286.95 ticks.",
                    "Do not stack this signal with the active long-meta paper-signal position.",
                ],
            }
        )
        policy_updates.update(
            {
                "thresholds": ThresholdConfig(global_threshold=0.40, regime_thresholds=None),
                "abstain_policy": _disabled_abstain_policy(),
            }
        )
    else:
        lineage_updates.update(
            {
                "policy_source_type": "ict_paper_signal_shadow_challenger",
                "notes": [
                    *source_policy.lineage.notes,
                    "This branch remains candidate/shadow in the 2026-08-13 paper-signal bundle.",
                ],
            }
        )

    policy_updates["lineage"] = source_policy.lineage.model_copy(update=lineage_updates)
    return source_policy.model_copy(deep=True, update=policy_updates)


def build_paper_policy_selection(
    source_payload: dict[str, Any],
    *,
    walk_forward_model_output: dict[str, Any],
    paper_policy: LivePolicy,
) -> dict[str, Any]:
    payload = deepcopy(source_payload)
    model_id = str(payload["model_id"])
    status = "active" if model_id == META_MODEL_ID else "candidate"
    reason = (
        META_PROMOTION_REASON
        if model_id == META_MODEL_ID
        else REVERSAL_HOLD_REASON
        if model_id == REVERSAL_MODEL_ID
        else str(payload.get("selection_reason") or "Retained as a shadow challenger.")
    )

    payload.update(
        {
            "selected_policy_name": paper_policy.lineage.selected_policy_name,
            "qualified_policy_names": paper_policy.lineage.qualified_policy_names,
            "static_vs_walk_forward_dominant_mismatch": False
            if model_id == META_MODEL_ID
            else payload.get("static_vs_walk_forward_dominant_mismatch", False),
            "selection_reason": reason,
            "thresholds": paper_policy.thresholds.model_dump(),
            "deployment_decision": {
                "effective_date": PROMOTION_DATE,
                "status": status,
                "mode": "paper_signal" if status == "active" else "shadow",
                "reason": reason,
                "broker_order_submission_authorized": False,
            },
        }
    )
    if model_id == META_MODEL_ID:
        payload["source_static_policy_selection"] = {
            "selected_policy_name": source_payload.get("selected_policy_name"),
            "qualified_policy_names": source_payload.get("qualified_policy_names", []),
            "selected_policy_metrics": source_payload.get("selected_policy_metrics", {}),
        }
        payload["selected_policy_metrics"] = _walk_forward_contract_metrics(
            walk_forward_model_output
        )
    return payload


def build_paper_run_summary(source_payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(source_payload)
    payload.update(
        {
            "generated_at_utc": DECISION_EFFECTIVE_AT_UTC.isoformat(),
            "output_root": _repo_relative_str(OUTPUT_REPORT_DIR),
            "registry_path": _repo_relative_str(OUTPUT_REGISTRY_PATH),
            "statuses": ["active", "candidate"],
            "source_bundle_id": SOURCE_BUNDLE_ID,
            "source_backtest_summary_path": _repo_relative_str(
                SOURCE_BACKTEST_SUMMARY_PATH
            ),
            "paper_signal_bundle": {
                "bundle_id": OUTPUT_BUNDLE_ID,
                "decision_effective_at_utc": DECISION_EFFECTIVE_AT_UTC.isoformat(),
                "active_model_ids": [META_MODEL_ID],
                "recommended_primary_model_id": META_MODEL_ID,
                "broker_order_submission_authorized": False,
                "overlapping_event_markouts_allowed": True,
                "broker_position_tracking": False,
                "markout_horizon_completed_bars": 20,
                "entry_price_source": "signal_bar_close",
                "exit_price_source": "horizon_bar_close",
                "stop_target_semantics": "not_applicable",
            },
        }
    )
    for model_output in payload.get("model_outputs", []):
        model_id = str(model_output["model_id"])
        if model_id not in EXPECTED_MODEL_IDS:
            raise ValueError(f"Unexpected model output in frozen ICT run summary: {model_id!r}.")
        model_output["paper_signal_status"] = "active" if model_id == META_MODEL_ID else "candidate"
        if model_id == META_MODEL_ID:
            model_output["selection_reason"] = META_PROMOTION_REASON
            model_output["live_registry_thresholds"] = {
                "global_threshold": 0.40,
                "regime_thresholds": None,
            }
        elif model_id == REVERSAL_MODEL_ID:
            model_output["selection_reason"] = REVERSAL_HOLD_REASON
            model_output["live_registry_thresholds"] = {
                "global_threshold": 0.40,
                "regime_thresholds": None,
            }
    return payload


def build_paper_decision_summary(registry_payload: dict[str, Any]) -> dict[str, Any]:
    records = {str(record["model_id"]): record for record in registry_payload["models"]}
    _validate_roster(records)
    return {
        "bundle_id": OUTPUT_BUNDLE_ID,
        "source_bundle_id": SOURCE_BUNDLE_ID,
        "decision_effective_at_utc": DECISION_EFFECTIVE_AT_UTC.isoformat(),
        "registry_path": _repo_relative_str(OUTPUT_REGISTRY_PATH),
        "policy_dir": _repo_relative_str(OUTPUT_POLICY_DIR),
        "runtime_manifest_dir": _repo_relative_str(OUTPUT_MANIFEST_DIR),
        "backtest_summary_path": _repo_relative_str(OUTPUT_RUN_SUMMARY_PATH),
        "recommended_primary_model_id": META_MODEL_ID,
        "active_model_ids": [META_MODEL_ID],
        "broker_order_submission_authorized": False,
        "trial_lifecycle": {
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
        },
        "portfolio_contract": {
            "overlapping_event_markouts_allowed": True,
            "broker_position_tracking": False,
            "stack_meta_and_reversal": False,
            "reversal_is_non_additive_shadow_challenger": True,
        },
        "confirmation_markout_contract": {
            "entry_price_source": "signal_bar_close",
            "exit_price_source": "horizon_bar_close",
            "horizon_completed_bars": 20,
            "stop_target_semantics": "not_applicable",
            "overlapping_events_allowed": True,
            "fixed_slippage_ticks": 0.25,
            "commission_ticks": 0.40,
            "spread_cost_source": "accepted_scheduled_round_trip_spread",
            "broker_orders_or_positions": False,
        },
        "overlap_evidence": {
            "match_key": ["entry_datetime", "source_row_idx"],
            "shared_trade_count": 612,
            "shared_net_pnl_ticks": 5758.20,
            "shared_share_of_meta_net_pnl_pct": 77.1747,
            "shared_share_of_reversal_net_pnl_pct": 105.2447,
            "reversal_unique_trade_count": 663,
            "reversal_unique_net_pnl_ticks": -286.95,
            "source_meta_trades": _walk_forward_model_artifact(
                META_MODEL_ID, "selected_test_trades.csv"
            ),
            "source_reversal_trades": _walk_forward_model_artifact(
                REVERSAL_MODEL_ID, "selected_test_trades.csv"
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


def _validate_frozen_source_bundle() -> None:
    required_paths = [
        SOURCE_REGISTRY_PATH,
        SOURCE_BACKTEST_SUMMARY_PATH,
        SOURCE_WALK_FORWARD_SUMMARY_PATH,
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
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Frozen ICT promotion inputs are incomplete: "
            + ", ".join(_repo_relative_str(path) for path in missing)
        )

    source_registry = load_ote_model_registry(SOURCE_REGISTRY_PATH)
    records = {record.model_id: record for record in source_registry.models}
    _validate_roster(records)
    non_candidate = sorted(
        record.model_id for record in source_registry.models if record.status != "candidate"
    )
    if non_candidate:
        raise ValueError(
            "The frozen July 30 source registry no longer has its all-candidate contract: "
            + ", ".join(non_candidate)
        )


def _validate_materialized_contract(
    *,
    registry_payload: dict[str, Any],
    paper_policies: dict[str, LivePolicy],
    direction_manifests: dict[str, DirectionRuntimeManifest],
) -> None:
    records = {str(record["model_id"]): record for record in registry_payload["models"]}
    statuses = {model_id: str(record["status"]) for model_id, record in records.items()}
    active_ids = sorted(model_id for model_id, status in statuses.items() if status == "active")
    if active_ids != [META_MODEL_ID]:
        raise ValueError(f"Paper-signal registry must activate only {META_MODEL_ID}: {active_ids}.")

    meta_policy = paper_policies[META_MODEL_ID]
    if (
        meta_policy.thresholds.global_threshold != 0.40
        or meta_policy.thresholds.regime_thresholds is not None
        or meta_policy.abstain_policy.enabled
    ):
        raise ValueError("Long-meta paper policy is not exact global 0.40/no-abstain.")

    long_manifest = direction_manifests.get("long")
    short_manifest = direction_manifests.get("short")
    if long_manifest is None or short_manifest is None:
        raise ValueError("Paper-signal bundle must contain both long and short direction manifests.")
    if long_manifest.recommendations.recommended_primary_model_id != META_MODEL_ID:
        raise ValueError("Long-meta must be the recommended long primary model.")
    manifest_statuses = {
        model.model_id: model.status
        for manifest in direction_manifests.values()
        for model in manifest.models
    }
    if manifest_statuses != statuses:
        raise ValueError("Runtime-manifest statuses do not match the paper-signal registry.")


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


def _walk_forward_contract_metrics(model_output: dict[str, Any]) -> dict[str, Any]:
    overall = dict(model_output.get("overall_test_metrics") or {})
    walk_forward = dict(model_output.get("walk_forward_efficiency") or {})
    return {
        "trade_count": overall.get("trade_count"),
        "net_pnl_units": overall.get("total_net_pnl_units"),
        "expectancy_units": overall.get("expectancy_units"),
        "monthly_sharpe": overall.get("monthly_sharpe"),
        "approx_deflated_sharpe": overall.get("approx_deflated_sharpe"),
        "max_drawdown_pct": overall.get("max_drawdown_pct"),
        "overall_wfe": walk_forward.get("overall_wfe"),
    }


def _disabled_abstain_policy() -> AbstainPolicy:
    return AbstainPolicy(
        enabled=False,
        abstain_high_stress=False,
        abstain_off_hours=False,
        cooldown_bars=0,
        minimum_expected_move_to_spread=0.0,
        abstain_session_regimes=[],
        abstain_composite_regimes=[],
        abstain_composite_session_pairs=[],
        abstain_composite_stress_pairs=[],
        minimum_probability_quantile=None,
    )


def _validate_roster(records: dict[str, Any]) -> None:
    actual = frozenset(records)
    if actual != EXPECTED_MODEL_IDS:
        raise ValueError(
            "Frozen ICT roster mismatch; expected "
            f"{sorted(EXPECTED_MODEL_IDS)}, got {sorted(actual)}."
        )


def _get_model_output(summary_payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    for model_output in summary_payload.get("model_outputs", []):
        if model_output.get("model_id") == model_id:
            return model_output
    raise KeyError(f"Could not find {model_id!r} in the ICT walk-forward summary.")


def _walk_forward_model_artifact(model_id: str, filename: str) -> str:
    return _repo_relative_str(SOURCE_WALK_FORWARD_SUMMARY_PATH.parent / model_id / filename)


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> bool:
    return _write_immutable_text(path, json.dumps(payload, indent=2) + "\n")


def _write_immutable_text(path: Path, text: str) -> bool:
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return False
        raise FileExistsError(
            "Refusing to mutate immutable ICT paper-signal artifact "
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
