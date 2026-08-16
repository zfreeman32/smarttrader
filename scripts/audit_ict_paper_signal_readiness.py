from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ID = "ict_es_paper_signal_20260813"
META_MODEL_ID = "ict_long_meta_xgb_v1"
REVERSAL_MODEL_ID = "ict_long_reversal_xgb_v1"
LONG_CONTINUATION_MODEL_ID = "ict_long_continuation_xgb_v1"
SHORT_META_MODEL_ID = "ict_short_meta_xgb_v1"
SHORT_REVERSAL_MODEL_ID = "ict_short_reversal_xgb_v1"
SHORT_CONTINUATION_MODEL_ID = "ict_short_continuation_xgb_v1"
EXPECTED_MODEL_IDS_BY_DIRECTION = {
    "long": frozenset(
        {
            META_MODEL_ID,
            REVERSAL_MODEL_ID,
            LONG_CONTINUATION_MODEL_ID,
        }
    ),
    "short": frozenset(
        {
            SHORT_META_MODEL_ID,
            SHORT_REVERSAL_MODEL_ID,
            SHORT_CONTINUATION_MODEL_ID,
        }
    ),
}
EXPECTED_MODEL_IDS = frozenset().union(*EXPECTED_MODEL_IDS_BY_DIRECTION.values())
EXPECTED_MODEL_STATUSES = {
    model_id: "active" if model_id == META_MODEL_ID else "candidate"
    for model_id in EXPECTED_MODEL_IDS
}
EXPECTED_REGISTRY_PATH = f"models/{BUNDLE_ID.replace('_20260813', '')}_registry_20260813.json"
EXPECTED_POLICY_BACKTEST_SUMMARY_PATH = (
    f"model_testing/reports/ict_paper_signal_bundles/{BUNDLE_ID}/run_summary.json"
)
DEFAULT_BUNDLE_DIR = REPO_ROOT / "ote_live" / "runtime_manifests" / BUNDLE_ID
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_HEARTBEAT_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_data"
    / "health"
    / "es_shared_live_signal_service_heartbeat.json"
)
DEFAULT_VALIDATION_FILENAME = "paper_signal_validation_summary.json"
DEFAULT_MAX_HEARTBEAT_AGE_SECONDS = 900.0
DEFAULT_CLEAN_HANDOFF_MAX_AGE_SECONDS = 120.0
EXPECTED_HEARTBEAT_IDENTITY = {
    "service_name": "es-shared-live-signal-service",
    "asset": "ES",
    "source_timeframe": "5m",
    "signal_timeframe": "5m",
}
_PUBLIC_ENV_KEYS = frozenset(
    {
        "ES_LIVE_ALL_MODELS_ACTIVE",
        "FRVP_LIVE_ALL_MODELS_ACTIVE",
        "IBKR_ALLOW_DELAYED_FALLBACK",
        "IBKR_ACCOUNT_MODE",
        "IBKR_ENABLED",
        "IBKR_PORT",
        "ICT_PAPER_SIGNAL_TRIAL_ENABLED",
    }
)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only readiness audit for the controlled ICT ES paper-signal bundle. "
            "The audit never connects to IBKR, starts services, or changes files."
        )
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help=f"Runtime-manifest directory for {BUNDLE_ID}.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Environment file inspected only for allowlisted boolean/mode facts.",
    )
    parser.add_argument(
        "--heartbeat-file",
        type=Path,
        default=DEFAULT_HEARTBEAT_PATH,
        help="Shared ES collector health snapshot.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=None,
        help=(
            "Saved bundle validation summary. Defaults to "
            f"<bundle-dir>/{DEFAULT_VALIDATION_FILENAME}."
        ),
    )
    parser.add_argument(
        "--max-heartbeat-age-seconds",
        type=float,
        default=DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
        help="Maximum age of a healthy collector snapshot.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Treat a disabled final launch switch as a separate activation step. "
            "When every substantive check passes, report "
            "status=ready_except_enable_switch and exit successfully."
        ),
    )
    parser.add_argument(
        "--allow-clean-stopped-handoff",
        action="store_true",
        help=(
            "Explicitly allow an identity-matched, healthy, non-stale, clean "
            "stopped ES collector snapshot no more than 120 seconds old. "
            "Normal audits continue to require service_status=running."
        ),
    )
    return parser


def audit_readiness(
    *,
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
    env_path: str | Path = DEFAULT_ENV_PATH,
    heartbeat_path: str | Path = DEFAULT_HEARTBEAT_PATH,
    validation_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
    max_heartbeat_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    preflight: bool = False,
    allow_clean_stopped_handoff: bool = False,
) -> dict[str, Any]:
    """Return a secret-free, fail-closed readiness result without mutating state."""

    checked_at = _ensure_utc(now or datetime.now(UTC))
    resolved_bundle_dir = _resolve_path(bundle_dir)
    resolved_env_path = _resolve_path(env_path)
    resolved_heartbeat_path = _resolve_path(heartbeat_path)
    resolved_validation_path = (
        _resolve_path(validation_path)
        if validation_path is not None
        else resolved_bundle_dir / DEFAULT_VALIDATION_FILENAME
    )
    reasons: list[dict[str, str]] = []
    activation_requirements: list[dict[str, str]] = []
    facts: dict[str, Any] = {
        "bundle_id": BUNDLE_ID,
        "bundle_dir": str(resolved_bundle_dir),
        "expected_active_model_id": META_MODEL_ID,
        "expected_shadow_challenger_model_id": REVERSAL_MODEL_ID,
        "expected_meta_global_threshold": 0.4,
        "expected_meta_abstain_enabled": False,
    }

    models = _load_bundle_models(resolved_bundle_dir, reasons)
    _audit_model_contract(models, reasons, facts)

    public_env = _load_public_environment(resolved_env_path, environ=environ)
    _audit_environment(
        public_env,
        reasons,
        activation_requirements,
        facts,
        preflight=preflight,
    )
    _audit_validation(resolved_validation_path, reasons, facts)
    _audit_heartbeat(
        resolved_heartbeat_path,
        reasons,
        facts,
        now=checked_at,
        max_age_seconds=max_heartbeat_age_seconds,
        allow_clean_stopped_handoff=allow_clean_stopped_handoff,
    )

    trial_enabled = facts.get("paper_signal_trial_enabled") is True
    preflight_ready = not reasons
    ready = preflight_ready and trial_enabled
    if ready:
        status = "ready_to_start"
    elif preflight and preflight_ready and activation_requirements:
        status = "ready_except_enable_switch"
    else:
        status = "blocked"
    return {
        "status": status,
        "ready_to_start": ready,
        "preflight_ready": preflight_ready,
        "preflight_mode": bool(preflight),
        "clean_stopped_handoff_mode": bool(allow_clean_stopped_handoff),
        "checked_at_utc": checked_at.isoformat(),
        "bundle_id": BUNDLE_ID,
        "blocking_reasons": reasons,
        "activation_requirements": activation_requirements,
        "facts": facts,
        "safety": {
            "read_only": True,
            "connects_to_ibkr": False,
            "starts_services": False,
            "secret_values_reported": False,
        },
    }


def _load_bundle_models(
    bundle_dir: Path,
    reasons: list[dict[str, str]],
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    if bundle_dir.name != BUNDLE_ID:
        _block(
            reasons,
            "bundle_directory_id_mismatch",
            f"Bundle directory must be named exactly {BUNDLE_ID}.",
        )

    manifest_files = {
        "long": bundle_dir / "live_runtime_manifest_long.json",
        "short": bundle_dir / "live_runtime_manifest_short.json",
    }
    for expected_direction, manifest_path in manifest_files.items():
        if not manifest_path.is_file():
            _block(
                reasons,
                "bundle_manifest_missing",
                f"Required bundle manifest is missing: {manifest_path.name}.",
            )
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            _block(
                reasons,
                "bundle_manifest_unreadable",
                f"Required bundle manifest is not readable valid JSON: {manifest_path.name}.",
            )
            continue
        if not isinstance(payload, dict):
            _block(
                reasons,
                "bundle_manifest_invalid",
                f"Bundle manifest must be a JSON object: {manifest_path.name}.",
            )
            continue
        _audit_manifest_contract(
            payload,
            expected_direction=expected_direction,
            manifest_name=manifest_path.name,
            reasons=reasons,
        )
        manifest_models = payload.get("models")
        if not isinstance(manifest_models, list):
            _block(
                reasons,
                "bundle_manifest_invalid",
                f"Bundle manifest has no models list: {manifest_path.name}.",
            )
            continue
        actual_direction_ids = [
            str(model.get("model_id") or "")
            for model in manifest_models
            if isinstance(model, dict)
        ]
        expected_direction_ids = EXPECTED_MODEL_IDS_BY_DIRECTION[expected_direction]
        if (
            len(actual_direction_ids) != len(expected_direction_ids)
            or frozenset(actual_direction_ids) != expected_direction_ids
        ):
            _block(
                reasons,
                "bundle_model_roster_mismatch",
                f"{manifest_path.name} must contain exactly the controlled "
                f"{expected_direction} roster: "
                + ", ".join(sorted(expected_direction_ids))
                + ".",
            )
        for model in manifest_models:
            if not isinstance(model, dict) or not str(model.get("model_id") or "").strip():
                _block(
                    reasons,
                    "bundle_model_invalid",
                    f"Bundle manifest contains a model record without a model_id: {manifest_path.name}.",
                )
                continue
            _audit_model_identity(
                model,
                expected_direction=expected_direction,
                manifest_name=manifest_path.name,
                reasons=reasons,
            )
            models.append(model)

    model_ids = [str(model["model_id"]) for model in models]
    duplicate_ids = sorted({model_id for model_id in model_ids if model_ids.count(model_id) > 1})
    if duplicate_ids:
        _block(
            reasons,
            "duplicate_model_ids",
            "Bundle contains duplicate model ids: " + ", ".join(duplicate_ids) + ".",
        )
    return models


def _audit_manifest_contract(
    payload: Mapping[str, Any],
    *,
    expected_direction: str,
    manifest_name: str,
    reasons: list[dict[str, str]],
) -> None:
    expected_values = {
        "direction": expected_direction,
        "asset": "ES",
        "timeframe": "5m",
        "registry_path": EXPECTED_REGISTRY_PATH,
        "policy_backtest_summary_path": EXPECTED_POLICY_BACKTEST_SUMMARY_PATH,
    }
    for field, expected in expected_values.items():
        actual = payload.get(field)
        if field.endswith("_path"):
            actual = _normalize_repo_reference(actual)
        if actual != expected:
            _block(
                reasons,
                "bundle_manifest_identity_mismatch",
                f"{manifest_name} must set {field}={expected}.",
            )


def _audit_model_identity(
    model: Mapping[str, Any],
    *,
    expected_direction: str,
    manifest_name: str,
    reasons: list[dict[str, str]],
) -> None:
    model_id = str(model.get("model_id") or "missing")
    expected_values = {
        "direction": expected_direction,
        "asset": "ES",
        "timeframe": "5m",
        "registry_path": EXPECTED_REGISTRY_PATH,
        "policy_backtest_summary_path": EXPECTED_POLICY_BACKTEST_SUMMARY_PATH,
    }
    for field, expected in expected_values.items():
        actual = model.get(field)
        if field.endswith("_path"):
            actual = _normalize_repo_reference(actual)
        if actual != expected:
            _block(
                reasons,
                "bundle_model_identity_mismatch",
                f"{model_id} in {manifest_name} must set {field}={expected}.",
            )


def _audit_model_contract(
    models: Sequence[dict[str, Any]],
    reasons: list[dict[str, str]],
    facts: dict[str, Any],
) -> None:
    statuses = {
        str(model.get("model_id")): str(model.get("status") or "unknown").lower()
        for model in models
    }
    active_model_ids = sorted(
        model_id for model_id, status in statuses.items() if status == "active"
    )
    facts["model_count"] = len(models)
    facts["model_ids"] = sorted(statuses)
    facts["expected_model_ids"] = sorted(EXPECTED_MODEL_IDS)
    facts["active_model_ids"] = active_model_ids
    facts["reversal_status"] = statuses.get(REVERSAL_MODEL_ID, "missing")

    if active_model_ids != [META_MODEL_ID]:
        actual = ", ".join(active_model_ids) if active_model_ids else "none"
        _block(
            reasons,
            "active_model_roster_mismatch",
            f"The sole active model must be {META_MODEL_ID}; actual active roster: {actual}.",
        )
    if len(models) != len(EXPECTED_MODEL_IDS) or frozenset(statuses) != EXPECTED_MODEL_IDS:
        _block(
            reasons,
            "bundle_model_roster_mismatch",
            "Bundle must contain exactly the six controlled ICT paper-signal models.",
        )
    status_mismatches = sorted(
        model_id
        for model_id, expected_status in EXPECTED_MODEL_STATUSES.items()
        if statuses.get(model_id) != expected_status
    )
    if status_mismatches:
        _block(
            reasons,
            "bundle_model_status_contract_mismatch",
            "The controlled bundle requires meta active and every other model candidate; "
            "mismatched models: " + ", ".join(status_mismatches) + ".",
        )
    if statuses.get(REVERSAL_MODEL_ID) != "candidate":
        _block(
            reasons,
            "reversal_status_mismatch",
            f"{REVERSAL_MODEL_ID} must remain candidate/shadow.",
        )

    meta_records = [model for model in models if model.get("model_id") == META_MODEL_ID]
    if len(meta_records) != 1:
        _block(
            reasons,
            "meta_model_record_mismatch",
            f"Bundle must contain exactly one {META_MODEL_ID} record.",
        )
        return

    policy = meta_records[0].get("live_policy")
    if not isinstance(policy, dict):
        _block(
            reasons,
            "meta_policy_missing",
            f"{META_MODEL_ID} has no embedded live policy.",
        )
        return

    facts["meta_policy_status"] = str(policy.get("policy_status") or "missing")
    if policy.get("policy_status") != "complete":
        _block(
            reasons,
            "meta_policy_incomplete",
            f"{META_MODEL_ID} live policy must have policy_status=complete.",
        )

    thresholds = policy.get("thresholds")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    threshold = _optional_float(thresholds.get("global_threshold"))
    facts["meta_global_threshold"] = threshold
    facts["meta_regime_thresholds_are_null"] = thresholds.get("regime_thresholds") is None
    if threshold is None or not math.isclose(threshold, 0.4, rel_tol=0.0, abs_tol=1e-12):
        _block(
            reasons,
            "meta_threshold_mismatch",
            f"{META_MODEL_ID} must use the exact global threshold 0.40.",
        )
    if thresholds.get("regime_thresholds") is not None:
        _block(
            reasons,
            "meta_regime_thresholds_enabled",
            f"{META_MODEL_ID} must not package regime thresholds for this trial.",
        )

    abstain_policy = policy.get("abstain_policy")
    if not isinstance(abstain_policy, dict):
        facts["meta_abstain_enabled"] = None
        _block(
            reasons,
            "meta_abstain_policy_missing",
            f"{META_MODEL_ID} must package an explicitly disabled abstain policy.",
        )
    else:
        abstain_enabled = abstain_policy.get("enabled")
        facts["meta_abstain_enabled"] = abstain_enabled
        if abstain_enabled is not False:
            _block(
                reasons,
                "meta_abstain_enabled",
                f"{META_MODEL_ID} abstention must be explicitly disabled.",
            )
        if not _is_exact_no_abstain_policy(abstain_policy):
            _block(
                reasons,
                "meta_no_abstain_contract_mismatch",
                f"{META_MODEL_ID} disabled abstain policy must contain only false/zero/empty filters.",
            )

    lineage = policy.get("lineage")
    selected_policy_name = (
        lineage.get("selected_policy_name") if isinstance(lineage, dict) else None
    )
    facts["meta_selected_policy_name"] = selected_policy_name
    if selected_policy_name != "global_threshold":
        _block(
            reasons,
            "meta_policy_name_mismatch",
            f"{META_MODEL_ID} selected policy must be global_threshold.",
        )


def _audit_environment(
    values: Mapping[str, str],
    reasons: list[dict[str, str]],
    activation_requirements: list[dict[str, str]],
    facts: dict[str, Any],
    *,
    preflight: bool,
) -> None:
    es_value = _nonempty(values.get("ES_LIVE_ALL_MODELS_ACTIVE"))
    fallback_value = _nonempty(values.get("FRVP_LIVE_ALL_MODELS_ACTIVE"))
    effective_value = es_value if es_value is not None else fallback_value
    try:
        all_models_active = _parse_bool(effective_value, default=False)
    except ValueError:
        all_models_active = None
        _block(
            reasons,
            "all_models_active_invalid",
            "Effective ES_LIVE_ALL_MODELS_ACTIVE is not a valid boolean.",
        )
    facts["effective_all_models_active"] = all_models_active
    if all_models_active is True:
        _block(
            reasons,
            "all_models_active_enabled",
            "ES_LIVE_ALL_MODELS_ACTIVE must be false so candidate models remain shadow-only.",
        )

    ibkr_enabled_value = _nonempty(values.get("IBKR_ENABLED"))
    try:
        ibkr_enabled = _parse_bool(ibkr_enabled_value, default=False)
    except ValueError:
        ibkr_enabled = None
        _block(
            reasons,
            "ibkr_enabled_invalid",
            "IBKR_ENABLED is not a valid boolean.",
        )
    facts["ibkr_enabled"] = ibkr_enabled
    if ibkr_enabled is False:
        _block(
            reasons,
            "ibkr_disabled",
            "IBKR_ENABLED must be true for the controlled paper-signal trial.",
        )

    account_mode = str(values.get("IBKR_ACCOUNT_MODE") or "paper").strip().lower()
    facts["ibkr_account_mode"] = account_mode
    if account_mode != "paper":
        _block(
            reasons,
            "ibkr_not_paper_mode",
            "IBKR_ACCOUNT_MODE must be paper before this trial can start.",
        )

    port_text = str(values.get("IBKR_PORT") or "4002").strip()
    try:
        ibkr_port = int(port_text)
    except ValueError:
        ibkr_port = None
    facts["ibkr_port"] = ibkr_port
    if ibkr_port not in {4002, 7497}:
        _block(
            reasons,
            "ibkr_not_paper_endpoint",
            "IBKR_PORT must be 4002 (IB Gateway paper) or 7497 (TWS paper).",
        )

    delayed_fallback_value = _nonempty(values.get("IBKR_ALLOW_DELAYED_FALLBACK"))
    try:
        delayed_fallback = _parse_bool(delayed_fallback_value, default=False)
    except ValueError:
        delayed_fallback = None
        _block(
            reasons,
            "ibkr_delayed_fallback_invalid",
            "IBKR_ALLOW_DELAYED_FALLBACK is not a valid boolean.",
        )
    facts["ibkr_allow_delayed_fallback"] = delayed_fallback
    if delayed_fallback is True:
        _block(
            reasons,
            "ibkr_delayed_fallback_enabled",
            "IBKR_ALLOW_DELAYED_FALLBACK must be false for this confirmation trial.",
        )

    trial_enabled_value = _nonempty(values.get("ICT_PAPER_SIGNAL_TRIAL_ENABLED"))
    try:
        trial_enabled = _parse_bool(trial_enabled_value, default=False)
    except ValueError:
        trial_enabled = None
        _block(
            reasons,
            "trial_enable_switch_invalid",
            "ICT_PAPER_SIGNAL_TRIAL_ENABLED is not a valid boolean.",
        )
    facts["paper_signal_trial_enabled"] = trial_enabled
    if trial_enabled is False and preflight:
        activation_requirements.append(
            {
                "code": "trial_enable_switch_disabled",
                "message": (
                    "Set ICT_PAPER_SIGNAL_TRIAL_ENABLED=true only for the final "
                    "readiness-bound collector launch."
                ),
            }
        )
    elif trial_enabled is not True:
        _block(
            reasons,
            "trial_enable_switch_disabled",
            "ICT_PAPER_SIGNAL_TRIAL_ENABLED must remain false until all other blockers are resolved, then be set true immediately before launch.",
        )


def _audit_validation(
    validation_path: Path,
    reasons: list[dict[str, str]],
    facts: dict[str, Any],
) -> None:
    if not validation_path.is_file():
        facts["validation"] = {"present": False}
        _block(
            reasons,
            "validation_summary_missing",
            "The saved paper-signal validation summary is missing.",
        )
        return
    try:
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        facts["validation"] = {"present": True, "valid_json": False}
        _block(
            reasons,
            "validation_summary_unreadable",
            "The saved paper-signal validation summary is not readable valid JSON.",
        )
        return
    if not isinstance(payload, dict) or payload.get("bundle_id") != BUNDLE_ID:
        facts["validation"] = {"present": True, "valid_json": True}
        _block(
            reasons,
            "validation_summary_bundle_mismatch",
            "The saved validation summary does not identify the expected bundle.",
        )
        return

    prediction_status = _nested_status(payload, "artifact_prediction_parity")
    feature_status = _nested_status(payload, "historical_feature_parity")
    dead_input_status = _nested_status(payload, "active_meta_dead_input_audit")
    policy_status = _nested_status(payload, "runtime_policy_regressions")
    ledger_status = _nested_status(payload, "paper_signal_ledger")
    start_authorized = payload.get("trial_start_authorized") is True
    facts["validation"] = {
        "present": True,
        "valid_json": True,
        "artifact_prediction_parity": prediction_status,
        "historical_feature_parity": feature_status,
        "active_meta_dead_input_audit": dead_input_status,
        "runtime_policy_regressions": policy_status,
        "paper_signal_ledger": ledger_status,
        "trial_start_authorized": start_authorized,
    }
    if prediction_status != "pass":
        _block(
            reasons,
            "artifact_prediction_parity_not_passed",
            "Saved model-artifact prediction parity must pass.",
        )
    accepted_feature_statuses = {"pass", "pass_with_documented_dead_inputs"}
    if feature_status not in accepted_feature_statuses:
        _block(
            reasons,
            "historical_feature_parity_not_passed",
            "Active-model historical feature/prediction parity must pass for the exact selected-feature contract.",
        )
    if feature_status == "pass_with_documented_dead_inputs" and dead_input_status != "pass":
        _block(
            reasons,
            "active_meta_dead_input_audit_not_passed",
            "Documented dead-input parity requires a passing zero-split and prediction-equivalence audit for active long-meta.",
        )
    if policy_status != "pass":
        _block(
            reasons,
            "runtime_policy_regressions_not_passed",
            "Runtime policy regression checks must pass.",
        )
    if ledger_status != "ready":
        _block(
            reasons,
            "paper_signal_ledger_not_ready",
            "A dedicated paper-signal confirmation ledger must be ready before the four-week clock starts.",
        )
    if not start_authorized:
        _block(
            reasons,
            "trial_start_not_authorized",
            "The saved validation summary does not authorize trial start.",
        )


def _audit_heartbeat(
    heartbeat_path: Path,
    reasons: list[dict[str, str]],
    facts: dict[str, Any],
    *,
    now: datetime,
    max_age_seconds: float,
    allow_clean_stopped_handoff: bool,
) -> None:
    if not heartbeat_path.is_file():
        _block(
            reasons,
            "heartbeat_missing",
            "Shared ES collector heartbeat is missing.",
        )
        facts["heartbeat"] = {"present": False}
        return
    try:
        payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _block(
            reasons,
            "heartbeat_unreadable",
            "Shared ES collector heartbeat is not readable valid JSON.",
        )
        facts["heartbeat"] = {"present": True, "valid_json": False}
        return
    if not isinstance(payload, dict):
        _block(
            reasons,
            "heartbeat_invalid",
            "Shared ES collector heartbeat must be a JSON object.",
        )
        facts["heartbeat"] = {"present": True, "valid_json": True}
        return

    generated_at = _optional_datetime(payload.get("generated_at_utc"))
    age_seconds = (now - generated_at).total_seconds() if generated_at is not None else None
    health_state = str(payload.get("health_state") or "missing").lower()
    service_status = str(payload.get("service_status") or "missing").lower()
    heartbeat_is_stale = payload.get("heartbeat_is_stale")
    heartbeat_identity = {
        key: str(payload.get(key) or "")
        for key in EXPECTED_HEARTBEAT_IDENTITY
    }
    identity_matches = heartbeat_identity == EXPECTED_HEARTBEAT_IDENTITY
    clean_handoff_age_valid = (
        age_seconds is not None
        and age_seconds >= -60.0
        and age_seconds <= DEFAULT_CLEAN_HANDOFF_MAX_AGE_SECONDS
    )
    running_age_valid = (
        age_seconds is not None
        and age_seconds >= -60.0
        and age_seconds <= float(max_age_seconds)
    )
    running_heartbeat_accepted = (
        service_status == "running"
        and health_state == "healthy"
        and heartbeat_is_stale is False
        and identity_matches
        and running_age_valid
    )
    clean_stopped_handoff_accepted = (
        bool(allow_clean_stopped_handoff)
        and service_status == "stopped"
        and health_state == "healthy"
        and heartbeat_is_stale is False
        and identity_matches
        and clean_handoff_age_valid
    )
    heartbeat_acceptance = (
        "running"
        if running_heartbeat_accepted
        else "clean_stopped_handoff"
        if clean_stopped_handoff_accepted
        else None
    )
    facts["heartbeat"] = {
        "present": True,
        "valid_json": True,
        "generated_at_utc": generated_at.isoformat() if generated_at is not None else None,
        "age_seconds": age_seconds,
        "max_age_seconds": float(max_age_seconds),
        "health_state": health_state,
        "service_status": service_status,
        "source_is_stale": heartbeat_is_stale,
        "identity": heartbeat_identity,
        "identity_matches": identity_matches,
        "clean_handoff_allowed": bool(allow_clean_stopped_handoff),
        "clean_handoff_max_age_seconds": DEFAULT_CLEAN_HANDOFF_MAX_AGE_SECONDS,
        "running_heartbeat_accepted": running_heartbeat_accepted,
        "clean_stopped_handoff_accepted": clean_stopped_handoff_accepted,
        "acceptance": heartbeat_acceptance,
    }

    if not identity_matches:
        _block(
            reasons,
            "heartbeat_identity_mismatch",
            "Shared ES collector heartbeat identity must match the controlled ES 5m service.",
        )

    if generated_at is None:
        _block(
            reasons,
            "heartbeat_timestamp_invalid",
            "Shared ES collector heartbeat has no valid timezone-aware generated_at_utc.",
        )
    elif age_seconds is not None and age_seconds < -60.0:
        _block(
            reasons,
            "heartbeat_from_future",
            "Shared ES collector heartbeat timestamp is unexpectedly ahead of the audit clock.",
        )
    elif (
        service_status == "stopped"
        and allow_clean_stopped_handoff
        and age_seconds is not None
        and age_seconds > DEFAULT_CLEAN_HANDOFF_MAX_AGE_SECONDS
    ):
        _block(
            reasons,
            "clean_handoff_too_old",
            "Clean stopped collector handoff is older than the 120-second launch window.",
        )
    elif age_seconds is not None and age_seconds > float(max_age_seconds):
        _block(
            reasons,
            "heartbeat_too_old",
            "Shared ES collector heartbeat is older than the allowed readiness window.",
        )
    if health_state != "healthy":
        _block(
            reasons,
            "heartbeat_unhealthy",
            "Shared ES collector health_state must be healthy.",
        )
    if service_status != "running" and not clean_stopped_handoff_accepted:
        _block(
            reasons,
            "collector_not_running",
            "Shared ES collector service_status must be running.",
        )
    if heartbeat_is_stale is not False:
        _block(
            reasons,
            "collector_source_stale",
            "Shared ES collector must explicitly report heartbeat_is_stale=false.",
        )


def _load_public_environment(
    env_path: Path,
    *,
    environ: Mapping[str, str] | None,
) -> dict[str, str]:
    values: dict[str, str] = {}
    if env_path.is_file():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            lines = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].lstrip()
            key, separator, raw_value = line.partition("=")
            key = key.strip()
            if not separator or key not in _PUBLIC_ENV_KEYS:
                continue
            values[key] = _strip_env_quotes(raw_value.strip())

    process_environment = os.environ if environ is None else environ
    for key in _PUBLIC_ENV_KEYS:
        if key in process_environment:
            values[key] = str(process_environment[key])
    return values


def _is_exact_no_abstain_policy(policy: Mapping[str, Any]) -> bool:
    false_fields = ("enabled", "abstain_high_stress", "abstain_off_hours")
    empty_fields = (
        "abstain_session_regimes",
        "abstain_composite_regimes",
        "abstain_composite_session_pairs",
        "abstain_composite_stress_pairs",
    )
    return (
        all(policy.get(field) is False for field in false_fields)
        and _optional_float(policy.get("cooldown_bars")) == 0.0
        and _optional_float(policy.get("minimum_expected_move_to_spread")) == 0.0
        and all(policy.get(field) == [] for field in empty_fields)
        and policy.get("minimum_probability_quantile") is None
    )


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError("invalid boolean")


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_status(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        return "missing"
    return str(value.get("status") or "missing").strip().lower()


def _optional_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(UTC)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _normalize_repo_reference(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _block(reasons: list[dict[str, str]], code: str, message: str) -> None:
    reasons.append({"code": code, "message": message})


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_readiness(
        bundle_dir=args.bundle_dir,
        env_path=args.env_file,
        heartbeat_path=args.heartbeat_file,
        validation_path=args.validation_file,
        max_heartbeat_age_seconds=args.max_heartbeat_age_seconds,
        preflight=args.preflight,
        allow_clean_stopped_handoff=args.allow_clean_stopped_handoff,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"ready_to_start", "ready_except_enable_switch"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
