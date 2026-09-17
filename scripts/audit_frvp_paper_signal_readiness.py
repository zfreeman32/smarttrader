from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_ict_paper_signal_readiness import (
    _audit_heartbeat,
    _ensure_utc,
    _resolve_path,
)


BUNDLE_ID = "frvp_es_paper_signal_20260816"
CONTINUATION_MODEL_ID = "frvp_long_continuation_xgb_v1"
REVERSAL_MODEL_ID = "frvp_long_reversal_xgb_v1"
LONG_META_MODEL_ID = "frvp_long_meta_xgb_v1"
SHORT_CONTINUATION_MODEL_ID = "frvp_short_continuation_tcn_v1"
SHORT_META_MODEL_ID = "frvp_short_meta_xgb_v1"
SHORT_REVERSAL_MODEL_ID = "frvp_short_reversal_xgb_v1"
EXPECTED_MODEL_IDS_BY_DIRECTION = {
    "long": frozenset({CONTINUATION_MODEL_ID, REVERSAL_MODEL_ID, LONG_META_MODEL_ID}),
    "short": frozenset({SHORT_CONTINUATION_MODEL_ID, SHORT_META_MODEL_ID}),
}
EXPECTED_MODEL_IDS = frozenset().union(*EXPECTED_MODEL_IDS_BY_DIRECTION.values())
EXPECTED_REGISTRY_MODEL_IDS = EXPECTED_MODEL_IDS | {SHORT_REVERSAL_MODEL_ID}
EXPECTED_MODEL_STATUSES = {
    CONTINUATION_MODEL_ID: "candidate",
    REVERSAL_MODEL_ID: "active",
    LONG_META_MODEL_ID: "candidate",
    SHORT_CONTINUATION_MODEL_ID: "candidate",
    SHORT_META_MODEL_ID: "candidate",
    SHORT_REVERSAL_MODEL_ID: "deprecated",
}
EXPECTED_REGISTRY_PATH = "models/frvp_es_paper_signal_registry_20260816.json"
EXPECTED_BACKTEST_SUMMARY_PATH = (
    f"model_testing/reports/frvp_paper_signal_bundles/{BUNDLE_ID}/run_summary.json"
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
EXPECTED_COSTS = {
    "fixed_slippage_pips_per_trade": 0.25,
    "commission_pips_per_trade": 0.40,
    "session_spread_pips": {
        "overlap": 1.0,
        "london": 1.0,
        "new_york": 1.0,
        "asia": 1.5,
        "off_hours": 2.0,
    },
}
ACTIVE_ARTIFACT_HASH_REFERENCE_KEYS = (
    "model_file",
    "scaler_file",
    "calibrator_file",
    "model_config_file",
    "training_summary_file",
)
EXPECTED_FILTERS = {
    CONTINUATION_MODEL_ID: {
        "global_threshold": 0.70,
        "preset": "frvp_long_continuation_xgb_overlap_composite_prune_v3",
        "sessions": {"overlap"},
        "composites": {"strong_down_medium", "strong_up_high"},
        "pairs": {
            ("strong_up_medium", "london"),
            ("strong_up_high", "london"),
            ("strong_down_medium", "london"),
            ("ranging_low", "london"),
            ("ranging_medium", "london"),
            ("ranging_medium", "new_york"),
            ("strong_up_low", "asia"),
        },
    },
    REVERSAL_MODEL_ID: {
        "global_threshold": 0.60,
        "preset": "frvp_long_reversal_xgb_recent2y_concentration_sdh_overlap_v1",
        "sessions": set(),
        "composites": {"ranging_high", "strong_up_low", "strong_up_medium"},
        "pairs": {
            ("strong_up_high", "new_york"),
            ("strong_down_high", "new_york"),
            ("ranging_low", "london"),
            ("strong_down_low", "asia"),
            ("ranging_medium", "asia"),
            ("ranging_medium", "london"),
            ("strong_down_low", "london"),
            ("strong_down_medium", "asia"),
            ("ranging_low", "asia"),
            ("ranging_medium", "new_york"),
            ("strong_down_high", "overlap"),
        },
    },
}
_PUBLIC_ENV_KEYS = frozenset(
    {
        "ES_LIVE_ALL_MODELS_ACTIVE",
        "ES_LIVE_ENABLE_SIGNAL_RUNTIME",
        "FRVP_LIVE_ASSET",
        "FRVP_LIVE_ALL_MODELS_ACTIVE",
        "FRVP_LIVE_DATA_SUPPLIER",
        "FRVP_LIVE_SOURCE_TIMEFRAME",
        "FRVP_PAPER_SIGNAL_TRIAL_ENABLED",
        "IBKR_ALLOW_DELAYED_FALLBACK",
        "IBKR_ACCOUNT_MODE",
        "IBKR_ENABLED",
        "IBKR_ES_BAR_SIZE",
        "IBKR_ES_CURRENCY",
        "IBKR_ES_EXCHANGE",
        "IBKR_ES_MULTIPLIER",
        "IBKR_ES_SECURITY_TYPE",
        "IBKR_ES_SYMBOL",
        "IBKR_ES_TRADING_CLASS",
        "IBKR_ES_USE_RTH",
        "IBKR_KEEP_UP_TO_DATE",
        "IBKR_MARKET_DATA_TYPE",
        "IBKR_PORT",
        "IBKR_WHAT_TO_SHOW",
    }
)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only readiness audit for the controlled FRVP ES paper-signal bundle. "
            "It never connects to IBKR, starts services, or changes files."
        )
    )
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--heartbeat-file", type=Path, default=DEFAULT_HEARTBEAT_PATH)
    parser.add_argument("--validation-file", type=Path, default=None)
    parser.add_argument(
        "--max-heartbeat-age-seconds",
        type=float,
        default=DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Report ready_except_enable_switch when only the final switch remains.",
    )
    parser.add_argument(
        "--allow-clean-stopped-handoff",
        action="store_true",
        help="Accept a healthy identity-matched clean stop no more than 120 seconds old.",
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
        "expected_active_model_ids": [REVERSAL_MODEL_ID],
        "broker_order_submission_authorized": False,
    }

    models = _load_bundle_models(resolved_bundle_dir, reasons, facts)
    _audit_registry_and_models(models, resolved_bundle_dir, reasons, facts)
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
    _audit_heartbeat_feed_contract(
        resolved_heartbeat_path,
        reasons,
        facts,
        allow_clean_stopped_handoff=allow_clean_stopped_handoff,
    )

    trial_enabled = facts.get("paper_signal_trial_enabled") is True
    substantive_ready = not reasons
    ready = substantive_ready and trial_enabled
    if ready:
        status = "ready_to_start"
    elif preflight and substantive_ready and activation_requirements:
        status = "ready_except_enable_switch"
    else:
        status = "blocked"
    return {
        "status": status,
        "ready_to_start": ready,
        "preflight_ready": substantive_ready,
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
            "broker_orders_authorized": False,
        },
    }


def _load_bundle_models(
    bundle_dir: Path,
    reasons: list[dict[str, str]],
    facts: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    directions: dict[str, list[str]] = {}
    for direction in ("long", "short"):
        path = bundle_dir / f"live_runtime_manifest_{direction}.json"
        if not path.is_file():
            _block(reasons, "direction_manifest_missing", f"Missing {direction} manifest.")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            _block(reasons, "direction_manifest_invalid", f"Invalid {direction} manifest JSON.")
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            _block(reasons, "direction_manifest_invalid", f"Malformed {direction} manifest.")
            continue
        if str(payload.get("direction")) != direction:
            _block(reasons, "direction_manifest_identity_mismatch", f"{direction} direction mismatch.")
        if str(payload.get("asset")) != "ES" or str(payload.get("timeframe")) != "5m":
            _block(reasons, "direction_manifest_market_mismatch", f"{direction} must be ES/5m.")
        if _normalize_reference(payload.get("registry_path")) != EXPECTED_REGISTRY_PATH:
            _block(reasons, "registry_reference_mismatch", f"{direction} registry reference is not exact.")
        if _normalize_reference(payload.get("policy_backtest_summary_path")) != EXPECTED_BACKTEST_SUMMARY_PATH:
            _block(reasons, "backtest_reference_mismatch", f"{direction} backtest reference is not exact.")
        ids: list[str] = []
        for raw_model in payload["models"]:
            if not isinstance(raw_model, dict):
                _block(reasons, "model_manifest_invalid", f"{direction} contains a malformed model.")
                continue
            model_id = str(raw_model.get("model_id") or "")
            if not model_id or model_id in models:
                _block(reasons, "model_roster_duplicate", "Model IDs must be present and unique.")
                continue
            models[model_id] = raw_model
            ids.append(model_id)
            nested_path = bundle_dir / model_id / "live_runtime_manifest.json"
            policy_path = bundle_dir / model_id / "live_policy.json"
            if not nested_path.is_file() or not policy_path.is_file():
                _block(reasons, "nested_contract_missing", f"Nested contract missing for {model_id}.")
                continue
            try:
                nested = json.loads(nested_path.read_text(encoding="utf-8"))
                packaged_policy = json.loads(policy_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                _block(reasons, "nested_contract_invalid", f"Nested contract invalid for {model_id}.")
                continue
            if nested != raw_model or packaged_policy != raw_model.get("live_policy"):
                _block(reasons, "nested_contract_mismatch", f"Nested contract differs for {model_id}.")
        directions[direction] = sorted(ids)
    facts["model_ids_by_direction"] = directions
    facts["model_ids"] = sorted(models)
    return models


def _audit_registry_and_models(
    models: Mapping[str, Mapping[str, Any]],
    bundle_dir: Path,
    reasons: list[dict[str, str]],
    facts: dict[str, Any],
) -> None:
    actual_ids = frozenset(models)
    if actual_ids != EXPECTED_MODEL_IDS:
        _block(reasons, "model_roster_mismatch", "Runtime bundle must contain the exact five loadable FRVP models.")
    for direction, expected_ids in EXPECTED_MODEL_IDS_BY_DIRECTION.items():
        actual_direction_ids = frozenset(
            model_id
            for model_id, model in models.items()
            if str(model.get("direction")) == direction
        )
        if actual_direction_ids != expected_ids:
            _block(reasons, "direction_roster_mismatch", f"{direction} FRVP roster is not exact.")

    statuses = {model_id: str(model.get("status") or "") for model_id, model in models.items()}
    facts["model_statuses"] = statuses
    if statuses != {key: value for key, value in EXPECTED_MODEL_STATUSES.items() if key in models}:
        _block(reasons, "model_status_mismatch", "FRVP active/candidate/deprecated statuses are not exact.")

    registry_path = REPO_ROOT / EXPECTED_REGISTRY_PATH
    if not registry_path.is_file():
        _block(reasons, "registry_missing", "Exact FRVP paper-signal registry is missing.")
    else:
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry_records = {
                str(item.get("model_id")): item
                for item in registry.get("models", [])
                if isinstance(item, dict)
            }
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            registry_records = {}
            _block(reasons, "registry_invalid", "FRVP paper-signal registry is invalid.")
        registry_statuses = {
            model_id: str(record.get("status") or "")
            for model_id, record in registry_records.items()
        }
        if (
            frozenset(registry_records) != EXPECTED_REGISTRY_MODEL_IDS
            or registry_statuses != EXPECTED_MODEL_STATUSES
        ):
            _block(reasons, "registry_contract_mismatch", "Registry roster/status contract is not exact.")

    for model_id, expected in EXPECTED_FILTERS.items():
        model = models.get(model_id)
        if model is None:
            continue
        if str(model.get("asset")) != "ES" or str(model.get("timeframe")) != "5m":
            _block(reasons, "active_market_mismatch", f"{model_id} must be ES/5m.")
        if str(model.get("backend")) != "xgboost" or str(model.get("direction")) != "long":
            _block(reasons, "active_model_identity_mismatch", f"{model_id} identity is not exact.")
        if _normalize_reference(model.get("registry_path")) != EXPECTED_REGISTRY_PATH:
            _block(reasons, "active_registry_reference_mismatch", f"{model_id} registry is not exact.")
        horizon = (
            model.get("context_requirements", {})
            .get("label_horizon_assumptions", {})
            .get("label_max_holding_bars")
        )
        if horizon != 120:
            _block(reasons, "active_horizon_mismatch", f"{model_id} must retain a 120-bar horizon.")
        _audit_active_policy(model_id, model.get("live_policy"), expected, reasons)
        refs = model.get("artifact_references")
        refs = refs if isinstance(refs, dict) else {}
        required_artifact_keys = (
            list(ACTIVE_ARTIFACT_HASH_REFERENCE_KEYS)
            if model_id == REVERSAL_MODEL_ID
            else ["model_file", "scaler_file", "model_config_file"]
        )
        if (
            model_id != REVERSAL_MODEL_ID
            and str(model.get("calibration_method") or "none").lower() != "none"
        ):
            required_artifact_keys.append("calibrator_file")
        for key in required_artifact_keys:
            value = refs.get(key)
            if not value or not _resolve_path(value).is_file():
                _block(reasons, "active_artifact_missing", f"{model_id} is missing {key}.")
        if model_id == REVERSAL_MODEL_ID:
            _audit_active_artifact_hashes(refs, reasons, facts)


def _audit_active_artifact_hashes(
    references: Mapping[str, Any],
    reasons: list[dict[str, str]],
    facts: dict[str, Any],
) -> None:
    raw_hashes = references.get("content_sha256")
    content_hashes = raw_hashes if isinstance(raw_hashes, dict) else {}
    expected_keys = set(ACTIVE_ARTIFACT_HASH_REFERENCE_KEYS)
    hash_contract_valid = set(content_hashes) == expected_keys
    hashes_match = hash_contract_valid
    if not hash_contract_valid:
        _block(
            reasons,
            "active_artifact_hash_contract_mismatch",
            "Active reversal must pin the exact runtime artifact SHA-256 set.",
        )

    checked_keys: list[str] = []
    for reference_key in ACTIVE_ARTIFACT_HASH_REFERENCE_KEYS:
        reference = references.get(reference_key)
        expected_hash = str(content_hashes.get(reference_key) or "").strip().lower()
        if (
            not reference
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            hash_contract_valid = False
            hashes_match = False
            continue
        artifact_path = _resolve_path(reference)
        if not artifact_path.is_file():
            hashes_match = False
            continue
        checked_keys.append(reference_key)
        actual_hash = _file_sha256(artifact_path)
        if actual_hash != expected_hash:
            hashes_match = False
            _block(
                reasons,
                "active_artifact_hash_mismatch",
                f"Active reversal artifact bytes changed for {reference_key}.",
            )
    if not hash_contract_valid and set(content_hashes) == expected_keys:
        _block(
            reasons,
            "active_artifact_hash_contract_mismatch",
            "Active reversal artifact SHA-256 pins must be exact lowercase hex digests.",
        )
    facts["active_artifact_integrity"] = {
        "model_id": REVERSAL_MODEL_ID,
        "algorithm": "sha256",
        "required_reference_keys": list(ACTIVE_ARTIFACT_HASH_REFERENCE_KEYS),
        "checked_reference_keys": checked_keys,
        "all_match": hashes_match,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_active_policy(
    model_id: str,
    policy: Any,
    expected: Mapping[str, Any],
    reasons: list[dict[str, str]],
) -> None:
    if not isinstance(policy, dict):
        _block(reasons, "active_policy_missing", f"{model_id} live policy is missing.")
        return
    thresholds = policy.get("thresholds")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    if not math.isclose(float(thresholds.get("global_threshold", -1.0)), float(expected["global_threshold"])):
        _block(reasons, "active_threshold_mismatch", f"{model_id} global threshold is not exact.")
    if thresholds.get("regime_thresholds") is not None:
        _block(reasons, "active_regime_threshold_mismatch", f"{model_id} must not use regime thresholds.")

    abstain = policy.get("abstain_policy")
    abstain = abstain if isinstance(abstain, dict) else {}
    exact_flags = (
        abstain.get("enabled") is True
        and abstain.get("abstain_high_stress") is True
        and abstain.get("abstain_off_hours") is True
        and abstain.get("cooldown_bars") == 4
        and math.isclose(float(abstain.get("minimum_expected_move_to_spread", -1.0)), 2.0)
        and abstain.get("minimum_probability_quantile") is None
        and set(map(str, abstain.get("abstain_session_regimes") or ())) == expected["sessions"]
        and set(map(str, abstain.get("abstain_composite_regimes") or ())) == expected["composites"]
        and _pair_set(abstain.get("abstain_composite_session_pairs")) == expected["pairs"]
        and not (abstain.get("abstain_composite_stress_pairs") or ())
    )
    if not exact_flags:
        _block(reasons, "active_abstain_policy_mismatch", f"{model_id} abstain filters are not exact.")

    costs = policy.get("cost_assumptions")
    costs = costs if isinstance(costs, dict) else {}
    schedule = costs.get("session_spread_pips")
    schedule = schedule if isinstance(schedule, dict) else {}
    exact_costs = (
        math.isclose(float(costs.get("fixed_slippage_pips_per_trade", -1.0)), 0.25)
        and math.isclose(float(costs.get("commission_pips_per_trade", -1.0)), 0.40)
        and {str(key): float(value) for key, value in schedule.items()}
        == EXPECTED_COSTS["session_spread_pips"]
        and str(costs.get("targeted_filter_preset")) == str(expected["preset"])
    )
    if not exact_costs:
        _block(reasons, "active_cost_policy_mismatch", f"{model_id} costs/preset are not exact.")


def _audit_environment(
    values: Mapping[str, str],
    reasons: list[dict[str, str]],
    activation_requirements: list[dict[str, str]],
    facts: dict[str, Any],
    *,
    preflight: bool,
) -> None:
    all_models_active = _parse_bool(
        values.get("ES_LIVE_ALL_MODELS_ACTIVE", values.get("FRVP_LIVE_ALL_MODELS_ACTIVE")),
        default=False,
    )
    trial_enabled = _parse_bool(values.get("FRVP_PAPER_SIGNAL_TRIAL_ENABLED"), default=False)
    ibkr_enabled = _parse_bool(values.get("IBKR_ENABLED"), default=False)
    delayed = _parse_bool(values.get("IBKR_ALLOW_DELAYED_FALLBACK"), default=False)
    signal_runtime_enabled = _parse_bool(
        values.get("ES_LIVE_ENABLE_SIGNAL_RUNTIME"), default=True
    )
    use_rth = _parse_bool(values.get("IBKR_ES_USE_RTH"), default=False)
    keep_up_to_date = _parse_bool(values.get("IBKR_KEEP_UP_TO_DATE"), default=True)
    data_supplier = str(values.get("FRVP_LIVE_DATA_SUPPLIER") or "IBKR").strip().upper()
    asset = str(values.get("FRVP_LIVE_ASSET") or "ES").strip().upper()
    source_timeframe = str(values.get("FRVP_LIVE_SOURCE_TIMEFRAME") or "5m").strip()
    market_data_type = str(values.get("IBKR_MARKET_DATA_TYPE") or "live").strip().lower()
    what_to_show = str(values.get("IBKR_WHAT_TO_SHOW") or "TRADES").strip().upper()
    bar_size = str(values.get("IBKR_ES_BAR_SIZE") or "5 mins").strip().lower()
    contract_values = {
        "symbol": str(values.get("IBKR_ES_SYMBOL") or "ES").strip().upper(),
        "security_type": str(values.get("IBKR_ES_SECURITY_TYPE") or "FUT").strip().upper(),
        "exchange": str(values.get("IBKR_ES_EXCHANGE") or "CME").strip().upper(),
        "currency": str(values.get("IBKR_ES_CURRENCY") or "USD").strip().upper(),
        "multiplier": str(values.get("IBKR_ES_MULTIPLIER") or "50").strip(),
        "trading_class": str(values.get("IBKR_ES_TRADING_CLASS") or "ES").strip().upper(),
    }
    account_mode = str(values.get("IBKR_ACCOUNT_MODE") or "paper").strip().lower()
    try:
        port = int(str(values.get("IBKR_PORT") or "4002"))
    except ValueError:
        port = 0
    facts.update(
        {
            "all_models_active": all_models_active,
            "paper_signal_trial_enabled": trial_enabled,
            "ibkr_enabled": ibkr_enabled,
            "ibkr_account_mode": account_mode or None,
            "ibkr_port": port or None,
            "ibkr_allow_delayed_fallback": delayed,
            "signal_runtime_enabled": signal_runtime_enabled,
            "data_supplier": data_supplier or None,
            "asset": asset or None,
            "source_timeframe": source_timeframe or None,
            "ibkr_market_data_type": market_data_type or None,
            "ibkr_what_to_show": what_to_show or None,
            "ibkr_use_rth": use_rth,
            "ibkr_bar_size": bar_size or None,
            "ibkr_keep_up_to_date": keep_up_to_date,
            "ibkr_contract": contract_values,
        }
    )
    if all_models_active:
        _block(reasons, "all_models_active_override", "ES_LIVE_ALL_MODELS_ACTIVE must be false.")
    if not ibkr_enabled:
        _block(reasons, "ibkr_disabled", "IBKR_ENABLED must be true.")
    if account_mode != "paper":
        _block(reasons, "ibkr_not_paper", "IBKR_ACCOUNT_MODE must be paper.")
    if port not in {4002, 7497}:
        _block(reasons, "ibkr_paper_port_invalid", "IBKR must use paper port 4002 or 7497.")
    if delayed:
        _block(reasons, "delayed_fallback_enabled", "Delayed fallback must be disabled.")
    if not signal_runtime_enabled:
        _block(reasons, "signal_runtime_disabled", "ES signal runtime must be enabled.")
    if data_supplier != "IBKR":
        _block(reasons, "data_supplier_mismatch", "FRVP data supplier must be IBKR.")
    if asset != "ES" or source_timeframe != "5m":
        _block(reasons, "market_contract_mismatch", "FRVP feed must be ES/5m.")
    if market_data_type not in {"live", "1"}:
        _block(reasons, "market_data_not_live", "IBKR market data type must be live.")
    if what_to_show != "TRADES":
        _block(reasons, "ibkr_bar_source_mismatch", "IBKR bars must use TRADES.")
    if use_rth:
        _block(reasons, "rth_only_enabled", "IBKR ES use_rth must be false.")
    if bar_size != "5 mins":
        _block(reasons, "ibkr_bar_size_mismatch", "IBKR bar size must be 5 mins.")
    if not keep_up_to_date:
        _block(reasons, "ibkr_streaming_disabled", "IBKR keep_up_to_date must be true.")
    if contract_values != {
        "symbol": "ES",
        "security_type": "FUT",
        "exchange": "CME",
        "currency": "USD",
        "multiplier": "50",
        "trading_class": "ES",
    }:
        _block(
            reasons,
            "ibkr_contract_mismatch",
            "IBKR contract must be ES/FUT/CME/USD/50/ES.",
        )
    if not trial_enabled:
        if preflight:
            activation_requirements.append(
                {
                    "code": "enable_trial_switch",
                    "message": "Set FRVP_PAPER_SIGNAL_TRIAL_ENABLED=true only for the final launch.",
                }
            )
        else:
            _block(reasons, "trial_enable_switch_disabled", "FRVP trial switch is disabled.")


def _audit_validation(
    path: Path,
    reasons: list[dict[str, str]],
    facts: dict[str, Any],
) -> None:
    if not path.is_file():
        _block(reasons, "validation_missing", "FRVP paper-signal validation summary is missing.")
        facts["validation"] = {"present": False}
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _block(reasons, "validation_invalid", "FRVP validation summary is invalid JSON.")
        facts["validation"] = {"present": True, "valid_json": False}
        return
    if not isinstance(payload, dict) or payload.get("bundle_id") != BUNDLE_ID:
        _block(reasons, "validation_bundle_mismatch", "Validation is not for the exact FRVP bundle.")
        return
    required = {
        "bundle_contract": "pass",
        "artifact_prediction_parity": "pass",
        "historical_feature_parity": "pass",
        "active_policy_parity": "pass",
        "runtime_policy_regressions": "pass",
        "paper_signal_ledger": "ready",
    }
    statuses = {
        key: str((payload.get(key) or {}).get("status") or "missing")
        if isinstance(payload.get(key), dict)
        else "missing"
        for key in required
    }
    facts["validation"] = {
        "present": True,
        "path": str(path),
        "statuses": statuses,
        "trial_start_authorized_after_readiness": (
            payload.get("trial_start_authorized_after_readiness") is True
        ),
        "trial_start_authorized_now": payload.get("trial_start_authorized_now") is True,
        "human_same_contract_signoff": payload.get("human_same_contract_signoff"),
    }
    for key, expected_status in required.items():
        if statuses[key] != expected_status:
            _block(reasons, f"{key}_not_ready", f"Validation {key} must report {expected_status}.")
    if payload.get("trial_start_authorized_after_readiness") is not True:
        _block(
            reasons,
            "trial_start_not_authorized",
            "FRVP decision has not authorized trial start after readiness passes.",
        )
    if payload.get("trial_start_authorized_now") is not False:
        _block(
            reasons,
            "validation_lifecycle_invalid",
            "Static validation must not claim that the readiness-bound trial is already start-authorized now.",
        )
    signoff = payload.get("human_same_contract_signoff")
    signoff = signoff if isinstance(signoff, dict) else {}
    if (
        signoff.get("status") != "deferred_for_controlled_paper_signal"
        or signoff.get("full_promotion_authorized") is not False
    ):
        _block(
            reasons,
            "human_signoff_disposition_missing",
            "Human signoff must be explicitly deferred for paper signals without authorizing full promotion.",
        )


def _audit_heartbeat_feed_contract(
    path: Path,
    reasons: list[dict[str, str]],
    facts: dict[str, Any],
    *,
    allow_clean_stopped_handoff: bool,
) -> None:
    """Require current-process feed identity without pinning a bootstrap bundle."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    runtime = payload.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    contract = runtime.get("collector_contract")
    contract = contract if isinstance(contract, dict) else {}
    feed = contract.get("feed")
    feed = feed if isinstance(feed, dict) else {}
    signal_runtime = contract.get("signal_runtime")
    signal_runtime = signal_runtime if isinstance(signal_runtime, dict) else {}
    es_contract = feed.get("contract")
    es_contract = es_contract if isinstance(es_contract, dict) else {}
    active_values = signal_runtime.get("active_model_ids")
    shadow_values = signal_runtime.get("shadow_model_ids")
    loaded_values = signal_runtime.get("loaded_model_ids")
    registry_values = signal_runtime.get("registry_paths")

    def _valid_identity_list(value: Any, *, allow_empty: bool) -> bool:
        return (
            isinstance(value, list)
            and (allow_empty or bool(value))
            and all(isinstance(item, str) and bool(item.strip()) for item in value)
            and len(value) == len(set(value))
        )

    active_values_valid = _valid_identity_list(active_values, allow_empty=True)
    shadow_values_valid = _valid_identity_list(shadow_values, allow_empty=True)
    loaded_values_valid = _valid_identity_list(loaded_values, allow_empty=False)
    registry_values_valid = _valid_identity_list(registry_values, allow_empty=False)
    active_model_ids = set(active_values) if active_values_valid else set()
    shadow_model_ids = set(shadow_values) if shadow_values_valid else set()
    loaded_model_ids = set(loaded_values) if loaded_values_valid else set()
    checks = {
        "schema_version": contract.get("schema_version") == 1,
        "data_supplier": str(feed.get("data_supplier") or "").upper() == "IBKR",
        "heartbeat_source": (
            str(payload.get("latest_heartbeat_source") or "")
            == "ibkr.historical.polling"
            and str(feed.get("heartbeat_source") or "")
            == "ibkr.historical.polling"
        ),
        "signal_runtime_enabled": signal_runtime.get("enabled") is True,
        "signal_runtime_loaded": (
            signal_runtime.get("loaded") is True
            and loaded_values_valid
            and active_values_valid
            and shadow_values_valid
            and registry_values_valid
            and not active_model_ids.intersection(shadow_model_ids)
            and loaded_model_ids == active_model_ids | shadow_model_ids
        ),
        "all_models_active": signal_runtime.get("all_models_active") is False,
        "market": (
            str(feed.get("asset") or "").upper() == "ES"
            and str(feed.get("source_timeframe") or "") == "5m"
        ),
        "ibkr_enabled": feed.get("ibkr_enabled") is True,
        "ibkr_account_mode": (
            str(feed.get("account_mode") or "").lower() == "paper"
        ),
        "market_data_type_requested": (
            str(feed.get("market_data_type_requested") or "").lower()
            in {"live", "1"}
        ),
        "market_data_type_received": (
            str(feed.get("market_data_type_received") or "").lower()
            == "live"
        ),
        "connection_state": (
            str(feed.get("connection_state") or "").lower()
            == "connected"
        ),
        "delayed_fallback": feed.get("allow_delayed_fallback") is False,
        "what_to_show": str(feed.get("what_to_show") or "").upper() == "TRADES",
        "use_rth": feed.get("use_rth") is False,
        "bar_size": str(feed.get("bar_size") or "").lower() == "5 mins",
        "keep_up_to_date": feed.get("keep_up_to_date") is True,
        "contract": (
            str(es_contract.get("symbol") or "").upper() == "ES"
            and str(es_contract.get("security_type") or "").upper() == "FUT"
            and str(es_contract.get("exchange") or "").upper() == "CME"
            and str(es_contract.get("currency") or "").upper() == "USD"
            and str(es_contract.get("multiplier") or "") == "50"
            and str(es_contract.get("trading_class") or "").upper() == "ES"
        ),
        "active_roster": (
            active_values_valid
            and active_model_ids.issubset({REVERSAL_MODEL_ID})
        ),
        "active_ledger": (
            not active_model_ids
            or (
                active_model_ids == {REVERSAL_MODEL_ID}
                and signal_runtime.get("frvp_paper_signal_ledger_ready") is True
            )
        ),
    }
    service_status = str(payload.get("service_status") or "").lower()
    if service_status == "stopped" and allow_clean_stopped_handoff:
        checks["clean_terminal_status"] = (
            str(runtime.get("terminal_status") or "").lower() == "completed"
        )
    matches = all(checks.values())
    heartbeat_facts = facts.get("heartbeat")
    if not isinstance(heartbeat_facts, dict):
        heartbeat_facts = {}
        facts["heartbeat"] = heartbeat_facts
    heartbeat_facts["feed_contract_matches"] = matches
    heartbeat_facts["feed_contract_checks"] = checks
    heartbeat_facts["collector_contract_schema_version"] = contract.get(
        "schema_version"
    )
    if not matches:
        _block(
            reasons,
            "heartbeat_feed_contract_mismatch",
            "Shared ES heartbeat does not prove the required current-process IBKR/live/TRADES/full-session signal-runtime contract.",
        )


def _load_public_environment(
    env_path: Path,
    *,
    environ: Mapping[str, str] | None,
) -> dict[str, str]:
    values: dict[str, str] = {}
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in _PUBLIC_ENV_KEYS:
                values[key] = _strip_env_quotes(value.strip())
    runtime_values = os.environ if environ is None else environ
    for key in _PUBLIC_ENV_KEYS:
        if key in runtime_values:
            values[key] = str(runtime_values[key])
    return values


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _pair_set(value: Any) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for item in value or ():
        if isinstance(item, (list, tuple)) and len(item) == 2:
            result.add((str(item[0]), str(item[1])))
    return result


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _normalize_reference(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def _block(reasons: list[dict[str, str]], code: str, message: str) -> None:
    if code not in {item["code"] for item in reasons}:
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
