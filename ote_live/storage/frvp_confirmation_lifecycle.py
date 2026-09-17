from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.frvp_paper_signal import (
    FRVP_PAPER_SIGNAL_BUNDLE_ID,
    FRVP_PAPER_SIGNAL_MODEL_IDS,
    FRVP_PAPER_SIGNAL_REGISTRY_PATH,
)


FRVP_CONFIRMATION_SCOPE = "frvp.paper_signal.confirmation"
FRVP_CONFIRMATION_STATE_KEY = FRVP_PAPER_SIGNAL_BUNDLE_ID
FRVP_CONFIRMATION_MINIMUM_DAYS = 28
FRVP_CONFIRMATION_MAX_HEARTBEAT_AGE_SECONDS = 120.0
FRVP_CONFIRMATION_SERVICE_NAME = "es-shared-live-signal-service"

FRVP_CONFIRMATION_LOADED_MODEL_IDS = frozenset(
    {
        "frvp_long_continuation_xgb_v1",
        "frvp_long_reversal_xgb_v1",
        "frvp_long_meta_xgb_v1",
        "frvp_short_continuation_tcn_v1",
        "frvp_short_meta_xgb_v1",
    }
)
FRVP_CONFIRMATION_ACTIVE_MODEL_IDS = FRVP_PAPER_SIGNAL_MODEL_IDS
FRVP_CONFIRMATION_SHADOW_MODEL_IDS = (
    FRVP_CONFIRMATION_LOADED_MODEL_IDS - FRVP_CONFIRMATION_ACTIVE_MODEL_IDS
)


@dataclass(frozen=True)
class FrvpConfirmationLifecycle:
    """Read-only view of the mutable FRVP confirmation clock."""

    started: bool
    newly_started: bool
    confirmation_start_utc: datetime | None
    earliest_minimum_completion_utc: datetime | None
    minimum_confirmation_days: int
    minimum_period_elapsed: bool
    broker_orders_authorized: bool
    refusal_reasons: tuple[str, ...] = ()


def record_frvp_confirmation_start(
    store: SQLiteLiveDataStore,
    *,
    heartbeat: Mapping[str, Any],
    bundle_id: str,
    first_healthy_final_heartbeat_confirmed: bool,
    recorded_at_utc: datetime | None = None,
) -> FrvpConfirmationLifecycle:
    """Persist the confirmation start once after the final heartbeat is written.

    The explicit confirmation flag is deliberately separate from heartbeat
    content.  A caller must affirm that this is the first final-process
    heartbeat, and the heartbeat must independently prove the frozen bundle,
    active roster, healthy running state, paper feed, and markout ledger.
    """

    observed_at = ensure_utc(recorded_at_utc or utc_now())
    existing = store.get_runtime_state(
        scope=FRVP_CONFIRMATION_SCOPE,
        state_key=FRVP_CONFIRMATION_STATE_KEY,
    )
    if existing is not None:
        return _lifecycle_from_payload(
            existing,
            now=observed_at,
            newly_started=False,
        )

    refusal_reasons: list[str] = []
    if first_healthy_final_heartbeat_confirmed is not True:
        refusal_reasons.append("explicit_first_healthy_final_heartbeat_required")
    heartbeat_start = _validate_final_heartbeat(
        heartbeat,
        bundle_id=bundle_id,
        observed_at=observed_at,
        refusal_reasons=refusal_reasons,
    )
    if refusal_reasons or heartbeat_start is None:
        return _not_started(tuple(refusal_reasons))

    earliest_completion = heartbeat_start + timedelta(
        days=FRVP_CONFIRMATION_MINIMUM_DAYS
    )
    payload = {
        "schema_version": 1,
        "state": "confirmation_running",
        "bundle_id": FRVP_PAPER_SIGNAL_BUNDLE_ID,
        "active_model_ids": sorted(FRVP_CONFIRMATION_ACTIVE_MODEL_IDS),
        "confirmation_start_utc": heartbeat_start.isoformat(),
        "earliest_minimum_completion_utc": earliest_completion.isoformat(),
        "minimum_confirmation_days": FRVP_CONFIRMATION_MINIMUM_DAYS,
        "broker_orders_authorized": False,
        "broker_order_submission_authorized": False,
        "execution_authorized": False,
        "start_evidence": {
            "explicit_first_healthy_final_heartbeat_confirmed": True,
            "heartbeat_generated_at_utc": heartbeat_start.isoformat(),
            "service_name": FRVP_CONFIRMATION_SERVICE_NAME,
            "service_status": "running",
            "health_state": "healthy",
            "heartbeat_is_stale": False,
            "registry_paths": [FRVP_PAPER_SIGNAL_REGISTRY_PATH],
            "loaded_model_ids": sorted(FRVP_CONFIRMATION_LOADED_MODEL_IDS),
            "active_model_ids": sorted(FRVP_CONFIRMATION_ACTIVE_MODEL_IDS),
            "frvp_paper_signal_ledger_ready": True,
        },
    }
    stored_at = observed_at.isoformat()
    cursor = store.connection.execute(
        """
        INSERT INTO runtime_state (
            scope, state_key, payload_json, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(scope, state_key) DO NOTHING
        """,
        (
            FRVP_CONFIRMATION_SCOPE,
            FRVP_CONFIRMATION_STATE_KEY,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            stored_at,
            stored_at,
        ),
    )
    store.connection.commit()

    persisted = store.get_runtime_state(
        scope=FRVP_CONFIRMATION_SCOPE,
        state_key=FRVP_CONFIRMATION_STATE_KEY,
    )
    if persisted is None:
        raise RuntimeError("Failed to persist the FRVP confirmation lifecycle state.")
    return _lifecycle_from_payload(
        persisted,
        now=observed_at,
        newly_started=cursor.rowcount == 1,
    )


def get_frvp_confirmation_lifecycle(
    store: SQLiteLiveDataStore,
    *,
    now: datetime | None = None,
) -> FrvpConfirmationLifecycle:
    """Return clock state without expanding paper-signal execution authority."""

    resolved_now = ensure_utc(now or utc_now())
    payload = store.get_runtime_state(
        scope=FRVP_CONFIRMATION_SCOPE,
        state_key=FRVP_CONFIRMATION_STATE_KEY,
    )
    if payload is None:
        return _not_started(())
    return _lifecycle_from_payload(payload, now=resolved_now, newly_started=False)


def _validate_final_heartbeat(
    heartbeat: Mapping[str, Any],
    *,
    bundle_id: str,
    observed_at: datetime,
    refusal_reasons: list[str],
) -> datetime | None:
    if str(bundle_id) != FRVP_PAPER_SIGNAL_BUNDLE_ID:
        _refuse(refusal_reasons, "bundle_id_mismatch")
    if str(heartbeat.get("service_name") or "") != FRVP_CONFIRMATION_SERVICE_NAME:
        _refuse(refusal_reasons, "service_identity_mismatch")
    if str(heartbeat.get("service_status") or "").lower() != "running":
        _refuse(refusal_reasons, "collector_not_running")
    if str(heartbeat.get("health_state") or "").lower() != "healthy":
        _refuse(refusal_reasons, "collector_unhealthy")
    if heartbeat.get("heartbeat_is_stale") is not False:
        _refuse(refusal_reasons, "collector_heartbeat_stale_or_unknown")
    if str(heartbeat.get("asset") or "").upper() != "ES":
        _refuse(refusal_reasons, "market_identity_mismatch")
    if str(heartbeat.get("source_timeframe") or "") != "5m":
        _refuse(refusal_reasons, "market_identity_mismatch")
    if str(heartbeat.get("signal_timeframe") or "") != "5m":
        _refuse(refusal_reasons, "market_identity_mismatch")
    if str(heartbeat.get("latest_heartbeat_source") or "") != "ibkr.historical.polling":
        _refuse(refusal_reasons, "heartbeat_source_mismatch")

    heartbeat_start = _parse_heartbeat_timestamp(
        heartbeat.get("generated_at_utc"), refusal_reasons
    )
    if heartbeat_start is not None:
        age_seconds = (observed_at - heartbeat_start).total_seconds()
        if age_seconds < 0.0:
            _refuse(refusal_reasons, "heartbeat_timestamp_in_future")
        elif age_seconds > FRVP_CONFIRMATION_MAX_HEARTBEAT_AGE_SECONDS:
            _refuse(refusal_reasons, "heartbeat_too_old_for_clock_start")

    runtime = heartbeat.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    if str(runtime.get("terminal_status") or "").lower() != "running":
        _refuse(refusal_reasons, "runtime_not_running")
    collector = runtime.get("collector_contract")
    collector = collector if isinstance(collector, Mapping) else {}
    if collector.get("schema_version") != 1:
        _refuse(refusal_reasons, "collector_contract_mismatch")

    feed = collector.get("feed")
    feed = feed if isinstance(feed, Mapping) else {}
    es_contract = feed.get("contract")
    es_contract = es_contract if isinstance(es_contract, Mapping) else {}
    feed_matches = (
        str(feed.get("data_supplier") or "").upper() == "IBKR"
        and str(feed.get("heartbeat_source") or "") == "ibkr.historical.polling"
        and str(feed.get("asset") or "").upper() == "ES"
        and str(feed.get("source_timeframe") or "") == "5m"
        and feed.get("ibkr_enabled") is True
        and str(feed.get("account_mode") or "").lower() == "paper"
        and str(feed.get("market_data_type_requested") or "").lower()
        in {"live", "1"}
        and str(feed.get("market_data_type_received") or "").lower() == "live"
        and str(feed.get("connection_state") or "").lower() == "connected"
        and feed.get("allow_delayed_fallback") is False
        and str(feed.get("what_to_show") or "").upper() == "TRADES"
        and feed.get("use_rth") is False
        and str(feed.get("bar_size") or "").lower() == "5 mins"
        and feed.get("keep_up_to_date") is True
        and str(es_contract.get("symbol") or "").upper() == "ES"
        and str(es_contract.get("security_type") or "").upper() == "FUT"
        and str(es_contract.get("exchange") or "").upper() == "CME"
        and str(es_contract.get("currency") or "").upper() == "USD"
        and str(es_contract.get("multiplier") or "") == "50"
        and str(es_contract.get("trading_class") or "").upper() == "ES"
    )
    if not feed_matches:
        _refuse(refusal_reasons, "paper_feed_contract_mismatch")

    signal_runtime = collector.get("signal_runtime")
    signal_runtime = signal_runtime if isinstance(signal_runtime, Mapping) else {}
    runtime_matches = (
        signal_runtime.get("enabled") is True
        and signal_runtime.get("loaded") is True
        and signal_runtime.get("all_models_active") is False
        and _exact_string_roster(
            signal_runtime.get("loaded_model_ids"),
            FRVP_CONFIRMATION_LOADED_MODEL_IDS,
        )
        and _exact_string_roster(
            signal_runtime.get("active_model_ids"),
            FRVP_CONFIRMATION_ACTIVE_MODEL_IDS,
        )
        and _exact_string_roster(
            signal_runtime.get("shadow_model_ids"),
            FRVP_CONFIRMATION_SHADOW_MODEL_IDS,
        )
        and _exact_string_roster(
            signal_runtime.get("registry_paths"),
            {FRVP_PAPER_SIGNAL_REGISTRY_PATH},
            normalize_paths=True,
        )
        and signal_runtime.get("frvp_paper_signal_ledger_ready") is True
    )
    if not runtime_matches:
        _refuse(refusal_reasons, "final_bundle_runtime_mismatch")
    if not _exact_string_roster(
        signal_runtime.get("active_model_ids"),
        FRVP_CONFIRMATION_ACTIVE_MODEL_IDS,
    ):
        _refuse(refusal_reasons, "active_model_roster_mismatch")

    return heartbeat_start


def _parse_heartbeat_timestamp(
    value: Any,
    refusal_reasons: list[str],
) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        _refuse(refusal_reasons, "heartbeat_timestamp_invalid")
        return None
    if parsed.tzinfo is None:
        _refuse(refusal_reasons, "heartbeat_timestamp_invalid")
        return None
    return ensure_utc(parsed)


def _exact_string_roster(
    value: Any,
    expected: set[str] | frozenset[str],
    *,
    normalize_paths: bool = False,
) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    normalized = [
        str(item).replace("\\", "/") if normalize_paths else str(item)
        for item in value
    ]
    return len(normalized) == len(expected) and frozenset(normalized) == expected


def _lifecycle_from_payload(
    payload: Mapping[str, Any],
    *,
    now: datetime,
    newly_started: bool,
) -> FrvpConfirmationLifecycle:
    try:
        start = ensure_utc(datetime.fromisoformat(str(payload["confirmation_start_utc"])))
        earliest = ensure_utc(
            datetime.fromisoformat(str(payload["earliest_minimum_completion_utc"]))
        )
        minimum_days = int(payload["minimum_confirmation_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Stored FRVP confirmation lifecycle is malformed.") from exc
    expected_earliest = start + timedelta(days=FRVP_CONFIRMATION_MINIMUM_DAYS)
    invariant_matches = (
        payload.get("schema_version") == 1
        and payload.get("state") == "confirmation_running"
        and payload.get("bundle_id") == FRVP_PAPER_SIGNAL_BUNDLE_ID
        and _exact_string_roster(
            payload.get("active_model_ids"),
            FRVP_CONFIRMATION_ACTIVE_MODEL_IDS,
        )
        and minimum_days == FRVP_CONFIRMATION_MINIMUM_DAYS
        and earliest == expected_earliest
        and payload.get("broker_orders_authorized") is False
        and payload.get("broker_order_submission_authorized") is False
        and payload.get("execution_authorized") is False
    )
    if not invariant_matches:
        raise RuntimeError(
            "Stored FRVP confirmation lifecycle violates its no-order/frozen-bundle contract."
        )
    return FrvpConfirmationLifecycle(
        started=True,
        newly_started=bool(newly_started),
        confirmation_start_utc=start,
        earliest_minimum_completion_utc=earliest,
        minimum_confirmation_days=minimum_days,
        minimum_period_elapsed=ensure_utc(now) >= earliest,
        broker_orders_authorized=False,
    )


def _not_started(refusal_reasons: tuple[str, ...]) -> FrvpConfirmationLifecycle:
    return FrvpConfirmationLifecycle(
        started=False,
        newly_started=False,
        confirmation_start_utc=None,
        earliest_minimum_completion_utc=None,
        minimum_confirmation_days=FRVP_CONFIRMATION_MINIMUM_DAYS,
        minimum_period_elapsed=False,
        broker_orders_authorized=False,
        refusal_reasons=refusal_reasons,
    )


def _refuse(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)
