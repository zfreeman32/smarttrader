from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.frvp_confirmation_lifecycle import (
    FRVP_CONFIRMATION_MINIMUM_DAYS,
    FRVP_CONFIRMATION_SCOPE,
    FRVP_CONFIRMATION_STATE_KEY,
    get_frvp_confirmation_lifecycle,
    record_frvp_confirmation_start,
)
from ote_live.storage.frvp_paper_signal import FRVP_PAPER_SIGNAL_BUNDLE_ID
from ote_live.ops.health import ServiceHealthSnapshot
from ote_live.scripts.run_es_live_collector import (
    _record_frvp_confirmation_clock_after_heartbeat,
)


START = datetime(2026, 8, 16, 14, 0, tzinfo=UTC)
ACTIVE_MODEL_ID = "frvp_long_reversal_xgb_v1"
LOADED_MODEL_IDS = [
    "frvp_long_continuation_xgb_v1",
    ACTIVE_MODEL_ID,
    "frvp_long_meta_xgb_v1",
    "frvp_short_continuation_tcn_v1",
    "frvp_short_meta_xgb_v1",
]


def test_records_first_exact_healthy_running_heartbeat_once(tmp_path: Path) -> None:
    with SQLiteLiveDataStore(tmp_path / "live.sqlite3") as store:
        first = record_frvp_confirmation_start(
            store,
            heartbeat=_healthy_final_heartbeat(),
            bundle_id=FRVP_PAPER_SIGNAL_BUNDLE_ID,
            first_healthy_final_heartbeat_confirmed=True,
            recorded_at_utc=START + timedelta(seconds=1),
        )
        row_before = store.connection.execute(
            """
            SELECT payload_json, created_at_utc, updated_at_utc
            FROM runtime_state WHERE scope = ? AND state_key = ?
            """,
            (FRVP_CONFIRMATION_SCOPE, FRVP_CONFIRMATION_STATE_KEY),
        ).fetchone()

        later_heartbeat = _healthy_final_heartbeat(
            generated_at=START + timedelta(hours=3)
        )
        repeated = record_frvp_confirmation_start(
            store,
            heartbeat=later_heartbeat,
            bundle_id=FRVP_PAPER_SIGNAL_BUNDLE_ID,
            first_healthy_final_heartbeat_confirmed=True,
            recorded_at_utc=START + timedelta(hours=3, seconds=1),
        )
        row_after = store.connection.execute(
            """
            SELECT payload_json, created_at_utc, updated_at_utc
            FROM runtime_state WHERE scope = ? AND state_key = ?
            """,
            (FRVP_CONFIRMATION_SCOPE, FRVP_CONFIRMATION_STATE_KEY),
        ).fetchone()

    assert first.started is True
    assert first.newly_started is True
    assert first.confirmation_start_utc == START
    assert first.earliest_minimum_completion_utc == START + timedelta(days=28)
    assert first.minimum_confirmation_days == FRVP_CONFIRMATION_MINIMUM_DAYS
    assert first.broker_orders_authorized is False
    assert repeated.started is True
    assert repeated.newly_started is False
    assert repeated.confirmation_start_utc == START
    assert tuple(row_before) == tuple(row_after)


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("not_explicit", "explicit_first_healthy_final_heartbeat_required"),
        ("bootstrap", "collector_not_running"),
        ("stopped", "collector_not_running"),
        ("unhealthy", "collector_unhealthy"),
        ("stale", "collector_heartbeat_stale_or_unknown"),
        ("wrong_bundle", "bundle_id_mismatch"),
        ("wrong_active_model", "active_model_roster_mismatch"),
    ],
)
def test_refuses_nonfinal_start_evidence(
    tmp_path: Path,
    case: str,
    expected_reason: str,
) -> None:
    heartbeat = _healthy_final_heartbeat()
    bundle_id = FRVP_PAPER_SIGNAL_BUNDLE_ID
    explicit = True
    if case == "not_explicit":
        explicit = False
    elif case == "bootstrap":
        heartbeat["service_status"] = "starting"
        heartbeat["health_state"] = "starting"
        heartbeat["runtime"]["terminal_status"] = "starting"
    elif case == "stopped":
        heartbeat["service_status"] = "stopped"
        heartbeat["runtime"]["terminal_status"] = "completed"
    elif case == "unhealthy":
        heartbeat["health_state"] = "unhealthy"
    elif case == "stale":
        heartbeat["heartbeat_is_stale"] = True
    elif case == "wrong_bundle":
        bundle_id = "frvp_es_shadow_20260721"
        heartbeat["runtime"]["collector_contract"]["signal_runtime"][
            "registry_paths"
        ] = ["models/frvp_es_shadow_live_registry_20260721.json"]
    elif case == "wrong_active_model":
        signal_runtime = heartbeat["runtime"]["collector_contract"][
            "signal_runtime"
        ]
        signal_runtime["active_model_ids"] = [
            "frvp_long_continuation_xgb_v1"
        ]
        signal_runtime["shadow_model_ids"] = [
            model_id
            for model_id in LOADED_MODEL_IDS
            if model_id != "frvp_long_continuation_xgb_v1"
        ]

    with SQLiteLiveDataStore(tmp_path / f"{case}.sqlite3") as store:
        result = record_frvp_confirmation_start(
            store,
            heartbeat=heartbeat,
            bundle_id=bundle_id,
            first_healthy_final_heartbeat_confirmed=explicit,
            recorded_at_utc=START + timedelta(seconds=1),
        )
        persisted = store.get_runtime_state(
            scope=FRVP_CONFIRMATION_SCOPE,
            state_key=FRVP_CONFIRMATION_STATE_KEY,
        )

    assert result.started is False
    assert result.newly_started is False
    assert result.broker_orders_authorized is False
    assert expected_reason in result.refusal_reasons
    assert persisted is None


def test_minimum_period_cannot_elapse_before_28_days(tmp_path: Path) -> None:
    with SQLiteLiveDataStore(tmp_path / "minimum.sqlite3") as store:
        record_frvp_confirmation_start(
            store,
            heartbeat=_healthy_final_heartbeat(),
            bundle_id=FRVP_PAPER_SIGNAL_BUNDLE_ID,
            first_healthy_final_heartbeat_confirmed=True,
            recorded_at_utc=START + timedelta(seconds=1),
        )
        too_early = get_frvp_confirmation_lifecycle(
            store,
            now=START + timedelta(days=28) - timedelta(microseconds=1),
        )
        boundary = get_frvp_confirmation_lifecycle(
            store,
            now=START + timedelta(days=28),
        )

    assert too_early.minimum_period_elapsed is False
    assert boundary.minimum_period_elapsed is True
    assert boundary.broker_orders_authorized is False


def test_refuses_old_running_heartbeat_replay(tmp_path: Path) -> None:
    with SQLiteLiveDataStore(tmp_path / "old.sqlite3") as store:
        result = record_frvp_confirmation_start(
            store,
            heartbeat=_healthy_final_heartbeat(),
            bundle_id=FRVP_PAPER_SIGNAL_BUNDLE_ID,
            first_healthy_final_heartbeat_confirmed=True,
            recorded_at_utc=START + timedelta(seconds=121),
        )

    assert result.started is False
    assert "heartbeat_too_old_for_clock_start" in result.refusal_reasons


def test_rejects_tampered_authority_in_existing_state(tmp_path: Path) -> None:
    with SQLiteLiveDataStore(tmp_path / "tampered.sqlite3") as store:
        record_frvp_confirmation_start(
            store,
            heartbeat=_healthy_final_heartbeat(),
            bundle_id=FRVP_PAPER_SIGNAL_BUNDLE_ID,
            first_healthy_final_heartbeat_confirmed=True,
            recorded_at_utc=START + timedelta(seconds=1),
        )
        payload = store.get_runtime_state(
            scope=FRVP_CONFIRMATION_SCOPE,
            state_key=FRVP_CONFIRMATION_STATE_KEY,
        )
        assert payload is not None
        payload["broker_orders_authorized"] = True
        store.upsert_runtime_state(
            scope=FRVP_CONFIRMATION_SCOPE,
            state_key=FRVP_CONFIRMATION_STATE_KEY,
            payload=payload,
        )

        with pytest.raises(RuntimeError, match="no-order/frozen-bundle"):
            get_frvp_confirmation_lifecycle(store, now=START + timedelta(days=29))


def test_collector_hook_starts_only_for_enabled_exact_healthy_snapshot(
    tmp_path: Path,
) -> None:
    with SQLiteLiveDataStore(tmp_path / "collector-hook.sqlite3") as store:
        runtime = SimpleNamespace(store=store)
        snapshot = _healthy_final_snapshot()

        _record_frvp_confirmation_clock_after_heartbeat(
            runtime,
            snapshot,
            enabled=False,
        )
        assert store.get_runtime_state(
            scope=FRVP_CONFIRMATION_SCOPE,
            state_key=FRVP_CONFIRMATION_STATE_KEY,
        ) is None

        _record_frvp_confirmation_clock_after_heartbeat(
            runtime,
            snapshot,
            enabled=True,
        )
        lifecycle = get_frvp_confirmation_lifecycle(store, now=START)

    assert lifecycle.started is True
    assert lifecycle.confirmation_start_utc == START
    assert lifecycle.broker_orders_authorized is False


def test_collector_hook_fails_closed_on_healthy_contract_mismatch(
    tmp_path: Path,
) -> None:
    with SQLiteLiveDataStore(tmp_path / "collector-hook-mismatch.sqlite3") as store:
        runtime = SimpleNamespace(store=store)
        heartbeat = _healthy_final_heartbeat()
        heartbeat["runtime"]["collector_contract"]["feed"][
            "market_data_type_received"
        ] = "delayed"
        snapshot = _snapshot_from_heartbeat(heartbeat)

        with pytest.raises(RuntimeError, match="paper_feed_contract_mismatch"):
            _record_frvp_confirmation_clock_after_heartbeat(
                runtime,
                snapshot,
                enabled=True,
            )

        assert store.get_runtime_state(
            scope=FRVP_CONFIRMATION_SCOPE,
            state_key=FRVP_CONFIRMATION_STATE_KEY,
        ) is None


def _healthy_final_heartbeat(
    *,
    generated_at: datetime = START,
) -> dict:
    heartbeat = {
        "generated_at_utc": generated_at.isoformat(),
        "service_name": "es-shared-live-signal-service",
        "service_status": "running",
        "health_state": "healthy",
        "heartbeat_is_stale": False,
        "asset": "ES",
        "source_timeframe": "5m",
        "signal_timeframe": "5m",
        "latest_heartbeat_source": "ibkr.historical.polling",
        "runtime": {
            "terminal_status": "running",
            "collector_contract": {
                "schema_version": 1,
                "feed": {
                    "data_supplier": "IBKR",
                    "heartbeat_source": "ibkr.historical.polling",
                    "asset": "ES",
                    "source_timeframe": "5m",
                    "ibkr_enabled": True,
                    "account_mode": "paper",
                    "market_data_type_requested": "live",
                    "market_data_type_received": "live",
                    "connection_state": "connected",
                    "allow_delayed_fallback": False,
                    "what_to_show": "TRADES",
                    "use_rth": False,
                    "bar_size": "5 mins",
                    "keep_up_to_date": True,
                    "contract": {
                        "symbol": "ES",
                        "security_type": "FUT",
                        "exchange": "CME",
                        "currency": "USD",
                        "multiplier": "50",
                        "trading_class": "ES",
                    },
                },
                "signal_runtime": {
                    "enabled": True,
                    "loaded": True,
                    "all_models_active": False,
                    "loaded_model_ids": LOADED_MODEL_IDS,
                    "active_model_ids": [ACTIVE_MODEL_ID],
                    "shadow_model_ids": [
                        model_id
                        for model_id in LOADED_MODEL_IDS
                        if model_id != ACTIVE_MODEL_ID
                    ],
                    "registry_paths": [
                        "models/frvp_es_paper_signal_registry_20260816.json"
                    ],
                    "frvp_paper_signal_ledger_ready": True,
                },
            },
        },
    }
    return deepcopy(heartbeat)


def _healthy_final_snapshot() -> ServiceHealthSnapshot:
    return _snapshot_from_heartbeat(_healthy_final_heartbeat())


def _snapshot_from_heartbeat(heartbeat: dict) -> ServiceHealthSnapshot:
    return ServiceHealthSnapshot(
        service_name=heartbeat["service_name"],
        service_status=heartbeat["service_status"],
        health_state=heartbeat["health_state"],
        generated_at_utc=datetime.fromisoformat(heartbeat["generated_at_utc"]),
        hostname="test-host",
        process_id=1234,
        db_path="test.sqlite3",
        asset=heartbeat["asset"],
        source_timeframe=heartbeat["source_timeframe"],
        signal_timeframe=heartbeat["signal_timeframe"],
        latest_source_bar_timestamp=START,
        latest_signal_timestamp=None,
        latest_prediction_timestamp=None,
        latest_heartbeat_source=heartbeat["latest_heartbeat_source"],
        latest_heartbeat_observed_at=START,
        heartbeat_lag_seconds=0.0,
        heartbeat_is_stale=heartbeat["heartbeat_is_stale"],
        unresolved_gap_count=0,
        recent_warning_count_24h=0,
        recent_error_count_24h=0,
        runtime=deepcopy(heartbeat["runtime"]),
    )
