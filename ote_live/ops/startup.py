from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

SERVICE_LIFECYCLE_SCOPE = "service_lifecycle"


@dataclass(frozen=True)
class StartupRecoveryState:
    service_name: str
    asset: str
    source_timeframe: str
    signal_timeframe: str
    latest_source_bar_timestamp: datetime | None
    latest_signal_timestamp: datetime | None
    unresolved_gap_count: int
    previous_status: str | None
    previous_started_at_utc: datetime | None
    previous_stopped_at_utc: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "asset": self.asset,
            "source_timeframe": self.source_timeframe,
            "signal_timeframe": self.signal_timeframe,
            "latest_source_bar_timestamp": _optional_isoformat(self.latest_source_bar_timestamp),
            "latest_signal_timestamp": _optional_isoformat(self.latest_signal_timestamp),
            "unresolved_gap_count": self.unresolved_gap_count,
            "previous_status": self.previous_status,
            "previous_started_at_utc": _optional_isoformat(self.previous_started_at_utc),
            "previous_stopped_at_utc": _optional_isoformat(self.previous_stopped_at_utc),
        }


def capture_startup_recovery_state(
    store: SQLiteLiveDataStore,
    *,
    service_name: str,
    asset: str,
    source_timeframe: str,
    signal_timeframe: str,
) -> StartupRecoveryState:
    previous_state = store.get_runtime_state(
        scope=SERVICE_LIFECYCLE_SCOPE,
        state_key=service_name,
    ) or {}
    gap_row = store.connection.execute(
        """
        SELECT COUNT(*) AS unresolved_gap_count
        FROM ingestion_gaps
        WHERE resolved_at_utc IS NULL
        """
    ).fetchone()
    return StartupRecoveryState(
        service_name=service_name,
        asset=asset,
        source_timeframe=source_timeframe,
        signal_timeframe=signal_timeframe,
        latest_source_bar_timestamp=store.get_latest_bar_timestamp(asset=asset, timeframe=source_timeframe),
        latest_signal_timestamp=_fetch_latest_timestamp(store, table_name="signal_decisions"),
        unresolved_gap_count=int(gap_row["unresolved_gap_count"]) if gap_row is not None else 0,
        previous_status=_read_optional_str(previous_state.get("status")),
        previous_started_at_utc=_parse_iso_datetime(previous_state.get("started_at_utc")),
        previous_stopped_at_utc=_parse_iso_datetime(previous_state.get("stopped_at_utc")),
    )


def record_service_starting(
    store: SQLiteLiveDataStore,
    audit_repository: LiveAuditRepository | None,
    *,
    service_name: str,
    db_path: str | Path,
    asset: str,
    source_timeframe: str,
    signal_timeframe: str,
    recovery_state: StartupRecoveryState,
    now: datetime | None = None,
) -> dict[str, Any]:
    resolved_now = ensure_utc(now or utc_now())
    payload = {
        "service_name": service_name,
        "status": "starting",
        "started_at_utc": resolved_now.isoformat(),
        "stopped_at_utc": None,
        "db_path": str(db_path),
        "asset": asset,
        "source_timeframe": source_timeframe,
        "signal_timeframe": signal_timeframe,
        "process_id": os.getpid(),
        "hostname": socket.gethostname(),
        "startup_recovery": recovery_state.to_dict(),
        "recovery_needed": bool(
            recovery_state.latest_source_bar_timestamp is not None
            or recovery_state.unresolved_gap_count > 0
            or recovery_state.previous_status == "running"
        ),
        "updated_at_utc": resolved_now.isoformat(),
    }
    merged = _merge_service_state(store, service_name=service_name, payload=payload)
    if audit_repository is not None:
        audit_repository.record_health_event(
            component="ops.startup",
            event_type="service_starting",
            severity="info",
            message="Live signal service startup initiated.",
            payload={
                "service_name": service_name,
                "asset": asset,
                "source_timeframe": source_timeframe,
                "previous_status": recovery_state.previous_status,
                "unresolved_gap_count": recovery_state.unresolved_gap_count,
            },
            event_timestamp_utc=resolved_now,
        )
        if recovery_state.previous_status == "running":
            audit_repository.record_health_event(
                component="ops.startup",
                event_type="unclean_restart_detected",
                severity="warning",
                message="The previous service lifecycle state still showed 'running' at startup.",
                payload={
                    "service_name": service_name,
                    "previous_started_at_utc": _optional_isoformat(recovery_state.previous_started_at_utc),
                    "previous_stopped_at_utc": _optional_isoformat(recovery_state.previous_stopped_at_utc),
                },
                event_timestamp_utc=resolved_now,
            )
    return merged


def record_service_started(
    store: SQLiteLiveDataStore,
    audit_repository: LiveAuditRepository | None,
    *,
    service_name: str,
    bootstrap_summary,
    now: datetime | None = None,
) -> dict[str, Any]:
    resolved_now = ensure_utc(now or utc_now())
    payload = {
        "service_name": service_name,
        "status": "running",
        "bootstrap": {
            "latest_stored_timestamp": _optional_isoformat(bootstrap_summary.latest_stored_timestamp),
            "seeded_history_bars": int(bootstrap_summary.seeded_history_bars),
            "catchup_bars": int(bootstrap_summary.catchup_bars),
            "stored_source_bars": int(bootstrap_summary.stored_source_bars),
            "stored_aggregated_bars": int(bootstrap_summary.stored_aggregated_bars),
            "startup_gap_recovery_attempts": int(getattr(bootstrap_summary, "startup_gap_recovery_attempts", 0)),
            "startup_gap_backfilled_bars": int(getattr(bootstrap_summary, "startup_gap_backfilled_bars", 0)),
            "resolved_startup_gaps": int(getattr(bootstrap_summary, "resolved_startup_gaps", 0)),
            "remaining_unresolved_gaps": int(getattr(bootstrap_summary, "remaining_unresolved_gaps", 0)),
        },
        "updated_at_utc": resolved_now.isoformat(),
    }
    merged = _merge_service_state(store, service_name=service_name, payload=payload)
    if audit_repository is not None:
        audit_repository.record_health_event(
            component="ops.startup",
            event_type="service_started",
            severity="info",
            message="Live signal service startup completed.",
            payload={
                "service_name": service_name,
                "seeded_history_bars": int(bootstrap_summary.seeded_history_bars),
                "catchup_bars": int(bootstrap_summary.catchup_bars),
                "stored_source_bars": int(bootstrap_summary.stored_source_bars),
                "stored_aggregated_bars": int(bootstrap_summary.stored_aggregated_bars),
                "startup_gap_recovery_attempts": int(getattr(bootstrap_summary, "startup_gap_recovery_attempts", 0)),
                "resolved_startup_gaps": int(getattr(bootstrap_summary, "resolved_startup_gaps", 0)),
                "remaining_unresolved_gaps": int(getattr(bootstrap_summary, "remaining_unresolved_gaps", 0)),
            },
            event_timestamp_utc=resolved_now,
        )
    return merged


def _merge_service_state(
    store: SQLiteLiveDataStore,
    *,
    service_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    existing = store.get_runtime_state(
        scope=SERVICE_LIFECYCLE_SCOPE,
        state_key=service_name,
    ) or {}
    merged = dict(existing)
    merged.update(payload)
    store.upsert_runtime_state(
        scope=SERVICE_LIFECYCLE_SCOPE,
        state_key=service_name,
        payload=merged,
    )
    return merged


def _fetch_latest_timestamp(store: SQLiteLiveDataStore, *, table_name: str) -> datetime | None:
    row = store.connection.execute(
        f"""
        SELECT timestamp_utc
        FROM {table_name}
        ORDER BY timestamp_utc DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return ensure_utc(datetime.fromisoformat(row["timestamp_utc"]))


def _optional_isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return ensure_utc(value).isoformat()


def _parse_iso_datetime(value: object) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return ensure_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


def _read_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
