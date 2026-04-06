from __future__ import annotations

from datetime import datetime
from typing import Any

from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.ops.startup import SERVICE_LIFECYCLE_SCOPE
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore


def record_service_shutdown(
    store: SQLiteLiveDataStore,
    audit_repository: LiveAuditRepository | None,
    *,
    service_name: str,
    summary,
    now: datetime | None = None,
) -> dict[str, Any]:
    resolved_now = ensure_utc(now or utc_now())
    service_status = _map_terminal_status_to_service_status(getattr(summary, "terminal_status", "completed"))
    payload = {
        "service_name": service_name,
        "status": service_status,
        "stopped_at_utc": resolved_now.isoformat(),
        "cycles_run": int(getattr(summary, "cycles_run", 0)),
        "evaluated_signals": int(getattr(summary, "evaluated_signals", 0)),
        "emitted_signals": int(getattr(summary, "emitted_signals", 0)),
        "sent_notifications": int(getattr(summary, "sent_notifications", 0)),
        "captured_media_artifacts": int(getattr(summary, "captured_media_artifacts", 0)),
        "flushed_aggregated_bars": int(getattr(summary, "flushed_aggregated_bars", 0)),
        "terminal_error": dict(getattr(summary, "terminal_error", None) or {}),
        "updated_at_utc": resolved_now.isoformat(),
    }
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

    if audit_repository is not None:
        event_type = f"service_{service_status}"
        severity = "error" if service_status == "failed" else "warning" if service_status == "cancelled" else "info"
        audit_repository.record_health_event(
            component="ops.shutdown",
            event_type=event_type,
            severity=severity,
            message=f"Live signal service transitioned to '{service_status}'.",
            payload={
                "service_name": service_name,
                "cycles_run": int(getattr(summary, "cycles_run", 0)),
                "flushed_aggregated_bars": int(getattr(summary, "flushed_aggregated_bars", 0)),
                "terminal_error": dict(getattr(summary, "terminal_error", None) or {}),
            },
            event_timestamp_utc=resolved_now,
        )
    return merged


def _map_terminal_status_to_service_status(status: str) -> str:
    if status == "completed":
        return "stopped"
    if status in {"cancelled", "failed"}:
        return status
    return "stopped"
