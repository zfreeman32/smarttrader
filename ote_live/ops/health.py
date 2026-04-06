from __future__ import annotations

import json
import os
import shutil
import socket
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEARTBEAT_PATH = REPO_ROOT / "ote_live" / "runtime_data" / "health" / "live_signal_service_heartbeat.json"
DISK_MONITOR_SCOPE = "ops.disk_monitor"


@dataclass(frozen=True)
class DiskUsageStatus:
    volume: str
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    minimum_free_bytes: int
    below_threshold: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume": self.volume,
            "path": self.path,
            "total_bytes": self.total_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "minimum_free_bytes": self.minimum_free_bytes,
            "below_threshold": self.below_threshold,
        }


@dataclass(frozen=True)
class ServiceHealthSnapshot:
    service_name: str
    service_status: str
    health_state: str
    generated_at_utc: datetime
    hostname: str
    process_id: int
    db_path: str
    asset: str
    source_timeframe: str
    signal_timeframe: str
    latest_source_bar_timestamp: datetime | None
    latest_signal_timestamp: datetime | None
    latest_prediction_timestamp: datetime | None
    latest_heartbeat_source: str | None
    latest_heartbeat_observed_at: datetime | None
    heartbeat_lag_seconds: float | None
    heartbeat_is_stale: bool | None
    unresolved_gap_count: int
    recent_warning_count_24h: int
    recent_error_count_24h: int
    disk_statuses: tuple[DiskUsageStatus, ...] = ()
    bootstrap: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "service_status": self.service_status,
            "health_state": self.health_state,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "hostname": self.hostname,
            "process_id": self.process_id,
            "db_path": self.db_path,
            "asset": self.asset,
            "source_timeframe": self.source_timeframe,
            "signal_timeframe": self.signal_timeframe,
            "latest_source_bar_timestamp": _optional_isoformat(self.latest_source_bar_timestamp),
            "latest_signal_timestamp": _optional_isoformat(self.latest_signal_timestamp),
            "latest_prediction_timestamp": _optional_isoformat(self.latest_prediction_timestamp),
            "latest_heartbeat_source": self.latest_heartbeat_source,
            "latest_heartbeat_observed_at": _optional_isoformat(self.latest_heartbeat_observed_at),
            "heartbeat_lag_seconds": self.heartbeat_lag_seconds,
            "heartbeat_is_stale": self.heartbeat_is_stale,
            "unresolved_gap_count": self.unresolved_gap_count,
            "recent_warning_count_24h": self.recent_warning_count_24h,
            "recent_error_count_24h": self.recent_error_count_24h,
            "disk_statuses": [item.to_dict() for item in self.disk_statuses],
            "bootstrap": _serialize_json_payload(self.bootstrap),
            "runtime": _serialize_json_payload(self.runtime),
        }


class DiskSpaceMonitor:
    def __init__(self, minimum_free_bytes: int) -> None:
        self.minimum_free_bytes = int(max(0, minimum_free_bytes))

    def evaluate(
        self,
        store: SQLiteLiveDataStore,
        audit_repository: LiveAuditRepository | None,
        *,
        paths: Iterable[str | Path],
        now: datetime | None = None,
    ) -> tuple[DiskUsageStatus, ...]:
        resolved_now = ensure_utc(now or utc_now())
        statuses = self.snapshot(paths)
        for status in statuses:
            state = store.get_runtime_state(
                scope=DISK_MONITOR_SCOPE,
                state_key=status.volume,
            ) or {}
            previous_below_threshold = bool(state.get("below_threshold"))
            store.upsert_runtime_state(
                scope=DISK_MONITOR_SCOPE,
                state_key=status.volume,
                payload={
                    "volume": status.volume,
                    "path": status.path,
                    "free_bytes": status.free_bytes,
                    "minimum_free_bytes": status.minimum_free_bytes,
                    "below_threshold": status.below_threshold,
                    "observed_at_utc": resolved_now.isoformat(),
                },
            )
            if audit_repository is None:
                continue
            if status.below_threshold and not previous_below_threshold:
                audit_repository.record_health_event(
                    component="ops.disk_monitor",
                    event_type="low_disk_space",
                    severity="warning",
                    message="Disk space dropped below the configured free-space threshold.",
                    payload={
                        "volume": status.volume,
                        "path": status.path,
                        "free_bytes": status.free_bytes,
                        "minimum_free_bytes": status.minimum_free_bytes,
                    },
                    event_timestamp_utc=resolved_now,
                )
            if not status.below_threshold and previous_below_threshold:
                audit_repository.record_health_event(
                    component="ops.disk_monitor",
                    event_type="disk_space_recovered",
                    severity="info",
                    message="Disk space recovered above the configured free-space threshold.",
                    payload={
                        "volume": status.volume,
                        "path": status.path,
                        "free_bytes": status.free_bytes,
                        "minimum_free_bytes": status.minimum_free_bytes,
                    },
                    event_timestamp_utc=resolved_now,
                )
        return statuses

    def snapshot(self, paths: Iterable[str | Path]) -> tuple[DiskUsageStatus, ...]:
        statuses: list[DiskUsageStatus] = []
        seen_volumes: set[str] = set()
        for raw_path in paths:
            target_path = _resolve_disk_usage_target(Path(raw_path))
            resolved_path = target_path.resolve()
            volume = resolved_path.anchor or str(resolved_path)
            if volume in seen_volumes:
                continue
            seen_volumes.add(volume)
            usage = shutil.disk_usage(target_path)
            free_bytes = int(usage.free)
            statuses.append(
                DiskUsageStatus(
                    volume=volume,
                    path=str(target_path),
                    total_bytes=int(usage.total),
                    used_bytes=int(usage.used),
                    free_bytes=free_bytes,
                    minimum_free_bytes=self.minimum_free_bytes,
                    below_threshold=free_bytes < self.minimum_free_bytes,
                )
            )
        return tuple(statuses)


class ServiceHeartbeatWriter:
    def __init__(self, heartbeat_path: str | Path = DEFAULT_HEARTBEAT_PATH) -> None:
        self.heartbeat_path = Path(heartbeat_path)

    def write_snapshot(self, snapshot: ServiceHealthSnapshot) -> Path:
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.heartbeat_path.with_suffix(self.heartbeat_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(snapshot.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.heartbeat_path)
        return self.heartbeat_path


def collect_service_health_snapshot(
    store: SQLiteLiveDataStore,
    *,
    service_name: str,
    service_status: str,
    db_path: str | Path,
    asset: str,
    source_timeframe: str,
    signal_timeframe: str,
    disk_statuses: Iterable[DiskUsageStatus] = (),
    bootstrap_payload: dict[str, Any] | None = None,
    runtime_payload: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> ServiceHealthSnapshot:
    resolved_now = ensure_utc(now or utc_now())
    heartbeat_row = store.connection.execute(
        """
        SELECT source, observed_at_utc, lag_seconds, is_stale
        FROM heartbeat_log
        ORDER BY observed_at_utc DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    unresolved_gap_row = store.connection.execute(
        """
        SELECT COUNT(*) AS unresolved_gap_count
        FROM ingestion_gaps
        WHERE resolved_at_utc IS NULL
        """
    ).fetchone()
    warning_count, error_count = _fetch_recent_health_counts(store, now=resolved_now)
    heartbeat_is_stale = (
        bool(heartbeat_row["is_stale"])
        if heartbeat_row is not None
        else None
    )
    unresolved_gap_count = (
        int(unresolved_gap_row["unresolved_gap_count"])
        if unresolved_gap_row is not None
        else 0
    )
    resolved_disk_statuses = tuple(disk_statuses)
    return ServiceHealthSnapshot(
        service_name=service_name,
        service_status=service_status,
        health_state=_derive_health_state(
            service_status=service_status,
            heartbeat_is_stale=heartbeat_is_stale,
            unresolved_gap_count=unresolved_gap_count,
            recent_warning_count_24h=warning_count,
            recent_error_count_24h=error_count,
            disk_statuses=resolved_disk_statuses,
        ),
        generated_at_utc=resolved_now,
        hostname=socket.gethostname(),
        process_id=os.getpid(),
        db_path=str(db_path),
        asset=asset,
        source_timeframe=source_timeframe,
        signal_timeframe=signal_timeframe,
        latest_source_bar_timestamp=store.get_latest_bar_timestamp(asset=asset, timeframe=source_timeframe),
        latest_signal_timestamp=_fetch_latest_timestamp(store, table_name="signal_decisions"),
        latest_prediction_timestamp=_fetch_latest_timestamp(store, table_name="model_predictions"),
        latest_heartbeat_source=heartbeat_row["source"] if heartbeat_row is not None else None,
        latest_heartbeat_observed_at=(
            ensure_utc(datetime.fromisoformat(heartbeat_row["observed_at_utc"]))
            if heartbeat_row is not None
            else None
        ),
        heartbeat_lag_seconds=(
            float(heartbeat_row["lag_seconds"])
            if heartbeat_row is not None
            else None
        ),
        heartbeat_is_stale=heartbeat_is_stale,
        unresolved_gap_count=unresolved_gap_count,
        recent_warning_count_24h=warning_count,
        recent_error_count_24h=error_count,
        disk_statuses=resolved_disk_statuses,
        bootstrap=dict(bootstrap_payload or {}),
        runtime=dict(runtime_payload or {}),
    )


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


def _fetch_recent_health_counts(
    store: SQLiteLiveDataStore,
    *,
    now: datetime,
) -> tuple[int, int]:
    cutoff = now - timedelta(hours=24)
    warning_row = store.connection.execute(
        """
        SELECT COUNT(*) AS warning_count
        FROM health_events
        WHERE severity = 'warning'
          AND event_timestamp_utc >= ?
        """,
        (cutoff.isoformat(),),
    ).fetchone()
    error_row = store.connection.execute(
        """
        SELECT COUNT(*) AS error_count
        FROM health_events
        WHERE severity = 'error'
          AND event_timestamp_utc >= ?
        """,
        (cutoff.isoformat(),),
    ).fetchone()
    return (
        int(warning_row["warning_count"]) if warning_row is not None else 0,
        int(error_row["error_count"]) if error_row is not None else 0,
    )


def _resolve_disk_usage_target(path: Path) -> Path:
    candidate = path
    if candidate.suffix:
        candidate = candidate.parent
    while not candidate.exists():
        if candidate.parent == candidate:
            return Path.cwd()
        candidate = candidate.parent
    return candidate


def _derive_health_state(
    *,
    service_status: str,
    heartbeat_is_stale: bool | None,
    unresolved_gap_count: int,
    recent_warning_count_24h: int,
    recent_error_count_24h: int,
    disk_statuses: tuple[DiskUsageStatus, ...],
) -> str:
    if service_status == "starting":
        return "starting"
    if service_status in {"failed", "cancelled"}:
        return "unhealthy"
    if heartbeat_is_stale:
        return "unhealthy"
    if any(status.below_threshold for status in disk_statuses):
        return "unhealthy"
    if unresolved_gap_count > 0 or recent_error_count_24h > 0 or recent_warning_count_24h > 0:
        return "degraded"
    return "healthy"


def _optional_isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return ensure_utc(value).isoformat()


def _serialize_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _serialize_json_value(value)
        for key, value in payload.items()
    }


def _serialize_json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return ensure_utc(value).isoformat()
    if isinstance(value, dict):
        return _serialize_json_payload(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_json_value(item) for item in value]
    return value
