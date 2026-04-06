from __future__ import annotations

import json
import sys
import uuid
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import HeartbeatStatus, IngestionGap
from ote_live.ingestion.runtime import BootstrapSummary, CollectorRunSummary
from ote_live.ops import (
    DISK_MONITOR_SCOPE,
    DiskSpaceMonitor,
    DiskUsageStatus,
    SERVICE_LIFECYCLE_SCOPE,
    ServiceHeartbeatWriter,
    capture_startup_recovery_state,
    collect_service_health_snapshot,
    record_service_shutdown,
    record_service_started,
    record_service_starting,
)
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore


def test_startup_and_shutdown_ops_persist_service_lifecycle_state() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_ops_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    source_bar = MarketBar(
        asset="EURUSD",
        timeframe="1m",
        timestamp=datetime(2026, 4, 5, 12, 0, tzinfo=UTC),
        open=1.1000,
        high=1.1003,
        low=1.0998,
        close=1.1002,
        volume=11.0,
        source="ops-test",
    )

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        store.upsert_bar(source_bar)
        store.record_gap(
            IngestionGap(
                asset="EURUSD",
                timeframe="1m",
                expected_timestamp=source_bar.timestamp + timedelta(minutes=1),
                observed_timestamp=source_bar.timestamp + timedelta(minutes=3),
                missing_timestamps=[
                    source_bar.timestamp + timedelta(minutes=1),
                    source_bar.timestamp + timedelta(minutes=2),
                ],
                gap_size=2,
            )
        )
        recovery_state = capture_startup_recovery_state(
            store,
            service_name="svc-test",
            asset="EURUSD",
            source_timeframe="1m",
            signal_timeframe="5m",
        )
        assert recovery_state.latest_source_bar_timestamp == source_bar.timestamp
        assert recovery_state.unresolved_gap_count == 1

        record_service_starting(
            store,
            audit,
            service_name="svc-test",
            db_path=db_path,
            asset="EURUSD",
            source_timeframe="1m",
            signal_timeframe="5m",
            recovery_state=recovery_state,
        )
        starting_state = store.get_runtime_state(
            scope=SERVICE_LIFECYCLE_SCOPE,
            state_key="svc-test",
        )
        assert starting_state is not None
        assert starting_state["status"] == "starting"
        assert starting_state["startup_recovery"]["unresolved_gap_count"] == 1

        bootstrap = BootstrapSummary(
            latest_stored_timestamp=source_bar.timestamp,
            seeded_history_bars=0,
            catchup_bars=2,
            stored_source_bars=2,
            stored_aggregated_bars=1,
        )
        record_service_started(
            store,
            audit,
            service_name="svc-test",
            bootstrap_summary=bootstrap,
        )
        running_state = store.get_runtime_state(
            scope=SERVICE_LIFECYCLE_SCOPE,
            state_key="svc-test",
        )
        assert running_state is not None
        assert running_state["status"] == "running"
        assert running_state["bootstrap"]["catchup_bars"] == 2

        record_service_shutdown(
            store,
            audit,
            service_name="svc-test",
            summary=CollectorRunSummary(
                bootstrap=bootstrap,
                cycles_run=3,
                flushed_aggregated_bars=1,
                terminal_status="completed",
            ),
        )
        stopped_state = store.get_runtime_state(
            scope=SERVICE_LIFECYCLE_SCOPE,
            state_key="svc-test",
        )
        assert stopped_state is not None
        assert stopped_state["status"] == "stopped"
        assert stopped_state["cycles_run"] == 3

        startup_events = audit.fetch_health_events(component="ops.startup")
        shutdown_events = audit.fetch_health_events(component="ops.shutdown")

    assert [event.event_type for event in startup_events] == ["service_starting", "service_started"]
    assert [event.event_type for event in shutdown_events] == ["service_stopped"]


def test_disk_space_monitor_records_transition_events_once_per_state_change(monkeypatch) -> None:
    tmp_root = ROOT / "tmp" / "ote_live_ops_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    usage_type = namedtuple("usage", ["total", "used", "free"])
    usage_samples = [
        usage_type(total=10_000, used=9_700, free=300),
        usage_type(total=10_000, used=9_650, free=350),
        usage_type(total=10_000, used=8_000, free=2_000),
    ]

    def _fake_disk_usage(path):
        return usage_samples.pop(0)

    monkeypatch.setattr("ote_live.ops.health.shutil.disk_usage", _fake_disk_usage)

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        monitor = DiskSpaceMonitor(minimum_free_bytes=500)
        first = monitor.evaluate(store, audit, paths=(tmp_root,))
        second = monitor.evaluate(store, audit, paths=(tmp_root,))
        third = monitor.evaluate(store, audit, paths=(tmp_root,))
        state = store.get_runtime_state(
            scope=DISK_MONITOR_SCOPE,
            state_key=Path(tmp_root).resolve().anchor,
        )
        health_events = audit.fetch_health_events(component="ops.disk_monitor")

    assert first[0].below_threshold is True
    assert second[0].below_threshold is True
    assert third[0].below_threshold is False
    assert state is not None
    assert state["below_threshold"] is False
    assert [event.event_type for event in health_events] == [
        "low_disk_space",
        "disk_space_recovered",
    ]


def test_record_service_starting_flags_unclean_restart_when_previous_state_was_running() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_ops_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    previous_start = datetime(2026, 4, 5, 8, 0, tzinfo=UTC)

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        store.upsert_runtime_state(
            scope=SERVICE_LIFECYCLE_SCOPE,
            state_key="svc-test",
            payload={
                "status": "running",
                "started_at_utc": previous_start.isoformat(),
                "stopped_at_utc": None,
            },
        )
        recovery_state = capture_startup_recovery_state(
            store,
            service_name="svc-test",
            asset="EURUSD",
            source_timeframe="1m",
            signal_timeframe="5m",
        )
        record_service_starting(
            store,
            audit,
            service_name="svc-test",
            db_path=db_path,
            asset="EURUSD",
            source_timeframe="1m",
            signal_timeframe="5m",
            recovery_state=recovery_state,
        )
        startup_events = audit.fetch_health_events(component="ops.startup")

    assert [event.event_type for event in startup_events] == [
        "service_starting",
        "unclean_restart_detected",
    ]


def test_service_heartbeat_writer_persists_health_snapshot_file() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_ops_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    heartbeat_path = tmp_root / f"{uuid.uuid4().hex}.json"
    source_bar = MarketBar(
        asset="EURUSD",
        timeframe="1m",
        timestamp=datetime(2026, 4, 5, 13, 0, tzinfo=UTC),
        open=1.2000,
        high=1.2004,
        low=1.1998,
        close=1.2001,
        volume=15.0,
        source="ops-test",
    )

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        store.upsert_bar(source_bar)
        store.record_heartbeat(
            HeartbeatStatus(
                source="collector-test",
                observed_at_utc=source_bar.timestamp + timedelta(seconds=10),
                stale_after_seconds=90.0,
                lag_seconds=10.0,
                is_stale=False,
            ),
            metadata={"service": "ops-test"},
        )
        audit.record_health_event(
            component="ops.test",
            event_type="warning_event",
            severity="warning",
            message="warning",
            payload={},
            event_timestamp_utc=source_bar.timestamp,
        )
        audit.record_health_event(
            component="ops.test",
            event_type="error_event",
            severity="error",
            message="error",
            payload={},
            event_timestamp_utc=source_bar.timestamp,
        )
        snapshot = collect_service_health_snapshot(
            store,
            service_name="svc-test",
            service_status="running",
            db_path=db_path,
            asset="EURUSD",
            source_timeframe="1m",
            signal_timeframe="5m",
            disk_statuses=(
                DiskUsageStatus(
                    volume=str(tmp_root.drive or tmp_root.anchor or tmp_root),
                    path=str(tmp_root),
                    total_bytes=10_000,
                    used_bytes=5_000,
                    free_bytes=5_000,
                    minimum_free_bytes=500,
                    below_threshold=False,
                ),
            ),
            bootstrap_payload={"catchup_bars": 2},
            runtime_payload={"terminal_status": "running"},
        )
        writer = ServiceHeartbeatWriter(heartbeat_path)
        writer.write_snapshot(snapshot)

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload["service_name"] == "svc-test"
    assert payload["service_status"] == "running"
    assert payload["health_state"] == "degraded"
    assert payload["latest_source_bar_timestamp"] == source_bar.timestamp.isoformat()
    assert payload["latest_heartbeat_source"] == "collector-test"
    assert payload["recent_warning_count_24h"] == 1
    assert payload["recent_error_count_24h"] == 1
    assert payload["disk_statuses"][0]["below_threshold"] is False
