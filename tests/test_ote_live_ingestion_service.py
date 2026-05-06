from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.aggregator import MultiTimeframeBarAggregator
from ote_live.ingestion.base import BackfillWindow, IngestionGap
from ote_live.ingestion.gap_detector import GapDetector
from ote_live.ingestion.heartbeat import HeartbeatMonitor
from ote_live.ingestion.service import LiveBarIngestionService
from ote_live.storage.repositories import LiveAuditRepository
from ote_live.storage.db import SQLiteLiveDataStore


def test_live_bar_ingestion_service_backfills_gap_and_stores_aggregated_bar() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_service_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")

    stream = _FakeStream(
        [
            _bar(0),
            _bar(2),
            _bar(3),
            _bar(4),
            _bar(5),
        ]
    )
    backfill = _FakeBackfill([_bar(1)])
    service = LiveBarIngestionService(
        stream=stream,
        backfill=backfill,
        store=store,
        gap_detector=GapDetector(timeframe="1m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("5m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-stream"),
    )

    result = asyncio.run(service.collect_once())

    one_minute_bars = store.fetch_bars(asset="EURUSD", timeframe="1m")
    five_minute_bars = store.fetch_bars(asset="EURUSD", timeframe="5m")
    gaps = store.fetch_gaps(asset="EURUSD", timeframe="1m", unresolved_only=False)
    store.close()

    assert result.polled_bars == 5
    assert result.gaps_detected == 1
    assert result.backfilled_bars == 1
    assert result.stored_source_bars == 6
    assert result.stored_aggregated_bars == 1
    assert len(one_minute_bars) == 6
    assert len(five_minute_bars) == 1
    assert len(gaps) == 1
    assert gaps[0].resolved_at_utc is not None
    assert five_minute_bars[0].timestamp == datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)


def test_live_bar_ingestion_service_records_heartbeat_health_events() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_service_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    audit = LiveAuditRepository(store)

    service = LiveBarIngestionService(
        stream=_FakeStream([]),
        backfill=_FakeBackfill([]),
        store=store,
        gap_detector=GapDetector(timeframe="1m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("5m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-stream"),
        audit_repository=audit,
    )

    result = asyncio.run(service.collect_once())
    health_events = audit.fetch_health_events(component="ingestion.heartbeat")
    store.close()

    assert result.polled_bars == 0
    assert len(health_events) == 1
    assert health_events[0].event_type == "heartbeat_stale"
    assert health_events[0].severity == "warning"
    assert health_events[0].payload["polled_bars"] == 0


def test_live_bar_ingestion_service_records_stream_poll_failures() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_service_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    audit = LiveAuditRepository(store)

    service = LiveBarIngestionService(
        stream=_FailingStream(RuntimeError("stream unavailable")),
        backfill=_FakeBackfill([]),
        store=store,
        gap_detector=GapDetector(timeframe="1m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("5m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-stream"),
        audit_repository=audit,
    )

    try:
        asyncio.run(service.collect_once())
    except RuntimeError as exc:
        assert "stream unavailable" in str(exc)
    else:
        raise AssertionError("Expected the failing stream to raise a RuntimeError.")

    health_events = audit.fetch_health_events(component="ingestion.stream", event_type="poll_failed")
    store.close()

    assert len(health_events) == 1
    assert health_events[0].severity == "error"
    assert health_events[0].payload["error_type"] == "RuntimeError"


def test_live_bar_ingestion_service_fails_fast_when_gap_backfill_is_incomplete() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_service_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    audit = LiveAuditRepository(store)

    service = LiveBarIngestionService(
        stream=_FakeStream([_bar(0), _bar(2)]),
        backfill=_FakeBackfill([]),
        store=store,
        gap_detector=GapDetector(timeframe="1m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("5m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-stream"),
        audit_repository=audit,
    )

    try:
        asyncio.run(service.collect_once())
    except RuntimeError as exc:
        assert "Incomplete gap recovery" in str(exc)
    else:
        raise AssertionError("Expected incomplete gap backfill to fail fast.")

    health_events = audit.fetch_health_events(component="ingestion.backfill", event_type="gap_recovery_incomplete")
    gaps = store.fetch_gaps(asset="EURUSD", timeframe="1m", unresolved_only=True)
    store.close()

    assert len(health_events) == 1
    assert len(gaps) == 1


def test_live_bar_ingestion_service_resolves_weekend_only_startup_gap_without_backfill() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_service_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    audit = LiveAuditRepository(store)
    backfill = _FakeBackfill([])

    service = LiveBarIngestionService(
        stream=_FakeStream([]),
        backfill=backfill,
        store=store,
        gap_detector=GapDetector(asset="EURUSD", timeframe="5m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("30m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-stream"),
        audit_repository=audit,
    )
    missing_timestamps = _timestamp_range(
        datetime(2026, 4, 10, 21, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 12, 21, 0, tzinfo=timezone.utc),
        step=timedelta(minutes=5),
    )
    store.record_gap(
        IngestionGap(
            asset="EURUSD",
            timeframe="5m",
            expected_timestamp=missing_timestamps[0],
            observed_timestamp=datetime(2026, 4, 12, 21, 5, tzinfo=timezone.utc),
            missing_timestamps=missing_timestamps,
            gap_size=len(missing_timestamps),
        )
    )

    unresolved_gaps = store.fetch_gaps(asset="EURUSD", timeframe="5m", unresolved_only=True)
    result = asyncio.run(service.recover_unresolved_gaps(list(unresolved_gaps)))
    gaps = store.fetch_gaps(asset="EURUSD", timeframe="5m", unresolved_only=False)
    health_events = audit.fetch_health_events(component="collector.bootstrap", event_type="gap_resolved_market_closed")
    store.close()

    assert backfill.calls == 0
    assert result.backfilled_bars == 0
    assert len(gaps) == 1
    assert gaps[0].resolved_at_utc is not None
    assert len(health_events) == 1
    assert health_events[0].payload["market_closed_timestamp_count"] == len(missing_timestamps)


class _FakeStream:
    def __init__(self, bars: list[MarketBar]) -> None:
        self._bars = bars
        self._called = False

    async def poll(self) -> list[MarketBar]:
        if self._called:
            return []
        self._called = True
        return self._bars


class _FakeBackfill:
    def __init__(self, bars: list[MarketBar]) -> None:
        self._bars = bars
        self.calls = 0

    async def backfill_bars(self, window: BackfillWindow) -> list[MarketBar]:
        self.calls += 1
        assert window.start == datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc)
        assert window.end == datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc)
        return self._bars


class _FailingStream:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def poll(self) -> list[MarketBar]:
        raise self.exc


def _bar(minute: int) -> MarketBar:
    open_price = 1.1000 + (minute * 0.0001)
    return MarketBar(
        asset="EURUSD",
        timeframe="1m",
        timestamp=datetime(2024, 1, 2, 10, minute, tzinfo=timezone.utc),
        open=open_price,
        high=open_price + 0.0003,
        low=open_price - 0.0002,
        close=open_price + 0.0001,
        volume=10.0 + minute,
    )


def _timestamp_range(start: datetime, end: datetime, *, step: timedelta) -> list[datetime]:
    values: list[datetime] = []
    cursor = start
    while cursor <= end:
        values.append(cursor)
        cursor += step
    return values
