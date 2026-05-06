from __future__ import annotations

import asyncio
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import IngestionGap
from ote_live.ingestion.aggregator import MultiTimeframeBarAggregator
from ote_live.ingestion.gap_detector import GapDetector
from ote_live.ingestion.heartbeat import HeartbeatMonitor
from ote_live.ingestion.runtime import LiveCollectorConfig, LiveCollectorRuntime
from ote_live.ingestion.service import CollectorCycleResult, LiveBarIngestionService
from ote_live.ingestion.signals import LiveSignalProcessor
from ote_live.storage.repositories import LiveAuditRepository
from ote_live.storage.db import SQLiteLiveDataStore


def test_runtime_bootstrap_seeds_empty_database_with_recent_history() -> None:
    runtime, store, stream, backfill = _build_runtime(
        client=_FakeClient(seed_bars=[_bar(0), _bar(1), _bar(2)]),
        stream=_FakeStream([]),
        backfill=_FakeBackfill({}),
    )

    summary = asyncio.run(runtime.bootstrap())
    stored_bars = store.fetch_bars(asset="EURUSD", timeframe="1m")
    asyncio.run(runtime.close())

    assert summary.seeded_history_bars == 3
    assert summary.catchup_bars == 0
    assert len(stored_bars) == 3
    assert stream.last_emitted_timestamp == datetime(2024, 1, 2, 10, 2, tzinfo=timezone.utc)
    assert backfill.windows == []


def test_live_collector_config_defaults_to_five_minute_polls_with_heartbeat_headroom() -> None:
    config = LiveCollectorConfig()

    assert config.source_timeframe == "5m"
    assert config.poll_interval_seconds == 300.0
    assert config.heartbeat_stale_after_seconds == 330.0
    assert config.finalized_bar_grace_seconds == 90.0
    assert config.signal_processing_delay_seconds == 120.0
    assert config.cycle_finalized_refresh_lookback_bars == 6


def test_live_collector_config_preserves_explicit_heartbeat_staleness() -> None:
    config = LiveCollectorConfig(
        poll_interval_seconds=120.0,
        heartbeat_stale_after_seconds=240.0,
    )

    assert config.heartbeat_stale_after_seconds == 240.0


def test_live_collector_config_preserves_explicit_signal_processing_delay() -> None:
    config = LiveCollectorConfig(
        poll_interval_seconds=30.0,
        finalized_bar_grace_seconds=75.0,
        signal_processing_delay_seconds=150.0,
    )

    assert config.finalized_bar_grace_seconds == 75.0
    assert config.signal_processing_delay_seconds == 150.0


def test_runtime_bootstrap_recovers_from_last_stored_bar(monkeypatch) -> None:
    fixed_now = datetime(2024, 1, 2, 10, 3, tzinfo=timezone.utc)
    monkeypatch.setattr("ote_live.ingestion.runtime.utc_now", lambda: fixed_now)
    monkeypatch.setattr("ote_live.ingestion.base.utc_now", lambda: fixed_now)

    runtime, store, stream, backfill = _build_runtime(
        client=_FakeClient(seed_bars=[]),
        stream=_FakeStream([]),
        backfill=_FakeBackfill(
            {
                (datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc), fixed_now): [_bar(1), _bar(2), _bar(3)],
            }
        ),
    )
    store.upsert_bar(_bar(0))

    summary = asyncio.run(runtime.bootstrap())
    stored_bars = store.fetch_bars(asset="EURUSD", timeframe="1m")
    asyncio.run(runtime.close())

    assert summary.seeded_history_bars == 0
    assert summary.catchup_bars == 2
    assert len(stored_bars) == 3
    assert stream.last_emitted_timestamp == datetime(2024, 1, 2, 10, 2, tzinfo=timezone.utc)
    assert backfill.windows[0][0] == datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc)
    assert backfill.windows[0][1] == fixed_now


def test_runtime_bootstrap_recovers_unresolved_gaps_before_resuming(monkeypatch) -> None:
    fixed_now = datetime(2024, 1, 2, 10, 5, tzinfo=timezone.utc)
    monkeypatch.setattr("ote_live.ingestion.runtime.utc_now", lambda: fixed_now)
    monkeypatch.setattr("ote_live.ingestion.base.utc_now", lambda: fixed_now)

    runtime, store, stream, backfill = _build_runtime(
        client=_FakeClient(seed_bars=[]),
        stream=_FakeStream([]),
        backfill=_FakeBackfill(
            {
                (
                    datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc),
                ): [_bar(1)],
            }
        ),
    )
    for minute in (0, 2, 3, 4, 5):
        store.upsert_bar(_bar(minute))
    store.record_gap(
        IngestionGap(
            asset="EURUSD",
            timeframe="1m",
            expected_timestamp=datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc),
            observed_timestamp=datetime(2024, 1, 2, 10, 2, tzinfo=timezone.utc),
            missing_timestamps=[datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc)],
            gap_size=1,
        )
    )

    summary = asyncio.run(runtime.bootstrap())
    source_bars = store.fetch_bars(asset="EURUSD", timeframe="1m")
    repaired_five_minute_bars = store.fetch_bars(asset="EURUSD", timeframe="5m")
    gaps = store.fetch_gaps(asset="EURUSD", timeframe="1m", unresolved_only=False)
    asyncio.run(runtime.close())

    assert summary.startup_gap_recovery_attempts == 1
    assert summary.startup_gap_backfilled_bars == 1
    assert summary.resolved_startup_gaps == 1
    assert summary.remaining_unresolved_gaps == 0
    assert len(source_bars) == 6
    assert len(repaired_five_minute_bars) == 1
    assert repaired_five_minute_bars[0].timestamp == datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    assert gaps[0].resolved_at_utc is not None
    assert stream.last_emitted_timestamp == datetime(2024, 1, 2, 10, 4, tzinfo=timezone.utc)


def test_runtime_bootstrap_records_health_event_when_seed_fetch_fails() -> None:
    runtime, store, _, _ = _build_runtime(
        client=_FailingClient(RuntimeError("seed fetch failed")),
        stream=_FakeStream([]),
        backfill=_FakeBackfill({}),
    )
    audit = LiveAuditRepository(store)

    try:
        asyncio.run(runtime.bootstrap())
    except RuntimeError as exc:
        assert "seed fetch failed" in str(exc)
    else:
        raise AssertionError("Expected bootstrap to propagate the client failure.")

    health_events = audit.fetch_health_events(component="collector.bootstrap", event_type="bootstrap_failed")
    asyncio.run(runtime.close())

    assert len(health_events) == 1
    assert health_events[0].severity == "error"
    assert health_events[0].payload["error_type"] == "RuntimeError"


def test_runtime_run_invokes_before_close_and_closes_store_when_bootstrap_fails() -> None:
    runtime, store, _, _ = _build_runtime(
        client=_FailingClient(RuntimeError("seed fetch failed")),
        stream=_FakeStream([]),
        backfill=_FakeBackfill({}),
    )
    captured: dict[str, object] = {}

    async def _before_close(summary) -> None:
        captured["terminal_status"] = summary.terminal_status
        captured["terminal_error"] = dict(summary.terminal_error or {})

    try:
        asyncio.run(runtime.run(before_close=_before_close))
    except RuntimeError as exc:
        assert "seed fetch failed" in str(exc)
    else:
        raise AssertionError("Expected runtime.run() to propagate the bootstrap failure.")

    assert captured["terminal_status"] == "failed"
    assert captured["terminal_error"] == {
        "error_type": "RuntimeError",
        "error_message": "seed fetch failed",
    }
    try:
        store.connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError("Expected runtime.run() to close the SQLite store on failure.")


def test_runtime_bootstrap_expands_seed_history_to_cover_signal_warmup_requirement() -> None:
    client = _FakeClient(seed_bars=[_five_minute_bar(index) for index in range(260)])
    tmp_root = ROOT / "tmp" / "ote_live_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    audit = LiveAuditRepository(store)
    stream = _FakeStream([])
    backfill = _FakeBackfill({})
    service = LiveBarIngestionService(
        stream=stream,
        backfill=backfill,
        store=store,
        gap_detector=GapDetector(asset="EURUSD", timeframe="5m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("30m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-runtime"),
        audit_repository=audit,
    )
    runtime = LiveCollectorRuntime(
        config=LiveCollectorConfig(
            asset="EURUSD",
            source_timeframe="5m",
            aggregation_timeframes=("30m",),
            signal_timeframe="5m",
            startup_history_bars=240,
            poll_interval_seconds=0.0,
            max_cycles=1,
        ),
        client=client,  # type: ignore[arg-type]
        stream=stream,  # type: ignore[arg-type]
        backfill_connector=backfill,  # type: ignore[arg-type]
        store=store,
        service=service,
        audit_repository=audit,
    )
    runtime.signal_processor = _WarmupAwareSignalProcessor(minimum_runtime_history_bars=250)  # type: ignore[assignment]

    summary = asyncio.run(runtime.bootstrap())
    asyncio.run(runtime.close())

    assert summary.seeded_history_bars == 260
    assert client.last_fetch_kwargs["outputsize"] == 260


def test_runtime_bootstrap_refreshes_recent_finalized_source_bars(monkeypatch) -> None:
    fixed_now = datetime(2024, 1, 2, 10, 7, tzinfo=timezone.utc)
    monkeypatch.setattr("ote_live.ingestion.runtime.utc_now", lambda: fixed_now)
    monkeypatch.setattr("ote_live.ingestion.base.utc_now", lambda: fixed_now)

    finalized_bar = _five_minute_bar(0)
    partial_bar = finalized_bar.model_copy(update={"close": finalized_bar.open})

    tmp_root = ROOT / "tmp" / "ote_live_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    audit = LiveAuditRepository(store)
    stream = _FakeStream([])
    backfill = _FakeBackfill({})
    client = _FakeClient(seed_bars=[finalized_bar])
    service = LiveBarIngestionService(
        stream=stream,
        backfill=backfill,
        store=store,
        gap_detector=GapDetector(asset="EURUSD", timeframe="5m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("30m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-runtime"),
        audit_repository=audit,
    )
    runtime = LiveCollectorRuntime(
        config=LiveCollectorConfig(
            asset="EURUSD",
            source_timeframe="5m",
            aggregation_timeframes=("30m",),
            signal_timeframe="5m",
            poll_interval_seconds=0.0,
            max_cycles=1,
        ),
        client=client,  # type: ignore[arg-type]
        stream=stream,  # type: ignore[arg-type]
        backfill_connector=backfill,  # type: ignore[arg-type]
        store=store,
        service=service,
        audit_repository=audit,
    )
    store.upsert_bar(partial_bar)

    summary = asyncio.run(runtime.bootstrap())
    repaired_bars = store.fetch_bars(asset="EURUSD", timeframe="5m")
    asyncio.run(runtime.close())

    assert summary.stored_source_bars >= 1
    assert repaired_bars[0].close == finalized_bar.close
    assert stream.last_emitted_timestamp == finalized_bar.timestamp


def test_runtime_processes_signal_cycle_when_source_and_signal_timeframes_match() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    audit = LiveAuditRepository(store)
    stream = _FakeStream([_five_minute_bar(0)])
    backfill = _FakeBackfill({})
    signal_processor = _RecordingSignalProcessor()
    service = LiveBarIngestionService(
        stream=stream,
        backfill=backfill,
        store=store,
        gap_detector=GapDetector(asset="EURUSD", timeframe="5m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("30m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-runtime"),
        audit_repository=audit,
    )
    runtime = LiveCollectorRuntime(
        config=LiveCollectorConfig(
            asset="EURUSD",
            source_timeframe="5m",
            aggregation_timeframes=("30m",),
            signal_timeframe="5m",
            startup_history_bars=0,
            poll_interval_seconds=0.0,
            max_cycles=1,
        ),
        client=_FakeClient(seed_bars=[]),  # type: ignore[arg-type]
        stream=stream,  # type: ignore[arg-type]
        backfill_connector=backfill,  # type: ignore[arg-type]
        store=store,
        service=service,
        audit_repository=audit,
        signal_processor=signal_processor,  # type: ignore[arg-type]
    )

    summary = asyncio.run(runtime.run())

    assert summary.cycles_run == 1
    assert signal_processor.process_calls == 1


def test_runtime_signal_cycle_passes_processing_cutoff_without_new_cycle_data(monkeypatch) -> None:
    fixed_now = datetime(2024, 1, 2, 10, 7, tzinfo=timezone.utc)
    monkeypatch.setattr("ote_live.ingestion.runtime.utc_now", lambda: fixed_now)

    tmp_root = ROOT / "tmp" / "ote_live_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    audit = LiveAuditRepository(store)
    stream = _FakeStream([])
    backfill = _FakeBackfill({})
    signal_processor = _RecordingSignalProcessor()
    service = LiveBarIngestionService(
        stream=stream,
        backfill=backfill,
        store=store,
        gap_detector=GapDetector(asset="EURUSD", timeframe="5m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("30m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-runtime"),
        audit_repository=audit,
    )
    runtime = LiveCollectorRuntime(
        config=LiveCollectorConfig(
            asset="EURUSD",
            source_timeframe="5m",
            aggregation_timeframes=("30m",),
            signal_timeframe="5m",
            poll_interval_seconds=30.0,
            finalized_bar_grace_seconds=90.0,
            signal_processing_delay_seconds=120.0,
            max_cycles=1,
        ),
        client=_FakeClient(seed_bars=[]),  # type: ignore[arg-type]
        stream=stream,  # type: ignore[arg-type]
        backfill_connector=backfill,  # type: ignore[arg-type]
        store=store,
        service=service,
        audit_repository=audit,
        signal_processor=signal_processor,  # type: ignore[arg-type]
    )

    runtime._process_signal_cycle(CollectorCycleResult())
    asyncio.run(runtime.close())

    assert signal_processor.process_calls == 1
    assert signal_processor.max_timestamps == [datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)]


def test_live_signal_processor_fetch_signal_bars_after_honors_max_timestamp() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    audit = LiveAuditRepository(store)
    for index in range(3):
        store.upsert_bar(_five_minute_bar(index))

    processor = LiveSignalProcessor.__new__(LiveSignalProcessor)
    processor.audit_repository = audit
    processor.asset = "EURUSD"
    processor.timeframe = "5m"

    selected = processor._fetch_signal_bars_after(
        datetime(2024, 1, 2, 9, 55, tzinfo=timezone.utc),
        max_timestamp=datetime(2024, 1, 2, 10, 5, tzinfo=timezone.utc),
    )
    store.close()

    assert [bar.timestamp for bar in selected] == [
        datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 10, 5, tzinfo=timezone.utc),
    ]


class _FakeClient:
    def __init__(self, *, seed_bars: list[MarketBar]) -> None:
        self.seed_bars = seed_bars
        self.last_fetch_kwargs = {}

    async def fetch_historical_chart(self, **kwargs) -> list[MarketBar]:
        self.last_fetch_kwargs = dict(kwargs)
        return self.seed_bars

    async def fetch_time_series(self, **kwargs) -> list[MarketBar]:
        return await self.fetch_historical_chart(**kwargs)

    async def aclose(self) -> None:
        return None


class _FailingClient(_FakeClient):
    def __init__(self, exc: Exception) -> None:
        super().__init__(seed_bars=[])
        self.exc = exc

    async def fetch_historical_chart(self, **kwargs) -> list[MarketBar]:
        raise self.exc

    async def fetch_time_series(self, **kwargs) -> list[MarketBar]:
        return await self.fetch_historical_chart(**kwargs)


class _FakeStream:
    def __init__(self, polled_bars: list[MarketBar]) -> None:
        self.polled_bars = polled_bars
        self.last_emitted_timestamp = None

    async def poll(self) -> list[MarketBar]:
        return self.polled_bars


class _FakeBackfill:
    def __init__(self, responses: dict[tuple[datetime, datetime], list[MarketBar]]) -> None:
        self.responses = responses
        self.windows: list[tuple[datetime, datetime]] = []

    async def backfill_bars(self, window) -> list[MarketBar]:
        key = (window.start, window.end)
        self.windows.append(key)
        return self.responses.get(key, [])


class _RecordingSignalProcessor:
    def __init__(self) -> None:
        self.process_calls = 0
        self.max_timestamps: list[datetime | None] = []

    def warm_from_store(self) -> int:
        return 0

    def process_new_bars_from_store(self, *, emit_operator_artifacts: bool, max_timestamp=None):
        self.process_calls += 1
        self.max_timestamps.append(max_timestamp)
        return ()


class _WarmupAwareSignalProcessor:
    def __init__(self, *, minimum_runtime_history_bars: int) -> None:
        self.bindings = [
            SimpleNamespace(
                loaded_model=SimpleNamespace(
                    manifest=SimpleNamespace(
                        context_requirements=SimpleNamespace(
                            minimum_runtime_history_bars=minimum_runtime_history_bars
                        )
                    )
                )
            )
        ]
        self.warm_calls = 0

    def warm_from_store(self) -> int:
        self.warm_calls += 1
        return 0


def _build_runtime(
    *,
    client: _FakeClient,
    stream: _FakeStream,
    backfill: _FakeBackfill,
) -> tuple[LiveCollectorRuntime, SQLiteLiveDataStore, _FakeStream, _FakeBackfill]:
    tmp_root = ROOT / "tmp" / "ote_live_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    audit = LiveAuditRepository(store)
    service = LiveBarIngestionService(
        stream=stream,
        backfill=backfill,
        store=store,
        gap_detector=GapDetector(asset="EURUSD", timeframe="1m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("5m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-runtime"),
        audit_repository=audit,
    )
    runtime = LiveCollectorRuntime(
        config=LiveCollectorConfig(
            asset="EURUSD",
            source_timeframe="1m",
            poll_interval_seconds=0.0,
            finalized_bar_grace_seconds=0.0,
            signal_processing_delay_seconds=0.0,
            cycle_finalized_refresh_lookback_bars=0,
            max_cycles=1,
        ),
        client=client,  # type: ignore[arg-type]
        stream=stream,  # type: ignore[arg-type]
        backfill_connector=backfill,  # type: ignore[arg-type]
        store=store,
        service=service,
        audit_repository=audit,
    )
    return runtime, store, stream, backfill


def _bar(minute: int) -> MarketBar:
    price = 1.1000 + (minute * 0.0001)
    return MarketBar(
        asset="EURUSD",
        timeframe="1m",
        timestamp=datetime(2024, 1, 2, 10, minute, tzinfo=timezone.utc),
        open=price,
        high=price + 0.0003,
        low=price - 0.0002,
        close=price + 0.0001,
        volume=10.0 + minute,
    )


def _five_minute_bar(index: int) -> MarketBar:
    price = 1.1000 + (index * 0.0005)
    return MarketBar(
        asset="EURUSD",
        timeframe="5m",
        timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=index * 5),
        open=price,
        high=price + 0.0003,
        low=price - 0.0002,
        close=price + 0.0001,
        volume=20.0 + index,
    )
