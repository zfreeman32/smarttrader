from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.aggregator import MultiTimeframeBarAggregator
from ote_live.ingestion.gap_detector import GapDetector
from ote_live.ingestion.heartbeat import HeartbeatMonitor
from ote_live.ingestion.runtime import LiveCollectorConfig, LiveCollectorRuntime
from ote_live.ingestion.service import LiveBarIngestionService
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


def test_runtime_bootstrap_recovers_from_last_stored_bar(monkeypatch) -> None:
    fixed_now = datetime(2024, 1, 2, 10, 3, tzinfo=timezone.utc)
    monkeypatch.setattr("ote_live.ingestion.runtime.utc_now", lambda: fixed_now)

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
    assert summary.catchup_bars == 3
    assert len(stored_bars) == 4
    assert stream.last_emitted_timestamp == fixed_now
    assert backfill.windows[0][0] == datetime(2024, 1, 2, 10, 1, tzinfo=timezone.utc)
    assert backfill.windows[0][1] == fixed_now


class _FakeClient:
    def __init__(self, *, seed_bars: list[MarketBar]) -> None:
        self.seed_bars = seed_bars

    async def fetch_time_series(self, **kwargs) -> list[MarketBar]:
        return self.seed_bars

    async def aclose(self) -> None:
        return None


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


def _build_runtime(
    *,
    client: _FakeClient,
    stream: _FakeStream,
    backfill: _FakeBackfill,
) -> tuple[LiveCollectorRuntime, SQLiteLiveDataStore, _FakeStream, _FakeBackfill]:
    tmp_root = ROOT / "tmp" / "ote_live_runtime_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(tmp_root / f"{uuid.uuid4().hex}.sqlite")
    service = LiveBarIngestionService(
        stream=stream,
        backfill=backfill,
        store=store,
        gap_detector=GapDetector(asset="EURUSD", timeframe="1m"),
        aggregator=MultiTimeframeBarAggregator(target_timeframes=("5m",)),
        heartbeat_monitor=HeartbeatMonitor(source="fake-runtime"),
    )
    runtime = LiveCollectorRuntime(
        config=LiveCollectorConfig(
            asset="EURUSD",
            source_timeframe="1m",
            poll_interval_seconds=0.0,
            max_cycles=1,
        ),
        client=client,  # type: ignore[arg-type]
        stream=stream,  # type: ignore[arg-type]
        backfill_connector=backfill,  # type: ignore[arg-type]
        store=store,
        service=service,
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
