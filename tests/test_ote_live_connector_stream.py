from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.connector_stream import FMPPollingBarStream


def test_polling_stream_waits_for_bar_finalization(monkeypatch) -> None:
    client = _FakeClient(
        [
            _five_minute_bar(datetime(2024, 1, 2, 10, 0, tzinfo=UTC)),
            _five_minute_bar(datetime(2024, 1, 2, 10, 5, tzinfo=UTC)),
        ]
    )
    stream = FMPPollingBarStream(
        client,  # type: ignore[arg-type]
        asset="EURUSD",
        timeframe="5m",
        outputsize=2,
        poll_interval_seconds=30.0,
    )

    monkeypatch.setattr("ote_live.ingestion.base.utc_now", lambda: datetime(2024, 1, 2, 10, 7, tzinfo=UTC))
    first_poll = asyncio.run(stream.poll())

    assert [bar.timestamp for bar in first_poll] == [datetime(2024, 1, 2, 10, 0, tzinfo=UTC)]
    assert stream.last_emitted_timestamp == datetime(2024, 1, 2, 10, 0, tzinfo=UTC)

    monkeypatch.setattr("ote_live.ingestion.base.utc_now", lambda: datetime(2024, 1, 2, 10, 10, tzinfo=UTC))
    second_poll = asyncio.run(stream.poll())

    assert [bar.timestamp for bar in second_poll] == [datetime(2024, 1, 2, 10, 5, tzinfo=UTC)]
    assert stream.last_emitted_timestamp == datetime(2024, 1, 2, 10, 5, tzinfo=UTC)


class _FakeClient:
    def __init__(self, bars: list[MarketBar]) -> None:
        self.bars = list(bars)

    async def fetch_historical_chart(self, **kwargs) -> list[MarketBar]:
        return list(self.bars)


def _five_minute_bar(timestamp: datetime) -> MarketBar:
    return MarketBar(
        asset="EURUSD",
        timeframe="5m",
        timestamp=timestamp,
        open=1.1000,
        high=1.1004,
        low=1.0998,
        close=1.1002,
        volume=10.0,
    )
