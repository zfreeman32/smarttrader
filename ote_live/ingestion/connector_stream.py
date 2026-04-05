from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncIterator

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import AbstractBarStream
from ote_live.ingestion.connector_backfill import TwelveDataRESTClient
from ote_live.ingestion.normalizer import (
    canonical_asset_to_twelvedata_symbol,
    canonical_timeframe_to_twelvedata_interval,
)


class TwelveDataPollingBarStream(AbstractBarStream):
    """
    Initial Phase 1 live collector built on Twelve Data long-polling.

    Twelve Data supports server-push WebSocket quotes, but this first slice keeps
    the collector deterministic by polling finalized 1-minute bars from
    `time_series` and letting the local runtime derive 5m/30m/1h bars.
    """

    def __init__(
        self,
        client: TwelveDataRESTClient,
        *,
        asset: str = "EURUSD",
        timeframe: str = "1m",
        outputsize: int = 10,
        poll_interval_seconds: float = 5.0,
        last_emitted_timestamp: datetime | None = None,
    ) -> None:
        self.client = client
        self.asset = asset
        self.timeframe = timeframe
        self.outputsize = outputsize
        self.poll_interval_seconds = poll_interval_seconds
        self.last_emitted_timestamp = last_emitted_timestamp

    async def poll(self) -> list[MarketBar]:
        bars = await self.client.fetch_time_series(
            symbol=canonical_asset_to_twelvedata_symbol(self.asset),
            interval=canonical_timeframe_to_twelvedata_interval(self.timeframe),  # type: ignore[arg-type]
            timezone=self.client.config.default_timezone,
            outputsize=self.outputsize,
            order="asc",
        )
        new_bars = [
            bar
            for bar in bars
            if self.last_emitted_timestamp is None or bar.timestamp > self.last_emitted_timestamp
        ]
        if new_bars:
            self.last_emitted_timestamp = new_bars[-1].timestamp
        return new_bars

    async def iterate(self) -> AsyncIterator[MarketBar]:
        while True:
            for bar in await self.poll():
                yield bar
            await asyncio.sleep(self.poll_interval_seconds)
