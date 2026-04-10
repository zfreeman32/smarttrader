from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncIterator

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import AbstractBarStream, filter_finalized_bars
from ote_live.ingestion.connector_backfill import FMPRESTClient
from ote_live.ingestion.normalizer import (
    canonical_asset_to_fmp_symbol,
    canonical_timeframe_to_fmp_interval,
)


class FMPPollingBarStream(AbstractBarStream):
    """
    Poll finalized intraday forex candles from Financial Modeling Prep.

    FMP exposes the bars through the historical chart endpoint rather than a
    dedicated streaming socket for this workflow, so the collector polls the
    latest finalized candles and emits only newly observed timestamps.
    """

    def __init__(
        self,
        client: FMPRESTClient,
        *,
        asset: str = "EURUSD",
        timeframe: str = "5m",
        outputsize: int = 2,
        poll_interval_seconds: float = 300.0,
        last_emitted_timestamp: datetime | None = None,
        finalized_bar_grace_seconds: float = 0.0,
    ) -> None:
        self.client = client
        self.asset = asset
        self.timeframe = timeframe
        self.outputsize = outputsize
        self.poll_interval_seconds = poll_interval_seconds
        self.last_emitted_timestamp = last_emitted_timestamp
        self.finalized_bar_grace_seconds = max(0.0, float(finalized_bar_grace_seconds))

    async def poll(self) -> list[MarketBar]:
        bars = await self.client.fetch_historical_chart(
            symbol=canonical_asset_to_fmp_symbol(self.asset),
            interval=canonical_timeframe_to_fmp_interval(self.timeframe),  # type: ignore[arg-type]
            outputsize=self.outputsize,
        )
        finalized_bars = filter_finalized_bars(
            bars,
            timeframe=self.timeframe,  # type: ignore[arg-type]
            grace_period_seconds=self.finalized_bar_grace_seconds,
        )
        new_bars = [
            bar
            for bar in finalized_bars
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


# Legacy alias kept so older imports continue to work during the provider migration.
TwelveDataPollingBarStream = FMPPollingBarStream
