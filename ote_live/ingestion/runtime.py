from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from ote_live.ingestion.aggregator import MultiTimeframeBarAggregator
from ote_live.ingestion.base import BackfillWindow, CanonicalTimeframe, timeframe_to_timedelta, utc_now
from ote_live.ingestion.connector_backfill import (
    TwelveDataBackfillConnector,
    TwelveDataConfig,
    TwelveDataRESTClient,
)
from ote_live.ingestion.connector_stream import TwelveDataPollingBarStream
from ote_live.ingestion.normalizer import canonical_asset_to_twelvedata_symbol, canonical_timeframe_to_twelvedata_interval
from ote_live.ingestion.gap_detector import GapDetector
from ote_live.ingestion.heartbeat import HeartbeatMonitor
from ote_live.ingestion.service import CollectorCycleResult, LiveBarIngestionService
from ote_live.storage.db import SQLiteLiveDataStore

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "ote_live" / "runtime_data" / "live_market_data.sqlite3"


@dataclass(frozen=True)
class LiveCollectorConfig:
    asset: str = "EURUSD"
    source_timeframe: CanonicalTimeframe = "1m"
    aggregation_timeframes: tuple[CanonicalTimeframe, ...] = ("5m", "30m", "1h")
    db_path: Path = DEFAULT_DB_PATH
    poll_interval_seconds: float = 5.0
    stream_outputsize: int = 10
    startup_history_bars: int = 240
    startup_warmup_lookback_bars: int = 90
    heartbeat_stale_after_seconds: float = 90.0
    startup_backfill_chunk_bars: int = 1000
    default_timezone: str = "UTC"
    log_level: str = "INFO"
    max_cycles: int | None = None


@dataclass
class BootstrapSummary:
    latest_stored_timestamp: object | None = None
    seeded_history_bars: int = 0
    catchup_bars: int = 0
    stored_source_bars: int = 0
    stored_aggregated_bars: int = 0


@dataclass
class CollectorRunSummary:
    bootstrap: BootstrapSummary
    cycles_run: int = 0
    cycle_results: list[CollectorCycleResult] = field(default_factory=list)
    flushed_aggregated_bars: int = 0


class LiveCollectorRuntime:
    def __init__(
        self,
        *,
        config: LiveCollectorConfig,
        client: TwelveDataRESTClient,
        stream: TwelveDataPollingBarStream,
        backfill_connector: TwelveDataBackfillConnector,
        store: SQLiteLiveDataStore,
        service: LiveBarIngestionService,
    ) -> None:
        self.config = config
        self.client = client
        self.stream = stream
        self.backfill_connector = backfill_connector
        self.store = store
        self.service = service

    @classmethod
    def from_config(
        cls,
        config: LiveCollectorConfig,
        *,
        api_key: str | None = None,
    ) -> "LiveCollectorRuntime":
        td_config = TwelveDataConfig(
            api_key=api_key or TwelveDataConfig().api_key,
            default_timezone=config.default_timezone,
        )
        client = TwelveDataRESTClient(td_config)
        stream = TwelveDataPollingBarStream(
            client,
            asset=config.asset,
            timeframe=config.source_timeframe,
            outputsize=config.stream_outputsize,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        backfill_connector = TwelveDataBackfillConnector(client)
        store = SQLiteLiveDataStore(config.db_path)
        gap_detector = GapDetector(asset=config.asset, timeframe=config.source_timeframe)
        aggregator = MultiTimeframeBarAggregator(target_timeframes=config.aggregation_timeframes)
        heartbeat_monitor = HeartbeatMonitor(
            source="twelvedata.polling",
            stale_after=timedelta(seconds=config.heartbeat_stale_after_seconds),
        )
        service = LiveBarIngestionService(
            stream=stream,
            backfill=backfill_connector,
            store=store,
            gap_detector=gap_detector,
            aggregator=aggregator,
            heartbeat_monitor=heartbeat_monitor,
        )
        return cls(
            config=config,
            client=client,
            stream=stream,
            backfill_connector=backfill_connector,
            store=store,
            service=service,
        )

    async def bootstrap(self) -> BootstrapSummary:
        latest_stored_timestamp = self.store.get_latest_bar_timestamp(
            asset=self.config.asset,
            timeframe=self.config.source_timeframe,
        )
        summary = BootstrapSummary(latest_stored_timestamp=latest_stored_timestamp)

        if latest_stored_timestamp is None:
            seed_bars = await self.client.fetch_time_series(
                symbol=canonical_asset_to_twelvedata_symbol(self.config.asset),
                interval=canonical_timeframe_to_twelvedata_interval(self.config.source_timeframe),
                timezone=self.config.default_timezone,
                outputsize=self.config.startup_history_bars,
                order="asc",
            )
            processed = await self.service.process_bars(seed_bars)
            summary.seeded_history_bars = len(seed_bars)
            summary.stored_source_bars += processed.stored_source_bars
            summary.stored_aggregated_bars += processed.stored_aggregated_bars
            if seed_bars:
                self.stream.last_emitted_timestamp = seed_bars[-1].timestamp
            LOGGER.info(
                "Bootstrapped empty live store with %s historical %s bars for %s.",
                len(seed_bars),
                self.config.source_timeframe,
                self.config.asset,
            )
            return summary

        self._warm_runtime_state(latest_stored_timestamp)
        self.stream.last_emitted_timestamp = latest_stored_timestamp

        start = latest_stored_timestamp + timeframe_to_timedelta(self.config.source_timeframe)
        end = utc_now()
        if start > end:
            LOGGER.info("Live collector resumed at %s with no startup catch-up needed.", latest_stored_timestamp)
            return summary

        for window in self._build_catchup_windows(start=start, end=end):
            recovered_bars = await self.backfill_connector.backfill_bars(window)
            processed = await self.service.process_bars(recovered_bars)
            summary.catchup_bars += len(recovered_bars)
            summary.stored_source_bars += processed.stored_source_bars
            summary.stored_aggregated_bars += processed.stored_aggregated_bars

        refreshed_timestamp = self.store.get_latest_bar_timestamp(
            asset=self.config.asset,
            timeframe=self.config.source_timeframe,
        )
        self.stream.last_emitted_timestamp = refreshed_timestamp or latest_stored_timestamp
        LOGGER.info(
            "Recovered %s startup catch-up bars for %s since %s.",
            summary.catchup_bars,
            self.config.asset,
            latest_stored_timestamp,
        )
        return summary

    async def run(self) -> CollectorRunSummary:
        bootstrap_summary = await self.bootstrap()
        summary = CollectorRunSummary(bootstrap=bootstrap_summary)
        cycle_index = 0
        try:
            while self.config.max_cycles is None or cycle_index < self.config.max_cycles:
                cycle_index += 1
                result = await self.service.collect_once()
                summary.cycles_run = cycle_index
                summary.cycle_results.append(result)
                LOGGER.info(
                    "Collector cycle %s: polled=%s stored_source=%s stored_aggregated=%s "
                    "backfilled=%s gaps=%s dup=%s out_of_order=%s",
                    cycle_index,
                    result.polled_bars,
                    result.stored_source_bars,
                    result.stored_aggregated_bars,
                    result.backfilled_bars,
                    result.gaps_detected,
                    result.duplicates,
                    result.out_of_order,
                )
                if self.config.max_cycles is not None and cycle_index >= self.config.max_cycles:
                    break
                await asyncio.sleep(self.config.poll_interval_seconds)
        finally:
            summary.flushed_aggregated_bars = self.service.flush()
            await self.close()
        return summary

    async def close(self) -> None:
        await self.client.aclose()
        self.store.close()

    def _warm_runtime_state(self, latest_stored_timestamp) -> None:
        lookback = timeframe_to_timedelta("1h") + (
            timeframe_to_timedelta(self.config.source_timeframe) * self.config.startup_warmup_lookback_bars
        )
        start = latest_stored_timestamp - lookback
        history = self.store.fetch_bars(
            asset=self.config.asset,
            timeframe=self.config.source_timeframe,
            start=start,
            end=latest_stored_timestamp,
        )
        for bar in history:
            self.service.gap_detector.observe(bar)
            self.service.aggregator.ingest_bar(bar)

    def _build_catchup_windows(self, *, start, end) -> list[BackfillWindow]:
        if start > end:
            return []

        step = timeframe_to_timedelta(self.config.source_timeframe)
        chunk_span = step * max(1, self.config.startup_backfill_chunk_bars - 1)
        windows: list[BackfillWindow] = []
        cursor = start
        while cursor <= end:
            window_end = min(end, cursor + chunk_span)
            windows.append(
                BackfillWindow(
                    asset=self.config.asset,
                    timeframe=self.config.source_timeframe,
                    start=cursor,
                    end=window_end,
                )
            )
            cursor = window_end + step
        return windows
