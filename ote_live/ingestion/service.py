from __future__ import annotations

from dataclasses import dataclass

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.aggregator import MultiTimeframeBarAggregator
from ote_live.ingestion.base import AbstractBackfillConnector, AbstractBarStream
from ote_live.ingestion.gap_detector import GapDetector
from ote_live.ingestion.heartbeat import HeartbeatMonitor
from ote_live.storage.db import SQLiteLiveDataStore


@dataclass
class CollectorCycleResult:
    polled_bars: int = 0
    stored_source_bars: int = 0
    stored_aggregated_bars: int = 0
    backfilled_bars: int = 0
    gaps_detected: int = 0
    duplicates: int = 0
    out_of_order: int = 0


class LiveBarIngestionService:
    def __init__(
        self,
        *,
        stream: AbstractBarStream,
        backfill: AbstractBackfillConnector,
        store: SQLiteLiveDataStore,
        gap_detector: GapDetector,
        aggregator: MultiTimeframeBarAggregator,
        heartbeat_monitor: HeartbeatMonitor,
    ) -> None:
        self.stream = stream
        self.backfill = backfill
        self.store = store
        self.gap_detector = gap_detector
        self.aggregator = aggregator
        self.heartbeat_monitor = heartbeat_monitor

    async def collect_once(self) -> CollectorCycleResult:
        result = CollectorCycleResult()
        new_bars = await self.stream.poll()
        result.polled_bars = len(new_bars)

        heartbeat_status = (
            self.heartbeat_monitor.beat() if new_bars else self.heartbeat_monitor.snapshot()
        )
        self.store.record_heartbeat(
            heartbeat_status,
            metadata={"polled_bars": len(new_bars)},
        )

        processing_result = await self.process_bars(new_bars)
        result.stored_source_bars = processing_result.stored_source_bars
        result.stored_aggregated_bars = processing_result.stored_aggregated_bars
        result.backfilled_bars = processing_result.backfilled_bars
        result.gaps_detected = processing_result.gaps_detected
        result.duplicates = processing_result.duplicates
        result.out_of_order = processing_result.out_of_order
        return result

    async def process_bars(self, bars: list[MarketBar]) -> CollectorCycleResult:
        result = CollectorCycleResult(polled_bars=len(bars))

        for bar in sorted(bars, key=lambda item: item.timestamp):
            observation = self.gap_detector.observe(bar)
            if observation.gap is not None:
                result.gaps_detected += 1
                self.store.record_gap(observation.gap)
                for missing_bar in await self.backfill.backfill_bars(observation.gap.to_backfill_window()):
                    self._store_source_bar(missing_bar)
                    result.backfilled_bars += 1
                    result.stored_source_bars += 1
                    result.stored_aggregated_bars += self._store_aggregated_bars(missing_bar)

            if observation.duplicate:
                result.duplicates += 1
                continue

            if observation.out_of_order:
                result.out_of_order += 1
                continue

            self._store_source_bar(bar)
            result.stored_source_bars += 1
            result.stored_aggregated_bars += self._store_aggregated_bars(bar)

        return result

    def flush(self) -> int:
        stored = 0
        for aggregated_bar in self.aggregator.flush():
            self.store.upsert_bar(aggregated_bar)
            stored += 1
        return stored

    def _store_source_bar(self, bar: MarketBar) -> None:
        self.store.upsert_bar(bar)

    def _store_aggregated_bars(self, bar: MarketBar) -> int:
        stored = 0
        for aggregated_bar in self.aggregator.ingest_bar(bar):
            self.store.upsert_bar(aggregated_bar)
            stored += 1
        return stored
