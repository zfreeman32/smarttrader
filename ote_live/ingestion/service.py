from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.aggregator import (
    MultiTimeframeBarAggregator,
    aggregate_bar_sequence,
    floor_timestamp_to_timeframe,
)
from ote_live.ingestion.base import AbstractBackfillConnector, AbstractBarStream, IngestionGap, ensure_utc, timeframe_to_timedelta
from ote_live.ingestion.gap_detector import GapDetector
from ote_live.ingestion.heartbeat import HeartbeatMonitor
from ote_live.storage.repositories import LiveAuditRepository
from ote_live.storage.db import SQLiteLiveDataStore, StoredIngestionGap


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
        audit_repository: LiveAuditRepository | None = None,
    ) -> None:
        self.stream = stream
        self.backfill = backfill
        self.store = store
        self.gap_detector = gap_detector
        self.aggregator = aggregator
        self.heartbeat_monitor = heartbeat_monitor
        self.audit_repository = audit_repository

    async def collect_once(self) -> CollectorCycleResult:
        result = CollectorCycleResult()
        try:
            new_bars = await self.stream.poll()
        except Exception as exc:
            self._record_health_event(
                component="ingestion.stream",
                event_type="poll_failed",
                severity="error",
                message="Polling the live bar stream failed.",
                payload={"error_type": type(exc).__name__, "error_message": str(exc)},
            )
            raise
        result.polled_bars = len(new_bars)

        heartbeat_status = (
            self.heartbeat_monitor.beat() if new_bars else self.heartbeat_monitor.snapshot()
        )
        self.store.record_heartbeat(
            heartbeat_status,
            metadata={"polled_bars": len(new_bars)},
        )
        self._record_heartbeat_health_event(heartbeat_status, polled_bars=len(new_bars))

        try:
            processing_result = await self.process_bars(new_bars)
        except Exception as exc:
            self._record_health_event(
                component="ingestion.processing",
                event_type="process_bars_failed",
                severity="error",
                message="Processing live bars failed.",
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "polled_bars": len(new_bars),
                },
            )
            raise
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
                gap_id = self.store.record_gap(observation.gap)
                self._record_health_event(
                    component="ingestion.gap_detector",
                    event_type="gap_detected",
                    severity="warning",
                    message="Detected a gap in the live ingestion stream.",
                    payload={
                        "asset": observation.gap.asset,
                        "timeframe": observation.gap.timeframe,
                        "gap_size": observation.gap.gap_size,
                        "expected_timestamp_utc": observation.gap.expected_timestamp.isoformat(),
                        "observed_timestamp_utc": observation.gap.observed_timestamp.isoformat(),
                    },
                )
                try:
                    recovered_bars = await self.backfill.backfill_bars(observation.gap.to_backfill_window())
                except Exception as exc:
                    self._record_health_event(
                        component="ingestion.backfill",
                        event_type="backfill_failed",
                        severity="error",
                        message="Backfill failed for a detected ingestion gap.",
                        payload={
                            "asset": observation.gap.asset,
                            "timeframe": observation.gap.timeframe,
                            "gap_size": observation.gap.gap_size,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    raise
                gap_recovery_bars = self._prepare_gap_recovery_bars(
                    gap=observation.gap,
                    recovered_bars=recovered_bars,
                    existing_bars=(),
                    component="ingestion.backfill",
                    gap_id=gap_id,
                )
                for missing_bar in gap_recovery_bars:
                    self._store_source_bar(missing_bar)
                    result.backfilled_bars += 1
                    result.stored_source_bars += 1
                    result.stored_aggregated_bars += self._store_aggregated_bars(missing_bar)
                self.store.mark_gap_resolved(gap_id)
                self._record_gap_resolved(
                    component="ingestion.backfill",
                    gap_id=gap_id,
                    gap=observation.gap,
                    recovered_bars=gap_recovery_bars,
                    repaired_aggregated_bars=0,
                    recovery_source="live_backfill",
                )

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

    async def recover_unresolved_gaps(self, gaps: list[StoredIngestionGap]) -> CollectorCycleResult:
        result = CollectorCycleResult()
        for gap in sorted(gaps, key=lambda item: item.expected_timestamp):
            existing_gap_bars = self._fetch_gap_source_bars(gap)
            complete_gap_bars: list[MarketBar]
            new_gap_bars: list[MarketBar] = []
            recovery_source = "existing_store"
            if self._has_full_gap_coverage(gap, existing_gap_bars):
                complete_gap_bars = list(existing_gap_bars)
            else:
                try:
                    recovered_bars = await self.backfill.backfill_bars(gap.to_backfill_window())
                except Exception as exc:
                    self._record_health_event(
                        component="collector.bootstrap",
                        event_type="startup_gap_recovery_failed",
                        severity="error",
                        message="Startup unresolved-gap recovery failed during backfill.",
                        payload={
                            "gap_id": gap.gap_id,
                            "asset": gap.asset,
                            "timeframe": gap.timeframe,
                            "gap_size": gap.gap_size,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    raise
                complete_gap_bars, new_gap_bars = self._partition_gap_recovery_bars(
                    gap=gap,
                    recovered_bars=recovered_bars,
                    existing_bars=existing_gap_bars,
                    component="collector.bootstrap",
                    gap_id=gap.gap_id,
                )
                recovery_source = "startup_backfill"

            for recovered_bar in new_gap_bars:
                self._store_source_bar(recovered_bar)
                result.backfilled_bars += 1
                result.stored_source_bars += 1

            repaired_aggregated_bars = self._repair_aggregates_for_gap_bars(complete_gap_bars)
            result.stored_aggregated_bars += repaired_aggregated_bars
            self.store.mark_gap_resolved(gap.gap_id)
            self._record_gap_resolved(
                component="collector.bootstrap",
                gap_id=gap.gap_id,
                gap=gap.to_ingestion_gap(),
                recovered_bars=complete_gap_bars,
                repaired_aggregated_bars=repaired_aggregated_bars,
                recovery_source=recovery_source,
            )
        return result

    def flush(self) -> int:
        stored = 0
        for aggregated_bar in self.aggregator.flush():
            self.store.upsert_bar(aggregated_bar)
            stored += 1
        return stored

    def repair_source_bars(self, bars: list[MarketBar]) -> CollectorCycleResult:
        """Overwrite recent source bars with finalized values and rebuild dependent aggregates."""
        if not bars:
            return CollectorCycleResult()

        deduped_by_timestamp: dict[datetime, MarketBar] = {}
        for bar in sorted(bars, key=lambda item: item.timestamp):
            deduped_by_timestamp[bar.timestamp] = bar
        repaired_bars = [deduped_by_timestamp[timestamp] for timestamp in sorted(deduped_by_timestamp)]

        result = CollectorCycleResult(polled_bars=len(repaired_bars))
        for bar in repaired_bars:
            self._store_source_bar(bar)
            result.stored_source_bars += 1
        result.stored_aggregated_bars += self._repair_aggregates_for_gap_bars(repaired_bars)
        return result

    def _store_source_bar(self, bar: MarketBar) -> None:
        self.store.upsert_bar(bar)

    def _store_aggregated_bars(self, bar: MarketBar) -> int:
        stored = 0
        for aggregated_bar in self.aggregator.ingest_bar(bar):
            self.store.upsert_bar(aggregated_bar)
            stored += 1
        return stored

    def _fetch_gap_source_bars(self, gap: IngestionGap | StoredIngestionGap) -> tuple[MarketBar, ...]:
        if not gap.missing_timestamps:
            return ()
        relevant_bars = self.store.fetch_bars(
            asset=gap.asset,
            timeframe=gap.timeframe,
            start=gap.missing_timestamps[0],
            end=gap.missing_timestamps[-1],
        )
        missing_timestamp_set = {ensure_utc(item) for item in gap.missing_timestamps}
        filtered = [bar for bar in relevant_bars if ensure_utc(bar.timestamp) in missing_timestamp_set]
        return tuple(sorted(filtered, key=lambda item: item.timestamp))

    def _prepare_gap_recovery_bars(
        self,
        *,
        gap: IngestionGap | StoredIngestionGap,
        recovered_bars: list[MarketBar],
        existing_bars: tuple[MarketBar, ...],
        component: str,
        gap_id: int,
    ) -> list[MarketBar]:
        complete_gap_bars, new_gap_bars = self._partition_gap_recovery_bars(
            gap=gap,
            recovered_bars=recovered_bars,
            existing_bars=existing_bars,
            component=component,
            gap_id=gap_id,
        )
        return new_gap_bars if isinstance(gap, IngestionGap) else complete_gap_bars

    def _partition_gap_recovery_bars(
        self,
        *,
        gap: IngestionGap | StoredIngestionGap,
        recovered_bars: list[MarketBar],
        existing_bars: tuple[MarketBar, ...],
        component: str,
        gap_id: int,
    ) -> tuple[list[MarketBar], list[MarketBar]]:
        existing_by_timestamp = {
            ensure_utc(bar.timestamp): bar
            for bar in existing_bars
        }
        recovered_by_timestamp: dict[datetime, MarketBar] = {}
        missing_timestamp_set = {ensure_utc(item) for item in gap.missing_timestamps}
        for bar in recovered_bars:
            timestamp = ensure_utc(bar.timestamp)
            if timestamp in missing_timestamp_set:
                recovered_by_timestamp[timestamp] = bar

        merged = dict(existing_by_timestamp)
        merged.update(recovered_by_timestamp)
        missing_after_recovery = [
            timestamp.isoformat()
            for timestamp in sorted(missing_timestamp_set)
            if timestamp not in merged
        ]
        if missing_after_recovery:
            self._record_health_event(
                component=component,
                event_type="gap_recovery_incomplete",
                severity="error",
                message="Gap recovery did not return every expected missing bar.",
                payload={
                    "gap_id": gap_id,
                    "asset": gap.asset,
                    "timeframe": gap.timeframe,
                    "gap_size": gap.gap_size,
                    "missing_after_recovery": missing_after_recovery,
                    "recovered_bar_count": len(recovered_by_timestamp),
                    "existing_bar_count": len(existing_by_timestamp),
                },
            )
            raise RuntimeError(
                f"Incomplete gap recovery for gap_id={gap_id}: missing {len(missing_after_recovery)} expected bars."
            )

        complete_gap_bars = [
            merged[timestamp]
            for timestamp in sorted(missing_timestamp_set)
        ]
        new_gap_bars = [
            recovered_by_timestamp[timestamp]
            for timestamp in sorted(recovered_by_timestamp)
            if timestamp not in existing_by_timestamp
        ]
        return complete_gap_bars, new_gap_bars

    def _has_full_gap_coverage(self, gap: IngestionGap | StoredIngestionGap, bars: tuple[MarketBar, ...]) -> bool:
        covered_timestamps = {ensure_utc(bar.timestamp) for bar in bars}
        return all(ensure_utc(timestamp) in covered_timestamps for timestamp in gap.missing_timestamps)

    def _repair_aggregates_for_gap_bars(self, bars: list[MarketBar]) -> int:
        if not bars:
            return 0

        asset = bars[0].asset
        source_timeframe = bars[0].timeframe
        source_step = timeframe_to_timedelta(source_timeframe)  # type: ignore[arg-type]
        latest_source_timestamp = self.store.get_latest_bar_timestamp(
            asset=asset,
            timeframe=source_timeframe,
        )
        if latest_source_timestamp is None:
            return 0

        stored = 0
        for target_timeframe in self.aggregator.target_timeframes:
            bucket_starts = sorted(
                {
                    floor_timestamp_to_timeframe(bar.timestamp, target_timeframe)
                    for bar in bars
                }
            )
            if not bucket_starts:
                continue
            start = bucket_starts[0]
            end = bucket_starts[-1] + timeframe_to_timedelta(target_timeframe) - source_step
            source_bars = self.store.fetch_bars(
                asset=asset,
                timeframe=source_timeframe,
                start=start,
                end=end,
            )
            if not source_bars:
                continue
            repaired_bars = aggregate_bar_sequence(
                source_bars,
                target_timeframe=target_timeframe,
                source=f"{self.aggregator.source_name}:repair:{source_timeframe}->{target_timeframe}",
            )
            for repaired_bar in repaired_bars:
                finalized_cutoff = repaired_bar.timestamp + timeframe_to_timedelta(target_timeframe)
                if (
                    repaired_bar.timestamp not in bucket_starts
                    or latest_source_timestamp < finalized_cutoff
                ):
                    continue
                self.store.upsert_bar(repaired_bar)
                stored += 1
        return stored

    def _record_gap_resolved(
        self,
        *,
        component: str,
        gap_id: int,
        gap: IngestionGap,
        recovered_bars: list[MarketBar],
        repaired_aggregated_bars: int,
        recovery_source: str,
    ) -> None:
        self._record_health_event(
            component=component,
            event_type="gap_resolved",
            severity="info",
            message="Recovered every missing bar for a recorded ingestion gap.",
            payload={
                "gap_id": gap_id,
                "asset": gap.asset,
                "timeframe": gap.timeframe,
                "gap_size": gap.gap_size,
                "recovered_bar_count": len(recovered_bars),
                "repaired_aggregated_bars": int(repaired_aggregated_bars),
                "recovery_source": recovery_source,
            },
        )

    def _record_heartbeat_health_event(self, heartbeat_status, *, polled_bars: int) -> None:
        event_type = "heartbeat_stale" if heartbeat_status.is_stale else "heartbeat_ok"
        severity = "warning" if heartbeat_status.is_stale else "info"
        message = (
            "Heartbeat marked the live ingestion feed as stale."
            if heartbeat_status.is_stale
            else "Heartbeat recorded live ingestion activity."
        )
        self._record_health_event(
            component="ingestion.heartbeat",
            event_type=event_type,
            severity=severity,
            message=message,
            payload={
                "source": heartbeat_status.source,
                "observed_at_utc": heartbeat_status.observed_at_utc.isoformat(),
                "lag_seconds": heartbeat_status.lag_seconds,
                "stale_after_seconds": heartbeat_status.stale_after_seconds,
                "polled_bars": int(polled_bars),
            },
        )

    def _record_health_event(
        self,
        *,
        component: str,
        event_type: str,
        severity: str,
        message: str,
        payload: dict[str, Any],
    ) -> None:
        if self.audit_repository is None:
            return
        self.audit_repository.record_health_event(
            component=component,
            event_type=event_type,
            severity=severity,
            message=message,
            payload=payload,
        )
