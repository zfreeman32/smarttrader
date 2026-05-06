from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

from ote_live.alerts import (
    EmailGatewaySmsTransport,
    LiveSignalEmailer,
    LiveSignalSmsSender,
    NotificationThrottle,
    RoutedSmsTransport,
    SmtpEmailTransport,
    TwilioSmsTransport,
    is_email_gateway_sms_recipient,
)
from ote_live.ingestion.aggregator import MultiTimeframeBarAggregator
from ote_live.ingestion.base import (
    BackfillWindow,
    CanonicalTimeframe,
    filter_finalized_bars,
    latest_finalized_bar_start,
    timeframe_to_timedelta,
    utc_now,
)
from ote_live.ingestion.connector_backfill import (
    FMPBackfillConnector,
    FMPConfig,
    FMPRESTClient,
)
from ote_live.ingestion.connector_stream import FMPPollingBarStream
from ote_live.ingestion.normalizer import canonical_asset_to_fmp_symbol, canonical_timeframe_to_fmp_interval
from ote_live.ingestion.gap_detector import GapDetector
from ote_live.ingestion.heartbeat import HeartbeatMonitor
from ote_live.ingestion.signals import LiveSignalProcessor, RuntimeSignalResult
from ote_live.ingestion.service import CollectorCycleResult, LiveBarIngestionService
from ote_live.media import PillowSignalAnnotator, PlotlySignalChartRenderer, SignalChartCaptureService
from ote_live.storage.db import SQLiteLiveDataStore
from ote_live.storage.repositories import LiveAuditRepository

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "ote_live" / "runtime_data" / "live_market_data.sqlite3"
DEFAULT_LONG_RUNTIME_MANIFEST_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_long.json"
DEFAULT_SHORT_RUNTIME_MANIFEST_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_short.json"
DEFAULT_SIGNAL_CHART_OUTPUT_ROOT = REPO_ROOT / "ote_live" / "runtime_data" / "charts"
DEFAULT_COLLECTOR_POLL_INTERVAL_SECONDS = 300.0
DEFAULT_HEARTBEAT_STALE_GRACE_SECONDS = 30.0
DEFAULT_FINALIZED_BAR_GRACE_SECONDS = 90.0
DEFAULT_SIGNAL_PROCESSING_EXTRA_DELAY_SECONDS = 30.0
DEFAULT_CYCLE_FINALIZED_REFRESH_LOOKBACK_BARS = 6


def resolve_heartbeat_stale_after_seconds(
    poll_interval_seconds: float,
    source_timeframe: CanonicalTimeframe = "5m",
    explicit_stale_after_seconds: float | None = None,
) -> float:
    if explicit_stale_after_seconds is not None:
        return float(explicit_stale_after_seconds)
    source_timeframe_seconds = timeframe_to_timedelta(source_timeframe).total_seconds()
    # For direct 5m ingestion, receiving no new bars for a couple of minutes is normal.
    return max(
        90.0,
        max(float(source_timeframe_seconds), float(poll_interval_seconds)) + DEFAULT_HEARTBEAT_STALE_GRACE_SECONDS,
    )


@dataclass(frozen=True)
class LiveCollectorConfig:
    asset: str = "EURUSD"
    source_timeframe: CanonicalTimeframe = "5m"
    aggregation_timeframes: tuple[CanonicalTimeframe, ...] = ("5m", "30m", "1h")
    db_path: Path = DEFAULT_DB_PATH
    poll_interval_seconds: float = DEFAULT_COLLECTOR_POLL_INTERVAL_SECONDS
    stream_outputsize: int = 2
    finalized_bar_grace_seconds: float = DEFAULT_FINALIZED_BAR_GRACE_SECONDS
    signal_processing_delay_seconds: float | None = None
    cycle_finalized_refresh_lookback_bars: int = DEFAULT_CYCLE_FINALIZED_REFRESH_LOOKBACK_BARS
    startup_history_bars: int = 240
    startup_warmup_lookback_bars: int = 90
    heartbeat_stale_after_seconds: float | None = None
    startup_backfill_chunk_bars: int = 1000
    default_timezone: str = "UTC"
    log_level: str = "INFO"
    max_cycles: int | None = None
    enable_signal_runtime: bool = True
    signal_timeframe: CanonicalTimeframe = "5m"
    long_runtime_manifest_path: Path = DEFAULT_LONG_RUNTIME_MANIFEST_PATH
    short_runtime_manifest_path: Path = DEFAULT_SHORT_RUNTIME_MANIFEST_PATH
    skip_unavailable_model_backends: bool = True
    enable_signal_chart_capture: bool = True
    signal_chart_output_root: Path = DEFAULT_SIGNAL_CHART_OUTPUT_ROOT
    signal_chart_lookback_bars: int = 120
    dashboard_url: str | None = None
    alert_email_recipients: tuple[str, ...] = ()
    alert_sms_recipients: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        poll_interval_seconds = max(0.0, float(self.poll_interval_seconds))
        object.__setattr__(self, "poll_interval_seconds", poll_interval_seconds)
        finalized_bar_grace_seconds = max(0.0, float(self.finalized_bar_grace_seconds))
        object.__setattr__(self, "finalized_bar_grace_seconds", finalized_bar_grace_seconds)
        signal_processing_delay_seconds = self.signal_processing_delay_seconds
        if signal_processing_delay_seconds is None:
            signal_processing_delay_seconds = finalized_bar_grace_seconds + min(
                poll_interval_seconds,
                DEFAULT_SIGNAL_PROCESSING_EXTRA_DELAY_SECONDS,
            )
        object.__setattr__(
            self,
            "signal_processing_delay_seconds",
            max(0.0, float(signal_processing_delay_seconds)),
        )
        object.__setattr__(
            self,
            "cycle_finalized_refresh_lookback_bars",
            max(0, int(self.cycle_finalized_refresh_lookback_bars)),
        )
        resolved_aggregation_timeframes = tuple(
            dict.fromkeys(
                timeframe
                for timeframe in self.aggregation_timeframes
                if timeframe != self.source_timeframe
            )
        )
        object.__setattr__(self, "aggregation_timeframes", resolved_aggregation_timeframes)
        object.__setattr__(
            self,
            "heartbeat_stale_after_seconds",
            resolve_heartbeat_stale_after_seconds(
                poll_interval_seconds,
                self.source_timeframe,
                self.heartbeat_stale_after_seconds,
            ),
        )


@dataclass
class BootstrapSummary:
    latest_stored_timestamp: object | None = None
    seeded_history_bars: int = 0
    catchup_bars: int = 0
    stored_source_bars: int = 0
    stored_aggregated_bars: int = 0
    startup_gap_recovery_attempts: int = 0
    startup_gap_backfilled_bars: int = 0
    resolved_startup_gaps: int = 0
    remaining_unresolved_gaps: int = 0


@dataclass
class CollectorRunSummary:
    bootstrap: BootstrapSummary
    cycles_run: int = 0
    cycle_results: list[CollectorCycleResult] = field(default_factory=list)
    signal_results: list[tuple[RuntimeSignalResult, ...]] = field(default_factory=list)
    evaluated_signals: int = 0
    emitted_signals: int = 0
    sent_notifications: int = 0
    captured_media_artifacts: int = 0
    flushed_aggregated_bars: int = 0
    terminal_status: str = "running"
    terminal_error: dict[str, Any] | None = None


RuntimeSummaryHook = Callable[[CollectorRunSummary], Any | Awaitable[Any]]
RuntimeCycleHook = Callable[[CollectorRunSummary, CollectorCycleResult, int], Any | Awaitable[Any]]


class LiveCollectorRuntime:
    def __init__(
        self,
        *,
        config: LiveCollectorConfig,
        client: FMPRESTClient,
        stream: FMPPollingBarStream,
        backfill_connector: FMPBackfillConnector,
        store: SQLiteLiveDataStore,
        service: LiveBarIngestionService,
        audit_repository: LiveAuditRepository | None = None,
        signal_processor: LiveSignalProcessor | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.stream = stream
        self.backfill_connector = backfill_connector
        self.store = store
        self.service = service
        self.audit_repository = audit_repository or service.audit_repository
        self.signal_processor = signal_processor

    @classmethod
    def from_config(
        cls,
        config: LiveCollectorConfig,
        *,
        api_key: str | None = None,
        signal_emailer: LiveSignalEmailer | None = None,
        signal_sms_sender: LiveSignalSmsSender | None = None,
        signal_chart_capture_service: SignalChartCaptureService | None = None,
    ) -> "LiveCollectorRuntime":
        fmp_config = FMPConfig(
            api_key=api_key or FMPConfig().api_key,
            default_timezone=config.default_timezone,
        )
        client = FMPRESTClient(fmp_config)
        stream = FMPPollingBarStream(
            client,
            asset=config.asset,
            timeframe=config.source_timeframe,
            outputsize=config.stream_outputsize,
            poll_interval_seconds=config.poll_interval_seconds,
            finalized_bar_grace_seconds=config.finalized_bar_grace_seconds,
        )
        backfill_connector = FMPBackfillConnector(client)
        store = SQLiteLiveDataStore(config.db_path)
        audit_repository = LiveAuditRepository(store)
        gap_detector = GapDetector(asset=config.asset, timeframe=config.source_timeframe)
        aggregator = MultiTimeframeBarAggregator(target_timeframes=config.aggregation_timeframes)
        heartbeat_monitor = HeartbeatMonitor(
            source="fmp.polling",
            stale_after=timedelta(seconds=config.heartbeat_stale_after_seconds),
        )
        service = LiveBarIngestionService(
            stream=stream,
            backfill=backfill_connector,
            store=store,
            gap_detector=gap_detector,
            aggregator=aggregator,
            heartbeat_monitor=heartbeat_monitor,
            audit_repository=audit_repository,
        )
        resolved_emailer = signal_emailer
        if resolved_emailer is None and config.alert_email_recipients:
            try:
                resolved_emailer = LiveSignalEmailer(
                    audit_repository=audit_repository,
                    throttle=NotificationThrottle(store),
                    transport=SmtpEmailTransport.from_environment(),
                    recipients=config.alert_email_recipients,
                )
            except Exception as exc:
                LOGGER.warning("Email alerts disabled: %s", exc)

        resolved_sms_sender = signal_sms_sender
        if resolved_sms_sender is None and config.alert_sms_recipients:
            try:
                resolved_sms_sender = LiveSignalSmsSender(
                    audit_repository=audit_repository,
                    throttle=NotificationThrottle(store),
                    transport=_build_sms_transport(config.alert_sms_recipients),
                    recipients=config.alert_sms_recipients,
                )
            except Exception as exc:
                LOGGER.warning("SMS alerts disabled: %s", exc)

        resolved_chart_capture_service = signal_chart_capture_service
        if resolved_chart_capture_service is None and config.enable_signal_chart_capture:
            annotator = None
            try:
                annotator = PillowSignalAnnotator()
            except ImportError:
                annotator = None
            resolved_chart_capture_service = SignalChartCaptureService(
                audit_repository=audit_repository,
                output_root=config.signal_chart_output_root,
                renderer=PlotlySignalChartRenderer(),
                annotator=annotator,
                lookback_bars=config.signal_chart_lookback_bars,
            )

        signal_processor = None
        if config.enable_signal_runtime:
            try:
                signal_processor = LiveSignalProcessor.from_direction_manifest_paths(
                    audit_repository=audit_repository,
                    long_manifest_path=config.long_runtime_manifest_path,
                    short_manifest_path=config.short_runtime_manifest_path,
                    emailer=resolved_emailer,
                    sms_sender=resolved_sms_sender,
                    chart_capture_service=resolved_chart_capture_service,
                    dashboard_url=config.dashboard_url,
                    skip_unavailable_backends=config.skip_unavailable_model_backends,
                )
                if signal_processor is None:
                    LOGGER.warning("Live signal runtime is enabled, but no eligible runtime models were loaded.")
                    audit_repository.record_health_event(
                        component="collector.signal_runtime",
                        event_type="signal_runtime_unavailable",
                        severity="warning",
                        message="Live signal runtime is enabled, but no eligible runtime models were loaded.",
                        payload={
                            "asset": config.asset,
                            "timeframe": config.signal_timeframe,
                            "long_runtime_manifest_path": str(config.long_runtime_manifest_path),
                            "short_runtime_manifest_path": str(config.short_runtime_manifest_path),
                        },
                    )
            except Exception as exc:
                LOGGER.warning("Live signal runtime initialization failed: %s", exc)
                audit_repository.record_health_event(
                    component="collector.signal_runtime",
                    event_type="signal_runtime_initialization_failed",
                    severity="error",
                    message="Live signal runtime initialization failed.",
                    payload={
                        "asset": config.asset,
                        "timeframe": config.signal_timeframe,
                        "long_runtime_manifest_path": str(config.long_runtime_manifest_path),
                        "short_runtime_manifest_path": str(config.short_runtime_manifest_path),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )

        return cls(
            config=config,
            client=client,
            stream=stream,
            backfill_connector=backfill_connector,
            store=store,
            service=service,
            audit_repository=audit_repository,
            signal_processor=signal_processor,
        )

    async def bootstrap(self) -> BootstrapSummary:
        try:
            latest_stored_timestamp = self.store.get_latest_bar_timestamp(
                asset=self.config.asset,
                timeframe=self.config.source_timeframe,
            )
            summary = BootstrapSummary(latest_stored_timestamp=latest_stored_timestamp)

            if latest_stored_timestamp is None:
                startup_history_bars = self._resolve_startup_history_bars()
                seed_bars = await self.client.fetch_historical_chart(
                    symbol=canonical_asset_to_fmp_symbol(self.config.asset),
                    interval=canonical_timeframe_to_fmp_interval(self.config.source_timeframe),
                    outputsize=startup_history_bars,
                )
                seed_bars = self._filter_finalized_source_bars(seed_bars)
                processed = await self.service.process_bars(seed_bars)
                summary.seeded_history_bars = len(seed_bars)
                summary.stored_source_bars += processed.stored_source_bars
                summary.stored_aggregated_bars += processed.stored_aggregated_bars
                if seed_bars:
                    self.stream.last_emitted_timestamp = seed_bars[-1].timestamp
                self._warm_signal_processor()
                LOGGER.info(
                    "Bootstrapped empty live store with %s historical %s bars for %s.",
                    len(seed_bars),
                    self.config.source_timeframe,
                    self.config.asset,
                )
                return summary

            unresolved_gaps = self.store.fetch_gaps(
                asset=self.config.asset,
                timeframe=self.config.source_timeframe,
                unresolved_only=True,
            )
            summary.startup_gap_recovery_attempts = len(unresolved_gaps)
            if unresolved_gaps:
                recovery_result = await self.service.recover_unresolved_gaps(list(unresolved_gaps))
                summary.startup_gap_backfilled_bars = recovery_result.backfilled_bars
                summary.stored_source_bars += recovery_result.stored_source_bars
                summary.stored_aggregated_bars += recovery_result.stored_aggregated_bars
                remaining_unresolved = self.store.fetch_gaps(
                    asset=self.config.asset,
                    timeframe=self.config.source_timeframe,
                    unresolved_only=True,
                )
                summary.remaining_unresolved_gaps = len(remaining_unresolved)
                summary.resolved_startup_gaps = (
                    summary.startup_gap_recovery_attempts - summary.remaining_unresolved_gaps
                )
                LOGGER.info(
                    "Recovered %s of %s unresolved startup gaps for %s.",
                    summary.resolved_startup_gaps,
                    summary.startup_gap_recovery_attempts,
                    self.config.asset,
                )
                if remaining_unresolved:
                    self._record_health_event(
                        component="collector.bootstrap",
                        event_type="startup_gap_recovery_incomplete",
                        severity="error",
                        message="Startup gap recovery left unresolved ingestion gaps in the store.",
                        payload={
                            "asset": self.config.asset,
                            "source_timeframe": self.config.source_timeframe,
                            "remaining_unresolved_gap_ids": [gap.gap_id for gap in remaining_unresolved],
                        },
                    )
                    raise RuntimeError(
                        f"Startup gap recovery left {len(remaining_unresolved)} unresolved ingestion gaps."
                    )

            refresh_result = await self._refresh_recent_finalized_source_bars(latest_stored_timestamp)
            summary.stored_source_bars += refresh_result.stored_source_bars
            summary.stored_aggregated_bars += refresh_result.stored_aggregated_bars

            latest_finalized_timestamp = self._latest_finalized_source_timestamp()
            if latest_finalized_timestamp is None:
                LOGGER.info("Live collector resumed with no finalized %s bars ready yet.", self.config.source_timeframe)
                return summary
            summary.latest_stored_timestamp = latest_finalized_timestamp

            self._warm_runtime_state(latest_finalized_timestamp)
            self.stream.last_emitted_timestamp = latest_finalized_timestamp

            start = latest_finalized_timestamp + timeframe_to_timedelta(self.config.source_timeframe)
            end = utc_now()
            if start > end:
                self._warm_signal_processor()
                LOGGER.info("Live collector resumed at %s with no startup catch-up needed.", latest_finalized_timestamp)
                return summary

            for window in self._build_catchup_windows(start=start, end=end):
                recovered_bars = await self.backfill_connector.backfill_bars(window)
                recovered_bars = self._filter_finalized_source_bars(recovered_bars)
                processed = await self.service.process_bars(recovered_bars)
                summary.catchup_bars += len(recovered_bars)
                summary.stored_source_bars += processed.stored_source_bars
                summary.stored_aggregated_bars += processed.stored_aggregated_bars

            refreshed_timestamp = self._latest_finalized_source_timestamp()
            self.stream.last_emitted_timestamp = refreshed_timestamp or latest_finalized_timestamp
            summary.latest_stored_timestamp = self.stream.last_emitted_timestamp
            self._warm_signal_processor()
            LOGGER.info(
                "Recovered %s startup catch-up bars for %s since %s.",
                summary.catchup_bars,
                self.config.asset,
                latest_finalized_timestamp,
            )
            return summary
        except Exception as exc:
            self._record_health_event(
                component="collector.bootstrap",
                event_type="bootstrap_failed",
                severity="error",
                message="Live collector bootstrap failed.",
                payload={
                    "asset": self.config.asset,
                    "source_timeframe": self.config.source_timeframe,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise

    async def run(
        self,
        *,
        after_bootstrap: RuntimeSummaryHook | None = None,
        after_cycle: RuntimeCycleHook | None = None,
        before_close: RuntimeSummaryHook | None = None,
    ) -> CollectorRunSummary:
        summary = CollectorRunSummary(bootstrap=BootstrapSummary())
        cycle_index = 0
        flush_error: Exception | None = None
        try:
            summary.bootstrap = await self.bootstrap()
            await self._invoke_runtime_hook(
                after_bootstrap,
                hook_name="after_bootstrap",
                args=(summary,),
            )
            while self.config.max_cycles is None or cycle_index < self.config.max_cycles:
                cycle_index += 1
                try:
                    result = await self.service.collect_once()
                except Exception as exc:
                    self._record_health_event(
                        component="collector.runtime",
                        event_type="cycle_failed",
                        severity="error",
                        message="A live collector cycle failed.",
                        payload={
                            "cycle_index": cycle_index,
                            "asset": self.config.asset,
                            "source_timeframe": self.config.source_timeframe,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    raise
                try:
                    refresh_result = await self._refresh_recent_finalized_cycle_bars()
                except Exception as exc:
                    self._record_health_event(
                        component="collector.runtime",
                        event_type="cycle_reconciliation_failed",
                        severity="error",
                        message="Refreshing recent finalized source bars failed during a live cycle.",
                        payload={
                            "cycle_index": cycle_index,
                            "asset": self.config.asset,
                            "source_timeframe": self.config.source_timeframe,
                            "lookback_bars": int(self.config.cycle_finalized_refresh_lookback_bars),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    raise
                result.reconciled_source_bars = refresh_result.stored_source_bars
                result.reconciled_aggregated_bars = refresh_result.stored_aggregated_bars
                summary.cycles_run = cycle_index
                summary.cycle_results.append(result)
                signal_results = self._process_signal_cycle(result)
                summary.signal_results.append(signal_results)
                summary.evaluated_signals += len(signal_results)
                summary.emitted_signals += sum(item.decision == "emit" for item in signal_results)
                summary.sent_notifications += sum(
                    (item.notification_status == "sent") + (item.sms_notification_status == "sent")
                    for item in signal_results
                )
                summary.captured_media_artifacts += sum(item.media_artifact_id is not None for item in signal_results)
                await self._invoke_runtime_hook(
                    after_cycle,
                    hook_name="after_cycle",
                    args=(summary, result, cycle_index),
                )
                LOGGER.info(
                    "Collector cycle %s: polled=%s stored_source=%s stored_aggregated=%s "
                    "reconciled_source=%s reconciled_aggregated=%s backfilled=%s gaps=%s dup=%s out_of_order=%s",
                    cycle_index,
                    result.polled_bars,
                    result.stored_source_bars,
                    result.stored_aggregated_bars,
                    result.reconciled_source_bars,
                    result.reconciled_aggregated_bars,
                    result.backfilled_bars,
                    result.gaps_detected,
                    result.duplicates,
                    result.out_of_order,
                )
                if self.config.max_cycles is not None and cycle_index >= self.config.max_cycles:
                    break
                await asyncio.sleep(self.config.poll_interval_seconds)
            summary.terminal_status = "completed"
            return summary
        except asyncio.CancelledError as exc:
            summary.terminal_status = "cancelled"
            summary.terminal_error = _error_payload(exc)
            raise
        except Exception as exc:
            summary.terminal_status = "failed"
            summary.terminal_error = _error_payload(exc)
            raise
        finally:
            try:
                summary.flushed_aggregated_bars = self.service.flush()
            except Exception as exc:
                flush_error = exc
                if summary.terminal_status == "running":
                    summary.terminal_status = "failed"
                if summary.terminal_error is None:
                    summary.terminal_error = _error_payload(exc)
                self._record_health_event(
                    component="collector.runtime",
                    event_type="flush_failed",
                    severity="error",
                    message="Flushing aggregated bars during shutdown failed.",
                    payload={
                        "asset": self.config.asset,
                        "source_timeframe": self.config.source_timeframe,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
            if summary.terminal_status == "running":
                summary.terminal_status = "completed"
            await self._invoke_runtime_hook(
                before_close,
                hook_name="before_close",
                args=(summary,),
            )
            await self.close()
            if flush_error is not None:
                raise flush_error

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

    def _filter_finalized_source_bars(self, bars: list) -> list:
        return filter_finalized_bars(
            bars,
            timeframe=self.config.source_timeframe,
            grace_period_seconds=self.config.finalized_bar_grace_seconds,
        )

    def _latest_finalized_source_timestamp(self):
        cutoff = latest_finalized_bar_start(
            self.config.source_timeframe,
            grace_period_seconds=self.config.finalized_bar_grace_seconds,
        )
        row = self.store.connection.execute(
            """
            SELECT timestamp_utc
            FROM canonical_bars
            WHERE asset = ? AND timeframe = ? AND timestamp_utc <= ?
            ORDER BY timestamp_utc DESC
            LIMIT 1
            """,
            (
                self.config.asset,
                self.config.source_timeframe,
                cutoff.isoformat(),
            ),
        ).fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(str(row["timestamp_utc"]))

    async def _refresh_recent_finalized_source_bars(
        self,
        latest_stored_timestamp,
        *,
        lookback_bars: int | None = None,
    ) -> CollectorCycleResult:
        resolved_lookback_bars = (
            max(2, int(self.config.startup_history_bars))
            if lookback_bars is None
            else max(2, int(lookback_bars))
        )
        source_step = timeframe_to_timedelta(self.config.source_timeframe)
        refresh_start = latest_stored_timestamp - (source_step * max(0, resolved_lookback_bars - 1))
        refreshed_bars = await self.client.fetch_historical_chart(
            symbol=canonical_asset_to_fmp_symbol(self.config.asset),
            interval=canonical_timeframe_to_fmp_interval(self.config.source_timeframe),
            start_date=refresh_start,
            end_date=utc_now(),
        )
        refreshed_bars = [
            bar
            for bar in self._filter_finalized_source_bars(refreshed_bars)
            if bar.timestamp >= refresh_start
        ]
        if not refreshed_bars:
            return CollectorCycleResult()
        result = self.service.repair_source_bars(refreshed_bars)
        log_method = LOGGER.info if lookback_bars is None else LOGGER.debug
        message = (
            "Refreshed %s recent finalized %s bars for %s."
            if lookback_bars is None
            else "Reconciled %s recent finalized %s bars for %s."
        )
        log_method(
            message,
            result.stored_source_bars,
            self.config.source_timeframe,
            self.config.asset,
        )
        return result

    async def _refresh_recent_finalized_cycle_bars(self) -> CollectorCycleResult:
        lookback_bars = int(self.config.cycle_finalized_refresh_lookback_bars)
        if lookback_bars <= 0:
            return CollectorCycleResult()
        latest_stored_timestamp = self.store.get_latest_bar_timestamp(
            asset=self.config.asset,
            timeframe=self.config.source_timeframe,
        )
        if latest_stored_timestamp is None:
            return CollectorCycleResult()
        return await self._refresh_recent_finalized_source_bars(
            latest_stored_timestamp,
            lookback_bars=lookback_bars,
        )

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

    def _warm_signal_processor(self) -> None:
        if self.signal_processor is None:
            return
        try:
            warmed_bars = self.signal_processor.warm_from_store()
            if warmed_bars:
                LOGGER.info("Warmed live signal processor with %s stored %s bars.", warmed_bars, self.config.signal_timeframe)
        except Exception as exc:
            self._record_health_event(
                component="collector.signal_runtime",
                event_type="signal_warmup_failed",
                severity="error",
                message="Warming the live signal processor failed.",
                payload={
                    "asset": self.config.asset,
                    "timeframe": self.config.signal_timeframe,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )

    def _resolve_startup_history_bars(self) -> int:
        startup_history_bars = int(self.config.startup_history_bars)
        if self.signal_processor is None:
            return startup_history_bars
        if self.config.source_timeframe != self.config.signal_timeframe:
            return startup_history_bars
        if not hasattr(self.signal_processor, "bindings"):
            return startup_history_bars

        required_signal_bars = max(
            (
                int(binding.loaded_model.manifest.context_requirements.minimum_runtime_history_bars)
                for binding in self.signal_processor.bindings
            ),
            default=0,
        )
        # Add a small cushion so the first live bar after bootstrap can evaluate immediately.
        return max(startup_history_bars, required_signal_bars + 10)

    def _process_signal_cycle(self, cycle_result: CollectorCycleResult) -> tuple[RuntimeSignalResult, ...]:
        if self.signal_processor is None:
            return ()

        emit_operator_artifacts = cycle_result.gaps_detected == 0 and cycle_result.backfilled_bars == 0
        try:
            return self.signal_processor.process_new_bars_from_store(
                emit_operator_artifacts=emit_operator_artifacts,
                max_timestamp=self._latest_signal_processing_timestamp(),
            )
        except Exception as exc:
            self._record_health_event(
                component="collector.signal_runtime",
                event_type="signal_cycle_failed",
                severity="error",
                message="Processing live signal decisions for the latest cycle failed.",
                payload={
                    "asset": self.config.asset,
                    "timeframe": self.config.signal_timeframe,
                    "stored_aggregated_bars": cycle_result.stored_aggregated_bars,
                    "gaps_detected": cycle_result.gaps_detected,
                    "backfilled_bars": cycle_result.backfilled_bars,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return ()

    def _latest_signal_processing_timestamp(self) -> datetime:
        return utc_now() - timeframe_to_timedelta(self.config.signal_timeframe) - timedelta(
            seconds=float(self.config.signal_processing_delay_seconds or 0.0)
        )

    async def _invoke_runtime_hook(
        self,
        hook: RuntimeSummaryHook | RuntimeCycleHook | None,
        *,
        hook_name: str,
        args: tuple[Any, ...],
    ) -> None:
        if hook is None:
            return
        try:
            outcome = hook(*args)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception as exc:
            self._record_health_event(
                component="collector.runtime",
                event_type="lifecycle_hook_failed",
                severity="error",
                message=f"Collector lifecycle hook '{hook_name}' failed.",
                payload={
                    "hook_name": hook_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )


def _error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _build_sms_transport(recipients: tuple[str, ...]):
    phone_recipients = tuple(
        recipient for recipient in recipients if not is_email_gateway_sms_recipient(recipient)
    )
    email_gateway_recipients = tuple(
        recipient for recipient in recipients if is_email_gateway_sms_recipient(recipient)
    )

    phone_transport = TwilioSmsTransport.from_environment() if phone_recipients else None
    email_gateway_transport = (
        EmailGatewaySmsTransport.from_environment()
        if email_gateway_recipients
        else None
    )

    if phone_transport is not None and email_gateway_transport is not None:
        return RoutedSmsTransport(
            phone_transport=phone_transport,
            email_gateway_transport=email_gateway_transport,
        )
    if phone_transport is not None:
        return phone_transport
    if email_gateway_transport is not None:
        return email_gateway_transport
    raise ValueError("At least one SMS recipient is required to build an SMS transport.")
