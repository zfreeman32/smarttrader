from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import (
    AbstractBackfillConnector,
    AbstractBarStream,
    BackfillWindow,
    CanonicalTimeframe,
    ensure_utc,
    filter_finalized_bars,
    timeframe_to_timedelta,
)
from ote_live.ingestion.ibkr import (
    IBKRConfig,
    IBKRError,
    IBKRMarketDataService,
    SQLiteIBKRSnapshotSink,
)

# Keep the established public name used by collector configuration and tests.
IBKRRuntimeConfig = IBKRConfig


class IBKRHistoricalBarClient:
    """Async compatibility adapter over the managed official-API service."""

    def __init__(
        self,
        config: IBKRRuntimeConfig,
        *,
        asset: str,
        timeframe: CanonicalTimeframe,
        snapshot_db_path: str | Path | None = None,
        service: IBKRMarketDataService | None = None,
    ) -> None:
        self.config = config
        self.asset = asset
        self.timeframe = timeframe
        self._snapshot_sink = (
            SQLiteIBKRSnapshotSink(snapshot_db_path, asset=asset)
            if snapshot_db_path is not None and config.enabled
            else None
        )
        self.service = service or IBKRMarketDataService(
            config,
            snapshot_sink=self._snapshot_sink,
        )
        self._start_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.service.stop)

    async def fetch_historical_chart(
        self,
        *,
        symbol: str,
        interval: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        outputsize: int | None = None,
    ) -> list[MarketBar]:
        del symbol, interval
        await self._ensure_ready()
        bars = [
            item.to_market_bar(asset=self.asset, timeframe=self.timeframe)
            for item in self.service.get_bars()
        ]
        if start_date is not None:
            start_utc = ensure_utc(start_date)
            bars = [bar for bar in bars if bar.timestamp >= start_utc]
        if end_date is not None:
            end_utc = ensure_utc(end_date)
            bars = [bar for bar in bars if bar.timestamp <= end_utc]
        bars.sort(
            key=lambda item: (
                item.timestamp,
                int(item.instrument_id or 0),
            )
        )
        if outputsize is not None:
            limit = max(0, int(outputsize))
            return bars[-limit:] if limit else []
        return bars

    async def fetch_time_series(self, **kwargs) -> list[MarketBar]:
        return await self.fetch_historical_chart(**kwargs)

    async def _ensure_ready(self) -> None:
        if not self.config.enabled:
            raise IBKRError(
                "IBKR ingestion is selected but IBKR_ENABLED=false. Set IBKR_ENABLED=true "
                "after starting TWS or IB Gateway in paper mode."
            )
        async with self._start_lock:
            if not self.service.is_running():
                self.service.start()
            ready = await asyncio.to_thread(
                self.service.wait_until_ready,
                self.config.history_timeout_seconds
                + self.config.contract_timeout_seconds
                + self.config.handshake_timeout_seconds,
            )
            if not ready:
                status = self.service.get_status()
                raise IBKRError(
                    "IBKR ES service did not become ready: "
                    f"{status.get('last_error_message') or status.get('connection_state')}"
                )


class IBKRPollingBarStream(AbstractBarStream):
    """Drains completed callback-store bars into the existing inference runtime."""

    def __init__(
        self,
        client: IBKRHistoricalBarClient,
        *,
        asset: str = "ES",
        timeframe: CanonicalTimeframe = "5m",
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
        self.finalized_bar_grace_seconds = max(
            0.0,
            float(finalized_bar_grace_seconds),
        )
        self._emitted_keys: set[tuple[datetime, int | None]] = set()

    async def poll(self) -> list[MarketBar]:
        bars = await self.client.fetch_historical_chart(
            symbol=self.asset,
            interval=self.timeframe,
            outputsize=max(self.outputsize, 24),
        )
        finalized_bars = filter_finalized_bars(
            bars,
            timeframe=self.timeframe,
            grace_period_seconds=self.finalized_bar_grace_seconds,
        )
        finalized_bars = _filter_bars_to_effective_roll_segments(
            finalized_bars,
            self.client.service.store.snapshot().get("roll_events") or [],
        )
        new_bars: list[MarketBar] = []
        for bar in finalized_bars:
            key = (bar.timestamp, bar.instrument_id)
            if key in self._emitted_keys:
                continue
            if (
                self.last_emitted_timestamp is not None
                and bar.timestamp < self.last_emitted_timestamp
            ):
                continue
            new_bars.append(bar)
            self._emitted_keys.add(key)
        if new_bars:
            self.last_emitted_timestamp = max(bar.timestamp for bar in new_bars)
        if len(self._emitted_keys) > 5_000:
            cutoff = sorted(self._emitted_keys)[-2_500:]
            self._emitted_keys = set(cutoff)
        return new_bars


class IBKRBackfillConnector(AbstractBackfillConnector):
    def __init__(self, client: IBKRHistoricalBarClient) -> None:
        self.client = client

    async def backfill_bars(self, window: BackfillWindow) -> list[MarketBar]:
        bars = await self.client.fetch_historical_chart(
            symbol=window.asset,
            interval=window.timeframe,
            start_date=window.start,
            end_date=window.end,
        )
        return [
            bar
            for bar in bars
            if ensure_utc(window.start) <= bar.timestamp <= ensure_utc(window.end)
        ]


def canonical_timeframe_to_ibkr_bar_size(
    timeframe: CanonicalTimeframe,
) -> str:
    return {
        "1m": "1 min",
        "5m": "5 mins",
        "30m": "30 mins",
        "1h": "1 hour",
    }[timeframe]


def _resolve_ibkr_duration(
    *,
    timeframe: CanonicalTimeframe,
    start_date: datetime | None,
    end_date: datetime | None,
    outputsize: int | None,
    default_duration: str,
) -> str:
    if start_date is not None and end_date is not None:
        span_seconds = max(
            1,
            int((ensure_utc(end_date) - ensure_utc(start_date)).total_seconds())
            + int(timeframe_to_timedelta(timeframe).total_seconds()),
        )
        return _duration_string_from_seconds(span_seconds)
    if outputsize is not None:
        span_seconds = int(timeframe_to_timedelta(timeframe).total_seconds()) * max(
            6,
            int(outputsize) + 4,
        )
        return _duration_string_from_seconds(span_seconds)
    return default_duration


def _duration_string_from_seconds(span_seconds: int) -> str:
    seconds = max(1, int(span_seconds))
    if seconds <= 86_400:
        return f"{seconds} S"
    return f"{max(1, (seconds + 86_399) // 86_400)} D"


def _format_ibkr_datetime(value: datetime) -> str:
    return ensure_utc(value).strftime("%Y%m%d %H:%M:%S UTC")


def _coerce_ibkr_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    text = str(value).strip()
    if not text:
        raise ValueError("IBKR historical bar was missing a timestamp.")
    if text.isdigit():
        if len(text) == 8:
            return datetime.strptime(text, "%Y%m%d").replace(tzinfo=UTC)
        return datetime.fromtimestamp(int(text), tz=UTC)
    return ensure_utc(datetime.fromisoformat(text.replace(" UTC", "+00:00")))


def _ibkr_bar_to_market_bar(
    payload: Any,
    *,
    asset: str,
    timeframe: CanonicalTimeframe,
    source: str,
    contract: Any,
) -> MarketBar:
    return MarketBar(
        asset=asset,
        timeframe=timeframe,
        timestamp=_coerce_ibkr_datetime(getattr(payload, "date")),
        open=float(getattr(payload, "open")),
        high=float(getattr(payload, "high")),
        low=float(getattr(payload, "low")),
        close=float(getattr(payload, "close")),
        volume=float(getattr(payload, "volume", 0.0) or 0.0),
        source=source,
        symbol=str(getattr(contract, "symbol", "") or "") or None,
        contract_symbol=str(getattr(contract, "localSymbol", "") or "") or None,
        instrument_id=(
            int(getattr(contract, "conId"))
            if getattr(contract, "conId", None) not in {None, 0}
            else None
        ),
    )


def _filter_bars_to_effective_roll_segments(
    bars: list[MarketBar],
    roll_events: list[dict[str, Any]],
) -> list[MarketBar]:
    filtered = list(bars)
    for event in roll_events:
        try:
            effective_at = ensure_utc(
                datetime.fromisoformat(str(event["effective_at_utc"]))
            )
            from_conid = int(event["from_conId"])
            to_conid = int(event["to_conId"])
        except (KeyError, TypeError, ValueError):
            continue
        filtered = [
            bar
            for bar in filtered
            if not (
                (bar.instrument_id == to_conid and bar.timestamp < effective_at)
                or (bar.instrument_id == from_conid and bar.timestamp >= effective_at)
            )
        ]
    return filtered
