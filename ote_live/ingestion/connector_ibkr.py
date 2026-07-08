from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

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


def _ib_classes():
    try:
        from ib_insync import Contract, Future, IB
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "IBKR live ingestion requires the 'ib_insync' package to be installed."
        ) from exc
    return IB, Future, Contract


@dataclass(frozen=True)
class IBKRRuntimeConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 17
    market_data_type: int = 1
    symbol: str = "ES"
    exchange: str = "CME"
    currency: str = "USD"
    local_symbol: str | None = None
    contract_month: str | None = None
    con_id: int | None = None
    use_rth: bool = False
    bar_size: str | None = None
    what_to_show: str = "TRADES"
    keep_up_to_date: bool = True
    backfill_duration: str = "2 D"
    readonly: bool = True


class IBKRHistoricalBarClient:
    def __init__(
        self,
        config: IBKRRuntimeConfig,
        *,
        asset: str,
        timeframe: CanonicalTimeframe,
    ) -> None:
        self.config = config
        self.asset = asset
        self.timeframe = timeframe
        self._ib = None
        self._contract = None
        self._connect_lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._ib is None:
            return
        if self._ib.isConnected():
            self._ib.disconnect()

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
        contract = await self._ensure_ready()
        ib = self._ib
        if ib is None:
            return []

        resolved_end = ensure_utc(end_date) if end_date is not None else None
        duration_str = _resolve_ibkr_duration(
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date,
            outputsize=outputsize,
            default_duration=self.config.backfill_duration,
        )
        bar_size = self.config.bar_size or canonical_timeframe_to_ibkr_bar_size(self.timeframe)
        end_date_time = "" if resolved_end is None else _format_ibkr_datetime(resolved_end)

        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_date_time,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=self.config.what_to_show,
            useRTH=bool(self.config.use_rth),
            formatDate=2,
            keepUpToDate=False,
        )
        normalized = [
            _ibkr_bar_to_market_bar(
                bar,
                asset=self.asset,
                timeframe=self.timeframe,
                source="ibkr.historical",
                contract=contract,
            )
            for bar in bars
        ]
        normalized.sort(key=lambda item: item.timestamp)

        if start_date is not None:
            start_utc = ensure_utc(start_date)
            normalized = [bar for bar in normalized if bar.timestamp >= start_utc]
        if end_date is not None:
            end_utc = ensure_utc(end_date)
            normalized = [bar for bar in normalized if bar.timestamp <= end_utc]
        if outputsize is not None:
            resolved_outputsize = max(0, int(outputsize))
            if resolved_outputsize == 0:
                return []
            normalized = normalized[-resolved_outputsize:]
        return normalized

    async def fetch_time_series(self, **kwargs) -> list[MarketBar]:
        return await self.fetch_historical_chart(**kwargs)

    async def _ensure_ready(self):
        async with self._connect_lock:
            IB, Future, Contract = _ib_classes()
            if self._ib is None:
                self._ib = IB()
            if not self._ib.isConnected():
                await self._ib.connectAsync(
                    self.config.host,
                    int(self.config.port),
                    clientId=int(self.config.client_id),
                    readonly=bool(self.config.readonly),
                )
                self._ib.reqMarketDataType(int(self.config.market_data_type))

            if self._contract is None:
                contract = self._build_contract(Future, Contract)
                qualified = await self._qualify_contract(contract)
                self._contract = qualified
            return self._contract

    def _build_contract(self, Future, Contract):
        if self.config.con_id is not None:
            contract = Contract()
            contract.secType = "FUT"
            contract.conId = int(self.config.con_id)
            contract.exchange = self.config.exchange
            contract.currency = self.config.currency
            return contract

        contract = Future(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            currency=self.config.currency,
        )
        if self.config.local_symbol:
            contract.localSymbol = self.config.local_symbol
        if self.config.contract_month:
            contract.lastTradeDateOrContractMonth = self.config.contract_month
        return contract

    async def _qualify_contract(self, contract):
        ib = self._ib
        if ib is None:
            raise RuntimeError("IBKR client is not connected.")

        if hasattr(ib, "qualifyContractsAsync"):
            qualified = await ib.qualifyContractsAsync(contract)
        else:  # pragma: no cover - legacy fallback
            qualified = await asyncio.to_thread(ib.qualifyContracts, contract)
        if not qualified:
            raise RuntimeError(
                "IBKR contract qualification returned no contracts. "
                "Set IBKR_ES_LOCAL_SYMBOL, IBKR_ES_CONTRACT_MONTH, or IBKR conId explicitly."
            )
        return qualified[0]


class IBKRPollingBarStream(AbstractBarStream):
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
        self.finalized_bar_grace_seconds = max(0.0, float(finalized_bar_grace_seconds))

    async def poll(self) -> list[MarketBar]:
        bars = await self.client.fetch_historical_chart(
            symbol=self.asset,
            interval=self.timeframe,
            outputsize=self.outputsize,
        )
        finalized_bars = filter_finalized_bars(
            bars,
            timeframe=self.timeframe,
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


def canonical_timeframe_to_ibkr_bar_size(timeframe: CanonicalTimeframe) -> str:
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
        buffer_bars = max(6, int(outputsize) + 4)
        span_seconds = int(timeframe_to_timedelta(timeframe).total_seconds()) * buffer_bars
        return _duration_string_from_seconds(span_seconds)

    return default_duration


def _duration_string_from_seconds(span_seconds: int) -> str:
    seconds = max(1, int(span_seconds))
    if seconds <= 86_400:
        return f"{seconds} S"
    days = max(1, (seconds + 86_399) // 86_400)
    return f"{days} D"


def _format_ibkr_datetime(value: datetime) -> str:
    return ensure_utc(value).strftime("%Y%m%d %H:%M:%S UTC")


def _coerce_ibkr_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    text = str(value).strip()
    if not text:
        raise ValueError("IBKR historical bar was missing a timestamp.")
    if text.isdigit() and len(text) == 8:
        parsed = datetime.strptime(text, "%Y%m%d")
        return ensure_utc(parsed)
    parsed = pd.to_datetime(text, errors="raise", utc=True)
    return ensure_utc(parsed.to_pydatetime())


def _ibkr_bar_to_market_bar(
    payload: Any,
    *,
    asset: str,
    timeframe: CanonicalTimeframe,
    source: str,
    contract,
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
