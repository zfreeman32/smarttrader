from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import logging
import math
from threading import RLock
from typing import Any, Callable

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.provenance import source_bar_version

from .contracts import QualifiedESContract

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IBKRBar:
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    wap: float | None
    trade_count: int | None
    conid: int
    local_symbol: str
    contract_month: str
    bar_size: str
    data_type: str = "TRADES"
    is_complete: bool = False
    source: str = "ibkr.historical"
    market_data_type: str | None = None
    received_at_utc: datetime = datetime.min.replace(tzinfo=UTC)
    first_observed_at_utc: datetime | None = None
    version: int = 1
    observation_kind: str = "unknown"
    trading_hours: str | None = None
    timezone: str | None = None

    @property
    def key(self) -> tuple[int, datetime, str, str]:
        return (self.conid, self.timestamp_utc, self.bar_size, self.data_type)

    def to_market_bar(self, *, asset: str = "ES", timeframe: str = "5m") -> MarketBar:
        bar = MarketBar(
            asset=asset,
            timeframe=timeframe,
            timestamp=self.timestamp_utc,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            source=self.source,
            symbol=asset,
            contract_symbol=self.local_symbol,
            instrument_id=self.conid,
            source_timestamp=self.timestamp_utc,
            is_complete=self.is_complete,
            feed_type=self.market_data_type,
            first_observed_at=self.first_observed_at_utc or (self.received_at_utc if self.received_at_utc.year > 1 else None),
            last_observed_at=self.received_at_utc if self.received_at_utc.year > 1 else None,
            observation_kind=self.observation_kind,
            feature_context={
                "ibkr_contract_month": self.contract_month,
                "ibkr_wap": self.wap,
                "ibkr_trade_count": self.trade_count,
                "ibkr_market_data_type": self.market_data_type,
                "ibkr_is_complete": self.is_complete,
                "ibkr_bar_version": self.version,
                "ibkr_trading_hours": self.trading_hours,
                "ibkr_timezone": self.timezone,
            },
        )
        return bar.model_copy(update={"bar_version": source_bar_version(bar)})


@dataclass(frozen=True)
class IBKRQuote:
    conid: int
    local_symbol: str
    contract_month: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    last_size: float | None = None
    session_high: float | None = None
    session_low: float | None = None
    previous_close: float | None = None
    reported_volume: float | None = None
    last_update_timestamp_utc: datetime | None = None
    market_data_type: str | None = None

    @property
    def midprice(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        if self.ask < self.bid:
            return None
        return (self.bid + self.ask) / 2.0


class IBKRMarketDataStore:
    """Bounded callback store; readers only receive immutable snapshots."""

    def __init__(self, *, max_bars: int = 2_000) -> None:
        self.max_bars = max(2, int(max_bars))
        self._lock = RLock()
        self._bars: OrderedDict[tuple[int, datetime, str, str], IBKRBar] = OrderedDict()
        self._quotes: dict[int, IBKRQuote] = {}
        self._active_conid: int | None = None
        self._status: dict[str, Any] = {}
        self._roll_events: list[dict[str, Any]] = []
        self._listeners: list[Callable[[dict[str, Any]], None]] = []

    def add_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def set_active_contract(self, contract: QualifiedESContract | None) -> None:
        with self._lock:
            self._active_conid = None if contract is None else contract.conid
            if contract is not None:
                self._quotes.setdefault(
                    contract.conid,
                    IBKRQuote(
                        conid=contract.conid,
                        local_symbol=contract.local_symbol,
                        contract_month=contract.contract_month,
                    ),
                )
        self._notify()

    def upsert_bar(self, bar: IBKRBar) -> None:
        changed_bars: list[IBKRBar] = []
        received = _utc(bar.received_at_utc) if bar.received_at_utc.year > 1 else datetime.now(UTC)
        normalized = replace(
            bar,
            timestamp_utc=_utc(bar.timestamp_utc),
            received_at_utc=received,
            first_observed_at_utc=bar.first_observed_at_utc or received,
        )
        with self._lock:
            prior_active_key = self._latest_key_for_contract(normalized.conid)
            if prior_active_key is not None:
                prior = self._bars[prior_active_key]
                if prior.timestamp_utc < normalized.timestamp_utc and not prior.is_complete:
                    finalized = replace(prior, is_complete=True, version=prior.version + 1, received_at_utc=received)
                    self._bars[prior_active_key] = finalized
                    changed_bars.append(finalized)
            existing = self._bars.get(normalized.key)
            if existing is not None:
                normalized = replace(normalized, is_complete=existing.is_complete or normalized.is_complete,
                                     first_observed_at_utc=existing.first_observed_at_utc,
                                     version=existing.version + 1)
            self._bars[normalized.key] = normalized
            changed_bars.append(normalized)
            self._bars.move_to_end(normalized.key)
            while len(self._bars) > self.max_bars:
                self._bars.popitem(last=False)
        self._notify_bar_observations(changed_bars)
        self._notify()

    def mark_request_history_complete(self, conid: int, *, observed_at_utc: datetime | None = None) -> None:
        # With keepUpToDate=True the newest bucket remains partial. Sequential
        # callback inserts already marked all preceding buckets complete.
        completed: list[IBKRBar] = []
        observed = _utc(observed_at_utc or datetime.now(UTC))
        with self._lock:
            keys = sorted((key for key in self._bars if key[0] == int(conid)), key=lambda key: key[1])
            for key in keys[:-1]:
                bar = self._bars[key]
                if not bar.is_complete:
                    finalized = replace(bar, is_complete=True, version=bar.version + 1, received_at_utc=observed)
                    self._bars[key] = finalized
                    completed.append(finalized)
        self._notify_bar_observations(completed)
        self._notify()

    def update_quote(
        self,
        contract: QualifiedESContract,
        *,
        field: str,
        value: float | None,
        received_at_utc: datetime,
        market_data_type: str | None = None,
    ) -> None:
        with self._lock:
            quote = self._quotes.get(
                contract.conid,
                IBKRQuote(
                    conid=contract.conid,
                    local_symbol=contract.local_symbol,
                    contract_month=contract.contract_month,
                ),
            )
            if field not in IBKRQuote.__dataclass_fields__:
                return
            resolved = _valid_quote_value(field, value)
            if resolved is None:
                return
            quote = replace(
                quote,
                **{
                    field: resolved,
                    "last_update_timestamp_utc": _utc(received_at_utc),
                    "market_data_type": market_data_type or quote.market_data_type,
                },
            )
            self._quotes[contract.conid] = quote
        self._notify()

    def set_market_data_type(self, conid: int, value: str) -> None:
        with self._lock:
            quote = self._quotes.get(int(conid))
            if quote is not None:
                self._quotes[int(conid)] = replace(quote, market_data_type=value)
            # Feed changes describe subsequent callbacks. Relabeling an already
            # observed delayed bar as live would erase its collection provenance.
        self._notify()

    def update_status(self, **values: Any) -> None:
        with self._lock:
            self._status.update(values)
        self._notify()

    def add_roll_event(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._roll_events.append(dict(payload))
            self._roll_events = self._roll_events[-50:]
        self._notify()

    def get_bars(
        self,
        *,
        limit: int | None = None,
        completed_only: bool = False,
    ) -> tuple[IBKRBar, ...]:
        with self._lock:
            bars = sorted(
                self._bars.values(),
                key=lambda item: (item.timestamp_utc, item.conid),
            )
            if completed_only:
                bars = [bar for bar in bars if bar.is_complete]
            if limit is not None:
                bars = bars[-max(0, int(limit)) :]
            return tuple(bars)

    def get_quote(self) -> IBKRQuote | None:
        with self._lock:
            if self._active_conid is None:
                return None
            return self._quotes.get(self._active_conid)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def snapshot(self, *, bar_limit: int = 400) -> dict[str, Any]:
        with self._lock:
            bars = sorted(
                self._bars.values(),
                key=lambda item: (item.timestamp_utc, item.conid),
            )[-max(0, int(bar_limit)) :]
            quote = (
                self._quotes.get(self._active_conid)
                if self._active_conid is not None
                else None
            )
            return {
                "status": _serialize(dict(self._status)),
                "quote": _serialize_quote(quote),
                "bars": [_serialize_bar(bar) for bar in bars],
                "roll_events": _serialize(list(self._roll_events)),
            }

    def _latest_key_for_contract(
        self,
        conid: int,
    ) -> tuple[int, datetime, str, str] | None:
        keys = [key for key in self._bars if key[0] == int(conid)]
        if not keys:
            return None
        return max(keys, key=lambda item: item[1])

    def _notify(self) -> None:
        with self._lock:
            listeners = tuple(self._listeners)
        if not listeners:
            return
        status = self.get_status()
        eligible = []
        for listener in listeners:
            should_publish = getattr(listener, "should_publish", None)
            if callable(should_publish) and not should_publish(status):
                continue
            eligible.append(listener)
        if not eligible:
            return
        snapshot = self.snapshot()
        for listener in eligible:
            try:
                listener(snapshot)
            except Exception:
                LOGGER.exception("IBKR market-data snapshot listener failed.")

    def _notify_bar_observations(self, bars: list[IBKRBar]) -> None:
        # Observation persistence bypasses the dashboard's throttled snapshots.
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            record = getattr(listener, "record_bar_observations", None)
            if callable(record):
                record(bars)


def _serialize_bar(bar: IBKRBar) -> dict[str, Any]:
    return _serialize(asdict(bar))


def _serialize_quote(quote: IBKRQuote | None) -> dict[str, Any] | None:
    if quote is None:
        return None
    payload = asdict(quote)
    payload["midprice"] = quote.midprice
    return _serialize(payload)


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return _utc(value).isoformat()
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _valid_quote_value(field: str, value: float | None) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved):
        return None
    if field in {
        "bid",
        "ask",
        "last",
        "session_high",
        "session_low",
        "previous_close",
    } and resolved <= 0:
        return None
    if field.endswith("_size") or field == "reported_volume":
        if resolved < 0:
            return None
    return resolved


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
