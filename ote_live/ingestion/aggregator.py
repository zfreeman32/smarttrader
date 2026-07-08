from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import CanonicalTimeframe
from ote_live.ingestion.normalizer import bars_to_dataframe, dataframe_to_market_bars

_TIMEFRAME_TO_RULE: dict[CanonicalTimeframe, str] = {
    "1m": "1min",
    "5m": "5min",
    "30m": "30min",
    "1h": "1h",
}


def floor_timestamp_to_timeframe(timestamp, timeframe: CanonicalTimeframe):
    return pd.Timestamp(timestamp).tz_convert("UTC").floor(_TIMEFRAME_TO_RULE[timeframe]).to_pydatetime()


def aggregate_bar_sequence(
    bars: list[MarketBar],
    *,
    target_timeframe: CanonicalTimeframe,
    source: str | None = None,
) -> list[MarketBar]:
    if not bars:
        return []

    _validate_bar_sequence(bars)
    frame = bars_to_dataframe(bars)
    frame = frame.sort_values("timestamp").set_index("timestamp")

    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "bid": "last",
        "ask": "last",
        "spread": "last",
    }
    resampled = (
        frame.resample(_TIMEFRAME_TO_RULE[target_timeframe], label="left", closed="left")
        .agg(aggregation)
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return dataframe_to_market_bars(
        resampled,
        asset=bars[0].asset,
        timeframe=target_timeframe,
        source=source or f"local_aggregation:{bars[0].timeframe}->{target_timeframe}",
    )


@dataclass
class _BucketState:
    bucket_start: object
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float | None
    ask: float | None
    spread: float | None
    symbol: str | None
    contract_symbol: str | None
    instrument_id: int | None

    @classmethod
    def from_bar(cls, bucket_start, bar: MarketBar) -> "_BucketState":
        return cls(
            bucket_start=bucket_start,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            bid=bar.bid,
            ask=bar.ask,
            spread=bar.spread,
            symbol=bar.symbol,
            contract_symbol=bar.contract_symbol,
            instrument_id=bar.instrument_id,
        )

    def update(self, bar: MarketBar) -> None:
        self.high = max(self.high, bar.high)
        self.low = min(self.low, bar.low)
        self.close = bar.close
        self.volume += bar.volume
        if bar.bid is not None:
            self.bid = bar.bid
        if bar.ask is not None:
            self.ask = bar.ask
        if bar.spread is not None:
            self.spread = bar.spread
        if bar.symbol is not None:
            self.symbol = bar.symbol
        if bar.contract_symbol is not None:
            self.contract_symbol = bar.contract_symbol
        if bar.instrument_id is not None:
            self.instrument_id = bar.instrument_id


class MultiTimeframeBarAggregator:
    """Incrementally roll finalized source bars into larger local bars."""

    def __init__(
        self,
        *,
        target_timeframes: tuple[CanonicalTimeframe, ...] = ("5m", "30m", "1h"),
        source_name: str = "local_aggregation",
    ) -> None:
        self.target_timeframes = target_timeframes
        self.source_name = source_name
        self._states: dict[CanonicalTimeframe, _BucketState] = {}
        self._source_timeframe: CanonicalTimeframe | None = None

    def ingest_bar(self, bar: MarketBar) -> list[MarketBar]:
        finalized: list[MarketBar] = []
        if self._source_timeframe is None:
            self._source_timeframe = bar.timeframe
        elif bar.timeframe != self._source_timeframe:
            raise ValueError(
                "MultiTimeframeBarAggregator expected a single source timeframe, "
                f"but received {bar.timeframe!r} after {self._source_timeframe!r}."
            )
        for timeframe in self.target_timeframes:
            if timeframe == bar.timeframe:
                continue
            bucket_start = floor_timestamp_to_timeframe(bar.timestamp, timeframe)
            state = self._states.get(timeframe)
            if state is None:
                self._states[timeframe] = _BucketState.from_bar(bucket_start, bar)
                continue

            if bucket_start != state.bucket_start:
                finalized.append(self._finalize_state(timeframe, state, bar))
                self._states[timeframe] = _BucketState.from_bar(bucket_start, bar)
                continue

            state.update(bar)
        return finalized

    def flush(self) -> list[MarketBar]:
        finalized = [
            self._finalize_state(timeframe, state)
            for timeframe, state in sorted(
                self._states.items(),
                key=lambda item: _target_timeframe_sort_key(item[0]),
            )
        ]
        self._states.clear()
        self._source_timeframe = None
        return finalized

    def _finalize_state(
        self,
        timeframe: CanonicalTimeframe,
        state: _BucketState,
        sample_bar: MarketBar | None = None,
    ) -> MarketBar:
        asset = sample_bar.asset if sample_bar is not None else "EURUSD"
        source_timeframe = (
            sample_bar.timeframe
            if sample_bar is not None
            else (self._source_timeframe or "1m")
        )
        return MarketBar(
            asset=asset,
            timeframe=timeframe,
            timestamp=state.bucket_start,
            open=state.open,
            high=state.high,
            low=state.low,
            close=state.close,
            volume=state.volume,
            bid=state.bid,
            ask=state.ask,
            spread=state.spread,
            source=f"{self.source_name}:{source_timeframe}->{timeframe}",
            symbol=state.symbol,
            contract_symbol=state.contract_symbol,
            instrument_id=state.instrument_id,
        )


def _validate_bar_sequence(bars: list[MarketBar]) -> None:
    assets = {bar.asset for bar in bars}
    timeframes = {bar.timeframe for bar in bars}
    if len(assets) != 1:
        raise ValueError(f"Expected a single asset in the bar sequence, found: {sorted(assets)}")
    if len(timeframes) != 1:
        raise ValueError(f"Expected a single source timeframe in the bar sequence, found: {sorted(timeframes)}")


def _target_timeframe_sort_key(timeframe: CanonicalTimeframe) -> int:
    return {"5m": 5, "30m": 30, "1h": 60}[timeframe]
