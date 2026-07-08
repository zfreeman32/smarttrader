from __future__ import annotations

from typing import Iterable

import pandas as pd

from ote_live.contracts.market_data import MarketBar


class FeatureRuntimeState:
    def __init__(self, *, max_history_bars: int) -> None:
        if max_history_bars <= 0:
            raise ValueError("max_history_bars must be positive.")
        self.max_history_bars = int(max_history_bars)
        self._bars: list[MarketBar] = []

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def latest_timestamp(self):
        if not self._bars:
            return None
        return self._bars[-1].timestamp

    def clear(self) -> None:
        self._bars.clear()

    def ingest_bar(self, bar: MarketBar) -> None:
        if self._bars:
            latest_timestamp = self._bars[-1].timestamp
            if bar.timestamp < latest_timestamp:
                raise ValueError(
                    "FeatureRuntimeState received an out-of-order bar. "
                    f"latest={latest_timestamp!r}, new={bar.timestamp!r}"
                )
            if bar.timestamp == latest_timestamp:
                self._bars[-1] = bar
                return
        self._bars.append(bar)
        self._trim()

    def extend(self, bars: Iterable[MarketBar]) -> None:
        for bar in bars:
            self.ingest_bar(bar)

    def to_frame(self) -> pd.DataFrame:
        if not self._bars:
            return pd.DataFrame(
                columns=[
                    "asset",
                    "timeframe",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "bid",
                    "ask",
                    "spread",
                    "source",
                    "symbol",
                    "contract_symbol",
                    "instrument_id",
                ]
            )

        rows = [
            {
                "asset": bar.asset,
                "timeframe": bar.timeframe,
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "bid": bar.bid,
                "ask": bar.ask,
                "spread": bar.spread,
                "source": bar.source,
                "symbol": bar.symbol,
                "contract_symbol": bar.contract_symbol,
                "instrument_id": bar.instrument_id,
            }
            for bar in self._bars
        ]
        return pd.DataFrame.from_records(rows)

    def _trim(self) -> None:
        overflow = len(self._bars) - self.max_history_bars
        if overflow > 0:
            del self._bars[:overflow]
