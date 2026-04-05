from __future__ import annotations

from datetime import datetime

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import GapCheckResult, IngestionGap, ensure_utc, timeframe_to_timedelta


class GapDetector:
    def __init__(self, *, asset: str = "EURUSD", timeframe: str = "1m") -> None:
        self.asset = asset
        self.timeframe = timeframe
        self._step = timeframe_to_timedelta(timeframe)  # type: ignore[arg-type]
        self._last_timestamp: datetime | None = None

    @property
    def last_timestamp(self) -> datetime | None:
        return self._last_timestamp

    def observe(self, bar: MarketBar) -> GapCheckResult:
        observed_timestamp = ensure_utc(bar.timestamp)
        if self._last_timestamp is None:
            self._last_timestamp = observed_timestamp
            return GapCheckResult(observed_timestamp=observed_timestamp)

        if observed_timestamp == self._last_timestamp:
            return GapCheckResult(observed_timestamp=observed_timestamp, duplicate=True)

        if observed_timestamp < self._last_timestamp:
            return GapCheckResult(observed_timestamp=observed_timestamp, out_of_order=True)

        expected_timestamp = self._last_timestamp + self._step
        gap: IngestionGap | None = None
        if observed_timestamp > expected_timestamp:
            missing_timestamps = []
            cursor = expected_timestamp
            while cursor < observed_timestamp:
                missing_timestamps.append(cursor)
                cursor += self._step
            gap = IngestionGap(
                asset=bar.asset,
                timeframe=bar.timeframe,  # type: ignore[arg-type]
                expected_timestamp=expected_timestamp,
                observed_timestamp=observed_timestamp,
                missing_timestamps=missing_timestamps,
                gap_size=len(missing_timestamps),
            )

        self._last_timestamp = observed_timestamp
        return GapCheckResult(observed_timestamp=observed_timestamp, gap=gap)


def detect_gaps(bars: list[MarketBar]) -> list[IngestionGap]:
    if not bars:
        return []

    detector = GapDetector(asset=bars[0].asset, timeframe=bars[0].timeframe)
    gaps: list[IngestionGap] = []
    for bar in sorted(bars, key=lambda item: item.timestamp):
        result = detector.observe(bar)
        if result.gap is not None:
            gaps.append(result.gap)
    return gaps
