from __future__ import annotations

from datetime import datetime, timedelta

from ote_live.ingestion.base import HeartbeatStatus, ensure_utc, utc_now


class HeartbeatMonitor:
    def __init__(self, *, source: str, stale_after: timedelta = timedelta(seconds=90)) -> None:
        self.source = source
        self.stale_after = stale_after
        self._last_observed_at_utc: datetime | None = None

    @property
    def last_observed_at_utc(self) -> datetime | None:
        return self._last_observed_at_utc

    def beat(self, observed_at_utc: datetime | None = None) -> HeartbeatStatus:
        observed = ensure_utc(observed_at_utc or utc_now())
        self._last_observed_at_utc = observed
        return self.snapshot(now=observed)

    def snapshot(self, *, now: datetime | None = None) -> HeartbeatStatus:
        current_time = ensure_utc(now or utc_now())
        if self._last_observed_at_utc is None:
            return HeartbeatStatus(
                source=self.source,
                observed_at_utc=current_time,
                stale_after_seconds=self.stale_after.total_seconds(),
                lag_seconds=float("inf"),
                is_stale=True,
            )

        lag_seconds = max(0.0, (current_time - self._last_observed_at_utc).total_seconds())
        return HeartbeatStatus(
            source=self.source,
            observed_at_utc=self._last_observed_at_utc,
            stale_after_seconds=self.stale_after.total_seconds(),
            lag_seconds=lag_seconds,
            is_stale=lag_seconds > self.stale_after.total_seconds(),
        )
