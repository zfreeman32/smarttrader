"""Observation-time provenance and the bar gate for prospective shadow research.

This gate is necessary, not sufficient: setup, feature, model and policy contracts
still apply. It deliberately does not manufacture an executable historical price.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math

from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import canonical_asset_symbol, ensure_utc, timeframe_to_timedelta, utc_now
from ote_live.ingestion.market_calendar import MARKET_CALENDAR_VERSION, broker_schedule_market_open, is_expected_market_bar_timestamp

BAR_PROVENANCE_VERSION = "source-bar-provenance-v2"
SHADOW_FRESHNESS_CONTRACT = "complete-observed-live-bar-90s-dated-calendar-v2"


def source_bar_version(bar: MarketBar) -> str:
    """Content identity; changing values, completion or feed creates a revision."""
    payload = bar.model_dump(mode="json", exclude={
        "bar_version", "first_observed_at", "last_observed_at", "observation_kind",
    })
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def observe_bar(bar: MarketBar, *, observed_at: datetime | None = None, observation_kind: str | None = None) -> MarketBar:
    """Stamp actual first ingestion without inferring completion or feed quality."""
    observed = ensure_utc(observed_at or utc_now())
    result = bar.model_copy(update={
        "source_timestamp": bar.source_timestamp or bar.timestamp,
        "first_observed_at": bar.first_observed_at or observed,
        "last_observed_at": bar.last_observed_at or observed,
        "observation_kind": observation_kind or bar.observation_kind,
    })
    return result.model_copy(update={"bar_version": result.bar_version or source_bar_version(result)})


@dataclass(frozen=True)
class ShadowBarEligibility:
    eligible: bool
    evaluation_kind: str
    reasons: tuple[str, ...]
    source_bar_timestamp: datetime
    source_bar_version: str | None
    first_observed_at: datetime | None
    prediction_recorded_at: datetime
    bar_close_at: datetime
    delay_seconds: float
    max_age_seconds: float
    feed_type: str | None
    is_complete: bool | None
    earliest_entry_at: datetime | None
    last_observed_at: datetime | None
    previous_bar_timestamp: datetime | None
    calendar_version: str = MARKET_CALENDAR_VERSION
    executable_entry_price: None = None
    contract_version: str = SHADOW_FRESHNESS_CONTRACT

    def to_payload(self) -> dict:
        return {key: value.isoformat() if isinstance(value, datetime) else list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items()}


def assess_shadow_bar_eligibility(
    bar: MarketBar, *, recorded_at: datetime | None = None,
    max_age_seconds: float = 90.0, previous_bar: MarketBar | None = None,
    previous_bar_timestamp: datetime | None = None,
) -> ShadowBarEligibility:
    if not math.isfinite(max_age_seconds) or max_age_seconds < 0:
        raise ValueError("max_age_seconds must be a finite nonnegative number")
    recorded = ensure_utc(recorded_at or utc_now())
    start = ensure_utc(bar.timestamp)
    close_at = start + timeframe_to_timedelta(bar.timeframe)
    delay = (recorded - close_at).total_seconds()
    feed = str(bar.feed_type or "unknown").lower().replace("-", "_")
    kind = str(bar.observation_kind or "unknown").lower()
    reasons: list[str] = []
    step = timeframe_to_timedelta(bar.timeframe)
    if start.timestamp() % step.total_seconds():
        reasons.append("source_bar_off_grid")
    if bar.is_complete is not True:
        reasons.append("bar_incomplete" if bar.is_complete is False else "bar_completion_unknown")
    if delay < 0:
        reasons.append("bar_not_closed")
    elif delay > max_age_seconds:
        reasons.append("bar_stale")
    if bar.first_observed_at is None:
        reasons.append("first_observation_unknown")
    elif ensure_utc(bar.first_observed_at) > recorded:
        reasons.append("observation_after_prediction")
    if bar.last_observed_at is None:
        reasons.append("last_observation_unknown")
    elif ensure_utc(bar.last_observed_at) > recorded:
        reasons.append("revision_after_prediction")
    if bar.first_observed_at is not None and bar.last_observed_at is not None and bar.first_observed_at > bar.last_observed_at:
        reasons.append("observation_times_reversed")
    if bar.is_complete is True and bar.last_observed_at is not None and ensure_utc(bar.last_observed_at) < close_at:
        reasons.append("completion_observed_before_bar_close")
    if not bar.bar_version or bar.source_timestamp is None:
        reasons.append("source_provenance_unknown")
    elif ensure_utc(bar.source_timestamp) != start:
        reasons.append("source_timestamp_mismatch")
    if feed not in {"live", "real_time", "realtime"}:
        reasons.append("feed_not_live")
    if kind != "live":
        reasons.append("backfilled_evaluation" if kind in {"backfill", "historical", "replay", "repair"} else "observation_kind_unknown")
    if not is_expected_market_bar_timestamp(start, asset=bar.asset, feature_context=bar.feature_context):
        reasons.append("source_bar_market_closed")
    if not is_expected_market_bar_timestamp(recorded, asset=bar.asset, feature_context=bar.feature_context):
        reasons.append("prediction_market_closed")
    # Weekly fallback remains useful for diagnostics, but cannot certify holidays.
    def dated_open(timestamp: datetime) -> bool | None:
        return broker_schedule_market_open(timestamp,
            trading_hours=bar.feature_context.get("ibkr_trading_hours"),
            timezone=bar.feature_context.get("ibkr_timezone"))

    if canonical_asset_symbol(bar.asset) == "ES":
        cursor = start
        while cursor < close_at:
            market_open = dated_open(cursor)
            if market_open is not True:
                reasons.append("market_calendar_unverified" if market_open is None else "source_bar_crosses_market_closure")
                break
            cursor += timedelta(minutes=1)
        if dated_open(recorded) is None:
            reasons.append("market_calendar_unverified")
    previous_timestamp = previous_bar.timestamp if previous_bar is not None else previous_bar_timestamp
    if previous_timestamp is not None:
        previous_timestamp = ensure_utc(previous_timestamp)
        if previous_timestamp >= start or (start - previous_timestamp) % step:
            reasons.append("source_bar_sequence_invalid")
        cursor = previous_timestamp + step
        while cursor < start:
            if canonical_asset_symbol(bar.asset) == "ES" and dated_open(cursor) is None:
                reasons.append("gap_calendar_unverified")
                break
            if is_expected_market_bar_timestamp(cursor, asset=bar.asset, feature_context=bar.feature_context):
                reasons.append("source_bar_gap")
                break
            cursor += timeframe_to_timedelta(bar.timeframe)
    reasons = list(dict.fromkeys(reasons))
    evaluation_kind = (
        "backfilled" if kind in {"backfill", "historical", "replay", "repair"}
        else "delayed" if delay > max_age_seconds or "delayed" in feed or "frozen" in feed
        else "provisional" if bar.is_complete is False or delay < 0
        else "live" if not reasons else "diagnostic"
    )
    return ShadowBarEligibility(
        eligible=not reasons, evaluation_kind=evaluation_kind, reasons=tuple(reasons),
        source_bar_timestamp=bar.source_timestamp or start, source_bar_version=bar.bar_version,
        first_observed_at=bar.first_observed_at, prediction_recorded_at=recorded,
        bar_close_at=close_at, delay_seconds=delay, max_age_seconds=max_age_seconds,
        feed_type=bar.feed_type, is_complete=bar.is_complete,
        earliest_entry_at=recorded if not reasons else None,
        last_observed_at=bar.last_observed_at, previous_bar_timestamp=previous_timestamp,
    )
