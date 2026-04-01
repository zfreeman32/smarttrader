from __future__ import annotations

import re
from datetime import timedelta, timezone, tzinfo
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


_FIXED_OFFSET_PATTERN = re.compile(
    r"^(?:(?:UTC|GMT)\s*)?([+-])\s*(\d{1,2})(?::?(\d{2}))?$",
    re.IGNORECASE,
)

_FIXED_TZ_ALIASES: dict[str, tzinfo] = {
    "UTC": timezone.utc,
    "GMT": timezone.utc,
    "Z": timezone.utc,
    "EST": timezone(timedelta(hours=-5)),
    "EDT": timezone(timedelta(hours=-4)),
    "CST": timezone(timedelta(hours=-6)),
    "CDT": timezone(timedelta(hours=-5)),
    "MST": timezone(timedelta(hours=-7)),
    "MDT": timezone(timedelta(hours=-6)),
    "PST": timezone(timedelta(hours=-8)),
    "PDT": timezone(timedelta(hours=-7)),
}


@lru_cache(maxsize=None)
def resolve_timezone(value: str | tzinfo | None) -> tzinfo:
    if isinstance(value, tzinfo):
        return value

    if value is None:
        return timezone.utc

    name = str(value).strip()
    if not name:
        return timezone.utc

    alias = _FIXED_TZ_ALIASES.get(name.upper())
    if alias is not None:
        return alias

    if match := _FIXED_OFFSET_PATTERN.match(name):
        sign = -1 if match.group(1) == "-" else 1
        hours = int(match.group(2))
        minutes = int(match.group(3) or 0)
        offset = timedelta(hours=hours, minutes=minutes) * sign
        return timezone(offset)

    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Unsupported timezone {value!r}. Use an IANA timezone like "
            "'UTC', 'America/New_York', 'Europe/London', or a fixed offset like 'GMT-6'."
        ) from exc


def normalize_datetime_series(
    values,
    *,
    source_timezone: str | tzinfo | None = "UTC",
    canonical_timezone: str | tzinfo = "UTC",
) -> pd.Series:
    if isinstance(values, pd.Series):
        raw = values.copy()
    else:
        raw = pd.Series(values)

    parsed = pd.to_datetime(raw, errors="coerce")
    if not isinstance(parsed, pd.Series):
        parsed = pd.Series(parsed, index=raw.index, name=getattr(raw, "name", None))

    if parsed.empty:
        return parsed

    if parsed.dt.tz is None:
        source_tz = resolve_timezone(source_timezone or canonical_timezone)
        try:
            localized = parsed.dt.tz_localize(
                source_tz,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Could not localize naive datetimes. Supply a correct fixed-offset or IANA "
                f"source timezone instead of relying on ambiguous local timestamps: {source_timezone!r}"
            ) from exc
    else:
        localized = parsed

    return localized.dt.tz_convert(resolve_timezone(canonical_timezone))


def convert_timezone(
    datetime_values,
    *,
    target_timezone: str | tzinfo,
    source_timezone: str | tzinfo | None = "UTC",
    canonical_timezone: str | tzinfo = "UTC",
) -> pd.Series:
    normalized = normalize_datetime_series(
        datetime_values,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    return normalized.dt.tz_convert(resolve_timezone(target_timezone))


def classify_session_regime(
    *,
    in_asian_session: pd.Series,
    in_london_session: pd.Series,
    in_newyork_session: pd.Series,
) -> pd.Series:
    asian = in_asian_session.fillna(0).astype(bool)
    london = in_london_session.fillna(0).astype(bool)
    new_york = in_newyork_session.fillna(0).astype(bool)

    regime = pd.Series("off_hours", index=in_london_session.index, dtype="object")
    regime.loc[asian & ~london & ~new_york] = "asia"
    regime.loc[london & ~new_york] = "london"
    regime.loc[london & new_york] = "overlap"
    regime.loc[new_york & ~london] = "new_york"
    return regime


def build_intraday_calendar_frame(
    datetime_values,
    *,
    source_timezone: str | tzinfo | None = "UTC",
    canonical_timezone: str | tzinfo = "UTC",
    feature_clock_timezone: str | tzinfo = "America/New_York",
    london_timezone: str | tzinfo = "Europe/London",
    new_york_timezone: str | tzinfo = "America/New_York",
    asia_session_reference_timezone: str | tzinfo = "America/New_York",
    asia_session_start_hour: int = 19,
    asia_session_end_hour: int = 2,
    london_session_start_hour: int = 7,
    london_session_end_hour: int = 16,
    new_york_session_start_hour: int = 8,
    new_york_session_end_hour: int = 17,
    london_killzone_reference_timezone: str | tzinfo = "America/New_York",
    london_killzone_start_hour: int = 2,
    london_killzone_end_hour: int = 5,
    new_york_killzone_reference_timezone: str | tzinfo = "America/New_York",
    new_york_killzone_start_hour: int = 8,
    new_york_killzone_end_hour: int = 11,
) -> pd.DataFrame:
    datetime_utc = normalize_datetime_series(
        datetime_values,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )

    feature_clock = datetime_utc.dt.tz_convert(resolve_timezone(feature_clock_timezone))
    london_local = datetime_utc.dt.tz_convert(resolve_timezone(london_timezone))
    new_york_local = datetime_utc.dt.tz_convert(resolve_timezone(new_york_timezone))
    asia_local = datetime_utc.dt.tz_convert(resolve_timezone(asia_session_reference_timezone))
    london_killzone_local = datetime_utc.dt.tz_convert(resolve_timezone(london_killzone_reference_timezone))
    new_york_killzone_local = datetime_utc.dt.tz_convert(resolve_timezone(new_york_killzone_reference_timezone))

    out = pd.DataFrame(index=datetime_utc.index)
    out["datetime_utc"] = datetime_utc
    out["feature_datetime"] = feature_clock
    out["feature_hour"] = feature_clock.dt.hour.astype("Int64")
    out["feature_minute"] = feature_clock.dt.minute.astype("Int64")
    out["feature_day_of_week"] = feature_clock.dt.dayofweek.astype("Int64")
    out["feature_month"] = feature_clock.dt.month.astype("Int64")
    out["in_asian_session"] = _time_window_mask(
        asia_local,
        start_hour=asia_session_start_hour,
        end_hour=asia_session_end_hour,
    ).astype(int)
    out["in_london_session"] = _time_window_mask(
        london_local,
        start_hour=london_session_start_hour,
        end_hour=london_session_end_hour,
    ).astype(int)
    out["in_newyork_session"] = _time_window_mask(
        new_york_local,
        start_hour=new_york_session_start_hour,
        end_hour=new_york_session_end_hour,
    ).astype(int)
    out["in_london_ny_overlap"] = (
        out["in_london_session"].astype(bool) & out["in_newyork_session"].astype(bool)
    ).astype(int)
    out["in_london_killzone"] = _time_window_mask(
        london_killzone_local,
        start_hour=london_killzone_start_hour,
        end_hour=london_killzone_end_hour,
    ).astype(int)
    out["in_newyork_killzone"] = _time_window_mask(
        new_york_killzone_local,
        start_hour=new_york_killzone_start_hour,
        end_hour=new_york_killzone_end_hour,
    ).astype(int)
    out["session_regime"] = classify_session_regime(
        in_asian_session=out["in_asian_session"],
        in_london_session=out["in_london_session"],
        in_newyork_session=out["in_newyork_session"],
    )
    return out


def build_market_day_close_labels(
    datetime_values,
    *,
    source_timezone: str | tzinfo | None = "UTC",
    canonical_timezone: str | tzinfo = "UTC",
    market_close_timezone: str | tzinfo = "America/New_York",
    market_close_hour: int = 17,
    market_close_minute: int = 0,
) -> pd.Series:
    datetime_utc = normalize_datetime_series(
        datetime_values,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    local = datetime_utc.dt.tz_convert(resolve_timezone(market_close_timezone))
    close_offset = pd.to_timedelta(int(market_close_hour), unit="h") + pd.to_timedelta(
        int(market_close_minute),
        unit="m",
    )
    shifted = local - close_offset
    market_day_midnight = shifted.dt.floor("D") + pd.Timedelta(days=1)
    market_day_close_local = market_day_midnight + close_offset
    return market_day_close_local.dt.tz_convert(resolve_timezone(canonical_timezone))


def build_market_week_close_labels(
    datetime_values,
    *,
    source_timezone: str | tzinfo | None = "UTC",
    canonical_timezone: str | tzinfo = "UTC",
    market_close_timezone: str | tzinfo = "America/New_York",
    market_close_hour: int = 17,
    market_close_minute: int = 0,
) -> pd.Series:
    datetime_utc = normalize_datetime_series(
        datetime_values,
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    local = datetime_utc.dt.tz_convert(resolve_timezone(market_close_timezone))
    close_offset = pd.to_timedelta(int(market_close_hour), unit="h") + pd.to_timedelta(
        int(market_close_minute),
        unit="m",
    )
    shifted = local - close_offset
    market_day_midnight = shifted.dt.floor("D") + pd.Timedelta(days=1)
    market_day_naive = market_day_midnight.dt.tz_localize(None)
    week_end_naive = market_day_naive.dt.to_period("W-FRI").dt.to_timestamp(how="end").dt.floor("D")
    week_close_naive = week_end_naive + close_offset
    week_close_local = week_close_naive.dt.tz_localize(resolve_timezone(market_close_timezone))
    return week_close_local.dt.tz_convert(resolve_timezone(canonical_timezone))


def resample_ohlcv_by_close_labels(
    frame: pd.DataFrame,
    close_labels: pd.Series,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    working = frame.copy()
    aligned_labels = close_labels.reindex(working.index)
    valid_mask = aligned_labels.notna()
    if not valid_mask.any():
        return working.iloc[0:0]

    working = working.loc[valid_mask].copy()
    working["_period_close"] = aligned_labels.loc[valid_mask].to_numpy()

    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in working.columns:
        aggregation["volume"] = "sum"

    out = working.groupby("_period_close", sort=True).agg(aggregation)
    out.index = pd.DatetimeIndex(out.index)
    return out.dropna(subset=["open", "high", "low", "close"])


def _time_window_mask(
    datetime_values: pd.Series,
    *,
    start_hour: int,
    end_hour: int,
    start_minute: int = 0,
    end_minute: int = 0,
) -> pd.Series:
    minutes = (datetime_values.dt.hour * 60) + datetime_values.dt.minute
    start = (int(start_hour) * 60) + int(start_minute)
    end = (int(end_hour) * 60) + int(end_minute)

    if start == end:
        mask = pd.Series(False, index=datetime_values.index)
    elif start < end:
        mask = (minutes >= start) & (minutes < end)
    else:
        mask = (minutes >= start) | (minutes < end)
    return mask.fillna(False)
