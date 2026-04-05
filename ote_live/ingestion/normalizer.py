from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from features.fx_calendar import normalize_datetime_series
from features.io import validate_ohlcv
from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import CanonicalTimeframe, canonical_asset_symbol

_TWELVEDATA_TO_CANONICAL_INTERVAL: dict[str, CanonicalTimeframe] = {
    "1min": "1m",
    "5min": "5m",
    "30min": "30m",
    "1h": "1h",
}

_CANONICAL_TO_TWELVEDATA_INTERVAL = {
    value: key for key, value in _TWELVEDATA_TO_CANONICAL_INTERVAL.items()
}


def twelvedata_interval_to_canonical_timeframe(interval: str) -> CanonicalTimeframe:
    normalized = interval.strip().lower()
    if normalized not in _TWELVEDATA_TO_CANONICAL_INTERVAL:
        raise ValueError(f"Unsupported Twelve Data interval: {interval!r}")
    return _TWELVEDATA_TO_CANONICAL_INTERVAL[normalized]


def canonical_timeframe_to_twelvedata_interval(timeframe: CanonicalTimeframe) -> str:
    return _CANONICAL_TO_TWELVEDATA_INTERVAL[timeframe]


def canonical_asset_to_twelvedata_symbol(asset: str) -> str:
    normalized = canonical_asset_symbol(asset)
    if len(normalized) == 6 and normalized.isalpha():
        return f"{normalized[:3]}/{normalized[3:]}"
    return normalized


def normalize_twelvedata_time_series_response(
    payload: Mapping[str, Any],
    *,
    source: str = "twelvedata.rest.time_series",
) -> list[MarketBar]:
    status = str(payload.get("status", "")).lower()
    if status and status != "ok":
        message = payload.get("message") or payload.get("code") or "Unknown Twelve Data API error."
        raise ValueError(f"Twelve Data returned an error response: {message}")

    meta = payload.get("meta") or {}
    symbol = meta.get("symbol")
    interval = meta.get("interval")
    if not symbol or not interval:
        raise ValueError("Twelve Data response is missing meta.symbol or meta.interval.")

    source_timezone = str(meta.get("exchange_timezone") or "UTC")
    bars = [
        _normalize_twelvedata_bar(
            row,
            symbol=symbol,
            interval=interval,
            source_timezone=source_timezone,
            source=source,
        )
        for row in payload.get("values", []) or []
    ]
    bars.sort(key=lambda bar: bar.timestamp)
    return bars


def bars_to_dataframe(bars: list[MarketBar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "asset",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "bid",
                "ask",
                "spread",
                "source",
            ]
        )

    rows = [
        {
            "timestamp": bar.timestamp,
            "asset": bar.asset,
            "timeframe": bar.timeframe,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "bid": bar.bid,
            "ask": bar.ask,
            "spread": bar.spread,
            "source": bar.source,
        }
        for bar in bars
    ]
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def dataframe_to_market_bars(
    frame: pd.DataFrame,
    *,
    asset: str,
    timeframe: CanonicalTimeframe,
    source: str | None = None,
) -> list[MarketBar]:
    if frame.empty:
        return []

    working = frame.copy()
    if "timestamp" not in working.columns:
        if "datetime" in working.columns:
            working = working.rename(columns={"datetime": "timestamp"})
        else:
            raise ValueError("Expected a timestamp or datetime column.")

    working["timestamp"] = normalize_datetime_series(
        working["timestamp"],
        source_timezone="UTC",
        canonical_timezone="UTC",
    )
    if "volume" not in working.columns:
        working["volume"] = 0.0

    bars: list[MarketBar] = []
    for row in working.itertuples(index=False):
        bars.append(
            MarketBar(
                asset=canonical_asset_symbol(asset),
                timeframe=timeframe,
                timestamp=getattr(row, "timestamp"),
                open=float(getattr(row, "open")),
                high=float(getattr(row, "high")),
                low=float(getattr(row, "low")),
                close=float(getattr(row, "close")),
                volume=float(getattr(row, "volume", 0.0) or 0.0),
                bid=_optional_float(getattr(row, "bid", None)),
                ask=_optional_float(getattr(row, "ask", None)),
                spread=_optional_float(getattr(row, "spread", None)),
                source=source,
            )
        )
    return bars


def load_local_csv_bars(
    path: str | Path,
    *,
    asset: str = "EURUSD",
    timeframe: CanonicalTimeframe,
    source_timezone: str = "UTC",
    canonical_timezone: str = "UTC",
    source: str | None = None,
    skiprows: int = 0,
    nrows: int | None = None,
) -> list[MarketBar]:
    path = Path(path)
    raw = _read_local_csv_frame(path, skiprows=skiprows, nrows=nrows)

    raw["timestamp"] = normalize_datetime_series(
        raw["timestamp"],
        source_timezone=source_timezone,
        canonical_timezone=canonical_timezone,
    )
    validated = validate_ohlcv(
        raw[["timestamp", "open", "high", "low", "close", "volume"]],
        drop_invalid=True,
    )
    validated = validated.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return dataframe_to_market_bars(
        validated,
        asset=asset,
        timeframe=timeframe,
        source=source or f"local_csv:{path.name}",
    )


def _normalize_twelvedata_bar(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    interval: str,
    source_timezone: str,
    source: str,
) -> MarketBar:
    timestamp = normalize_datetime_series(
        pd.Series([payload["datetime"]]),
        source_timezone=source_timezone,
        canonical_timezone="UTC",
    ).iloc[0]

    return MarketBar(
        asset=canonical_asset_symbol(symbol),
        timeframe=twelvedata_interval_to_canonical_timeframe(interval),
        timestamp=timestamp,
        open=float(payload["open"]),
        high=float(payload["high"]),
        low=float(payload["low"]),
        close=float(payload["close"]),
        volume=float(payload.get("volume") or 0.0),
        source=source,
    )


def _read_local_csv_frame(path: Path, *, skiprows: int = 0, nrows: int | None = None) -> pd.DataFrame:
    first_line = path.open("r", encoding="utf-8").readline().strip()

    if first_line.lower().startswith("date,time,"):
        raw = pd.read_csv(path, skiprows=skiprows, nrows=nrows)
        date_column = next(column for column in raw.columns if str(column).strip().lower() == "date")
        time_column = next(column for column in raw.columns if str(column).strip().lower() == "time")
        parsed = _parse_date_time_columns(raw[date_column], raw[time_column])
        raw = raw.rename(
            columns={
                date_column: "date",
                time_column: "time",
                next(column for column in raw.columns if str(column).strip().lower() == "open"): "open",
                next(column for column in raw.columns if str(column).strip().lower() == "high"): "high",
                next(column for column in raw.columns if str(column).strip().lower() == "low"): "low",
                next(column for column in raw.columns if str(column).strip().lower() == "close"): "close",
                next(column for column in raw.columns if str(column).strip().lower() == "volume"): "volume",
            }
        )
    else:
        raw = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["date", "time", "open", "high", "low", "close", "volume"],
            skiprows=skiprows,
            nrows=nrows,
        )
        parsed = _parse_date_time_columns(raw["date"], raw["time"])

    frame = pd.DataFrame(
        {
            "timestamp": parsed,
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw["volume"], errors="coerce").fillna(0.0),
        }
    )
    return frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(drop=True)


def _parse_date_time_columns(date_values: pd.Series, time_values: pd.Series) -> pd.Series:
    date_values = date_values.astype(str).str.strip()
    time_values = time_values.astype(str).str.strip()
    combined = date_values + " " + time_values

    has_slashes = date_values.str.contains("/", regex=False).any()
    if has_slashes:
        parsed = pd.to_datetime(combined, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        if parsed.isna().any():
            parsed = pd.to_datetime(combined, errors="coerce", dayfirst=True)
        return pd.Series(parsed)

    parsed = pd.to_datetime(combined, format="%Y%m%d %H:%M:%S", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(combined, errors="coerce")
    return pd.Series(parsed)


def _optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)
