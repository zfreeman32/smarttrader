from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .fx_calendar import normalize_datetime_series


REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close")


def load_market_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV into a DataFrame."""
    return pd.read_csv(path)


def standardize_market_frame(
    df: pd.DataFrame,
    *,
    source_timezone: str | None = "UTC",
    canonical_timezone: str = "UTC",
) -> pd.DataFrame:
    """Normalize common OHLCV column names and sort by time if available."""
    canonical = {
        "date": "date",
        "time": "time",
        "datetime": "datetime",
        "timestamp": "datetime",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }

    rename_map: Dict[str, str] = {}
    for column in df.columns:
        normalized = column.strip().lower()
        if normalized in canonical:
            rename_map[column] = canonical[normalized]

    df = df.rename(columns=rename_map)

    if "datetime" not in df.columns:
        if {"date", "time"}.issubset(df.columns):
            combined = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
            parsed = pd.to_datetime(combined, format="%Y%m%d %H:%M:%S", errors="coerce")
            if parsed.isna().all():
                parsed = pd.to_datetime(combined, errors="coerce")
            df["datetime"] = parsed
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")

    if "datetime" in df.columns:
        df["datetime"] = normalize_datetime_series(
            df["datetime"],
            source_timezone=source_timezone,
            canonical_timezone=canonical_timezone,
        )
        if "date" in df.columns:
            df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
        if "time" in df.columns:
            df["time"] = df["datetime"].dt.strftime("%H:%M:%S")
        if not df["datetime"].is_monotonic_increasing:
            df = df.sort_values("datetime").reset_index(drop=True)

    for column in ("open", "high", "low", "close", "volume"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def validate_ohlcv(df: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    """Validate OHLC relationships and optionally drop broken rows."""
    invalid = pd.Series(False, index=df.index)

    missing_price_cols = [column for column in REQUIRED_PRICE_COLUMNS if column not in df.columns]
    if missing_price_cols:
        raise ValueError(f"Missing required price columns: {missing_price_cols}")

    invalid |= df[list(REQUIRED_PRICE_COLUMNS)].isna().any(axis=1)
    invalid |= df["high"] < df[["open", "close", "low"]].max(axis=1)
    invalid |= df["low"] > df[["open", "close", "high"]].min(axis=1)

    valid_df = df.loc[~invalid].copy() if drop_invalid else df.copy()
    valid_df.attrs["invalid_rows_removed"] = int(invalid.sum())
    return valid_df.reset_index(drop=True)


def save_dataset(
    df: pd.DataFrame,
    metadata: Dict[str, object],
    output_path: str | Path,
) -> Tuple[Path, Path]:
    """Save the feature dataset plus a metadata sidecar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    metadata_path = output_path.with_suffix(".metadata.json")
    payload = dict(metadata)
    payload["saved_at_utc"] = datetime.now(timezone.utc).isoformat()

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return output_path, metadata_path
