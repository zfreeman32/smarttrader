from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set


@register_feature_set(
    name="session",
    category="context",
    description="Session, overlap, kill-zone, and cyclical time features",
    required_columns=("datetime",),
)
def build_session(
    df: pd.DataFrame,
    _: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    dt = pd.to_datetime(df["datetime"], errors="coerce")
    hour = dt.dt.hour.fillna(0).astype(int)
    minute = dt.dt.minute.fillna(0).astype(int)
    day_of_week = dt.dt.dayofweek.fillna(0).astype(int)
    month = dt.dt.month.fillna(0).astype(int)

    out["hour"] = hour
    out["minute"] = minute
    out["day_of_week"] = day_of_week
    out["month"] = month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    out["in_asian_session"] = ((hour >= 19) | (hour < 2)).astype(int)
    out["in_london_session"] = ((hour >= 2) & (hour < 11)).astype(int)
    out["in_newyork_session"] = ((hour >= 8) & (hour < 17)).astype(int)
    out["in_london_ny_overlap"] = ((hour >= 8) & (hour < 11)).astype(int)
    out["in_london_killzone"] = ((hour >= 2) & (hour < 5)).astype(int)
    out["in_newyork_killzone"] = ((hour >= 8) & (hour < 11)).astype(int)

    return out
