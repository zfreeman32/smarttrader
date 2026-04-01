from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..fx_calendar import build_intraday_calendar_frame
from ..registry import register_feature_set


@register_feature_set(
    name="session",
    category="context",
    description="Session, overlap, kill-zone, and cyclical time features",
    required_columns=("datetime",),
)
def build_session(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    calendar = build_intraday_calendar_frame(
        df["datetime"],
        source_timezone=config.source_timezone,
        canonical_timezone=config.canonical_timezone,
        feature_clock_timezone=config.feature_clock_timezone,
        london_timezone=config.london_timezone,
        new_york_timezone=config.new_york_timezone,
        asia_session_reference_timezone=config.asia_session_reference_timezone,
        asia_session_start_hour=config.asia_session_start_hour,
        asia_session_end_hour=config.asia_session_end_hour,
        london_session_start_hour=config.london_session_start_hour,
        london_session_end_hour=config.london_session_end_hour,
        new_york_session_start_hour=config.new_york_session_start_hour,
        new_york_session_end_hour=config.new_york_session_end_hour,
        london_killzone_reference_timezone=config.london_killzone_reference_timezone,
        london_killzone_start_hour=config.london_killzone_start_hour,
        london_killzone_end_hour=config.london_killzone_end_hour,
        new_york_killzone_reference_timezone=config.new_york_killzone_reference_timezone,
        new_york_killzone_start_hour=config.new_york_killzone_start_hour,
        new_york_killzone_end_hour=config.new_york_killzone_end_hour,
    )
    hour = calendar["feature_hour"].fillna(0).astype(int)
    minute = calendar["feature_minute"].fillna(0).astype(int)
    day_of_week = calendar["feature_day_of_week"].fillna(0).astype(int)
    month = calendar["feature_month"].fillna(0).astype(int)

    out["hour"] = hour
    out["minute"] = minute
    out["day_of_week"] = day_of_week
    out["month"] = month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    out["in_asian_session"] = calendar["in_asian_session"].astype(int)
    out["in_london_session"] = calendar["in_london_session"].astype(int)
    out["in_newyork_session"] = calendar["in_newyork_session"].astype(int)
    out["in_london_ny_overlap"] = calendar["in_london_ny_overlap"].astype(int)
    out["in_london_killzone"] = calendar["in_london_killzone"].astype(int)
    out["in_newyork_killzone"] = calendar["in_newyork_killzone"].astype(int)

    return out
