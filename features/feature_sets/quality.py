from __future__ import annotations

import pandas as pd

from ..config import FeatureBuilderConfig
from ..fx_calendar import normalize_datetime_series
from ..registry import register_feature_set
from ..transforms import safe_divide


@register_feature_set(
    name="quality",
    category="governance",
    description="Anomaly, bar-quality, and data-integrity flags",
    required_columns=("open", "high", "low", "close"),
)
def build_quality(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    bar_range = df["high"] - df["low"]
    atr_like = df["atr_14"] if "atr_14" in df.columns else bar_range.rolling(14).mean()

    out["anomaly_range_gt_10atr"] = (bar_range > (atr_like * 10)).astype(int)
    out["close_outside_bar_flag"] = ((df["close"] > df["high"]) | (df["close"] < df["low"])).astype(int)
    out["open_outside_bar_flag"] = ((df["open"] > df["high"]) | (df["open"] < df["low"])).astype(int)
    out["range_vs_atr"] = safe_divide(bar_range, atr_like)

    if "volume" in df.columns:
        out["zero_volume_flag"] = (df["volume"].fillna(0) <= 0).astype(int)
    else:
        out["zero_volume_flag"] = 0

    if "datetime" in df.columns:
        dt = normalize_datetime_series(
            df["datetime"],
            source_timezone=config.source_timezone,
            canonical_timezone=config.canonical_timezone,
        )
        out["large_time_gap_flag"] = (dt.diff().dt.total_seconds().fillna(0) > 60 * 60).astype(int)

    return out
