from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import safe_divide


@register_feature_set(
    name="price_action",
    category="core",
    description="Returns, candle anatomy, and close-location features",
    required_columns=("open", "high", "low", "close"),
)
def build_price_action(
    df: pd.DataFrame,
    _: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    candle_range = df["high"] - df["low"]
    candle_body = df["close"] - df["open"]
    body_abs = candle_body.abs()
    upper_body = pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
    lower_body = pd.concat([df["open"], df["close"]], axis=1).min(axis=1)
    positive_ratio = (df["close"] / df["close"].shift(1)).where(lambda series: series > 0)

    out["close_return_1"] = df["close"].pct_change(1)
    out["close_return_3"] = df["close"].pct_change(3)
    out["close_return_5"] = df["close"].pct_change(5)
    out["close_return_10"] = df["close"].pct_change(10)
    out["log_return_1"] = np.log(positive_ratio)
    out["candle_body"] = candle_body
    out["candle_body_pct"] = safe_divide(candle_body, df["open"])
    out["candle_range"] = candle_range
    out["upper_wick"] = df["high"] - upper_body
    out["lower_wick"] = lower_body - df["low"]
    out["body_to_range"] = safe_divide(body_abs, candle_range)
    out["close_location"] = safe_divide(df["close"] - df["low"], candle_range, fill_value=0.5)
    out["gap_return_1"] = safe_divide(df["open"] - df["close"].shift(1), df["close"].shift(1))
    out["range_expansion_1"] = safe_divide(candle_range, candle_range.shift(1))
    out["bullish_candle"] = (df["close"] > df["open"]).astype(int)
    out["bearish_candle"] = (df["close"] < df["open"]).astype(int)
    out["doji_like"] = (out["body_to_range"] < 0.1).astype(int)

    return out
