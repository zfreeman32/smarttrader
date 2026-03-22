from __future__ import annotations

import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import bars_since_event, safe_divide


@register_feature_set(
    name="structure",
    category="core",
    description="Causal structure, sweep, premium/discount, and displacement features",
    required_columns=("open", "high", "low", "close"),
)
def build_structure(
    df: pd.DataFrame,
    _: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    atr_like = df["atr_14"] if "atr_14" in df.columns else (df["high"] - df["low"]).rolling(14).mean()
    prior_high_20 = df["high"].shift(1).rolling(20).max()
    prior_low_20 = df["low"].shift(1).rolling(20).min()
    prior_high_50 = df["high"].shift(1).rolling(50).max()
    prior_low_50 = df["low"].shift(1).rolling(50).min()
    range_size_50 = prior_high_50 - prior_low_50
    candle_range = df["high"] - df["low"]
    body_to_range = safe_divide((df["close"] - df["open"]).abs(), candle_range)
    range_atr = safe_divide(candle_range, atr_like)

    out["prior_high_20"] = prior_high_20
    out["prior_low_20"] = prior_low_20
    out["prior_high_50"] = prior_high_50
    out["prior_low_50"] = prior_low_50
    out["dist_to_prior_high_20_atr"] = safe_divide(prior_high_20 - df["close"], atr_like)
    out["dist_to_prior_low_20_atr"] = safe_divide(df["close"] - prior_low_20, atr_like)
    out["dist_to_prior_high_50_atr"] = safe_divide(prior_high_50 - df["close"], atr_like)
    out["dist_to_prior_low_50_atr"] = safe_divide(df["close"] - prior_low_50, atr_like)
    out["price_position_50"] = safe_divide(df["close"] - prior_low_50, range_size_50, fill_value=0.5)
    out["in_discount_zone"] = (out["price_position_50"] < 0.35).astype(int)
    out["in_premium_zone"] = (out["price_position_50"] > 0.65).astype(int)
    out["breakout_high_20"] = (df["high"] > prior_high_20).astype(int)
    out["breakout_low_20"] = (df["low"] < prior_low_20).astype(int)
    out["sweep_high_20"] = ((df["high"] > prior_high_20) & (df["close"] < prior_high_20)).astype(int)
    out["sweep_low_20"] = ((df["low"] < prior_low_20) & (df["close"] > prior_low_20)).astype(int)
    out["bars_since_sweep_high"] = bars_since_event(out["sweep_high_20"])
    out["bars_since_sweep_low"] = bars_since_event(out["sweep_low_20"])
    out["displacement_bullish"] = (
        (df["close"] > df["open"])
        & (body_to_range > 0.65)
        & (range_atr > 1.5)
    ).astype(int)
    out["displacement_bearish"] = (
        (df["close"] < df["open"])
        & (body_to_range > 0.65)
        & (range_atr > 1.5)
    ).astype(int)
    out["structure_pressure"] = out["breakout_high_20"] - out["breakout_low_20"]

    return out
