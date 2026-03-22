from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import safe_divide


@register_feature_set(
    name="trend",
    category="core",
    description="EMA distance, slope, alignment, and MACD trend features",
    required_columns=("close",),
)
def build_trend(
    df: pd.DataFrame,
    _: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    atr_like = df["atr_14"] if "atr_14" in df.columns else df["close"].diff().abs().rolling(14).mean()
    ema_8 = df["close"].ewm(span=8, adjust=False).mean()
    ema_21 = df["close"].ewm(span=21, adjust=False).mean()
    ema_50 = df["close"].ewm(span=50, adjust=False).mean()
    ema_200 = df["close"].ewm(span=200, adjust=False).mean()
    sma_20 = df["close"].rolling(20).mean()
    sma_50 = df["close"].rolling(50).mean()

    macd_fast = df["close"].ewm(span=12, adjust=False).mean()
    macd_slow = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    out["ema_8"] = ema_8
    out["ema_21"] = ema_21
    out["ema_50"] = ema_50
    out["ema_200"] = ema_200
    out["sma_20"] = sma_20
    out["sma_50"] = sma_50
    out["close_vs_ema_8_atr"] = safe_divide(df["close"] - ema_8, atr_like)
    out["close_vs_ema_21_atr"] = safe_divide(df["close"] - ema_21, atr_like)
    out["close_vs_ema_50_atr"] = safe_divide(df["close"] - ema_50, atr_like)
    out["close_vs_sma_20_atr"] = safe_divide(df["close"] - sma_20, atr_like)
    out["ema_spread_8_21_atr"] = safe_divide(ema_8 - ema_21, atr_like)
    out["ema_spread_21_50_atr"] = safe_divide(ema_21 - ema_50, atr_like)
    out["ema_slope_8_3"] = safe_divide(ema_8.diff(3), atr_like)
    out["ema_slope_21_5"] = safe_divide(ema_21.diff(5), atr_like)
    out["ema_alignment"] = np.where(
        (ema_8 > ema_21) & (ema_21 > ema_50),
        1,
        np.where((ema_8 < ema_21) & (ema_21 < ema_50), -1, 0),
    )
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    return out
