from __future__ import annotations

import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import safe_divide


@register_feature_set(
    name="volatility",
    category="core",
    description="ATR, rolling volatility, and range-normalized volatility features",
    required_columns=("high", "low", "close"),
)
def build_volatility(
    df: pd.DataFrame,
    _: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_14 = true_range.rolling(14).mean()
    atr_50 = true_range.rolling(50).mean()
    returns = df["close"].pct_change()
    rolling_std_20 = returns.rolling(20).std()
    rolling_std_60 = returns.rolling(60).std()
    rolling_mean_20 = df["close"].rolling(20).mean()
    rolling_std_close_20 = df["close"].rolling(20).std()

    out["true_range"] = true_range
    out["atr_14"] = atr_14
    out["atr_50"] = atr_50
    out["atr_ratio_14_50"] = safe_divide(atr_14, atr_50)
    out["range_atr_14"] = safe_divide(df["high"] - df["low"], atr_14)
    out["rolling_vol_20"] = rolling_std_20
    out["rolling_vol_60"] = rolling_std_60
    out["volatility_expansion_20"] = safe_divide(rolling_std_20, rolling_std_60)
    out["bb_width_20"] = safe_divide(rolling_std_close_20 * 4.0, rolling_mean_20)
    out["donchian_width_20"] = safe_divide(
        df["high"].rolling(20).max() - df["low"].rolling(20).min(),
        atr_14,
    )
    out["range_shock_20"] = safe_divide(
        df["high"] - df["low"],
        (df["high"] - df["low"]).rolling(20).mean(),
    )

    return out
