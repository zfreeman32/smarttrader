from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import safe_divide


@register_feature_set(
    name="momentum",
    category="core",
    description="RSI, ROC, acceleration, and exhaustion-style momentum features",
    required_columns=("close",),
)
def build_momentum(
    df: pd.DataFrame,
    _: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    atr_like = df["atr_14"] if "atr_14" in df.columns else df["close"].diff().abs().rolling(14).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    low_14 = df["close"].rolling(14).min()
    high_14 = df["close"].rolling(14).max()

    out["rsi_14"] = rsi
    out["rsi_from_mid"] = rsi - 50
    out["rsi_delta_3"] = rsi.diff(3)
    out["roc_5"] = df["close"].pct_change(5)
    out["roc_10"] = df["close"].pct_change(10)
    out["momentum_atr_5"] = safe_divide(df["close"] - df["close"].shift(5), atr_like)
    out["stoch_position_14"] = safe_divide(df["close"] - low_14, high_14 - low_14, fill_value=0.5)
    out["price_acceleration_3"] = out["roc_5"] - out["roc_5"].shift(3)

    if "close_vs_ema_8_atr" in df.columns:
        out["exhaustion_score"] = (
            df["close_vs_ema_8_atr"].abs()
            * out["rsi_from_mid"].abs()
            * out["price_acceleration_3"].fillna(0).mul(-1)
        )
    else:
        out["exhaustion_score"] = out["rsi_from_mid"].abs() * out["price_acceleration_3"].fillna(0).mul(-1)

    return out
