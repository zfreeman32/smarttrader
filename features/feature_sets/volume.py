from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import safe_divide


@register_feature_set(
    name="volume",
    category="core",
    description="Volume intensity, imbalance, and money-flow features",
    required_columns=("close", "volume"),
)
def build_volume(
    df: pd.DataFrame,
    _: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    volume = df["volume"].fillna(0)
    if float(volume.abs().sum()) == 0:
        out["log_volume"] = 0.0
        out["volume_relative_20"] = 0.0
        out["volume_zscore_50"] = 0.0
        out["up_volume_share_10"] = 0.5
        out["down_volume_share_10"] = 0.5
        out["volume_imbalance_10"] = 0.0
        out["money_flow_10"] = 0.0
        out["dollar_volume"] = 0.0
        return out

    price_change = df["close"].diff().fillna(0.0)
    up_volume = volume.where(price_change > 0, 0.0)
    down_volume = volume.where(price_change < 0, 0.0)
    total_volume_10 = volume.rolling(10).sum()
    log_volume = np.log1p(volume.clip(lower=0))

    out["log_volume"] = log_volume
    out["volume_relative_20"] = safe_divide(volume, volume.rolling(20).mean())
    out["volume_zscore_50"] = safe_divide(
        log_volume - log_volume.rolling(50).mean(),
        log_volume.rolling(50).std(),
    )
    out["up_volume_share_10"] = safe_divide(up_volume.rolling(10).sum(), total_volume_10, fill_value=0.5)
    out["down_volume_share_10"] = safe_divide(down_volume.rolling(10).sum(), total_volume_10, fill_value=0.5)
    out["volume_imbalance_10"] = out["up_volume_share_10"] - out["down_volume_share_10"]
    out["money_flow_10"] = ((df["close"] - df.get("open", df["close"])) * volume).rolling(10).sum()
    out["dollar_volume"] = df["close"] * volume

    return out
