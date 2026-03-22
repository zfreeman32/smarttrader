from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import safe_divide


@register_feature_set(
    name="microstructure",
    category="context",
    description="Spread, impact, illiquidity, and intrabar microstructure proxies",
    required_columns=("open", "high", "low", "close", "volume"),
)
def build_microstructure(
    df: pd.DataFrame,
    _: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    bar_range = df["high"] - df["low"]
    volume = df["volume"].replace(0, np.nan)

    out["approx_spread"] = bar_range
    out["relative_spread_bps"] = safe_divide(bar_range, df["close"]) * 10000.0
    out["price_impact_proxy"] = safe_divide(df["close"] - df["open"], volume)
    out["amihud_proxy"] = safe_divide(df["close"].pct_change().abs(), volume)
    out["kyle_lambda_proxy"] = safe_divide((df["close"] - df["open"]).abs(), volume)
    out["order_flow_toxicity"] = df["close"].diff().fillna(0.0) * df["volume"].fillna(0.0)
    out["intrabar_efficiency"] = safe_divide((df["close"] - df["open"]).abs(), bar_range)
    out["range_to_volume"] = safe_divide(bar_range, volume)
    out["close_location_micro"] = safe_divide(df["close"] - df["low"], bar_range, fill_value=0.5)

    return out
