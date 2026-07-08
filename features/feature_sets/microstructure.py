from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import safe_divide
from frvp.config.instruments import get_instrument_config


@register_feature_set(
    name="microstructure",
    category="context",
    description="Spread, impact, illiquidity, and intrabar microstructure proxies",
    required_columns=("open", "high", "low", "close", "volume"),
)
def build_microstructure(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    bar_range = df["high"] - df["low"]
    volume = df["volume"].replace(0, np.nan)
    approx_spread = _resolve_approx_spread_proxy(bar_range, config)

    out["approx_spread"] = approx_spread
    out["relative_spread_bps"] = safe_divide(approx_spread, df["close"]) * 10000.0
    out["price_impact_proxy"] = safe_divide(df["close"] - df["open"], volume)
    out["amihud_proxy"] = safe_divide(df["close"].pct_change().abs(), volume)
    out["kyle_lambda_proxy"] = safe_divide((df["close"] - df["open"]).abs(), volume)
    out["order_flow_toxicity"] = df["close"].diff().fillna(0.0) * df["volume"].fillna(0.0)
    out["intrabar_efficiency"] = safe_divide((df["close"] - df["open"]).abs(), bar_range)
    out["range_to_volume"] = safe_divide(bar_range, volume)
    out["close_location_micro"] = safe_divide(df["close"] - df["low"], bar_range, fill_value=0.5)

    return out


def _resolve_approx_spread_proxy(
    bar_range: pd.Series,
    config: FeatureBuilderConfig,
) -> pd.Series:
    instrument = str(config.instrument or "").strip().lower()
    if instrument not in {"es", "6e"}:
        return bar_range

    tick_size = float(get_instrument_config(instrument).tick_size)
    return pd.to_numeric(bar_range, errors="coerce").clip(
        lower=tick_size,
        upper=(2.0 * tick_size),
    )
