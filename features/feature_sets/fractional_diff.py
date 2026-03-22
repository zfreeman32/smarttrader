from __future__ import annotations

import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set
from ..transforms import apply_fractional_diff


@register_feature_set(
    name="fractional_diff",
    category="transforms",
    description="Fractionally differentiated close series for non-stationarity treatment",
    required_columns=("close",),
)
def build_fractional_diff(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    fracdiff_close = apply_fractional_diff(
        df["close"],
        d=config.fractional_diff_d,
        threshold=config.fractional_diff_threshold,
        max_size=config.fractional_diff_max_size,
    )
    out["fracdiff_close"] = fracdiff_close
    out["fracdiff_close_return_1"] = fracdiff_close.diff()
    return out
