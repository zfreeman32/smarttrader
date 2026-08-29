from __future__ import annotations

import pandas as pd

from ict.feature_sets.ict_context import build_ict_context_features

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set


@register_feature_set(
    name="ict_context",
    category="context",
    description="ICT proximity features for FVGs, order blocks, liquidity pools, and structure breaks",
    required_columns=("open", "high", "low", "close"),
)
def build_ict_context(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    """Thin registry shim for the package-native ICT context family."""

    return build_ict_context_features(df, config)
