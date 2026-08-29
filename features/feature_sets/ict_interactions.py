from __future__ import annotations

import pandas as pd

from ict.feature_sets.ict_interactions import build_ict_interaction_features

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set


@register_feature_set(
    name="ict_interactions",
    category="context",
    description="ICT interaction features layered on top of the base ICT context family",
)
def build_ict_interactions(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    """Thin registry shim for the package-native ICT interaction family."""

    return build_ict_interaction_features(df, config)
