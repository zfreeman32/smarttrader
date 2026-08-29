from __future__ import annotations

import pandas as pd

from ..config import FeatureBuilderConfig
from ..registry import register_feature_set


@register_feature_set(
    name="frvp_context",
    category="context",
    description="FRVP profile-context feature family built on the Phase 0 continuity layer",
    required_columns=("datetime", "open", "high", "low", "close", "volume"),
)
def build_frvp_context(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    """Thin registry shim for the real FRVP feature family implementation."""

    # Import lazily because the real implementation imports sibling feature
    # modules. Importing it while ``features.feature_sets`` is still
    # registering modules creates a package-initialization cycle.
    from frvp.feature_sets.frvp_context import build_frvp_context_features

    return build_frvp_context_features(df, config, instrument=config.instrument)
