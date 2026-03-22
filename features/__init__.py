"""Modular feature-generation toolkit for trading datasets."""

from .builder import FeatureDatasetBuilder
from .config import FeatureBuilderConfig
from .preprocessing import FeaturePreprocessingPipeline, PreprocessingConfig
from .registry import FEATURE_REGISTRY, register_feature_set
from .strategy_registry import STRATEGY_REGISTRY

__all__ = [
    "FEATURE_REGISTRY",
    "FeatureBuilderConfig",
    "FeatureDatasetBuilder",
    "FeaturePreprocessingPipeline",
    "PreprocessingConfig",
    "STRATEGY_REGISTRY",
    "register_feature_set",
]
