"""Modular feature-generation toolkit for trading datasets."""

from __future__ import annotations

from typing import Any

__all__ = [
    "FEATURE_REGISTRY",
    "FeatureBuilderConfig",
    "FeatureDatasetBuilder",
    "FeaturePreprocessingPipeline",
    "PreprocessingConfig",
    "STRATEGY_REGISTRY",
    "register_feature_set",
]


def __getattr__(name: str) -> Any:
    if name == "FeatureDatasetBuilder":
        from .builder import FeatureDatasetBuilder

        return FeatureDatasetBuilder
    if name == "FeatureBuilderConfig":
        from .config import FeatureBuilderConfig

        return FeatureBuilderConfig
    if name in {"FeaturePreprocessingPipeline", "PreprocessingConfig"}:
        from .preprocessing import FeaturePreprocessingPipeline, PreprocessingConfig

        return {
            "FeaturePreprocessingPipeline": FeaturePreprocessingPipeline,
            "PreprocessingConfig": PreprocessingConfig,
        }[name]
    if name in {"FEATURE_REGISTRY", "register_feature_set"}:
        from .registry import FEATURE_REGISTRY, register_feature_set

        return {
            "FEATURE_REGISTRY": FEATURE_REGISTRY,
            "register_feature_set": register_feature_set,
        }[name]
    if name == "STRATEGY_REGISTRY":
        from .strategy_registry import STRATEGY_REGISTRY

        return STRATEGY_REGISTRY
    raise AttributeError(f"module 'features' has no attribute {name!r}")
