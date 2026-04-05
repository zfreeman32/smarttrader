from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from features.config import FeatureBuilderConfig
from features.strategy_registry import STRATEGY_REGISTRY, StrategyRegistry, slugify_strategy_name

STRATEGY_FEATURE_PREFIX = "strategy__"


def is_strategy_feature_name(feature_name: str) -> bool:
    return feature_name.startswith(STRATEGY_FEATURE_PREFIX)


def extract_strategy_slug(feature_name: str) -> str:
    if not is_strategy_feature_name(feature_name):
        raise ValueError(f"Feature '{feature_name}' is not a strategy feature.")

    parts = feature_name.split("__")
    if len(parts) < 3 or not parts[1]:
        raise ValueError(f"Strategy feature '{feature_name}' does not contain a strategy slug.")
    return parts[1]


@dataclass(frozen=True)
class StrategyFeatureAdapter:
    strategy_feature_names: tuple[str, ...]
    strategy_prefixes: tuple[str, ...]
    strategy_ids: tuple[str, ...]

    @property
    def has_strategy_features(self) -> bool:
        return bool(self.strategy_ids)

    @classmethod
    def from_feature_names(
        cls,
        feature_names: Iterable[str],
        *,
        registry: StrategyRegistry | None = None,
    ) -> "StrategyFeatureAdapter":
        registry = registry or STRATEGY_REGISTRY
        strategy_feature_names = tuple(name for name in feature_names if is_strategy_feature_name(name))
        if not strategy_feature_names:
            return cls(
                strategy_feature_names=(),
                strategy_prefixes=(),
                strategy_ids=(),
            )

        slug_index = _build_strategy_slug_index(registry)
        strategy_prefixes = tuple(dict.fromkeys(extract_strategy_slug(name) for name in strategy_feature_names))
        strategy_ids: list[str] = []
        for prefix in strategy_prefixes:
            matches = slug_index.get(prefix, ())
            if not matches:
                raise KeyError(f"Could not resolve strategy slug '{prefix}' from live-selected features.")
            if len(matches) > 1:
                options = ", ".join(matches)
                raise ValueError(
                    f"Strategy slug '{prefix}' is ambiguous for live-selected features. Matches: {options}"
                )
            strategy_ids.append(matches[0])

        return cls(
            strategy_feature_names=strategy_feature_names,
            strategy_prefixes=strategy_prefixes,
            strategy_ids=tuple(strategy_ids),
        )

    def apply_to_config(self, config: FeatureBuilderConfig) -> FeatureBuilderConfig:
        config.strategy_ids = list(self.strategy_ids)
        if self.strategy_ids:
            if "strategy_signals" not in config.feature_sets:
                config.feature_sets.append("strategy_signals")
        elif "strategy_signals" in config.feature_sets:
            config.feature_sets = [name for name in config.feature_sets if name != "strategy_signals"]
        return config


def _build_strategy_slug_index(registry: StrategyRegistry) -> dict[str, tuple[str, ...]]:
    slug_index: dict[str, list[str]] = {}
    for strategy_id in registry.names():
        slug = slugify_strategy_name(strategy_id)
        slug_index.setdefault(slug, [])
        if strategy_id not in slug_index[slug]:
            slug_index[slug].append(strategy_id)
    return {slug: tuple(values) for slug, values in slug_index.items()}
