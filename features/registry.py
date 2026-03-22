from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pandas as pd

from .config import FeatureBuilderConfig


FeatureBuilder = Callable[[pd.DataFrame, FeatureBuilderConfig], pd.DataFrame]


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    category: str
    description: str
    required_columns: Tuple[str, ...]
    builder: FeatureBuilder


class FeatureRegistry:
    """Registry of reusable feature-set builders."""

    def __init__(self) -> None:
        self._specs: Dict[str, FeatureSetSpec] = {}

    def register(
        self,
        *,
        name: str,
        category: str,
        description: str,
        required_columns: Tuple[str, ...],
        builder: FeatureBuilder,
    ) -> FeatureBuilder:
        if name in self._specs:
            raise ValueError(f"Feature set '{name}' is already registered.")
        self._specs[name] = FeatureSetSpec(
            name=name,
            category=category,
            description=description,
            required_columns=required_columns,
            builder=builder,
        )
        return builder

    def get(self, name: str) -> FeatureSetSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._specs))
            raise KeyError(f"Unknown feature set '{name}'. Available: {available}") from exc

    def names(self) -> List[str]:
        return sorted(self._specs)

    def describe(self) -> List[FeatureSetSpec]:
        return [self._specs[name] for name in self.names()]

    def build(self, name: str, df: pd.DataFrame, config: FeatureBuilderConfig) -> pd.DataFrame:
        spec = self.get(name)
        missing = [column for column in spec.required_columns if column not in df.columns]
        if missing:
            raise KeyError(f"Feature set '{name}' requires columns {missing}.")

        built = spec.builder(df, config)
        if not isinstance(built, pd.DataFrame):
            raise TypeError(f"Feature set '{name}' must return a pandas DataFrame.")
        return built


FEATURE_REGISTRY = FeatureRegistry()


def register_feature_set(
    *,
    name: str,
    category: str,
    description: str,
    required_columns: Tuple[str, ...] = (),
) -> Callable[[FeatureBuilder], FeatureBuilder]:
    """Decorator for registering a feature-set builder."""

    def decorator(builder: FeatureBuilder) -> FeatureBuilder:
        return FEATURE_REGISTRY.register(
            name=name,
            category=category,
            description=description,
            required_columns=required_columns,
            builder=builder,
        )

    return decorator
