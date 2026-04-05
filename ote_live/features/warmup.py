from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from ote_live.features.manifest import LiveRuntimeManifest

DEFAULT_STRATEGY_RECOMPUTE_BARS = 1000


@dataclass(frozen=True)
class WarmupStatus:
    ready: bool
    history_ready: bool
    features_ready: bool
    bars_available: int
    minimum_history_bars: int
    required_feature_count: int
    valid_feature_count: int
    missing_feature_names: tuple[str, ...]


def resolve_required_history_bars(manifests: Iterable[LiveRuntimeManifest]) -> int:
    manifests = tuple(manifests)
    if not manifests:
        raise ValueError("At least one live manifest is required.")
    return max(int(manifest.context_requirements.minimum_runtime_history_bars) for manifest in manifests)


def resolve_runtime_history_bars(
    manifests: Iterable[LiveRuntimeManifest],
    *,
    rolling_window_bars: int | None = None,
    has_strategy_features: bool = False,
    inferred_history_bars: int = 0,
) -> int:
    minimum_history_bars = resolve_required_history_bars(manifests)
    minimum_history_bars = max(minimum_history_bars, int(inferred_history_bars))
    if rolling_window_bars is not None:
        return max(minimum_history_bars, int(rolling_window_bars))
    if has_strategy_features:
        return max(minimum_history_bars, DEFAULT_STRATEGY_RECOMPUTE_BARS)
    return minimum_history_bars


def evaluate_feature_warmup(
    feature_frame: pd.DataFrame,
    required_feature_names: Sequence[str],
    *,
    minimum_history_bars: int,
) -> WarmupStatus:
    required_feature_names = tuple(dict.fromkeys(required_feature_names))
    bars_available = int(len(feature_frame))
    history_ready = bars_available >= int(minimum_history_bars)

    if bars_available <= 0:
        return WarmupStatus(
            ready=False,
            history_ready=False,
            features_ready=False,
            bars_available=0,
            minimum_history_bars=int(minimum_history_bars),
            required_feature_count=len(required_feature_names),
            valid_feature_count=0,
            missing_feature_names=required_feature_names,
        )

    latest = feature_frame.iloc[-1]
    missing_feature_names: list[str] = []
    for feature_name in required_feature_names:
        if feature_name not in feature_frame.columns:
            missing_feature_names.append(feature_name)
            continue
        if pd.isna(latest[feature_name]):
            missing_feature_names.append(feature_name)

    valid_feature_count = len(required_feature_names) - len(missing_feature_names)
    features_ready = not missing_feature_names
    return WarmupStatus(
        ready=history_ready and features_ready,
        history_ready=history_ready,
        features_ready=features_ready,
        bars_available=bars_available,
        minimum_history_bars=int(minimum_history_bars),
        required_feature_count=len(required_feature_names),
        valid_feature_count=valid_feature_count,
        missing_feature_names=tuple(missing_feature_names),
    )
