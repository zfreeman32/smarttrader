from __future__ import annotations

from dataclasses import dataclass

from ote_live.features.manifest import LivePolicy


@dataclass(frozen=True)
class ThresholdResolution:
    threshold: float
    source: str
    regime: str | None = None
    note: str | None = None


def resolve_live_threshold(
    live_policy: LivePolicy,
    *,
    regime: str | None = None,
) -> ThresholdResolution:
    regime_thresholds = live_policy.thresholds.regime_thresholds or {}
    if regime is not None and regime in regime_thresholds:
        return ThresholdResolution(
            threshold=float(regime_thresholds[regime]),
            source="regime",
            regime=regime,
        )

    if live_policy.thresholds.global_threshold is not None:
        note = None
        if regime is None and regime_thresholds:
            note = "regime_unavailable_global_fallback"
        elif regime is not None and regime_thresholds:
            note = "regime_threshold_missing_global_fallback"
        return ThresholdResolution(
            threshold=float(live_policy.thresholds.global_threshold),
            source="global",
            regime=regime,
            note=note,
        )

    if regime_thresholds:
        available = ", ".join(sorted(regime_thresholds))
        raise ValueError(
            f"Live policy for {live_policy.model_id} only has regime thresholds. "
            f"Received regime={regime!r}; available regimes: {available}."
        )

    raise ValueError(f"Live policy for {live_policy.model_id} does not define any thresholds.")


def clears_threshold(
    probability: float,
    resolution: ThresholdResolution,
) -> bool:
    return float(probability) >= float(resolution.threshold)
