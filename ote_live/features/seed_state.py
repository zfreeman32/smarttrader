from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ote_live.features.incremental_engine import IncrementalFeatureEngine, ManifestFeaturePlan
from ote_live.ingestion.base import ensure_utc
from ote_live.storage.db import SQLiteLiveDataStore

FEATURE_SEED_STATE_SCOPE = "feature_seed_offsets"


@dataclass(frozen=True)
class FeatureSeedState:
    state_key: str
    asset: str
    timeframe: str
    runtime_history_bars: int
    selected_feature_names: tuple[str, ...]
    additive_seed_feature_names: tuple[str, ...]
    feature_seed_offsets: dict[str, float]
    latest_timestamp: datetime | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "runtime_history_bars": int(self.runtime_history_bars),
            "selected_feature_names": list(self.selected_feature_names),
            "additive_seed_feature_names": list(self.additive_seed_feature_names),
            "feature_seed_offsets": {
                feature_name: float(value)
                for feature_name, value in self.feature_seed_offsets.items()
            },
            "latest_timestamp": (
                ensure_utc(self.latest_timestamp).isoformat()
                if self.latest_timestamp is not None
                else None
            ),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FeatureSeedState":
        latest_timestamp_raw = payload.get("latest_timestamp")
        return cls(
            state_key=str(payload["state_key"]),
            asset=str(payload["asset"]),
            timeframe=str(payload["timeframe"]),
            runtime_history_bars=int(payload["runtime_history_bars"]),
            selected_feature_names=tuple(payload.get("selected_feature_names", [])),
            additive_seed_feature_names=tuple(payload.get("additive_seed_feature_names", [])),
            feature_seed_offsets={
                str(feature_name): float(value)
                for feature_name, value in dict(payload.get("feature_seed_offsets", {})).items()
            },
            latest_timestamp=(
                ensure_utc(datetime.fromisoformat(str(latest_timestamp_raw)))
                if latest_timestamp_raw
                else None
            ),
        )


@dataclass(frozen=True)
class FeatureSeedBootstrapResult:
    loaded_bars: int
    restored_from_store: bool
    derived_from_history: bool
    latest_timestamp: datetime | None
    feature_seed_offsets: dict[str, float]


def build_feature_seed_state_key(plan: ManifestFeaturePlan) -> str:
    payload = {
        "asset": plan.asset,
        "timeframe": plan.timeframe,
        "runtime_history_bars": int(plan.runtime_history_bars),
        "selected_feature_names": list(plan.selected_feature_names),
        "additive_seed_feature_names": list(plan.additive_seed_feature_names),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{plan.asset}:{plan.timeframe}:{digest}"


def capture_feature_seed_state(engine: IncrementalFeatureEngine) -> FeatureSeedState | None:
    if not engine.plan.additive_seed_feature_names:
        return None
    return FeatureSeedState(
        state_key=build_feature_seed_state_key(engine.plan),
        asset=engine.plan.asset,
        timeframe=engine.plan.timeframe,
        runtime_history_bars=engine.plan.runtime_history_bars,
        selected_feature_names=engine.plan.selected_feature_names,
        additive_seed_feature_names=engine.plan.additive_seed_feature_names,
        feature_seed_offsets=engine.get_feature_seed_offsets(),
        latest_timestamp=engine.state.latest_timestamp,
    )


def persist_feature_seed_state(
    store: SQLiteLiveDataStore,
    engine: IncrementalFeatureEngine,
) -> FeatureSeedState | None:
    seed_state = capture_feature_seed_state(engine)
    if seed_state is None:
        return None
    store.upsert_runtime_state(
        scope=FEATURE_SEED_STATE_SCOPE,
        state_key=seed_state.state_key,
        payload=seed_state.to_payload(),
    )
    return seed_state


def load_feature_seed_state(
    store: SQLiteLiveDataStore,
    engine: IncrementalFeatureEngine,
) -> FeatureSeedState | None:
    payload = store.get_runtime_state(
        scope=FEATURE_SEED_STATE_SCOPE,
        state_key=build_feature_seed_state_key(engine.plan),
    )
    if payload is None:
        return None
    seed_state = FeatureSeedState.from_payload(payload)
    if seed_state.additive_seed_feature_names != engine.plan.additive_seed_feature_names:
        return None
    return seed_state


def restore_feature_seed_state(
    store: SQLiteLiveDataStore,
    engine: IncrementalFeatureEngine,
    *,
    expected_timestamp: datetime | None = None,
) -> FeatureSeedState | None:
    seed_state = load_feature_seed_state(store, engine)
    if seed_state is None:
        return None
    if expected_timestamp is not None:
        expected_timestamp = ensure_utc(expected_timestamp)
        if seed_state.latest_timestamp != expected_timestamp:
            return None
    engine.set_feature_seed_offsets(seed_state.feature_seed_offsets)
    return seed_state


def warm_feature_engine_from_store(
    store: SQLiteLiveDataStore,
    engine: IncrementalFeatureEngine,
) -> FeatureSeedBootstrapResult:
    bars = store.fetch_bars(asset=engine.plan.asset, timeframe=engine.plan.timeframe)
    engine.extend(bars)

    restored = restore_feature_seed_state(
        store,
        engine,
        expected_timestamp=engine.state.latest_timestamp,
    )
    derived = False
    if (
        restored is None
        and engine.plan.additive_seed_feature_names
        and bars
        and len(bars) > len(engine.state)
    ):
        history_frame = engine.market_frame_from_bars(bars)
        derived_offsets = engine.derive_feature_seed_offsets_from_market_frame(history_frame)
        if derived_offsets:
            engine.set_feature_seed_offsets(derived_offsets)
            derived = True

    return FeatureSeedBootstrapResult(
        loaded_bars=len(bars),
        restored_from_store=restored is not None,
        derived_from_history=derived,
        latest_timestamp=engine.state.latest_timestamp,
        feature_seed_offsets=engine.get_feature_seed_offsets(),
    )
