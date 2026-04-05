from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.io import standardize_market_frame
from ote_live.contracts.market_data import MarketBar
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.features.seed_state import persist_feature_seed_state, warm_feature_engine_from_store
from ote_live.storage.db import SQLiteLiveDataStore

LONG_LSTM_V2_MANIFEST_PATH = (
    ROOT / "ote_live" / "runtime_manifests" / "long_ote_lstm_v2_candidate" / "live_runtime_manifest.json"
)
EURUSD_5M_PATH = ROOT / "data" / "currency_data" / "eurusd-5m.csv"


def _load_manifest(path: Path) -> LiveRuntimeManifest:
    return LiveRuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _subset_manifest(
    manifest: LiveRuntimeManifest,
    selected_feature_names: list[str],
) -> LiveRuntimeManifest:
    feature_manifest = manifest.feature_manifest.model_copy(
        update={
            "selected_feature_names": list(selected_feature_names),
            "selected_feature_count": len(selected_feature_names),
            "strategy_feature_count": sum(name.startswith("strategy__") for name in selected_feature_names),
        }
    )
    return manifest.model_copy(update={"feature_manifest": feature_manifest})


def test_incremental_engine_rolls_additive_seed_offsets_forward() -> None:
    manifest = _subset_manifest(
        _load_manifest(LONG_LSTM_V2_MANIFEST_PATH),
        [
            "strategy__pvi_signals__pvi",
            "strategy__ad_indicator_signals__ad_line",
        ],
    )
    engine = IncrementalFeatureEngine([manifest], rolling_window_bars=120)
    bootstrap_rows = engine.plan.runtime_history_bars + 70
    market_frame = _load_standardized_market_frame().tail(bootstrap_rows + 40).reset_index(drop=True)
    bootstrap_bars = [
        _market_bar_from_row(row, asset=manifest.asset, timeframe=manifest.timeframe)
        for row in market_frame.head(bootstrap_rows).itertuples(index=False)
    ]
    engine.extend(bootstrap_bars)
    engine.set_feature_seed_offsets(
        engine.derive_feature_seed_offsets_from_market_frame(market_frame.head(bootstrap_rows))
    )

    seed_before_roll = engine.get_feature_seed_offsets()
    engine.ingest_bar(
        _market_bar_from_row(
            market_frame.iloc[bootstrap_rows],
            asset=manifest.asset,
            timeframe=manifest.timeframe,
        )
    )
    seed_after_roll = engine.get_feature_seed_offsets()

    assert seed_before_roll
    assert seed_after_roll
    assert seed_after_roll != seed_before_roll
    assert seed_after_roll["strategy__pvi_signals__pvi"] >= seed_before_roll["strategy__pvi_signals__pvi"]
    assert seed_after_roll["strategy__ad_indicator_signals__ad_line"] != seed_before_roll[
        "strategy__ad_indicator_signals__ad_line"
    ]


def test_feature_seed_state_round_trips_through_live_store_bootstrap() -> None:
    manifest = _subset_manifest(
        _load_manifest(LONG_LSTM_V2_MANIFEST_PATH),
        [
            "strategy__pvi_signals__pvi",
            "strategy__ad_indicator_signals__ad_line",
            "ema_200",
        ],
    )
    probe_engine = IncrementalFeatureEngine([manifest], rolling_window_bars=120)
    market_frame = _load_standardized_market_frame().tail(probe_engine.plan.runtime_history_bars + 120).reset_index(
        drop=True
    )
    bars = [
        _market_bar_from_row(row, asset=manifest.asset, timeframe=manifest.timeframe)
        for row in market_frame.itertuples(index=False)
    ]

    tmp_root = ROOT / "tmp" / "ote_live_feature_seed_state_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        store.upsert_bars(bars)

        engine = IncrementalFeatureEngine([manifest], rolling_window_bars=120)
        bootstrap = warm_feature_engine_from_store(store, engine)
        baseline_feature_frame = engine.build_feature_frame()
        persisted_state = persist_feature_seed_state(store, engine)

        restored_engine = IncrementalFeatureEngine([manifest], rolling_window_bars=120)
        restored_bootstrap = warm_feature_engine_from_store(store, restored_engine)
        restored_feature_frame = restored_engine.build_feature_frame()

    assert bootstrap.loaded_bars == len(bars)
    assert bootstrap.restored_from_store is False
    assert bootstrap.derived_from_history is True
    assert persisted_state is not None

    assert restored_bootstrap.loaded_bars == len(bars)
    assert restored_bootstrap.restored_from_store is True
    assert restored_bootstrap.derived_from_history is False
    assert restored_bootstrap.feature_seed_offsets == bootstrap.feature_seed_offsets

    pd.testing.assert_frame_equal(
        baseline_feature_frame.reset_index(drop=True),
        restored_feature_frame.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        atol=1e-9,
        rtol=1e-9,
    )


def _load_standardized_market_frame() -> pd.DataFrame:
    raw = pd.read_csv(EURUSD_5M_PATH)
    standardized = standardize_market_frame(raw, source_timezone="GMT-6", canonical_timezone="UTC")
    standardized["datetime"] = pd.to_datetime(standardized["datetime"], errors="coerce", utc=True)
    return standardized.dropna(subset=["datetime"]).reset_index(drop=True)


def _market_bar_from_row(row, *, asset: str, timeframe: str) -> MarketBar:
    return MarketBar(
        asset=asset,
        timeframe=timeframe,
        timestamp=pd.Timestamp(getattr(row, "datetime")).to_pydatetime(),
        open=float(getattr(row, "open")),
        high=float(getattr(row, "high")),
        low=float(getattr(row, "low")),
        close=float(getattr(row, "close")),
        volume=float(getattr(row, "volume", 0.0) or 0.0),
        bid=None,
        ask=None,
        spread=None,
        source=getattr(row, "source", None),
    )
