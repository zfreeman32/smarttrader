"""Section 4 prompt 5 runtime fixes (opt-in, versioned): HTF bar-timestamp semantics pass-through
(s4-htf-semantics-v1) and the history refresh behind the cursor (s4-history-refresh-v1). Off = legacy behavior."""
from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.ingestion.signals import LiveSignalProcessor, SignalRuntimeModelBinding
from ote_live.models.loaders import LoadedRuntimeModel, load_live_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

MANIFEST = ROOT / "ote_live/runtime_manifests/short_reversal_xgb_v2_20260525/live_runtime_manifest.json"
HTF = "htf_1h_ema_spread_21_50_atr"
SELECTED = [HTF, "rsi_14", "dist_to_prior_high_20_atr"]
BAR = timedelta(minutes=5)


class _InputSumBooster:
    def predict(self, dmatrix) -> np.ndarray:
        return np.asarray([np.tanh(float(np.nansum(dmatrix.get_data().data)) / 50.0)], dtype=np.float32)


def _loaded() -> LoadedRuntimeModel:
    base = load_live_runtime_manifest(MANIFEST)
    manifest = base.model_copy(update={
        "feature_manifest": base.feature_manifest.model_copy(update={
            "selected_feature_names": list(SELECTED), "selected_feature_count": len(SELECTED),
            "strategy_feature_count": 0, "lag_steps": [0, 1], "delta_feature_count": 0}),
        "live_policy": base.live_policy.model_copy(update={
            "thresholds": base.live_policy.thresholds.model_copy(update={"global_threshold": 0.5, "regime_thresholds": {}}),
            "abstain_policy": base.live_policy.abstain_policy.model_copy(update={"enabled": False})}),
    })
    return LoadedRuntimeModel(manifest=manifest, model=_InputSumBooster(), scaler=None, calibrator=None,
                              training_summary={}, scale_clip=0.0, batch_size=1, use_amp=False)


LOADED = _loaded()


def _bars(count: int) -> list[MarketBar]:
    start = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    out = []
    for index in range(count):
        ts = start + BAR * index
        price = 1.1 + 0.0004 * np.sin(index / 7.0) + 0.00001 * index
        observed = ts + BAR + timedelta(seconds=90)
        out.append(MarketBar(asset="EURUSD", timeframe="5m", timestamp=ts, open=price, high=price + 0.0003,
                             low=price - 0.0003, close=price + 0.00005, volume=100.0 + index, symbol="EURUSD",
                             source="test", source_timestamp=ts, is_complete=True, feed_type="historical",
                             observation_kind="historical", first_observed_at=observed, last_observed_at=observed))
    return out


def _processor(store: SQLiteLiveDataStore, **kwargs) -> LiveSignalProcessor:
    return LiveSignalProcessor(bindings=[SignalRuntimeModelBinding(loaded_model=LOADED, shadow_mode=True)],
                               audit_repository=LiveAuditRepository(store), rolling_window_bars=400, force_shadow_mode=True,
                               enable_ict_paper_signal_ledger=False, carry_bar_provenance=True, group_name="TEST", **kwargs)


def test_htf_semantics_off_is_unchanged_and_on_changes_only_htf_columns() -> None:
    bars = _bars(500)
    frames = {}
    for semantics in (None, "bar_open"):
        engine = IncrementalFeatureEngine([LOADED.manifest], rolling_window_bars=500, htf_bar_timestamp_semantics=semantics)
        engine.extend(bars)
        frames[semantics] = engine.build_feature_frame()
        if semantics is None:
            default = IncrementalFeatureEngine([LOADED.manifest], rolling_window_bars=500)
            assert engine.config.to_dict() == default.config.to_dict()  # None = recipe default, bit-for-bit config
    off, on = frames[None], frames["bar_open"]
    differing = [c for c in off.columns if c in on.columns and not np.array_equal(
        pd.to_numeric(off[c], errors="coerce").to_numpy(float), pd.to_numeric(on[c], errors="coerce").to_numpy(float), equal_nan=True)]
    assert HTF in differing
    assert all(c.startswith(("htf_", "continuation_")) for c in differing), differing


def _run(tmp_path: Path, name: str, *, refresh: bool, backfill: bool):
    bars = _bars(330)
    store = SQLiteLiveDataStore(tmp_path / f"{name}.sqlite3")
    store.upsert_bars(bars[:300])
    processor = _processor(store, refresh_history_behind_cursor=refresh)
    processor.process_new_bars_from_store(emit_operator_artifacts=False)  # warm
    withheld = set(range(305, 308)) if backfill else set()
    results = []
    for index in range(300, 330):
        if index in withheld:
            continue
        if backfill and index == 310:
            store.upsert_bars([bars[i] for i in sorted(withheld)])  # behind the cursor
        store.upsert_bars([bars[index]])
        results.extend(processor.process_new_bars_from_store(emit_operator_artifacts=False))
    state = {bar.timestamp for bar in processor.feature_engine.state._bars}
    health = store.connection.execute("SELECT COUNT(*) FROM health_events WHERE event_type='history_refreshed_behind_cursor'").fetchone()[0] \
        if store.connection.execute("SELECT name FROM sqlite_master WHERE name='health_events'").fetchone() else None
    scores = [(r.timestamp, round(float(p.calibrated_probability), 12)) for r in results for p in getattr(r, "predictions", ())]
    frame = processor.feature_engine.build_feature_frame()
    store.close()
    return state, health, scores, frame, {bars[i].timestamp for i in withheld}


def test_history_refresh_brings_backfilled_bars_into_state(tmp_path: Path) -> None:
    state_on, health_on, _, frame_on, withheld = _run(tmp_path, "on", refresh=True, backfill=True)
    state_off, _, _, _, _ = _run(tmp_path, "off", refresh=False, backfill=True)
    assert withheld <= state_on                      # fix: backfilled bars are in the retained history
    assert not (withheld & state_off)                # legacy: they never enter it (the EURUSD stale-history bug)
    assert health_on is None or health_on >= 1
    clean_state, _, _, clean_frame, _ = _run(tmp_path, "clean", refresh=False, backfill=False)
    assert state_on == clean_state                   # after the refresh the history equals an uninterrupted feed
    pd.testing.assert_frame_equal(frame_on.reset_index(drop=True), clean_frame.reset_index(drop=True))


def test_history_refresh_is_value_neutral_without_behind_cursor_writes(tmp_path: Path) -> None:
    on = _run(tmp_path, "neutral_on", refresh=True, backfill=False)
    off = _run(tmp_path, "neutral_off", refresh=False, backfill=False)
    assert on[0] == off[0] and on[2] == off[2]
    pd.testing.assert_frame_equal(on[3], off[3])
