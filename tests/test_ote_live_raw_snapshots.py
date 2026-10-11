"""Section 4 R1: raw pre-imputation snapshots are copy-only and carry explicit state."""
from __future__ import annotations

import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.builder import FeatureDatasetBuilder
from ote_live.contracts.market_data import MarketBar
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.features.raw_snapshots import (
    RAW_SNAPSHOT_SCHEMA_VERSION,
    VALUE_PREFIX,
    RawSnapshotRecorder,
    capture_pre_fill,
    eurusd_calendar_v1,
)
from ote_live.ingestion.signals import LiveSignalProcessor, SignalRuntimeModelBinding, _bar_from_row
from ote_live.models.loaders import LoadedRuntimeModel, load_live_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

SHORT_XGB_LIVE_MANIFEST_PATH = (
    ROOT / "ote_live" / "runtime_manifests" / "short_reversal_xgb_v2_20260525" / "live_runtime_manifest.json"
)
SELECTED = ["rsi_14", "dist_to_prior_high_20_atr", "atr_ratio_14_50_roll_mean_100"]
HELPER = "htf_confluence_long_score"
TMP_ROOT = ROOT / "tmp" / "ote_live_raw_snapshot_tests"


class _InputSumBooster:
    """Score depends on every scaled lagged input, so any input change moves it."""

    def predict(self, dmatrix) -> np.ndarray:
        data = dmatrix.get_data()
        return np.asarray([np.tanh(float(np.nansum(data.data)) / 50.0)], dtype=np.float32)


def _bar(index: int, *, complete: bool | None = None) -> MarketBar:
    timestamp = datetime(2024, 1, 2, 0, 0, tzinfo=UTC) + timedelta(minutes=5 * index)
    price = 1.1000 + 0.0004 * np.sin(index / 7.0) + 0.00001 * index
    return MarketBar(
        asset="EURUSD", timeframe="5m", timestamp=timestamp, open=price, high=price + 0.0003,
        low=price - 0.0003, close=price + 0.00005, volume=100.0 + index, source="raw-snapshot-test",
        is_complete=complete,
    )


def _processor(store: SQLiteLiveDataStore, *, carry_bar_provenance: bool = False,
               selected: list[str] = SELECTED) -> LiveSignalProcessor:
    base = load_live_runtime_manifest(SHORT_XGB_LIVE_MANIFEST_PATH)
    manifest = base.model_copy(update={
        "feature_manifest": base.feature_manifest.model_copy(update={
            "selected_feature_names": list(selected), "selected_feature_count": len(selected),
            "strategy_feature_count": 0, "lag_steps": [0, 1, 3], "delta_feature_count": 0,
        }),
        "live_policy": base.live_policy.model_copy(update={
            "thresholds": base.live_policy.thresholds.model_copy(update={"global_threshold": 0.5, "regime_thresholds": {}}),
            "abstain_policy": base.live_policy.abstain_policy.model_copy(update={"enabled": False}),
        }),
    })
    loaded = LoadedRuntimeModel(manifest=manifest, model=_InputSumBooster(), scaler=None, calibrator=None,
                                training_summary={}, scale_clip=0.0, batch_size=1, use_amp=False)
    return LiveSignalProcessor(
        bindings=[SignalRuntimeModelBinding(loaded_model=loaded, shadow_mode=True)],
        audit_repository=LiveAuditRepository(store), rolling_window_bars=260, force_shadow_mode=True,
        enable_ict_paper_signal_ledger=False, carry_bar_provenance=carry_bar_provenance,
    )


def _run(*, snapshot: bool, bars: list[MarketBar], tail: int = 6, selected: list[str] = SELECTED):
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(TMP_ROOT / f"{uuid.uuid4().hex}.sqlite")
    store.upsert_bars(bars[:-tail])
    processor = _processor(store, selected=selected)
    recorder = None
    if snapshot:
        recorder = RawSnapshotRecorder(stream="EURUSD", group="TEST", calendar=eurusd_calendar_v1())
        processor.attach_raw_snapshot_recorder(recorder)
    frames, scaled = [], []
    runner = processor._runner_by_model_id[processor.bindings[0].loaded_model.model_id]
    original_scale = runner._scale_matrix
    runner._scale_matrix = lambda matrix: scaled.append(original_scale(matrix)) or scaled[-1]
    original_prepare = processor.prepare_ingested_bar_snapshot

    def prepare(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        frames.append(prepared.feature_frame.copy())
        return prepared

    processor.prepare_ingested_bar_snapshot = prepare
    processor.warm_from_store()
    store.upsert_bars(bars[-tail:])
    processor.process_new_bars_from_store(emit_operator_artifacts=False)
    predictions = pd.read_sql_query(
        "SELECT mp.timestamp_utc, mp.raw_score, mp.calibrated_probability, mp.threshold_applied, sd.decision "
        "FROM model_predictions mp LEFT JOIN signal_decisions sd ON sd.prediction_id = mp.id ORDER BY mp.id",
        store.connection,
    )
    snapshots = pd.read_sql_query(
        "SELECT timestamp_utc, feature_values_json FROM feature_snapshots ORDER BY id", store.connection)
    store.close()
    return dict(frames=frames, scaled=scaled, predictions=predictions, snapshots=snapshots, recorder=recorder)


def test_capture_pre_fill_copies_tail_with_nan_and_never_mutates() -> None:
    working = pd.DataFrame({
        "datetime": pd.date_range("2024-01-02", periods=5, freq="5min", tz="UTC"),
        "a": [1.0, np.nan, 3.0, np.inf, np.nan],
        "flag": [True, False, True, False, True],
        "contract_symbol": ["X", "X", "Y", "Y", "Y"],
    })
    before = working.copy()
    capture = capture_pre_fill(working, {"a": "feature_set:demo"}, ["a", "flag", "absent"], 3)
    pd.testing.assert_frame_equal(working, before)
    assert capture.window_positions == (2, 3, 4)
    assert np.isnan(capture.values[2, 0]) and capture.values[0, 0] == 3.0
    assert capture.values[:, 1].tolist() == [1.0, 0.0, 1.0]
    assert capture.produced == (True, True, False) and np.isnan(capture.values[:, 2]).all()
    assert capture.producers["absent"] == "missing" and capture.previous_contract_symbol == "X"


def test_builder_observer_sees_pre_fill_values_and_is_value_neutral() -> None:
    engine = IncrementalFeatureEngine([_processor_manifest()], rolling_window_bars=260)
    frame = IncrementalFeatureEngine.market_frame_from_bars([_bar(i) for i in range(120)])
    seen = {}
    plain, _ = FeatureDatasetBuilder(engine.config).build(frame)
    observed, _ = FeatureDatasetBuilder(engine.config).build(
        frame, pre_fill_observer=lambda working, producers: seen.update(rsi=working["rsi_14"].copy(), producers=producers))
    pd.testing.assert_frame_equal(plain, observed, check_exact=True)
    assert engine.config.fillna_numeric is True  # EURUSD fills inside the builder
    assert seen["rsi"].isna().any() and not plain["rsi_14"].isna().any()
    assert seen["producers"]["rsi_14"].startswith("feature_set:")


def test_processor_snapshot_on_off_is_bit_identical_and_records_explicit_state() -> None:
    bars = [_bar(i, complete=True if i % 2 else None) for i in range(300)]
    off, on = _run(snapshot=False, bars=bars), _run(snapshot=True, bars=bars)
    assert len(off["frames"]) == len(on["frames"]) == 6
    for left, right in zip(off["frames"], on["frames"]):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert len(off["scaled"]) == len(on["scaled"]) > 0
    for left, right in zip(off["scaled"], on["scaled"]):
        assert left.dtype == right.dtype and left.tobytes() == right.tobytes()
    pd.testing.assert_frame_equal(off["predictions"], on["predictions"], check_exact=True)
    assert off["snapshots"].equals(on["snapshots"])
    assert off["predictions"]["raw_score"].nunique() > 1

    recorder = on["recorder"]
    assert recorder.errors == [] and recorder.pending_builds == 6
    out = TMP_ROOT / uuid.uuid4().hex
    hashes = recorder.flush(out, "chunk_0")
    assert recorder.pending_builds == 0 and len(hashes) == 6
    index = json.loads((out / "chunk_0.index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == RAW_SNAPSHOT_SCHEMA_VERSION and index["feature_names"] == SELECTED
    assert index["models"][0]["feature_positions"] == [0, 1, 2] and index["models"][0]["consumed_rows"] == 4
    assert index["calendar_id"].startswith("eurusd_5m_feed_clock_week_v1:provisional")
    builds = pd.read_parquet(out / "chunk_0.builds.parquet")
    rows = pd.read_parquet(out / "chunk_0.rows.parquet")
    states = pd.read_parquet(out / "chunk_0.feature_states.parquet")
    model_states = pd.read_parquet(out / "chunk_0.model_states.parquet")
    categories = pd.read_parquet(out / "chunk_0.model_feature_states.parquet")
    assert set(builds.fill_mode) == {"ffill_zero_in_builder"}
    assert (builds.endpoint_timestamp.map(pd.Timestamp) == [b.timestamp for b in bars[-6:]]).all()
    assert set(states.group_state) == {"produced"} and states.producer_module_sha256.notna().all()
    assert set(categories.contract_category) == {"available"}
    # Raw values keep warmup NaN that the builder later forward-fills / zero-fills.
    assert rows.groupby("build_id").size().eq(index["models"][0]["capture_rows"]).all()
    # Store provenance and source_row_idx come from canonical rows, not from the engine bar.
    endpoint_rows = rows.loc[rows.row_offset == 0].reset_index(drop=True)
    assert (endpoint_rows.store_source_row_idx.to_numpy() == builds.endpoint_source_row_idx.to_numpy()).all()
    assert endpoint_rows.store_is_complete.tolist() == [bool(i % 2) or None for i in range(294, 300)]
    assert endpoint_rows.engine_is_complete.isna().all()  # D8 off: provenance not carried to the engine
    assert set(rows.gap_state) <= {"none", "window_start"} and model_states.consumed_rows_captured.all()
    assert model_states.warmup_ready.all()


def test_missing_helper_state_is_explicit_and_never_a_zero_value() -> None:
    bars = [_bar(i) for i in range(300)]
    run = _run(snapshot=True, bars=bars, tail=2, selected=[*SELECTED, HELPER])
    assert run["predictions"].empty  # the helper keeps the model out of warmup
    out = TMP_ROOT / uuid.uuid4().hex
    run["recorder"].flush(out, "chunk_0")
    rows = pd.read_parquet(out / "chunk_0.rows.parquet")
    states = pd.read_parquet(out / "chunk_0.feature_states.parquet")
    categories = pd.read_parquet(out / "chunk_0.model_feature_states.parquet")
    models = pd.read_parquet(out / "chunk_0.model_states.parquet")
    helper = states.loc[states.feature == HELPER]
    assert len(helper) == 2 and set(helper.group_state) == {"fallback_zero_default"} and not helper.produced.any()
    assert helper.producer.eq("missing").all() and helper.producer_module_sha256.isna().all()
    assert rows[VALUE_PREFIX + HELPER].isna().all()
    assert set(categories.loc[categories.feature == HELPER, "contract_category"]) == {"missing_producer"}
    assert not models.warmup_ready.any() and models.contract_status.eq("diagnostic_only").all()
    assert set(states.loc[states.feature != HELPER, "group_state"]) == {"produced"}


def test_raw_values_preserve_nan_the_builder_fills() -> None:
    bars = [_bar(i) for i in range(300)]
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(TMP_ROOT / f"{uuid.uuid4().hex}.sqlite")
    store.upsert_bars(bars)
    processor = _processor(store, selected=["atr_ratio_14_50_roll_mean_100"])
    processor.feature_engine.enable_raw_capture(300)
    assert processor.warm_from_store() == 260
    frame = processor.feature_engine.build_feature_frame()
    capture = processor.feature_engine.last_raw_capture
    store.close()
    raw = capture.values[:, 0]
    assert len(raw) == 260 and np.isnan(raw[:50]).all() and np.isfinite(raw[-10:]).all()
    assert not frame["atr_ratio_14_50_roll_mean_100"].isna().any()  # builder ffill+0
    assert (frame["atr_ratio_14_50_roll_mean_100"].to_numpy()[:50] == 0.0).all()


def test_d8_provenance_flag_defaults_off_and_carries_canonical_provenance_when_on() -> None:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(TMP_ROOT / f"{uuid.uuid4().hex}.sqlite")
    store.upsert_bars([_bar(i, complete=True) for i in range(3)])
    legacy, carried = _processor(store), _processor(store, carry_bar_provenance=True)
    off = legacy._fetch_signal_bars_after(datetime(2024, 1, 1, tzinfo=UTC))
    on = carried._fetch_signal_bars_after(datetime(2024, 1, 1, tzinfo=UTC))
    row = store.connection.execute("SELECT * FROM canonical_bars ORDER BY timestamp_utc LIMIT 1").fetchone()
    store.close()
    assert off[0] == _bar_from_row(row)
    assert off[0].is_complete is None and off[0].bar_version is None and off[0].observation_kind == "unknown"
    assert on[0].is_complete is True and on[0].bar_version == row["bar_version"] and on[0].first_observed_at is not None
    assert [bar.model_dump(include={"timestamp", "open", "high", "low", "close", "volume"}) for bar in off] == [
        bar.model_dump(include={"timestamp", "open", "high", "low", "close", "volume"}) for bar in on]


def _processor_manifest():
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(TMP_ROOT / f"{uuid.uuid4().hex}.sqlite")
    manifest = _processor(store).bindings[0].loaded_model.manifest
    store.close()
    return manifest
