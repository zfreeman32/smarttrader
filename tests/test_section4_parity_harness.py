"""Section 4 parity harness validation: known synthetic defects must be detected; faults run on the real app path."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.features.raw_snapshots import consumed_rows, eurusd_calendar_v1
from ote_live.ingestion.signals import LiveSignalProcessor, SignalRuntimeModelBinding
from ote_live.models.loaders import LoadedRuntimeModel, load_live_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
from scripts import section4_parity_harness as harness
from scripts.run_section3_smoke_replay import file_hash
from scripts.section4_parity_compare import (
    compare_app_runs, compare_segment, compare_setup_events, compare_source, load_app,
)

MANIFEST = ROOT / "ote_live/runtime_manifests/short_reversal_xgb_v2_20260525/live_runtime_manifest.json"
SELECTED = ["rsi_14", "dist_to_prior_high_20_atr", "atr_ratio_14_50_roll_mean_100"]
PREFIX, ENDPOINTS, CHUNK = 300, 9, 3
APP_ARTIFACTS = {"model.json": "a" * 64}
SET_SHA = "s" * 64


class _InputSumBooster:
    def predict(self, dmatrix) -> np.ndarray:
        data = dmatrix.get_data()
        return np.asarray([np.tanh(float(np.nansum(data.data)) / 50.0)], dtype=np.float32)


def _loaded() -> LoadedRuntimeModel:
    base = load_live_runtime_manifest(MANIFEST)
    manifest = base.model_copy(update={
        "feature_manifest": base.feature_manifest.model_copy(update={
            "selected_feature_names": list(SELECTED), "selected_feature_count": len(SELECTED),
            "strategy_feature_count": 0, "lag_steps": [0, 1, 3], "delta_feature_count": 0}),
        "live_policy": base.live_policy.model_copy(update={
            "thresholds": base.live_policy.thresholds.model_copy(update={"global_threshold": 0.5, "regime_thresholds": {}}),
            "abstain_policy": base.live_policy.abstain_policy.model_copy(update={"enabled": False})}),
    })
    return LoadedRuntimeModel(manifest=manifest, model=_InputSumBooster(), scaler=None, calibrator=None,
                              training_summary={}, scale_clip=0.0, batch_size=1, use_amp=False)


LOADED = _loaded()
MODEL_ID = LOADED.model_id


def _factory(store: SQLiteLiveDataStore) -> LiveSignalProcessor:
    return LiveSignalProcessor(
        bindings=[SignalRuntimeModelBinding(loaded_model=LOADED, shadow_mode=True)],
        audit_repository=LiveAuditRepository(store), rolling_window_bars=260, force_shadow_mode=True,
        enable_ict_paper_signal_ledger=False, carry_bar_provenance=True, group_name="TEST")


def _bars(count: int) -> list[MarketBar]:
    start = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    bars = []
    for index in range(count):
        timestamp = start + timedelta(minutes=5 * index)
        price = 1.1 + 0.0004 * np.sin(index / 7.0) + 0.00001 * index
        observed = timestamp + harness.BAR + harness.REPLAY_GRACE
        bars.append(MarketBar(asset="EURUSD", timeframe="5m", timestamp=timestamp, open=price, high=price + 0.0003,
                              low=price - 0.0003, close=price + 0.00005, volume=100.0 + index, symbol="EURUSD",
                              source="section4_training_source:synthetic", source_timestamp=timestamp, is_complete=True,
                              feed_type="historical", observation_kind="historical", first_observed_at=observed,
                              last_observed_at=observed))
    return bars


def _run(root: Path, name: str, *, fault=None, chunk=CHUNK, restart_each_chunk=False, prefix=PREFIX,
         failure=None, allow_resume=False) -> Path:
    bars = _bars(prefix + ENDPOINTS)
    segment = dict(segment_id=name, stream="EURUSD", source="H", rule="test",
                   endpoints=[harness.iso(b.timestamp) for b in bars[prefix:]], fault=fault)
    models = [dict(model_id=MODEL_ID, threshold=0.5, consumed_rows=consumed_rows(LOADED.manifest),
                   artifact_sha256=dict(APP_ARTIFACTS), reviewed_artifact_set_sha256=SET_SHA)]
    harness.run_app_segment(segment=segment, group="TEST", models=models, processor_factory=_factory,
                            seg_dir=root / name, chunk_endpoints=chunk, prefix_bars=bars[:prefix],
                            endpoint_bars=bars[prefix:], restart_each_chunk=restart_each_chunk,
                            allow_resume=allow_resume, calendar=eurusd_calendar_v1(), failure=failure,
                            log=lambda message: None)
    return root / name


def _self_reference(clean: Path, target: Path, offset: int = 1000) -> None:
    """Reference tables equal to the clean continuous app run (row ids shifted by a constant offset)."""
    app = load_app(clean / "app")
    endpoint = app["builds"].assign(e=app["builds"].endpoint_timestamp.map(harness.iso)).set_index("build_id").e
    rows = app["rows"].assign(endpoint=app["rows"].build_id.map(endpoint))
    io = app["io"][MODEL_ID].drop(columns=["part", "row_timestamp", "row_source_row_idx"])
    io = io.merge(rows[["endpoint", "row_offset", "timestamp", "store_source_row_idx"]], on=["endpoint", "row_offset"])
    io = io.rename(columns={"timestamp": "row_timestamp"}).assign(row_source_row_idx=io.store_source_row_idx + offset)
    ref = target / "ref"
    ref.mkdir(parents=True, exist_ok=True)
    io.drop(columns=["store_source_row_idx"]).to_parquet(ref / f"{MODEL_ID}.io.parquet", index=False)
    scores = app["scores"][["model_id", "endpoint", "source_row_idx", "raw_score", "calibrated_probability",
                            "threshold", "decision"]]
    scores.to_parquet(ref / f"{MODEL_ID}.scores.parquet", index=False)
    (ref / f"{MODEL_ID}.run.json").write_text(json.dumps(dict(
        mode="self_test", verified=dict(b4_freeze_files_verified=dict(APP_ARTIFACTS), reviewed_artifact_set_sha256=SET_SHA),
        source_history_sha256=None)), encoding="utf-8")


def _spec() -> dict:
    return dict(model_id=MODEL_ID, queue_id=None, stream="EURUSD", threshold=0.5,
                feature_classes={name: "C1_continuous" for name in SELECTED}, producers={},
                reference_fill_stage="pre_fill", source_row_offset=0)


def _expected(model_sha: str = "a" * 64) -> dict:
    return {MODEL_ID: {"artifacts": {"sha256": {"model": model_sha, "calibrator": None}, "reviewed_artifact_set_sha256": SET_SHA}}}


def _compare(seg: Path, expected=None) -> dict:
    summary = compare_segment(seg, [_spec()], harness.load_protocol(), expected_artifacts=expected or _expected())
    return summary["models"][MODEL_ID]


def _reseal(app_dir: Path) -> None:
    """A defective app writes consistent files: recompute the done-marker hashes after the edit."""
    for path in app_dir.glob("chunk_*.done"):
        marker = json.loads(path.read_text(encoding="utf-8"))
        marker["files"] = {name: file_hash(app_dir / name) for name in marker["files"]}
        path.write_text(json.dumps(marker), encoding="utf-8")


def _edit(seg: Path, table: str, edit) -> None:
    path = sorted((seg / "app").glob(f"chunk_000_p0.{table}.parquet"))[0]
    frame = pd.read_parquet(path)
    edit(frame).to_parquet(path, index=False)
    _reseal(seg / "app")


@pytest.fixture(scope="module")
def clean(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("s4h")
    seg = _run(root, "clean")
    _self_reference(seg, seg)
    return seg


def _defect_copy(clean: Path, tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(clean, target)
    return target


# ---------------------------------------------------------------- comparator: clean and known defects

def test_clean_run_compares_with_no_mismatch(clean: Path) -> None:
    result = _compare(clean)
    assert result["headline"]["status"] == "no_mismatch", result["headline"]
    assert result["headline"]["comparable"] == ENDPOINTS
    for stage in ("S1", "S3", "S4", "S4_mask", "S5", "S6", "S7"):
        assert result["stages"][stage]["compared"] > 0 and result["stages"][stage]["mismatched"] == 0
    assert result["stages"]["S3"]["compared"] == ENDPOINTS * consumed_rows(LOADED.manifest) * len(SELECTED)


def test_detects_injected_value_change(clean: Path, tmp_path: Path) -> None:
    seg = _defect_copy(clean, tmp_path, "value")
    endpoint = load_app(seg / "app")["builds"].build_id.iloc[1]
    _edit(seg, "rows", lambda f: f.assign(**{"f:rsi_14": np.where((f.build_id == endpoint) & (f.row_offset == 0),
                                                                  f["f:rsi_14"] + 0.5, f["f:rsi_14"])}))
    result = _compare(seg)
    group = next(g for g in result["mismatch_groups"] if g["stage"] == "S3" and g["feature"] == "rsi_14")
    assert group["count"] == 1 and abs(group["max_abs"] - 0.5) < 1e-9  # raw float64 on both sides
    assert group["first_endpoint"] == result_endpoint(seg, endpoint) and group["classification"] == "unclassified"
    assert (seg / "diff" / f"{MODEL_ID}.S3.parquet").exists()


def result_endpoint(seg: Path, build_id: str) -> str:
    builds = load_app(seg / "app")["builds"].set_index("build_id")
    return harness.iso(builds.loc[build_id, "endpoint_timestamp"])


def test_detects_identity_change(clean: Path, tmp_path: Path) -> None:
    seg = _defect_copy(clean, tmp_path, "identity")
    target = load_app(seg / "app")["builds"].build_id.iloc[2]
    _edit(seg, "rows", lambda f: f.assign(store_source_row_idx=np.where(
        (f.build_id == target) & (f.row_offset == -1), f.store_source_row_idx + 7, f.store_source_row_idx)))
    result = _compare(seg)
    features = {g["feature"] for g in result["mismatch_groups"] if g["stage"] == "S1"}
    assert {"source_row_idx_offset", "source_row_idx_consecutive"} <= features


def test_detects_missing_snapshot(clean: Path, tmp_path: Path) -> None:
    seg = _defect_copy(clean, tmp_path, "missing")
    target = load_app(seg / "app")["builds"].build_id.iloc[0]
    _edit(seg, "rows", lambda f: f.loc[f.build_id != target])
    result = _compare(seg)
    assert result["checks"]["missing_snapshot"] and "missing_snapshot" in result["headline"]["failed_checks"]
    assert result["headline"]["comparable"] == ENDPOINTS - 1


def test_detects_stale_helper(clean: Path, tmp_path: Path) -> None:
    seg = _defect_copy(clean, tmp_path, "stale")
    _edit(seg, "rows", lambda f: f.assign(engine_htf_30m_source_age_minutes=np.where(
        f.row_offset == 0, 95.0, 10.0), engine_htf_30m_source_complete=1))
    result = _compare(seg)
    assert result["checks"]["stale_helper"] > 0 and "stale_helper" in result["headline"]["failed_checks"]


def test_detects_unexplained_fallback(clean: Path, tmp_path: Path) -> None:
    seg = _defect_copy(clean, tmp_path, "fallback")
    target = load_app(seg / "app")["builds"].build_id.iloc[2]
    _edit(seg, "feature_states", lambda f: f.assign(group_state=np.where(
        (f.build_id == target) & (f.feature == "rsi_14"), "fallback_zero_default", f.group_state)))
    result = _compare(seg)
    assert result["checks"]["fallback"] == 1 and result["stages"]["S3_state"]["mismatched"] == 1
    assert "fallback" in result["headline"]["failed_checks"]


def test_detects_wrong_artifact_hash(clean: Path) -> None:
    result = _compare(clean, expected=_expected(model_sha="b" * 64))
    assert not result["checks"]["artifact_hash"]["ok"]
    assert "app_artifact_sha256_not_b4_v2" in result["checks"]["artifact_hash"]["reasons"]
    assert "artifact_hash" in result["headline"]["failed_checks"]


def test_detects_score_and_decision_flip(clean: Path, tmp_path: Path) -> None:
    seg = _defect_copy(clean, tmp_path, "flip")
    endpoint = load_app(seg / "app")["scores"].endpoint.iloc[1]
    _edit(seg, "scores", lambda f: f.assign(calibrated_probability=np.where(
        f.endpoint == endpoint, np.where(f.calibrated_probability >= 0.5, 0.1, 0.9), f.calibrated_probability)))
    result = _compare(seg)
    assert result["stages"]["S6"]["mismatched"] == 1 and len(result["decision_flips"]) == 1


def test_tampered_chunk_without_marker_update_is_refused(clean: Path, tmp_path: Path) -> None:
    seg = _defect_copy(clean, tmp_path, "tamper")
    path = seg / "app" / "chunk_000_p0.rows.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[:1].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="changed after their done markers"):
        _compare(seg)


# ---------------------------------------------------------------- restart, chunking, resume

def test_restart_and_chunked_runs_equal_continuous(clean: Path, tmp_path: Path) -> None:
    continuous = _run(tmp_path, "one_chunk", chunk=ENDPOINTS)
    chunked = _run(tmp_path, "restart_each_chunk", restart_each_chunk=True)
    mid = _run(tmp_path, "mid_session", fault=dict(kind="restart", at=4, point="mid_session"))
    boundary = _run(tmp_path, "chunk_boundary", fault=dict(kind="restart", at=CHUNK, point="chunk_boundary"))
    for other in (clean, chunked, mid, boundary):
        result = compare_app_runs(continuous / "app", other / "app")
        assert result["equal"], (other.name, {k: v["differing_columns"] for k, v in result["tables"].items() if not v["equal"]})
    assert len(load_app(mid / "app")["restarted_parts"]) == 1
    assert len(load_app(chunked / "app")["restarted_parts"]) == ENDPOINTS // CHUNK - 1


def test_resume_after_crash_restores_checkpoint_and_requires_restart_test(clean: Path, tmp_path: Path, monkeypatch) -> None:
    original, calls = harness.AppDriver.apply, {"n": 0}

    def crash(self, step):
        calls["n"] += 1
        if calls["n"] == CHUNK + 2:
            raise KeyboardInterrupt("simulated crash in chunk 1")
        return original(self, step)

    monkeypatch.setattr(harness.AppDriver, "apply", crash)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, "resume")
    monkeypatch.setattr(harness.AppDriver, "apply", original)
    with pytest.raises(RuntimeError, match="restart test"):
        _run(tmp_path, "resume")
    seg = _run(tmp_path, "resume", allow_resume=True)
    assert json.loads((seg / "app" / "run.json").read_text())["resumes"][0]["chunk"] == 1
    assert compare_app_runs(clean / "app", seg / "app")["equal"]


# ---------------------------------------------------------------- fault injection on the app path

def test_dropped_bars_are_flagged_as_gap_and_injected(clean: Path, tmp_path: Path) -> None:
    seg = _run(tmp_path, "dropped", fault=dict(kind="dropped", at=3, count=2))
    shutil.copytree(clean / "ref", seg / "ref")
    app = load_app(seg / "app")
    assert (app["rows"].gap_state == "gap").any()
    assert int((~app["truth"].delivered).sum()) == 2
    result = _compare(seg)
    assert result["endpoint_status"]["not_delivered"] == 2
    s1 = [g for g in result["mismatch_groups"] if g["stage"] == "S1"]
    assert s1 and all(g["evidence_at_first"]["injected_before_endpoint"] for g in s1)


def test_backfill_behind_cursor_is_stored_and_detected(clean: Path, tmp_path: Path) -> None:
    seg = _run(tmp_path, "backfill", fault=dict(kind="backfill", at=2, count=2, release_after=1))
    shutil.copytree(clean / "ref", seg / "ref")
    store = SQLiteLiveDataStore(seg / "parity.sqlite3")
    kinds = pd.read_sql_query("SELECT observation_kind, COUNT(*) n FROM canonical_bars GROUP BY 1", store.connection)
    store.close()
    assert int(kinds.set_index("observation_kind").n.get("backfill", 0)) == 2
    result = _compare(seg)
    assert result["headline"]["status"] == "mismatches_unclassified"
    assert all(g["evidence_at_first"]["injected_before_endpoint"] for g in result["mismatch_groups"])


def test_revision_provisional_then_final_is_logged_and_detected(clean: Path, tmp_path: Path) -> None:
    seg = _run(tmp_path, "revision", fault=dict(kind="revision", at=6, earlier_lag=3))
    shutil.copytree(clean / "ref", seg / "ref")
    store = SQLiteLiveDataStore(seg / "parity.sqlite3")
    revisions = store.connection.execute("SELECT COUNT(*) FROM source_bar_history WHERE event_type = 'revision'").fetchone()[0]
    store.close()
    assert revisions >= 2
    run = json.loads((seg / "app" / "run.json").read_text())
    assert {i["revised"] for i in run["injections"] if i["action"] == "provisional_processed"} == {"latest", "earlier"}
    result = _compare(seg)
    assert result["mismatch_groups"] and all(g["evidence_at_first"]["injected_before_endpoint"] for g in result["mismatch_groups"])


def test_reconnect_batch_scores_every_endpoint_like_continuous(clean: Path, tmp_path: Path) -> None:
    seg = _run(tmp_path, "reconnect", fault=dict(kind="reconnect", at=2, count=4))
    result = compare_app_runs(clean / "app", seg / "app")
    assert result["tables"]["scores"]["equal"] and result["tables"][f"io:{MODEL_ID}"]["equal"]
    assert len(load_app(seg / "app")["scores"]) == ENDPOINTS
    assert set(result["tables"]["rows"]["differing_columns"]) <= {
        "store_observation_kind", "engine_observation_kind", "store_bar_version", "engine_bar_version",
        "store_first_observed_at", "store_last_observed_at", "engine_first_observed_at", "engine_last_observed_at"}


def test_cold_start_is_incomplete_history_not_a_pass(tmp_path: Path) -> None:
    seg = _run(tmp_path, "cold", prefix=20)
    app = load_app(seg / "app")
    assert not app["model_states"].warmup_ready.astype(bool).any()
    assert app["scores"].empty


def test_producer_failure_surfaces_and_is_never_silent(clean: Path, tmp_path: Path) -> None:
    producer = load_app(clean / "app")["feature_states"].query("feature == 'rsi_14'").producer.iloc[0]
    feature_set = producer.split(":", 1)[1]
    seg = _run(tmp_path, "producer_failure", failure=dict(feature_set=feature_set, mode="raise", **{"from": 2, "to": 4}))
    shutil.copytree(clean / "ref", seg / "ref")
    result = _compare(seg)
    # The app records feature_build_failed and skips the bar: surfaced as health errors + unevaluated endpoints.
    assert result["checks"]["health_errors"] == 2 and len(result["checks"]["not_evaluated"]) == 2
    assert {"health_errors", "not_evaluated"} <= set(result["headline"]["failed_checks"])
    assert result["checks"]["health_events"]["signal_runtime.features:feature_build_failed:error"] == 2


# ---------------------------------------------------------------- S2, S layer, windows, protocol

def test_setup_event_comparator_matches_and_reports_differences() -> None:
    app = pd.DataFrame(dict(event_key=["a", "a", "b"], revision=[0, 1, 0], event_kind=["observed", "revised", "observed"],
                            strategy=["ICT"] * 3, source_timestamp_utc=["2026-06-01T10:00:00+00:00"] * 2 + ["2026-06-01T11:00:00+00:00"],
                            observed_at_utc=[None] * 3, setup_type=["sweep_reclaim"] * 2 + ["ifvg_reversal"],
                            setup_side=[-1, -1, 1], setup_family=["reversal"] * 3, rule_confidence=[0.8, 0.9, 0.7],
                            detector_identity=["d"] * 3, detector_version=["v"] * 3,
                            geometry_json=['{"stop_reference": 10.0}', '{"stop_reference": 11.0}', '{"stop_reference": 5.0}']))
    ref = pd.DataFrame(dict(strategy=["ICT", "ICT"], side=[-1, 1], source_ts=["2026-06-01 10:00:00+00:00", "2026-06-01 12:00:00+00:00"],
                            setup_type=["sweep_reclaim", "ifvg_reversal"], **{"geo:stop_reference": [10.0, 5.0]}))
    result = compare_setup_events(app, ref, harness.load_protocol())
    assert result["matched"] == 1 and result["only_app"] == 1 and result["only_ref"] == 1
    assert result["mismatches"]["geo:stop_reference"] == 0 and result["revised_app_events"] == 1


def test_source_comparator_reports_identity_completion_and_price() -> None:
    times = pd.date_range("2026-06-01", periods=6, freq="5min", tz="UTC")
    training = pd.DataFrame(dict(timestamp=times, open=5000.0, high=5001.0, low=4999.0, close=5000.5, volume=10.0,
                                 contract_symbol="ESM6"))
    app = training.drop(index=2).assign(is_complete=pd.Series(True, index=training.index, dtype=object))
    app.loc[3, "close"] = 5000.75
    app.loc[4, "is_complete"] = None
    app.loc[5, "contract_symbol"] = "ESU6"
    result = compare_source(app, training, harness.load_protocol(), stream="ES")
    assert result["only_training_count"] == 1 and result["contract_mismatches"] == 1
    assert result["completion_unknown"] == 1 and result["prices"]["close"]["out_of_tolerance"] == 1
    assert result["status"] == "incomplete"


def test_windows_file_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "windows.json"
    segment = dict(segment_id="x", endpoints=["2026-01-01T00:00:00+00:00"])
    digest = harness.write_windows(path, [segment])
    assert harness.write_windows(path, [segment]) == digest
    with pytest.raises(ValueError, match="different windows"):
        harness.write_windows(path, [dict(segment, segment_id="y")])


def test_protocol_v2_lock_is_verified() -> None:
    assert harness.load_protocol()["version"] == 2


# ---------------------------------------------------------------- prompt 4 matrix extensions

def test_multi_fault_program_equals_single_faults_and_rejects_overlap() -> None:
    bars = _bars(40)
    singles = [dict(kind="dropped", at=1, count=1), dict(kind="backfill", at=4, count=2, release_after=1),
               dict(kind="revision", at=12, earlier_lag=3), dict(kind="reconnect", at=16, count=3)]
    _, multi_log, multi_final = harness.feed_program(bars, dict(kind="multi", faults=singles), stream="EURUSD",
                                                     chunk_endpoints=CHUNK)
    single_log = [entry for fault in singles for entry in
                  harness.feed_program(bars, fault, stream="EURUSD", chunk_endpoints=CHUNK)[1]]
    key = lambda entry: (entry["timestamp"], entry["kind"], entry["action"])
    assert sorted(map(key, multi_log)) == sorted(map(key, single_log))
    assert len(multi_final) == len(bars) - 1
    with pytest.raises(ValueError, match="overlap"):
        harness.feed_program(bars, dict(kind="multi", faults=[dict(kind="dropped", at=3, count=3),
                                                              dict(kind="backfill", at=5, count=1)]),
                             stream="EURUSD", chunk_endpoints=CHUNK)


def test_multi_point_restart_equals_continuous_and_subset_compare(clean: Path, tmp_path: Path) -> None:
    continuous = _run(tmp_path, "one_chunk", chunk=ENDPOINTS)
    points = _run(tmp_path, "points", restart_each_chunk=True,
                  fault=dict(kind="restart", points=[dict(at=3, point="a"), dict(at=5, point="b")]))
    result = compare_app_runs(continuous / "app", points / "app")
    assert result["equal"], {k: v["differing_columns"] for k, v in result["tables"].items() if not v["equal"]}
    assert sum(1 for e in json.loads((points / "app" / "run.json").read_text())["injections"] if e["kind"] == "restart") == 2
    subset = compare_app_runs(continuous / "app", points / "app", models=[MODEL_ID], common_endpoints=True,
                              row_columns=["timestamp", "store_row_found", "window_position"])
    assert subset["equal"] and set(subset["tables"]) == {"rows", f"io:{MODEL_ID}", "scores"}
