"""Section 4 R1: prove raw snapshotting is value-neutral on the actual app path with B4 v2 bytes.

For each protocol group (ES_FRVP, ES_ICT, EURUSD_OTE) this builds the group's LiveSignalProcessor from
staged manifests that load the hash-verified B4 v2 artifacts, runs the same bars twice on two isolated
SQLite stores (snapshotting off, then on), and compares features, scaled inputs, scores and threshold
decisions bitwise. The 'on' run's snapshot tables are flushed as R1 evidence. Never touches the live
DB, installed manifests or the registry; shadow is forced and no notifier is attached.

Usage: python scripts/run_section4_r1_value_neutrality.py <out_dir> [endpoints=3] [groups=ES_ICT,ES_FRVP,EURUSD_OTE]
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.market_data import MarketBar
from ote_live.features.raw_snapshots import RAW_SNAPSHOT_SCHEMA_VERSION, RawSnapshotRecorder, es_app_calendar, eurusd_calendar_v1
from ote_live.ingestion.signals import LiveSignalProcessor, SignalRuntimeModelBinding
from ote_live.models.loaders import LoadedRuntimeModel, load_live_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
from scripts.audit_b5_b6_live_feature_parity import candidate_manifest
from scripts.run_section3_smoke_replay import file_hash, roster, verify_bytes

PROTOCOL = ROOT / "configs/section4_parity_protocol_v1.json"
PROTOCOL_SHA256 = "55f9d226701c620ba3939de4de6b2bfb43289c52badc61ef25262c93cfbcdd2f"
EVIDENCE_VERSION = "s4-r1-value-neutrality-v1"
BASES = {2: "frvp_es_paper_signal_20260816/frvp_long_continuation_xgb_v1",
         3: "frvp_es_paper_signal_20260816/frvp_long_meta_xgb_v1",
         5: "frvp_es_setup_family_20260829/frvp_long_continuation_setup2_xgb_v1",
         10: "frvp_es_paper_signal_20260816/frvp_long_reversal_xgb_v1",
         12: "ict_es_paper_signal_20260813/ict_long_meta_xgb_v1", 13: "ict_es_paper_signal_20260813/ict_long_reversal_xgb_v1",
         14: "ict_es_paper_signal_20260813/ict_short_continuation_xgb_v1", 15: "ict_es_paper_signal_20260813/ict_short_meta_xgb_v1",
         16: "ict_es_paper_signal_20260813/ict_short_reversal_xgb_v1",
         18: "ict_short_setup_family_20260811_audit/ict_short_reversal_ifvg_reversal_xgb_v1",
         19: "ict_short_setup_family_20260811_audit/ict_short_reversal_sweep_reclaim_xgb_v1",
         21: "long_ote_meta_tcn_champion", 22: "long_reversal_tcn_v2_20260525_narrow48",
         23: "long_ote_union_tcn_candidate_20260523", 25: "short_ote_meta_tcn_champion", 26: "short_reversal_xgb_v2_20260525"}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def staged_model(model: dict) -> tuple[LoadedRuntimeModel, dict]:
    """B4 v2 bytes (hash-verified first) behind an installed base manifest; staged in memory only."""
    verified = verify_bytes(model)
    summary = json.loads((model["model_dir"] / "training_summary.json").read_text(encoding="utf-8"))
    if model["stream"] == "ES":
        from scripts.audit_b2_candidate_history import runtime
    else:
        from scripts.run_b2_ote_candidate_history import runtime_model as runtime
    frozen = runtime(model["model_dir"], summary).loaded_model
    base = load_live_runtime_manifest(ROOT / "ote_live/runtime_manifests" / BASES[model["queue_id"]] / "live_runtime_manifest.json")
    manifest = candidate_manifest(base, model["model_dir"], summary)
    artifacts = model["freeze"]["artifacts"]
    calibrator = model["model_dir"] / artifacts["calibrator_file"] if artifacts.get("calibrator_file") else None
    manifest = manifest.model_copy(update={
        "model_id": model["model_id"],
        # Staged, never installed: a staging registry path keeps a B4 v2 replacement from posing as the
        # controlled FRVP paper-signal manifest (whose guard fails closed on the installed path).
        "registry_path": f"artifacts/s4parity_20261002_v1/staged_manifests/{model['queue_id']:02d}_{model['model_id']}.json",
        "artifact_references": manifest.artifact_references.model_copy(update={
            "model_file": rel(model["model_dir"] / artifacts["model_file"]),
            "scaler_file": rel(model["model_dir"] / artifacts["scaler_file"]),
            "calibrator_file": rel(calibrator) if calibrator is not None and calibrator.exists() else None,
        }),
        "live_policy": manifest.live_policy.model_copy(update={
            "thresholds": manifest.live_policy.thresholds.model_copy(update={
                "global_threshold": float(model["threshold"]), "regime_thresholds": {}}),
        }),
    })
    loaded = LoadedRuntimeModel(manifest=manifest, model=frozen.model, scaler=frozen.scaler, calibrator=frozen.calibrator,
                                training_summary=summary, scale_clip=frozen.scale_clip, batch_size=frozen.batch_size,
                                use_amp=frozen.use_amp)
    return loaded, verified


def source_bars(stream: str, count: int) -> list[MarketBar]:
    if stream == "ES":
        raw = pd.read_csv(ROOT / "data/futures_data/ES-5m-tagged.csv").tail(count)
        times = pd.to_datetime(raw.ts_event, utc=True)
        ohlcv = raw[["open", "high", "low", "close", "volume"]]
        contracts = raw.contract_symbol.tolist()
    else:
        raw = pd.read_csv(ROOT / "data/currency_data/eurusd-5m.csv").tail(count)
        times = pd.to_datetime(raw.Date.astype(str) + " " + raw.Time, utc=True)
        ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]].set_axis(["open", "high", "low", "close", "volume"], axis=1)
        contracts = [None] * len(raw)
    return [MarketBar(asset=stream, timeframe="5m", timestamp=times.iloc[i].to_pydatetime().astimezone(timezone.utc),
                      open=float(ohlcv.open.iloc[i]), high=float(ohlcv.high.iloc[i]), low=float(ohlcv.low.iloc[i]),
                      close=float(ohlcv.close.iloc[i]), volume=float(ohlcv.volume.iloc[i]), symbol=stream,
                      contract_symbol=contracts[i], source="section4-r1-training-source")
            for i in range(len(raw))]


def frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for column in frame.columns:
        values = frame[column].to_numpy()
        digest.update(f"{column}|{values.dtype}|".encode())
        digest.update(values.tobytes() if values.dtype != object else repr(values.tolist()).encode())
    digest.update(repr(list(frame.index)).encode())
    return digest.hexdigest()


def run_once(group: str, stream: str, loaded: list[LoadedRuntimeModel], bars: list[MarketBar], endpoints: int,
             db_path: Path, recorder: RawSnapshotRecorder | None) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    store = SQLiteLiveDataStore(db_path)
    store.upsert_bars(bars[:-endpoints])
    processor = LiveSignalProcessor(
        bindings=[SignalRuntimeModelBinding(loaded_model=model, shadow_mode=True) for model in loaded],
        audit_repository=LiveAuditRepository(store), group_name=group, force_shadow_mode=True,
        enable_ict_paper_signal_ledger=False, data_supplier="SECTION4_R1")
    if recorder is not None:
        processor.attach_raw_snapshot_recorder(recorder)
    frames, scaled = [], {model.model_id: [] for model in loaded}
    for model_id, runner in processor._runner_by_model_id.items():
        original = runner._scale_matrix
        runner._scale_matrix = (lambda fn, sink: (lambda matrix: sink.append(fn(matrix)) or sink[-1]))(original, scaled[model_id])
    prepare = processor.prepare_ingested_bar_snapshot

    def recording_prepare(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        if prepared is not None:
            frames.append(frame_digest(prepared.feature_frame))
        return prepared

    processor.prepare_ingested_bar_snapshot = recording_prepare
    warmed = processor.warm_from_store()
    store.upsert_bars(bars[-endpoints:])
    started = time.perf_counter()
    processor.process_new_bars_from_store(emit_operator_artifacts=False)
    elapsed = time.perf_counter() - started
    predictions = pd.read_sql_query(
        "SELECT mp.model_id, mp.timestamp_utc, mp.source_row_idx, mp.raw_score, mp.calibrated_probability, "
        "mp.threshold_applied, sd.decision, sd.threshold FROM model_predictions mp "
        "LEFT JOIN signal_decisions sd ON sd.prediction_id = mp.id ORDER BY mp.model_id, mp.timestamp_utc", store.connection)
    snapshots = pd.read_sql_query("SELECT runtime_manifest_id, direction, timestamp_utc, source_row_idx, feature_values_json, "
                                  "valid_feature_count FROM feature_snapshots ORDER BY id", store.connection)
    store.close()
    return dict(warmed=warmed, seconds=round(elapsed, 2), frames=frames, predictions=predictions, snapshots=snapshots,
                scaled={key: [hashlib.sha256(a.dtype.str.encode() + a.tobytes()).hexdigest() for a in value]
                        for key, value in scaled.items()})


def main() -> None:
    out = Path(sys.argv[1])
    endpoints = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    groups = sys.argv[3].split(",") if len(sys.argv) > 3 else ["ES_ICT", "ES_FRVP", "EURUSD_OTE"]
    if file_hash(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Section 4 protocol v1 does not match its lock")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    models = {model["queue_id"]: model for model in roster()}
    report = dict(evidence_version=EVIDENCE_VERSION, raw_snapshot_schema=RAW_SNAPSHOT_SCHEMA_VERSION,
                  protocol_sha256=PROTOCOL_SHA256, endpoints_per_group=endpoints,
                  code_sha256={path: file_hash(ROOT / path) for path in (
                      "features/builder.py", "ote_live/features/incremental_engine.py", "ote_live/features/raw_snapshots.py",
                      "ote_live/ingestion/signals.py", "scripts/run_section4_r1_value_neutrality.py")},
                  bar_source="training-source tail (ES-5m-tagged.csv / eurusd-5m.csv); value-neutrality only, not parity",
                  groups={})
    for group in groups:
        spec = protocol["engine_composition"]["groups"][group]
        stream = spec["stream"]
        loaded, verified = zip(*(staged_model(models[q]) for q in spec["queue_ids"]))
        bars = source_bars(stream, int(spec["runtime_history_bars"]) + endpoints)
        print(group, "models", len(loaded), "bars", len(bars), flush=True)
        off = run_once(group, stream, list(loaded), bars, endpoints, out / group / "off.sqlite3", None)
        print(group, "off done", off["seconds"], "s", flush=True)
        calendar = es_app_calendar() if stream == "ES" else eurusd_calendar_v1()
        recorder = RawSnapshotRecorder(stream=stream, group=group, calendar=calendar)
        on = run_once(group, stream, list(loaded), bars, endpoints, out / group / "on.sqlite3", recorder)
        print(group, "on done", on["seconds"], "s", flush=True)
        hashes = recorder.flush(out / group / "snapshots", "chunk_0")
        rows = pd.read_parquet(out / group / "snapshots" / "chunk_0.rows.parquet")
        states = pd.read_parquet(out / group / "snapshots" / "chunk_0.feature_states.parquet")
        model_states = pd.read_parquet(out / group / "snapshots" / "chunk_0.model_states.parquet")
        values = rows.filter(like="f:")
        thresholds = {models[q]["model_id"]: float(models[q]["threshold"]) for q in spec["queue_ids"]}
        for result in (off, on):  # B4 v2 rule: calibrated_probability >= frozen threshold
            frame = result["predictions"]
            result["threshold_decision"] = (frame.calibrated_probability >= frame.model_id.map(thresholds)).tolist()
        identical = dict(
            feature_frames=off["frames"] == on["frames"],
            scaled_inputs=off["scaled"] == on["scaled"],
            scores_and_decisions=off["predictions"].equals(on["predictions"]),
            threshold_decisions=off["threshold_decision"] == on["threshold_decision"],
            persisted_feature_snapshots=off["snapshots"].equals(on["snapshots"]),
        )
        report["groups"][group] = dict(
            stream=stream, queue_ids=spec["queue_ids"], b4_v2_bytes_verified=[v["b4_freeze_files_verified"] for v in verified],
            eurusd_calendar_status="provisional (configs/eurusd_5m_expected_open_calendar_v1.json)" if stream == "EURUSD" else None,
            warmed_bars=off["warmed"], seconds_off=off["seconds"], seconds_on=on["seconds"],
            builds=len(off["frames"]), predictions=int(len(off["predictions"])),
            app_decisions=off["predictions"]["decision"].value_counts(dropna=False).to_dict(),
            threshold_decisions_true=int(sum(off["threshold_decision"])),
            scaled_matrices={key: len(value) for key, value in off["scaled"].items()},
            bit_identical=identical, value_neutral=all(identical.values()),
            snapshot=dict(tables_sha256=hashes, recorder_errors=recorder.errors, rows=int(len(rows)),
                          feature_columns=int(values.shape[1]), raw_nan_cells=int(values.isna().sum().sum()),
                          group_states=states.group_state.value_counts().to_dict(),
                          warmup_ready=model_states.groupby("model_id").warmup_ready.all().astype(bool).to_dict(),
                          gap_states=rows.gap_state.value_counts().to_dict(),
                          store_rows_found=bool(rows.store_row_found.all())),
        )
        print(group, "VALUE_NEUTRAL" if all(identical.values()) else "NOT_NEUTRAL", identical, flush=True)
        (out / "value_neutrality.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    report["value_neutral_all_groups"] = all(g["value_neutral"] for g in report["groups"].values())
    (out / "value_neutrality.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print("R1 VALUE NEUTRALITY COMPLETE", report["value_neutral_all_groups"], flush=True)


if __name__ == "__main__":
    main()
