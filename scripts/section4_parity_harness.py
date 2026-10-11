"""Section 4 R2/R4 parity harness (protocol v2): the actual app path vs the frozen offline reference.

Extends ``scripts/run_section4_r1_value_neutrality.py`` (staged B4 v2 manifests over hash-verified bytes,
isolated SQLite stores, raw pre-imputation snapshots) into a bar-by-bar, chunked and checkpointed driver.

- App: one ``LiveSignalProcessor`` per protocol group on an isolated parity DB (``carry_bar_provenance=True``,
  shadow forced, no notifier, no ledgers). Raw snapshots stay on (S1/S3 and states). Read-only taps on each
  ``RuntimeModelRunner`` instance record the model-input rows before the frozen imputer (S4 input), the
  imputed rows and mask (S4), the scaled and clipped rows (S5), raw/calibrated scores (S6) and the B4 v2
  threshold decision ``calibrated >= frozen threshold`` (S7). Taps copy values only.
- Reference: the frozen production runner (``scripts.audit_b2_candidate_history.runtime`` for ES,
  ``scripts.run_b2_ote_candidate_history.runtime_model`` for EURUSD) with the same taps, on frozen
  ``source_history.parquet`` (historical windows) or on the frozen dataset config recomputed over the full
  contiguous series (recorded live windows, after the anchor check).
- Faults: deterministic feed programs on the same bars (dropped bars, backfill behind the cursor,
  provisional-then-final revisions, reconnect batch, restart through a fresh processor at a chunk boundary or
  mid-session, cold start, producer failure / strategy timeout).

Layout: ``<out>/<stream>/<group>/<segment>/{app,ref,diff}/`` + ``summary.json``. Each chunk writes
``chunk_<n>_p<k>.*`` tables and ``chunk_<n>.done`` (file sha256); resume skips finished chunks and restarts the
processor from the verified checkpoint DB. Never opens the live DB writable (``mode=ro``), never touches
installed manifests or the registry, never sends anything.

Usage:
  python scripts/section4_parity_harness.py plumbing <queue_id> [endpoints=3] [chunk_endpoints=2]
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_training.source_history import contract_sha256, impute_matrix
from ote_live.contracts.market_data import MarketBar
from ote_live.features.raw_snapshots import (
    RAW_SNAPSHOT_SCHEMA_VERSION, RawSnapshotRecorder, consumed_rows, es_app_calendar, eurusd_calendar_v1,
)
from ote_live.ingestion.provenance import observe_bar
from ote_live.ingestion.signals import LiveSignalProcessor, SignalRuntimeModelBinding, _bar_from_row
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
from scripts.run_section3_smoke_replay import file_hash, roster, verify_bytes

HARNESS_VERSION = "s4-parity-harness-v1"
RUN_DATE = os.environ.get("S4_RUN_DATE", "20261002")  # prompt 5 re-runs use a new run id (S4_RUN_DATE=20261005)
# Prompt 5 app fixes (opt-in runtime flags): "p5" = EURUSD htf_bar_timestamp_semantics from the frozen dataset config
# (s4-htf-semantics-v1) + refresh_history_behind_cursor (s4-history-refresh-v1). Empty = prompt 4 configuration.
APP_FIXES = os.environ.get("S4_APP_FIXES", "")
RUN_ROOT = ROOT / f"artifacts/s4parity_{RUN_DATE}_v1"
PROTOCOL = ROOT / "configs/section4_parity_protocol_v2.json"
PROTOCOL_SHA256 = "e25b5e48905d44de5d6c4369a5ac24b49bf711074722710b149494fccb34eff5"
PROTOCOL_LOCK = ROOT / "configs/section4_parity_protocol_v2.lock.json"
PROTOCOL_LOCK_SHA256 = "f9b65ced680b65e46ca8f84167c4abd5de15a9372d3d88f98369d2304acbff38"
FREEZE = ROOT / "configs/b4_broad_full_session_shadow_freeze_v2.json"
LIVE_DB = ROOT / "ote_live/runtime_data/live_market_data.sqlite3"
# Pinned training sources; row position == frozen source_row_idx (checked per model against source_history).
TRAINING_SOURCES = {"ES": "data/futures_data/ES-5m-tagged.csv", "EURUSD": "artifacts/b2_ote_labels_20260920_v2/labels.csv.gz"}
# Frozen dataset configs for the recompute path, resolved by the contract's source_dataset_sha256.
DATASET_DIRS = ("artifacts/b2_frvp_july_prepared_20260921_v1", "artifacts/b2_frvp_setups_prepared_20260921_v1",
                "artifacts/b2_ict_prepared_20260921_v1", "artifacts/b2_ict_specialists_prepared_20260921_v1",
                "artifacts/b2_ote_prepared_20260921_v1")
BAR = timedelta(minutes=5)
# Replay clock for copied bars: observed once at bar close + the app's 90 s finalized-bar grace. Isolated
# parity DBs only; never live-qualification evidence.
REPLAY_GRACE = timedelta(seconds=90)
PRICE_STEP = {"ES": 0.25, "EURUSD": 0.00005}
CODE_FILES = (
    "scripts/section4_parity_harness.py", "scripts/section4_parity_compare.py",
    "scripts/run_section4_r1_value_neutrality.py", "scripts/run_section3_smoke_replay.py",
    "features/builder.py", "ote_live/features/incremental_engine.py", "ote_live/features/raw_snapshots.py",
    "ote_live/ingestion/signals.py", "ote_live/ingestion/provenance.py", "ote_live/models/runners.py",
    "ote_live/models/loaders.py", "ote_live/storage/db.py", "model_training/source_history.py",
)
LIVE_COLUMNS = ("asset, timeframe, timestamp_utc, open, high, low, close, volume, bid, ask, spread, source, symbol, "
                "contract_symbol, instrument_id, feature_context_json, source_timestamp_utc, bar_version, is_complete, "
                "feed_type, first_observed_at_utc, last_observed_at_utc, observation_kind")


# ---------------------------------------------------------------- protocol, hashes, small io

def load_protocol() -> dict:
    """Protocol v2, verified against its lock before anything else runs."""
    if file_hash(PROTOCOL) != PROTOCOL_SHA256 or file_hash(PROTOCOL_LOCK) != PROTOCOL_LOCK_SHA256:
        raise ValueError("Section 4 protocol v2 does not match its lock")
    if json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8")).get("sha256") != PROTOCOL_SHA256:
        raise ValueError("Section 4 protocol v2 lock names a different protocol")
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def code_hashes() -> dict[str, str]:
    return {path: file_hash(ROOT / path) for path in CODE_FILES if (ROOT / path).exists()}


def dump(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str, allow_nan=True) + "\n", encoding="utf-8")
    return file_hash(path)


def iso(value) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat() if pd.Timestamp(value).tzinfo else pd.Timestamp(value).tz_localize("UTC").isoformat()


def manifest_sha256(manifest) -> str:
    return hashlib.sha256(manifest.model_dump_json().encode()).hexdigest()


def _text(value) -> str | None:
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


# ---------------------------------------------------------------- bars

@lru_cache(maxsize=2)
def training_frame(stream: str) -> pd.DataFrame:
    path = ROOT / TRAINING_SOURCES[stream]
    if stream == "ES":
        raw = pd.read_csv(path, usecols=["ts_event", "open", "high", "low", "close", "volume", "contract_symbol"])
        raw["timestamp"] = pd.to_datetime(raw.pop("ts_event"), utc=True)
    else:
        raw = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume"])
        raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
        raw["contract_symbol"] = None
    if not raw.timestamp.is_monotonic_increasing or raw.timestamp.duplicated().any():
        raise ValueError(f"{stream} training source is not a strictly increasing bar series")
    return raw.reset_index(drop=True)


def historical_bar(stream: str, row, *, observed_at: datetime | None = None, kind: str = "historical",
                   is_complete: bool | None = True, close: float | None = None) -> MarketBar:
    """Training-source bar as copied into an isolated parity DB (protocol v2 historical_source_completion)."""
    timestamp = pd.Timestamp(row.timestamp).to_pydatetime()
    observed = observed_at or timestamp + BAR + REPLAY_GRACE
    close_value = float(row.close) if close is None else float(close)
    return MarketBar(
        asset=stream, timeframe="5m", timestamp=timestamp, open=float(row.open),
        high=max(float(row.high), close_value), low=min(float(row.low), close_value), close=close_value,
        volume=float(row.volume), symbol=stream, contract_symbol=_text(row.contract_symbol),
        source=f"section4_training_source:{TRAINING_SOURCES[stream]}", source_timestamp=timestamp,
        is_complete=is_complete, feed_type="historical", observation_kind=kind,
        first_observed_at=observed, last_observed_at=observed,
    )


def live_bars(stream: str, *, end: str | None = None, limit: int | None = None) -> list[MarketBar]:
    """Recorded bars from the live DB, opened read-only; provenance carried as recorded."""
    connection = sqlite3.connect(f"file:{LIVE_DB.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        query = f"SELECT {LIVE_COLUMNS} FROM canonical_bars WHERE asset = ? AND timeframe = '5m'"
        params: list[Any] = [stream]
        if end is not None:
            query += " AND timestamp_utc <= ?"
            params.append(iso(end))
        query += " ORDER BY timestamp_utc DESC" + (f" LIMIT {int(limit)}" if limit else "")
        rows = connection.execute(query, params).fetchall()
    finally:
        connection.close()
    return [_bar_from_row(row, carry_provenance=True) for row in reversed(rows)]


# ---------------------------------------------------------------- windows

def historical_segment(stream: str, segment_id: str, event: str, *, before: int, after: int, prefix: int,
                       rule: str, scenarios: Sequence[str] = (), fault: dict | None = None,
                       plumbing: bool = False) -> dict:
    """Bars [event-before, event+after] of the training source plus ``prefix`` earlier bars (selection by rule)."""
    times = training_frame(stream).timestamp
    position = int(times.searchsorted(pd.Timestamp(event)))
    if position >= len(times) or times.iloc[position] != pd.Timestamp(event):
        raise ValueError(f"{event} is not a {stream} training-source bar")
    first, last = position - before, position + after
    if first - prefix < 0 or last >= len(times):
        raise ValueError(f"Segment {segment_id} exceeds the {stream} training source")
    return dict(segment_id=segment_id, stream=stream, source="H", rule=rule, event=iso(event),
                prefix_start_pos=first - prefix, first_endpoint_pos=first, last_endpoint_pos=last, prefix_bars=prefix,
                endpoints=[iso(t) for t in times.iloc[first:last + 1]], scenarios=list(scenarios), fault=fault,
                plumbing=bool(plumbing), training_source=TRAINING_SOURCES[stream])


def segment_bars(segment: dict) -> tuple[list[MarketBar], list[MarketBar]]:
    """(prefix bars, endpoint bars) for a segment; H from the training source, R from the live DB (read-only)."""
    stream = segment["stream"]
    if segment["source"] == "H":
        frame = training_frame(stream)
        prefix = frame.iloc[segment["prefix_start_pos"]:segment["first_endpoint_pos"]]
        endpoints = frame.iloc[segment["first_endpoint_pos"]:segment["last_endpoint_pos"] + 1]
        return ([historical_bar(stream, row) for row in prefix.itertuples(index=False)],
                [historical_bar(stream, row) for row in endpoints.itertuples(index=False)])
    if segment["source"] == "R":
        bars = live_bars(stream, end=segment["endpoints"][-1], limit=segment["prefix_bars"] + len(segment["endpoints"]))
        # Recorded rows without an observation time (legacy rows; ESU6 and most EURUSD) would otherwise be stamped
        # with the wall clock at copy time, so two runs of one window would differ in input provenance. They get the
        # replay clock instead (bar close + grace, as for H bars); everything else stays as recorded.
        bars = [bar if bar.first_observed_at is not None else bar.model_copy(update={
            "first_observed_at": bar.timestamp + BAR + REPLAY_GRACE,
            "last_observed_at": bar.last_observed_at or bar.timestamp + BAR + REPLAY_GRACE}) for bar in bars]
        split = [iso(bar.timestamp) for bar in bars].index(segment["endpoints"][0])
        if [iso(bar.timestamp) for bar in bars[split:]] != segment["endpoints"]:
            raise ValueError(f"Recorded bars changed under segment {segment['segment_id']}")
        return bars[:split], bars[split:]
    raise ValueError(f"Unsupported segment source {segment['source']!r}")


def write_windows(path: Path, segments: Sequence[dict], *, protocol_sha256: str = PROTOCOL_SHA256) -> str:
    """windows.json is written by rule before any comparison and never rewritten with different content."""
    payload = dict(schema="s4-parity-windows-v1", harness_version=HARNESS_VERSION, protocol_sha256=protocol_sha256,
                   selection="calendar/event presence only, never outcomes or scores", segments=list(segments))
    text = json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"{path} already exists with different windows; publish a new version instead")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    digest = file_hash(path)
    path.with_suffix(".json.sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


# ---------------------------------------------------------------- fault programs

def feed_program(bars: Sequence[MarketBar], fault: dict | None, *, stream: str, chunk_endpoints: int,
                 restart_each_chunk: bool = False) -> tuple[list[tuple[int, dict]], list[dict], dict[str, MarketBar]]:
    """Deterministic delivery steps ``(endpoint_index, step)`` plus an injection log and the final bar per timestamp.

    Steps: ``insert`` (upsert bars, then optionally one ``process_new_bars_from_store``) and ``restart`` (fresh
    processor on the same DB: ``warm_from_store`` then later ``process_new_bars_from_store``).
    """
    fault = dict(fault or {})
    # A ``multi`` fault applies several single faults at non-overlapping endpoint ranges in one segment;
    # a ``restart`` fault may name several ``points`` ({at, point}). Single faults behave exactly as before.
    faults = [dict(f) for f in fault.get("faults", [])] if fault.get("kind") == "multi" else ([fault] if fault else [])
    restarts = sorted(((int(p["at"]), p.get("point", "mid_session")) for f in faults if f.get("kind") == "restart"
                       for p in (f.get("points") or [dict(at=f.get("at", -1), point=f.get("point"))])))
    restarts = [(at_, point or "mid_session") for at_, point in restarts]
    spans = []
    for f in faults:
        f.setdefault("count", 1)
        at, count = int(f.get("at", -1)), int(f["count"])
        if f.get("kind") == "revision":
            spans.append((at - int(f.get("earlier_lag", 3)), at + int(f.get("earlier_lag", 3)), f))
        elif f.get("kind") == "backfill":
            spans.append((at, at + count + int(f.get("release_after", 0)), f))
        elif f.get("kind") in {"dropped", "reconnect"}:
            spans.append((at, at + count - 1, f))
        elif f.get("kind") != "restart":
            raise ValueError(f"Unsupported fault {f}")
    spans.sort(key=lambda item: item[0])
    if any(a[1] >= b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError(f"Fault ranges overlap: {[(a, b, f['kind']) for a, b, f in spans]}")
    steps: list[tuple[int, dict]] = []
    log: list[dict] = []
    final: dict[str, MarketBar] = {}
    withheld: dict[int, list[MarketBar]] = {id(f): [] for _, _, f in spans}
    delayed: dict[int, list[MarketBar]] = {}

    def deliver(index: int, delivered: Sequence[MarketBar], process: bool, note: str | None = None) -> None:
        steps.append((index, {"op": "insert", "bars": list(delivered), "process": bool(process), "note": note}))
        for bar in delivered:
            final[iso(bar.timestamp)] = bar

    def relabel(bar: MarketBar, at_bar: MarketBar, kind_: str, offset: timedelta = timedelta(0)) -> MarketBar:
        observed = at_bar.timestamp + BAR + REPLAY_GRACE + offset
        return bar.model_copy(update={"observation_kind": kind_, "first_observed_at": observed,
                                      "last_observed_at": observed, "bar_version": None})

    def active(index: int, kind_: str) -> dict | None:
        return next((f for _, _, f in spans if f["kind"] == kind_
                     and int(f["at"]) <= index < int(f["at"]) + int(f["count"])), None)

    for index, bar in enumerate(bars):
        stamp = iso(bar.timestamp)
        if restart_each_chunk and index and index % chunk_endpoints == 0:
            steps.append((index, {"op": "restart", "note": "chunk_boundary"}))
        for at_, point in restarts:
            if index == at_:
                steps.append((index, {"op": "restart", "note": point}))
                log.append(dict(kind="restart", timestamp=stamp, action="fresh_processor_before_bar", point=point))
        for late in delayed.pop(index, []):
            deliver(index, [late], process=False, note="revision_behind_cursor")
            log.append(dict(kind="revision", timestamp=iso(late.timestamp), action="final_upserted_behind_cursor",
                            delivered_before=stamp))
        if active(index, "dropped"):
            log.append(dict(kind="dropped", timestamp=stamp, action="never_delivered"))
            continue
        f = active(index, "backfill")
        if f:
            withheld[id(f)].append(bar)
            log.append(dict(kind="backfill", timestamp=stamp, action="withheld"))
            continue
        for _, _, f in spans:
            if (f["kind"] == "backfill" and withheld[id(f)]
                    and index == int(f["at"]) + int(f["count"]) + int(f.get("release_after", 0))):
                released = [relabel(item, bar, "backfill", timedelta(seconds=-30)) for item in withheld[id(f)]]
                deliver(index, released, process=False, note="backfill_behind_cursor")
                log.extend(dict(kind="backfill", timestamp=iso(item.timestamp), action="upserted_behind_cursor",
                                delivered_before=stamp) for item in released)
                withheld[id(f)] = []
        f = active(index, "reconnect")
        if f:
            withheld[id(f)].append(bar)
            log.append(dict(kind="reconnect", timestamp=stamp, action="missed_while_disconnected"))
            if index == int(f["at"]) + int(f["count"]) - 1:
                batch = [relabel(item, bar, "backfill", timedelta(seconds=30)) for item in withheld[id(f)]]
                deliver(index, batch, process=True, note="reconnect_backfill_batch")
                withheld[id(f)] = []
            continue
        f = next((f for _, _, f in spans if f["kind"] == "revision"
                  and index in (int(f["at"]), int(f["at"]) - int(f.get("earlier_lag", 3)))), None)
        if f:
            latest = index == int(f["at"])
            provisional = bar.model_copy(update={
                "close": bar.close + PRICE_STEP[stream], "high": max(bar.high, bar.close + PRICE_STEP[stream]),
                "is_complete": None, "observation_kind": "live", "bar_version": None,
                "first_observed_at": bar.timestamp + BAR + timedelta(seconds=5),
                "last_observed_at": bar.timestamp + BAR + timedelta(seconds=5)})
            deliver(index, [provisional], process=True, note="provisional")
            log.append(dict(kind="revision", timestamp=stamp, action="provisional_processed",
                            revised="latest" if latest else "earlier"))
            final_bar = bar.model_copy(update={"first_observed_at": provisional.first_observed_at, "bar_version": None})
            if latest:  # the final version lands before the next bar is processed
                deliver(index, [final_bar], process=False, note="revision_latest")
                log.append(dict(kind="revision", timestamp=stamp, action="final_upserted_latest"))
            else:
                delayed.setdefault(min(index + int(f.get("earlier_lag", 3)), len(bars) - 1), []).append(final_bar)
            continue
        deliver(index, [bar], process=True)
    if any(withheld.values()):
        raise ValueError(f"Fault {fault} withholds bars past the end of the segment")
    return steps, log, final


@contextlib.contextmanager
def producer_failure(feature_set: str, mode: str = "raise", *, active: Callable[[], bool] = lambda: True):
    """F producer_failure: one feature set raises (``raise``) or runs with a near-zero strategy timeout (``timeout``)."""
    from features.registry import FEATURE_REGISTRY

    original = FEATURE_REGISTRY.build

    def build(name, df, config):
        if name == feature_set and active():
            if mode == "raise":
                raise RuntimeError(f"section4 injected producer failure: {feature_set}")
            if mode == "timeout":
                return original(name, df, dataclasses.replace(config, strategy_timeout_seconds=1e-9))
        return original(name, df, config)

    FEATURE_REGISTRY.build = build
    try:
        yield
    finally:
        FEATURE_REGISTRY.build = original


# ---------------------------------------------------------------- taps

class RunnerTap:
    """Read-only taps on one ``RuntimeModelRunner`` instance (S4 input/imputed/mask, S5 rows, S6, S7)."""

    def __init__(self, runner, *, model_id: str, consumed: int, threshold: float) -> None:
        self.model_id, self.consumed, self.threshold = model_id, int(consumed), float(threshold)
        self.feature_names = list(runner.loaded_model.selected_feature_names)
        self.contract = getattr(runner.loaded_model.scaler, "history_contract", None)
        self.records: list[dict] = []
        self.errors: list[dict] = []
        self._pending: dict | None = None
        original_predict, original_scale = runner.predict_latest, runner._scale_matrix

        def scale(matrix):
            scaled = original_scale(matrix)
            if self._pending is not None:
                self._pending["input"] = np.array(matrix[-self.consumed:], dtype=np.float32, copy=True)
                self._pending["scaled"] = np.array(scaled[-self.consumed:], dtype=np.float32, copy=True)
            return scaled

        def predict(frame, *args, **kwargs):
            self._pending = {"frame_tail": frame.tail(self.consumed)}
            try:
                prediction = original_predict(frame, *args, **kwargs)
            except Exception as exc:
                self.errors.append(dict(model_id=self.model_id, timestamp=str(kwargs.get("timestamp")),
                                        error_type=type(exc).__name__, error=str(exc)))
                self._pending = None
                raise
            pending, self._pending = self._pending, None
            self._record(prediction, pending)
            return prediction

        runner._scale_matrix, runner.predict_latest = scale, predict

    def _record(self, prediction, pending: dict) -> None:
        values = pending.get("input")
        imputed = impute_matrix(values, self.contract) if self.contract is not None and values is not None else values
        tail = pending["frame_tail"]
        times = (pd.to_datetime(tail["datetime"], utc=True).map(iso).tolist() if "datetime" in tail.columns
                 else [None] * len(tail))
        ids = (pd.to_numeric(tail["source_row_idx"], errors="coerce").tolist() if "source_row_idx" in tail.columns
               else [None] * len(tail))
        # Values as handed to the runner, before its float32 cast (reference: raw source history; app: post-fill).
        present = [name for name in self.feature_names if name in tail.columns]
        raw = (tail.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)[-self.consumed:]
               if len(present) == len(self.feature_names) else None)
        calibrated = float(prediction.calibrated_probability)
        self.records.append(dict(
            endpoint=iso(prediction.timestamp), source_row_idx=prediction.source_row_idx,
            raw_score=float(prediction.raw_score), calibrated_probability=calibrated, threshold=self.threshold,
            decision=bool(calibrated >= self.threshold), raw=raw, input=values, imputed=imputed, scaled=pending.get("scaled"),
            row_timestamps=times[-self.consumed:], row_source_row_idx=ids[-self.consumed:]))

    def frames(self, **extra) -> tuple[pd.DataFrame, list[dict]]:
        """(wide io frame: one row per endpoint x consumed row, score records)."""
        rows, scores = [], []
        for record in self.records:
            scores.append({key: record[key] for key in ("endpoint", "source_row_idx", "raw_score",
                                                        "calibrated_probability", "threshold", "decision")}
                          | {"model_id": self.model_id, **extra})
            count = 0 if record["input"] is None else len(record["input"])
            for position in range(count):
                row = {"endpoint": record["endpoint"], "row_offset": position - count + 1,
                       "row_timestamp": record["row_timestamps"][position] if position < len(record["row_timestamps"]) else None,
                       "row_source_row_idx": record["row_source_row_idx"][position] if position < len(record["row_source_row_idx"]) else None}
                raw = record["raw"]
                if raw is not None and position < len(raw):
                    row.update((f"raw:{name}", float(raw[position, column])) for column, name in enumerate(self.feature_names))
                for stage, key in (("in", "input"), ("imp", "imputed"), ("sc", "scaled")):
                    if record[key] is not None:
                        row.update((f"{stage}:{name}", float(record[key][position, column]))
                                   for column, name in enumerate(self.feature_names))
                rows.append(row)
        return pd.DataFrame.from_records(rows), scores


# ---------------------------------------------------------------- models

def freeze_entries() -> dict[int, dict]:
    return {entry["queue_id"]: entry for entry in json.loads(FREEZE.read_text(encoding="utf-8"))["frozen_replacements"]}


def roster_by_id() -> dict[int, dict]:
    return {model["queue_id"]: model for model in roster()}


def staged_group(group: str, protocol: dict) -> tuple[list, list[dict]]:
    """Staged B4 v2 runtime models for a protocol group (bytes hash-verified before load)."""
    from scripts.run_section4_r1_value_neutrality import staged_model

    models = roster_by_id()
    loaded, records = [], []
    for queue_id in protocol["engine_composition"]["groups"][group]["queue_ids"]:
        model = models[queue_id]
        runtime_model, verified = staged_model(model)
        loaded.append(runtime_model)
        records.append(dict(queue_id=queue_id, model_id=model["model_id"], threshold=float(model["threshold"]),
                            reviewed_artifact_set_sha256=verified["reviewed_artifact_set_sha256"],
                            artifact_sha256=verified["b4_freeze_files_verified"],
                            prepared_manifest_sha256=verified["prepared_manifest_sha256"],
                            staged_manifest_sha256=manifest_sha256(runtime_model.manifest),
                            consumed_rows=consumed_rows(runtime_model.manifest)))
    return loaded, records


def app_fix_kwargs(group: str) -> dict:
    """Prompt 5 opt-in runtime flags for a group (empty unless S4_APP_FIXES contains 'p5' and the group is EURUSD)."""
    if "p5" not in APP_FIXES or not group.startswith("EURUSD"):
        return {}
    protocol = load_protocol()
    models = roster_by_id()
    values = set()
    for queue_id in protocol["engine_composition"]["groups"][group]["queue_ids"]:
        reference = FrozenReference(models[queue_id])
        values.add(dataset_metadata_for(reference.contract)[1]["config"].get("htf_bar_timestamp_semantics"))
    if len(values) != 1 or None in values:
        raise ValueError(f"{group} frozen dataset configs disagree on htf_bar_timestamp_semantics: {values}")
    return dict(refresh_history_behind_cursor=True, htf_bar_timestamp_semantics=values.pop())


def group_processor_factory(group: str, loaded: Sequence) -> Callable[[SQLiteLiveDataStore], LiveSignalProcessor]:
    fixes = app_fix_kwargs(group)

    def factory(store: SQLiteLiveDataStore) -> LiveSignalProcessor:
        return LiveSignalProcessor(
            bindings=[SignalRuntimeModelBinding(loaded_model=model, shadow_mode=True) for model in loaded],
            audit_repository=LiveAuditRepository(store), group_name=group, force_shadow_mode=True,
            enable_ict_paper_signal_ledger=False, data_supplier="SECTION4_PARITY", carry_bar_provenance=True, **fixes)
    factory.app_fixes = fixes
    return factory


def stream_calendar(stream: str):
    return es_app_calendar() if stream == "ES" else eurusd_calendar_v1()  # EURUSD calendar v1 is provisional


# ---------------------------------------------------------------- app driver

class AppDriver:
    """Drives the actual app classes on an isolated parity DB; one shared producer pass per group."""

    def __init__(self, *, group: str, stream: str, db_path: Path, processor_factory, thresholds: dict[str, float],
                 calendar=None) -> None:
        self.group, self.stream, self.db_path = group, stream, Path(db_path)
        self.processor_factory, self.thresholds = processor_factory, dict(thresholds)
        self.calendar = calendar if calendar is not None else stream_calendar(stream)
        self.instance, self.part = 0, 0
        self.store = self.processor = self.recorder = None
        self.taps: dict[str, RunnerTap] = {}
        self.process_errors: list[dict] = []
        self.warmed: list[int] = []
        self.health_cursor = 0

    def start(self) -> None:
        """Real start/restart path: fresh store connection and processor, then ``warm_from_store``."""
        self.stop()
        self.store = SQLiteLiveDataStore(self.db_path)
        self.processor = self.processor_factory(self.store)
        self.recorder = RawSnapshotRecorder(stream=self.stream, group=self.group, calendar=self.calendar)
        self.processor.attach_raw_snapshot_recorder(self.recorder)
        self.taps = {}
        for binding in self.processor.bindings:
            model_id = binding.loaded_model.model_id
            self.taps[model_id] = RunnerTap(self.processor._runner_by_model_id[model_id], model_id=model_id,
                                            consumed=consumed_rows(binding.loaded_model.manifest),
                                            threshold=self.thresholds[model_id])
        self.instance += 1
        row = self.store.connection.execute("SELECT COALESCE(MAX(id), 0) FROM health_events").fetchone()
        self.health_cursor = int(row[0])
        self.warmed.append(int(self.processor.warm_from_store()))

    def stop(self) -> None:
        if self.store is not None:
            self.store.close()
        self.store = self.processor = None

    def apply(self, step: dict) -> None:
        if step["op"] == "restart":
            return
        self.store.upsert_bars(step["bars"])
        if step["process"]:
            try:
                self.processor.process_new_bars_from_store(emit_operator_artifacts=False)
            except Exception as exc:  # recorded, never swallowed silently; the next bar continues the session
                self.process_errors.append(dict(instance=self.instance, last_bar=iso(step["bars"][-1].timestamp),
                                                error_type=type(exc).__name__, error=str(exc),
                                                trace=traceback.format_exc(limit=3)))

    def flush(self, directory: Path, prefix: str) -> dict[str, str]:
        """Write the recorder tables, per-model io, scores (with persisted cross-check) and run notes."""
        hashes = self.recorder.flush(directory, prefix)
        scores = []
        for model_id, tap in self.taps.items():
            io, model_scores = tap.frames(instance=self.instance, part=prefix)
            path = directory / f"{prefix}.io.{model_id}.parquet"
            io.to_parquet(path, index=False)
            hashes[path.name] = file_hash(path)
            scores.extend(model_scores)
            tap.records = []
        frame = pd.DataFrame.from_records(scores)
        if len(frame):
            persisted = pd.read_sql_query(
                "SELECT id, model_id, timestamp_utc, raw_score AS persisted_raw_score, "
                "calibrated_probability AS persisted_calibrated_probability FROM model_predictions", self.store.connection)
            persisted["endpoint"] = persisted.timestamp_utc.map(iso)
            persisted = persisted.sort_values("id").drop_duplicates(["model_id", "endpoint"], keep="last")
            frame = frame.merge(persisted[["model_id", "endpoint", "persisted_raw_score", "persisted_calibrated_probability"]],
                                on=["model_id", "endpoint"], how="left")
        path = directory / f"{prefix}.scores.parquet"
        frame.to_parquet(path, index=False)
        hashes[path.name] = file_hash(path)
        health = pd.read_sql_query("SELECT id, component, event_type, severity, payload_json FROM health_events "
                                   "WHERE id > ? ORDER BY id", self.store.connection, params=(self.health_cursor,))
        if len(health):
            self.health_cursor = int(health.id.max())
        notes = dict(instance=self.instance, warmed=self.warmed[-1] if self.warmed else None,
                     process_errors=self.process_errors,
                     health_events=[dict(component=r.component, event_type=r.event_type, severity=r.severity,
                                         timestamp=json.loads(r.payload_json or "{}").get("timestamp_utc"))
                                    for r in health.itertuples()],
                     tap_errors=[error for tap in self.taps.values() for error in tap.errors])
        hashes[f"{prefix}.notes.json"] = dump(directory / f"{prefix}.notes.json", notes)
        self.process_errors = []
        for tap in self.taps.values():
            tap.errors = []
        return hashes


def build_parity_db(path: Path, bars: Sequence[MarketBar]) -> str:
    """Isolated parity DB (schema + migrations via SQLiteLiveDataStore) holding only the copied prefix bars."""
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteLiveDataStore(path)
    try:
        store.upsert_bars(list(bars))
    finally:
        store.close()
    return file_hash(path)


def _done_ok(app_dir: Path, chunk: int) -> dict | None:
    marker = app_dir / f"chunk_{chunk:03d}.done"
    if not marker.exists():
        return None
    done = json.loads(marker.read_text(encoding="utf-8"))
    if any(not (app_dir / name).exists() or file_hash(app_dir / name) != digest for name, digest in done["files"].items()):
        return None
    return done


def run_app_segment(*, segment: dict, group: str, models: Sequence[dict], processor_factory, seg_dir: Path,
                    chunk_endpoints: int, prefix_bars: Sequence[MarketBar], endpoint_bars: Sequence[MarketBar],
                    restart_each_chunk: bool = False, allow_resume: bool = False, calendar=None,
                    failure: dict | None = None, log: Callable[[str], None] = print) -> dict:
    """Bar-by-bar app replay of one segment in checkpointed chunks; returns the app run record."""
    app_dir = seg_dir / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    run_path = app_dir / "run.json"
    stream, fault = segment["stream"], segment.get("fault")
    steps, injections, final = feed_program(endpoint_bars, fault, stream=stream, chunk_endpoints=chunk_endpoints,
                                            restart_each_chunk=restart_each_chunk)
    chunks = (len(endpoint_bars) + chunk_endpoints - 1) // chunk_endpoints
    db_path, checkpoint = seg_dir / "parity.sqlite3", seg_dir / "checkpoint.sqlite3"
    done = [_done_ok(app_dir, n) for n in range(chunks)]
    start_chunk = next((n for n, marker in enumerate(done) if marker is None), chunks)
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() and start_chunk else None
    if run is not None and run.get("segment") != json.loads(json.dumps(segment, default=str)):
        raise ValueError("An existing app run in this directory belongs to a different segment definition")
    if run is not None and start_chunk == chunks and run.get("finished_at_utc"):
        log(f"{segment['segment_id']} {group} app run already complete; reusing sealed chunks")
        return run
    if run is None:
        prefix_sha = build_parity_db(db_path, prefix_bars)
        run = dict(harness_version=HARNESS_VERSION, raw_snapshot_schema=RAW_SNAPSHOT_SCHEMA_VERSION,
                   protocol=dict(version=2, sha256=PROTOCOL_SHA256, lock_sha256=PROTOCOL_LOCK_SHA256),
                   segment=segment, group=group, stream=stream, models=list(models), chunk_endpoints=chunk_endpoints,
                   app_fixes=dict(getattr(processor_factory, "app_fixes", {}) or {}),
                   restart_each_chunk=restart_each_chunk, carry_bar_provenance=True, calendar_id=None,
                   parity_db_prefix_sha256=prefix_sha, prefix_bars=len(prefix_bars), code_sha256=code_hashes(),
                   injections=injections, failure=failure, resumes=[], started_at_utc=datetime.now(timezone.utc).isoformat())
        _write_truth(app_dir / "truth_bars.parquet", prefix_bars, endpoint_bars, final, injections,
                     max(int(m["consumed_rows"]) for m in models) + 1)
        start_chunk = 0
        dump(run_path, run)  # written first so an interrupted run resumes instead of silently starting over
    elif start_chunk < chunks:
        if not allow_resume:
            raise RuntimeError("Chunk resume is a restart; allowed only after the stream's restart test passed")
        previous = done[start_chunk - 1]
        if file_hash(checkpoint) != previous["db_sha256"]:
            raise ValueError("Checkpoint DB does not match the last finished chunk")
        for suffix in ("-wal", "-shm"):
            Path(f"{db_path}{suffix}").unlink(missing_ok=True)
        shutil.copyfile(checkpoint, db_path)
        run["resumes"].append(dict(chunk=start_chunk, at_utc=datetime.now(timezone.utc).isoformat()))
        log(f"resume at chunk {start_chunk} from verified checkpoint (a restart)")
    driver = AppDriver(group=group, stream=stream, db_path=db_path, processor_factory=processor_factory,
                       thresholds={m["model_id"]: m["threshold"] for m in models}, calendar=calendar)
    run["calendar_id"] = driver.calendar[0]
    active = {"on": False}
    guard = (producer_failure(failure["feature_set"], failure.get("mode", "raise"), active=lambda: active["on"])
             if failure else contextlib.nullcontext())
    with guard:
        # One processor for the whole invocation (continuous); it restarts only on an explicit restart step.
        # The first (re)start of an invocation is chunk 0's cold open or the resume restart.
        driver.start()
        just_started = True
        try:
            for chunk in range(start_chunk, chunks):
                started = time.perf_counter()
                parts, files = [], {}
                lo, hi = chunk * chunk_endpoints, min((chunk + 1) * chunk_endpoints, len(endpoint_bars))
                for index, step in steps:
                    if not lo <= index < hi:
                        continue
                    if step["op"] == "restart":
                        if not just_started:
                            prefix = f"chunk_{chunk:03d}_p{len(parts)}"
                            files.update(driver.flush(app_dir, prefix))
                            parts.append(dict(prefix=prefix, instance=driver.instance))
                            driver.start()
                            just_started = True
                        continue
                    if failure:
                        active["on"] = int(failure.get("from", -1)) <= index < int(failure.get("to", -1))
                    driver.apply(step)
                    just_started = False
                prefix = f"chunk_{chunk:03d}_p{len(parts)}"
                files.update(driver.flush(app_dir, prefix))
                parts.append(dict(prefix=prefix, instance=driver.instance))
                checkpoint.unlink(missing_ok=True)
                target = sqlite3.connect(checkpoint)
                try:
                    driver.store.connection.backup(target)
                finally:
                    target.close()
                marker = dict(chunk=chunk, endpoints=[iso(endpoint_bars[lo].timestamp), iso(endpoint_bars[hi - 1].timestamp)],
                              parts=parts, instance=driver.instance, files=files, db_sha256=file_hash(checkpoint),
                              seconds=round(time.perf_counter() - started, 2), harness_version=HARNESS_VERSION)
                dump(app_dir / f"chunk_{chunk:03d}.done", marker)
                done[chunk] = marker
                log(f"{segment['segment_id']} {group} chunk {chunk + 1}/{chunks} done in {marker['seconds']} s")
        finally:
            driver.stop()  # a crash must not leave the parity DB locked
    run["chunks"] = chunks
    run["parity_db_sha256"] = file_hash(db_path)
    run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_events(app_dir, db_path, stream)
    dump(run_path, run)
    return run


def _write_truth(path: Path, prefix: Sequence[MarketBar], endpoints: Sequence[MarketBar], final: dict[str, MarketBar],
                 injections: Sequence[dict], prefix_tail: int) -> None:
    """Expected canonical rows (final delivered version) for S1; dropped bars carry delivered=False."""
    injected = {}
    for item in injections:
        injected.setdefault(item["timestamp"], []).append(f"{item['kind']}:{item['action']}")
    records = []
    for bar, is_endpoint in [*((b, False) for b in list(prefix)[-prefix_tail:]), *((b, True) for b in endpoints)]:
        stamp = iso(bar.timestamp)
        delivered = final.get(stamp, bar if not is_endpoint else None)
        stored = observe_bar(delivered) if delivered is not None else None
        records.append(dict(timestamp=stamp, endpoint=is_endpoint, delivered=stored is not None,
                            contract_symbol=bar.contract_symbol,
                            expected_is_complete=None if stored is None else stored.is_complete,
                            expected_observation_kind=None if stored is None else stored.observation_kind,
                            expected_feed_type=None if stored is None else stored.feed_type,
                            expected_bar_version=None if stored is None else stored.bar_version,
                            injected=";".join(injected.get(stamp, [])) or None))
    pd.DataFrame.from_records(records).to_parquet(path, index=False)


def _write_events(app_dir: Path, db_path: Path, stream: str) -> None:
    """S2 app side: setup events the app's detectors recorded on the replayed bars (ES only, D5)."""
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        events = pd.read_sql_query(
            "SELECT event_key, revision, event_kind, strategy, source_timestamp_utc, observed_at_utc, setup_type, "
            "setup_side, setup_family, rule_confidence, detector_identity, detector_version, geometry_json "
            "FROM setup_events ORDER BY id", connection)
    finally:
        connection.close()
    events.to_parquet(app_dir / "setup_events.parquet", index=False)


# ---------------------------------------------------------------- reference

def dataset_metadata_for(contract: dict) -> tuple[Path, dict]:
    """Frozen dataset config whose lineage names the source history's source_dataset_sha256."""
    digest = contract["source_dataset_sha256"]
    for directory in DATASET_DIRS:
        for name in ("manifest.json", "dataset.metadata.json"):
            path = ROOT / directory / name
            if path.exists() and digest in path.read_text(encoding="utf-8"):
                return ROOT / directory / "dataset.metadata.json", json.loads((ROOT / directory / "dataset.metadata.json").read_text(encoding="utf-8"))
    raise ValueError(f"No frozen dataset config names source dataset {digest}")


_RECOMPUTE_CACHE: dict[tuple, tuple[pd.DataFrame, pd.Series]] = {}


def _recompute_full(builder_config, bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One producer pass of a frozen dataset config over a bar series, shared by every model that uses it."""
    from features.builder import FeatureDatasetBuilder

    key = (hashlib.sha256(json.dumps(builder_config.to_dict() if hasattr(builder_config, "to_dict") else repr(builder_config),
                                     sort_keys=True, default=str).encode()).hexdigest(),
           hashlib.sha256(pd.util.hash_pandas_object(bars, index=False).to_numpy().tobytes()).hexdigest())
    if key not in _RECOMPUTE_CACHE:
        _RECOMPUTE_CACHE.clear()  # one window at a time; keeps memory bounded
        market = bars.rename(columns={"timestamp": "datetime"})[["datetime", "open", "high", "low", "close", "volume",
                                                                  *(["contract_symbol"] if "contract_symbol" in bars else [])]]
        built, _ = FeatureDatasetBuilder(builder_config).build(market)
        times = pd.to_datetime(built["datetime"], utc=True).reset_index(drop=True)
        _RECOMPUTE_CACHE[key] = (built.reset_index(drop=True), times)
    return _RECOMPUTE_CACHE[key]


def recompute_history(model_dir: Path, contract: dict, bars: pd.DataFrame, *, builder_config=None) -> pd.DataFrame:
    """Recompute path: the frozen dataset config on the full contiguous bar series, frozen selection, no fills."""
    from features.config import FeatureBuilderConfig
    from preprocessing.feature_selection import encode_candidate_features

    summary = json.loads((model_dir / "training_summary.json").read_text(encoding="utf-8"))
    names = list(summary["selected_feature_names"])
    if builder_config is None:
        builder_config = FeatureBuilderConfig.from_dict(dataset_metadata_for(contract)[1]["config"])
    built, times = _recompute_full(builder_config, bars)
    encoded, encoding = encode_candidate_features(built[names], names)
    if encoding["encoded_columns"]:
        raise ValueError("Recompute path requires numeric selected predictors")
    positions = pd.Series(range(len(bars)), index=pd.to_datetime(bars.timestamp, utc=True)).reindex(times)
    if positions.isna().any():
        raise ValueError("Recomputed rows do not map onto the input bars")
    return pd.concat([pd.DataFrame({"source_row_idx": positions.to_numpy(dtype=np.int64), "datetime": times}),
                      encoded.reset_index(drop=True)], axis=1)


class FrozenReference:
    """Frozen production runner (hash-verified B4 v2 bytes) with the app's taps, on a source-bar history."""

    def __init__(self, model: dict) -> None:
        self.model = model
        self.verified = verify_bytes(model)
        self.summary = json.loads((model["model_dir"] / "training_summary.json").read_text(encoding="utf-8"))
        if model["stream"] == "ES":
            from scripts.audit_b2_candidate_history import runtime
        else:
            from scripts.run_b2_ote_candidate_history import runtime_model as runtime
        self.runner = runtime(model["model_dir"], self.summary)
        pins = model["freeze"]["source_data_pins"]
        self.prepared = (ROOT / pins["prepared_manifest"]).parent / "prepared" / model["freeze"]["target"]
        contract_path = self.prepared / "source_history_contract.json"
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        # The freeze pins the contract object and its canonical-JSON hash (model_training.source_history).
        if (contract_sha256(self.contract) != pins["source_history_contract_sha256"]
                or self.contract != pins["source_history_contract"]):
            raise ValueError("Source history contract does not match the B4 v2 freeze")
        self.consumed = consumed_rows(self.runner.loaded_model.manifest)
        self.tap = RunnerTap(self.runner, model_id=model["model_id"], consumed=self.consumed, threshold=float(model["threshold"]))

    def frozen_history(self, first: int, last: int) -> pd.DataFrame:
        """Rows ``first..last`` (positions == source_row_idx) of the hash-verified frozen source history."""
        path = self.prepared / "source_history.parquet"
        if file_hash(path) != self.contract["source_history_sha256"]:
            raise ValueError("Source history does not match its frozen contract")
        columns = ["source_row_idx", "datetime", *self.summary["selected_feature_names"]]
        frame = pd.read_parquet(path, columns=columns, filters=[("source_row_idx", ">=", int(first)),
                                                                 ("source_row_idx", "<=", int(last))])
        frame = frame.sort_values("source_row_idx").reset_index(drop=True)
        if not np.array_equal(frame.source_row_idx.to_numpy(), np.arange(first, last + 1)):
            raise ValueError("Frozen source history is not consecutive over the requested rows")
        return frame

    def score(self, history: pd.DataFrame, endpoints: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Score each endpoint on ``history.iloc[end-context:end+1]``; unknown endpoints are reported, not scored."""
        context = int(self.summary["context_rows"])
        times = pd.to_datetime(history.datetime, utc=True).map(iso)
        position = {stamp: index for index, stamp in enumerate(times)}
        missing, short = [], []
        for endpoint in endpoints:
            index = position.get(endpoint)
            if index is None:
                missing.append(endpoint)
            elif index < context:
                short.append(endpoint)
            else:
                self.runner.predict_latest(history.iloc[index - context:index + 1])
        io, scores = self.tap.frames()
        self.tap.records = []
        return io, pd.DataFrame.from_records(scores), dict(missing_endpoints=missing, short_history_endpoints=short)


def build_reference(model: dict, segment: dict, seg_dir: Path, *, mode: str = "frozen",
                    bars: pd.DataFrame | None = None) -> dict:
    """Reference io/scores for one model on a segment's endpoints (``frozen`` or ``recompute``)."""
    ref_dir = seg_dir / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    reference = FrozenReference(model)
    context = int(reference.summary["context_rows"])
    if mode == "frozen":
        if segment["source"] != "H":
            raise ValueError("Frozen source history covers historical (H) windows only")
        history = reference.frozen_history(segment["first_endpoint_pos"] - context, segment["last_endpoint_pos"])
        frame = training_frame(segment["stream"])
        expected = frame.timestamp.iloc[history.source_row_idx].map(iso).to_numpy()
        if not (pd.to_datetime(history.datetime, utc=True).map(iso).to_numpy() == expected).all():
            raise ValueError("Frozen source history timestamps differ from the pinned training source")
    elif mode == "recompute":
        history = recompute_history(model["model_dir"], reference.contract, bars)
    else:
        raise ValueError(mode)
    io, scores, notes = reference.score(history, segment["endpoints"])
    io_path, scores_path = ref_dir / f"{model['model_id']}.io.parquet", ref_dir / f"{model['model_id']}.scores.parquet"
    io.to_parquet(io_path, index=False)
    scores.to_parquet(scores_path, index=False)
    record = dict(harness_version=HARNESS_VERSION, mode=mode, model_id=model["model_id"], queue_id=model["queue_id"],
                  verified=reference.verified, source_history_sha256=reference.contract["source_history_sha256"],
                  source_row_offset=reference.contract.get("source_row_offset"), consumed_rows=reference.consumed,
                  context_rows=context, files={io_path.name: file_hash(io_path), scores_path.name: file_hash(scores_path)},
                  **notes)
    dump(ref_dir / f"{model['model_id']}.run.json", record)
    return record


# ---------------------------------------------------------------- plumbing CLI

def plumbing(queue_id: int, endpoints: int = 3, chunk_endpoints: int = 2) -> dict:
    """Tiny real-data run (training-source tail, H) through app, reference and comparator; output recorded only."""
    from scripts.section4_parity_compare import compare_segment, model_spec, reference_setup_events

    protocol = load_protocol()
    group = next(name for name, spec in protocol["engine_composition"]["groups"].items() if queue_id in spec["queue_ids"])
    stream = protocol["engine_composition"]["groups"][group]["stream"]
    history_bars = int(protocol["engine_composition"]["groups"][group]["runtime_history_bars"])
    frame = training_frame(stream)
    segment = historical_segment(
        stream, f"plumbing_{queue_id:02d}_{stream}_tail", iso(frame.timestamp.iloc[-1]), before=endpoints - 1, after=0,
        prefix=history_bars, rule="plumbing only: the last N bars of the pinned training source (not a protocol segment)",
        plumbing=True)
    root = RUN_ROOT / "plumbing"
    windows_sha = write_windows(root / f"windows_{queue_id:02d}.json", [segment])
    seg_dir = root / stream / group / segment["segment_id"]
    print(f"plumbing {queue_id} {group} windows {windows_sha[:12]}", flush=True)
    loaded, records = staged_group(group, protocol)
    prefix_bars, endpoint_bars = segment_bars(segment)
    started = time.perf_counter()
    run = run_app_segment(segment=segment, group=group, models=records, processor_factory=group_processor_factory(group, loaded),
                          seg_dir=seg_dir, chunk_endpoints=chunk_endpoints, prefix_bars=prefix_bars,
                          endpoint_bars=endpoint_bars, log=lambda m: print(m, flush=True))
    print(f"app done {time.perf_counter() - started:.1f} s", flush=True)
    model = roster_by_id()[queue_id]
    reference = build_reference(model, segment, seg_dir)
    print(f"reference done; missing {len(reference['missing_endpoints'])}", flush=True)
    setup_reference = (reference_setup_events(model, segment["endpoints"][0], segment["endpoints"][-1])
                       if stream == "ES" else None)
    summary = compare_segment(seg_dir, [model_spec(model, protocol)], protocol, windows_sha256=windows_sha,
                              expected_artifacts={model["model_id"]: freeze_entries()[queue_id]},
                              setup_reference=setup_reference)
    print(json.dumps(summary["models"][model["model_id"]]["headline"], default=str), flush=True)
    print("S2", json.dumps({k: v for k, v in summary["S2"].items() if k != "mismatches"}, default=str), flush=True)
    print(f"PLUMBING {queue_id} COMPLETE", flush=True)
    return summary


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "plumbing":
        plumbing(int(sys.argv[2]), *(int(a) for a in sys.argv[3:5]))
    else:
        print(__doc__)
        sys.exit(2)
