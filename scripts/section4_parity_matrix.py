"""Section 4 prompt 4: the full protocol-v2 parity matrix for all 16 roster versions (R2-R5 runs).

Drives ``scripts/section4_parity_harness.py`` (app path, frozen/recompute reference, faults) and
``scripts/section4_parity_compare.py`` (row-level comparison) over the windows written by rule in
``artifacts/s4parity_20261002_v1/windows.json``. One job, one segment at a time, chunked and checkpointed:

  1. per stream: verify the B4 v2 bytes (staged manifests over hash-verified bytes)
  2. anchor check: recompute path vs frozen source_history.parquet on one H segment per group
  3. the stream's restart gate: base R segment(s) continuous, restart segments (explicit points plus a restart at
     every chunk boundary), bitwise compare_app_runs; chunk resumes are allowed only after it passes
  4. every remaining segment x group: app run, references for the group's models, compare_segment
  5. ES group composition: single-model engines vs the group engine on sampled endpoints (app vs app)

State (pending -> running -> compared | failed | blocked) goes to ``matrix_state.json``. Markers:
``SEGMENT <id> <group> COMPLETE``, ``STREAM <stream> COMPLETE``, ``MATRIX COMPLETE``. A lock file refuses a
second instance. No fixes, classification or reports here (classification and reports follow the run).

Usage:
  python scripts/section4_parity_matrix.py windows        # write windows.json by rule (immutable)
  python scripts/section4_parity_matrix.py run [ES|EURUSD]  # run the matrix (background job)
  python scripts/section4_parity_matrix.py status
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import section4_parity_harness as H  # noqa: E402
from scripts.run_section3_smoke_replay import file_hash  # noqa: E402
from scripts.section4_parity_compare import (  # noqa: E402
    CLASS_TOLERANCE, compare_app_runs, compare_segment, model_spec, reference_setup_events, tolerance, within,
)

MATRIX_VERSION = "s4-parity-matrix-v1"
WINDOWS = H.RUN_ROOT / "windows.json"
STATE = H.RUN_ROOT / "matrix_state.json"
LOCK = H.RUN_ROOT / "matrix.lock"
LOG = ROOT / f"logs/section4_matrix_{H.RUN_DATE}.log"
SHAPE = {"ES": (48, 96), "EURUSD": (32, 64)}
GROUPS = {"ES": ("ES_FRVP", "ES_ICT"), "EURUSD": ("EURUSD_OTE",)}
ANCHOR_SEGMENT = {"ES": "ES_H_rth_maint", "EURUSD": "EU_H_session_dst"}
MIN_FREE_GB = 5.0
COMPOSITION_SAMPLE = (60, 6)  # ES_R_rth_maint endpoint index and count (engine_composition: sampled endpoints)

_original_training_frame = H.training_frame


@lru_cache(maxsize=2)
def _cached_training_frame(stream: str) -> pd.DataFrame:
    return _original_training_frame(stream)


H.training_frame = lambda stream: _cached_training_frame(stream).copy()  # read once per process, never mutated


def log(message: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}"
    print(line, flush=True)


# ---------------------------------------------------------------- windows by rule

@lru_cache(maxsize=2)
def _live_times(stream: str) -> tuple[str, ...]:
    return tuple(H.iso(bar.timestamp) for bar in H.live_bars(stream))


def live_segment(stream: str, segment_id: str, event: str, *, rule: str, scenarios, prefix: int | None,
                 fault: dict | None = None, base: str | None = None, endpoint_slice: tuple[int, int] | None = None,
                 reference_bars: str = "full_recorded_series") -> dict:
    """Recorded (R) bars [event-before, event+after] from the live DB; ``prefix=None`` = everything recorded."""
    before, after = SHAPE[stream]
    times = _live_times(stream)
    position = times.index(H.iso(event))
    first, last = position - before, position + after
    if endpoint_slice is not None:
        first, last = first + endpoint_slice[0], first + endpoint_slice[0] + endpoint_slice[1] - 1
    if first < 0 or last >= len(times):
        raise ValueError(f"{segment_id} exceeds the recorded {stream} bars")
    prefix_bars = first if prefix is None else min(int(prefix), first)
    return dict(segment_id=segment_id, stream=stream, source="R", rule=rule, event=H.iso(event), prefix_bars=prefix_bars,
                endpoints=list(times[first:last + 1]), scenarios=list(scenarios), fault=fault, plumbing=False,
                base_segment=base, reference="recompute", reference_bars=reference_bars,
                prefix_start=times[first - prefix_bars] if prefix_bars else None, live_db=str(H.LIVE_DB.relative_to(ROOT)))


def build_windows() -> list[dict]:
    """Every protocol segment, chosen by calendar/event presence only (protocol segments.selection)."""
    segs: list[dict] = []
    es_b, es_a = SHAPE["ES"]
    eu_b, eu_a = SHAPE["EURUSD"]
    hist = lambda stream, sid, event, rule, scen: {**H.historical_segment(
        stream, sid, event, before=SHAPE[stream][0], after=SHAPE[stream][1], prefix=8000, rule=rule, scenarios=scen),
        "reference": "frozen", "base_segment": None}
    # ES historical (frozen source_history.parquet reference)
    segs += [
        hist("ES", "ES_H_rth_maint", "2026-06-16T13:30:00+00:00",
             "first 13:30 UTC RTH open in the last full training week; the 96 bars after reach the 21:00-22:00 break",
             ["overnight_rth_transitions", "daily_maintenance_break", "htf_publication"]),
        hist("ES", "ES_H_spring_dst", "2026-03-08T22:00:00+00:00",
             "first bar after the weekend in the spring DST week (reopen 22:00 UTC under EDT)",
             ["spring_dst", "weekend_reopen", "htf_publication"]),
        hist("ES", "ES_H_fall_dst", "2025-11-02T23:00:00+00:00",
             "first bar after the weekend in the fall DST week (reopen 23:00 UTC under EST)",
             ["fall_dst", "weekend_reopen", "htf_publication"]),
        hist("ES", "ES_H_holiday", "2025-11-27T23:00:00+00:00",
             "first bar after the Thanksgiving 2025-11-27 planned closure (17:55 -> 23:00 UTC)",
             ["holiday_planned_closure"]),
        hist("ES", "ES_H_roll", "2026-06-17T00:00:00+00:00",
             "first contract_symbol change in June 2026 (ESM6 -> ESU6)", ["quarterly_roll", "daily_maintenance_break"]),
    ]
    # ES recorded (recompute reference over everything recorded up to the segment end)
    r_rule = "recorded bars; prefix = everything recorded (protocol segments.prefix)"
    segs += [
        live_segment("ES", "ES_R_rth_maint", "2026-09-15T13:30:00+00:00", prefix=None,
                     rule=f"R complete week (ESZ6, is_complete known): 2026-09-15 RTH open through the break; {r_rule}",
                     scenarios=["overnight_rth_transitions", "daily_maintenance_break", "htf_publication"]),
        live_segment("ES", "ES_R_roll_weekend", "2026-09-13T22:00:00+00:00", prefix=None,
                     rule=f"R roll ESU6 -> ESZ6 at the Sunday reopen; {r_rule}", scenarios=["quarterly_roll", "weekend_reopen"]),
        live_segment("ES", "ES_R_labor_day", "2026-09-07T22:00:00+00:00", prefix=None,
                     rule=f"R Labor Day closure 16:55 -> 22:00 UTC (ESU6, completion unknown); {r_rule}",
                     scenarios=["holiday_planned_closure"]),
        live_segment("ES", "ES_R_gap_backfill", "2026-09-18T11:40:00+00:00", prefix=None,
                     rule=f"R gap/backfill candidate 2026-09-18 11:40-13:00, bars stored by repair; {r_rule}",
                     scenarios=["missing_bars", "reconnect_backfill", "revision"]),
    ]
    base = "ES_R_rth_maint"
    f_rule = "F on ES_R_rth_maint bars; reference recomputed on the final delivered series"
    segs += [
        live_segment("ES", "ES_F_restart_rth", "2026-09-15T13:30:00+00:00", prefix=None, base=base,
                     rule=f"{f_rule}; restart at every chunk boundary plus mid-RTH, inside a full lag window, after the break",
                     scenarios=["restart"], fault=dict(kind="restart", restart_each_chunk=True, points=[
                         dict(at=84, point="mid_rth"), dict(at=100, point="inside_full_lag_window"),
                         dict(at=139, point="after_maintenance_break")])),
        live_segment("ES", "ES_F_restart_roll", "2026-09-13T22:00:00+00:00", prefix=None, base="ES_R_roll_weekend",
                     rule="F on ES_R_roll_weekend bars; restart at every chunk boundary plus after the roll boundary",
                     scenarios=["restart"], fault=dict(kind="restart", restart_each_chunk=True, points=[
                         dict(at=50, point="after_roll_boundary")])),
        live_segment("ES", "ES_F_faults", "2026-09-15T13:30:00+00:00", prefix=None, base=base,
                     rule=f"{f_rule}; dropped 1/3/12 non-closure bars, backfill behind the cursor, revision latest+earlier, reconnect batch",
                     scenarios=["missing_bars", "reconnect_backfill", "revision"], fault=dict(kind="multi", faults=[
                         dict(kind="dropped", at=8, count=1), dict(kind="dropped", at=20, count=3),
                         dict(kind="backfill", at=30, count=4, release_after=2), dict(kind="revision", at=52, earlier_lag=3),
                         dict(kind="reconnect", at=66, count=5), dict(kind="dropped", at=100, count=12)])),
        live_segment("ES", "ES_F_cold_start", "2026-09-15T13:30:00+00:00", prefix=100, base=base, reference_bars="app_view",
                     rule="F cold start: 100 prefix bars (< FRVP minimum_runtime_history_bars 132); reference on the same bars",
                     scenarios=["cold_start_warmup"]),
        {**live_segment("ES", "ES_F_producer_failure", "2026-09-15T13:30:00+00:00", prefix=None, base=base,
                        rule=f"{f_rule}; one feature set raises for endpoints 60-65 (ES_FRVP frvp_context, ES_ICT ict_context)",
                        scenarios=["producer_failure"]),
         "failure": {"ES_FRVP": dict(feature_set="frvp_context", mode="raise", **{"from": 60, "to": 66}),
                     "ES_ICT": dict(feature_set="ict_context", mode="raise", **{"from": 60, "to": 66})}},
        live_segment("ES", "ES_F_composition", "2026-09-15T13:30:00+00:00", prefix=None, base=base,
                     endpoint_slice=COMPOSITION_SAMPLE,
                     rule="F group composition: single-model engines on 6 sampled ES_R_rth_maint endpoints (index 60-65) vs the group engine",
                     scenarios=["group_composition"]),
    ]
    # EURUSD historical
    segs += [
        hist("EURUSD", "EU_H_weekend_dst", "2026-03-08T21:00:00+00:00",
             "first bar after the weekend in the US/EU DST mismatch weeks (US DST began 2026-03-08)",
             ["weekend_close_open", "us_eu_dst_mismatch", "completed_30m_1h_atr_publication"]),
        hist("EURUSD", "EU_H_session_dst", "2026-03-10T12:00:00+00:00",
             "12:00 UTC New York open under EDT while London is on GMT; the 32 bars before are London",
             ["session_transitions_asia_london_ny", "us_eu_dst_mismatch", "completed_30m_1h_atr_publication"]),
        hist("EURUSD", "EU_H_holiday", "2025-12-25T23:00:00+00:00",
             "first bar after the Christmas closure (12-24 18:00 -> 12-25 23:00 UTC)", ["holiday_closure_calendar_v1"]),
    ]
    er = "recorded bars; prefix up to 8000 (protocol segments.prefix); EURUSD completion unknown; calendar v1 provisional"
    segs += [
        live_segment("EURUSD", "EU_R_session", "2026-09-15T12:00:00+00:00", prefix=8000,
                     rule=f"R week 2026-09-15: London -> New York at 12:00 UTC; {er}",
                     scenarios=["session_transitions_asia_london_ny", "completed_30m_1h_atr_publication"]),
        live_segment("EURUSD", "EU_R_weekend", "2026-09-20T21:05:00+00:00", prefix=8000,
                     rule=f"R weekend 2026-09-18 -> 09-20 (first recorded bar after the weekend); {er}",
                     scenarios=["weekend_close_open"]),
        live_segment("EURUSD", "EU_R_outage_edge", "2026-05-14T22:10:00+00:00", prefix=8000,
                     rule=f"R outage edge 2026-05-14 22:10 (first bar after a recording gap); {er}",
                     scenarios=["missing_bars", "reconnect_backfill"]),
    ]
    base = "EU_R_session"
    ef = "F on EU_R_session bars; reference recomputed on the final delivered series"
    segs += [
        live_segment("EURUSD", "EU_F_restart_session", "2026-09-15T12:00:00+00:00", prefix=8000, base=base,
                     rule=f"{ef}; restart at every chunk boundary plus London, inside a TCN window, New York",
                     scenarios=["restart"], fault=dict(kind="restart", restart_each_chunk=True, points=[
                         dict(at=8, point="london"), dict(at=20, point="inside_tcn_window"), dict(at=36, point="new_york")])),
        live_segment("EURUSD", "EU_F_restart_weekend", "2026-09-20T21:05:00+00:00", prefix=8000, base="EU_R_weekend",
                     rule="F on EU_R_weekend bars; restart at every chunk boundary plus after the weekend reopen",
                     scenarios=["restart"], fault=dict(kind="restart", restart_each_chunk=True, points=[
                         dict(at=34, point="after_weekend_reopen")])),
        live_segment("EURUSD", "EU_F_faults", "2026-09-15T12:00:00+00:00", prefix=8000, base=base,
                     rule=f"{ef}; dropped 1/3/12 bars, backfill behind the cursor, revision latest+earlier, reconnect batch",
                     scenarios=["missing_bars", "reconnect_backfill", "revision"], fault=dict(kind="multi", faults=[
                         dict(kind="dropped", at=6, count=1), dict(kind="dropped", at=12, count=3),
                         dict(kind="backfill", at=20, count=3, release_after=2), dict(kind="revision", at=32, earlier_lag=3),
                         dict(kind="reconnect", at=44, count=4), dict(kind="dropped", at=64, count=12)])),
        live_segment("EURUSD", "EU_F_cold_start", "2026-09-15T12:00:00+00:00", prefix=200, base=base, reference_bars="app_view",
                     rule="F cold start: 200 prefix bars (< minimum_runtime_history_bars 250); reference on the same bars",
                     scenarios=["cold_start_warmup"]),
        {**live_segment("EURUSD", "EU_F_strategy_timeout", "2026-09-15T12:00:00+00:00", prefix=8000, base=base,
                        rule=f"{ef}; strategy_signals runs with a near-zero strategy timeout for endpoints 40-45",
                        scenarios=["strategy_timeout_producer_failure"]),
         "failure": {"EURUSD_OTE": dict(feature_set="strategy_signals", mode="timeout", **{"from": 40, "to": 46})}},
    ]
    for seg in segs:
        fault = seg.get("fault") or {}
        seg["restart_each_chunk"] = bool(fault.pop("restart_each_chunk", False)) if fault else False
        seg["groups"] = list(GROUPS[seg["stream"]])
    return segs


def load_windows() -> tuple[list[dict], str]:
    digest = file_hash(WINDOWS)
    if (WINDOWS.with_suffix(".json.sha256").read_text(encoding="utf-8").strip() != digest):
        raise ValueError("windows.json does not match its sha256 sidecar")
    payload = json.loads(WINDOWS.read_text(encoding="utf-8"))
    if payload["protocol_sha256"] != H.PROTOCOL_SHA256:
        raise ValueError("windows.json was written under a different protocol")
    return payload["segments"], digest


# ---------------------------------------------------------------- state

def read_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else dict(matrix_version=MATRIX_VERSION, items={})


def set_state(key: str, **fields) -> None:
    state = read_state()
    item = state["items"].setdefault(key, {})
    item.update(fields, updated_at_utc=datetime.now(timezone.utc).isoformat())
    STATE.write_text(json.dumps(state, indent=1, default=str) + "\n", encoding="utf-8")


def state_of(key: str) -> str | None:
    return read_state()["items"].get(key, {}).get("status")


# ---------------------------------------------------------------- bars

def _frame(bars, *, stream: str) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp": [b.timestamp for b in bars], "open": [b.open for b in bars],
                          "high": [b.high for b in bars], "low": [b.low for b in bars], "close": [b.close for b in bars],
                          "volume": [b.volume for b in bars], "contract_symbol": [b.contract_symbol for b in bars]})
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    return frame.drop(columns=["contract_symbol"]) if stream == "EURUSD" else frame


def reference_frame(seg: dict, prefix_bars, endpoint_bars, chunk: int) -> pd.DataFrame:
    """Recompute input: everything recorded up to the segment end (or the app's own view for cold start), with the
    final delivered version of each endpoint bar (dropped bars absent, revised bars final)."""
    _, _, final = H.feed_program(endpoint_bars, seg.get("fault"), stream=seg["stream"], chunk_endpoints=chunk,
                                 restart_each_chunk=seg.get("restart_each_chunk", False))
    delivered = [final[H.iso(b.timestamp)] for b in endpoint_bars if H.iso(b.timestamp) in final]
    if seg.get("reference_bars") == "app_view":
        history = list(prefix_bars)
    else:
        history = [b for b in H.live_bars(seg["stream"], end=seg["endpoints"][0]) if H.iso(b.timestamp) < seg["endpoints"][0]]
    return _frame([*history, *delivered], stream=seg["stream"])


# ---------------------------------------------------------------- anchor check

def anchor_check(group: str, seg: dict, models: list[dict], protocol: dict) -> dict:
    """Recompute path vs frozen source_history.parquet on one H segment (same 8000-bar span as the app)."""
    out = H.RUN_ROOT / "anchor" / seg["stream"] / group / seg["segment_id"]
    if (out / "anchor.json").exists():
        return json.loads((out / "anchor.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    frame = H.training_frame(seg["stream"]).iloc[seg["prefix_start_pos"]:seg["last_endpoint_pos"] + 1].reset_index(drop=True)
    if seg["stream"] == "EURUSD":
        frame = frame.drop(columns=["contract_symbol"])
    result = dict(matrix_version=MATRIX_VERSION, group=group, segment=seg["segment_id"], bars=len(frame), models={})
    for model in models:
        spec = model_spec(model, protocol)
        reference = H.FrozenReference(model)
        context = int(reference.summary["context_rows"])
        frozen = reference.frozen_history(seg["first_endpoint_pos"] - context, seg["last_endpoint_pos"])
        recomputed = H.recompute_history(model["model_dir"], reference.contract, frame)
        recomputed = recomputed.set_index(pd.to_datetime(recomputed.datetime, utc=True))
        aligned = recomputed.reindex(pd.to_datetime(frozen.datetime, utc=True))
        features, diffs = {}, []
        for name in reference.summary["selected_feature_names"]:
            cls = spec["feature_classes"].get(name, "C1_continuous")
            atol, rtol = tolerance(protocol, CLASS_TOLERANCE[cls])
            app, ref = aligned[name].to_numpy(float), frozen[name].to_numpy(float)
            bad = ~within(app, ref, atol, rtol)
            if bad.any():
                with np.errstate(invalid="ignore"):
                    abs_err = np.abs(app - ref)
                features[name] = dict(mismatches=int(bad.sum()), cells=int(len(bad)), tolerance_class=cls,
                                      producer=spec["producers"].get(name),
                                      first_ts=H.iso(frozen.datetime.iloc[int(np.argmax(bad))]),
                                      max_abs=float(np.nanmax(abs_err[bad])) if np.isfinite(abs_err[bad]).any() else None)
                diffs.append(pd.DataFrame({"feature": name, "datetime": frozen.datetime[bad].map(H.iso).to_numpy(),
                                           "recompute": app[bad], "frozen": ref[bad]}))
        _, frozen_scores, _ = reference.score(frozen, seg["endpoints"])
        rec_hist = aligned.reset_index(drop=True).assign(datetime=frozen.datetime.to_numpy(),
                                                         source_row_idx=frozen.source_row_idx.to_numpy())
        _, rec_scores, _ = reference.score(rec_hist, seg["endpoints"])
        atol, rtol = tolerance(protocol, "S6_scores")
        score_bad = ~within(rec_scores.calibrated_probability.to_numpy(float), frozen_scores.calibrated_probability.to_numpy(float), atol, rtol)
        thr = float(model["threshold"])
        flips = int(((rec_scores.calibrated_probability >= thr) != (frozen_scores.calibrated_probability >= thr)).sum())
        if diffs:
            path = out / f"{model['model_id']}.anchor_diff.parquet"
            pd.concat(diffs, ignore_index=True).to_parquet(path, index=False)
        result["models"][model["model_id"]] = dict(
            queue_id=model["queue_id"], rows=len(frozen), features_compared=len(reference.summary["selected_feature_names"]),
            features_mismatched=len(features), cells_mismatched=sum(f["mismatches"] for f in features.values()),
            mismatched_features=features, endpoints_scored=len(frozen_scores),
            score_mismatches=int(score_bad.sum()), decision_flips=flips,
            anchored=not features and not score_bad.any())
        log(f"anchor {group} {model['queue_id']}: features mismatched {len(features)}, score mismatches {int(score_bad.sum())}, flips {flips}")
    H.dump(out / "anchor.json", result)
    return result


# ---------------------------------------------------------------- segments

def seg_dir(seg: dict, group: str) -> Path:
    return H.RUN_ROOT / seg["stream"] / group / seg["segment_id"]


def run_segment(seg: dict, group: str, staged: dict, protocol: dict, windows_sha: str, *, allow_resume: bool) -> dict | None:
    key = f"{seg['segment_id']}|{group}"
    if state_of(key) == "compared" and (seg_dir(seg, group) / "summary.json").exists():
        return json.loads((seg_dir(seg, group) / "summary.json").read_text(encoding="utf-8"))
    free_gb = shutil.disk_usage(H.RUN_ROOT).free / 1e9
    if free_gb < MIN_FREE_GB:  # a transient full disk must stop the run, not fail every later segment
        log(f"BLOCKED: {free_gb:.1f} GB free < {MIN_FREE_GB} GB; stopping before {key}")
        raise SystemExit(3)
    loaded, records = staged[group]
    chunk = int(protocol["engine_composition"]["groups"][group]["chunk_endpoints"])
    directory = seg_dir(seg, group)
    set_state(key, status="running", segment=seg["segment_id"], group=group, stream=seg["stream"])
    log(f"SEGMENT {seg['segment_id']} {group} START ({len(seg['endpoints'])} endpoints, chunk {chunk})")
    started = time.perf_counter()
    prefix_bars, endpoint_bars = H.segment_bars(seg)
    run = H.run_app_segment(segment=seg, group=group, models=records, processor_factory=H.group_processor_factory(group, loaded),
                            seg_dir=directory, chunk_endpoints=chunk, prefix_bars=prefix_bars, endpoint_bars=endpoint_bars,
                            restart_each_chunk=seg.get("restart_each_chunk", False), allow_resume=allow_resume,
                            failure=(seg.get("failure") or {}).get(group), log=log)
    log(f"{seg['segment_id']} {group} app done in {time.perf_counter() - started:.0f} s; references")
    roster = H.roster_by_id()
    models = [roster[r["queue_id"]] for r in records]
    bars = None if seg["reference"] == "frozen" else reference_frame(seg, prefix_bars, endpoint_bars, chunk)
    for model in models:
        H.build_reference(model, seg, directory, mode=seg["reference"], bars=bars)
    setup_reference = None
    if seg["stream"] == "ES" and seg["source"] == "H":
        setup_reference = reference_setup_events(models[0], seg["endpoints"][0], seg["endpoints"][-1])
    freeze = H.freeze_entries()
    summary = compare_segment(directory, [model_spec(m, protocol) for m in models], protocol, windows_sha256=windows_sha,
                              expected_artifacts={m["model_id"]: freeze[m["queue_id"]] for m in models},
                              setup_reference=setup_reference)
    heads = {m["queue_id"]: summary["models"][m["model_id"]]["headline"] for m in models}
    set_state(key, status="compared", summary_sha256=file_hash(directory / "summary.json"),
              parity_db_sha256=run.get("parity_db_sha256"), seconds=round(time.perf_counter() - started),
              headline={q: dict(comparable=h["comparable"], endpoints=h["endpoints"], status=h["status"],
                                mismatches=h["mismatches"], flips=h["decision_flips"], failed_checks=h["failed_checks"])
                        for q, h in heads.items()})
    log(f"SEGMENT {seg['segment_id']} {group} COMPLETE in {time.perf_counter() - started:.0f} s: "
        + "; ".join(f"{q}:{h['status']}/{h['comparable']}of{h['endpoints']}" for q, h in heads.items()))
    return summary


def restart_gate(stream: str, segs: list[dict]) -> bool:
    """Bitwise app-vs-app equality of every restart segment with its continuous base, per group."""
    key = f"restart_gate|{stream}"
    results = {}
    for seg in (s for s in segs if s["stream"] == stream and "restart" in s["scenarios"]):
        base = next(s for s in segs if s["segment_id"] == seg["base_segment"])
        for group in GROUPS[stream]:
            result = compare_app_runs(seg_dir(base, group) / "app", seg_dir(seg, group) / "app")
            results[f"{seg['segment_id']}|{group}"] = dict(
                equal=result["equal"], base=base["segment_id"],
                differing={t: v["differing_columns"] for t, v in result["tables"].items() if not v["equal"]})
            log(f"restart gate {seg['segment_id']} {group} vs {base['segment_id']}: equal={result['equal']}")
    passed = bool(results) and all(r["equal"] for r in results.values())
    H.dump(H.RUN_ROOT / f"restart_gate_{stream}.json", dict(matrix_version=MATRIX_VERSION, passed=passed, results=results))
    set_state(key, status="passed" if passed else "failed", results={k: v["equal"] for k, v in results.items()})
    return passed


def composition(seg: dict, staged: dict, protocol: dict, segs: list[dict]) -> dict:
    """ES: each roster model alone in its own engine on the sampled endpoints vs the group engine (bitwise)."""
    from scripts.run_section4_r1_value_neutrality import staged_model

    out_path = H.RUN_ROOT / "composition_ES.json"
    base = next(s for s in segs if s["segment_id"] == seg["base_segment"])
    prefix_bars, endpoint_bars = H.segment_bars(seg)
    roster = H.roster_by_id()
    results = {}
    for group in GROUPS["ES"]:
        for record in staged[group][1]:
            key = f"{seg['segment_id']}|single_{record['queue_id']:02d}"
            directory = H.RUN_ROOT / "ES" / f"single_{record['queue_id']:02d}" / seg["segment_id"]
            set_state(key, status="running", segment=seg["segment_id"], group=f"single_{record['queue_id']:02d}", stream="ES")
            runtime_model, _ = staged_model(roster[record["queue_id"]])
            H.run_app_segment(segment=seg, group=f"single_{record['queue_id']:02d}", models=[record],
                              processor_factory=H.group_processor_factory(group, [runtime_model]), seg_dir=directory,
                              chunk_endpoints=len(endpoint_bars), prefix_bars=prefix_bars, endpoint_bars=endpoint_bars, log=log)
            single_rows = pd.read_parquet(next((directory / "app").glob("chunk_000_p0.rows.parquet")))
            index = json.loads(next((directory / "app").glob("chunk_000_p0.index.json")).read_text(encoding="utf-8"))
            entry = next(m for m in index["models"] if m["model_id"] == record["model_id"])
            features = {index["feature_names"][p] for p in entry["feature_positions"]}
            columns = [c for c in single_rows.columns if not c.startswith("f:") or c[2:] in features]
            result = compare_app_runs(seg_dir(base, group) / "app", directory / "app", models=[record["model_id"]],
                                      common_endpoints=True, row_columns=columns)
            results[record["queue_id"]] = dict(group=group, equal=result["equal"], endpoints=len(seg["endpoints"]),
                                               differing={t: v["differing_columns"] for t, v in result["tables"].items() if not v["equal"]})
            set_state(key, status="compared", equal=result["equal"])
            log(f"composition {record['queue_id']} alone vs {group}: equal={result['equal']}")
    payload = dict(matrix_version=MATRIX_VERSION, segment=seg["segment_id"], base=base["segment_id"], results=results,
                   passed=all(r["equal"] for r in results.values()))
    H.dump(out_path, payload)
    return payload


# ---------------------------------------------------------------- driver

def acquire_lock() -> None:
    if LOCK.exists():
        pid = int(json.loads(LOCK.read_text(encoding="utf-8"))["pid"])
        try:
            os.kill(pid, 0)
            alive = True
        except (OSError, SystemError):
            alive = False
        if alive:
            raise SystemExit(f"matrix already running (pid {pid}); refusing a second instance")
    LOCK.write_text(json.dumps(dict(pid=os.getpid(), started_at_utc=datetime.now(timezone.utc).isoformat())), encoding="utf-8")


def run(streams: list[str]) -> None:
    acquire_lock()
    try:
        protocol = H.load_protocol()
        segs, windows_sha = load_windows()
        log(f"MATRIX START {MATRIX_VERSION} windows {windows_sha[:12]} protocol {H.PROTOCOL_SHA256[:12]} streams {streams}")
        for stream in streams:
            staged = {}
            for group in GROUPS[stream]:
                staged[group] = H.staged_group(group, protocol)  # verify_bytes on every B4 v2 artifact before load
                set_state(f"bytes|{group}", status="verified", models={r["queue_id"]: r["artifact_sha256"] for r in staged[group][1]})
                log(f"B4 v2 bytes verified for {group}: {[r['queue_id'] for r in staged[group][1]]}")
            mine = [s for s in segs if s["stream"] == stream]
            anchor_seg = next(s for s in mine if s["segment_id"] == ANCHOR_SEGMENT[stream])
            roster = H.roster_by_id()
            for group in GROUPS[stream]:
                key = f"anchor|{group}"
                if state_of(key) != "compared":
                    set_state(key, status="running")
                    try:
                        result = anchor_check(group, anchor_seg, [roster[r["queue_id"]] for r in staged[group][1]], protocol)
                        set_state(key, status="compared", anchored={m["queue_id"]: m["anchored"] for m in result["models"].values()})
                    except Exception as exc:
                        set_state(key, status="failed", error=repr(exc)[:400])
                        log(f"ANCHOR {group} FAILED: {exc!r} " + f"{traceback.format_exc()[-1500:]}")
            restart_segs = [s for s in mine if "restart" in s["scenarios"]]
            bases = [s for s in mine if s["segment_id"] in {r["base_segment"] for r in restart_segs}]
            gate = state_of(f"restart_gate|{stream}") == "passed"
            for seg in [*bases, *restart_segs]:
                for group in GROUPS[stream]:
                    try:
                        run_segment(seg, group, staged, protocol, windows_sha, allow_resume=gate)
                    except Exception as exc:  # a failing branch stays diagnostic and does not block the others
                        set_state(f"{seg['segment_id']}|{group}", status="failed", error=repr(exc)[:400])
                        log(f"SEGMENT {seg['segment_id']} {group} FAILED: {exc!r}\n{traceback.format_exc()[-1500:]}")
            if state_of(f"restart_gate|{stream}") != "passed":
                try:
                    gate = restart_gate(stream, segs)
                except Exception as exc:
                    gate = False
                    set_state(f"restart_gate|{stream}", status="failed", error=repr(exc)[:400])
                    log(f"RESTART GATE {stream} FAILED: {exc!r}")
            log(f"restart gate {stream}: {'passed' if gate else 'FAILED (chunk resumes stay disallowed)'}")
            for seg in mine:
                if seg in bases or seg in restart_segs:
                    continue
                if "group_composition" in seg["scenarios"]:
                    if state_of(f"composition|{stream}") != "compared":
                        try:
                            result = composition(seg, staged, protocol, segs)
                            set_state(f"composition|{stream}", status="compared", passed=result["passed"])
                        except Exception as exc:
                            set_state(f"composition|{stream}", status="failed", error=repr(exc)[:400])
                            log(f"COMPOSITION FAILED: {exc!r}\n{traceback.format_exc()[-1500:]}")
                    continue
                for group in GROUPS[stream]:
                    try:
                        run_segment(seg, group, staged, protocol, windows_sha, allow_resume=gate)
                    except Exception as exc:
                        set_state(f"{seg['segment_id']}|{group}", status="failed", error=repr(exc)[:400])
                        log(f"SEGMENT {seg['segment_id']} {group} FAILED: {exc!r}\n{traceback.format_exc()[-1500:]}")
            set_state(f"stream|{stream}", status="complete")
            log(f"STREAM {stream} COMPLETE")
        if all(state_of(f"stream|{s}") == "complete" for s in ("ES", "EURUSD")):
            log("MATRIX COMPLETE")
    finally:
        LOCK.unlink(missing_ok=True)


def status() -> None:
    state = read_state()["items"]
    counts: dict[str, int] = {}
    for item in state.values():
        counts[item.get("status", "?")] = counts.get(item.get("status", "?"), 0) + 1
    print(json.dumps(counts))
    for key, item in state.items():
        if item.get("status") in {"running", "failed", "blocked"}:
            print(key, item.get("status"), item.get("error", "")[:200])


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "windows":
        segments = build_windows()
        digest = H.write_windows(WINDOWS, segments)
        print(f"windows.json {digest} segments {len(segments)}")
        for s in segments:
            print(s["segment_id"], s["source"], len(s["endpoints"]), s["endpoints"][0], s["endpoints"][-1], s.get("prefix_bars"))
    elif command == "run":
        run(sys.argv[2:] or ["ES", "EURUSD"])
    elif command == "status":
        status()
    else:
        print(__doc__)
        sys.exit(2)
