"""Section 4 R3 comparator (protocol v2): app vs reference on every stage and every consumed row.

Reads one segment written by ``scripts/section4_parity_harness.py`` and compares, per model and comparable
endpoint, every consumed row (XGB max(lag)+1, TCN window_size):

- S1 identity/provenance: row timestamp, source_row_idx (constant app-ref offset, consecutive), contract,
  completion, observation kind, feed type and bar version against the delivered truth rows, plus engine vs
  store provenance (stale provenance after revision/backfill).
- S3 raw pre-imputation values (per-feature tolerance class, frozen before any comparison) and explicit
  producer states (missing producer, fallback). Where the frozen reference stores post-builder-fill values
  (EURUSD ``fillna_numeric=True``), app NaN cells are counted ``not_compared_ref_post_fill`` and compared at S4.
- S4 imputed values (class of the raw cell) and imputation mask (exact); S5 scaled rows; S6 raw/calibrated
  scores; S7 threshold decision with the near-threshold band and flips.
- Checks: B4 v2 artifact hashes, missing snapshots/scores, recorder/process/tap errors, stale HTF helpers,
  persisted vs tapped scores.

Every out-of-tolerance cell goes to ``diff/<model_id>.<stage>.parquet``; only summaries (counts per stage and
feature, first offending timestamp, max abs/rel error, evidence flags) go to ``summary.json``. Mismatches are
left ``unclassified``; classification is a later step.

Also: ``compare_app_runs`` (restart / continuous-vs-chunked, bitwise), ``compare_setup_events`` (S2) and
``compare_source`` (S layer, D2).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_section3_smoke_replay import file_hash

COMPARATOR_VERSION = "s4-parity-comparator-v1"
INVENTORY = ROOT / "docs/audits/section4_feature_inventory_20261002.json"
CLASS_TOLERANCE = {"C0_exact": "E_exact", "C1_continuous": "C1_continuous", "C2_price_level": "C2_price_level",
                   "C3_volume_scale": "C3_volume_scale"}
S1_FIELDS = (("contract_symbol", "contract_symbol"), ("is_complete", "expected_is_complete"),
             ("observation_kind", "expected_observation_kind"), ("feed_type", "expected_feed_type"),
             ("bar_version", "expected_bar_version"))
ENGINE_STORE_FIELDS = ("contract_symbol", "is_complete", "observation_kind", "feed_type", "bar_version")
HELPERS = (("htf_30m", 30.0), ("htf_1h", 60.0))
EVIDENCE_COLUMNS = ("row_timestamp", "gap_state", "roll_boundary", "store_is_complete", "store_observation_kind",
                    "engine_htf_30m_source_age_minutes", "engine_htf_1h_source_age_minutes", "injected_row",
                    "injected_window", "injected_before_endpoint", "part", "instance", "after_restart", "warmup_ready")


# ---------------------------------------------------------------- tolerances

def tolerance(protocol: dict, name: str) -> tuple[float, float]:
    spec = protocol["tolerances"][name]
    return float(spec["atol"]), float(spec["rtol"])


def within(app, ref, atol: float, rtol: float) -> np.ndarray:
    """Protocol rule: |app-ref| <= atol + rtol*|ref|; NaN only matches NaN; inf only the same-sign inf."""
    a, r = np.asarray(app, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    a_nan, r_nan = np.isnan(a), np.isnan(r)
    a_inf, r_inf = np.isinf(a), np.isinf(r)
    with np.errstate(invalid="ignore"):
        close = np.abs(a - r) <= atol + rtol * np.abs(r)
    finite = ~(a_nan | r_nan | a_inf | r_inf)
    return np.where(a_nan | r_nan, a_nan & r_nan, np.where(a_inf | r_inf, a_inf & r_inf & (a == r), finite & close))


def errors(app, ref) -> tuple[np.ndarray, np.ndarray]:
    a, r = np.asarray(app, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        absolute = np.abs(a - r)
        relative = absolute / np.abs(r)
    return absolute, relative


def _same(a, b) -> np.ndarray:
    """Exact equality for identity fields (None/NaN equal only to None/NaN)."""
    left = pd.Series(a, dtype=object).map(_norm)
    right = pd.Series(b, dtype=object).map(_norm)
    return (left.to_numpy() == right.to_numpy())


def _field(name: str, values) -> np.ndarray:
    """Identity field values as plain objects; completion as True/False/None whatever the storage dtype."""
    values = np.asarray(values, dtype=object)
    if name != "is_complete":
        return values
    return np.asarray([None if _norm(v) == "<null>" else bool(float(v)) if not isinstance(v, str) else v == "True"
                       for v in values], dtype=object)


def _norm(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA or value is pd.NaT:
        return "<null>"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and float(value).is_integer()):
        return str(int(value)) if not isinstance(value, (bool, np.bool_)) else str(value)
    return str(value)


# ---------------------------------------------------------------- inputs

def load_inventory(protocol: dict) -> dict[int, dict]:
    pin = protocol["inputs"]["feature_inventory_tolerance_classes"]
    if file_hash(ROOT / pin["path"]) != pin["sha256"]:
        raise ValueError("Feature inventory does not match the protocol's frozen tolerance-class input")
    return {entry["queue_id"]: entry for entry in json.loads((ROOT / pin["path"]).read_text(encoding="utf-8"))["models"]}


def model_spec(model: dict, protocol: dict) -> dict:
    """Comparison spec for one roster model: frozen tolerance classes, fill stage, offsets, threshold."""
    from scripts.section4_parity_harness import dataset_metadata_for

    entry = load_inventory(protocol)[model["queue_id"]]
    pins = model["freeze"]["source_data_pins"]
    prepared = (ROOT / pins["prepared_manifest"]).parent / "prepared" / model["freeze"]["target"]
    contract = json.loads((prepared / "source_history_contract.json").read_text(encoding="utf-8"))
    config = dataset_metadata_for(contract)[1]["config"]
    return dict(model_id=model["model_id"], queue_id=model["queue_id"], stream=model["stream"],
                threshold=float(model["threshold"]),
                feature_classes={f["feature"]: f["tolerance_class"] for f in entry["features"]},
                producers={f["feature"]: f["producer"] for f in entry["features"]},
                reference_fill_stage="post_builder_fill" if config.get("fillna_numeric") else "pre_fill",
                source_row_offset=int(contract.get("source_row_offset") or 0))


def _concat(paths: Sequence[Path], **tags) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        for key, value in tags.items():
            frame[key] = value(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_app(app_dir: Path) -> dict:
    """Concatenate every chunk part (verified against its done marker)."""
    run = json.loads((app_dir / "run.json").read_text(encoding="utf-8"))
    markers = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(app_dir.glob("chunk_*.done"))]
    bad = [name for marker in markers for name, digest in marker["files"].items()
           if not (app_dir / name).exists() or file_hash(app_dir / name) != digest]
    if bad:
        raise ValueError(f"App chunk files changed after their done markers: {bad[:5]}")
    parts = [part for marker in markers for part in marker["parts"]]
    instance = {part["prefix"]: part["instance"] for part in parts}
    part_of = lambda path: path.name.split(".")[0]  # noqa: E731
    tables = {}
    for table in ("builds", "rows", "feature_states", "model_states", "model_feature_states", "scores"):
        tables[table] = _concat([app_dir / f"{p['prefix']}.{table}.parquet" for p in parts
                                 if (app_dir / f"{p['prefix']}.{table}.parquet").exists()], part=part_of)
    io = {}
    for path in sorted(app_dir.glob("chunk_*.io.*.parquet")):
        io.setdefault(path.name.split(".io.")[1][: -len(".parquet")], []).append(path)
    tables["io"] = {model: _concat(paths, part=part_of) for model, paths in io.items()}
    tables["index"] = [json.loads((app_dir / f"{p['prefix']}.index.json").read_text(encoding="utf-8")) for p in parts]
    tables["notes"] = [json.loads((app_dir / f"{p['prefix']}.notes.json").read_text(encoding="utf-8")) for p in parts]
    tables["truth"] = pd.read_parquet(app_dir / "truth_bars.parquet")
    tables["run"], tables["markers"], tables["instance"] = run, markers, instance
    first_parts = {marker["parts"][0]["prefix"] for marker in markers}
    resumed = {f"chunk_{r['chunk']:03d}_p0" for r in run.get("resumes", [])}
    tables["restarted_parts"] = {p["prefix"] for p in parts if p["prefix"] not in first_parts or p["prefix"] in resumed}
    return tables


# ---------------------------------------------------------------- per-model comparison

def compare_model(app: dict, ref_dir: Path, spec: dict, protocol: dict, diff_dir: Path,
                  expected_artifact: dict | None = None) -> dict:
    model_id, stream = spec["model_id"], spec["stream"]
    ref_run = json.loads((ref_dir / f"{model_id}.run.json").read_text(encoding="utf-8"))
    ref_io = pd.read_parquet(ref_dir / f"{model_id}.io.parquet")
    ref_scores = pd.read_parquet(ref_dir / f"{model_id}.scores.parquet")
    run = app["run"]
    model_record = next((m for m in run["models"] if m["model_id"] == model_id), None)
    if model_record is None:
        raise ValueError(f"{model_id} was not part of the app run")
    consumed = int(model_record["consumed_rows"])
    index = next(m for m in app["index"][0]["models"] if m["model_id"] == model_id)
    features = [app["index"][0]["feature_names"][p] for p in index["feature_positions"]]
    checks: dict[str, Any] = {}

    # Artifacts: the app and the reference must both have loaded the hash-verified B4 v2 bytes.
    checks["artifact_hash"] = _artifact_check(model_record, ref_run, expected_artifact)

    # Endpoint classification.
    builds = app["builds"].copy()
    builds["endpoint"] = builds.endpoint_timestamp.map(_iso)
    builds = builds.drop_duplicates("endpoint", keep="last").set_index("endpoint")
    states = app["model_states"]
    states = states.loc[states.model_id == model_id].set_index("build_id")
    scores = app["scores"]
    scores = scores.loc[scores.model_id == model_id].drop_duplicates("endpoint", keep="last").set_index("endpoint") if len(scores) else pd.DataFrame()
    ref_scores = ref_scores.drop_duplicates("endpoint", keep="last").set_index("endpoint") if len(ref_scores) else ref_scores
    truth = app["truth"].set_index("timestamp")
    app_io = app["io"].get(model_id, pd.DataFrame())
    ref_first_idx = ref_io.loc[ref_io.row_offset == -(consumed - 1)].set_index("endpoint").row_source_row_idx if len(ref_io) else pd.Series(dtype=float)
    status: dict[str, str] = {}
    for endpoint in run["segment"]["endpoints"]:
        build = builds.loc[endpoint] if endpoint in builds.index else None
        state = states.loc[build.build_id] if build is not None and build.build_id in states.index else None
        if endpoint in truth.index and not bool(truth.loc[endpoint, "delivered"]):
            status[endpoint] = "not_delivered"
        elif build is None:
            status[endpoint] = "missing_snapshot" if endpoint in scores.index else "not_evaluated"
        elif state is None or not bool(state.consumed_rows_captured):
            status[endpoint] = "missing_snapshot"
        elif not bool(state.warmup_ready):
            status[endpoint] = "warmup_not_ready"
        elif endpoint not in scores.index:
            status[endpoint] = "missing_app_score"
        elif endpoint not in ref_scores.index:
            status[endpoint] = "ref_missing"
        elif spec.get("source_row_offset") and ref_first_idx.get(endpoint, -1) < spec["source_row_offset"]:
            status[endpoint] = "below_source_row_offset"
        else:
            status[endpoint] = "comparable"
    counts = pd.Series(status).value_counts().to_dict()
    comparable = [e for e, s in status.items() if s == "comparable"]
    checks["missing_snapshot"] = [e for e, s in status.items() if s == "missing_snapshot"]
    checks["missing_app_score"] = [e for e, s in status.items() if s == "missing_app_score"]

    # Aligned consumed rows (positional: what the model consumes) with evidence.
    rows = app["rows"]
    build_ids = builds.loc[[e for e in comparable], "build_id"] if comparable else pd.Series(dtype=object)
    app_rows = rows.loc[rows.build_id.isin(build_ids.values) & (rows.row_offset > -consumed)].copy()
    app_rows["endpoint"] = app_rows.build_id.map({b: e for e, b in build_ids.items()})
    app_io = app_io.loc[app_io.endpoint.isin(comparable)].drop_duplicates(["endpoint", "row_offset"], keep="last")
    ref_rows = ref_io.loc[ref_io.endpoint.isin(comparable)].drop_duplicates(["endpoint", "row_offset"], keep="last")
    aligned = (app_rows.merge(app_io, on=["endpoint", "row_offset"], how="outer", suffixes=("", "_io"))
               .merge(ref_rows.add_prefix("ref_").rename(columns={"ref_endpoint": "endpoint", "ref_row_offset": "row_offset"}),
                      on=["endpoint", "row_offset"], how="outer"))
    short = aligned.groupby("endpoint").size()
    incomplete_rows = sorted(set(short.index[short < consumed]) | set(aligned.loc[aligned.build_id.isna() | aligned["ref_row_timestamp"].isna(), "endpoint"]))
    if incomplete_rows:
        checks["missing_snapshot"] = sorted(set(checks["missing_snapshot"]) | set(aligned.loc[aligned.build_id.isna(), "endpoint"]))
        for endpoint in incomplete_rows:
            status[endpoint] = "incomplete_history"
        aligned = aligned.loc[~aligned.endpoint.isin(incomplete_rows)]
        comparable = [e for e in comparable if e not in incomplete_rows]
        counts = pd.Series(status).value_counts().to_dict()
    aligned = aligned.sort_values(["endpoint", "row_offset"]).reset_index(drop=True)
    evidence = _evidence(aligned, truth, app, states, consumed)

    stage_rows: dict[str, list[pd.DataFrame]] = {}
    stats: dict[str, dict] = {}

    def record(stage: str, feature: str, mask: np.ndarray, app_values, ref_values, cls: str | None = None,
               compared: int | None = None, extra: dict | None = None) -> None:
        mask = np.asarray(mask, dtype=bool)
        entry = stats.setdefault(stage, dict(compared=0, matched=0, mismatched=0, not_compared=0, max_abs=0.0,
                                             max_rel=0.0, first_offending_ts=None))
        total = len(mask) if compared is None else compared
        entry["compared"] += int(total)
        entry["mismatched"] += int(mask.sum())
        entry["matched"] += int(total - mask.sum())
        if not mask.any():
            return
        frame = evidence.loc[mask].copy()
        frame.insert(0, "feature", feature)
        frame.insert(0, "stage", stage)
        frame["app"] = pd.Series(app_values, dtype=object).loc[mask].map(_plain).to_numpy() if not np.isscalar(app_values) else app_values
        frame["ref"] = pd.Series(ref_values, dtype=object).loc[mask].map(_plain).to_numpy() if not np.isscalar(ref_values) else ref_values
        numeric = pd.to_numeric(frame.app, errors="coerce"), pd.to_numeric(frame.ref, errors="coerce")
        frame["abs_err"], frame["rel_err"] = errors(*numeric)
        frame["tolerance_class"] = cls
        for key, value in (extra or {}).items():
            frame[key] = np.asarray(value)[mask] if not np.isscalar(value) else value
        stage_rows.setdefault(stage, []).append(frame)
        finite_abs, finite_rel = frame.abs_err[np.isfinite(frame.abs_err)], frame.rel_err[np.isfinite(frame.rel_err)]
        entry["max_abs"] = max(entry["max_abs"], float(finite_abs.max()) if len(finite_abs) else 0.0)
        entry["max_rel"] = max(entry["max_rel"], float(finite_rel.max()) if len(finite_rel) else 0.0)
        first = frame.endpoint.min()
        entry["first_offending_ts"] = first if entry["first_offending_ts"] is None else min(entry["first_offending_ts"], first)

    if len(aligned):
        # S1 identity and provenance.
        record("S1", "timestamp", ~_same(aligned.timestamp, aligned.ref_row_timestamp), aligned.timestamp, aligned.ref_row_timestamp)
        app_idx = pd.to_numeric(aligned.store_source_row_idx, errors="coerce")
        ref_idx = pd.to_numeric(aligned.ref_row_source_row_idx, errors="coerce")
        offset = (app_idx - ref_idx).dropna()  # anchored at the segment's first comparable row (aligned is sorted)
        base = float(offset.iloc[0]) if len(offset) else np.nan
        record("S1", "source_row_idx_offset", ~within(app_idx - ref_idx, np.full(len(aligned), base), 0.0, 0.0),
               app_idx, ref_idx, extra=dict(expected_offset=base))
        steps = aligned.groupby("endpoint")["store_source_row_idx"].transform(lambda s: pd.to_numeric(s).diff())
        record("S1", "source_row_idx_consecutive", steps.notna().to_numpy() & (steps.to_numpy() != 1), steps, 1)
        expected = truth.reindex(aligned.ref_row_timestamp.map(_iso))
        for app_field, truth_field in S1_FIELDS:
            app_values, truth_values = _field(app_field, aligned[f"store_{app_field}"]), _field(app_field, expected[truth_field])
            record("S1", app_field, ~_same(app_values, truth_values), app_values, truth_values)
        for name in ENGINE_STORE_FIELDS:
            if f"engine_{name}" in aligned:
                engine, store = _field(name, aligned[f"engine_{name}"]), _field(name, aligned[f"store_{name}"])
                record("S1", f"engine_vs_store:{name}", ~_same(engine, store), engine, store)

        # S3 raw values, S4 imputed + mask, S5 scaled.
        for feature in features:
            cls = spec["feature_classes"].get(feature)
            if cls is None:
                raise ValueError(f"{model_id}: no frozen tolerance class for {feature}")
            atol, rtol = tolerance(protocol, CLASS_TOLERANCE[cls])
            ref_column = f"ref_raw:{feature}" if f"ref_raw:{feature}" in aligned else f"ref_in:{feature}"
            raw, ref_raw = aligned[f"f:{feature}"].to_numpy(float), aligned[ref_column].to_numpy(float)
            skip = np.isnan(raw) if spec["reference_fill_stage"] == "post_builder_fill" else np.zeros(len(raw), bool)
            bad = ~within(raw, ref_raw, atol, rtol) & ~skip
            stats.setdefault("S3", dict(compared=0, matched=0, mismatched=0, not_compared=0, max_abs=0.0, max_rel=0.0,
                                        first_offending_ts=None))["not_compared"] += int(skip.sum())
            record("S3", feature, bad, raw, ref_raw, cls, compared=int((~skip).sum()))
            app_in, ref_in = aligned[f"in:{feature}"].to_numpy(float), aligned[f"ref_in:{feature}"].to_numpy(float)
            prefill = np.isnan(raw) & np.isfinite(app_in)
            record("S4_mask", feature, (~np.isfinite(app_in)) != (~np.isfinite(ref_in)), ~np.isfinite(app_in),
                   ~np.isfinite(ref_in), "E", extra=dict(app_prefill=prefill))
            record("S4", feature, ~within(aligned[f"imp:{feature}"], aligned[f"ref_imp:{feature}"], atol, rtol),
                   aligned[f"imp:{feature}"], aligned[f"ref_imp:{feature}"], cls, extra=dict(app_prefill=prefill))
            satol, srtol = tolerance(protocol, "S5_scaled")
            record("S5", feature, ~within(aligned[f"sc:{feature}"], aligned[f"ref_sc:{feature}"], satol, srtol),
                   aligned[f"sc:{feature}"], aligned[f"ref_sc:{feature}"], "S5", extra=dict(app_prefill=prefill))

    # S3 states: explicit producer/fallback state for this model's features at comparable endpoints.
    feature_states = app["feature_states"]
    fs = feature_states.loc[feature_states.build_id.isin(build_ids.values) & feature_states.feature.isin(features)]
    bad_state = fs.loc[fs.group_state != "produced"]
    stats["S3_state"] = dict(compared=int(len(fs)), matched=int(len(fs) - len(bad_state)), mismatched=int(len(bad_state)),
                             not_compared=0, max_abs=None, max_rel=None,
                             first_offending_ts=None if bad_state.empty else _first_endpoint(bad_state.build_id, build_ids))
    if len(bad_state):
        frame = bad_state.assign(stage="S3_state", endpoint=bad_state.build_id.map({b: e for e, b in build_ids.items()}),
                                 app=bad_state.group_state, ref="produced")
        stage_rows.setdefault("S3_state", []).append(frame)
    checks["fallback"] = int((bad_state.group_state == "fallback_zero_default").sum())
    checks["missing_producer"] = int((bad_state.group_state == "missing_producer").sum())

    # S6 scores and S7 decisions on comparable endpoints.
    near, flips = [], []
    if comparable:
        app_s, ref_s = scores.loc[comparable], ref_scores.loc[comparable]
        ev = pd.DataFrame({"endpoint": comparable}).merge(
            evidence.loc[evidence.row_offset == 0] if len(evidence) else pd.DataFrame(columns=["endpoint"]), on="endpoint", how="left")
        atol, rtol = tolerance(protocol, "S6_scores")
        for column in ("raw_score", "calibrated_probability"):
            bad = ~within(app_s[column].to_numpy(float), ref_s[column].to_numpy(float), atol, rtol)
            _record_endpoint(stage_rows, stats, "S6", column, bad, comparable, app_s[column], ref_s[column], ev)
        threshold = spec["threshold"]
        band = 1e-6 + 1e-5 * threshold
        in_band = ((ref_s.calibrated_probability - threshold).abs() <= band) | ((app_s.calibrated_probability - threshold).abs() <= band)
        app_decision = app_s.calibrated_probability.to_numpy(float) >= threshold
        ref_decision = ref_s.calibrated_probability.to_numpy(float) >= threshold
        flip = app_decision != ref_decision
        near = [dict(endpoint=e, app_calibrated=float(app_s.loc[e, "calibrated_probability"]),
                     ref_calibrated=float(ref_s.loc[e, "calibrated_probability"]), app_decision=bool(a), ref_decision=bool(r))
                for e, a, r, b in zip(comparable, app_decision, ref_decision, in_band) if b]
        flips = [dict(endpoint=e, app_calibrated=float(app_s.loc[e, "calibrated_probability"]),
                      ref_calibrated=float(ref_s.loc[e, "calibrated_probability"]), app_decision=bool(a), ref_decision=bool(r))
                 for e, a, r, f, b in zip(comparable, app_decision, ref_decision, flip, in_band) if f and not b]
        compared7 = ~in_band.to_numpy()
        _record_endpoint(stage_rows, stats, "S7", "threshold_decision", flip & compared7, comparable, app_decision,
                         ref_decision, ev, compared=int(compared7.sum()))
        stats["S7"]["not_compared"] = int(in_band.sum())
        persisted = app_s[["persisted_raw_score", "persisted_calibrated_probability"]].to_numpy(float)
        tapped = app_s[["raw_score", "calibrated_probability"]].to_numpy(float)
        checks["persisted_vs_tapped_mismatch"] = int((~np.isclose(persisted, tapped, rtol=0, atol=0, equal_nan=True)).any(axis=1).sum())

    checks["stale_helper"] = _stale_helpers(rows.loc[rows.build_id.isin(builds.build_id)])
    checks["recorder_errors"] = sum(len(i.get("errors") or []) for i in app["index"])
    checks["process_errors"] = sum(len(n["process_errors"]) for n in app["notes"])
    checks["tap_errors"] = sum(len(n["tap_errors"]) for n in app["notes"] if any(e["model_id"] == model_id for e in n["tap_errors"]))
    checks["warmup_not_ready"] = int(counts.get("warmup_not_ready", 0))
    checks["not_evaluated"] = [e for e, s in status.items() if s == "not_evaluated"]
    health = [h for n in app["notes"] for h in n.get("health_events", [])]
    checks["health_errors"] = sum(1 for h in health if h["severity"] == "error")
    checks["health_events"] = {f"{h['component']}:{h['event_type']}:{h['severity']}": 0 for h in health}
    for h in health:
        checks["health_events"][f"{h['component']}:{h['event_type']}:{h['severity']}"] += 1

    # Write diffs and summarize.
    diff_dir.mkdir(parents=True, exist_ok=True)
    groups, feature_counts, diff_files = [], {}, {}
    producers = {}
    if len(fs):
        producers = fs.drop_duplicates("feature").set_index("feature")[["producer", "producer_module"]].to_dict("index")
    for stage, frames in stage_rows.items():
        frame = pd.concat(frames, ignore_index=True)
        path = diff_dir / f"{model_id}.{stage}.parquet"
        frame.astype({c: str for c in frame.columns if frame[c].dtype == object}).to_parquet(path, index=False)
        diff_files[path.name] = file_hash(path)
        for feature, group in frame.groupby("feature", sort=False):
            first = group.sort_values(["endpoint", "row_offset"] if "row_offset" in group else ["endpoint"]).iloc[0]
            abs_err = pd.to_numeric(group.get("abs_err"), errors="coerce") if "abs_err" in group else pd.Series(dtype=float)
            rel_err = pd.to_numeric(group.get("rel_err"), errors="coerce") if "rel_err" in group else pd.Series(dtype=float)
            groups.append(dict(
                stage=stage, feature=feature, count=int(len(group)),
                producer=producers.get(feature, {}).get("producer") or spec.get("producers", {}).get(feature),
                producer_module=producers.get(feature, {}).get("producer_module"),
                tolerance_class=spec["feature_classes"].get(feature), first_endpoint=first.endpoint,
                first_row_timestamp=_plain(first.get("row_timestamp")),
                max_abs=_finite_max(abs_err), max_rel=_finite_max(rel_err),
                evidence_at_first={k: _plain(first.get(k)) for k in (*EVIDENCE_COLUMNS, "app", "ref", "app_prefill",
                                                                    "group_state", "contract_category") if k in first.index},
                classification="unclassified"))
            if stage in {"S3", "S4", "S4_mask", "S5", "S3_state"}:
                feature_counts.setdefault(feature, dict(mismatches=0, tolerance_class=spec["feature_classes"].get(feature)))
                feature_counts[feature]["mismatches"] += int(len(group))
    failed_checks = sorted(k for k, v in checks.items()
                           if (k == "artifact_hash" and not v["ok"])
                           or (k not in {"artifact_hash", "warmup_not_ready", "health_events"} and v))
    headline = dict(model_id=model_id, endpoints=len(status), comparable=len(comparable),
                    mismatches={s: v["mismatched"] for s, v in stats.items() if v["mismatched"]},
                    near_threshold=len(near), decision_flips=len(flips), failed_checks=failed_checks,
                    status="mismatches_unclassified" if groups or failed_checks
                    else ("incomplete" if len(comparable) < len(status) else "no_mismatch"))
    return dict(comparator_version=COMPARATOR_VERSION, model_id=model_id, queue_id=spec.get("queue_id"), stream=stream,
                consumed_rows=consumed, reference=dict(mode=ref_run["mode"], source_history_sha256=ref_run.get("source_history_sha256"),
                                                       fill_stage=spec["reference_fill_stage"]),
                endpoint_status=counts, stages=stats, features=feature_counts, mismatch_groups=groups,
                near_threshold=near, decision_flips=flips, checks=checks, diff_files=diff_files, headline=headline)


def _artifact_check(model_record: dict, ref_run: dict, expected: dict | None) -> dict:
    result = dict(ok=True, reasons=[])
    app_files, ref_files = model_record.get("artifact_sha256") or {}, (ref_run.get("verified") or {}).get("b4_freeze_files_verified") or {}
    if expected is not None:
        frozen = {v for v in expected["artifacts"]["sha256"].values() if v}
        set_hash = expected["artifacts"]["reviewed_artifact_set_sha256"]
        for side, files, record in (("app", app_files, model_record), ("ref", ref_files, ref_run.get("verified") or {})):
            if set(files.values()) != frozen:
                result["reasons"].append(f"{side}_artifact_sha256_not_b4_v2")
            if record.get("reviewed_artifact_set_sha256") != set_hash:
                result["reasons"].append(f"{side}_reviewed_artifact_set_not_b4_v2")
    elif app_files != ref_files:
        result["reasons"].append("app_ref_artifacts_differ")
    result["ok"] = not result["reasons"]
    return result


def _evidence(aligned: pd.DataFrame, truth: pd.DataFrame, app: dict, states: pd.DataFrame, consumed: int) -> pd.DataFrame:
    frame = pd.DataFrame({"endpoint": aligned.endpoint, "row_offset": aligned.row_offset})
    frame["row_timestamp"] = aligned.get("ref_row_timestamp", aligned.get("timestamp"))
    for column in ("gap_state", "roll_boundary", "store_is_complete", "store_observation_kind",
                   "engine_htf_30m_source_age_minutes", "engine_htf_1h_source_age_minutes", "part"):
        frame[column] = aligned[column].to_numpy() if column in aligned else None
    injected = truth.injected.dropna()
    frame["injected_row"] = frame.row_timestamp.map(lambda ts: injected.get(_iso(ts)) if isinstance(ts, str) else None)
    window_hit = frame.assign(hit=frame.injected_row.notna()).groupby("endpoint").hit.transform("any")
    frame["injected_window"] = window_hit.to_numpy()
    first_injection = min((_iso(ts) for ts in injected.index), default=None)
    frame["injected_before_endpoint"] = (frame.endpoint >= first_injection).to_numpy() if first_injection else False
    frame["instance"] = frame.part.map(app["instance"]) if "part" in frame else None
    frame["after_restart"] = frame.part.isin(app["restarted_parts"]) if "part" in frame else False
    frame["warmup_ready"] = aligned.build_id.map(states.warmup_ready).to_numpy() if "build_id" in aligned else None
    return frame.reset_index(drop=True)


def _record_endpoint(stage_rows, stats, stage, feature, mask, endpoints, app_values, ref_values, evidence, compared=None):
    mask = np.asarray(mask, dtype=bool)
    entry = stats.setdefault(stage, dict(compared=0, matched=0, mismatched=0, not_compared=0, max_abs=0.0, max_rel=0.0,
                                         first_offending_ts=None))
    total = len(mask) if compared is None else compared
    entry["compared"] += total
    entry["mismatched"] += int(mask.sum())
    entry["matched"] += int(total - mask.sum())
    if not mask.any():
        return
    frame = evidence.loc[mask].copy()
    frame["stage"], frame["feature"] = stage, feature
    frame["app"] = np.asarray(app_values, dtype=object)[mask]
    frame["ref"] = np.asarray(ref_values, dtype=object)[mask]
    frame["abs_err"], frame["rel_err"] = errors(pd.to_numeric(frame.app, errors="coerce"), pd.to_numeric(frame.ref, errors="coerce"))
    stage_rows.setdefault(stage, []).append(frame)
    entry["max_abs"] = max(entry["max_abs"], _finite_max(frame.abs_err) or 0.0)
    entry["max_rel"] = max(entry["max_rel"], _finite_max(frame.rel_err) or 0.0)
    first = frame.endpoint.min()
    entry["first_offending_ts"] = first if entry["first_offending_ts"] is None else min(entry["first_offending_ts"], first)


def _stale_helpers(rows: pd.DataFrame) -> int:
    """HTF helper older than its timeframe although the preceding timeframe's rows were contiguous."""
    if rows.empty:
        return 0
    stale = np.zeros(len(rows), dtype=bool)
    ordered = rows.sort_values(["build_id", "row_offset"])
    for prefix, minutes in HELPERS:
        age_column = f"engine_{prefix}_source_age_minutes"
        if age_column not in ordered:
            continue
        span = int(minutes // 5)
        gaps = (ordered.gap_state.ne("none") & ordered.gap_state.ne("window_start")).astype(float)
        recent_gap = gaps.groupby(ordered.build_id).transform(lambda s: s.rolling(span, min_periods=1).max()).astype(bool)
        age = pd.to_numeric(ordered[age_column], errors="coerce")
        complete_column = f"engine_{prefix}_source_complete"
        incomplete = pd.to_numeric(ordered[complete_column], errors="coerce").eq(0) if complete_column in ordered else False
        stale |= (((age >= minutes) & ~recent_gap) | incomplete).to_numpy()
    return int(stale.sum())


def _first_endpoint(build_column: pd.Series, build_ids: pd.Series) -> str | None:
    lookup = {b: e for e, b in build_ids.items()}
    values = [lookup[b] for b in build_column if b in lookup]
    return min(values) if values else None


def _finite_max(values) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series[np.isfinite(series)]
    return float(series.max()) if len(series) else None


def _plain(value):
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _iso(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    stamp = pd.Timestamp(value)
    return (stamp.tz_convert("UTC") if stamp.tzinfo else stamp.tz_localize("UTC")).isoformat()


# ---------------------------------------------------------------- segment

def compare_segment(seg_dir: Path, specs: Sequence[dict], protocol: dict, *, windows_sha256: str | None = None,
                    expected_artifacts: dict[str, dict] | None = None, setup_reference: pd.DataFrame | None = None,
                    write: bool = True) -> dict:
    """Compare every requested model of one segment; write diff/ files and ``summary.json``."""
    seg_dir = Path(seg_dir)
    app = load_app(seg_dir / "app")
    models = {spec["model_id"]: compare_model(app, seg_dir / "ref", spec, protocol, seg_dir / "diff",
                                              (expected_artifacts or {}).get(spec["model_id"]))
              for spec in specs}
    run = app["run"]
    summary = dict(comparator_version=COMPARATOR_VERSION, segment=run["segment"]["segment_id"], stream=run["stream"],
                   group=run["group"], windows_sha256=windows_sha256, protocol=run["protocol"],
                   parity_db_sha256=run.get("parity_db_sha256"), code_sha256=run["code_sha256"],
                   eurusd_calendar=(dict(id=run.get("calendar_id"), status="provisional") if run["stream"] == "EURUSD" else None),
                   injections=len(run.get("injections") or []), resumes=run.get("resumes"),
                   S2=segment_setup_parity(seg_dir, run, protocol, setup_reference), models=models)
    if write:
        (seg_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------- app vs app (restart, continuous vs chunked)

def compare_app_runs(dir_a: Path, dir_b: Path, *, models: Sequence[str] | None = None,
                     common_endpoints: bool = False, row_columns: Sequence[str] | None = None) -> dict:
    """Bitwise app-vs-app equality on S1-S7 content (protocol restart_vs_continuous, atol = rtol = 0).

    Defaults compare everything. The composition check (engine_composition.composition_invariance) restricts
    to ``models``, to the endpoints both runs evaluated (``common_endpoints``) and to ``row_columns`` (the
    model's selected ``f:`` columns plus the bar/provenance fields both runs record).
    """
    a, b = load_app(Path(dir_a)), load_app(Path(dir_b))
    result: dict[str, Any] = {}

    def keyed_rows(app):
        endpoint = app["builds"].assign(endpoint=app["builds"].endpoint_timestamp.map(_iso)).set_index("build_id").endpoint
        rows = app["rows"].assign(endpoint=app["rows"].build_id.map(endpoint))
        return rows.drop_duplicates(["endpoint", "row_offset"], keep="last").set_index(["endpoint", "row_offset"]).sort_index()

    ignore = {"build_id", "part", "generation_id"}
    tables = {"rows": (keyed_rows(a), keyed_rows(b))}
    for model in sorted(set(a["io"]) | set(b["io"])):
        tables[f"io:{model}"] = tuple(x["io"].get(model, pd.DataFrame(columns=["endpoint", "row_offset"]))
                                      .drop_duplicates(["endpoint", "row_offset"], keep="last")
                                      .set_index(["endpoint", "row_offset"]).sort_index() for x in (a, b))
    tables["scores"] = tuple(x["scores"].drop_duplicates(["model_id", "endpoint"], keep="last")
                             .set_index(["model_id", "endpoint"]).sort_index() for x in (a, b))
    if models is not None:
        tables = {name: pair for name, pair in tables.items() if name == "rows" or name.split(":", 1)[-1] in set(models)
                  or name == "scores"}
        tables["scores"] = tuple(t.loc[t.index.get_level_values("model_id").isin(list(models))] for t in tables["scores"])
    if row_columns is not None:
        tables["rows"] = tuple(t[[c for c in row_columns if c in t.columns]] for t in tables["rows"])
    for name, (left, right) in tables.items():
        columns = sorted((set(left.columns) | set(right.columns)) - ignore - {"instance", "persisted_raw_score",
                                                                                  "persisted_calibrated_probability"})
        if common_endpoints:
            level = "endpoint"
            shared = set(left.index.get_level_values(level)) & set(right.index.get_level_values(level))
            left = left.loc[left.index.get_level_values(level).isin(shared)]
            right = right.loc[right.index.get_level_values(level).isin(shared)]
        keys = left.index.union(right.index)
        left, right = left.reindex(keys), right.reindex(keys)
        differing = {}
        for column in columns:
            x = left[column] if column in left else pd.Series(None, index=keys, dtype=object)
            y = right[column] if column in right else pd.Series(None, index=keys, dtype=object)
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                same = np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=0, atol=0, equal_nan=True)
            else:
                same = _same(x.to_numpy(object), y.to_numpy(object))
            if not same.all():
                first = keys[~same][0]
                differing[column] = dict(cells=int((~same).sum()), first_key=[str(k) for k in (first if isinstance(first, tuple) else (first,))])
        result[name] = dict(keys=len(keys), differing_columns=differing, equal=not differing)
    return dict(equal=all(v["equal"] for v in result.values()), tables=result)


# ---------------------------------------------------------------- S2 setup events

def app_setup_events(events: pd.DataFrame) -> pd.DataFrame:
    """First observation of each app event (causal), normalized to (strategy, side, source ts, type)."""
    if events.empty:
        return pd.DataFrame(columns=["strategy", "side", "source_ts", "setup_type", "family", "rule_confidence",
                                     "detector_identity", "detector_version", "geometry", "revisions"])
    revisions = events.groupby("event_key").revision.max()
    first = events.sort_values(["event_key", "revision"]).drop_duplicates("event_key")
    return pd.DataFrame({
        "strategy": first.strategy.str.upper(), "side": first.setup_side.astype(int),
        "source_ts": first.source_timestamp_utc.map(_iso), "setup_type": first.setup_type.map(_setup_type),
        "family": first.setup_family, "rule_confidence": pd.to_numeric(first.rule_confidence, errors="coerce"),
        "detector_identity": first.detector_identity, "detector_version": first.detector_version,
        "geometry": first.geometry_json.map(lambda g: json.loads(g) if isinstance(g, str) and g else {}),
        "revisions": first.event_key.map(revisions).to_numpy()}).reset_index(drop=True)


def _setup_type(value) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def compare_setup_events(app_events: pd.DataFrame, ref_events: pd.DataFrame, protocol: dict,
                         diff_path: Path | None = None) -> dict:
    """S2: categorical fields and timestamps exact; rule confidence and geometry within tolerance.

    ``ref_events`` columns: strategy, side, source_ts, setup_type and optionally family, rule_confidence and
    geometry keys as ``geo:<key>`` (C2_price_level)."""
    key = ["strategy", "side", "source_ts", "setup_type"]
    app = app_setup_events(app_events) if "event_key" in app_events else app_events
    ref = ref_events.assign(setup_type=ref_events.setup_type.map(_setup_type), source_ts=ref_events.source_ts.map(_iso))
    merged = app.merge(ref, on=key, how="outer", suffixes=("", "_ref"), indicator=True)
    out = dict(app_events=int(len(app)), ref_events=int(len(ref)), matched=int((merged._merge == "both").sum()),
               only_app=int((merged._merge == "left_only").sum()), only_ref=int((merged._merge == "right_only").sum()),
               mismatches={})
    both = merged.loc[merged._merge == "both"]
    if "family_ref" in both and len(both):
        out["mismatches"]["family"] = int((~_same(both.family, both.family_ref)).sum())
    if "rule_confidence_ref" in both and len(both):
        atol, rtol = tolerance(protocol, "S2_rule_confidence")
        out["mismatches"]["rule_confidence"] = int((~within(both.rule_confidence, both.rule_confidence_ref, atol, rtol)).sum())
    atol, rtol = tolerance(protocol, "C2_price_level")
    for column in [c for c in ref.columns if c.startswith("geo:")]:
        name = column[4:]
        app_values = both.geometry.map(lambda g: (g or {}).get(name, np.nan)) if len(both) else pd.Series(dtype=float)
        out["mismatches"][column] = int((~within(pd.to_numeric(app_values, errors="coerce"), both[column], atol, rtol)).sum())
    first = merged.loc[merged._merge != "both"].source_ts.min() if (merged._merge != "both").any() else None
    out["first_unmatched_ts"] = first
    out["revised_app_events"] = int((app.revisions > 0).sum()) if "revisions" in app else 0
    if diff_path is not None:
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        merged.loc[merged._merge != "both"].astype(str).to_parquet(diff_path, index=False)
    return out


def reference_setup_events(model: dict, start: str, end: str) -> pd.DataFrame:
    """Offline detector events section 3 used (hash-verified), for one ES model's strategy and a time window.

    FRVP: the prepared source dataset's detector columns (frvp_setup_type > 0, frvp_setup_side);
    ICT: the corrected fired-event file (setup type/side, event time, stop/target references)."""
    from scripts import run_b2_retraining as old

    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if model["binding"].strategy == "frvp":
        prepared = json.loads((ROOT / model["freeze"]["source_data_pins"]["prepared_manifest"]).read_text(encoding="utf-8"))
        dataset = ROOT / Path(prepared["source_dataset"].replace("\\", "/"))
        if file_hash(dataset) != prepared["source_dataset_sha256"]:
            raise ValueError("FRVP source dataset does not match the prepared manifest")
        rows = pd.read_parquet(dataset, columns=["datetime", "frvp_setup_type", "frvp_setup_side"])
        rows["datetime"] = pd.to_datetime(rows.datetime, utc=True)
        rows = rows.loc[rows.datetime.between(lo, hi) & rows.frvp_setup_type.gt(0)]
        return pd.DataFrame({"strategy": "FRVP", "side": rows.frvp_setup_side.astype(int).to_numpy(),
                             "source_ts": rows.datetime.map(_iso).to_numpy(), "setup_type": rows.frvp_setup_type.to_numpy()})
    old.require_hash(old.EVENTS, old.EVENTS_SHA256)
    events = pd.read_csv(old.EVENTS, usecols=["setup_type", "setup_side", "event_time", "stop_reference", "target_reference"])
    events["event_time"] = pd.to_datetime(events.event_time, utc=True)
    events = events.loc[events.event_time.between(lo, hi)]
    return pd.DataFrame({"strategy": "ICT", "side": events.setup_side.astype(int).to_numpy(),
                         "source_ts": events.event_time.map(_iso).to_numpy(), "setup_type": events.setup_type.to_numpy(),
                         "geo:stop_reference": events.stop_reference.to_numpy(),
                         "geo:target_reference": events.target_reference.to_numpy()})


def segment_setup_parity(seg_dir: Path, run: dict, protocol: dict, reference: pd.DataFrame | None) -> dict:
    """S2 for one segment: ES app events vs the offline detector events; EURUSD incomplete until D5."""
    if run["stream"] != "ES":
        return dict(status="incomplete", reason="no live OTE setup-event producer (protocol setup_events / D5)")
    if reference is None:
        return dict(status="not_run", reason="no reference events supplied")
    app_events = pd.read_parquet(seg_dir / "app" / "setup_events.parquet")
    strategies = set(reference.strategy) if len(reference) else set()
    app_events = app_events.loc[app_events.strategy.str.upper().isin(strategies)] if strategies else app_events
    endpoints = run["segment"]["endpoints"]
    app_events = app_events.loc[app_events.source_timestamp_utc.map(_iso).between(endpoints[0], endpoints[-1])]
    result = compare_setup_events(app_events, reference, protocol, diff_path=seg_dir / "diff" / "S2.unmatched.parquet")
    result["status"] = "compared_unclassified"
    return result


# ---------------------------------------------------------------- S source parity (D2)

def compare_source(app_bars: pd.DataFrame, training: pd.DataFrame, protocol: dict, *, stream: str,
                   diff_path: Path | None = None) -> dict:
    """S layer: identity, timestamps and completion exact; price (C2) and volume (C3) reported with tolerance.

    Both frames need timestamp, open, high, low, close, volume, contract_symbol; ``app_bars`` also is_complete.
    Only the overlapping time range is compared."""
    a = app_bars.assign(timestamp=app_bars.timestamp.map(_iso)).set_index("timestamp")
    t = training.assign(timestamp=training.timestamp.map(_iso)).set_index("timestamp")
    lo, hi = max(a.index.min(), t.index.min()), min(a.index.max(), t.index.max())
    a, t = a.loc[(a.index >= lo) & (a.index <= hi)], t.loc[(t.index >= lo) & (t.index <= hi)]
    both = a.index.intersection(t.index)
    out = dict(stream=stream, overlap_start=lo, overlap_end=hi, app_bars=int(len(a)), training_bars=int(len(t)),
               matched_timestamps=int(len(both)), only_app=sorted(a.index.difference(t.index))[:20],
               only_training=sorted(t.index.difference(a.index))[:20],
               only_app_count=int(len(a.index.difference(t.index))), only_training_count=int(len(t.index.difference(a.index))))
    x, y = a.loc[both], t.loc[both]
    rows = []
    contract_bad = ~_same(x.contract_symbol, y.contract_symbol) if stream == "ES" else np.zeros(len(both), bool)
    complete = x.is_complete.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v))
    out["contract_mismatches"] = int(contract_bad.sum())
    out["completion_unknown"] = int(complete.isna().sum())
    out["completion_false"] = int((complete == False).sum())  # noqa: E712
    out["prices"] = {}
    for column, name in (("open", "C2_price_level"), ("high", "C2_price_level"), ("low", "C2_price_level"),
                         ("close", "C2_price_level"), ("volume", "C3_volume_scale")):
        atol, rtol = tolerance(protocol, name)
        bad = ~within(x[column], y[column], atol, rtol)
        abs_err, rel_err = errors(x[column], y[column])
        out["prices"][column] = dict(out_of_tolerance=int(bad.sum()), max_abs=_finite_max(abs_err), max_rel=_finite_max(rel_err),
                                     first_ts=str(both[bad][0]) if bad.any() else None)
        if bad.any():
            rows.append(pd.DataFrame({"timestamp": both[bad], "field": column, "app": x[column][bad].to_numpy(),
                                      "training": y[column][bad].to_numpy(), "abs_err": abs_err[bad]}))
    if contract_bad.any():
        rows.append(pd.DataFrame({"timestamp": both[contract_bad], "field": "contract_symbol",
                                  "app": x.contract_symbol[contract_bad].astype(str).to_numpy(),
                                  "training": y.contract_symbol[contract_bad].astype(str).to_numpy(), "abs_err": np.nan}))
    out["identity_ok"] = not (out["only_app_count"] or out["only_training_count"] or out["contract_mismatches"])
    out["completion_ok"] = not (out["completion_unknown"] or out["completion_false"])
    out["status"] = ("incomplete" if out["completion_unknown"] else
                     "identity_or_completion_mismatch" if not (out["identity_ok"] and out["completion_ok"]) else
                     "source_differences_reported" if any(v["out_of_tolerance"] for v in out["prices"].values()) else "equal")
    if diff_path is not None:
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["timestamp", "field", "app", "training", "abs_err"])
        frame.astype({"app": str, "training": str}).to_parquet(diff_path, index=False)
    return out


def source_impact(ref_on_app: Path, ref_on_training: Path, spec: dict, protocol: dict) -> dict:
    """Per-feature and threshold-decision impact of source differences: the same frozen reference run on the
    app-recorded bars vs on the training bars (both written by ``build_reference(..., mode='recompute')``)."""
    model_id = spec["model_id"]
    io_a, io_b = (pd.read_parquet(Path(d) / f"{model_id}.io.parquet").set_index(["endpoint", "row_offset"]) for d in (ref_on_app, ref_on_training))
    sc_a, sc_b = (pd.read_parquet(Path(d) / f"{model_id}.scores.parquet").set_index("endpoint") for d in (ref_on_app, ref_on_training))
    keys, endpoints = io_a.index.intersection(io_b.index), sc_a.index.intersection(sc_b.index)
    features = {}
    for column in [c for c in io_a.columns if c.startswith("in:")]:
        cls = spec["feature_classes"].get(column[3:], "C1_continuous")
        atol, rtol = tolerance(protocol, CLASS_TOLERANCE[cls])
        bad = ~within(io_a.loc[keys, column], io_b.loc[keys, column], atol, rtol)
        if bad.any():
            features[column[3:]] = int(bad.sum())
    threshold = spec["threshold"]
    da = sc_a.loc[endpoints].calibrated_probability >= threshold
    db = sc_b.loc[endpoints].calibrated_probability >= threshold
    flips = [dict(endpoint=e, app_bars_calibrated=float(sc_a.loc[e, "calibrated_probability"]),
                  training_bars_calibrated=float(sc_b.loc[e, "calibrated_probability"])) for e in endpoints[(da != db).to_numpy()]]
    return dict(model_id=model_id, endpoints=int(len(endpoints)), features_differing=features,
                decision_flips_for_review=flips, classification="source_difference")
