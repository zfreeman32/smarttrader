"""Prompt 4 step 2: classify every mismatch group of every compared segment x model (protocol v2 taxonomy).

Inputs: segment summaries/diffs (artifacts/s4parity_20261002_v1) and the classification diagnostics in
``classification/``: gate_test_<seg> (ES R: recompute with recorded is_complete), variant_plain_<seg> (ES H:
recompute on the app's 8,000-bar span), variant_htflegacy_<seg> (EURUSD: recompute with the app's default
htf_bar_timestamp_semantics), truncation/anchor results. Rules, first match wins per (segment, model, feature):

  1. positional row-index predictor                      -> training_side_noncausal / positional_index
  2. verified lookahead predictor (code or tail signature) -> training_side_noncausal / lookahead
  3. ES R: app == recompute with recorded completion      -> declared_contract_behavior / completion_gate
  4. ES H: app == recompute on the app's 8,000-bar span     -> app_producer_bug / window_truncation
  5. EURUSD: app == recompute with legacy HTF semantics     -> app_producer_bug / htf_timestamp_semantics
  6. F segment: feature classified in its base segment      -> base class; injected window -> source_difference
  7. otherwise                                              -> unexplained
Downstream stages (S4, S4_mask, S5, S6, S7) at a segment inherit that segment's S3 classes; a downstream mismatch
with no S3 mismatch at the same segment is unexplained. Writes classification/classified.json (summaries only).
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/s4parity_20261002_v1"
CLS = RUN / "classification"
WINDOWS = {s["segment_id"]: s for s in json.loads((RUN / "windows.json").read_text(encoding="utf-8"))["segments"]}

POSITIONAL = re.compile(r"^ict_(nearest|latest)_.*(_formed_index|_inversion_index|_displacement_id)$")
LOOKAHEAD = {
    "ict_raw_swing_high": "ict/structure/swings.py:66-82 centered pivot (confirmed_* is the 3-bar-shifted causal form); mismatches only at row offsets -2..0",
    "ict_raw_swing_low": "ict/structure/swings.py:66-82 centered pivot; mismatches only at row offsets -2..0",
    "frvp_swing_poc_vs_session_poc": "value at bar t is revised when later bars arrive (141/145 bars on ES_R_rth_maint, history start fixed); frozen values computed on the full series",
}
LOOKAHEAD_PREFIX = {
    "strategy__high_volume_points__": "features/strategies/high_volume_points.py:24-25 rolling(center=True) pivots; mismatches only at row offsets -14..0",
    "strategy__tesla_stock_boom_signals__": "features/strategies/tesla_stock_boom_signals.py:18 shift(-6); mismatches only at row offsets -5..0",
    "strategy__ichimoku_signals_ichimoku_strategy_strategy__": "mismatches only at row offsets -25..0 (26-bar displacement) with no app repaint; value at t depends on bars t+1..t+25",
}
DECLARED_GATE = ("features/feature_sets/htf_context.py:47 (pre-existing completion gate) under protocol v2 "
                 "runtime_changes_approved_in_principle.D8_provenance + eurusd_completion analogue; recorded ESU6 rows have "
                 "is_complete NULL, so this segment is not known-completion evidence")
HTF_SEMANTICS = ("ote_live/features/incremental_engine.py:313-336 build_runtime_feature_config never sets "
                 "htf_bar_timestamp_semantics, so features/feature_sets/htf_context.py:140-142 falls back to legacy_right_closed "
                 "for EURUSD; the frozen dataset config (artifacts/b2_ote_prepared_20260921_v1/dataset.metadata.json) uses bar_open")
STALE_HISTORY = {"EU_F_faults": (
    "after the backfill was upserted behind the cursor (endpoint index 25) the app's consumed rows still lack the 3 "
    "backfilled bars (gap rows at endpoint indexes 26, 32, 50; ES_F_faults holds all 4); ote_live/ingestion/signals.py:997 "
    "_reconcile_setup_history returns unless asset == 'ES', so ote_live/ingestion/setup_revisions.py:149-178 (engine.state "
    "clear + extend from the store) never refreshes the EURUSD history; the same path leaves the earlier-revision "
    "provisional bar in place")}
TRUNCATION = ("the app's 8,000-bar runtime window (runtime_history_bars) truncates history the frozen full-series values "
              "use; app equals the offline recompute on the same span")


def lookahead_note(feature: str) -> str | None:
    if feature in LOOKAHEAD:
        return LOOKAHEAD[feature]
    return next((note for prefix, note in LOOKAHEAD_PREFIX.items() if feature.startswith(prefix)), None)


def expected_endpoints(window: dict) -> set[str]:
    """Endpoints processed before an injection resolved (decision-time history differs from the final series).
    backfill: withheld..release-1; revision: the provisional endpoints (latest: k; earlier: k-lag..k-1);
    reconnect: none (the batch is processed complete); dropped: none (the final series lacks the bars too)."""
    fault = window.get("fault") or {}
    faults = fault.get("faults", []) if fault.get("kind") == "multi" else [fault]
    eps, out = window["endpoints"], set()
    for f in faults:
        at = int(f.get("at", -1))
        if f.get("kind") == "backfill":
            out.update(eps[at:at + int(f.get("count", 1)) + int(f.get("release_after", 0))])
        elif f.get("kind") == "revision":
            lag = int(f.get("earlier_lag", 3))
            out.update(eps[at - lag:at + 1])
    return out


def load(name: str) -> dict | None:
    path = CLS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def explained_counts(segment: str, model_id: str, stream: str, source: str) -> tuple[dict, str | None, str | None]:
    """Per-feature explained cell counts from the diagnostic that applies to this segment."""
    if stream == "ES" and source == "R":
        data = load(f"gate_test_{segment}.json") or {}
        return ({f: x["gate_explained"] for f, x in data.get(model_id, {}).get("features", {}).items()},
                "declared_contract_behavior/completion_gate", f"classification/gate_test_{segment}.json")
    if stream == "ES" and source == "H":
        data = (load(f"variant_plain_{segment}.json") or {}).get("models", {})
        return ({f: x["explained"] for f, x in data.get(model_id, {}).get("features", {}).items()},
                "app_producer_bug/window_truncation", f"classification/variant_plain_{segment}.json")
    if stream == "EURUSD":
        data = (load(f"variant_htflegacy_{segment}.json") or {}).get("models", {})
        return ({f: x["explained"] for f, x in data.get(model_id, {}).get("features", {}).items()},
                "app_producer_bug/htf_timestamp_semantics", f"classification/variant_htflegacy_{segment}.json")
    return {}, None, None


NOTES = {"declared_contract_behavior/completion_gate": DECLARED_GATE, "app_producer_bug/htf_timestamp_semantics": HTF_SEMANTICS,
         "app_producer_bug/window_truncation": TRUNCATION}


PRECISION = json.loads((CLS / "precision_check.json").read_text(encoding="utf-8")) if (CLS / "precision_check.json").exists() else {}
EXACT_WINDOW = {}
for _path in CLS.glob("exact_window_*.json"):
    _segment = _path.stem[len("exact_window_"):].rsplit("_", 1)[0]
    _data = json.loads(_path.read_text(encoding="utf-8"))
    _features = {}
    for _endpoint, _res in _data.items():
        for _feature, _x in _res.items():
            _features.setdefault(_feature, []).append(_x["cells"] == _x["app_equals_exact_window"])
    for _feature, _oks in _features.items():
        if all(_oks):
            EXACT_WINDOW[(_segment, _feature)] = f"classification/{_path.name}"


def classify_segment(seg_dir: Path, base_classes: dict) -> dict:
    summary = json.loads((seg_dir / "summary.json").read_text(encoding="utf-8"))
    segment, stream = summary["segment"], summary["stream"]
    window = WINDOWS[segment]
    is_f = segment.split("_")[1] == "F"
    out = {}
    for model_id, m in summary["models"].items():
        groups = []
        s3 = [g for g in m["mismatch_groups"] if g["stage"] == "S3"]
        counts, diag_class, diag_path = explained_counts(segment, model_id, stream, window["source"])
        base = base_classes.get(window.get("base_segment"), {}).get(model_id, {}) if is_f else {}
        injected_cells = {}
        if is_f and s3:
            expected = expected_endpoints(window)
            d = pd.read_parquet(seg_dir / "diff" / f"{model_id}.S3.parquet", columns=["feature", "endpoint"])
            injected_cells = d.assign(i=d.endpoint.isin(expected)).groupby("feature").i.sum().to_dict()
        for g in s3:
            feature, count = g["feature"], int(g["count"])
            parts = []
            if POSITIONAL.match(feature):
                parts.append((count, "training_side_noncausal/positional_index",
                              "row position within the input frame (ict/detectors/displacement.py:86, ict/detectors/fvg.py:263); frozen values are ordinals in the 2017-onward training file"))
            elif lookahead_note(feature):
                parts.append((count, "training_side_noncausal/lookahead", lookahead_note(feature)))
            elif is_f:
                if feature in base:
                    parts.append((count, base[feature], f"same feature classified in base segment {window['base_segment']}"))
                else:
                    inj = int(injected_cells.get(feature, 0))
                    if inj:
                        parts.append((inj, "source_difference/fault_pre_resolution", "F injection: endpoint processed before the backfill/revision resolved, so its decision-time bars differ from the final series"))
                    if count - inj and segment in STALE_HISTORY:
                        parts.append((count - inj, "app_producer_bug/stale_history_after_backfill", STALE_HISTORY[segment]))
                    elif count - inj:
                        parts.append((count - inj, "unexplained", "F cell after resolution and not classified in the base segment (stale-history candidate)"))
            else:
                explained = min(int(counts.get(feature, 0)), count)
                if explained:
                    parts.append((explained, diag_class, NOTES[diag_class] + f" (evidence {diag_path})"))
                if count - explained and (segment, feature) in EXACT_WINDOW:
                    parts.append((count - explained, "app_producer_bug/window_truncation", TRUNCATION + f" (exact app-window recompute reproduces every checked cell: {EXACT_WINDOW[(segment, feature)]})"))
                elif count - explained:
                    parts.append((count - explained, "unexplained", "not reproduced by the applicable diagnostic"))
            for n, cls, note in parts:
                groups.append(dict(stage="S3", feature=feature, producer=g.get("producer"), count=n, cls=cls, causal_note=note,
                                   first_ts=g.get("first_row_timestamp"), first_endpoint=g.get("first_endpoint"),
                                   max_abs=g.get("max_abs"), evidence_path=str((seg_dir / "diff" / f"{model_id}.S3.parquet").relative_to(ROOT))))
        s3_classes = sorted({x["cls"] for x in groups})
        upstream_of = {"S4": ("S3",), "S4_mask": ("S3",), "S5": ("S3", "S4", "S4_mask"),
                       "S6": ("S3", "S4", "S4_mask", "S5"), "S7": ("S3", "S4", "S4_mask", "S5")}
        stage_endpoints = {}
        for stage in ("S3", "S4", "S4_mask", "S5"):
            path = seg_dir / "diff" / f"{model_id}.{stage}.parquet"
            stage_endpoints[stage] = set(pd.read_parquet(path, columns=["endpoint"]).endpoint) if path.exists() else set()
        precision = PRECISION.get(f"{segment}|{summary['group']}|{model_id}", {}).get("verified", [])
        for g in m["mismatch_groups"]:
            if g["stage"] == "S3":
                continue
            base = dict(stage=g["stage"], feature=g["feature"], first_ts=g.get("first_row_timestamp"), first_endpoint=g.get("first_endpoint"),
                        max_abs=g.get("max_abs"), evidence_path=str((seg_dir / "diff" / f"{model_id}.{g['stage']}.parquet").relative_to(ROOT)))
            if g["stage"] == "S1":
                inj = bool(g["evidence_at_first"].get("injected_window"))
                cls = "source_difference/fault_pre_resolution" if inj and is_f else "unexplained"
                note = "F injection (dropped/backfill/revision/reconnect) changes delivered rows; truth marks them" if inj else "S1 mismatch outside an injection"
                groups.append(dict(base, count=int(g["count"]), cls=cls, causal_note=note))
                continue
            if g["stage"] not in upstream_of:
                groups.append(dict(base, count=int(g["count"]), cls="downstream:" + "+".join(s3_classes) if s3_classes else "unexplained",
                                   causal_note="state mismatch"))
                continue
            d = pd.read_parquet(seg_dir / "diff" / f"{model_id}.{g['stage']}.parquet", columns=["feature", "endpoint"])
            eps = d.loc[d.feature == g["feature"], "endpoint"]
            upstream = set().union(*(stage_endpoints[s] for s in upstream_of[g["stage"]]))
            n_down = int(eps.isin(upstream).sum())
            n_prec = int((~eps.isin(upstream) & eps.isin(precision)).sum()) if g["stage"] == "S6" else 0
            n_rest = len(eps) - n_down - n_prec
            if n_down:
                groups.append(dict(base, count=n_down, cls="downstream:" + "+".join(s3_classes) if s3_classes else "downstream:upstream_stage",
                                   causal_note="endpoint has upstream (S3-S5) mismatches in this segment"))
            if n_prec:
                groups.append(dict(base, count=n_prec, cls="declared_contract_behavior/float32_tree_split",
                                   causal_note="protocol v3 mismatch_taxonomy.declared_contract_behavior.float32_tree_split; verified in classification/precision_check.json"))
            if n_rest:
                groups.append(dict(base, count=n_rest, cls="unexplained",
                                   causal_note=f"{g['stage']} mismatch at endpoints without upstream mismatch" + (" (S7 flips are never covered by the v3 clause)" if g["stage"] == "S7" else "")))
        checks = []
        for name in m["headline"]["failed_checks"]:
            value = m["checks"][name]
            n = len(value) if isinstance(value, list) else value
            if name in {"health_errors", "not_evaluated"} and window.get("failure"):
                cls, note = "declared_contract_behavior/producer_failure_surfaced", "protocol v2 scenarios.F_injections.producer_failure: feature_build_failed error, endpoint skipped, never a silent fill"
            elif name == "not_evaluated" and is_f and window.get("fault"):
                cls, note = "source_difference/fault_pre_resolution", "withheld backfill bars are delivered behind the cursor and not re-scored"
            elif name == "stale_helper" and stream == "ES" and window["source"] == "R" or (name == "stale_helper" and is_f and stream == "ES"):
                cls, note = "declared_contract_behavior/completion_gate", DECLARED_GATE
            elif name == "stale_helper" and window["source"] == "H":
                cls, note = "declared_contract_behavior/post_closure_helper_age", "HTF helper age after a planned closure (first hour after reopen); HTF values equal the reference"
            else:
                cls, note = "unexplained", "failed check"
            checks.append(dict(check=name, count=n, cls=cls, causal_note=note))
        out[model_id] = dict(queue_id=m["queue_id"], endpoint_status=m["endpoint_status"], comparable=m["headline"]["comparable"],
                             endpoints=m["headline"]["endpoints"], stages=m["stages"], groups=groups, checks=checks,
                             near_threshold=m["near_threshold"], decision_flips=m["decision_flips"],
                             artifact_check=m["checks"]["artifact_hash"], diff_files=m["diff_files"])
    return dict(segment=segment, stream=stream, group=summary["group"], source=window["source"], scenarios=window["scenarios"],
                S2=summary["S2"], summary_path=str((seg_dir / "summary.json").relative_to(ROOT)), models=out,
                eurusd_calendar=summary.get("eurusd_calendar"))


def main() -> None:
    dirs = [p.parent for p in sorted(RUN.glob("*/*/*/summary.json"))
            if "superseded" not in str(p) and "plumbing" not in str(p) and "r1_value" not in str(p) and p.parent.name in WINDOWS]
    results, base_classes = {}, {}
    for seg_dir in sorted(dirs, key=lambda p: (p.name.split("_")[1] == "F", p.name)):  # bases before F segments
        result = classify_segment(seg_dir, base_classes)
        results[f"{result['segment']}|{result['group']}"] = result
        for model_id, m in result["models"].items():
            per = defaultdict(set)
            for g in m["groups"]:
                if g["stage"] == "S3":
                    per[g["feature"]].add(g["cls"])
            base_classes.setdefault(result["segment"], {})[model_id] = {f: next(iter(c)) for f, c in per.items() if len(c) == 1 and "unexplained" not in c}
    (CLS / "classified.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    tally = defaultdict(Counter)
    for key, r in results.items():
        for model_id, m in r["models"].items():
            for g in m["groups"]:
                tally[m["queue_id"]][g["cls"].split(":")[0] if g["cls"].startswith("downstream") else g["cls"]] += g["count"]
            for c in m["checks"]:
                tally[m["queue_id"]]["check:" + c["cls"]] += 1
    for q in sorted(tally):
        print(q, dict(tally[q]))


if __name__ == "__main__":
    main()
