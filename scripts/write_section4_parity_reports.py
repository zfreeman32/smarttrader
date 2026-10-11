"""Section 4 prompt 4 steps 3-4: one immutable report per exact roster model version plus a manifest.

Reads the classified matrix (artifacts/s4parity_20261002_v1/classification/classified.json), windows.json, the
restart gates, the composition re-check and the protocol v2 lock. Writes
``docs/audits/section4_parity_<model_id>_<reviewed_set_sha12>_<date>.json`` (+ ``.sha256``) per model and
``docs/audits/section4_parity_manifest_<date>.json``. Refuses to overwrite (protocol report_schema.immutability).
Status: pass | fail:<class> | incomplete. A pass is not scored eligibility (B6, A12, A13, B7 still required).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import section4_parity_harness as H  # noqa: E402

DATE = "20261005"
RUN = H.RUN_ROOT
OUT = ROOT / "docs/audits"
V3 = ROOT / "configs/section4_parity_protocol_v3.json"
V3_LOCK = ROOT / "configs/section4_parity_protocol_v3.lock.json"
FAIL_ORDER = ("training_side_noncausal", "app_producer_bug", "unexplained")
SCENARIO_KEYS_SKIP = {"candidates", "restart_points", "completion"}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lock = json.loads(V3_LOCK.read_text(encoding="utf-8"))
    if file_sha(V3) != lock["sha256"]:
        raise SystemExit("protocol v3 does not match its lock")
    protocol = json.loads(V3.read_text(encoding="utf-8"))
    protocol_ref = dict(version=3, sha256=lock["sha256"], lock_sha256=file_sha(V3_LOCK),
                        executed_under=dict(version=2, sha256=H.PROTOCOL_SHA256, lock_sha256=H.PROTOCOL_LOCK_SHA256),
                        note="run executed under v2 (tolerances identical); classified under v3, which adds only the "
                             "float32_tree_split declared-behavior clause (Q-P1, user-approved 2026-10-05)")
    classified = json.loads((RUN / "classification/classified.json").read_text(encoding="utf-8"))
    windows = {s["segment_id"]: s for s in json.loads((RUN / "windows.json").read_text(encoding="utf-8"))["segments"]}
    windows_sha = file_sha(RUN / "windows.json")
    gates = {s: json.loads((RUN / f"restart_gate_{s}.json").read_text(encoding="utf-8")) for s in ("ES", "EURUSD")}
    composition = json.loads((RUN / "classification/composition_recheck_ES.json").read_text(encoding="utf-8"))
    roster = H.roster_by_id()
    model_records, run_meta = {}, defaultdict(dict)
    for key, seg in classified.items():
        run = json.loads((ROOT / seg["summary_path"]).parent.joinpath("app/run.json").read_text(encoding="utf-8"))
        for record in run["models"]:
            model_records.setdefault(record["queue_id"], record)
        run_meta[key] = dict(parity_db_sha256=run.get("parity_db_sha256"), code_sha256=run["code_sha256"],
                             resumes=run.get("resumes"), started=run.get("started_at_utc"), finished=run.get("finished_at_utc"))
    manifest = dict(schema="section4-parity-manifest-v1", date=DATE, protocol=protocol_ref, windows_sha256=windows_sha,
                    run_id=RUN.name, reports=[])
    for queue_id in [2, 3, 5, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22, 23, 25, 26]:
        model = roster[queue_id]
        record = model_records[queue_id]
        stream = model["stream"]
        segs = {k: v for k, v in classified.items() if model["model_id"] in v["models"]}
        stages, features, groups, near, flips, win_rows, code_sets = {}, {}, {}, [], [], [], defaultdict(list)
        for key, seg in sorted(segs.items()):
            m = seg["models"][model["model_id"]]
            w = windows[seg["segment"]]
            win_rows.append(dict(segment_id=seg["segment"], group=seg["group"], source=w["source"], rule=w["rule"],
                                 start=w["endpoints"][0], end=w["endpoints"][-1], endpoints=m["endpoints"],
                                 comparable_endpoints=m["comparable"], endpoint_status=m["endpoint_status"],
                                 parity_db_sha256=run_meta[key]["parity_db_sha256"], summary=seg["summary_path"]))
            code_sets[canonical(run_meta[key]["code_sha256"])].append(seg["segment"])
            for stage, s in m["stages"].items():
                agg = stages.setdefault(stage, dict(compared=0, matched=0, mismatched=0, not_compared=0, max_abs=0.0, max_rel=0.0, first_offending_ts=None))
                for k in ("compared", "matched", "mismatched", "not_compared"):
                    agg[k] += int(s.get(k) or 0)
                for k in ("max_abs", "max_rel"):
                    if s.get(k) is not None and s[k] == s[k]:
                        agg[k] = max(agg[k], float(s[k]))
                first = s.get("first_offending_ts") or s.get("first_ts")
                if first and (agg["first_offending_ts"] is None or str(first) < agg["first_offending_ts"]):
                    agg["first_offending_ts"] = str(first)
            for g in m["groups"]:
                cls = g["cls"]
                gk = (cls, g["stage"], g["feature"] if g["stage"] in {"S1", "S3"} else "*")
                e = groups.setdefault(gk, dict(cls=cls, stage=g["stage"], feature=gk[2], count=0, first_ts=None, segments=[],
                                               evidence_paths=[], causal_note=g["causal_note"]))
                e["count"] += g["count"]
                if g.get("first_ts") and (e["first_ts"] is None or str(g["first_ts"]) < e["first_ts"]):
                    e["first_ts"] = str(g["first_ts"])
                if seg["segment"] not in e["segments"]:
                    e["segments"].append(seg["segment"])
                    if len(e["evidence_paths"]) < 3:
                        e["evidence_paths"].append(g["evidence_path"])
                if g["stage"] == "S3":
                    f = features.setdefault(g["feature"], dict(mismatches=0, classes=[]))
                    f["mismatches"] += g["count"]
                    if cls not in f["classes"]:
                        f["classes"].append(cls)
            for c in m["checks"]:
                gk = (c["cls"], "check", c["check"])
                e = groups.setdefault(gk, dict(cls=c["cls"], stage="check", feature=c["check"], count=0, first_ts=None, segments=[],
                                               evidence_paths=[], causal_note=c["causal_note"]))
                e["count"] += int(c["count"] or 0)
                if seg["segment"] not in e["segments"]:
                    e["segments"].append(seg["segment"])
                    e["evidence_paths"].append(seg["summary_path"]) if len(e["evidence_paths"]) < 3 else None
            near += [dict(segment=seg["segment"], **x) for x in m["near_threshold"]]
            flips += [dict(segment=seg["segment"], **x) for x in m["decision_flips"]]
        # coverage
        scenarios = [k for k in protocol["scenarios"][stream] if k not in SCENARIO_KEYS_SKIP]
        coverage = {}
        for scenario in scenarios:
            hits = [r for r in win_rows if scenario in windows[r["segment_id"]]["scenarios"] and r["comparable_endpoints"] > 0]
            reason = None
            if scenario == "restart":
                gate = gates[stream]
                equal = [v["equal"] for k, v in gate["results"].items()]
                coverage[scenario] = dict(status="exercised" if hits and gate["passed"] else "unexercised",
                                          reason=f"restart gate {'passed' if gate['passed'] else 'failed'} ({sum(equal)}/{len(equal)} bitwise equal)",
                                          segments=[r["segment_id"] for r in hits])
                continue
            if scenario == "group_composition":
                c = composition.get(str(queue_id))
                ok = bool(c) and not c["differing_columns"]
                coverage[scenario] = dict(status="exercised" if ok else "unexercised",
                                          reason=(f"single-model engine equal to group engine on {c['shared_keys']} shared rows; io/scores bitwise equal "
                                                  "(classification/composition_recheck_ES.json)") if c else "not run")
                continue
            if scenario in {"producer_failure", "strategy_timeout_producer_failure"}:
                reason = "surfaced as feature_build_failed error; affected endpoints skipped (never a silent fill)"
            coverage[scenario] = dict(status="exercised" if hits else "unexercised",
                                      reason=reason or (f"{len(hits)} segment(s) with comparable endpoints" if hits else "no segment with comparable endpoints"),
                                      segments=[r["segment_id"] for r in hits])
        s2 = [classified[k]["S2"] for k in segs if classified[k]["S2"].get("status") not in (None, "not_run")]
        if stream == "ES":
            matched = sum(int(x.get("matched") or 0) for x in s2)
            ref = sum(int(x.get("ref_events") or 0) for x in s2)
            s2_status = dict(status="compared_reference_recall", reference_events=ref, matched=matched,
                             only_ref=sum(int(x.get("only_ref") or 0) for x in s2), only_app=sum(int(x.get("only_app") or 0) for x in s2),
                             note="H segments only (offline detector events exist for the training period). Reference = events section 3 used "
                                  "(FRVP detector columns; ICT corrected fired-event file), so app-only events are outside the reference "
                                  "population; geometry mismatches 0. The single only_ref/only_app pair (ES_H_fall_dst FRVP 2025-10-31 17:05, "
                                  "setup_type app 1 vs ref 4, first endpoint) is attributed in the summary.")
        else:
            s2_status = dict(status="incomplete", reason="no live OTE setup-event producer (protocol setup_events / D5)")
        source_parity = dict(status="incomplete", reason=(
            "protocol source_parity.fallback: IBKR not running (no listener on 4001) for ES 2026-06-01..06-19; the app's FMP connector "
            "(FMPRESTClient.fetch_historical_chart) sends no from/to and filters the latest response, so EURUSD 2026-03-02..03-24 is "
            "unreachable through the app's own connector. No outward fetch was made."))
        restart = {k: dict(equal=v["equal"], base=v["base"]) for k, v in gates[stream]["results"].items()}
        # verdict
        present = defaultdict(int)
        for e in groups.values():
            top = e["cls"].split("/")[0]
            if e["cls"].startswith("downstream:"):
                continue
            present[top] += e["count"]
        fails = [c for c in FAIL_ORDER if present.get(c)]
        reasons = [f"{c}: {present[c]} cells/checks" for c in fails]
        unexercised = [s for s, v in coverage.items() if v["status"] != "exercised"]
        if fails:
            status = f"fail:{fails[0]}"
        else:
            status = "incomplete" if unexercised or source_parity["status"] != "exercised" or s2_status["status"] == "incomplete" else "pass"
        reasons += [f"scenario unexercised: {s}" for s in unexercised]
        reasons.append("source parity incomplete (protocol fallback)")
        if s2_status["status"] == "incomplete":
            reasons.append("S2 incomplete (D5)")
        if any(g["cls"].startswith("declared_contract_behavior/completion_gate") for g in groups.values()):
            reasons.append("ES R segments are not known-completion evidence (recorded ESU6 completion unknown)")
        report = dict(
            id=protocol["report_schema"]["id"], date=DATE,
            model=dict(queue_id=queue_id, model_id=model["model_id"], stream=stream,
                       reviewed_artifact_set_sha256=record["reviewed_artifact_set_sha256"], artifact_sha256=record["artifact_sha256"],
                       staged_manifest_sha256=record["staged_manifest_sha256"], prepared_manifest_sha256=record.get("prepared_manifest_sha256"),
                       threshold=record["threshold"], consumed_rows=record["consumed_rows"]),
            protocol=protocol_ref,
            run=dict(id=RUN.name, dates=["2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"], windows_sha256=windows_sha,
                     code_sha256_sets=[dict(code_sha256=json.loads(k), segments=v) for k, v in code_sets.items()],
                     logs=["logs/section4_matrix_20261002.log", "logs/section4_matrix_20261003.log", "logs/section4_diagnostics_20261005.log"]),
            windows=win_rows, coverage=coverage, stages=stages, features=features,
            mismatches=sorted(groups.values(), key=lambda e: (e["cls"], e["stage"], -e["count"])),
            near_threshold=near, decision_flips=flips, restart=restart, setup_events=s2_status, source_parity=source_parity,
            verdict=status, reasons=reasons,
            scored_eligibility="not granted: a parity verdict is not scored eligibility; B6, A12, A13 and B7 are still required",
        )
        if stream == "EURUSD":
            cal = protocol["inputs"]["eurusd_calendar_v1"]
            report["eurusd_calendar"] = dict(id="eurusd_5m_feed_clock_week_v1", path=cal["path"], sha256=cal["sha256"], status="provisional")
        report["report_sha256"] = hashlib.sha256(canonical(report).encode()).hexdigest()
        path = OUT / f"section4_parity_{model['model_id']}_{record['reviewed_artifact_set_sha256'][:12]}_{DATE}.json"
        text = json.dumps(report, indent=1, default=str) + "\n"
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_text(text, encoding="utf-8")
        digest = file_sha(path)
        path.with_suffix(".json.sha256").write_text(digest + "\n", encoding="utf-8")
        manifest["reports"].append(dict(queue_id=queue_id, model_id=model["model_id"], stream=stream, verdict=status,
                                        path=str(path.relative_to(ROOT)).replace("\\", "/"), sha256=digest, report_sha256=report["report_sha256"]))
        print(queue_id, model["model_id"], status, digest[:12])
    mpath = OUT / f"section4_parity_manifest_{DATE}.json"
    if mpath.exists():
        raise SystemExit(f"refusing to overwrite {mpath}")
    mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    mpath.with_suffix(".json.sha256").write_text(file_sha(mpath) + "\n", encoding="utf-8")
    print("manifest", file_sha(mpath)[:12])


if __name__ == "__main__":
    main()
