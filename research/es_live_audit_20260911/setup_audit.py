"""Extract archived ES setup evidence without rerunning either detector.

Run from repo root: ote_venv/Scripts/python.exe research/es_live_audit_20260911/setup_audit.py
The snapshot is opened read-only; this script writes only sibling audit artifacts.
"""
from __future__ import annotations

import collections
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
FRVP_FAMILIES = {1: "reversal", 2: "continuation", 3: "continuation", 4: "reversal", 5: "continuation", 6: "reversal"}
FRVP_NAMES = {1: "balanced profile value-edge fade", 2: "breakout acceptance/retest", 3: "value-area volume breakout", 4: "failed auction/reentry", 5: "low-volume-node traverse", 6: "high-volume-node magnet"}
NY = ZoneInfo("America/New_York")


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (HERE / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def model_route(model):
    parts = model.split("_")
    strategy, direction, family = parts[:3]
    remainder = "_".join(parts[3:])
    match = re.match(r"setup([1-6])_", remainder)
    if strategy == "frvp" and match:
        setup_type = match.group(1)
    elif strategy == "ict" and not re.match(r"(?:xgb|tcn|lstm)_v", remainder):
        setup_type = re.sub(r"_(?:xgb|tcn|lstm)_v\d+.*$", "", remainder)
    else:
        setup_type = None
    return strategy.upper(), direction, family, setup_type


def main():
    conn = sqlite3.connect((HERE / "es_audit_snapshot.sqlite3").as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    predictions = collections.defaultdict(list)
    labels_by_bar = collections.defaultdict(set)
    archived = collections.defaultdict(list)
    context_counts = collections.Counter()
    missing_keys = collections.Counter()
    duplicate_keys = collections.Counter()
    specialist_counts = collections.Counter()
    for row in conn.execute("""
        SELECT p.*,d.id AS decision_id,d.decision,d.reasons_json
        FROM model_predictions p LEFT JOIN signal_decisions d ON d.prediction_id=p.id
        ORDER BY p.recorded_at_utc,p.id
    """):
        row = dict(row)
        meta = json.loads(row.pop("metadata_json") or "{}")
        ctx = meta.get("policy_context", {})
        strategy = row["model_id"].split("_")[0].upper()
        if ctx.get("asset") != "ES":
            continue
        context_counts[strategy] += 1
        setup_keys = [key for key in ctx if "setup" in key]
        if not setup_keys:
            missing_keys[strategy] += 1
        row.pop("prediction_json", None)
        row["context_close"] = ctx.get("close")
        row["context_low"] = ctx.get("low")
        row["context_high"] = ctx.get("high")
        row["bar_complete"] = ctx.get("ibkr_is_complete")
        row["market_data_type"] = ctx.get("ibkr_market_data_type")
        row["shadow_mode"] = meta.get("shadow_mode")
        row["regime"] = ctx.get("composite_regime", row["regime"])
        row["session_regime"] = ctx.get("session_regime")
        row["setup_type"] = str(ctx.get("frvp_setup_type", "")) if strategy == "FRVP" else ""
        row["setup_side"] = ctx.get("frvp_setup_side", 0) if strategy == "FRVP" else 0
        row["confidence"] = clean(ctx.get("frvp_setup_confidence_rule"))
        route_strategy, route_direction, route_family, route_setup = model_route(row["model_id"])
        if route_strategy == "FRVP" and route_setup is not None:
            specialist_counts["predictions"] += 1
            if row["calibrated_probability"] >= row["threshold_applied"]:
                specialist_counts["threshold_passes"] += 1
                matched = row["setup_type"] == route_setup and row["setup_side"] == (1 if route_direction == "long" else -1)
                specialist_counts["threshold_passes_matching_setup" if matched else "threshold_passes_off_setup"] += 1
        predictions[(strategy, row["timestamp_utc"])].append(row)
        duplicate_keys[(row["model_id"], row["timestamp_utc"])] += 1
        if strategy == "FRVP":
            labels_by_bar[(strategy, row["timestamp_utc"])].add((row["setup_type"], row["setup_side"]))
            if row["setup_type"] not in {"", "0", "None"} and row["setup_side"] in {-1, 1}:
                key = strategy, row["timestamp_utc"], row["setup_type"], row["setup_side"]
                archived[key].append(row)

    dashboard = collections.defaultdict(list)
    dashboard_meta = {}
    for row in conn.execute("SELECT * FROM runtime_state WHERE scope='dashboard_view_state'"):
        payload = json.loads(row["payload_json"])
        strategy = row["state_key"]
        if payload.get("asset") != "ES":
            continue
        dashboard_meta[strategy] = {
            "updated_at_utc": row["updated_at_utc"],
            "event_count": len(payload.get("recent_setups", [])),
            "latest_bar_timestamp_utc": payload.get("latest_bar_timestamp_utc"),
        }
        for ev in payload.get("recent_setups", []):
            key = strategy, ev["timestamp_utc"], str(ev["setup_type"]), ev["setup_side"]
            dashboard[key].append(ev)

    events = []
    probability_rows = []
    for key in sorted(set(archived) | set(dashboard), key=lambda k: (k[1], k[0], k[2], k[3])):
        strategy, ts, setup_type, side = key
        source = archived.get(key, [])
        state = dashboard.get(key, [])
        first = source[0] if source else None
        last = source[-1] if source else None
        ev = state[0] if state else {}
        model_rows = predictions.get((strategy, ts), [])
        family = FRVP_FAMILIES[int(setup_type)] if strategy == "FRVP" else ev.get("setup_family")
        event_id = f"{strategy}:{ts}:{setup_type}:{side:+d}"
        close = first["context_close"] if first else ev.get("bar_close")
        confidence = first["confidence"] if first else ev.get("confidence")
        provenance = "prediction_policy_context+dashboard_retained" if source and state else "prediction_policy_context" if source else "dashboard_retained_only"
        event = {
            "setup_id": event_id, "timestamp_utc": ts,
            "timestamp_et": stamp(ts).astimezone(NY).isoformat(), "strategy": strategy,
            "setup_type": setup_type, "setup_side": side, "setup_family": family,
            "setup_name": FRVP_NAMES[int(setup_type)] if strategy == "FRVP" else setup_type.replace("_", " "),
            "confidence": confidence, "provenance": provenance, "close": close,
            "entry_price": ev.get("entry_price", close),
            "stop_price": ev.get("stop_reference"), "target_price": ev.get("target_reference"),
            "bar_low": first["context_low"] if first else ev.get("bar_low"),
            "bar_high": first["context_high"] if first else ev.get("bar_high"),
            "anchor_level": ev.get("anchor_level"), "reference_level": ev.get("reference_level"),
            "reference_level_type": ev.get("reference_level_type"), "htf_context": ev.get("htf_context"),
            "sweep_type": ev.get("sweep_type"), "session_phase": ev.get("session_phase"),
            "original_mark_recorded_at_utc": first["recorded_at_utc"] if first else None,
            "last_matching_prediction_recorded_at_utc": last["recorded_at_utc"] if last else None,
            "original_mark_recorded_time_known": bool(first),
            "first_model_recorded_at_utc": model_rows[0]["recorded_at_utc"] if model_rows else None,
            "dashboard_state_updated_at_utc": dashboard_meta.get(strategy, {}).get("updated_at_utc") if state else None,
            "prediction_context_observations": len(source), "prediction_model_count": len(model_rows),
            "dashboard_copies": len(state),
            "context_label_variants_on_bar": len(labels_by_bar[(strategy, ts)]) if strategy == "FRVP" else None,
            "context_confidence_variants": len(set(r["confidence"] for r in source)) if source else None,
            "context_close_variants": len(set(r["context_close"] for r in source)) if source else None,
            "dashboard_confidence_matches_context": abs(ev["confidence"] - confidence) < 1e-10 if source and state else None,
            "dashboard_close_matches_context": ev["bar_close"] == close if source and state else None,
        }
        events.append(event)
        for row in model_rows:
            _, model_direction, model_family, model_setup = model_route(row["model_id"])
            direction_match = (model_direction == "long") == (side == 1)
            family_match = model_family in {family, "meta"}
            route_match = direction_match and family_match and (model_setup is None or model_setup == setup_type)
            probability_rows.append({
                "setup_id": event_id, "timestamp_utc": ts, "strategy": strategy,
                "setup_type": setup_type, "setup_side": side, "setup_family": family,
                "prediction_id": row["id"], "model_id": row["model_id"],
                "runtime_manifest_id": row["runtime_manifest_id"], "direction": row["direction"],
                "model_family": model_family, "model_setup_type": model_setup,
                "is_setup_specific": model_setup is not None,
                "direction_match": direction_match, "family_match": family_match, "route_match": route_match,
                "probability": row["calibrated_probability"], "raw_score": row["raw_score"],
                "threshold": row["threshold_applied"],
                "threshold_pass": row["calibrated_probability"] >= row["threshold_applied"] if row["threshold_applied"] is not None else None,
                "decision": row["decision"], "decision_id": row["decision_id"],
                "shadow_mode": row["shadow_mode"], "reasons_json": row["reasons_json"],
                "recorded_at_utc": row["recorded_at_utc"],
                "lag_from_bar_timestamp_minutes": (stamp(row["recorded_at_utc"]) - stamp(ts)).total_seconds() / 60,
                "context_close": row["context_close"], "context_close_matches_mark": row["context_close"] == close,
                "bar_complete": row["bar_complete"], "market_data_type": row["market_data_type"],
                "regime": row["regime"], "session_regime": row["session_regime"],
                "setup_provenance": provenance,
                "setup_mark_recorded_time_known": bool(first),
            })

    write_csv("setup_events.csv", events)
    write_csv("setup_model_probabilities.csv", probability_rows)
    summary = {
        "context_rows_by_strategy": dict(context_counts),
        "context_rows_without_setup_keys": dict(missing_keys),
        "total_events": len(events),
        "event_counts": {system: sum(e["strategy"] == system for e in events) for system in context_counts},
        "event_type_counts": dict(collections.Counter(f"{e['strategy']}:{e['setup_type']}:{e['setup_side']}" for e in events)),
        "provenance_counts": dict(collections.Counter(e["provenance"] for e in events)),
        "dashboard_metadata": dashboard_meta,
        "dashboard_duplicate_entries": sum(max(0, len(v) - 1) for v in dashboard.values()),
        "model_timestamp_duplicates": sum(n - 1 for n in duplicate_keys.values()),
        "frvp_setup_bars_with_label_variants": sum(e["strategy"] == "FRVP" and e["context_label_variants_on_bar"] > 1 for e in events),
        "dashboard_context_disagreements": sum(e["dashboard_confidence_matches_context"] is False or e["dashboard_close_matches_context"] is False for e in events),
        "probability_rows_at_mark_timestamp": len(probability_rows),
        "setup_specific_matched_observations": sum(r["is_setup_specific"] and r["route_match"] for r in probability_rows),
        "setup_specific_matched_threshold_passes": sum(r["is_setup_specific"] and r["route_match"] and r["threshold_pass"] for r in probability_rows),
        "setup_specific_matched_by_model": {},
        "all_frvp_specialist_rows": dict(specialist_counts),
    }
    for model in sorted({r["model_id"] for r in probability_rows if r["is_setup_specific"]}):
        rows = [r for r in probability_rows if r["model_id"] == model and r["route_match"]]
        summary["setup_specific_matched_by_model"][model] = {
            "n": len(rows), "threshold_passes": sum(r["threshold_pass"] for r in rows),
            "mean_probability": sum(r["probability"] for r in rows) / len(rows) if rows else None,
        }
    (HERE / "setup_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    family_rows = []
    for model in sorted({r["model_id"] for r in probability_rows if not r["is_setup_specific"]}):
        rows = [r for r in probability_rows if r["model_id"] == model and r["route_match"]]
        family_rows.append({
            "model_id": model, "matching_setup_count": len(rows),
            "threshold_passes_at_matching_setup": sum(r["threshold_pass"] for r in rows),
            "mean_probability_at_matching_setup": statistics.mean(r["probability"] for r in rows) if rows else None,
        })
    write_csv("setup_family_model_agreement.csv", family_rows)
    report = [
        "# ES archived setup evidence audit",
        "",
        "Snapshot: `es_audit_snapshot.sqlite3`. This extraction never calls either setup detector, rebuilds features, or generates new historical model predictions. All times below use America/New_York unless marked UTC.",
        "",
        "**Recoverable scope.** There are 87 unique retained setup events: 47 FRVP (31 short, 16 long) and 40 ICT (17 short, 23 long). FRVP type/side/confidence are present in all 19,284 FRVP prediction policy contexts; 433 prediction rows contain a fired setup, collapsing to 47 setup bars. All 40 retained FRVP dashboard markings exactly match their archived context close and confidence. The other seven FRVP events precede the rolling dashboard retention window. ICT type/side/family are absent from all 16,683 ICT prediction policy contexts. Only 40 retained ICT dashboard markings can be recovered as setup events; older ICT setup markings cannot be fully audited from this snapshot.",
        "",
        "**What the dashboard evidence means.** Current source calls dashboard persistence on each processed bar before the model loop (`ote_live/ingestion/signals.py:641`). The dashboard detector result's latest row is appended to recent_setups only when fired, and each strategy retains at most 40 events (`ote_live/dashboard/view_state.py:14`, `:18`, `:94`). These are retained outputs from historical processing, not labels reconstructed by this audit. They may originate from delayed processing or catch-up: a bar timestamp is not its original wall-clock observation time. The current dashboard state's Sep 11 update timestamp is not the original timestamp when each older setup was marked. FVG history in the same ICT payload is structural-zone lifecycle data, not additional setup/trade events.",
        "",
        "**Revision limitations.** No duplicate model/timestamp prediction records or conflicting FRVP labels/close values are present in this snapshot. The dashboard append helper ignores a new fire when the last event already has the same timestamp (`view_state.py:533`); an earlier fire is not removed if the revised bar subsequently reports no setup. Therefore absence of revisions in storage does not establish that the detector or incomplete bar never changed. ICT dashboard events do not carry per-event recorded_at timestamps or detector-version identifiers. `original_mark_recorded_at_utc` in setup_events.csv means the first archived FRVP prediction record containing that setup, not an exact chart-render timestamp; it is blank for ICT.",
        "",
        "**Price and timing association.** Every recovered event has model outputs at its exact source bar timestamp: 772 event/model records, 433 FRVP and 339 ICT. Every model context close matches the recorded setup close. However, at least one associated context has ibkr_is_complete=False on 30/47 FRVP and 25/40 ICT marks. First associated model output lag from the source bar timestamp has median 17.33 minutes for FRVP and 17.15 minutes for ICT, maximum 291.38 and 419.07 minutes. These associations support studying model behavior at the source setup bar; they do not establish fills executable at that historical close. A five-minute bar's timestamp may be its opening timestamp, so the lag from nominal bar completion is five minutes smaller.",
        "",
        "**Setup confidence and model probability are different fields.** Rule confidence ranges from 0.4831 to 0.96 for FRVP and 0.5325 to 0.87 for ICT. These values come from rule-based scoring formulas in the setup detectors; they are not calibrated probabilities of making money. Use `confidence` for rule score and `probability` for the separately archived model output. The family and specialist models may target different labels/horizons, so their probabilities are not interchangeable without checking manifest target semantics.",
        "",
        "**The largest observed source of apparent model noise is off-setup shadow output.** All 8,454 FRVP specialist model predictions were evaluated across bars. There were 3,306 threshold passes, but only six of those passes occurred on that model's actual setup number and long direction. The other 3,300 (99.82%) were off its intended setup. Shadow decisions return immediately after checking the threshold (`ote_live/policies/decision_engine.py:105`), before setup holds, abstention rules, and cooldown. The caller also suppresses setup force-hold reasons for shadow mode (`ote_live/ingestion/signals.py:711`). Thus the archive's candidate_passed_threshold flag alone is not a strategy-qualified entry. This behavior is consistent with the current shadow implementation; it does not mean 3,300 unintended trades were placed.",
        "",
        "**Families and specialists.** FRVP reversal pools S1/S4/S6; continuation pools S2/S3/S5 (`frvp/target_lanes.py:7`). A family model emits a score over the broader label family; a specialist targets one setup and direction. ICT reversal includes sweep reclaim and IFVG reversal here; premium/discount continuation belongs to continuation. Meta models combine family-level targets. The setup gate recognizes only specialist model IDs; broad family model IDs return a matched gate without checking the current setup (`ote_live/models/setup_family.py:131`). For meaningful setup comparisons, this audit explicitly requires strategy, direction, family (or meta), and specialist setup type to match.",
        "",
        "**Coverage is asymmetric.** There are six FRVP long specialist models and three ICT short specialist models. No FRVP short specialist and no ICT long specialist appears in this archive. The FRVP short reversal family has only 31 archived scores, all outside the 47 fired FRVP setup bars, so no FRVP short reversal probability exists at these retained reversal marks. The 47 FRVP events have either five or eleven same-bar model scores, and the 40 ICT events have six or nine. Long S1 and long S2 specialists have zero matching event observations during their active coverage, despite producing 1,409 scores each. The count of raw scores substantially overstates the count of relevant validation examples.",
        "",
        "**Recovered setup composition**",
        "",
        "| Strategy | Setup | Family | Long | Short |",
        "|---|---|---|---:|---:|",
    ]
    for strategy in ("FRVP", "ICT"):
        for setup_type in sorted({e["setup_type"] for e in events if e["strategy"] == strategy}):
            rows = [e for e in events if e["strategy"] == strategy and e["setup_type"] == setup_type]
            report.append(f"| {strategy} | {setup_type} | {rows[0]['setup_family']} | {sum(e['setup_side'] == 1 for e in rows)} | {sum(e['setup_side'] == -1 for e in rows)} |")
    report.extend([
        "", "**Matching specialist output at the setup mark**", "",
        "| Model | Matching marks | Threshold passes | Mean model probability |",
        "|---|---:|---:|---:|",
    ])
    for model, values in summary["setup_specific_matched_by_model"].items():
        mean = f"{values['mean_probability']:.3f}" if values["mean_probability"] is not None else "—"
        report.append(f"| {model} | {values['n']} | {values['threshold_passes']} | {mean} |")
    report.extend([
        "", "Only 25 specialist observations match an archived setup: ten FRVP and fifteen ICT. Eight pass threshold. This is too small and uneven to rank all specialists reliably.",
        "", "**Matching family output at the setup mark**", "",
        "| Model | Matching marks | Threshold passes | Mean model probability |",
        "|---|---:|---:|---:|",
    ])
    for row in family_rows:
        report.append(f"| {row['model_id']} | {row['matching_setup_count']} | {row['threshold_passes_at_matching_setup']} | {row['mean_probability_at_matching_setup']:.3f} |")
    report.extend([
        "", "ICT family reversal filters agree with very few markings: long reversal passes one of twenty and short reversal zero of fifteen. Long continuation passes all three matching continuation marks; short continuation zero of two. On FRVP, the long/short meta models pass 12/16 and 25/31 respective setup marks. This describes model selectivity; it does not by itself prove which filters improve returns.",
        "", "**Eight specialist threshold passes on their matching setup**", "",
        "| Source bar ET | Model | Probability | Threshold |",
        "|---|---|---:|---:|",
    ])
    for row in probability_rows:
        if row["is_setup_specific"] and row["route_match"] and row["threshold_pass"]:
            report.append(f"| {stamp(row['timestamp_utc']).astimezone(NY).strftime('%Y-%m-%d %H:%M')} | {row['model_id']} | {row['probability']:.4f} | {row['threshold']:.4f} |")
    report.extend([
        "", "**Trade geometry and outcomes.** All 40 ICT retained marks contain entry, stop reference, and target reference with positive directional risk and reward. These are setup references, not recorded broker orders or fills. FRVP dashboard events do not contain an executable bracket. Outcome evaluation must use bars after the setup, flag gaps and same-bar stop/target ambiguity, and account for recorded-time availability. See the companion root audit for price-action markouts, simulated bracket behavior, trade ledger results, and profitability.",
        "", "**Audit improvements indicated by the evidence.** Persist immutable setup events for both systems with detector/version hash, source bar version and completion flag, observed-at timestamp, exact model prediction IDs, gate match/result, setup geometry, and revision events. Keep threshold candidates separate from gate-qualified shadow entries in analytics, and evaluate the actual abstention/cooldown policy for shadow execution studies. Preserve more than the latest 40 ICT events. These recommendations follow concrete limitations above; no runtime changes were made by this audit.",
        "", "Artifacts: `setup_events.csv` is one row per unique recovered setup; `setup_model_probabilities.csv` has all same-bar model outputs, including off-direction/family/type rows explicitly flagged; `setup_family_model_agreement.csv` and `setup_audit_summary.json` summarize coverage; `setup_audit.py` reproduces extraction from the read-only frozen snapshot.",
    ])
    (HERE / "setup_findings.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
