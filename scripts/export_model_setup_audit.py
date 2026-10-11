"""Export a single live collection read-only, without filtering scores or decisions."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_testing.outcome_contracts import content_hash, file_hash


def export_collection(database: Path, output_dir: Path, *, collection: str) -> dict:
    # URI mode=ro avoids migrations, pragmas that write, and accidental DB creation.
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        read = lambda query, args=(): [dict(row) for row in db.execute(query, args)]
        predictions = read("SELECT * FROM model_predictions WHERE collection_version=? ORDER BY id", (collection,))
        if not predictions:
            raise ValueError("Collection has no persisted predictions; no inferred collection fallback")
        snapshots = read("SELECT * FROM feature_snapshots WHERE id IN (SELECT feature_snapshot_id FROM model_predictions WHERE collection_version=?) ORDER BY id", (collection,))
        decisions = read("SELECT * FROM signal_decisions WHERE prediction_id IN (SELECT id FROM model_predictions WHERE collection_version=?) ORDER BY id", (collection,))
        setups = read("SELECT * FROM setup_events WHERE collection_version=? ORDER BY id", (collection,))
        links = read("SELECT * FROM setup_prediction_links WHERE prediction_id IN (SELECT id FROM model_predictions WHERE collection_version=?) ORDER BY id", (collection,))
        manifests = read("SELECT * FROM runtime_manifests WHERE id IN (SELECT runtime_manifest_id FROM model_predictions WHERE collection_version=?) ORDER BY id", (collection,))
        frames = {"model_predictions": predictions, "feature_snapshots": snapshots,
                  "signal_decisions": decisions, "setup_events": setups,
                  "setup_prediction_links": links, "runtime_manifests": manifests}
        frames["collection_versions"] = read("SELECT * FROM collection_versions WHERE collection_version=?", (collection,))
        # Raw snapshots support future response reconstruction but do not prove
        # exchange-calendar coverage or historical availability by themselves.
        source_times = pd.to_datetime([p["timestamp_utc"] for p in predictions], utc=True)
        start, end = source_times.min() - pd.Timedelta(days=7), source_times.max()
        assets = {(f["asset"], f["timeframe"]) for f in snapshots}
        bars, gaps = [], []
        for asset, timeframe in sorted(assets):
            instrument_bars = read("SELECT * FROM canonical_bars WHERE asset=? AND timeframe=? AND timestamp_utc>=? AND timestamp_utc<=? ORDER BY timestamp_utc",
                                   (asset, timeframe, start.isoformat(), end.isoformat()))
            # Preserve observed follow-up rows across weekends/session closures;
            # their existence still does not establish that expected bars are complete.
            instrument_bars += read("SELECT * FROM canonical_bars WHERE asset=? AND timeframe=? AND timestamp_utc>? ORDER BY timestamp_utc LIMIT 60",
                                    (asset, timeframe, end.isoformat()))
            bars.extend(instrument_bars)
            last_bar = instrument_bars[-1]["timestamp_utc"] if instrument_bars else end.isoformat()
            gaps.extend(read("SELECT * FROM ingestion_gaps WHERE asset=? AND timeframe=? AND observed_timestamp_utc>=? AND expected_timestamp_utc<=? ORDER BY id",
                             (asset, timeframe, start.isoformat(), last_bar)))
        frames["canonical_bars_snapshot"] = bars
        frames["ingestion_gaps"] = gaps
        snapshots_by_id = {row["id"]: row for row in snapshots}
        setups_by_id = {row["id"]: row for row in setups}
        manifests_by_id = {row["id"]: row for row in manifests}
        links_by_prediction, decisions_by_prediction = {}, {}
        for row in links:
            links_by_prediction.setdefault(row["prediction_id"], []).append(row)
        for row in decisions:
            decisions_by_prediction.setdefault(row["prediction_id"], []).append(row)
        observations = []
        for prediction in predictions:
            source = snapshots_by_id[prediction["feature_snapshot_id"]]
            manifest = manifests_by_id.get(prediction["runtime_manifest_id"], {})
            events = decisions_by_prediction.get(prediction["id"], [])
            decision_payload = json.loads(events[-1]["signal_json"]) if events else {}
            shadow = decision_payload.get("shadow_evaluation") or {}
            regimes = (shadow.get("model_setup_audit_context") or {}).get("regimes", {})
            model_payload = json.loads(prediction["prediction_json"])
            base = dict(collection_id=collection, prediction_id=prediction["id"],
                model_id=prediction["model_id"], model_manifest_hash=manifest.get("manifest_hash"),
                policy_hash=shadow.get("policy_sha256"), side=prediction["direction"],
                instrument=source["asset"], timeframe=source["timeframe"],
                source_timestamp=prediction["timestamp_utc"], prediction_recorded_at=model_payload.get("prediction_recorded_at_utc"),
                raw_score=prediction["raw_score"], calibrated_probability=prediction["calibrated_probability"],
                threshold=prediction["threshold_applied"], threshold_source=prediction["threshold_source"],
                prediction_regime=prediction["regime"], snapshot_hash=content_hash(source),
                threshold_crossed=shadow.get("threshold_crossed"),
                full_policy_passed=shadow.get("full_policy_passed"),
                rejection_reasons=json.dumps(shadow.get("rejection_reasons", decision_payload.get("reasons", []))),
                context_status="recorded" if regimes else "UNKNOWN_LEGACY",
                forward_response_status="NOT_EVALUATED_requires_B1_calendar_and_causal_inputs",
                **{name: regimes.get(name) for name in ("trend_regime", "volatility_regime", "composite_regime", "session_regime", "stress_regime")})
            candidates = links_by_prediction.get(prediction["id"], [])
            if not candidates:
                observations.append(dict(
                    base, relationship="no_persisted_setup_link", setup_event_id=None,
                    routing_contract=None, routing_role="no_persisted_setup_link",
                    primary_executable_setup_event_id=None, counterfactual_markout_only=False,
                    later_setup_not_attached=False, executable_route=False,
                ))
            for link in candidates:
                event = setups_by_id[link["setup_event_id"]]
                same_side = event["setup_side"] == (1 if prediction["direction"] == "long" else -1)
                link_payload = json.loads(link["payload_json"])
                shadow_evaluation = link_payload.get("shadow_evaluation") or {}
                setup_match = shadow_evaluation.get("setup_match") or {}
                candidate = next((item for item in setup_match.get("candidates", [])
                                  if item.get("event_id") == event["id"]), {})
                prediction_time = model_payload.get("prediction_recorded_at_utc")
                availability = ("unknown" if prediction_time is None else
                                "not_known_at_prediction" if pd.Timestamp(event["observed_at_utc"]) > pd.Timestamp(prediction_time)
                                else "known_at_prediction")
                # A later event is never relabeled as the prediction's setup.
                # Keep legacy relationship names for known rows while exposing
                # the explicit executable/counterfactual routing role.
                relationship = ("not_known_at_prediction" if availability != "known_at_prediction" else
                                "selected_same_side" if same_side and event["selected"] else
                                "other_same_side" if same_side else "opposite_side")
                routing_role = candidate.get("routing_role")
                if routing_role is None:
                    routing_role = ("counterfactual_same_side" if same_side and availability == "known_at_prediction"
                                    and not event["selected"] else "unknown")
                observations.append(dict(base, setup_event_id=event["id"], setup_strategy=event["strategy"],
                    setup_type=event["setup_type"], setup_family=event["setup_family"], setup_side=event["setup_side"],
                    setup_selected=bool(event["selected"]), setup_revision=event["revision"],
                    detector_version=event["detector_version"], setup_observed_at=event["observed_at_utc"],
                    source_bar_version=event["source_bar_version"], setup_event_kind=event["event_kind"],
                    shadow_setup_matched=link_payload.get("shadow_setup_matched"),
                    relationship_availability=availability,
                    relationship=relationship,
                    routing_contract=setup_match.get("routing_contract"),
                    routing_role=routing_role,
                    primary_executable_setup_event_id=setup_match.get("primary_event_id"),
                    counterfactual_markout_only=(
                        routing_role == "counterfactual_same_side"
                        or event["id"] in setup_match.get("counterfactual_event_ids", [])
                    ),
                    later_setup_not_attached=(
                        routing_role == "not_known_at_decision"
                        or event["id"] in setup_match.get("later_event_ids", [])
                    ),
                    executable_route=(
                        routing_role == "primary_executable"
                        or event["id"] == setup_match.get("primary_event_id")
                    )))
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in frames.items():
        with (output_dir / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    pd.DataFrame(observations).to_csv(output_dir / "model_setup_live_observations.csv", index=False)
    report = dict(schema_version="model_setup_live_export_v1", collection_id=collection,
        prediction_count=len(predictions), setup_count=len(setups), link_count=len(links),
        no_link_prediction_count=sum(p["id"] not in links_by_prediction for p in predictions),
        table_counts={key: len(value) for key, value in frames.items()},
        limitations=["No-link is not proof of no setup; inspect detector availability.",
                     "Observed models are not proof the expected model roster was fully evaluated.",
                     "Raw bars are latest database snapshots, not causal replay attestations.",
                     "No performance ranking or trade P&L is inferred; use B1 forward outcomes and explicit denominators."],
        artifact_hashes={p.name: file_hash(p) for p in sorted(output_dir.iterdir())})
    report["export_hash"] = content_hash(report)
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_collection(args.database, args.output_dir, collection=args.collection), indent=2))


if __name__ == "__main__":
    main()
