"""Publish B3 protocol v3 (v2 carried verbatim plus the approved section-3 simulation bindings).

v1, v2 and their locks are never edited.  Every v2 section is copied
unchanged; the surfaces, ledgers, per-ledger limits, capacity rule and
smoke-replay window rule come from the approved 2026-10-02 draft.  Refuses to
overwrite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_testing.b3_comparison import build_b3_comparison_plan, load_json, sha256_file
from model_testing.outcome_contracts import CATALOG
from model_testing.outcome_routes import ROUTE_BINDINGS_V2

PROTOCOL_V1 = ROOT / "configs/b3_b4_candidate_review_v1_20260929.json"
PROTOCOL_V2 = ROOT / "configs/b3_b4_candidate_review_v2_20261002.json"
DRAFT_V3 = ROOT / "configs/b3_b4_candidate_review_v3_draft_20261002.json"
PROTOCOL_V3 = ROOT / "configs/b3_b4_candidate_review_v3_20261002.json"
LOCK_V3 = ROOT / "configs/b3_b4_candidate_review_v3_20261002.lock.json"
ROSTER = ROOT / "configs/broad_full_session_shadow_selection_v1.json"
SELECTION_V2 = ROOT / "configs/broad_full_session_shadow_v2.json"
FREEZE_V2 = ROOT / "configs/b4_broad_full_session_shadow_freeze_v2.json"

STREAM_KEYS = ("calendar", "per_ledger_limits", "capacity", "cost_stress")
CARRIED = ("outcome_inspection_guard", "scope", "comparison_branches", "streams", "cross_stream_rules",
           "experimental_variants", "predeclared_outputs", "promotion")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def build() -> dict:
    v2, draft = load_json(PROTOCOL_V2), load_json(DRAFT_V3)
    if draft["parent_protocol_sha256"] != sha256_file(PROTOCOL_V2):
        raise ValueError("Draft was written against a different v2 protocol")
    freeze = load_json(FREEZE_V2)
    status = {str(row["queue_id"]): row["policy"]["full_policy"] for row in freeze["frozen_replacements"]}
    if any(not value.startswith("not_defined") for value in status.values()):
        raise ValueError("B4 freeze declares a full policy that this protocol records as not_defined")
    declared = {**draft["full_policy_status"]["ES"], **draft["full_policy_status"]["EURUSD"]}
    if set(declared) != set(status):
        raise ValueError("Full-policy status roster does not match the B4 freeze")
    if draft["roster"]["selection_order"] != v2["scope"]["selected_recipe_roster"]["candidate_ids"]:
        raise ValueError("Selection order does not match the v2 roster")
    smoke = json.loads(json.dumps(draft["smoke_replay"]))
    smoke["window_rule"]["cap_weeks_per_model"] = 4
    smoke["window_rule"]["candidate_weeks"] = "Restricted to the model's held-out period with complete source bars."
    smoke["eurusd_calendar_rule"] = ("EURUSD smoke replay is not_run until one versioned EURUSD 5m expected-open "
                                     "calendar is pinned; closures are never inferred from bars.")
    ledgers = json.loads(json.dumps(draft["ledgers"]))
    ledgers["instrument_capacity_rule"] = {
        "feed": "The intended full policy where exactly defined, otherwise setup_threshold; the feeding variant is labeled on every entry.",
        "priority": "Earliest decision time (known_at); ties broken by frozen selection order, then opportunity key.",
        "selection_order": draft["roster"]["selection_order"],
        "blocking": "An open or unresolved position blocks every other entry on that instrument, same side or opposite, until it closes. No pyramiding, no reversal, zero cooldown. A same-bar exit does not free capacity at that bar's open.",
        "supersedes": "For the instrument-capacity ledger only, this priority replaces the tie-breakers in streams.*.execution_assumptions.portfolio.capacity_rule.",
        "engine": "Implemented as an optional ordering key in the shared position simulator, published as b1-v5; b1-v4 stays reproducible from archived engine source.",
    }
    ledgers["independent_model_variant_rule"] = "One position at a time, one unit, no opposite side, zero cooldown inside each ledger."
    return {
        "protocol_id": "b3_b4_candidate_review_v3_20261002",
        "protocol_version": 3,
        "status": v2["status"],
        "declared_at": "2026-10-02",
        "parent_protocol": rel(PROTOCOL_V2),
        "parent_protocol_sha256": sha256_file(PROTOCOL_V2),
        "purpose": draft["purpose"],
        "change_from_v2": [
            "Every v2 section is carried verbatim.",
            "Adds the four persisted surfaces, ledger kinds and naming, per-ledger limits, the instrument-capacity rule, per-model full-policy status and the outcome-blind smoke-replay window rule.",
            "Records the ES session-clock alias; no behavior changes.",
        ],
        **{key: v2[key] for key in CARRIED},
        "section3_roster": draft["roster"],
        "section3_stream_bindings": {name: {key: stream[key] for key in STREAM_KEYS}
                                     for name, stream in draft["streams"].items()},
        "surfaces": draft["surfaces"],
        "full_policy_status": {**draft["full_policy_status"],
                               "rule": "All 16 stay not_defined during section 3. An exact binding needs per-model policy manifests and a new B4 freeze version."},
        "ledgers": ledgers,
        "smoke_replay": smoke,
        "session_coverage": draft["session_coverage"],
        "session_clock_alias": {
            "london_07_16_ny_08_17_asia_ny_v1": "london_07_16_ny_08_17_asia_ny_19_02_v1",
            "rule": "Both spellings name the single frozen clock implemented by outcome_engine.session_at.",
        },
        "approved_decisions": {"approved_at": "2026-10-02", "outcome": "as_recommended",
                               "items": draft["decisions_needing_approval"]},
    }


def main() -> None:
    for path in (PROTOCOL_V3, LOCK_V3):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen publication: {path}")
    protocol = build()
    PROTOCOL_V3.write_bytes((json.dumps(protocol, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    plan = build_b3_comparison_plan(load_json(PROTOCOL_V3), load_json(ROSTER), selection_path=ROSTER)
    lock = {
        protocol["protocol_id"]: sha256_file(PROTOCOL_V3),
        "b3_b4_candidate_review_v2_20261002": sha256_file(PROTOCOL_V2),
        "b3_b4_candidate_review_v1_20260929": sha256_file(PROTOCOL_V1),
        "broad_full_session_shadow_selection_v1": sha256_file(ROSTER),
        "broad_full_session_shadow_v2": sha256_file(SELECTION_V2),
        "b4_broad_full_session_shadow_freeze_v2": sha256_file(FREEZE_V2),
        "broad_full_session_shadow_route_bindings_v2": sha256_file(ROUTE_BINDINGS_V2),
        "b1_v3": sha256_file(CATALOG),
    }
    LOCK_V3.write_bytes((json.dumps(lock, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({
        "protocol_sha256": lock[protocol["protocol_id"]],
        "lock_sha256": sha256_file(LOCK_V3),
        "candidate_count": plan["candidate_count"],
        "streams": plan["streams"],
    }, indent=2))


if __name__ == "__main__":
    main()
