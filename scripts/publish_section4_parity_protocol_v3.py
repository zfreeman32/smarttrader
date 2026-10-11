"""Publish and lock section 4 live-parity protocol v3 (user approved 2026-10-05, decision Q-P1 option (b)).

v3 = locked v2 plus one declared-behavior clause for float32 tree-split score differences. Numeric tolerances,
segments, groups, stages, scenarios, verdict rules and the report schema are unchanged (asserted).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs/section4_parity_protocol_v2.json"
V2_LOCK = ROOT / "configs/section4_parity_protocol_v2.lock.json"
OUT = ROOT / "configs/section4_parity_protocol_v3.json"
LOCK = ROOT / "configs/section4_parity_protocol_v3.lock.json"
DATE = "2026-10-05"
CLAUSE = (
    "declared_contract_behavior/float32_tree_split: an S6 difference outside S6_scores is declared behavior at an endpoint "
    "only if (a) every S1-S5 cell of that model at that endpoint (all consumed rows) is within tolerance, and (b) rebuilding "
    "the model input from each side's recorded S5 rows and predicting with the hash-verified frozen model reproduces that "
    "side's recorded raw score exactly. Cause: the frozen source_history.parquet stores float32 values computed on the "
    "offline path (model_training/source_history.py casts to float32) while the app computes float64 and casts, so inputs "
    "can differ by about one float32 ULP (inside S3/S5 tolerance) and a tree split can sit between the two values. "
    "An S7 decision flip at such an endpoint is never covered by this clause: it still counts as a flip (near-threshold "
    "band rule unchanged).")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    lock = json.loads(V2_LOCK.read_text(encoding="utf-8"))
    if sha(V2) != lock["sha256"]:
        raise ValueError("Protocol v2 does not match its lock")
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    v3 = copy.deepcopy(v2)
    v3.update(version=3, status="locked", approved_at=DATE,
              approval="user 2026-10-05: decision Q-P1 answered '(b)' (recommendation in docs/section4_progress.md, prompt 4)",
              supersedes={"protocol": "configs/section4_parity_protocol_v2.json", "sha256": lock["sha256"]})
    v3["inputs"]["protocol_v2"] = {"path": "configs/section4_parity_protocol_v2.json", "sha256": lock["sha256"]}
    v3["inputs"]["protocol_v2_lock"] = {"path": "configs/section4_parity_protocol_v2.lock.json", "sha256": sha(V2_LOCK)}
    v3["mismatch_taxonomy"]["declared_contract_behavior"]["float32_tree_split"] = CLAUSE
    v3["changes_from_v2"] = {
        "decision": "Q-P1",
        "reason": ("Run s4parity_20261002_v1 found S6-only differences (IDs 12 and 15; 7 endpoints on H windows, no decision "
                   "flips) whose S1-S5 inputs were within tolerance; each side's recorded S5 rows reproduced its own score "
                   "exactly, and the inputs differed by one float32 ULP between the float32 frozen history and the app's "
                   "float64 path. v2 had no class for this, so it was 'unexplained'. Approved after seeing results, with "
                   "this reason recorded, as the protocol's change_control requires."),
        "changed_keys": ["mismatch_taxonomy.declared_contract_behavior.float32_tree_split"],
        "unchanged": "tolerances (numeric), segments, scenarios, groups, stages, verdicts, report schema, retention",
    }
    if v3["tolerances"] != v2["tolerances"]:
        raise AssertionError("v3 must not change numeric tolerances")
    return v3


def main() -> int:
    if OUT.exists() or LOCK.exists():
        raise SystemExit("Protocol v3 already published; publish a new version instead.")
    payload = protocol()
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOCK.write_text(json.dumps({"path": "configs/section4_parity_protocol_v3.json", "sha256": sha(OUT), "locked_at": DATE,
                                "inputs_sha256": {k: v["sha256"] for k, v in payload["inputs"].items()}}, indent=2) + "\n",
                    encoding="utf-8")
    print("PUBLISHED", sha(OUT), sha(LOCK))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
