"""One-off: publish route bindings v2 (v1 routes carried forward verbatim + IDs 5/23/18/14)."""
import json
import sys

sys.path.insert(0, ".")
from model_testing.outcome_contracts import CATALOG, ROOT, file_hash, load_contracts
from model_testing.outcome_routes import ROUTE_BINDINGS_V1, ROUTE_BINDINGS_V2, load_route_bindings, route_binding_audit

v1 = json.loads(ROUTE_BINDINGS_V1.read_text(encoding="utf-8"))
load_route_bindings(ROUTE_BINDINGS_V1)  # v1 must still load against its lock and pins
contracts = load_contracts()
selection = ROOT / "configs/broad_full_session_shadow_v2.json"
lock = json.loads((ROOT / "configs/b4_broad_full_session_shadow_freeze_v2.lock.json").read_text(encoding="utf-8"))
assert file_hash(selection) == lock["selection_file_sha256"]
assert file_hash(ROOT / lock["freeze"]) == lock["freeze_file_sha256"]

by_id = {b["queue_id"]: b for b in v1["bindings"]}
OTE = by_id[21]["setup_bindings"]
new = [
    {"queue_id": 5, "model_id": "frvp_long_continuation_setup2_xgb_v1", "instrument": "ES", "strategy": "frvp",
     "side": "long", "model_family": "continuation_setup2", "route_selection": "exact_contemporaneous_same_side",
     "setup_bindings": {"setup2": "es_frvp_setup2_eval_v2"}},
    {"queue_id": 23, "model_id": "long_union_tcn_champion", "instrument": "EURUSD", "strategy": "ote",
     "side": "long", "model_family": "union", "route_selection": by_id[21]["route_selection"],
     "setup_bindings": dict(OTE)},
    {"queue_id": 18, "model_id": "ict_short_reversal_ifvg_reversal_xgb_v1", "instrument": "ES", "strategy": "ict",
     "side": "short", "model_family": "reversal_ifvg_reversal", "route_selection": by_id[19]["route_selection"],
     "setup_bindings": {"ifvg_reversal": "es_ict_ifvg_reversal_eval_v1"}},
    {"queue_id": 14, "model_id": "ict_short_continuation_xgb_v1", "instrument": "ES", "strategy": "ict",
     "side": "short", "model_family": "continuation", "route_selection": by_id[16]["route_selection"],
     "setup_bindings": {
         "premium_discount_continuation": "es_ict_premium_discount_continuation_eval_v1",
         "displacement_continuation_after_raid": "es_ict_displacement_continuation_after_raid_eval_v1"}},
]
for binding in new:
    for contract_id in binding["setup_bindings"].values():
        assert contract_id in contracts, contract_id

v2 = {
    "binding_id": "broad_full_session_shadow_route_bindings_v2",
    "version": 2,
    "selected_collection_id": "broad_full_session_shadow_v2",
    "selection_path": "configs/broad_full_session_shadow_v2.json",
    "selection_sha256": file_hash(selection),
    "catalog_path": v1["catalog_path"],
    "catalog_sha256": file_hash(CATALOG),
    "catalog_lock_sha256": file_hash(CATALOG.with_suffix(".lock.json")),
    "pooled_routing_contract": v1["pooled_routing_contract"],
    "bindings": [*v1["bindings"], *new],
}
assert (v2["catalog_sha256"], v2["catalog_lock_sha256"]) == (v1["catalog_sha256"], v1["catalog_lock_sha256"])
assert v2["bindings"][:12] == v1["bindings"]
lock_path = ROUTE_BINDINGS_V2.with_suffix(".lock.json")
for path in (ROUTE_BINDINGS_V2, lock_path):
    if path.exists():
        raise FileExistsError(path)
ROUTE_BINDINGS_V2.write_bytes((json.dumps(v2, indent=2) + "\n").encode("utf-8"))
lock_path.write_bytes((json.dumps({v2["binding_id"]: file_hash(ROUTE_BINDINGS_V2)}, indent=2) + "\n").encode("utf-8"))

for path in (ROUTE_BINDINGS_V1, ROUTE_BINDINGS_V2):
    manifest = load_route_bindings(path)
    audit = route_binding_audit(manifest)
    streams = {}
    for row in audit["rows"]:
        key = f"{row['instrument']}/{row['strategy']}"
        streams[key] = streams.get(key, 0) + 1
    print(path.name, file_hash(path), "content", manifest.sha256, "lock", file_hash(path.with_suffix(".lock.json")),
          "models", audit["selected_model_count"], "routes", audit["route_count"], streams)
