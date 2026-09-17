"""Inventory required HTF/helper inputs without loading weights or starting collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ote_live.features.input_contract import INPUT_CONTRACT_VERSION


DEFAULT_BUNDLES = (
    ("frvp_es_shadow_20260721", ("long", "short")),
    ("frvp_es_setup_family_20260829", ("long",)),
    ("ict_es_paper_signal_20260813", ("long", "short")),
    ("ict_short_setup_family_20260811_audit", ("short",)),
)


def audit_dependencies(paths: list[Path]) -> dict:
    models = []
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(raw)
        for manifest in payload.get("models", [payload]):
            features = manifest["feature_manifest"]["selected_feature_names"]
            corrected = [name for name in features if name.startswith(("htf_", "ict_"))]
            helpers = [name for name in features if name.startswith("htf_confluence_")]
            version = manifest["feature_manifest"].get("input_contract_version")
            reasons = []
            if corrected and version != INPUT_CONTRACT_VERSION:
                reasons.append("corrected_feature_lineage_unverified")
            if helpers:
                reasons.append("live_helper_producer_not_packaged")
            models.append({
                "model_id": manifest["model_id"], "backend": manifest["backend"],
                "manifest_status": manifest["status"],
                "retired": manifest["status"] in {"deprecated", "retired"},
                "manifest_path": path.resolve().relative_to(ROOT).as_posix(),
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "input_contract_version": version, "selected_feature_count": len(features),
                "htf_features": [name for name in features if name.startswith("htf_")],
                "external_helper_features": helpers, "corrected_feature_count": len(corrected),
                "static_blockers": reasons,
                "status": "diagnostic_only" if reasons else "requires_runtime_input_check",
            })
    return {"contract_version": INPUT_CONTRACT_VERSION,
            "scope": "Configured ES family and specialist bundles; static dependency inventory, not artifact parity or live qualification.",
            "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [ROOT / "ote_live/runtime_manifests" / bundle / f"live_runtime_manifest_{side}.json"
             for bundle, sides in DEFAULT_BUNDLES for side in sides]
    payload = audit_dependencies(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(payload['models'])} manifests; wrote {args.output}")


if __name__ == "__main__":
    main()
