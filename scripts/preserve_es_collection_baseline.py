"""Freeze existing audit evidence and model bytes without opening the live database."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ote_live.storage.collection import COLLECTION_CONTRACT, file_sha256


def preserve(audit: Path, destination: Path, *, root: Path = ROOT) -> dict:
    if not (audit / "current_model_semantics.csv").is_file():
        raise FileNotFoundError(f"Audit roster not found: {audit}")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace frozen baseline: {destination}")
    destination.mkdir(parents=True)
    objects = destination / "objects"
    objects.mkdir()
    files: dict = {}

    def capture(path: Path, label: str) -> None:
        if not path.is_file():
            files[label] = {"missing": True, "source": str(path)}
            return
        digest = file_sha256(path)
        target = objects / digest
        if not target.exists():
            shutil.copyfile(path, target)
        if file_sha256(target) != digest:
            raise RuntimeError(f"Snapshot copy failed verification: {label}")
        files[label] = {"source": str(path), "sha256": digest, "bytes": path.stat().st_size,
                        "object": "objects/" + digest}

    for path in sorted(audit.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            capture(path, "audit/" + path.relative_to(audit).as_posix())
    with (audit / "current_model_semantics.csv").open(encoding="utf-8", newline="") as handle:
        roster = list(csv.DictReader(handle))
    manifests = []
    for row in roster:
        path = root / row["manifest_path"]
        if not path.is_file():
            capture(path, "manifest/" + row["model_id"])
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        candidates = document.get("models", [document])
        payload = next((item for item in candidates if item["model_id"] == row["model_id"]), None)
        if payload is None:
            raise ValueError(f"Missing model in manifest: {row['model_id']}")
        manifests.append(payload)
        capture(path, "manifest/" + row["model_id"])
        for name, reference in payload["artifact_references"].items():
            if name.endswith("_file") and reference:
                capture(root / reference, "artifact/" + row["model_id"] + "/" + name)
        for name in ("registry_path", "policy_backtest_summary_path"):
            if payload.get(name):
                capture(root / payload[name], "policy/" + row["model_id"] + "/" + name)
        for name, reference in payload.get("live_policy", {}).get("lineage", {}).items():
            if name.endswith("_path") and reference:
                capture(root / reference, "policy_lineage/" + row["model_id"] + "/" + name)
    # Preserve the actual input/policy implementation alongside the model bytes.
    for directory in ("features", "frvp", "ict", "ote_live/features", "ote_live/contracts",
                      "ote_live/ingestion", "ote_live/models", "ote_live/policies", "ote_live/storage"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".sql"}:
                capture(path, "source/" + path.relative_to(root).as_posix())
    for reference in ("model_testing/ote_threshold_policy.py", "model_testing/ote_policy_backtest.py",
                      "scripts/ote_targeted_filter_presets.py", "ote_live/scripts/run_es_live_collector.py"):
        if (root / reference).is_file():
            capture(root / reference, "source/" + reference)
    result = {"baseline_id": destination.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "collection_version": "legacy-unversioned", "qualified_period_started": False,
              "next_collection_contract": COLLECTION_CONTRACT,
              "activation": "shadow-only; existing paper-trial restrictions preserved",
              "artifact_capture_note": "Current bytes at preservation time; audit-era byte identity requires B2 review.",
              "files": files, "manifests": manifests}
    with (destination / "baseline_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    (destination / "README.md").write_text(
        "# Frozen ES baseline\n\nContent-addressed copies of the September 11 audit, its frozen SQLite snapshot, "
        "and available baseline model, calibration, manifest and policy artifacts.\n\n"
        "`baseline_manifest.json` maps names to SHA-256-verified objects. Do not overwrite this directory. "
        "These legacy observations are diagnostic and precede the corrected input contract. "
        "Artifact lineage remains subject to B2; no qualified collection or paper trial is started.\n", encoding="utf-8")
    return result


def verify(destination: Path) -> dict:
    """Verify frozen objects without depending on mutable source files."""
    manifest = json.loads((destination / "baseline_manifest.json").read_text(encoding="utf-8"))
    missing = []
    for label, item in manifest["files"].items():
        if item.get("missing"):
            missing.append(label)
            continue
        path = destination / item["object"]
        if not path.is_file() or file_sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Frozen baseline integrity failure: {label}")
    return {"baseline": str(destination), "files": len(manifest["files"]),
            "models": len(manifest["manifests"]), "missing": missing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "research/es_live_audit_20260911")
    parser.add_argument("--destination", type=Path, default=ROOT / "artifacts/es_collection_baseline_20260912")
    parser.add_argument("--verify", action="store_true", help="Verify an existing snapshot without changing it")
    args = parser.parse_args()
    if not args.verify:
        preserve(args.audit, args.destination)
    result = verify(args.destination)
    print(json.dumps(result))
    if result["missing"]:
        sys.exit(1)
