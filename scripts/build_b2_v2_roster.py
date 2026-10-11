"""Retrain R2c: the v2 model data roster (16 section 4 roster models -> feature contract v2 prepared roots).

Each row keeps the model's saved recipe and frozen selection boundary from the pinned v1 validation manifest
(``artifacts/b2_validation_prepared_20260922_v2``) and swaps only the prepared root: v1 R2b root -> v2 R2b root
(``*_prepared_v2_20261006``, same labels/events/recipes, v2 feature tapes). Refuses to overwrite.

Usage: python scripts/build_b2_v2_roster.py [--output docs/audits/b2_v2_model_data_roster_20261006.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_testing.artifact_lineage_audit import digest  # noqa: E402

V1_MANIFEST = ROOT / "artifacts/b2_validation_prepared_20260922_v2/manifest.json"
V1_MANIFEST_SHA256 = "e99d03f09d543efdf809e3c0a315e85bc1c0be6ae463223f821327cd2e2b1936"
ROSTER_IDS = (2, 3, 5, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22, 23, 25, 26)
V2_ROOTS = {
    "b2_frvp_july_prepared_20260921_v1": "b2_frvp_july_prepared_v2_20261006",
    "b2_frvp_recency_prepared_20260921_v1": "b2_frvp_recency_prepared_v2_20261006",
    "b2_frvp_setups_prepared_20260921_v1": "b2_frvp_setups_prepared_v2_20261006",
    "b2_ict_prepared_20260921_v1": "b2_ict_prepared_v2_20261006",
    "b2_ict_specialists_prepared_20260921_v1": "b2_ict_specialists_prepared_v2_20261006",
    "b2_ote_prepared_20260921_v1": "b2_ote_prepared_v2_20261006",
}
FIELDS = ("model_id", "stream", "target", "backend", "recipe", "recipe_sha256", "selection_fit_rows",
          "v1_prepared_dir", "v1_validation_prepared_dir", "v2_prepared_dir")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "docs/audits/b2_v2_model_data_roster_20261006.csv")
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Roster exists; versions are immutable.")
    if digest(V1_MANIFEST) != V1_MANIFEST_SHA256:
        raise SystemExit("Pinned v1 validation manifest changed.")
    v1 = json.loads(V1_MANIFEST.read_text())["models"]
    rows = []
    for model_id in ROSTER_IDS:
        job = v1[str(model_id)]
        old = Path(job["old_prepared_dir"])
        root = old.parts[1]
        new = Path("artifacts") / V2_ROOTS[root] / Path(*old.parts[2:])
        recipe = ROOT / job["recipe"]
        if digest(recipe) != job["recipe_sha256"]:
            raise SystemExit(f"Recipe changed for model {model_id}: {recipe}")
        if not (ROOT / new / "train.csv").exists():
            raise SystemExit(f"Missing v2 prepared target for model {model_id}: {new}")
        config = json.loads(recipe.read_text())["config"]
        backend = config["backend"] if config["backend"] != "torch" else config.get("model_type", "torch")
        rows.append(dict(model_id=model_id, stream="EURUSD" if root.startswith("b2_ote") else "ES", target=old.name,
                         backend=backend, recipe=job["recipe"], recipe_sha256=job["recipe_sha256"],
                         selection_fit_rows=job["selection_fit_rows"], v1_prepared_dir=str(old),
                         v1_validation_prepared_dir=job["prepared_dir"], v2_prepared_dir=str(new)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output.relative_to(ROOT)} ({len(rows)} models) sha256 {digest(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
