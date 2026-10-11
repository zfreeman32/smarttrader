"""Retrain R2c part 1: frozen-prefix validation views on the feature contract v2 prepared roots.

Mirrors ``scripts/prepare_b2_validation_views.py`` (v1, unchanged) for the 16 roster models in the v2 roster
(``docs/audits/b2_v2_model_data_roster_20261006.csv``): per prepared root, the preprocessing recipe reruns with the
per-target selection boundary pinned from the v1 manifest, so fills, low-variance/collinearity pruning and the
importance ranking are fit on the first ``selection_fit_rows`` training rows only. Checks per model: split identity
(source_row_idx, target, sample_weight) equals both the R2b v2 split and the v1 validation view; all values finite;
FRVP long reversal keeps its 730-day/0.20 recency weighting. On success it writes a lock with the manifest hash.

Usage: python scripts/prepare_b2_v2_validation_views.py [--output-root artifacts/b2_validation_prepared_v2_20261006]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
from model_testing.artifact_lineage_audit import digest  # noqa: E402
from preprocessing.config import PreprocessingConfig  # noqa: E402
from scripts.prepare_b2_validation_views import preserve_recency, write_json  # noqa: E402
from scripts.rebuild_b2_prepared import ProgressPipeline  # noqa: E402

ROSTER = ROOT / "docs/audits/b2_v2_model_data_roster_20261006.csv"
ROSTER_SHA256 = "8d50d873f376c91def632db9eff3217bc3cdfa2c9398d12c822faf641e72e5ae"
DEFAULT_OUTPUT = ROOT / "artifacts/b2_validation_prepared_v2_20261006"
LOCK = ROOT / "configs/b2_v2_validation_views_lock_20261006.json"
SPLIT_KEYS = ["source_row_idx", "target", "sample_weight"]
CODE = ("scripts/prepare_b2_v2_validation_views.py", "scripts/prepare_b2_validation_views.py", "preprocessing/pipeline.py",
        "preprocessing/config.py", "preprocessing/feature_selection.py", "preprocessing/feature_importance.py",
        "scripts/materialize_frvp_recency_prepared_root.py", "scripts/build_b2_v2_roster.py")


def same_split(left: Path, right: Path, split: str) -> None:
    before = pd.read_csv(left / f"{split}.csv", usecols=SPLIT_KEYS)
    after = pd.read_csv(right / f"{split}.csv", usecols=SPLIT_KEYS)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def prepare(output: Path, only: set[str] | None = None) -> dict:
    output = output.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Use a version under artifacts.")
    if digest(ROSTER) != ROSTER_SHA256:
        raise ValueError("v2 roster changed; review before preparing.")
    rows = list(csv.DictReader(ROSTER.open(encoding="utf-8")))
    groups: dict[Path, list[dict]] = {}
    for row in rows:
        if digest(ROOT / row["recipe"]) != row["recipe_sha256"]:
            raise ValueError(f"Recipe changed: {row['recipe']}")
        groups.setdefault((ROOT / row["v2_prepared_dir"]).parent, []).append(row)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():  # resume: completed roots are kept, never rebuilt
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] == "prepared_and_checked" or manifest["roster_sha256"] != ROSTER_SHA256:
            raise ValueError("Output is complete or from another roster; use a new version.")
    else:
        output.mkdir(parents=True)
        manifest = dict(status="preparing", roster=str(ROSTER.relative_to(ROOT)), roster_sha256=ROSTER_SHA256,
                        feature_contract="v2", models={}, parents={}, trading_model_training_run=False,
                        trusted_baseline=False, preparation_code_sha256={p: digest(ROOT / p) for p in CODE})
        write_json(manifest_path, manifest)
    if manifest["preparation_code_sha256"] != {p: digest(ROOT / p) for p in CODE}:
        raise ValueError("Preparation code changed since this version started.")
    for old_root, jobs in groups.items():
        name = old_root.parent.name
        if only and name not in only:
            continue
        if all(row["model_id"] in manifest["models"] for row in jobs):
            print(f"{name}: already prepared", flush=True)
            continue
        new_root = output / name / "prepared"
        if new_root.exists():
            raise ValueError(f"Partial view root exists, delete it before resuming: {new_root}")
        summary_path = old_root / "summary.json"
        summary = json.loads(summary_path.read_text())
        dataset, metadata = Path(summary["input_file"]), Path(summary["metadata_file"])
        parent_manifest = dataset.parent / "manifest.json"
        expected = json.loads(parent_manifest.read_text())
        for path, sha in ((dataset, expected["dataset_sha256"]), (metadata, expected["metadata_sha256"])):
            if digest(path) != sha:
                raise ValueError(f"Prepared parent changed: {path}")
        targets = list(dict.fromkeys(row["target"] for row in jobs))
        fit_rows = {}
        for row in jobs:
            if fit_rows.setdefault(row["target"], int(row["selection_fit_rows"])) != int(row["selection_fit_rows"]):
                raise ValueError(f"Models sharing {row['target']} disagree on the selection boundary.")
        cfg = PreprocessingConfig(**summary["config"])
        cfg.target_columns = ["label_" + target for target in targets]
        cfg.target_selection_fit_rows = fit_rows
        print(f"{name}: preparing {targets} with fit rows {fit_rows}", flush=True)
        ProgressPipeline(cfg).run(dataset, new_root, metadata_path=metadata)
        recency = {target: preserve_recency(summary, dataset, new_root, target) for target in targets}
        for path in (dataset, metadata, summary_path, parent_manifest):
            manifest["parents"][str(path.relative_to(ROOT))] = digest(path)
        for row in jobs:
            target = new_root / row["target"]
            r2b, v1_view = ROOT / row["v2_prepared_dir"], ROOT / row["v1_validation_prepared_dir"]
            for file in sorted(p for p in r2b.iterdir() if p.is_file()):
                manifest["parents"][str(file.relative_to(ROOT))] = digest(file)
            for split in ("train", "val", "test"):
                same_split(r2b, target, split)
                same_split(v1_view, target, split)
                values = pd.read_csv(target / f"{split}.csv")
                if not np.isfinite(values.to_numpy(dtype=float)).all():
                    raise ValueError(f"Nonfinite prepared values: {target}/{split}")
            contract = json.loads((target / "report.json").read_text())["selection_contract"]
            if contract.get("mode") != "frozen_development_prefix" or contract["fit_rows"] != int(row["selection_fit_rows"]):
                raise ValueError(f"Frozen selection contract mismatch for model {row['model_id']}")
            manifest["models"][row["model_id"]] = dict(
                prepared_dir=str(target.relative_to(ROOT)), old_prepared_dir=row["v2_prepared_dir"],
                v1_validation_prepared_dir=row["v1_validation_prepared_dir"], recipe=row["recipe"],
                recipe_sha256=row["recipe_sha256"], prepared_hashes={p.name: digest(p) for p in target.iterdir() if p.is_file()},
                summary_sha256=digest(new_root / "summary.json"), split_identity_labels_weights_unchanged=True,
                split_identity_equals_v1_validation_view=True, selection_fit_rows=int(row["selection_fit_rows"]),
                recency_weighting=recency[row["target"]])
            print(f"  model {row['model_id']} {row['target']}: split identity == R2b v2 == v1 view; finite", flush=True)
        write_json(manifest_path, manifest)
    if set(manifest["models"]) == {row["model_id"] for row in rows}:
        manifest["status"] = "prepared_and_checked"
        write_json(manifest_path, manifest)
        lock = dict(manifest=str(manifest_path.relative_to(ROOT)), manifest_sha256=digest(manifest_path),
                    roster=str(ROSTER.relative_to(ROOT)), roster_sha256=ROSTER_SHA256, models=sorted(manifest["models"], key=int))
        with LOCK.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(lock, indent=2) + "\n")
        print(f"VALIDATION VIEWS COMPLETE: {len(manifest['models'])} models; lock {LOCK.relative_to(ROOT)}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", nargs="*", help="Prepared root names to run (default: all)")
    args = parser.parse_args()
    prepare(args.output_root, set(args.only) if args.only else None)
