"""Build versioned frozen-selection views for retained B2 jobs, preserving old artifacts."""
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

from model_testing.artifact_lineage_audit import digest
from preprocessing.config import PreprocessingConfig
from scripts.rebuild_b2_prepared import ProgressPipeline
from scripts.materialize_frvp_recency_prepared_root import load_source_timestamps, reweight_split

PRIORITY_IDS = (2, 1, 11, 17)
ROSTER = ROOT / "docs/audits/b2_repaired_model_data_roster_20260921.csv"
ROSTER_SHA256 = "e4da1f4d83dbd3ccc3baf44033f552b9e6a48e81b7a768b8926018ad0de4525a"
DEFAULT_OUTPUT = ROOT / "artifacts/b2_validation_prepared_20260922_v1"
PRIORITY_MANIFEST = DEFAULT_OUTPUT / "manifest.json"
PRIORITY_MANIFEST_SHA256 = "3c2ddda7dcf50ecce087475112e863fd7fe42ae8bc808d30172303ad7c4becad"
LATER_IDS = tuple(i for i in range(1, 28) if i not in PRIORITY_IDS)


def selection_boundaries(jobs):
    """Shared OTE meta/union targets use the earliest recipe's training boundary."""
    boundaries = {}
    for _, old, _, saved in jobs:
        boundaries[old.name] = min(boundaries.get(old.name, int(saved["cv_initial_train_rows"])),
                                   int(saved["cv_initial_train_rows"]))
    return boundaries


def preserve_recency(summary, dataset, new_root, target):
    weighting = summary.get("recency_weighting")
    if not weighting:
        if target == "long_frvp_reversal":
            raise ValueError("FRVP long reversal requires its saved recency weighting.")
        return None
    if target != "long_frvp_reversal" or weighting["half_life_days"] != 730 or weighting["floor"] != .20:
        raise ValueError("Unexpected recency contract; review before preparation.")
    timestamps = load_source_timestamps(dataset)
    if timestamps.max().isoformat() != weighting["latest_ts_utc"]:
        raise ValueError("Recency reference timestamp changed.")
    result = dict(weighting, output_prepared_root=str(new_root), source_dataset=str(dataset), splits={})
    for split in ("train", "val"):
        result["splits"][split] = reweight_split(split_path=new_root / target / f"{split}.csv",
            timestamps=timestamps, latest_ts=timestamps.max(), half_life_days=730, floor=.20)
    write_json(new_root / target / "recency_weighting_summary.json", result)
    new_summary = json.loads((new_root / "summary.json").read_text())
    new_summary["recency_weighting"] = result
    write_json(new_root / "summary.json", new_summary)
    return dict(half_life_days=730, floor=.20, latest_ts_utc=weighting["latest_ts_utc"],
                weighted_splits=["train", "val"], test_weights_unchanged=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(output: Path, group="priority") -> dict:
    output = output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        raise ValueError("Choose a new output version under artifacts; existing runs are preserved.")
    if digest(ROSTER) != ROSTER_SHA256:
        raise ValueError("Retained roster changed; review model IDs before preparing.")
    rows = list(csv.DictReader(ROSTER.open()))
    groups = {}
    for model_id in (PRIORITY_IDS if group == "priority" else LATER_IDS):
        row = rows[model_id - 1]
        old = ROOT / row["prepared_dir"]
        recipe = ROOT / row["artifact_path"] / "training_summary.json"
        saved = json.loads(recipe.read_text())["config"]
        groups.setdefault(old.parent, []).append((model_id, old, recipe, saved))
    output.mkdir(parents=True)
    manifest = dict(status="preparing", roster_sha256=digest(ROSTER), models={}, parents={},
                    trading_model_training_run=False, trusted_baseline=False)
    if group == "later":
        if digest(PRIORITY_MANIFEST) != PRIORITY_MANIFEST_SHA256:
            raise ValueError("Priority preparation changed; review before extending lineage.")
        priority = json.loads(PRIORITY_MANIFEST.read_text())
        manifest["models"].update(priority["models"])
        manifest["parents"].update(priority["parents"])
        manifest["parents"][str(PRIORITY_MANIFEST.relative_to(ROOT))] = PRIORITY_MANIFEST_SHA256
    manifest["preparation_code_sha256"] = {name: digest(ROOT / name) for name in (
        "scripts/prepare_b2_validation_views.py", "preprocessing/pipeline.py", "preprocessing/config.py",
        "preprocessing/feature_selection.py", "preprocessing/feature_importance.py",
        "scripts/materialize_frvp_recency_prepared_root.py")}
    write_json(output / "manifest.json", manifest)
    old_validation = json.loads((ROOT / "docs/audits/b2_repair_validation_20260921.json").read_text())["targets"]
    for old_root, jobs in groups.items():
        summary_path = old_root / "summary.json"
        summary = json.loads(summary_path.read_text())
        dataset, metadata = Path(summary["input_file"]), Path(summary["metadata_file"])
        parent_manifest = dataset.parent / "manifest.json"
        expected_parent = json.loads(parent_manifest.read_text())
        for path, expected in ((dataset, expected_parent["dataset_sha256"]),
                               (metadata, expected_parent["metadata_sha256"])):
            if digest(path) != expected:
                raise ValueError(f"Corrected pre-selection parent changed: {path}")
        cfg = PreprocessingConfig(**summary["config"])
        cfg.target_columns = list(dict.fromkeys("label_" + old.name for _, old, _, _ in jobs))
        cfg.target_selection_fit_rows = selection_boundaries(jobs)
        new_root = output / old_root.parent.name / "prepared"
        ProgressPipeline(cfg).run(dataset, new_root, metadata_path=metadata)
        recency = {old.name: preserve_recency(summary, dataset, new_root, old.name)
                   for old in dict.fromkeys(job[1] for job in jobs)}
        for path in (dataset, metadata, summary_path, parent_manifest):
            manifest["parents"][str(path.relative_to(ROOT))] = digest(path)
        for model_id, old, recipe, saved in jobs:
            target = new_root / old.name
            previous = old_validation[str(old.relative_to(ROOT))]["hashes"]
            for name, expected in previous.items():
                if digest(old / name) != expected:
                    raise ValueError(f"Historical prepared file changed: {old / name}")
                manifest["parents"][str((old / name).relative_to(ROOT))] = expected
            for split in ("train", "val", "test"):
                columns = ["source_row_idx", "target", "sample_weight"]
                before = pd.read_csv(old / f"{split}.csv", usecols=columns)
                after = pd.read_csv(target / f"{split}.csv")
                pd.testing.assert_frame_equal(before[columns], after[columns], check_exact=True)
                if not np.isfinite(after.to_numpy()).all():
                    raise ValueError(f"Nonfinite prepared values: {target}/{split}")
            manifest["models"][str(model_id)] = dict(
                prepared_dir=str(target.relative_to(ROOT)), old_prepared_dir=str(old.relative_to(ROOT)),
                recipe=str(recipe.relative_to(ROOT)), recipe_sha256=digest(recipe),
                prepared_hashes={p.name: digest(p) for p in target.iterdir() if p.is_file()},
                summary_sha256=digest(new_root / "summary.json"),
                split_identity_labels_weights_unchanged=True,
                selection_fit_rows=cfg.target_selection_fit_rows[old.name],
                recency_weighting=recency[old.name],
            )
        write_json(output / "manifest.json", manifest)
    manifest["status"] = "prepared_and_checked"
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group", choices=("priority", "later"), default="priority")
    args = parser.parse_args()
    result = prepare(args.output_root, args.group)
    print(f"Prepared and checked lineage for {len(result['models'])} model jobs.", flush=True)
