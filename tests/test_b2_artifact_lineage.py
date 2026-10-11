from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_testing.artifact_lineage_audit import Inventory, build_inventory, feature_base, feature_routes, write_outputs


def save(root: Path, path: str, data: object) -> Path:
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def fixture_model(root: Path, model="candidate", *, prepared="data/prepared/base/long_ict_continuation"):
    artifact = "models/" + model
    save(root, artifact + "/training_summary.json", {
        "target": "long_ict_continuation", "prepared_dir": prepared,
        "selected_feature_names": ["ict_prior_week_high_dist_atr_lag_2", "close"],
        "source_lineage": {"feature_csv": "data/features/features.csv"},
    })
    save(root, artifact + "/model_config.json", {"selected_feature_names": ["ict_prior_week_high_dist_atr_lag_2", "close"]})
    save(root, artifact + "/model.json", {"weight": 1})
    save(root, prepared + "/features.json", {"features": ["ict_prior_week_high_dist_atr_lag_2", "close"]})
    save(root, prepared + "/report.json", {})
    save(root, str(Path(prepared).parent / "summary.json"), {"input_file": "data/features/features.csv"})
    save(root, "data/features/features.metadata.json", {
        "source_path": "data/raw.csv", "prepared_contract": {"label_source_path": "data/labeling/labeled_data/labels.csv"},
    })
    for path in ["data/features/features.csv", "data/raw.csv", "data/labeling/labeled_data/labels.csv", prepared + "/train.csv"]:
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("source_row_idx,target\n0,1\n", encoding="utf-8")
    return artifact


def test_nested_transforms_and_non_prefix_dependencies():
    assert feature_base("ict_rth_open_dist_atr_lag_2_roll_mean_20") == "ict_rth_open_dist_atr"
    assert "ICT_REFERENCE_DIRECT" in feature_routes("ict_rth_open_dist_atr_lag_2_roll_mean_20")
    assert "HTF_FEATURE_PUBLICATION" in feature_routes("continuation_bull_retest_quality_zscore_20")
    assert "HTF_FEATURE_PUBLICATION" in feature_routes("interaction_newyork_bull_setup")
    assert "LIVE_CONFLUENCE_PRODUCER_MISMATCH" in feature_routes("htf_confluence_long_frvp_continuation_lag_1")


@pytest.mark.parametrize("name", ["ict_bos_bull", "ict_choch_bull", "ict_displacement_score", "ict_in_ote_band", "dist_to_bull_fvg_atr", "dist_to_bull_order_block_atr"])
def test_shared_producer_argument_is_not_enough_to_claim_reference_dependency(name):
    assert not any(r.startswith("ICT_REFERENCE") for r in feature_routes(name))


def test_distinct_artifacts_are_not_collapsed_by_model_id(tmp_path):
    a, b = fixture_model(tmp_path, "a"), fixture_model(tmp_path, "b")
    save(tmp_path, "models/example_registry.json", {"models": [
        {"model_id": "same_id", "artifact_path": a}, {"model_id": "same_id", "artifact_path": b},
        {"model_id": "alias", "artifact_path": a},
    ]})
    result = build_inventory(tmp_path)
    assert result["summary"]["model_bundles"] == 2
    assert result["summary"]["registrations"] == 3
    assert all(m["baseline_clearance"] is False for m in result["models"])
    assert result["models"][0]["observed_bundle_sha256"] == result["models"][1]["observed_bundle_sha256"]


def test_nested_label_source_is_traced_and_inputs_unchanged(tmp_path):
    fixture_model(tmp_path)
    before = {str(p):p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = build_inventory(tmp_path)
    model = result["models"][0]
    assert "data/labeling/labeled_data/labels.csv" in model["label_source_paths"]
    assert "historical_training_label_hash_or_version_missing" in model["evidence_gaps"]
    assert all(p.read_bytes() == before[str(p)] for p in tmp_path.rglob("*") if p.is_file())
    output = tmp_path / "audit"
    write_outputs(result, output)
    with pytest.raises(FileExistsError):
        write_outputs(result, output)


def test_missing_exact_source_is_not_replaced_with_similarly_named_file(tmp_path):
    fixture_model(tmp_path)
    (tmp_path / "data/features/features.csv").unlink()
    (tmp_path / "data/features/features.csv.gz").write_bytes(b"not-proof-of-original-bytes")
    result = build_inventory(tmp_path, hash_files=False)
    assert "data/features/features.csv" in result["models"][0]["missing_ancestor_paths"]
    assert result["models"][0]["baseline_clearance"] is False


def test_recency_parent_and_external_research_model_are_included(tmp_path):
    artifact = fixture_model(tmp_path)
    summary = json.loads((tmp_path / artifact / "training_summary.json").read_text())
    save(tmp_path, "model_testing/reports/frvp_placebo_readouts/shuffle/training_summary.json", summary)
    save(tmp_path, "data/prepared/base/long_ict_continuation/recency_weighting_summary.json", {
        "base_prepared_root": "data/prepared/original", "source_dataset": "data/features/features.csv",
    })
    result = build_inventory(tmp_path, hash_files=False)
    assert result["summary"]["model_bundles"] == 2
    assert result["summary"]["research_placebo_bundles"] == 1
    assert any(e["relationship"] == "recency_materialization_parent" for e in result["edges"])


def test_foreign_paths_not_read(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "foreign"
    save(outside, "training_summary.json", {"secret": "must_not_read"})
    save(root, "models/foreign_registry.json", {"models": [{"model_id": "foreign", "artifact_path": str(outside)}]})
    result = build_inventory(root, hash_files=False)
    assert result["models"][0]["model_files"] == []
    assert "must_not_read" not in json.dumps(result)


def test_capture_objects_resolve_relative_to_their_manifest(tmp_path):
    manifest = save(tmp_path, "artifacts/snapshot/baseline_manifest.json", {"files": {
        "model": {"source": "models/old/model.pt", "object": "objects/abc", "sha256": "abc"},
    }})
    obj = tmp_path / "artifacts/snapshot/objects/abc"
    obj.parent.mkdir()
    obj.write_bytes(b"snapshot")
    inv = Inventory(tmp_path)
    inv.document(manifest)
    assert inv.nodes["artifacts/snapshot/objects/abc"]["exists"]
    assert "objects/abc" not in inv.nodes


def test_temporary_trained_smoke_is_included_but_metadata_fixture_is_not(tmp_path):
    save(tmp_path, "tmp/smoke/training_summary.json", {"target": "long_frvp_reversal"})
    save(tmp_path, "tmp/smoke/model.json", {"trees": []})
    save(tmp_path, "tmp/test_fixture/training_summary.json", {"target": "long_reversal"})
    result = build_inventory(tmp_path, hash_files=False)
    assert result["summary"]["model_bundles"] == 1
    assert result["summary"]["temporary_saved_training_bundles"] == 1
    assert result["discovery_exclusions"] == [{"path": "tmp/test_fixture/training_summary.json", "reason": "temporary_metadata_without_saved_model_weights"}]
