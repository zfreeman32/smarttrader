"""Publish and verify the reviewed broad full-session shadow replacement freeze.

This is an identity freeze, not an installation step.  It refuses to publish
until every selected core replacement has a passing artifact review, then
binds the actual model/preprocessor/calibrator/feature bytes and the evidence
used to reproduce folds, OOF threshold selection, and production-runner
decisions.  Later B3-B7 gates remain fail-closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SELECTION_V0 = ROOT / "configs/broad_full_session_shadow_selection_v0.json"
SELECTION_V1 = ROOT / "configs/broad_full_session_shadow_v1.json"
FREEZE_V1 = ROOT / "configs/b4_broad_full_session_shadow_freeze_v1.json"
LOCK_V1 = ROOT / "configs/b4_broad_full_session_shadow_freeze_v1.lock.json"
AUDIT_V1 = ROOT / "docs/audits/broad_full_session_shadow_v1_verification.json"
REPORT_V1 = ROOT / "docs/B4_broad_full_session_shadow_freeze_v1_20261001.md"
ROSTER_V1 = ROOT / "configs/broad_full_session_shadow_selection_v1.json"
SELECTION_V2 = ROOT / "configs/broad_full_session_shadow_v2.json"
FREEZE_V2 = ROOT / "configs/b4_broad_full_session_shadow_freeze_v2.json"
LOCK_V2 = ROOT / "configs/b4_broad_full_session_shadow_freeze_v2.lock.json"
AUDIT_V2 = ROOT / "docs/audits/broad_full_session_shadow_v2_verification.json"
REPORT_V2 = ROOT / "docs/B4_broad_full_session_shadow_freeze_v2_20261001.md"

SELECTED_IDS = (2, 10, 12, 13, 15, 16, 22, 26, 3, 19, 21, 25)
# Latency-approved extension, in v0 reserve then diagnostic-companion order.
EXTENSION_IDS = (5, 23, 18, 14)
MODEL_ROOTS = {
    2: ROOT / "models/b2_candidate_history_20260928_v1",
    3: ROOT / "models/b2_candidate_history_minimum_core_es_20260929_v1",
    10: ROOT / "models/b2_candidate_history_minimum_core_es_20260929_v1",
    12: ROOT / "models/b2_candidate_history_minimum_core_es_20260929_v1",
    13: ROOT / "models/b2_candidate_history_minimum_core_es_20260929_v1",
    15: ROOT / "models/b2_candidate_history_minimum_core_es_20260929_v1",
    16: ROOT / "models/b2_candidate_history_minimum_core_es_20260929_v1",
    19: ROOT / "models/b2_candidate_history_minimum_core_es_20260929_v1",
    22: ROOT / "models/b2_candidate_history_minimum_core_ote_20260930_v3",
    26: ROOT / "models/b2_candidate_history_minimum_core_ote_20260930_v3",
    21: ROOT / "models/b2_candidate_history_expanded_core_ote_20261001_v1",
    25: ROOT / "models/b2_candidate_history_expanded_core_ote_20261001_v1",
    5: ROOT / "models/b2_candidate_history_catchup_es_20261001_v1",
    14: ROOT / "models/b2_candidate_history_catchup_es_20261001_v1",
    18: ROOT / "models/b2_candidate_history_catchup_es_20261001_v1",
    23: ROOT / "models/b2_candidate_history_promoted_reserve_ote_20261001_v1",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def rendered_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_match(parent: Path, pattern: str) -> Path:
    matches = sorted(parent.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern!r} below {parent}, found {len(matches)}")
    return matches[0]


def review_path(model_id: int) -> Path:
    if model_id in EXTENSION_IDS:
        return unique_match(ROOT / "docs/audits", f"b2_catchup_{model_id}_review_[0-9]*.json")
    return unique_match(ROOT / "docs/audits", f"b2_candidate_history_review_{model_id:02d}_*.json")


def candidate_dir(model_id: int) -> Path:
    root = MODEL_ROOTS[model_id]
    if not root.is_dir():
        raise FileNotFoundError(f"Selected replacement root is missing for ID {model_id}: {root}")
    return unique_match(root, f"{model_id:02d}_*")


def training_manifest(candidate: Path) -> Path:
    choices = [
        candidate / "candidate_history_training_manifest.json",
        candidate / "candidate_ote_training_manifest.json",
    ]
    matches = [path for path in choices if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one replacement training manifest below {candidate}")
    return matches[0]


def artifact_binding(model_id: int, selected: dict[str, Any]) -> dict[str, Any]:
    review_file = review_path(model_id)
    review = load_json(review_file)
    allowed_status = {
        "candidate_history_artifact_review_passed",
        "candidate_ote_history_artifact_review_passed",
    }
    if review.get("status") not in allowed_status:
        raise ValueError(f"ID {model_id} has no passing replacement review")
    required_true = (
        "threshold_decisions_equal",
        "OOF_threshold_reproduced",
        "scaler_fit_scope_verified",
        "selected_feature_order_verified",
    )
    if not all(review.get(name) is True for name in required_true):
        raise ValueError(f"ID {model_id} review is missing a required exact check")
    if not (review.get("purged_holdout_and_cv_verified") is True
            or review.get("purged_holdout_and_original_cv_verified") is True):
        raise ValueError(f"ID {model_id} review did not verify folds and purge gaps")

    candidate = candidate_dir(model_id)
    run_file = training_manifest(candidate)
    run = load_json(run_file)
    expected_run_status = {
        "trained_pending_candidate_history_review",
        "trained_pending_candidate_ote_history_review",
    }
    if run.get("status") not in expected_run_status or int(run.get("model_id")) != model_id:
        raise ValueError(f"ID {model_id} replacement training manifest is incomplete")
    if file_sha256(run_file) != review.get("training_manifest_sha256"):
        raise ValueError(f"ID {model_id} reviewed a different training manifest")

    target = str(review["target"])
    model_dir = candidate / target
    summary_file = model_dir / "training_summary.json"
    summary = load_json(summary_file)
    if summary.get("target") != target:
        raise ValueError(f"ID {model_id} target changed after review")
    if summary.get("selected_feature_names") is None:
        raise ValueError(f"ID {model_id} has no frozen selected-feature order")
    if float(summary["threshold"]) != float(review["threshold"]):
        raise ValueError(f"ID {model_id} threshold differs from its review")

    output_hashes = dict(review["model_outputs_sha256"])
    for name, expected in output_hashes.items():
        path = model_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"ID {model_id} reviewed output changed: {name}")
    model_name = "model.json" if (model_dir / "model.json").is_file() else "model.pt"
    if model_name not in output_hashes:
        raise ValueError(f"ID {model_id} model artifact was not covered by its review")
    scaler_name = "scaler.joblib"
    if scaler_name not in output_hashes:
        raise ValueError(f"ID {model_id} scaler artifact was not covered by its review")

    calibration = dict(summary["calibration"])
    resolved_method = str(calibration["resolved_method"])
    calibrator_hash = output_hashes.get("calibrator.joblib")
    if resolved_method != "none" and not calibrator_hash:
        raise ValueError(f"ID {model_id} calibrated score lacks a reviewed calibrator")
    if resolved_method != "none" and "calibrated_prediction_max_abs_error" not in review:
        raise ValueError(f"ID {model_id} calibrated replay lacks a calibrated-score error")

    features = list(summary["selected_feature_names"])
    fold_contract = {
        "cv_splitter": summary["cv_splitter"],
        "resolved_purge_bars": summary["resolved_purge_bars"],
        "label_horizon_assumptions": summary["label_horizon_assumptions"],
        "final_holdout": summary["final_holdout"],
    }
    artifact_hashes = {
        "model": output_hashes[model_name],
        "scaler": output_hashes[scaler_name],
        "calibrator": calibrator_hash,
        "model_config": output_hashes["model_config.json"],
        "training_summary": output_hashes["training_summary.json"],
        "oof_predictions": output_hashes["oof_predictions.csv"],
        "test_predictions": output_hashes["test_predictions.csv"],
        "cv_fold_manifest_json": output_hashes["cv_fold_manifest.json"],
        "cv_fold_manifest_csv": output_hashes["cv_fold_manifest.csv"],
    }
    source_contract = run.get("source_history_contract") or summary["model_config"].get("source_history_contract")
    if not source_contract:
        raise ValueError(f"ID {model_id} has no frozen source-history contract")
    prepared_root = Path(run["config"]["prepared_root"])
    prepared_root = prepared_root if prepared_root.is_absolute() else ROOT / prepared_root
    prepared_manifest_file = prepared_root.parent / "manifest.json"
    if file_sha256(prepared_manifest_file) != review["prepared_manifest_sha256"]:
        raise ValueError(f"ID {model_id} prepared manifest changed after review")
    prepared_manifest = load_json(prepared_manifest_file)
    source_history_file = prepared_root / target / "source_history.parquet"
    if file_sha256(source_history_file) != source_contract["source_history_sha256"]:
        raise ValueError(f"ID {model_id} source-history bytes changed after review")
    if prepared_manifest.get("source_dataset"):
        source_dataset = Path(prepared_manifest["source_dataset"])
        source_dataset = source_dataset if source_dataset.is_absolute() else ROOT / source_dataset
        if file_sha256(source_dataset) != prepared_manifest["source_dataset_sha256"]:
            raise ValueError(f"ID {model_id} pinned source dataset changed after review")
    corrected_publication = run.get("corrected_publication")
    if corrected_publication:
        for path_key, hash_key in (
            ("feature_artifact", "feature_artifact_sha256"),
            ("feature_metadata", "feature_metadata_sha256"),
            ("label_artifact", "label_artifact_sha256"),
            ("label_metadata", "label_metadata_sha256"),
        ):
            pinned_file = ROOT / corrected_publication[path_key]
            if file_sha256(pinned_file) != corrected_publication[hash_key]:
                raise ValueError(f"ID {model_id} corrected publication pin changed: {path_key}")
    code_pins = dict(run.get("code_sha256", {}))
    if not code_pins:
        raise ValueError(f"ID {model_id} has no training code snapshot pins")
    for name, expected in code_pins.items():
        snapshot = candidate / "code_snapshot" / name
        if file_sha256(snapshot) != expected:
            raise ValueError(f"ID {model_id} code snapshot changed after review: {name}")

    return {
        "queue_id": model_id,
        "model_id": selected["model_id"],
        "target": target,
        "instrument": selected["instrument"],
        "side": selected["side"],
        "family": selected["family"],
        "order_group": selected["order_group"],
        "artifact_root": relative(model_dir),
        "artifacts": {
            "model_file": model_name,
            "scaler_file": scaler_name,
            "calibrator_file": "calibrator.joblib" if calibrator_hash else None,
            "sha256": artifact_hashes,
            "reviewed_artifact_set_sha256": canonical_sha256(artifact_hashes),
        },
        "source_data_pins": {
            "b2_recipe_and_data_parent": selected["b2"],
            "prepared_manifest_sha256": review["prepared_manifest_sha256"],
            "prepared_manifest": relative(prepared_manifest_file),
            "source_history_contract": source_contract,
            "source_history_contract_sha256": canonical_sha256(source_contract),
            "corrected_publication": corrected_publication,
        },
        "code_pins": {
            "snapshot_sha256": code_pins,
            "snapshot_set_sha256": canonical_sha256(code_pins),
        },
        "features": {
            "ordered_names": features,
            "count": len(features),
            "ordered_names_sha256": canonical_sha256(features),
        },
        "folds_and_purge": {
            **fold_contract,
            "contract_sha256": canonical_sha256(fold_contract),
            "review_verified": True,
        },
        "calibration": {
            **calibration,
            "calibrator_sha256": calibrator_hash,
            "selection_source": review.get("calibration_selection_source", "development_OOF_only"),
        },
        "threshold": {
            "value": float(review["threshold"]),
            "value_hex": float(review["threshold"]).hex(),
            "comparison": "calibrated_probability >= threshold",
            "selection_source": "development_OOF_only",
            "oof_reproduced": True,
            "exact_saved_decisions_replayed": True,
        },
        "production_runner_replay": {
            "review": relative(review_file),
            "review_sha256": file_sha256(review_file),
            "prediction_rows": int(review["prediction_rows"]),
            "raw_score_max_abs_error": float(review["prediction_max_abs_error"]),
            "calibrated_score_max_abs_error": float(
                review.get("calibrated_prediction_max_abs_error", review["prediction_max_abs_error"])
            ),
            "threshold_decisions_equal": True,
        },
        "quality_metrics": {
            "treatment": "recorded_advisory_only_not_a_research_collection_veto",
            "oof_summary": summary.get("oof_summary"),
            "threshold_metrics": summary.get("threshold_metrics"),
            "test_metrics": summary.get("test_metrics"),
        },
        "policy": {
            "setup_only": "predeclared",
            "model_threshold": "frozen_by_this_record",
            "full_policy": "not_defined_pending_exact_B3_trading_rule_binding",
            "full_policy_ledger_allowed": False,
            "historical_policy_substitution_allowed": False,
        },
        "gate_status": {
            "replacement_review": "passed",
            "b5_live_feature_parity": "pending",
            "b6_staging": "pending",
            "b7_readiness": "pending",
            "scored_collection_eligible": False,
            "paper_or_order_authority": False,
        },
    }


def payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    selection_v0 = load_json(SELECTION_V0)
    candidates = {int(item["queue_id"]): item for item in selection_v0["candidates"]}
    if tuple(model_id for group in (
        selection_v0["order"]["minimum_core_order"],
        selection_v0["order"]["expanded_core_if_latency_permits_order"],
    ) for model_id in group) != SELECTED_IDS:
        raise ValueError("v0 selected core order changed; publish a new version deliberately")
    bindings = [artifact_binding(model_id, candidates[model_id]) for model_id in SELECTED_IDS]
    selection = {
        "selection_id": "broad_full_session_shadow_v1",
        "selection_version": 1,
        "declared_at": "2026-10-01",
        "status": "replacement_artifacts_frozen_pending_B3_through_B7",
        "parent_selection": relative(SELECTION_V0),
        "parent_selection_sha256": file_sha256(SELECTION_V0),
        "purpose": "Bind every selected core replacement to reviewed immutable bytes without granting collection, paper, alert, or order authority.",
        "selected_order": list(SELECTED_IDS),
        "instrument_separation": selection_v0["instrument_separation"],
        "comparison_protocol_intent": selection_v0["comparison_protocol_intent"],
        "collection_controls": selection_v0["collection_controls"],
        "replacement_bindings": bindings,
        "admission_rule": "B2 review freezes identity only. Scored shadow admission remains fail-closed until exact B3/B4 trading rules, B5 parity, B6 staging, A12 latency, and B7 readiness pass for these same hashes.",
        "legacy_swap_rule": "Any byte, feature order, calibrator, threshold, matcher, or policy change requires a new selection/freeze version and clean collection boundary.",
    }
    freeze = {
        "freeze_id": "b4_broad_full_session_shadow_freeze_v1",
        "freeze_version": 1,
        "declared_at": "2026-10-01",
        "status": "frozen_replacement_identity_pending_policy_and_readiness_gates",
        "selection": relative(SELECTION_V1),
        "selection_sha256": hashlib.sha256(rendered_bytes(selection)).hexdigest(),
        "parent_b4": "configs/b4_baseline_freeze_20260929.json",
        "parent_b4_sha256": file_sha256(ROOT / "configs/b4_baseline_freeze_20260929.json"),
        "scope": "12 selected minimum-plus-expanded core replacements across separate ES and EURUSD research streams",
        "frozen_replacements": bindings,
        "threshold_decision_rule": "Use the exact per-binding float and calibrated_probability >= threshold; equality is accepted.",
        "policy_status": "full_policy_not_defined_for_all_selected_replacements_pending_section_3_binding",
        "undefined_policy_rule": "Do not create a full-policy ledger, report zero P&L, claim a pass, or substitute a legacy policy while status is not_defined.",
        "historical_quality_rule": "Recorded classifier and economic quality remains advisory for this research collection and did not veto a technically reviewed replacement.",
        "authority": {
            "research_identity_freeze": True,
            "scored_collection": False,
            "paper": False,
            "alerts": False,
            "broker_orders": False,
        },
        "change_control": "Never overwrite or silently replace frozen bytes. Publish a new version and clean boundary for any binding change.",
    }
    return selection, freeze


def payloads_v2() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the 16-model roster, selection and freeze; the 12 v1 entries carry forward unchanged."""
    selection_v0 = load_json(SELECTION_V0)
    candidates = {int(item["queue_id"]): item for item in selection_v0["candidates"]}
    lock_v1 = load_json(LOCK_V1)
    if (file_sha256(SELECTION_V1) != lock_v1["selection_file_sha256"]
            or file_sha256(FREEZE_V1) != lock_v1["freeze_file_sha256"]):
        raise ValueError("Published v1 selection/freeze no longer matches its lock")
    frozen_v1 = load_json(FREEZE_V1)["frozen_replacements"]
    carried = [artifact_binding(model_id, candidates[model_id]) for model_id in SELECTED_IDS]
    for old, new in zip(frozen_v1, carried, strict=True):
        if canonical_bytes(old) != canonical_bytes(new):
            raise ValueError(f"ID {new['queue_id']} bytes on disk differ from its v1 freeze entry")
    extension = [artifact_binding(model_id, candidates[model_id]) for model_id in EXTENSION_IDS]
    bindings = carried + extension
    selected_ids = [*SELECTED_IDS, *EXTENSION_IDS]

    def roster_entry(binding: dict[str, Any]) -> dict[str, Any]:
        model_id = int(binding["queue_id"])
        replay = binding["production_runner_replay"]
        return {
            **candidates[model_id],
            "selection_role": "core" if model_id in SELECTED_IDS else "latency_approved_extension",
            "review_evidence": {
                "review": replay["review"],
                "review_sha256": replay["review_sha256"],
                "status": "reviewed_pass",
                "artifact_root": binding["artifact_root"],
                "prepared_manifest": binding["source_data_pins"]["prepared_manifest"],
                "prepared_manifest_sha256": binding["source_data_pins"]["prepared_manifest_sha256"],
                "reviewed_artifact_set_sha256": binding["artifacts"]["reviewed_artifact_set_sha256"],
            },
        }

    roster = {
        "selection_id": "broad_full_session_shadow_selection_v1",
        "selection_version": 1,
        "declared_at": "2026-10-01",
        "status": "selection_with_reviewed_replacement_artifacts_pending_B3_through_B7",
        "parent_selection": relative(SELECTION_V0),
        "parent_selection_sha256": file_sha256(SELECTION_V0),
        "purpose": "Extend the research-only broad full-session shadow roster from the 12 core models to 16 by adding the latency-approved extension IDs 5/23/14/18 after their B2 backfill reviews passed.",
        "order": {
            "core_order": list(SELECTED_IDS),
            "latency_approved_extension_order": list(EXTENSION_IDS),
            "selected_order": selected_ids,
        },
        "instrument_separation": selection_v0["instrument_separation"],
        "comparison_protocol_intent": selection_v0["comparison_protocol_intent"],
        "collection_controls": selection_v0["collection_controls"],
        "candidates": [roster_entry(binding) for binding in bindings],
        "authority": "Selection only. No collection, registry, paper, alert, broker, or promotion authority.",
    }
    selection = {
        "selection_id": "broad_full_session_shadow_v2",
        "selection_version": 2,
        "declared_at": "2026-10-01",
        "status": "replacement_artifacts_frozen_pending_B3_through_B7",
        "parent_selection": relative(ROSTER_V1),
        "parent_selection_sha256": hashlib.sha256(rendered_bytes(roster)).hexdigest(),
        "purpose": "Bind every selected core and extension replacement to reviewed immutable bytes without granting collection, paper, alert, or order authority.",
        "selected_order": selected_ids,
        "instrument_separation": selection_v0["instrument_separation"],
        "comparison_protocol_intent": selection_v0["comparison_protocol_intent"],
        "collection_controls": selection_v0["collection_controls"],
        "replacement_bindings": bindings,
        "admission_rule": "B2 review freezes identity only. Scored shadow admission remains fail-closed until exact B3/B4 trading rules, B5 parity, B6 staging, A12 latency, and B7 readiness pass for these same hashes.",
        "legacy_swap_rule": "Any byte, feature order, calibrator, threshold, matcher, or policy change requires a new selection/freeze version and clean collection boundary.",
    }
    freeze = {
        "freeze_id": "b4_broad_full_session_shadow_freeze_v2",
        "freeze_version": 2,
        "declared_at": "2026-10-01",
        "status": "frozen_replacement_identity_pending_policy_and_readiness_gates",
        "selection": relative(SELECTION_V2),
        "selection_sha256": hashlib.sha256(rendered_bytes(selection)).hexdigest(),
        "parent_b4": relative(FREEZE_V1),
        "parent_b4_sha256": file_sha256(FREEZE_V1),
        "scope": "16 selected replacements (12 core carried forward unchanged from v1 plus extension IDs 5/23/18/14) across separate ES and EURUSD research streams",
        "frozen_replacements": bindings,
        "threshold_decision_rule": "Use the exact per-binding float and calibrated_probability >= threshold; equality is accepted.",
        "policy_status": "full_policy_not_defined_for_all_selected_replacements_pending_section_3_binding",
        "undefined_policy_rule": "Do not create a full-policy ledger, report zero P&L, claim a pass, or substitute a legacy policy while status is not_defined.",
        "historical_quality_rule": "Recorded classifier and economic quality remains advisory for this research collection and did not veto a technically reviewed replacement.",
        "authority": {
            "research_identity_freeze": True,
            "scored_collection": False,
            "paper": False,
            "alerts": False,
            "broker_orders": False,
        },
        "change_control": "Never overwrite or silently replace frozen bytes. Publish a new version and clean boundary for any binding change.",
    }
    return roster, selection, freeze


def write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen publication: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered_bytes(value))


def markdown(selection: dict[str, Any], freeze: dict[str, Any]) -> str:
    rows = []
    for item in selection["replacement_bindings"]:
        rows.append(
            f"| {item['queue_id']} | `{item['model_id']}` | {item['instrument']} | "
            f"{item['calibration']['resolved_method']} | {item['threshold']['value']:.17g} | "
            f"{item['features']['count']} | `{item['artifacts']['sha256']['model'][:12]}` | "
            f"{item['policy']['full_policy']} |"
        )
    return "\n".join([
        "# Broad full-session shadow v1 / expanded B4 freeze — 2026-10-01",
        "",
        "All 12 selected replacement artifacts passed their candidate-specific B2 review. The freeze binds actual model, scaler, calibrator, selected-feature, fold/purge, OOF-threshold, code, data, and replay evidence. Historical quality metrics are retained as advisory evidence and were not used as a collection veto.",
        "",
        "This is not scored-shadow admission. Every full-policy branch remains `not_defined` pending the exact B3 trading-rule binding, and B5/B6/B7 remain fail-closed. No legacy policy or artifact may be substituted.",
        "",
        "| Queue | Model | Stream | Calibration | Exact threshold | Features | Model SHA-256 | Full policy |",
        "|---:|---|---|---|---:|---:|---|---|",
        *rows,
        "",
        f"Selection: `{relative(SELECTION_V1)}` (`{freeze['selection_sha256']}`)",
        "",
        f"Expanded B4 freeze: `{relative(FREEZE_V1)}`",
        "",
        f"Verification: `{relative(AUDIT_V1)}`",
        "",
    ])


def verify(expected_selection: dict[str, Any] | None = None, expected_freeze: dict[str, Any] | None = None) -> dict[str, Any]:
    rebuilt_selection, rebuilt_freeze = payloads()
    selection = expected_selection if expected_selection is not None else load_json(SELECTION_V1)
    freeze = expected_freeze if expected_freeze is not None else load_json(FREEZE_V1)
    if selection != rebuilt_selection:
        raise ValueError("Published broad_full_session_shadow_v1 differs from current reviewed bytes")
    if freeze != rebuilt_freeze:
        raise ValueError("Published expanded B4 freeze differs from current reviewed bytes")
    return {
        "status": "broad_full_session_shadow_v1_verification_passed",
        "verified_at": "2026-10-01",
        "selected_ids": list(SELECTED_IDS),
        "replacement_count": len(SELECTED_IDS),
        "selection_sha256": hashlib.sha256(rendered_bytes(selection)).hexdigest(),
        "freeze_sha256": hashlib.sha256(rendered_bytes(freeze)).hexdigest(),
        "all_source_data_code_feature_fold_calibration_threshold_replay_pins_verified": True,
        "all_exact_threshold_decisions_verified": True,
        "all_full_policies_not_defined": True,
        "scored_collection_eligible": False,
    }


def publish() -> None:
    selection, freeze = payloads()
    audit = verify(selection, freeze)
    lock = {
        "lock_id": "b4_broad_full_session_shadow_freeze_v1_lock",
        "selection": relative(SELECTION_V1),
        "selection_file_sha256": hashlib.sha256(rendered_bytes(selection)).hexdigest(),
        "freeze": relative(FREEZE_V1),
        "freeze_file_sha256": hashlib.sha256(rendered_bytes(freeze)).hexdigest(),
        "replacement_count": len(SELECTED_IDS),
    }
    write_new(SELECTION_V1, selection)
    write_new(FREEZE_V1, freeze)
    write_new(LOCK_V1, lock)
    write_new(AUDIT_V1, audit)
    if REPORT_V1.exists():
        raise FileExistsError(f"Refusing to overwrite frozen publication report: {REPORT_V1}")
    REPORT_V1.write_text(markdown(selection, freeze), encoding="utf-8")
    print(json.dumps(audit, indent=2))


def verify_v2(expected: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None) -> dict[str, Any]:
    rebuilt = payloads_v2()
    roster, selection, freeze = expected or (load_json(ROSTER_V1), load_json(SELECTION_V2), load_json(FREEZE_V2))
    if (roster, selection, freeze) != rebuilt:
        raise ValueError("Published 16-model selection/freeze differs from current reviewed bytes")
    return {
        "status": "broad_full_session_shadow_v2_verification_passed",
        "verified_at": "2026-10-01",
        "selected_ids": selection["selected_order"],
        "replacement_count": len(selection["replacement_bindings"]),
        "roster_sha256": hashlib.sha256(rendered_bytes(roster)).hexdigest(),
        "selection_sha256": hashlib.sha256(rendered_bytes(selection)).hexdigest(),
        "freeze_sha256": hashlib.sha256(rendered_bytes(freeze)).hexdigest(),
        "v1_core_entries_rehashed_and_carried_forward_unchanged": list(SELECTED_IDS),
        "v1_freeze_sha256": file_sha256(FREEZE_V1),
        "all_source_data_code_feature_fold_calibration_threshold_replay_pins_verified": True,
        "all_exact_threshold_decisions_verified": True,
        "all_full_policies_not_defined": True,
        "scored_collection_eligible": False,
    }


def publish_v2() -> None:
    roster, selection, freeze = payloads_v2()
    audit = verify_v2((roster, selection, freeze))
    lock = {
        "lock_id": "b4_broad_full_session_shadow_freeze_v2_lock",
        "roster": relative(ROSTER_V1),
        "roster_file_sha256": audit["roster_sha256"],
        "selection": relative(SELECTION_V2),
        "selection_file_sha256": audit["selection_sha256"],
        "freeze": relative(FREEZE_V2),
        "freeze_file_sha256": audit["freeze_sha256"],
        "replacement_count": audit["replacement_count"],
    }
    if REPORT_V2.exists():
        raise FileExistsError(f"Refusing to overwrite frozen publication report: {REPORT_V2}")
    write_new(ROSTER_V1, roster)
    write_new(SELECTION_V2, selection)
    write_new(FREEZE_V2, freeze)
    write_new(LOCK_V2, lock)
    write_new(AUDIT_V2, audit)
    report = markdown(selection, freeze)
    for old, new in (
        ("shadow v1 / expanded", "shadow v2 / expanded"),
        ("All 12 selected", "All 16 selected"),
        (relative(SELECTION_V1), relative(SELECTION_V2)),
        (relative(FREEZE_V1), relative(FREEZE_V2)),
        (relative(AUDIT_V1), relative(AUDIT_V2)),
    ):
        report = report.replace(old, new)
    report += (
        f"16-model selection roster: `{relative(ROSTER_V1)}` (`{audit['roster_sha256']}`), parent "
        f"`{relative(SELECTION_V0)}`.\n\nThe 12 v1 entries were re-hashed from disk and carried forward "
        f"byte-identical to `{relative(FREEZE_V1)}`; IDs 5/23/18/14 are the new entries.\n"
    )
    REPORT_V2.write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("publish", "verify"))
    parser.add_argument("--version", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.version == 2:
        if args.mode == "publish":
            publish_v2()
        else:
            print(json.dumps(verify_v2(), indent=2))
    elif args.mode == "publish":
        publish()
    else:
        print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
