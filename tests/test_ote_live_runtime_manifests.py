from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ote_live.features.manifest import LivePolicy
from ote_live.models.registry import (
    DEFAULT_ACTIVE_REGISTRY_PATH,
    DEFAULT_CANDIDATE_REGISTRY_PATH,
    build_direction_runtime_manifests,
    validate_manifest_for_live_decisions,
    write_direction_runtime_manifests,
)
from ote_live.policies.packager import package_candidate_policy_artifacts
from ote_live.scripts.export_live_runtime_manifests import build_parser as build_export_manifest_parser


def _make_local_tmp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "tmp" / "ote_live_manifest_tests"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_packaged_manifests() -> dict[str, object]:
    packaged_policy_dir = _make_local_tmp_dir()
    package_candidate_policy_artifacts(output_dir=packaged_policy_dir)
    return build_direction_runtime_manifests(packaged_policy_dir=packaged_policy_dir)


def test_candidate_registry_exports_direction_manifests() -> None:
    manifests = _build_packaged_manifests()

    assert set(manifests) == {"long", "short"}
    assert len(manifests["long"].models) == 5
    assert len(manifests["short"].models) == 3
    assert manifests["long"].recommendations.recommended_primary_model_id == "long_reversal_tcn_v2_20260525_narrow48"
    assert manifests["short"].recommendations.recommended_primary_model_id == "short_reversal_xgb_v2_20260525"
    assert manifests["long"].recommendations.recommended_strongest_v2_model_id == "long_reversal_tcn_v2_20260525_narrow48"
    assert manifests["short"].recommendations.recommended_strongest_v2_model_id == "short_reversal_xgb_v2_20260525"

    long_primary = next(
        model for model in manifests["long"].models if model.model_id == "long_reversal_tcn_v2_20260525_narrow48"
    )
    short_primary = next(
        model for model in manifests["short"].models if model.model_id == "short_reversal_xgb_v2_20260525"
    )
    long_model_ids = {model.model_id for model in manifests["long"].models}
    short_model_ids = {model.model_id for model in manifests["short"].models}
    long_breakout = next(model for model in manifests["long"].models if model.model_id == "long_breakout_tcn_champion")
    long_union = next(
        model for model in manifests["long"].models if model.model_id == "long_ote_union_tcn_candidate_20260523"
    )

    assert long_primary.live_policy.policy_status == "complete"
    assert short_primary.live_policy.policy_status == "complete"
    assert long_breakout.live_policy.policy_status == "complete"
    assert long_union.live_policy.policy_status == "complete"
    assert long_model_ids == {
        "long_reversal_tcn_v2_20260525_narrow48",
        "long_ote_union_tcn_candidate_20260523",
        "long_breakout_tcn_champion",
        "long_ote_meta_tcn_champion",
        "long_breakout_xgb_v1",
    }
    assert short_model_ids == {
        "short_reversal_xgb_v2_20260525",
        "short_ote_meta_tcn_champion",
        "short_ote_union_tcn_candidate_20260520",
    }
    assert long_primary.status == "active"
    assert short_primary.status == "active"
    assert long_breakout.status == "candidate"
    assert long_primary.live_policy.thresholds.global_threshold == pytest.approx(0.85)
    assert short_primary.live_policy.thresholds.global_threshold == pytest.approx(0.80)
    assert short_primary.live_policy.abstain_policy.enabled is False

    output_dir = write_direction_runtime_manifests(manifests, output_dir=_make_local_tmp_dir())
    assert (output_dir / "live_runtime_manifest_long.json").exists()
    assert (output_dir / "live_runtime_manifest_short.json").exists()
    assert (output_dir / long_primary.model_id / "live_policy.json").exists()


def test_selected_features_exist_in_canonical_feature_catalog() -> None:
    manifests = _build_packaged_manifests()

    for direction_manifest in manifests.values():
        for model_manifest in direction_manifest.models:
            validation = model_manifest.feature_manifest.validation
            assert validation.selected_features_in_canonical_catalog
            assert validation.missing_from_canonical_catalog == []
            if validation.selected_features_in_direction_feature_list:
                assert validation.missing_from_direction_feature_list == []
            else:
                assert validation.missing_from_direction_feature_list
                assert any("stale direction list" in note for note in model_manifest.notes)


def test_live_registry_allows_stale_direction_feature_lists() -> None:
    manifests = build_direction_runtime_manifests(packaged_policy_dir=_make_local_tmp_dir())

    clean_direction_models: list[str] = []
    stale_direction_models: list[str] = []
    for direction_manifest in manifests.values():
        for model_manifest in direction_manifest.models:
            validation = model_manifest.feature_manifest.validation
            if validation.selected_features_in_direction_feature_list:
                clean_direction_models.append(model_manifest.model_id)
                continue
            stale_direction_models.append(model_manifest.model_id)
            assert validation.missing_from_canonical_catalog == []
            assert any("stale direction list" in note for note in model_manifest.notes)

    assert clean_direction_models
    assert stale_direction_models


def test_live_policy_schema_requires_abstain_fields_and_context_rows() -> None:
    manifests = _build_packaged_manifests()
    primary_long = next(
        model for model in manifests["long"].models if model.model_id == "long_reversal_tcn_v2_20260525_narrow48"
    )

    validate_manifest_for_live_decisions(primary_long, require_complete_policy=True)

    invalid_manifest = primary_long.model_copy(
        update={
            "context_requirements": primary_long.context_requirements.model_copy(update={"context_rows": 0}),
        }
    )
    with pytest.raises(ValueError, match="context_rows"):
        validate_manifest_for_live_decisions(invalid_manifest, require_complete_policy=True)

    invalid_policy_payload = primary_long.live_policy.model_dump()
    invalid_policy_payload["abstain_policy"].pop("cooldown_bars")
    with pytest.raises(ValidationError):
        LivePolicy.model_validate(invalid_policy_payload)


def test_written_policy_file_is_versioned_json() -> None:
    manifests = _build_packaged_manifests()
    output_dir = write_direction_runtime_manifests(manifests, output_dir=_make_local_tmp_dir())

    policy_path = output_dir / "long_reversal_tcn_v2_20260525_narrow48" / "live_policy.json"
    payload = json.loads(policy_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0.0"
    assert payload["model_id"] == "long_reversal_tcn_v2_20260525_narrow48"


def test_legacy_active_registry_still_exports_direction_manifests() -> None:
    packaged_policy_dir = _make_local_tmp_dir()
    package_candidate_policy_artifacts(output_dir=packaged_policy_dir)
    manifests = build_direction_runtime_manifests(
        registry_path=DEFAULT_ACTIVE_REGISTRY_PATH,
        packaged_policy_dir=packaged_policy_dir,
    )

    assert set(manifests) == {"long", "short"}

    long_manifest = manifests["long"]
    short_manifest = manifests["short"]
    long_ids = {model.model_id for model in long_manifest.models}
    short_ids = {model.model_id for model in short_manifest.models}

    assert long_manifest.recommendations.recommended_primary_model_id in long_ids
    assert short_manifest.recommendations.recommended_primary_model_id in short_ids
    assert long_manifest.recommendations.recommended_primary_model_id != "long_ote_tcn_v2_candidate"
    assert short_manifest.recommendations.recommended_primary_model_id != "short_ote_tcn_v2_candidate"
    assert all(model.live_policy.policy_status in {"complete", "provisional"} for model in long_manifest.models)
    assert all(model.live_policy.policy_status in {"complete", "provisional"} for model in short_manifest.models)


def test_export_script_defaults_to_candidate_registry() -> None:
    parser = build_export_manifest_parser()
    args = parser.parse_args([])

    assert Path(args.registry_path).resolve() == DEFAULT_CANDIDATE_REGISTRY_PATH.resolve()


def test_live_registry_long_breakout_uses_v3_pairwise_abstains() -> None:
    manifests = build_direction_runtime_manifests()

    long_breakout = next(model for model in manifests["long"].models if model.model_id == "long_breakout_tcn_champion")

    assert long_breakout.live_policy.thresholds.global_threshold == pytest.approx(0.92)
    assert long_breakout.live_policy.cost_assumptions.targeted_filter_preset == "long_breakout_regime_prune_v3"
    assert long_breakout.live_policy.abstain_policy.abstain_composite_session_pairs == [
        ("strong_up_medium", "asia")
    ]
    assert long_breakout.live_policy.abstain_policy.abstain_composite_stress_pairs == [
        ("strong_up_medium", "elevated")
    ]
