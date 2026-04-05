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
    build_direction_runtime_manifests,
    validate_manifest_for_live_decisions,
    write_direction_runtime_manifests,
)
from ote_live.policies.packager import package_candidate_policy_artifacts


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
    assert len(manifests["short"].models) == 4
    assert manifests["long"].recommendations.recommended_primary_model_id == "long_ote_tcn_v2_candidate"
    assert manifests["short"].recommendations.recommended_primary_model_id == "short_ote_tcn_v2_candidate"
    assert manifests["long"].recommendations.recommended_strongest_v2_model_id == "long_ote_tcn_v2_candidate"
    assert manifests["short"].recommendations.recommended_strongest_v2_model_id == "short_ote_tcn_v2_candidate"

    long_v2 = next(model for model in manifests["long"].models if model.model_id == "long_ote_tcn_v2_candidate")
    short_v2 = next(model for model in manifests["short"].models if model.model_id == "short_ote_tcn_v2_candidate")
    long_v1 = next(model for model in manifests["long"].models if model.model_id == "long_ote_tcn_v1_candidate")
    short_v1 = next(model for model in manifests["short"].models if model.model_id == "short_ote_tcn_v1_candidate")
    long_model_ids = {model.model_id for model in manifests["long"].models}

    assert long_v2.live_policy.policy_status == "complete"
    assert short_v2.live_policy.policy_status == "complete"
    assert long_v1.live_policy.policy_status == "complete"
    assert short_v1.live_policy.policy_status == "complete"
    assert "long_ote_xgb_v1_candidate" not in long_model_ids
    assert long_v2.live_policy.thresholds.regime_thresholds["strong_down_high"] == 0.8
    assert short_v2.live_policy.thresholds.global_threshold == 0.75
    assert short_v2.live_policy.abstain_policy.enabled is False

    output_dir = write_direction_runtime_manifests(manifests, output_dir=_make_local_tmp_dir())
    assert (output_dir / "live_runtime_manifest_long.json").exists()
    assert (output_dir / "live_runtime_manifest_short.json").exists()
    assert (output_dir / long_v2.model_id / "live_policy.json").exists()


def test_selected_features_exist_in_canonical_feature_catalog() -> None:
    manifests = _build_packaged_manifests()

    for direction_manifest in manifests.values():
        for model_manifest in direction_manifest.models:
            validation = model_manifest.feature_manifest.validation
            assert validation.selected_features_in_canonical_catalog
            assert validation.selected_features_in_direction_feature_list
            assert validation.missing_from_canonical_catalog == []
            assert validation.missing_from_direction_feature_list == []


def test_live_policy_schema_requires_abstain_fields_and_context_rows() -> None:
    manifests = _build_packaged_manifests()
    primary_long = next(model for model in manifests["long"].models if model.model_id == "long_ote_tcn_v2_candidate")

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

    policy_path = output_dir / "long_ote_tcn_v2_candidate" / "live_policy.json"
    payload = json.loads(policy_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0.0"
    assert payload["model_id"] == "long_ote_tcn_v2_candidate"
