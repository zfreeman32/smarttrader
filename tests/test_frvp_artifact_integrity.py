from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from ote_live.models.loaders import (
    load_direction_runtime_manifest,
    load_live_runtime_manifest,
    load_runtime_model,
    requires_frozen_frvp_preload_validation,
    validate_runtime_artifact_integrity,
)
from ote_live.models.ensemble import load_direction_models
from scripts.build_frvp_paper_signal_bundle import (
    ACTIVE_ARTIFACT_HASH_REFERENCE_KEYS,
    REVERSAL_MODEL_ID,
)


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MANIFEST_PATH = (
    ROOT
    / "ote_live"
    / "runtime_manifests"
    / "frvp_es_paper_signal_20260816"
    / REVERSAL_MODEL_ID
    / "live_runtime_manifest.json"
)
DIRECTION_MANIFEST_PATH = ACTIVE_MANIFEST_PATH.parents[1] / "live_runtime_manifest_long.json"
LEGACY_SHADOW_DIRECTION_PATH = (
    ROOT
    / "ote_live"
    / "runtime_manifests"
    / "frvp_es_shadow_20260721"
    / "live_runtime_manifest_long.json"
)


def test_active_reversal_manifest_pins_every_runtime_artifact() -> None:
    manifest = load_live_runtime_manifest(ACTIVE_MANIFEST_PATH)

    assert tuple(manifest.artifact_references.content_sha256) == (
        ACTIVE_ARTIFACT_HASH_REFERENCE_KEYS
    )
    assert "test_predictions_file" not in manifest.artifact_references.content_sha256
    validate_runtime_artifact_integrity(manifest)


def test_legacy_shadow_reversal_is_not_claimed_by_frozen_active_preflight() -> None:
    direction = load_direction_runtime_manifest(LEGACY_SHADOW_DIRECTION_PATH)
    legacy_reversal = next(
        model for model in direction.models if model.model_id == REVERSAL_MODEL_ID
    )

    assert legacy_reversal.status == "candidate"
    assert not requires_frozen_frvp_preload_validation(legacy_reversal)


@pytest.mark.parametrize("reference_key", ACTIVE_ARTIFACT_HASH_REFERENCE_KEYS)
def test_runtime_loader_rejects_pinned_artifact_byte_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reference_key: str,
) -> None:
    manifest = load_live_runtime_manifest(ACTIVE_MANIFEST_PATH)
    original_reference = getattr(manifest.artifact_references, reference_key)
    original_path = ROOT / str(original_reference)
    mutated_path = tmp_path / original_path.name
    shutil.copyfile(original_path, mutated_path)
    with mutated_path.open("ab") as handle:
        handle.write(b"frvp-integrity-mutation")

    from ote_live.models import loaders

    original_resolver = loaders._resolve_repo_path

    def redirect_selected_artifact(path):
        if str(path) == str(original_reference):
            return mutated_path
        return original_resolver(path)

    monkeypatch.setattr(loaders, "_resolve_repo_path", redirect_selected_artifact)

    with pytest.raises(
        ValueError,
        match=rf"artifact SHA-256 mismatch for '{reference_key}'",
    ):
        load_runtime_model(manifest, require_complete_policy=True)


def test_direction_loader_rejects_coordinated_manifest_and_artifact_mutation_before_any_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    direction = load_direction_runtime_manifest(DIRECTION_MANIFEST_PATH)
    active = next(model for model in direction.models if model.model_id == REVERSAL_MODEL_ID)
    mutated_model_path = tmp_path / "coordinated-mutated-model.json"
    shutil.copyfile(ROOT / active.artifact_references.model_file, mutated_model_path)
    with mutated_model_path.open("ab") as handle:
        handle.write(b"coordinated-frvp-mutation")
    mutated_hash = hashlib.sha256(mutated_model_path.read_bytes()).hexdigest()
    pins = dict(active.artifact_references.content_sha256)
    pins["model_file"] = mutated_hash
    references = active.artifact_references.model_copy(
        update={"model_file": str(mutated_model_path), "content_sha256": pins}
    )
    mutated_active = active.model_copy(update={"artifact_references": references})
    reordered = direction.model_copy(
        update={
            "models": [
                *[model for model in direction.models if model.model_id != REVERSAL_MODEL_ID],
                mutated_active,
            ]
        }
    )
    load_calls: list[str] = []

    def unexpected_load(manifest, **_kwargs):
        load_calls.append(manifest.model_id)
        pytest.fail("No model or joblib deserializer may be reached before FRVP preflight.")

    monkeypatch.setattr("ote_live.models.ensemble.load_runtime_model", unexpected_load)

    with pytest.raises(ValueError, match="Frozen active FRVP manifest SHA-256 mismatch"):
        load_direction_models(reordered, require_complete_policy=True)

    assert load_calls == []


def test_direct_loader_rejects_coordinated_mutation_before_deserialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    active = load_live_runtime_manifest(ACTIVE_MANIFEST_PATH)
    mutated_model_path = tmp_path / "direct-mutated-model.json"
    shutil.copyfile(ROOT / active.artifact_references.model_file, mutated_model_path)
    with mutated_model_path.open("ab") as handle:
        handle.write(b"coordinated-direct-loader-mutation")
    pins = dict(active.artifact_references.content_sha256)
    pins["model_file"] = hashlib.sha256(mutated_model_path.read_bytes()).hexdigest()
    references = active.artifact_references.model_copy(
        update={"model_file": str(mutated_model_path), "content_sha256": pins}
    )
    mutated = active.model_copy(update={"artifact_references": references})
    deserialize_calls: list[str] = []

    def unexpected_read_json(_path):
        deserialize_calls.append("json")
        pytest.fail("No runtime artifact may be deserialized before frozen validation.")

    monkeypatch.setattr("ote_live.models.loaders._read_json", unexpected_read_json)

    with pytest.raises(ValueError, match="Frozen active FRVP manifest SHA-256 mismatch"):
        load_runtime_model(mutated, require_complete_policy=True)

    assert deserialize_calls == []
