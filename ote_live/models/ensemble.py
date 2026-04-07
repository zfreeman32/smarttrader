from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ote_live.features.manifest import DirectionRuntimeManifest
from ote_live.models.loaders import (
    LoadedRuntimeModel,
    load_direction_runtime_manifest,
    load_runtime_model,
)


@dataclass
class LoadedDirectionModels:
    direction_manifest: DirectionRuntimeManifest
    loaded_models: dict[str, LoadedRuntimeModel] = field(default_factory=dict)
    unavailable_models: dict[str, str] = field(default_factory=dict)

    @property
    def primary_model(self) -> LoadedRuntimeModel | None:
        primary_id = self.direction_manifest.recommendations.recommended_primary_model_id
        if primary_id:
            return self.loaded_models.get(primary_id)

        for manifest in self.direction_manifest.models:
            loaded = self.loaded_models.get(manifest.model_id)
            if loaded is not None:
                return loaded
        return None

    @property
    def shadow_models(self) -> tuple[LoadedRuntimeModel, ...]:
        primary = self.primary_model
        primary_id = primary.model_id if primary is not None else None
        ordered: list[LoadedRuntimeModel] = []
        for manifest in self.direction_manifest.models:
            if manifest.model_id == primary_id:
                continue
            loaded = self.loaded_models.get(manifest.model_id)
            if loaded is not None:
                ordered.append(loaded)
        return tuple(ordered)

    def get_model(self, model_id: str) -> LoadedRuntimeModel:
        try:
            return self.loaded_models[model_id]
        except KeyError as exc:
            raise KeyError(f"Direction bundle does not contain a loaded runtime model {model_id!r}.") from exc


def load_direction_models(
    direction_manifest_or_path: DirectionRuntimeManifest | str | Path,
    *,
    model_ids: tuple[str, ...] | list[str] | None = None,
    skip_unavailable_backends: bool = False,
    require_complete_policy: bool = False,
) -> LoadedDirectionModels:
    direction_manifest = (
        direction_manifest_or_path
        if isinstance(direction_manifest_or_path, DirectionRuntimeManifest)
        else load_direction_runtime_manifest(direction_manifest_or_path)
    )

    requested_ids = set(model_ids) if model_ids is not None else None
    loaded_models: dict[str, LoadedRuntimeModel] = {}
    unavailable_models: dict[str, str] = {}

    for manifest in direction_manifest.models:
        if requested_ids is not None and manifest.model_id not in requested_ids:
            continue

        try:
            loaded_models[manifest.model_id] = load_runtime_model(
                manifest,
                require_complete_policy=require_complete_policy,
            )
        except ImportError as exc:
            if not skip_unavailable_backends:
                raise
            unavailable_models[manifest.model_id] = str(exc)

    return LoadedDirectionModels(
        direction_manifest=direction_manifest,
        loaded_models=loaded_models,
        unavailable_models=unavailable_models,
    )
