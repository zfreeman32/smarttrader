from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ote_live.features.manifest import DirectionRuntimeManifest, LiveRuntimeManifest
from ote_live.models.calibrators import load_probability_calibrator
from ote_live.models.registry import validate_manifest_for_live_decisions

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCALE_CLIP = 8.0
DEFAULT_BATCH_SIZE = 256


@dataclass(frozen=True)
class LoadedRuntimeModel:
    manifest: LiveRuntimeManifest
    model: Any
    scaler: Any | None
    calibrator: Any | None
    training_summary: dict[str, Any]
    scale_clip: float
    batch_size: int
    use_amp: bool

    @property
    def model_id(self) -> str:
        return self.manifest.model_id

    @property
    def backend(self) -> str:
        return self.manifest.backend

    @property
    def selected_feature_names(self) -> tuple[str, ...]:
        return tuple(self.manifest.feature_manifest.selected_feature_names)

    @property
    def allows_nan_feature_values(self) -> bool:
        return self.backend == "xgboost" and self.model_id.lower().startswith(("frvp_", "ict_"))

    @property
    def context_rows(self) -> int:
        return int(self.manifest.context_requirements.context_rows)

    @property
    def window_size(self) -> int:
        return int(self.manifest.context_requirements.window_size)


def load_live_runtime_manifest(path: str | Path) -> LiveRuntimeManifest:
    payload = _read_json(_resolve_repo_path(path))
    return LiveRuntimeManifest.model_validate(payload)


def load_direction_runtime_manifest(path: str | Path) -> DirectionRuntimeManifest:
    payload = _read_json(_resolve_repo_path(path))
    return DirectionRuntimeManifest.model_validate(payload)


def load_runtime_model(
    manifest_or_path: LiveRuntimeManifest | str | Path,
    *,
    require_complete_policy: bool = False,
) -> LoadedRuntimeModel:
    manifest = (
        manifest_or_path
        if isinstance(manifest_or_path, LiveRuntimeManifest)
        else load_live_runtime_manifest(manifest_or_path)
    )
    validate_manifest_for_live_decisions(
        manifest,
        require_complete_policy=require_complete_policy,
    )

    training_summary_path = _resolve_repo_path(manifest.artifact_references.training_summary_file)
    training_summary = _read_json(training_summary_path)

    model = _load_backend_model(manifest)
    scaler = _load_optional_joblib(manifest.artifact_references.scaler_file)
    calibrator = load_probability_calibrator(
        _resolve_repo_path(manifest.artifact_references.calibrator_file)
        if manifest.artifact_references.calibrator_file
        else None
    )

    return LoadedRuntimeModel(
        manifest=manifest,
        model=model,
        scaler=scaler,
        calibrator=calibrator,
        training_summary=training_summary,
        scale_clip=float(_first_non_none(
            _nested_get(training_summary, "config", "scale_clip"),
            DEFAULT_SCALE_CLIP,
        )),
        batch_size=int(_first_non_none(
            _nested_get(training_summary, "model_config", "trainer", "batch_size"),
            _nested_get(training_summary, "config", "batch_size"),
            DEFAULT_BATCH_SIZE,
        )),
        use_amp=_resolve_runtime_amp_usage(
            manifest=manifest,
            training_summary=training_summary,
        ),
    )


def _load_backend_model(manifest: LiveRuntimeManifest) -> Any:
    model_file = _resolve_repo_path(manifest.artifact_references.model_file)

    if manifest.backend == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                f"Model {manifest.model_id} requires the 'xgboost' package for live inference."
            ) from exc

        booster = xgb.Booster()
        booster.load_model(str(model_file))
        return booster

    if manifest.backend in {"tcn", "lstm"}:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                f"Model {manifest.model_id} requires the 'torch' package for live inference."
            ) from exc

        from model_training.ote_training.torch_trainer import load_torch_model_from_checkpoint

        checkpoint = torch.load(model_file, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Torch artifact for {manifest.model_id} does not contain a valid checkpoint payload."
            )
        return load_torch_model_from_checkpoint(checkpoint, device=torch.device("cpu"))

    raise ValueError(f"Unsupported runtime backend {manifest.backend!r} for model {manifest.model_id}.")


def _load_optional_joblib(path: str | Path | None) -> Any | None:
    if path is None:
        return None
    resolved = _resolve_repo_path(path)
    if not resolved.exists():
        return None
    return joblib.load(resolved)


def _resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (REPO_ROOT / path_obj).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_runtime_amp_usage(
    *,
    manifest: LiveRuntimeManifest,
    training_summary: dict[str, Any],
) -> bool:
    requested_use_amp = bool(
        _first_non_none(
            _nested_get(training_summary, "model_config", "trainer", "use_amp"),
            _nested_get(training_summary, "config", "use_amp"),
            False,
        )
    )
    if manifest.backend not in {"tcn", "lstm"}:
        return False

    try:
        import torch
        from model_training.ote_training.torch_trainer import resolve_amp_usage
    except ImportError:
        return False

    model_type = str(
        _first_non_none(
            _nested_get(training_summary, "model_config", "model_type"),
            manifest.backend,
        )
    )
    return bool(
        resolve_amp_usage(
            model_type=model_type,
            requested_use_amp=requested_use_amp,
            device=torch.device("cpu"),
        )
    )
