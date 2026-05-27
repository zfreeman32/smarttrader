from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

DEFAULT_REGISTRY_PATH = Path(__file__).with_name("ote_model_registry.json")
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("ote_model_registry.schema.json")
REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_DIRECTIONS = {"long", "short"}
VALID_ROLES = {"champion", "challenger", "candidate", "benchmark"}
VALID_BACKENDS = {"xgboost", "tcn", "lstm"}
VALID_CALIBRATION_METHODS = {"platt", "isotonic", "none"}
VALID_STATUSES = {"active", "deprecated", "candidate"}


@dataclass(frozen=True)
class OTEModelRecord:
    model_id: str
    direction: str
    role: str
    backend: str
    artifact_path: str
    cv_mean_ap: float
    cv_mean_event_f05: float
    test_ap: float
    test_event_f05: float
    global_threshold: Optional[float]
    regime_thresholds: Optional[Dict[str, Any]]
    abstain_policy: Optional[Dict[str, Any]]
    calibration_method: str
    promotion_date: str
    promotion_reason: str
    status: str

    def resolve_artifact_path(self, repo_root: Path | None = None) -> Path:
        root = repo_root or REPO_ROOT
        return (root / self.artifact_path).resolve()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "direction": self.direction,
            "role": self.role,
            "backend": self.backend,
            "artifact_path": self.artifact_path,
            "cv_mean_ap": self.cv_mean_ap,
            "cv_mean_event_f05": self.cv_mean_event_f05,
            "test_ap": self.test_ap,
            "test_event_f05": self.test_event_f05,
            "global_threshold": self.global_threshold,
            "regime_thresholds": self.regime_thresholds,
            "abstain_policy": self.abstain_policy,
            "calibration_method": self.calibration_method,
            "promotion_date": self.promotion_date,
            "promotion_reason": self.promotion_reason,
            "status": self.status,
        }


@dataclass(frozen=True)
class OTEPromotionRules:
    min_cv_splits: int
    min_test_event_f05: float
    require_regime_robustness: bool
    require_post_cost_profitability: bool
    require_paper_trading_confirmation: bool


class OTEModelRegistry:
    def __init__(
        self,
        models: Iterable[OTEModelRecord],
        promotion_rules: OTEPromotionRules,
        registry_path: Path,
    ) -> None:
        self._models = tuple(models)
        self._promotion_rules = promotion_rules
        self._registry_path = registry_path
        self._by_id = {model.model_id: model for model in self._models}

    @property
    def models(self) -> Tuple[OTEModelRecord, ...]:
        return self._models

    @property
    def promotion_rules(self) -> OTEPromotionRules:
        return self._promotion_rules

    @property
    def registry_path(self) -> Path:
        return self._registry_path

    def get_model(self, model_id: str) -> OTEModelRecord:
        try:
            return self._by_id[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown OTE model_id: {model_id}") from exc

    def find_models(
        self,
        *,
        direction: Optional[str] = None,
        role: Optional[str] = None,
        backend: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Tuple[OTEModelRecord, ...]:
        matches = []
        for model in self._models:
            if direction is not None and model.direction != direction:
                continue
            if role is not None and model.role != role:
                continue
            if backend is not None and model.backend != backend:
                continue
            if status is not None and model.status != status:
                continue
            matches.append(model)
        return tuple(matches)

    def get_direction_role(
        self,
        direction: str,
        role: str,
        *,
        status: Optional[str] = None,
    ) -> OTEModelRecord:
        matches = self.find_models(direction=direction, role=role, status=status)
        if not matches:
            raise LookupError(
                f"No OTE registry entry found for direction={direction!r}, role={role!r}, status={status!r}."
            )
        if len(matches) > 1:
            raise LookupError(
                f"Multiple OTE registry entries found for direction={direction!r}, role={role!r}, status={status!r}."
            )
        return matches[0]

    def resolve_artifact_path(self, model_id: str) -> Path:
        return self.get_model(model_id).resolve_artifact_path()


def load_ote_model_registry(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    *,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    validate: bool = True,
) -> OTEModelRegistry:
    registry_path = Path(registry_path)
    # Accept UTF-8 files with or without a BOM so PowerShell-authored registry
    # files can still be loaded without manual re-encoding.
    payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    if validate:
        validate_ote_model_registry_payload(
            payload,
            registry_path=registry_path,
            schema_path=Path(schema_path),
        )

    models = tuple(OTEModelRecord(**record) for record in payload["models"])
    promotion_rules = OTEPromotionRules(**payload["promotion_rules"])
    return OTEModelRegistry(
        models=models,
        promotion_rules=promotion_rules,
        registry_path=registry_path,
    )


def validate_ote_model_registry_payload(
    payload: Dict[str, Any],
    *,
    registry_path: Path,
    schema_path: Path,
) -> None:
    _validate_with_jsonschema(payload, schema_path)

    if not isinstance(payload, dict):
        raise ValueError(f"OTE registry at {registry_path} must contain a JSON object.")

    models = payload.get("models")
    promotion_rules = payload.get("promotion_rules")
    if not isinstance(models, list) or not models:
        raise ValueError(f"OTE registry at {registry_path} must contain a non-empty 'models' list.")
    if not isinstance(promotion_rules, dict):
        raise ValueError(f"OTE registry at {registry_path} must contain a 'promotion_rules' object.")

    required_model_fields = {
        "model_id",
        "direction",
        "role",
        "backend",
        "artifact_path",
        "cv_mean_ap",
        "cv_mean_event_f05",
        "test_ap",
        "test_event_f05",
        "global_threshold",
        "regime_thresholds",
        "abstain_policy",
        "calibration_method",
        "promotion_date",
        "promotion_reason",
        "status",
    }

    seen_ids = set()
    for index, record in enumerate(models):
        if not isinstance(record, dict):
            raise ValueError(f"Registry model entry #{index} must be an object.")
        missing = required_model_fields - set(record)
        if missing:
            raise ValueError(f"Registry model entry #{index} is missing required fields: {sorted(missing)}")

        model_id = record["model_id"]
        if model_id in seen_ids:
            raise ValueError(f"Duplicate OTE model_id in registry: {model_id}")
        seen_ids.add(model_id)

        _validate_enum(record["direction"], VALID_DIRECTIONS, "direction", model_id)
        _validate_enum(record["role"], VALID_ROLES, "role", model_id)
        _validate_enum(record["backend"], VALID_BACKENDS, "backend", model_id)
        _validate_enum(
            record["calibration_method"],
            VALID_CALIBRATION_METHODS,
            "calibration_method",
            model_id,
        )
        _validate_enum(record["status"], VALID_STATUSES, "status", model_id)

        for metric_key in ("cv_mean_ap", "cv_mean_event_f05", "test_ap", "test_event_f05"):
            metric_value = float(record[metric_key])
            if metric_value < 0.0:
                raise ValueError(f"Registry model {model_id} has negative metric {metric_key}={metric_value}.")

        threshold = record["global_threshold"]
        if threshold is not None:
            threshold_value = float(threshold)
            if not 0.0 <= threshold_value <= 1.0:
                raise ValueError(
                    f"Registry model {model_id} has global_threshold outside [0, 1]: {threshold_value}."
                )

        artifact_path = Path(record["artifact_path"])
        if artifact_path.is_absolute():
            raise ValueError(f"Registry model {model_id} must use a relative artifact_path.")
        resolved_artifact_path = (REPO_ROOT / artifact_path).resolve()
        if not resolved_artifact_path.exists():
            raise ValueError(
                f"Registry model {model_id} points to a missing artifact path: {resolved_artifact_path}"
            )

    required_rules = {
        "min_cv_splits",
        "min_test_event_f05",
        "require_regime_robustness",
        "require_post_cost_profitability",
        "require_paper_trading_confirmation",
    }
    missing_rules = required_rules - set(promotion_rules)
    if missing_rules:
        raise ValueError(f"OTE registry promotion_rules missing required keys: {sorted(missing_rules)}")


def _validate_enum(value: Any, allowed: set[str], field_name: str, model_id: str) -> None:
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(
            f"Registry model {model_id} has invalid {field_name}={value!r}. Allowed values: {allowed_values}"
        )


def _validate_with_jsonschema(payload: Dict[str, Any], schema_path: Path) -> None:
    if not schema_path.exists():
        return

    try:
        import jsonschema
    except ImportError:
        return

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=payload, schema=schema)
