from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ote_registry_loader import (
    DEFAULT_REGISTRY_PATH,
    DEFAULT_SCHEMA_PATH,
    load_ote_model_registry,
    validate_ote_model_registry_payload,
)


def test_default_ote_registry_loads_and_resolves_artifacts() -> None:
    registry = load_ote_model_registry()

    champion = registry.get_direction_role("long", "champion")
    short_champion = registry.get_direction_role("short", "champion", status="active")
    short_challenger = registry.get_direction_role("short", "challenger", status="active")

    assert champion.model_id == "long_ote_champion_v1"
    assert champion.backend == "tcn"
    assert champion.resolve_artifact_path().exists()
    assert champion.abstain_policy is None
    assert short_champion.backend == "tcn"
    assert short_challenger.backend == "xgboost"


def test_ote_registry_validation_rejects_duplicate_model_ids(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
    duplicate = dict(payload["models"][0])
    payload["models"].append(duplicate)

    registry_path = tmp_path / "ote_model_registry.json"
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate OTE model_id"):
        validate_ote_model_registry_payload(
            payload,
            registry_path=registry_path,
            schema_path=DEFAULT_SCHEMA_PATH,
        )
