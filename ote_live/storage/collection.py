"""Content-addressed collection contracts and explicit report partition selection."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from ote_live.ingestion.market_calendar import MARKET_CALENDAR_VERSION
from ote_live.ingestion.provenance import BAR_PROVENANCE_VERSION, SHADOW_FRESHNESS_CONTRACT
from ote_live.models.frvp_research import FRVP_RESEARCH_ROSTER_VERSION
from ote_live.models.ict_research import ICT_RESEARCH_ROSTER_VERSION

ROOT = Path(__file__).resolve().parents[2]
COLLECTION_CONTRACT = "es-input-observation-v1"
LEGACY_COLLECTION = "legacy-unversioned"


class MixedCollectionError(ValueError):
    """A performance report needs an explicit observation partition."""


def file_sha256(path: Path) -> str:
    stat = path.stat()
    return _file_sha256_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4096)
def _file_sha256_cached(path: str, modified_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def collection_identity(manifests: Iterable[Any], *, root: Path = ROOT) -> tuple[str, dict]:
    models = []
    for manifest in sorted(manifests, key=lambda item: item.model_id):
        payload = manifest.model_dump(mode="json")
        artifacts = {}
        for name, reference in payload["artifact_references"].items():
            if not name.endswith("_file") or not reference:
                continue
            path = root / reference
            artifacts[name] = {"path": reference, "sha256": file_sha256(path) if path.is_file() else None}
        models.append({"manifest": payload, "artifacts": artifacts})
    # Hash producers, detector definitions, inference and policies, not only filenames.
    source_hashes = {}
    for directory in ("features", "ict", "frvp", "ote_live/features", "ote_live/ingestion",
                      "ote_live/models", "ote_live/policies", "ote_live/storage",
                      "ote_live/contracts"):
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".json", ".sql"}:
                continue
            source_hashes[path.relative_to(root).as_posix()] = file_sha256(path)
    for reference in ("model_testing/ote_threshold_policy.py", "model_testing/ote_policy_backtest.py",
                      "scripts/ote_targeted_filter_presets.py", "ote_live/scripts/run_es_live_collector.py"):
        path = root / reference
        if path.is_file():
            source_hashes[reference] = file_sha256(path)
    payload = {"contract": COLLECTION_CONTRACT, "models": models, "source_sha256": source_hashes,
               "bar_provenance_contract": BAR_PROVENANCE_VERSION,
               "shadow_freshness_contract": SHADOW_FRESHNESS_CONTRACT,
               "market_calendar_version": MARKET_CALENDAR_VERSION,
               "frvp_research_roster_version": FRVP_RESEARCH_ROSTER_VERSION,
               "ict_research_roster_version": ICT_RESEARCH_ROSTER_VERSION,
               "qualified_period_started": False, "research_only": True,
               "freshness_max_age_seconds": 90, "entry_contract": "unassigned-pending-B1-B3"}
    version = COLLECTION_CONTRACT + ":" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return version, payload


def register_collection(store, version: str, payload: dict) -> None:
    serialized = canonical_json(payload)
    existing = store.connection.execute(
        "SELECT contract_json FROM collection_versions WHERE collection_version = ?", (version,)
    ).fetchone()
    if existing is not None and existing[0] != serialized:
        raise ValueError("Collection version already identifies a different immutable contract")
    store.connection.execute(
        "INSERT OR IGNORE INTO collection_versions VALUES (?, ?, ?)",
        (version, serialized, datetime.now(timezone.utc).isoformat()),
    )
    store.connection.commit()


def select_report_collection(store, *, collection_version: str | None = None,
                             model_id: str | None = None) -> str:
    query = "SELECT DISTINCT collection_version FROM model_predictions"
    params = ()
    if model_id is not None:
        query += " WHERE model_id = ?"
        params = (model_id,)
    versions = {row[0] for row in store.connection.execute(query, params)}
    if collection_version is not None:
        if collection_version not in versions:
            raise ValueError("Requested collection has no matching predictions")
        return collection_version
    if len(versions) > 1:
        raise MixedCollectionError("Mixed collection versions: choose an explicit collection_version for this report")
    return next(iter(versions), LEGACY_COLLECTION)


def fetch_collection_predictions(store, *, collection_version: str | None = None,
                                 model_id: str | None = None) -> list[dict]:
    version = select_report_collection(store, collection_version=collection_version, model_id=model_id)
    query = "SELECT * FROM model_predictions WHERE collection_version = ?"
    params: list = [version]
    if model_id is not None:
        query += " AND model_id = ?"
        params.append(model_id)
    query += " ORDER BY recorded_at_utc, id"
    return [dict(row) for row in store.connection.execute(query, params)]
