"""Append-only rule-setup observations, prediction links and diagnostic outcomes.

This history is independent of dashboard retention and paper-trial ledgers. A
revision appends a new observation; it never rewrites what was known previously.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from frvp.setups.detector import detect_frvp_setups
from frvp.target_lanes import FRVP_SETUP_BROAD_FAMILY_BY_TYPE
from ict.setups.detector import detect_ict_setups
from ict.taxonomy import infer_ict_setup_family
from ote_live.contracts.market_data import MarketBar
from ote_live.ingestion.base import ensure_utc, utc_now
from ote_live.ingestion.provenance import source_bar_version
from ote_live.storage.db import SQLiteLiveDataStore


SETUP_HISTORY_CONTRACT_VERSION = "setup_history_v1"
SETUP_HISTORY_TABLES = ("setup_events", "setup_prediction_links", "setup_event_outcomes")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS setup_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    previous_event_id INTEGER REFERENCES setup_events(id),
    event_kind TEXT NOT NULL CHECK(event_kind IN ('observed','revised','invalidated')),
    collection_version TEXT NOT NULL,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    strategy TEXT NOT NULL,
    source_timestamp_utc TEXT NOT NULL,
    source_bar_version TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    first_observed_at_utc TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    setup_side INTEGER NOT NULL,
    setup_family TEXT NOT NULL,
    rule_confidence REAL,
    detector_identity TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    selected INTEGER NOT NULL,
    geometry_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    UNIQUE(event_key, revision)
);
CREATE INDEX IF NOT EXISTS idx_setup_events_source
ON setup_events(asset, timeframe, collection_version, source_timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_setup_events_key ON setup_events(event_key, revision);
CREATE TABLE IF NOT EXISTS setup_prediction_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_event_id INTEGER NOT NULL REFERENCES setup_events(id),
    prediction_id INTEGER NOT NULL REFERENCES model_predictions(id),
    model_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    rejection_reasons_json TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(setup_event_id, prediction_id)
);
CREATE TABLE IF NOT EXISTS setup_event_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_event_id INTEGER NOT NULL REFERENCES setup_events(id),
    outcome_type TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    UNIQUE(setup_event_id, outcome_type, content_hash)
);
"""


@dataclass(frozen=True)
class SetupEventRecord:
    event_id: int
    event_key: str
    revision: int
    event_kind: str
    strategy: str
    setup_type: str
    setup_side: int
    setup_family: str
    selected: bool
    source_timestamp: datetime
    source_bar_version: str
    observed_at: datetime
    first_observed_at: datetime
    collection_version: str
    payload: dict[str, Any]


class SetupEventRepository:
    def __init__(self, store: SQLiteLiveDataStore) -> None:
        self.store = store
        store.connection.executescript(_SCHEMA)
        for table in SETUP_HISTORY_TABLES:
            for operation in ("UPDATE", "DELETE"):
                store.connection.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()} "
                    f"BEFORE {operation} ON {table} BEGIN "
                    "SELECT RAISE(ABORT, 'setup history is append-only'); END"
                )
            # REPLACE may silently delete conflicts when recursive triggers are
            # disabled. Guard conflicts at INSERT as well, on every connection.
            conflict = {
                "setup_events": "event_key=NEW.event_key AND revision=NEW.revision",
                "setup_prediction_links": "setup_event_id=NEW.setup_event_id AND prediction_id=NEW.prediction_id",
                "setup_event_outcomes": "setup_event_id=NEW.setup_event_id AND outcome_type=NEW.outcome_type AND content_hash=NEW.content_hash",
            }[table]
            store.connection.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_replace BEFORE INSERT ON {table} "
                f"WHEN EXISTS(SELECT 1 FROM {table} WHERE id=NEW.id OR ({conflict})) "
                "BEGIN SELECT RAISE(IGNORE); END"
            )
        store.connection.commit()

    def collect_from_frame(
        self,
        policy_frame: pd.DataFrame,
        *,
        bar: MarketBar,
        collection_version: str,
        observed_at: datetime | None = None,
        source_bar_version: object | None = None,
        strategies: Sequence[str] = ("FRVP", "ICT"),
    ) -> tuple[SetupEventRecord, ...]:
        """Collect all current-bar candidates before any model or policy filter.

        Detector failures propagate: an unavailable producer must not invalidate
        previous events as though it had successfully reported no candidates.
        """
        if policy_frame.empty:
            return ()
        records: list[SetupEventRecord] = []
        for strategy in strategies:
            observations = extract_setup_observations(policy_frame, strategy=strategy)
            records.extend(self.record_observations(
                observations, bar=bar, strategy=strategy,
                collection_version=collection_version, observed_at=observed_at,
                source_bar_version=source_bar_version,
            ))
        return tuple(records)

    def record_observations(
        self,
        observations: Iterable[Mapping[str, Any]],
        *,
        bar: MarketBar,
        strategy: str,
        collection_version: str,
        observed_at: datetime | None = None,
        source_bar_version: object | None = None,
    ) -> tuple[SetupEventRecord, ...]:
        """Reconcile one complete detector result for one source bar.

        The supplied result may be empty after a source correction. Disappearing
        candidates are then invalidated, while all prior observations remain.
        """
        # A failed generator/validation/insert must never leave half a detector
        # result to be committed by a subsequent health event or prediction.
        with self.store.connection:
            return self._record_observations(
                observations, bar=bar, strategy=strategy, collection_version=collection_version,
                observed_at=observed_at, source_bar_version=source_bar_version,
            )

    def _record_observations(self, observations, *, bar, strategy, collection_version,
                             observed_at, source_bar_version):
        strategy = strategy.upper()
        if strategy not in {"FRVP", "ICT"} or not collection_version:
            raise ValueError("FRVP/ICT strategy and collection version are required")
        now = ensure_utc(observed_at or utc_now())
        timestamp = ensure_utc(bar.timestamp).isoformat()
        version = str(source_bar_version if source_bar_version is not None else _bar_version(bar))
        previous = self.store.connection.execute(
            "SELECT * FROM setup_events s WHERE asset=? AND timeframe=? AND strategy=? "
            "AND collection_version=? AND source_timestamp_utc=? "
            "AND revision=(SELECT MAX(revision) FROM setup_events WHERE event_key=s.event_key)",
            (bar.asset, bar.timeframe, strategy, collection_version, timestamp),
        ).fetchall()
        prior_by_key = {row["event_key"]: row for row in previous}
        seen: set[str] = set()
        current: list[SetupEventRecord] = []
        for observation in observations:
            payload = _clean(dict(observation))
            setup_type = str(payload.get("setup_type") or "")
            side = int(payload.get("setup_side") or 0)
            if not setup_type or side not in {-1, 1}:
                raise ValueError("Setup observations require a type and side -1/+1")
            family = str(payload.get("setup_family") or (
                FRVP_SETUP_BROAD_FAMILY_BY_TYPE.get(int(setup_type), "unknown")
                if strategy == "FRVP" else infer_ict_setup_family(setup_type)
            ) or "unknown")
            payload.update({"setup_type": setup_type, "setup_side": side, "setup_family": family})
            event_key = _hash([collection_version, bar.asset, bar.timeframe, strategy,
                               timestamp, setup_type, side, family, payload.get("research", False)])
            if event_key in seen:
                raise ValueError("Duplicate setup candidate identity in detector result")
            seen.add(event_key)
            payload.update({
                "detector_identity": payload.get("detector_identity") or f"{strategy.lower()}.setups.detector",
                "detector_version": payload.get("detector_version") or detector_source_version(strategy),
                "history_contract_version": SETUP_HISTORY_CONTRACT_VERSION,
                "source_bar_version": version,
                "source_bar": _clean(bar.model_dump(mode="json")),
            })
            prior = prior_by_key.get(event_key)
            if prior is not None and prior["event_kind"] != "invalidated" and prior["content_hash"] == _event_hash(payload):
                current.append(_record(prior))
                continue
            current.append(self._append(
                event_key=event_key, payload=payload, prior=prior,
                event_kind="observed" if prior is None else "revised",
                bar=bar, strategy=strategy, collection_version=collection_version,
                source_bar_version=version, now=now,
            ))
        for key, prior in prior_by_key.items():
            if key in seen or prior["event_kind"] == "invalidated":
                continue
            payload = json.loads(prior["payload_json"])
            payload.update({"invalidation_reason": "absent_from_recomputed_source_bar",
                            "source_bar_version": version,
                            "source_bar": _clean(bar.model_dump(mode="json"))})
            self._append(event_key=key, payload=payload, prior=prior, event_kind="invalidated",
                         bar=bar, strategy=strategy, collection_version=collection_version,
                         source_bar_version=version, now=now)
        return tuple(current)

    def _append(self, *, event_key, payload, prior, event_kind, bar, strategy,
                collection_version, source_bar_version, now) -> SetupEventRecord:
        detector_identity = str(payload.get("detector_identity") or f"{strategy.lower()}.setups.detector")
        detector_version = str(payload.get("detector_version") or detector_source_version(strategy))
        cursor = self.store.connection.execute(
            "INSERT INTO setup_events (event_key, revision, previous_event_id, event_kind, "
            "collection_version, asset, timeframe, strategy, source_timestamp_utc, source_bar_version, "
            "observed_at_utc, first_observed_at_utc, setup_type, setup_side, setup_family, rule_confidence, "
            "detector_identity, detector_version, selected, geometry_json, payload_json, content_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (event_key, int(prior["revision"]) + 1 if prior else 1, prior["id"] if prior else None,
             event_kind, collection_version, bar.asset, bar.timeframe, strategy,
             ensure_utc(bar.timestamp).isoformat(), source_bar_version, now.isoformat(),
             prior["first_observed_at_utc"] if prior else now.isoformat(),
             payload["setup_type"], payload["setup_side"], payload["setup_family"],
             payload.get("confidence"), detector_identity, detector_version,
             int(bool(payload.get("selected", payload.get("fired", False)))),
             _json(payload.get("geometry", {})), _json(payload), _event_hash(payload)),
        )
        return _record(self.store.connection.execute("SELECT * FROM setup_events WHERE id=?",
                                                     (cursor.lastrowid,)).fetchone())

    def list_events(self, *, strategy: str | None = None, collection_version: str | None = None,
                    latest_only: bool = False) -> tuple[SetupEventRecord, ...]:
        query = "SELECT * FROM setup_events s WHERE 1=1"
        parameters: list[Any] = []
        if strategy is not None:
            query += " AND strategy=?"
            parameters.append(strategy.upper())
        if collection_version is not None:
            query += " AND collection_version=?"
            parameters.append(collection_version)
        if latest_only:
            query += " AND revision=(SELECT MAX(revision) FROM setup_events WHERE event_key=s.event_key)"
        return tuple(_record(row) for row in self.store.connection.execute(query + " ORDER BY id", parameters))

    def history(self, event_key: str) -> tuple[SetupEventRecord, ...]:
        return tuple(_record(row) for row in self.store.connection.execute(
            "SELECT * FROM setup_events WHERE event_key=? ORDER BY revision", (event_key,)))

    def link_prediction(self, event_id: int, prediction_id: int, *, model_id: str,
                        decision: str, rejection_reasons: Sequence[str] = (),
                        observed_at: datetime | None = None,
                        payload: Mapping[str, Any] | None = None) -> int:
        """Link every evaluated candidate, including mismatches and rejections."""
        self._require_event(event_id)
        prediction = self.store.connection.execute(
            "SELECT p.*, f.asset, f.timeframe FROM model_predictions p "
            "JOIN feature_snapshots f ON f.id=p.feature_snapshot_id WHERE p.id=?", (prediction_id,)
        ).fetchone()
        if prediction is None or prediction["model_id"] != model_id:
            raise ValueError("Prediction linkage requires an existing prediction for the supplied model")
        event = self.store.connection.execute("SELECT * FROM setup_events WHERE id=?", (event_id,)).fetchone()
        prediction_payload = json.loads(prediction["prediction_json"])
        if any(prediction[field] != event[field] for field in ("asset", "timeframe", "collection_version")) or (
            ensure_utc(datetime.fromisoformat(prediction_payload["timestamp"].replace("Z", "+00:00"))).isoformat()
            != event["source_timestamp_utc"]
        ):
            raise ValueError("Prediction and setup must share asset, timeframe, source timestamp and collection version")
        link_payload = {**dict(payload or {}), "prediction": prediction_payload,
                        "prediction_metadata": json.loads(prediction["metadata_json"] or "{}")}
        self.store.connection.execute(
            "INSERT OR IGNORE INTO setup_prediction_links "
            "(setup_event_id,prediction_id,model_id,decision,rejection_reasons_json,observed_at_utc,payload_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (event_id, prediction_id, model_id, decision, _json(list(rejection_reasons)),
             ensure_utc(observed_at or utc_now()).isoformat(), _json(link_payload)),
        )
        self.store.connection.commit()
        return int(self.store.connection.execute(
            "SELECT id FROM setup_prediction_links WHERE setup_event_id=? AND prediction_id=?",
            (event_id, prediction_id)).fetchone()["id"])

    def record_outcome(self, event_id: int, outcome_type: str, payload: Mapping[str, Any], *,
                       observed_at: datetime | None = None) -> int:
        """Append a named outcome contract; amended outcomes remain new records."""
        self._require_event(event_id)
        content_hash = _event_hash(payload)
        self.store.connection.execute(
            "INSERT OR IGNORE INTO setup_event_outcomes "
            "(setup_event_id,outcome_type,observed_at_utc,payload_json,content_hash) VALUES (?,?,?,?,?)",
            (event_id, outcome_type, ensure_utc(observed_at or utc_now()).isoformat(),
             _json(payload), content_hash),
        )
        self.store.connection.commit()
        return int(self.store.connection.execute(
            "SELECT id FROM setup_event_outcomes WHERE setup_event_id=? AND outcome_type=? AND content_hash=?",
            (event_id, outcome_type, content_hash)).fetchone()["id"])

    def record_followup_bar(self, bar: MarketBar, *, collection_version: str,
                            observed_at: datetime | None = None,
                            source_bar_version: object | None = None,
                            horizon_minutes: int = 60) -> int:
        """Keep raw one-hour follow-up candles for fired and rejected setups.

        These are descriptive price observations, not execution prices, label
        outcomes, exits or simulated P&L. Wall-clock offsets preserve gaps; no
        interpolation or stop/target ordering is inferred.
        """
        end = ensure_utc(bar.timestamp)
        rows = self.store.connection.execute(
            "SELECT * FROM setup_events WHERE asset=? AND timeframe=? AND collection_version=? "
            "AND event_kind != 'invalidated' AND source_timestamp_utc >= ? AND source_timestamp_utc < ?",
            (bar.asset, bar.timeframe, collection_version,
             (end - timedelta(minutes=horizon_minutes)).isoformat(), end.isoformat()),
        ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            reference_close = payload.get("source_bar", {}).get("close")
            self.record_outcome(int(row["id"]), "raw_followup_bar_v1", {
                "contract": "diagnostic_raw_candles_elapsed_60m_v1" if horizon_minutes == 60
                            else f"diagnostic_raw_candles_elapsed_{horizon_minutes}m_v1",
                "source_bar": _clean(bar.model_dump(mode="json")),
                "source_bar_version": str(source_bar_version if source_bar_version is not None else _bar_version(bar)),
                "elapsed_seconds": (end - datetime.fromisoformat(row["source_timestamp_utc"])).total_seconds(),
                "reference_close": reference_close,
                "directional_close_change": ((bar.close - reference_close) * row["setup_side"]
                                             if reference_close is not None else None),
                "executable_entry": False,
            }, observed_at=observed_at)
        return len(rows)

    def prediction_links(self, event_id: int) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self.store.connection.execute(
            "SELECT * FROM setup_prediction_links WHERE setup_event_id=? ORDER BY id", (event_id,)))

    def outcomes(self, event_id: int) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self.store.connection.execute(
            "SELECT * FROM setup_event_outcomes WHERE setup_event_id=? ORDER BY id", (event_id,)))

    def _require_event(self, event_id: int) -> None:
        if self.store.connection.execute("SELECT id FROM setup_events WHERE id=?", (event_id,)).fetchone() is None:
            raise ValueError(f"Unknown setup event {event_id}")


def extract_setup_observations(policy_frame: pd.DataFrame, *, strategy: str) -> tuple[dict[str, Any], ...]:
    """Extract the complete latest-bar candidate surface, including rejected rules."""
    strategy = strategy.upper()
    detector = {"FRVP": detect_frvp_setups, "ICT": detect_ict_setups}[strategy]
    detected = detector(policy_frame)
    latest_position = len(policy_frame) - 1
    latest = policy_frame.iloc[-1]
    observations: list[dict[str, Any]] = []
    for prefix, research in (("", False), ("research_", True)):
        candidates = _records(detected.attrs.get(prefix + "candidate_events"))
        selected = _records(detected.attrs.get(prefix + "fired_events"))
        selected_keys = {(_position(item), str(item.get("setup_type")), int(item.get("setup_side") or 0))
                         for item in selected}
        for item in candidates:
            if _position(item) != latest_position:
                continue
            item = dict(item)
            key = (_position(item), str(item.get("setup_type")), int(item.get("setup_side") or 0))
            item["selected"] = bool(item.get("selected") or key in selected_keys)
            item["research"] = research
            if not item["selected"] and not item.get("rejection_reasons"):
                item["rejection_reasons"] = ["detector_priority" if item.get("eligible") else "detector_ineligible"]
            observations.append(item)
    if not observations and not any(key in detected.attrs for key in ("candidate_events", "research_candidate_events")):
        row = detected.iloc[-1].to_dict()
        if bool(row.get("fired")):
            row.update({"selected": True, "research": False})
            observations.append(row)
    geometry = {str(key): _clean(value) for key, value in latest.items()
                if str(key).startswith(strategy.lower() + "_")
                or str(key) in {"open", "high", "low", "close", "atr_14"}}
    for item in observations:
        item["geometry"] = {**geometry, **{key: _clean(value) for key, value in item.items()
                            if key not in {"rejection_reasons", "eligible", "selected", "research"}}}
        item["detector_identity"] = f"{strategy.lower()}.setups.detector.{detector.__name__}"
        item["detector_version"] = detector_source_version(strategy)
        item["detector_config"] = _clean(detected.attrs.get("detector_config", {}))
    return tuple(_clean(item) for item in observations)


@lru_cache(maxsize=2)
def detector_source_version(strategy: str) -> str:
    detector = detect_frvp_setups if strategy.upper() == "FRVP" else detect_ict_setups
    source_file = inspect.getsourcefile(detector)
    return "sha256:" + hashlib.sha256(Path(source_file).read_bytes()).hexdigest()


def _bar_version(bar: MarketBar) -> object:
    explicit = getattr(bar, "bar_version", None)
    return explicit if explicit is not None else source_bar_version(bar)


def _records(value: object) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return [dict(item) for item in value] if isinstance(value, (list, tuple)) else []


def _position(item: Mapping[str, Any]) -> int:
    return int(item.get("bar_index", item.get("source_row_idx", -1)))


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(_clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _event_hash(payload: Mapping[str, Any]) -> str:
    identity = dict(payload)
    if isinstance(identity.get("source_bar"), Mapping):
        identity["source_bar"] = {key: value for key, value in identity["source_bar"].items()
                                  if key != "last_observed_at"}
    return _hash(identity)


def _record(row) -> SetupEventRecord:
    return SetupEventRecord(
        event_id=int(row["id"]), event_key=row["event_key"], revision=int(row["revision"]),
        event_kind=row["event_kind"], strategy=row["strategy"], setup_type=row["setup_type"],
        setup_side=int(row["setup_side"]), setup_family=row["setup_family"], selected=bool(row["selected"]),
        source_timestamp=datetime.fromisoformat(row["source_timestamp_utc"]),
        source_bar_version=row["source_bar_version"], observed_at=datetime.fromisoformat(row["observed_at_utc"]),
        first_observed_at=datetime.fromisoformat(row["first_observed_at_utc"]),
        collection_version=row["collection_version"], payload=json.loads(row["payload_json"]),
    )
