"""Raw pre-imputation feature snapshots (section 4 R1).

Capture point: inside ``FeatureDatasetBuilder.build`` after every producer
(feature sets, transforms, inf -> NaN) and before any numeric fill: the
EURUSD in-builder ``ffill().fillna(0)``, the ES post-contract ``ffill+0``,
live default materialization, additive seed offsets and the frozen imputer.
Producer-internal HTF as-of alignment is part of each producer's definition and
is not undone. NaN is preserved.

Storage (``s4-raw-preimpute-snapshot-v1``) is normalized so no value is stored
twice across models: one producer snapshot per engine group and build (the
tail rows of the union of selected features) plus a per-model selected-feature
index. Missing-producer, fallback and stale-input states are explicit fields;
they are never inferred from values.

Capture is disabled unless a recorder is attached. It only copies values, so
features, scaled inputs, scores and decisions are unchanged.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

RAW_SNAPSHOT_SCHEMA_VERSION = "s4-raw-preimpute-snapshot-v1"
CAPTURE_POINT = "builder_after_producers_before_numeric_fill"
VALUE_PREFIX = "f:"
ROW_METADATA_COLUMNS = (
    "asset", "symbol", "contract_symbol", "instrument_id", "source", "source_timestamp", "bar_version",
    "is_complete", "feed_type", "first_observed_at", "last_observed_at", "observation_kind",
)
HTF_HELPER_COLUMNS = (
    "htf_30m_source_complete", "htf_30m_source_age_minutes", "htf_1h_source_complete", "htf_1h_source_age_minutes",
)
STORE_PROVENANCE_COLUMNS = (
    "source_row_idx", "contract_symbol", "source_timestamp", "bar_version", "is_complete", "feed_type",
    "first_observed_at", "last_observed_at", "observation_kind",
)
CONTRACT_CATEGORIES = (
    "available", "event_dependent_missing", "not_yet_observed", "missing_producer",
    "unverified_producer", "missing_history_or_input", "stale_input",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAX_GAP_SLOTS = 20_000

ExpectedOpen = Callable[[pd.Timestamp], bool]


@dataclass(frozen=True)
class RawFeatureCapture:
    """Tail rows of one build, copied before numeric fill."""

    feature_names: tuple[str, ...]
    values: np.ndarray
    produced: tuple[bool, ...]
    nonnumeric_features: tuple[str, ...]
    producers: dict[str, str]
    timestamps: tuple[pd.Timestamp, ...]
    window_positions: tuple[int, ...]
    window_rows: int
    row_metadata: dict[str, list[Any]]
    previous_timestamp: pd.Timestamp | None
    previous_contract_symbol: Any


def capture_pre_fill(
    working: pd.DataFrame,
    producers: Mapping[str, str],
    feature_names: Sequence[str],
    rows: int,
) -> RawFeatureCapture:
    """Copy the last ``rows`` rows of ``feature_names``; never mutates ``working``."""
    rows = max(1, int(rows))
    total = len(working)
    start = max(0, total - rows)
    tail = working.iloc[start:]
    values = np.full((len(tail), len(feature_names)), np.nan, dtype=np.float64)
    produced: list[bool] = []
    nonnumeric: list[str] = []
    for position, name in enumerate(feature_names):
        if name not in tail.columns:
            produced.append(False)
            continue
        produced.append(True)
        series = tail[name]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            values[:, position] = series.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            nonnumeric.append(name)
            values[:, position] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    time_column = "datetime" if "datetime" in working.columns else "timestamp"
    times = pd.to_datetime(working[time_column], errors="coerce", utc=True) if time_column in working.columns else None
    timestamps = tuple(times.iloc[start:]) if times is not None else tuple(pd.NaT for _ in range(len(tail)))
    metadata = {
        column: [_plain(value) for value in tail[column].tolist()]
        for column in (*ROW_METADATA_COLUMNS, *HTF_HELPER_COLUMNS)
        if column in tail.columns
    }
    previous_timestamp = times.iloc[start - 1] if times is not None and start > 0 else None
    previous_contract = (
        _plain(working["contract_symbol"].iloc[start - 1])
        if start > 0 and "contract_symbol" in working.columns else None
    )
    return RawFeatureCapture(
        feature_names=tuple(feature_names),
        values=values,
        produced=tuple(produced),
        nonnumeric_features=tuple(nonnumeric),
        producers={name: producers.get(name, "missing") for name in feature_names},
        timestamps=timestamps,
        window_positions=tuple(range(start, total)),
        window_rows=total,
        row_metadata=metadata,
        previous_timestamp=previous_timestamp,
        previous_contract_symbol=previous_contract,
    )


def consumed_rows(manifest) -> int:
    """Rows of the feature history one prediction consumes (XGB max lag + 1, TCN window)."""
    if str(manifest.backend).lower() == "xgboost":
        lags = list(manifest.feature_manifest.lag_steps or [0])
        return int(max(lags)) + 1
    return int(manifest.context_requirements.window_size)


def es_app_calendar() -> tuple[str, ExpectedOpen]:
    from ote_live.ingestion.market_calendar import MARKET_CALENDAR_VERSION, is_expected_market_bar_timestamp

    def expected_open(timestamp: pd.Timestamp) -> bool:
        return bool(is_expected_market_bar_timestamp(timestamp.to_pydatetime(), asset="ES", feature_context={}))

    return f"{MARKET_CALENDAR_VERSION}:no_bar_trading_hours", expected_open


def eurusd_calendar_v1(path: str | Path | None = None) -> tuple[str, ExpectedOpen]:
    """Pinned provisional EURUSD 5m expected-open calendar (lock-verified)."""
    path = Path(path) if path is not None else _REPO_ROOT / "configs" / "eurusd_5m_expected_open_calendar_v1.json"
    lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    if _file_sha256(path) != lock["sha256"]:
        raise ValueError("EURUSD calendar does not match its lock")
    rule = json.loads(path.read_text(encoding="utf-8"))
    open_minute = int(rule["weekly_open_utc"]["time"][:2]) * 60
    close_minute = int(rule["weekly_close_utc"]["time"][:2]) * 60
    holidays = [(int(item["month"]), int(item["day"])) for item in rule["holiday_closures"]]

    def expected_open(timestamp: pd.Timestamp) -> bool:
        ts = pd.Timestamp(timestamp).tz_convert("UTC")
        minute = ts.hour * 60 + ts.minute
        if ts.weekday() == 5 or (ts.weekday() == 6 and minute < open_minute) or (ts.weekday() == 4 and minute >= close_minute):
            return False
        for month, day in holidays:
            for year in (ts.year - 1, ts.year, ts.year + 1):
                holiday = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
                if holiday - pd.Timedelta(hours=2) <= ts < holiday + pd.Timedelta(hours=23):
                    return False
        return True

    return f"{rule['calendar_id']}:{rule['status']}", expected_open


class RawSnapshotRecorder:
    """Buffers per-build raw snapshots for one engine group; ``flush`` writes parquet tables."""

    def __init__(
        self,
        *,
        stream: str,
        group: str,
        calendar: tuple[str, ExpectedOpen] | None = None,
        bar_minutes: float = 5.0,
    ) -> None:
        self.stream = str(stream)
        self.group = str(group)
        self.calendar_id, self._expected_open = calendar if calendar is not None else ("none", None)
        self.step = pd.Timedelta(minutes=float(bar_minutes))
        self.feature_names: tuple[str, ...] = ()
        self.models: list[dict[str, Any]] = []
        self.capture_rows = 1
        self.errors: list[dict[str, Any]] = []
        self._producer_modules: dict[str, dict[str, Any]] = {}
        self._gap_cache: dict[tuple[pd.Timestamp, pd.Timestamp], int | None] = {}
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._builds: list[dict[str, Any]] = []
        self._rows: list[dict[str, Any]] = []
        self._feature_states: list[dict[str, Any]] = []
        self._model_states: list[dict[str, Any]] = []
        self._model_feature_states: list[dict[str, Any]] = []

    @property
    def pending_builds(self) -> int:
        return len(self._builds)

    def register_models(self, *, feature_names: Sequence[str], models: Sequence[Mapping[str, Any]]) -> None:
        self.feature_names = tuple(feature_names)
        position = {name: index for index, name in enumerate(self.feature_names)}
        self.models = []
        for model in models:
            entry = dict(model)
            entry["feature_positions"] = [position[name] for name in entry.pop("selected_feature_names")]
            self.models.append(entry)
        self.capture_rows = max([1, *(int(m["capture_rows"]) for m in self.models)])

    def record(
        self,
        *,
        build_id: str,
        generation_id: int | None,
        endpoint_timestamp: datetime,
        endpoint_source_row_idx: int | None,
        capture: RawFeatureCapture | None,
        build_metadata: Mapping[str, Any],
        warmup: Mapping[str, Any],
        cache_hit: bool,
        collection_version: str | None,
        fill_mode: str,
        store_provenance: Mapping[pd.Timestamp, Mapping[str, Any]],
    ) -> None:
        if capture is None:
            raise ValueError("Raw snapshot capture is missing for a built feature frame.")
        if capture.feature_names != self.feature_names:
            raise ValueError("Raw snapshot capture feature order differs from the registered group features.")
        endpoint = pd.Timestamp(endpoint_timestamp).tz_convert("UTC")
        if capture.timestamps and capture.timestamps[-1] != endpoint:
            raise ValueError(f"Raw snapshot capture ends at {capture.timestamps[-1]}, not the endpoint {endpoint}.")
        missing_producers = sorted(str(name) for name in build_metadata.get("missing_producer_feature_names") or ())
        fallbacks = sorted(str(name) for name in build_metadata.get("fallback_feature_names") or ())
        strategy_report = dict((build_metadata.get("feature_set_reports") or {}).get("strategy_signals") or {})
        self._builds.append({
            "schema_version": RAW_SNAPSHOT_SCHEMA_VERSION,
            "stream": self.stream,
            "group": self.group,
            "build_id": str(build_id),
            "generation_id": None if generation_id is None else int(generation_id),
            "endpoint_timestamp": endpoint.isoformat(),
            "endpoint_source_row_idx": None if endpoint_source_row_idx is None else int(endpoint_source_row_idx),
            "collection_version": collection_version,
            "cache_hit": bool(cache_hit),
            "capture_point": CAPTURE_POINT,
            "fill_mode": fill_mode,
            "window_rows": int(capture.window_rows),
            "captured_rows": int(len(capture.timestamps)),
            "input_contract_version": build_metadata.get("input_contract_version"),
            "missing_producer_features_json": json.dumps(missing_producers),
            "fallback_features_json": json.dumps(fallbacks),
            "nonnumeric_features_json": json.dumps(list(capture.nonnumeric_features)),
            "skipped_strategies_json": json.dumps(strategy_report.get("skipped_strategies") or [], default=str),
            "built_strategy_count": strategy_report.get("built"),
            "calendar_id": self.calendar_id,
        })
        previous_time, previous_contract = capture.previous_timestamp, capture.previous_contract_symbol
        last = len(capture.timestamps) - 1
        for index, timestamp in enumerate(capture.timestamps):
            stored = dict(store_provenance.get(timestamp) or {})
            engine_contract = capture.row_metadata.get("contract_symbol", [None] * (last + 1))[index]
            row: dict[str, Any] = {
                "build_id": str(build_id),
                "row_offset": index - last,
                "window_position": int(capture.window_positions[index]),
                "timestamp": timestamp.isoformat() if not pd.isna(timestamp) else None,
                "store_row_found": bool(stored),
                **{f"store_{name}": _plain(stored.get(name)) for name in STORE_PROVENANCE_COLUMNS},
                **{f"engine_{name}": values[index] for name, values in capture.row_metadata.items()},
                **self._gap_state(previous_time, timestamp),
                "roll_boundary": (
                    None if previous_contract is None or engine_contract is None
                    else bool(engine_contract != previous_contract)
                ),
            }
            row.update(
                (VALUE_PREFIX + name, float(capture.values[index, position]))
                for position, name in enumerate(self.feature_names)
            )
            self._rows.append(row)
            previous_time, previous_contract = timestamp, engine_contract
        missing_set, fallback_set = set(missing_producers), set(fallbacks)
        for position, name in enumerate(self.feature_names):
            producer = capture.producers.get(name, "missing")
            module = self._producer_module(producer)
            self._feature_states.append({
                "build_id": str(build_id),
                "feature": name,
                "producer": producer,
                "producer_module": module["path"],
                "producer_module_sha256": module["sha256"],
                "produced": bool(capture.produced[position]),
                "group_state": (
                    "fallback_zero_default" if name in fallback_set
                    else "missing_producer" if name in missing_set or not capture.produced[position]
                    else "produced"
                ),
            })
        contracts = dict(build_metadata.get("model_input_contracts") or {})
        for model in self.models:
            model_id = model["model_id"]
            contract = dict(contracts.get(model_id) or {})
            status = dict(warmup.get(model_id) or {})
            self._model_states.append({
                "build_id": str(build_id),
                "model_id": model_id,
                "warmup_ready": status.get("ready"),
                "history_ready": status.get("history_ready"),
                "features_ready": status.get("features_ready"),
                "bars_available": status.get("bars_available"),
                "minimum_history_bars": status.get("minimum_history_bars"),
                "valid_feature_count": status.get("valid_feature_count"),
                "consumed_rows": int(model["consumed_rows"]),
                "consumed_rows_captured": bool(len(capture.timestamps) >= int(model["consumed_rows"])),
                "contract_status": contract.get("status", "input_contract_unavailable"),
                "contract_reasons_json": json.dumps(list(contract.get("reasons") or [])),
                "contract_fallback_features_json": json.dumps(list(contract.get("fallback_features") or [])),
            })
            category = {
                feature: name
                for name in CONTRACT_CATEGORIES
                for feature in (contract.get("feature_states") or {}).get(name, ())
            }
            for position in model["feature_positions"]:
                feature = self.feature_names[position]
                self._model_feature_states.append({
                    "build_id": str(build_id),
                    "model_id": model_id,
                    "feature": feature,
                    "contract_category": category.get(feature, "not_evaluated"),
                })

    def note_failure(self, *, build_id: str, endpoint_timestamp: datetime, error: BaseException) -> None:
        self.errors.append({
            "build_id": str(build_id),
            "endpoint_timestamp": pd.Timestamp(endpoint_timestamp).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
        })

    def flush(self, directory: str | Path, prefix: str) -> dict[str, str]:
        """Write buffered tables as ``<prefix>.<table>.parquet`` plus ``<prefix>.index.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        tables = {
            "builds": self._builds,
            "rows": self._rows,
            "feature_states": self._feature_states,
            "model_states": self._model_states,
            "model_feature_states": self._model_feature_states,
        }
        for name, records in tables.items():
            path = directory / f"{prefix}.{name}.parquet"
            pd.DataFrame.from_records(records).to_parquet(path, index=False)
            hashes[path.name] = _file_sha256(path)
        index = {
            "schema_version": RAW_SNAPSHOT_SCHEMA_VERSION,
            "capture_point": CAPTURE_POINT,
            "stream": self.stream,
            "group": self.group,
            "calendar_id": self.calendar_id,
            "value_column_prefix": VALUE_PREFIX,
            "feature_names": list(self.feature_names),
            "models": self.models,
            "producer_modules": self._producer_modules,
            "errors": self.errors,
            "tables_sha256": hashes,
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        index_path = directory / f"{prefix}.index.json"
        index_path.write_text(json.dumps(index, indent=1, default=str) + "\n", encoding="utf-8")
        hashes[index_path.name] = _file_sha256(index_path)
        self._reset_buffers()
        self.errors = []
        return hashes

    def _gap_state(self, previous: pd.Timestamp | None, current: pd.Timestamp) -> dict[str, Any]:
        if previous is None or pd.isna(previous) or pd.isna(current):
            return {"gap_before_slots": None, "gap_expected_open_slots": None, "gap_state": "window_start"}
        slots = int((current - previous) / self.step) - 1
        if slots <= 0:
            return {"gap_before_slots": max(slots, 0), "gap_expected_open_slots": 0,
                    "gap_state": "none" if slots == 0 else "sequence_invalid"}
        key = (previous, current)
        if key not in self._gap_cache:
            if self._expected_open is None or slots > _MAX_GAP_SLOTS:
                self._gap_cache[key] = None
            else:
                missing = pd.date_range(previous + self.step, current - self.step, freq=self.step)
                self._gap_cache[key] = int(sum(bool(self._expected_open(ts)) for ts in missing))
        expected = self._gap_cache[key]
        state = "calendar_unavailable" if expected is None else "closure" if expected == 0 else "gap"
        return {"gap_before_slots": slots, "gap_expected_open_slots": expected, "gap_state": state}

    def _producer_module(self, producer: str) -> dict[str, Any]:
        if producer not in self._producer_modules:
            kind, _, name = producer.partition(":")
            path: Path | None = None
            if kind == "feature_set":
                from features.registry import FEATURE_REGISTRY

                try:
                    source = inspect.getsourcefile(FEATURE_REGISTRY.get(name).builder)
                    path = Path(source) if source else None
                except (KeyError, TypeError):
                    path = None
            elif kind == "transform":
                path = _REPO_ROOT / "features" / "transforms.py"
            relative = None
            if path is not None and path.is_file():
                try:
                    relative = path.resolve().relative_to(_REPO_ROOT).as_posix()
                except ValueError:
                    relative = path.as_posix()
            self._producer_modules[producer] = {
                "path": relative,
                "sha256": _file_sha256(path) if path is not None and path.is_file() else None,
            }
        return self._producer_modules[producer]


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
