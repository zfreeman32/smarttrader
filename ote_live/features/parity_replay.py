from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from features.io import standardize_market_frame
from ote_live.contracts.market_data import MarketBar
from ote_live.features.incremental_engine import IncrementalFeatureEngine
from ote_live.features.manifest import LiveRuntimeManifest

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FeatureParityMismatch:
    timestamp: datetime
    feature_name: str
    expected_value: object
    actual_value: object
    abs_diff: float | None = None


@dataclass(frozen=True)
class FeatureParityFeatureResult:
    feature_name: str
    compared_rows: int
    matched_rows: int
    mismatched_rows: int
    max_abs_diff: float | None = None


@dataclass(frozen=True)
class FeatureParityReport:
    asset: str
    timeframe: str
    model_ids: tuple[str, ...]
    selected_feature_names: tuple[str, ...]
    requested_evaluation_bars: int
    replay_bars_processed: int
    rows_ready_for_comparison: int
    rows_evaluated: int
    missing_reference_rows: int
    compared_cells: int
    matched_cells: int
    mismatched_cells: int
    matched_cell_ratio: float
    max_abs_diff: float | None
    feature_results: tuple[FeatureParityFeatureResult, ...]
    sample_mismatches: tuple[FeatureParityMismatch, ...]


def load_reference_feature_frame(
    manifests: Sequence[LiveRuntimeManifest],
    *,
    repo_root: Path = REPO_ROOT,
) -> pd.DataFrame:
    manifests = tuple(manifests)
    if not manifests:
        raise ValueError("At least one live manifest is required.")

    reference_path = _resolve_repo_relative_path(manifests[0].source_lineage.feature_csv, repo_root=repo_root)
    selected_feature_names = _selected_feature_union(manifests)
    usecols = {"datetime", *selected_feature_names}
    frame = pd.read_csv(
        reference_path,
        usecols=lambda column: column in usecols,
    )
    if "datetime" not in frame.columns:
        raise ValueError(f"Reference feature frame '{reference_path}' is missing a datetime column.")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["datetime"]).reset_index(drop=True)
    return frame


def load_replay_market_frame(
    manifests: Sequence[LiveRuntimeManifest],
    *,
    repo_root: Path = REPO_ROOT,
) -> pd.DataFrame:
    manifests = tuple(manifests)
    if not manifests:
        raise ValueError("At least one live manifest is required.")

    manifest = manifests[0]
    market_path = _resolve_repo_relative_path(manifest.source_lineage.upstream_source_path, repo_root=repo_root)
    raw = pd.read_csv(market_path)
    standardized = standardize_market_frame(
        raw,
        source_timezone=manifest.source_lineage.upstream_timezone_contract.source_timezone,
        canonical_timezone=manifest.timezone_contract.canonical_timezone,
    )
    if "datetime" not in standardized.columns:
        raise ValueError(f"Replay market frame '{market_path}' did not produce a datetime column.")
    standardized["datetime"] = pd.to_datetime(standardized["datetime"], errors="coerce", utc=True)
    standardized = standardized.dropna(subset=["datetime"]).reset_index(drop=True)
    return standardized


def replay_feature_parity(
    manifests: Iterable[LiveRuntimeManifest],
    *,
    market_frame: pd.DataFrame | None = None,
    reference_feature_frame: pd.DataFrame | None = None,
    evaluation_bars: int = 25,
    rolling_window_bars: int | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    max_mismatch_examples: int = 20,
) -> FeatureParityReport:
    manifests = tuple(manifests)
    if not manifests:
        raise ValueError("At least one live manifest is required.")
    if evaluation_bars <= 0:
        raise ValueError("evaluation_bars must be positive.")

    engine = IncrementalFeatureEngine(
        manifests,
        rolling_window_bars=rolling_window_bars,
    )
    selected_feature_names = engine.plan.selected_feature_names
    market_frame = market_frame.copy() if market_frame is not None else load_replay_market_frame(manifests)
    reference_feature_frame = (
        reference_feature_frame.copy()
        if reference_feature_frame is not None
        else load_reference_feature_frame(manifests)
    )

    replay_input = _select_replay_input_window(
        market_frame,
        history_prefill_bars=engine.plan.runtime_history_bars + engine.plan.additive_seed_prefix_bars,
        evaluation_bars=evaluation_bars,
    )
    actual_frame, rows_ready_for_comparison = _replay_selected_feature_rows(
        engine,
        replay_input,
        history_prefill_bars=engine.plan.runtime_history_bars + engine.plan.additive_seed_prefix_bars,
        reference_feature_frame=reference_feature_frame,
    )
    if not actual_frame.empty and len(actual_frame) > evaluation_bars:
        actual_frame = actual_frame.iloc[-evaluation_bars:].reset_index(drop=True)

    if actual_frame.empty:
        return FeatureParityReport(
            asset=engine.plan.asset,
            timeframe=engine.plan.timeframe,
            model_ids=tuple(manifest.model_id for manifest in manifests),
            selected_feature_names=selected_feature_names,
            requested_evaluation_bars=int(evaluation_bars),
            replay_bars_processed=int(len(replay_input)),
            rows_ready_for_comparison=rows_ready_for_comparison,
            rows_evaluated=0,
            missing_reference_rows=0,
            compared_cells=0,
            matched_cells=0,
            mismatched_cells=0,
            matched_cell_ratio=1.0,
            max_abs_diff=None,
            feature_results=(),
            sample_mismatches=(),
        )

    prepared_reference = reference_feature_frame.copy()
    prepared_reference["datetime"] = pd.to_datetime(prepared_reference["datetime"], errors="coerce", utc=True)
    prepared_reference = prepared_reference.dropna(subset=["datetime"]).reset_index(drop=True)

    actual_indexed = actual_frame.set_index("datetime")
    reference_indexed = prepared_reference.set_index("datetime")
    missing_reference_rows = int((~actual_indexed.index.isin(reference_indexed.index)).sum())
    joined_index = actual_indexed.index.intersection(reference_indexed.index)
    actual_aligned = actual_indexed.loc[joined_index, list(selected_feature_names)]
    reference_aligned = reference_indexed.loc[joined_index, list(selected_feature_names)]

    feature_results: list[FeatureParityFeatureResult] = []
    sample_mismatches: list[FeatureParityMismatch] = []
    compared_cells = 0
    matched_cells = 0
    mismatched_cells = 0
    global_max_abs_diff: float | None = None

    for feature_name in selected_feature_names:
        expected = reference_aligned[feature_name]
        actual = actual_aligned[feature_name]
        result, mismatches = _compare_feature_series(
            feature_name=feature_name,
            expected=expected,
            actual=actual,
            atol=atol,
            rtol=rtol,
            max_examples=max_mismatch_examples - len(sample_mismatches),
        )
        feature_results.append(result)
        compared_cells += result.compared_rows
        matched_cells += result.matched_rows
        mismatched_cells += result.mismatched_rows
        if result.max_abs_diff is not None:
            global_max_abs_diff = (
                result.max_abs_diff
                if global_max_abs_diff is None
                else max(global_max_abs_diff, result.max_abs_diff)
            )
        if mismatches and len(sample_mismatches) < max_mismatch_examples:
            sample_mismatches.extend(mismatches[: max_mismatch_examples - len(sample_mismatches)])

    matched_cell_ratio = 1.0 if compared_cells == 0 else matched_cells / compared_cells
    return FeatureParityReport(
        asset=engine.plan.asset,
        timeframe=engine.plan.timeframe,
        model_ids=tuple(manifest.model_id for manifest in manifests),
        selected_feature_names=selected_feature_names,
        requested_evaluation_bars=int(evaluation_bars),
        replay_bars_processed=int(len(replay_input)),
        rows_ready_for_comparison=rows_ready_for_comparison,
        rows_evaluated=int(len(joined_index)),
        missing_reference_rows=missing_reference_rows,
        compared_cells=compared_cells,
        matched_cells=matched_cells,
        mismatched_cells=mismatched_cells,
        matched_cell_ratio=matched_cell_ratio,
        max_abs_diff=global_max_abs_diff,
        feature_results=tuple(feature_results),
        sample_mismatches=tuple(sample_mismatches),
    )


def _selected_feature_union(manifests: Sequence[LiveRuntimeManifest]) -> tuple[str, ...]:
    selected: list[str] = []
    for manifest in manifests:
        for feature_name in manifest.feature_manifest.selected_feature_names:
            if feature_name not in selected:
                selected.append(feature_name)
    return tuple(selected)


def _resolve_repo_relative_path(path_value: str, *, repo_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _select_replay_input_window(
    market_frame: pd.DataFrame,
    *,
    history_prefill_bars: int,
    evaluation_bars: int,
) -> pd.DataFrame:
    required_rows = int(history_prefill_bars) + int(evaluation_bars)
    if len(market_frame) <= required_rows:
        return market_frame.reset_index(drop=True)
    return market_frame.iloc[-required_rows:].reset_index(drop=True)


def _replay_selected_feature_rows(
    engine: IncrementalFeatureEngine,
    market_frame: pd.DataFrame,
    *,
    history_prefill_bars: int,
    reference_feature_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    rows_ready_for_comparison = 0
    seed_offsets_initialized = False
    reference_indexed = reference_feature_frame.copy()
    reference_indexed["datetime"] = pd.to_datetime(reference_indexed["datetime"], errors="coerce", utc=True)
    reference_indexed = reference_indexed.dropna(subset=["datetime"]).set_index("datetime")

    for bar_index, row in enumerate(market_frame.itertuples(index=False), start=1):
        engine.ingest_bar(_market_bar_from_row(row, asset=engine.plan.asset, timeframe=engine.plan.timeframe))
        if bar_index < history_prefill_bars:
            continue
        feature_frame = engine.build_feature_frame()
        if not seed_offsets_initialized and engine.plan.additive_seed_feature_names:
            seed_offsets = _derive_additive_feature_seed_offsets(
                engine=engine,
                feature_frame=feature_frame,
                reference_feature_frame=reference_indexed,
            )
            engine.set_feature_seed_offsets(seed_offsets)
            seed_offsets_initialized = True
            if seed_offsets:
                feature_frame = engine.build_feature_frame()
        status = engine.warmup_status(feature_frame=feature_frame)
        if not status.ready:
            continue

        rows_ready_for_comparison += 1
        latest = feature_frame.iloc[-1]
        row_payload: dict[str, object] = {"datetime": pd.Timestamp(engine.state.latest_timestamp)}
        for feature_name in engine.plan.selected_feature_names:
            row_payload[feature_name] = latest[feature_name]
        rows.append(row_payload)

    return pd.DataFrame(rows), rows_ready_for_comparison


def _derive_additive_feature_seed_offsets(
    *,
    engine: IncrementalFeatureEngine,
    feature_frame: pd.DataFrame,
    reference_feature_frame: pd.DataFrame,
) -> dict[str, float]:
    if feature_frame.empty or not engine.plan.additive_seed_feature_names:
        return {}

    state_frame = engine.state.to_frame()
    if state_frame.empty:
        return {}

    window_start_timestamp = pd.Timestamp(state_frame["timestamp"].iloc[0])
    if window_start_timestamp not in reference_feature_frame.index:
        return {}

    reference_row = reference_feature_frame.loc[window_start_timestamp]
    if isinstance(reference_row, pd.DataFrame):
        reference_row = reference_row.iloc[-1]

    actual_row = feature_frame.iloc[0]
    offsets: dict[str, float] = {}
    for feature_name in engine.plan.additive_seed_feature_names:
        if feature_name not in actual_row.index or feature_name not in reference_row.index:
            continue
        actual_value = pd.to_numeric(pd.Series([actual_row[feature_name]]), errors="coerce").iloc[0]
        reference_value = pd.to_numeric(pd.Series([reference_row[feature_name]]), errors="coerce").iloc[0]
        if pd.isna(actual_value) or pd.isna(reference_value):
            continue
        offsets[feature_name] = float(reference_value - actual_value)
    return offsets


def _market_bar_from_row(row, *, asset: str, timeframe: str) -> MarketBar:
    return MarketBar(
        asset=asset,
        timeframe=timeframe,
        timestamp=pd.Timestamp(getattr(row, "datetime")).to_pydatetime(),
        open=float(getattr(row, "open")),
        high=float(getattr(row, "high")),
        low=float(getattr(row, "low")),
        close=float(getattr(row, "close")),
        volume=float(getattr(row, "volume", 0.0) or 0.0),
        bid=_optional_float(getattr(row, "bid", None)),
        ask=_optional_float(getattr(row, "ask", None)),
        spread=_optional_float(getattr(row, "spread", None)),
        source=getattr(row, "source", None),
    )


def _compare_feature_series(
    *,
    feature_name: str,
    expected: pd.Series,
    actual: pd.Series,
    atol: float,
    rtol: float,
    max_examples: int,
) -> tuple[FeatureParityFeatureResult, list[FeatureParityMismatch]]:
    expected_numeric = pd.to_numeric(expected, errors="coerce")
    actual_numeric = pd.to_numeric(actual, errors="coerce")

    both_na = expected.isna() & actual.isna()
    numeric_mask = expected_numeric.notna() & actual_numeric.notna()
    numeric_match = pd.Series(False, index=expected.index)
    if numeric_mask.any():
        numeric_match.loc[numeric_mask] = np.isclose(
            expected_numeric.loc[numeric_mask].to_numpy(dtype=float),
            actual_numeric.loc[numeric_mask].to_numpy(dtype=float),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        )
    exact_match = expected.astype("string").fillna("<NA>") == actual.astype("string").fillna("<NA>")
    matches = both_na | numeric_match | exact_match

    compared_rows = int(len(expected))
    matched_rows = int(matches.sum())
    mismatched_rows = compared_rows - matched_rows

    abs_diff: pd.Series | None = None
    max_abs_diff: float | None = None
    if numeric_mask.any():
        abs_diff = (expected_numeric.loc[numeric_mask] - actual_numeric.loc[numeric_mask]).abs()
        if not abs_diff.empty:
            max_abs_diff = float(abs_diff.max())

    mismatches: list[FeatureParityMismatch] = []
    if mismatched_rows and max_examples > 0:
        mismatched_index = expected.index[~matches]
        for timestamp in mismatched_index[:max_examples]:
            diff_value: float | None = None
            if abs_diff is not None and timestamp in abs_diff.index and pd.notna(abs_diff.loc[timestamp]):
                diff_value = float(abs_diff.loc[timestamp])
            mismatches.append(
                FeatureParityMismatch(
                    timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                    feature_name=feature_name,
                    expected_value=_to_python_scalar(expected.loc[timestamp]),
                    actual_value=_to_python_scalar(actual.loc[timestamp]),
                    abs_diff=diff_value,
                )
            )

    return (
        FeatureParityFeatureResult(
            feature_name=feature_name,
            compared_rows=compared_rows,
            matched_rows=matched_rows,
            mismatched_rows=mismatched_rows,
            max_abs_diff=max_abs_diff,
        ),
        mismatches,
    )


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _to_python_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
