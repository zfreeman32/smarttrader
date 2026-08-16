from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

POSITIVE_OUTCOMES = frozenset({"tp", "timeout_profit"})
PREPARED_SPLIT_NAMES = ("train", "val", "test")
DEFAULT_TARGET_NAMES = (
    "long_ict_reversal",
    "short_ict_reversal",
    "long_ict_continuation",
    "short_ict_continuation",
    "long_ict_meta",
    "short_ict_meta",
)


def build_ict_leakage_control_audit(
    events: pd.DataFrame,
    *,
    phase02_metadata: Mapping[str, Any] | None = None,
    prepared_root: str | Path | None = None,
    target_names: tuple[str, ...] = DEFAULT_TARGET_NAMES,
    bootstrap_sample_size: int = 256,
    bootstrap_max_events: int = 3000,
    bootstrap_random_state: int = 42,
) -> dict[str, Any]:
    event_windows = build_ict_event_window_frame(events)
    phase02_metadata = phase02_metadata or {}
    swing_confirm_bars = resolve_ict_swing_confirm_bars(phase02_metadata)
    prepared_root_path = Path(prepared_root) if prepared_root is not None else None

    target_rows: list[dict[str, Any]] = []
    leakage_target_count = 0
    prepared_target_count = 0
    overall_max_window_bars = int(event_windows["realized_window_bars"].max()) if not event_windows.empty else 0

    for target_name in target_names:
        target_events = event_windows.loc[event_windows["target_name"].eq(target_name)].copy()
        target_summary = _build_target_leakage_summary(
            target_name=target_name,
            target_events=target_events,
            swing_confirm_bars=swing_confirm_bars,
            prepared_root=prepared_root_path,
            bootstrap_sample_size=bootstrap_sample_size,
            bootstrap_max_events=bootstrap_max_events,
            bootstrap_random_state=bootstrap_random_state,
        )
        split_audit = target_summary.get("prepared_split_audit") or {}
        if split_audit.get("available"):
            prepared_target_count += 1
            if not split_audit.get("passes_all_boundaries", False):
                leakage_target_count += 1
        target_rows.append(target_summary)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "headline": {
            "targets_covered": int(len(target_rows)),
            "prepared_targets_audited": int(prepared_target_count),
            "targets_with_boundary_leakage_or_embargo_failure": int(leakage_target_count),
            "overall_max_realized_window_bars": int(overall_max_window_bars),
            "overall_recommended_embargo_bars": int(overall_max_window_bars + swing_confirm_bars),
        },
        "contract": {
            "swing_confirm_bars": int(swing_confirm_bars),
            "overall_max_realized_window_bars": int(overall_max_window_bars),
            "overall_recommended_embargo_bars": int(overall_max_window_bars + swing_confirm_bars),
            "formula": "recommended_embargo_bars = max(realized_window_bars) + swing_confirm_bars",
        },
        "artifact_context": {
            "prepared_root": str(prepared_root_path) if prepared_root_path is not None else None,
            "usable_event_rows": int(len(event_windows)),
        },
        "targets": _json_safe(target_rows),
    }
    return _json_safe(summary)


def render_ict_leakage_control_markdown(summary: Mapping[str, Any]) -> str:
    headline = summary["headline"]
    contract = summary["contract"]
    lines = [
        "# ICT Leakage-Control Audit",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        "## Headline",
        "",
        f"- Targets covered: `{headline['targets_covered']}`",
        f"- Prepared targets audited: `{headline['prepared_targets_audited']}`",
        f"- Targets with boundary leakage or embargo failure: `{headline['targets_with_boundary_leakage_or_embargo_failure']}`",
        f"- Max realized window: `{headline['overall_max_realized_window_bars']}` bars",
        f"- Recommended embargo: `{headline['overall_recommended_embargo_bars']}` bars",
        "",
        "## Contract",
        "",
        f"- Swing confirm bars: `{contract['swing_confirm_bars']}`",
        f"- Formula: `{contract['formula']}`",
        "",
        "## Target Summaries",
        "",
    ]

    for row in summary["targets"]:
        lines.append(
            f"- `{row['target_name']}`: usable=`{row['usable_events']}`, base_rate=`{_fmt(row['base_rate_pct'])}%`, "
            f"max_window=`{row['realized_window_bars']['max']}`, embargo=`{row['recommended_embargo_bars']}`"
        )
        split_audit = row.get("prepared_split_audit") or {}
        if split_audit.get("available"):
            lines.append(
                f"  prepared_split_pass=`{split_audit['passes_all_boundaries']}` "
                f"matched_events=`{split_audit['matched_event_count']}/{row['usable_events']}`"
            )
            for boundary in split_audit.get("boundaries", []):
                lines.append(
                    f"  `{boundary['boundary_name']}`: gap=`{boundary['raw_source_row_gap_bars']}`, "
                    f"needed=`{boundary['required_embargo_bars']}`, overlap_events=`{boundary['overlap_event_count']}`, "
                    f"pass=`{boundary['passes_boundary']}`"
                )
        bootstrap = row.get("bootstrap_diagnostic") or {}
        if bootstrap.get("status") == "ok":
            lines.append(
                f"  bootstrap_uniqueness=`{_fmt(bootstrap['bootstrap_avg_uniqueness'])}` vs "
                f"chrono=`{_fmt(bootstrap['chronological_avg_uniqueness'])}` "
                f"(delta=`{_fmt(bootstrap['uniqueness_delta'])}`)"
            )
        elif bootstrap:
            lines.append(f"  bootstrap_status=`{bootstrap.get('status')}`")
    return "\n".join(lines) + "\n"


def write_ict_leakage_control_audit(
    *,
    output_dir: str | Path,
    summary: Mapping[str, Any],
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_json = output_path / "ict_leakage_control_summary.json"
    summary_md = output_path / "ict_leakage_control_summary.md"

    summary_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    summary_md.write_text(render_ict_leakage_control_markdown(summary), encoding="utf-8")

    return {
        "summary_json": str(summary_json),
        "summary_markdown": str(summary_md),
    }


def build_ict_event_window_frame(events: pd.DataFrame) -> pd.DataFrame:
    working = events.copy()
    working["excluded"] = _as_bool_series(working.get("excluded", False), index=working.index)
    working = working.loc[~working["excluded"]].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "target_name",
                "event_direction",
                "label_family",
                "signal_index",
                "barrier_end_index",
                "realized_window_bars",
                "max_holding_bars",
                "positive_event",
            ]
        )

    for column in ("signal_index", "barrier_end_index", "max_holding_bars"):
        working[column] = pd.to_numeric(working.get(column), errors="coerce")
    working = working.loc[working["signal_index"].notna() & working["barrier_end_index"].notna()].copy()
    if working.empty:
        return pd.DataFrame()

    working["signal_index"] = working["signal_index"].astype(int)
    working["barrier_end_index"] = working["barrier_end_index"].astype(int)
    working["realized_window_bars"] = (working["barrier_end_index"] - working["signal_index"]) + 1
    working["positive_event"] = working.get("tb_outcome", "").fillna("").astype(str).isin(POSITIVE_OUTCOMES)

    family_rows = working.loc[:, [
        "event_direction",
        "label_family",
        "signal_index",
        "barrier_end_index",
        "realized_window_bars",
        "max_holding_bars",
        "positive_event",
        "event_time",
        "setup_type",
    ]].copy()
    family_rows["target_name"] = family_rows["event_direction"].astype(str) + "_" + family_rows["label_family"].astype(str)

    meta_rows = family_rows.copy()
    meta_rows["label_family"] = "ict_meta"
    meta_rows["target_name"] = meta_rows["event_direction"].astype(str) + "_ict_meta"

    out = pd.concat([family_rows, meta_rows], ignore_index=True)
    out = out.sort_values(["target_name", "signal_index", "barrier_end_index"]).reset_index(drop=True)
    return out


def resolve_ict_swing_confirm_bars(phase02_metadata: Mapping[str, Any] | None) -> int:
    config = {}
    if isinstance(phase02_metadata, Mapping):
        config = phase02_metadata.get("config", {}) or {}
    try:
        return max(1, int(config.get("swing_window", 3)))
    except (TypeError, ValueError):
        return 3


def resolve_ict_recommended_embargo_bars(
    target_events: pd.DataFrame,
    *,
    swing_confirm_bars: int,
) -> int:
    if target_events.empty:
        return int(max(1, swing_confirm_bars))
    max_window = int(pd.to_numeric(target_events["realized_window_bars"], errors="coerce").dropna().max())
    return int(max_window + max(int(swing_confirm_bars), 0))


def compute_average_uniqueness(intervals: pd.DataFrame) -> float:
    if intervals.empty:
        return 0.0
    starts = intervals["signal_index"].to_numpy(dtype=int, copy=False)
    ends = intervals["barrier_end_index"].to_numpy(dtype=int, copy=False)
    concurrency = np.zeros(int(np.max(ends)) + 1, dtype=np.int32)
    for start, end in zip(starts, ends):
        concurrency[start : end + 1] += 1
    uniqueness_scores = np.empty(len(intervals), dtype=np.float64)
    for index, (start, end) in enumerate(zip(starts, ends)):
        uniqueness_scores[index] = float(np.mean(1.0 / np.maximum(concurrency[start : end + 1], 1)))
    return float(np.mean(uniqueness_scores))


def sequential_bootstrap_sample(
    intervals: pd.DataFrame,
    *,
    sample_size: int,
    random_state: int = 42,
    replace: bool = False,
) -> np.ndarray:
    if intervals.empty or sample_size <= 0:
        return np.array([], dtype=np.int64)

    starts = intervals["signal_index"].to_numpy(dtype=int, copy=False)
    ends = intervals["barrier_end_index"].to_numpy(dtype=int, copy=False)
    n_events = len(intervals)
    sample_size = int(sample_size)
    if not replace:
        sample_size = min(sample_size, n_events)

    offset = int(np.min(starts))
    local_starts = starts - offset
    local_ends = ends - offset
    lengths = (local_ends - local_starts + 1).astype(np.int64, copy=False)
    timeline_length = int(np.max(local_ends)) + 1

    coverage: list[list[int]] = [[] for _ in range(timeline_length)]
    for event_idx, (start, end) in enumerate(zip(local_starts, local_ends)):
        for bar in range(int(start), int(end) + 1):
            coverage[bar].append(int(event_idx))

    current_concurrency = np.zeros(timeline_length, dtype=np.int32)
    score_sums = lengths.astype(np.float64, copy=True)
    active_mask = np.ones(n_events, dtype=bool)
    selected: list[int] = []
    rng = np.random.default_rng(random_state)

    for _ in range(sample_size):
        candidate_indices = np.flatnonzero(active_mask) if not replace else np.arange(n_events, dtype=np.int64)
        if candidate_indices.size == 0:
            break

        scores = score_sums[candidate_indices] / lengths[candidate_indices]
        if not np.isfinite(scores).all() or float(scores.sum()) <= 0.0:
            probabilities = np.full(len(candidate_indices), 1.0 / max(len(candidate_indices), 1), dtype=np.float64)
        else:
            probabilities = scores / scores.sum()

        chosen_idx = int(rng.choice(candidate_indices, p=probabilities))
        selected.append(chosen_idx)

        for bar in range(int(local_starts[chosen_idx]), int(local_ends[chosen_idx]) + 1):
            old_concurrency = int(current_concurrency[bar])
            current_concurrency[bar] = old_concurrency + 1
            delta = (1.0 / float(old_concurrency + 2)) - (1.0 / float(old_concurrency + 1))
            if delta == 0.0:
                continue
            covered_candidates = coverage[bar]
            if not covered_candidates:
                continue
            score_sums[np.asarray(covered_candidates, dtype=np.int64)] += delta

        if not replace:
            active_mask[chosen_idx] = False

    return np.asarray(selected, dtype=np.int64)


def _build_target_leakage_summary(
    *,
    target_name: str,
    target_events: pd.DataFrame,
    swing_confirm_bars: int,
    prepared_root: Path | None,
    bootstrap_sample_size: int,
    bootstrap_max_events: int,
    bootstrap_random_state: int,
) -> dict[str, Any]:
    target_summary = {
        "target_name": target_name,
        "usable_events": int(len(target_events)),
        "positive_events": int(target_events["positive_event"].sum()) if not target_events.empty else 0,
        "base_rate_pct": float(target_events["positive_event"].mean() * 100.0) if not target_events.empty else 0.0,
        "realized_window_bars": {
            "median": _quantile(target_events.get("realized_window_bars"), 0.50),
            "p60": _quantile(target_events.get("realized_window_bars"), 0.60),
            "p75": _quantile(target_events.get("realized_window_bars"), 0.75),
            "max": int(pd.to_numeric(target_events.get("realized_window_bars"), errors="coerce").dropna().max())
            if not target_events.empty
            else 0,
        },
        "recommended_embargo_bars": int(
            resolve_ict_recommended_embargo_bars(target_events, swing_confirm_bars=swing_confirm_bars)
        ),
        "bootstrap_diagnostic": _bootstrap_diagnostic(
            target_events,
            sample_size=bootstrap_sample_size,
            max_events=bootstrap_max_events,
            random_state=bootstrap_random_state,
        ),
        "prepared_split_audit": {
            "available": False,
        },
    }

    if prepared_root is not None:
        split_audit = _audit_prepared_target_splits(
            prepared_root=prepared_root,
            target_name=target_name,
            target_events=target_events,
            required_embargo_bars=target_summary["recommended_embargo_bars"],
        )
        target_summary["prepared_split_audit"] = split_audit

    return target_summary


def _bootstrap_diagnostic(
    target_events: pd.DataFrame,
    *,
    sample_size: int,
    max_events: int,
    random_state: int,
) -> dict[str, Any]:
    if target_events.empty:
        return {"status": "no_events"}
    if len(target_events) > int(max_events):
        return {"status": "skipped_too_many_events", "event_count": int(len(target_events))}

    sample_size = min(int(sample_size), int(len(target_events)))
    chronological = target_events.sort_values(["signal_index", "barrier_end_index"]).head(sample_size).reset_index(drop=True)
    bootstrap_indices = sequential_bootstrap_sample(
        target_events.reset_index(drop=True),
        sample_size=sample_size,
        random_state=random_state,
        replace=False,
    )
    bootstrap_sample = target_events.reset_index(drop=True).iloc[bootstrap_indices].copy()
    chronological_uniqueness = compute_average_uniqueness(chronological)
    bootstrap_uniqueness = compute_average_uniqueness(bootstrap_sample)
    return {
        "status": "ok",
        "sample_size": int(sample_size),
        "replace": False,
        "chronological_avg_uniqueness": float(chronological_uniqueness),
        "bootstrap_avg_uniqueness": float(bootstrap_uniqueness),
        "uniqueness_delta": float(bootstrap_uniqueness - chronological_uniqueness),
    }


def _audit_prepared_target_splits(
    *,
    prepared_root: Path,
    target_name: str,
    target_events: pd.DataFrame,
    required_embargo_bars: int,
) -> dict[str, Any]:
    target_dir = prepared_root / target_name
    report_path = target_dir / "report.json"
    if not report_path.exists():
        return {"available": False, "reason": f"missing_report:{report_path}"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    split_rows = _load_prepared_split_source_rows(target_dir)
    if split_rows is None:
        return {"available": False, "reason": "missing_split_csv"}

    split_sets = {name: set(values.tolist()) for name, values in split_rows.items()}
    matched_counts = {
        name: int(target_events["signal_index"].isin(values).sum())
        for name, values in split_rows.items()
    }
    matched_event_count = int(sum(matched_counts.values()))

    boundaries = []
    passes_all_boundaries = True
    for current_name, next_name in (("train", "val"), ("val", "test")):
        current_rows = split_rows[current_name]
        next_rows = split_rows[next_name]
        current_events = target_events.loc[target_events["signal_index"].isin(split_sets[current_name])].copy()
        if current_rows.empty or next_rows.empty:
            boundaries.append(
                {
                    "boundary_name": f"{current_name}_to_{next_name}",
                    "available": False,
                }
            )
            continue

        current_end = int(current_rows.max())
        next_start = int(next_rows.min())
        raw_source_gap = int(next_start - current_end - 1)
        overlap_events = current_events.loc[current_events["barrier_end_index"].ge(next_start)].copy()
        observed_post_window_gap = (
            int(next_start - int(current_events["barrier_end_index"].max()) - 1)
            if not current_events.empty
            else raw_source_gap
        )
        max_overlap_bars = (
            int((overlap_events["barrier_end_index"] - next_start + 1).max())
            if not overlap_events.empty
            else 0
        )
        passes_boundary = raw_source_gap >= int(required_embargo_bars) and overlap_events.empty
        passes_all_boundaries = passes_all_boundaries and passes_boundary
        boundaries.append(
            {
                "boundary_name": f"{current_name}_to_{next_name}",
                "available": True,
                "current_split_event_count": int(len(current_events)),
                "next_split_row_count": int(len(next_rows)),
                "current_split_source_row_end": int(current_end),
                "next_split_source_row_start": int(next_start),
                "raw_source_row_gap_bars": int(raw_source_gap),
                "observed_post_window_gap_bars": int(observed_post_window_gap),
                "required_embargo_bars": int(required_embargo_bars),
                "overlap_event_count": int(len(overlap_events)),
                "max_overlap_bars": int(max_overlap_bars),
                "passes_boundary": bool(passes_boundary),
            }
        )

    split_ranges = {}
    for name, values in split_rows.items():
        if values.empty:
            split_ranges[name] = None
            continue
        split_ranges[name] = {
            "start_source_row_idx": int(values.min()),
            "end_source_row_idx": int(values.max()),
            "row_count": int(len(values)),
        }

    return {
        "available": True,
        "target_report_path": str(report_path),
        "matched_event_count": int(matched_event_count),
        "unmatched_event_count": int(max(len(target_events) - matched_event_count, 0)),
        "matched_event_count_by_split": matched_counts,
        "report_split_counts": report.get("split_counts", {}),
        "split_source_row_ranges": split_ranges,
        "boundaries": boundaries,
        "passes_all_boundaries": bool(passes_all_boundaries),
    }


def _load_prepared_split_source_rows(target_dir: Path) -> dict[str, pd.Series] | None:
    rows: dict[str, pd.Series] = {}
    for split_name in PREPARED_SPLIT_NAMES:
        split_path = target_dir / f"{split_name}.csv"
        if not split_path.exists():
            return None
        frame = pd.read_csv(split_path, usecols=["source_row_idx"])
        rows[split_name] = pd.to_numeric(frame["source_row_idx"], errors="coerce").dropna().astype(np.int64)
    return rows


def _as_bool_series(value: Any, *, index: pd.Index | None = None) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value.fillna(False)
        normalized = value.fillna(False).astype(str).str.strip().str.lower()
        return normalized.isin({"1", "true", "t", "yes"})
    if index is None:
        return pd.Series(bool(value))
    return pd.Series(bool(value), index=index)


def _quantile(values: Any, q: float) -> float | None:
    series = pd.Series(dtype=float) if values is None else pd.to_numeric(values, errors="coerce")
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return None
    return float(cleaned.quantile(q))


def _fmt(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2f}"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_list())
    return str(value)
