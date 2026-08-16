from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ict.setups.detector import SETUP_MIN_SPACING


STRUCTURAL_SESSION_SETUP_TYPES = frozenset(
    {
        "session_open_manipulation_pre_ib",
        "session_open_manipulation_post_ib",
    }
)
POSITIVE_OUTCOMES = frozenset({"tp", "timeout_profit"})


def build_ict_spacing_cooldown_diagnostics(
    events: pd.DataFrame,
    *,
    diagnostics: dict[str, Any] | None = None,
    current_spacing: Mapping[str, int] | None = None,
    min_refit_events: int = 50,
    min_gap_observations: int = 20,
    alignment_tolerance_bars: int = 1,
    low_overlap_tolerance_pct: float = 5.0,
    minimum_spacing_bars: int = 2,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build realized barrier-timing and clustering diagnostics for ICT setup spacing."""

    current_spacing_map = {str(key): int(value) for key, value in (current_spacing or SETUP_MIN_SPACING).items()}
    diagnostics = diagnostics or {}
    usable = _prepare_spacing_frame(events)

    setup_rows: list[dict[str, Any]] = []
    for setup_type, group in usable.groupby("setup_type", dropna=False, sort=True):
        setup_type = str(setup_type)
        side_rows = [_build_side_metrics(side_group) for _, side_group in group.groupby("setup_side", dropna=False, sort=True)]
        pooled = _build_side_metrics(group)
        current_value = int(current_spacing_map.get(setup_type, 0))
        recommendation = _recommend_spacing(
            setup_type=setup_type,
            current_spacing=current_value,
            pooled_metrics=pooled,
            min_refit_events=min_refit_events,
            min_gap_observations=min_gap_observations,
            alignment_tolerance_bars=alignment_tolerance_bars,
            low_overlap_tolerance_pct=low_overlap_tolerance_pct,
            minimum_spacing_bars=minimum_spacing_bars,
        )
        setup_rows.append(
            {
                "setup_type": setup_type,
                "label_family": str(group["label_family"].mode(dropna=False).iloc[0]) if not group.empty else "",
                "current_spacing_bars": current_value,
                **pooled,
                **recommendation,
                "side_metrics": side_rows,
            }
        )

    setup_frame = pd.DataFrame(setup_rows)
    if not setup_frame.empty:
        setup_frame = setup_frame.sort_values(["recommended_spacing_bars", "setup_type"], ascending=[False, True])
    changed = setup_frame.loc[setup_frame["recommended_spacing_bars"] != setup_frame["current_spacing_bars"]].copy()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_diagnostics": diagnostics,
        "headline": {
            "usable_events": int(len(usable)),
            "setup_types_covered": int(setup_frame["setup_type"].nunique()),
            "changed_setup_count": int(len(changed)),
        },
        "recommendation_policy": {
            "min_refit_events": int(min_refit_events),
            "min_gap_observations": int(min_gap_observations),
            "alignment_tolerance_bars": int(alignment_tolerance_bars),
            "low_overlap_tolerance_pct": float(low_overlap_tolerance_pct),
            "minimum_spacing_bars": int(minimum_spacing_bars),
            "formula": (
                "raw_spacing = round(min(p60_first_barrier_bars, p25_same_setup_side_gap_bars)); "
                "keep current spacing when the raw recommendation is within tolerance or when overlap is already low; "
                "keep session-open patterns structural once-per-session."
            ),
        },
        "recommended_spacing": {
            row["setup_type"]: int(row["recommended_spacing_bars"])
            for row in setup_rows
        },
        "changed_setups": [
            {
                "setup_type": str(row["setup_type"]),
                "current_spacing_bars": int(row["current_spacing_bars"]),
                "recommended_spacing_bars": int(row["recommended_spacing_bars"]),
                "recommendation_status": str(row["recommendation_status"]),
                "recommendation_reason": str(row["recommendation_reason"]),
            }
            for row in changed.to_dict(orient="records")
        ],
        "setup_rows": _json_safe(setup_rows),
    }
    return _json_safe(summary), setup_frame


def render_ict_spacing_cooldown_markdown(summary: dict[str, Any], setup_frame: pd.DataFrame) -> str:
    headline = summary["headline"]
    changed = setup_frame.loc[setup_frame["recommended_spacing_bars"] != setup_frame["current_spacing_bars"]].copy()
    ordered = setup_frame.sort_values(["setup_type"]).copy()

    lines = [
        "# ICT Spacing / Cooldown Diagnostics",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        "## Headline",
        "",
        f"- Usable events analyzed: `{headline['usable_events']}`",
        f"- Setup types covered: `{headline['setup_types_covered']}`",
        f"- Setup types with changed recommendations: `{headline['changed_setup_count']}`",
        "",
        "## Recommendations",
        "",
    ]
    if changed.empty:
        lines.append("- No spacing changes recommended by the current policy.")
    else:
        for row in changed.itertuples(index=False):
            lines.append(
                f"- `{row.setup_type}`: `{row.current_spacing_bars} -> {row.recommended_spacing_bars}` "
                f"({row.recommendation_status}; p60_first_barrier=`{_fmt(row.p60_first_barrier_bars)}`, "
                f"p25_gap=`{_fmt(row.p25_same_setup_side_gap_bars)}`, overlap=`{_fmt(row.overlap_rate_prev_pct)}%`)"
            )
    lines.extend(["", "## Setup Table", ""])
    for row in ordered.itertuples(index=False):
        lines.append(
            f"- `{row.setup_type}`: current=`{row.current_spacing_bars}`, recommended=`{row.recommended_spacing_bars}`, "
            f"count=`{row.usable_event_count}`, median_first=`{_fmt(row.median_first_barrier_bars)}`, "
            f"p60_first=`{_fmt(row.p60_first_barrier_bars)}`, p25_gap=`{_fmt(row.p25_same_setup_side_gap_bars)}`, "
            f"overlap=`{_fmt(row.overlap_rate_prev_pct)}%`, status=`{row.recommendation_status}`"
        )
    return "\n".join(lines) + "\n"


def write_ict_spacing_cooldown_diagnostics(
    *,
    output_dir: str | Path,
    summary: dict[str, Any],
    setup_frame: pd.DataFrame,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_json = output_path / "ict_spacing_cooldown_summary.json"
    summary_md = output_path / "ict_spacing_cooldown_summary.md"
    setup_csv = output_path / "ict_spacing_cooldown_setup_rows.csv"

    summary_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    summary_md.write_text(render_ict_spacing_cooldown_markdown(summary, setup_frame), encoding="utf-8")
    setup_frame.drop(columns=["side_metrics"], errors="ignore").to_csv(setup_csv, index=False)

    return {
        "summary_json": str(summary_json),
        "summary_markdown": str(summary_md),
        "setup_rows_csv": str(setup_csv),
    }


def _prepare_spacing_frame(events: pd.DataFrame) -> pd.DataFrame:
    working = events.copy()
    working["excluded"] = _as_bool_series(working.get("excluded", False), index=working.index)
    working = working.loc[~working["excluded"]].copy()
    if working.empty:
        return working

    for column in (
        "signal_index",
        "entry_index",
        "target_hit_index",
        "barrier_end_index",
        "tb_bars_held",
        "setup_side",
    ):
        working[column] = pd.to_numeric(working.get(column), errors="coerce")

    target_hit = working["target_hit_index"].to_numpy(dtype=float, copy=False)
    barrier_end = working["barrier_end_index"].to_numpy(dtype=float, copy=False)
    first_index = np.where(np.isfinite(target_hit), target_hit, barrier_end)
    working["first_barrier_index"] = first_index
    working["first_barrier_bars"] = first_index - working["entry_index"].to_numpy(dtype=float, copy=False) + 1.0
    working["positive_event"] = working["tb_outcome"].fillna("").astype(str).isin(POSITIVE_OUTCOMES)
    working = working.sort_values(["setup_type", "setup_side", "signal_index"]).reset_index(drop=True)
    return working


def _build_side_metrics(group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values("signal_index").copy()
    gaps = ordered["signal_index"].diff()
    prev_signal = ordered["signal_index"].shift(1)
    prev_first = ordered["first_barrier_bars"].shift(1)
    overlap = ((ordered["signal_index"] - prev_signal) < prev_first).fillna(False)

    return {
        "setup_side": int(ordered["setup_side"].mode(dropna=False).iloc[0]) if "setup_side" in ordered.columns else 0,
        "usable_event_count": int(len(ordered)),
        "gap_observation_count": int(gaps.notna().sum()),
        "base_rate_pct": float(ordered["positive_event"].mean() * 100.0) if len(ordered) else 0.0,
        "median_first_barrier_bars": _median(ordered["first_barrier_bars"]),
        "p60_first_barrier_bars": _quantile(ordered["first_barrier_bars"], 0.60),
        "p75_first_barrier_bars": _quantile(ordered["first_barrier_bars"], 0.75),
        "median_tb_bars_held": _median(ordered["tb_bars_held"]),
        "median_same_setup_side_gap_bars": _median(gaps),
        "p25_same_setup_side_gap_bars": _quantile(gaps, 0.25),
        "p10_same_setup_side_gap_bars": _quantile(gaps, 0.10),
        "overlap_rate_prev_pct": float(overlap.mean() * 100.0) if len(overlap) else 0.0,
    }


def _recommend_spacing(
    *,
    setup_type: str,
    current_spacing: int,
    pooled_metrics: Mapping[str, Any],
    min_refit_events: int,
    min_gap_observations: int,
    alignment_tolerance_bars: int,
    low_overlap_tolerance_pct: float,
    minimum_spacing_bars: int,
) -> dict[str, Any]:
    count = int(pooled_metrics.get("usable_event_count", 0) or 0)
    gap_count = int(pooled_metrics.get("gap_observation_count", 0) or 0)

    if setup_type in STRUCTURAL_SESSION_SETUP_TYPES:
        return {
            "raw_recommended_spacing_bars": int(current_spacing),
            "recommended_spacing_bars": int(current_spacing),
            "recommendation_status": "structural_once_per_session_keep",
            "recommendation_reason": "Session-open manipulation setups remain capped as once-per-session opening patterns.",
        }

    if count < int(min_refit_events) or gap_count < int(min_gap_observations):
        return {
            "raw_recommended_spacing_bars": int(current_spacing),
            "recommended_spacing_bars": int(current_spacing),
            "recommendation_status": "insufficient_sample_keep",
            "recommendation_reason": "Sample size or gap observations are too small to refit spacing reliably.",
        }

    p60_first = _safe_float(pooled_metrics.get("p60_first_barrier_bars"))
    p25_gap = _safe_float(pooled_metrics.get("p25_same_setup_side_gap_bars"))
    overlap_rate = _safe_float(pooled_metrics.get("overlap_rate_prev_pct"))
    if p60_first is None:
        return {
            "raw_recommended_spacing_bars": int(current_spacing),
            "recommended_spacing_bars": int(current_spacing),
            "recommendation_status": "missing_barrier_metrics_keep",
            "recommendation_reason": "No usable barrier timing metrics were available for this setup.",
        }

    raw_spacing = p60_first if p25_gap is None else min(p60_first, p25_gap)
    raw_spacing_bars = max(int(minimum_spacing_bars), int(round(float(raw_spacing))))

    if abs(raw_spacing_bars - int(current_spacing)) <= int(alignment_tolerance_bars):
        return {
            "raw_recommended_spacing_bars": int(raw_spacing_bars),
            "recommended_spacing_bars": int(current_spacing),
            "recommendation_status": "keep_current_aligned",
            "recommendation_reason": "Current spacing is already aligned with realized first-barrier timing within tolerance.",
        }

    if raw_spacing_bars < int(current_spacing) and overlap_rate <= float(low_overlap_tolerance_pct):
        return {
            "raw_recommended_spacing_bars": int(raw_spacing_bars),
            "recommended_spacing_bars": int(current_spacing),
            "recommendation_status": "keep_current_low_overlap",
            "recommendation_reason": "Current spacing is slightly conservative, but overlap is already low so no reduction is needed.",
        }

    status = "raise_spacing_from_barrier_timing" if raw_spacing_bars > int(current_spacing) else "lower_spacing_from_barrier_timing"
    reason = (
        "Realized first-barrier timing materially exceeds the current spacing."
        if raw_spacing_bars > int(current_spacing)
        else "Realized first-barrier timing supports a tighter spacing without excessive clustering."
    )
    return {
        "raw_recommended_spacing_bars": int(raw_spacing_bars),
        "recommended_spacing_bars": int(raw_spacing_bars),
        "recommendation_status": status,
        "recommendation_reason": reason,
    }


def _as_bool_series(value: Any, *, index: pd.Index | None = None) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value.fillna(False)
        normalized = value.fillna(False).astype(str).str.strip().str.lower()
        return normalized.isin({"1", "true", "t", "yes"})
    if index is None:
        return pd.Series(bool(value))
    return pd.Series(bool(value), index=index)


def _median(values: pd.Series) -> float | None:
    cleaned = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return None
    return float(cleaned.median())


def _quantile(values: pd.Series, q: float) -> float | None:
    cleaned = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return None
    return float(cleaned.quantile(q))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _fmt(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2f}"


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
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_list())
    return str(value)
