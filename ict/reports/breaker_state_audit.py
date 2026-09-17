from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.io import standardize_market_frame


BREAKER_REQUIRED_COLUMNS = (
    "dist_to_bull_breaker_atr",
    "dist_to_bear_breaker_atr",
    "bull_breaker_age_bars",
    "bear_breaker_age_bars",
    "ict_bull_breaker_retest_count",
    "ict_bear_breaker_retest_count",
    "ict_active_bull_breaker_count",
    "ict_active_bear_breaker_count",
    "ict_nearest_bull_breaker_id",
    "ict_nearest_bear_breaker_id",
    "ict_nearest_bull_breaker_source_order_block_id",
    "ict_nearest_bear_breaker_source_order_block_id",
    "ict_nearest_bull_breaker_lower",
    "ict_nearest_bull_breaker_upper",
    "ict_nearest_bear_breaker_lower",
    "ict_nearest_bear_breaker_upper",
    "ict_nearest_bull_breaker_formed_index",
    "ict_nearest_bear_breaker_formed_index",
    "ict_nearest_bull_breaker_source_order_block_formed_index",
    "ict_nearest_bear_breaker_source_order_block_formed_index",
    "ict_bull_breaker_retest_event",
    "ict_bear_breaker_retest_event",
    "ict_bull_breaker_created_event",
    "ict_bear_breaker_created_event",
)

OPTIONAL_CONTEXT_COLUMNS = (
    "datetime",
    "timestamp",
    "ts_event",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "session_date",
    "ict_session_phase_code",
    "ict_is_rth",
)

REVIEW_COLUMNS = (
    "review_bucket",
    "event_time",
    "side",
    "event_kind",
    "row_index",
    "breaker_id",
    "source_order_block_id",
    "source_order_block_age_at_break_bars",
    "retest_lag_bars",
    "breaker_age_bars",
    "retest_count",
    "dist_to_breaker_atr",
    "close",
    "breaker_lower",
    "breaker_upper",
)


def load_ict_breaker_feature_surface(path: str | Path) -> pd.DataFrame:
    """Load only columns needed for breaker-state auditing."""

    return standardize_market_frame(
        pd.read_csv(
            path,
            usecols=lambda column: _use_breaker_audit_column(str(column)),
        )
    )


def build_ict_breaker_state_audit(
    features: pd.DataFrame,
    *,
    min_total_retests: int = 100,
    min_side_retests: int = 25,
    min_retest_years: int = 3,
    top_n_review_rows: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize causal failed-order-block breaker state breadth and integrity."""

    working = features.reset_index(drop=True).copy()
    missing = [column for column in BREAKER_REQUIRED_COLUMNS if column not in working.columns]
    event_time = _event_time(working)

    base_summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_contract": {
            "rows": int(len(working)),
            "start_time_utc": _timestamp_or_none(event_time.min()) if event_time is not None else None,
            "end_time_utc": _timestamp_or_none(event_time.max()) if event_time is not None else None,
            "present_required_columns": [
                column for column in BREAKER_REQUIRED_COLUMNS if column in working.columns
            ],
            "missing_required_columns": missing,
            "has_required_breaker_surface": not missing,
        },
        "thresholds": {
            "min_total_retests": int(min_total_retests),
            "min_side_retests": int(min_side_retests),
            "min_retest_years": int(min_retest_years),
        },
    }
    if missing:
        reasons = [f"missing_required_columns:{len(missing)}"]
        base_summary.update(
            {
                "headline": _empty_headline(),
                "breadth": _empty_breadth(),
                "integrity": _empty_integrity(),
                "readiness": {
                    "ready_for_classic_breaker_label_design": False,
                    "decision": "blocked_missing_columns",
                    "reasons": reasons,
                },
            }
        )
        return _json_safe(base_summary), pd.DataFrame(columns=REVIEW_COLUMNS)

    bull = _build_side_frame(working, event_time, side="bull")
    bear = _build_side_frame(working, event_time, side="bear")
    events = pd.concat([bull, bear], ignore_index=True)
    created = events.loc[events["event_kind"].eq("created")].copy()
    retests = events.loc[events["event_kind"].eq("retest")].copy()

    integrity = _build_integrity_summary(events, working)
    breadth = _build_breadth_summary(created, retests, event_time)
    headline = _build_headline(created, retests, working)
    readiness = _build_readiness_summary(
        headline,
        breadth,
        integrity,
        min_total_retests=min_total_retests,
        min_side_retests=min_side_retests,
        min_retest_years=min_retest_years,
    )
    review_rows = _build_review_rows(created, retests, top_n=top_n_review_rows)

    summary = {
        **base_summary,
        "headline": headline,
        "breadth": breadth,
        "integrity": integrity,
        "readiness": readiness,
        "review_rows_preview": _json_safe(review_rows.head(16).to_dict(orient="records")),
    }
    return _json_safe(summary), review_rows


def render_ict_breaker_state_audit_markdown(summary: dict[str, Any]) -> str:
    headline = summary["headline"]
    contract = summary["data_contract"]
    breadth = summary["breadth"]
    integrity = summary["integrity"]
    readiness = summary["readiness"]

    lines = [
        "# ICT Breaker State Audit",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        "## Data Contract",
        "",
        f"- Rows: `{contract['rows']}`",
        f"- Window: `{contract['start_time_utc']}` to `{contract['end_time_utc']}`",
        f"- Required breaker surface present: `{contract['has_required_breaker_surface']}`",
        f"- Missing required columns: `{len(contract['missing_required_columns'])}`",
        "",
        "## Headline",
        "",
        f"- Created event bars: `{headline['total_created_event_bars']}`",
        f"- Retest event bars: `{headline['total_retest_event_bars']}`",
        f"- Bullish created / retests: `{headline['bull_created_event_bars']}` / `{headline['bull_retest_event_bars']}`",
        f"- Bearish created / retests: `{headline['bear_created_event_bars']}` / `{headline['bear_retest_event_bars']}`",
        f"- Max active bull / bear breakers: `{headline['max_active_bull_breakers']}` / `{headline['max_active_bear_breakers']}`",
        "",
        "## Breadth",
        "",
        f"- Years with retests: `{breadth['years_with_retests']}`",
        f"- Months with retests: `{breadth['months_with_retests']}`",
        f"- Retest event bars by year: `{_format_year_counts(breadth['retest_event_bars_by_year'])}`",
        "",
        "## Integrity",
        "",
        f"- Same-bar create and retest events: `{integrity['same_bar_create_and_retest_events']}`",
        f"- Retests after breaker formation: `{integrity['retest_after_formation_pct']:.2f}%`",
        f"- Retests with source order block: `{integrity['retest_has_source_order_block_pct']:.2f}%`",
        f"- Retests with finite bounds: `{integrity['retest_has_finite_bounds_pct']:.2f}%`",
        f"- Median retest lag bars: `{_fmt_optional(integrity['median_retest_lag_bars'])}`",
        "",
        "## Decision",
        "",
        f"- Ready for classic-breaker label design: `{readiness['ready_for_classic_breaker_label_design']}`",
        f"- Decision: `{readiness['decision']}`",
        f"- Reasons: `{', '.join(readiness['reasons']) if readiness['reasons'] else 'none'}`",
        "",
    ]
    return "\n".join(lines)


def write_ict_breaker_state_audit(
    *,
    output_dir: str | Path,
    summary: dict[str, Any],
    review_rows: pd.DataFrame,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_json = output_path / "ict_breaker_state_audit_summary.json"
    summary_md = output_path / "ict_breaker_state_audit_summary.md"
    review_csv = output_path / "ict_breaker_state_audit_review_rows.csv"

    summary_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    summary_md.write_text(render_ict_breaker_state_audit_markdown(summary), encoding="utf-8")
    review_rows.to_csv(review_csv, index=False)

    return {
        "summary_json": str(summary_json),
        "summary_markdown": str(summary_md),
        "review_rows_csv": str(review_csv),
    }


def _use_breaker_audit_column(column: str) -> bool:
    normalized = column.strip().lower()
    return normalized in set(BREAKER_REQUIRED_COLUMNS) | set(OPTIONAL_CONTEXT_COLUMNS)


def _build_side_frame(
    frame: pd.DataFrame,
    event_time: pd.Series | None,
    *,
    side: str,
) -> pd.DataFrame:
    side_label = "long" if side == "bull" else "short"
    created = _event_mask(frame[f"ict_{side}_breaker_created_event"])
    retest = _event_mask(frame[f"ict_{side}_breaker_retest_event"])
    created_rows = _event_rows(frame, event_time, side=side, side_label=side_label, event_kind="created", mask=created)
    retest_rows = _event_rows(frame, event_time, side=side, side_label=side_label, event_kind="retest", mask=retest)
    return pd.concat([created_rows, retest_rows], ignore_index=True)


def _event_rows(
    frame: pd.DataFrame,
    event_time: pd.Series | None,
    *,
    side: str,
    side_label: str,
    event_kind: str,
    mask: pd.Series,
) -> pd.DataFrame:
    rows = frame.loc[mask].copy()
    if rows.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    row_index = rows.index.to_series(index=rows.index).astype(int)
    formed_index = _numeric(rows[f"ict_nearest_{side}_breaker_formed_index"])
    source_formed_index = _numeric(rows[f"ict_nearest_{side}_breaker_source_order_block_formed_index"])
    result = pd.DataFrame(
        {
            "review_bucket": f"{side_label}_{event_kind}_sample",
            "event_time": _format_times(event_time.loc[rows.index]) if event_time is not None else "",
            "side": side_label,
            "event_kind": event_kind,
            "row_index": row_index.to_numpy(dtype=int),
            "breaker_id": _numeric(rows[f"ict_nearest_{side}_breaker_id"]),
            "source_order_block_id": _numeric(rows[f"ict_nearest_{side}_breaker_source_order_block_id"]),
            "source_order_block_age_at_break_bars": formed_index - source_formed_index,
            "retest_lag_bars": row_index.astype(float) - formed_index,
            "breaker_age_bars": _numeric(rows[f"{side}_breaker_age_bars"]),
            "retest_count": _numeric(rows[f"ict_{side}_breaker_retest_count"]),
            "dist_to_breaker_atr": _numeric(rows[f"dist_to_{side}_breaker_atr"]),
            "close": _numeric(rows.get("close", pd.Series(np.nan, index=rows.index))),
            "breaker_lower": _numeric(rows[f"ict_nearest_{side}_breaker_lower"]),
            "breaker_upper": _numeric(rows[f"ict_nearest_{side}_breaker_upper"]),
        },
        index=rows.index,
    )
    return result.reset_index(drop=True)


def _build_headline(created: pd.DataFrame, retests: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    bull_created = int((created["side"] == "long").sum()) if "side" in created.columns else 0
    bear_created = int((created["side"] == "short").sum()) if "side" in created.columns else 0
    bull_retests = int((retests["side"] == "long").sum()) if "side" in retests.columns else 0
    bear_retests = int((retests["side"] == "short").sum()) if "side" in retests.columns else 0
    return {
        "total_created_event_bars": int(len(created)),
        "total_retest_event_bars": int(len(retests)),
        "bull_created_event_bars": bull_created,
        "bear_created_event_bars": bear_created,
        "bull_retest_event_bars": bull_retests,
        "bear_retest_event_bars": bear_retests,
        "unique_breaker_ids_on_retest_rows": int(retests["breaker_id"].dropna().nunique()) if not retests.empty else 0,
        "unique_source_order_block_ids_on_retest_rows": (
            int(retests["source_order_block_id"].dropna().nunique()) if not retests.empty else 0
        ),
        "max_active_bull_breakers": _max_int(frame["ict_active_bull_breaker_count"]),
        "max_active_bear_breakers": _max_int(frame["ict_active_bear_breaker_count"]),
        "first_retest_time_utc": _first_time(retests),
        "last_retest_time_utc": _last_time(retests),
    }


def _build_breadth_summary(
    created: pd.DataFrame,
    retests: pd.DataFrame,
    event_time: pd.Series | None,
) -> dict[str, Any]:
    del event_time
    return {
        "years_with_created_events": _distinct_periods(created, "Y"),
        "years_with_retests": _distinct_periods(retests, "Y"),
        "months_with_retests": _distinct_periods(retests, "M"),
        "created_event_bars_by_year": _period_counts(created, "Y"),
        "retest_event_bars_by_year": _period_counts(retests, "Y"),
        "retest_event_bars_by_side": _side_counts(retests),
    }


def _build_integrity_summary(events: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        retests = events.copy()
    else:
        retests = events.loc[events["event_kind"].eq("retest")].copy()
    created_and_retest = int(
        (
            _event_mask(frame["ict_bull_breaker_created_event"])
            & _event_mask(frame["ict_bull_breaker_retest_event"])
        ).sum()
        + (
            _event_mask(frame["ict_bear_breaker_created_event"])
            & _event_mask(frame["ict_bear_breaker_retest_event"])
        ).sum()
    )
    retest_count = int(len(retests))
    after_formation = _numeric(retests.get("retest_lag_bars", pd.Series(dtype=float))).gt(0)
    source_present = _numeric(retests.get("source_order_block_id", pd.Series(dtype=float))).notna()
    bounds_present = _numeric(retests.get("breaker_lower", pd.Series(dtype=float))).notna() & _numeric(
        retests.get("breaker_upper", pd.Series(dtype=float))
    ).notna()
    source_age = _numeric(retests.get("source_order_block_age_at_break_bars", pd.Series(dtype=float)))

    return {
        "same_bar_create_and_retest_events": created_and_retest,
        "retest_after_formation_pct": _pct(after_formation.sum(), retest_count),
        "retest_has_source_order_block_pct": _pct(source_present.sum(), retest_count),
        "retest_has_finite_bounds_pct": _pct(bounds_present.sum(), retest_count),
        "source_order_block_formed_before_breaker_pct": _pct(source_age.gt(0).sum(), retest_count),
        "median_retest_lag_bars": _median_or_none(retests.get("retest_lag_bars", pd.Series(dtype=float))),
        "median_source_order_block_age_at_break_bars": _median_or_none(source_age),
        "median_retest_distance_atr": _median_or_none(retests.get("dist_to_breaker_atr", pd.Series(dtype=float))),
    }


def _build_readiness_summary(
    headline: dict[str, Any],
    breadth: dict[str, Any],
    integrity: dict[str, Any],
    *,
    min_total_retests: int,
    min_side_retests: int,
    min_retest_years: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if int(headline["total_retest_event_bars"]) < int(min_total_retests):
        reasons.append("thin_total_retest_sample")
    if int(headline["bull_retest_event_bars"]) < int(min_side_retests):
        reasons.append("thin_bull_retest_sample")
    if int(headline["bear_retest_event_bars"]) < int(min_side_retests):
        reasons.append("thin_bear_retest_sample")
    if int(breadth["years_with_retests"]) < int(min_retest_years):
        reasons.append("insufficient_year_coverage")
    if int(integrity["same_bar_create_and_retest_events"]) > 0:
        reasons.append("same_bar_create_retest_leakage")
    if float(integrity["retest_after_formation_pct"]) < 100.0 and int(headline["total_retest_event_bars"]) > 0:
        reasons.append("noncausal_retest_ordering")
    if float(integrity["retest_has_source_order_block_pct"]) < 100.0 and int(headline["total_retest_event_bars"]) > 0:
        reasons.append("missing_source_order_block_on_retest")
    if float(integrity["retest_has_finite_bounds_pct"]) < 100.0 and int(headline["total_retest_event_bars"]) > 0:
        reasons.append("missing_breaker_bounds_on_retest")

    if any(reason.startswith("thin") or reason == "insufficient_year_coverage" for reason in reasons):
        decision = "thin_sample"
    elif reasons:
        decision = "integrity_blocked"
    else:
        decision = "ready_for_label_design"
    return {
        "ready_for_classic_breaker_label_design": decision == "ready_for_label_design",
        "decision": decision,
        "reasons": reasons,
    }


def _build_review_rows(created: pd.DataFrame, retests: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    buckets: list[pd.DataFrame] = []
    for side in ("long", "short"):
        side_retests = retests.loc[retests["side"].eq(side)].sort_values("event_time")
        if not side_retests.empty:
            bucket = side_retests.head(top_n).copy()
            bucket["review_bucket"] = f"{side}_retest_sample"
            buckets.append(bucket)
    if not created.empty:
        bucket = created.sort_values("event_time").head(top_n).copy()
        bucket["review_bucket"] = "created_sample"
        buckets.append(bucket)
    if not retests.empty:
        delayed = retests.sort_values("retest_lag_bars", ascending=False).head(top_n).copy()
        delayed["review_bucket"] = "longest_retest_lag"
        buckets.append(delayed)
    if not buckets:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    review = pd.concat(buckets, ignore_index=True)
    return review.loc[:, [column for column in REVIEW_COLUMNS if column in review.columns]].copy()


def _event_time(frame: pd.DataFrame) -> pd.Series | None:
    if "datetime" not in frame.columns:
        return None
    return pd.to_datetime(frame["datetime"], errors="coerce", utc=True)


def _event_mask(value: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce").fillna(0)
    text = value.fillna("").astype(str).str.strip().str.lower()
    return numeric.ne(0) | text.isin({"true", "t", "yes"})


def _numeric(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    return pd.Series(value)


def _format_times(times: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(times, errors="coerce", utc=True)
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _first_time(events: pd.DataFrame) -> str | None:
    if events.empty or "event_time" not in events.columns:
        return None
    value = pd.to_datetime(events["event_time"], errors="coerce", utc=True).dropna()
    return None if value.empty else _timestamp_or_none(value.min())


def _last_time(events: pd.DataFrame) -> str | None:
    if events.empty or "event_time" not in events.columns:
        return None
    value = pd.to_datetime(events["event_time"], errors="coerce", utc=True).dropna()
    return None if value.empty else _timestamp_or_none(value.max())


def _timestamp_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _period_counts(events: pd.DataFrame, freq: str) -> list[dict[str, Any]]:
    if events.empty or "event_time" not in events.columns:
        return []
    times = pd.to_datetime(events["event_time"], errors="coerce", utc=True).dropna()
    if times.empty:
        return []
    labels = times.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period(freq).astype(str)
    counts = labels.value_counts().sort_index()
    return [{"period": str(period), "count": int(count)} for period, count in counts.items()]


def _distinct_periods(events: pd.DataFrame, freq: str) -> int:
    return len(_period_counts(events, freq))


def _side_counts(events: pd.DataFrame) -> list[dict[str, Any]]:
    if events.empty or "side" not in events.columns:
        return []
    counts = events["side"].fillna("missing").astype(str).value_counts().sort_index()
    total = int(counts.sum())
    return [{"side": str(side), "count": int(count), "share_pct": _pct(count, total)} for side, count in counts.items()]


def _format_year_counts(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    return ", ".join(f"{row['period']}={row['count']}" for row in rows)


def _max_int(values: pd.Series) -> int:
    cleaned = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return 0 if cleaned.empty else int(cleaned.max())


def _median_or_none(values: Any) -> float | None:
    cleaned = _numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
    return None if cleaned.empty else round(float(cleaned.median()), 6)


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _pct(numerator: Any, denominator: Any) -> float:
    try:
        denom = float(denominator)
    except (TypeError, ValueError):
        return 0.0
    if denom <= 0:
        return 0.0
    return round(float(numerator) / denom * 100.0, 4)


def _empty_headline() -> dict[str, Any]:
    return {
        "total_created_event_bars": 0,
        "total_retest_event_bars": 0,
        "bull_created_event_bars": 0,
        "bear_created_event_bars": 0,
        "bull_retest_event_bars": 0,
        "bear_retest_event_bars": 0,
        "unique_breaker_ids_on_retest_rows": 0,
        "unique_source_order_block_ids_on_retest_rows": 0,
        "max_active_bull_breakers": 0,
        "max_active_bear_breakers": 0,
        "first_retest_time_utc": None,
        "last_retest_time_utc": None,
    }


def _empty_breadth() -> dict[str, Any]:
    return {
        "years_with_created_events": 0,
        "years_with_retests": 0,
        "months_with_retests": 0,
        "created_event_bars_by_year": [],
        "retest_event_bars_by_year": [],
        "retest_event_bars_by_side": [],
    }


def _empty_integrity() -> dict[str, Any]:
    return {
        "same_bar_create_and_retest_events": 0,
        "retest_after_formation_pct": 0.0,
        "retest_has_source_order_block_pct": 0.0,
        "retest_has_finite_bounds_pct": 0.0,
        "source_order_block_formed_before_breaker_pct": 0.0,
        "median_retest_lag_bars": None,
        "median_source_order_block_age_at_break_bars": None,
        "median_retest_distance_atr": None,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return _timestamp_or_none(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_list())
    return str(value)
