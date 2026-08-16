from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from features.io import standardize_market_frame
from ict.labeling.ict_labeling_engine import ICTLabelingConfig, build_ict_labels, ict_events_to_frame
from ict.config.setups import ICTSetupDetectorConfig
from ict.setups.detector import detect_ict_setups


DEFAULT_REVIEW_COLUMNS = (
    "review_bucket",
    "event_time",
    "event_direction",
    "label_family",
    "setup_type",
    "setup_side",
    "htf_context",
    "entry_price",
    "stop_reference",
    "stop_price",
    "target_reference",
    "target_price",
    "rr_ratio",
    "horizon_scale",
    "tb_outcome",
    "exit_reason",
    "excluded",
    "exclude_reasons",
    "stop_adjustment_ticks",
    "target_adjustment_ticks",
    "total_adjustment_ticks",
)
POSITIVE_OUTCOMES = frozenset({"tp", "timeout_profit"})


def refresh_ict_phase03_labeling_artifacts(
    *,
    phase03_dir: str | Path,
    market_5m_path: str | Path,
    market_1m_path: str | Path | None = None,
    setup_feature_path: str | Path | None = None,
    config: ICTLabelingConfig | None = None,
) -> dict[str, Any]:
    """Regenerate ICT Phase 3 labels/events/diagnostics from the current labeler."""

    phase03_path = Path(phase03_dir)
    phase03_path.mkdir(parents=True, exist_ok=True)
    labeling_config = config or ICTLabelingConfig()

    market_5m = _load_label_market_frame(market_5m_path)
    market_1m = None
    if market_1m_path is not None:
        market_1m = _load_execution_market_frame(market_1m_path)

    setup_output = None
    setup_source_rows = 0
    if setup_feature_path is not None:
        setup_surface = _load_setup_surface_frame(setup_feature_path)
        setup_source_rows = int(len(setup_surface))
        setup_output = detect_ict_setups(
            setup_surface,
            config=ICTSetupDetectorConfig(instrument=labeling_config.instrument),
        )

    labels, diagnostics, events = build_ict_labels(
        market_5m,
        params=labeling_config,
        df_1m=market_1m,
        setup_output=setup_output,
        verbose=True,
    )
    labels_reset = labels.reset_index()
    if "datetime" not in labels_reset.columns and len(labels_reset.columns) > 0:
        labels_reset = labels_reset.rename(columns={labels_reset.columns[0]: "datetime"})
    labels_reset["datetime"] = pd.to_datetime(labels_reset["datetime"], errors="coerce", utc=True)

    events_frame = ict_events_to_frame(events)

    labels_path = phase03_path / "ict_es_labels.csv"
    events_path = phase03_path / "ict_es_events.csv"
    diagnostics_path = phase03_path / "ict_es_labeling_diagnostics.json"

    labels_reset.to_csv(labels_path, index=False)
    events_frame.to_csv(events_path, index=False)
    diagnostics_path.write_text(json.dumps(_json_safe(diagnostics), indent=2), encoding="utf-8")

    return {
        "phase03_dir": str(phase03_path),
        "labels_csv": str(labels_path),
        "events_csv": str(events_path),
        "diagnostics_json": str(diagnostics_path),
        "market_5m_path": str(Path(market_5m_path)),
        "market_1m_path": str(Path(market_1m_path)) if market_1m_path is not None else None,
        "setup_feature_path": str(Path(setup_feature_path)) if setup_feature_path is not None else None,
        "diagnostics": _json_safe(diagnostics),
        "rows": {
            "market_5m_rows": int(len(market_5m)),
            "market_1m_rows": int(len(market_1m)) if market_1m is not None else 0,
            "setup_source_rows": int(setup_source_rows),
            "labels_rows": int(len(labels_reset)),
            "events_rows": int(len(events_frame)),
        },
    }


def load_ict_phase03_outputs(phase03_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    phase03_path = Path(phase03_dir)
    events_path = phase03_path / "ict_es_events.csv"
    diagnostics_path = phase03_path / "ict_es_labeling_diagnostics.json"
    events = pd.read_csv(events_path)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    paths = {
        "phase03_dir": str(phase03_path),
        "events_csv": str(events_path),
        "diagnostics_json": str(diagnostics_path),
    }
    return events, diagnostics, paths


def build_ict_event_sample_audit(
    events: pd.DataFrame,
    *,
    diagnostics: dict[str, Any] | None = None,
    tick_size: float = 0.25,
    top_n_contexts: int = 5,
    top_n_review_rows: int = 12,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize ICT event geometry and context quality for manual review."""

    working = _prepare_event_metrics_frame(events, tick_size=tick_size)
    diagnostics = diagnostics or {}
    usable = working.loc[working["usable_event"]].copy()
    continuation = usable.loc[usable["label_family"].eq("ict_continuation")].copy()

    review_rows = _build_review_rows(working, top_n_per_bucket=top_n_review_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "headline": _build_headline_summary(working, diagnostics),
        "htf_context": {
            "overall_top_contexts": _top_contexts(working, top_n=top_n_contexts),
            "by_branch": _build_context_group_summary(
                working,
                group_columns=("event_direction", "label_family"),
                top_n=top_n_contexts,
            ),
            "by_setup_branch": _build_context_group_summary(
                working,
                group_columns=("event_direction", "setup_type"),
                top_n=top_n_contexts,
            ),
        },
        "reference_geometry": {
            "overall": _build_reference_geometry_summary(working),
            "by_branch": _build_group_geometry_summary(
                working,
                group_columns=("event_direction", "label_family"),
            ),
            "by_setup_branch": _build_group_geometry_summary(
                working,
                group_columns=("event_direction", "setup_type"),
            ),
        },
        "barrier_geometry": {
            "barrier_family_counts": _value_count_records(usable["barrier_family"]),
            "exit_reason_counts": _value_count_records(usable["exit_reason"]),
            "by_branch": _build_group_barrier_summary(
                working,
                group_columns=("event_direction", "label_family"),
            ),
            "by_setup_branch": _build_group_barrier_summary(
                working,
                group_columns=("event_direction", "setup_type"),
            ),
        },
        "continuation_management": {
            "usable_events": int(len(continuation)),
            "target_activation_events": int(continuation["target_hit_index"].notna().sum()),
            "target_activation_share_pct": _pct(continuation["target_hit_index"].notna().sum(), len(continuation)),
            "exit_reason_counts": _value_count_records(continuation["exit_reason"]),
            "median_extension_bars_after_target": _median_or_none(
                continuation["tb_bars_held"] - (continuation["target_hit_index"] - continuation["entry_index"] + 1.0)
            ),
        },
        "exclusions": {
            "exclude_reason_counts": _count_pipe_delimited_values(working["exclude_reasons"]),
            "excluded_rows": int(working["excluded"].sum()),
            "ambiguous_rows": int(working["tb_outcome"].eq("ambiguous").sum()),
            "ambiguous_without_1m": int(diagnostics.get("events_excluded_ambiguous_5m_without_1m", 0) or 0),
            "unresolved_intrabar_1m": int(diagnostics.get("events_excluded_unresolved_intrabar_1m", 0) or 0),
            "macro_event_window": int(diagnostics.get("events_excluded_macro_event_window", 0) or 0),
            "half_day_or_holiday_window": int(diagnostics.get("events_excluded_half_day_or_holiday_window", 0) or 0),
            "lunch_dominated_window": int(diagnostics.get("events_excluded_lunch_dominated_window", 0) or 0),
            "thin_session_window": int(diagnostics.get("events_excluded_thin_session_window", 0) or 0),
        },
        "review_buckets": {
            bucket: int((review_rows["review_bucket"] == bucket).sum())
            for bucket in sorted(review_rows["review_bucket"].unique())
        },
        "review_rows_preview": _json_safe(review_rows.head(16).to_dict(orient="records")),
    }
    return _json_safe(summary), review_rows


def render_ict_event_sample_audit_markdown(summary: dict[str, Any]) -> str:
    headline = summary["headline"]
    overall_reference = summary["reference_geometry"]["overall"]
    continuation = summary["continuation_management"]
    exclusions = summary["exclusions"]
    branch_rows = summary["barrier_geometry"]["by_branch"]
    context_rows = summary["htf_context"]["by_branch"]

    lines = [
        "# ICT Event Sample Audit",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        "## Headline",
        "",
        f"- Total events: `{headline['total_events']}`",
        f"- Usable events: `{headline['usable_events']}`",
        f"- Excluded events: `{headline['excluded_events']}`",
        f"- Positive events: `{headline['positive_events']}`",
        f"- Base rate: `{headline['base_rate_pct']:.2f}%`",
        f"- Mean label quality: `{headline['label_quality_mean']:.4f}`",
        "",
        "## Reference Geometry",
        "",
        f"- Raw stop on expected side: `{overall_reference['raw_stop_expected_pct']:.2f}%`",
        f"- Raw target on expected side: `{overall_reference['raw_target_expected_pct']:.2f}%`",
        f"- Raw reference pair valid: `{overall_reference['raw_reference_valid_pct']:.2f}%`",
        f"- Final stop/target geometry valid: `{overall_reference['final_geometry_valid_pct']:.2f}%`",
        f"- Stop adjusted from raw reference: `{overall_reference['stop_adjusted_pct']:.2f}%`",
        f"- Target adjusted from raw reference: `{overall_reference['target_adjusted_pct']:.2f}%`",
        "",
        "## Branch Geometry",
        "",
    ]
    for row in branch_rows:
        branch = f"{row['event_direction']} {row['group_value']}"
        lines.append(
            f"- `{branch}`: usable=`{row['usable_events']}`, base_rate=`{row['base_rate_pct']:.2f}%`, "
            f"median_rr=`{_fmt_optional(row['median_rr_ratio'])}`, median_stop_ticks=`{_fmt_optional(row['median_stop_distance_ticks'])}`, "
            f"median_target_ticks=`{_fmt_optional(row['median_target_distance_ticks'])}`, median_horizon_scale=`{_fmt_optional(row['median_horizon_scale'])}`"
        )
    lines.extend(["", "## HTF Context", ""])
    for row in context_rows:
        branch = f"{row['event_direction']} {row['group_value']}"
        top_context = row["top_contexts"][0]["htf_context"] if row["top_contexts"] else "missing"
        top_share = row["top_contexts"][0]["share_pct"] if row["top_contexts"] else 0.0
        lines.append(
            f"- `{branch}`: missing=`{row['missing_context_pct']:.2f}%`, dominant_context=`{top_context}` (`{top_share:.2f}%`)"
        )
    lines.extend(
        [
            "",
            "## Continuation Management",
            "",
            f"- Usable continuation events: `{continuation['usable_events']}`",
            f"- Target activations: `{continuation['target_activation_events']}` (`{continuation['target_activation_share_pct']:.2f}%`)",
            f"- Median extension bars after target: `{_fmt_optional(continuation['median_extension_bars_after_target'])}`",
            "",
            "## Exclusions",
            "",
            f"- Macro windows: `{exclusions['macro_event_window']}`",
            f"- Half-day / holiday windows: `{exclusions['half_day_or_holiday_window']}`",
            f"- Lunch-dominated windows: `{exclusions['lunch_dominated_window']}`",
            f"- Thin-session windows: `{exclusions['thin_session_window']}`",
            f"- Ambiguous without 1m: `{exclusions['ambiguous_without_1m']}`",
            f"- Unresolved even with 1m: `{exclusions['unresolved_intrabar_1m']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def write_ict_event_sample_audit(
    *,
    output_dir: str | Path,
    summary: dict[str, Any],
    review_rows: pd.DataFrame,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_json = output_path / "ict_event_sample_audit_summary.json"
    summary_md = output_path / "ict_event_sample_audit_summary.md"
    review_csv = output_path / "ict_event_sample_audit_review_rows.csv"

    summary_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    summary_md.write_text(render_ict_event_sample_audit_markdown(summary), encoding="utf-8")
    review_rows.to_csv(review_csv, index=False)

    return {
        "summary_json": str(summary_json),
        "summary_markdown": str(summary_md),
        "review_rows_csv": str(review_csv),
    }


def _load_label_market_frame(path: str | Path) -> pd.DataFrame:
    return standardize_market_frame(
        pd.read_csv(
            path,
            usecols=lambda column: str(column).strip().lower()
            in {"ts_event", "datetime", "timestamp", "open", "high", "low", "close", "volume", "contract_id", "warmup_mask"},
        )
    )


def _load_execution_market_frame(path: str | Path) -> pd.DataFrame:
    return standardize_market_frame(
        pd.read_csv(
            path,
            usecols=lambda column: str(column).strip().lower()
            in {"ts_event", "datetime", "timestamp", "open", "high", "low", "close", "volume"},
        )
    )


def _load_setup_surface_frame(path: str | Path) -> pd.DataFrame:
    return standardize_market_frame(
        pd.read_csv(
            path,
            usecols=lambda column: _use_setup_surface_column(str(column)),
        )
    )


def _use_setup_surface_column(column: str) -> bool:
    normalized = column.strip().lower()
    if normalized.startswith("ict_"):
        return True
    return normalized in {
        "ts_event",
        "datetime",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr_14",
        "session_date",
        "dist_to_bull_fvg_atr",
        "dist_to_bull_order_block_atr",
        "dist_to_bear_fvg_atr",
        "dist_to_bear_order_block_atr",
        "htf_30m_ema_alignment",
        "htf_1h_ema_alignment",
    }


def _prepare_event_metrics_frame(events: pd.DataFrame, *, tick_size: float) -> pd.DataFrame:
    working = events.copy()
    for column in (
        "label_family",
        "setup_type",
        "event_direction",
        "htf_context",
        "tb_outcome",
        "exit_reason",
        "barrier_family",
        "exclude_reasons",
    ):
        if column not in working.columns:
            working[column] = ""
    for column in (
        "event_time",
        "entry_time",
        "target_hit_time",
        "barrier_end_time",
    ):
        if column in working.columns:
            working[column] = pd.to_datetime(working[column], errors="coerce", utc=True)

    for column in (
        "setup_side",
        "entry_price",
        "stop_reference",
        "stop_price",
        "target_reference",
        "target_price",
        "rr_ratio",
        "horizon_scale",
        "tb_bars_held",
        "label_quality",
        "target_hit_index",
        "entry_index",
        "signal_index",
    ):
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working["excluded"] = _as_bool_series(working.get("excluded", False))
    working["usable_event"] = ~working["excluded"]
    working["positive_event"] = working.get("tb_outcome", pd.Series("", index=working.index)).astype(str).isin(POSITIVE_OUTCOMES)
    working["htf_context"] = working.get("htf_context", pd.Series("", index=working.index)).fillna("").astype(str)
    working["missing_htf_context"] = working["htf_context"].str.strip().eq("")

    entry = working["entry_price"]
    stop_ref = working["stop_reference"]
    target_ref = working["target_reference"]
    stop_price = working["stop_price"]
    target_price = working["target_price"]
    side = working["setup_side"]

    raw_stop_expected = np.where(side.gt(0), stop_ref.lt(entry), stop_ref.gt(entry))
    raw_target_expected = np.where(side.gt(0), target_ref.gt(entry), target_ref.lt(entry))
    final_geometry_valid = np.where(
        side.gt(0),
        stop_price.lt(entry) & entry.lt(target_price),
        target_price.lt(entry) & entry.lt(stop_price),
    )
    raw_stop_expected = pd.Series(raw_stop_expected, index=working.index) & _finite_pair(entry, stop_ref)
    raw_target_expected = pd.Series(raw_target_expected, index=working.index) & _finite_pair(entry, target_ref)
    final_geometry_valid = pd.Series(final_geometry_valid, index=working.index) & _finite_triplet(entry, stop_price, target_price)

    working["raw_stop_expected_side"] = raw_stop_expected
    working["raw_target_expected_side"] = raw_target_expected
    working["raw_reference_valid"] = raw_stop_expected & raw_target_expected
    working["final_geometry_valid"] = final_geometry_valid
    working["stop_adjustment_ticks"] = (stop_price - stop_ref).abs() / float(tick_size)
    working["target_adjustment_ticks"] = (target_price - target_ref).abs() / float(tick_size)
    working["total_adjustment_ticks"] = working["stop_adjustment_ticks"].fillna(0.0) + working["target_adjustment_ticks"].fillna(0.0)
    working["stop_distance_ticks"] = (entry - stop_price).abs() / float(tick_size)
    working["target_distance_ticks"] = (target_price - entry).abs() / float(tick_size)

    tolerance = 1e-9
    working["stop_adjusted"] = working["stop_adjustment_ticks"].gt(tolerance)
    working["target_adjusted"] = working["target_adjustment_ticks"].gt(tolerance)
    return working


def _build_headline_summary(working: pd.DataFrame, diagnostics: dict[str, Any]) -> dict[str, Any]:
    usable = working.loc[working["usable_event"]]
    positives = usable.loc[usable["positive_event"]]
    return {
        "total_events": int(len(working)),
        "usable_events": int(len(usable)),
        "excluded_events": int(working["excluded"].sum()),
        "positive_events": int(len(positives)),
        "base_rate_pct": _pct(len(positives), len(usable)),
        "label_quality_mean": float(usable["label_quality"].mean()) if not usable.empty else 0.0,
        "diagnostics_total_events_sampled": int(diagnostics.get("total_events_sampled", len(working)) or len(working)),
        "diagnostics_usable_events": int(diagnostics.get("usable_events", len(usable)) or len(usable)),
    }


def _build_reference_geometry_summary(working: pd.DataFrame) -> dict[str, Any]:
    evaluable_raw_stop = int(_finite_pair(working["entry_price"], working["stop_reference"]).sum())
    evaluable_raw_target = int(_finite_pair(working["entry_price"], working["target_reference"]).sum())
    evaluable_raw_pair = int(
        (_finite_pair(working["entry_price"], working["stop_reference"]) & _finite_pair(working["entry_price"], working["target_reference"])).sum()
    )
    evaluable_final = int(_finite_triplet(working["entry_price"], working["stop_price"], working["target_price"]).sum())
    return {
        "evaluable_raw_stop_rows": evaluable_raw_stop,
        "evaluable_raw_target_rows": evaluable_raw_target,
        "evaluable_raw_pair_rows": evaluable_raw_pair,
        "evaluable_final_rows": evaluable_final,
        "raw_stop_expected_pct": _pct(working["raw_stop_expected_side"].sum(), evaluable_raw_stop),
        "raw_target_expected_pct": _pct(working["raw_target_expected_side"].sum(), evaluable_raw_target),
        "raw_reference_valid_pct": _pct(working["raw_reference_valid"].sum(), evaluable_raw_pair),
        "final_geometry_valid_pct": _pct(working["final_geometry_valid"].sum(), evaluable_final),
        "stop_adjusted_pct": _pct(working["stop_adjusted"].sum(), evaluable_final),
        "target_adjusted_pct": _pct(working["target_adjusted"].sum(), evaluable_final),
        "median_stop_adjustment_ticks": _median_or_none(working["stop_adjustment_ticks"]),
        "median_target_adjustment_ticks": _median_or_none(working["target_adjustment_ticks"]),
        "p90_stop_adjustment_ticks": _quantile_or_none(working["stop_adjustment_ticks"], 0.90),
        "p90_target_adjustment_ticks": _quantile_or_none(working["target_adjustment_ticks"], 0.90),
    }


def _build_group_geometry_summary(
    working: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in _iter_groups(working, group_columns):
        usable = group.loc[group["usable_event"]]
        evaluable_pair = int(
            (_finite_pair(group["entry_price"], group["stop_reference"]) & _finite_pair(group["entry_price"], group["target_reference"])).sum()
        )
        evaluable_final = int(_finite_triplet(group["entry_price"], group["stop_price"], group["target_price"]).sum())
        row = _keyed_row(keys, group_columns)
        row.update(
            {
                "total_events": int(len(group)),
                "usable_events": int(len(usable)),
                "excluded_events": int(group["excluded"].sum()),
                "positive_events": int(usable["positive_event"].sum()),
                "base_rate_pct": _pct(usable["positive_event"].sum(), len(usable)),
                "raw_reference_valid_pct": _pct(group["raw_reference_valid"].sum(), evaluable_pair),
                "final_geometry_valid_pct": _pct(group["final_geometry_valid"].sum(), evaluable_final),
                "stop_adjusted_pct": _pct(group["stop_adjusted"].sum(), evaluable_final),
                "target_adjusted_pct": _pct(group["target_adjusted"].sum(), evaluable_final),
                "median_stop_adjustment_ticks": _median_or_none(group["stop_adjustment_ticks"]),
                "median_target_adjustment_ticks": _median_or_none(group["target_adjustment_ticks"]),
            }
        )
        rows.append(row)
    return rows


def _build_group_barrier_summary(
    working: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in _iter_groups(working, group_columns):
        usable = group.loc[group["usable_event"]]
        row = _keyed_row(keys, group_columns)
        row.update(
            {
                "total_events": int(len(group)),
                "usable_events": int(len(usable)),
                "excluded_events": int(group["excluded"].sum()),
                "base_rate_pct": _pct(usable["positive_event"].sum(), len(usable)),
                "median_rr_ratio": _median_or_none(usable["rr_ratio"]),
                "median_stop_distance_ticks": _median_or_none(usable["stop_distance_ticks"]),
                "median_target_distance_ticks": _median_or_none(usable["target_distance_ticks"]),
                "median_horizon_scale": _median_or_none(usable["horizon_scale"]),
                "median_bars_held": _median_or_none(usable["tb_bars_held"]),
                "median_label_quality": _median_or_none(usable["label_quality"]),
                "barrier_family_counts": _value_count_records(usable["barrier_family"]),
                "exit_reason_counts": _value_count_records(usable["exit_reason"]),
            }
        )
        rows.append(row)
    return rows


def _build_context_group_summary(
    working: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in _iter_groups(working, group_columns):
        row = _keyed_row(keys, group_columns)
        row.update(
            {
                "total_events": int(len(group)),
                "missing_context_pct": _pct(group["missing_htf_context"].sum(), len(group)),
                "top_contexts": _top_contexts(group, top_n=top_n),
            }
        )
        rows.append(row)
    return rows


def _build_review_rows(working: pd.DataFrame, *, top_n_per_bucket: int) -> pd.DataFrame:
    buckets: list[pd.DataFrame] = []

    raw_failures = working.loc[~working["raw_reference_valid"]].sort_values(
        ["total_adjustment_ticks", "event_time"],
        ascending=[False, True],
    )
    if not raw_failures.empty:
        bucket = raw_failures.head(top_n_per_bucket).copy()
        bucket["review_bucket"] = "raw_reference_side_failure"
        buckets.append(bucket)

    missing_context = working.loc[working["missing_htf_context"]].sort_values("event_time")
    if not missing_context.empty:
        bucket = missing_context.head(top_n_per_bucket).copy()
        bucket["review_bucket"] = "missing_htf_context"
        buckets.append(bucket)

    large_adjustments = working.loc[working["usable_event"]].sort_values("total_adjustment_ticks", ascending=False)
    if not large_adjustments.empty:
        bucket = large_adjustments.head(top_n_per_bucket).copy()
        bucket["review_bucket"] = "large_barrier_adjustment"
        buckets.append(bucket)

    excluded_windows = working.loc[working["excluded"]].sort_values("event_time")
    if not excluded_windows.empty:
        bucket = excluded_windows.head(top_n_per_bucket).copy()
        bucket["review_bucket"] = "excluded_window_sample"
        buckets.append(bucket)

    if not buckets:
        return pd.DataFrame(columns=DEFAULT_REVIEW_COLUMNS)

    review_rows = pd.concat(buckets, ignore_index=True)
    review_rows = review_rows.loc[:, [column for column in DEFAULT_REVIEW_COLUMNS if column in review_rows.columns]].copy()
    if "event_time" in review_rows.columns:
        review_rows["event_time"] = pd.to_datetime(review_rows["event_time"], errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return review_rows


def _top_contexts(frame: pd.DataFrame, *, top_n: int) -> list[dict[str, Any]]:
    normalized = frame["htf_context"].fillna("").astype(str).str.strip()
    normalized = normalized.mask(normalized.eq(""), "missing")
    counts = normalized.value_counts(dropna=False).head(top_n)
    total = int(len(frame))
    return [
        {
            "htf_context": str(context),
            "count": int(count),
            "share_pct": _pct(count, total),
        }
        for context, count in counts.items()
    ]


def _count_pipe_delimited_values(values: pd.Series) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values.fillna("").astype(str):
        for item in [piece.strip() for piece in value.split("|") if piece.strip()]:
            counts[item] = counts.get(item, 0) + 1
    return [
        {"value": key, "count": int(value)}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _value_count_records(values: pd.Series) -> list[dict[str, Any]]:
    counts = values.fillna("missing").astype(str).value_counts(dropna=False)
    total = int(counts.sum())
    return [
        {
            "value": str(value),
            "count": int(count),
            "share_pct": _pct(count, total),
        }
        for value, count in counts.items()
    ]


def _iter_groups(frame: pd.DataFrame, group_columns: tuple[str, ...]) -> Iterable[tuple[tuple[Any, ...], pd.DataFrame]]:
    grouped = frame.groupby(list(group_columns), dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        yield keys, group.copy()


def _keyed_row(keys: tuple[Any, ...], columns: tuple[str, ...]) -> dict[str, Any]:
    row = {str(column): _json_safe(key) for column, key in zip(columns, keys, strict=True)}
    if len(columns) >= 2:
        row["group_value"] = _json_safe(keys[-1])
    return row


def _as_bool_series(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value.fillna(False)
        normalized = value.fillna(False).astype(str).str.strip().str.lower()
        return normalized.isin({"1", "true", "t", "yes"})
    return pd.Series(bool(value))


def _finite_pair(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)


def _finite_triplet(first: pd.Series, second: pd.Series, third: pd.Series) -> pd.Series:
    return _finite_pair(first, second) & third.notna() & np.isfinite(third)


def _median_or_none(values: pd.Series) -> float | None:
    cleaned = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return None
    return float(cleaned.median())


def _quantile_or_none(values: pd.Series, q: float) -> float | None:
    cleaned = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return None
    return float(cleaned.quantile(q))


def _pct(numerator: float | int, denominator: float | int) -> float:
    if float(denominator) <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def _fmt_optional(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if not pd.isna(value) else None
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
