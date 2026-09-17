from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from model_testing.ote_regime_labeler import label_regimes


CLASSIC_BREAKER_SETUP_TYPE = "classic_breaker"
CLASSIC_BREAKER_LABEL_FAMILY = "ict_classic_breaker"
POSITIVE_OUTCOMES = frozenset({"tp", "timeout_profit"})
CANONICAL_LABEL_COLUMNS = (
    "label_long_ict_reversal",
    "label_short_ict_reversal",
    "label_long_ict_continuation",
    "label_short_ict_continuation",
    "label_long_ict_meta",
    "label_short_ict_meta",
)
CLASSIC_LABEL_COLUMNS = (
    "label_long_ict_classic_breaker",
    "label_short_ict_classic_breaker",
)
CANONICAL_FAMILIES = frozenset({"ict_reversal", "ict_continuation"})
SUSPICIOUS_FEATURE_TOKENS = (
    "label_",
    "sample_weight",
    "label_quality",
    "exclude_",
    "neg_ok_",
    "concurrency_",
    "tb_",
    "target_hit",
    "barrier_end",
    "exit_",
    "future",
    "mfe",
    "mae",
)


def build_ict_classic_breaker_phase3_audit(
    events: pd.DataFrame,
    *,
    labels: pd.DataFrame | None = None,
    canonical_events: pd.DataFrame | None = None,
    canonical_labels: pd.DataFrame | None = None,
    regime_source: pd.DataFrame | None = None,
    prepared_root: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    working = _prepare_events(events)
    classic = working.loc[
        working["setup_type"].eq(CLASSIC_BREAKER_SETUP_TYPE)
        | working["label_family"].eq(CLASSIC_BREAKER_LABEL_FAMILY)
    ].copy()
    canonical = working.loc[~working.index.isin(classic.index)].copy()
    if regime_source is not None and not classic.empty:
        classic = _attach_regime_context(classic, regime_source)

    usable = classic.loc[~classic["excluded"]].copy()
    canonical_events_prepared = _prepare_events(canonical_events) if canonical_events is not None else canonical
    isolation = _build_isolation_summary(
        classic=classic,
        usable=usable,
        labels=labels,
        canonical_events=canonical_events_prepared,
        canonical_labels=canonical_labels,
    )
    integrity = _build_integrity_summary(classic)
    leakage = _build_leakage_summary(events=events, labels=labels, prepared_root=prepared_root)
    duplicates = _build_duplicate_overlap_summary(classic, canonical_events_prepared)
    sample = _build_sample_summary(classic, usable)
    verdict = _build_verdict(
        headline=sample["headline"],
        integrity=integrity,
        isolation=isolation,
        leakage=leakage,
        duplicates=duplicates,
    )
    review_rows = _build_review_rows(classic)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": "ict_classic_breaker_phase3",
        "setup_type": CLASSIC_BREAKER_SETUP_TYPE,
        "label_family": CLASSIC_BREAKER_LABEL_FAMILY,
        "verdict": verdict,
        "sample": sample,
        "integrity": integrity,
        "canonical_isolation": isolation,
        "duplicates_and_overlap": duplicates,
        "leakage": leakage,
        "review_rows_preview": _json_safe(review_rows.head(20).to_dict(orient="records")),
    }
    return _json_safe(summary), review_rows


def write_ict_classic_breaker_phase3_audit(
    *,
    output_dir: str | Path,
    summary: Mapping[str, Any],
    review_rows: pd.DataFrame,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_json = output_path / "ict_classic_breaker_phase3_audit_summary.json"
    summary_md = output_path / "ict_classic_breaker_phase3_audit_summary.md"
    review_csv = output_path / "ict_classic_breaker_phase3_review_rows.csv"

    summary_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    summary_md.write_text(render_ict_classic_breaker_phase3_audit_markdown(summary), encoding="utf-8")
    review_rows.to_csv(review_csv, index=False)
    return {
        "summary_json": str(summary_json),
        "summary_markdown": str(summary_md),
        "review_rows_csv": str(review_csv),
    }


def render_ict_classic_breaker_phase3_audit_markdown(summary: Mapping[str, Any]) -> str:
    headline = summary["sample"]["headline"]
    integrity = summary["integrity"]
    isolation = summary["canonical_isolation"]
    leakage = summary["leakage"]
    verdict = summary["verdict"]

    lines = [
        "# ICT Classic Breaker Phase 3 Audit",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        "## Verdict",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Failures: `{', '.join(verdict['failures']) if verdict['failures'] else 'none'}`",
        f"- Caveats: `{', '.join(verdict['caveats']) if verdict['caveats'] else 'none'}`",
        "",
        "## Sample",
        "",
        f"- Total classic breaker events: `{headline['total_events']}`",
        f"- Usable events: `{headline['usable_events']}`",
        f"- Positive events: `{headline['positive_events']}`",
        f"- Base rate: `{headline['base_rate_pct']:.2f}%`",
        f"- Long usable / short usable: `{headline['usable_long_events']}` / `{headline['usable_short_events']}`",
        f"- First event / last event: `{headline['first_event_time']}` / `{headline['last_event_time']}`",
        "",
        "## Integrity",
        "",
        f"- Source OB before activation: `{integrity['source_order_block_before_activation_pct']:.2f}%`",
        f"- Activation before retest: `{integrity['activation_before_retest_pct']:.2f}%`",
        f"- Signal equals retest index: `{integrity['signal_equals_retest_index_pct']:.2f}%`",
        f"- Bounds finite and ordered: `{integrity['finite_ordered_zone_bounds_pct']:.2f}%`",
        f"- First retest only: `{integrity['first_retest_only_pct']:.2f}%`",
        "",
        "## Isolation",
        "",
        f"- Classic label columns present: `{isolation['classic_label_columns_present']}`",
        f"- Canonical meta leakage on classic-only bars: `{isolation['classic_only_canonical_meta_leak_count']}`",
        f"- Canonical reversal/continuation leakage on classic-only bars: `{isolation['classic_only_canonical_family_leak_count']}`",
        f"- Baseline canonical columns equal: `{isolation['canonical_columns_equal_baseline']}`",
        "",
        "## Leakage",
        "",
        f"- Prepared feature audit available: `{leakage['prepared_feature_audit']['available']}`",
        f"- Suspicious prepared feature references: `{leakage['prepared_feature_audit']['suspicious_reference_count']}`",
        f"- Event outcome columns remain event-only: `{leakage['event_outcome_columns_present']}`",
        "",
    ]
    return "\n".join(lines)


def _prepare_events(events: pd.DataFrame | None) -> pd.DataFrame:
    if events is None:
        return pd.DataFrame()
    working = events.copy()
    for column in (
        "setup_type",
        "label_family",
        "event_direction",
        "tb_outcome",
        "exit_reason",
        "exclude_reasons",
        "htf_context",
    ):
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].fillna("").astype(str).str.strip().str.lower()

    for column in (
        "signal_index",
        "entry_index",
        "barrier_end_index",
        "max_holding_bars",
        "setup_side",
        "breaker_id",
        "breaker_source_order_block_id",
        "breaker_activation_index",
        "breaker_source_order_block_formed_index",
        "breaker_retest_index",
        "breaker_age_bars",
        "breaker_retest_count",
        "breaker_zone_lower",
        "breaker_zone_upper",
        "displacement_index",
        "entry_price",
        "stop_price",
        "target_price",
        "stop_reference",
        "target_reference",
        "rr_ratio",
        "tb_bars_held",
        "tb_return",
        "label_quality",
        "sample_weight",
    ):
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")

    if "event_time" not in working.columns:
        working["event_time"] = pd.NaT
    working["event_time"] = pd.to_datetime(working["event_time"], errors="coerce", utc=True)
    working["event_year"] = working["event_time"].dt.year.astype("Int64")
    working["event_month"] = working["event_time"].dt.tz_convert(None).dt.to_period("M").astype(str)
    working["excluded"] = _as_bool_series(working.get("excluded", False), index=working.index)
    working["positive_event"] = working["tb_outcome"].isin(POSITIVE_OUTCOMES)
    return working


def _attach_regime_context(classic: pd.DataFrame, regime_source: pd.DataFrame) -> pd.DataFrame:
    out = classic.copy()
    regime_columns = [
        "trend_regime",
        "vol_regime",
        "session_regime",
        "stress_regime",
        "composite_regime",
    ]
    try:
        regimes = label_regimes(regime_source.reset_index(drop=True))
    except Exception as exc:  # pragma: no cover - audit should report, not fail, on optional context.
        out["regime_context_error"] = str(exc)
        return out
    for column in regime_columns:
        out[column] = ""
    valid = out["signal_index"].dropna().astype(int)
    valid = valid.loc[valid.between(0, len(regimes) - 1)]
    for column in regime_columns:
        out.loc[valid.index, column] = regimes.iloc[valid.to_numpy()][column].to_numpy()
    return out


def _build_sample_summary(classic: pd.DataFrame, usable: pd.DataFrame) -> dict[str, Any]:
    first_time = classic["event_time"].min() if not classic.empty else pd.NaT
    last_time = classic["event_time"].max() if not classic.empty else pd.NaT
    return {
        "headline": {
            "total_events": int(len(classic)),
            "usable_events": int(len(usable)),
            "excluded_events": int(classic["excluded"].sum()) if not classic.empty else 0,
            "positive_events": int(usable["positive_event"].sum()) if not usable.empty else 0,
            "base_rate_pct": _pct(usable["positive_event"].sum(), len(usable)) if not usable.empty else 0.0,
            "long_events": int(classic["event_direction"].eq("long").sum()) if not classic.empty else 0,
            "short_events": int(classic["event_direction"].eq("short").sum()) if not classic.empty else 0,
            "usable_long_events": int(usable["event_direction"].eq("long").sum()) if not usable.empty else 0,
            "usable_short_events": int(usable["event_direction"].eq("short").sum()) if not usable.empty else 0,
            "first_event_time": first_time.isoformat() if not pd.isna(first_time) else None,
            "last_event_time": last_time.isoformat() if not pd.isna(last_time) else None,
        },
        "outcomes": {
            "tb_outcome_counts": _value_counts(usable.get("tb_outcome", pd.Series(dtype=str))),
            "base_rate_by_direction": _group_rate(usable, "event_direction"),
            "events_by_year": _value_counts(classic.get("event_year", pd.Series(dtype=object))),
            "events_by_month": _value_counts(classic.get("event_month", pd.Series(dtype=object))),
        },
        "quality": {
            "label_quality": _quantile_summary(usable.get("label_quality")),
            "sample_weight": _quantile_summary(usable.get("sample_weight")),
            "rr_ratio": _quantile_summary(usable.get("rr_ratio")),
            "tb_bars_held": _quantile_summary(usable.get("tb_bars_held")),
            "stop_distance": _quantile_summary((usable["entry_price"] - usable["stop_price"]).abs()) if not usable.empty else {},
            "target_distance": _quantile_summary((usable["target_price"] - usable["entry_price"]).abs()) if not usable.empty else {},
        },
        "regimes": {
            column: _group_rate(usable, column)
            for column in ("trend_regime", "vol_regime", "session_regime", "stress_regime", "composite_regime")
            if column in usable.columns
        },
    }


def _build_integrity_summary(classic: pd.DataFrame) -> dict[str, Any]:
    total = int(len(classic))
    if classic.empty:
        return {
            "total_events_checked": 0,
            "missing_breaker_id_count": 0,
            "missing_source_order_block_id_count": 0,
            "source_order_block_before_activation_pct": 0.0,
            "activation_before_retest_pct": 0.0,
            "signal_equals_retest_index_pct": 0.0,
            "finite_ordered_zone_bounds_pct": 0.0,
            "first_retest_only_pct": 0.0,
            "displacement_not_after_signal_pct": 0.0,
            "entry_stop_target_geometry_valid_pct": 0.0,
        }

    source_before_activation = classic["breaker_source_order_block_formed_index"].lt(classic["breaker_activation_index"])
    activation_before_retest = classic["breaker_activation_index"].lt(classic["signal_index"])
    signal_equals_retest = classic["breaker_retest_index"].eq(classic["signal_index"])
    finite_bounds = _finite_pair(classic["breaker_zone_lower"], classic["breaker_zone_upper"])
    ordered_bounds = finite_bounds & classic["breaker_zone_lower"].lt(classic["breaker_zone_upper"])
    first_retest = classic["breaker_retest_count"].eq(1)
    displacement_not_after_signal = classic["displacement_index"].isna() | classic["displacement_index"].le(classic["signal_index"])
    long_geometry = classic["event_direction"].eq("long") & classic["stop_price"].lt(classic["entry_price"]) & classic["entry_price"].lt(classic["target_price"])
    short_geometry = classic["event_direction"].eq("short") & classic["target_price"].lt(classic["entry_price"]) & classic["entry_price"].lt(classic["stop_price"])
    valid_geometry = (long_geometry | short_geometry) & _finite_triplet(classic["entry_price"], classic["stop_price"], classic["target_price"])

    return {
        "total_events_checked": total,
        "missing_breaker_id_count": int(classic["breaker_id"].isna().sum()),
        "missing_source_order_block_id_count": int(classic["breaker_source_order_block_id"].isna().sum()),
        "source_order_block_before_activation_pct": _pct(source_before_activation.sum(), total),
        "activation_before_retest_pct": _pct(activation_before_retest.sum(), total),
        "signal_equals_retest_index_pct": _pct(signal_equals_retest.sum(), total),
        "finite_ordered_zone_bounds_pct": _pct(ordered_bounds.sum(), total),
        "first_retest_only_pct": _pct(first_retest.sum(), total),
        "displacement_not_after_signal_pct": _pct(displacement_not_after_signal.sum(), total),
        "entry_stop_target_geometry_valid_pct": _pct(valid_geometry.sum(), total),
    }


def _build_isolation_summary(
    *,
    classic: pd.DataFrame,
    usable: pd.DataFrame,
    labels: pd.DataFrame | None,
    canonical_events: pd.DataFrame,
    canonical_labels: pd.DataFrame | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "labels_available": labels is not None,
        "classic_label_columns_present": False,
        "canonical_label_columns_present": False,
        "canonical_baseline_comparison_mode": "not_requested",
        "classic_label_positive_counts": {},
        "canonical_target_positive_counts": {},
        "classic_only_canonical_meta_leak_count": 0,
        "classic_only_canonical_family_leak_count": 0,
        "canonical_columns_equal_baseline": None,
        "canonical_column_sum_deltas_vs_baseline": {},
    }
    if labels is None:
        return result

    labels = labels.reset_index(drop=True).copy()
    result["classic_label_columns_present"] = all(column in labels.columns for column in CLASSIC_LABEL_COLUMNS)
    result["canonical_label_columns_present"] = all(column in labels.columns for column in CANONICAL_LABEL_COLUMNS)
    result["classic_label_positive_counts"] = {
        column: int(pd.to_numeric(labels.get(column, 0), errors="coerce").fillna(0).sum())
        for column in CLASSIC_LABEL_COLUMNS
        if column in labels.columns
    }
    result["canonical_target_positive_counts"] = {
        column: int(pd.to_numeric(labels.get(column, 0), errors="coerce").fillna(0).sum())
        for column in CANONICAL_LABEL_COLUMNS
        if column in labels.columns
    }

    canonical_pairs = set(
        zip(
            canonical_events.get("signal_index", pd.Series(dtype=float)).dropna().astype(int),
            canonical_events.loc[canonical_events["signal_index"].notna(), "event_direction"].astype(str),
            strict=False,
        )
    )
    meta_leaks = 0
    family_leaks = 0
    for _, event in usable.iterrows():
        signal_index = int(event["signal_index"])
        direction = str(event["event_direction"])
        if (signal_index, direction) in canonical_pairs or signal_index < 0 or signal_index >= len(labels):
            continue
        meta_column = f"label_{direction}_ict_meta"
        reversal_column = f"label_{direction}_ict_reversal"
        continuation_column = f"label_{direction}_ict_continuation"
        if meta_column in labels.columns and _scalar_int(labels.loc[signal_index, meta_column]) != 0:
            meta_leaks += 1
        family_values = [
            _scalar_int(labels.loc[signal_index, column])
            for column in (reversal_column, continuation_column)
            if column in labels.columns
        ]
        if any(value != 0 for value in family_values):
            family_leaks += 1
    result["classic_only_canonical_meta_leak_count"] = int(meta_leaks)
    result["classic_only_canonical_family_leak_count"] = int(family_leaks)

    if canonical_labels is not None:
        canonical_labels = canonical_labels.reset_index(drop=True)
        comparable_columns = [
            column
            for column in CANONICAL_LABEL_COLUMNS
            if column in labels.columns and column in canonical_labels.columns and len(labels) == len(canonical_labels)
        ]
        if comparable_columns:
            current_sums = {
                column: int(pd.to_numeric(labels[column], errors="coerce").fillna(0).sum())
                for column in comparable_columns
            }
            baseline_sums = {
                column: int(pd.to_numeric(canonical_labels[column], errors="coerce").fillna(0).sum())
                for column in comparable_columns
            }
            if sum(current_sums.values()) == 0 and sum(baseline_sums.values()) > 0:
                result["canonical_baseline_comparison_mode"] = "not_applicable_classic_only_artifact"
                result["canonical_columns_equal_baseline"] = None
            else:
                result["canonical_baseline_comparison_mode"] = "direct_column_equality"
                result["canonical_columns_equal_baseline"] = bool(labels[comparable_columns].equals(canonical_labels[comparable_columns]))
            result["canonical_column_sum_deltas_vs_baseline"] = {
                column: int(current_sums[column] - baseline_sums[column])
                for column in comparable_columns
            }
    return result


def _build_duplicate_overlap_summary(classic: pd.DataFrame, canonical_events: pd.DataFrame) -> dict[str, Any]:
    if classic.empty:
        return {
            "duplicate_same_breaker_side_event_count": 0,
            "long_short_same_bar_collision_count": 0,
            "adjacent_same_breaker_fire_count": 0,
            "canonical_same_signal_direction_overlap_count": 0,
            "canonical_setup4_same_signal_direction_overlap_count": 0,
        }
    keyed = classic.dropna(subset=["breaker_id"]).copy()
    duplicate_same_breaker = (
        int(keyed.groupby(["event_direction", "breaker_id"], dropna=False).size().sub(1).clip(lower=0).sum())
        if not keyed.empty
        else 0
    )
    signal_direction_dupes = int(classic.duplicated(subset=["signal_index", "event_direction"]).sum())
    collisions = (
        classic.drop_duplicates(subset=["signal_index", "event_direction"])
        .groupby("signal_index")["event_direction"]
        .nunique()
        .gt(1)
        .sum()
    )
    adjacent = 0
    for _, group in keyed.sort_values("signal_index").groupby(["event_direction", "breaker_id"], dropna=False):
        diffs = group["signal_index"].diff().dropna()
        adjacent += int(diffs.le(1).sum())

    canonical_pairs = set(
        zip(
            canonical_events.get("signal_index", pd.Series(dtype=float)).dropna().astype(int),
            canonical_events.loc[canonical_events["signal_index"].notna(), "event_direction"].astype(str),
            strict=False,
        )
    )
    canonical_setup4 = canonical_events.loc[canonical_events.get("setup_type", "").eq("ob_retest_after_mss")].copy()
    setup4_pairs = set(
        zip(
            canonical_setup4.get("signal_index", pd.Series(dtype=float)).dropna().astype(int),
            canonical_setup4.loc[canonical_setup4["signal_index"].notna(), "event_direction"].astype(str),
            strict=False,
        )
    )
    classic_pairs = {
        (int(row["signal_index"]), str(row["event_direction"]))
        for _, row in classic.loc[classic["signal_index"].notna()].iterrows()
    }
    return {
        "duplicate_same_breaker_side_event_count": duplicate_same_breaker,
        "same_signal_direction_duplicate_count": signal_direction_dupes,
        "long_short_same_bar_collision_count": int(collisions),
        "adjacent_same_breaker_fire_count": int(adjacent),
        "canonical_same_signal_direction_overlap_count": int(len(classic_pairs.intersection(canonical_pairs))),
        "canonical_setup4_same_signal_direction_overlap_count": int(len(classic_pairs.intersection(setup4_pairs))),
    }


def _build_leakage_summary(
    *,
    events: pd.DataFrame,
    labels: pd.DataFrame | None,
    prepared_root: str | Path | None,
) -> dict[str, Any]:
    event_columns = [str(column) for column in events.columns]
    suspicious_event_columns = [
        column
        for column in event_columns
        if any(token in column.lower() for token in ("tb_", "target_hit", "barrier_end", "exit_", "label_quality", "sample_weight"))
    ]
    label_columns = [] if labels is None else [str(column) for column in labels.columns]
    return {
        "event_outcome_columns_present": suspicious_event_columns,
        "label_columns_present": {
            "classic": [column for column in CLASSIC_LABEL_COLUMNS if column in label_columns],
            "canonical": [column for column in CANONICAL_LABEL_COLUMNS if column in label_columns],
        },
        "prepared_feature_audit": _inspect_prepared_features(prepared_root),
    }


def _inspect_prepared_features(prepared_root: str | Path | None) -> dict[str, Any]:
    if prepared_root is None:
        return {
            "available": False,
            "reason": "prepared_root_not_provided",
            "suspicious_reference_count": 0,
            "suspicious_references": [],
        }
    root = Path(prepared_root)
    if not root.exists():
        return {
            "available": False,
            "reason": f"missing_prepared_root:{root}",
            "suspicious_reference_count": 0,
            "suspicious_references": [],
        }

    suspicious: list[dict[str, str]] = []
    inspected_paths: list[str] = []
    inspected_feature_count = 0
    for target_name in ("long_ict_classic_breaker", "short_ict_classic_breaker"):
        target_dir = root / target_name
        if not target_dir.exists():
            continue
        feature_sources = _load_prepared_feature_names(target_dir)
        for path, feature_names in feature_sources:
            inspected_paths.append(str(path))
            inspected_feature_count += len(feature_names)
            for feature_name in feature_names:
                lowered = str(feature_name).strip().lower()
                if not lowered:
                    continue
                for token in SUSPICIOUS_FEATURE_TOKENS:
                    if token in lowered:
                        suspicious.append(
                            {
                                "target_name": target_name,
                                "path": str(path),
                                "feature": str(feature_name),
                                "token": token,
                            }
                        )
                        break
    return {
        "available": True,
        "prepared_root": str(root),
        "inspected_paths": inspected_paths,
        "inspected_feature_count": int(inspected_feature_count),
        "suspicious_reference_count": int(len(suspicious)),
        "suspicious_references": suspicious[:50],
    }


def _load_prepared_feature_names(target_dir: Path) -> list[tuple[Path, list[str]]]:
    sources: list[tuple[Path, list[str]]] = []

    features_path = target_dir / "features.json"
    if features_path.exists():
        try:
            payload = json.loads(features_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        features = payload.get("features") if isinstance(payload, Mapping) else None
        if isinstance(features, list):
            sources.append((features_path, [str(feature) for feature in features]))

    importance_path = target_dir / "feature_importance.csv"
    if importance_path.exists():
        try:
            ranking = pd.read_csv(importance_path, usecols=["feature"])
        except (ValueError, pd.errors.EmptyDataError):
            ranking = pd.DataFrame()
        if "feature" in ranking.columns:
            sources.append((importance_path, ranking["feature"].dropna().astype(str).tolist()))

    return sources


def _build_verdict(
    *,
    headline: Mapping[str, Any],
    integrity: Mapping[str, Any],
    isolation: Mapping[str, Any],
    leakage: Mapping[str, Any],
    duplicates: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    caveats: list[str] = []
    if int(headline.get("total_events", 0) or 0) == 0:
        failures.append("no_classic_breaker_events")
    if int(headline.get("usable_events", 0) or 0) == 0:
        failures.append("no_usable_classic_breaker_events")
    for key in (
        "source_order_block_before_activation_pct",
        "activation_before_retest_pct",
        "signal_equals_retest_index_pct",
        "finite_ordered_zone_bounds_pct",
        "first_retest_only_pct",
    ):
        if float(integrity.get(key, 0.0) or 0.0) < 100.0 and int(headline.get("total_events", 0) or 0) > 0:
            failures.append(key.replace("_pct", "_failed"))
    if not isolation.get("classic_label_columns_present", False):
        failures.append("classic_label_columns_missing")
    if int(isolation.get("classic_only_canonical_meta_leak_count", 0) or 0) > 0:
        failures.append("classic_events_leak_into_ict_meta")
    if int(isolation.get("classic_only_canonical_family_leak_count", 0) or 0) > 0:
        failures.append("classic_events_leak_into_canonical_family_targets")
    if int(leakage.get("prepared_feature_audit", {}).get("suspicious_reference_count", 0) or 0) > 0:
        failures.append("prepared_feature_leakage_tokens")

    if int(headline.get("usable_long_events", 0) or 0) < 25:
        caveats.append("sparse_long_sample")
    if int(headline.get("usable_short_events", 0) or 0) < 25:
        caveats.append("sparse_short_sample")
    base_rate = float(headline.get("base_rate_pct", 0.0) or 0.0)
    if base_rate <= 5.0 or base_rate >= 95.0:
        caveats.append("extreme_base_rate")
    if int(duplicates.get("duplicate_same_breaker_side_event_count", 0) or 0) > 0:
        caveats.append("duplicate_same_breaker_side_events")

    status = "FAIL / REVISE LABEL DESIGN" if failures else ("PASS WITH CAVEATS" if caveats else "PASS")
    return {
        "status": status,
        "failures": failures,
        "caveats": caveats,
    }


def _build_review_rows(classic: pd.DataFrame, *, per_bucket: int = 5) -> pd.DataFrame:
    if classic.empty:
        return pd.DataFrame()
    buckets: list[pd.DataFrame] = []
    group_columns = ["event_direction", "event_year", "tb_outcome"]
    for _, group in classic.sort_values("event_time").groupby(group_columns, dropna=False):
        buckets.append(group.head(per_bucket).copy())
    review = pd.concat(buckets, ignore_index=True) if buckets else classic.head(50).copy()
    columns = [
        "event_time",
        "event_direction",
        "tb_outcome",
        "excluded",
        "exclude_reasons",
        "signal_index",
        "entry_index",
        "barrier_end_index",
        "breaker_id",
        "breaker_source_order_block_id",
        "breaker_source_order_block_formed_index",
        "breaker_activation_index",
        "breaker_retest_index",
        "breaker_age_bars",
        "breaker_retest_count",
        "breaker_zone_lower",
        "breaker_zone_upper",
        "entry_price",
        "stop_price",
        "target_price",
        "rr_ratio",
        "label_quality",
        "sample_weight",
        "trend_regime",
        "vol_regime",
        "session_regime",
        "stress_regime",
        "composite_regime",
    ]
    review = review.loc[:, [column for column in columns if column in review.columns]].copy()
    if "event_time" in review.columns:
        review["event_time"] = pd.to_datetime(review["event_time"], errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return review


def _group_rate(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, dropna=False, sort=True):
        rows.append(
            {
                "value": _json_safe(value),
                "events": int(len(group)),
                "positive_events": int(group["positive_event"].sum()),
                "base_rate_pct": _pct(group["positive_event"].sum(), len(group)),
            }
        )
    return rows


def _value_counts(values: pd.Series) -> list[dict[str, Any]]:
    if values is None:
        return []
    series = pd.Series(values).astype("object")
    counts = series.where(series.notna(), "missing").astype(str).value_counts(dropna=False)
    total = int(counts.sum())
    return [
        {"value": str(value), "count": int(count), "share_pct": _pct(count, total)}
        for value, count in counts.items()
    ]


def _quantile_summary(values: Any) -> dict[str, float | None]:
    cleaned = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return {"count": 0, "p10": None, "p50": None, "p90": None, "mean": None}
    return {
        "count": int(len(cleaned)),
        "p10": float(cleaned.quantile(0.10)),
        "p50": float(cleaned.quantile(0.50)),
        "p90": float(cleaned.quantile(0.90)),
        "mean": float(cleaned.mean()),
    }


def _as_bool_series(value: Any, *, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value.fillna(False)
        normalized = value.fillna(False).astype(str).str.strip().str.lower()
        return normalized.isin({"1", "true", "t", "yes"})
    return pd.Series(bool(value), index=index)


def _scalar_int(value: Any) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0
    return int(numeric)


def _finite_pair(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)


def _finite_triplet(first: pd.Series, second: pd.Series, third: pd.Series) -> pd.Series:
    return _finite_pair(first, second) & third.notna() & np.isfinite(third)


def _pct(numerator: float | int, denominator: float | int) -> float:
    if float(denominator) <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


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
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_list())
    return str(value)
