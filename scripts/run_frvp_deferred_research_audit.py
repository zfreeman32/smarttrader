from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SETUP_NAME_BY_TYPE = {
    1: "Setup 1",
    2: "Setup 2",
    3: "Setup 3",
    4: "Setup 4",
    5: "Setup 5",
    6: "Setup 6",
}
DEFAULT_SETUP6B_MAX_DISTANCE_ATR = 1.5
GAMMA_COLUMN_CANDIDATES = (
    "es_gamma_pin_flag",
    "es_gamma_strike_dist_atr",
    "es_gamma_abs_notional_rank",
    "frvp_magnet_is_gamma_pin",
)


@dataclass(frozen=True)
class BranchSpec:
    run_id: str
    label: str
    model_id: str
    backtest_root: Path
    label_family: str
    direction: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the saved FRVP artifact state for the deferred research cluster: "
            "Setup 6b, gamma-context features, pooled-vs-family splits, roll handling, and "
            "the standalone Setup 4 retest."
        )
    )
    parser.add_argument(
        "--phase02-dataset",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "frvp_es_primary_refresh_20260701"
        / "phase02"
        / "es_primary_frvp_phase04_dataset.csv.gz",
    )
    parser.add_argument(
        "--phase02-metadata",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "frvp_es_primary_refresh_20260701"
        / "phase02"
        / "es_primary_frvp_phase04_dataset.csv.metadata.json",
    )
    parser.add_argument(
        "--phase03-events",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "frvp_es_primary_refresh_20260701"
        / "phase03"
        / "es_primary_frvp_events.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "model_testing"
        / "reports"
        / "frvp_deferred_research_audit"
        / f"frvp_deferred_research_audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
    )
    parser.add_argument(
        "--setup6b-max-distance-atr",
        type=float,
        default=DEFAULT_SETUP6B_MAX_DISTANCE_ATR,
        help="Directional naked-VPOC distance threshold used for the Setup 6b readiness proxy.",
    )
    return parser


def default_branch_specs() -> tuple[BranchSpec, ...]:
    study_root = (
        REPO_ROOT
        / "model_testing"
        / "reports"
        / "frvp_regime_gated_deployment"
        / "frvp_regime_gated_deployment_20260721"
        / "backtest"
    )
    return (
        BranchSpec(
            run_id="long_continuation_v3_baseline",
            label="Long continuation v3 baseline",
            model_id="frvp_long_continuation_xgb_v1",
            backtest_root=study_root / "long_continuation_v3_baseline" / "frvp_long_continuation_xgb_v1",
            label_family="frvp_continuation",
            direction="long",
        ),
        BranchSpec(
            run_id="long_reversal_fullspan_baseline",
            label="Long reversal full-span control",
            model_id="frvp_long_reversal_xgb_v1",
            backtest_root=study_root / "long_reversal_fullspan_baseline" / "frvp_long_reversal_xgb_v1",
            label_family="frvp_reversal",
            direction="long",
        ),
        BranchSpec(
            run_id="long_reversal_recent2y_baseline",
            label="Long reversal recent-2y control",
            model_id="frvp_long_reversal_xgb_v1",
            backtest_root=study_root / "long_reversal_recent2y_baseline" / "frvp_long_reversal_xgb_v1",
            label_family="frvp_reversal",
            direction="long",
        ),
        BranchSpec(
            run_id="long_reversal_recent2y_operational",
            label="Long reversal recent-2y operational prune",
            model_id="frvp_long_reversal_xgb_v1",
            backtest_root=study_root / "long_reversal_recent2y_sdh_overlap_prune_v1" / "frvp_long_reversal_xgb_v1",
            label_family="frvp_reversal",
            direction="long",
        ),
    )


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    features = load_phase02_features(args.phase02_dataset)
    metadata = load_phase02_metadata(args.phase02_metadata)
    events = load_phase03_events(args.phase03_events)
    event_context = attach_feature_context(
        events,
        features,
        setup6b_max_distance_atr=float(args.setup6b_max_distance_atr),
    )

    branch_specs = default_branch_specs()
    branch_trade_frames: list[pd.DataFrame] = []
    branch_rows: list[dict[str, Any]] = []

    for spec in branch_specs:
        trades = load_selected_trades(spec.backtest_root / "selected_test_trades.csv")
        annotated = annotate_selected_trades(trades, event_context, spec=spec)
        branch_trade_frames.append(annotated)
        branch_rows.append(
            summarize_branch(
                annotated,
                spec=spec,
            )
        )

    all_branch_trades = pd.concat(branch_trade_frames, ignore_index=True, sort=False) if branch_trade_frames else pd.DataFrame()
    matched_branch_trades = (
        all_branch_trades.loc[all_branch_trades["annotation_matched"]].copy()
        if not all_branch_trades.empty and "annotation_matched" in all_branch_trades.columns
        else pd.DataFrame()
    )
    branch_summary = pd.DataFrame(branch_rows).sort_values("run_id").reset_index(drop=True)
    setup_breakdown = summarize_trade_groups(
        matched_branch_trades,
        group_columns=("run_id", "label", "model_id", "setup_type", "setup_name"),
    )
    roll_breakdown = summarize_trade_groups(
        matched_branch_trades,
        group_columns=("run_id", "label", "model_id", "flag_roll_bracket"),
    )
    setup_roll_breakdown = summarize_trade_groups(
        matched_branch_trades,
        group_columns=("run_id", "label", "model_id", "setup_type", "setup_name", "flag_roll_bracket"),
    )
    pooling_summary = build_pooling_summary(setup_breakdown)
    setup4_summary = summarize_setup4(event_context, matched_branch_trades)
    setup6b_readiness = summarize_setup6b_readiness(event_context)
    gamma_summary = summarize_gamma_readiness(metadata)
    readiness_summary = build_readiness_summary(
        event_context=event_context,
        branch_summary=branch_summary,
        pooling_summary=pooling_summary,
        setup4_summary=setup4_summary,
        setup6b_readiness=setup6b_readiness,
        gamma_summary=gamma_summary,
    )

    branch_summary.to_csv(output_root / "branch_summary.csv", index=False)
    setup_breakdown.to_csv(output_root / "branch_setup_breakdown.csv", index=False)
    roll_breakdown.to_csv(output_root / "branch_roll_breakdown.csv", index=False)
    setup_roll_breakdown.to_csv(output_root / "branch_setup_roll_breakdown.csv", index=False)
    pooling_summary.to_csv(output_root / "pooling_summary.csv", index=False)
    setup4_summary.to_csv(output_root / "setup4_summary.csv", index=False)
    setup6b_readiness.to_csv(output_root / "setup6b_readiness.csv", index=False)
    (output_root / "readiness_summary.json").write_text(
        json.dumps(readiness_summary, indent=2),
        encoding="utf-8",
    )
    (output_root / "DEFERRED_RESEARCH_SUMMARY.md").write_text(
        build_markdown_summary(
            generated_at_utc=datetime.now(timezone.utc),
            branch_summary=branch_summary,
            pooling_summary=pooling_summary,
            setup4_summary=setup4_summary,
            setup6b_readiness=setup6b_readiness,
            gamma_summary=gamma_summary,
            readiness_summary=readiness_summary,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_root": _repo_relative_str(output_root),
                "branch_summary_path": _repo_relative_str(output_root / "branch_summary.csv"),
                "setup_breakdown_path": _repo_relative_str(output_root / "branch_setup_breakdown.csv"),
                "roll_breakdown_path": _repo_relative_str(output_root / "branch_roll_breakdown.csv"),
                "summary_markdown_path": _repo_relative_str(output_root / "DEFERRED_RESEARCH_SUMMARY.md"),
                "readiness_summary_path": _repo_relative_str(output_root / "readiness_summary.json"),
                "branch_count": len(branch_summary),
                "gamma_columns_found": gamma_summary["gamma_columns_found"],
                "setup_types_present": readiness_summary["setup_types_present"],
            },
            indent=2,
        )
    )
    return 0


def load_phase02_features(path: Path) -> pd.DataFrame:
    usecols = [
        "datetime",
        "frvp_setup_type",
        "frvp_setup_side",
        "frvp_naked_vpoc_dist_above_atr",
        "frvp_naked_vpoc_dist_below_atr",
        "frvp_naked_vpoc_age_sessions",
        "frvp_naked_vpoc_count",
        "frvp_setup_confidence_rule",
    ]
    frame = pd.read_csv(
        path,
        usecols=usecols,
        compression="infer",
    )
    frame["source_row_idx"] = np.arange(len(frame), dtype=np.int64)
    frame["entry_datetime"] = _parse_datetime(frame["datetime"])
    return frame


def load_phase02_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase03_events(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["source_row_idx"] = pd.to_numeric(frame["confirm_index"], errors="coerce").astype("Int64")
    frame["entry_datetime"] = _parse_datetime(frame["entry_time"])
    frame["swing_datetime"] = _parse_datetime(frame["swing_time"])
    frame["confirm_datetime"] = _parse_datetime(frame["confirm_time"])
    frame["direction"] = np.where(pd.to_numeric(frame["setup_side"], errors="coerce").fillna(0) > 0, "long", "short")
    frame["setup_name"] = pd.to_numeric(frame["setup_type"], errors="coerce").fillna(0).astype(int).map(SETUP_NAME_BY_TYPE).fillna("Unknown")
    frame["excluded"] = _coerce_bool_series(frame.get("excluded"))
    frame["flag_roll_bracket"] = _coerce_bool_series(frame.get("flag_roll_bracket"))
    frame["flag_roll_span"] = _coerce_bool_series(frame.get("flag_roll_span"))
    frame["is_tp"] = frame["tb_outcome"].astype(str).eq("tp")
    return dedupe_trade_event_annotations(frame)


def load_selected_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["entry_datetime"] = _parse_datetime(frame["entry_datetime"])
    if "source_row_idx" in frame.columns:
        frame["source_row_idx"] = pd.to_numeric(frame["source_row_idx"], errors="coerce").astype("Int64")
    return frame


def dedupe_trade_event_annotations(events: pd.DataFrame) -> pd.DataFrame:
    working = events.copy()
    sort_columns = ["entry_datetime", "direction", "label_family", "excluded", "label_quality"]
    ascending = [True, True, True, True, False]
    for column in ("label_quality",):
        if column not in working.columns:
            working[column] = np.nan
    working = working.sort_values(sort_columns, ascending=ascending, kind="stable")
    dedupe_keys = ["source_row_idx", "direction", "label_family"] if "source_row_idx" in working.columns else ["entry_datetime", "direction", "label_family"]
    return working.drop_duplicates(subset=dedupe_keys, keep="first").reset_index(drop=True)


def attach_feature_context(
    events: pd.DataFrame,
    features: pd.DataFrame,
    *,
    setup6b_max_distance_atr: float,
) -> pd.DataFrame:
    keep_columns = [
        "entry_datetime",
        "frvp_setup_type",
        "frvp_setup_side",
        "frvp_naked_vpoc_dist_above_atr",
        "frvp_naked_vpoc_dist_below_atr",
        "frvp_naked_vpoc_age_sessions",
        "frvp_naked_vpoc_count",
        "frvp_setup_confidence_rule",
    ]
    feature_columns = keep_columns + (["source_row_idx"] if "source_row_idx" in features.columns else [])
    join_key = "source_row_idx" if "source_row_idx" in events.columns and "source_row_idx" in features.columns else "entry_datetime"
    merged = events.merge(
        features.loc[:, feature_columns],
        on=join_key,
        how="left",
        suffixes=("", "_feature"),
        validate="many_to_one",
    )
    above = pd.to_numeric(merged["frvp_naked_vpoc_dist_above_atr"], errors="coerce")
    below = pd.to_numeric(merged["frvp_naked_vpoc_dist_below_atr"], errors="coerce")
    merged["directional_naked_vpoc_distance_atr"] = np.where(merged["direction"].eq("long"), above, below)
    merged["nearest_naked_vpoc_distance_atr"] = pd.concat([above, below], axis=1).min(axis=1, skipna=True)
    merged["directional_naked_vpoc_in_reach"] = _distance_in_reach(
        merged["directional_naked_vpoc_distance_atr"],
        max_distance_atr=setup6b_max_distance_atr,
    )
    merged["any_naked_vpoc_in_reach"] = _distance_in_reach(
        merged["nearest_naked_vpoc_distance_atr"],
        max_distance_atr=setup6b_max_distance_atr,
    )
    return merged


def annotate_selected_trades(
    trades: pd.DataFrame,
    event_context: pd.DataFrame,
    *,
    spec: BranchSpec,
) -> pd.DataFrame:
    event_subset = event_context.loc[
        event_context["label_family"].astype(str).eq(spec.label_family)
        & event_context["direction"].astype(str).eq(spec.direction)
    ].copy()
    columns = [
        "entry_datetime",
        "setup_type",
        "setup_name",
        "direction",
        "label_family",
        "barrier_family",
        "label_quality",
        "setup_confidence",
        "flag_roll_bracket",
        "flag_roll_span",
        "directional_naked_vpoc_distance_atr",
        "directional_naked_vpoc_in_reach",
        "any_naked_vpoc_in_reach",
        "frvp_naked_vpoc_count",
        "frvp_naked_vpoc_age_sessions",
        "tb_outcome",
    ]
    join_key = "source_row_idx" if "source_row_idx" in trades.columns and "source_row_idx" in event_subset.columns else "entry_datetime"
    event_columns = columns + ([join_key] if join_key not in columns else [])
    annotated = trades.merge(
        event_subset.loc[:, event_columns],
        on=join_key,
        how="left",
        validate="many_to_one",
    )
    if "direction" not in annotated.columns:
        if "direction_x" in annotated.columns:
            annotated["direction"] = annotated["direction_x"]
        elif "direction_y" in annotated.columns:
            annotated["direction"] = annotated["direction_y"]
    if "entry_datetime" not in annotated.columns:
        if "entry_datetime_x" in annotated.columns:
            annotated["entry_datetime"] = annotated["entry_datetime_x"]
        elif "entry_datetime_y" in annotated.columns:
            annotated["entry_datetime"] = annotated["entry_datetime_y"]
    annotated["run_id"] = spec.run_id
    annotated["label"] = spec.label
    annotated["model_id"] = spec.model_id
    annotated["expected_label_family"] = spec.label_family
    annotated["annotation_matched"] = annotated["setup_type"].notna()
    annotated["flag_roll_bracket"] = _coerce_bool_series(annotated.get("flag_roll_bracket"))
    annotated["flag_roll_span"] = _coerce_bool_series(annotated.get("flag_roll_span"))
    annotated["directional_naked_vpoc_in_reach"] = _coerce_bool_series(annotated.get("directional_naked_vpoc_in_reach"))
    annotated["any_naked_vpoc_in_reach"] = _coerce_bool_series(annotated.get("any_naked_vpoc_in_reach"))
    return annotated


def summarize_branch(
    trades: pd.DataFrame,
    *,
    spec: BranchSpec,
) -> dict[str, Any]:
    total_trades = int(len(trades))
    matched_trades = int(trades["annotation_matched"].sum())
    return {
        "run_id": spec.run_id,
        "label": spec.label,
        "model_id": spec.model_id,
        "label_family": spec.label_family,
        "direction": spec.direction,
        "trade_count": total_trades,
        "annotated_trade_count": matched_trades,
        "annotation_match_rate": _safe_share(matched_trades, total_trades),
        "net_pnl_units": _safe_sum(trades.get("net_pnl_units")),
        "expectancy_units": _safe_mean(trades.get("net_pnl_units")),
        "win_rate": _safe_mean(pd.to_numeric(trades.get("net_pnl_units"), errors="coerce") > 0),
        "roll_bracket_trade_share": _safe_mean(trades.get("flag_roll_bracket")),
        "directional_naked_vpoc_trade_share": _safe_mean(trades.get("directional_naked_vpoc_in_reach")),
        "distinct_setups_traded": int(pd.to_numeric(trades.get("setup_type"), errors="coerce").dropna().astype(int).nunique()),
        "backtest_root": _repo_relative_str(spec.backtest_root),
    }


def summarize_trade_groups(
    trades: pd.DataFrame,
    *,
    group_columns: Iterable[str],
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    working = trades.copy()
    working["win_flag"] = pd.to_numeric(working["net_pnl_units"], errors="coerce") > 0
    summary = (
        working.groupby(list(group_columns), dropna=False, sort=True)
        .agg(
            trade_count=("net_pnl_units", "size"),
            win_rate=("win_flag", "mean"),
            net_pnl_units=("net_pnl_units", "sum"),
            expectancy_units=("net_pnl_units", "mean"),
            roll_bracket_trade_share=("flag_roll_bracket", "mean"),
            directional_naked_vpoc_trade_share=("directional_naked_vpoc_in_reach", "mean"),
            setup_quality_mean=("label_quality", "mean"),
        )
        .reset_index()
    )
    summary["win_rate"] = summary["win_rate"].astype(float)
    return summary


def build_pooling_summary(setup_breakdown: pd.DataFrame) -> pd.DataFrame:
    if setup_breakdown.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for run_id, group in setup_breakdown.groupby("run_id", sort=True):
        ordered = group.sort_values("trade_count", ascending=False, kind="stable").reset_index(drop=True)
        total_trades = int(ordered["trade_count"].sum())
        positive = ordered.loc[ordered["expectancy_units"] > 0]
        negative = ordered.loc[ordered["expectancy_units"] < 0]
        dominant = ordered.iloc[0]
        best = ordered.sort_values("expectancy_units", ascending=False, kind="stable").iloc[0]
        worst = ordered.sort_values("expectancy_units", ascending=True, kind="stable").iloc[0]
        rows.append(
            {
                "run_id": run_id,
                "label": str(ordered["label"].iloc[0]),
                "model_id": str(ordered["model_id"].iloc[0]),
                "setup_count": int(len(ordered)),
                "trade_count": total_trades,
                "dominant_setup_type": int(dominant["setup_type"]),
                "dominant_setup_name": str(dominant["setup_name"]),
                "dominant_setup_trade_share": _safe_share(int(dominant["trade_count"]), total_trades),
                "best_setup_type": int(best["setup_type"]),
                "best_setup_name": str(best["setup_name"]),
                "best_setup_expectancy_units": float(best["expectancy_units"]),
                "worst_setup_type": int(worst["setup_type"]),
                "worst_setup_name": str(worst["setup_name"]),
                "worst_setup_expectancy_units": float(worst["expectancy_units"]),
                "positive_setup_count": int(len(positive)),
                "negative_setup_count": int(len(negative)),
                "mixed_expectancy_signs": bool(not positive.empty and not negative.empty),
                "split_recommended": bool(
                    len(ordered) >= 2
                    and (
                        (not positive.empty and not negative.empty)
                        or _safe_share(int(dominant["trade_count"]), total_trades) >= 0.65
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


def summarize_setup4(event_context: pd.DataFrame, branch_trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    setup4_events = event_context.loc[pd.to_numeric(event_context["setup_type"], errors="coerce").eq(4)].copy()
    usable_setup4 = setup4_events.loc[~setup4_events["excluded"]].copy()
    for direction, group in usable_setup4.groupby("direction", sort=True):
        rows.append(
            {
                "scope": "usable_events",
                "direction": direction,
                "trade_run_id": "",
                "count": int(len(group)),
                "base_rate_tp": _safe_mean(group["is_tp"]),
                "quality_mean": _safe_mean(group["label_quality"]),
                "roll_bracket_share": _safe_mean(group["flag_roll_bracket"]),
                "directional_naked_vpoc_share": _safe_mean(group["directional_naked_vpoc_in_reach"]),
                "net_pnl_units": np.nan,
                "expectancy_units": np.nan,
            }
        )
    setup4_trades = branch_trades.loc[pd.to_numeric(branch_trades["setup_type"], errors="coerce").eq(4)].copy()
    for run_id, group in setup4_trades.groupby("run_id", sort=True):
        rows.append(
            {
                "scope": "selected_test_trades",
                "direction": str(group["direction"].iloc[0]),
                "trade_run_id": run_id,
                "count": int(len(group)),
                "base_rate_tp": _safe_mean(pd.to_numeric(group["net_pnl_units"], errors="coerce") > 0),
                "quality_mean": _safe_mean(group["label_quality"]),
                "roll_bracket_share": _safe_mean(group["flag_roll_bracket"]),
                "directional_naked_vpoc_share": _safe_mean(group["directional_naked_vpoc_in_reach"]),
                "net_pnl_units": _safe_sum(group["net_pnl_units"]),
                "expectancy_units": _safe_mean(group["net_pnl_units"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_setup6b_readiness(event_context: pd.DataFrame) -> pd.DataFrame:
    usable = event_context.loc[~event_context["excluded"]].copy()
    if usable.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for family, group in usable.groupby("label_family", sort=True):
        in_reach = group.loc[group["directional_naked_vpoc_in_reach"]].copy()
        not_in_reach = group.loc[~group["directional_naked_vpoc_in_reach"]].copy()
        rows.append(
            {
                "label_family": family,
                "usable_events": int(len(group)),
                "directional_naked_vpoc_events": int(len(in_reach)),
                "directional_naked_vpoc_share": _safe_share(int(len(in_reach)), int(len(group))),
                "tp_rate_when_in_reach": _safe_mean(in_reach["is_tp"]),
                "tp_rate_when_not_in_reach": _safe_mean(not_in_reach["is_tp"]),
                "quality_mean_when_in_reach": _safe_mean(in_reach["label_quality"]),
                "median_naked_vpoc_age_sessions_when_in_reach": _safe_median(in_reach["frvp_naked_vpoc_age_sessions"]),
            }
        )
    return pd.DataFrame(rows).sort_values("label_family").reset_index(drop=True)


def summarize_gamma_readiness(metadata: dict[str, Any]) -> dict[str, Any]:
    feature_columns = list(metadata.get("feature_columns", []))
    found = sorted(
        [
            str(column)
            for column in feature_columns
            if str(column) in GAMMA_COLUMN_CANDIDATES or str(column).startswith("es_gamma_")
        ]
    )
    return {
        "gamma_columns_found": found,
        "gamma_columns_present": bool(found),
        "phase02_feature_count": int(len(feature_columns)),
    }


def build_readiness_summary(
    *,
    event_context: pd.DataFrame,
    branch_summary: pd.DataFrame,
    pooling_summary: pd.DataFrame,
    setup4_summary: pd.DataFrame,
    setup6b_readiness: pd.DataFrame,
    gamma_summary: dict[str, Any],
) -> dict[str, Any]:
    setup_types = sorted(pd.to_numeric(event_context["setup_type"], errors="coerce").dropna().astype(int).unique().tolist())
    setup6b_implemented = any(value > 6 for value in setup_types)
    split_candidates = (
        pooling_summary.loc[pooling_summary["split_recommended"], "run_id"].astype(str).tolist()
        if not pooling_summary.empty
        else []
    )
    setup4_event_rows = setup4_summary.loc[setup4_summary["scope"].eq("usable_events")] if not setup4_summary.empty else pd.DataFrame()
    setup4_trade_rows = setup4_summary.loc[setup4_summary["scope"].eq("selected_test_trades")] if not setup4_summary.empty else pd.DataFrame()
    reversal_readiness = (
        setup6b_readiness.loc[setup6b_readiness["label_family"].eq("frvp_reversal")]
        if not setup6b_readiness.empty
        else pd.DataFrame()
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "setup_types_present": setup_types,
        "setup6b_implemented": bool(setup6b_implemented),
        "gamma_columns_found": gamma_summary["gamma_columns_found"],
        "gamma_columns_present": gamma_summary["gamma_columns_present"],
        "branch_count": int(len(branch_summary)),
        "split_candidate_run_ids": split_candidates,
        "setup4_usable_event_count": int(setup4_event_rows["count"].sum()) if not setup4_event_rows.empty else 0,
        "setup4_selected_trade_count": int(setup4_trade_rows["count"].sum()) if not setup4_trade_rows.empty else 0,
        "setup6b_reversal_directional_share": (
            float(reversal_readiness["directional_naked_vpoc_share"].iloc[0]) if not reversal_readiness.empty else 0.0
        ),
        "recommended_order": [
            "Per-setup-family audit/training lane before any new pooled-family optimization.",
            "Standalone Setup 4 retest once the per-setup path exists.",
            "Roll translation vs reset A/B on the control branches that still show roll-bracket drag, especially continuation and the full-span reversal control.",
            "Setup 6b only after the naked-VPOC target path is promoted from context to event logic.",
            "Gamma-context features only after a concrete options-data contract exists.",
        ],
    }


def build_markdown_summary(
    *,
    generated_at_utc: datetime,
    branch_summary: pd.DataFrame,
    pooling_summary: pd.DataFrame,
    setup4_summary: pd.DataFrame,
    setup6b_readiness: pd.DataFrame,
    gamma_summary: dict[str, Any],
    readiness_summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# FRVP Deferred Research Audit")
    lines.append("")
    lines.append(f"Generated: `{generated_at_utc.isoformat()}`")
    lines.append("")
    lines.append("This audit revisits the deferred FRVP research cluster using only saved repo artifacts.")
    lines.append("")
    lines.append("## Repo-State Snapshot")
    lines.append("")
    lines.append(f"- Saved Phase 3 setup types present: `{', '.join(str(value) for value in readiness_summary['setup_types_present'])}`")
    lines.append(f"- Setup 6b implemented in saved events: `{readiness_summary['setup6b_implemented']}`")
    lines.append(f"- Gamma-context columns present in Phase 2 feature metadata: `{gamma_summary['gamma_columns_present']}`")
    if gamma_summary["gamma_columns_found"]:
        lines.append(f"- Gamma columns found: `{', '.join(gamma_summary['gamma_columns_found'])}`")
    else:
        lines.append("- Gamma columns found: none")
    lines.append("")
    lines.append("## Pooled Vs. Per-Setup Read")
    lines.append("")
    if pooling_summary.empty:
        lines.append("- No pooling summary was generated.")
    else:
        for row in pooling_summary.itertuples(index=False):
            lines.append(
                f"- `{row.run_id}`: dominant `{row.dominant_setup_name}` share `{row.dominant_setup_trade_share:.3f}`, "
                f"best `{row.best_setup_name}` expectancy `{row.best_setup_expectancy_units:.2f}`, "
                f"worst `{row.worst_setup_name}` expectancy `{row.worst_setup_expectancy_units:.2f}`, "
                f"split recommended `{row.split_recommended}`."
            )
    lines.append("")
    lines.append("## Branch Snapshot")
    lines.append("")
    if branch_summary.empty:
        lines.append("- No branch summary was generated.")
    else:
        for row in branch_summary.itertuples(index=False):
            lines.append(
                f"- `{row.run_id}`: trades `{row.trade_count}`, net `{row.net_pnl_units:.2f}`, "
                f"expectancy `{row.expectancy_units:.2f}`, roll-bracket share `{row.roll_bracket_trade_share:.3f}`, "
                f"directional naked-VPOC share `{row.directional_naked_vpoc_trade_share:.3f}`."
            )
    lines.append("")
    lines.append("## Setup 4 Standalone Retest")
    lines.append("")
    if setup4_summary.empty:
        lines.append("- No Setup 4 summary was generated.")
    else:
        for row in setup4_summary.itertuples(index=False):
            if row.scope == "usable_events":
                lines.append(
                    f"- Usable Setup 4 `{row.direction}` events: `{row.count}` with TP rate `{row.base_rate_tp:.3f}` "
                    f"and quality `{row.quality_mean:.3f}`."
                )
            else:
                lines.append(
                    f"- `{row.trade_run_id}` Setup 4 selected trades: `{row.count}` with net `{row.net_pnl_units:.2f}` "
                    f"and expectancy `{row.expectancy_units:.2f}`."
                )
    lines.append("")
    lines.append("## Setup 6b Readiness Proxy")
    lines.append("")
    if setup6b_readiness.empty:
        lines.append("- No Setup 6b readiness summary was generated.")
    else:
        for row in setup6b_readiness.itertuples(index=False):
            lines.append(
                f"- `{row.label_family}`: directional naked-VPOC-in-reach share `{row.directional_naked_vpoc_share:.3f}` "
                f"({row.directional_naked_vpoc_events}/{row.usable_events}), TP in-reach `{row.tp_rate_when_in_reach:.3f}` "
                f"vs. not-in-reach `{row.tp_rate_when_not_in_reach:.3f}`."
            )
    lines.append("")
    lines.append("## Recommended Order")
    lines.append("")
    for item in readiness_summary["recommended_order"]:
        lines.append(f"1. {item}")
    return "\n".join(lines) + "\n"


def _distance_in_reach(values, *, max_distance_atr: float) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return series.ge(0.0) & series.le(float(max_distance_atr))


def _coerce_bool_series(values) -> pd.Series:
    series = pd.Series(values) if not isinstance(values, pd.Series) else values.copy()
    if series.empty:
        return series.astype(bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y"}
    return normalized.isin(true_values)


def _parse_datetime(values) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def _safe_sum(values) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if series.dropna().empty:
        return 0.0
    return float(series.sum())


def _safe_mean(values) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if series.dropna().empty:
        return 0.0
    return float(series.mean())


def _safe_median(values) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if series.dropna().empty:
        return 0.0
    return float(series.median())


def _safe_share(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _repo_relative_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("/", "\\")
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
