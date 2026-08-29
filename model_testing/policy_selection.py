from __future__ import annotations

from typing import Any

import pandas as pd

from model_testing.ote_policy_metrics import add_unit_aliases

POLICY_BASELINE_NAME = "global_threshold"
EVENT_F05_DEGRADATION_TOLERANCE = 0.02


def select_policy_variant(
    evaluation: pd.DataFrame,
    *,
    split_name: str,
    min_trades_per_week: float,
    apply_to_base_policy_variants: bool = False,
    reference_split_name: str | None = None,
) -> dict[str, Any]:
    split_rows = evaluation.loc[evaluation["dataset_split"] == split_name].copy()
    if split_rows.empty:
        raise ValueError(f"Policy selection is missing the {split_name!r} split rows.")

    baseline = split_rows.loc[split_rows["policy_name"] == POLICY_BASELINE_NAME]
    if baseline.empty:
        raise ValueError("Policy selection requires a global_threshold baseline row.")
    baseline_row = baseline.iloc[0]

    candidate_rows = split_rows.loc[split_rows["policy_name"] != POLICY_BASELINE_NAME].copy()
    if candidate_rows.empty:
        return _fallback_selection(
            baseline_row,
            split_name=split_name,
            apply_to_base_policy_variants=apply_to_base_policy_variants,
        )

    reference_rows = None
    reference_baseline_row = None
    if reference_split_name is not None:
        reference_rows = evaluation.loc[evaluation["dataset_split"] == reference_split_name].copy()
        if not reference_rows.empty:
            reference_baseline = reference_rows.loc[
                reference_rows["policy_name"] == POLICY_BASELINE_NAME
            ]
            if not reference_baseline.empty:
                reference_baseline_row = reference_baseline.iloc[0]
                reference_lookup = reference_rows.set_index("policy_name", drop=False)
                candidate_rows["reference_post_cost_expectancy_pips"] = candidate_rows["policy_name"].map(
                    lambda value: _coerce_metric(
                        reference_lookup.at[value, "post_cost_expectancy_pips"]
                    )
                    if value in reference_lookup.index
                    else float("nan")
                )
                candidate_rows["reference_event_f05"] = candidate_rows["policy_name"].map(
                    lambda value: _coerce_metric(reference_lookup.at[value, "event_f05"])
                    if value in reference_lookup.index
                    else float("nan")
                )
                candidate_rows["reference_net_pnl_pips"] = candidate_rows["policy_name"].map(
                    lambda value: _coerce_metric(reference_lookup.at[value, "net_pnl_pips"])
                    if value in reference_lookup.index
                    else float("nan")
                )

    candidate_rows = candidate_rows.copy()
    candidate_rows["post_cost_expectancy_pips"] = candidate_rows["post_cost_expectancy_pips"].map(_coerce_metric)
    candidate_rows["event_f05"] = candidate_rows["event_f05"].map(_coerce_metric)
    candidate_rows["net_pnl_pips"] = candidate_rows["net_pnl_pips"].map(_coerce_metric)
    candidate_rows["trades_per_week"] = candidate_rows["trades_per_week"].map(_coerce_metric)
    baseline_expectancy = _coerce_metric(baseline_row["post_cost_expectancy_pips"])
    baseline_event_f05 = _coerce_metric(baseline_row["event_f05"])

    candidate_rows["expectancy_delta_vs_baseline"] = candidate_rows["post_cost_expectancy_pips"] - baseline_expectancy
    candidate_rows["event_f05_delta_vs_baseline"] = candidate_rows["event_f05"] - baseline_event_f05
    candidate_rows["passes_frequency_floor"] = (
        candidate_rows["trades_per_week"] >= float(min_trades_per_week)
    )
    candidate_rows["passes_expectancy_gate"] = (
        (candidate_rows["post_cost_expectancy_pips"] > baseline_expectancy)
        & (candidate_rows["post_cost_expectancy_pips"] > 0.0)
        & (candidate_rows["net_pnl_pips"] > 0.0)
    )
    candidate_rows["passes_event_f05_floor"] = (
        candidate_rows["event_f05"] >= (baseline_event_f05 - EVENT_F05_DEGRADATION_TOLERANCE)
    )
    candidate_rows["is_hard_pruned_base_policy"] = (
        bool(apply_to_base_policy_variants)
        & candidate_rows["policy_name"].astype(str).isin({"global_threshold", "regime_threshold"})
    )

    if reference_baseline_row is not None and "reference_post_cost_expectancy_pips" in candidate_rows.columns:
        reference_baseline_expectancy = _coerce_metric(reference_baseline_row["post_cost_expectancy_pips"])
        reference_baseline_event_f05 = _coerce_metric(reference_baseline_row["event_f05"])
        candidate_rows["passes_reference_expectancy_gate"] = (
            (candidate_rows["reference_post_cost_expectancy_pips"].fillna(float("-inf")) > 0.0)
            & (
                candidate_rows["reference_post_cost_expectancy_pips"].fillna(float("-inf"))
                >= reference_baseline_expectancy
            )
        )
        candidate_rows["passes_reference_event_f05_floor"] = (
            candidate_rows["reference_event_f05"].fillna(float("-inf"))
            >= (reference_baseline_event_f05 - EVENT_F05_DEGRADATION_TOLERANCE)
        )
        candidate_rows["reference_expectancy_delta_vs_baseline"] = (
            candidate_rows["reference_post_cost_expectancy_pips"] - reference_baseline_expectancy
        )
        candidate_rows["min_split_post_cost_expectancy_pips"] = candidate_rows[
            ["post_cost_expectancy_pips", "reference_post_cost_expectancy_pips"]
        ].min(axis=1)
        candidate_rows["min_split_event_f05"] = candidate_rows[
            ["event_f05", "reference_event_f05"]
        ].min(axis=1)
    else:
        candidate_rows["passes_reference_expectancy_gate"] = True
        candidate_rows["passes_reference_event_f05_floor"] = True
        candidate_rows["reference_expectancy_delta_vs_baseline"] = candidate_rows[
            "expectancy_delta_vs_baseline"
        ]
        candidate_rows["min_split_post_cost_expectancy_pips"] = candidate_rows["post_cost_expectancy_pips"]
        candidate_rows["min_split_event_f05"] = candidate_rows["event_f05"]

    candidate_rows["passes_robustness_gate"] = (
        candidate_rows["passes_event_f05_floor"]
        & candidate_rows["passes_reference_expectancy_gate"]
        & candidate_rows["passes_reference_event_f05_floor"]
    )
    candidate_rows["qualifies"] = (
        candidate_rows["passes_frequency_floor"]
        & candidate_rows["passes_expectancy_gate"]
        & candidate_rows["passes_robustness_gate"]
    )

    qualified_rows = candidate_rows.loc[candidate_rows["qualifies"]].copy()
    if qualified_rows.empty:
        return _fallback_selection(
            baseline_row,
            split_name=split_name,
            apply_to_base_policy_variants=apply_to_base_policy_variants,
        )

    qualified_rows = qualified_rows.sort_values(
        [
            "min_split_post_cost_expectancy_pips",
            "post_cost_expectancy_pips",
            "reference_expectancy_delta_vs_baseline",
            "is_hard_pruned_base_policy",
            "min_split_event_f05",
            "event_f05",
            "net_pnl_pips",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    selected = qualified_rows.iloc[0]

    return {
        "qualified_policy_names": qualified_rows["policy_name"].astype(str).tolist(),
        "selected_policy_name": str(selected["policy_name"]),
        "selected_policy_metrics": _policy_metrics_from_row(selected),
        "selection_reason": _selected_policy_reason(
            split_name=split_name,
            is_hard_pruned_base_policy=bool(selected["is_hard_pruned_base_policy"]),
            has_reference_split=reference_baseline_row is not None,
        ),
    }


def _fallback_selection(
    baseline_row: pd.Series,
    *,
    split_name: str,
    apply_to_base_policy_variants: bool,
) -> dict[str, Any]:
    return {
        "qualified_policy_names": [],
        "selected_policy_name": POLICY_BASELINE_NAME,
        "selected_policy_metrics": _policy_metrics_from_row(baseline_row),
        "selection_reason": _fallback_selection_reason(
            split_name=split_name,
            apply_to_base_policy_variants=apply_to_base_policy_variants,
        ),
    }


def _selected_policy_reason(
    *,
    split_name: str,
    is_hard_pruned_base_policy: bool,
    has_reference_split: bool,
) -> str:
    split_label = "test" if split_name == "test" else split_name
    if is_hard_pruned_base_policy:
        if has_reference_split:
            return f"best_{split_label}_hard_pruned_base_policy_by_post_cost_expectancy_and_split_robustness"
        return f"best_{split_label}_hard_pruned_base_policy_by_post_cost_expectancy_and_robustness"
    if has_reference_split:
        return f"best_{split_label}_policy_by_post_cost_expectancy_and_split_robustness"
    return f"best_{split_label}_policy_by_post_cost_expectancy_and_robustness"


def _fallback_selection_reason(
    *,
    split_name: str,
    apply_to_base_policy_variants: bool,
) -> str:
    split_label = "test" if split_name == "test" else split_name
    if apply_to_base_policy_variants:
        return (
            f"hard_pruned_base_policy_retained_no_non_global_variant_met_{split_label}"
            "_expectancy_frequency_and_robustness_requirements"
        )
    return (
        f"no_non_global_policy_met_{split_label}"
        "_expectancy_frequency_and_robustness_requirements"
    )


def _policy_metrics_from_row(row: pd.Series) -> dict[str, Any]:
    return add_unit_aliases(
        {
            "event_f05": float(row["event_f05"]),
            "post_cost_expectancy_pips": float(row["post_cost_expectancy_pips"]),
            "net_pnl_pips": float(row["net_pnl_pips"]),
            "trades_per_week": float(row["trades_per_week"]),
        }
    )


def _coerce_metric(value: object) -> float:
    return float(pd.to_numeric(value, errors="coerce"))
