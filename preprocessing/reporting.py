from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def format_summary_report(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append("FEATURE PREPROCESSING SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Input file: {summary['input_file']}")
    lines.append(f"Metadata file: {summary['metadata_file']}")
    timezone_contract = summary.get("timezone_contract") or {}
    if timezone_contract:
        lines.append(
            "Timezone contract: "
            f"source={timezone_contract.get('source_timezone')} "
            f"canonical={timezone_contract.get('canonical_timezone')} "
            f"feature_clock={timezone_contract.get('feature_clock_timezone')}"
        )
    source_lineage = summary.get("source_lineage") or {}
    if source_lineage:
        lines.append(f"Builder source: {source_lineage.get('feature_builder_source_path')}")
        if source_lineage.get("upstream_source_path"):
            lines.append(f"Upstream source: {source_lineage.get('upstream_source_path')}")
    lines.append("")
    lines.append("Upstream preprocessing detected:")
    for key, value in summary["upstream_preprocessing"].items():
        lines.append(f"  - {key}: {value}")
    lines.append("")
    lines.append("Feature pool:")
    for key, value in summary["feature_pool"].items():
        lines.append(f"  - {key}: {value}")
    lines.append("")
    lines.append("Target readiness:")
    for target_name, result in summary["targets"].items():
        lines.append(
            f"  - {target_name}: readiness={result['readiness_score']}/100 "
            f"({result['readiness_grade']}), usable_rows={result['usable_rows']}, "
            f"features={result['selected_features']}, positives={result.get('positive_count')}"
        )
    return "\n".join(lines)


def format_target_report(
    payload: Dict[str, Any],
    importance_df: pd.DataFrame,
    top_n_features: int,
) -> str:
    lines = []
    lines.append(f"TARGET REPORT: {payload['target_name']}")
    lines.append("=" * 80)
    lines.append(f"Target column: {payload['target_column']}")
    lines.append(f"Direction: {payload['direction']}")
    lines.append(f"Label kind: {payload['label_kind']}")
    lines.append(f"Binary target: {payload['is_binary']}")
    timezone_contract = payload.get("timezone_contract") or {}
    if timezone_contract:
        lines.append(
            "Timezone contract: "
            f"source={timezone_contract.get('source_timezone')} "
            f"canonical={timezone_contract.get('canonical_timezone')} "
            f"feature_clock={timezone_contract.get('feature_clock_timezone')}"
        )
    lines.append("")
    lines.append("Row counts:")
    for key, value in payload["row_counts"].items():
        lines.append(f"  - {key}: {value}")
    lines.append("")
    lines.append("Split counts:")
    for key, value in payload["split_counts"].items():
        date_range = payload["split_date_ranges"].get(key)
        if date_range:
            lines.append(f"  - {key}: {value} ({date_range['start']} -> {date_range['end']})")
        else:
            lines.append(f"  - {key}: {value}")
    lines.append("")
    lines.append("Class balance:")
    for key, value in payload["class_balance"].items():
        lines.append(f"  - {key}: {value}")
    lines.append("")
    lines.append("Cleaning:")
    lines.append(f"  - columns filled: {payload['fill_report']['columns_filled']}")
    lines.append(f"  - low variance removed: {payload['low_variance']['removed_count']}")
    lines.append(f"  - collinear removed: {len(payload['collinearity']['dropped_columns'])}")
    lines.append(f"  - max remaining correlation: {payload['collinearity']['max_remaining_correlation']:.4f}")
    lines.append("")
    lines.append("Readiness:")
    lines.append(f"  - score: {payload['readiness']['score']}")
    lines.append(f"  - grade: {payload['readiness']['grade']}")
    if payload["readiness"]["open_issues"]:
        lines.append("  - open issues:")
        for issue in payload["readiness"]["open_issues"]:
            lines.append(f"      * {issue}")
    lines.append("")
    lines.append("Validations:")
    for item in payload["validations"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"  - [{status}] {item['check']}: {item['detail']}")
    lines.append("")
    lines.append("Top features:")
    if importance_df.empty:
        lines.append("  - no importance scores available")
    else:
        for _, row in importance_df.head(top_n_features).iterrows():
            lines.append(
                f"  - {row['feature']}: composite={row['composite_score']:.4f}, "
                f"assoc={row['association']:.4f}, mi={row['mutual_information']:.4f}, "
                f"rf={row['rf_importance']:.4f}"
            )
    return "\n".join(lines)


__all__ = [
    "format_summary_report",
    "format_target_report",
    "save_json",
]
