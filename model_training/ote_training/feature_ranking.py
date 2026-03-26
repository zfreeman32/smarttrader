from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_RANKING_FILE_CANDIDATES: tuple[str, ...] = (
    "feature_importance_merged.csv",
    "feature_importance.csv",
)


def load_feature_allowlist(prepared_dir: Path) -> List[str]:
    payload = json.loads((prepared_dir / "features.json").read_text(encoding="utf-8"))
    return list(payload.get("features", []))


def resolve_ranking_path(
    prepared_dir: Path,
    candidate_filenames: Sequence[str] = DEFAULT_RANKING_FILE_CANDIDATES,
) -> Path | None:
    for filename in candidate_filenames:
        candidate = prepared_dir / filename
        if candidate.exists():
            return candidate
    return None


def ranking_sort_column(columns: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in ("merged_score", "composite_score", "mean_abs_shap_all", "rf_importance", "mutual_information"):
        if candidate in available:
            return candidate
    return None


def _rank_to_unit_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    output = pd.Series(0.0, index=series.index, dtype=float)
    if valid.empty:
        return output
    if len(valid) == 1:
        output.loc[valid.index] = 1.0
        return output

    ranks = valid.rank(method="average", ascending=False)
    scores = 1.0 - ((ranks - 1.0) / float(len(valid) - 1))
    output.loc[valid.index] = scores.astype(float)
    return output


def merge_feature_rankings(
    base_ranking: pd.DataFrame,
    shap_ranking: pd.DataFrame,
    allowed_features: Sequence[str],
    *,
    base_weight: float = 0.50,
    shap_weight: float = 0.35,
    shap_positive_weight: float = 0.15,
) -> pd.DataFrame:
    if not allowed_features:
        raise ValueError("No allowed features were provided for ranking merge.")

    base = base_ranking.copy()
    shap = shap_ranking.copy()

    if "feature" not in base.columns:
        raise ValueError("Base ranking must include a 'feature' column.")
    if "feature" not in shap.columns:
        raise ValueError("SHAP ranking must include a 'feature' column.")

    base = base.drop_duplicates(subset=["feature"], keep="first").set_index("feature")
    shap = shap.drop_duplicates(subset=["feature"], keep="first").set_index("feature")

    feature_list = [
        feature
        for feature in allowed_features
        if feature in base.index or feature in shap.index
    ]
    if not feature_list:
        raise ValueError("No overlap was found between the allowed features and ranking inputs.")

    merged = pd.DataFrame(index=pd.Index(feature_list, name="feature"))
    merged = merged.join(base, how="left")
    merged = merged.join(
        shap[
            [
                column
                for column in (
                    "mean_abs_shap_all",
                    "mean_signed_shap_all",
                    "mean_feature_value_all",
                    "mean_abs_shap_positive",
                    "mean_signed_shap_positive",
                    "mean_feature_value_positive",
                    "mean_abs_shap_negative",
                    "mean_signed_shap_negative",
                    "mean_feature_value_negative",
                )
                if column in shap.columns
            ]
        ],
        how="left",
    )

    if "composite_score" in merged.columns:
        base_component = _rank_to_unit_score(merged["composite_score"])
    else:
        fallback_components = []
        for column in ("mutual_information", "rf_importance", "association"):
            if column not in merged.columns:
                continue
            values = merged[column].abs() if column == "association" else merged[column]
            fallback_components.append(_rank_to_unit_score(values))
        if fallback_components:
            base_component = sum(fallback_components) / float(len(fallback_components))
        else:
            base_component = pd.Series(1.0, index=merged.index, dtype=float)

    shap_component = (
        _rank_to_unit_score(merged["mean_abs_shap_all"])
        if "mean_abs_shap_all" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )
    shap_positive_component = (
        _rank_to_unit_score(merged["mean_abs_shap_positive"])
        if "mean_abs_shap_positive" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )

    active_weight_sum = base_weight
    if float(shap_component.max()) > 0.0:
        active_weight_sum += shap_weight
    else:
        shap_weight = 0.0
    if float(shap_positive_component.max()) > 0.0:
        active_weight_sum += shap_positive_weight
    else:
        shap_positive_weight = 0.0

    if active_weight_sum <= 0.0:
        raise ValueError("At least one ranking component must contribute positive weight.")

    merged["base_rank_score"] = base_component
    merged["shap_rank_score"] = shap_component
    merged["shap_positive_rank_score"] = shap_positive_component
    merged["merged_score"] = (
        (base_weight * merged["base_rank_score"])
        + (shap_weight * merged["shap_rank_score"])
        + (shap_positive_weight * merged["shap_positive_rank_score"])
    ) / active_weight_sum

    merged = merged.reset_index()
    sort_columns = ["merged_score"]
    ascending = [False]
    for column in ("mean_abs_shap_all", "composite_score", "rf_importance", "mutual_information"):
        if column in merged.columns:
            sort_columns.append(column)
            ascending.append(False)
    sort_columns.append("feature")
    ascending.append(True)
    merged = merged.sort_values(sort_columns, ascending=ascending, kind="stable").reset_index(drop=True)

    merged["merged_rank"] = np.arange(1, len(merged) + 1, dtype=int)
    return merged
