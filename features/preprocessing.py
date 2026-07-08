from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from preprocessing.config import PreprocessingConfig
from preprocessing.feature_importance import association_scores, prepare_feature_importance_frame
from preprocessing.pipeline import FeaturePreprocessingPipeline as _BaseFeaturePreprocessingPipeline


def compute_feature_importance(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series,
    is_binary: bool,
    config: PreprocessingConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if X_train.empty:
        empty = pd.DataFrame(
            columns=[
                "feature",
                "association",
                "mutual_information",
                "rf_importance",
                "composite_score",
            ]
        )
        return empty, {
            "analysis_rows": 0,
            "max_association": 0.0,
            "max_mutual_information": 0.0,
            "rf_oob_score": None,
        }

    analysis_rows = min(len(X_train), max(1, int(config.max_analysis_rows)))
    X_analysis = X_train.iloc[-analysis_rows:]
    y_analysis = y_train.iloc[-analysis_rows:]
    weight_analysis = sample_weight.iloc[-analysis_rows:]

    association = association_scores(X_analysis, y_analysis)
    prepared_analysis, discrete_features = prepare_feature_importance_frame(X_analysis)

    mi_scores = pd.Series(0.0, index=X_analysis.columns, dtype=float)
    if y_analysis.nunique(dropna=True) >= 2 and len(X_analysis) > 3:
        try:
            neighbors = min(config.mutual_info_neighbors, max(1, len(X_analysis) - 1))
            if is_binary:
                mi_values = mutual_info_classif(
                    prepared_analysis,
                    y_analysis,
                    random_state=42,
                    n_neighbors=neighbors,
                    discrete_features=discrete_features,
                )
            else:
                mi_values = mutual_info_regression(
                    prepared_analysis,
                    y_analysis,
                    random_state=42,
                    n_neighbors=neighbors,
                    discrete_features=discrete_features,
                )
            mi_scores = pd.Series(mi_values, index=X_analysis.columns, dtype=float)
        except Exception:
            mi_scores = pd.Series(0.0, index=X_analysis.columns, dtype=float)

    rf_scores = pd.Series(0.0, index=X_analysis.columns, dtype=float)
    rf_oob_score: Optional[float] = None
    if y_analysis.nunique(dropna=True) >= 2 and len(X_analysis) >= max(
        25,
        config.rf_min_samples_leaf * 2,
    ):
        try:
            if is_binary:
                model = RandomForestClassifier(
                    n_estimators=config.rf_n_estimators,
                    max_depth=config.rf_max_depth,
                    min_samples_leaf=config.rf_min_samples_leaf,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                    oob_score=True,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=config.rf_n_estimators,
                    max_depth=config.rf_max_depth,
                    min_samples_leaf=config.rf_min_samples_leaf,
                    n_jobs=-1,
                    random_state=42,
                    oob_score=True,
                )

            model.fit(prepared_analysis, y_analysis, sample_weight=weight_analysis)
            rf_scores = pd.Series(
                model.feature_importances_,
                index=X_analysis.columns,
                dtype=float,
            )
            rf_oob_score = float(getattr(model, "oob_score_", np.nan))
            if np.isnan(rf_oob_score):
                rf_oob_score = None
        except Exception:
            rf_scores = pd.Series(0.0, index=X_analysis.columns, dtype=float)
            rf_oob_score = None

    importance_df = pd.DataFrame(
        {
            "feature": X_analysis.columns,
            "association": association.reindex(X_analysis.columns).fillna(0.0).values,
            "mutual_information": mi_scores.reindex(X_analysis.columns).fillna(0.0).values,
            "rf_importance": rf_scores.reindex(X_analysis.columns).fillna(0.0).values,
        }
    )

    if not importance_df.empty:
        ranks = []
        for column in ("association", "mutual_information", "rf_importance"):
            series = importance_df[column]
            if float(series.abs().sum()) == 0.0:
                ranks.append(pd.Series(0.0, index=importance_df.index))
            else:
                ranks.append(series.rank(method="average", pct=True))
        importance_df["composite_score"] = sum(ranks) / len(ranks)
        importance_df = importance_df.sort_values(
            ["composite_score", "association", "mutual_information", "rf_importance"],
            ascending=False,
        ).reset_index(drop=True)
    else:
        importance_df["composite_score"] = []

    summary = {
        "analysis_rows": int(analysis_rows),
        "max_association": float(importance_df["association"].abs().max())
        if not importance_df.empty
        else 0.0,
        "max_mutual_information": float(importance_df["mutual_information"].max())
        if not importance_df.empty
        else 0.0,
        "rf_oob_score": rf_oob_score,
    }
    return importance_df, summary


class FeaturePreprocessingPipeline(_BaseFeaturePreprocessingPipeline):
    def _compute_feature_importance(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: pd.Series,
        is_binary: bool,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return compute_feature_importance(
            X_train=X_train,
            y_train=y_train,
            sample_weight=sample_weight,
            is_binary=is_binary,
            config=self.config,
        )


__all__ = [
    "FeaturePreprocessingPipeline",
    "PreprocessingConfig",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "association_scores",
    "compute_feature_importance",
    "mutual_info_classif",
    "mutual_info_regression",
]
