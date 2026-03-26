from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from .io import standardize_market_frame


def _default_target_columns() -> List[str]:
    return [
        "label_long_entry",
        "label_short_entry",
        "label_long_ote",
        "label_short_ote",
    ]


@dataclass
class PreprocessingConfig:
    """Configuration for target-aware feature preprocessing."""

    target_columns: List[str] = field(default_factory=_default_target_columns)
    time_column: str = "datetime"
    max_analysis_rows: int = 10_000
    additional_skip_rows: int = 0
    respect_upstream_warmup: bool = True

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    use_time_based_split: bool = True

    similarity_threshold: float = 0.995
    correlation_threshold: float = 0.98
    variance_threshold: float = 1e-9

    min_usable_rows: int = 250
    min_train_rows: int = 100
    min_positive_samples: int = 25
    top_n_features: int = 50

    scaler_type: str = "none"  # one of: none, robust, standard

    rf_n_estimators: int = 250
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 8
    mutual_info_neighbors: int = 5

    include_base_price_columns: bool = False
    save_scaler: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetDatasetSpec:
    """Metadata about one label target and its helper columns."""

    name: str
    target_column: str
    direction: str
    label_kind: str
    sample_weight_column: Optional[str] = None
    quality_column: Optional[str] = None
    exclude_column: Optional[str] = None
    safe_negative_column: Optional[str] = None


class FeaturePreprocessingPipeline:
    """Prepare generated feature datasets for target-specific model training."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        metadata_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_path)
        df = standardize_market_frame(df)
        df = self._optimize_loaded_frame(df)
        metadata = self._load_metadata(input_path, metadata_path)

        upstream_info = self._detect_upstream_preprocessing(metadata)
        df, row_window_info = self._apply_row_windowing(df, upstream_info)

        feature_columns = self._resolve_feature_columns(df, metadata)
        encoded_features, encoding_info = self._encode_candidate_features(df, feature_columns)
        encoded_features, constant_info = self._remove_global_constant_features(encoded_features)
        encoded_features, duplicate_info = self._remove_exact_duplicate_features(encoded_features)

        target_specs = self._discover_targets(df)
        if not target_specs:
            raise ValueError(
                "No supported target columns were found. "
                f"Looked for: {self.config.target_columns}"
            )

        target_context = self._build_target_context(df, target_specs)
        del df
        gc.collect()

        target_results: Dict[str, Any] = {}
        for spec in target_specs:
            target_output_dir = output_dir / spec.name
            target_output_dir.mkdir(parents=True, exist_ok=True)
            target_results[spec.name] = self._prepare_target_dataset(
                df=target_context,
                encoded_features=encoded_features,
                spec=spec,
                output_dir=target_output_dir,
            )

        summary = {
            "input_file": str(input_path),
            "metadata_file": str(metadata_path) if metadata_path is not None else str(input_path.with_suffix(".metadata.json")),
            "config": self.config.to_dict(),
            "upstream_preprocessing": upstream_info,
            "row_windowing": row_window_info,
            "feature_pool": {
                "metadata_feature_count": len(feature_columns),
                "encoded_feature_count": int(encoded_features.shape[1]),
                "encoded_categorical_columns": encoding_info["encoded_columns"],
                "duplicate_columns_removed": duplicate_info["removed_count"],
                "constant_columns_removed": constant_info["removed_count"],
            },
            "targets": target_results,
        }

        self._save_json(output_dir / "summary.json", summary)
        (output_dir / "summary_report.txt").write_text(
            self._format_summary_report(summary),
            encoding="utf-8",
        )
        (output_dir / "encoders.json").write_text(
            json.dumps(encoding_info["encoders"], indent=2),
            encoding="utf-8",
        )

        return summary

    def _load_metadata(
        self,
        input_path: Path,
        metadata_path: str | Path | None,
    ) -> Dict[str, Any]:
        if metadata_path is None:
            metadata_path = input_path.with_suffix(".metadata.json")

        path = Path(metadata_path)
        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _detect_upstream_preprocessing(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        config = metadata.get("config", {})
        feature_columns = metadata.get("feature_columns", [])
        return {
            "metadata_detected": bool(metadata),
            "feature_columns_from_metadata": len(feature_columns),
            "warmup_rows_already_dropped": bool(config.get("drop_warmup_rows")),
            "warmup_rows_configured": int(config.get("warmup_rows", 0)),
            "numeric_fill_applied_upstream": bool(config.get("fillna_numeric")),
            "duplicate_column_names_blocked_in_builder": True,
        }

    def _apply_row_windowing(
        self,
        df: pd.DataFrame,
        upstream_info: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        info = {
            "rows_before": int(len(df)),
            "additional_skip_rows_requested": int(self.config.additional_skip_rows),
            "additional_skip_rows_applied": 0,
            "analysis_row_cap": int(self.config.max_analysis_rows),
        }

        skip_rows = int(self.config.additional_skip_rows)
        if skip_rows > 0:
            warmup_already_done = upstream_info.get("warmup_rows_already_dropped", False)
            if warmup_already_done and self.config.respect_upstream_warmup:
                info["additional_skip_rows_reason"] = "skipped_because_upstream_builder_already_removed_warmup_rows"
            else:
                df = df.iloc[skip_rows:].reset_index(drop=True)
                info["additional_skip_rows_applied"] = skip_rows

        info["rows_after"] = int(len(df))
        return df, info

    def _resolve_feature_columns(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> List[str]:
        metadata_features = [
            column
            for column in metadata.get("feature_columns", [])
            if column in df.columns
        ]
        if metadata_features:
            features = metadata_features
        else:
            features = self._fallback_feature_columns(df)

        if self.config.include_base_price_columns:
            for column in ("open", "high", "low", "close", "volume"):
                if column in df.columns and column not in features:
                    features.append(column)

        return list(dict.fromkeys(features))

    def _fallback_feature_columns(self, df: pd.DataFrame) -> List[str]:
        excluded_prefixes = (
            "label_",
            "sample_weight",
            "entry_quality_",
            "label_quality_",
            "exclude_",
            "neg_ok_",
            "htf_confluence_",
            "bars_since_1h_swing_",
            "bars_since_30m_swing_",
            "concurrency_",
        )
        excluded_columns = {
            "datetime",
            "date",
            "time",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "warmup_mask",
            "is_anomaly",
            "atr",
            "structural_atr",
        }
        return [
            column
            for column in df.columns
            if column not in excluded_columns and not column.startswith(excluded_prefixes)
        ]

    def _encode_candidate_features(
        self,
        df: pd.DataFrame,
        feature_columns: Iterable[str],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        encoded_parts: List[pd.Series] = []
        encoders: Dict[str, Dict[str, int]] = {}
        encoded_columns: List[str] = []

        for column in feature_columns:
            if column not in df.columns:
                continue

            series = df[column]
            if pd.api.types.is_bool_dtype(series):
                encoded_parts.append(series.fillna(False).astype(np.int8).rename(column))
                continue

            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                encoded_parts.append(self._downcast_numeric_series(numeric).rename(column))
                continue

            cleaned = series.fillna("__missing__").astype(str).str.strip()
            if cleaned.nunique(dropna=False) <= 1:
                encoded_parts.append(pd.Series(0, index=df.index, dtype=np.int8, name=column))
                continue

            encoder = LabelEncoder()
            encoded = pd.Series(encoder.fit_transform(cleaned), index=df.index, name=column)
            encoded_parts.append(self._downcast_numeric_series(encoded))
            encoders[column] = {
                value: int(code)
                for code, value in enumerate(encoder.classes_.tolist())
            }
            encoded_columns.append(column)

        encoded = pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=df.index)
        return encoded, {
            "encoders": encoders,
            "encoded_columns": encoded_columns,
        }

    def _remove_exact_duplicate_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df.empty or df.shape[1] < 2:
            return df, {
                "probe_rows": int(len(df)),
                "removed_count": 0,
                "removed_columns": [],
            }

        probe_positions = self._analysis_probe_positions(len(df))
        probe_frame = df.iloc[probe_positions]

        signatures: Dict[str, List[str]] = {}
        keep: List[str] = []
        removed: List[Dict[str, str]] = []

        for column in df.columns:
            signature = self._column_signature(probe_frame[column])
            candidates = signatures.setdefault(signature, [])
            original = next((candidate for candidate in candidates if df[column].equals(df[candidate])), None)
            if original is not None:
                removed.append({"dropped": column, "kept": original})
                continue

            candidates.append(column)
            keep.append(column)

        return df.loc[:, keep], {
            "probe_rows": int(len(probe_positions)),
            "removed_count": len(removed),
            "removed_columns": removed,
        }

    def _remove_global_constant_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        nunique = df.nunique(dropna=False)
        constant_columns = nunique[nunique <= 1].index.tolist()
        kept = [column for column in df.columns if column not in constant_columns]
        return df.loc[:, kept], {
            "removed_count": len(constant_columns),
            "removed_columns": constant_columns,
        }

    def _build_target_context(
        self,
        df: pd.DataFrame,
        target_specs: List[TargetDatasetSpec],
    ) -> pd.DataFrame:
        required_columns: List[str] = []

        for column in (self.config.time_column, "warmup_mask"):
            if column in df.columns:
                required_columns.append(column)

        for spec in target_specs:
            for column in (
                spec.target_column,
                spec.sample_weight_column,
                spec.quality_column,
                spec.exclude_column,
                spec.safe_negative_column,
            ):
                if column and column in df.columns:
                    required_columns.append(column)

        return df.loc[:, list(dict.fromkeys(required_columns))].copy()

    def _discover_targets(self, df: pd.DataFrame) -> List[TargetDatasetSpec]:
        specs: List[TargetDatasetSpec] = []
        for target_column in self.config.target_columns:
            if target_column not in df.columns:
                continue

            direction = "long" if "_long_" in target_column else "short"
            label_kind = "entry" if target_column.endswith("_entry") else "ote"

            sample_weight_column = f"sample_weight_{direction}"
            quality_column = f"label_quality_{direction}"
            if label_kind == "entry":
                sample_weight_column = f"sample_weight_entry_{direction}"
                quality_column = f"entry_quality_{direction}"

            specs.append(
                TargetDatasetSpec(
                    name=f"{direction}_{label_kind}",
                    target_column=target_column,
                    direction=direction,
                    label_kind=label_kind,
                    sample_weight_column=sample_weight_column if sample_weight_column in df.columns else None,
                    quality_column=quality_column if quality_column in df.columns else None,
                    exclude_column=f"exclude_{direction}" if f"exclude_{direction}" in df.columns else None,
                    safe_negative_column=f"neg_ok_{direction}" if f"neg_ok_{direction}" in df.columns else None,
                )
            )

        return specs

    def _prepare_target_dataset(
        self,
        *,
        df: pd.DataFrame,
        encoded_features: pd.DataFrame,
        spec: TargetDatasetSpec,
        output_dir: Path,
    ) -> Dict[str, Any]:
        target_raw = pd.to_numeric(df[spec.target_column], errors="coerce")
        positive_mask = target_raw.fillna(0).astype(float) > 0
        warmup_mask = (
            df["warmup_mask"].fillna(False).astype(bool)
            if "warmup_mask" in df.columns
            else pd.Series(False, index=df.index)
        )

        negative_mask = ~positive_mask
        if spec.safe_negative_column:
            negative_mask &= df[spec.safe_negative_column].fillna(False).astype(bool)
        if spec.exclude_column:
            negative_mask &= ~df[spec.exclude_column].fillna(False).astype(bool)

        usable_mask = ~warmup_mask & (positive_mask | negative_mask)
        ordered_positions, ordered_time_values = self._ordered_usable_positions(df, usable_mask)
        target_series = target_raw.iloc[ordered_positions].fillna(0)

        is_binary = self._detect_binary_target(target_series)
        if is_binary:
            target_series = target_series.astype(int)

        sample_weight = self._resolve_sample_weight(
            sample_weight_source=df[spec.sample_weight_column]
            if spec.sample_weight_column and spec.sample_weight_column in df.columns
            else None,
            positive_mask=positive_mask,
        ).iloc[ordered_positions]

        usable_rows = int(len(ordered_positions))
        split_indices = self._build_split_indices(usable_rows)
        train_idx = split_indices["train"]
        val_idx = split_indices["val"]
        test_idx = split_indices["test"]

        X_train_raw = encoded_features.iloc[ordered_positions[train_idx]]
        X_val_raw = encoded_features.iloc[ordered_positions[val_idx]]
        X_test_raw = encoded_features.iloc[ordered_positions[test_idx]]
        y_train = target_series.iloc[train_idx].reset_index(drop=True)
        y_val = target_series.iloc[val_idx].reset_index(drop=True)
        y_test = target_series.iloc[test_idx].reset_index(drop=True)
        w_train = sample_weight.iloc[train_idx].reset_index(drop=True)
        w_val = sample_weight.iloc[val_idx].reset_index(drop=True)
        w_test = sample_weight.iloc[test_idx].reset_index(drop=True)

        fill_values, fill_report = self._build_fill_values(X_train_raw)
        X_train = self._apply_fill_values(X_train_raw, fill_values)
        X_val = self._apply_fill_values(X_val_raw, fill_values)
        X_test = self._apply_fill_values(X_test_raw, fill_values)

        X_train, low_variance_info = self._remove_low_variance_columns(X_train)
        X_val = X_val.loc[:, X_train.columns]
        X_test = X_test.loc[:, X_train.columns]

        collinearity_info = self._prune_collinear_features(
            X_train=X_train,
            y_train=y_train,
            sample_weight=w_train,
            is_binary=is_binary,
        )
        selected_features = [column for column in X_train.columns if column not in collinearity_info["dropped_columns"]]
        X_train = X_train.loc[:, selected_features]
        X_val = X_val.loc[:, selected_features]
        X_test = X_test.loc[:, selected_features]

        scaler, X_train_scaled, X_val_scaled, X_test_scaled = self._maybe_scale(X_train, X_val, X_test)
        importance_df, importance_summary = self._compute_feature_importance(
            X_train=X_train,
            y_train=y_train,
            sample_weight=w_train,
            is_binary=is_binary,
        )

        split_date_ranges = self._split_date_ranges(ordered_time_values, split_indices)
        class_balance = self._class_balance_report(
            y_all=target_series.reset_index(drop=True),
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            sample_weight_all=sample_weight.reset_index(drop=True),
            is_binary=is_binary,
        )
        validations = self._validate_target_dataset(
            usable_rows=usable_rows,
            selected_features=selected_features,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            importance_summary=importance_summary,
            max_remaining_corr=collinearity_info["max_remaining_correlation"],
            is_binary=is_binary,
        )
        readiness = self._readiness_report(
            usable_rows=usable_rows,
            positive_count=int((target_series > 0).sum()) if is_binary else None,
            validations=validations,
            class_balance=class_balance,
            importance_summary=importance_summary,
            is_binary=is_binary,
        )

        self._save_split_csv(output_dir / "train.csv", X_train_scaled, y_train, w_train)
        self._save_split_csv(output_dir / "val.csv", X_val_scaled, y_val, w_val)
        self._save_split_csv(output_dir / "test.csv", X_test_scaled, y_test, w_test)
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        self._save_json(output_dir / "features.json", {"features": selected_features, "n_features": len(selected_features)})

        report_payload = {
            "target_column": spec.target_column,
            "target_name": spec.name,
            "direction": spec.direction,
            "label_kind": spec.label_kind,
            "is_binary": is_binary,
            "row_counts": {
                "rows_usable": usable_rows,
                "rows_positive": int(positive_mask.iloc[ordered_positions].sum()),
                "rows_negative": int((~positive_mask.iloc[ordered_positions]).sum()),
                "rows_excluded_by_warmup": int(warmup_mask.sum()),
                "rows_excluded_as_ambiguous": int((~warmup_mask & ~(positive_mask | negative_mask)).sum()),
            },
            "split_counts": {
                "train": int(len(y_train)),
                "val": int(len(y_val)),
                "test": int(len(y_test)),
            },
            "split_date_ranges": split_date_ranges,
            "sample_weight_column": spec.sample_weight_column,
            "quality_column": spec.quality_column,
            "fill_report": fill_report,
            "low_variance": low_variance_info,
            "collinearity": collinearity_info,
            "class_balance": class_balance,
            "importance_summary": importance_summary,
            "validations": validations,
            "readiness": readiness,
        }

        self._save_json(output_dir / "report.json", report_payload)
        (output_dir / "report.txt").write_text(
            self._format_target_report(report_payload, importance_df),
            encoding="utf-8",
        )

        if scaler is not None and self.config.save_scaler:
            joblib.dump(scaler, output_dir / "scaler.joblib")

        return {
            "target_column": spec.target_column,
            "usable_rows": usable_rows,
            "selected_features": int(len(selected_features)),
            "readiness_score": readiness["score"],
            "readiness_grade": readiness["grade"],
            "positive_rate": class_balance.get("positive_rate"),
            "positive_count": class_balance.get("positive_count"),
        }

    def _resolve_sample_weight(
        self,
        sample_weight_source: Optional[pd.Series],
        positive_mask: pd.Series,
    ) -> pd.Series:
        if sample_weight_source is not None:
            weights = pd.to_numeric(sample_weight_source, errors="coerce").fillna(1.0)
        else:
            weights = pd.Series(1.0, index=positive_mask.index)

        weights = weights.clip(lower=0.0)
        weights = weights.where(weights > 0, 1.0)
        weights.loc[~positive_mask] = weights.loc[~positive_mask].replace(0.0, 1.0)
        return weights.astype(np.float32)

    def _ordered_usable_positions(
        self,
        df: pd.DataFrame,
        usable_mask: pd.Series,
    ) -> Tuple[np.ndarray, Optional[pd.Series]]:
        usable_positions = np.flatnonzero(usable_mask.to_numpy())
        if self.config.time_column not in df.columns or len(usable_positions) == 0:
            return usable_positions, None

        time_values = pd.to_datetime(
            df[self.config.time_column].iloc[usable_positions],
            errors="coerce",
        ).reset_index(drop=True)
        if time_values.is_monotonic_increasing:
            return usable_positions, time_values
        sort_order = time_values.sort_values(kind="stable").index.to_numpy(dtype=int, copy=False)
        ordered_positions = usable_positions[sort_order]
        ordered_time_values = time_values.iloc[sort_order].reset_index(drop=True)
        return ordered_positions, ordered_time_values

    def _analysis_probe_positions(self, n_rows: int) -> np.ndarray:
        if n_rows <= 0:
            return np.array([], dtype=int)

        probe_rows = min(n_rows, max(1, int(self.config.max_analysis_rows)))
        if probe_rows == n_rows:
            return np.arange(n_rows, dtype=int)

        return np.unique(np.linspace(0, n_rows - 1, num=probe_rows, dtype=int))

    def _build_split_indices(self, n_rows: int) -> Dict[str, np.ndarray]:
        if n_rows == 0:
            empty = np.array([], dtype=int)
            return {"train": empty, "val": empty, "test": empty}

        train_end = int(n_rows * self.config.train_size)
        val_end = int(n_rows * (self.config.train_size + self.config.val_size))

        if n_rows >= 3:
            train_end = min(max(train_end, 1), n_rows - 2)
            val_end = min(max(val_end, train_end + 1), n_rows - 1)
        else:
            train_end = n_rows
            val_end = n_rows

        if not self.config.use_time_based_split:
            order = np.random.default_rng(42).permutation(n_rows)
        else:
            order = np.arange(n_rows)

        return {
            "train": order[:train_end],
            "val": order[train_end:val_end],
            "test": order[val_end:n_rows],
        }

    def _build_fill_values(self, df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if df.empty:
            return {}, {"columns_filled": 0, "strategies": {}}

        clean = df.replace([np.inf, -np.inf], np.nan)
        missing_mask = clean.isna()
        has_missing = missing_mask.any(axis=0)
        all_missing = missing_mask.all(axis=0)

        means = clean.mean(axis=0)
        medians = clean.median(axis=0)
        valid_counts = clean.notna().sum(axis=0)
        skews = clean.skew(axis=0).where(valid_counts > 2, other=0.0).fillna(0.0)

        fill_series = means.astype(np.float64, copy=True)
        fill_series.loc[all_missing] = 0.0

        use_median = has_missing & ~all_missing & (skews.abs() > 1.0)
        fill_series.loc[use_median] = medians.loc[use_median]
        fill_series = fill_series.fillna(0.0).astype(np.float32)

        strategies: Dict[str, str] = {}
        for column in clean.columns[has_missing]:
            if bool(all_missing.loc[column]):
                strategies[column] = "zero_all_missing"
            elif bool(use_median.loc[column]):
                strategies[column] = "median"
            else:
                strategies[column] = "mean"

        return (
            {column: float(value) for column, value in fill_series.items()},
            {
                "columns_filled": int(has_missing.sum()),
                "strategies": strategies,
            },
        )

    def _apply_fill_values(
        self,
        df: pd.DataFrame,
        fill_values: Dict[str, float],
    ) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        clean = df.replace([np.inf, -np.inf], np.nan)
        fill_series = (
            pd.Series(fill_values, dtype=np.float32)
            .reindex(clean.columns)
            .fillna(np.float32(0.0))
        )
        return clean.fillna(fill_series).astype(np.float32, copy=False)

    def _remove_low_variance_columns(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        variances = df.var()
        low_variance = variances[variances <= self.config.variance_threshold].index.tolist()
        kept = [column for column in df.columns if column not in low_variance]
        return df.loc[:, kept], {
            "removed_count": len(low_variance),
            "removed_columns": low_variance,
        }

    def _prune_collinear_features(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: pd.Series,
        is_binary: bool,
    ) -> Dict[str, Any]:
        if X_train.empty or X_train.shape[1] < 2:
            return {
                "dropped_columns": [],
                "dropped_pairs": [],
                "similarity_pairs": [],
                "max_remaining_correlation": 0.0,
            }

        analysis_rows = min(len(X_train), max(1, int(self.config.max_analysis_rows)))
        X_analysis = X_train.iloc[-analysis_rows:].copy()
        y_analysis = y_train.iloc[-analysis_rows:].copy()

        association_scores = self._association_scores(X_analysis, y_analysis)
        composite_scores = association_scores.fillna(0.0).to_dict()

        corr = X_analysis.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        pairs: List[Tuple[str, str, float]] = []
        similarity_pairs: List[Dict[str, Any]] = []
        for column in upper.columns:
            series = upper[column].dropna()
            hits = series[series >= self.config.correlation_threshold]
            for other, corr_value in hits.items():
                value = float(corr_value)
                pairs.append((other, column, value))
                if value >= self.config.similarity_threshold:
                    similarity_pairs.append({"feature_a": other, "feature_b": column, "correlation": value})

        pairs.sort(key=lambda item: item[2], reverse=True)

        dropped: set[str] = set()
        drop_records: List[Dict[str, Any]] = []
        for left, right, corr_value in pairs:
            if left in dropped or right in dropped:
                continue

            left_score = float(composite_scores.get(left, 0.0))
            right_score = float(composite_scores.get(right, 0.0))

            if left_score > right_score:
                drop_feature, keep_feature = right, left
            elif right_score > left_score:
                drop_feature, keep_feature = left, right
            else:
                drop_feature, keep_feature = sorted([left, right])[-1], sorted([left, right])[0]

            dropped.add(drop_feature)
            drop_records.append(
                {
                    "dropped": drop_feature,
                    "kept": keep_feature,
                    "correlation": float(corr_value),
                    "dropped_score": float(composite_scores.get(drop_feature, 0.0)),
                    "kept_score": float(composite_scores.get(keep_feature, 0.0)),
                }
            )

        remaining = [column for column in X_train.columns if column not in dropped]
        if len(remaining) >= 2:
            remaining_corr = X_analysis[remaining].corr().abs()
            remaining_upper = remaining_corr.where(np.triu(np.ones(remaining_corr.shape), k=1).astype(bool))
            max_remaining = float(remaining_upper.max().max()) if not remaining_upper.empty else 0.0
            if np.isnan(max_remaining):
                max_remaining = 0.0
        else:
            max_remaining = 0.0

        return {
            "analysis_rows": int(analysis_rows),
            "dropped_columns": sorted(dropped),
            "dropped_pairs": drop_records[:200],
            "similarity_pairs": similarity_pairs[:100],
            "max_remaining_correlation": max_remaining,
        }

    def _maybe_scale(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scaler = None
        scaler_type = self.config.scaler_type.lower()

        if scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "standard":
            scaler = StandardScaler()
        else:
            return None, X_train, X_val, X_test

        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        return scaler, X_train_scaled, X_val_scaled, X_test_scaled

    def _compute_feature_importance(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: pd.Series,
        is_binary: bool,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if X_train.empty:
            empty = pd.DataFrame(columns=["feature", "association", "mutual_information", "rf_importance", "composite_score"])
            return empty, {
                "analysis_rows": 0,
                "max_association": 0.0,
                "max_mutual_information": 0.0,
                "rf_oob_score": None,
            }

        analysis_rows = min(len(X_train), max(1, int(self.config.max_analysis_rows)))
        X_analysis = X_train.iloc[-analysis_rows:]
        y_analysis = y_train.iloc[-analysis_rows:]
        weight_analysis = sample_weight.iloc[-analysis_rows:]

        association = self._association_scores(X_analysis, y_analysis)

        mi_scores = pd.Series(0.0, index=X_analysis.columns, dtype=float)
        if y_analysis.nunique(dropna=True) >= 2 and len(X_analysis) > 3:
            try:
                neighbors = min(self.config.mutual_info_neighbors, max(1, len(X_analysis) - 1))
                if is_binary:
                    mi_values = mutual_info_classif(X_analysis, y_analysis, random_state=42, n_neighbors=neighbors)
                else:
                    mi_values = mutual_info_regression(X_analysis, y_analysis, random_state=42, n_neighbors=neighbors)
                mi_scores = pd.Series(mi_values, index=X_analysis.columns, dtype=float)
            except Exception:
                mi_scores = pd.Series(0.0, index=X_analysis.columns, dtype=float)

        rf_scores = pd.Series(0.0, index=X_analysis.columns, dtype=float)
        rf_oob_score: Optional[float] = None
        if y_analysis.nunique(dropna=True) >= 2 and len(X_analysis) >= max(25, self.config.rf_min_samples_leaf * 2):
            try:
                if is_binary:
                    model = RandomForestClassifier(
                        n_estimators=self.config.rf_n_estimators,
                        max_depth=self.config.rf_max_depth,
                        min_samples_leaf=self.config.rf_min_samples_leaf,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                        oob_score=True,
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=self.config.rf_n_estimators,
                        max_depth=self.config.rf_max_depth,
                        min_samples_leaf=self.config.rf_min_samples_leaf,
                        n_jobs=-1,
                        random_state=42,
                        oob_score=True,
                    )

                model.fit(X_analysis, y_analysis, sample_weight=weight_analysis)
                rf_scores = pd.Series(model.feature_importances_, index=X_analysis.columns, dtype=float)
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
            "max_association": float(importance_df["association"].abs().max()) if not importance_df.empty else 0.0,
            "max_mutual_information": float(importance_df["mutual_information"].max()) if not importance_df.empty else 0.0,
            "rf_oob_score": rf_oob_score,
        }
        return importance_df, summary

    def _class_balance_report(
        self,
        *,
        y_all: pd.Series,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        sample_weight_all: pd.Series,
        is_binary: bool,
    ) -> Dict[str, Any]:
        if not is_binary:
            return {
                "target_type": "regression",
                "target_mean": float(y_all.mean()) if len(y_all) else 0.0,
                "target_std": float(y_all.std()) if len(y_all) else 0.0,
            }

        positive_count = int(y_all.sum())
        total_count = int(len(y_all))
        negative_count = int(total_count - positive_count)
        positive_rate = float(positive_count / total_count) if total_count else 0.0
        weighted_positive_rate = float(
            sample_weight_all[y_all == 1].sum() / sample_weight_all.sum()
        ) if total_count and sample_weight_all.sum() > 0 else 0.0
        suggested_pos_weight = float(negative_count / positive_count) if positive_count else None

        return {
            "target_type": "classification",
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_rate,
            "weighted_positive_rate": weighted_positive_rate,
            "suggested_positive_class_weight": suggested_pos_weight,
            "split_positive_rates": {
                "train": float(y_train.mean()) if len(y_train) else 0.0,
                "val": float(y_val.mean()) if len(y_val) else 0.0,
                "test": float(y_test.mean()) if len(y_test) else 0.0,
            },
        }

    def _validate_target_dataset(
        self,
        *,
        usable_rows: int,
        selected_features: List[str],
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        importance_summary: Dict[str, Any],
        max_remaining_corr: float,
        is_binary: bool,
    ) -> List[Dict[str, Any]]:
        validations: List[Dict[str, Any]] = []

        def add(name: str, passed: bool, detail: str) -> None:
            validations.append({"check": name, "passed": bool(passed), "detail": detail})

        add(
            "usable_rows",
            usable_rows >= self.config.min_usable_rows,
            f"usable_rows={usable_rows}, threshold={self.config.min_usable_rows}",
        )
        add(
            "selected_features",
            len(selected_features) > 0,
            f"selected_features={len(selected_features)}",
        )
        add(
            "train_rows",
            len(y_train) >= self.config.min_train_rows,
            f"train_rows={len(y_train)}, threshold={self.config.min_train_rows}",
        )
        add("val_rows", len(y_val) > 0, f"val_rows={len(y_val)}")
        add("test_rows", len(y_test) > 0, f"test_rows={len(y_test)}")
        add(
            "max_remaining_correlation",
            max_remaining_corr < self.config.correlation_threshold,
            f"max_remaining_corr={max_remaining_corr:.4f}, threshold={self.config.correlation_threshold:.4f}",
        )

        if is_binary:
            add(
                "train_has_both_classes",
                y_train.nunique(dropna=True) >= 2,
                f"train_unique_classes={int(y_train.nunique(dropna=True))}",
            )
            add(
                "enough_positive_samples_train",
                int(y_train.sum()) >= self.config.min_positive_samples,
                f"train_positive_count={int(y_train.sum())}, threshold={self.config.min_positive_samples}",
            )
        else:
            add(
                "train_target_variation",
                float(y_train.std()) > 0,
                f"train_target_std={float(y_train.std()):.6f}",
            )

        max_mi = float(importance_summary.get("max_mutual_information") or 0.0)
        add("feature_signal", max_mi > 0.0, f"max_mutual_information={max_mi:.6f}")

        return validations

    def _readiness_report(
        self,
        *,
        usable_rows: int,
        positive_count: Optional[int],
        validations: List[Dict[str, Any]],
        class_balance: Dict[str, Any],
        importance_summary: Dict[str, Any],
        is_binary: bool,
    ) -> Dict[str, Any]:
        passed_ratio = (
            sum(1 for item in validations if item["passed"]) / len(validations)
            if validations
            else 0.0
        )
        score = 35.0 * passed_ratio
        score += 25.0 * min(usable_rows / max(self.config.min_usable_rows, 1), 1.0)

        if is_binary:
            positives = float(positive_count or 0)
            score += 25.0 * min(positives / max(self.config.min_positive_samples, 1), 1.0)
            score += 15.0 * min((importance_summary.get("max_mutual_information") or 0.0) / 0.02, 1.0)
        else:
            score += 15.0 * min(abs(class_balance.get("target_std", 0.0)) / 0.10, 1.0)
            score += 25.0 * min((importance_summary.get("max_mutual_information") or 0.0) / 0.05, 1.0)

        failed_checks = {item["check"] for item in validations if not item["passed"]}
        if "train_has_both_classes" in failed_checks or "selected_features" in failed_checks:
            score = min(score, 39.0)
        elif {"usable_rows", "train_rows", "enough_positive_samples_train"} & failed_checks:
            score = min(score, 59.0)

        score = round(float(min(max(score, 0.0), 100.0)), 1)
        if score >= 80:
            grade = "green"
        elif score >= 60:
            grade = "yellow"
        else:
            grade = "red"

        reasons = [item["detail"] for item in validations if not item["passed"]][:6]
        return {"score": score, "grade": grade, "open_issues": reasons}

    def _save_split_csv(
        self,
        path: Path,
        features: pd.DataFrame,
        target: pd.Series,
        sample_weight: pd.Series,
    ) -> None:
        # Prepared split files are intentionally model-only tables.
        # We keep temporal ordering through the row sequence and record date ranges
        # in the companion reports, but exclude raw datetime columns from the CSVs
        # to prevent accidental timestamp leakage and to keep training inputs purely numeric.
        frame = features.reset_index(drop=True)
        frame["target"] = target.reset_index(drop=True)
        frame["sample_weight"] = sample_weight.reset_index(drop=True)
        frame.to_csv(path, index=False)

    def _split_date_ranges(
        self,
        time_values: Optional[pd.Series],
        split_indices: Dict[str, np.ndarray],
    ) -> Dict[str, Optional[Dict[str, str]]]:
        if time_values is None:
            return {"train": None, "val": None, "test": None}

        result: Dict[str, Optional[Dict[str, str]]] = {}
        for split_name, indices in split_indices.items():
            if len(indices) == 0:
                result[split_name] = None
                continue
            subset = pd.to_datetime(time_values.iloc[indices], errors="coerce").dropna()
            if subset.empty:
                result[split_name] = None
            else:
                result[split_name] = {
                    "start": subset.min().isoformat(),
                    "end": subset.max().isoformat(),
                }
        return result

    def _optimize_loaded_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            df[column] = self._downcast_numeric_series(df[column])
        return df

    @staticmethod
    def _downcast_numeric_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
            return series
        if pd.api.types.is_bool_dtype(series):
            return series
        if pd.api.types.is_integer_dtype(series):
            return pd.to_numeric(series, downcast="integer")
        if pd.api.types.is_float_dtype(series):
            return pd.to_numeric(series, downcast="float")
        return series

    def _detect_binary_target(self, target: pd.Series) -> bool:
        unique_values = set(pd.Series(target).dropna().unique().tolist())
        return unique_values.issubset({0, 1, 0.0, 1.0})

    def _association_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        if X.empty or y.nunique(dropna=True) < 2:
            return pd.Series(0.0, index=X.columns, dtype=float)

        with np.errstate(invalid="ignore", divide="ignore"):
            scores = X.corrwith(pd.Series(y).astype(float)).abs()
        return scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _column_signature(series: pd.Series) -> str:
        hashed = pd.util.hash_pandas_object(series, index=False).values.tobytes()
        return hashlib.blake2b(hashed, digest_size=16).hexdigest()

    @staticmethod
    def _save_json(path: Path, payload: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _format_summary_report(self, summary: Dict[str, Any]) -> str:
        lines = []
        lines.append("FEATURE PREPROCESSING SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Input file: {summary['input_file']}")
        lines.append(f"Metadata file: {summary['metadata_file']}")
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

    def _format_target_report(
        self,
        payload: Dict[str, Any],
        importance_df: pd.DataFrame,
    ) -> str:
        lines = []
        lines.append(f"TARGET REPORT: {payload['target_name']}")
        lines.append("=" * 80)
        lines.append(f"Target column: {payload['target_column']}")
        lines.append(f"Direction: {payload['direction']}")
        lines.append(f"Label kind: {payload['label_kind']}")
        lines.append(f"Binary target: {payload['is_binary']}")
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
            for _, row in importance_df.head(self.config.top_n_features).iterrows():
                lines.append(
                    f"  - {row['feature']}: composite={row['composite_score']:.4f}, "
                    f"assoc={row['association']:.4f}, mi={row['mutual_information']:.4f}, "
                    f"rf={row['rf_importance']:.4f}"
                )
        return "\n".join(lines)
