from __future__ import annotations

import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import PreprocessingConfig, TargetDatasetSpec
from .feature_importance import compute_feature_importance
from .feature_selection import (
    apply_fill_values,
    build_fill_values,
    build_split_indices,
    build_target_context,
    detect_binary_target,
    discover_targets,
    encode_candidate_features,
    maybe_scale,
    optimize_loaded_frame,
    ordered_usable_positions,
    prune_collinear_features,
    remove_exact_duplicate_features,
    remove_global_constant_features,
    remove_low_variance_columns,
    resolve_source_row_idx,
    resolve_feature_columns,
    resolve_sample_weight,
)
from .reporting import format_summary_report, format_target_report, save_json


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
        from features.io import standardize_market_frame

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        resolved_metadata_path = (
            Path(metadata_path) if metadata_path is not None else input_path.with_suffix(".metadata.json")
        )

        metadata = self._load_metadata(resolved_metadata_path)
        active_config, runtime_tuning = self._resolve_runtime_config(metadata)
        read_plan = self._build_csv_read_plan(
            input_path=input_path,
            metadata=metadata,
            time_column=active_config.time_column,
            load_time_column=bool(active_config.load_time_column),
        )
        df = pd.read_csv(
            input_path,
            usecols=read_plan["usecols"],
            dtype=read_plan["dtypes"],
            parse_dates=read_plan["parse_dates"],
        )
        df = standardize_market_frame(df)
        source_row_idx = resolve_source_row_idx(df)
        df = optimize_loaded_frame(df)
        timezone_contract = self._resolve_timezone_contract(metadata)
        source_lineage = self._build_source_lineage(
            input_path=input_path,
            metadata_path=resolved_metadata_path,
            metadata=metadata,
        )

        upstream_info = self._detect_upstream_preprocessing(metadata)
        df, source_row_idx, row_window_info = self._apply_row_windowing(
            df,
            source_row_idx,
            upstream_info,
            config=active_config,
        )

        target_specs = discover_targets(df, active_config)
        if not target_specs:
            raise ValueError(
                "No supported target columns were found. "
                f"Looked for: {active_config.target_columns}"
            )

        metadata_feature_columns = [
            column
            for column in metadata.get("feature_columns", [])
            if column in df.columns
        ]
        feature_columns = resolve_feature_columns(
            df,
            metadata,
            active_config,
            target_specs=target_specs,
        )
        carry_through_feature_columns = [
            column for column in feature_columns if column not in set(metadata_feature_columns)
        ]
        encoded_features, encoding_info = encode_candidate_features(df, feature_columns)
        encoded_features, constant_info = remove_global_constant_features(encoded_features)
        encoded_features, duplicate_info = remove_exact_duplicate_features(
            encoded_features,
            active_config.max_analysis_rows,
        )

        target_context = build_target_context(df, target_specs, active_config.time_column)
        target_context["source_row_idx"] = source_row_idx.to_numpy(copy=False)
        del df
        del source_row_idx
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
                timezone_contract=timezone_contract,
                source_lineage=source_lineage,
                config=active_config,
            )

        summary = {
            "input_file": str(input_path),
            "metadata_file": str(resolved_metadata_path),
            "config": active_config.to_dict(),
            "upstream_preprocessing": upstream_info,
            "row_windowing": row_window_info,
            "runtime_bar_interval_tuning": runtime_tuning,
            "feature_pool": {
                "metadata_feature_count": len(metadata_feature_columns),
                "carry_through_feature_count": len(carry_through_feature_columns),
                "carry_through_feature_columns": carry_through_feature_columns,
                "resolved_feature_count": len(feature_columns),
                "encoded_feature_count": int(encoded_features.shape[1]),
                "encoded_categorical_columns": encoding_info["encoded_columns"],
                "duplicate_columns_removed": duplicate_info["removed_count"],
                "constant_columns_removed": constant_info["removed_count"],
            },
            "timezone_contract": timezone_contract,
            "source_lineage": source_lineage,
            "targets": target_results,
        }

        save_json(output_dir / "summary.json", summary)
        (output_dir / "summary_report.txt").write_text(
            format_summary_report(summary),
            encoding="utf-8",
        )
        (output_dir / "encoders.json").write_text(
            json.dumps(encoding_info["encoders"], indent=2),
            encoding="utf-8",
        )

        return summary

    def _load_metadata(self, metadata_path: Path) -> Dict[str, Any]:
        if not metadata_path.exists():
            return {}

        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_csv_read_plan(
        self,
        *,
        input_path: Path,
        metadata: Dict[str, Any],
        time_column: str,
        load_time_column: bool,
    ) -> Dict[str, Any]:
        header = pd.read_csv(input_path, nrows=0)
        available_columns = [str(column) for column in header.columns]
        prepared_contract = metadata.get("prepared_contract")
        if not isinstance(prepared_contract, dict):
            prepared_contract = metadata.get("phase02_contract", {})
        if not isinstance(prepared_contract, dict):
            prepared_contract = {}

        declared_feature_columns = [
            str(column)
            for column in metadata.get("feature_columns", [])
            if str(column) in available_columns
        ]
        target_context_columns = [
            str(column)
            for column in prepared_contract.get("target_context_columns", [])
            if str(column) in available_columns
        ]

        helper_prefixes = (
            "label_",
            "label_quality_",
            "sample_weight_",
            "exclude_",
            "neg_ok_",
            "concurrency_",
            "htf_confluence_",
        )
        requested_columns = {
            "warmup_mask",
            *declared_feature_columns,
            *target_context_columns,
            *self.config.target_columns,
        }
        if load_time_column:
            requested_columns.add(str(time_column))
        requested_columns.update(
            column
            for column in available_columns
            if any(column.startswith(prefix) for prefix in helper_prefixes)
        )
        usecols = [column for column in available_columns if column in requested_columns]
        if not declared_feature_columns and not target_context_columns:
            usecols = available_columns

        sample = pd.read_csv(input_path, nrows=64, usecols=usecols)
        dtype_map: Dict[str, Any] = {}
        numeric_feature_candidates = set(declared_feature_columns) | set(target_context_columns)
        for column in usecols:
            if column == time_column:
                dtype_map[column] = "string"
                continue
            if column not in sample.columns:
                continue
            sample_series = sample[column]
            if column.startswith("label_quality_") or column.startswith("sample_weight_"):
                dtype_map[column] = np.float32
                continue
            if column.startswith("concurrency_"):
                dtype_map[column] = np.int16
                continue
            if column.startswith("exclude_") or column.startswith("neg_ok_") or column == "warmup_mask":
                dtype_map[column] = np.int8
                continue
            if column.startswith("label_"):
                dtype_map[column] = np.int8
                continue
            if column in numeric_feature_candidates and pd.api.types.is_numeric_dtype(sample_series):
                dtype_map[column] = np.float32

        return {
            "usecols": usecols,
            "dtypes": dtype_map,
            "parse_dates": None,
        }

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

    def _resolve_runtime_config(
        self,
        metadata: Dict[str, Any],
    ) -> tuple[PreprocessingConfig, Dict[str, Any]]:
        input_bar_minutes_raw = metadata.get("input_bar_minutes", metadata.get("base_bar_minutes"))
        try:
            input_bar_minutes = float(input_bar_minutes_raw)
        except (TypeError, ValueError):
            input_bar_minutes = float(self.config.bar_interval_reference_minutes)

        reference_minutes = float(self.config.bar_interval_reference_minutes)
        input_timeframe = (
            metadata.get("input_timeframe")
            if isinstance(metadata.get("input_timeframe"), str)
            else f"{int(round(input_bar_minutes))}m"
        )
        info: Dict[str, Any] = {
            "input_bar_minutes": input_bar_minutes,
            "input_timeframe": input_timeframe,
            "reference_bar_minutes": reference_minutes,
            "applied": False,
            "updated_fields": [],
        }

        if not self.config.auto_retune_for_bar_interval:
            return self.config, info
        if input_bar_minutes <= 0 or input_bar_minutes >= reference_minutes:
            return replace(self.config, auto_retune_for_bar_interval=False), info

        default_config = PreprocessingConfig()
        updates: Dict[str, Any] = {}
        if self.config.max_analysis_rows == default_config.max_analysis_rows:
            scale = reference_minutes / max(input_bar_minutes, 1e-6)
            updates["max_analysis_rows"] = int(round(self.config.max_analysis_rows * scale))

        active_config = replace(
            self.config,
            **updates,
            auto_retune_for_bar_interval=False,
        )
        if updates:
            info["applied"] = True
            info["updated_fields"] = sorted(updates.keys())
        return active_config, info

    def _resolve_timezone_contract(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        timezone_contract = metadata.get("timezone_contract", {})
        return timezone_contract if isinstance(timezone_contract, dict) else {}

    def _build_source_lineage(
        self,
        *,
        input_path: Path,
        metadata_path: Path,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "feature_csv": str(input_path),
            "feature_metadata_file": str(metadata_path),
            "feature_builder_source_path": metadata.get("source_path"),
            "feature_builder_source_metadata_file": metadata.get("source_metadata_file"),
            "upstream_source_path": metadata.get("upstream_source_path"),
            "upstream_metadata_file": metadata.get("upstream_metadata_file"),
            "upstream_bar_timestamp_semantics": metadata.get("upstream_bar_timestamp_semantics"),
            "upstream_timezone_contract": metadata.get("upstream_timezone_contract"),
        }

    def _apply_row_windowing(
        self,
        df: pd.DataFrame,
        source_row_idx: pd.Series,
        upstream_info: Dict[str, Any],
        *,
        config: PreprocessingConfig,
    ) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        info = {
            "rows_before": int(len(df)),
            "additional_skip_rows_requested": int(config.additional_skip_rows),
            "additional_skip_rows_applied": 0,
            "analysis_row_cap": int(config.max_analysis_rows),
        }

        skip_rows = int(config.additional_skip_rows)
        if skip_rows > 0:
            warmup_already_done = upstream_info.get("warmup_rows_already_dropped", False)
            if warmup_already_done and config.respect_upstream_warmup:
                info["additional_skip_rows_reason"] = (
                    "skipped_because_upstream_builder_already_removed_warmup_rows"
                )
            else:
                df = df.iloc[skip_rows:].reset_index(drop=True)
                source_row_idx = source_row_idx.iloc[skip_rows:].reset_index(drop=True)
                info["additional_skip_rows_applied"] = skip_rows

        info["rows_after"] = int(len(df))
        return df, source_row_idx, info

    def _combine_numeric_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> pd.Series | None:
        available = [column for column in columns if column in df.columns]
        if not available:
            return None

        frame = pd.concat(
            [pd.to_numeric(df[column], errors="coerce").rename(column) for column in available],
            axis=1,
        )
        return frame.max(axis=1, skipna=True)

    def _combine_boolean_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        *,
        mode: str,
    ) -> pd.Series | None:
        available = [column for column in columns if column in df.columns]
        if not available:
            return None

        frame = pd.concat(
            [df[column].fillna(False).astype(bool).rename(column) for column in available],
            axis=1,
        )
        if mode == "all":
            return frame.all(axis=1)
        if mode == "any":
            return frame.any(axis=1)
        raise ValueError(f"Unsupported boolean combine mode: {mode}")

    def _resolve_target_inputs(
        self,
        df: pd.DataFrame,
        spec: TargetDatasetSpec,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series | None]:
        if spec.is_synthetic:
            target_raw = self._combine_numeric_columns(df, spec.component_target_columns)
            if target_raw is None:
                raise KeyError(
                    f"Synthetic target {spec.name!r} is missing component columns: "
                    f"{spec.component_target_columns}"
                )
            sample_weight_source = self._combine_numeric_columns(
                df,
                spec.component_sample_weight_columns,
            )
            positive_mask = target_raw.fillna(0).astype(float) > 0
            component_negative_masks: list[pd.Series] = []
            for index, target_column in enumerate(spec.component_target_columns):
                component_positive = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(float) > 0
                safe_negative_column = (
                    spec.component_safe_negative_columns[index]
                    if index < len(spec.component_safe_negative_columns)
                    else None
                )
                exclude_column = (
                    spec.component_exclude_columns[index]
                    if index < len(spec.component_exclude_columns)
                    else None
                )
                safe_negative = (
                    df[safe_negative_column].fillna(False).astype(bool)
                    if safe_negative_column and safe_negative_column in df.columns
                    else pd.Series(False, index=df.index)
                )
                exclude_mask = (
                    df[exclude_column].fillna(False).astype(bool)
                    if exclude_column and exclude_column in df.columns
                    else pd.Series(False, index=df.index)
                )
                component_negative_masks.append((~component_positive) & safe_negative & ~exclude_mask)

            negative_mask = (~positive_mask) & (
                pd.concat(component_negative_masks, axis=1).any(axis=1)
                if component_negative_masks
                else pd.Series(False, index=df.index)
            )
        else:
            target_raw = pd.to_numeric(df[spec.target_column], errors="coerce")
            sample_weight_source = (
                df[spec.sample_weight_column]
                if spec.sample_weight_column and spec.sample_weight_column in df.columns
                else None
            )
            safe_negative_mask = (
                df[spec.safe_negative_column].fillna(False).astype(bool)
                if spec.safe_negative_column and spec.safe_negative_column in df.columns
                else None
            )
            exclude_mask = (
                df[spec.exclude_column].fillna(False).astype(bool)
                if spec.exclude_column and spec.exclude_column in df.columns
                else None
            )
            positive_mask = target_raw.fillna(0).astype(float) > 0
            negative_mask = ~positive_mask
            if safe_negative_mask is not None:
                negative_mask &= safe_negative_mask
            if exclude_mask is not None:
                negative_mask &= ~exclude_mask
        return target_raw, positive_mask, negative_mask, sample_weight_source

    def _target_construction_report(self, spec: TargetDatasetSpec) -> Dict[str, Any]:
        if not spec.is_synthetic:
            return {
                "type": "direct_column",
                "source_target_column": spec.target_column,
            }

        return {
            "type": "synthetic_any_positive_union",
            "source_target_columns": spec.component_target_columns,
            "component_sample_weight_columns": spec.component_sample_weight_columns,
            "component_quality_columns": spec.component_quality_columns,
            "component_exclude_columns": spec.component_exclude_columns,
            "component_safe_negative_columns": spec.component_safe_negative_columns,
            "positive_label_rule": "rowwise_any_component_positive",
            "negative_label_rule": "not_positive AND any_component_safe_negative_without_component_exclude",
            "sample_weight_rule": "rowwise_max_component_sample_weight",
        }

    def _prepare_target_dataset(
        self,
        *,
        df: pd.DataFrame,
        encoded_features: pd.DataFrame,
        spec: TargetDatasetSpec,
        output_dir: Path,
        timezone_contract: Dict[str, Any],
        source_lineage: Dict[str, Any],
        config: PreprocessingConfig,
    ) -> Dict[str, Any]:
        target_raw, positive_mask, negative_mask, sample_weight_source = self._resolve_target_inputs(
            df,
            spec,
        )
        warmup_mask = (
            df["warmup_mask"].fillna(False).astype(bool)
            if "warmup_mask" in df.columns
            else pd.Series(False, index=df.index)
        )

        usable_mask = ~warmup_mask & (positive_mask | negative_mask)
        ordered_positions, ordered_time_values = ordered_usable_positions(
            df,
            usable_mask,
            config.time_column,
        )
        target_series = target_raw.iloc[ordered_positions].fillna(0)

        is_binary = detect_binary_target(target_series)
        if is_binary:
            target_series = target_series.astype(int)

        sample_weight = resolve_sample_weight(
            sample_weight_source=sample_weight_source,
            positive_mask=positive_mask,
        ).iloc[ordered_positions]
        source_row_idx = df["source_row_idx"].iloc[ordered_positions].reset_index(drop=True)

        usable_rows = int(len(ordered_positions))
        applied_embargo_bars = self._resolve_target_split_embargo_bars(config, spec.name)
        split_indices = build_split_indices(
            usable_rows,
            config,
            source_row_idx=source_row_idx,
            target_name=spec.name,
        )
        train_idx = split_indices["train"]
        val_idx = split_indices["val"]
        test_idx = split_indices["test"]
        train_positions = ordered_positions[train_idx]
        val_positions = ordered_positions[val_idx]
        test_positions = ordered_positions[test_idx]

        def take_feature_rows(row_positions: np.ndarray, columns: Optional[List[str]] = None) -> pd.DataFrame:
            row_index = encoded_features.index.take(row_positions)
            if columns is None:
                return encoded_features.loc[row_index]
            return encoded_features.loc[row_index, columns]

        X_train_raw = take_feature_rows(train_positions)
        y_train = target_series.iloc[train_idx].reset_index(drop=True)
        y_val = target_series.iloc[val_idx].reset_index(drop=True)
        y_test = target_series.iloc[test_idx].reset_index(drop=True)
        w_train = sample_weight.iloc[train_idx].reset_index(drop=True)
        w_val = sample_weight.iloc[val_idx].reset_index(drop=True)
        w_test = sample_weight.iloc[test_idx].reset_index(drop=True)
        source_row_idx_train = source_row_idx.iloc[train_idx].reset_index(drop=True)
        source_row_idx_val = source_row_idx.iloc[val_idx].reset_index(drop=True)
        source_row_idx_test = source_row_idx.iloc[test_idx].reset_index(drop=True)

        fill_values, fill_report = build_fill_values(X_train_raw)
        X_train = apply_fill_values(X_train_raw, fill_values)
        del X_train_raw

        X_train, low_variance_info = remove_low_variance_columns(
            X_train,
            config.variance_threshold,
        )

        collinearity_info = prune_collinear_features(
            X_train=X_train,
            y_train=y_train,
            config=config,
        )
        selected_features = [
            column
            for column in X_train.columns
            if column not in collinearity_info["dropped_columns"]
        ]
        X_train = X_train.loc[:, selected_features]
        selected_fill_values = {
            column: fill_values[column]
            for column in selected_features
            if column in fill_values
        }

        X_val = apply_fill_values(
            take_feature_rows(val_positions, selected_features),
            selected_fill_values,
        )
        X_test = apply_fill_values(
            take_feature_rows(test_positions, selected_features),
            selected_fill_values,
        )

        scaler, X_train_scaled, X_val_scaled, X_test_scaled = maybe_scale(
            X_train,
            X_val,
            X_test,
            config.scaler_type,
        )
        importance_df, importance_summary = compute_feature_importance(
            X_train=X_train,
            y_train=y_train,
            sample_weight=w_train,
            is_binary=is_binary,
            config=config,
        )

        split_date_ranges = self._split_date_ranges(ordered_time_values, split_indices)
        split_geometry = self._split_geometry_report(
            source_row_idx=source_row_idx,
            split_indices=split_indices,
            applied_embargo_bars=applied_embargo_bars,
        )
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
            config=config,
        )
        readiness = self._readiness_report(
            usable_rows=usable_rows,
            positive_count=int((target_series > 0).sum()) if is_binary else None,
            validations=validations,
            class_balance=class_balance,
            importance_summary=importance_summary,
            is_binary=is_binary,
            config=config,
        )

        self._save_split_csv(output_dir / "train.csv", X_train_scaled, y_train, w_train, source_row_idx_train)
        self._save_split_csv(output_dir / "val.csv", X_val_scaled, y_val, w_val, source_row_idx_val)
        self._save_split_csv(output_dir / "test.csv", X_test_scaled, y_test, w_test, source_row_idx_test)
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        save_json(
            output_dir / "features.json",
            {"features": selected_features, "n_features": len(selected_features)},
        )

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
            "split_geometry": split_geometry,
            "timezone_contract": timezone_contract,
            "source_lineage": source_lineage,
            "row_identity": {
                "source_row_idx_column": "source_row_idx",
                "source_row_idx_origin": "standardized_input_frame",
            },
            "target_construction": self._target_construction_report(spec),
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

        save_json(output_dir / "report.json", report_payload)
        (output_dir / "report.txt").write_text(
            format_target_report(
                report_payload,
                importance_df,
                config.top_n_features,
            ),
            encoding="utf-8",
        )

        if scaler is not None and config.save_scaler:
            joblib.dump(scaler, output_dir / "scaler.joblib")

        del X_train
        del X_val
        del X_test
        del X_train_scaled
        del X_val_scaled
        del X_test_scaled
        gc.collect()

        return {
            "target_column": spec.target_column,
            "usable_rows": usable_rows,
            "selected_features": int(len(selected_features)),
            "readiness_score": readiness["score"],
            "readiness_grade": readiness["grade"],
            "positive_rate": class_balance.get("positive_rate"),
            "positive_count": class_balance.get("positive_count"),
        }

    def _resolve_target_split_embargo_bars(
        self,
        config: PreprocessingConfig,
        target_name: str,
    ) -> int:
        target_map = config.target_split_embargo_bars or {}
        if target_name in target_map:
            try:
                return max(int(target_map[target_name]), 0)
            except (TypeError, ValueError):
                return 0
        try:
            return max(int(config.split_embargo_bars), 0)
        except (TypeError, ValueError):
            return 0

    def _split_geometry_report(
        self,
        *,
        source_row_idx: pd.Series,
        split_indices: Dict[str, np.ndarray],
        applied_embargo_bars: int,
    ) -> Dict[str, Any]:
        source_rows = pd.to_numeric(source_row_idx, errors="coerce").reset_index(drop=True)
        split_ranges: Dict[str, Optional[Dict[str, int]]] = {}
        boundaries: List[Dict[str, Any]] = []

        for split_name, indices in split_indices.items():
            if len(indices) == 0:
                split_ranges[split_name] = None
                continue
            values = source_rows.iloc[indices]
            split_ranges[split_name] = {
                "start_source_row_idx": int(values.min()),
                "end_source_row_idx": int(values.max()),
                "row_count": int(len(indices)),
            }

        for current_name, next_name in (("train", "val"), ("val", "test")):
            current_idx = split_indices[current_name]
            next_idx = split_indices[next_name]
            if len(current_idx) == 0 or len(next_idx) == 0:
                boundaries.append(
                    {
                        "boundary_name": f"{current_name}_to_{next_name}",
                        "available": False,
                    }
                )
                continue

            current_last_pos = int(current_idx[-1])
            next_first_pos = int(next_idx[0])
            current_last_source = int(source_rows.iloc[current_last_pos])
            next_first_source = int(source_rows.iloc[next_first_pos])
            raw_gap = int(next_first_source - current_last_source - 1)
            dropped_usable_rows = int(max(next_first_pos - current_last_pos - 1, 0))
            boundaries.append(
                {
                    "boundary_name": f"{current_name}_to_{next_name}",
                    "available": True,
                    "current_split_last_position": current_last_pos,
                    "next_split_first_position": next_first_pos,
                    "current_split_last_source_row_idx": current_last_source,
                    "next_split_first_source_row_idx": next_first_source,
                    "raw_source_row_gap_bars": raw_gap,
                    "dropped_usable_rows_between_splits": dropped_usable_rows,
                    "required_embargo_bars": int(applied_embargo_bars),
                    "passes_embargo": bool(raw_gap >= int(applied_embargo_bars)),
                }
            )

        return {
            "applied_embargo_bars": int(applied_embargo_bars),
            "split_source_row_ranges": split_ranges,
            "boundaries": boundaries,
        }

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
        weighted_positive_rate = (
            float(sample_weight_all[y_all == 1].sum() / sample_weight_all.sum())
            if total_count and sample_weight_all.sum() > 0
            else 0.0
        )
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
        config: PreprocessingConfig,
    ) -> List[Dict[str, Any]]:
        validations: List[Dict[str, Any]] = []

        def add(name: str, passed: bool, detail: str) -> None:
            validations.append({"check": name, "passed": bool(passed), "detail": detail})

        add(
            "usable_rows",
            usable_rows >= config.min_usable_rows,
            f"usable_rows={usable_rows}, threshold={config.min_usable_rows}",
        )
        add(
            "selected_features",
            len(selected_features) > 0,
            f"selected_features={len(selected_features)}",
        )
        add(
            "train_rows",
            len(y_train) >= config.min_train_rows,
            f"train_rows={len(y_train)}, threshold={config.min_train_rows}",
        )
        add("val_rows", len(y_val) > 0, f"val_rows={len(y_val)}")
        add("test_rows", len(y_test) > 0, f"test_rows={len(y_test)}")
        add(
            "max_remaining_correlation",
            max_remaining_corr < config.correlation_threshold,
            f"max_remaining_corr={max_remaining_corr:.4f}, threshold={config.correlation_threshold:.4f}",
        )

        if is_binary:
            add(
                "train_has_both_classes",
                y_train.nunique(dropna=True) >= 2,
                f"train_unique_classes={int(y_train.nunique(dropna=True))}",
            )
            add(
                "enough_positive_samples_train",
                int(y_train.sum()) >= config.min_positive_samples,
                f"train_positive_count={int(y_train.sum())}, threshold={config.min_positive_samples}",
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
        config: PreprocessingConfig,
    ) -> Dict[str, Any]:
        passed_ratio = (
            sum(1 for item in validations if item["passed"]) / len(validations)
            if validations
            else 0.0
        )
        score = 35.0 * passed_ratio
        score += 25.0 * min(usable_rows / max(config.min_usable_rows, 1), 1.0)

        if is_binary:
            positives = float(positive_count or 0)
            score += 25.0 * min(positives / max(config.min_positive_samples, 1), 1.0)
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
        source_row_idx: pd.Series,
    ) -> None:
        # Keep a stable row identity in prepared splits so downstream prediction
        # exports can be joined back to the source feature frame without putting
        # timestamps into the model-training matrices themselves.
        frame = features.reset_index(drop=True)
        frame.insert(0, "source_row_idx", source_row_idx.reset_index(drop=True).astype(np.int64))
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


__all__ = ["FeaturePreprocessingPipeline"]
