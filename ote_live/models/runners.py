from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.prediction import ModelPrediction
from ote_live.models.calibrators import apply_probability_calibrator
from ote_live.models.loaders import LoadedRuntimeModel


class RuntimeModelRunner:
    def __init__(self, loaded_model: LoadedRuntimeModel) -> None:
        self.loaded_model = loaded_model

    def build_latest_snapshot(
        self,
        feature_frame: pd.DataFrame,
        *,
        valid_feature_count: int | None = None,
    ) -> FeatureSnapshot:
        ordered = self._coerce_ordered_feature_frame(feature_frame)
        latest_values = ordered.iloc[-1]
        resolved_valid_feature_count = (
            int(valid_feature_count)
            if valid_feature_count is not None
            else int(latest_values.notna().sum())
        )
        return FeatureSnapshot(
            asset=self.loaded_model.manifest.asset,
            timeframe=self.loaded_model.manifest.timeframe,
            direction=self.loaded_model.manifest.direction,
            timestamp=self._extract_latest_timestamp(feature_frame),
            source_row_idx=self._extract_latest_source_row_idx(feature_frame),
            feature_values={
                feature_name: _to_python_scalar(value)
                for feature_name, value in latest_values.items()
            },
            valid_feature_count=resolved_valid_feature_count,
        )

    def predict_latest(
        self,
        feature_frame: pd.DataFrame,
        *,
        timestamp: datetime | None = None,
        source_row_idx: int | None = None,
        regime: str | None = None,
    ) -> ModelPrediction:
        ordered = self._coerce_ordered_feature_frame(feature_frame)
        raw_probability = self._predict_latest_raw_probability(ordered)
        calibrated_probability = float(
            apply_probability_calibrator(
                self.loaded_model.calibrator,
                np.asarray([raw_probability], dtype=np.float32),
            )[0]
        )

        resolved_timestamp = timestamp or self._extract_latest_timestamp(feature_frame)
        resolved_source_row_idx = (
            source_row_idx
            if source_row_idx is not None
            else self._extract_latest_source_row_idx(feature_frame)
        )

        return ModelPrediction(
            model_id=self.loaded_model.model_id,
            direction=self.loaded_model.manifest.direction,
            backend=self.loaded_model.manifest.backend,
            timestamp=resolved_timestamp,
            source_row_idx=resolved_source_row_idx,
            regime=regime,
            raw_score=float(raw_probability),
            calibrated_probability=calibrated_probability,
            threshold_applied=None,
            threshold_source="unknown",
        )

    def predict_from_snapshot_history(
        self,
        snapshots: Sequence[FeatureSnapshot],
        *,
        regime: str | None = None,
    ) -> ModelPrediction:
        if not snapshots:
            raise ValueError("At least one feature snapshot is required for live prediction.")

        latest = snapshots[-1]
        frame = pd.DataFrame(
            [{feature: snapshot.feature_values.get(feature) for feature in self.loaded_model.selected_feature_names} for snapshot in snapshots]
        )
        return self.predict_latest(
            frame,
            timestamp=latest.timestamp,
            source_row_idx=latest.source_row_idx,
            regime=regime,
        )

    def _predict_latest_raw_probability(self, ordered_feature_frame: pd.DataFrame) -> float:
        matrix = ordered_feature_frame.to_numpy(dtype=np.float32, copy=True)
        scaled = self._scale_matrix(matrix)

        if self.loaded_model.backend == "xgboost":
            lag_steps = self.loaded_model.manifest.feature_manifest.lag_steps or [0]
            lagged = _build_sparse_lag_view(
                scaled,
                lag_steps=lag_steps,
                delta_feature_count=self.loaded_model.manifest.feature_manifest.delta_feature_count,
            )
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise ImportError(
                    f"Model {self.loaded_model.model_id} requires the 'xgboost' package for live inference."
                ) from exc

            dmatrix = xgb.DMatrix(lagged[-1:])
            logits = np.asarray(self.loaded_model.model.predict(dmatrix), dtype=np.float32)
            return float(_sigmoid(logits)[0])

        if self.loaded_model.backend in {"tcn", "lstm"}:
            sequences = _build_sequence_windows(
                scaled,
                window_size=self.loaded_model.window_size,
            )
            try:
                from model_training.ote_training.torch_trainer import predict_torch_model
            except ImportError as exc:
                raise ImportError(
                    f"Model {self.loaded_model.model_id} requires the 'torch' package for live inference."
                ) from exc

            probabilities = predict_torch_model(
                model=self.loaded_model.model,
                X=sequences[-1:],
                batch_size=self.loaded_model.batch_size,
                use_amp=self.loaded_model.use_amp,
            )
            return float(np.asarray(probabilities, dtype=np.float32)[0])

        raise ValueError(f"Unsupported runtime backend {self.loaded_model.backend!r}.")

    def _coerce_ordered_feature_frame(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        feature_names = list(self.loaded_model.selected_feature_names)
        missing = [feature_name for feature_name in feature_names if feature_name not in feature_frame.columns]
        if missing:
            raise ValueError(
                "Feature frame is missing required live model features: "
                + ", ".join(missing[:20])
            )

        ordered = feature_frame.loc[:, feature_names].copy()
        ordered = ordered.apply(pd.to_numeric, errors="coerce")
        if ordered.isna().any().any():
            raise ValueError(f"Feature frame for {self.loaded_model.model_id} contains NaN values after coercion.")

        minimum_rows = self.loaded_model.context_rows + 1
        if len(ordered) < minimum_rows:
            raise ValueError(
                f"Model {self.loaded_model.model_id} requires at least {minimum_rows} feature rows, "
                f"but received {len(ordered)}."
            )
        return ordered

    def _scale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if self.loaded_model.scaler is None:
            scaled = np.asarray(matrix, dtype=np.float32)
        else:
            scaled = self.loaded_model.scaler.transform(matrix).astype(np.float32, copy=False)

        clip_value = float(self.loaded_model.scale_clip)
        if clip_value > 0:
            np.clip(scaled, -clip_value, clip_value, out=scaled)
        return np.ascontiguousarray(scaled, dtype=np.float32)

    @staticmethod
    def _extract_latest_timestamp(feature_frame: pd.DataFrame) -> datetime:
        for column in ("timestamp", "datetime"):
            if column in feature_frame.columns:
                series = pd.to_datetime(feature_frame[column], errors="coerce")
                if series.notna().any():
                    return series.iloc[-1].to_pydatetime()

        if isinstance(feature_frame.index, pd.DatetimeIndex) and len(feature_frame.index) > 0:
            return feature_frame.index[-1].to_pydatetime()

        raise ValueError("Feature frame does not contain a usable timestamp or datetime column.")

    @staticmethod
    def _extract_latest_source_row_idx(feature_frame: pd.DataFrame) -> int | None:
        if "source_row_idx" in feature_frame.columns:
            series = pd.to_numeric(feature_frame["source_row_idx"], errors="coerce")
            if series.notna().any():
                return int(series.iloc[-1])

        if isinstance(feature_frame.index, pd.RangeIndex):
            return int(feature_frame.index[-1])
        return int(len(feature_frame) - 1)


def _build_sequence_windows(
    matrix: np.ndarray,
    *,
    window_size: int,
) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")
    if matrix.shape[0] < window_size:
        raise ValueError(
            f"Need at least {window_size} rows to build live sequence windows, got {matrix.shape[0]}."
        )

    windows = np.lib.stride_tricks.sliding_window_view(matrix, window_shape=window_size, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    return np.ascontiguousarray(windows, dtype=np.float32)


def _build_sparse_lag_view(
    matrix: np.ndarray,
    *,
    lag_steps: Sequence[int],
    delta_feature_count: int = 0,
) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")

    max_lag = max(int(lag) for lag in lag_steps)
    if matrix.shape[0] <= max_lag:
        raise ValueError(
            f"Need more than {max_lag} rows to build live lag features, got {matrix.shape[0]}."
        )

    rows = matrix.shape[0] - max_lag
    parts = [matrix[max_lag - int(lag) : max_lag - int(lag) + rows] for lag in lag_steps]

    if delta_feature_count > 0 and len(lag_steps) > 1:
        delta_count = min(int(delta_feature_count), matrix.shape[1])
        current = matrix[max_lag : max_lag + rows, :delta_count]
        for lag in lag_steps[1:]:
            lagged = matrix[max_lag - int(lag) : max_lag - int(lag) + rows, :delta_count]
            parts.append(current - lagged)

    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _to_python_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value
