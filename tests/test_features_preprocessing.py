from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import features.preprocessing as preprocessing_module
from features.preprocessing import FeaturePreprocessingPipeline, PreprocessingConfig


def _write_dataset(base_dir: Path) -> Path:
    rows = 30
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="5min")
    long_target = [0] * rows
    short_target = [0] * rows
    for index in (5, 11, 18, 25):
        long_target[index] = 1
    for index in (7, 14, 21, 27):
        short_target[index] = 1

    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "open": [1.10 + i * 0.0001 for i in range(rows)],
            "high": [1.11 + i * 0.0001 for i in range(rows)],
            "low": [1.09 + i * 0.0001 for i in range(rows)],
            "close": [1.10 + i * 0.0001 for i in range(rows)],
            "volume": [100 + i for i in range(rows)],
            "label_long_entry": long_target,
            "label_short_entry": short_target,
            "exclude_long": [False] * rows,
            "exclude_short": [False] * rows,
            "neg_ok_long": [i not in {2, 3, 8} and long_target[i] == 0 for i in range(rows)],
            "neg_ok_short": [i not in {4, 9, 10} and short_target[i] == 0 for i in range(rows)],
            "sample_weight_entry_long": [2.0 if long_target[i] else 1.0 for i in range(rows)],
            "sample_weight_entry_short": [2.5 if short_target[i] else 1.0 for i in range(rows)],
            "warmup_mask": [i < 2 for i in range(rows)],
            "feature_signal": [float(i % 3 == 0) for i in range(rows)],
            "feature_signal_dup": [float(i % 3 == 0) for i in range(rows)],
            "feature_constant": [1.0] * rows,
            "feature_cat": ["a" if i % 2 == 0 else "b" for i in range(rows)],
        }
    )

    dataset_path = base_dir / "sample_features.csv"
    metadata_path = base_dir / "sample_features.metadata.json"
    df.to_csv(dataset_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": [
                    "feature_signal",
                    "feature_signal_dup",
                    "feature_constant",
                    "feature_cat",
                ],
                "config": {
                    "drop_warmup_rows": False,
                    "warmup_rows": 0,
                    "fillna_numeric": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_path


def test_preprocessing_removes_duplicate_and_constant_features(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    output_dir = tmp_path / "prepared"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_entry"],
            min_usable_rows=5,
            min_train_rows=3,
            min_positive_samples=1,
            top_n_features=5,
        )
    )
    summary = pipeline.run(dataset_path, output_dir)

    feature_file = output_dir / "long_entry" / "features.json"
    features_payload = json.loads(feature_file.read_text(encoding="utf-8"))
    selected = set(features_payload["features"])

    assert "feature_signal" in selected
    assert "feature_signal_dup" not in selected
    assert "feature_constant" not in selected
    assert "feature_cat" in selected
    assert summary["targets"]["long_entry"]["usable_rows"] > 0

    encoders = json.loads((output_dir / "encoders.json").read_text(encoding="utf-8"))
    assert "feature_cat" in encoders


def test_preprocessing_uses_safe_negatives_and_builds_multiple_targets(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    output_dir = tmp_path / "prepared_multi"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_entry", "label_short_entry"],
            min_usable_rows=5,
            min_train_rows=3,
            min_positive_samples=1,
        )
    )
    summary = pipeline.run(dataset_path, output_dir)

    assert set(summary["targets"].keys()) == {"long_entry", "short_entry"}

    long_dir = output_dir / "long_entry"
    short_dir = output_dir / "short_entry"
    assert long_dir.exists()
    assert short_dir.exists()

    long_total = sum(
        len(pd.read_csv(long_dir / filename))
        for filename in ("train.csv", "val.csv", "test.csv")
    )
    short_total = sum(
        len(pd.read_csv(short_dir / filename))
        for filename in ("train.csv", "val.csv", "test.csv")
    )

    long_report = json.loads((long_dir / "report.json").read_text(encoding="utf-8"))
    short_report = json.loads((short_dir / "report.json").read_text(encoding="utf-8"))

    assert long_total == long_report["row_counts"]["rows_usable"]
    assert short_total == short_report["row_counts"]["rows_usable"]
    assert long_report["row_counts"]["rows_excluded_as_ambiguous"] > 0
    assert short_report["row_counts"]["rows_excluded_as_ambiguous"] > 0


def test_preprocessing_writes_source_row_idx_to_each_split(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    output_dir = tmp_path / "prepared_source_rows"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_entry"],
            min_usable_rows=5,
            min_train_rows=3,
            min_positive_samples=1,
        )
    )
    pipeline.run(dataset_path, output_dir)

    long_dir = output_dir / "long_entry"
    splits = [pd.read_csv(long_dir / filename) for filename in ("train.csv", "val.csv", "test.csv")]

    for split in splits:
        assert "source_row_idx" in split.columns
        assert pd.api.types.is_integer_dtype(split["source_row_idx"])
        assert (split["source_row_idx"] >= 0).all()

    all_source_rows = pd.concat([split["source_row_idx"] for split in splits], ignore_index=True)
    assert all_source_rows.is_unique

    report = json.loads((long_dir / "report.json").read_text(encoding="utf-8"))
    assert report["row_identity"]["source_row_idx_column"] == "source_row_idx"


def test_feature_importance_respects_analysis_row_cap(monkeypatch) -> None:
    calls: dict[str, int] = {}

    def fake_mutual_info(X, y, random_state=None, n_neighbors=None):
        calls["mi_rows"] = len(X)
        return np.linspace(0.1, 0.3, X.shape[1], dtype=float)

    class DummyRandomForestClassifier:
        def __init__(self, *args, **kwargs) -> None:
            self.feature_importances_ = np.array([], dtype=float)
            self.oob_score_ = 0.42

        def fit(self, X, y, sample_weight=None):
            calls["rf_rows"] = len(X)
            calls["rf_weight_rows"] = len(sample_weight) if sample_weight is not None else 0
            self.feature_importances_ = np.linspace(0.9, 0.3, X.shape[1], dtype=float)

    monkeypatch.setattr(preprocessing_module, "mutual_info_classif", fake_mutual_info)
    monkeypatch.setattr(preprocessing_module, "RandomForestClassifier", DummyRandomForestClassifier)

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            max_analysis_rows=26,
            rf_min_samples_leaf=1,
        )
    )

    X_train = pd.DataFrame(
        {
            "feature_a": np.arange(40, dtype=float),
            "feature_b": np.arange(40, dtype=float) % 3,
            "feature_c": np.linspace(1.0, 2.0, 40, dtype=float),
        }
    )
    y_train = pd.Series(([0, 1] * 20), dtype=int)
    sample_weight = pd.Series(np.ones(40, dtype=np.float32))

    importance_df, summary = pipeline._compute_feature_importance(
        X_train=X_train,
        y_train=y_train,
        sample_weight=sample_weight,
        is_binary=True,
    )

    assert summary["analysis_rows"] == 26
    assert calls["mi_rows"] == 26
    assert calls["rf_rows"] == 26
    assert calls["rf_weight_rows"] == 26
    assert set(importance_df["feature"]) == {"feature_a", "feature_b", "feature_c"}
