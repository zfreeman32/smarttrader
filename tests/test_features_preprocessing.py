from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import features.preprocessing as preprocessing_module
from features.preprocessing import FeaturePreprocessingPipeline, PreprocessingConfig
from preprocessing.feature_selection import discover_targets


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
                "timezone_contract": {
                    "source_timezone": "UTC",
                    "canonical_timezone": "UTC",
                    "feature_clock_timezone": "America/New_York",
                    "market_close_timezone": "America/New_York",
                },
                "source_path": "data/labeling/labeled_data/sample_labels.csv",
                "source_metadata_file": "data/labeling/labeled_data/sample_labels.metadata.json",
                "upstream_source_path": "data/currency_data/sample_raw.csv",
                "upstream_metadata_file": "data/labeling/labeled_data/sample_labels.metadata.json",
                "upstream_bar_timestamp_semantics": "bar_open",
                "upstream_timezone_contract": {
                    "source_timezone": "GMT-6",
                    "canonical_timezone": "UTC",
                },
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


def _write_meta_zone_dataset(base_dir: Path) -> Path:
    rows = 36
    timestamps = pd.date_range("2024-02-01 00:00:00", periods=rows, freq="5min")

    long_reversal = [0] * rows
    long_continuation = [0] * rows
    long_breakout = [0] * rows
    short_reversal = [0] * rows
    short_continuation = [0] * rows
    short_breakout = [0] * rows

    for index in (5, 6, 18):
        long_reversal[index] = 1
    for index in (10, 18, 19):
        long_continuation[index] = 1
    for index in (14, 15):
        long_breakout[index] = 1

    for index in (8, 9):
        short_reversal[index] = 1
    for index in (20, 21):
        short_continuation[index] = 1
    for index in (26, 27):
        short_breakout[index] = 1

    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "open": [1.20 + i * 0.0001 for i in range(rows)],
            "high": [1.21 + i * 0.0001 for i in range(rows)],
            "low": [1.19 + i * 0.0001 for i in range(rows)],
            "close": [1.20 + i * 0.0001 for i in range(rows)],
            "volume": [200 + i for i in range(rows)],
            "label_long_reversal": long_reversal,
            "label_long_continuation_pullback": long_continuation,
            "label_long_breakout": long_breakout,
            "label_short_reversal": short_reversal,
            "label_short_continuation_pullback": short_continuation,
            "label_short_breakout": short_breakout,
            "sample_weight_long_reversal": [2.5 if value else 1.0 for value in long_reversal],
            "sample_weight_long_continuation": [1.8 if value else 1.0 for value in long_continuation],
            "sample_weight_long_breakout": [3.0 if value else 1.0 for value in long_breakout],
            "sample_weight_short_reversal": [2.2 if value else 1.0 for value in short_reversal],
            "sample_weight_short_continuation": [1.7 if value else 1.0 for value in short_continuation],
            "sample_weight_short_breakout": [2.8 if value else 1.0 for value in short_breakout],
            "label_quality_long_reversal": [0.8 if value else 0.0 for value in long_reversal],
            "label_quality_long_continuation": [0.7 if value else 0.0 for value in long_continuation],
            "label_quality_long_breakout": [0.9 if value else 0.0 for value in long_breakout],
            "label_quality_short_reversal": [0.75 if value else 0.0 for value in short_reversal],
            "label_quality_short_continuation": [0.65 if value else 0.0 for value in short_continuation],
            "label_quality_short_breakout": [0.85 if value else 0.0 for value in short_breakout],
            "exclude_long_reversal": [False] * rows,
            "exclude_long_continuation": [False] * rows,
            "exclude_long_breakout": [False] * rows,
            "exclude_short_reversal": [False] * rows,
            "exclude_short_continuation": [False] * rows,
            "exclude_short_breakout": [False] * rows,
            "neg_ok_long_reversal": [i not in {2, 3} and not long_reversal[i] for i in range(rows)],
            "neg_ok_long_continuation": [i not in {3, 11} and not long_continuation[i] for i in range(rows)],
            "neg_ok_long_breakout": [i not in {4, 16} and not long_breakout[i] for i in range(rows)],
            "neg_ok_short_reversal": [i not in {7, 11} and not short_reversal[i] for i in range(rows)],
            "neg_ok_short_continuation": [i not in {19, 22} and not short_continuation[i] for i in range(rows)],
            "neg_ok_short_breakout": [i not in {25, 28} and not short_breakout[i] for i in range(rows)],
            "warmup_mask": [i < 3 for i in range(rows)],
            "feature_trend": np.linspace(0.0, 1.0, rows),
            "feature_structure": np.sin(np.linspace(0.0, 2.0, rows)),
            "feature_binary": [float(i % 4 == 0) for i in range(rows)],
        }
    )

    dataset_path = base_dir / "meta_zone_features.csv"
    metadata_path = base_dir / "meta_zone_features.metadata.json"
    df.to_csv(dataset_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": [
                    "feature_trend",
                    "feature_structure",
                    "feature_binary",
                ],
                "timezone_contract": {
                    "source_timezone": "UTC",
                    "canonical_timezone": "UTC",
                    "feature_clock_timezone": "America/New_York",
                    "market_close_timezone": "America/New_York",
                },
                "source_path": "data/labeling/labeled_data/meta_zone_labels.csv",
                "source_metadata_file": "data/labeling/labeled_data/meta_zone_labels.metadata.json",
                "upstream_source_path": "data/currency_data/meta_zone_raw.csv",
                "upstream_metadata_file": "data/labeling/labeled_data/meta_zone_labels.metadata.json",
                "upstream_bar_timestamp_semantics": "bar_open",
                "upstream_timezone_contract": {
                    "source_timezone": "GMT-6",
                    "canonical_timezone": "UTC",
                },
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


def test_preprocessing_runtime_tuning_expands_analysis_cap_for_1m_metadata(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    metadata_path = dataset_path.with_suffix(".metadata.json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["input_bar_minutes"] = 1.0
    payload["input_timeframe"] = "1m"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_dir = tmp_path / "prepared_1m"
    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_entry"],
            min_usable_rows=5,
            min_train_rows=3,
            min_positive_samples=1,
        )
    )
    summary = pipeline.run(dataset_path, output_dir)

    assert summary["runtime_bar_interval_tuning"]["applied"] is True
    assert summary["config"]["max_analysis_rows"] == 500000


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


def test_preprocessing_propagates_timezone_contract_and_source_lineage() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "tmp" / f"test_preprocessing_timezone_lineage_{uuid4().hex}"
    base_dir.mkdir(parents=True)

    try:
        dataset_path = _write_dataset(base_dir)
        output_dir = base_dir / "prepared"

        pipeline = FeaturePreprocessingPipeline(
            PreprocessingConfig(
                target_columns=["label_long_entry"],
                min_usable_rows=5,
                min_train_rows=3,
                min_positive_samples=1,
            )
        )
        summary = pipeline.run(dataset_path, output_dir)
        report = json.loads((output_dir / "long_entry" / "report.json").read_text(encoding="utf-8"))

        assert summary["timezone_contract"]["canonical_timezone"] == "UTC"
        assert summary["timezone_contract"]["feature_clock_timezone"] == "America/New_York"
        assert summary["source_lineage"]["feature_builder_source_path"] == "data/labeling/labeled_data/sample_labels.csv"
        assert summary["source_lineage"]["upstream_source_path"] == "data/currency_data/sample_raw.csv"
        assert report["timezone_contract"]["canonical_timezone"] == "UTC"
        assert report["source_lineage"]["upstream_bar_timestamp_semantics"] == "bar_open"
        assert report["source_lineage"]["upstream_timezone_contract"]["source_timezone"] == "GMT-6"
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


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


def test_discover_targets_supports_reversal_continuation_and_breakout_families() -> None:
    df = pd.DataFrame(
        columns=[
            "label_long_reversal",
            "sample_weight_long_reversal",
            "label_quality_long_reversal",
            "exclude_long_reversal",
            "neg_ok_long_reversal",
            "label_long_continuation_pullback",
            "sample_weight_long_continuation",
            "label_quality_long_continuation",
            "exclude_long_continuation",
            "neg_ok_long_continuation",
            "label_short_breakout_entry",
            "sample_weight_entry_short_breakout",
            "entry_quality_short_breakout",
            "exclude_short_breakout",
            "neg_ok_short_breakout",
        ]
    )

    specs = discover_targets(df, PreprocessingConfig())
    specs_by_name = {spec.name: spec for spec in specs}

    assert {"long_reversal", "long_continuation_pullback", "short_breakout_entry"} <= set(specs_by_name)

    reversal = specs_by_name["long_reversal"]
    assert reversal.target_column == "label_long_reversal"
    assert reversal.sample_weight_column == "sample_weight_long_reversal"
    assert reversal.quality_column == "label_quality_long_reversal"
    assert reversal.exclude_column == "exclude_long_reversal"
    assert reversal.safe_negative_column == "neg_ok_long_reversal"
    assert reversal.label_kind == "reversal"

    continuation = specs_by_name["long_continuation_pullback"]
    assert continuation.target_column == "label_long_continuation_pullback"
    assert continuation.sample_weight_column == "sample_weight_long_continuation"
    assert continuation.quality_column == "label_quality_long_continuation"
    assert continuation.exclude_column == "exclude_long_continuation"
    assert continuation.safe_negative_column == "neg_ok_long_continuation"
    assert continuation.label_kind == "continuation_pullback"

    breakout_entry = specs_by_name["short_breakout_entry"]
    assert breakout_entry.target_column == "label_short_breakout_entry"
    assert breakout_entry.sample_weight_column == "sample_weight_entry_short_breakout"
    assert breakout_entry.quality_column == "entry_quality_short_breakout"
    assert breakout_entry.exclude_column == "exclude_short_breakout"
    assert breakout_entry.safe_negative_column == "neg_ok_short_breakout"
    assert breakout_entry.label_kind == "breakout_entry"


def test_discover_targets_supports_breakout_sustained_family() -> None:
    df = pd.DataFrame(
        columns=[
            "label_long_breakout_sustained",
            "sample_weight_long_breakout_sustained",
            "label_quality_long_breakout_sustained",
            "exclude_long_breakout_sustained",
            "neg_ok_long_breakout_sustained",
        ]
    )

    specs = discover_targets(
        df,
        PreprocessingConfig(target_columns=["label_long_breakout_sustained"]),
    )
    specs_by_name = {spec.name: spec for spec in specs}

    assert "long_breakout_sustained" in specs_by_name
    sustained = specs_by_name["long_breakout_sustained"]
    assert sustained.target_column == "label_long_breakout_sustained"
    assert sustained.sample_weight_column == "sample_weight_long_breakout_sustained"
    assert sustained.quality_column == "label_quality_long_breakout_sustained"
    assert sustained.exclude_column == "exclude_long_breakout_sustained"
    assert sustained.safe_negative_column == "neg_ok_long_breakout_sustained"
    assert sustained.label_kind == "breakout_sustained"


def test_discover_targets_synthesizes_meta_ote_targets_from_family_zone_labels() -> None:
    df = pd.DataFrame(
        columns=[
            "label_long_reversal",
            "sample_weight_long_reversal",
            "label_quality_long_reversal",
            "exclude_long_reversal",
            "neg_ok_long_reversal",
            "label_long_continuation_pullback",
            "sample_weight_long_continuation",
            "label_quality_long_continuation",
            "exclude_long_continuation",
            "neg_ok_long_continuation",
            "label_long_breakout",
            "sample_weight_long_breakout",
            "label_quality_long_breakout",
            "exclude_long_breakout",
            "neg_ok_long_breakout",
            "label_short_reversal",
            "sample_weight_short_reversal",
            "label_quality_short_reversal",
            "exclude_short_reversal",
            "neg_ok_short_reversal",
            "label_short_continuation_pullback",
            "sample_weight_short_continuation",
            "label_quality_short_continuation",
            "exclude_short_continuation",
            "neg_ok_short_continuation",
            "label_short_breakout",
            "sample_weight_short_breakout",
            "label_quality_short_breakout",
            "exclude_short_breakout",
            "neg_ok_short_breakout",
        ]
    )

    specs = discover_targets(
        df,
        PreprocessingConfig(target_columns=["label_long_ote", "label_short_ote"]),
    )
    specs_by_name = {spec.name: spec for spec in specs}

    assert {"long_ote", "short_ote"} == set(specs_by_name)

    long_meta = specs_by_name["long_ote"]
    assert long_meta.is_synthetic is True
    assert long_meta.target_column == "label_long_ote"
    assert long_meta.component_target_columns == [
        "label_long_reversal",
        "label_long_continuation_pullback",
        "label_long_breakout",
    ]
    assert long_meta.component_sample_weight_columns == [
        "sample_weight_long_reversal",
        "sample_weight_long_continuation",
        "sample_weight_long_breakout",
    ]
    assert long_meta.component_exclude_columns == [
        "exclude_long_reversal",
        "exclude_long_continuation",
        "exclude_long_breakout",
    ]
    assert long_meta.component_safe_negative_columns == [
        "neg_ok_long_reversal",
        "neg_ok_long_continuation",
        "neg_ok_long_breakout",
    ]


def test_preprocessing_builds_meta_ote_target_from_union_of_family_zone_labels(tmp_path: Path) -> None:
    dataset_path = _write_meta_zone_dataset(tmp_path)
    output_dir = tmp_path / "prepared_meta"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_ote", "label_short_ote"],
            min_usable_rows=8,
            min_train_rows=4,
            min_positive_samples=2,
            top_n_features=5,
        )
    )
    summary = pipeline.run(dataset_path, output_dir)

    assert set(summary["targets"]) == {"long_ote", "short_ote"}

    long_dir = output_dir / "long_ote"
    long_report = json.loads((long_dir / "report.json").read_text(encoding="utf-8"))
    long_target_construction = long_report["target_construction"]

    assert long_target_construction["type"] == "synthetic_any_positive_union"
    assert long_target_construction["source_target_columns"] == [
        "label_long_reversal",
        "label_long_continuation_pullback",
        "label_long_breakout",
    ]

    long_positive_union = len({5, 6, 10, 14, 15, 18, 19})
    assert long_report["row_counts"]["rows_positive"] == long_positive_union
    assert long_report["row_counts"]["rows_excluded_as_ambiguous"] > 0

    all_long_rows = pd.concat(
        [
            pd.read_csv(long_dir / "train.csv"),
            pd.read_csv(long_dir / "val.csv"),
            pd.read_csv(long_dir / "test.csv"),
        ],
        ignore_index=True,
    )
    assert "source_row_idx" in all_long_rows.columns
    assert int(all_long_rows["target"].sum()) == long_positive_union
    assert float(all_long_rows.loc[all_long_rows["target"] == 1, "sample_weight"].max()) == 3.0
