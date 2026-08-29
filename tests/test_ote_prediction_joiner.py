from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.preprocessing import FeaturePreprocessingPipeline, PreprocessingConfig
from model_testing.ote_prediction_joiner import join_prediction_file


def _make_local_tmp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "tmp" / "ote_prediction_joiner_tests"
    path = root / "case"
    suffix = 0
    while path.exists():
        suffix += 1
        path = root / f"case_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_ote_source_dataset(base_dir: Path) -> Path:
    rows = 24
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="5min")
    long_target = [0] * rows
    for index in (4, 9, 14, 19):
        long_target[index] = 1

    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "open": [1.10 + i * 0.0001 for i in range(rows)],
            "high": [1.11 + i * 0.0001 for i in range(rows)],
            "low": [1.09 + i * 0.0001 for i in range(rows)],
            "close": [1.10 + i * 0.0001 for i in range(rows)],
            "volume": [100 + i for i in range(rows)],
            "label_long_ote": long_target,
            "exclude_long": [False] * rows,
            "neg_ok_long": [i not in {2, 6, 11} and long_target[i] == 0 for i in range(rows)],
            "sample_weight_long": [2.0 if long_target[i] else 1.0 for i in range(rows)],
            "warmup_mask": [i < 2 for i in range(rows)],
            "feature_signal": [float(i % 3 == 0) for i in range(rows)],
        }
    )

    dataset_path = base_dir / "ote_features.csv"
    metadata_path = base_dir / "ote_features.metadata.json"
    df.to_csv(dataset_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": ["feature_signal"],
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


def _prepare_ote_dataset(tmp_path: Path) -> tuple[Path, Path]:
    source_path = _write_ote_source_dataset(tmp_path)
    prepared_root = tmp_path / "prepared"
    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_ote"],
            min_usable_rows=5,
            min_train_rows=3,
            min_positive_samples=1,
        )
    )
    pipeline.run(source_path, prepared_root)
    return source_path, prepared_root


def _write_family_source_dataset(base_dir: Path) -> Path:
    rows = 24
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="5min")
    long_reversal = [0] * rows
    for index in (4, 9, 14, 19):
        long_reversal[index] = 1
    long_continuation_pullback = [0] * rows
    long_breakout = [0] * rows

    def _negative_mask(target_values: list[int]) -> list[bool]:
        return [i not in {2, 6, 11} and target_values[i] == 0 for i in range(rows)]

    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "open": [1.10 + i * 0.0001 for i in range(rows)],
            "high": [1.11 + i * 0.0001 for i in range(rows)],
            "low": [1.09 + i * 0.0001 for i in range(rows)],
            "close": [1.10 + i * 0.0001 for i in range(rows)],
            "volume": [100 + i for i in range(rows)],
            "label_long_reversal": long_reversal,
            "exclude_long_reversal": [False] * rows,
            "neg_ok_long_reversal": _negative_mask(long_reversal),
            "sample_weight_long_reversal": [2.0 if long_reversal[i] else 1.0 for i in range(rows)],
            "label_quality_long_reversal": [1.0] * rows,
            "label_long_continuation_pullback": long_continuation_pullback,
            "exclude_long_continuation": [False] * rows,
            "neg_ok_long_continuation": _negative_mask(long_continuation_pullback),
            "sample_weight_long_continuation": [1.0] * rows,
            "label_quality_long_continuation": [1.0] * rows,
            "label_long_breakout": long_breakout,
            "exclude_long_breakout": [False] * rows,
            "neg_ok_long_breakout": _negative_mask(long_breakout),
            "sample_weight_long_breakout": [1.0] * rows,
            "label_quality_long_breakout": [1.0] * rows,
            "warmup_mask": [i < 2 for i in range(rows)],
            "feature_signal": [float(i % 3 == 0) for i in range(rows)],
        }
    )

    dataset_path = base_dir / "family_features.csv"
    metadata_path = base_dir / "family_features.metadata.json"
    df.to_csv(dataset_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": ["feature_signal"],
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


def _prepare_family_dataset(tmp_path: Path) -> tuple[Path, Path]:
    source_path = _write_family_source_dataset(tmp_path)
    prepared_root = tmp_path / "prepared"
    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_reversal", "label_long_ote"],
            min_usable_rows=5,
            min_train_rows=3,
            min_positive_samples=1,
        )
    )
    pipeline.run(source_path, prepared_root)
    return source_path, prepared_root


def test_join_prediction_file_uses_direct_source_row_idx() -> None:
    tmp_path = _make_local_tmp_dir()
    source_path, prepared_root = _prepare_ote_dataset(tmp_path)
    target_dir = prepared_root / "long_ote"

    train_split = pd.read_csv(target_dir / "train.csv")
    val_split = pd.read_csv(target_dir / "val.csv")
    dev_split = pd.concat([train_split, val_split], ignore_index=True)

    model_dir = tmp_path / "models" / "long_ote"
    model_dir.mkdir(parents=True)
    prediction_path = model_dir / "oof_predictions.csv"
    pd.DataFrame(
        {
            "source_row_idx": dev_split["source_row_idx"].iloc[:3].to_list(),
            "row_index_in_dev": [0, 1, 2],
            "fold": [1, 1, 1],
            "target": [0, 1, 0],
            "sample_weight": [1.0, 2.0, 1.0],
            "oof_raw_probability": [0.2, 0.7, 0.3],
            "oof_calibrated_probability": [0.25, 0.75, 0.35],
        }
    ).to_csv(prediction_path, index=False)

    joined = join_prediction_file(prediction_path, source_file=source_path)
    source_frame = pd.read_csv(source_path)
    source_datetime = pd.to_datetime(
        source_frame.iloc[joined["source_row_idx"]]["datetime"],
        errors="coerce",
        utc=True,
    )
    joined_datetime = pd.to_datetime(joined["datetime"], errors="coerce", utc=True)

    assert "datetime" in joined.columns
    assert joined["source_row_idx"].tolist() == dev_split["source_row_idx"].iloc[:3].tolist()
    assert joined_datetime.tolist() == source_datetime.tolist()


def test_join_prediction_file_reconstructs_legacy_test_indices() -> None:
    tmp_path = _make_local_tmp_dir()
    source_path, prepared_root = _prepare_ote_dataset(tmp_path)
    target_dir = prepared_root / "long_ote"
    test_split = pd.read_csv(target_dir / "test.csv")

    model_dir = tmp_path / "models" / "long_ote"
    model_dir.mkdir(parents=True)
    prediction_path = model_dir / "test_predictions.csv"
    pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "target": [0, 1, 0],
            "sample_weight": [1.0, 2.0, 1.0],
            "raw_probability": [0.4, 0.8, 0.3],
            "calibrated_probability": [0.45, 0.85, 0.35],
            "predicted_label": [0, 1, 0],
        }
    ).to_csv(prediction_path, index=False)

    (model_dir / "training_summary.json").write_text(
        json.dumps({"prepared_dir": str(target_dir)}, indent=2),
        encoding="utf-8",
    )

    joined = join_prediction_file(prediction_path)
    source_frame = pd.read_csv(source_path)
    expected_source_rows = test_split["source_row_idx"].iloc[:3].tolist()
    source_datetime = pd.to_datetime(
        source_frame.iloc[expected_source_rows]["datetime"],
        errors="coerce",
        utc=True,
    )
    joined_datetime = pd.to_datetime(joined["datetime"], errors="coerce", utc=True)

    assert joined["source_row_idx"].tolist() == expected_source_rows
    assert joined_datetime.tolist() == source_datetime.tolist()


def test_join_prediction_file_infers_family_target_from_training_summary() -> None:
    tmp_path = _make_local_tmp_dir()
    source_path, prepared_root = _prepare_family_dataset(tmp_path)
    target_dir = prepared_root / "long_reversal"
    test_split = pd.read_csv(target_dir / "test.csv")

    model_dir = tmp_path / "models" / "long_reversal_tcn_champion"
    model_dir.mkdir(parents=True)
    prediction_path = model_dir / "test_predictions.csv"
    pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "target": [0, 1, 0],
            "sample_weight": [1.0, 2.0, 1.0],
            "raw_probability": [0.4, 0.8, 0.3],
            "calibrated_probability": [0.45, 0.85, 0.35],
            "predicted_label": [0, 1, 0],
        }
    ).to_csv(prediction_path, index=False)

    (model_dir / "training_summary.json").write_text(
        json.dumps({"prepared_dir": str(target_dir), "target": "long_reversal"}, indent=2),
        encoding="utf-8",
    )

    joined = join_prediction_file(prediction_path)
    source_frame = pd.read_csv(source_path)
    expected_source_rows = test_split["source_row_idx"].iloc[:3].tolist()
    source_datetime = pd.to_datetime(
        source_frame.iloc[expected_source_rows]["datetime"],
        errors="coerce",
        utc=True,
    )
    joined_datetime = pd.to_datetime(joined["datetime"], errors="coerce", utc=True)

    assert joined["source_row_idx"].tolist() == expected_source_rows
    assert joined_datetime.tolist() == source_datetime.tolist()


def test_join_prediction_file_falls_back_to_feature_builder_source_for_missing_price_columns() -> None:
    tmp_path = _make_local_tmp_dir()
    source_path, prepared_root = _prepare_ote_dataset(tmp_path)
    target_dir = prepared_root / "long_ote"
    test_split = pd.read_csv(target_dir / "test.csv")

    fallback_source = tmp_path / "prepared_input_without_prices.csv"
    base_source = pd.read_csv(source_path)
    base_source.loc[:, ["datetime", "label_long_ote", "exclude_long", "neg_ok_long", "sample_weight_long", "warmup_mask", "feature_signal"]].to_csv(
        fallback_source,
        index=False,
    )

    summary_path = prepared_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["input_file"] = str(fallback_source)
    summary["source_lineage"] = {
        "feature_builder_source_path": str(source_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_dir = tmp_path / "models" / "long_ote"
    model_dir.mkdir(parents=True)
    prediction_path = model_dir / "test_predictions.csv"
    pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "target": [0, 1, 0],
            "sample_weight": [1.0, 2.0, 1.0],
            "raw_probability": [0.4, 0.8, 0.3],
            "calibrated_probability": [0.45, 0.85, 0.35],
            "predicted_label": [0, 1, 0],
        }
    ).to_csv(prediction_path, index=False)

    (model_dir / "training_summary.json").write_text(
        json.dumps({"prepared_dir": str(target_dir)}, indent=2),
        encoding="utf-8",
    )

    joined = join_prediction_file(
        prediction_path,
        source_columns=["datetime", "close", "feature_signal"],
    )

    expected_source_rows = test_split["source_row_idx"].iloc[:3].tolist()
    expected_close = base_source.iloc[expected_source_rows]["close"].to_list()

    assert joined["source_row_idx"].tolist() == expected_source_rows
    assert joined["close"].tolist() == expected_close
    assert "feature_signal" in joined.columns


def test_join_prediction_file_reconstructs_synthetic_ote_target_from_training_summary() -> None:
    tmp_path = _make_local_tmp_dir()
    source_path, prepared_root = _prepare_family_dataset(tmp_path)
    target_dir = prepared_root / "long_ote"
    test_split = pd.read_csv(target_dir / "test.csv")

    model_dir = tmp_path / "models" / "long_tcn_champion"
    model_dir.mkdir(parents=True)
    prediction_path = model_dir / "test_predictions.csv"
    pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "target": [0, 1, 0],
            "sample_weight": [1.0, 2.0, 1.0],
            "raw_probability": [0.4, 0.8, 0.3],
            "calibrated_probability": [0.45, 0.85, 0.35],
            "predicted_label": [0, 1, 0],
        }
    ).to_csv(prediction_path, index=False)

    (model_dir / "training_summary.json").write_text(
        json.dumps({"prepared_dir": str(target_dir), "target": "long_ote"}, indent=2),
        encoding="utf-8",
    )

    joined = join_prediction_file(prediction_path)
    source_frame = pd.read_csv(source_path)
    expected_source_rows = test_split["source_row_idx"].iloc[:3].tolist()
    source_datetime = pd.to_datetime(
        source_frame.iloc[expected_source_rows]["datetime"],
        errors="coerce",
        utc=True,
    )
    joined_datetime = pd.to_datetime(joined["datetime"], errors="coerce", utc=True)

    assert joined["source_row_idx"].tolist() == expected_source_rows
    assert joined_datetime.tolist() == source_datetime.tolist()
