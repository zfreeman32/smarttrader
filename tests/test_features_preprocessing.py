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
from preprocessing.feature_selection import build_split_indices, discover_targets


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


def _write_frvp_dataset(base_dir: Path, *, include_metadata: bool = True) -> Path:
    rows = 96
    timestamps = pd.date_range("2024-03-01 00:00:00", periods=rows, freq="5min")

    def _marks(indices: set[int]) -> list[int]:
        return [1 if index in indices else 0 for index in range(rows)]

    long_reversal_idx = {8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88}
    short_reversal_idx = {12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92}
    long_continuation_idx = {10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90}
    short_continuation_idx = {14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94}

    long_reversal = _marks(long_reversal_idx)
    short_reversal = _marks(short_reversal_idx)
    long_continuation = _marks(long_continuation_idx)
    short_continuation = _marks(short_continuation_idx)

    frvp_dist_poc_session_atr = []
    frvp_dist_poc_day_atr = []
    frvp_open_type = []
    frvp_day_type = []
    frvp_poc_plus_ict_fvg_confluence = []
    htf_confluence_long_reversal = []
    htf_confluence_short_reversal = []
    htf_confluence_long_continuation = []
    htf_confluence_short_continuation = []

    for index in range(rows):
        long_rev_hit = index in long_reversal_idx
        short_rev_hit = index in short_reversal_idx
        long_cont_hit = index in long_continuation_idx
        short_cont_hit = index in short_continuation_idx

        frvp_dist_poc_session_atr.append(
            1.25 if long_rev_hit else (-1.10 if short_rev_hit else ((index % 11) - 5) / 5.0)
        )
        frvp_dist_poc_day_atr.append(
            0.95 if long_cont_hit else (-0.90 if short_cont_hit else np.sin(index / 7.0))
        )
        frvp_open_type.append(
            1 if (long_rev_hit or long_cont_hit) else (-1 if (short_rev_hit or short_cont_hit) else (index % 3) - 1)
        )
        frvp_day_type.append(
            3 if (long_cont_hit or short_cont_hit) else (2 if (long_rev_hit or short_rev_hit) else index % 4)
        )
        frvp_poc_plus_ict_fvg_confluence.append(
            1 if (long_rev_hit or short_rev_hit or long_cont_hit or short_cont_hit) else int(index % 5 == 0)
        )
        htf_confluence_long_reversal.append(1 if long_rev_hit or index in {7, 9, 39, 41} else 0)
        htf_confluence_short_reversal.append(1 if short_rev_hit or index in {11, 13, 43, 45} else 0)
        htf_confluence_long_continuation.append(1 if long_cont_hit or index in {17, 19, 49, 51} else 0)
        htf_confluence_short_continuation.append(1 if short_cont_hit or index in {21, 23, 53, 55} else 0)

    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "ts_event": timestamps,
            "open": 5000.0 + np.arange(rows, dtype=float) * 0.25,
            "high": 5000.5 + np.arange(rows, dtype=float) * 0.25,
            "low": 4999.5 + np.arange(rows, dtype=float) * 0.25,
            "close": 5000.1 + np.arange(rows, dtype=float) * 0.25,
            "volume": 1000 + np.arange(rows, dtype=int) * 5,
            "symbol": ["ES.v.0"] * rows,
            "instrument_id": [35903] * rows,
            "contract_symbol": ["ESH4"] * rows,
            "contract_expiration": ["2024-03-15T13:30:00+00:00"] * rows,
            "is_roll_boundary": [False] * rows,
            "bars_since_roll": list(range(rows)),
            "market_day_close": [timestamps[min(index + 1, rows - 1)] for index in range(rows)],
            "market_day_index": [index // 12 for index in range(rows)],
            "in_roll_bracket": [False] * rows,
            "frvp_atr": np.linspace(1.0, 2.0, rows),
            "frvp_structural_atr": np.linspace(1.2, 2.2, rows),
            "frvp_dist_poc_session_atr": frvp_dist_poc_session_atr,
            "frvp_dist_poc_day_atr": frvp_dist_poc_day_atr,
            "frvp_open_type": frvp_open_type,
            "frvp_day_type": frvp_day_type,
            "frvp_poc_plus_ict_fvg_confluence": frvp_poc_plus_ict_fvg_confluence,
            "label_long_frvp_reversal": long_reversal,
            "label_short_frvp_reversal": short_reversal,
            "label_long_frvp_continuation": long_continuation,
            "label_short_frvp_continuation": short_continuation,
            "sample_weight_long_frvp_reversal": [2.5 if value else 1.0 for value in long_reversal],
            "sample_weight_short_frvp_reversal": [2.5 if value else 1.0 for value in short_reversal],
            "sample_weight_long_frvp_continuation": [2.0 if value else 1.0 for value in long_continuation],
            "sample_weight_short_frvp_continuation": [2.0 if value else 1.0 for value in short_continuation],
            "label_quality_long_frvp_reversal": [0.8 if value else 0.0 for value in long_reversal],
            "label_quality_short_frvp_reversal": [0.78 if value else 0.0 for value in short_reversal],
            "label_quality_long_frvp_continuation": [0.76 if value else 0.0 for value in long_continuation],
            "label_quality_short_frvp_continuation": [0.74 if value else 0.0 for value in short_continuation],
            "exclude_long_frvp_reversal": [False] * rows,
            "exclude_short_frvp_reversal": [False] * rows,
            "exclude_long_frvp_continuation": [False] * rows,
            "exclude_short_frvp_continuation": [False] * rows,
            "neg_ok_long_frvp_reversal": [not value for value in long_reversal],
            "neg_ok_short_frvp_reversal": [not value for value in short_reversal],
            "neg_ok_long_frvp_continuation": [not value for value in long_continuation],
            "neg_ok_short_frvp_continuation": [not value for value in short_continuation],
            "htf_confluence_long_frvp_reversal": htf_confluence_long_reversal,
            "htf_confluence_short_frvp_reversal": htf_confluence_short_reversal,
            "htf_confluence_long_frvp_continuation": htf_confluence_long_continuation,
            "htf_confluence_short_frvp_continuation": htf_confluence_short_continuation,
            "warmup_mask": [index < 4 for index in range(rows)],
        }
    )

    dataset_path = base_dir / "frvp_features.csv"
    df.to_csv(dataset_path, index=False)

    if include_metadata:
        metadata_path = base_dir / "frvp_features.metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "feature_columns": [
                        "frvp_dist_poc_session_atr",
                        "frvp_dist_poc_day_atr",
                        "frvp_open_type",
                        "frvp_day_type",
                        "frvp_poc_plus_ict_fvg_confluence",
                    ],
                    "timezone_contract": {
                        "source_timezone": "UTC",
                        "canonical_timezone": "UTC",
                        "feature_clock_timezone": "America/New_York",
                        "market_close_timezone": "America/New_York",
                    },
                    "config": {
                        "drop_warmup_rows": False,
                        "warmup_rows": 0,
                        "fillna_numeric": False,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return dataset_path


def _write_ict_dataset(base_dir: Path, *, include_metadata: bool = True) -> Path:
    rows = 96
    timestamps = pd.date_range("2024-05-01 00:00:00", periods=rows, freq="5min")

    def _marks(indices: set[int]) -> list[int]:
        return [1 if index in indices else 0 for index in range(rows)]

    long_reversal_idx = {8, 16, 24, 32, 48, 64, 80}
    short_reversal_idx = {12, 20, 36, 44, 60, 76}
    long_continuation_idx = {16, 18, 34, 48, 50, 66, 82}
    short_continuation_idx = {20, 22, 38, 60, 62, 78}

    long_reversal = _marks(long_reversal_idx)
    short_reversal = _marks(short_reversal_idx)
    long_continuation = _marks(long_continuation_idx)
    short_continuation = _marks(short_continuation_idx)

    long_meta_idx = long_reversal_idx | long_continuation_idx
    short_meta_idx = short_reversal_idx | short_continuation_idx
    long_meta = _marks(long_meta_idx)
    short_meta = _marks(short_meta_idx)

    ict_total_confluence = []
    ict_zone_balance = []
    ict_bull_sweep_plus_fvg = []
    ict_sweep_plus_choch = []
    htf_confluence_long_reversal = []
    htf_confluence_short_reversal = []
    htf_confluence_long_continuation = []
    htf_confluence_short_continuation = []
    htf_confluence_long_meta = []
    htf_confluence_short_meta = []
    sample_weight_long_meta = []
    sample_weight_short_meta = []
    label_quality_long_meta = []
    label_quality_short_meta = []

    for index in range(rows):
        long_rev_hit = index in long_reversal_idx
        short_rev_hit = index in short_reversal_idx
        long_cont_hit = index in long_continuation_idx
        short_cont_hit = index in short_continuation_idx
        long_meta_hit = index in long_meta_idx
        short_meta_hit = index in short_meta_idx

        ict_total_confluence.append(
            3.0 if long_meta_hit else (-2.5 if short_meta_hit else np.sin(index / 6.0))
        )
        ict_zone_balance.append(
            1.2 if long_rev_hit else (-1.1 if short_rev_hit else ((index % 9) - 4) / 4.0)
        )
        ict_bull_sweep_plus_fvg.append(int(long_rev_hit or long_cont_hit or index in {7, 15, 47, 65}))
        ict_sweep_plus_choch.append(int(short_rev_hit or short_cont_hit or index in {11, 19, 59, 77}))
        htf_confluence_long_reversal.append(1 if long_rev_hit or index in {7, 9, 47, 49} else 0)
        htf_confluence_short_reversal.append(1 if short_rev_hit or index in {11, 13, 59, 61} else 0)
        htf_confluence_long_continuation.append(1 if long_cont_hit or index in {17, 19, 65, 67} else 0)
        htf_confluence_short_continuation.append(1 if short_cont_hit or index in {21, 23, 77, 79} else 0)
        htf_confluence_long_meta.append(int(long_meta_hit or index in {17, 47, 67}))
        htf_confluence_short_meta.append(int(short_meta_hit or index in {21, 61, 79}))

        sample_weight_long_meta.append(2.4 if long_rev_hit else (1.9 if long_cont_hit else 1.0))
        sample_weight_short_meta.append(2.3 if short_rev_hit else (1.8 if short_cont_hit else 1.0))
        label_quality_long_meta.append(0.82 if long_rev_hit else (0.74 if long_cont_hit else 0.0))
        label_quality_short_meta.append(0.79 if short_rev_hit else (0.71 if short_cont_hit else 0.0))

    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "open": 5100.0 + np.arange(rows, dtype=float) * 0.20,
            "high": 5100.5 + np.arange(rows, dtype=float) * 0.20,
            "low": 5099.5 + np.arange(rows, dtype=float) * 0.20,
            "close": 5100.1 + np.arange(rows, dtype=float) * 0.20,
            "volume": 1200 + np.arange(rows, dtype=int) * 6,
            "ict_total_confluence_1atr": ict_total_confluence,
            "ict_zone_balance_1atr": ict_zone_balance,
            "ict_bull_sweep_plus_fvg": ict_bull_sweep_plus_fvg,
            "ict_sweep_plus_choch": ict_sweep_plus_choch,
            "label_long_ict_reversal": long_reversal,
            "label_short_ict_reversal": short_reversal,
            "label_long_ict_continuation": long_continuation,
            "label_short_ict_continuation": short_continuation,
            "label_long_ict_meta": long_meta,
            "label_short_ict_meta": short_meta,
            "sample_weight_long_ict_reversal": [2.4 if value else 1.0 for value in long_reversal],
            "sample_weight_short_ict_reversal": [2.3 if value else 1.0 for value in short_reversal],
            "sample_weight_long_ict_continuation": [1.9 if value else 1.0 for value in long_continuation],
            "sample_weight_short_ict_continuation": [1.8 if value else 1.0 for value in short_continuation],
            "sample_weight_long_ict_meta": sample_weight_long_meta,
            "sample_weight_short_ict_meta": sample_weight_short_meta,
            "label_quality_long_ict_reversal": [0.82 if value else 0.0 for value in long_reversal],
            "label_quality_short_ict_reversal": [0.79 if value else 0.0 for value in short_reversal],
            "label_quality_long_ict_continuation": [0.74 if value else 0.0 for value in long_continuation],
            "label_quality_short_ict_continuation": [0.71 if value else 0.0 for value in short_continuation],
            "label_quality_long_ict_meta": label_quality_long_meta,
            "label_quality_short_ict_meta": label_quality_short_meta,
            "exclude_long_ict_reversal": [False] * rows,
            "exclude_short_ict_reversal": [False] * rows,
            "exclude_long_ict_continuation": [False] * rows,
            "exclude_short_ict_continuation": [False] * rows,
            "exclude_long_ict_meta": [False] * rows,
            "exclude_short_ict_meta": [False] * rows,
            "neg_ok_long_ict_reversal": [not value for value in long_reversal],
            "neg_ok_short_ict_reversal": [not value for value in short_reversal],
            "neg_ok_long_ict_continuation": [not value for value in long_continuation],
            "neg_ok_short_ict_continuation": [not value for value in short_continuation],
            "neg_ok_long_ict_meta": [not value for value in long_meta],
            "neg_ok_short_ict_meta": [not value for value in short_meta],
            "htf_confluence_long_ict_reversal": htf_confluence_long_reversal,
            "htf_confluence_short_ict_reversal": htf_confluence_short_reversal,
            "htf_confluence_long_ict_continuation": htf_confluence_long_continuation,
            "htf_confluence_short_ict_continuation": htf_confluence_short_continuation,
            "htf_confluence_long_ict_meta": htf_confluence_long_meta,
            "htf_confluence_short_ict_meta": htf_confluence_short_meta,
            "warmup_mask": [index < 4 for index in range(rows)],
        }
    )

    dataset_path = base_dir / "ict_features.csv"
    df.to_csv(dataset_path, index=False)

    if include_metadata:
        metadata_path = base_dir / "ict_features.metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "feature_columns": [
                        "ict_total_confluence_1atr",
                        "ict_zone_balance_1atr",
                        "ict_bull_sweep_plus_fvg",
                        "ict_sweep_plus_choch",
                    ],
                    "timezone_contract": {
                        "source_timezone": "UTC",
                        "canonical_timezone": "UTC",
                        "feature_clock_timezone": "America/New_York",
                        "market_close_timezone": "America/New_York",
                    },
                    "config": {
                        "drop_warmup_rows": False,
                        "warmup_rows": 0,
                        "fillna_numeric": False,
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


def test_build_split_indices_can_reserve_source_row_embargo_gaps() -> None:
    config = PreprocessingConfig(
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
        split_embargo_bars=2,
    )
    source_rows = pd.Series(np.arange(12, dtype=np.int64))

    splits = build_split_indices(
        len(source_rows),
        config,
        source_row_idx=source_rows,
        target_name="long_entry",
    )

    assert splits["train"].tolist() == [0, 1, 2, 3, 4, 5]
    assert splits["val"].tolist() == [8]
    assert splits["test"].tolist() == [11]


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
    calls: dict[str, object] = {}

    def fake_mutual_info(X, y, random_state=None, n_neighbors=None, discrete_features=None):
        calls["mi_rows"] = len(X)
        calls["mi_discrete_features"] = list(discrete_features)
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
    assert calls["mi_discrete_features"] == [False, True, False]
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


def test_discover_targets_supports_frvp_direct_families() -> None:
    df = pd.DataFrame(
        columns=[
            "label_long_frvp_reversal",
            "sample_weight_long_frvp_reversal",
            "label_quality_long_frvp_reversal",
            "exclude_long_frvp_reversal",
            "neg_ok_long_frvp_reversal",
            "label_short_frvp_continuation",
            "sample_weight_short_frvp_continuation",
            "label_quality_short_frvp_continuation",
            "exclude_short_frvp_continuation",
            "neg_ok_short_frvp_continuation",
        ]
    )

    specs = discover_targets(
        df,
        PreprocessingConfig(
            target_columns=[
                "label_long_frvp_reversal",
                "label_short_frvp_continuation",
            ]
        ),
    )
    specs_by_name = {spec.name: spec for spec in specs}

    assert {"long_frvp_reversal", "short_frvp_continuation"} == set(specs_by_name)

    long_reversal = specs_by_name["long_frvp_reversal"]
    assert long_reversal.target_column == "label_long_frvp_reversal"
    assert long_reversal.sample_weight_column == "sample_weight_long_frvp_reversal"
    assert long_reversal.quality_column == "label_quality_long_frvp_reversal"
    assert long_reversal.exclude_column == "exclude_long_frvp_reversal"
    assert long_reversal.safe_negative_column == "neg_ok_long_frvp_reversal"
    assert long_reversal.label_kind == "frvp_reversal"

    short_continuation = specs_by_name["short_frvp_continuation"]
    assert short_continuation.target_column == "label_short_frvp_continuation"
    assert short_continuation.sample_weight_column == "sample_weight_short_frvp_continuation"
    assert short_continuation.quality_column == "label_quality_short_frvp_continuation"
    assert short_continuation.exclude_column == "exclude_short_frvp_continuation"
    assert short_continuation.safe_negative_column == "neg_ok_short_frvp_continuation"
    assert short_continuation.label_kind == "frvp_continuation"


def test_discover_targets_supports_ict_direct_families_and_meta() -> None:
    df = pd.DataFrame(
        columns=[
            "label_long_ict_reversal",
            "sample_weight_long_ict_reversal",
            "label_quality_long_ict_reversal",
            "exclude_long_ict_reversal",
            "neg_ok_long_ict_reversal",
            "label_short_ict_continuation",
            "sample_weight_short_ict_continuation",
            "label_quality_short_ict_continuation",
            "exclude_short_ict_continuation",
            "neg_ok_short_ict_continuation",
            "label_long_ict_meta",
            "sample_weight_long_ict_meta",
            "label_quality_long_ict_meta",
            "exclude_long_ict_meta",
            "neg_ok_long_ict_meta",
        ]
    )

    specs = discover_targets(df, PreprocessingConfig())
    specs_by_name = {spec.name: spec for spec in specs}

    assert {"long_ict_reversal", "short_ict_continuation", "long_ict_meta"} <= set(specs_by_name)

    long_reversal = specs_by_name["long_ict_reversal"]
    assert long_reversal.target_column == "label_long_ict_reversal"
    assert long_reversal.sample_weight_column == "sample_weight_long_ict_reversal"
    assert long_reversal.quality_column == "label_quality_long_ict_reversal"
    assert long_reversal.exclude_column == "exclude_long_ict_reversal"
    assert long_reversal.safe_negative_column == "neg_ok_long_ict_reversal"
    assert long_reversal.label_kind == "ict_reversal"

    short_continuation = specs_by_name["short_ict_continuation"]
    assert short_continuation.target_column == "label_short_ict_continuation"
    assert short_continuation.sample_weight_column == "sample_weight_short_ict_continuation"
    assert short_continuation.quality_column == "label_quality_short_ict_continuation"
    assert short_continuation.exclude_column == "exclude_short_ict_continuation"
    assert short_continuation.safe_negative_column == "neg_ok_short_ict_continuation"
    assert short_continuation.label_kind == "ict_continuation"

    long_meta = specs_by_name["long_ict_meta"]
    assert long_meta.target_column == "label_long_ict_meta"
    assert long_meta.sample_weight_column == "sample_weight_long_ict_meta"
    assert long_meta.quality_column == "label_quality_long_ict_meta"
    assert long_meta.exclude_column == "exclude_long_ict_meta"
    assert long_meta.safe_negative_column == "neg_ok_long_ict_meta"
    assert long_meta.label_kind == "ict_meta"


def test_discover_targets_synthesizes_frvp_meta_targets_from_family_labels() -> None:
    df = pd.DataFrame(
        columns=[
            "label_long_frvp_reversal",
            "sample_weight_long_frvp_reversal",
            "label_quality_long_frvp_reversal",
            "exclude_long_frvp_reversal",
            "neg_ok_long_frvp_reversal",
            "label_long_frvp_continuation",
            "sample_weight_long_frvp_continuation",
            "label_quality_long_frvp_continuation",
            "exclude_long_frvp_continuation",
            "neg_ok_long_frvp_continuation",
            "label_short_frvp_reversal",
            "sample_weight_short_frvp_reversal",
            "label_quality_short_frvp_reversal",
            "exclude_short_frvp_reversal",
            "neg_ok_short_frvp_reversal",
            "label_short_frvp_continuation",
            "sample_weight_short_frvp_continuation",
            "label_quality_short_frvp_continuation",
            "exclude_short_frvp_continuation",
            "neg_ok_short_frvp_continuation",
        ]
    )

    specs = discover_targets(
        df,
        PreprocessingConfig(target_columns=["label_long_frvp_meta", "label_short_frvp_meta"]),
    )
    specs_by_name = {spec.name: spec for spec in specs}

    assert {"long_frvp_meta", "short_frvp_meta"} == set(specs_by_name)

    long_meta = specs_by_name["long_frvp_meta"]
    assert long_meta.is_synthetic is True
    assert long_meta.component_target_columns == [
        "label_long_frvp_reversal",
        "label_long_frvp_continuation",
    ]
    assert long_meta.component_sample_weight_columns == [
        "sample_weight_long_frvp_reversal",
        "sample_weight_long_frvp_continuation",
    ]
    assert long_meta.component_exclude_columns == [
        "exclude_long_frvp_reversal",
        "exclude_long_frvp_continuation",
    ]
    assert long_meta.component_safe_negative_columns == [
        "neg_ok_long_frvp_reversal",
        "neg_ok_long_frvp_continuation",
    ]


def test_discover_targets_synthesizes_ict_meta_targets_from_family_labels() -> None:
    df = pd.DataFrame(
        columns=[
            "label_long_ict_reversal",
            "sample_weight_long_ict_reversal",
            "label_quality_long_ict_reversal",
            "exclude_long_ict_reversal",
            "neg_ok_long_ict_reversal",
            "label_long_ict_continuation",
            "sample_weight_long_ict_continuation",
            "label_quality_long_ict_continuation",
            "exclude_long_ict_continuation",
            "neg_ok_long_ict_continuation",
            "label_short_ict_reversal",
            "sample_weight_short_ict_reversal",
            "label_quality_short_ict_reversal",
            "exclude_short_ict_reversal",
            "neg_ok_short_ict_reversal",
            "label_short_ict_continuation",
            "sample_weight_short_ict_continuation",
            "label_quality_short_ict_continuation",
            "exclude_short_ict_continuation",
            "neg_ok_short_ict_continuation",
        ]
    )

    specs = discover_targets(
        df,
        PreprocessingConfig(target_columns=["label_long_ict_meta", "label_short_ict_meta"]),
    )
    specs_by_name = {spec.name: spec for spec in specs}

    assert {"long_ict_meta", "short_ict_meta"} == set(specs_by_name)

    long_meta = specs_by_name["long_ict_meta"]
    assert long_meta.is_synthetic is True
    assert long_meta.component_target_columns == [
        "label_long_ict_reversal",
        "label_long_ict_continuation",
    ]
    assert long_meta.component_sample_weight_columns == [
        "sample_weight_long_ict_reversal",
        "sample_weight_long_ict_continuation",
    ]
    assert long_meta.component_exclude_columns == [
        "exclude_long_ict_reversal",
        "exclude_long_ict_continuation",
    ]
    assert long_meta.component_safe_negative_columns == [
        "neg_ok_long_ict_reversal",
        "neg_ok_long_ict_continuation",
    ]


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
    assert long_report["row_counts"]["rows_negative"] > 0

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


def test_preprocessing_builds_frvp_meta_target_from_union_of_family_labels(tmp_path: Path) -> None:
    dataset_path = _write_frvp_dataset(tmp_path, include_metadata=True)
    output_dir = tmp_path / "prepared_frvp_meta"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_frvp_meta", "label_short_frvp_meta"],
            min_usable_rows=20,
            min_train_rows=10,
            min_positive_samples=3,
            top_n_features=10,
        )
    )
    summary = pipeline.run(dataset_path, output_dir)

    assert set(summary["targets"]) == {"long_frvp_meta", "short_frvp_meta"}

    long_dir = output_dir / "long_frvp_meta"
    long_report = json.loads((long_dir / "report.json").read_text(encoding="utf-8"))

    assert long_report["target_construction"]["type"] == "synthetic_any_positive_union"
    assert long_report["target_construction"]["source_target_columns"] == [
        "label_long_frvp_reversal",
        "label_long_frvp_continuation",
    ]

    long_positive_union = len({8, 10, 16, 18, 24, 26, 32, 34, 40, 42, 48, 50, 56, 58, 64, 66, 72, 74, 80, 82, 88, 90})
    assert long_report["row_counts"]["rows_positive"] == long_positive_union
    assert long_report["row_counts"]["rows_negative"] > 0

    all_long_rows = pd.concat(
        [
            pd.read_csv(long_dir / "train.csv"),
            pd.read_csv(long_dir / "val.csv"),
            pd.read_csv(long_dir / "test.csv"),
        ],
        ignore_index=True,
    )
    assert int(all_long_rows["target"].sum()) == long_positive_union
    assert float(all_long_rows.loc[all_long_rows["target"] == 1, "sample_weight"].max()) == 2.5


def test_preprocessing_builds_ict_direct_and_meta_targets(tmp_path: Path) -> None:
    dataset_path = _write_ict_dataset(tmp_path, include_metadata=True)
    output_dir = tmp_path / "prepared_ict"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=[
                "label_long_ict_reversal",
                "label_long_ict_continuation",
                "label_long_ict_meta",
            ],
            min_usable_rows=20,
            min_train_rows=10,
            min_positive_samples=3,
            top_n_features=10,
        )
    )
    summary = pipeline.run(dataset_path, output_dir)

    assert set(summary["targets"]) == {
        "long_ict_reversal",
        "long_ict_continuation",
        "long_ict_meta",
    }

    meta_dir = output_dir / "long_ict_meta"
    meta_report = json.loads((meta_dir / "report.json").read_text(encoding="utf-8"))
    meta_features = json.loads((meta_dir / "features.json").read_text(encoding="utf-8"))
    reversal_features = json.loads((output_dir / "long_ict_reversal" / "features.json").read_text(encoding="utf-8"))
    continuation_features = json.loads((output_dir / "long_ict_continuation" / "features.json").read_text(encoding="utf-8"))

    assert meta_report["target_construction"]["type"] == "direct_column"
    assert meta_report["target_construction"]["source_target_column"] == "label_long_ict_meta"
    assert "htf_confluence_long_ict_meta" in meta_features["features"]
    assert "htf_confluence_long_ict_reversal" in reversal_features["features"]
    assert "htf_confluence_long_ict_continuation" in continuation_features["features"]
    assert "ict_total_confluence_1atr" in meta_features["features"]
    assert "ict_bull_sweep_plus_fvg" in reversal_features["features"]

    long_positive_union = len({8, 16, 18, 24, 32, 34, 48, 50, 64, 66, 80, 82})
    assert meta_report["row_counts"]["rows_positive"] == long_positive_union

    all_meta_rows = pd.concat(
        [
            pd.read_csv(meta_dir / "train.csv"),
            pd.read_csv(meta_dir / "val.csv"),
            pd.read_csv(meta_dir / "test.csv"),
        ],
        ignore_index=True,
    )
    assert int(all_meta_rows["target"].sum()) == long_positive_union
    assert float(all_meta_rows.loc[all_meta_rows["target"] == 1, "sample_weight"].max()) == 2.4


def test_preprocessing_synthetic_union_keeps_component_local_negative_rows_when_other_family_is_excluded(tmp_path: Path) -> None:
    dataset_path = tmp_path / "synthetic_component_local_union.csv"
    metadata_path = tmp_path / "synthetic_component_local_union.metadata.json"

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-04-01 00:00:00", periods=8, freq="5min"),
            "open": np.linspace(5000.0, 5001.4, 8),
            "high": np.linspace(5000.3, 5001.7, 8),
            "low": np.linspace(4999.7, 5001.1, 8),
            "close": np.linspace(5000.1, 5001.5, 8),
            "volume": np.linspace(1000, 1070, 8),
            "label_long_frvp_reversal": [1, 0, 0, 0, 0, 0, 0, 0],
            "label_long_frvp_continuation": [0, 0, 1, 0, 0, 0, 0, 0],
            "sample_weight_long_frvp_reversal": [2.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "sample_weight_long_frvp_continuation": [0.0, 0.0, 2.2, 1.6, 0.0, 0.0, 0.0, 0.0],
            "label_quality_long_frvp_reversal": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "label_quality_long_frvp_continuation": [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
            "exclude_long_frvp_reversal": [False, False, True, True, False, False, False, False],
            "exclude_long_frvp_continuation": [True, True, False, False, False, False, False, False],
            "neg_ok_long_frvp_reversal": [True, True, False, False, False, False, False, False],
            "neg_ok_long_frvp_continuation": [False, False, True, True, False, False, False, False],
            "warmup_mask": [False] * 8,
            "feature_signal": np.linspace(0.0, 1.0, 8),
        }
    )
    df.to_csv(dataset_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": ["feature_signal"],
                "timezone_contract": {"canonical_timezone": "UTC"},
                "config": {"drop_warmup_rows": False, "warmup_rows": 0, "fillna_numeric": False},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_frvp_meta"],
            min_usable_rows=2,
            min_train_rows=1,
            min_positive_samples=1,
            top_n_features=3,
        )
    )
    summary = pipeline.run(dataset_path, tmp_path / "prepared_component_local_union")
    report = json.loads(
        (tmp_path / "prepared_component_local_union" / "long_frvp_meta" / "report.json").read_text(encoding="utf-8")
    )

    assert summary["targets"]["long_frvp_meta"]["positive_rate"] == 0.5
    assert report["row_counts"]["rows_positive"] == 2
    assert report["row_counts"]["rows_negative"] == 2
    assert report["row_counts"]["rows_usable"] == 4


def test_preprocessing_appends_target_specific_htf_confluence_for_frvp_targets(tmp_path: Path) -> None:
    dataset_path = _write_frvp_dataset(tmp_path, include_metadata=True)
    output_dir = tmp_path / "prepared_frvp"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=[
                "label_long_frvp_reversal",
                "label_long_frvp_continuation",
            ],
            min_usable_rows=20,
            min_train_rows=10,
            min_positive_samples=3,
            top_n_features=10,
        )
    )
    pipeline.run(dataset_path, output_dir)

    long_reversal_features = json.loads((output_dir / "long_frvp_reversal" / "features.json").read_text(encoding="utf-8"))
    long_continuation_features = json.loads((output_dir / "long_frvp_continuation" / "features.json").read_text(encoding="utf-8"))

    assert "htf_confluence_long_frvp_reversal" in long_reversal_features["features"]
    assert "htf_confluence_long_frvp_continuation" in long_continuation_features["features"]
    assert "frvp_dist_poc_session_atr" in long_reversal_features["features"]
    assert "frvp_open_type" in long_continuation_features["features"]


def test_preprocessing_fallback_excludes_frvp_leakage_like_and_contract_lineage_columns(tmp_path: Path) -> None:
    dataset_path = _write_frvp_dataset(tmp_path, include_metadata=False)
    output_dir = tmp_path / "prepared_frvp_fallback"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_frvp_reversal"],
            min_usable_rows=20,
            min_train_rows=10,
            min_positive_samples=3,
            top_n_features=10,
        )
    )
    pipeline.run(dataset_path, output_dir)

    features_payload = json.loads((output_dir / "long_frvp_reversal" / "features.json").read_text(encoding="utf-8"))
    selected = set(features_payload["features"])

    assert "htf_confluence_long_frvp_reversal" in selected
    assert "frvp_dist_poc_session_atr" in selected
    assert "frvp_day_type" in selected

    for forbidden in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ts_event",
        "symbol",
        "instrument_id",
        "contract_symbol",
        "contract_expiration",
        "is_roll_boundary",
        "bars_since_roll",
        "market_day_close",
        "market_day_index",
        "in_roll_bracket",
        "frvp_atr",
        "frvp_structural_atr",
        "label_quality_long_frvp_reversal",
        "sample_weight_long_frvp_reversal",
        "exclude_long_frvp_reversal",
        "neg_ok_long_frvp_reversal",
    ):
        assert forbidden not in selected


def test_preprocessing_can_skip_loading_time_column(tmp_path: Path) -> None:
    dataset_path = _write_frvp_dataset(tmp_path, include_metadata=True)
    output_dir = tmp_path / "prepared_no_time_column"

    pipeline = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=["label_long_frvp_reversal"],
            load_time_column=False,
            min_usable_rows=20,
            min_train_rows=10,
            min_positive_samples=3,
            top_n_features=10,
        )
    )
    summary = pipeline.run(dataset_path, output_dir)
    report = json.loads((output_dir / "long_frvp_reversal" / "report.json").read_text(encoding="utf-8"))

    assert summary["targets"]["long_frvp_reversal"]["usable_rows"] > 0
    assert report["split_date_ranges"] == {"train": None, "val": None, "test": None}
