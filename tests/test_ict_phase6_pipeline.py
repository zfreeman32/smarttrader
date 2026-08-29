from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ict.pipelines import ICTESPrimaryPhase06Config, run_es_primary_phase06  # noqa: E402
from ict.pipelines.es_primary_phase06 import _load_ict_label_dataset  # noqa: E402


def _marks(rows: int, hits: set[int]) -> list[int]:
    return [1 if index in hits else 0 for index in range(rows)]


def _write_phase06_inputs(
    base_dir: Path,
    *,
    unnamed_label_datetime: bool = False,
) -> tuple[Path, Path, Path]:
    rows = 72
    timestamps = pd.date_range("2024-06-03 13:30:00", periods=rows, freq="5min", tz="UTC")

    long_reversal_idx = {8, 16, 24, 32, 40, 48, 56, 64}
    long_continuation_idx = {10, 18, 26, 34, 42, 50, 58, 66}
    long_meta_idx = long_reversal_idx | long_continuation_idx

    feature_df = pd.DataFrame(
        {
            "datetime": timestamps,
            "ict_total_confluence_1atr": np.sin(np.arange(rows) / 5.0) + [0.35 if i in long_meta_idx else 0.0 for i in range(rows)],
            "ict_zone_balance": np.cos(np.arange(rows) / 7.0) + [0.20 if i in long_reversal_idx else -0.10 for i in range(rows)],
            "ict_bull_sweep_plus_fvg": [int(i in long_reversal_idx or i in {7, 23, 39, 55}) for i in range(rows)],
            "ict_sweep_plus_choch": [int(i % 9 == 0) for i in range(rows)],
            "ict_displacement_ratio": [((i % 11) - 5) / 5.0 for i in range(rows)],
            "ignored_column": np.arange(rows),
        }
    )

    feature_csv_path = base_dir / "ict_phase06_features.csv"
    feature_metadata_path = base_dir / "ict_phase06_features.metadata.json"
    feature_df.to_csv(feature_csv_path, index=False)
    feature_metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": [
                    "ict_total_confluence_1atr",
                    "ict_zone_balance",
                    "ict_bull_sweep_plus_fvg",
                    "ict_sweep_plus_choch",
                    "ict_displacement_ratio",
                ],
                "timezone_contract": {
                    "source_timezone": "UTC",
                    "canonical_timezone": "UTC",
                    "feature_clock_timezone": "America/New_York",
                    "market_close_timezone": "America/New_York",
                },
                "source_path": "artifacts/ict_phase02/features.csv",
                "source_metadata_file": "artifacts/ict_phase02/features.metadata.json",
                "upstream_source_path": "data/futures_data/ES-5m-tagged.csv",
                "upstream_metadata_file": "data/futures_data/ES-5m-tagged.metadata.json",
                "upstream_bar_timestamp_semantics": "bar_open",
                "upstream_timezone_contract": {
                    "source_timezone": "UTC",
                    "canonical_timezone": "UTC",
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

    long_reversal = _marks(rows, long_reversal_idx)
    long_continuation = _marks(rows, long_continuation_idx)
    long_meta = _marks(rows, long_meta_idx)

    labels_df = pd.DataFrame(
        {
            "datetime": timestamps,
            "open": 5310.0 + np.arange(rows, dtype=float) * 0.25,
            "high": 5310.5 + np.arange(rows, dtype=float) * 0.25,
            "low": 5309.5 + np.arange(rows, dtype=float) * 0.25,
            "close": 5310.2 + np.arange(rows, dtype=float) * 0.25,
            "volume": 1200 + np.arange(rows, dtype=int) * 8,
            "label_long_ict_reversal": long_reversal,
            "label_long_ict_continuation": long_continuation,
            "label_long_ict_meta": long_meta,
            "sample_weight_long_ict_reversal": [2.3 if value else 1.0 for value in long_reversal],
            "sample_weight_long_ict_continuation": [1.9 if value else 1.0 for value in long_continuation],
            "sample_weight_long_ict_meta": [
                2.3 if index in long_reversal_idx else (1.9 if index in long_continuation_idx else 1.0)
                for index in range(rows)
            ],
            "label_quality_long_ict_reversal": [0.82 if value else 0.0 for value in long_reversal],
            "label_quality_long_ict_continuation": [0.74 if value else 0.0 for value in long_continuation],
            "label_quality_long_ict_meta": [
                0.82 if index in long_reversal_idx else (0.74 if index in long_continuation_idx else 0.0)
                for index in range(rows)
            ],
            "exclude_long_ict_reversal": [False] * rows,
            "exclude_long_ict_continuation": [False] * rows,
            "exclude_long_ict_meta": [False] * rows,
            "neg_ok_long_ict_reversal": [not value for value in long_reversal],
            "neg_ok_long_ict_continuation": [not value for value in long_continuation],
            "neg_ok_long_ict_meta": [not value for value in long_meta],
            "concurrency_long_ict_reversal": [1 if index in long_reversal_idx else 0 for index in range(rows)],
            "concurrency_long_ict_meta": [1 if index in long_meta_idx else 0 for index in range(rows)],
            "htf_confluence_long_ict_reversal": [
                int(index in long_reversal_idx or index in {7, 25, 41, 57}) for index in range(rows)
            ],
            "htf_confluence_long_ict_meta": [
                int(index in long_meta_idx or index in {9, 27, 45, 63}) for index in range(rows)
            ],
            "warmup_mask": [index < 4 for index in range(rows)],
        }
    )

    if unnamed_label_datetime:
        labels_df = labels_df.rename(columns={"datetime": "Unnamed: 0"})

    labels_csv_path = base_dir / "ict_phase06_labels.csv"
    labels_df.to_csv(labels_csv_path, index=False)

    event_rows = []
    for signal_index in sorted(long_reversal_idx):
        event_rows.append(
            {
                "event_direction": "long",
                "label_family": "ict_reversal",
                "signal_index": signal_index,
                "barrier_end_index": signal_index + 5,
                "max_holding_bars": 6,
                "tb_outcome": "tp",
                "excluded": False,
                "event_time": timestamps[signal_index].isoformat(),
                "setup_type": "reversal",
            }
        )
    for signal_index in sorted(long_continuation_idx):
        event_rows.append(
            {
                "event_direction": "long",
                "label_family": "ict_continuation",
                "signal_index": signal_index,
                "barrier_end_index": signal_index + 7,
                "max_holding_bars": 8,
                "tb_outcome": "tp",
                "excluded": False,
                "event_time": timestamps[signal_index].isoformat(),
                "setup_type": "continuation",
            }
        )
    pd.DataFrame(event_rows).to_csv(base_dir / "ict_es_events.csv", index=False)
    return feature_csv_path, feature_metadata_path, labels_csv_path


def test_ict_phase6_entry_path_writes_clean_merge_and_prepared_artifacts(tmp_path: Path) -> None:
    feature_csv_path, feature_metadata_path, labels_csv_path = _write_phase06_inputs(tmp_path)

    summary = run_es_primary_phase06(
        ICTESPrimaryPhase06Config(
            feature_csv_path=feature_csv_path,
            labels_csv_path=labels_csv_path,
            feature_metadata_path=feature_metadata_path,
            base_dir=tmp_path / "artifacts",
            run_id="ict_phase06_case",
            target_columns=("label_long_ict_reversal", "label_long_ict_meta"),
            min_usable_rows=20,
            min_train_rows=10,
            min_positive_samples=3,
            top_n_features=8,
        )
    )

    artifact_root = Path(summary["artifact_root"])
    prepared_root = Path(summary["prepared_root"])
    cleaned_csv_path = Path(summary["artifacts"]["cleaned_dataset_csv"])
    cleaned_metadata_path = Path(summary["artifacts"]["cleaned_dataset_metadata"])
    phase06_summary_path = Path(summary["artifacts"]["phase06_summary"])

    assert artifact_root == tmp_path / "artifacts" / "ict_phase06_case"
    assert prepared_root == artifact_root / "phase04_prepared" / "prepared"
    assert cleaned_csv_path.exists()
    assert cleaned_metadata_path.exists()
    assert phase06_summary_path.exists()
    assert (prepared_root / "summary.json").exists()
    assert (prepared_root / "long_ict_reversal" / "train.csv").exists()
    assert (prepared_root / "long_ict_meta" / "train.csv").exists()

    cleaned_df = pd.read_csv(cleaned_csv_path)
    assert "open" not in cleaned_df.columns
    assert "high" not in cleaned_df.columns
    assert "low" not in cleaned_df.columns
    assert "close" not in cleaned_df.columns
    assert "volume" not in cleaned_df.columns
    assert "ict_total_confluence_1atr" in cleaned_df.columns
    assert "htf_confluence_long_ict_reversal" in cleaned_df.columns
    assert "htf_confluence_long_ict_meta" in cleaned_df.columns
    assert cleaned_df["warmup_mask"].sum() == 4

    cleaned_metadata = json.loads(cleaned_metadata_path.read_text(encoding="utf-8"))
    prepared_contract = cleaned_metadata["prepared_contract"]
    assert prepared_contract["label_source_path"] == str(labels_csv_path)
    assert prepared_contract["target_context_columns"] == [
        "htf_confluence_long_ict_reversal",
        "htf_confluence_long_ict_meta",
    ]
    assert "open" in prepared_contract["raw_price_columns_removed"]

    prepared_summary = json.loads((prepared_root / "summary.json").read_text(encoding="utf-8"))
    assert set(prepared_summary["targets"]) == {"long_ict_reversal", "long_ict_meta"}
    assert summary["leakage_control"]["available"] is True
    assert summary["leakage_control"]["target_split_embargo_bars"]["long_ict_reversal"] == 9
    assert summary["leakage_control"]["target_split_embargo_bars"]["long_ict_meta"] == 11

    reversal_features = json.loads((prepared_root / "long_ict_reversal" / "features.json").read_text(encoding="utf-8"))
    meta_features = json.loads((prepared_root / "long_ict_meta" / "features.json").read_text(encoding="utf-8"))
    reversal_report = json.loads((prepared_root / "long_ict_reversal" / "report.json").read_text(encoding="utf-8"))
    meta_report = json.loads((prepared_root / "long_ict_meta" / "report.json").read_text(encoding="utf-8"))
    assert "htf_confluence_long_ict_reversal" in reversal_features["features"]
    assert "htf_confluence_long_ict_meta" in meta_features["features"]
    assert "ict_total_confluence_1atr" in meta_features["features"]
    assert "source_row_idx" in pd.read_csv(prepared_root / "long_ict_meta" / "train.csv").columns
    assert reversal_report["split_geometry"]["applied_embargo_bars"] == 9
    assert meta_report["split_geometry"]["applied_embargo_bars"] == 11
    assert all(
        boundary.get("passes_embargo", True)
        for boundary in reversal_report["split_geometry"]["boundaries"]
        if boundary.get("available")
    )


def test_ict_phase6_label_loader_accepts_unnamed_datetime_column(tmp_path: Path) -> None:
    _, _, labels_csv_path = _write_phase06_inputs(tmp_path, unnamed_label_datetime=True)

    loaded = _load_ict_label_dataset(labels_csv_path)

    assert "datetime" in loaded.columns
    assert loaded["datetime"].is_monotonic_increasing
    dtype_text = str(loaded["datetime"].dtype)
    assert dtype_text.startswith("datetime64[")
    assert "UTC" in dtype_text
    assert "label_long_ict_reversal" in loaded.columns
