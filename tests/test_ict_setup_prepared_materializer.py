from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_ict_setup_prepared_root import materialize_ict_setup_prepared_root  # noqa: E402


def _write_base_phase04_root(base_root: Path) -> None:
    phase04 = base_root / "phase04_prepared"
    phase04.mkdir(parents=True)

    rows = 60
    row_number = np.arange(rows)
    source_row_idx = np.arange(100, 100 + rows, dtype=np.int64)
    dataset = pd.DataFrame(
        {
            "source_row_idx": source_row_idx,
            "datetime": pd.date_range("2024-01-03T14:30:00Z", periods=rows, freq="5min"),
            "ict_feature_alpha": np.sin(row_number / 3.0),
            "ict_feature_beta": np.cos(row_number / 5.0),
            "ict_feature_cycle": row_number % 7,
            "warmup_mask": False,
        }
    )
    dataset.to_csv(phase04 / "ict_es_phase06_merged_dataset.csv", index=False)
    (phase04 / "ict_es_phase06_merged_dataset.metadata.json").write_text(
        json.dumps(
            {
                "feature_columns": [
                    "ict_feature_alpha",
                    "ict_feature_beta",
                    "ict_feature_cycle",
                ],
                "config": {"swing_window": 3},
                "prepared_contract": {
                    "target_context_columns": [],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_classic_breaker_phase03(phase03_dir: Path) -> list[int]:
    phase03_dir.mkdir(parents=True)
    signal_rows = [102, 110, 118, 126, 134, 142]
    events = pd.DataFrame(
        [
            {
                "event_direction": "short",
                "label_family": "ict_classic_breaker",
                "setup_type": "classic_breaker",
                "signal_index": signal_row,
                "barrier_end_index": signal_row + 2,
                "max_holding_bars": 8,
                "tb_outcome": "tp" if index < 3 else "sl",
                "excluded": False,
                "event_time": pd.Timestamp("2024-01-03T14:30:00Z")
                + pd.Timedelta(minutes=5 * (signal_row - 100)),
                "label_quality": 0.8 + index * 0.01,
                "sample_weight": 1.2 + index * 0.01,
                "ict_concurrency": index,
                "htf_confluence_flag": index % 2 == 0,
            }
            for index, signal_row in enumerate(signal_rows)
        ]
    )
    events.to_csv(phase03_dir / "ict_es_events.csv", index=False)
    return signal_rows


def test_materialize_ict_setup_prepared_root_supports_classic_breaker_source_split(
    tmp_path: Path,
) -> None:
    base_root = tmp_path / "canonical_feature_root"
    source_phase03_dir = tmp_path / "classic_phase03"
    _write_base_phase04_root(base_root)
    signal_rows = _write_classic_breaker_phase03(source_phase03_dir)

    summary = materialize_ict_setup_prepared_root(
        base_root=base_root,
        source_phase03_dir=source_phase03_dir,
        artifact_base_dir=tmp_path / "artifacts",
        run_id="classic_breaker_phase4_case",
        direction="short",
        min_usable_rows=6,
        min_train_rows=4,
        min_positive_samples=3,
        top_n_features=3,
    )

    prepared_root = Path(summary["artifacts"]["prepared_root"])
    target_dir = prepared_root / "short_ict_classic_breaker"
    wrong_suffix_dir = prepared_root / "short_ict_classic_breaker_classic_breaker"

    assert summary["base_root"] == str(base_root)
    assert summary["source_phase03_dir"] == str(source_phase03_dir)
    assert summary["target_columns"] == ["label_short_ict_classic_breaker"]
    assert summary["counts"]["targets"]["short_ict_classic_breaker"] == {
        "usable_rows": 6,
        "positive_rows": 3,
        "positive_rate": 0.5,
    }
    assert summary["target_split_embargo_bars"]["short_ict_classic_breaker"] == 6
    assert target_dir.exists()
    assert not wrong_suffix_dir.exists()

    report = json.loads((target_dir / "report.json").read_text(encoding="utf-8"))
    train = pd.read_csv(target_dir / "train.csv")
    val = pd.read_csv(target_dir / "val.csv")
    test = pd.read_csv(target_dir / "test.csv")

    assert report["target_name"] == "short_ict_classic_breaker"
    assert report["target_column"] == "label_short_ict_classic_breaker"
    assert report["sample_weight_column"] == "sample_weight_short_ict_classic_breaker"
    assert report["split_geometry"]["applied_embargo_bars"] == 6
    assert set(train["source_row_idx"]).issubset(set(signal_rows))
    assert set(val["source_row_idx"]).issubset(set(signal_rows))
    assert set(test["source_row_idx"]).issubset(set(signal_rows))
    assert int(train["target"].sum()) == 3
