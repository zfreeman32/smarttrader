from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ict.reports.leakage_control import (  # noqa: E402
    build_ict_event_window_frame,
    build_ict_leakage_control_audit,
    compute_average_uniqueness,
    resolve_ict_swing_confirm_bars,
    sequential_bootstrap_sample,
)


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_direction": "long",
                "label_family": "ict_continuation",
                "signal_index": 10,
                "barrier_end_index": 18,
                "max_holding_bars": 12,
                "tb_outcome": "tp",
                "excluded": False,
                "event_time": "2024-01-03T14:30:00Z",
                "setup_type": "premium_discount_continuation",
            },
            {
                "event_direction": "long",
                "label_family": "ict_continuation",
                "signal_index": 20,
                "barrier_end_index": 34,
                "max_holding_bars": 16,
                "tb_outcome": "timeout_loss",
                "excluded": False,
                "event_time": "2024-01-03T15:00:00Z",
                "setup_type": "premium_discount_continuation",
            },
            {
                "event_direction": "long",
                "label_family": "ict_continuation",
                "signal_index": 40,
                "barrier_end_index": 46,
                "max_holding_bars": 8,
                "tb_outcome": "timeout_profit",
                "excluded": False,
                "event_time": "2024-01-03T15:30:00Z",
                "setup_type": "displacement_continuation_after_raid",
            },
            {
                "event_direction": "short",
                "label_family": "ict_reversal",
                "signal_index": 60,
                "barrier_end_index": 66,
                "max_holding_bars": 8,
                "tb_outcome": "sl",
                "excluded": False,
                "event_time": "2024-01-03T16:00:00Z",
                "setup_type": "ifvg_reversal",
            },
        ]
    )


def _write_prepared_target(target_dir: Path, *, train: list[int], val: list[int], test: list[int]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "target_name": target_dir.name,
        "split_counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
    }
    (target_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split_name, rows in {"train": train, "val": val, "test": test}.items():
        pd.DataFrame({"source_row_idx": rows, "target": [1] * len(rows), "sample_weight": [1.0] * len(rows)}).to_csv(
            target_dir / f"{split_name}.csv",
            index=False,
        )


def test_resolve_ict_swing_confirm_bars_prefers_phase02_metadata() -> None:
    metadata = {"config": {"swing_window": 5}}
    assert resolve_ict_swing_confirm_bars(metadata) == 5


def test_build_ict_event_window_frame_synthesizes_meta_targets() -> None:
    frame = build_ict_event_window_frame(_sample_events())
    assert "long_ict_continuation" in set(frame["target_name"])
    assert "long_ict_meta" in set(frame["target_name"])
    assert "short_ict_meta" in set(frame["target_name"])


def test_build_ict_leakage_control_audit_detects_split_boundary_leakage(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    _write_prepared_target(
        prepared_root / "long_ict_continuation",
        train=[10, 20],
        val=[30],
        test=[50],
    )

    summary = build_ict_leakage_control_audit(
        _sample_events(),
        phase02_metadata={"config": {"swing_window": 3}},
        prepared_root=prepared_root,
        target_names=("long_ict_continuation",),
        bootstrap_max_events=10,
    )

    target = summary["targets"][0]
    assert target["recommended_embargo_bars"] == 18
    split_audit = target["prepared_split_audit"]
    assert split_audit["available"] is True
    assert split_audit["passes_all_boundaries"] is False
    boundary = split_audit["boundaries"][0]
    assert boundary["boundary_name"] == "train_to_val"
    assert boundary["overlap_event_count"] == 1
    assert boundary["passes_boundary"] is False


def test_sequential_bootstrap_sample_improves_uniqueness_on_overlapping_events() -> None:
    intervals = pd.DataFrame(
        {
            "signal_index": [0, 1, 2, 20],
            "barrier_end_index": [6, 7, 8, 24],
        }
    )
    chronological = intervals.iloc[:3].reset_index(drop=True)
    bootstrap_idx = sequential_bootstrap_sample(intervals, sample_size=3, random_state=7, replace=False)
    bootstrapped = intervals.iloc[bootstrap_idx].reset_index(drop=True)

    assert len(bootstrap_idx) == 3
    assert compute_average_uniqueness(bootstrapped) >= compute_average_uniqueness(chronological)
