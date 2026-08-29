from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ict.reports.spacing_cooldown_diagnostics import build_ict_spacing_cooldown_diagnostics  # noqa: E402


def _build_events(
    *,
    setup_type: str,
    setup_side: int,
    count: int,
    signal_step: int,
    first_barrier_bars: int,
    label_family: str,
    include_excluded: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(count):
        signal_index = idx * signal_step
        entry_index = signal_index + 1
        first_barrier_index = entry_index + first_barrier_bars - 1
        row = {
            "label_family": label_family,
            "setup_type": setup_type,
            "setup_side": setup_side,
            "signal_index": signal_index,
            "entry_index": entry_index,
            "target_hit_index": first_barrier_index,
            "barrier_end_index": first_barrier_index,
            "tb_outcome": "tp" if idx % 2 == 0 else "timeout_loss",
            "tb_bars_held": first_barrier_bars,
        }
        if include_excluded:
            row["excluded"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_spacing_diagnostics_raises_continuation_spacing_when_overlap_is_high() -> None:
    events = _build_events(
        setup_type="premium_discount_continuation",
        setup_side=1,
        count=60,
        signal_step=12,
        first_barrier_bars=16,
        label_family="ict_continuation",
    )

    summary, setup_frame = build_ict_spacing_cooldown_diagnostics(
        events,
        current_spacing={"premium_discount_continuation": 6},
    )

    row = setup_frame.loc[setup_frame["setup_type"].eq("premium_discount_continuation")].iloc[0]
    assert summary["headline"]["usable_events"] == 60
    assert row["recommended_spacing_bars"] == 12
    assert row["raw_recommended_spacing_bars"] == 12
    assert row["recommendation_status"] == "raise_spacing_from_barrier_timing"


def test_build_spacing_diagnostics_keeps_structural_session_open_spacing() -> None:
    events = _build_events(
        setup_type="session_open_manipulation_pre_ib",
        setup_side=1,
        count=60,
        signal_step=20,
        first_barrier_bars=4,
        label_family="ict_reversal",
    )

    _, setup_frame = build_ict_spacing_cooldown_diagnostics(
        events,
        current_spacing={"session_open_manipulation_pre_ib": 60},
    )

    row = setup_frame.loc[setup_frame["setup_type"].eq("session_open_manipulation_pre_ib")].iloc[0]
    assert row["recommended_spacing_bars"] == 60
    assert row["recommendation_status"] == "structural_once_per_session_keep"


def test_build_spacing_diagnostics_treats_missing_excluded_column_as_usable() -> None:
    events = _build_events(
        setup_type="sweep_reclaim",
        setup_side=1,
        count=60,
        signal_step=16,
        first_barrier_bars=5,
        label_family="ict_reversal",
        include_excluded=False,
    )

    summary, setup_frame = build_ict_spacing_cooldown_diagnostics(
        events,
        current_spacing={"sweep_reclaim": 6},
    )

    row = setup_frame.loc[setup_frame["setup_type"].eq("sweep_reclaim")].iloc[0]
    assert summary["headline"]["usable_events"] == 60
    assert row["recommended_spacing_bars"] == 6
    assert row["recommendation_status"] == "keep_current_aligned"
