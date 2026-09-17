from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ict.reports.breaker_state_audit import (  # noqa: E402
    BREAKER_REQUIRED_COLUMNS,
    build_ict_breaker_state_audit,
    render_ict_breaker_state_audit_markdown,
)


def _sample_breaker_surface() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=8, freq="5min", tz="UTC"),
            "open": [100.0, 99.8, 99.5, 100.7, 100.2, 101.4, 100.6, 100.9],
            "high": [100.2, 100.0, 100.8, 101.0, 100.7, 101.6, 101.0, 101.1],
            "low": [99.8, 99.0, 99.2, 100.0, 99.9, 100.1, 100.0, 100.3],
            "close": [99.9, 99.2, 100.6, 100.2, 100.5, 100.3, 100.8, 100.7],
        }
    )
    for column in BREAKER_REQUIRED_COLUMNS:
        frame[column] = pd.NA

    for side in ("bull", "bear"):
        frame[f"ict_{side}_breaker_created_event"] = 0
        frame[f"ict_{side}_breaker_retest_event"] = 0
        frame[f"ict_active_{side}_breaker_count"] = 0

    frame.loc[2:, "ict_active_bull_breaker_count"] = 1
    frame.loc[2:, "ict_nearest_bull_breaker_id"] = 10
    frame.loc[2:, "ict_nearest_bull_breaker_source_order_block_id"] = 4
    frame.loc[2:, "ict_nearest_bull_breaker_lower"] = 99.8
    frame.loc[2:, "ict_nearest_bull_breaker_upper"] = 100.4
    frame.loc[2:, "ict_nearest_bull_breaker_formed_index"] = 2
    frame.loc[2:, "ict_nearest_bull_breaker_source_order_block_formed_index"] = 1
    frame.loc[2:, "bull_breaker_age_bars"] = frame.index[2:] - 2
    frame.loc[2:, "ict_bull_breaker_retest_count"] = 0
    frame.loc[2:, "dist_to_bull_breaker_atr"] = 0.0
    frame.loc[2, "ict_bull_breaker_created_event"] = 1
    frame.loc[4, "ict_bull_breaker_retest_event"] = 1
    frame.loc[4, "ict_bull_breaker_retest_count"] = 1

    frame.loc[3:, "ict_active_bear_breaker_count"] = 1
    frame.loc[3:, "ict_nearest_bear_breaker_id"] = 11
    frame.loc[3:, "ict_nearest_bear_breaker_source_order_block_id"] = 5
    frame.loc[3:, "ict_nearest_bear_breaker_lower"] = 100.6
    frame.loc[3:, "ict_nearest_bear_breaker_upper"] = 101.2
    frame.loc[3:, "ict_nearest_bear_breaker_formed_index"] = 3
    frame.loc[3:, "ict_nearest_bear_breaker_source_order_block_formed_index"] = 2
    frame.loc[3:, "bear_breaker_age_bars"] = frame.index[3:] - 3
    frame.loc[3:, "ict_bear_breaker_retest_count"] = 0
    frame.loc[3:, "dist_to_bear_breaker_atr"] = 0.0
    frame.loc[3, "ict_bear_breaker_created_event"] = 1
    frame.loc[6, "ict_bear_breaker_retest_event"] = 1
    frame.loc[6, "ict_bear_breaker_retest_count"] = 1

    return frame


def test_build_ict_breaker_state_audit_summarizes_breadth_and_integrity() -> None:
    summary, review_rows = build_ict_breaker_state_audit(
        _sample_breaker_surface(),
        min_total_retests=2,
        min_side_retests=1,
        min_retest_years=1,
    )

    assert summary["data_contract"]["has_required_breaker_surface"] is True
    assert summary["headline"]["total_created_event_bars"] == 2
    assert summary["headline"]["total_retest_event_bars"] == 2
    assert summary["headline"]["bull_retest_event_bars"] == 1
    assert summary["headline"]["bear_retest_event_bars"] == 1
    assert summary["integrity"]["same_bar_create_and_retest_events"] == 0
    assert summary["integrity"]["retest_after_formation_pct"] == 100.0
    assert summary["readiness"]["decision"] == "ready_for_label_design"
    assert set(review_rows["review_bucket"]) >= {"long_retest_sample", "short_retest_sample", "created_sample"}


def test_build_ict_breaker_state_audit_reports_missing_columns() -> None:
    summary, review_rows = build_ict_breaker_state_audit(pd.DataFrame({"close": [1.0, 2.0]}))

    assert summary["data_contract"]["has_required_breaker_surface"] is False
    assert summary["readiness"]["decision"] == "blocked_missing_columns"
    assert summary["data_contract"]["missing_required_columns"]
    assert review_rows.empty


def test_render_ict_breaker_state_audit_markdown_includes_decision() -> None:
    summary, _ = build_ict_breaker_state_audit(
        _sample_breaker_surface(),
        min_total_retests=2,
        min_side_retests=1,
        min_retest_years=1,
    )

    markdown = render_ict_breaker_state_audit_markdown(summary)

    assert "# ICT Breaker State Audit" in markdown
    assert "## Integrity" in markdown
    assert "ready_for_label_design" in markdown
