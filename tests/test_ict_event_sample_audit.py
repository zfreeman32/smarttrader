from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ict.reports.event_sample_audit import build_ict_event_sample_audit, render_ict_event_sample_audit_markdown  # noqa: E402


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label_family": "ict_reversal",
                "setup_type": "sweep_reclaim",
                "setup_side": 1,
                "event_direction": "long",
                "event_time": "2024-01-03T14:30:00Z",
                "signal_index": 10,
                "htf_context": "aligned_discount",
                "stop_reference": 99.0,
                "target_reference": 104.0,
                "entry_index": 11,
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 104.0,
                "barrier_family": "reversal",
                "horizon_scale": 1.10,
                "rr_ratio": 4.0,
                "tb_outcome": "tp",
                "tb_bars_held": 3,
                "exit_reason": "target_touch",
                "label_quality": 0.85,
                "excluded": False,
                "exclude_reasons": "",
            },
            {
                "label_family": "ict_continuation",
                "setup_type": "displacement_continuation_after_raid",
                "setup_side": -1,
                "event_direction": "short",
                "event_time": "2024-01-03T15:00:00Z",
                "signal_index": 20,
                "htf_context": "neutral",
                "stop_reference": 101.0,
                "target_reference": 102.0,
                "entry_index": 21,
                "entry_price": 100.0,
                "stop_price": 101.0,
                "target_price": 97.0,
                "barrier_family": "continuation",
                "horizon_scale": 0.95,
                "rr_ratio": 3.0,
                "target_hit_index": 22,
                "tb_outcome": "tp",
                "tb_bars_held": 5,
                "exit_reason": "continuation_trailing_stop",
                "label_quality": 0.72,
                "excluded": False,
                "exclude_reasons": "",
            },
            {
                "label_family": "ict_continuation",
                "setup_type": "premium_discount_continuation",
                "setup_side": 1,
                "event_direction": "long",
                "event_time": "2024-01-03T16:00:00Z",
                "signal_index": 30,
                "htf_context": "aligned_discount",
                "stop_reference": 99.5,
                "target_reference": 103.5,
                "entry_index": 31,
                "entry_price": 100.0,
                "stop_price": 99.5,
                "target_price": 103.5,
                "barrier_family": "continuation",
                "horizon_scale": 1.05,
                "rr_ratio": 7.0,
                "tb_outcome": "skip",
                "tb_bars_held": 0,
                "exit_reason": "",
                "label_quality": 0.60,
                "excluded": True,
                "exclude_reasons": "macro_event_window",
            },
            {
                "label_family": "ict_reversal",
                "setup_type": "ifvg_reversal",
                "setup_side": -1,
                "event_direction": "short",
                "event_time": "2024-01-03T16:30:00Z",
                "signal_index": 40,
                "htf_context": "",
                "stop_reference": 99.0,
                "target_reference": 102.0,
                "entry_index": 41,
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 102.0,
                "barrier_family": "reversal",
                "horizon_scale": 1.25,
                "rr_ratio": 1.0,
                "tb_outcome": "skip",
                "tb_bars_held": 0,
                "exit_reason": "",
                "label_quality": 0.20,
                "excluded": True,
                "exclude_reasons": "invalid_barrier_geometry",
            },
        ]
    )


def test_build_ict_event_sample_audit_summarizes_geometry_and_contexts() -> None:
    events = _sample_events()
    diagnostics = {
        "total_events_sampled": 4,
        "usable_events": 2,
        "events_excluded_macro_event_window": 1,
        "events_excluded_invalid_barrier_geometry": 1,
        "events_excluded_ambiguous_5m_without_1m": 0,
        "continuation_trail_activations": 1,
    }

    summary, review_rows = build_ict_event_sample_audit(events, diagnostics=diagnostics)

    assert summary["headline"]["total_events"] == 4
    assert summary["headline"]["usable_events"] == 2
    assert summary["headline"]["positive_events"] == 2
    assert summary["headline"]["base_rate_pct"] == 100.0

    overall = summary["reference_geometry"]["overall"]
    assert overall["raw_reference_valid_pct"] == 50.0
    assert overall["final_geometry_valid_pct"] == 75.0
    assert overall["target_adjusted_pct"] == 25.0

    continuation = summary["continuation_management"]
    assert continuation["usable_events"] == 1
    assert continuation["target_activation_events"] == 1
    assert continuation["target_activation_share_pct"] == 100.0

    exclusion_counts = {row["value"]: row["count"] for row in summary["exclusions"]["exclude_reason_counts"]}
    assert exclusion_counts["macro_event_window"] == 1
    assert exclusion_counts["invalid_barrier_geometry"] == 1

    branch_rows = summary["barrier_geometry"]["by_branch"]
    long_reversal = next(
        row for row in branch_rows if row["event_direction"] == "long" and row["group_value"] == "ict_reversal"
    )
    assert long_reversal["median_rr_ratio"] == 4.0
    assert long_reversal["median_stop_distance_ticks"] == 4.0

    assert set(review_rows["review_bucket"]) >= {
        "raw_reference_side_failure",
        "missing_htf_context",
        "excluded_window_sample",
    }


def test_render_ict_event_sample_audit_markdown_includes_key_sections() -> None:
    summary, _ = build_ict_event_sample_audit(_sample_events(), diagnostics={"usable_events": 2})

    markdown = render_ict_event_sample_audit_markdown(summary)

    assert "# ICT Event Sample Audit" in markdown
    assert "## Reference Geometry" in markdown
    assert "Raw reference pair valid" in markdown
    assert "## Continuation Management" in markdown
