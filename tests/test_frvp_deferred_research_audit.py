from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frvp_deferred_research_audit import (  # noqa: E402
    BranchSpec,
    annotate_selected_trades,
    attach_feature_context,
    build_pooling_summary,
    dedupe_trade_event_annotations,
    summarize_trade_groups,
)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def test_attach_feature_context_marks_directional_naked_vpoc_reach() -> None:
    events = pd.DataFrame(
        [
            {
                "entry_datetime": _ts("2026-07-01 14:00:00"),
                "direction": "long",
                "label_family": "frvp_reversal",
                "setup_type": 1,
                "setup_name": "Setup 1",
                "excluded": False,
                "flag_roll_bracket": False,
                "flag_roll_span": False,
                "label_quality": 0.7,
                "setup_confidence": 0.8,
                "tb_outcome": "tp",
            },
            {
                "entry_datetime": _ts("2026-07-01 14:05:00"),
                "direction": "short",
                "label_family": "frvp_reversal",
                "setup_type": 4,
                "setup_name": "Setup 4",
                "excluded": False,
                "flag_roll_bracket": True,
                "flag_roll_span": False,
                "label_quality": 0.4,
                "setup_confidence": 0.5,
                "tb_outcome": "sl",
            },
        ]
    )
    features = pd.DataFrame(
        [
            {
                "entry_datetime": _ts("2026-07-01 14:00:00"),
                "frvp_setup_type": 1,
                "frvp_setup_side": 1,
                "frvp_naked_vpoc_dist_above_atr": 1.2,
                "frvp_naked_vpoc_dist_below_atr": 3.0,
                "frvp_naked_vpoc_age_sessions": 2.0,
                "frvp_naked_vpoc_count": 1.0,
                "frvp_setup_confidence_rule": 0.8,
            },
            {
                "entry_datetime": _ts("2026-07-01 14:05:00"),
                "frvp_setup_type": 4,
                "frvp_setup_side": -1,
                "frvp_naked_vpoc_dist_above_atr": 2.4,
                "frvp_naked_vpoc_dist_below_atr": 1.1,
                "frvp_naked_vpoc_age_sessions": 5.0,
                "frvp_naked_vpoc_count": 2.0,
                "frvp_setup_confidence_rule": 0.5,
            },
        ]
    )

    annotated = attach_feature_context(events, features, setup6b_max_distance_atr=1.5)

    assert annotated.loc[0, "directional_naked_vpoc_distance_atr"] == 1.2
    assert bool(annotated.loc[0, "directional_naked_vpoc_in_reach"]) is True
    assert bool(annotated.loc[1, "directional_naked_vpoc_in_reach"]) is True
    assert bool(annotated.loc[1, "any_naked_vpoc_in_reach"]) is True


def test_dedupe_trade_event_annotations_prefers_usable_row() -> None:
    duplicated = pd.DataFrame(
        [
            {
                "entry_datetime": _ts("2026-07-01 14:00:00"),
                "direction": "long",
                "label_family": "frvp_reversal",
                "excluded": True,
                "label_quality": 0.9,
            },
            {
                "entry_datetime": _ts("2026-07-01 14:00:00"),
                "direction": "long",
                "label_family": "frvp_reversal",
                "excluded": False,
                "label_quality": 0.4,
            },
        ]
    )

    deduped = dedupe_trade_event_annotations(duplicated)

    assert len(deduped) == 1
    assert bool(deduped.iloc[0]["excluded"]) is False


def test_annotate_selected_trades_and_pooling_summary_capture_setup_heterogeneity() -> None:
    event_context = pd.DataFrame(
        [
            {
                "entry_datetime": _ts("2026-07-01 14:00:00"),
                "direction": "long",
                "label_family": "frvp_reversal",
                "setup_type": 1,
                "setup_name": "Setup 1",
                "barrier_family": "mean_reversion",
                "label_quality": 0.7,
                "setup_confidence": 0.8,
                "flag_roll_bracket": False,
                "flag_roll_span": False,
                "directional_naked_vpoc_distance_atr": 1.2,
                "directional_naked_vpoc_in_reach": True,
                "any_naked_vpoc_in_reach": True,
                "frvp_naked_vpoc_count": 1.0,
                "frvp_naked_vpoc_age_sessions": 2.0,
                "tb_outcome": "tp",
            },
            {
                "entry_datetime": _ts("2026-07-01 14:05:00"),
                "direction": "long",
                "label_family": "frvp_reversal",
                "setup_type": 4,
                "setup_name": "Setup 4",
                "barrier_family": "failed_auction",
                "label_quality": 0.4,
                "setup_confidence": 0.5,
                "flag_roll_bracket": True,
                "flag_roll_span": False,
                "directional_naked_vpoc_distance_atr": 2.4,
                "directional_naked_vpoc_in_reach": False,
                "any_naked_vpoc_in_reach": False,
                "frvp_naked_vpoc_count": 0.0,
                "frvp_naked_vpoc_age_sessions": 0.0,
                "tb_outcome": "sl",
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {"entry_datetime": _ts("2026-07-01 14:00:00"), "net_pnl_units": 150.0, "direction": "long"},
            {"entry_datetime": _ts("2026-07-01 14:05:00"), "net_pnl_units": -75.0, "direction": "long"},
        ]
    )
    spec = BranchSpec(
        run_id="long_reversal_fullspan_baseline",
        label="Long reversal full-span control",
        model_id="frvp_long_reversal_xgb_v1",
        backtest_root=Path("dummy"),
        label_family="frvp_reversal",
        direction="long",
    )

    annotated = annotate_selected_trades(trades, event_context, spec=spec)
    setup_breakdown = summarize_trade_groups(
        annotated,
        group_columns=("run_id", "label", "model_id", "setup_type", "setup_name"),
    )
    pooling = build_pooling_summary(setup_breakdown)

    assert annotated["annotation_matched"].tolist() == [True, True]
    assert setup_breakdown["trade_count"].sum() == 2
    assert bool(pooling.iloc[0]["mixed_expectancy_signs"]) is True
    assert bool(pooling.iloc[0]["split_recommended"]) is True
