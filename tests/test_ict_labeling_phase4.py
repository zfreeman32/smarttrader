from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ict.labeling.ict_labeling_engine import ICTLabelingConfig, build_ict_labels, ict_events_to_frame  # noqa: E402
from ict.setups.setup_types import build_empty_setup_frame  # noqa: E402


def _market_frame(
    rows: list[tuple[float, float, float, float]],
    *,
    start: str = "2024-01-03 14:30:00",
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame.insert(0, "datetime", timestamps)
    frame["atr_14"] = 1.0
    frame["volume"] = 1000.0
    frame["warmup_mask"] = False
    frame["contract_id"] = "ESH24"
    return frame


def _minute_frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame["volume"] = 100.0
    return frame


def _setup_frame(market: pd.DataFrame) -> pd.DataFrame:
    return build_empty_setup_frame(market.index, event_time=market["datetime"]).reset_index(drop=True)


def _configure_event(
    setup: pd.DataFrame,
    row: int,
    *,
    setup_type: str,
    setup_family: str,
    side: int,
    confidence: float,
    anchor_level: float,
    stop_reference: float,
    target_reference: float,
    htf_context: str = "aligned",
    reference_level: float | None = None,
    reference_level_type: str = "liq",
    sweep_type: str = "",
) -> None:
    setup.loc[row, "fired"] = True
    setup.loc[row, "setup_type"] = setup_type
    setup.loc[row, "setup_family"] = setup_family
    setup.loc[row, "setup_side"] = side
    setup.loc[row, "confidence"] = confidence
    setup.loc[row, "anchor_level"] = anchor_level
    setup.loc[row, "entry_price"] = np.nan
    setup.loc[row, "stop_reference"] = stop_reference
    setup.loc[row, "target_reference"] = target_reference
    setup.loc[row, "reference_level"] = anchor_level if reference_level is None else reference_level
    setup.loc[row, "reference_level_type"] = reference_level_type
    setup.loc[row, "sweep_type"] = sweep_type
    setup.loc[row, "htf_context"] = htf_context
    setup.loc[row, "displacement_volume_z"] = 1.5


def _params(**overrides: object) -> ICTLabelingConfig:
    defaults: dict[str, object] = {
        "instrument": "es",
        "reversal_fallback_target_atr": 1.0,
        "continuation_fallback_target_atr": 1.0,
        "reversal_max_bars": 3,
        "session_open_reversal_max_bars": 2,
        "continuation_max_bars": 3,
        "warmup_bars": 0,
    }
    defaults.update(overrides)
    return ICTLabelingConfig(**defaults)


def test_build_ict_labels_materializes_reversal_event_frame_and_tp_label() -> None:
    market = _market_frame(
        [
            (100.0, 100.3, 99.7, 100.0),
            (100.0, 100.5, 99.9, 100.2),
            (100.2, 101.6, 100.1, 101.3),
            (101.3, 101.4, 100.9, 101.1),
        ]
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.90,
        anchor_level=99.8,
        stop_reference=99.6,
        target_reference=101.5,
        htf_context="aligned_bull",
        reference_level_type="prior_rth_low",
        sweep_type="sell_side",
    )

    labeled, diagnostics, events = build_ict_labels(market, params=_params(), setup_output=setup, verbose=False)
    event_frame = ict_events_to_frame(events).set_index("setup_type")

    assert diagnostics["usable_events"] == 1
    assert event_frame.loc["sweep_reclaim", "entry_index"] == 1
    assert float(event_frame.loc["sweep_reclaim", "entry_price"]) == pytest.approx(100.0)
    assert float(event_frame.loc["sweep_reclaim", "stop_price"]) == pytest.approx(99.6)
    assert float(event_frame.loc["sweep_reclaim", "target_price"]) == pytest.approx(101.5)
    assert str(event_frame.loc["sweep_reclaim", "barrier_family"]) == "reversal"
    assert str(event_frame.loc["sweep_reclaim", "tb_outcome"]) == "tp"
    assert int(labeled.iloc[0]["label_long_ict_reversal"]) == 1
    assert int(labeled.iloc[0]["label_long_ict_meta"]) == 1
    assert bool(labeled.iloc[0]["exclude_long_ict_reversal"]) is False
    assert bool(labeled.iloc[0]["neg_ok_long_ict_reversal"]) is True


def test_build_ict_labels_routes_continuation_events_into_continuation_and_meta_columns() -> None:
    market = _market_frame(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.3, 99.9, 100.1),
            (100.1, 101.3, 100.0, 101.2),
            (101.2, 101.3, 100.8, 101.0),
        ]
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="premium_discount_continuation",
        setup_family="continuation",
        side=1,
        confidence=0.80,
        anchor_level=99.9,
        stop_reference=99.5,
        target_reference=101.2,
        htf_context="aligned_bull",
        reference_level_type="discount_fvg",
    )

    labeled, diagnostics, events = build_ict_labels(market, params=_params(), setup_output=setup, verbose=False)
    event_frame = ict_events_to_frame(events).set_index("setup_type")

    assert diagnostics["events_ict_continuation"] == 1
    assert str(event_frame.loc["premium_discount_continuation", "barrier_family"]) == "continuation"
    assert str(event_frame.loc["premium_discount_continuation", "tb_outcome"]) == "tp"
    assert int(labeled.iloc[0]["label_long_ict_continuation"]) == 1
    assert int(labeled.iloc[0]["label_long_ict_meta"]) == 1
    assert int(labeled.iloc[0]["label_long_ict_reversal"]) == 0
    assert bool(labeled.iloc[0]["exclude_long_ict_continuation"]) is False


def test_build_ict_labels_excludes_ambiguous_barriers_without_1m_resolution() -> None:
    market = _market_frame(
        [
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 101.2, 99.6, 100.4),
            (100.4, 100.6, 100.1, 100.5),
            (100.5, 100.6, 100.2, 100.4),
        ]
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.75,
        anchor_level=99.8,
        stop_reference=99.75,
        target_reference=101.0,
        htf_context="aligned_bull",
        sweep_type="sell_side",
    )

    labeled, diagnostics, events = build_ict_labels(market, params=_params(), setup_output=setup, verbose=False)

    assert diagnostics["usable_events"] == 0
    assert diagnostics["events_excluded_ambiguous_5m_without_1m"] == 1
    assert events[0].excluded is True
    assert "ambiguous_5m_without_1m" in (events[0].exclude_reasons or [])
    assert int(labeled.iloc[0]["label_long_ict_reversal"]) == 0
    assert bool(labeled.iloc[0]["exclude_long_ict_reversal"]) is True


def test_build_ict_labels_resolves_ambiguous_barriers_with_1m_ordering() -> None:
    market = _market_frame(
        [
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 101.2, 99.6, 100.4),
            (100.4, 100.6, 100.1, 100.5),
            (100.5, 100.6, 100.2, 100.4),
        ]
    )
    minute = _minute_frame(
        [
            ("2024-01-03 14:35:00+00:00", 100.0, 100.4, 99.9, 100.3),
            ("2024-01-03 14:36:00+00:00", 100.3, 101.05, 100.2, 100.9),
            ("2024-01-03 14:37:00+00:00", 100.9, 101.1, 99.7, 100.1),
            ("2024-01-03 14:38:00+00:00", 100.1, 100.3, 99.9, 100.0),
            ("2024-01-03 14:39:00+00:00", 100.0, 100.2, 99.8, 100.1),
        ]
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.75,
        anchor_level=99.8,
        stop_reference=99.75,
        target_reference=101.0,
        htf_context="aligned_bull",
        sweep_type="sell_side",
    )

    labeled, diagnostics, events = build_ict_labels(
        market,
        params=_params(),
        df_1m=minute,
        setup_output=setup,
        verbose=False,
    )

    assert diagnostics["usable_events"] == 1
    assert events[0].excluded is False
    assert str(events[0].tb_outcome) == "tp"
    assert float(events[0].exit_price or 0.0) == pytest.approx(101.0)
    assert int(labeled.iloc[0]["label_long_ict_reversal"]) == 1
    assert bool(labeled.iloc[0]["exclude_long_ict_reversal"]) is False


def test_build_ict_labels_marks_positive_timeouts_under_pnl_sign_policy() -> None:
    market = _market_frame(
        [
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 100.5, 99.8, 100.1),
            (100.1, 100.6, 100.0, 100.3),
            (100.3, 100.4, 100.1, 100.2),
        ]
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.70,
        anchor_level=99.7,
        stop_reference=99.5,
        target_reference=101.5,
        htf_context="neutral",
        sweep_type="sell_side",
    )

    labeled, _, events = build_ict_labels(
        market,
        params=_params(reversal_max_bars=2, breakeven_plus_cost_atr=0.0),
        setup_output=setup,
        verbose=False,
    )

    assert str(events[0].tb_outcome) == "timeout_profit"
    assert float(events[0].tb_return or 0.0) == pytest.approx(0.3)
    assert int(labeled.iloc[0]["label_long_ict_reversal"]) == 1
    assert int(labeled.iloc[0]["label_long_ict_meta"]) == 1


def test_build_ict_labels_keeps_family_and_meta_weights_distinct_under_overlap() -> None:
    market = _market_frame(
        [
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 100.2, 99.9, 100.1),
            (100.1, 100.3, 100.0, 100.2),
            (100.2, 100.4, 100.1, 100.35),
            (100.35, 100.5, 100.2, 100.4),
        ]
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=1.0,
        anchor_level=99.8,
        stop_reference=99.5,
        target_reference=101.5,
        htf_context="aligned_bull",
        sweep_type="sell_side",
    )
    _configure_event(
        setup,
        1,
        setup_type="premium_discount_continuation",
        setup_family="continuation",
        side=1,
        confidence=1.0,
        anchor_level=99.9,
        stop_reference=99.6,
        target_reference=101.6,
        htf_context="aligned_bull",
        reference_level_type="discount_fvg",
    )

    labeled, _, events = build_ict_labels(
        market,
        params=_params(reversal_max_bars=2, continuation_max_bars=2),
        setup_output=setup,
        verbose=False,
    )
    event_frame = ict_events_to_frame(events).set_index("setup_type")

    continuation_weight = float(labeled.iloc[1]["sample_weight_long_ict_continuation"])
    meta_weight = float(labeled.iloc[1]["sample_weight_long_ict_meta"])

    assert int(labeled.iloc[1]["concurrency_long_ict_continuation"]) == 1
    assert int(labeled.iloc[1]["concurrency_long_ict_meta"]) == 2
    assert continuation_weight > meta_weight
    assert continuation_weight == pytest.approx(meta_weight * 2.0)
    assert float(event_frame.loc["premium_discount_continuation", "sample_weight"]) == pytest.approx(continuation_weight)


def test_build_ict_labels_scales_horizon_by_phase_volatility() -> None:
    market = _market_frame([(100.0, 100.2, 99.8, 100.0)] * 12)
    market["atr_14"] = 1.0
    market.loc[0, "atr_14"] = 0.5
    market.loc[6, "atr_14"] = 2.0
    market["ict_session_phase_code"] = 3

    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.80,
        anchor_level=99.7,
        stop_reference=99.4,
        target_reference=103.0,
        htf_context="aligned_bull",
        sweep_type="sell_side",
    )
    _configure_event(
        setup,
        6,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.80,
        anchor_level=99.7,
        stop_reference=99.4,
        target_reference=103.0,
        htf_context="aligned_bull",
        sweep_type="sell_side",
    )

    _, _, events = build_ict_labels(
        market,
        params=_params(reversal_max_bars=4),
        setup_output=setup,
        verbose=False,
    )
    event_frame = ict_events_to_frame(events).set_index("signal_index")

    assert int(event_frame.loc[0, "max_holding_bars"]) == 7
    assert int(event_frame.loc[6, "max_holding_bars"]) == 3
    assert float(event_frame.loc[0, "horizon_scale"]) > float(event_frame.loc[6, "horizon_scale"])


def test_build_ict_labels_uses_continuation_trailing_exit_after_target_activation() -> None:
    market = _market_frame(
        [
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 100.4, 99.9, 100.2),
            (100.2, 101.25, 100.2, 101.0),
            (101.0, 102.6, 101.8, 102.4),
            (102.4, 102.5, 101.5, 101.7),
            (101.7, 101.9, 101.4, 101.6),
        ]
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="premium_discount_continuation",
        setup_family="continuation",
        side=1,
        confidence=0.85,
        anchor_level=99.9,
        stop_reference=99.5,
        target_reference=101.2,
        htf_context="aligned_bull",
        reference_level_type="discount_fvg",
    )

    labeled, diagnostics, events = build_ict_labels(
        market,
        params=_params(
            continuation_max_bars=2,
            continuation_max_extension_bars=3,
            continuation_trailing_stop_atr=1.0,
        ),
        setup_output=setup,
        verbose=False,
    )
    event = events[0]

    assert diagnostics["continuation_trail_activations"] == 1
    assert event.target_hit_index == 2
    assert event.exit_reason == "continuation_trailing_stop"
    assert event.tb_outcome == "tp"
    assert float(event.exit_price or 0.0) == pytest.approx(101.6)
    assert float(event.exit_price or 0.0) > float(event.target_price or 0.0)
    assert int(labeled.iloc[0]["label_long_ict_continuation"]) == 1


def test_build_ict_labels_excludes_macro_window_events() -> None:
    market = _market_frame(
        [
            (100.0, 100.3, 99.8, 100.0),
            (100.0, 100.4, 99.9, 100.2),
            (100.2, 100.5, 100.0, 100.3),
            (100.3, 100.4, 100.1, 100.2),
        ],
        start="2025-06-18 18:00:00",
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.75,
        anchor_level=99.8,
        stop_reference=99.5,
        target_reference=101.5,
        htf_context="aligned_bull",
        sweep_type="sell_side",
    )

    labeled, diagnostics, events = build_ict_labels(market, params=_params(reversal_max_bars=2), setup_output=setup, verbose=False)

    assert diagnostics["events_excluded_macro_event_window"] == 1
    assert events[0].excluded is True
    assert "macro_event_window" in (events[0].exclude_reasons or [])
    assert bool(labeled.iloc[0]["exclude_long_ict_reversal"]) is True


def test_build_ict_labels_excludes_half_day_events() -> None:
    market = _market_frame(
        [
            (100.0, 100.3, 99.8, 100.0),
            (100.0, 100.4, 99.9, 100.2),
            (100.2, 100.5, 100.0, 100.3),
            (100.3, 100.4, 100.1, 100.2),
        ],
        start="2025-07-03 14:30:00",
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.75,
        anchor_level=99.8,
        stop_reference=99.5,
        target_reference=101.5,
        htf_context="aligned_bull",
        sweep_type="sell_side",
    )

    _, diagnostics, events = build_ict_labels(market, params=_params(reversal_max_bars=2), setup_output=setup, verbose=False)

    assert diagnostics["events_excluded_half_day_or_holiday_window"] == 1
    assert events[0].excluded is True
    assert "half_day_or_holiday_window" in (events[0].exclude_reasons or [])


def test_build_ict_labels_excludes_lunch_dominated_windows() -> None:
    market = _market_frame(
        [(100.0, 100.2, 99.8, 100.0)] * 24,
        start="2024-01-03 16:50:00",
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.70,
        anchor_level=99.7,
        stop_reference=99.5,
        target_reference=103.0,
        htf_context="aligned_bull",
        sweep_type="sell_side",
    )

    _, diagnostics, events = build_ict_labels(market, params=_params(reversal_max_bars=4), setup_output=setup, verbose=False)

    assert diagnostics["events_excluded_lunch_dominated_window"] == 1
    assert events[0].excluded is True
    assert "lunch_dominated_window" in (events[0].exclude_reasons or [])


def test_build_ict_labels_allows_overnight_events_when_thin_session_gate_is_entry_anchored() -> None:
    market = _market_frame(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.3, 99.9, 100.1),
            (100.1, 101.4, 100.0, 101.2),
            (101.2, 101.3, 100.9, 101.0),
        ],
        start="2024-01-03 00:00:00",
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.80,
        anchor_level=99.8,
        stop_reference=99.5,
        target_reference=101.3,
        htf_context="aligned_bull",
        sweep_type="sell_side",
    )

    labeled, diagnostics, events = build_ict_labels(market, params=_params(reversal_max_bars=2), setup_output=setup, verbose=False)

    assert diagnostics["usable_events"] == 1
    assert "events_excluded_thin_session_window" not in diagnostics
    assert events[0].excluded is False
    assert bool(labeled.iloc[0]["exclude_long_ict_reversal"]) is False


def test_build_ict_labels_still_excludes_rth_close_tail_entries() -> None:
    market = _market_frame(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.3, 99.9, 100.1),
            (100.1, 100.4, 100.0, 100.2),
            (100.2, 100.4, 100.1, 100.3),
        ],
        start="2024-01-03 20:25:00",
    )
    setup = _setup_frame(market)
    _configure_event(
        setup,
        0,
        setup_type="sweep_reclaim",
        setup_family="reversal",
        side=1,
        confidence=0.80,
        anchor_level=99.8,
        stop_reference=99.5,
        target_reference=101.2,
        htf_context="aligned_bull",
        sweep_type="sell_side",
    )

    _, diagnostics, events = build_ict_labels(market, params=_params(reversal_max_bars=2), setup_output=setup, verbose=False)

    assert diagnostics["events_excluded_thin_session_window"] == 1
    assert events[0].excluded is True
    assert "thin_session_window" in (events[0].exclude_reasons or [])
