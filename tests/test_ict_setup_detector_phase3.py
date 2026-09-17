from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ict.config.setups import ICTSetupDetectorConfig  # noqa: E402
from ict.setups.detector import detect_ict_setups, summarize_setup_fire_rates  # noqa: E402


def _setup_input(rows: int = 3) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-03 14:30:00", periods=rows, freq="5min", tz="UTC"),
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 101.5),
            "low": np.full(rows, 99.5),
            "close": np.full(rows, 100.75),
            "atr_14": np.full(rows, 1.0),
            "volume": np.full(rows, 100.0),
            "ict_buy_side_sweep": np.zeros(rows, dtype=int),
            "ict_sell_side_sweep": np.zeros(rows, dtype=int),
            "ict_sweep_direction": np.zeros(rows, dtype=int),
            "ict_sweep_level_code": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_sweep_level_value": np.full(rows, np.nan),
            "ict_sweep_penetration_atr": np.full(rows, np.nan),
            "ict_bars_since_buy_side_sweep": np.full(rows, np.nan),
            "ict_bars_since_sell_side_sweep": np.full(rows, np.nan),
            "ict_structure_state": np.zeros(rows, dtype=int),
            "ict_impulse_direction": np.zeros(rows, dtype=int),
            "dist_to_bull_fvg_atr": np.full(rows, np.nan),
            "dist_to_bear_fvg_atr": np.full(rows, np.nan),
            "dist_to_bull_order_block_atr": np.full(rows, np.nan),
            "dist_to_bear_order_block_atr": np.full(rows, np.nan),
            "ict_nearest_bull_fvg_is_ifvg": np.zeros(rows, dtype=int),
            "ict_nearest_bear_fvg_is_ifvg": np.zeros(rows, dtype=int),
            "ict_nearest_bull_fvg_id": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bear_fvg_id": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bull_fvg_lower": np.full(rows, np.nan),
            "ict_nearest_bull_fvg_upper": np.full(rows, np.nan),
            "ict_nearest_bear_fvg_lower": np.full(rows, np.nan),
            "ict_nearest_bear_fvg_upper": np.full(rows, np.nan),
            "ict_nearest_bull_fvg_ce": np.full(rows, np.nan),
            "ict_nearest_bear_fvg_ce": np.full(rows, np.nan),
            "ict_nearest_bull_fvg_created_by_displacement": np.zeros(rows, dtype=int),
            "ict_nearest_bear_fvg_created_by_displacement": np.zeros(rows, dtype=int),
            "ict_nearest_bull_fvg_inversion_index": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bear_fvg_inversion_index": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_bars_since_displacement_bull": np.full(rows, np.nan),
            "ict_bars_since_displacement_bear": np.full(rows, np.nan),
            "ict_latest_bull_displacement_index": np.full(rows, np.nan),
            "ict_latest_bear_displacement_index": np.full(rows, np.nan),
            "ict_latest_bull_displacement_id": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_latest_bear_displacement_id": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_latest_bull_displacement_origin": np.full(rows, np.nan),
            "ict_latest_bear_displacement_origin": np.full(rows, np.nan),
            "ict_displacement_volume_zscore": np.full(rows, 0.0),
            "ict_displacement_score": np.full(rows, 0.0),
            "ict_bars_since_choch_bull": np.full(rows, np.nan),
            "ict_bars_since_choch_bear": np.full(rows, np.nan),
            "ict_bars_since_mss_bull": np.full(rows, np.nan),
            "ict_bars_since_mss_bear": np.full(rows, np.nan),
            "ict_bull_order_block_retest_event": np.zeros(rows, dtype=int),
            "ict_bear_order_block_retest_event": np.zeros(rows, dtype=int),
            "ict_nearest_bull_order_block_id": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bear_order_block_id": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bull_order_block_lower": np.full(rows, np.nan),
            "ict_nearest_bull_order_block_upper": np.full(rows, np.nan),
            "ict_nearest_bear_order_block_lower": np.full(rows, np.nan),
            "ict_nearest_bear_order_block_upper": np.full(rows, np.nan),
            "dist_to_bull_breaker_atr": np.full(rows, np.nan),
            "dist_to_bear_breaker_atr": np.full(rows, np.nan),
            "bull_breaker_age_bars": np.full(rows, np.nan),
            "bear_breaker_age_bars": np.full(rows, np.nan),
            "ict_bull_breaker_retest_count": np.full(rows, np.nan),
            "ict_bear_breaker_retest_count": np.full(rows, np.nan),
            "ict_active_bull_breaker_count": np.zeros(rows, dtype=int),
            "ict_active_bear_breaker_count": np.zeros(rows, dtype=int),
            "ict_nearest_bull_breaker_id": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bear_breaker_id": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bull_breaker_source_order_block_id": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bear_breaker_source_order_block_id": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bull_breaker_lower": np.full(rows, np.nan),
            "ict_nearest_bull_breaker_upper": np.full(rows, np.nan),
            "ict_nearest_bear_breaker_lower": np.full(rows, np.nan),
            "ict_nearest_bear_breaker_upper": np.full(rows, np.nan),
            "ict_nearest_bull_breaker_formed_index": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bear_breaker_formed_index": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bull_breaker_source_order_block_formed_index": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_nearest_bear_breaker_source_order_block_formed_index": pd.Series([pd.NA] * rows, dtype="Int64"),
            "ict_bull_breaker_retest_event": np.zeros(rows, dtype=int),
            "ict_bear_breaker_retest_event": np.zeros(rows, dtype=int),
            "ict_bull_breaker_created_event": np.zeros(rows, dtype=int),
            "ict_bear_breaker_created_event": np.zeros(rows, dtype=int),
            "ict_discount_zone": np.zeros(rows, dtype=int),
            "ict_premium_zone": np.zeros(rows, dtype=int),
            "ict_in_ote_band": np.zeros(rows, dtype=int),
            "ict_is_rth": np.zeros(rows, dtype=int),
            "ict_ib_complete": np.ones(rows, dtype=int),
            "ict_lunch_lull_flag": np.zeros(rows, dtype=int),
            "ict_close_ramp_flag": np.zeros(rows, dtype=int),
            "ict_session_phase_code": np.full(rows, 3, dtype=int),
            "ict_session_vwap": np.full(rows, 100.5),
            "ict_dol_level_up": np.full(rows, 104.0),
            "ict_dol_level_down": np.full(rows, 97.0),
            "ict_latest_swing_high": np.full(rows, 104.0),
            "ict_latest_swing_low": np.full(rows, 97.0),
            "session_date": pd.Timestamp("2024-01-03"),
        }
    )
    return frame


def _mark_bull_classic_breaker(frame: pd.DataFrame, row: int = 4) -> None:
    frame.loc[row - 1, "close"] = 101.2
    frame.loc[row, "high"] = 101.2
    frame.loc[row, "low"] = 100.4
    frame.loc[row, "close"] = 101.1
    frame.loc[row, "ict_bull_breaker_retest_event"] = 1
    frame.loc[row, "ict_active_bull_breaker_count"] = 1
    frame.loc[row, "ict_nearest_bull_breaker_id"] = 5
    frame.loc[row, "ict_nearest_bull_breaker_source_order_block_id"] = 17
    frame.loc[row, "ict_nearest_bull_breaker_lower"] = 100.5
    frame.loc[row, "ict_nearest_bull_breaker_upper"] = 101.0
    frame.loc[row, "ict_nearest_bull_breaker_formed_index"] = row - 1
    frame.loc[row, "ict_nearest_bull_breaker_source_order_block_formed_index"] = row - 3
    frame.loc[row, "bull_breaker_age_bars"] = 1.0
    frame.loc[row, "ict_bull_breaker_retest_count"] = 1.0
    frame.loc[row, "dist_to_bull_breaker_atr"] = 0.0
    frame.loc[row, "ict_bars_since_mss_bull"] = 1.0
    frame.loc[row, "ict_latest_bull_displacement_index"] = row - 1
    frame.loc[row, "ict_latest_bull_displacement_id"] = 31
    frame.loc[row, "ict_discount_zone"] = 1


def _mark_bear_classic_breaker(frame: pd.DataFrame, row: int = 4) -> None:
    frame.loc[row - 1, "close"] = 98.8
    frame.loc[row, "high"] = 99.4
    frame.loc[row, "low"] = 98.7
    frame.loc[row, "close"] = 98.9
    frame.loc[row, "ict_bear_breaker_retest_event"] = 1
    frame.loc[row, "ict_active_bear_breaker_count"] = 1
    frame.loc[row, "ict_nearest_bear_breaker_id"] = 6
    frame.loc[row, "ict_nearest_bear_breaker_source_order_block_id"] = 18
    frame.loc[row, "ict_nearest_bear_breaker_lower"] = 99.0
    frame.loc[row, "ict_nearest_bear_breaker_upper"] = 99.5
    frame.loc[row, "ict_nearest_bear_breaker_formed_index"] = row - 1
    frame.loc[row, "ict_nearest_bear_breaker_source_order_block_formed_index"] = row - 3
    frame.loc[row, "bear_breaker_age_bars"] = 1.0
    frame.loc[row, "ict_bear_breaker_retest_count"] = 1.0
    frame.loc[row, "dist_to_bear_breaker_atr"] = 0.0
    frame.loc[row, "ict_bars_since_mss_bear"] = 1.0
    frame.loc[row, "ict_latest_bear_displacement_index"] = row - 1
    frame.loc[row, "ict_latest_bear_displacement_id"] = 32
    frame.loc[row, "ict_premium_zone"] = 1


def test_detect_ict_setups_fires_generic_sweep_reclaim_long() -> None:
    frame = _setup_input(rows=1)
    frame.loc[0, "ict_sell_side_sweep"] = 1
    frame.loc[0, "ict_sweep_direction"] = -1
    frame.loc[0, "ict_sweep_level_code"] = 13
    frame.loc[0, "ict_sweep_level_value"] = 99.0
    frame.loc[0, "ict_sweep_penetration_atr"] = 0.5
    frame.loc[0, "ict_discount_zone"] = 1

    result = detect_ict_setups(frame, ICTSetupDetectorConfig(instrument="es"))

    assert bool(result.loc[0, "fired"]) is True
    assert str(result.loc[0, "setup_type"]) == "sweep_reclaim"
    assert str(result.loc[0, "setup_family"]) == "reversal"
    assert int(result.loc[0, "setup_side"]) == 1
    assert float(result.loc[0, "anchor_level"]) == 99.0
    assert str(result.loc[0, "reference_level_type"]) == "prior_rth_low"
    assert str(result.loc[0, "sweep_type"]) == "sell_side"
    assert float(result.loc[0, "target_reference"]) == 104.0


def test_detect_ict_setups_prefers_pre_ib_open_manipulation_over_generic_sweep() -> None:
    frame = _setup_input(rows=1)
    frame.loc[0, "ict_is_rth"] = 1
    frame.loc[0, "ict_ib_complete"] = 0
    frame.loc[0, "ict_sell_side_sweep"] = 1
    frame.loc[0, "ict_sweep_direction"] = -1
    frame.loc[0, "ict_sweep_level_code"] = 14
    frame.loc[0, "ict_sweep_level_value"] = 98.5

    result = detect_ict_setups(frame, ICTSetupDetectorConfig(instrument="es"))

    assert bool(result.loc[0, "fired"]) is True
    assert str(result.loc[0, "setup_type"]) == "session_open_manipulation_pre_ib"
    assert int(result.loc[0, "setup_side"]) == 1
    assert str(result.loc[0, "reference_level_type"]) == "overnight_low"


def test_detect_ict_setups_fires_ifvg_reversal_long() -> None:
    frame = _setup_input(rows=3)
    frame.loc[:, "ict_structure_state"] = 1
    frame.loc[:, "low"] = [100.9, 100.8, 100.7]
    frame.loc[:, "high"] = [101.3, 101.4, 101.5]
    frame.loc[:, "close"] = [101.1, 101.2, 101.25]
    frame.loc[:, "dist_to_bull_fvg_atr"] = [0.1, 0.0, 0.0]
    frame.loc[:, "ict_nearest_bull_fvg_is_ifvg"] = 1
    frame.loc[:, "ict_nearest_bull_fvg_id"] = 7
    frame.loc[:, "ict_nearest_bull_fvg_lower"] = 100.5
    frame.loc[:, "ict_nearest_bull_fvg_upper"] = 101.5
    frame.loc[:, "ict_nearest_bull_fvg_ce"] = 101.0
    frame.loc[:, "ict_nearest_bull_fvg_inversion_index"] = 0

    result = detect_ict_setups(frame, ICTSetupDetectorConfig(instrument="es"))

    assert bool(result.loc[1, "fired"]) is True
    assert str(result.loc[1, "setup_type"]) == "ifvg_reversal"
    assert int(result.loc[1, "setup_side"]) == 1
    assert int(result.loc[1, "fvg_id"]) == 7
    assert float(result.loc[1, "ce_price"]) == 101.0


def test_detect_ict_setups_deduplicates_consecutive_same_ifvg_signal() -> None:
    frame = _setup_input(rows=3)
    frame.loc[:, "ict_structure_state"] = 1
    frame.loc[:, "low"] = [100.9, 100.8, 100.8]
    frame.loc[:, "high"] = [101.3, 101.4, 101.4]
    frame.loc[:, "close"] = [101.1, 101.2, 101.2]
    frame.loc[:, "dist_to_bull_fvg_atr"] = [0.1, 0.0, 0.0]
    frame.loc[:, "ict_nearest_bull_fvg_is_ifvg"] = 1
    frame.loc[:, "ict_nearest_bull_fvg_id"] = 11
    frame.loc[:, "ict_nearest_bull_fvg_lower"] = 100.5
    frame.loc[:, "ict_nearest_bull_fvg_upper"] = 101.5
    frame.loc[:, "ict_nearest_bull_fvg_ce"] = 101.0
    frame.loc[:, "ict_nearest_bull_fvg_inversion_index"] = 0

    result = detect_ict_setups(frame, ICTSetupDetectorConfig(instrument="es"))

    assert result["fired"].tolist() == [False, True, False]


def test_detect_ict_setups_fires_premium_discount_continuation_and_summary() -> None:
    frame = _setup_input(rows=1)
    frame.loc[0, "ict_structure_state"] = 1
    frame.loc[0, "ict_impulse_direction"] = 1
    frame.loc[0, "ict_discount_zone"] = 1
    frame.loc[0, "ict_in_ote_band"] = 1
    frame.loc[0, "dist_to_bull_fvg_atr"] = 0.2
    frame.loc[0, "ict_nearest_bull_fvg_id"] = 21
    frame.loc[0, "ict_nearest_bull_fvg_lower"] = 99.5
    frame.loc[0, "ict_nearest_bull_fvg_upper"] = 100.8
    frame.loc[0, "ict_nearest_bull_fvg_ce"] = 100.15
    frame.loc[0, "ict_latest_bull_displacement_origin"] = 99.4

    result = detect_ict_setups(frame, ICTSetupDetectorConfig(instrument="es"))
    summary = summarize_setup_fire_rates(result)

    assert bool(result.loc[0, "fired"]) is True
    assert str(result.loc[0, "setup_type"]) == "premium_discount_continuation"
    assert str(result.loc[0, "setup_family"]) == "continuation"
    assert int(result.loc[0, "setup_side"]) == 1
    assert not summary.empty
    assert str(summary.iloc[0]["setup_type"]) == "premium_discount_continuation"


def test_classic_breaker_is_not_enabled_by_default() -> None:
    frame = _setup_input(rows=5)
    _mark_bull_classic_breaker(frame)

    result = detect_ict_setups(frame, ICTSetupDetectorConfig(instrument="es"))

    assert "classic_breaker" not in set(result["setup_type"].astype(str))
    assert bool(result.loc[4, "fired"]) is False
    assert result.attrs["research_fired_events"].empty


def test_classic_breaker_fires_as_explicit_long_research_setup() -> None:
    frame = _setup_input(rows=5)
    _mark_bull_classic_breaker(frame)

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(instrument="es", enabled_setup_types=("classic_breaker",)),
    )

    assert bool(result.loc[4, "fired"]) is True
    assert str(result.loc[4, "setup_type"]) == "classic_breaker"
    assert str(result.loc[4, "setup_family"]) == "reversal"
    assert int(result.loc[4, "setup_side"]) == 1
    assert int(result.loc[4, "order_block_id"]) == 17
    assert int(result.loc[4, "breaker_id"]) == 5
    assert int(result.loc[4, "breaker_source_order_block_id"]) == 17
    assert int(result.loc[4, "breaker_activation_index"]) == 3
    assert int(result.loc[4, "breaker_source_order_block_formed_index"]) == 1
    assert int(result.loc[4, "breaker_retest_index"]) == 4
    assert float(result.loc[4, "breaker_zone_lower"]) == 100.5
    assert float(result.loc[4, "breaker_zone_upper"]) == 101.0
    assert int(result.loc[4, "displacement_index"]) == 3


def test_classic_breaker_fires_as_explicit_short_research_setup() -> None:
    frame = _setup_input(rows=5)
    _mark_bear_classic_breaker(frame)

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(instrument="es", enabled_setup_types=("classic_breaker",)),
    )

    assert bool(result.loc[4, "fired"]) is True
    assert str(result.loc[4, "setup_type"]) == "classic_breaker"
    assert int(result.loc[4, "setup_side"]) == -1
    assert int(result.loc[4, "breaker_id"]) == 6
    assert int(result.loc[4, "breaker_source_order_block_id"]) == 18
    assert int(result.loc[4, "breaker_activation_index"]) == 3
    assert int(result.loc[4, "breaker_retest_index"]) == 4


def test_classic_breaker_rejects_same_bar_activation() -> None:
    frame = _setup_input(rows=5)
    _mark_bull_classic_breaker(frame)
    frame.loc[4, "ict_nearest_bull_breaker_formed_index"] = 4
    frame.loc[4, "bull_breaker_age_bars"] = 0.0

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(instrument="es", enabled_setup_types=("classic_breaker",)),
    )

    assert bool(result.loc[4, "fired"]) is False


def test_classic_breaker_requires_side_correct_rejection_close() -> None:
    frame = _setup_input(rows=5)
    _mark_bull_classic_breaker(frame)
    frame.loc[4, "close"] = 100.8

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(instrument="es", enabled_setup_types=("classic_breaker",)),
    )

    assert bool(result.loc[4, "fired"]) is False


def test_classic_breaker_can_allow_non_rejection_close_for_expanded_research() -> None:
    frame = _setup_input(rows=5)
    _mark_bull_classic_breaker(frame)
    frame.loc[4, "close"] = 100.8

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(
            instrument="es",
            enabled_setup_types=("classic_breaker",),
            classic_breaker_require_rejection_close=False,
        ),
    )

    assert bool(result.loc[4, "fired"]) is True
    assert str(result.loc[4, "setup_type"]) == "classic_breaker"


def test_classic_breaker_first_retest_filter_blocks_later_retests() -> None:
    frame = _setup_input(rows=5)
    _mark_bull_classic_breaker(frame)
    frame.loc[4, "ict_bull_breaker_retest_count"] = 2.0

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(instrument="es", enabled_setup_types=("classic_breaker",)),
    )

    assert bool(result.loc[4, "fired"]) is False


def test_classic_breaker_can_allow_later_retests_for_expanded_research() -> None:
    frame = _setup_input(rows=5)
    _mark_bull_classic_breaker(frame)
    frame.loc[4, "ict_bull_breaker_retest_count"] = 2.0

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(
            instrument="es",
            enabled_setup_types=("classic_breaker",),
            classic_breaker_first_retest_only=False,
        ),
    )

    assert bool(result.loc[4, "fired"]) is True
    assert str(result.loc[4, "setup_type"]) == "classic_breaker"


def test_classic_breaker_stale_breaker_does_not_fire() -> None:
    frame = _setup_input(rows=5)
    _mark_bull_classic_breaker(frame)
    frame.loc[4, "bull_breaker_age_bars"] = 121.0

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(instrument="es", enabled_setup_types=("classic_breaker",), classic_breaker_max_age=120),
    )

    assert bool(result.loc[4, "fired"]) is False


def test_classic_breaker_requires_source_order_block_lineage() -> None:
    frame = _setup_input(rows=5)
    _mark_bull_classic_breaker(frame)
    frame.loc[4, "ict_nearest_bull_breaker_source_order_block_id"] = pd.NA

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(instrument="es", enabled_setup_types=("classic_breaker",)),
    )

    assert bool(result.loc[4, "fired"]) is False


def test_classic_breaker_mss_and_displacement_without_breaker_retest_do_not_fire() -> None:
    frame = _setup_input(rows=5)
    frame.loc[4, "ict_bars_since_mss_bull"] = 1.0
    frame.loc[4, "ict_latest_bull_displacement_index"] = 3
    frame.loc[4, "ict_latest_bull_displacement_id"] = 31

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(instrument="es", enabled_setup_types=("classic_breaker",)),
    )

    assert bool(result.loc[4, "fired"]) is False


def test_classic_breaker_rejects_future_displacement_index() -> None:
    frame = _setup_input(rows=5)
    _mark_bull_classic_breaker(frame)
    frame.loc[4, "ict_latest_bull_displacement_index"] = 6

    result = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(instrument="es", enabled_setup_types=("classic_breaker",)),
    )

    assert bool(result.loc[4, "fired"]) is False


def test_classic_breaker_sidecar_does_not_change_primary_canonical_selection() -> None:
    frame = _setup_input(rows=5)
    frame.loc[0, "ict_sell_side_sweep"] = 1
    frame.loc[0, "ict_sweep_direction"] = -1
    frame.loc[0, "ict_sweep_level_code"] = 13
    frame.loc[0, "ict_sweep_level_value"] = 99.0
    frame.loc[0, "ict_sweep_penetration_atr"] = 0.5
    _mark_bull_classic_breaker(frame)

    canonical = detect_ict_setups(frame, ICTSetupDetectorConfig(instrument="es"))
    with_classic = detect_ict_setups(
        frame,
        ICTSetupDetectorConfig(
            instrument="es",
            enabled_setup_types=tuple(ICTSetupDetectorConfig().enabled_setup_types) + ("classic_breaker",),
        ),
    )

    pd.testing.assert_series_equal(canonical["fired"], with_classic["fired"])
    pd.testing.assert_series_equal(canonical["setup_type"], with_classic["setup_type"])
    research = with_classic.attrs["research_fired_events"]
    assert len(research) == 1
    assert str(research.iloc[0]["setup_type"]) == "classic_breaker"
    assert int(research.iloc[0]["breaker_id"]) == 5
