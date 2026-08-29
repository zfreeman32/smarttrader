from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frvp.setups.detector import detect_frvp_setups, summarize_setup_fire_rates  # noqa: E402


def _base_setup_row(timestamp: pd.Timestamp, session_date: pd.Timestamp) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "session_date": session_date,
        "contract_id": "ESH24",
        "open": 100.0,
        "high": 100.5,
        "low": 99.5,
        "close": 100.0,
        "atr_14": 10.0,
        "frvp_profile_shape": 0,
        "frvp_open_type": 0,
        "frvp_open_drive_flag": 0,
        "frvp_in_va": 0,
        "frvp_above_vah": 0,
        "frvp_below_val": 0,
        "frvp_dist_vah_atr": 1.0,
        "frvp_dist_val_atr": 1.0,
        "frvp_dist_nearest_lvn_atr": 1.0,
        "frvp_hvn_above_close": np.nan,
        "frvp_hvn_below_close": np.nan,
        "volume_zscore_50": 0.0,
        "displacement_bullish": 0,
        "displacement_bearish": 0,
        "bars_since_sweep_high": np.nan,
        "bars_since_sweep_low": np.nan,
    }


def _setup_fixture() -> pd.DataFrame:
    session_date = pd.Timestamp("2024-01-03")
    start = pd.Timestamp("2024-01-03 09:30:00", tz="UTC")
    rows = [_base_setup_row(start + pd.Timedelta(minutes=5 * index), session_date) for index in range(9)]

    rows[1].update(
        {
            "close": 103.0,
            "high": 103.4,
            "low": 102.5,
            "frvp_in_va": 1,
            "frvp_dist_vah_atr": 0.1,
            "frvp_dist_val_atr": 0.7,
            "volume_zscore_50": 0.2,
        }
    )
    rows[2].update(
        {
            "close": 105.6,
            "high": 105.8,
            "low": 104.3,
            "frvp_above_vah": 1,
            "frvp_dist_vah_atr": -0.12,
            "frvp_dist_val_atr": 0.92,
            "volume_zscore_50": 2.2,
            "frvp_open_drive_flag": 1,
            "displacement_bullish": 1,
        }
    )
    rows[3].update(
        {
            "close": 104.2,
            "high": 104.8,
            "low": 103.95,
            "frvp_above_vah": 1,
            "frvp_dist_vah_atr": -0.02,
            "frvp_dist_val_atr": 0.82,
            "volume_zscore_50": 0.6,
        }
    )
    rows[4].update(
        {
            "close": 105.0,
            "high": 105.5,
            "low": 104.6,
            "frvp_above_vah": 1,
            "frvp_dist_vah_atr": -0.1,
            "frvp_dist_val_atr": 0.9,
            "volume_zscore_50": 0.4,
        }
    )
    rows[5].update(
        {
            "close": 103.7,
            "high": 104.1,
            "low": 103.2,
            "frvp_in_va": 1,
            "frvp_dist_vah_atr": 0.08,
            "frvp_dist_val_atr": 0.77,
            "volume_zscore_50": 0.3,
            "bars_since_sweep_high": 2.0,
        }
    )
    rows[6].update(
        {
            "close": 99.0,
            "high": 100.0,
            "low": 98.6,
            "frvp_profile_shape": 1,
            "frvp_open_type": 1,
            "frvp_dist_vah_atr": 0.5,
            "frvp_dist_val_atr": 0.3,
            "frvp_dist_nearest_lvn_atr": 0.2,
            "displacement_bullish": 1,
        }
    )
    rows[7].update(
        {
            "close": 100.0,
            "high": 100.2,
            "low": 99.8,
            "frvp_profile_shape": 0,
            "frvp_open_type": 0,
            "frvp_in_va": 1,
            "frvp_hvn_above_close": 1.4,
            "frvp_hvn_below_close": 0.85,
        }
    )
    rows[8].update(
        {
            "close": 100.1,
            "high": 100.3,
            "low": 99.9,
            "frvp_profile_shape": 0,
            "frvp_open_type": 0,
            "frvp_in_va": 1,
            "frvp_hvn_above_close": 1.35,
            "frvp_hvn_below_close": 0.8,
        }
    )
    return pd.DataFrame(rows)


def test_detect_frvp_setups_emits_expected_types_sides_and_confidence() -> None:
    frame = _setup_fixture()
    result = detect_frvp_setups(frame)

    assert result["fired"].tolist() == [False, True, True, True, False, True, True, True, False]
    assert result["setup_type"].tolist() == [0, 1, 3, 2, 0, 4, 5, 6, 0]
    assert result["setup_side"].tolist() == [0, -1, 1, 1, 0, -1, 1, -1, 0]
    assert (result.loc[result["fired"], "confidence"] > 0.0).all()
    assert (result.loc[result["fired"], "confidence"] <= 1.0).all()


def test_failed_auction_detection_does_not_peek_at_future_bar() -> None:
    base = _setup_fixture().iloc[:6].copy()
    future = _base_setup_row(
        pd.Timestamp("2024-01-03 10:05:00", tz="UTC"),
        pd.Timestamp("2024-01-03"),
    )
    future.update(
        {
            "close": 110.0,
            "high": 112.0,
            "low": 103.5,
            "frvp_above_vah": 1,
            "frvp_dist_vah_atr": -0.6,
            "frvp_dist_val_atr": 1.4,
            "volume_zscore_50": 3.5,
        }
    )
    extended = pd.concat([base, pd.DataFrame([future])], ignore_index=True)

    base_result = detect_frvp_setups(base)
    extended_result = detect_frvp_setups(extended)

    assert bool(base_result.iloc[-1]["fired"]) is True
    assert int(base_result.iloc[-1]["setup_type"]) == 4
    assert base_result.iloc[-1].to_dict() == extended_result.iloc[len(base_result) - 1].to_dict()


def test_fire_rate_summary_counts_fires_per_session_side() -> None:
    first = _setup_fixture()
    second = _setup_fixture().copy()
    second["timestamp"] = second["timestamp"] + pd.Timedelta(days=1)
    second["session_date"] = pd.Timestamp("2024-01-04")
    combined = pd.concat([first, second], ignore_index=True)

    summary = summarize_setup_fire_rates(combined)
    setup1_short = summary.loc[(summary["setup_type"] == 1) & (summary["setup_side"] == -1)].iloc[0]
    setup6_short = summary.loc[(summary["setup_type"] == 6) & (summary["setup_side"] == -1)].iloc[0]

    assert int(setup1_short["total_fires"]) == 2
    assert float(setup1_short["avg_fires_per_session_side"]) == 1.0
    assert bool(setup1_short["within_target_band"]) is True
    assert int(setup6_short["total_fires"]) == 2
    assert float(setup6_short["avg_fires_per_session_side"]) == 1.0


def test_setup6_requires_balanced_in_value_non_drive_context() -> None:
    frame = _setup_fixture()
    frame.loc[7, "frvp_in_va"] = 0

    result = detect_frvp_setups(frame)

    assert bool(result.iloc[7]["fired"]) is False
    assert int(result.iloc[7]["setup_type"]) == 0


def test_setup6_rejects_ambiguous_hvn_distances() -> None:
    frame = _setup_fixture()
    frame.loc[7, "frvp_hvn_above_close"] = 1.05
    frame.loc[7, "frvp_hvn_below_close"] = 0.90

    result = detect_frvp_setups(frame)

    assert bool(result.iloc[7]["fired"]) is False
    assert int(result.iloc[7]["setup_type"]) == 0


def test_setup6_rejects_high_volume_or_displacement_context() -> None:
    frame = _setup_fixture()
    frame.loc[7, "volume_zscore_50"] = 0.8
    frame.loc[7, "displacement_bearish"] = 1

    result = detect_frvp_setups(frame)

    assert bool(result.iloc[7]["fired"]) is False
    assert int(result.iloc[7]["setup_type"]) == 0


def test_setup1_allows_small_edge_reentry_when_opened_in_value() -> None:
    row = _base_setup_row(
        pd.Timestamp("2024-01-03 10:10:00", tz="UTC"),
        pd.Timestamp("2024-01-03"),
    )
    row.update(
        {
            "frvp_in_va": 0,
            "frvp_above_vah": 1,
            "frvp_dist_vah_atr": -0.12,
            "frvp_dist_val_atr": 0.9,
            "volume_zscore_50": 0.2,
        }
    )

    result = detect_frvp_setups(pd.DataFrame([row]))

    assert bool(result.iloc[0]["fired"]) is True
    assert int(result.iloc[0]["setup_type"]) == 1
    assert int(result.iloc[0]["setup_side"]) == -1


def test_setup2_accepts_later_retest_with_retuned_window() -> None:
    session_date = pd.Timestamp("2024-01-03")
    start = pd.Timestamp("2024-01-03 09:30:00", tz="UTC")
    rows = [_base_setup_row(start + pd.Timedelta(minutes=5 * index), session_date) for index in range(12)]

    rows[1].update(
        {
            "close": 105.6,
            "high": 105.8,
            "low": 104.3,
            "frvp_above_vah": 1,
            "frvp_dist_vah_atr": -0.12,
            "frvp_dist_val_atr": 0.92,
            "volume_zscore_50": 1.4,
            "displacement_bullish": 1,
        }
    )
    rows[11].update(
        {
            "close": 104.2,
            "high": 104.7,
            "low": 103.95,
            "frvp_profile_shape": 1,
            "frvp_dist_vah_atr": -0.02,
            "frvp_dist_val_atr": 0.82,
            "volume_zscore_50": 0.3,
        }
    )

    result = detect_frvp_setups(pd.DataFrame(rows))

    assert bool(result.iloc[11]["fired"]) is True
    assert int(result.iloc[11]["setup_type"]) == 2
    assert int(result.iloc[11]["setup_side"]) == 1


def test_setup3_uses_retuned_volume_threshold() -> None:
    row = _base_setup_row(
        pd.Timestamp("2024-01-03 10:15:00", tz="UTC"),
        pd.Timestamp("2024-01-03"),
    )
    row.update(
        {
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 104.8,
            "frvp_profile_shape": 1,
            "frvp_above_vah": 1,
            "frvp_dist_vah_atr": -0.12,
            "frvp_dist_val_atr": 0.92,
            "volume_zscore_50": 1.3,
            "displacement_bullish": 1,
        }
    )

    result = detect_frvp_setups(pd.DataFrame([row]))

    assert bool(result.iloc[0]["fired"]) is True
    assert int(result.iloc[0]["setup_type"]) == 3
    assert int(result.iloc[0]["setup_side"]) == 1


def test_setup3_requires_real_displacement_close_efficiency() -> None:
    row = _base_setup_row(
        pd.Timestamp("2024-01-03 10:20:00", tz="UTC"),
        pd.Timestamp("2024-01-03"),
    )
    row.update(
        {
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 102.2,
            "frvp_profile_shape": 1,
            "frvp_above_vah": 1,
            "frvp_dist_vah_atr": -0.08,
            "frvp_dist_val_atr": 0.92,
            "volume_zscore_50": 1.6,
            "displacement_bullish": 1,
        }
    )

    result = detect_frvp_setups(pd.DataFrame([row]))

    assert bool(result.iloc[0]["fired"]) is False
    assert int(result.iloc[0]["setup_type"]) == 0


def test_setup4_requires_same_side_sweep() -> None:
    rows = _setup_fixture().iloc[:6].copy()
    rows.loc[5, "bars_since_sweep_high"] = np.nan
    rows.loc[5, "bars_since_sweep_low"] = 2.0

    result = detect_frvp_setups(rows)

    assert bool(result.iloc[5]["fired"]) is False
    assert int(result.iloc[5]["setup_type"]) == 0


def test_setup4_rejects_high_volume_reentry() -> None:
    rows = _setup_fixture().iloc[:6].copy()
    rows.loc[5, "volume_zscore_50"] = 1.1

    result = detect_frvp_setups(rows)

    assert bool(result.iloc[5]["fired"]) is False
    assert int(result.iloc[5]["setup_type"]) == 0


def test_setup4_requires_measurable_reentry_depth_inside_value() -> None:
    rows = _setup_fixture().iloc[:6].copy()
    rows.loc[5, "frvp_dist_vah_atr"] = 0.03

    result = detect_frvp_setups(rows)

    assert bool(result.iloc[5]["fired"]) is False
    assert int(result.iloc[5]["setup_type"]) == 0


def test_setup2_keeps_near_edge_hold_and_blocks_failed_auction_flip() -> None:
    session_date = pd.Timestamp("2024-01-03")
    start = pd.Timestamp("2024-01-03 09:30:00", tz="UTC")
    rows = [_base_setup_row(start + pd.Timedelta(minutes=5 * index), session_date) for index in range(4)]

    rows[1].update(
        {
            "close": 105.6,
            "high": 105.8,
            "low": 104.3,
            "frvp_above_vah": 1,
            "frvp_dist_vah_atr": -0.12,
            "frvp_dist_val_atr": 0.92,
            "volume_zscore_50": 1.4,
            "displacement_bullish": 1,
        }
    )
    rows[2].update(
        {
            "close": 103.9,
            "high": 104.3,
            "low": 103.7,
            "frvp_in_va": 1,
            "frvp_dist_vah_atr": 0.04,
            "frvp_dist_val_atr": 0.74,
            "volume_zscore_50": 0.2,
            "bars_since_sweep_high": 1.0,
        }
    )

    result = detect_frvp_setups(pd.DataFrame(rows))

    assert bool(result.iloc[2]["fired"]) is True
    assert int(result.iloc[2]["setup_type"]) == 2
    assert int(result.iloc[2]["setup_side"]) == 1
