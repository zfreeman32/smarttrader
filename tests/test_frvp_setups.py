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
        "frvp_dist_vah_atr": 0.4,
        "frvp_dist_val_atr": 0.4,
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
            "close": 105.2,
            "high": 105.8,
            "low": 104.3,
            "frvp_above_vah": 1,
            "frvp_dist_vah_atr": -0.12,
            "frvp_dist_val_atr": 0.92,
            "volume_zscore_50": 2.2,
            "frvp_open_drive_flag": 1,
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
            "frvp_dist_vah_atr": 0.03,
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
            "frvp_profile_shape": 1,
            "frvp_open_type": 1,
            "frvp_hvn_above_close": 1.1,
            "frvp_hvn_below_close": 0.7,
        }
    )
    rows[8].update(
        {
            "close": 100.1,
            "high": 100.3,
            "low": 99.9,
            "frvp_profile_shape": 1,
            "frvp_open_type": 1,
            "frvp_hvn_above_close": 1.0,
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
