from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frvp.continuity import (  # noqa: E402
    CoordinateMismatchError,
    RollBoundaryError,
    build_continuous_contract,
    build_continuous_contract_from_tagged_series,
)


def _contract_frame(
    timestamps: list[pd.Timestamp],
    closes: list[float],
    volumes: list[float],
) -> pd.DataFrame:
    closes_array = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes_array - 0.5,
            "high": closes_array + 0.5,
            "low": closes_array - 1.0,
            "close": closes_array,
            "volume": np.asarray(volumes, dtype=float),
        }
    )


def _build_synthetic_contracts() -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp]]:
    timestamps = [
        pd.Timestamp("2024-01-02 23:00:00+00:00"),
        pd.Timestamp("2024-01-02 23:05:00+00:00"),
        pd.Timestamp("2024-01-03 23:00:00+00:00"),
        pd.Timestamp("2024-01-03 23:05:00+00:00"),
        pd.Timestamp("2024-01-04 23:00:00+00:00"),
        pd.Timestamp("2024-01-04 23:05:00+00:00"),
        pd.Timestamp("2024-01-05 23:00:00+00:00"),
        pd.Timestamp("2024-01-05 23:05:00+00:00"),
    ]
    contracts = {
        "ESH24": _contract_frame(
            timestamps=timestamps,
            closes=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 105.0],
            volumes=[600.0, 400.0, 500.0, 400.0, 200.0, 200.0, 100.0, 100.0],
        ),
        "ESM24": _contract_frame(
            timestamps=timestamps,
            closes=[106.0, 107.0, 108.0, 109.0, 109.0, 110.0, 110.0, 111.0],
            volumes=[80.0, 120.0, 150.0, 150.0, 600.0, 600.0, 700.0, 800.0],
        ),
    }
    return contracts, timestamps


def test_profiles_stay_within_one_contract_and_cross_roll_slices_fail() -> None:
    contracts, timestamps = _build_synthetic_contracts()
    result = build_continuous_contract(
        contracts,
        initial_lead_contract="ESH24",
        roll_bracket_sessions=2,
    )

    session_three_slice = result.raw_profile_bars.profile_slice(timestamps[4], timestamps[5])
    assert session_three_slice.contract_id == "ESH24"
    assert session_three_slice.bars["contract_id"].nunique() == 1

    with pytest.raises(RollBoundaryError):
        result.raw_profile_bars.profile_slice(timestamps[4], timestamps[6])


def test_absolute_levels_cannot_cross_coordinate_systems() -> None:
    contracts, timestamps = _build_synthetic_contracts()
    result = build_continuous_contract(
        contracts,
        initial_lead_contract="ESH24",
    )

    pre_roll_level = result.raw_profile_bars.level_from_close(timestamps[5])
    same_contract_distance = result.raw_profile_bars.distance_to_close(pre_roll_level, timestamps[4])
    assert same_contract_distance == pytest.approx(-1.0)

    with pytest.raises(CoordinateMismatchError):
        result.raw_profile_bars.distance_to_close(pre_roll_level, timestamps[6])


def test_lead_assignment_is_causal_by_completed_session() -> None:
    contracts, timestamps = _build_synthetic_contracts()
    result = build_continuous_contract(
        contracts,
        initial_lead_contract="ESH24",
    )

    raw = result.raw_profile_bars.bars
    assert raw.loc[raw["timestamp"] == timestamps[5], "contract_id"].iloc[0] == "ESH24"
    assert raw.loc[raw["timestamp"] == timestamps[6], "contract_id"].iloc[0] == "ESM24"

    lead_schedule = result.lead_schedule.copy()
    session_four = lead_schedule.iloc[3]
    session_three_close = lead_schedule.iloc[2]["session_close"]
    esm24_session_three_volume = result.session_metrics.loc[
        (result.session_metrics["session_close"] == session_three_close)
        & (result.session_metrics["contract_id"] == "ESM24"),
        "metric_value",
    ].iloc[0]

    assert session_four["lead_contract"] == "ESM24"
    assert pd.Timestamp(session_four["decision_session_close"]) == pd.Timestamp(session_three_close)
    assert float(session_four["decision_metric_value"]) == pytest.approx(float(esm24_session_three_volume))


def test_back_adjusted_levels_track_cumulative_roll_spreads() -> None:
    contracts, timestamps = _build_synthetic_contracts()
    result = build_continuous_contract(
        contracts,
        initial_lead_contract="ESH24",
    )

    raw = result.raw_profile_bars.bars.reset_index(drop=True)
    adjusted = result.path_bars.bars.reset_index(drop=True)
    level_diff = raw["close"] - adjusted["close"]

    assert level_diff.iloc[:6].tolist() == pytest.approx([0.0] * 6)
    assert level_diff.iloc[6:].tolist() == pytest.approx([6.0, 6.0])

    raw_segment_diffs = raw.groupby("roll_segment_id")["close"].diff()
    adjusted_segment_diffs = adjusted.groupby("roll_segment_id")["close"].diff()
    np.testing.assert_allclose(
        raw_segment_diffs.to_numpy(dtype=float),
        adjusted_segment_diffs.to_numpy(dtype=float),
        equal_nan=True,
    )

    assert result.rolls["roll_spread"].tolist() == pytest.approx([6.0])
    assert pd.Timestamp(result.rolls["effective_from"].iloc[0]) == timestamps[6]


def test_roll_spanning_event_windows_are_flagged_for_exclusion() -> None:
    contracts, timestamps = _build_synthetic_contracts()
    result = build_continuous_contract(
        contracts,
        initial_lead_contract="ESH24",
    )

    flags = result.flag_event_windows(
        [timestamps[3], timestamps[5]],
        [timestamps[4], timestamps[6]],
    )

    assert flags.tolist() == [False, True]

    raw = result.raw_profile_bars.bars
    assert bool(raw.loc[raw["timestamp"] == timestamps[4], "is_roll_bracket"].iloc[0]) is True
    assert bool(raw.loc[raw["timestamp"] == timestamps[6], "is_roll_bracket"].iloc[0]) is True


def test_flat_contract_frame_prefers_contract_symbol_over_continuous_symbol() -> None:
    contracts, timestamps = _build_synthetic_contracts()
    flat_rows: list[pd.DataFrame] = []
    for instrument_id, (contract_symbol, frame) in enumerate(contracts.items(), start=1):
        tagged = frame.copy()
        tagged["symbol"] = "ES.v.0"
        tagged["contract_symbol"] = contract_symbol
        tagged["instrument_id"] = instrument_id
        flat_rows.append(tagged)

    flat_frame = pd.concat(flat_rows, ignore_index=True, sort=False)
    result = build_continuous_contract(
        flat_frame,
        timestamp_col="timestamp",
        initial_lead_contract="ESH24",
    )

    raw = result.raw_profile_bars.bars
    assert raw.loc[raw["timestamp"] == timestamps[5], "contract_id"].iloc[0] == "ESH24"
    assert raw.loc[raw["timestamp"] == timestamps[6], "contract_id"].iloc[0] == "ESM24"
    assert result.rolls["from_contract"].tolist() == ["ESH24"]
    assert result.rolls["to_contract"].tolist() == ["ESM24"]


def test_tagged_continuous_series_builds_directly_from_contract_tags() -> None:
    frame = pd.DataFrame(
        {
            "ts_event": [
                pd.Timestamp("2024-03-12 23:00:00+00:00"),
                pd.Timestamp("2024-03-12 23:05:00+00:00"),
                pd.Timestamp("2024-03-13 00:00:00+00:00"),
                pd.Timestamp("2024-03-13 00:05:00+00:00"),
            ],
            "open": [100.0, 101.0, 110.0, 111.0],
            "high": [101.5, 102.5, 111.5, 112.5],
            "low": [99.5, 100.5, 109.5, 110.5],
            "close": [101.0, 102.0, 111.0, 112.0],
            "volume": [1000, 900, 1200, 1100],
            "symbol": ["ES.v.0", "ES.v.0", "ES.v.0", "ES.v.0"],
            "instrument_id": [111, 111, 222, 222],
            "contract_symbol": ["ESH24", "ESH24", "ESM24", "ESM24"],
            "contract_expiration": [
                pd.Timestamp("2024-03-15 13:30:00+00:00"),
                pd.Timestamp("2024-03-15 13:30:00+00:00"),
                pd.Timestamp("2024-06-21 13:30:00+00:00"),
                pd.Timestamp("2024-06-21 13:30:00+00:00"),
            ],
            "is_roll_boundary": [False, False, True, False],
            "bars_since_roll": [0, 1, 0, 1],
            "market_day_close": [
                pd.Timestamp("2024-03-13 21:00:00+00:00"),
                pd.Timestamp("2024-03-13 21:00:00+00:00"),
                pd.Timestamp("2024-03-13 21:00:00+00:00"),
                pd.Timestamp("2024-03-13 21:00:00+00:00"),
            ],
            "market_day_index": [0, 0, 0, 0],
            "in_roll_bracket": [True, True, True, True],
        }
    )

    result = build_continuous_contract_from_tagged_series(frame)

    raw = result.raw_profile_bars.bars
    adjusted = result.path_bars.bars

    assert raw["contract_id"].tolist() == ["ESH24", "ESH24", "ESM24", "ESM24"]
    assert result.rolls["roll_spread"].tolist() == pytest.approx([8.0])
    assert result.rolls["from_contract"].tolist() == ["ESH24"]
    assert result.rolls["to_contract"].tolist() == ["ESM24"]
    assert adjusted["close"].tolist() == pytest.approx([101.0, 102.0, 103.0, 104.0])
