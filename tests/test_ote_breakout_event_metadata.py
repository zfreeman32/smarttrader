import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_testing.ote_breakout_event_metadata import build_breakout_event_metadata_frame


def test_build_breakout_event_metadata_frame_maps_zone_rows_and_derives_event_exit() -> None:
    label_frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 00:00:00+00:00", periods=8, freq="5min"),
            "close": [1.1000, 1.0995, 1.0990, 1.0985, 1.0980, 1.0975, 1.0970, 1.0965],
        }
    )
    events_frame = pd.DataFrame(
        {
            "label_family": ["breakout"],
            "breakout_direction": ["down"],
            "confirm_time": [pd.Timestamp("2024-01-01 00:10:00+00:00")],
            "confirm_index": [2],
            "entry_time": [pd.Timestamp("2024-01-01 00:20:00+00:00")],
            "entry_index": [4],
            "entry_price": [1.0980],
            "entry_score": [4.2],
            "tb_bars_held": [5],
            "tb_outcome": ["tp"],
            "tb_return": [0.0015],
            "label_quality": [0.85],
            "label_tier": ["A"],
        }
    )

    metadata = build_breakout_event_metadata_frame(
        label_frame,
        events_frame,
        direction="short",
        label_kind="breakout",
        zone_pre_bars=0,
        zone_post_bars=2,
    )

    assert metadata["breakout_event_confirm_index"].tolist() == [2, 2, 2]
    assert metadata["breakout_event_entry_index"].tolist() == [4, 4, 4]
    assert metadata["breakout_event_exit_index"].tolist() == [7, 7, 7]
    assert metadata["breakout_event_entry_delay_bars"].tolist() == [2, 2, 2]
    assert metadata["breakout_event_remaining_bars_after_entry"].tolist() == [3, 3, 3]
    assert metadata["breakout_event_trade_available"].tolist() == [True, True, True]


def test_build_breakout_event_metadata_frame_maps_breakout_entry_only_to_entry_bar() -> None:
    label_frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 00:00:00+00:00", periods=6, freq="5min"),
            "close": [1.1000, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025],
        }
    )
    events_frame = pd.DataFrame(
        {
            "label_family": ["breakout"],
            "breakout_direction": ["up"],
            "confirm_time": [pd.Timestamp("2024-01-01 00:05:00+00:00")],
            "confirm_index": [1],
            "entry_time": [pd.Timestamp("2024-01-01 00:15:00+00:00")],
            "entry_index": [3],
            "entry_price": [1.1015],
            "tb_bars_held": [4],
            "tb_outcome": ["tp"],
            "tb_return": [0.0010],
        }
    )

    metadata = build_breakout_event_metadata_frame(
        label_frame,
        events_frame,
        direction="long",
        label_kind="breakout_entry",
        zone_pre_bars=0,
        zone_post_bars=2,
    )

    assert len(metadata) == 1
    assert int(metadata.loc[0, "breakout_event_entry_index"]) == 3
    assert bool(metadata.loc[0, "breakout_event_trade_available"]) is True
    assert metadata.loc[0, "datetime"] == pd.Timestamp("2024-01-01 00:15:00+00:00")
