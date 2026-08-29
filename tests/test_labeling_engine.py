from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.labeling.labeling_engine import (
    build_default_params,
    build_output_metadata,
    filter_date_range,
    load_fx_csv,
    retune_combined_params,
    write_metadata,
)


def test_load_fx_csv_normalizes_gmt_minus_6_input_and_builds_metadata() -> None:
    test_root = ROOT / "tmp" / f"test_load_fx_csv_normalizes_{uuid4().hex}"
    test_root.mkdir(parents=True)

    try:
        input_path = test_root / "eurusd_5m.csv"
        pd.DataFrame(
            {
                "Date": ["20240101", "20240101"],
                "Time": ["00:00:00", "00:05:00"],
                "Open": [1.1000, 1.1005],
                "High": [1.1010, 1.1015],
                "Low": [1.0990, 1.0995],
                "Close": [1.1004, 1.1008],
                "Volume": [100, 120],
            }
        ).to_csv(input_path, index=False)

        df_5m = load_fx_csv(
            input_path,
            source_timezone="GMT-6",
            canonical_timezone="UTC",
        )
        df_30m = df_5m.resample("30min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        df_1h = df_5m.resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

        metadata = build_output_metadata(
            output_kind="bar_labels",
            source_path=input_path,
            output_path=test_root / "labels.csv",
            df_5m=df_5m,
            df_30m=df_30m,
            df_1hr=df_1h,
            params=build_default_params(),
            source_timezone="GMT-6",
            canonical_timezone="UTC",
            bar_timestamp_semantics="bar_open",
            output_rows=len(df_5m),
        )

        assert str(df_5m.index[0]) == "2024-01-01 06:00:00+00:00"
        assert str(df_5m.index[1]) == "2024-01-01 06:05:00+00:00"
        assert df_5m.attrs["source_timezone"] == "GMT-6"
        assert df_5m.attrs["canonical_timezone"] == "UTC"
        assert metadata["timezone_contract"]["source_timezone"] == "GMT-6"
        assert metadata["timezone_contract"]["canonical_timezone"] == "UTC"
        assert metadata["timezone_contract"]["csv_timestamps_are_timezone_aware"] is False
        assert metadata["bar_timestamp_semantics"] == "bar_open"
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def test_filter_date_range_keeps_requested_canonical_window_and_updates_metadata() -> None:
    index = pd.date_range("2018-12-31 23:55:00+00:00", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [1, 1, 1, 1, 1],
        },
        index=index,
    )
    df.attrs["source_row_count"] = len(df)
    df.attrs["anomaly_count"] = 0

    filtered = filter_date_range(
        df,
        start_date="2019-01-01",
        end_date="2019-01-01",
        canonical_timezone="UTC",
    )

    assert list(filtered.index.astype(str)) == [
        "2019-01-01 00:00:00+00:00",
        "2019-01-01 00:05:00+00:00",
        "2019-01-01 00:10:00+00:00",
        "2019-01-01 00:15:00+00:00",
    ]
    assert filtered.attrs["rows_before_date_filter"] == 5
    assert filtered.attrs["rows_removed_by_date_filter"] == 1
    assert filtered.attrs["requested_start_date"] == "2019-01-01"
    assert filtered.attrs["requested_end_date"] == "2019-01-01"
    assert filtered.attrs["date_filter_start_canonical"] == "2019-01-01T00:00:00+00:00"
    assert filtered.attrs["date_filter_end_canonical"] == "2019-01-01T00:00:00+00:00"


def test_write_metadata_handles_numpy_and_pandas_scalars() -> None:
    test_root = ROOT / "tmp" / f"test_write_metadata_handles_numpy_scalars_{uuid4().hex}"
    test_root.mkdir(parents=True)

    try:
        output_path = test_root / "labels.csv"
        payload = {
            "family_diagnostics": {
                "reversal": {
                    "sanity_checks": [
                        ("no_overlap", np.bool_(True), np.int64(0)),
                        ("distribution_ok", np.bool_(False), np.float64(2.5)),
                    ],
                    "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                }
            }
        }

        metadata_path = write_metadata(output_path, payload)
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))

        assert saved["family_diagnostics"]["reversal"]["sanity_checks"] == [
            ["no_overlap", True, 0],
            ["distribution_ok", False, 2.5],
        ]
        assert saved["family_diagnostics"]["reversal"]["timestamp"] == "2024-01-01T00:00:00+00:00"
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def test_retune_combined_params_scales_5m_bar_counts_for_1m_runtime() -> None:
    params = retune_combined_params(build_default_params(), 1.0)

    assert params.reversal.min_bars_between_swings == 90
    assert params.reversal.tb_max_bars == 600
    assert params.continuation.max_pullback_bars == 240
    assert params.breakout.breakout_lookback_bars == 100
    assert params.breakout.compression_lookback_bars == 60
    assert params.reversal.auto_scale_bar_counts is False
