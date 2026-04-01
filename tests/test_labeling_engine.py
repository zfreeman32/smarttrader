from __future__ import annotations

import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.labeling.labeling_engine import build_default_params, build_output_metadata, load_fx_csv


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
