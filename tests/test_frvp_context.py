from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.builder import FeatureDatasetBuilder  # noqa: E402
from features.config import FeatureBuilderConfig  # noqa: E402
from frvp.feature_sets.frvp_context import build_frvp_context_features  # noqa: E402


def _ts(local_value: str) -> pd.Timestamp:
    return pd.Timestamp(local_value, tz="America/New_York").tz_convert("UTC")


def _synthetic_frvp_frame() -> pd.DataFrame:
    timestamps = pd.date_range(_ts("2024-01-02 09:30:00"), _ts("2024-01-03 20:00:00"), freq="5min")
    kept: list[pd.Timestamp] = []
    for timestamp in timestamps:
        local = timestamp.tz_convert("America/New_York")
        minute_of_day = local.hour * 60 + local.minute
        if 17 * 60 <= minute_of_day < 18 * 60:
            continue
        kept.append(timestamp)

    closes: list[float] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    volumes: list[float] = []
    previous_close = 5000.0

    for index, timestamp in enumerate(kept):
        local = timestamp.tz_convert("America/New_York")
        session_day = (local.date() - date(2024, 1, 2)).days
        minute_of_day = local.hour * 60 + local.minute
        in_rth = (9 * 60 + 30) <= minute_of_day < (16 * 60)

        if session_day == 0 and in_rth:
            minutes_since_open = minute_of_day - (9 * 60 + 30)
            base = 5000.0 + (minutes_since_open / 390.0) * 4.0 + np.sin(minutes_since_open / 35.0) * 0.6
            volume = 240 + (1.0 - abs((minutes_since_open - 195.0) / 195.0)) * 120
        elif session_day == 0:
            base = 5004.5 + np.sin(index / 11.0) * 0.8
            volume = 90 + abs(np.cos(index / 7.0)) * 30
        elif in_rth:
            minutes_since_open = minute_of_day - (9 * 60 + 30)
            base = 5010.5 + (minutes_since_open / 390.0) * 2.5 + np.cos(minutes_since_open / 28.0) * 0.5
            volume = 260 + (1.0 - abs((minutes_since_open - 195.0) / 195.0)) * 140
        else:
            base = 5008.0 + np.sin(index / 9.0) * 0.7
            volume = 95 + abs(np.sin(index / 5.0)) * 35

        open_price = previous_close
        close_price = base + np.sin(index / 4.0) * 0.15
        high_price = max(open_price, close_price) + 0.35 + (index % 4) * 0.03
        low_price = min(open_price, close_price) - 0.35 - (index % 3) * 0.03

        opens.append(float(open_price))
        closes.append(float(close_price))
        highs.append(float(high_price))
        lows.append(float(low_price))
        volumes.append(float(volume))
        previous_close = close_price

    return pd.DataFrame(
        {
            "datetime": kept,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "contract_id": "ESH24",
            "symbol": "ESH24",
        }
    )


def _frvp_only_config() -> FeatureBuilderConfig:
    return FeatureBuilderConfig(
        feature_sets=["frvp_context"],
        warmup_rows=0,
        drop_warmup_rows=False,
        fillna_numeric=False,
        enable_lags=False,
        enable_rolling_stats=False,
        enable_zscores=False,
        enable_winsorization=False,
        enable_percentile_ranks=False,
        enable_atr_normalization=False,
        enable_sigma_normalization=False,
        enable_interactions=False,
    )


def test_frvp_context_registered_output_has_expected_columns_and_masks_ib() -> None:
    frame = _synthetic_frvp_frame()
    dataset, _ = FeatureDatasetBuilder(_frvp_only_config()).build(frame)

    expected_columns = {
        "frvp_dist_poc_session_atr",
        "frvp_dist_poc_day_atr",
        "frvp_dist_poc_swing_atr",
        "frvp_dist_vah_atr",
        "frvp_dist_val_atr",
        "frvp_price_position_va",
        "frvp_profile_shape",
        "frvp_va_width_atr",
        "frvp_open_type",
        "frvp_open_drive_flag",
        "frvp_ib_extension",
        "frvp_dist_ib_high_atr",
        "frvp_dist_ib_low_atr",
        "frvp_session_phase",
        "frvp_open_vs_prior_poc_atr",
        "frvp_open_gap_atr",
        "frvp_gap_into_value",
        "frvp_naked_vpoc_dist_above_atr",
        "frvp_naked_vpoc_dist_below_atr",
        "frvp_naked_vpoc_age_sessions",
        "frvp_naked_vpoc_count",
        "frvp_setup_type",
        "frvp_setup_side",
        "frvp_setup_confidence_rule",
        "frvp_failed_auction_with_sweep",
        "frvp_day_type",
    }
    assert expected_columns.issubset(set(dataset.columns))
    assert str(dataset["frvp_profile_shape"].dtype) == "Int64"
    assert str(dataset["frvp_open_type"].dtype) == "Int64"
    assert str(dataset["frvp_ib_extension"].dtype) == "Int64"
    assert str(dataset["frvp_setup_type"].dtype) == "Int64"
    assert str(dataset["frvp_setup_side"].dtype) == "Int64"

    pre_ib = dataset.loc[dataset["datetime"] == _ts("2024-01-03 10:25:00")].iloc[0]
    post_ib = dataset.loc[dataset["datetime"] == _ts("2024-01-03 10:30:00")].iloc[0]

    assert pd.isna(pre_ib["frvp_dist_ib_high_atr"])
    assert pd.isna(pre_ib["frvp_dist_ib_low_atr"])
    assert pd.isna(pre_ib["frvp_ib_extension"])
    assert pd.notna(post_ib["frvp_dist_ib_high_atr"])
    assert pd.notna(post_ib["frvp_dist_ib_low_atr"])
    assert pd.notna(post_ib["frvp_ib_extension"])


def test_frvp_context_synthetic_session_values_are_sane() -> None:
    frame = _synthetic_frvp_frame()
    features = build_frvp_context_features(frame, _frvp_only_config())
    target = features.loc[frame["datetime"] == _ts("2024-01-03 10:35:00")].iloc[0]

    assert np.isfinite(target["frvp_dist_poc_session_atr"])
    assert np.isfinite(target["frvp_dist_poc_day_atr"])
    assert np.isfinite(target["frvp_price_position_va"])
    assert np.isfinite(target["frvp_open_vs_prior_poc_atr"])
    assert int(target["frvp_open_type"]) in {-1, 0, 1}
    assert int(target["frvp_session_phase"]) in {1, 2, 3, 4, 5}
    assert int(target["frvp_day_type"]) in {1, 2, 3, 4, 5}
    assert float(target["frvp_naked_vpoc_count"]) >= 0.0
    assert 0.0 <= float(target["frvp_setup_confidence_rule"]) <= 1.0
    if pd.notna(target["frvp_rth_eth_value_overlap"]):
        assert 0.0 <= float(target["frvp_rth_eth_value_overlap"]) <= 1.0


def test_frvp_context_keeps_next_session_preopen_open_type_blank_after_1600_roll() -> None:
    frame = _synthetic_frvp_frame()
    features = build_frvp_context_features(frame, _frvp_only_config())

    pre_open = features.loc[frame["datetime"] == _ts("2024-01-03 08:00:00")].iloc[0]
    next_session_preopen = features.loc[frame["datetime"] == _ts("2024-01-03 16:05:00")].iloc[0]

    assert pd.isna(pre_open["frvp_open_type"])
    assert pd.isna(pre_open["frvp_open_vs_prior_poc_atr"])
    assert pd.isna(pre_open["frvp_open_gap_atr"])
    assert pd.isna(pre_open["frvp_gap_into_value"])

    assert pd.isna(next_session_preopen["frvp_open_type"])
    assert pd.isna(next_session_preopen["frvp_open_vs_prior_poc_atr"])
    assert pd.isna(next_session_preopen["frvp_open_gap_atr"])
    assert pd.isna(next_session_preopen["frvp_gap_into_value"])


def test_frvp_context_does_not_change_existing_rows_when_future_bar_is_appended() -> None:
    frame = _synthetic_frvp_frame()
    base = frame.iloc[:-1].copy()
    future = frame.iloc[[-1]].copy()
    future.loc[:, "close"] = future["close"] + 25.0
    future.loc[:, "high"] = future["high"] + 25.0
    future.loc[:, "low"] = future["low"] + 25.0
    future.loc[:, "volume"] = future["volume"] * 20.0
    extended = pd.concat([base, future], ignore_index=True)

    features_base = build_frvp_context_features(base, _frvp_only_config())
    features_extended = build_frvp_context_features(extended, _frvp_only_config())

    compare_timestamp = pd.Timestamp(base["datetime"].iloc[-1])
    row_base = features_base.loc[base["datetime"] == compare_timestamp].iloc[0]
    row_extended = features_extended.loc[extended["datetime"] == compare_timestamp].iloc[0]
    compare_columns = [
        "frvp_dist_poc_session_atr",
        "frvp_dist_poc_day_atr",
        "frvp_dist_vah_atr",
        "frvp_dist_val_atr",
        "frvp_price_position_va",
        "frvp_profile_shape",
        "frvp_open_type",
        "frvp_open_vs_prior_poc_atr",
        "frvp_naked_vpoc_count",
        "frvp_setup_type",
        "frvp_setup_side",
        "frvp_setup_confidence_rule",
        "frvp_failed_auction_with_sweep",
    ]

    for column in compare_columns:
        left = row_base[column]
        right = row_extended[column]
        if pd.isna(left) and pd.isna(right):
            continue
        assert left == right
