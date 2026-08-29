from __future__ import annotations

import shutil
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.builder import FeatureDatasetBuilder
from features.config import FeatureBuilderConfig
from features.feature_sets.microstructure import build_microstructure
from features.feature_sets.strategy_signals import _execute_strategy_jobs, build_strategy_signals
from features.io import standardize_market_frame
from features.progress import progress_context
from features.strategy_registry import STRATEGY_REGISTRY, prepare_strategy_input
from features.transforms import (
    add_atr_normalized_features,
    add_lag_features,
    add_rolling_percentile_rank_features,
    add_rolling_statistics,
    add_rolling_winsorized_features,
    add_rolling_zscores,
    add_sigma_normalized_features,
    calculate_atr,
    rolling_sigma_normalize,
    rolling_winsorize_series,
    safe_divide,
)
from frvp.feature_sets import summarize_frvp_feature_dataset


def _sample_market_frame(rows: int = 360) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="5min")
    trend = 1.08 + (index * 0.00008)
    cycle = np.sin(index / 9.0) * 0.0015
    open_ = trend + cycle
    close = trend + np.cos(index / 11.0) * 0.0013
    high = np.maximum(open_, close) + 0.0007 + (index % 5) * 0.00002
    low = np.minimum(open_, close) - 0.0007 - (index % 7) * 0.00002
    volume = 900 + ((index.astype(int) % 24) * 35)
    return pd.DataFrame(
        {
            "datetime": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _ts_ny(local_value: str) -> pd.Timestamp:
    return pd.Timestamp(local_value, tz="America/New_York").tz_convert("UTC")


def _synthetic_frvp_futures_frame() -> pd.DataFrame:
    timestamps = pd.date_range(_ts_ny("2024-01-02 09:30:00"), _ts_ny("2024-01-03 11:00:00"), freq="5min")
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
            "ts_event": kept,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "contract_id": "ESH24",
            "symbol": "ES.v.0",
        }
    )


def _reference_frame(index: pd.Index, columns: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for name, series in columns:
        out[name] = series
    return out


def _reference_rolling_percentile_rank_features(
    df: pd.DataFrame,
    columns: list[str],
    window: int,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        rolling = pd.to_numeric(df[column], errors="coerce").rolling(window, min_periods=window)
        try:
            out[f"{column}_pct_rank_{window}"] = rolling.rank(pct=True)
        except AttributeError:
            out[f"{column}_pct_rank_{window}"] = rolling.apply(
                lambda values: float(np.sum(values[~np.isnan(values)] <= values[~np.isnan(values)][-1]) / len(values[~np.isnan(values)]))
                if len(values[~np.isnan(values)]) > 0
                else np.nan,
                raw=True,
            )
    return out


def _reference_transform_outputs(df: pd.DataFrame, config: FeatureBuilderConfig) -> dict[str, pd.DataFrame]:
    atr = df["atr_14"] if "atr_14" in df.columns else calculate_atr(df)

    lag_columns: list[tuple[str, pd.Series]] = []
    for column in config.lag_columns:
        if column not in df.columns:
            continue
        for period in config.lag_periods:
            lag_columns.append((f"{column}_lag_{period}", df[column].shift(period)))

    rolling_stat_columns: list[tuple[str, pd.Series]] = []
    for column in config.rolling_stat_columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        for window in config.rolling_windows:
            rolling = numeric.rolling(window)
            rolling_stat_columns.append((f"{column}_roll_mean_{window}", rolling.mean()))
            rolling_stat_columns.append((f"{column}_roll_std_{window}", rolling.std()))

    zscore_columns: list[tuple[str, pd.Series]] = []
    for column in config.zscore_columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        mean = numeric.rolling(config.zscore_window).mean()
        std = numeric.rolling(config.zscore_window).std().replace(0, np.nan)
        zscore_columns.append((f"{column}_zscore_{config.zscore_window}", (numeric - mean) / std))

    winsor_columns = [
        (
            f"{column}_winsor_{config.winsorization_window}",
            rolling_winsorize_series(
                df[column],
                window=config.winsorization_window,
                lower_quantile=config.winsorization_lower_quantile,
                upper_quantile=config.winsorization_upper_quantile,
            ),
        )
        for column in config.winsorize_columns
        if column in df.columns
    ]

    atr_norm_columns = [
        (f"{column}_atr_norm", safe_divide(df[column], atr))
        for column in config.atr_normalization_columns
        if column in df.columns
    ]

    sigma_columns = [
        (
            f"{column}_sigma_norm_{config.sigma_normalization_window}",
            rolling_sigma_normalize(df[column], config.sigma_normalization_window),
        )
        for column in config.sigma_normalization_columns
        if column in df.columns
    ]

    return {
        "winsorization": _reference_frame(df.index, winsor_columns),
        "percentile_ranks": _reference_rolling_percentile_rank_features(
            df,
            list(config.percentile_rank_columns),
            config.percentile_rank_window,
        ),
        "atr_normalization": _reference_frame(df.index, atr_norm_columns),
        "sigma_normalization": _reference_frame(df.index, sigma_columns),
        "lags": _reference_frame(df.index, lag_columns),
        "rolling_stats": _reference_frame(df.index, rolling_stat_columns),
        "zscores": _reference_frame(df.index, zscore_columns),
    }


def test_builder_threaded_transforms_match_serial_output() -> None:
    market = _sample_market_frame()

    serial_config = FeatureBuilderConfig(
        warmup_rows=0,
        drop_warmup_rows=False,
        transform_workers=1,
    )
    threaded_config = FeatureBuilderConfig.from_dict(serial_config.to_dict())
    threaded_config.transform_workers = 2

    serial_dataset, serial_metadata = FeatureDatasetBuilder(serial_config).build(market)
    threaded_dataset, threaded_metadata = FeatureDatasetBuilder(threaded_config).build(market)

    assert serial_dataset.columns.tolist() == threaded_dataset.columns.tolist()
    numeric_columns = serial_dataset.select_dtypes(include=[np.number]).columns
    np.testing.assert_allclose(
        serial_dataset.loc[:, numeric_columns].to_numpy(),
        threaded_dataset.loc[:, numeric_columns].to_numpy(),
        equal_nan=True,
    )
    assert serial_metadata["transform_counts"] == threaded_metadata["transform_counts"]
    assert "total" in threaded_metadata["build_timings_seconds"]
    assert "transform:lags" in threaded_metadata["build_timings_seconds"]


def test_builder_can_downcast_generated_feature_dtypes() -> None:
    market = _sample_market_frame()
    config = FeatureBuilderConfig(
        warmup_rows=0,
        drop_warmup_rows=False,
        optimize_feature_dtypes=True,
    )

    dataset, metadata = FeatureDatasetBuilder(config).build(market)

    assert metadata["memory_usage_bytes"] <= metadata["memory_usage_bytes_before_dtype_optimization"]
    assert metadata["feature_memory_usage_bytes"] > 0
    assert dataset["atr_14"].dtype == np.float32
    assert dataset["bullish_candle"].dtype == np.int8
    assert "feature_set:price_action" in metadata["build_timings_seconds"]
    assert "transform:rolling_stats" in metadata["build_timings_seconds"]


def test_transform_refactor_matches_reference_outputs_with_float32_tolerance() -> None:
    market = _sample_market_frame(rows=360)
    base_config = FeatureBuilderConfig(
        warmup_rows=0,
        drop_warmup_rows=False,
        enable_lags=False,
        enable_rolling_stats=False,
        enable_zscores=False,
        enable_winsorization=False,
        enable_percentile_ranks=False,
        enable_atr_normalization=False,
        enable_sigma_normalization=False,
        enable_interactions=False,
    )
    feature_frame, _ = FeatureDatasetBuilder(base_config).build(market)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        actual = {
            "winsorization": add_rolling_winsorized_features(
                feature_frame,
                base_config.winsorize_columns,
                base_config.winsorization_window,
                base_config.winsorization_lower_quantile,
                base_config.winsorization_upper_quantile,
            ),
            "percentile_ranks": add_rolling_percentile_rank_features(
                feature_frame,
                base_config.percentile_rank_columns,
                base_config.percentile_rank_window,
            ),
            "atr_normalization": add_atr_normalized_features(
                feature_frame,
                base_config.atr_normalization_columns,
                atr_column=base_config.atr_normalization_source,
            ),
            "sigma_normalization": add_sigma_normalized_features(
                feature_frame,
                base_config.sigma_normalization_columns,
                base_config.sigma_normalization_window,
            ),
            "lags": add_lag_features(
                feature_frame,
                base_config.lag_columns,
                base_config.lag_periods,
            ),
            "rolling_stats": add_rolling_statistics(
                feature_frame,
                base_config.rolling_stat_columns,
                base_config.rolling_windows,
            ),
            "zscores": add_rolling_zscores(
                feature_frame,
                base_config.zscore_columns,
                base_config.zscore_window,
            ),
        }

    expected = _reference_transform_outputs(feature_frame, base_config)

    for name, actual_frame in actual.items():
        expected_frame = expected[name]
        assert actual_frame.columns.tolist() == expected_frame.columns.tolist()
        np.testing.assert_allclose(
            actual_frame.to_numpy(dtype=np.float64),
            expected_frame.to_numpy(dtype=np.float64),
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        )
        assert all(dtype == np.float32 for dtype in actual_frame.dtypes)


def test_builder_emits_progress_events() -> None:
    market = _sample_market_frame(rows=96)
    events = []
    config = FeatureBuilderConfig(
        warmup_rows=0,
        drop_warmup_rows=False,
        transform_workers=1,
    )

    FeatureDatasetBuilder(config, progress_callback=events.append).build(market)

    assert any(event.stage == "build" and event.action == "start" for event in events)
    assert any(
        event.stage == "feature_set" and event.action == "start" and event.name == "price_action"
        for event in events
    )
    assert any(
        event.stage == "transform" and event.action == "complete" and event.name == "lags"
        for event in events
    )
    assert events[-1].stage == "build"
    assert events[-1].action == "complete"


def test_standardize_market_frame_normalizes_gmt_minus_6_source_to_utc() -> None:
    raw = pd.DataFrame(
        {
            "datetime": ["2024-01-01 00:00:00", "2024-01-01 00:05:00"],
            "open": [1.10, 1.11],
            "high": [1.12, 1.13],
            "low": [1.09, 1.10],
            "close": [1.11, 1.12],
            "volume": [100, 120],
        }
    )

    standardized = standardize_market_frame(raw, source_timezone="GMT-6", canonical_timezone="UTC")

    assert str(standardized["datetime"].iloc[0]) == "2024-01-01 06:00:00+00:00"
    assert str(standardized["datetime"].iloc[1]) == "2024-01-01 06:05:00+00:00"


def test_builder_produces_identical_time_features_for_equivalent_utc_and_gmt_minus_6_inputs() -> None:
    market_utc = _sample_market_frame(rows=96)
    market_gmt6 = market_utc.copy()
    market_gmt6["datetime"] = (
        pd.to_datetime(market_utc["datetime"], errors="coerce") - pd.Timedelta(hours=6)
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    base_config = FeatureBuilderConfig(
        feature_sets=["session", "temporal_context", "quality"],
        warmup_rows=0,
        drop_warmup_rows=False,
        enable_lags=False,
        enable_rolling_stats=False,
        enable_zscores=False,
        enable_winsorization=False,
        enable_percentile_ranks=False,
        enable_atr_normalization=False,
        enable_sigma_normalization=False,
        enable_interactions=False,
        fillna_numeric=False,
    )

    utc_config = FeatureBuilderConfig.from_dict(base_config.to_dict())
    utc_config.source_timezone = "UTC"
    gmt6_config = FeatureBuilderConfig.from_dict(base_config.to_dict())
    gmt6_config.source_timezone = "GMT-6"

    utc_dataset, _ = FeatureDatasetBuilder(utc_config).build(market_utc)
    gmt6_dataset, _ = FeatureDatasetBuilder(gmt6_config).build(market_gmt6)

    compare_columns = [
        "hour",
        "minute",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "in_asian_session",
        "in_london_session",
        "in_newyork_session",
        "in_london_ny_overlap",
        "in_london_killzone",
        "in_newyork_killzone",
        "in_london_early",
        "in_london_mid",
        "in_london_late",
        "in_newyork_early",
        "in_newyork_mid",
        "in_newyork_late",
        "large_time_gap_flag",
    ]

    np.testing.assert_allclose(
        utc_dataset.loc[:, compare_columns].to_numpy(dtype=float),
        gmt6_dataset.loc[:, compare_columns].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_builder_runtime_tuning_scales_feature_windows_for_1m_inputs() -> None:
    market = _sample_market_frame(rows=360)
    market["datetime"] = pd.date_range("2024-01-01 00:00:00", periods=len(market), freq="1min")
    config = FeatureBuilderConfig(
        feature_sets=["price_action", "volatility", "session"],
        drop_warmup_rows=False,
        enable_interactions=False,
        enable_sigma_normalization=False,
        enable_atr_normalization=False,
    )

    _, metadata = FeatureDatasetBuilder(config).build(market)

    assert metadata["input_bar_minutes"] == 1.0
    assert metadata["input_timeframe"] == "1m"
    assert metadata["runtime_bar_interval_tuning"]["applied"] is True
    assert metadata["config"]["warmup_rows"] == 1000
    assert metadata["config"]["lag_periods"] == [5, 15, 25, 50]
    assert metadata["config"]["rolling_windows"] == [25, 50, 100, 250]


def test_futures_microstructure_spread_proxy_is_tick_clipped_not_full_bar_range() -> None:
    frame = pd.DataFrame(
        {
            "open": [5000.00, 5001.00],
            "high": [5001.50, 5002.25],
            "low": [4999.50, 5000.25],
            "close": [5000.75, 5001.50],
            "volume": [1000.0, 1200.0],
        }
    )

    out = build_microstructure(frame, FeatureBuilderConfig(instrument="es"))

    assert float(out.loc[0, "approx_spread"]) == 0.50
    assert float(out.loc[1, "approx_spread"]) == 0.50
    assert float(out.loc[0, "approx_spread"]) < float(frame.loc[0, "high"] - frame.loc[0, "low"])


def test_builder_can_build_frvp_meta_recipe_from_ts_event_futures_input(tmp_path) -> None:
    market = _synthetic_frvp_futures_frame()
    config = FeatureBuilderConfig.from_recipe(ROOT / "features" / "recipes" / "frvp_meta.json")
    config.instrument = "es"
    config.transform_workers = 1

    dataset, metadata = FeatureDatasetBuilder(config).build(market)

    expected_columns = {
        "datetime",
        "hour",
        "htf_30m_ema_alignment",
        "frvp_dist_poc_session_atr",
        "frvp_setup_type",
        "frvp_setup_confidence_rule",
    }
    assert expected_columns.issubset(dataset.columns)
    assert metadata["config"]["instrument"] == "es"
    assert metadata["input_timeframe"] == "5m"

    audit_path = tmp_path / "frvp_phase2_audit.json"
    summary = summarize_frvp_feature_dataset(dataset, output_path=audit_path)

    assert summary["frvp_column_count"] > 0
    assert "frvp_setup_type" in summary["coverage"]
    assert audit_path.exists()


def test_frvp_feature_audit_preserves_open_type_missing_bucket_for_mi(tmp_path: Path) -> None:
    rows = 48
    timestamps = pd.date_range("2024-03-01 00:00:00", periods=rows, freq="5min", tz="UTC")
    open_type = ([np.nan] * 30) + ([1] * 9) + ([-1] * 9)
    target = ([0, 1] * 15) + ([1] * 9) + ([0] * 9)

    dataset = pd.DataFrame(
        {
            "datetime": timestamps,
            "frvp_open_type": open_type,
            "frvp_dist_poc_session_atr": np.linspace(-1.0, 1.0, rows),
            "htf_confluence_long_frvp_reversal": [1 if index % 6 == 0 else 0 for index in range(rows)],
            "label_long_frvp_reversal": target,
            "sample_weight_long_frvp_reversal": [1.5 if value else 1.0 for value in target],
            "label_quality_long_frvp_reversal": [0.8 if value else 0.0 for value in target],
            "exclude_long_frvp_reversal": [False] * rows,
            "neg_ok_long_frvp_reversal": [not bool(value) for value in target],
            "warmup_mask": [False] * rows,
        }
    )

    summary = summarize_frvp_feature_dataset(dataset, output_path=tmp_path / "phase2_audit.json")

    assert summary["coverage"]["frvp_open_type"]["null_count"] == 30
    assert summary["key_feature_mi"]["long_frvp_reversal"]["frvp_open_type_mi"] > 0.0


def test_prepare_strategy_input_preserves_original_columns_and_adds_aliases() -> None:
    market = _sample_market_frame()
    market["extra_feature"] = np.arange(len(market), dtype=float)

    prepared = prepare_strategy_input(market)

    assert "extra_feature" in prepared.columns
    assert "datetime" in prepared.columns
    assert "Open" in prepared.columns
    assert "High" in prepared.columns
    assert "Low" in prepared.columns
    assert "Close" in prepared.columns
    assert "Volume" in prepared.columns
    assert isinstance(prepared.index, pd.DatetimeIndex)


def test_prepare_strategy_input_avoids_fragmentation_warning() -> None:
    market = _sample_market_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for idx in range(150):
            market[f"extra_{idx}"] = idx

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        prepared = prepare_strategy_input(market)

    assert "Open" in prepared.columns
    assert "Date" in prepared.columns
    assert "Time" in prepared.columns


def test_strategy_signals_threaded_execution_matches_serial_output(monkeypatch) -> None:
    market = _sample_market_frame(rows=64)
    market["extra_feature"] = np.arange(len(market), dtype=float)
    seen_columns: list[tuple[str, ...]] = []
    progress_events = []

    def fake_resolve(name: str) -> str:
        return name

    def fake_build(name: str, df: pd.DataFrame, *, copy_input: bool = True) -> pd.DataFrame:
        assert copy_input is True
        seen_columns.append(tuple(df.columns))
        if name == "alpha":
            signal = np.where(np.arange(len(df)) % 2 == 0, "long", "neutral")
            return pd.DataFrame({"signal": signal}, index=df.index)
        return pd.DataFrame({"score": np.arange(len(df), dtype=float)}, index=df.index)

    monkeypatch.setattr(STRATEGY_REGISTRY, "resolve", fake_resolve)
    monkeypatch.setattr(STRATEGY_REGISTRY, "build", fake_build)

    serial = build_strategy_signals(
        market,
        FeatureBuilderConfig(strategy_ids=["alpha", "beta"], transform_workers=1, strategy_timeout_seconds=None),
    )
    with progress_context(progress_events.append):
        threaded = build_strategy_signals(
            market,
            FeatureBuilderConfig(strategy_ids=["alpha", "beta"], transform_workers=2, strategy_timeout_seconds=None),
        )

    assert serial.columns.tolist() == threaded.columns.tolist()
    numeric_columns = serial.select_dtypes(include=[np.number]).columns
    np.testing.assert_allclose(
        serial.loc[:, numeric_columns].to_numpy(),
        threaded.loc[:, numeric_columns].to_numpy(),
        equal_nan=True,
    )
    assert serial.loc[:, serial.columns.difference(numeric_columns)].equals(
        threaded.loc[:, threaded.columns.difference(numeric_columns)]
    )
    assert [column for column in threaded.columns if column.startswith("strategy__alpha__")] == threaded.columns[:2].tolist()
    assert threaded.columns[-1] == "strategy__beta__score"
    assert threaded.attrs["feature_build_report"]["requested"] == 2
    assert threaded.attrs["feature_build_report"]["built"] == 2
    assert threaded.attrs["feature_build_report"]["skipped"] == 0
    assert threaded.attrs["feature_build_report"]["wall_clock_seconds"] >= 0.0
    assert len(threaded.attrs["feature_build_report"]["slowest_strategy_attempts"]) == 2
    assert any(
        event.stage == "strategy_batch" and event.action == "start"
        for event in progress_events
    )
    assert any(
        event.stage == "strategy" and event.action == "start" and event.name == "alpha"
        for event in progress_events
    )
    assert any(
        event.stage == "strategy" and event.action == "complete" and event.name == "alpha"
        for event in progress_events
    )

    for columns in seen_columns:
        assert "extra_feature" in columns
        assert "Open" in columns
        assert "Close" in columns


def test_strategy_registry_build_handles_legacy_chained_assignment_under_copy_on_write(monkeypatch) -> None:
    market = _sample_market_frame(rows=6)

    def legacy_builder(df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        signals["signal"] = 0
        signals["signal"][1:] = np.where(df["close"].iloc[1:] > df["close"].iloc[0], 1, 0)
        return signals

    monkeypatch.setattr(STRATEGY_REGISTRY, "load", lambda _name: legacy_builder)

    with pd.option_context("mode.copy_on_write", True):
        with warnings.catch_warnings():
            warnings.simplefilter("error", getattr(pd.errors, "ChainedAssignmentError"))
            warnings.filterwarnings(
                "error",
                message=r".*ChainedAssignmentError.*",
                category=FutureWarning,
            )
            built = STRATEGY_REGISTRY.build("legacy_strategy", market, copy_input=True)

    assert built["signal"].tolist() == [0, 1, 1, 1, 1, 1]


def test_strategy_timeout_skips_long_running_strategy() -> None:
    strategies_dir = ROOT / "data" / ".tmp_strategy_timeout_test"
    if strategies_dir.exists():
        shutil.rmtree(strategies_dir)
    strategies_dir.mkdir(parents=True)

    try:
        (strategies_dir / "sleepy_strategy.py").write_text(
            "\n".join(
                [
                    "import time",
                    "import pandas as pd",
                    "",
                    "def sleepy_signals(stock_df):",
                    "    time.sleep(0.25)",
                    "    return pd.DataFrame({'signal': 'long'}, index=stock_df.index)",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        market = _sample_market_frame(rows=24)
        strategy_input = prepare_strategy_input(market)

        results = _execute_strategy_jobs(
            requested_strategy_ids=["sleepy_strategy"],
            strategy_input=strategy_input,
            target_index=market.index,
            input_columns=set(strategy_input.columns),
            skip_failed_strategies=True,
            max_workers=1,
            timeout_seconds=0.05,
            strategies_dir=strategies_dir,
        )

        assert len(results) == 1
        assert results[0].failure is not None
        assert results[0].failure["error_type"] == "TimeoutError"
    finally:
        shutil.rmtree(strategies_dir, ignore_errors=True)
