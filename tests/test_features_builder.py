from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.builder import FeatureDatasetBuilder
from features.config import FeatureBuilderConfig
from features.feature_sets.strategy_signals import _execute_strategy_jobs, build_strategy_signals
from features.io import standardize_market_frame
from features.progress import progress_context
from features.strategy_registry import STRATEGY_REGISTRY, prepare_strategy_input


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
