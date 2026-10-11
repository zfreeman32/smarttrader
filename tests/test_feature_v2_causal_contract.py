"""Feature contract v2 (retrain R1): anchored weekly warm-start, bars-since, causal strategies, std floor, v1 identity."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.builder import FeatureDatasetBuilder  # noqa: E402
from features.causal_contract import (  # noqa: E402
    V2_DROPPED_STRATEGIES, add_bars_since_columns, v2_predictor_excluded, v2_strategy_ids, weekly_block_ids,
)
from features.config import FeatureBuilderConfig  # noqa: E402
from features.registry import FEATURE_REGISTRY  # noqa: E402
from features.transforms import add_rolling_zscores, rolling_sigma_normalize  # noqa: E402


def _bars(n: int, start: str = "2026-01-05 00:00", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame({"datetime": times, "open": close, "high": close + 0.3, "low": close - 0.3, "close": close,
                         "volume": rng.integers(100, 1000, n).astype(float)})


def test_weekly_anchor_is_saturday_noon_new_york() -> None:
    times = pd.Series(pd.to_datetime(["2026-03-07 16:59", "2026-03-07 17:00", "2026-03-13 21:55", "2026-03-14 15:59",
                                      "2026-03-14 17:00"], utc=True))  # 2026-03-07 is a Saturday (EST, UTC-5)
    blocks = weekly_block_ids(times)
    assert blocks[0] != blocks[1] and blocks[1] == blocks[2] and blocks[3] == blocks[2] and blocks[4] != blocks[3]
    # after US DST (2026-03-08) Saturday noon is 16:00 UTC: 15:59 is the old block, 17:00 the new
    assert blocks[3] == "2026-03-07" and blocks[4] == "2026-03-14"


@pytest.fixture()
def cumulative_feature_set():
    """A deliberately unbounded-memory producer (running sum), swapped in for the bounded v2 set 'temporal_context'."""
    import dataclasses

    saved = FEATURE_REGISTRY._specs["temporal_context"]

    def build(df, config):
        return pd.DataFrame({"toy_running_sum": pd.to_numeric(df["close"]).cumsum()}, index=df.index)

    FEATURE_REGISTRY._specs["temporal_context"] = dataclasses.replace(saved, builder=build, required_columns=("close",))
    try:
        yield
    finally:
        FEATURE_REGISTRY._specs["temporal_context"] = saved


def _toy_build(bars: pd.DataFrame, version: int) -> pd.Series:
    config = FeatureBuilderConfig(feature_sets=["temporal_context"], feature_version=version, bounded_memory_warmup_bars=500,
                                  drop_warmup_rows=False, fillna_numeric=False)
    for flag in ("enable_lags", "enable_rolling_stats", "enable_zscores", "enable_sigma_normalization", "enable_interactions",
                 "enable_atr_normalization", "enable_winsorization", "enable_percentile_ranks", "enable_fractional_diff"):
        if hasattr(config, flag):
            setattr(config, flag, False)
    frame, _ = FeatureDatasetBuilder(config).build(bars)
    return frame.set_index(pd.to_datetime(frame["datetime"], utc=True))["toy_running_sum"]


def test_v2_bounded_memory_is_window_invariant_and_v1_is_not(cumulative_feature_set) -> None:
    bars = _bars(9000)
    full_v2, late_v2 = _toy_build(bars, 2), _toy_build(bars.iloc[4000:], 2)
    tail = full_v2.index[-300:]
    np.testing.assert_allclose(full_v2.loc[tail].to_numpy(), late_v2.loc[tail].to_numpy(), rtol=0, atol=1e-9)
    full_v1, late_v1 = _toy_build(bars, 1), _toy_build(bars.iloc[4000:], 1)
    assert not np.allclose(full_v1.loc[tail].to_numpy(), late_v1.loc[tail].to_numpy())


def test_bars_since_is_translation_invariant() -> None:
    frame = pd.DataFrame({"ict_nearest_bull_fvg_formed_index": [np.nan, 0, 0, 2, 2], "other": range(5)})
    out = add_bars_since_columns(frame)
    assert out["ict_nearest_bull_fvg_formed_bars_since"].tolist()[1:] == [1.0, 2.0, 1.0, 2.0]
    shifted = add_bars_since_columns(pd.concat([pd.DataFrame({"ict_nearest_bull_fvg_formed_index": [np.nan] * 3, "other": 0}),
                                                frame.assign(ict_nearest_bull_fvg_formed_index=frame.ict_nearest_bull_fvg_formed_index + 3)],
                                               ignore_index=True))
    assert shifted["ict_nearest_bull_fvg_formed_bars_since"].tolist()[4:] == [1.0, 2.0, 1.0, 2.0]


def test_predictor_exclusions_and_strategy_ids() -> None:
    for name in ("ict_nearest_bull_fvg_formed_index", "ict_latest_bear_displacement_id", "ict_raw_swing_high",
                 "ict_raw_swing_high_lag_3", "ict_nearest_bull_fvg_formed_index_zscore_100", "frvp_open_drive_flag"):
        assert v2_predictor_excluded(name), name
    for name in ("ict_nearest_bull_fvg_formed_bars_since", "ict_confirmed_swing_high", "rsi_14", "frvp_swing_poc_vs_session_poc"):
        assert not v2_predictor_excluded(name), name
    ids = v2_strategy_ids(["rsi_signals", "high_volume_points", "tesla_stock_boom_signals", "inflation_trading_signals",
                           "ichimoku_signals__ichimoku_strategy_strategy", "rsi_signals"])
    assert ids == ["rsi_signals", "high_volume_points_causal_v2", "tesla_stock_boom_causal_v2", "ichimoku_strategy_causal_v2"]
    assert "inflation_trading_signals" in V2_DROPPED_STRATEGIES


@pytest.mark.parametrize("module", ["high_volume_points_causal_v2", "ichimoku_strategy_causal_v2", "tesla_stock_boom_causal_v2"])
def test_v2_strategies_are_causal(module: str) -> None:
    from features.strategy_registry import STRATEGY_REGISTRY

    bars = _bars(1200, seed=3)
    frame = pd.DataFrame({"Open": bars.open, "High": bars.high, "Low": bars.low, "Close": bars.close, "Volume": bars.volume})
    builder = STRATEGY_REGISTRY.load(module)
    full = builder(frame.copy())
    for cut in (700, 950, 1100):
        prefix = builder(frame.iloc[:cut].copy())
        pd.testing.assert_frame_equal(prefix, full.iloc[:cut])


def test_exact_rolling_v2_is_history_independent() -> None:
    from features.causal_contract import exact_rolling_mean_std
    from features.transforms import add_rolling_statistics

    rng = np.random.default_rng(4)  # this draw makes pandas report std ~0.045 for the constant windows below
    calm = [2.7] * 120
    with_history = pd.Series(list(rng.normal(0, 1e6, 200)) + calm)
    # v1: pandas' online variance leaks the earlier scale into a constant window
    assert (with_history.rolling(60).std().iloc[-40:] > 0).all()
    assert np.isfinite(rolling_sigma_normalize(with_history, 60).iloc[-40:]).any()
    # v2: per-window values; a constant window has std 0 -> NaN, regardless of history
    assert rolling_sigma_normalize(with_history, 60, exact=True).iloc[-40:].isna().all()
    assert add_rolling_zscores(pd.DataFrame({"x": with_history}), ["x"], 60, exact=True)["x_zscore_60"].iloc[-40:].isna().all()
    stats = add_rolling_statistics(pd.DataFrame({"x": with_history}), ["x"], [60], exact=True)
    assert (stats["x_roll_std_60"].iloc[-40:] == 0).all() and np.allclose(stats["x_roll_mean_60"].iloc[-40:], 2.7)
    mixed = pd.Series(rng.normal(0, 1, 500))
    em, es = exact_rolling_mean_std(mixed, 30)
    np.testing.assert_allclose(em, mixed.rolling(30).mean(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(es, mixed.rolling(30).std(), rtol=1e-10, atol=1e-12)


def test_absent_strategy_one_hots_are_zero_only_when_the_output_was_built() -> None:
    from features.causal_contract import add_absent_strategy_one_hots, strategy_one_hot_prefix

    assert strategy_one_hot_prefix("strategy__rsi_signals__signal__long") == "strategy__rsi_signals__signal__"
    assert strategy_one_hot_prefix("strategy__rocwb_signals__rocwb_buy_signal") is None
    frame = pd.DataFrame({"strategy__rsi_signals__signal__neutral": [1, 1], "x": [0.5, 0.6]})
    vocabulary = ["strategy__rsi_signals__signal__long", "strategy__rsi_signals__signal__neutral",
                  "strategy__emv_signals__emv_signal__long"]  # emv output not built: stays missing
    out = add_absent_strategy_one_hots(frame, vocabulary)
    assert out["strategy__rsi_signals__signal__long"].tolist() == [0, 0]
    assert "strategy__emv_signals__emv_signal__long" not in out
    assert add_absent_strategy_one_hots(frame, None) is frame


def test_v1_config_serialization_unchanged() -> None:
    v1 = FeatureBuilderConfig().to_dict()
    assert "feature_version" not in v1 and "bounded_memory_warmup_bars" not in v1
    assert "v2_strategy_one_hot_columns" not in v1 and "v2_strategy_one_hot_columns" not in FeatureBuilderConfig(feature_version=2).to_dict()
    assert FeatureBuilderConfig(feature_version=2).to_dict()["feature_version"] == 2
    assert FeatureBuilderConfig.from_dict(v1).feature_version == 1
