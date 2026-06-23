from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.config import FeatureBuilderConfig
from ote_live.features.incremental_engine import (
    DEFAULT_RECIPE_PATH,
    IncrementalFeatureEngine,
    build_manifest_feature_plan,
    build_runtime_feature_config,
    infer_required_history_bars,
)
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.features.strategy_adapter import StrategyFeatureAdapter

REPO_ROOT = Path(__file__).resolve().parents[1]
LONG_V2_MANIFEST_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_manifests"
    / "long_reversal_tcn_v2_20260525_narrow48"
    / "live_runtime_manifest.json"
)
SHORT_V2_MANIFEST_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_manifests"
    / "short_reversal_xgb_v2_20260525"
    / "live_runtime_manifest.json"
)
EURUSD_5M_PATH = REPO_ROOT / "data" / "currency_data" / "eurusd-5m.csv"

_LAG_PATTERN = re.compile(r"^(?P<base>.+)_lag_(?P<period>\d+)$")
_ROLLING_PATTERN = re.compile(r"^(?P<base>.+)_roll_(?P<stat>mean|std)_(?P<window>\d+)$")
_ZSCORE_PATTERN = re.compile(r"^(?P<base>.+)_zscore_(?P<window>\d+)$")
_PERCENTILE_PATTERN = re.compile(r"^(?P<base>.+)_pct_rank_(?P<window>\d+)$")
_ATR_PATTERN = re.compile(r"^(?P<base>.+)_atr_norm$")
_SIGMA_PATTERN = re.compile(r"^(?P<base>.+)_sigma_norm_(?P<window>\d+)$")
_WINSOR_PATTERN = re.compile(r"^(?P<base>.+)_winsor_(?P<window>\d+)$")


def _load_manifest(path: Path) -> LiveRuntimeManifest:
    return LiveRuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _subset_manifest(
    manifest: LiveRuntimeManifest,
    selected_feature_names: list[str],
) -> LiveRuntimeManifest:
    feature_manifest = manifest.feature_manifest.model_copy(
        update={
            "selected_feature_names": list(selected_feature_names),
            "selected_feature_count": len(selected_feature_names),
            "strategy_feature_count": sum(name.startswith("strategy__") for name in selected_feature_names),
        }
    )
    return manifest.model_copy(update={"feature_manifest": feature_manifest})


def _collect_columns(
    feature_names: tuple[str, ...],
    pattern: re.Pattern[str],
    *,
    allowed_bases: set[str] | None = None,
) -> set[str]:
    columns: set[str] = set()
    for feature_name in feature_names:
        match = pattern.match(feature_name)
        if not match:
            continue
        base = match.group("base")
        if allowed_bases is not None and base not in allowed_bases:
            continue
        columns.add(base)
    return columns


def _collect_columns_and_values(
    feature_names: tuple[str, ...],
    pattern: re.Pattern[str],
    value_group: str,
    *,
    allowed_bases: set[str] | None = None,
) -> tuple[set[str], set[int]]:
    columns: set[str] = set()
    values: set[int] = set()
    for feature_name in feature_names:
        match = pattern.match(feature_name)
        if not match:
            continue
        base = match.group("base")
        if allowed_bases is not None and base not in allowed_bases:
            continue
        columns.add(base)
        values.add(int(match.group(value_group)))
    return columns, values


def test_strategy_adapter_resolves_real_candidate_manifest_prefixes() -> None:
    manifest = _load_manifest(LONG_V2_MANIFEST_PATH)

    adapter = StrategyFeatureAdapter.from_feature_names(manifest.feature_manifest.selected_feature_names)

    expected_strategy_feature_count = sum(
        name.startswith("strategy__") for name in manifest.feature_manifest.selected_feature_names
    )
    expected_unique_prefixes = {
        name.split("__")[1]
        for name in manifest.feature_manifest.selected_feature_names
        if name.startswith("strategy__")
    }

    assert adapter.has_strategy_features
    assert len(adapter.strategy_feature_names) == expected_strategy_feature_count
    assert len(adapter.strategy_ids) == len(expected_unique_prefixes)
    assert set(adapter.strategy_prefixes) == expected_unique_prefixes
    assert "moving_average_strategy" in adapter.strategy_prefixes
    assert "ichimoku_signals__ichimoku_strategy_strategy" in adapter.strategy_ids


def test_runtime_feature_config_prunes_transforms_to_selected_manifest_features() -> None:
    long_manifest = _load_manifest(LONG_V2_MANIFEST_PATH)
    short_manifest = _load_manifest(SHORT_V2_MANIFEST_PATH)

    plan = build_manifest_feature_plan([long_manifest, short_manifest])
    config = build_runtime_feature_config(plan)
    default_config = FeatureBuilderConfig.from_recipe(DEFAULT_RECIPE_PATH)

    expected_feature_names = plan.built_feature_names

    lag_columns, lag_periods = _collect_columns_and_values(
        expected_feature_names,
        _LAG_PATTERN,
        "period",
        allowed_bases=set(default_config.lag_columns),
    )
    rolling_columns, rolling_windows = _collect_columns_and_values(
        expected_feature_names,
        _ROLLING_PATTERN,
        "window",
        allowed_bases=set(default_config.rolling_stat_columns),
    )
    zscore_columns, zscore_windows = _collect_columns_and_values(
        expected_feature_names,
        _ZSCORE_PATTERN,
        "window",
        allowed_bases=set(default_config.zscore_columns),
    )
    percentile_columns, percentile_windows = _collect_columns_and_values(
        expected_feature_names,
        _PERCENTILE_PATTERN,
        "window",
        allowed_bases=set(default_config.percentile_rank_columns),
    )
    if {
        "interaction_lowvol_bull_reversion",
        "interaction_lowvol_bear_reversion",
    } & set(expected_feature_names):
        percentile_columns.add("rolling_vol_20")
        percentile_windows.add(60)
    atr_columns = _collect_columns(
        expected_feature_names,
        _ATR_PATTERN,
        allowed_bases=set(default_config.atr_normalization_columns),
    )
    sigma_columns, sigma_windows = _collect_columns_and_values(
        expected_feature_names,
        _SIGMA_PATTERN,
        "window",
        allowed_bases=set(default_config.sigma_normalization_columns),
    )
    winsor_columns, winsor_windows = _collect_columns_and_values(
        expected_feature_names,
        _WINSOR_PATTERN,
        "window",
        allowed_bases=set(default_config.winsorize_columns),
    )

    assert set(config.strategy_ids) == set(plan.strategy_adapter.strategy_ids)
    assert 0 < len(config.strategy_ids) < 50
    assert "strategy_signals" in config.feature_sets

    assert config.enable_lags == bool(lag_columns)
    assert set(config.lag_columns) == lag_columns
    assert set(config.lag_periods) == lag_periods

    assert config.enable_rolling_stats == bool(rolling_columns)
    assert set(config.rolling_stat_columns) == rolling_columns
    assert set(config.rolling_windows) == rolling_windows

    assert config.enable_zscores == bool(zscore_columns)
    if zscore_columns:
        assert set(config.zscore_columns) == zscore_columns
        assert config.zscore_window == next(iter(zscore_windows))

    assert config.enable_percentile_ranks == bool(percentile_columns)
    if percentile_columns:
        assert set(config.percentile_rank_columns) == percentile_columns
        assert config.percentile_rank_window == next(iter(percentile_windows))

    assert config.enable_atr_normalization == bool(atr_columns)
    if atr_columns:
        assert set(config.atr_normalization_columns) == atr_columns

    assert config.enable_sigma_normalization == bool(sigma_columns)
    if sigma_columns:
        assert set(config.sigma_normalization_columns) == sigma_columns
        assert config.sigma_normalization_window == next(iter(sigma_windows))

    assert config.enable_winsorization == bool(winsor_columns)
    if winsor_columns:
        assert set(config.winsorize_columns) == winsor_columns
        assert config.winsorization_window == next(iter(winsor_windows))

    assert config.enable_interactions == any(
        feature_name.startswith("interaction_") for feature_name in plan.selected_feature_names
    )


def test_infer_required_history_bars_expands_for_htf_context_features() -> None:
    assert infer_required_history_bars(["htf_30m_ema_spread_21_50_atr"]) >= 1000
    assert infer_required_history_bars(["htf_1h_ema_spread_21_50_atr"]) >= 2000
    assert infer_required_history_bars(["htf_alignment_score"]) >= 2000


def test_manifest_plan_prefers_htf_history_over_strategy_default_budget() -> None:
    manifest = _subset_manifest(
        _load_manifest(LONG_V2_MANIFEST_PATH),
        [
            "strategy__smi_signals__smi",
            "htf_1h_ema_spread_21_50_atr",
        ],
    )

    plan = build_manifest_feature_plan([manifest])

    assert plan.inferred_history_bars >= 2000
    assert plan.runtime_history_bars >= 2000


def test_incremental_engine_builds_selected_subset_with_single_strategy_feature() -> None:
    base_manifest = _load_manifest(LONG_V2_MANIFEST_PATH)
    selected_feature_names = [
        "strategy__smi_signals__smi",
        "ema_200",
        "close_vs_ema_8_atr_lag_1",
    ]
    manifest = _subset_manifest(base_manifest, selected_feature_names)

    engine = IncrementalFeatureEngine([manifest], rolling_window_bars=600)
    market_frame = pd.read_csv(EURUSD_5M_PATH).tail(600).reset_index(drop=True)

    feature_frame = engine.build_feature_frame(market_frame)
    warmup_status = engine.warmup_status(model_id=manifest.model_id, feature_frame=feature_frame)
    snapshots = engine.compute_latest_snapshots(market_frame)

    assert engine.config.strategy_ids == ["smi_signals"]
    assert "strategy_signals" in engine.config.feature_sets
    assert list(feature_frame.columns) == selected_feature_names
    assert warmup_status.ready
    assert manifest.model_id in snapshots
    assert set(snapshots[manifest.model_id].feature_values) == set(selected_feature_names)
    assert snapshots[manifest.model_id].valid_feature_count == len(selected_feature_names)


def test_incremental_engine_can_include_policy_context_features_without_changing_model_subset() -> None:
    base_manifest = _load_manifest(LONG_V2_MANIFEST_PATH)
    selected_feature_names = [
        "rsi_14",
        "dist_to_prior_high_20_atr",
        "atr_ratio_14_50_roll_mean_100",
    ]
    manifest = _subset_manifest(base_manifest, selected_feature_names)

    engine = IncrementalFeatureEngine([manifest], rolling_window_bars=600)
    market_frame = pd.read_csv(EURUSD_5M_PATH).tail(600).reset_index(drop=True)

    selected_feature_frame = engine.build_feature_frame(market_frame)
    policy_feature_frame = engine.build_feature_frame(market_frame, include_policy_features=True)

    assert list(selected_feature_frame.columns) == selected_feature_names
    assert list(policy_feature_frame.columns[: len(selected_feature_names)]) == selected_feature_names
    assert {
        "ema_alignment",
        "atr_14",
        "range_shock_20",
        "close_vs_ema_50_atr",
        "hour",
        "hour_sin",
        "hour_cos",
    }.issubset(policy_feature_frame.columns)
    latest_policy_row = policy_feature_frame.iloc[-1]
    assert pd.notna(latest_policy_row["ema_alignment"])
    assert pd.notna(latest_policy_row["atr_14"])
    assert pd.notna(latest_policy_row["range_shock_20"])
