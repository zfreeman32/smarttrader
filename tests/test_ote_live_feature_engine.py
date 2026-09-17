from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.io import standardize_market_frame
from features.config import FeatureBuilderConfig
from ote_live.features.incremental_engine import (
    DEFAULT_RECIPE_PATH,
    IncrementalFeatureEngine,
    _materialize_live_default_features,
    _manifest_allows_nan_feature_values,
    build_manifest_feature_plan,
    build_runtime_feature_config,
    infer_required_history_bars,
)
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.features.strategy_adapter import StrategyFeatureAdapter
from ote_live.models.loaders import LoadedRuntimeModel, load_runtime_model
from ote_live.models.runners import RuntimeModelRunner

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
FRVP_SHORT_META_MANIFEST_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_manifests"
    / "frvp_es_shadow_20260715"
    / "frvp_short_meta_xgb_v1"
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


def _resolve_repo_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


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


def test_ict_xgboost_runtime_preserves_the_frozen_nan_contract() -> None:
    source_manifest = _load_manifest(FRVP_SHORT_META_MANIFEST_PATH)
    ict_manifest = source_manifest.model_copy(update={"model_id": "ict_nan_contract_xgb", "backend": "xgboost"})

    engine = IncrementalFeatureEngine([ict_manifest], rolling_window_bars=300)
    loaded_model = LoadedRuntimeModel(
        manifest=ict_manifest,
        model=object(),
        scaler=None,
        calibrator=None,
        training_summary={},
        scale_clip=8.0,
        batch_size=1,
        use_amp=False,
    )

    assert _manifest_allows_nan_feature_values(ict_manifest) is True
    assert engine.config.fillna_numeric is False
    assert loaded_model.allows_nan_feature_values is True


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


def test_runtime_feature_config_prunes_family_context_sets_for_generic_manifest_subset() -> None:
    manifest = _subset_manifest(
        _load_manifest(SHORT_V2_MANIFEST_PATH),
        [
            "rsi_14",
            "dist_to_prior_high_20_atr",
            "atr_ratio_14_50_roll_mean_100",
        ],
    )

    plan = build_manifest_feature_plan([manifest])
    config = build_runtime_feature_config(plan)

    assert "frvp_context" not in config.feature_sets
    assert "ict_context" not in config.feature_sets


def test_runtime_feature_config_keeps_family_context_sets_when_requested() -> None:
    manifest = _subset_manifest(
        _load_manifest(SHORT_V2_MANIFEST_PATH),
        [
            "frvp_dist_poc_session_atr",
            "ict_nearest_bull_fvg_ce",
            "dist_to_bull_fvg_atr",
        ],
    )

    plan = build_manifest_feature_plan([manifest])
    config = build_runtime_feature_config(plan)

    assert "frvp_context" in config.feature_sets
    assert "ict_context" in config.feature_sets


def test_runtime_feature_config_keeps_frvp_context_for_target_specific_htf_confluence_features() -> None:
    manifest = _subset_manifest(
        _load_manifest(SHORT_V2_MANIFEST_PATH),
        [
            "htf_confluence_short_frvp_continuation",
            "htf_confluence_short_frvp_reversal",
        ],
    )

    plan = build_manifest_feature_plan([manifest])
    config = build_runtime_feature_config(plan)

    assert "frvp_context" in config.feature_sets
    assert "ict_context" not in config.feature_sets


def test_runtime_feature_config_keeps_ict_context_for_interaction_only_manifest_subset() -> None:
    manifest = _subset_manifest(
        _load_manifest(FRVP_SHORT_META_MANIFEST_PATH),
        [
            "interaction_bull_structure_proximity",
        ],
    )

    plan = build_manifest_feature_plan([manifest])
    config = build_runtime_feature_config(plan)

    assert "ict_context" in config.feature_sets


def test_runtime_feature_config_enables_requested_ict_interaction_family() -> None:
    manifest = _subset_manifest(
        _load_manifest(FRVP_SHORT_META_MANIFEST_PATH),
        ["ict_bear_zone_in_premium"],
    )

    plan = build_manifest_feature_plan((manifest,))
    config = build_runtime_feature_config(plan)

    assert "ict_context" in config.feature_sets
    assert "ict_interactions" in config.feature_sets


def test_runtime_feature_config_enables_ict_context_derived_transforms() -> None:
    manifest = _subset_manifest(
        _load_manifest(FRVP_SHORT_META_MANIFEST_PATH),
        [
            "ict_total_confluence_1atr_lag_1",
            "ict_total_confluence_1atr_roll_mean_20",
            "ict_zone_balance_1atr_roll_std_50",
            "ict_zone_balance_1atr_zscore_60",
        ],
    )

    config = build_runtime_feature_config(build_manifest_feature_plan([manifest]))

    assert config.lag_columns == ["ict_total_confluence_1atr"]
    assert config.lag_periods == [1]
    assert config.rolling_stat_columns == [
        "ict_total_confluence_1atr",
        "ict_zone_balance_1atr",
    ]
    assert config.rolling_windows == [20, 50]
    assert config.zscore_columns == ["ict_zone_balance_1atr"]
    assert config.zscore_window == 60


def test_live_feature_defaults_materialize_sparse_htf_confluence_helpers() -> None:
    source = pd.DataFrame({"close": [6000.0, 6001.0]})

    materialized = _materialize_live_default_features(
        source,
        requested_feature_names=(
            "close",
            "htf_confluence_long_frvp_continuation",
            "htf_confluence_short_ict_reversal",
        ),
    )

    assert materialized["htf_confluence_long_frvp_continuation"].tolist() == [0, 0]
    assert materialized["htf_confluence_short_ict_reversal"].tolist() == [0, 0]


def test_infer_required_history_bars_expands_for_htf_context_features() -> None:
    assert infer_required_history_bars(["frvp_va_width_zscore_20"]) >= 8000
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


def test_manifest_plan_expands_for_frvp_prior_rth_history_features() -> None:
    manifest = _subset_manifest(
        _load_manifest(FRVP_SHORT_META_MANIFEST_PATH),
        [
            "frvp_va_width_zscore_20",
            "htf_confluence_short_frvp_continuation",
        ],
    )

    plan = build_manifest_feature_plan([manifest])

    assert plan.inferred_history_bars >= 8000
    assert plan.runtime_history_bars >= 8000


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


def test_incremental_engine_reuses_snapshot_scoped_feature_cache_for_identical_market_frame() -> None:
    base_manifest = _load_manifest(LONG_V2_MANIFEST_PATH)
    selected_feature_names = [
        "rsi_14",
        "dist_to_prior_high_20_atr",
        "atr_ratio_14_50_roll_mean_100",
    ]
    manifest = _subset_manifest(base_manifest, selected_feature_names)

    engine = IncrementalFeatureEngine([manifest], rolling_window_bars=300)
    market_frame = pd.read_csv(EURUSD_5M_PATH).tail(300).reset_index(drop=True)

    first = engine.build_feature_frame(market_frame, include_policy_features=True)
    second = engine.build_feature_frame(market_frame.copy(), include_policy_features=True)

    pd.testing.assert_frame_equal(first, second)
    assert engine.feature_build_count == 1
    assert engine.feature_cache_hit_count == 1

    repaired = market_frame.copy()
    repaired.loc[repaired.index[-1], "Close"] += 0.0001
    engine.build_feature_frame(repaired, include_policy_features=True)

    assert engine.feature_build_count == 2


def test_incremental_engine_preserves_frvp_htf_confluence_carry_through_features() -> None:
    base_manifest = _load_manifest(FRVP_SHORT_META_MANIFEST_PATH)
    selected_feature_names = [
        "htf_confluence_short_frvp_continuation",
        "htf_confluence_short_frvp_reversal",
    ]
    manifest = _subset_manifest(base_manifest, selected_feature_names)

    upstream_path = _resolve_repo_path(manifest.source_lineage.upstream_source_path)
    feature_csv_path = _resolve_repo_path(manifest.source_lineage.feature_csv)
    upstream_market_frame = standardize_market_frame(
        pd.read_csv(upstream_path),
        source_timezone=manifest.source_lineage.upstream_timezone_contract.source_timezone,
        canonical_timezone=manifest.timezone_contract.canonical_timezone,
    )
    upstream_market_frame["datetime"] = pd.to_datetime(
        upstream_market_frame["datetime"],
        errors="coerce",
        utc=True,
    )
    upstream_market_frame = upstream_market_frame.dropna(subset=["datetime"])

    feature_context_frame = pd.read_csv(
        feature_csv_path,
        usecols=[
            "datetime",
            *selected_feature_names,
        ],
    )
    feature_context_frame["datetime"] = pd.to_datetime(
        feature_context_frame["datetime"],
        errors="coerce",
        utc=True,
    )
    feature_context_frame = feature_context_frame.dropna(subset=["datetime"])
    feature_context_frame = feature_context_frame.drop_duplicates(subset=["datetime"], keep="last")

    market_frame = (
        upstream_market_frame.merge(
            feature_context_frame,
            on="datetime",
            how="left",
        )
        .tail(600)
        .reset_index(drop=True)
    )

    engine = IncrementalFeatureEngine([manifest], rolling_window_bars=600)
    feature_frame = engine.build_feature_frame(market_frame)

    assert list(feature_frame.columns) == selected_feature_names
    pd.testing.assert_series_equal(
        feature_frame["htf_confluence_short_frvp_continuation"].reset_index(drop=True),
        market_frame["htf_confluence_short_frvp_continuation"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        feature_frame["htf_confluence_short_frvp_reversal"].reset_index(drop=True),
        market_frame["htf_confluence_short_frvp_reversal"].reset_index(drop=True),
        check_names=False,
    )


def test_runtime_model_runner_accepts_frvp_xgboost_selected_nan_features() -> None:
    manifest = _load_manifest(FRVP_SHORT_META_MANIFEST_PATH)
    loaded_model = load_runtime_model(manifest)
    runner = RuntimeModelRunner(loaded_model)

    feature_csv_path = _resolve_repo_path(manifest.source_lineage.feature_csv)
    feature_frame = pd.read_csv(
        feature_csv_path,
        usecols=lambda candidate: candidate in {"datetime", *manifest.feature_manifest.selected_feature_names},
    )
    feature_frame["datetime"] = pd.to_datetime(
        feature_frame["datetime"],
        errors="coerce",
        utc=True,
    )
    feature_frame = feature_frame.dropna(subset=["datetime"]).reset_index(drop=True)

    target_timestamp = pd.Timestamp("2026-06-17T00:10:00+00:00")
    target_index = int(feature_frame.index[feature_frame["datetime"] == target_timestamp][0])
    window = feature_frame.iloc[target_index - 30 : target_index + 1].reset_index(drop=True)
    window["timestamp"] = window["datetime"]

    prediction = runner.predict_latest(
        window,
        timestamp=target_timestamp.to_pydatetime(),
        source_row_idx=target_index,
        regime="ranging_medium",
    )

    assert prediction.model_id == manifest.model_id


def test_runtime_model_runner_snapshot_normalizes_frvp_selected_nan_features_to_none() -> None:
    manifest = _load_manifest(FRVP_SHORT_META_MANIFEST_PATH)
    loaded_model = load_runtime_model(manifest)
    runner = RuntimeModelRunner(loaded_model)

    feature_csv_path = _resolve_repo_path(manifest.source_lineage.feature_csv)
    feature_frame = pd.read_csv(
        feature_csv_path,
        usecols=lambda candidate: candidate in {"datetime", *manifest.feature_manifest.selected_feature_names},
    )
    feature_frame["datetime"] = pd.to_datetime(
        feature_frame["datetime"],
        errors="coerce",
        utc=True,
    )
    feature_frame = feature_frame.dropna(subset=["datetime"]).reset_index(drop=True)

    target_timestamp = pd.Timestamp("2026-06-17T00:10:00+00:00")
    target_index = int(feature_frame.index[feature_frame["datetime"] == target_timestamp][0])
    window = feature_frame.iloc[target_index - 30 : target_index + 1].reset_index(drop=True)
    window["timestamp"] = window["datetime"]

    snapshot = runner.build_latest_snapshot(window)

    assert snapshot.feature_values["frvp_dist_ib_low_atr"] is None
