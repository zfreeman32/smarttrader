from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from features.builder import FeatureDatasetBuilder
from features.config import FeatureBuilderConfig
from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.features.manifest import LiveRuntimeManifest
from ote_live.features.runtime_state import FeatureRuntimeState
from ote_live.features.strategy_adapter import StrategyFeatureAdapter
from ote_live.features.warmup import (
    WarmupStatus,
    evaluate_feature_warmup,
    resolve_required_history_bars,
    resolve_runtime_history_bars,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE_PATH = REPO_ROOT / "features" / "recipes" / "ote_extended.json"
DEFAULT_WEEKLY_CONTEXT_HISTORY_BARS = 8000
DEFAULT_DAILY_CONTEXT_HISTORY_BARS = 2000
DEFAULT_HTF_30M_CONTEXT_HISTORY_BARS = 1000
DEFAULT_HTF_1H_CONTEXT_HISTORY_BARS = 2000
DEFAULT_FRVP_PRIOR_RTH_HISTORY_BARS = 8000
ADDITIVE_SEED_PREFIX_BARS = 1
ADDITIVE_CUMULATIVE_FEATURE_NAMES = frozenset(
    {
        "strategy__pvi_signals__pvi",
        "strategy__ad_indicator_signals__ad_line",
    }
)

_LAG_PATTERN = re.compile(r"^(?P<base>.+)_lag_(?P<period>\d+)$")
_ROLLING_PATTERN = re.compile(r"^(?P<base>.+)_roll_(?P<stat>mean|std)_(?P<window>\d+)$")
_ZSCORE_PATTERN = re.compile(r"^(?P<base>.+)_zscore_(?P<window>\d+)$")
_PERCENTILE_PATTERN = re.compile(r"^(?P<base>.+)_pct_rank_(?P<window>\d+)$")
_ATR_PATTERN = re.compile(r"^(?P<base>.+)_atr_norm$")
_SIGMA_PATTERN = re.compile(r"^(?P<base>.+)_sigma_norm_(?P<window>\d+)$")
_WINSOR_PATTERN = re.compile(r"^(?P<base>.+)_winsor_(?P<window>\d+)$")
_LOWVOL_INTERACTIONS = {
    "interaction_lowvol_bull_reversion",
    "interaction_lowvol_bear_reversion",
}
_RUNTIME_CONTEXT_TRANSFORM_BASE_FEATURE_NAMES = (
    "ict_total_confluence_1atr",
    "ict_zone_balance_1atr",
)
_LIVE_ZERO_DEFAULT_FEATURE_PREFIXES = (
    "htf_confluence_",
)
_ICT_INTERACTION_FEATURE_NAMES = frozenset(
    {
        "ict_bull_sweep_plus_fvg",
        "ict_bear_sweep_plus_fvg",
        "ict_bull_sweep_plus_ob",
        "ict_bear_sweep_plus_ob",
        "ict_bull_choch_after_sweep",
        "ict_bear_choch_after_sweep",
        "ict_sweep_plus_choch",
        "ict_displacement_after_sweep",
        "ict_fvg_retrace_after_displacement",
        "ict_bull_zone_in_discount",
        "ict_bear_zone_in_premium",
    }
)
_FRVP_PRIOR_RTH_HISTORY_FEATURE_NAMES = frozenset(
    {
        "frvp_day_type",
        "frvp_poc_migration_vel",
        "frvp_va_expansion",
        "frvp_va_migration_vel",
        "frvp_va_width_zscore_20",
    }
)
POLICY_CONTEXT_FEATURE_NAMES = (
    "ema_alignment",
    "atr_14",
    "range_shock_20",
    "close_vs_ema_50_atr",
    "hour",
    "hour_sin",
    "hour_cos",
    "in_asian_session",
    "in_london_session",
    "in_newyork_session",
)
FRVP_DASHBOARD_EXTRA_FEATURE_NAMES = (
    "atr_14",
    "volume_zscore_50",
    "displacement_bullish",
    "displacement_bearish",
    "bars_since_sweep_high",
    "bars_since_sweep_low",
    "frvp_dist_poc_session_atr",
    "frvp_dist_vah_atr",
    "frvp_dist_val_atr",
    "frvp_dist_ib_high_atr",
    "frvp_dist_ib_low_atr",
    "frvp_naked_vpoc_dist_above_atr",
    "frvp_naked_vpoc_dist_below_atr",
    "frvp_naked_vpoc_count",
    "frvp_naked_vpoc_age_sessions",
    "frvp_setup_type",
    "frvp_setup_side",
    "frvp_setup_confidence_rule",
    "frvp_profile_shape",
    "frvp_open_type",
    "frvp_open_drive_flag",
    "frvp_in_va",
    "frvp_above_vah",
    "frvp_below_val",
    "frvp_dist_nearest_lvn_atr",
    "frvp_hvn_above_close",
    "frvp_hvn_below_close",
)
ICT_DASHBOARD_EXTRA_FEATURE_NAMES = (
    "atr_14",
    "htf_30m_ema_alignment",
    "htf_1h_ema_alignment",
    "dist_to_bull_fvg_atr",
    "dist_to_bear_fvg_atr",
    "dist_to_bull_order_block_atr",
    "dist_to_bear_order_block_atr",
    "ict_buy_side_sweep",
    "ict_sell_side_sweep",
    "ict_sweep_direction",
    "ict_sweep_level_code",
    "ict_sweep_level_value",
    "ict_sweep_penetration_atr",
    "ict_structure_state",
    "ict_impulse_direction",
    "ict_displacement_score",
    "ict_displacement_volume_zscore",
    "ict_ib_complete",
    "ict_is_rth",
    "ict_lunch_lull_flag",
    "ict_close_ramp_flag",
    "ict_in_ote_band",
    "ict_premium_zone",
    "ict_discount_zone",
    "ict_session_phase_code",
    "ict_session_vwap",
    "ict_bars_since_buy_side_sweep",
    "ict_bars_since_sell_side_sweep",
    "ict_bars_since_displacement_bull",
    "ict_bars_since_displacement_bear",
    "ict_bars_since_choch_bull",
    "ict_bars_since_choch_bear",
    "ict_bars_since_mss_bull",
    "ict_bars_since_mss_bear",
    "ict_latest_bull_displacement_id",
    "ict_latest_bull_displacement_index",
    "ict_latest_bull_displacement_origin",
    "ict_latest_bear_displacement_id",
    "ict_latest_bear_displacement_index",
    "ict_latest_bear_displacement_origin",
    "ict_nearest_bull_fvg_id",
    "ict_nearest_bull_fvg_lower",
    "ict_nearest_bull_fvg_upper",
    "ict_nearest_bull_fvg_ce",
    "ict_nearest_bull_fvg_is_ifvg",
    "ict_nearest_bull_fvg_created_by_displacement",
    "ict_nearest_bull_fvg_inversion_index",
    "ict_nearest_bear_fvg_id",
    "ict_nearest_bear_fvg_lower",
    "ict_nearest_bear_fvg_upper",
    "ict_nearest_bear_fvg_ce",
    "ict_nearest_bear_fvg_is_ifvg",
    "ict_nearest_bear_fvg_created_by_displacement",
    "ict_nearest_bear_fvg_inversion_index",
    "ict_nearest_bull_order_block_id",
    "ict_nearest_bull_order_block_lower",
    "ict_nearest_bull_order_block_upper",
    "ict_nearest_bear_order_block_id",
    "ict_nearest_bear_order_block_lower",
    "ict_nearest_bear_order_block_upper",
    "ict_bull_order_block_retest_event",
    "ict_bear_order_block_retest_event",
    "ict_prior_rth_high",
    "ict_prior_rth_low",
    "ict_prior_rth_close",
    "ict_rth_open",
    "ict_overnight_high",
    "ict_overnight_low",
    "ict_prior_week_high",
    "ict_prior_week_low",
    "ict_ib_high",
    "ict_ib_low",
    "ict_midnight_open",
    "ict_open_0830",
    "ict_dol_level_up",
    "ict_dol_level_down",
    "ict_latest_swing_high",
    "ict_latest_swing_low",
)
ICT_CONTEXT_NONPREFIX_FEATURE_NAMES = frozenset(
    {
        "dist_to_bull_fvg_atr",
        "dist_to_bear_fvg_atr",
        "dist_to_bull_order_block_atr",
        "dist_to_bear_order_block_atr",
    }
)
_ASSET_TO_FEATURE_INSTRUMENT = {
    "ES": "es",
    "6E": "6e",
}


@dataclass(frozen=True)
class ManifestFeaturePlan:
    asset: str
    timeframe: str
    selected_feature_names: tuple[str, ...]
    extra_feature_names: tuple[str, ...]
    policy_feature_names: tuple[str, ...]
    built_feature_names: tuple[str, ...]
    model_feature_names: dict[str, tuple[str, ...]]
    strategy_adapter: StrategyFeatureAdapter
    additive_seed_feature_names: tuple[str, ...]
    additive_seed_prefix_bars: int
    inferred_history_bars: int
    minimum_history_bars: int
    runtime_history_bars: int


def build_manifest_feature_plan(
    manifests: Iterable[LiveRuntimeManifest],
    *,
    rolling_window_bars: int | None = None,
    extra_feature_names: Sequence[str] = (),
) -> ManifestFeaturePlan:
    manifests = tuple(manifests)
    if not manifests:
        raise ValueError("At least one live manifest is required.")

    asset = manifests[0].asset
    timeframe = manifests[0].timeframe
    for manifest in manifests[1:]:
        if manifest.asset != asset:
            raise ValueError(f"All manifests must share the same asset. Got '{asset}' and '{manifest.asset}'.")
        if manifest.timeframe != timeframe:
            raise ValueError(
                f"All manifests must share the same timeframe. Got '{timeframe}' and '{manifest.timeframe}'."
            )

    selected_feature_names: list[str] = []
    model_feature_names: dict[str, tuple[str, ...]] = {}
    for manifest in manifests:
        feature_names = tuple(manifest.feature_manifest.selected_feature_names)
        model_feature_names[manifest.model_id] = feature_names
        for feature_name in feature_names:
            if feature_name not in selected_feature_names:
                selected_feature_names.append(feature_name)

    resolved_extra_feature_names = tuple(
        feature_name
        for feature_name in dict.fromkeys(str(name) for name in extra_feature_names if str(name))
        if feature_name not in selected_feature_names
    )

    policy_feature_names = tuple(
        feature_name
        for feature_name in POLICY_CONTEXT_FEATURE_NAMES
        if feature_name not in selected_feature_names and feature_name not in resolved_extra_feature_names
    )
    built_feature_names = tuple((*selected_feature_names, *resolved_extra_feature_names, *policy_feature_names))
    strategy_adapter = StrategyFeatureAdapter.from_feature_names(built_feature_names)
    additive_seed_feature_names = tuple(
        feature_name
        for feature_name in built_feature_names
        if feature_name in ADDITIVE_CUMULATIVE_FEATURE_NAMES
    )
    additive_seed_prefix_bars = ADDITIVE_SEED_PREFIX_BARS if additive_seed_feature_names else 0
    inferred_history_bars = infer_required_history_bars(built_feature_names)
    minimum_history_bars = max(resolve_required_history_bars(manifests), inferred_history_bars)
    runtime_history_bars = resolve_runtime_history_bars(
        manifests,
        rolling_window_bars=rolling_window_bars,
        has_strategy_features=strategy_adapter.has_strategy_features,
        inferred_history_bars=inferred_history_bars,
    )

    return ManifestFeaturePlan(
        asset=asset,
        timeframe=timeframe,
        selected_feature_names=tuple(selected_feature_names),
        extra_feature_names=resolved_extra_feature_names,
        policy_feature_names=policy_feature_names,
        built_feature_names=built_feature_names,
        model_feature_names=model_feature_names,
        strategy_adapter=strategy_adapter,
        additive_seed_feature_names=additive_seed_feature_names,
        additive_seed_prefix_bars=additive_seed_prefix_bars,
        inferred_history_bars=inferred_history_bars,
        minimum_history_bars=minimum_history_bars,
        runtime_history_bars=runtime_history_bars,
    )


def build_runtime_feature_config(
    plan: ManifestFeaturePlan,
    *,
    recipe_path: str | Path = DEFAULT_RECIPE_PATH,
    transform_workers: int | None = None,
    strategy_timeout_seconds: float | None = None,
) -> FeatureBuilderConfig:
    config = FeatureBuilderConfig.from_recipe(recipe_path)
    config.drop_warmup_rows = False
    config.fillna_numeric = True

    if transform_workers is not None:
        config.transform_workers = max(1, int(transform_workers))
    if strategy_timeout_seconds is not None:
        config.strategy_timeout_seconds = None if strategy_timeout_seconds <= 0 else float(strategy_timeout_seconds)

    _prune_feature_sets(config, plan.built_feature_names)
    _prune_transform_settings(config, plan.built_feature_names)
    plan.strategy_adapter.apply_to_config(config)
    if any(feature_name.startswith(("frvp_", "ict_")) for feature_name in plan.built_feature_names):
        config.instrument = _ASSET_TO_FEATURE_INSTRUMENT.get(plan.asset.upper(), config.instrument)
    return config


class IncrementalFeatureEngine:
    def __init__(
        self,
        manifests: Sequence[LiveRuntimeManifest],
        *,
        recipe_path: str | Path = DEFAULT_RECIPE_PATH,
        rolling_window_bars: int | None = None,
        transform_workers: int | None = None,
        strategy_timeout_seconds: float | None = None,
        feature_seed_offsets: dict[str, float] | None = None,
        extra_feature_names: Sequence[str] = (),
    ) -> None:
        self.manifests = tuple(manifests)
        self.plan = build_manifest_feature_plan(
            self.manifests,
            rolling_window_bars=rolling_window_bars,
            extra_feature_names=extra_feature_names,
        )
        self.config = build_runtime_feature_config(
            self.plan,
            recipe_path=recipe_path,
            transform_workers=transform_workers,
            strategy_timeout_seconds=strategy_timeout_seconds,
        )
        if _manifests_preserve_numeric_nans(self.manifests):
            self.config.fillna_numeric = False
        self.builder = FeatureDatasetBuilder(self.config)
        self.state = FeatureRuntimeState(
            max_history_bars=self.plan.runtime_history_bars + self.plan.additive_seed_prefix_bars
        )
        self.feature_seed_offsets = {
            feature_name: float(value)
            for feature_name, value in (feature_seed_offsets or {}).items()
            if feature_name in self.plan.additive_seed_feature_names
        }
        self._cached_feature_frame: pd.DataFrame | None = None
        self._cached_state_signature: tuple[object | None, int] | None = None

    def ingest_bar(self, bar: MarketBar) -> None:
        next_seed_offsets = self._rolling_seed_offsets_for_next_window(bar)
        self.state.ingest_bar(bar)
        if next_seed_offsets is not None:
            self.feature_seed_offsets = next_seed_offsets
        self._invalidate_cache()

    def extend(self, bars: Iterable[MarketBar]) -> None:
        for bar in bars:
            self.ingest_bar(bar)

    def build_feature_frame(
        self,
        market_frame: pd.DataFrame | None = None,
        *,
        include_policy_features: bool = False,
    ) -> pd.DataFrame:
        if market_frame is None:
            signature = (self.state.latest_timestamp, len(self.state))
            if self._cached_state_signature == signature and self._cached_feature_frame is not None:
                cached = self._cached_feature_frame.copy()
                if include_policy_features:
                    return cached
                return cached.loc[:, list(self.plan.selected_feature_names)].copy()
            source_frame = self.state.to_frame()
        else:
            signature = None
            source_frame = market_frame

        if source_frame.empty:
            columns = self.plan.built_feature_names if include_policy_features else self.plan.selected_feature_names
            empty = pd.DataFrame(columns=list(columns))
            if signature is not None:
                self._cached_feature_frame = empty
                self._cached_state_signature = signature
            return empty.copy()

        bounded = self._slice_runtime_window(source_frame)
        built_frame, _ = self.builder.build(bounded)
        built_frame = self._apply_feature_seed_offsets(built_frame)
        built_frame = _materialize_live_default_features(
            built_frame,
            requested_feature_names=self.plan.built_feature_names,
        )
        missing_feature_names = [
            feature_name for feature_name in self.plan.built_feature_names if feature_name not in built_frame.columns
        ]
        if missing_feature_names:
            raise ValueError(
                "Live feature build did not produce all selected features: "
                + ", ".join(missing_feature_names[:20])
            )

        available = built_frame.loc[:, list(self.plan.built_feature_names)].copy()
        if len(available) > self.plan.runtime_history_bars:
            available = available.iloc[-self.plan.runtime_history_bars :].reset_index(drop=True)
        if signature is not None:
            self._cached_feature_frame = available
            self._cached_state_signature = signature
        if include_policy_features:
            return available.copy()
        return available.loc[:, list(self.plan.selected_feature_names)].copy()

    def warmup_status(
        self,
        *,
        model_id: str | None = None,
        feature_frame: pd.DataFrame | None = None,
    ) -> WarmupStatus:
        feature_frame = feature_frame if feature_frame is not None else self.build_feature_frame()
        if model_id is None:
            return evaluate_feature_warmup(
                feature_frame,
                self.plan.selected_feature_names,
                minimum_history_bars=self.plan.minimum_history_bars,
                allow_nan_values=all(_manifest_allows_nan_feature_values(manifest) for manifest in self.manifests),
            )

        manifest = self._manifest_by_id(model_id)
        return evaluate_feature_warmup(
            feature_frame,
            manifest.feature_manifest.selected_feature_names,
            minimum_history_bars=manifest.context_requirements.minimum_runtime_history_bars,
            allow_nan_values=_manifest_allows_nan_feature_values(manifest),
        )

    def compute_latest_snapshots(
        self,
        market_frame: pd.DataFrame | None = None,
        *,
        require_ready: bool = True,
    ) -> dict[str, FeatureSnapshot]:
        source_frame = self.state.to_frame() if market_frame is None else market_frame
        feature_frame = self.build_feature_frame(market_frame)
        if feature_frame.empty:
            return {}

        timestamp = self._extract_latest_timestamp(source_frame)
        source_row_idx = int(len(feature_frame) - 1)
        snapshots: dict[str, FeatureSnapshot] = {}

        for manifest in self.manifests:
            status = self.warmup_status(model_id=manifest.model_id, feature_frame=feature_frame)
            if require_ready and not status.ready:
                continue

            latest_values = feature_frame.iloc[-1].loc[list(manifest.feature_manifest.selected_feature_names)]
            snapshots[manifest.model_id] = FeatureSnapshot(
                asset=manifest.asset,
                timeframe=manifest.timeframe,
                direction=manifest.direction,
                timestamp=timestamp,
                source_row_idx=source_row_idx,
                feature_values={
                    feature_name: _to_python_scalar(value)
                    for feature_name, value in latest_values.items()
                },
                valid_feature_count=status.valid_feature_count,
            )

        return snapshots

    def _slice_runtime_window(self, market_frame: pd.DataFrame) -> pd.DataFrame:
        runtime_window_bars = self.plan.runtime_history_bars + self.plan.additive_seed_prefix_bars
        if len(market_frame) <= runtime_window_bars:
            return market_frame.copy()
        return market_frame.iloc[-runtime_window_bars:].reset_index(drop=True)

    def _invalidate_cache(self) -> None:
        self._cached_feature_frame = None
        self._cached_state_signature = None

    def set_feature_seed_offsets(self, feature_seed_offsets: dict[str, float] | None) -> None:
        self.feature_seed_offsets = {
            feature_name: float(value)
            for feature_name, value in (feature_seed_offsets or {}).items()
            if feature_name in self.plan.additive_seed_feature_names
        }
        self._invalidate_cache()

    def get_feature_seed_offsets(self) -> dict[str, float]:
        return dict(self.feature_seed_offsets)

    def derive_feature_seed_offsets_from_market_frame(self, market_frame: pd.DataFrame) -> dict[str, float]:
        if not self.plan.additive_seed_feature_names:
            return {}

        current_window = self.state.to_frame()
        if current_window.empty:
            return {}
        if len(market_frame) < len(current_window):
            raise ValueError("market_frame must contain at least as many rows as the current runtime window.")
        if len(market_frame) == len(current_window):
            return {}

        built_frame, _ = self.builder.build(market_frame.copy())
        if len(built_frame) < len(current_window):
            return {}

        window_start_row = built_frame.iloc[-len(current_window)]
        offsets: dict[str, float] = {}
        for feature_name in self.plan.additive_seed_feature_names:
            if feature_name not in window_start_row.index:
                continue
            value = _coerce_numeric_scalar(window_start_row[feature_name])
            if value is None:
                continue
            offsets[feature_name] = value
        return offsets

    @staticmethod
    def market_frame_from_bars(bars: Iterable[MarketBar]) -> pd.DataFrame:
        rows = [
            {
                "asset": bar.asset,
                "timeframe": bar.timeframe,
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "bid": bar.bid,
                "ask": bar.ask,
                "spread": bar.spread,
                "source": bar.source,
                "symbol": bar.symbol,
                "contract_symbol": bar.contract_symbol,
                "instrument_id": bar.instrument_id,
            }
            for bar in bars
        ]
        return pd.DataFrame.from_records(rows)

    def _apply_feature_seed_offsets(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_seed_offsets:
            return feature_frame

        adjusted = feature_frame.copy()
        for feature_name, offset in self.feature_seed_offsets.items():
            if feature_name not in adjusted.columns:
                continue
            adjusted[feature_name] = pd.to_numeric(adjusted[feature_name], errors="coerce") + float(offset)
        return adjusted

    def _rolling_seed_offsets_for_next_window(self, incoming_bar: MarketBar) -> dict[str, float] | None:
        if not self.plan.additive_seed_feature_names:
            return None
        if not self.feature_seed_offsets:
            return None
        if len(self.state) < self.state.max_history_bars:
            return None
        if self.state.latest_timestamp is None or incoming_bar.timestamp <= self.state.latest_timestamp:
            return None

        feature_frame = self.build_feature_frame()
        rollover_index = 0 if self.plan.additive_seed_prefix_bars else 1
        if len(feature_frame) <= rollover_index:
            return None

        next_offsets: dict[str, float] = {}
        rollover_row = feature_frame.iloc[rollover_index]
        for feature_name in self.plan.additive_seed_feature_names:
            value = _coerce_numeric_scalar(rollover_row.get(feature_name))
            if value is not None:
                next_offsets[feature_name] = value
            elif feature_name in self.feature_seed_offsets:
                next_offsets[feature_name] = float(self.feature_seed_offsets[feature_name])
        return next_offsets

    def _manifest_by_id(self, model_id: str) -> LiveRuntimeManifest:
        for manifest in self.manifests:
            if manifest.model_id == model_id:
                return manifest
        raise KeyError(f"Unknown live manifest model_id '{model_id}'.")

    @staticmethod
    def _extract_latest_timestamp(market_frame: pd.DataFrame):
        if market_frame.empty:
            raise ValueError("Cannot build a feature snapshot from an empty market frame.")

        for column in ("timestamp", "datetime"):
            if column in market_frame.columns:
                series = pd.to_datetime(market_frame[column], errors="coerce")
                if series.notna().any():
                    return series.iloc[-1].to_pydatetime()
        for date_column, time_column in (("date", "time"), ("Date", "Time")):
            if date_column in market_frame.columns and time_column in market_frame.columns:
                combined = (
                    market_frame[date_column].astype(str).str.strip()
                    + " "
                    + market_frame[time_column].astype(str).str.strip()
                )
                series = pd.to_datetime(combined, errors="coerce")
                if series.notna().any():
                    return series.iloc[-1].to_pydatetime()
        for column in ("date", "Date"):
            if column in market_frame.columns:
                series = pd.to_datetime(market_frame[column], errors="coerce")
                if series.notna().any():
                    return series.iloc[-1].to_pydatetime()
        raise ValueError("Live market frame does not contain a usable timestamp or datetime column.")


def _prune_transform_settings(config: FeatureBuilderConfig, selected_feature_names: Sequence[str]) -> None:
    selected_feature_names = tuple(dict.fromkeys(selected_feature_names))

    # The ICT confluence bases are emitted by the ict_context feature set rather
    # than the base OTE recipe. Add them to the transform allowlists before
    # pruning so manifest-selected lag/rolling/z-score derivatives are built.
    config.lag_columns = list(
        dict.fromkeys((*config.lag_columns, *_RUNTIME_CONTEXT_TRANSFORM_BASE_FEATURE_NAMES))
    )
    config.rolling_stat_columns = list(
        dict.fromkeys((*config.rolling_stat_columns, *_RUNTIME_CONTEXT_TRANSFORM_BASE_FEATURE_NAMES))
    )
    config.zscore_columns = list(
        dict.fromkeys((*config.zscore_columns, *_RUNTIME_CONTEXT_TRANSFORM_BASE_FEATURE_NAMES))
    )

    lag_bases, lag_periods = _collect_columns_and_periods(
        selected_feature_names,
        _LAG_PATTERN,
        "period",
        allowed_bases=set(config.lag_columns),
    )
    config.enable_lags = bool(lag_bases)
    if lag_bases:
        config.lag_columns = _ordered_strings(config.lag_columns, lag_bases)
        config.lag_periods = _ordered_ints(config.lag_periods, lag_periods)

    rolling_bases, rolling_windows = _collect_columns_and_periods(
        selected_feature_names,
        _ROLLING_PATTERN,
        "window",
        allowed_bases=set(config.rolling_stat_columns),
    )
    config.enable_rolling_stats = bool(rolling_bases)
    if rolling_bases:
        config.rolling_stat_columns = _ordered_strings(config.rolling_stat_columns, rolling_bases)
        config.rolling_windows = _ordered_ints(config.rolling_windows, rolling_windows)

    zscore_bases, zscore_windows = _collect_columns_and_periods(
        selected_feature_names,
        _ZSCORE_PATTERN,
        "window",
        allowed_bases=set(config.zscore_columns),
    )
    config.enable_zscores = bool(zscore_bases)
    if zscore_bases:
        config.zscore_columns = _ordered_strings(config.zscore_columns, zscore_bases)
        config.zscore_window = _resolve_single_window(zscore_windows, transform_name="z-score")

    percentile_bases, percentile_windows = _collect_columns_and_periods(
        selected_feature_names,
        _PERCENTILE_PATTERN,
        "window",
        allowed_bases=set(config.percentile_rank_columns),
    )
    if any(name in _LOWVOL_INTERACTIONS for name in selected_feature_names):
        percentile_bases.add("rolling_vol_20")
        percentile_windows.add(60)
    config.enable_percentile_ranks = bool(percentile_bases)
    if percentile_bases:
        config.percentile_rank_columns = _ordered_strings(config.percentile_rank_columns, percentile_bases)
        config.percentile_rank_window = _resolve_single_window(
            percentile_windows,
            transform_name="percentile-rank",
        )

    atr_bases = _collect_columns(
        selected_feature_names,
        _ATR_PATTERN,
        allowed_bases=set(config.atr_normalization_columns),
    )
    config.enable_atr_normalization = bool(atr_bases)
    if atr_bases:
        config.atr_normalization_columns = _ordered_strings(config.atr_normalization_columns, atr_bases)

    sigma_bases, sigma_windows = _collect_columns_and_periods(
        selected_feature_names,
        _SIGMA_PATTERN,
        "window",
        allowed_bases=set(config.sigma_normalization_columns),
    )
    config.enable_sigma_normalization = bool(sigma_bases)
    if sigma_bases:
        config.sigma_normalization_columns = _ordered_strings(config.sigma_normalization_columns, sigma_bases)
        config.sigma_normalization_window = _resolve_single_window(
            sigma_windows,
            transform_name="sigma-normalization",
        )

    winsor_bases, winsor_windows = _collect_columns_and_periods(
        selected_feature_names,
        _WINSOR_PATTERN,
        "window",
        allowed_bases=set(config.winsorize_columns),
    )
    config.enable_winsorization = bool(winsor_bases)
    if winsor_bases:
        config.winsorize_columns = _ordered_strings(config.winsorize_columns, winsor_bases)
        config.winsorization_window = _resolve_single_window(
            winsor_windows,
            transform_name="winsorization",
        )

    config.enable_interactions = any(name.startswith("interaction_") for name in selected_feature_names)


def _prune_feature_sets(config: FeatureBuilderConfig, selected_feature_names: Sequence[str]) -> None:
    selected_feature_names = tuple(dict.fromkeys(str(name) for name in selected_feature_names if str(name)))
    requires_interaction_context = any(name.startswith("interaction_") for name in selected_feature_names)
    requires_ict_interactions = any(
        name in _ICT_INTERACTION_FEATURE_NAMES
        for name in selected_feature_names
    )
    requires_frvp_context = any(
        name.startswith("frvp_") or "_frvp_" in name
        for name in selected_feature_names
    )
    requires_ict_context = any(
        name.startswith("ict_")
        or "_ict_" in name
        or name in ICT_CONTEXT_NONPREFIX_FEATURE_NAMES
        for name in selected_feature_names
    ) or requires_interaction_context

    pruned_feature_sets: list[str] = []
    for feature_set_name in config.feature_sets:
        if feature_set_name == "frvp_context" and not requires_frvp_context:
            continue
        if feature_set_name == "ict_context" and not requires_ict_context:
            continue
        pruned_feature_sets.append(feature_set_name)
    if requires_ict_interactions and "ict_interactions" not in pruned_feature_sets:
        pruned_feature_sets.append("ict_interactions")
    config.feature_sets = pruned_feature_sets


def _materialize_live_default_features(
    built_frame: pd.DataFrame,
    *,
    requested_feature_names: Sequence[str],
) -> pd.DataFrame:
    """Materialize causal defaults for offline helper columns absent from raw bars.

    HTF confluence helpers are sparse event-time flags in the training datasets:
    every non-event row is zero. Live IBKR bars do not carry the offline label
    helper columns, so zero is the only causal value until a live event-specific
    confluence producer is available.
    """

    missing_default_names = [
        feature_name
        for feature_name in requested_feature_names
        if feature_name not in built_frame.columns
        and feature_name.startswith(_LIVE_ZERO_DEFAULT_FEATURE_PREFIXES)
    ]
    if not missing_default_names:
        return built_frame

    materialized = built_frame.copy()
    for feature_name in missing_default_names:
        materialized[feature_name] = np.zeros(len(materialized), dtype=np.int8)
    return materialized


def infer_required_history_bars(selected_feature_names: Sequence[str]) -> int:
    selected_feature_names = tuple(dict.fromkeys(selected_feature_names))
    required_bars = 0

    if any(feature_name in _FRVP_PRIOR_RTH_HISTORY_FEATURE_NAMES for feature_name in selected_feature_names):
        required_bars = max(required_bars, DEFAULT_FRVP_PRIOR_RTH_HISTORY_BARS)
    if any("rolling_weekly" in feature_name for feature_name in selected_feature_names):
        required_bars = max(required_bars, DEFAULT_WEEKLY_CONTEXT_HISTORY_BARS)
    if any("rolling_daily" in feature_name for feature_name in selected_feature_names):
        required_bars = max(required_bars, DEFAULT_DAILY_CONTEXT_HISTORY_BARS)
    if any(feature_name.startswith("htf_30m_") for feature_name in selected_feature_names):
        required_bars = max(required_bars, DEFAULT_HTF_30M_CONTEXT_HISTORY_BARS)
    if any(
        feature_name == "htf_alignment_score" or feature_name.startswith("htf_1h_")
        for feature_name in selected_feature_names
    ):
        required_bars = max(required_bars, DEFAULT_HTF_1H_CONTEXT_HISTORY_BARS)

    return required_bars


def _collect_columns(
    feature_names: Sequence[str],
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


def _collect_columns_and_periods(
    feature_names: Sequence[str],
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


def _manifest_allows_nan_feature_values(manifest: LiveRuntimeManifest) -> bool:
    model_id = str(manifest.model_id).lower()
    return manifest.backend == "xgboost" and model_id.startswith(("frvp_", "ict_"))


def _manifests_preserve_numeric_nans(manifests: Sequence[LiveRuntimeManifest]) -> bool:
    return bool(manifests) and all(_manifest_allows_nan_feature_values(manifest) for manifest in manifests)


def _ordered_strings(existing_values: Sequence[str], selected_values: set[str]) -> list[str]:
    ordered = [value for value in existing_values if value in selected_values]
    extras = sorted(selected_values - set(ordered))
    return ordered + extras


def _ordered_ints(existing_values: Sequence[int], selected_values: set[int]) -> list[int]:
    ordered = [int(value) for value in existing_values if int(value) in selected_values]
    extras = sorted(selected_values - set(ordered))
    return ordered + extras


def _resolve_single_window(windows: set[int], *, transform_name: str) -> int:
    if not windows:
        raise ValueError(f"{transform_name} pruning requires at least one window.")
    if len(windows) > 1:
        ordered = ", ".join(str(value) for value in sorted(windows))
        raise ValueError(
            f"Selected live features reference multiple {transform_name} windows ({ordered}). "
            "The current runtime pruning path supports one window per transform family."
        )
    return next(iter(windows))


def _to_python_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_numeric_scalar(value) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)
