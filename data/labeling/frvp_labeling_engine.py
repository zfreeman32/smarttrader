from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from features.config import FeatureBuilderConfig

from frvp.continuity.continuous_contract import BackAdjustedPathBars, RawProfileBars
from frvp.continuity.types import RollBoundaryError
from frvp.feature_sets.frvp_context import _prepare_frvp_context, build_frvp_context_features
from frvp.target_lanes import (
    FRVP_POOLED_TARGET_FAMILIES,
    FRVP_SETUP_TARGET_FAMILIES,
    FRVP_SETUP_TYPES,
    pooled_target_family,
    setup_target_family,
)

try:
    from .reversal_labeling_engine import (
        REFERENCE_BAR_MINUTES,
        ReversalParams,
        SwingPoint,
        _prep,
        annotate_swing_context,
        compute_atr,
        compute_htf_swing_match,
        compute_sample_weights,
        detect_swings_zigzag,
        format_bar_timeframe,
        infer_bar_minutes,
        map_htf_atr_to_ltf,
        retune_bar_count_params,
        trend_scan_swings,
    )
except ImportError:
    from reversal_labeling_engine import (  # type: ignore
        REFERENCE_BAR_MINUTES,
        ReversalParams,
        SwingPoint,
        _prep,
        annotate_swing_context,
        compute_atr,
        compute_htf_swing_match,
        compute_sample_weights,
        detect_swings_zigzag,
        format_bar_timeframe,
        infer_bar_minutes,
        map_htf_atr_to_ltf,
        retune_bar_count_params,
        trend_scan_swings,
    )


FRVP_REVERSAL_FAMILY = pooled_target_family(1)
FRVP_CONTINUATION_FAMILY = pooled_target_family(2)
FRVP_QUALITY_TARGET = 0.65
MEAN_REVERSION_SETUPS = frozenset({1, 6})
CONTINUATION_SETUPS = frozenset({2, 3, 5})
FAILED_AUCTION_SETUPS = frozenset({4})
SETUP1_TYPE = 1
SETUP6_TYPE = 6
PAST_HTF_CONFLUENCE_SETUPS = frozenset({SETUP1_TYPE, SETUP6_TYPE})
SUPPORTED_FRVP_INSTRUMENTS = frozenset({"es", "6e"})
MACRO_FLAG_COLUMNS = frozenset(
    {
        "macro_event_flag",
        "major_macro_event_flag",
        "fomc_flag",
        "cpi_flag",
        "nfp_flag",
        "fed_statement_flag",
        "statement_flag",
        "fed_presser_flag",
        "presser_flag",
    }
)
HALF_DAY_FLAG_COLUMNS = frozenset(
    {
        "half_day_flag",
        "early_close_flag",
        "equity_half_day_flag",
        "equity_early_close_flag",
    }
)
QUAD_FLAG_COLUMNS = frozenset({"quad_witching_flag"})
REBALANCE_FLAG_COLUMNS = frozenset({"index_rebalance_flag", "rebalance_flag"})


@dataclass
class FRVPLabelingParams(ReversalParams):
    """Setup-driven FRVP meta-labeling parameters."""

    enabled: bool = True
    instrument: Optional[str] = None

    tb_profit_atr: float = 0.8
    tb_stop_atr: float = 1.0
    tb_max_bars: int = 60
    trend_scan_enforce: bool = False
    min_label_quality: float = 0.25
    event_match_bars: int = 1

    mean_reversion_profit_atr: float = 0.8
    mean_reversion_stop_atr: float = 1.0
    mean_reversion_max_bars: int = 60

    continuation_profit_atr: float = 1.2
    continuation_stop_atr: float = 1.0
    continuation_max_bars: int = 96
    continuation_setup3_long_profit_atr: float | None = None
    continuation_setup3_long_stop_atr: float | None = None
    continuation_setup3_long_max_bars: int | None = None
    continuation_setup3_short_profit_atr: float | None = None
    continuation_setup3_short_stop_atr: float | None = None
    continuation_setup3_short_max_bars: int | None = None
    continuation_setup5_long_profit_atr: float | None = None
    continuation_setup5_long_stop_atr: float | None = None
    continuation_setup5_long_max_bars: int | None = None
    continuation_setup5_short_profit_atr: float | None = None
    continuation_setup5_short_stop_atr: float | None = None
    continuation_setup5_short_max_bars: int | None = None

    enable_failed_auction_labels: bool = True
    failed_auction_profit_atr: float = 1.0
    failed_auction_stop_atr: float = 0.9
    failed_auction_max_bars: int = 48
    setup6_reversal_cooldown_bars: int = 0
    enable_reversal_past_htf_confluence_gate: bool = False
    setup1_past_htf_30m_bars: int = 48
    setup1_past_htf_1h_bars: int = 96
    setup1_short_past_htf_30m_bars: int = 36
    setup1_short_past_htf_1h_bars: int = 72
    setup6_past_htf_30m_bars: int = 48
    setup6_past_htf_1h_bars: int = 96

    first_rth_exclusion_bars: int = 5
    roll_bracket_weight: float = 0.50
    quad_rebalance_weight: float = 0.75
    thin_session_volume_frac: float = 0.20
    thin_session_lookback_sessions: int = 20
    empirical_half_day_ratio: float = 0.80

    def validate(self):
        super().validate()
        assert self.instrument is None or self.instrument.lower() in SUPPORTED_FRVP_INSTRUMENTS
        assert self.mean_reversion_profit_atr > 0
        assert self.mean_reversion_stop_atr > 0
        assert self.mean_reversion_max_bars >= 1
        assert self.continuation_profit_atr > 0
        assert self.continuation_stop_atr > 0
        assert self.continuation_max_bars >= 1
        for value in (
            self.continuation_setup3_long_profit_atr,
            self.continuation_setup3_long_stop_atr,
            self.continuation_setup3_short_profit_atr,
            self.continuation_setup3_short_stop_atr,
            self.continuation_setup5_long_profit_atr,
            self.continuation_setup5_long_stop_atr,
            self.continuation_setup5_short_profit_atr,
            self.continuation_setup5_short_stop_atr,
        ):
            assert value is None or value > 0
        for value in (
            self.continuation_setup3_long_max_bars,
            self.continuation_setup3_short_max_bars,
            self.continuation_setup5_long_max_bars,
            self.continuation_setup5_short_max_bars,
        ):
            assert value is None or value >= 1
        assert self.failed_auction_profit_atr > 0
        assert self.failed_auction_stop_atr > 0
        assert self.failed_auction_max_bars >= 1
        assert self.setup6_reversal_cooldown_bars >= 0
        assert self.setup1_past_htf_30m_bars >= 0
        assert self.setup1_past_htf_1h_bars >= 0
        assert self.setup1_short_past_htf_30m_bars >= 0
        assert self.setup1_short_past_htf_1h_bars >= 0
        assert self.setup6_past_htf_30m_bars >= 0
        assert self.setup6_past_htf_1h_bars >= 0
        assert self.first_rth_exclusion_bars >= 0
        assert 0 < self.roll_bracket_weight <= 1.0
        assert 0 < self.quad_rebalance_weight <= 1.0
        assert 0 < self.thin_session_volume_frac <= 1.0
        assert self.thin_session_lookback_sessions >= 1
        assert 0 < self.empirical_half_day_ratio <= 1.0


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def _frvp_only_config(*, instrument: Optional[str]) -> FeatureBuilderConfig:
    return FeatureBuilderConfig(
        feature_sets=["frvp_context"],
        instrument=instrument,
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


def _infer_supported_instrument(df: pd.DataFrame, params: FRVPLabelingParams) -> Optional[str]:
    if params.instrument:
        instrument = str(params.instrument).strip().lower()
        return instrument if instrument in SUPPORTED_FRVP_INSTRUMENTS else None

    for column in ("symbol", "contract_symbol", "contract_id"):
        if column not in df.columns:
            continue
        values = df[column].dropna().astype(str)
        if values.empty:
            continue
        token = values.iloc[0].upper()
        if token.startswith("ES"):
            return "es"
        if token.startswith("6E"):
            return "6e"
    return None


def _as_bool_series(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    matches = [column for column in columns if column in df.columns]
    if not matches:
        return pd.Series(False, index=df.index, dtype=bool)
    frame = pd.concat(
        [
            pd.Series(df[column], index=df.index).where(pd.Series(df[column], index=df.index).notna(), False).astype(bool).rename(column)
            for column in matches
        ],
        axis=1,
    )
    return frame.any(axis=1)


def _third_friday(date_value: pd.Timestamp) -> bool:
    ts = pd.Timestamp(date_value)
    return ts.weekday() == 4 and 15 <= ts.day <= 21 and ts.month in {3, 6, 9, 12}


def _last_business_day_of_quarter(date_value: pd.Timestamp) -> bool:
    ts = pd.Timestamp(date_value)
    if ts.month not in {3, 6, 9, 12}:
        return False
    next_business = ts + pd.offsets.BDay(1)
    return next_business.month != ts.month


def _barrier_spec(
    event: SwingPoint,
    params: FRVPLabelingParams,
) -> tuple[str, float, float, int]:
    setup_type = int(getattr(event, "setup_type", 0))
    if setup_type in MEAN_REVERSION_SETUPS:
        return (
            "mean_reversion",
            float(params.mean_reversion_profit_atr),
            float(params.mean_reversion_stop_atr),
            int(params.mean_reversion_max_bars),
        )
    if setup_type in CONTINUATION_SETUPS:
        direction = "long" if int(getattr(event, "setup_side", 0)) > 0 else "short"
        prefix = f"continuation_setup{setup_type}_{direction}"
        tp_atr = getattr(params, f"{prefix}_profit_atr", None)
        sl_atr = getattr(params, f"{prefix}_stop_atr", None)
        max_bars = getattr(params, f"{prefix}_max_bars", None)
        return (
            "continuation",
            float(tp_atr if tp_atr is not None else params.continuation_profit_atr),
            float(sl_atr if sl_atr is not None else params.continuation_stop_atr),
            int(max_bars if max_bars is not None else params.continuation_max_bars),
        )
    if setup_type in FAILED_AUCTION_SETUPS:
        return (
            "failed_auction",
            float(params.failed_auction_profit_atr),
            float(params.failed_auction_stop_atr),
            int(params.failed_auction_max_bars),
        )
    raise ValueError(f"Unsupported FRVP setup type: {setup_type}")


def _event_target_family(setup_type: int) -> str:
    return pooled_target_family(int(setup_type))


def _target_columns(
    direction: str,
    family: str,
) -> dict[str, str]:
    return {
        "label": f"label_{direction}_{family}",
        "quality": f"label_quality_{direction}_{family}",
        "sample_weight": f"sample_weight_{direction}_{family}",
        "exclude": f"exclude_{direction}_{family}",
        "neg_ok": f"neg_ok_{direction}_{family}",
        "concurrency": f"concurrency_{direction}_{family}",
        "htf_confluence": f"htf_confluence_{direction}_{family}",
    }


def _prepare_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    source = df.copy()
    if isinstance(source.index, pd.DatetimeIndex):
        if "datetime" not in source.columns:
            source["datetime"] = source.index
        if "ts_event" not in source.columns:
            source["ts_event"] = source.index
    if "timestamp" in source.columns and "datetime" in source.columns:
        source = source.drop(columns=["timestamp"])
    return source.reset_index(drop=True)


def _detect_htf_swings(
    path_5m: pd.DataFrame,
    params: FRVPLabelingParams,
) -> tuple[pd.Series, pd.Series, pd.Series, list[SwingPoint], list[SwingPoint], pd.DataFrame, pd.DataFrame]:
    path_30m = _resample_ohlcv(path_5m, "30min")
    path_1hr = _resample_ohlcv(path_5m, "1h")

    atr_5m = compute_atr(path_5m, params.atr_period, params.atr_smoothing)
    atr_30m = compute_atr(path_30m, params.atr_period, params.atr_smoothing)
    atr_1hr = compute_atr(path_1hr, params.atr_period, params.atr_smoothing)
    if params.structural_atr_tf == "1hr":
        structural_atr_5m = map_htf_atr_to_ltf(atr_1hr, path_5m.index)
        structural_atr_30m = map_htf_atr_to_ltf(atr_1hr, path_30m.index)
    else:
        structural_atr_5m = map_htf_atr_to_ltf(atr_30m, path_5m.index)
        structural_atr_30m = atr_30m

    htf_warmup_bars = max(int(params.warmup_bars), int(params.atr_period) * 2)
    sw30 = detect_swings_zigzag(
        path_30m,
        structural_atr_30m,
        params.htf_confirm_atr_mult,
        params.htf_min_swing_atr,
        params.min_swing_distance_atr * 0.8,
        params.htf_min_bars_between,
        htf_warmup_bars,
        params.confirm_use_close,
        "30m",
    )
    sw1h = detect_swings_zigzag(
        path_1hr,
        atr_1hr,
        params.htf_confirm_atr_mult,
        params.htf_min_swing_atr * 0.7,
        params.min_swing_distance_atr * 0.7,
        max(2, params.htf_min_bars_between // 2),
        htf_warmup_bars,
        params.confirm_use_close,
        "1hr",
    )
    return atr_5m, atr_30m, structural_atr_5m, sw30, sw1h, path_30m, path_1hr


def _build_event_candidates(
    market: pd.DataFrame,
    features: pd.DataFrame,
    structural_atr: pd.Series,
) -> list[SwingPoint]:
    timestamps = pd.DatetimeIndex(market["timestamp"])
    closes = pd.to_numeric(market["close"], errors="coerce")
    opens = pd.to_numeric(market["open"], errors="coerce")
    highs = pd.to_numeric(market["high"], errors="coerce")
    lows = pd.to_numeric(market["low"], errors="coerce")
    setup_type = pd.to_numeric(features["frvp_setup_type"], errors="coerce").fillna(0).astype(int)
    setup_side = pd.to_numeric(features["frvp_setup_side"], errors="coerce").fillna(0).astype(int)
    confidence = pd.to_numeric(features["frvp_setup_confidence_rule"], errors="coerce").fillna(0.0).astype(float)

    events: list[SwingPoint] = []
    for idx in range(len(market)):
        event_setup = int(setup_type.iloc[idx])
        side = int(setup_side.iloc[idx])
        if event_setup <= 0 or side == 0:
            continue

        atr_at = float(structural_atr.iloc[idx]) if idx < len(structural_atr) else np.nan
        if not np.isfinite(atr_at) or atr_at <= 0:
            atr_at = float("nan")

        event = SwingPoint(
            swing_type="low" if side > 0 else "high",
            swing_time=pd.Timestamp(timestamps[idx]),
            swing_index=idx,
            swing_price=float(closes.iloc[idx]),
            confirm_time=pd.Timestamp(timestamps[idx]),
            confirm_index=idx,
            confirm_lag=1,
            atr_at_swing=atr_at,
            swing_size_atr=float((highs.iloc[idx] - lows.iloc[idx]) / atr_at) if np.isfinite(atr_at) and atr_at > 0 else 0.0,
            source_tf="frvp",
        )
        event.setup_type = event_setup
        event.setup_side = side
        event.setup_confidence = float(confidence.iloc[idx])
        event.target_family = _event_target_family(event_setup)
        event.event_direction = "long" if side > 0 else "short"
        event.fired_at_open = float(opens.iloc[idx])
        event.fired_at_close = float(closes.iloc[idx])
        event.fired_at_high = float(highs.iloc[idx])
        event.fired_at_low = float(lows.iloc[idx])
        event.excluded = False
        event.exclude_reasons = []
        event.frvp_concurrency = 0
        events.append(event)
    return events


def _evaluate_event_barriers(
    raw_df: pd.DataFrame,
    structural_atr: pd.Series,
    events: list[SwingPoint],
    params: FRVPLabelingParams,
) -> list[SwingPoint]:
    if not events:
        return events

    n = len(raw_df)
    opens = pd.to_numeric(raw_df["open"], errors="coerce").to_numpy(dtype=float, copy=False)
    highs = pd.to_numeric(raw_df["high"], errors="coerce").to_numpy(dtype=float, copy=False)
    lows = pd.to_numeric(raw_df["low"], errors="coerce").to_numpy(dtype=float, copy=False)
    closes = pd.to_numeric(raw_df["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    timestamps = pd.DatetimeIndex(raw_df.index)
    atr_values = structural_atr.to_numpy(dtype=float, copy=False)

    for event in events:
        entry_idx = int(event.swing_index) + 1
        if entry_idx >= n:
            event.tb_outcome = "skip"
            event.tb_return = 0.0
            event.tb_signed_move_atr = 0.0
            event.tb_bars_held = 0
            event.entry_index = None
            event.entry_time = None
            event.entry_price = None
            event.excluded = True
            event.exclude_reasons.append("no_next_open")
            continue

        barrier_family, tp_atr, sl_atr, max_bars = _barrier_spec(event, params)
        atr_at = atr_values[event.swing_index] if event.swing_index < len(atr_values) else np.nan
        if not np.isfinite(atr_at) or atr_at <= 0:
            atr_at = atr_values[entry_idx] if entry_idx < len(atr_values) else np.nan
        if not np.isfinite(atr_at) or atr_at <= 0:
            event.tb_outcome = "skip"
            event.tb_return = 0.0
            event.tb_signed_move_atr = 0.0
            event.tb_bars_held = 0
            event.entry_index = entry_idx
            event.entry_time = pd.Timestamp(timestamps[entry_idx])
            event.entry_price = float(opens[entry_idx])
            event.excluded = True
            event.exclude_reasons.append("invalid_structural_atr")
            continue

        direction_sign = 1.0 if int(getattr(event, "setup_side", 0)) > 0 else -1.0
        entry = float(opens[entry_idx])
        if direction_sign > 0:
            tp_price = entry + tp_atr * atr_at
            sl_price = entry - sl_atr * atr_at
        else:
            tp_price = entry - tp_atr * atr_at
            sl_price = entry + sl_atr * atr_at

        end_exclusive = min(entry_idx + max_bars, n)
        end_index = max(entry_idx, end_exclusive - 1)
        outcome = "timeout"
        exit_price = float(closes[end_index])
        held_bars = end_exclusive - entry_idx

        for bar_idx in range(entry_idx, end_exclusive):
            if direction_sign > 0:
                if highs[bar_idx] >= tp_price:
                    outcome = "tp"
                    exit_price = float(tp_price)
                    held_bars = (bar_idx - entry_idx) + 1
                    end_index = bar_idx
                    break
                if lows[bar_idx] <= sl_price:
                    outcome = "sl"
                    exit_price = float(sl_price)
                    held_bars = (bar_idx - entry_idx) + 1
                    end_index = bar_idx
                    break
            else:
                if lows[bar_idx] <= tp_price:
                    outcome = "tp"
                    exit_price = float(tp_price)
                    held_bars = (bar_idx - entry_idx) + 1
                    end_index = bar_idx
                    break
                if highs[bar_idx] >= sl_price:
                    outcome = "sl"
                    exit_price = float(sl_price)
                    held_bars = (bar_idx - entry_idx) + 1
                    end_index = bar_idx
                    break

        event.entry_index = entry_idx
        event.entry_time = pd.Timestamp(timestamps[entry_idx])
        event.entry_price = entry
        event.barrier_family = barrier_family
        event.tp_atr = float(tp_atr)
        event.sl_atr = float(sl_atr)
        event.max_holding_bars = int(max_bars)
        event.tb_outcome = outcome
        event.tb_bars_held = int(max(held_bars, 0))
        event.barrier_end_index = int(end_index)
        event.barrier_end_time = pd.Timestamp(timestamps[end_index])
        event.tb_signed_move_atr = float(direction_sign * (exit_price - entry) / atr_at)
        event.tb_return = float(direction_sign * (exit_price - entry) / max(abs(entry), 1e-8))
    return events


def _score_event_entries(
    raw_df: pd.DataFrame,
    structural_atr: pd.Series,
    events: list[SwingPoint],
) -> list[SwingPoint]:
    if not events:
        return events

    opens = pd.to_numeric(raw_df["open"], errors="coerce").to_numpy(dtype=float, copy=False)
    highs = pd.to_numeric(raw_df["high"], errors="coerce").to_numpy(dtype=float, copy=False)
    lows = pd.to_numeric(raw_df["low"], errors="coerce").to_numpy(dtype=float, copy=False)
    closes = pd.to_numeric(raw_df["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    atr_values = structural_atr.to_numpy(dtype=float, copy=False)
    n = len(raw_df)

    for event in events:
        entry_idx = getattr(event, "entry_index", None)
        if entry_idx is None or entry_idx >= n:
            event.entry_rr = 0.0
            event.entry_followthrough_atr = 0.0
            event.entry_score = 0.0
            continue

        atr_at = atr_values[event.swing_index] if event.swing_index < len(atr_values) else np.nan
        if not np.isfinite(atr_at) or atr_at <= 0:
            atr_at = atr_values[entry_idx] if entry_idx < len(atr_values) else np.nan
        if not np.isfinite(atr_at) or atr_at <= 0:
            event.entry_rr = 0.0
            event.entry_followthrough_atr = 0.0
            event.entry_score = 0.0
            continue

        eval_end = min(n, entry_idx + max(int(getattr(event, "max_holding_bars", 1)), 12))
        future_high = float(np.max(highs[entry_idx:eval_end]))
        future_low = float(np.min(lows[entry_idx:eval_end]))
        direction_sign = 1.0 if int(getattr(event, "setup_side", 0)) > 0 else -1.0
        entry = float(getattr(event, "entry_price", opens[entry_idx]))

        if direction_sign > 0:
            favorable = max(0.0, future_high - entry)
            adverse = max(0.0, entry - future_low)
            adverse_gap = max(0.0, opens[entry_idx] - closes[event.swing_index])
        else:
            favorable = max(0.0, entry - future_low)
            adverse = max(0.0, future_high - entry)
            adverse_gap = max(0.0, closes[event.swing_index] - opens[entry_idx])

        rr = favorable / max(adverse, 0.10 * atr_at)
        followthrough_atr = favorable / atr_at
        rr_target = max(float(getattr(event, "tp_atr", 1.0)) / max(float(getattr(event, "sl_atr", 1.0)), 1e-8), 1.0)
        rr_score = float(np.clip(rr / rr_target, 0.0, 1.0))
        follow_score = float(np.clip(followthrough_atr / max(float(getattr(event, "tp_atr", 1.0)), 0.25), 0.0, 1.0))
        gap_score = 1.0 - float(np.clip(adverse_gap / max(atr_at * float(getattr(event, "sl_atr", 1.0)), 1e-8), 0.0, 1.0))

        event.entry_rr = float(rr)
        event.entry_followthrough_atr = float(followthrough_atr)
        event.entry_score = float(np.clip(0.45 * rr_score + 0.35 * follow_score + 0.20 * gap_score, 0.0, 1.0))
    return events


def _compute_event_quality(
    events: list[SwingPoint],
    params: FRVPLabelingParams,
) -> list[SwingPoint]:
    for event in events:
        tb_outcome = getattr(event, "tb_outcome", None)
        if tb_outcome == "tp":
            tb_score = 1.0
        elif tb_outcome == "sl":
            tb_score = 0.0
        elif tb_outcome == "timeout":
            signed_move_atr = float(getattr(event, "tb_signed_move_atr", 0.0))
            tp_atr = max(float(getattr(event, "tp_atr", 1.0)), 1e-8)
            tb_score = float(np.clip(0.5 + 0.5 * (signed_move_atr / tp_atr), 0.0, 1.0))
        else:
            tb_score = 0.35

        trend_t = float(getattr(event, "trend_scan_t", 0.0) or 0.0)
        trend_score = float(np.clip(trend_t / 4.0, 0.0, 1.0))
        htf_score = 0.5 * float(bool(getattr(event, "htf_match_30m", False))) + 0.5 * float(
            bool(getattr(event, "htf_match_1h", False))
        )
        entry_score = float(np.clip(float(getattr(event, "entry_score", 0.0) or 0.0), 0.0, 1.0))
        setup_score = float(np.clip(float(getattr(event, "setup_confidence", 0.0) or 0.0), 0.0, 1.0))

        quality = 0.40 * tb_score + 0.25 * trend_score + 0.20 * htf_score + 0.10 * entry_score + 0.05 * setup_score
        event.label_quality = float(np.clip(quality, 0.0, 1.0))
        if event.label_quality >= 0.80:
            event.label_tier = "A"
        elif event.label_quality >= 0.65:
            event.label_tier = "B"
        elif event.label_quality >= 0.50:
            event.label_tier = "C"
        else:
            event.label_tier = "D"
    return events


def _session_context(
    market: pd.DataFrame,
    base_bar_minutes: float,
    params: FRVPLabelingParams,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = market.copy().reset_index(drop=True)
    working["session_date"] = pd.to_datetime(working["session_date"], errors="coerce").dt.normalize()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=True)
    if "is_rth" in working.columns:
        working["is_rth"] = working["is_rth"].fillna(False).astype(bool)
    else:
        working["is_rth"] = False
    working["rth_volume"] = np.where(working["is_rth"], pd.to_numeric(working["volume"], errors="coerce").fillna(0.0), 0.0)
    has_explicit_half_day_columns = any(column in working.columns for column in HALF_DAY_FLAG_COLUMNS)
    working["explicit_half_day"] = _as_bool_series(working, HALF_DAY_FLAG_COLUMNS)
    working["explicit_quad"] = _as_bool_series(working, QUAD_FLAG_COLUMNS)
    working["explicit_rebalance"] = _as_bool_series(working, REBALANCE_FLAG_COLUMNS)
    working["macro_event"] = _as_bool_series(working, MACRO_FLAG_COLUMNS)
    rth_bar_number = (
        working.loc[working["is_rth"]]
        .groupby("session_date", sort=False)
        .cumcount()
        .add(1)
        .reindex(working.index)
    )
    working["rth_bar_number"] = rth_bar_number.fillna(0).astype(int)

    session_summary = (
        working.groupby("session_date", sort=True)
        .agg(
            rth_bars=("is_rth", "sum"),
            rth_volume=("rth_volume", "sum"),
            explicit_half_day=("explicit_half_day", "max"),
            explicit_quad=("explicit_quad", "max"),
            explicit_rebalance=("explicit_rebalance", "max"),
            macro_event=("macro_event", "max"),
        )
        .reset_index()
    )
    expected_rth_bars = max(int(round((6.5 * 60.0) / max(base_bar_minutes, 1e-6))), 1)
    empirical_half_day = (
        session_summary["rth_bars"].gt(0)
        & session_summary["rth_bars"].lt(int(round(expected_rth_bars * params.empirical_half_day_ratio)))
    )
    session_summary["empirical_half_day"] = empirical_half_day & (not has_explicit_half_day_columns)
    session_summary["half_day_or_early_close"] = (
        session_summary["explicit_half_day"].fillna(False).astype(bool)
        | session_summary["empirical_half_day"].fillna(False).astype(bool)
    )

    trailing_median = (
        session_summary["rth_volume"]
        .shift(1)
        .rolling(params.thin_session_lookback_sessions, min_periods=params.thin_session_lookback_sessions)
        .median()
    )
    session_summary["thin_session"] = (
        trailing_median.gt(0)
        & session_summary["rth_volume"].lt(trailing_median * float(params.thin_session_volume_frac))
    )
    session_summary["session_date"] = pd.to_datetime(session_summary["session_date"], errors="coerce").dt.normalize()
    session_summary["quad_witching"] = session_summary["session_date"].map(_third_friday).fillna(False).astype(bool)
    session_summary["index_rebalance"] = session_summary["session_date"].map(_last_business_day_of_quarter).fillna(False).astype(bool)
    session_summary["quad_or_rebalance"] = (
        session_summary["quad_witching"]
        | session_summary["index_rebalance"]
        | session_summary["explicit_quad"].fillna(False).astype(bool)
        | session_summary["explicit_rebalance"].fillna(False).astype(bool)
    )
    session_summary["macro_event"] = session_summary["macro_event"].fillna(False).astype(bool)

    diagnostics = {
        "macro_flag_columns_used": [column for column in MACRO_FLAG_COLUMNS if column in working.columns],
        "half_day_flag_columns_used": [column for column in HALF_DAY_FLAG_COLUMNS if column in working.columns],
        "quad_flag_columns_used": [column for column in QUAD_FLAG_COLUMNS if column in working.columns],
        "rebalance_flag_columns_used": [column for column in REBALANCE_FLAG_COLUMNS if column in working.columns],
        "todo_macro_flags_unavailable": not any(column in working.columns for column in MACRO_FLAG_COLUMNS),
        "todo_halfday_calendar_overrides_pending": not has_explicit_half_day_columns,
        "halfday_detection_uses_empirical_fallback": bool(session_summary["empirical_half_day"].any()),
        "expected_rth_bars": int(expected_rth_bars),
    }
    return session_summary, diagnostics


def _event_within_past_htf_window(event: SwingPoint, *, bars_30m: int, bars_1h: int) -> bool:
    prev_30m = pd.to_numeric(pd.Series([getattr(event, "bars_since_prev_30m_same", np.nan)]), errors="coerce").iloc[0]
    prev_1h = pd.to_numeric(pd.Series([getattr(event, "bars_since_prev_1h_same", np.nan)]), errors="coerce").iloc[0]
    has_30m = bool(np.isfinite(prev_30m) and float(prev_30m) <= float(max(bars_30m, 0)))
    has_1h = bool(np.isfinite(prev_1h) and float(prev_1h) <= float(max(bars_1h, 0)))
    return has_30m or has_1h


def _passes_reversal_past_htf_confluence_gate(
    event: SwingPoint,
    params: FRVPLabelingParams,
) -> bool:
    if not bool(params.enable_reversal_past_htf_confluence_gate):
        return True
    if str(getattr(event, "target_family", "")) != FRVP_REVERSAL_FAMILY:
        return True

    setup_type = int(getattr(event, "setup_type", 0))
    setup_side = int(getattr(event, "setup_side", 0))
    if setup_type not in PAST_HTF_CONFLUENCE_SETUPS:
        return True

    if setup_type == SETUP1_TYPE and setup_side < 0:
        return _event_within_past_htf_window(
            event,
            bars_30m=int(params.setup1_short_past_htf_30m_bars),
            bars_1h=int(params.setup1_short_past_htf_1h_bars),
        )
    if setup_type == SETUP1_TYPE:
        return _event_within_past_htf_window(
            event,
            bars_30m=int(params.setup1_past_htf_30m_bars),
            bars_1h=int(params.setup1_past_htf_1h_bars),
        )
    return _event_within_past_htf_window(
        event,
        bars_30m=int(params.setup6_past_htf_30m_bars),
        bars_1h=int(params.setup6_past_htf_1h_bars),
    )


def _apply_exclusion_masks(
    market: pd.DataFrame,
    events: list[SwingPoint],
    continuous: Any,
    base_bar_minutes: float,
    params: FRVPLabelingParams,
) -> tuple[list[SwingPoint], dict[str, Any]]:
    if not events:
        diagnostics = {
            "events_excluded_total": 0,
            "events_excluded_roll_span": 0,
            "events_excluded_first_rth": 0,
            "events_excluded_macro": 0,
            "events_excluded_half_day": 0,
            "events_excluded_thin_session": 0,
            "events_excluded_disabled_setup": 0,
            "events_excluded_past_htf_confluence": 0,
            "events_excluded_reversal_cooldown": 0,
            "events_roll_bracket": 0,
            "events_quad_or_rebalance": 0,
        }
        return events, diagnostics

    session_summary, session_diag = _session_context(market, base_bar_minutes, params)
    session_lookup = session_summary.set_index("session_date")
    market_working = market.copy().reset_index(drop=True)
    market_working["session_date"] = pd.to_datetime(market_working["session_date"], errors="coerce").dt.normalize()
    if "is_rth" in market_working.columns:
        market_working["is_rth"] = market_working["is_rth"].fillna(False).astype(bool)
    else:
        market_working["is_rth"] = False
    market_working["rth_bar_number"] = (
        market_working.loc[market_working["is_rth"]]
        .groupby("session_date", sort=False)
        .cumcount()
        .add(1)
        .reindex(market_working.index)
        .fillna(0)
        .astype(int)
    )
    in_roll_bracket_series = (
        market_working["in_roll_bracket"].fillna(False).astype(bool)
        if "in_roll_bracket" in market_working.columns
        else pd.Series(False, index=market_working.index, dtype=bool)
    )
    is_roll_bracket_series = (
        market_working["is_roll_bracket"].fillna(False).astype(bool)
        if "is_roll_bracket" in market_working.columns
        else pd.Series(False, index=market_working.index, dtype=bool)
    )
    in_roll_bracket = (
        in_roll_bracket_series
        | is_roll_bracket_series
    )

    counts = {
        "events_excluded_total": 0,
        "events_excluded_roll_span": 0,
        "events_excluded_first_rth": 0,
        "events_excluded_macro": 0,
        "events_excluded_half_day": 0,
        "events_excluded_thin_session": 0,
        "events_excluded_disabled_setup": 0,
        "events_excluded_past_htf_confluence": 0,
        "events_excluded_reversal_cooldown": 0,
        "events_roll_bracket": 0,
        "events_quad_or_rebalance": 0,
    }
    for event in events:
        event_idx = int(event.swing_index)
        event_session = pd.Timestamp(market_working.at[event_idx, "session_date"]).normalize()
        session_row = session_lookup.loc[event_session] if event_session in session_lookup.index else None

        flag_first_rth = bool(
            market_working.at[event_idx, "is_rth"]
            and int(market_working.at[event_idx, "rth_bar_number"]) <= int(params.first_rth_exclusion_bars)
        )
        flag_roll_bracket = bool(in_roll_bracket.iloc[event_idx])
        flag_macro = bool(session_row["macro_event"]) if session_row is not None else False
        flag_half_day = bool(session_row["half_day_or_early_close"]) if session_row is not None else False
        flag_thin_session = bool(session_row["thin_session"]) if session_row is not None else False
        flag_quad_or_rebalance = bool(session_row["quad_or_rebalance"]) if session_row is not None else False
        flag_failed_auction_disabled = (
            not bool(params.enable_failed_auction_labels)
            and int(getattr(event, "setup_type", 0)) in FAILED_AUCTION_SETUPS
        )
        flag_past_htf_confluence = not _passes_reversal_past_htf_confluence_gate(event, params)

        flag_roll_span = False
        entry_time = getattr(event, "entry_time", None)
        barrier_end_time = getattr(event, "barrier_end_time", None)
        if entry_time is not None and barrier_end_time is not None:
            try:
                continuous.raw_profile_bars.profile_slice(entry_time, barrier_end_time)
            except RollBoundaryError:
                flag_roll_span = True
            except KeyError:
                flag_roll_span = True

        event.flag_first_rth = flag_first_rth
        event.flag_roll_bracket = flag_roll_bracket
        event.flag_roll_span = flag_roll_span
        event.flag_macro = flag_macro
        event.flag_half_day = flag_half_day
        event.flag_thin_session = flag_thin_session
        event.flag_quad_or_rebalance = flag_quad_or_rebalance
        event.flag_failed_auction_disabled = flag_failed_auction_disabled
        event.flag_past_htf_confluence = flag_past_htf_confluence
        event.flag_reversal_cooldown = False

        if flag_roll_bracket:
            counts["events_roll_bracket"] += 1
        if flag_quad_or_rebalance:
            counts["events_quad_or_rebalance"] += 1
        if flag_first_rth:
            counts["events_excluded_first_rth"] += 1
            event.exclude_reasons.append("first_rth_bars")
        if flag_roll_span:
            counts["events_excluded_roll_span"] += 1
            event.exclude_reasons.append("roll_spanning_window")
        if flag_macro:
            counts["events_excluded_macro"] += 1
            event.exclude_reasons.append("macro_event")
        if flag_half_day:
            counts["events_excluded_half_day"] += 1
            event.exclude_reasons.append("half_day_or_early_close")
        if flag_thin_session:
            counts["events_excluded_thin_session"] += 1
            event.exclude_reasons.append("thin_session")
        if flag_failed_auction_disabled:
            counts["events_excluded_disabled_setup"] += 1
            event.exclude_reasons.append("failed_auction_setup_disabled")
        if flag_past_htf_confluence:
            counts["events_excluded_past_htf_confluence"] += 1
            event.exclude_reasons.append("past_htf_confluence_gate")

        event.excluded = event.excluded or any(
            [
                flag_first_rth,
                flag_roll_span,
                flag_macro,
                flag_half_day,
                flag_thin_session,
                flag_failed_auction_disabled,
                flag_past_htf_confluence,
            ]
        )
        if event.excluded:
            counts["events_excluded_total"] += 1

    cooldown_bars = max(int(params.setup6_reversal_cooldown_bars), 0)
    if cooldown_bars > 0:
        last_kept_reversal: dict[str, int] = {"long": -(cooldown_bars + 1), "short": -(cooldown_bars + 1)}
        for event in sorted(events, key=lambda current: int(current.swing_index)):
            if bool(getattr(event, "excluded", False)):
                continue
            if str(getattr(event, "target_family", "")) != FRVP_REVERSAL_FAMILY:
                continue

            direction = str(getattr(event, "event_direction", ""))
            if direction not in last_kept_reversal:
                continue

            event_idx = int(event.swing_index)
            if (
                int(getattr(event, "setup_type", 0)) == SETUP6_TYPE
                and (event_idx - last_kept_reversal[direction]) < cooldown_bars
            ):
                event.flag_reversal_cooldown = True
                event.exclude_reasons.append("setup6_reversal_cooldown")
                event.excluded = True
                counts["events_excluded_reversal_cooldown"] += 1
                counts["events_excluded_total"] += 1
                continue

            last_kept_reversal[direction] = event_idx

    counts.update(session_diag)
    return events, counts


def _apply_event_concurrency_and_weights(
    index: pd.DatetimeIndex,
    events: list[SwingPoint],
    params: FRVPLabelingParams,
) -> dict[tuple[str, str], np.ndarray]:
    groups: dict[tuple[str, str], list[SwingPoint]] = {}
    for event in events:
        groups.setdefault((str(getattr(event, "event_direction")), str(getattr(event, "target_family"))), []).append(event)

    concurrency_arrays: dict[tuple[str, str], np.ndarray] = {}
    for key, group in groups.items():
        conc = np.zeros(len(index), dtype=np.int32)
        event_mask = np.zeros(len(index), dtype=np.int8)
        for event in group:
            if getattr(event, "entry_index", None) is None:
                continue
            start = int(event.swing_index)
            end = int(getattr(event, "barrier_end_index", start))
            conc[start : end + 1] += 1
            event_mask[int(event.swing_index)] = 1

        concurrency_arrays[key] = conc
        event_weights = compute_sample_weights(conc, event_mask)
        for event in group:
            event.frvp_concurrency = int(conc[int(event.swing_index)]) if int(event.swing_index) < len(conc) else 0
            base_weight = float(event_weights[int(event.swing_index)]) if int(event.swing_index) < len(event_weights) else 1.0
            quality_weight = max(float(getattr(event, "label_quality", 0.0)), float(params.min_label_quality))
            penalty = 1.0
            if bool(getattr(event, "flag_roll_bracket", False)):
                penalty *= float(params.roll_bracket_weight)
            if bool(getattr(event, "flag_quad_or_rebalance", False)):
                penalty *= float(params.quad_rebalance_weight)
            event.sample_weight = float(base_weight * quality_weight * penalty)
    return concurrency_arrays


def _materialize_frvp_targets(
    market: pd.DataFrame,
    events: list[SwingPoint],
    concurrency_arrays: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    output_index = pd.DatetimeIndex(market["timestamp"])
    directions = ("long", "short")
    column_data: dict[str, np.ndarray] = {}

    for direction in directions:
        for target_family in (*FRVP_POOLED_TARGET_FAMILIES, *FRVP_SETUP_TARGET_FAMILIES):
            columns = _target_columns(direction, target_family)
            if target_family in FRVP_POOLED_TARGET_FAMILIES:
                concurrency_family = target_family
            else:
                setup_type = FRVP_SETUP_TYPES[FRVP_SETUP_TARGET_FAMILIES.index(target_family)]
                concurrency_family = pooled_target_family(setup_type)
            column_data[columns["label"]] = np.zeros(len(output_index), dtype=np.int8)
            column_data[columns["quality"]] = np.zeros(len(output_index), dtype=np.float32)
            column_data[columns["sample_weight"]] = np.ones(len(output_index), dtype=np.float32)
            column_data[columns["exclude"]] = np.ones(len(output_index), dtype=bool)
            column_data[columns["neg_ok"]] = np.zeros(len(output_index), dtype=bool)
            # Setup targets intentionally inherit the pooled-family uniqueness
            # weights/concurrency. This keeps the first split-first comparison
            # limited to target routing instead of silently adding a weighting
            # policy change.
            column_data[columns["concurrency"]] = concurrency_arrays.get(
                (direction, concurrency_family),
                np.zeros(len(output_index), dtype=np.int32),
            )
            column_data[columns["htf_confluence"]] = np.zeros(len(output_index), dtype=np.int8)

    out = pd.DataFrame(column_data, index=output_index)

    for event in events:
        direction = str(getattr(event, "event_direction"))
        event_idx = int(event.swing_index)
        if event_idx >= len(out):
            continue

        quality = float(getattr(event, "label_quality", 0.0))
        sample_weight = float(getattr(event, "sample_weight", 1.0))
        usable = not bool(getattr(event, "excluded", False))
        target_families = (
            str(getattr(event, "target_family")),
            setup_target_family(int(getattr(event, "setup_type"))),
        )
        for target_family in target_families:
            columns = _target_columns(direction, target_family)
            out.iat[event_idx, out.columns.get_loc(columns["quality"])] = quality
            out.iat[event_idx, out.columns.get_loc(columns["sample_weight"])] = sample_weight
            out.iat[event_idx, out.columns.get_loc(columns["htf_confluence"])] = int(
                bool(getattr(event, "htf_match_30m", False)) or bool(getattr(event, "htf_match_1h", False))
            )

            if usable:
                out.iat[event_idx, out.columns.get_loc(columns["exclude"])] = False
                out.iat[event_idx, out.columns.get_loc(columns["neg_ok"])] = True

            if usable and str(getattr(event, "tb_outcome", "")) == "tp":
                out.iat[event_idx, out.columns.get_loc(columns["label"])] = 1

    return out


def frvp_events_to_frame(events: Sequence[SwingPoint]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        rows.append(
            {
                "label_family": str(getattr(event, "target_family", "frvp")),
                "setup_target_family": setup_target_family(int(getattr(event, "setup_type", 0))),
                "setup_type": int(getattr(event, "setup_type", 0)),
                "setup_side": int(getattr(event, "setup_side", 0)),
                "barrier_family": getattr(event, "barrier_family", None),
                "swing_type": event.swing_type,
                "source_tf": event.source_tf,
                "swing_time": event.swing_time,
                "swing_index": event.swing_index,
                "swing_price": event.swing_price,
                "confirm_time": event.confirm_time,
                "confirm_index": event.confirm_index,
                "confirm_lag_bars": event.confirm_lag,
                "atr_at_swing": event.atr_at_swing,
                "swing_size_atr": event.swing_size_atr,
                "entry_time": getattr(event, "entry_time", None),
                "entry_index": getattr(event, "entry_index", None),
                "entry_price": getattr(event, "entry_price", None),
                "tp_atr": getattr(event, "tp_atr", None),
                "sl_atr": getattr(event, "sl_atr", None),
                "max_holding_bars": getattr(event, "max_holding_bars", None),
                "tb_outcome": event.tb_outcome,
                "tb_return": event.tb_return,
                "tb_signed_move_atr": getattr(event, "tb_signed_move_atr", None),
                "tb_bars_held": event.tb_bars_held,
                "trend_scan_t": event.trend_scan_t,
                "trend_scan_window": event.trend_scan_window,
                "trend_scan_pass": event.trend_scan_pass,
                "htf_match_30m": event.htf_match_30m,
                "htf_match_1h": event.htf_match_1h,
                "bars_since_prev_30m_same": event.bars_since_prev_30m_same,
                "bars_since_prev_1h_same": event.bars_since_prev_1h_same,
                "entry_score": getattr(event, "entry_score", None),
                "entry_rr": getattr(event, "entry_rr", None),
                "entry_followthrough_atr": getattr(event, "entry_followthrough_atr", None),
                "setup_confidence": getattr(event, "setup_confidence", None),
                "label_quality": event.label_quality,
                "label_tier": event.label_tier,
                "sample_weight": getattr(event, "sample_weight", None),
                "frvp_concurrency": getattr(event, "frvp_concurrency", None),
                "excluded": bool(getattr(event, "excluded", False)),
                "exclude_reasons": "|".join(getattr(event, "exclude_reasons", [])),
                "flag_first_rth": bool(getattr(event, "flag_first_rth", False)),
                "flag_roll_bracket": bool(getattr(event, "flag_roll_bracket", False)),
                "flag_roll_span": bool(getattr(event, "flag_roll_span", False)),
                "flag_macro": bool(getattr(event, "flag_macro", False)),
                "flag_half_day": bool(getattr(event, "flag_half_day", False)),
                "flag_quad_or_rebalance": bool(getattr(event, "flag_quad_or_rebalance", False)),
                "flag_thin_session": bool(getattr(event, "flag_thin_session", False)),
                "flag_failed_auction_disabled": bool(getattr(event, "flag_failed_auction_disabled", False)),
                "flag_past_htf_confluence": bool(getattr(event, "flag_past_htf_confluence", False)),
                "flag_reversal_cooldown": bool(getattr(event, "flag_reversal_cooldown", False)),
            }
        )
    return pd.DataFrame(rows)


def build_frvp_labels(
    df_5m: pd.DataFrame,
    df_30m: pd.DataFrame | None = None,
    df_1hr: pd.DataFrame | None = None,
    params: FRVPLabelingParams | None = None,
    verbose: bool = True,
):
    raw_params = params or FRVPLabelingParams()
    prepared_5m = _prep(df_5m)
    base_bar_minutes = infer_bar_minutes(prepared_5m.index)
    base_tf = format_bar_timeframe(base_bar_minutes)
    params = retune_bar_count_params(
        raw_params,
        base_bar_minutes,
        extra_count_fields=(
            "mean_reversion_max_bars",
            "continuation_max_bars",
            "continuation_setup3_long_max_bars",
            "continuation_setup3_short_max_bars",
            "continuation_setup5_long_max_bars",
            "continuation_setup5_short_max_bars",
            "failed_auction_max_bars",
            "setup6_reversal_cooldown_bars",
            "setup1_past_htf_30m_bars",
            "setup1_past_htf_1h_bars",
            "setup1_short_past_htf_30m_bars",
            "setup1_short_past_htf_1h_bars",
            "setup6_past_htf_30m_bars",
            "setup6_past_htf_1h_bars",
            "first_rth_exclusion_bars",
        ),
    )
    params.validate()

    instrument = _infer_supported_instrument(prepared_5m, params)
    if not params.enabled or instrument is None:
        diagnostics = {
            "status": "skipped",
            "reason": "unsupported_or_disabled_instrument",
            "base_timeframe": base_tf,
        }
        return pd.DataFrame(index=prepared_5m.index), diagnostics, []

    if verbose:
        print(f"[FRVP] {len(prepared_5m):,} bars on {base_tf} ({instrument.upper()})")
        if abs(base_bar_minutes - REFERENCE_BAR_MINUTES) >= 1e-9:
            print(f"[FRVP] Runtime bar-count tuning: {format_bar_timeframe(REFERENCE_BAR_MINUTES)} -> {base_tf}")

    source = _prepare_source_frame(prepared_5m)
    config = _frvp_only_config(instrument=instrument)
    frvp_features = build_frvp_context_features(source, config, instrument=instrument).reset_index(drop=True)
    prepared_context = _prepare_frvp_context(source, config, instrument=instrument)
    market = prepared_context.market.reset_index(drop=True).copy()
    if len(frvp_features) != len(market):
        raise ValueError(
            "FRVP labeling requires one-to-one alignment between context features and market bars. "
            f"Got {len(frvp_features)} features vs {len(market)} bars."
        )

    market["timestamp"] = pd.to_datetime(market["timestamp"], errors="coerce", utc=True)
    market = market.set_index("timestamp", drop=False)

    path_5m = prepared_context.continuous.path_bars.bars.copy()
    path_5m["timestamp"] = pd.to_datetime(path_5m["timestamp"], errors="coerce", utc=True)
    path_5m = path_5m.set_index("timestamp")
    path_5m = path_5m.loc[:, ["open", "high", "low", "close", "volume"]].copy()

    atr_5m, atr_30m, structural_atr_5m, sw30, sw1h, path_30m, path_1hr = _detect_htf_swings(path_5m, params)
    events = _build_event_candidates(market, frvp_features, structural_atr_5m.reindex(market.index))

    htf_match_30m = compute_htf_swing_match(events, sw30, params.htf_confluence_window_minutes) if events else {}
    htf_match_1h = (
        compute_htf_swing_match(events, sw1h, params.htf_confluence_window_minutes * 2) if events and sw1h else {}
    )
    events = annotate_swing_context(
        events,
        sw30,
        sw1h,
        htf_match_30m,
        htf_match_1h,
        base_bar_minutes=base_bar_minutes,
    )
    events = trend_scan_swings(path_5m, events, params)
    events = _evaluate_event_barriers(market.loc[:, ["open", "high", "low", "close"]], structural_atr_5m.reindex(market.index), events, params)
    events = _score_event_entries(market.loc[:, ["open", "high", "low", "close"]], structural_atr_5m.reindex(market.index), events)
    events = _compute_event_quality(events, params)
    events, exclusion_diag = _apply_exclusion_masks(market.reset_index(drop=True), events, prepared_context.continuous, base_bar_minutes, params)
    concurrency_arrays = _apply_event_concurrency_and_weights(market.index, events, params)
    label_frame = _materialize_frvp_targets(market.reset_index(drop=True), events, concurrency_arrays)
    label_frame["atr"] = atr_5m.reindex(label_frame.index)
    label_frame["structural_atr"] = structural_atr_5m.reindex(label_frame.index)
    label_frame["warmup_mask"] = False
    if int(params.warmup_bars) > 0 and len(label_frame):
        label_frame.iloc[: int(params.warmup_bars), label_frame.columns.get_loc("warmup_mask")] = True

    usable_events = [event for event in events if not bool(getattr(event, "excluded", False))]
    usable_by_target: dict[str, list[SwingPoint]] = {}
    for event in usable_events:
        direction = str(getattr(event, "event_direction"))
        pooled_family = str(getattr(event, "target_family"))
        setup_family = setup_target_family(int(getattr(event, "setup_type")))
        usable_by_target.setdefault(f"{direction}_{pooled_family}", []).append(event)
        usable_by_target.setdefault(f"{direction}_{setup_family}", []).append(event)

    diagnostics: dict[str, Any] = {
        "status": "ok",
        "instrument": instrument,
        "base_timeframe": base_tf,
        "total_events_sampled": int(len(events)),
        "usable_events": int(len(usable_events)),
        "base_rate_pct": 100.0
        * (
            sum(1 for event in usable_events if str(getattr(event, "tb_outcome", "")) == "tp") / max(len(usable_events), 1)
        ),
        "label_quality_mean": float(np.mean([event.label_quality for event in usable_events])) if usable_events else 0.0,
        "event_cluster_cv": 0.0,
        "htf_swings_30m": int(len(sw30)),
        "htf_swings_1h": int(len(sw1h)),
        "path_rows_30m": int(len(path_30m)),
        "path_rows_1h": int(len(path_1hr)),
    }
    if len(usable_events) > 2:
        ordered = sorted(usable_events, key=lambda event: int(event.swing_index))
        gaps = np.diff([int(event.swing_index) for event in ordered])
        if len(gaps) > 0 and float(np.mean(gaps)) > 0:
            diagnostics["event_cluster_cv"] = float(np.std(gaps) / np.mean(gaps))

    for direction in ("long", "short"):
        for target_family in (*FRVP_POOLED_TARGET_FAMILIES, *FRVP_SETUP_TARGET_FAMILIES):
            key = f"{direction}_{target_family}"
            group = usable_by_target.get(key, [])
            diagnostics[f"events_{key}"] = int(len(group))
            diagnostics[f"base_rate_{key}_pct"] = 100.0 * (
                sum(1 for event in group if str(getattr(event, "tb_outcome", "")) == "tp") / max(len(group), 1)
            )
            diagnostics[f"quality_mean_{key}"] = (
                float(np.mean([event.label_quality for event in group])) if group else 0.0
            )

    diagnostics.update(exclusion_diag)
    if verbose:
        print(
            "[FRVP] Sampled "
            f"{diagnostics['total_events_sampled']} events, kept {diagnostics['usable_events']} "
            f"after exclusions, base rate={diagnostics['base_rate_pct']:.1f}%, "
            f"quality={diagnostics['label_quality_mean']:.3f}"
        )

    return label_frame, diagnostics, events


def build_frvp_diagnostic_report(
    df_labeled: pd.DataFrame,
    events: Sequence[SwingPoint],
) -> pd.DataFrame:
    event_frame = frvp_events_to_frame(events)
    if event_frame.empty:
        return pd.DataFrame(
            [
                {
                    "direction": direction,
                    "family": family,
                    "events": 0,
                    "events_per_year": 0.0,
                    "base_rate_pct": 0.0,
                    "quality_mean": 0.0,
                    "roll_spanning_excluded": 0,
                    "all_excluded": 0,
                }
                for direction in ("long", "short")
                for family in (FRVP_REVERSAL_FAMILY, FRVP_CONTINUATION_FAMILY)
            ]
        )

    frame = event_frame.copy()
    frame["direction"] = np.where(frame["setup_side"] > 0, "long", "short")
    frame["event_year"] = pd.to_datetime(frame["swing_time"], errors="coerce", utc=True).dt.year
    total_years = max(int(frame["event_year"].nunique()), 1)
    if not df_labeled.empty and isinstance(df_labeled.index, pd.DatetimeIndex):
        elapsed_years = max((df_labeled.index.max() - df_labeled.index.min()).total_seconds() / (365.25 * 86400.0), 0.0)
        if elapsed_years > 0:
            total_years = max(float(elapsed_years), 1.0)

    summaries: list[dict[str, Any]] = []
    for direction in ("long", "short"):
        for family in (FRVP_REVERSAL_FAMILY, FRVP_CONTINUATION_FAMILY):
            subset = frame.loc[(frame["direction"] == direction) & (frame["label_family"] == family)].copy()
            usable = subset.loc[~subset["excluded"]].copy()
            roll_span_excluded_ok = True
            if subset["flag_roll_span"].any():
                roll_span_excluded_ok = bool(subset.loc[subset["flag_roll_span"], "excluded"].all())
            summaries.append(
                {
                    "direction": direction,
                    "family": family,
                    "events": int(len(usable)),
                    "events_per_year": float(len(usable) / total_years),
                    "base_rate_pct": 100.0 * float(usable["tb_outcome"].eq("tp").mean()) if not usable.empty else 0.0,
                    "quality_mean": float(pd.to_numeric(usable["label_quality"], errors="coerce").mean())
                    if not usable.empty
                    else 0.0,
                    "roll_spanning_excluded": int(subset["flag_roll_span"].sum()),
                    "all_excluded": int(subset["excluded"].sum()),
                    "roll_bracket_flagged": int(subset["flag_roll_bracket"].sum()),
                    "quad_rebalance_flagged": int(subset["flag_quad_or_rebalance"].sum()),
                    "macro_excluded": int(subset["flag_macro"].sum()),
                    "half_day_excluded": int(subset["flag_half_day"].sum()),
                    "thin_session_excluded": int(subset["flag_thin_session"].sum()),
                    "disabled_setup_excluded": int(subset["flag_failed_auction_disabled"].sum()),
                    "past_htf_confluence_excluded": int(subset["flag_past_htf_confluence"].sum()),
                    "reversal_cooldown_excluded": int(subset["flag_reversal_cooldown"].sum()),
                    "event_cluster_cv": float(
                        np.std(np.diff(sorted(usable["swing_index"].astype(int).tolist())))
                        / np.mean(np.diff(sorted(usable["swing_index"].astype(int).tolist())))
                    )
                    if len(usable) > 2 and np.mean(np.diff(sorted(usable["swing_index"].astype(int).tolist()))) > 0
                    else 0.0,
                    "gate_events_per_year": bool(250.0 <= (len(usable) / total_years) <= 750.0),
                    "gate_base_rate": bool(
                        usable["tb_outcome"].eq("tp").mean() >= 0.45 and usable["tb_outcome"].eq("tp").mean() <= 0.60
                    )
                    if not usable.empty
                    else False,
                    # The quality target is preserved in the diagnostics as an advisory reference point.
                    "gate_quality_mean": bool(
                        pd.to_numeric(usable["label_quality"], errors="coerce").mean() >= FRVP_QUALITY_TARGET
                    )
                    if not usable.empty
                    else False,
                    "quality_target_mean": float(FRVP_QUALITY_TARGET),
                    "quality_target_met": bool(
                        pd.to_numeric(usable["label_quality"], errors="coerce").mean() >= FRVP_QUALITY_TARGET
                    )
                    if not usable.empty
                    else False,
                    "quality_target_is_blocking": False,
                    "gate_roll_spanning_excluded": roll_span_excluded_ok,
                    "gate_not_excessively_clustered": bool(
                        (
                            np.std(np.diff(sorted(usable["swing_index"].astype(int).tolist())))
                            / np.mean(np.diff(sorted(usable["swing_index"].astype(int).tolist())))
                        )
                        <= 2.5
                    )
                    if len(usable) > 2 and np.mean(np.diff(sorted(usable["swing_index"].astype(int).tolist()))) > 0
                    else True,
                }
            )
    return pd.DataFrame(summaries)


def build_frvp_setup_diagnostic_report(
    df_labeled: pd.DataFrame,
    events: Sequence[SwingPoint],
) -> pd.DataFrame:
    """Report setup-specific target coverage without changing legacy gates."""

    event_frame = frvp_events_to_frame(events)
    if event_frame.empty:
        return pd.DataFrame(
            [
                {
                    "direction": direction,
                    "setup_type": setup_type,
                    "target_family": setup_target_family(setup_type),
                    "pooled_family": pooled_target_family(setup_type),
                    "barrier_family": None,
                    "events": 0,
                    "events_per_year": 0.0,
                    "base_rate_pct": 0.0,
                    "quality_mean": 0.0,
                    "events_total": 0,
                    "events_excluded": 0,
                    "roll_spanning_excluded": 0,
                    "sample_weight_scope": "pooled_family",
                }
                for setup_type in FRVP_SETUP_TYPES
                for direction in ("long", "short")
            ]
        )

    frame = event_frame.copy()
    frame["direction"] = np.where(frame["setup_side"] > 0, "long", "short")
    frame["event_year"] = pd.to_datetime(frame["swing_time"], errors="coerce", utc=True).dt.year
    total_years: float = float(max(int(frame["event_year"].nunique()), 1))
    if not df_labeled.empty and isinstance(df_labeled.index, pd.DatetimeIndex):
        elapsed_years = max(
            (df_labeled.index.max() - df_labeled.index.min()).total_seconds() / (365.25 * 86400.0),
            0.0,
        )
        if elapsed_years > 0:
            total_years = max(float(elapsed_years), 1.0)

    rows: list[dict[str, Any]] = []
    for setup_type in FRVP_SETUP_TYPES:
        for direction in ("long", "short"):
            subset = frame.loc[
                frame["setup_type"].eq(setup_type) & frame["direction"].eq(direction)
            ].copy()
            usable = subset.loc[~subset["excluded"]].copy()
            rows.append(
                {
                    "direction": direction,
                    "setup_type": setup_type,
                    "target_family": setup_target_family(setup_type),
                    "pooled_family": pooled_target_family(setup_type),
                    "barrier_family": (
                        str(subset["barrier_family"].dropna().iloc[0])
                        if not subset["barrier_family"].dropna().empty
                        else None
                    ),
                    "events": int(len(usable)),
                    "events_per_year": float(len(usable) / total_years),
                    "base_rate_pct": (
                        100.0 * float(usable["tb_outcome"].eq("tp").mean())
                        if not usable.empty
                        else 0.0
                    ),
                    "quality_mean": (
                        float(pd.to_numeric(usable["label_quality"], errors="coerce").mean())
                        if not usable.empty
                        else 0.0
                    ),
                    "events_total": int(len(subset)),
                    "events_excluded": int(subset["excluded"].sum()),
                    "roll_spanning_excluded": int(subset["flag_roll_span"].sum()),
                    "sample_weight_scope": "pooled_family",
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "FRVPLabelingParams",
    "FRVP_CONTINUATION_FAMILY",
    "FRVP_QUALITY_TARGET",
    "FRVP_REVERSAL_FAMILY",
    "build_frvp_diagnostic_report",
    "build_frvp_setup_diagnostic_report",
    "build_frvp_labels",
    "frvp_events_to_frame",
]
