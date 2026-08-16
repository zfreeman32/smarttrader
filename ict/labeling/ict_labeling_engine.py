from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from frvp.calendars.macro import annotate_us_macro_event_flags

from ..common import session_phase_codes_from_frame
from ..config.instruments import get_ict_instrument_config
from ..config.setups import ICTSetupDetectorConfig
from ..sessions.equity import build_ict_equity_session_frame
from ..setups.detector import detect_ict_setups
from ..setups.setup_types import build_empty_setup_frame


ICT_LABEL_TARGET_COLUMNS = (
    "label_long_ict_reversal",
    "label_short_ict_reversal",
    "label_long_ict_continuation",
    "label_short_ict_continuation",
    "label_long_ict_meta",
    "label_short_ict_meta",
)

ICT_REVERSAL_FAMILY = "ict_reversal"
ICT_CONTINUATION_FAMILY = "ict_continuation"
ICT_META_FAMILY = "ict_meta"
REVERSAL_SETUP_TYPES = frozenset(
    {
        "sweep_reclaim",
        "sweep_displacement_fvg",
        "ob_retest_after_mss",
        "ifvg_reversal",
        "session_open_manipulation_pre_ib",
        "session_open_manipulation_post_ib",
    }
)
CONTINUATION_SETUP_TYPES = frozenset(
    {
        "premium_discount_continuation",
        "displacement_continuation_after_raid",
    }
)
SESSION_OPEN_SETUP_TYPES = frozenset(
    {
        "session_open_manipulation_pre_ib",
        "session_open_manipulation_post_ib",
    }
)
SETUP_OUTPUT_REQUIRED_COLUMNS = (
    "fired",
    "setup_type",
    "setup_family",
    "setup_side",
    "confidence",
    "anchor_level",
    "entry_price",
    "stop_reference",
    "target_reference",
    "reference_level",
    "reference_level_type",
    "sweep_type",
    "htf_context",
    "fvg_id",
    "ce_price",
    "order_block_id",
    "displacement_id",
    "displacement_volume_z",
    "session_phase",
)


@dataclass(frozen=True)
class ICTLabelingConfig:
    """Config for the Phase 4 ICT event and labeling engine."""

    instrument: str = "es"
    primary_timeframe: str = "5m"
    execution_timeframe: str = "1m"
    htf_timeframes: tuple[str, ...] = ("30m", "1h")
    source_timezone: str = "UTC"
    canonical_timezone: str = "UTC"
    enabled: bool = True
    timeout_policy: str = "pnl_sign"
    breakeven_plus_cost_atr: float = 0.0
    min_stop_atr: float = 0.25
    min_stop_ticks: int = 1
    min_target_rr: float = 1.25
    reversal_fallback_target_atr: float = 2.0
    continuation_fallback_target_atr: float = 2.5
    reversal_max_bars: int = 18
    session_open_reversal_max_bars: int = 12
    continuation_max_bars: int = 24
    continuation_max_extension_bars: int = 24
    continuation_trailing_stop_atr: float = 1.0
    continuation_trailing_stop_ticks: int = 4
    continuation_lock_in_entry_after_target: bool = True
    horizon_vol_scale_min: float = 0.67
    horizon_vol_scale_max: float = 1.75
    macro_event_exclusion_enabled: bool = True
    half_day_exclusion_enabled: bool = True
    lunch_window_exclusion_enabled: bool = True
    thin_session_exclusion_enabled: bool = True
    thin_session_exclude_overnight: bool = False
    freeze_lunch_clock: bool = True
    lunch_dominated_share_threshold: float = 0.5
    thin_session_minutes_until_close: int = 60
    thin_session_min_rth_share: float = 0.75
    warmup_bars: int = 0
    require_intrabar_resolution: bool = True
    compute_uniqueness: bool = True

    def validate(self) -> None:
        assert self.timeout_policy in {"pnl_sign", "zero"}
        assert self.min_stop_atr > 0
        assert self.min_stop_ticks >= 1
        assert self.min_target_rr > 0
        assert self.reversal_fallback_target_atr > 0
        assert self.continuation_fallback_target_atr > 0
        assert self.reversal_max_bars >= 1
        assert self.session_open_reversal_max_bars >= 1
        assert self.continuation_max_bars >= 1
        assert self.continuation_max_extension_bars >= 0
        assert self.continuation_trailing_stop_atr > 0
        assert self.continuation_trailing_stop_ticks >= 1
        assert self.horizon_vol_scale_min > 0
        assert self.horizon_vol_scale_max >= self.horizon_vol_scale_min
        assert 0.0 <= self.lunch_dominated_share_threshold <= 1.0
        assert 0.0 <= self.thin_session_min_rth_share <= 1.0
        assert self.thin_session_minutes_until_close >= 0
        assert self.warmup_bars >= 0


@dataclass
class ICTEvent:
    label_family: str
    setup_type: str
    setup_family: str
    setup_side: int
    event_direction: str
    event_time: pd.Timestamp
    signal_index: int
    anchor_level: float
    reference_level: float
    reference_level_type: str
    sweep_type: str
    setup_confidence: float
    stop_reference: float
    target_reference: float
    htf_context: str
    fvg_id: int | None
    ce_price: float
    order_block_id: int | None
    displacement_id: int | None
    displacement_volume_z: float
    session_phase: int | None
    entry_index: int | None = None
    entry_time: pd.Timestamp | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    barrier_family: str | None = None
    max_holding_bars: int | None = None
    horizon_scale: float | None = None
    rr_ratio: float | None = None
    target_hit_index: int | None = None
    target_hit_time: pd.Timestamp | None = None
    tb_outcome: str | None = None
    tb_return: float | None = None
    tb_bars_held: int | None = None
    barrier_end_index: int | None = None
    barrier_end_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    label_quality: float = 0.0
    sample_weight: float | None = None
    ict_concurrency: int = 0
    excluded: bool = False
    exclude_reasons: list[str] | None = None
    htf_confluence_flag: bool = False

    def __post_init__(self) -> None:
        if self.exclude_reasons is None:
            self.exclude_reasons = []


def get_ict_helper_column_names(direction: str, family: str) -> dict[str, str]:
    normalized_direction = str(direction).strip().lower()
    normalized_family = str(family).strip().lower()
    return {
        "sample_weight": f"sample_weight_{normalized_direction}_{normalized_family}",
        "exclude": f"exclude_{normalized_direction}_{normalized_family}",
        "neg_ok": f"neg_ok_{normalized_direction}_{normalized_family}",
        "concurrency": f"concurrency_{normalized_direction}_{normalized_family}",
        "htf_confluence": f"htf_confluence_{normalized_direction}_{normalized_family}",
    }


def build_ict_labels(
    df_5m: pd.DataFrame,
    params: ICTLabelingConfig | None = None,
    *,
    df_1m: pd.DataFrame | None = None,
    setup_output: pd.DataFrame | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], list[ICTEvent]]:
    config = params or ICTLabelingConfig()
    config.validate()

    market = _prepare_market_frame(df_5m, config)
    if not config.enabled:
        diagnostics = {"status": "skipped", "reason": "disabled", "total_events_sampled": 0}
        return _empty_label_frame(market), diagnostics, []

    exec_market = _prepare_execution_frame(df_1m, config) if df_1m is not None else None
    setup_frame = _resolve_setup_output(market, config, setup_output=setup_output)
    events = _build_ict_events(market, setup_frame)
    events = _evaluate_event_barriers(market, events, config, exec_market=exec_market)
    events = _apply_basic_exclusions(market, events, config)
    events = _compute_event_quality(events)
    concurrency_arrays = _apply_event_concurrency_and_weights(market, events, config)
    label_frame = _materialize_ict_targets(market, events, concurrency_arrays)
    diagnostics = _build_labeling_diagnostics(market, events, config)

    if verbose:
        usable = diagnostics["usable_events"]
        total = diagnostics["total_events_sampled"]
        base_rate = diagnostics["base_rate_pct"]
        print(f"[ICT] Sampled {total} setup events, kept {usable}, base rate={base_rate:.1f}%")

    return label_frame, diagnostics, events


def ict_events_to_frame(events: Sequence[ICTEvent]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        rows.append(
            {
                "label_family": event.label_family,
                "setup_type": event.setup_type,
                "setup_family": event.setup_family,
                "setup_side": int(event.setup_side),
                "event_direction": event.event_direction,
                "event_time": event.event_time,
                "signal_index": event.signal_index,
                "anchor_level": event.anchor_level,
                "reference_level": event.reference_level,
                "reference_level_type": event.reference_level_type,
                "sweep_type": event.sweep_type,
                "setup_confidence": event.setup_confidence,
                "stop_reference": event.stop_reference,
                "target_reference": event.target_reference,
                "htf_context": event.htf_context,
                "fvg_id": event.fvg_id,
                "ce_price": event.ce_price,
                "order_block_id": event.order_block_id,
                "displacement_id": event.displacement_id,
                "displacement_volume_z": event.displacement_volume_z,
                "session_phase": event.session_phase,
                "entry_index": event.entry_index,
                "entry_time": event.entry_time,
                "entry_price": event.entry_price,
                "stop_price": event.stop_price,
                "target_price": event.target_price,
                "barrier_family": event.barrier_family,
                "max_holding_bars": event.max_holding_bars,
                "horizon_scale": event.horizon_scale,
                "rr_ratio": event.rr_ratio,
                "target_hit_index": event.target_hit_index,
                "target_hit_time": event.target_hit_time,
                "tb_outcome": event.tb_outcome,
                "tb_return": event.tb_return,
                "tb_bars_held": event.tb_bars_held,
                "barrier_end_index": event.barrier_end_index,
                "barrier_end_time": event.barrier_end_time,
                "exit_price": event.exit_price,
                "exit_reason": event.exit_reason,
                "label_quality": event.label_quality,
                "sample_weight": event.sample_weight,
                "ict_concurrency": event.ict_concurrency,
                "htf_confluence_flag": event.htf_confluence_flag,
                "excluded": bool(event.excluded),
                "exclude_reasons": "|".join(event.exclude_reasons or []),
            }
        )
    return pd.DataFrame(rows)


def _prepare_market_frame(df: pd.DataFrame, config: ICTLabelingConfig) -> pd.DataFrame:
    working = df.copy()
    if "datetime" in working.columns:
        datetime_values = pd.to_datetime(working["datetime"], errors="coerce", utc=True)
    elif "timestamp" in working.columns:
        datetime_values = pd.to_datetime(working["timestamp"], errors="coerce", utc=True)
        working["datetime"] = datetime_values
    elif isinstance(working.index, pd.DatetimeIndex):
        datetime_values = pd.to_datetime(working.index, errors="coerce", utc=True)
        working["datetime"] = datetime_values
    else:
        raise ValueError("ICT labeling requires a datetime-aware index or a 'datetime'/'timestamp' column.")

    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(working.columns))
    if missing:
        raise ValueError(f"ICT labeling is missing required columns: {', '.join(missing)}")

    working["datetime"] = datetime_values
    working = working.loc[working["datetime"].notna()].copy()
    working = working.sort_values("datetime").reset_index(drop=True)
    if "volume" not in working.columns:
        working["volume"] = 0.0

    session = build_ict_equity_session_frame(
        working["datetime"],
        instrument=config.instrument,
        source_timezone=config.source_timezone,
        canonical_timezone=config.canonical_timezone,
    )
    macro_flags = annotate_us_macro_event_flags(
        working["datetime"],
        source_timezone=config.canonical_timezone,
        canonical_timezone=config.canonical_timezone,
    )
    local_dt = pd.to_datetime(session["equity_datetime"], errors="coerce")
    local_minutes = (local_dt.dt.hour * 60) + local_dt.dt.minute

    for column in (
        "session_date",
        "is_rth",
        "is_overnight",
        "is_ib",
        "ib_complete",
        "minutes_since_rth_open",
        "minutes_until_rth_close",
        "equity_holiday_flag",
        "equity_half_day_flag",
        "equity_early_close_flag",
        "equity_datetime",
    ):
        if column not in working.columns:
            working[column] = session[column].to_numpy()
    for column in macro_flags.columns:
        if column not in working.columns:
            working[column] = macro_flags[column].to_numpy()
    if "ict_lunch_lull_flag" not in working.columns:
        working["ict_lunch_lull_flag"] = local_minutes.between(12 * 60, 13 * 60 + 30, inclusive="left").astype(np.int8)
    if "ict_close_ramp_flag" not in working.columns:
        working["ict_close_ramp_flag"] = (
            pd.Series(working["is_rth"], index=working.index).fillna(False).astype(bool)
            & pd.to_numeric(working["minutes_until_rth_close"], errors="coerce").le(60)
        ).astype(np.int8)
    if "ict_session_phase_code" not in working.columns:
        working["ict_session_phase_code"] = session_phase_codes_from_frame(session)
    return working


def _prepare_execution_frame(df: pd.DataFrame, config: ICTLabelingConfig) -> pd.DataFrame:
    exec_market = _prepare_market_frame(df, config)
    exec_market = exec_market.loc[:, ["datetime", "open", "high", "low", "close", "volume"]].copy()
    return exec_market


def _resolve_setup_output(
    market: pd.DataFrame,
    config: ICTLabelingConfig,
    *,
    setup_output: pd.DataFrame | None,
) -> pd.DataFrame:
    if setup_output is not None:
        setup_frame = setup_output.reset_index(drop=True).copy()
    elif set(SETUP_OUTPUT_REQUIRED_COLUMNS).issubset(market.columns):
        setup_frame = market.loc[:, SETUP_OUTPUT_REQUIRED_COLUMNS].reset_index(drop=True).copy()
    else:
        detector_config = ICTSetupDetectorConfig(instrument=config.instrument)
        setup_frame = detect_ict_setups(market.copy(), config=detector_config).reset_index(drop=True)

    if len(setup_frame) != len(market):
        raise ValueError(
            "ICT setup output must align one-to-one with the 5m market frame. "
            f"Got {len(setup_frame)} setup rows vs {len(market)} market rows."
        )

    missing = [column for column in SETUP_OUTPUT_REQUIRED_COLUMNS if column not in setup_frame.columns]
    if missing:
        event_time = market["datetime"]
        repaired = build_empty_setup_frame(setup_frame.index, event_time=event_time)
        for column in setup_frame.columns:
            if column in repaired.columns:
                repaired[column] = setup_frame[column]
        setup_frame = repaired
    return setup_frame


def _build_ict_events(market: pd.DataFrame, setup_frame: pd.DataFrame) -> list[ICTEvent]:
    events: list[ICTEvent] = []
    timestamps = pd.DatetimeIndex(market["datetime"])
    for idx, row in setup_frame.iterrows():
        if not _as_bool(row.get("fired")):
            continue
        setup_type = str(row.get("setup_type", "")).strip().lower()
        setup_side = _to_int(row.get("setup_side"), default=0)
        if setup_type in REVERSAL_SETUP_TYPES:
            label_family = ICT_REVERSAL_FAMILY
        elif setup_type in CONTINUATION_SETUP_TYPES:
            label_family = ICT_CONTINUATION_FAMILY
        else:
            continue
        if setup_side not in {-1, 1}:
            continue

        event = ICTEvent(
            label_family=label_family,
            setup_type=setup_type,
            setup_family=str(row.get("setup_family", "")),
            setup_side=setup_side,
            event_direction="long" if setup_side > 0 else "short",
            event_time=pd.Timestamp(timestamps[idx]),
            signal_index=int(idx),
            anchor_level=_to_float(row.get("anchor_level")),
            reference_level=_to_float(row.get("reference_level")),
            reference_level_type=str(row.get("reference_level_type", "")),
            sweep_type=str(row.get("sweep_type", "")),
            setup_confidence=_to_float(row.get("confidence"), default=0.0),
            stop_reference=_to_float(row.get("stop_reference")),
            target_reference=_to_float(row.get("target_reference")),
            htf_context=str(row.get("htf_context", "")),
            fvg_id=_nullable_int(row.get("fvg_id")),
            ce_price=_to_float(row.get("ce_price")),
            order_block_id=_nullable_int(row.get("order_block_id")),
            displacement_id=_nullable_int(row.get("displacement_id")),
            displacement_volume_z=_to_float(row.get("displacement_volume_z")),
            session_phase=_nullable_int(row.get("session_phase")),
        )
        event.htf_confluence_flag = "aligned" in event.htf_context
        events.append(event)
    return events


def _evaluate_event_barriers(
    market: pd.DataFrame,
    events: list[ICTEvent],
    config: ICTLabelingConfig,
    *,
    exec_market: pd.DataFrame | None,
) -> list[ICTEvent]:
    if not events:
        return events

    n_rows = len(market)
    opens = pd.to_numeric(market["open"], errors="coerce").to_numpy(dtype=float, copy=False)
    highs = pd.to_numeric(market["high"], errors="coerce").to_numpy(dtype=float, copy=False)
    lows = pd.to_numeric(market["low"], errors="coerce").to_numpy(dtype=float, copy=False)
    closes = pd.to_numeric(market["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    atr = _resolve_atr_like(market)
    timestamps = pd.DatetimeIndex(market["datetime"])
    bar_minutes = _infer_bar_minutes(timestamps)
    tick_size = float(get_ict_instrument_config(config.instrument).tick_size)

    for event in events:
        signal_idx = int(event.signal_index)
        entry_idx = signal_idx + 1
        event.barrier_family = _event_barrier_family(event)

        if entry_idx >= n_rows or not np.isfinite(opens[entry_idx]):
            event.excluded = True
            event.exclude_reasons.append("no_next_open")
            event.tb_outcome = "skip"
            continue

        entry_price = float(opens[entry_idx])
        atr_at_event = float(atr.iloc[signal_idx]) if signal_idx < len(atr) else np.nan
        if not np.isfinite(atr_at_event) or atr_at_event <= 0:
            atr_at_event = float(atr.iloc[entry_idx]) if entry_idx < len(atr) else np.nan
        if not np.isfinite(atr_at_event) or atr_at_event <= 0:
            event.excluded = True
            event.exclude_reasons.append("invalid_atr")
            event.tb_outcome = "skip"
            continue

        horizon_bars, horizon_scale = _event_horizon_bars(
            event,
            market=market,
            atr=atr,
            signal_idx=signal_idx,
            atr_at_event=atr_at_event,
            config=config,
        )
        event.max_holding_bars = int(horizon_bars)
        event.horizon_scale = float(horizon_scale)

        window_indices = _build_horizon_window_indices(
            market=market,
            entry_idx=entry_idx,
            holding_bars=int(event.max_holding_bars),
            config=config,
        )
        if len(window_indices) < int(event.max_holding_bars):
            event.excluded = True
            event.exclude_reasons.append("incomplete_horizon")
            event.tb_outcome = "skip"
            continue

        stop_price, target_price = _resolve_barrier_prices(
            event,
            entry_price=entry_price,
            atr_at_event=atr_at_event,
            tick_size=tick_size,
            config=config,
        )
        if not np.isfinite(stop_price) or not np.isfinite(target_price):
            event.excluded = True
            event.exclude_reasons.append("invalid_barrier_geometry")
            event.tb_outcome = "skip"
            continue

        if event.setup_side > 0 and not (stop_price < entry_price < target_price):
            event.excluded = True
            event.exclude_reasons.append("invalid_barrier_geometry")
            event.tb_outcome = "skip"
            continue
        if event.setup_side < 0 and not (target_price < entry_price < stop_price):
            event.excluded = True
            event.exclude_reasons.append("invalid_barrier_geometry")
            event.tb_outcome = "skip"
            continue

        event.entry_index = entry_idx
        event.entry_time = pd.Timestamp(timestamps[entry_idx])
        event.entry_price = entry_price
        event.stop_price = float(stop_price)
        event.target_price = float(target_price)
        event.rr_ratio = abs(target_price - entry_price) / max(abs(entry_price - stop_price), 1e-9)

        resolved = False
        continuation_target_activated = False
        continuation_trailing_stop = float(stop_price)
        continuation_extension_loaded = False
        continuation_high_watermark = float(entry_price)
        continuation_low_watermark = float(entry_price)
        bar_pointer = 0
        while True:
            while bar_pointer < len(window_indices):
                bar_idx = int(window_indices[bar_pointer])
                if continuation_target_activated:
                    continuation_high_watermark = max(
                        continuation_high_watermark,
                        highs[bar_idx] if np.isfinite(highs[bar_idx]) else continuation_high_watermark,
                    )
                    continuation_low_watermark = min(
                        continuation_low_watermark,
                        lows[bar_idx] if np.isfinite(lows[bar_idx]) else continuation_low_watermark,
                    )
                    continuation_trailing_stop = _continuation_trailing_stop(
                        side=event.setup_side,
                        current_stop=continuation_trailing_stop,
                        entry_price=entry_price,
                        high_watermark=continuation_high_watermark,
                        low_watermark=continuation_low_watermark,
                        atr_at_event=atr_at_event,
                        tick_size=tick_size,
                        config=config,
                    )
                    if _stop_hit(event.setup_side, highs[bar_idx], lows[bar_idx], continuation_trailing_stop):
                        _finalize_event_resolution(
                            event,
                            outcome="tp",
                            exit_price=float(continuation_trailing_stop),
                            bar_idx=bar_idx,
                            timestamps=timestamps,
                            entry_idx=entry_idx,
                            entry_price=entry_price,
                            exit_reason="continuation_trailing_stop",
                        )
                        resolved = True
                        break
                    bar_pointer += 1
                    continue

                hit_target = _target_hit(event.setup_side, highs[bar_idx], lows[bar_idx], target_price)
                hit_stop = _stop_hit(event.setup_side, highs[bar_idx], lows[bar_idx], stop_price)
                if not hit_target and not hit_stop:
                    bar_pointer += 1
                    continue

                if hit_target and hit_stop:
                    bar_start = pd.Timestamp(timestamps[bar_idx])
                    resolution = _resolve_intrabar_order(
                        event,
                        exec_market=exec_market,
                        bar_start=bar_start,
                        bar_minutes=bar_minutes,
                        stop_price=stop_price,
                        target_price=target_price,
                        config=config,
                    )
                    if resolution is None:
                        event.excluded = True
                        reason = "ambiguous_5m_without_1m" if exec_market is None else "unresolved_intrabar_1m"
                        event.exclude_reasons.append(reason)
                        event.tb_outcome = "ambiguous"
                        resolved = True
                        break
                    outcome, exit_price = resolution
                    if (
                        outcome == "tp"
                        and event.label_family == ICT_CONTINUATION_FAMILY
                        and config.continuation_max_extension_bars > 0
                    ):
                        continuation_target_activated = True
                        event.target_hit_index = bar_idx
                        event.target_hit_time = pd.Timestamp(timestamps[bar_idx])
                        continuation_high_watermark = max(continuation_high_watermark, target_price, highs[bar_idx])
                        continuation_low_watermark = min(continuation_low_watermark, target_price, lows[bar_idx])
                        continuation_trailing_stop = _continuation_trailing_stop(
                            side=event.setup_side,
                            current_stop=stop_price,
                            entry_price=entry_price,
                            high_watermark=continuation_high_watermark,
                            low_watermark=continuation_low_watermark,
                            atr_at_event=atr_at_event,
                            tick_size=tick_size,
                            config=config,
                        )
                        bar_pointer += 1
                        continue
                    exit_reason = "target_touch" if outcome == "tp" else "stop_touch"
                elif hit_target:
                    if (
                        event.label_family == ICT_CONTINUATION_FAMILY
                        and config.continuation_max_extension_bars > 0
                    ):
                        continuation_target_activated = True
                        event.target_hit_index = bar_idx
                        event.target_hit_time = pd.Timestamp(timestamps[bar_idx])
                        continuation_high_watermark = max(continuation_high_watermark, target_price, highs[bar_idx])
                        continuation_low_watermark = min(continuation_low_watermark, target_price, lows[bar_idx])
                        continuation_trailing_stop = _continuation_trailing_stop(
                            side=event.setup_side,
                            current_stop=stop_price,
                            entry_price=entry_price,
                            high_watermark=continuation_high_watermark,
                            low_watermark=continuation_low_watermark,
                            atr_at_event=atr_at_event,
                            tick_size=tick_size,
                            config=config,
                        )
                        bar_pointer += 1
                        continue
                    outcome, exit_price, exit_reason = "tp", target_price, "target_touch"
                else:
                    outcome, exit_price, exit_reason = "sl", stop_price, "stop_touch"

                _finalize_event_resolution(
                    event,
                    outcome=outcome,
                    exit_price=float(exit_price),
                    bar_idx=bar_idx,
                    timestamps=timestamps,
                    entry_idx=entry_idx,
                    entry_price=entry_price,
                    exit_reason=exit_reason,
                )
                resolved = True
                break

            if resolved:
                break
            if continuation_target_activated and not continuation_extension_loaded and config.continuation_max_extension_bars > 0:
                continuation_extension_loaded = True
                extension_start = (window_indices[-1] + 1) if window_indices else (entry_idx + 1)
                extension_indices = _build_horizon_window_indices(
                    market=market,
                    entry_idx=extension_start,
                    holding_bars=int(config.continuation_max_extension_bars),
                    config=config,
                )
                if extension_indices:
                    window_indices.extend(extension_indices)
                    continue
            break

        if resolved:
            continue

        end_idx = window_indices[-1]
        exit_price = float(closes[end_idx])
        if continuation_target_activated:
            if end_idx == int(event.target_hit_index or end_idx):
                exit_price = float(target_price)
                exit_reason = "target_touch_no_extension"
            else:
                exit_price = _continuation_timeout_exit_price(
                    side=event.setup_side,
                    close_price=exit_price,
                    trailing_stop=continuation_trailing_stop,
                    entry_price=entry_price,
                    config=config,
                )
                exit_reason = "continuation_timeout_after_target"
            outcome = "tp"
        elif config.timeout_policy == "zero":
            outcome = "timeout_loss"
            exit_reason = "timeout_zero_policy"
        else:
            threshold = float(config.breakeven_plus_cost_atr) * atr_at_event
            outcome = "timeout_profit" if _signed_return(event.setup_side, entry_price, exit_price) >= threshold else "timeout_loss"
            exit_reason = "timeout_pnl_sign"
        signed_return = _signed_return(event.setup_side, entry_price, exit_price)
        event.barrier_end_index = int(end_idx)
        event.barrier_end_time = pd.Timestamp(timestamps[end_idx])
        event.tb_bars_held = int((end_idx - entry_idx) + 1)
        event.exit_price = exit_price
        event.tb_return = float(signed_return)
        event.tb_outcome = outcome
        event.exit_reason = exit_reason

    return events


def _apply_basic_exclusions(
    market: pd.DataFrame,
    events: list[ICTEvent],
    config: ICTLabelingConfig,
) -> list[ICTEvent]:
    if not events:
        return events

    warmup = pd.to_numeric(market.get("warmup_mask", pd.Series(False, index=market.index)), errors="coerce").fillna(0).astype(bool)
    contract = market["contract_id"].astype(str) if "contract_id" in market.columns else None

    for event in events:
        if event.excluded:
            continue
        if event.signal_index < int(config.warmup_bars) or (event.signal_index < len(warmup) and bool(warmup.iloc[event.signal_index])):
            event.excluded = True
            event.exclude_reasons.append("warmup_mask")
            continue
        if event.barrier_end_index is None:
            continue

        window_slice = slice(int(event.signal_index), int(event.barrier_end_index) + 1)
        if contract is not None:
            window = contract.iloc[window_slice]
            if window.nunique(dropna=False) > 1:
                event.excluded = True
                event.exclude_reasons.append("roll_spanning_window")
                continue

        window_frame = market.iloc[window_slice]
        if config.macro_event_exclusion_enabled and _window_has_flag(window_frame, "macro_event_flag"):
            event.excluded = True
            event.exclude_reasons.append("macro_event_window")
            continue
        if config.half_day_exclusion_enabled and (
            _window_has_flag(window_frame, "equity_half_day_flag")
            or _window_has_flag(window_frame, "equity_early_close_flag")
            or _window_has_flag(window_frame, "equity_holiday_flag")
        ):
            event.excluded = True
            event.exclude_reasons.append("half_day_or_holiday_window")
            continue
        if config.lunch_window_exclusion_enabled and _window_flag_share(window_frame, "ict_lunch_lull_flag") >= float(
            config.lunch_dominated_share_threshold
        ):
            event.excluded = True
            event.exclude_reasons.append("lunch_dominated_window")
            continue
        if config.thin_session_exclusion_enabled and _is_thin_session_window(window_frame, config):
            event.excluded = True
            event.exclude_reasons.append("thin_session_window")

    return events


def _compute_event_quality(events: list[ICTEvent]) -> list[ICTEvent]:
    for event in events:
        rr_score = 0.0 if event.rr_ratio is None or not np.isfinite(event.rr_ratio) else min(float(event.rr_ratio) / 3.0, 1.0)
        htf_score = 1.0 if event.htf_confluence_flag else (0.5 if "neutral" in event.htf_context else 0.0)
        event.label_quality = _clamp01((0.55 * _clamp01(event.setup_confidence)) + (0.25 * rr_score) + (0.20 * htf_score))
    return events


def _apply_event_concurrency_and_weights(
    market: pd.DataFrame,
    events: list[ICTEvent],
    config: ICTLabelingConfig,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    n_rows = len(market)
    usable = [event for event in events if not event.excluded and event.barrier_end_index is not None]
    groups: dict[tuple[str, str], list[ICTEvent]] = {}
    for event in usable:
        groups.setdefault((event.event_direction, event.label_family), []).append(event)
        groups.setdefault((event.event_direction, ICT_META_FAMILY), []).append(event)

    outputs: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, group in groups.items():
        conc = np.zeros(n_rows, dtype=np.int32)
        event_mask = np.zeros(n_rows, dtype=np.int8)
        for event in group:
            start = int(event.signal_index)
            end = int(event.barrier_end_index or event.signal_index)
            conc[start : end + 1] += 1
            event_mask[start] = 1
        group_weights = np.ones(n_rows, dtype=np.float32)
        base_weights = _compute_sample_weights(conc, event_mask) if config.compute_uniqueness else np.ones(n_rows, dtype=np.float64)
        for event in group:
            signal_index = int(event.signal_index)
            quality_weight = max(float(event.label_quality), 0.1)
            sample_weight = float(base_weights[signal_index] * quality_weight)
            group_weights[signal_index] = sample_weight
            if key[1] == event.label_family:
                event.ict_concurrency = int(conc[signal_index])
                event.sample_weight = sample_weight

        outputs[key] = {
            "concurrency": conc,
            "sample_weight": group_weights,
        }

    return outputs


def _materialize_ict_targets(
    market: pd.DataFrame,
    events: list[ICTEvent],
    group_arrays: dict[tuple[str, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.DatetimeIndex(market["datetime"]))
    directions = ("long", "short")
    families = (ICT_REVERSAL_FAMILY, ICT_CONTINUATION_FAMILY, ICT_META_FAMILY)

    for direction in directions:
        for family in families:
            label_column = f"label_{direction}_{family}"
            helpers = get_ict_helper_column_names(direction, family)
            quality_column = f"label_quality_{direction}_{family}"
            group_state = group_arrays.get((direction, family), {})
            out[label_column] = np.zeros(len(out), dtype=np.int8)
            out[quality_column] = np.zeros(len(out), dtype=np.float32)
            out[helpers["sample_weight"]] = group_state.get(
                "sample_weight",
                np.ones(len(out), dtype=np.float32),
            )
            out[helpers["exclude"]] = np.ones(len(out), dtype=bool)
            out[helpers["neg_ok"]] = np.zeros(len(out), dtype=bool)
            out[helpers["concurrency"]] = group_state.get(
                "concurrency",
                np.zeros(len(out), dtype=np.int32),
            )
            out[helpers["htf_confluence"]] = np.zeros(len(out), dtype=np.int8)

    for event in events:
        direction = event.event_direction
        label_value = 1 if _event_is_positive(event) else 0
        family_columns = get_ict_helper_column_names(direction, event.label_family)
        meta_columns = get_ict_helper_column_names(direction, ICT_META_FAMILY)
        family_label = f"label_{direction}_{event.label_family}"
        meta_label = f"label_{direction}_{ICT_META_FAMILY}"
        family_quality = f"label_quality_{direction}_{event.label_family}"
        meta_quality = f"label_quality_{direction}_{ICT_META_FAMILY}"
        idx = int(event.signal_index)

        if event.excluded:
            continue

        out.iat[idx, out.columns.get_loc(family_label)] = label_value
        out.iat[idx, out.columns.get_loc(meta_label)] = label_value
        out.iat[idx, out.columns.get_loc(family_quality)] = float(event.label_quality)
        out.iat[idx, out.columns.get_loc(meta_quality)] = float(event.label_quality)
        out.iat[idx, out.columns.get_loc(family_columns["exclude"])] = False
        out.iat[idx, out.columns.get_loc(meta_columns["exclude"])] = False
        out.iat[idx, out.columns.get_loc(family_columns["neg_ok"])] = True
        out.iat[idx, out.columns.get_loc(meta_columns["neg_ok"])] = True
        out.iat[idx, out.columns.get_loc(family_columns["htf_confluence"])] = int(event.htf_confluence_flag)
        out.iat[idx, out.columns.get_loc(meta_columns["htf_confluence"])] = int(event.htf_confluence_flag)

    out["warmup_mask"] = pd.to_numeric(market.get("warmup_mask", pd.Series(False, index=market.index)), errors="coerce").fillna(0).astype(bool).to_numpy()
    return out


def _build_labeling_diagnostics(
    market: pd.DataFrame,
    events: list[ICTEvent],
    config: ICTLabelingConfig,
) -> dict[str, Any]:
    usable = [event for event in events if not event.excluded]
    positives = [event for event in usable if _event_is_positive(event)]
    diagnostics: dict[str, Any] = {
        "status": "ok",
        "instrument": config.instrument,
        "total_rows": int(len(market)),
        "total_events_sampled": int(len(events)),
        "usable_events": int(len(usable)),
        "positive_events": int(len(positives)),
        "base_rate_pct": 100.0 * (len(positives) / max(len(usable), 1)),
        "label_quality_mean": float(np.mean([event.label_quality for event in usable])) if usable else 0.0,
    }

    for direction in ("long", "short"):
        direction_events = [event for event in usable if event.event_direction == direction]
        diagnostics[f"events_{direction}"] = int(len(direction_events))
        diagnostics[f"base_rate_{direction}_pct"] = 100.0 * (
            sum(1 for event in direction_events if _event_is_positive(event)) / max(len(direction_events), 1)
        )

    for family in (ICT_REVERSAL_FAMILY, ICT_CONTINUATION_FAMILY):
        family_events = [event for event in usable if event.label_family == family]
        diagnostics[f"events_{family}"] = int(len(family_events))
        diagnostics[f"base_rate_{family}_pct"] = 100.0 * (
            sum(1 for event in family_events if _event_is_positive(event)) / max(len(family_events), 1)
        )
        diagnostics[f"horizon_{family}_mean_bars"] = (
            float(np.mean([event.max_holding_bars for event in family_events if event.max_holding_bars is not None]))
            if family_events
            else 0.0
        )

    exit_reason_counts: dict[str, int] = {}
    for event in usable:
        if event.exit_reason:
            exit_reason_counts[event.exit_reason] = exit_reason_counts.get(event.exit_reason, 0) + 1
    for reason, count in sorted(exit_reason_counts.items()):
        diagnostics[f"events_exit_{reason}"] = int(count)
    diagnostics["continuation_trail_activations"] = int(
        sum(1 for event in usable if event.label_family == ICT_CONTINUATION_FAMILY and event.target_hit_index is not None)
    )

    reason_counts: dict[str, int] = {}
    for event in events:
        for reason in event.exclude_reasons or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in sorted(reason_counts.items()):
        diagnostics[f"events_excluded_{reason}"] = int(count)

    return diagnostics


def _empty_label_frame(market: pd.DataFrame) -> pd.DataFrame:
    return _materialize_ict_targets(market, [], {})


def _resolve_barrier_prices(
    event: ICTEvent,
    *,
    entry_price: float,
    atr_at_event: float,
    tick_size: float,
    config: ICTLabelingConfig,
) -> tuple[float, float]:
    stop_floor = max(float(config.min_stop_atr) * atr_at_event, float(config.min_stop_ticks) * tick_size)
    target_floor_atr = (
        float(config.reversal_fallback_target_atr)
        if event.label_family == ICT_REVERSAL_FAMILY
        else float(config.continuation_fallback_target_atr)
    )

    if event.setup_side > 0:
        raw_stop = event.stop_reference if np.isfinite(event.stop_reference) and event.stop_reference < entry_price else np.nan
        stop_price = raw_stop if np.isfinite(raw_stop) and (entry_price - raw_stop) >= stop_floor else entry_price - stop_floor
        min_target = max(entry_price + (target_floor_atr * atr_at_event), entry_price + (entry_price - stop_price) * float(config.min_target_rr))
        raw_target = event.target_reference if np.isfinite(event.target_reference) and event.target_reference > entry_price else np.nan
        target_price = raw_target if np.isfinite(raw_target) and raw_target >= min_target else min_target
        return stop_price, target_price

    raw_stop = event.stop_reference if np.isfinite(event.stop_reference) and event.stop_reference > entry_price else np.nan
    stop_price = raw_stop if np.isfinite(raw_stop) and (raw_stop - entry_price) >= stop_floor else entry_price + stop_floor
    min_target = min(entry_price - (target_floor_atr * atr_at_event), entry_price - (stop_price - entry_price) * float(config.min_target_rr))
    raw_target = event.target_reference if np.isfinite(event.target_reference) and event.target_reference < entry_price else np.nan
    target_price = raw_target if np.isfinite(raw_target) and raw_target <= min_target else min_target
    return stop_price, target_price


def _resolve_intrabar_order(
    event: ICTEvent,
    *,
    exec_market: pd.DataFrame | None,
    bar_start: pd.Timestamp,
    bar_minutes: float,
    stop_price: float,
    target_price: float,
    config: ICTLabelingConfig,
) -> tuple[str, float] | None:
    if exec_market is None:
        return None

    bar_end = bar_start + pd.Timedelta(minutes=bar_minutes)
    minute_window = exec_market.loc[(exec_market["datetime"] >= bar_start) & (exec_market["datetime"] < bar_end)].copy()
    if minute_window.empty:
        return None

    for _, minute_bar in minute_window.iterrows():
        high = _to_float(minute_bar.get("high"))
        low = _to_float(minute_bar.get("low"))
        hit_target = _target_hit(event.setup_side, high, low, target_price)
        hit_stop = _stop_hit(event.setup_side, high, low, stop_price)
        if not hit_target and not hit_stop:
            continue
        if hit_target and hit_stop:
            return None if config.require_intrabar_resolution else ("sl", stop_price)
        if hit_target:
            return "tp", target_price
        return "sl", stop_price
    return None


def _event_barrier_family(event: ICTEvent) -> str:
    if event.setup_type in SESSION_OPEN_SETUP_TYPES:
        return "session_open_reversal"
    if event.label_family == ICT_REVERSAL_FAMILY:
        return "reversal"
    return "continuation"


def _event_horizon_bars(
    event: ICTEvent,
    *,
    market: pd.DataFrame,
    atr: pd.Series,
    signal_idx: int,
    atr_at_event: float,
    config: ICTLabelingConfig,
) -> tuple[int, float]:
    phase_series = pd.to_numeric(market.get("ict_session_phase_code"), errors="coerce")
    phase_value = phase_series.iloc[signal_idx] if signal_idx < len(phase_series) else np.nan
    if np.isfinite(phase_value):
        phase_mask = phase_series.eq(phase_value)
        phase_reference_atr = float(pd.to_numeric(atr.loc[phase_mask], errors="coerce").dropna().median())
    else:
        phase_reference_atr = np.nan
    if not np.isfinite(phase_reference_atr) or phase_reference_atr <= 0:
        phase_reference_atr = float(pd.to_numeric(atr, errors="coerce").dropna().median())
    if not np.isfinite(phase_reference_atr) or phase_reference_atr <= 0:
        phase_reference_atr = float(atr_at_event)

    scale = 1.0
    if np.isfinite(atr_at_event) and atr_at_event > 0 and np.isfinite(phase_reference_atr) and phase_reference_atr > 0:
        scale = _clip(
            phase_reference_atr / atr_at_event,
            lower=float(config.horizon_vol_scale_min),
            upper=float(config.horizon_vol_scale_max),
        )

    if event.setup_type in SESSION_OPEN_SETUP_TYPES:
        return max(1, int(round(float(config.session_open_reversal_max_bars) * scale))), scale
    if event.label_family == ICT_REVERSAL_FAMILY:
        return max(1, int(round(float(config.reversal_max_bars) * scale))), scale
    return max(1, int(round(float(config.continuation_max_bars) * scale))), scale


def _build_horizon_window_indices(
    *,
    market: pd.DataFrame,
    entry_idx: int,
    holding_bars: int,
    config: ICTLabelingConfig,
) -> list[int]:
    if holding_bars <= 0 or entry_idx >= len(market):
        return []

    indices: list[int] = []
    counted_bars = 0
    for bar_idx in range(int(entry_idx), len(market)):
        indices.append(int(bar_idx))
        if _freeze_clock_on_bar(market.iloc[bar_idx], config):
            continue
        counted_bars += 1
        if counted_bars >= int(holding_bars):
            break
    return indices


def _freeze_clock_on_bar(row: pd.Series, config: ICTLabelingConfig) -> bool:
    return bool(config.freeze_lunch_clock) and _as_bool(row.get("ict_lunch_lull_flag"))


def _continuation_trailing_stop(
    *,
    side: int,
    current_stop: float,
    entry_price: float,
    high_watermark: float,
    low_watermark: float,
    atr_at_event: float,
    tick_size: float,
    config: ICTLabelingConfig,
) -> float:
    trail_buffer = max(
        float(config.continuation_trailing_stop_atr) * float(atr_at_event),
        float(config.continuation_trailing_stop_ticks) * float(tick_size),
    )
    if side > 0:
        candidate = float(high_watermark) - trail_buffer
        protected = max(float(current_stop), candidate)
        if config.continuation_lock_in_entry_after_target:
            protected = max(protected, float(entry_price))
        return protected

    candidate = float(low_watermark) + trail_buffer
    protected = min(float(current_stop), candidate)
    if config.continuation_lock_in_entry_after_target:
        protected = min(protected, float(entry_price))
    return protected


def _continuation_timeout_exit_price(
    *,
    side: int,
    close_price: float,
    trailing_stop: float,
    entry_price: float,
    config: ICTLabelingConfig,
) -> float:
    if side > 0:
        floor_price = max(float(trailing_stop), float(entry_price)) if config.continuation_lock_in_entry_after_target else float(trailing_stop)
        return max(float(close_price), floor_price)

    ceiling_price = min(float(trailing_stop), float(entry_price)) if config.continuation_lock_in_entry_after_target else float(trailing_stop)
    return min(float(close_price), ceiling_price)


def _finalize_event_resolution(
    event: ICTEvent,
    *,
    outcome: str,
    exit_price: float,
    bar_idx: int,
    timestamps: pd.DatetimeIndex,
    entry_idx: int,
    entry_price: float,
    exit_reason: str,
) -> None:
    event.tb_outcome = str(outcome)
    event.exit_price = float(exit_price)
    event.barrier_end_index = int(bar_idx)
    event.barrier_end_time = pd.Timestamp(timestamps[bar_idx])
    event.tb_bars_held = int((bar_idx - entry_idx) + 1)
    event.tb_return = float(_signed_return(event.setup_side, entry_price, float(exit_price)))
    event.exit_reason = exit_reason


def _window_has_flag(window_frame: pd.DataFrame, column: str) -> bool:
    if column not in window_frame.columns:
        return False
    series = pd.to_numeric(window_frame[column], errors="coerce").fillna(0).astype(bool)
    return bool(series.any())


def _window_flag_share(window_frame: pd.DataFrame, column: str) -> float:
    if column not in window_frame.columns or window_frame.empty:
        return 0.0
    series = pd.to_numeric(window_frame[column], errors="coerce").fillna(0).astype(bool)
    return float(series.mean())


def _is_thin_session_window(window_frame: pd.DataFrame, config: ICTLabelingConfig) -> bool:
    if window_frame.empty:
        return False
    # Evaluate thin-session risk off the actual entry bar when possible.
    probe_row = window_frame.iloc[1] if len(window_frame) > 1 else window_frame.iloc[0]
    probe_is_rth = bool(_as_bool(probe_row.get("is_rth")))
    probe_is_overnight = bool(_as_bool(probe_row.get("is_overnight")))
    probe_minutes_until_close = _to_float(probe_row.get("minutes_until_rth_close"))
    if probe_is_rth and np.isfinite(probe_minutes_until_close):
        return probe_minutes_until_close <= float(config.thin_session_minutes_until_close)
    if not bool(getattr(config, "thin_session_exclude_overnight", False)) or (not probe_is_overnight):
        return False

    is_rth = (
        pd.to_numeric(window_frame.get("is_rth", pd.Series(False, index=window_frame.index)), errors="coerce")
        .fillna(0)
        .astype(bool)
    )
    rth_share = float(is_rth.mean()) if len(is_rth) else 0.0
    return rth_share < float(config.thin_session_min_rth_share)


def _event_is_positive(event: ICTEvent) -> bool:
    return str(event.tb_outcome) in {"tp", "timeout_profit"}


def _resolve_atr_like(df: pd.DataFrame) -> pd.Series:
    if "atr_14" in df.columns:
        return pd.to_numeric(df["atr_14"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(14, min_periods=14).mean()


def _infer_bar_minutes(index: pd.DatetimeIndex) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 5.0
    diffs = index.to_series().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return 5.0
    return float((diffs.dt.total_seconds() / 60.0).mode().iloc[0])


def _compute_sample_weights(concurrency: np.ndarray, labels: np.ndarray) -> np.ndarray:
    weights = np.ones(len(labels), dtype=np.float64)
    positives = labels.astype(bool)
    weights[positives] = 1.0 / np.maximum(concurrency[positives], 1)
    return weights


def _target_hit(side: int, high: float, low: float, target_price: float) -> bool:
    if side > 0:
        return np.isfinite(high) and high >= target_price
    return np.isfinite(low) and low <= target_price


def _stop_hit(side: int, high: float, low: float, stop_price: float) -> bool:
    if side > 0:
        return np.isfinite(low) and low <= stop_price
    return np.isfinite(high) and high >= stop_price


def _signed_return(side: int, entry_price: float, exit_price: float) -> float:
    if side > 0:
        return float(exit_price - entry_price)
    return float(entry_price - exit_price)


def _to_float(value: Any, *, default: float = np.nan) -> float:
    if value is None or value is pd.NA or pd.isna(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, *, default: int = 0) -> int:
    if value is None or value is pd.NA or pd.isna(value):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _nullable_int(value: Any) -> int | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    return _to_int(value, default=0) == 1


def _clamp01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _clip(value: float, *, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


__all__ = [
    "ICTLabelingConfig",
    "ICT_LABEL_TARGET_COLUMNS",
    "ICT_CONTINUATION_FAMILY",
    "ICT_META_FAMILY",
    "ICT_REVERSAL_FAMILY",
    "ICTEvent",
    "build_ict_labels",
    "get_ict_helper_column_names",
    "ict_events_to_frame",
]
