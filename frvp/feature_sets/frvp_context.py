from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from features.config import FeatureBuilderConfig
from features.feature_sets.session import build_session
from features.feature_sets.structure import build_structure
from features.feature_sets.volatility import build_volatility
from features.feature_sets.volume import build_volume
from features.fx_calendar import normalize_datetime_series
from features.io import standardize_market_frame

from ..calendars.macro import annotate_us_macro_event_flags
from ..config.instruments import InstrumentConfig, get_instrument_config
from ..sessions.equity import build_equity_market_day_labels, build_equity_session_frame

if TYPE_CHECKING:
    from ..continuity.continuous_contract import ContinuousContractResult
    from ..continuity.types import ProfileSlice
    from ..profiles.anchors import AnchorWindow, FRVPAnchorEngine, NakedVPOCLevel, NakedVPOCTracker
    from ..profiles.builder import VolumeProfile, VolumeProfileBuilder


ROLL_BRACKET_SESSIONS = 3
SESSION_PHASE_OPEN = 1
SESSION_PHASE_MORNING = 2
SESSION_PHASE_MIDDAY = 3
SESSION_PHASE_AFTERNOON = 4
SESSION_PHASE_CLOSE_RAMP = 5
SHAPE_ENCODING = {"D": 0, "P": 1, "b": -1}
DAY_TYPE_NORMAL = 1
DAY_TYPE_NORMAL_VARIATION = 2
DAY_TYPE_TREND = 3
DAY_TYPE_DOUBLE_DISTRIBUTION_TREND = 4
DAY_TYPE_NEUTRAL = 5
SCHEDULE_PATHS = {
    "es": Path("data/futures_data/es_roll_schedule.json"),
    "6e": Path("data/futures_data/6e_roll_schedule.json"),
}


@dataclass(frozen=True)
class _PreparedFRVPContext:
    instrument: str
    instrument_config: InstrumentConfig
    market: pd.DataFrame
    tagged: pd.DataFrame
    continuous: Any
    anchor_engine: Any
    profile_builder: Any
    session_index_by_date: dict[pd.Timestamp, int]


def build_frvp_context_features(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
    *,
    instrument: str | None = None,
) -> pd.DataFrame:
    """Build the Phase 1 FRVP context feature family."""

    from ..profiles.anchors import NakedVPOCLevel, NakedVPOCTracker
    from ..setups.detector import detect_frvp_setups

    prepared = _prepare_frvp_context(df, config, instrument=instrument)
    market = prepared.market.reset_index(drop=True)
    anchor_engine = prepared.anchor_engine
    profile_builder = prepared.profile_builder
    out = pd.DataFrame(index=market.index)

    timestamps_ns = pd.DatetimeIndex(market["timestamp"]).asi8
    rth_end_ns = pd.DatetimeIndex(anchor_engine.rth_sessions["end"]).asi8 if not anchor_engine.rth_sessions.empty else np.array([], dtype=np.int64)
    overnight_end_ns = (
        pd.DatetimeIndex(anchor_engine.overnight_sessions["end"]).asi8
        if not anchor_engine.overnight_sessions.empty
        else np.array([], dtype=np.int64)
    )
    prior_rth_indices = np.searchsorted(rth_end_ns, timestamps_ns, side="left") - 1 if len(rth_end_ns) else np.full(len(market), -1)
    overnight_indices = (
        np.searchsorted(overnight_end_ns, timestamps_ns, side="left") - 1
        if len(overnight_end_ns)
        else np.full(len(market), -1)
    )

    session_open_context = _build_session_open_context(market)
    session_prior_rth_summary_indices = _build_session_prior_rth_summary_indices(anchor_engine.rth_sessions)
    profile_cache: dict[tuple[str, str, pd.Timestamp, pd.Timestamp], Any] = {}
    anchor_window_cache: dict[tuple[str, object], Any] = {}
    tracker = NakedVPOCTracker()
    registered_vpocs: set[tuple[str, str, pd.Timestamp]] = set()
    prior_rth_history: list[dict[str, float | pd.Timestamp]] = []
    prior_rth_history_by_session: dict[pd.Timestamp, dict[str, float | pd.Timestamp]] = {}

    float_columns = [
        "frvp_dist_poc_session_atr",
        "frvp_dist_poc_day_atr",
        "frvp_dist_poc_swing_atr",
        "frvp_dist_vah_atr",
        "frvp_dist_val_atr",
        "frvp_dist_nearest_hvn_atr",
        "frvp_dist_nearest_lvn_atr",
        "frvp_dist_lvn_above_atr",
        "frvp_dist_lvn_below_atr",
        "frvp_naked_poc_dist_session_atr",
        "frvp_naked_poc_dist_day_atr",
        "frvp_price_position_va",
        "frvp_va_overshoot_atr",
        "frvp_vol_skew",
        "frvp_vol_concentration_top_pct",
        "frvp_vol_concentration_bot_pct",
        "frvp_poc_vol_pct",
        "frvp_hvn_above_close",
        "frvp_hvn_below_close",
        "frvp_va_width_atr",
        "frvp_va_width_zscore_20",
        "frvp_va_migration_vel",
        "frvp_poc_migration_vel",
        "frvp_poc_anchor_diff_atr",
        "frvp_va_overlap_pct",
        "frvp_swing_poc_vs_session_poc",
        "frvp_dist_ib_high_atr",
        "frvp_dist_ib_low_atr",
        "frvp_rth_eth_value_overlap",
        "frvp_open_vs_prior_poc_atr",
        "frvp_open_gap_atr",
        "frvp_naked_vpoc_dist_above_atr",
        "frvp_naked_vpoc_dist_below_atr",
        "frvp_naked_vpoc_age_sessions",
        "frvp_setup_confidence_rule",
    ]
    int_columns = [
        "frvp_above_poc",
        "frvp_in_va",
        "frvp_above_vah",
        "frvp_below_val",
        "frvp_profile_shape",
        "frvp_hvn_count",
        "frvp_lvn_count",
        "frvp_hvn_count_1atr",
        "frvp_lvn_count_1atr",
        "frvp_va_expansion",
        "frvp_multi_poc_cluster",
        "frvp_setup_type",
        "frvp_setup_side",
        "frvp_open_type",
        "frvp_open_drive_flag",
        "frvp_ib_extension",
        "frvp_session_phase",
        "frvp_gap_into_value",
        "frvp_naked_vpoc_count",
        "frvp_poc_plus_ict_fvg_confluence",
        "frvp_vah_plus_ob_confluence",
        "frvp_val_plus_ob_confluence",
        "frvp_failed_auction_with_sweep",
        "frvp_in_killzone",
        "frvp_day_type",
    ]
    for column in float_columns:
        out[column] = np.nan
    for column in int_columns:
        out[column] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    for index, row in market.iterrows():
        current_ts = pd.Timestamp(row["timestamp"])
        atr = float(row["atr_14"]) if pd.notna(row["atr_14"]) and float(row["atr_14"]) != 0.0 else np.nan
        current_price = float(row["close"])
        current_contract = str(row["contract_id"])
        session_date = pd.Timestamp(row["session_date"])

        prior_rth = _resolve_segment_window(
            anchor_engine=anchor_engine,
            cache=anchor_window_cache,
            anchor_name="prior_rth",
            summary_frame=anchor_engine.rth_sessions,
            summary_index=int(prior_rth_indices[index]),
        )
        session_prior_rth = _resolve_segment_window(
            anchor_engine=anchor_engine,
            cache=anchor_window_cache,
            anchor_name="prior_rth",
            summary_frame=anchor_engine.rth_sessions,
            summary_index=int(session_prior_rth_summary_indices.get(session_date, -1)),
        )
        overnight_eth = _resolve_segment_window(
            anchor_engine=anchor_engine,
            cache=anchor_window_cache,
            anchor_name="overnight_eth",
            summary_frame=anchor_engine.overnight_sessions,
            summary_index=int(overnight_indices[index]),
        )
        initial_balance = _resolve_initial_balance_window(
            anchor_engine=anchor_engine,
            cache=anchor_window_cache,
            row=row,
        )
        swing_window = _resolve_swing_window(
            anchor_engine=anchor_engine,
            cache=anchor_window_cache,
            current_ts=current_ts,
        )
        composite_window = _resolve_composite_window(
            anchor_engine=anchor_engine,
            cache=anchor_window_cache,
            prior_rth_summary_index=int(prior_rth_indices[index]),
        )

        profile_bundle = {
            "prior_rth": _resolve_profile(profile_builder, profile_cache, prior_rth),
            "overnight_eth": _resolve_profile(profile_builder, profile_cache, overnight_eth),
            "initial_balance": _resolve_profile(profile_builder, profile_cache, initial_balance),
            "swing_to_swing": _resolve_profile(profile_builder, profile_cache, swing_window),
            "rolling_composite": _resolve_profile(profile_builder, profile_cache, composite_window),
        }

        for anchor_name in ("prior_rth", "overnight_eth", "initial_balance"):
            anchor_window = {
                "prior_rth": prior_rth,
                "overnight_eth": overnight_eth,
                "initial_balance": initial_balance,
            }[anchor_name]
            profile = profile_bundle[anchor_name]
            if anchor_window is None or profile is None:
                continue
            session_key = pd.Timestamp(anchor_window.metadata.get("session_date")) if "session_date" in anchor_window.metadata else None
            register_key = (anchor_name, profile.contract_id, pd.Timestamp(anchor_window.completed_at))
            if register_key in registered_vpocs:
                continue
            tracker.register_level(
                NakedVPOCLevel(
                    price=float(profile.poc),
                    contract_id=profile.contract_id,
                    formed_at=pd.Timestamp(anchor_window.completed_at),
                    anchor_name=anchor_name,
                    session_date=session_key,
                )
            )
            registered_vpocs.add(register_key)

        tracker.process_bar(row)
        active_vpocs = tracker.active_levels(contract_id=current_contract)

        prior_rth_profile = profile_bundle["prior_rth"]
        overnight_profile = profile_bundle["overnight_eth"]
        swing_profile = profile_bundle["swing_to_swing"]
        composite_profile = profile_bundle["rolling_composite"]
        session_prior_rth_profile = _resolve_profile(profile_builder, profile_cache, session_prior_rth)

        if prior_rth is not None and prior_rth_profile is not None:
            history_metrics = _ensure_prior_rth_history_metrics(
                history=prior_rth_history,
                history_by_session=prior_rth_history_by_session,
                session_index_by_date=prepared.session_index_by_date,
                market=market,
                anchor_window=prior_rth,
                profile=prior_rth_profile,
                ib_minutes=prepared.instrument_config.ib_minutes,
            )
            out.at[index, "frvp_dist_poc_session_atr"] = _safe_ratio(current_price - prior_rth_profile.poc, atr)
            out.at[index, "frvp_dist_vah_atr"] = _safe_ratio(prior_rth_profile.vah - current_price, atr)
            out.at[index, "frvp_dist_val_atr"] = _safe_ratio(current_price - prior_rth_profile.val, atr)
            out.at[index, "frvp_price_position_va"] = _price_position_in_range(
                current_price,
                prior_rth_profile.val,
                prior_rth_profile.vah,
            )
            out.at[index, "frvp_above_poc"] = int(current_price > prior_rth_profile.poc)
            out.at[index, "frvp_in_va"] = int(prior_rth_profile.val <= current_price <= prior_rth_profile.vah)
            out.at[index, "frvp_above_vah"] = int(current_price > prior_rth_profile.vah)
            out.at[index, "frvp_below_val"] = int(current_price < prior_rth_profile.val)
            out.at[index, "frvp_va_overshoot_atr"] = _va_overshoot(
                current_price,
                prior_rth_profile.val,
                prior_rth_profile.vah,
                atr,
            )
            out.at[index, "frvp_profile_shape"] = SHAPE_ENCODING.get(prior_rth_profile.shape, 0)
            out.at[index, "frvp_vol_skew"] = prior_rth_profile.vol_skew
            out.at[index, "frvp_vol_concentration_top_pct"] = prior_rth_profile.vol_concentration_top_pct
            out.at[index, "frvp_vol_concentration_bot_pct"] = prior_rth_profile.vol_concentration_bot_pct
            out.at[index, "frvp_poc_vol_pct"] = prior_rth_profile.poc_vol_pct
            out.at[index, "frvp_hvn_count"] = len(prior_rth_profile.hvn_levels)
            out.at[index, "frvp_lvn_count"] = len(prior_rth_profile.lvn_levels)
            out.at[index, "frvp_hvn_count_1atr"] = _count_levels_within(prior_rth_profile.hvn_levels, current_price, atr, 1.0)
            out.at[index, "frvp_lvn_count_1atr"] = _count_levels_within(prior_rth_profile.lvn_levels, current_price, atr, 1.0)
            out.at[index, "frvp_hvn_above_close"] = _distance_above(prior_rth_profile.hvn_levels, current_price, atr)
            out.at[index, "frvp_hvn_below_close"] = _distance_below(prior_rth_profile.hvn_levels, current_price, atr)
            out.at[index, "frvp_dist_nearest_hvn_atr"] = _distance_nearest(prior_rth_profile.hvn_levels, current_price, atr)
            out.at[index, "frvp_dist_nearest_lvn_atr"] = _distance_nearest(prior_rth_profile.lvn_levels, current_price, atr)
            out.at[index, "frvp_dist_lvn_above_atr"] = _distance_above(prior_rth_profile.lvn_levels, current_price, atr)
            out.at[index, "frvp_dist_lvn_below_atr"] = _distance_below(prior_rth_profile.lvn_levels, current_price, atr)
            out.at[index, "frvp_va_width_atr"] = _safe_ratio(prior_rth_profile.vah - prior_rth_profile.val, atr)
            out.at[index, "frvp_va_width_zscore_20"] = history_metrics["width_atr_zscore_20"]
            out.at[index, "frvp_poc_migration_vel"] = _safe_ratio(
                prior_rth_profile.poc - history_metrics["previous_poc"],
                atr,
            )
            out.at[index, "frvp_va_migration_vel"] = _safe_ratio(
                ((prior_rth_profile.vah + prior_rth_profile.val) / 2.0) - history_metrics["previous_va_mid"],
                atr,
            )
            out.at[index, "frvp_va_expansion"] = int(
                pd.notna(history_metrics["previous_width_median"])
                and (prior_rth_profile.vah - prior_rth_profile.val) > float(history_metrics["previous_width_median"])
            )
            if int(history_metrics["day_type"]) > 0:
                out.at[index, "frvp_day_type"] = int(history_metrics["day_type"])
            out.at[index, "frvp_naked_poc_dist_session_atr"] = _distance_to_active_anchor_vpoc(
                active_vpocs,
                anchor_name="prior_rth",
                current_price=current_price,
                atr=atr,
            )

        if overnight_profile is not None:
            out.at[index, "frvp_dist_poc_day_atr"] = _safe_ratio(current_price - overnight_profile.poc, atr)
            out.at[index, "frvp_naked_poc_dist_day_atr"] = _distance_to_active_anchor_vpoc(
                active_vpocs,
                anchor_name="overnight_eth",
                current_price=current_price,
                atr=atr,
            )

        if swing_profile is not None:
            out.at[index, "frvp_dist_poc_swing_atr"] = _safe_ratio(current_price - swing_profile.poc, atr)

        if prior_rth_profile is not None and overnight_profile is not None:
            va_overlap = _interval_overlap_fraction(
                prior_rth_profile.val,
                prior_rth_profile.vah,
                overnight_profile.val,
                overnight_profile.vah,
            )
            out.at[index, "frvp_poc_anchor_diff_atr"] = _safe_ratio(prior_rth_profile.poc - overnight_profile.poc, atr)
            out.at[index, "frvp_va_overlap_pct"] = va_overlap
            out.at[index, "frvp_rth_eth_value_overlap"] = va_overlap

        if prior_rth_profile is not None and swing_profile is not None and overnight_profile is not None:
            out.at[index, "frvp_swing_poc_vs_session_poc"] = _safe_ratio(swing_profile.poc - prior_rth_profile.poc, atr)
            out.at[index, "frvp_multi_poc_cluster"] = int(
                _clustered_levels(
                    [prior_rth_profile.poc, overnight_profile.poc, swing_profile.poc],
                    atr=atr,
                    tolerance_atr=0.5,
                )
            )

        session_context = session_open_context.get(session_date, {})
        if session_prior_rth_profile is not None and session_context:
            rth_open_ts = pd.Timestamp(session_context["rth_open_ts"])
            if current_ts >= rth_open_ts:
                rth_open = float(session_context["open_price"])
                prior_rth_close = float(session_prior_rth_profile.source_bars["close"].iloc[-1])
                out.at[index, "frvp_open_type"] = _classify_open_type(
                    open_price=rth_open,
                    value_low=session_prior_rth_profile.val,
                    value_high=session_prior_rth_profile.vah,
                )
                out.at[index, "frvp_open_vs_prior_poc_atr"] = _safe_ratio(rth_open - session_prior_rth_profile.poc, atr)
                out.at[index, "frvp_open_gap_atr"] = _safe_ratio(rth_open - prior_rth_close, atr)
                out.at[index, "frvp_gap_into_value"] = int(
                    session_prior_rth_profile.val <= rth_open <= session_prior_rth_profile.vah
                )

        if row["is_rth"]:
            phase_code = _session_phase_code(float(row["minutes_since_rth_open"])) if pd.notna(row["minutes_since_rth_open"]) else pd.NA
            if phase_code is not pd.NA:
                out.at[index, "frvp_session_phase"] = int(phase_code)

            if session_context and pd.notna(session_context.get("open_drive_flag")):
                out.at[index, "frvp_open_drive_flag"] = int(session_context["open_drive_flag"])

            if initial_balance is not None and profile_bundle["initial_balance"] is not None and current_ts >= pd.Timestamp(row["ib_end"]):
                ib_profile = profile_bundle["initial_balance"]
                ib_high = float(session_context["ib_high"])
                ib_low = float(session_context["ib_low"])
                del ib_profile
                out.at[index, "frvp_dist_ib_high_atr"] = _safe_ratio(ib_high - current_price, atr)
                out.at[index, "frvp_dist_ib_low_atr"] = _safe_ratio(current_price - ib_low, atr)
                out.at[index, "frvp_ib_extension"] = _ib_extension(
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=current_price,
                    ib_high=ib_high,
                    ib_low=ib_low,
                )

        vpoc_above = _distance_active_vpoc_above(active_vpocs, current_price, atr)
        vpoc_below = _distance_active_vpoc_below(active_vpocs, current_price, atr)
        nearest_active = _nearest_active_vpoc(
            active_vpocs,
            current_price=current_price,
        )
        out.at[index, "frvp_naked_vpoc_dist_above_atr"] = vpoc_above
        out.at[index, "frvp_naked_vpoc_dist_below_atr"] = vpoc_below
        out.at[index, "frvp_naked_vpoc_count"] = _count_active_vpocs(active_vpocs, current_price, atr, distance_atr=3.0)
        if nearest_active is not None:
            formed_session_index = prepared.session_index_by_date.get(pd.Timestamp(nearest_active.session_date), np.nan)
            current_session_index = prepared.session_index_by_date.get(session_date, np.nan)
            if pd.notna(formed_session_index) and pd.notna(current_session_index):
                out.at[index, "frvp_naked_vpoc_age_sessions"] = float(current_session_index - formed_session_index)

        out.at[index, "frvp_poc_plus_ict_fvg_confluence"] = int(
            _within_abs(out.at[index, "frvp_dist_poc_session_atr"], 0.5)
            and (
                _within_abs(market.at[index, "dist_to_bull_fvg_atr"], 0.5)
                or _within_abs(market.at[index, "dist_to_bear_fvg_atr"], 0.5)
            )
        )
        out.at[index, "frvp_vah_plus_ob_confluence"] = int(
            _within_abs(out.at[index, "frvp_dist_vah_atr"], 0.3)
            and _within_abs(market.at[index, "dist_to_bear_order_block_atr"], 0.5)
        )
        out.at[index, "frvp_val_plus_ob_confluence"] = int(
            _within_abs(out.at[index, "frvp_dist_val_atr"], 0.3)
            and _within_abs(market.at[index, "dist_to_bull_order_block_atr"], 0.5)
        )
        out.at[index, "frvp_in_killzone"] = int(
            bool(market.at[index, "in_london_killzone"]) or bool(market.at[index, "in_newyork_killzone"])
        )

        if composite_profile is not None:
            del composite_profile

    setup_input = pd.concat(
        [
            market.reset_index(drop=True),
            out.reset_index(drop=True),
        ],
        axis=1,
    )
    setup_output = detect_frvp_setups(setup_input)
    out["frvp_setup_type"] = pd.Series(setup_output["setup_type"].to_numpy(), index=out.index, dtype="Int64")
    out["frvp_setup_side"] = pd.Series(setup_output["setup_side"].to_numpy(), index=out.index, dtype="Int64")
    out["frvp_setup_confidence_rule"] = setup_output["confidence"].to_numpy(dtype=float, copy=False)
    failed_auction_with_sweep = (
        setup_output["setup_type"].eq(4)
        & (
            pd.to_numeric(market["bars_since_sweep_high"], errors="coerce").le(3)
            | pd.to_numeric(market["bars_since_sweep_low"], errors="coerce").le(3)
        )
    ).astype(int)
    out["frvp_failed_auction_with_sweep"] = pd.Series(
        failed_auction_with_sweep.to_numpy(),
        index=out.index,
        dtype="Int64",
    )

    return out


def dump_sampled_profile_diagnostics(
    df: pd.DataFrame,
    *,
    output_path: str | Path,
    config: FeatureBuilderConfig | None = None,
    samples: int = 10,
    instrument: str | None = None,
) -> Path:
    """Dump sampled completed-session profile levels for manual TradingView checks."""

    active_config = config or FeatureBuilderConfig()
    prepared = _prepare_frvp_context(df, active_config, instrument=instrument)
    anchor_engine = prepared.anchor_engine
    sessions = anchor_engine.rth_sessions
    if sessions.empty:
        raise ValueError("No completed RTH sessions are available for diagnostics.")

    sample_count = max(1, min(int(samples), len(sessions)))
    sample_positions = sorted(set(np.linspace(0, len(sessions) - 1, num=sample_count, dtype=int).tolist()))
    profile_cache: dict[tuple[str, str, pd.Timestamp, pd.Timestamp], Any] = {}
    rows: list[dict[str, object]] = []
    for position in sample_positions:
        window = _resolve_segment_window(
            anchor_engine=anchor_engine,
            cache={},
            anchor_name="prior_rth",
            summary_frame=sessions,
            summary_index=int(position),
        )
        profile = _resolve_profile(prepared.profile_builder, profile_cache, window)
        if window is None or profile is None:
            continue
        rows.append(
            {
                "anchor_name": "prior_rth",
                "session_date": pd.Timestamp(window.metadata["session_date"]).isoformat(),
                "contract_id": profile.contract_id,
                "start": profile.start.isoformat(),
                "end": profile.end.isoformat(),
                "poc": profile.poc,
                "vah": profile.vah,
                "val": profile.val,
                "hvn_levels": "|".join(f"{value:g}" for value in profile.hvn_levels),
                "lvn_levels": "|".join(f"{value:g}" for value in profile.lvn_levels),
                "shape": profile.shape,
                "total_volume": profile.total_volume,
                "bar_count": len(profile.source_bars),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _prepare_frvp_context(
    df: pd.DataFrame,
    config: FeatureBuilderConfig,
    *,
    instrument: str | None = None,
) -> _PreparedFRVPContext:
    from ..continuity.continuous_contract import build_continuous_contract_from_tagged_series
    from ..profiles.anchors import FRVPAnchorEngine
    from ..profiles.builder import VolumeProfileBuilder

    working = df.copy()
    if "datetime" not in working.columns:
        if "ts_event" in working.columns:
            working["datetime"] = working["ts_event"]
        elif "timestamp" in working.columns:
            working["datetime"] = working["timestamp"]
    working = standardize_market_frame(
        working,
        source_timezone=config.source_timezone,
        canonical_timezone=config.canonical_timezone,
    )
    if "datetime" not in working.columns:
        raise KeyError("FRVP context requires a datetime, timestamp, or ts_event column.")

    inferred_instrument = instrument or _infer_instrument(working)
    instrument_config = get_instrument_config(inferred_instrument)
    tagged = _prepare_tagged_continuous_frame(
        working,
        instrument_config=instrument_config,
        canonical_timezone=config.canonical_timezone,
    )
    continuous = build_continuous_contract_from_tagged_series(
        tagged,
        timestamp_col="ts_event",
        contract_col="contract_id",
        volume_col="volume",
        source_timezone=config.canonical_timezone,
        canonical_timezone=config.canonical_timezone,
        market_close_timezone=instrument_config.market_close_timezone,
        market_close_hour=instrument_config.market_close_hour,
        market_close_minute=instrument_config.market_close_minute,
        roll_bracket_sessions=ROLL_BRACKET_SESSIONS,
    )
    anchor_engine = FRVPAnchorEngine(
        continuous.raw_profile_bars,
        instrument=instrument_config,
        source_timezone=config.canonical_timezone,
        canonical_timezone=config.canonical_timezone,
    )
    profile_builder = VolumeProfileBuilder(instrument=instrument_config)

    session_frame = build_equity_session_frame(
        working["datetime"],
        instrument=instrument_config,
        source_timezone=config.canonical_timezone,
        canonical_timezone=config.canonical_timezone,
    ).drop(columns=["datetime_utc"])
    support = pd.concat([working.reset_index(drop=True), session_frame.reset_index(drop=True)], axis=1)
    macro_flags = annotate_us_macro_event_flags(
        working["datetime"],
        source_timezone=config.canonical_timezone,
        canonical_timezone=config.canonical_timezone,
    )
    support = _merge_missing_columns(support, macro_flags.reset_index(drop=True))
    support["timestamp"] = support["datetime"]
    if "market_day_close" not in support.columns:
        support["market_day_close"] = build_equity_market_day_labels(
            support["timestamp"],
            source_timezone=config.canonical_timezone,
            canonical_timezone=config.canonical_timezone,
            market_close_timezone=instrument_config.market_close_timezone,
            market_close_hour=instrument_config.market_close_hour,
            market_close_minute=instrument_config.market_close_minute,
        )
    if "market_day_index" not in support.columns:
        session_codes, _ = pd.factorize(support["market_day_close"], sort=True)
        support["market_day_index"] = session_codes.astype(int)

    volatility = build_volatility(support, config)
    support = _merge_missing_columns(support, volatility)
    volume = build_volume(support, config)
    support = _merge_missing_columns(support, volume)
    structure = build_structure(support, config)
    support = _merge_missing_columns(support, structure)
    session_block = build_session(support, config)
    support = _merge_missing_columns(support, session_block)

    default_optional = {
        "dist_to_bull_fvg_atr": np.nan,
        "dist_to_bear_fvg_atr": np.nan,
        "dist_to_bull_order_block_atr": np.nan,
        "dist_to_bear_order_block_atr": np.nan,
        "in_london_killzone": 0,
        "in_newyork_killzone": 0,
    }
    for column, fill_value in default_optional.items():
        if column not in support.columns:
            support[column] = fill_value

    merge_columns = [
        "timestamp",
        "datetime",
        "session_date",
        "is_rth",
        "is_overnight",
        "is_ib",
        "ib_complete",
        "rth_start",
        "rth_end",
        "overnight_start",
        "overnight_end",
        "ib_start",
        "ib_end",
        "minutes_since_rth_open",
        "minutes_until_rth_close",
        "equity_holiday_flag",
        "equity_half_day_flag",
        "equity_early_close_flag",
        "market_day_close",
        "market_day_index",
        "atr_14",
        "atr_50",
        "volume_zscore_50",
        "displacement_bullish",
        "displacement_bearish",
        "bars_since_sweep_high",
        "bars_since_sweep_low",
        "dist_to_bull_fvg_atr",
        "dist_to_bear_fvg_atr",
        "dist_to_bull_order_block_atr",
        "dist_to_bear_order_block_atr",
        "in_london_killzone",
        "in_newyork_killzone",
        "cpi_flag",
        "nfp_flag",
        "fomc_flag",
        "fed_statement_flag",
        "fed_presser_flag",
        "macro_event_flag",
        "major_macro_event_flag",
    ]
    support_merge = support.loc[:, [column for column in merge_columns if column in support.columns]].copy()
    market = continuous.raw_profile_bars.bars.merge(
        support_merge,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    if market["timestamp"].isna().any():
        raise ValueError("FRVP context failed to align support features to raw profile bars.")

    session_index_by_date = (
        market.groupby("session_date", sort=True)["market_day_index"].first().to_dict()
        if "session_date" in market.columns and "market_day_index" in market.columns
        else {}
    )
    return _PreparedFRVPContext(
        instrument=instrument_config.instrument,
        instrument_config=instrument_config,
        market=market.reset_index(drop=True),
        tagged=tagged.reset_index(drop=True),
        continuous=continuous,
        anchor_engine=anchor_engine,
        profile_builder=profile_builder,
        session_index_by_date={pd.Timestamp(key): int(value) for key, value in session_index_by_date.items()},
    )


def _prepare_tagged_continuous_frame(
    working: pd.DataFrame,
    *,
    instrument_config: InstrumentConfig,
    canonical_timezone: str,
) -> pd.DataFrame:
    from ..continuity.reconstruct_boundaries import tag_bars

    source = working.copy().reset_index(drop=True)
    source["ts_event"] = normalize_datetime_series(
        source["datetime"],
        source_timezone=canonical_timezone,
        canonical_timezone=canonical_timezone,
    )
    if "symbol" not in source.columns:
        source["symbol"] = instrument_config.instrument.upper()

    schedule_path = SCHEDULE_PATHS.get(instrument_config.instrument)
    if "contract_symbol" not in source.columns and "instrument_id" in source.columns and schedule_path is not None and schedule_path.exists():
        schedule = _load_schedule_dataframe(schedule_path)
        tagged = tag_bars(
            source.loc[:, ["ts_event", "open", "high", "low", "close", "volume", "symbol", "instrument_id"]].copy(),
            schedule,
            roll_bracket_sessions=ROLL_BRACKET_SESSIONS,
        )
        tagged["contract_id"] = tagged["contract_symbol"].astype(str)
        return tagged

    tagged = source.copy()
    if "contract_symbol" in tagged.columns:
        tagged["contract_id"] = tagged["contract_symbol"].astype(str)
    elif "contract_id" in tagged.columns:
        tagged["contract_id"] = tagged["contract_id"].astype(str)
    elif "instrument_id" in tagged.columns:
        tagged["contract_id"] = tagged["instrument_id"].astype(str)
    else:
        tagged["contract_id"] = "single_contract"

    tagged["market_day_close"] = build_equity_market_day_labels(
        tagged["ts_event"],
        source_timezone=canonical_timezone,
        canonical_timezone=canonical_timezone,
        market_close_timezone=instrument_config.market_close_timezone,
        market_close_hour=instrument_config.market_close_hour,
        market_close_minute=instrument_config.market_close_minute,
    )
    market_day_codes, _ = pd.factorize(tagged["market_day_close"], sort=True)
    tagged["market_day_index"] = market_day_codes.astype(int)
    tagged["is_roll_boundary"] = tagged["contract_id"].ne(tagged["contract_id"].shift()).fillna(False)
    if not tagged.empty:
        tagged.loc[0, "is_roll_boundary"] = False
    roll_sessions = sorted(set(tagged.loc[tagged["is_roll_boundary"], "market_day_index"].astype(int).tolist()))
    if roll_sessions:
        distances = [
            min(abs(int(session_index) - roll_session) for roll_session in roll_sessions)
            for session_index in tagged["market_day_index"].tolist()
        ]
        tagged["in_roll_bracket"] = pd.Series(distances, index=tagged.index).le(ROLL_BRACKET_SESSIONS)
    else:
        tagged["in_roll_bracket"] = False
    return tagged


def _merge_missing_columns(base: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for column in additions.columns:
        if column in out.columns:
            continue
        out[column] = additions[column].to_numpy()
    return out


def _infer_instrument(df: pd.DataFrame) -> str:
    for column in ("symbol", "contract_symbol", "contract_id"):
        if column not in df.columns:
            continue
        values = df[column].dropna().astype(str)
        if values.empty:
            continue
        sample = values.iloc[0].upper()
        if sample.startswith("6E"):
            return "6e"
        if sample.startswith("ES"):
            return "es"
    return "es"


def _load_schedule_dataframe(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return schedule
    schedule["instrument_id"] = pd.to_numeric(schedule["instrument_id"], errors="raise").astype("Int64")
    schedule["raw_symbol"] = schedule["raw_symbol"].astype(str)
    schedule["expiration"] = pd.to_datetime(schedule["expiration"], utc=True, errors="raise")
    schedule["start"] = pd.to_datetime(schedule["start"], utc=True, errors="raise")
    schedule["end"] = pd.to_datetime(schedule["end"], utc=True, errors="raise")
    return schedule


def _resolve_segment_window(
    *,
    anchor_engine,
    cache: dict[tuple[str, object], Any],
    anchor_name: str,
    summary_frame: pd.DataFrame,
    summary_index: int,
) -> Any:
    key = (anchor_name, summary_index)
    if key in cache:
        return cache[key]
    if summary_index < 0 or summary_frame.empty or summary_index >= len(summary_frame):
        cache[key] = None
        return None
    row = summary_frame.iloc[int(summary_index)]
    cache[key] = anchor_engine._build_time_window(
        anchor_name=anchor_name,
        start=row["start"],
        end=row["end"],
        completed_at=row["end"],
        metadata={"session_date": row["session_date"]},
    )
    return cache[key]


def _resolve_initial_balance_window(
    *,
    anchor_engine,
    cache: dict[tuple[str, object], Any],
    row: pd.Series,
) -> Any:
    if not bool(row["is_rth"]) or pd.isna(row["ib_end"]) or pd.Timestamp(row["timestamp"]) < pd.Timestamp(row["ib_end"]):
        return None
    session_date = pd.Timestamp(row["session_date"])
    key = ("initial_balance", session_date)
    if key in cache:
        return cache[key]
    cache[key] = anchor_engine._build_time_window(
        anchor_name="initial_balance",
        start=row["ib_start"],
        end=row["ib_end"],
        completed_at=row["ib_end"],
        metadata={"session_date": session_date},
    )
    return cache[key]


def _resolve_swing_window(
    *,
    anchor_engine,
    cache: dict[tuple[str, object], Any],
    current_ts: pd.Timestamp,
) -> Any:
    from ..continuity.types import ProfileSlice
    from ..profiles.anchors import AnchorWindow

    current_position = np.searchsorted(anchor_engine._timestamp_ns, pd.Timestamp(current_ts).value, side="left")
    cutoff = np.searchsorted(np.asarray(anchor_engine._swing_confirmed_indices, dtype=int), current_position, side="left")
    key = ("swing_to_swing", int(cutoff))
    if key in cache:
        return cache[key]
    if cutoff < 2:
        cache[key] = None
        return None
    previous_event = anchor_engine._swing_events[cutoff - 2]
    current_event = anchor_engine._swing_events[cutoff - 1]
    start_index = int(previous_event["swing_index"])
    end_index = int(current_event["swing_index"])
    if end_index <= start_index:
        cache[key] = None
        return None
    window = anchor_engine._bars.iloc[start_index : end_index + 1].copy()
    if window.empty or window["contract_id"].astype(str).nunique() != 1:
        cache[key] = None
        return None
    cache[key] = AnchorWindow(
        anchor_name="swing_to_swing",
        profile_slice=ProfileSlice(
            contract_id=str(window["contract_id"].iloc[0]),
            start=pd.Timestamp(window["timestamp"].iloc[0]),
            end=pd.Timestamp(window["timestamp"].iloc[-1]),
            bars=window.reset_index(drop=True),
        ),
        completed_at=current_ts,
        metadata={
            "start_swing_kind": previous_event["swing_kind"],
            "start_swing_level": previous_event["swing_level"],
            "end_swing_kind": current_event["swing_kind"],
            "end_swing_level": current_event["swing_level"],
        },
    )
    return cache[key]


def _resolve_composite_window(
    *,
    anchor_engine,
    cache: dict[tuple[str, object], Any],
    prior_rth_summary_index: int,
) -> Any:
    from ..continuity.types import ProfileSlice
    from ..profiles.anchors import AnchorWindow
    key = ("rolling_composite", int(prior_rth_summary_index))
    if key in cache:
        return cache[key]
    if prior_rth_summary_index < 0 or anchor_engine.rth_sessions.empty:
        cache[key] = None
        return None

    completed = anchor_engine.rth_sessions.iloc[: prior_rth_summary_index + 1].copy()
    selected_dates: list[pd.Timestamp] = []
    contract_id: str | None = None
    for row in reversed(list(completed.itertuples(index=False))):
        if row.contract_count != 1 or row.contract_id is None:
            if contract_id is not None:
                break
            continue
        if contract_id is None:
            contract_id = str(row.contract_id)
        if str(row.contract_id) != contract_id:
            break
        selected_dates.append(pd.Timestamp(row.session_date))
        if len(selected_dates) >= anchor_engine.composite_sessions:
            break

    if not selected_dates:
        cache[key] = None
        return None
    window = anchor_engine._bars.loc[
        anchor_engine._bars["is_rth"] & anchor_engine._bars["session_date"].isin(selected_dates)
    ].copy()
    if window.empty or window["contract_id"].astype(str).nunique() != 1:
        cache[key] = None
        return None
    cache[key] = AnchorWindow(
        anchor_name="rolling_composite",
        profile_slice=ProfileSlice(
            contract_id=str(window["contract_id"].iloc[0]),
            start=pd.Timestamp(window["timestamp"].iloc[0]),
            end=pd.Timestamp(window["timestamp"].iloc[-1]),
            bars=window.reset_index(drop=True),
        ),
        completed_at=pd.Timestamp(completed.iloc[-1]["end"]),
        metadata={"session_dates": tuple(reversed(selected_dates))},
    )
    return cache[key]


def _resolve_profile(
    profile_builder,
    cache: dict[tuple[str, str, pd.Timestamp, pd.Timestamp], Any],
    anchor_window,
) -> Any:
    if anchor_window is None:
        return None
    key = (
        anchor_window.anchor_name,
        anchor_window.contract_id,
        pd.Timestamp(anchor_window.start),
        pd.Timestamp(anchor_window.end),
    )
    if key not in cache:
        cache[key] = profile_builder.build(anchor_window.profile_slice)
    return cache[key]


def _ensure_prior_rth_history_metrics(
    *,
    history: list[dict[str, float | pd.Timestamp]],
    history_by_session: dict[pd.Timestamp, dict[str, float | pd.Timestamp]],
    session_index_by_date: dict[pd.Timestamp, int],
    market: pd.DataFrame,
    anchor_window,
    profile,
    ib_minutes: int,
) -> dict[str, float]:
    session_date = pd.Timestamp(anchor_window.metadata["session_date"])
    if session_date in history_by_session:
        metrics = history_by_session[session_date]
        return {
            "width_atr_zscore_20": float(metrics["width_atr_zscore_20"]),
            "previous_poc": float(metrics["previous_poc"]),
            "previous_va_mid": float(metrics["previous_va_mid"]),
            "previous_width_median": float(metrics["previous_width_median"]) if pd.notna(metrics["previous_width_median"]) else np.nan,
            "day_type": float(metrics["day_type"]),
        }

    profile_end = pd.Timestamp(profile.source_bars["timestamp"].iloc[-1])
    atr_match = market.loc[market["timestamp"] == profile_end, "atr_14"]
    reference_atr = float(atr_match.iloc[-1]) if not atr_match.empty and pd.notna(atr_match.iloc[-1]) and float(atr_match.iloc[-1]) != 0.0 else np.nan
    width_points = float(profile.vah - profile.val)
    width_atr_reference = _safe_ratio(width_points, reference_atr)
    previous_entries = history[-20:]
    previous_width_atr_values = [float(item["width_atr_reference"]) for item in previous_entries if pd.notna(item["width_atr_reference"])]
    previous_width_points = [float(item["width_points"]) for item in previous_entries]
    width_atr_zscore_20 = _zscore_against_history(width_atr_reference, previous_width_atr_values)
    previous_poc = float(previous_entries[-1]["poc"]) if previous_entries else np.nan
    previous_va_mid = float(previous_entries[-1]["va_mid"]) if previous_entries else np.nan
    previous_width_median = float(np.median(previous_width_points)) if previous_width_points else np.nan
    day_type = _classify_prior_day_type(profile=profile, ib_minutes=ib_minutes)

    metrics = {
        "session_date": session_date,
        "session_index": float(session_index_by_date.get(session_date, np.nan)),
        "poc": float(profile.poc),
        "va_mid": float((profile.vah + profile.val) / 2.0),
        "width_points": width_points,
        "width_atr_reference": width_atr_reference,
        "width_atr_zscore_20": width_atr_zscore_20,
        "previous_poc": previous_poc,
        "previous_va_mid": previous_va_mid,
        "previous_width_median": previous_width_median,
        "day_type": float(day_type),
    }
    history.append(metrics)
    history_by_session[session_date] = metrics
    return {
        "width_atr_zscore_20": float(width_atr_zscore_20),
        "previous_poc": float(previous_poc),
        "previous_va_mid": float(previous_va_mid),
        "previous_width_median": float(previous_width_median) if pd.notna(previous_width_median) else np.nan,
        "day_type": float(day_type),
    }


def _classify_prior_day_type(
    *,
    profile,
    ib_minutes: int,
) -> int:
    session_bars = profile.source_bars.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if session_bars.empty:
        return 0

    highs = pd.to_numeric(session_bars["high"], errors="coerce")
    lows = pd.to_numeric(session_bars["low"], errors="coerce")
    closes = pd.to_numeric(session_bars["close"], errors="coerce")
    timestamps = pd.to_datetime(session_bars["timestamp"], errors="coerce", utc=True)
    if highs.isna().all() or lows.isna().all() or closes.isna().all() or timestamps.isna().all():
        return 0

    session_high = float(highs.max())
    session_low = float(lows.min())
    session_close = float(closes.iloc[-1])
    session_range = session_high - session_low
    if session_range <= 0.0:
        return 0

    session_open = float(pd.to_numeric(session_bars["open"], errors="coerce").iloc[0])
    start = pd.Timestamp(timestamps.iloc[0])
    ib_cutoff = start + pd.Timedelta(minutes=int(ib_minutes))
    ib_bars = session_bars.loc[pd.to_datetime(session_bars["timestamp"], errors="coerce", utc=True) < ib_cutoff].copy()
    if ib_bars.empty:
        ib_bars = session_bars.iloc[: max(int(round(ib_minutes / 5.0)), 1)].copy()

    ib_high = float(pd.to_numeric(ib_bars["high"], errors="coerce").max())
    ib_low = float(pd.to_numeric(ib_bars["low"], errors="coerce").min())
    ib_range = ib_high - ib_low
    broke_above = session_high > ib_high
    broke_below = session_low < ib_low
    close_position = (session_close - session_low) / session_range
    close_above_value = session_close > float(profile.vah)
    close_below_value = session_close < float(profile.val)
    extended_range = session_range >= (1.25 * max(ib_range, 1e-8))
    trend_range = session_range >= (1.80 * max(ib_range, 1e-8))
    one_time_up = session_close > session_open and close_position >= 0.75 and not broke_below
    one_time_down = session_close < session_open and close_position <= 0.25 and not broke_above
    double_distribution = (
        trend_range
        and ((profile.shape == "P" and one_time_up) or (profile.shape == "b" and one_time_down))
        and (close_above_value or close_below_value)
    )

    if double_distribution:
        return DAY_TYPE_DOUBLE_DISTRIBUTION_TREND
    if trend_range and ((close_above_value and one_time_up) or (close_below_value and one_time_down)):
        return DAY_TYPE_TREND
    if broke_above and broke_below:
        return DAY_TYPE_NEUTRAL
    if extended_range and (broke_above or broke_below):
        return DAY_TYPE_NORMAL_VARIATION
    return DAY_TYPE_NORMAL


def _build_session_open_context(market: pd.DataFrame) -> dict[pd.Timestamp, dict[str, object]]:
    context: dict[pd.Timestamp, dict[str, object]] = {}
    rth = market.loc[market["is_rth"]].copy()
    for session_date, frame in rth.groupby("session_date", sort=True):
        group = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if group.empty:
            continue
        open_price = float(group["open"].iloc[0])
        first_30 = group.loc[group["minutes_since_rth_open"].lt(30)].copy()
        first_60 = group.loc[group["is_ib"]].copy()
        open_drive_flag = pd.NA
        if len(first_30) >= 6:
            bullish = (
                float(first_30["low"].min()) >= open_price
                and float(first_30["close"].iloc[-1]) > open_price
                and first_30["close"].diff().fillna(0.0).ge(0.0).all()
            )
            bearish = (
                float(first_30["high"].max()) <= open_price
                and float(first_30["close"].iloc[-1]) < open_price
                and first_30["close"].diff().fillna(0.0).le(0.0).all()
            )
            open_drive_flag = int(bullish or bearish)
        context[pd.Timestamp(session_date)] = {
            "rth_open_ts": pd.Timestamp(group["timestamp"].iloc[0]),
            "open_price": open_price,
            "open_drive_flag": open_drive_flag,
            "ib_high": float(first_60["high"].max()) if not first_60.empty else np.nan,
            "ib_low": float(first_60["low"].min()) if not first_60.empty else np.nan,
        }
    return context


def _build_session_prior_rth_summary_indices(rth_sessions: pd.DataFrame) -> dict[pd.Timestamp, int]:
    if rth_sessions.empty:
        return {}
    return {
        pd.Timestamp(session_date): (index - 1)
        for index, session_date in enumerate(rth_sessions["session_date"].tolist())
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator is None or not np.isfinite(denominator) or float(denominator) == 0.0:
        return np.nan
    return float(numerator) / float(denominator)


def _price_position_in_range(price: float, low: float, high: float) -> float:
    if pd.isna(low) or pd.isna(high) or high == low:
        return 0.5
    return float(price - low) / float(high - low)


def _distance_nearest(levels: tuple[float, ...], current_price: float, atr: float) -> float:
    if not levels:
        return np.nan
    nearest = min(levels, key=lambda level: abs(float(level) - current_price))
    return _safe_ratio(abs(float(nearest) - current_price), atr)


def _distance_above(levels: tuple[float, ...], current_price: float, atr: float) -> float:
    above = [float(level) for level in levels if float(level) > current_price]
    if not above:
        return np.nan
    return _safe_ratio(min(above) - current_price, atr)


def _distance_below(levels: tuple[float, ...], current_price: float, atr: float) -> float:
    below = [float(level) for level in levels if float(level) < current_price]
    if not below:
        return np.nan
    return _safe_ratio(current_price - max(below), atr)


def _count_levels_within(levels: tuple[float, ...], current_price: float, atr: float, distance_atr: float) -> int:
    if not levels or not np.isfinite(atr) or atr <= 0.0:
        return 0
    threshold = float(distance_atr) * float(atr)
    return int(sum(abs(float(level) - current_price) <= threshold for level in levels))


def _interval_overlap_fraction(low_a: float, high_a: float, low_b: float, high_b: float) -> float:
    start = max(float(low_a), float(low_b))
    end = min(float(high_a), float(high_b))
    overlap = max(end - start, 0.0)
    union = max(float(high_a), float(high_b)) - min(float(low_a), float(low_b))
    if union <= 0.0:
        return 1.0 if overlap == 0.0 and low_a == low_b and high_a == high_b else 0.0
    return overlap / union


def _clustered_levels(levels: list[float], *, atr: float, tolerance_atr: float) -> bool:
    valid = [float(level) for level in levels if pd.notna(level)]
    if len(valid) < len(levels) or not np.isfinite(atr) or atr <= 0.0:
        return False
    return (max(valid) - min(valid)) <= (float(tolerance_atr) * float(atr))


def _zscore_against_history(value: float, history: list[float]) -> float:
    if not np.isfinite(value) or len(history) < 2:
        return np.nan
    mean = float(np.mean(history))
    std = float(np.std(history, ddof=0))
    if std == 0.0:
        return np.nan
    return (float(value) - mean) / std


def _va_overshoot(price: float, value_low: float, value_high: float, atr: float) -> float:
    if price > value_high:
        return _safe_ratio(price - value_high, atr)
    if price < value_low:
        return -_safe_ratio(value_low - price, atr)
    return 0.0


def _distance_to_active_anchor_vpoc(
    active_levels: tuple[Any, ...],
    *,
    anchor_name: str,
    current_price: float,
    atr: float,
) -> float:
    candidates = [level for level in active_levels if level.anchor_name == anchor_name]
    if not candidates:
        return np.nan
    nearest = min(candidates, key=lambda level: abs(level.price - current_price))
    return _safe_ratio(current_price - nearest.price, atr)


def _classify_open_type(*, open_price: float, value_low: float, value_high: float) -> int:
    if value_low <= open_price <= value_high:
        return 0
    return 1 if open_price > value_high else -1


def _session_phase_code(minutes_since_open: float) -> int:
    if minutes_since_open < 30:
        return SESSION_PHASE_OPEN
    if minutes_since_open < 120:
        return SESSION_PHASE_MORNING
    if minutes_since_open < 240:
        return SESSION_PHASE_MIDDAY
    if minutes_since_open < 330:
        return SESSION_PHASE_AFTERNOON
    return SESSION_PHASE_CLOSE_RAMP


def _ib_extension(
    *,
    high: float,
    low: float,
    close: float,
    ib_high: float,
    ib_low: float,
) -> int:
    broke_high = high > ib_high
    broke_low = low < ib_low
    if broke_high and not broke_low:
        return 1
    if broke_low and not broke_high:
        return -1
    if broke_high and broke_low:
        midpoint = (ib_high + ib_low) / 2.0
        return 1 if close >= midpoint else -1
    return 0


def _distance_active_vpoc_above(
    active_levels: tuple[Any, ...],
    current_price: float,
    atr: float,
) -> float:
    above = [level.price for level in active_levels if level.price > current_price]
    if not above:
        return np.nan
    return _safe_ratio(min(above) - current_price, atr)


def _distance_active_vpoc_below(
    active_levels: tuple[Any, ...],
    current_price: float,
    atr: float,
) -> float:
    below = [level.price for level in active_levels if level.price < current_price]
    if not below:
        return np.nan
    return _safe_ratio(current_price - max(below), atr)


def _nearest_active_vpoc(
    active_levels: tuple[Any, ...],
    *,
    current_price: float,
) -> Any:
    if not active_levels:
        return None
    return min(active_levels, key=lambda level: abs(level.price - current_price))


def _count_active_vpocs(
    active_levels: tuple[Any, ...],
    current_price: float,
    atr: float,
    *,
    distance_atr: float,
) -> int:
    if not np.isfinite(atr) or atr <= 0.0:
        return 0
    threshold = float(distance_atr) * float(atr)
    return int(sum(abs(level.price - current_price) <= threshold for level in active_levels))


def _within_abs(value, threshold: float) -> bool:
    return pd.notna(value) and abs(float(value)) <= float(threshold)
