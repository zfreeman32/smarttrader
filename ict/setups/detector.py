from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..common import get_tick_size, resolve_event_time
from ..feature_sets.ict_context import build_ict_context_features
from ..sessions.equity import build_ict_equity_session_frame
from ..config.setups import ICTSetupDetectorConfig
from .setup_types import ICTSetupFamily, ICTSetupSide, ICTSetupType, build_empty_setup_frame
from .validators import validate_ict_market_frame, validate_ict_setup_output


LEVEL_CODE_TO_NAME = {
    1: "swing_high",
    2: "equal_high",
    3: "prior_rth_high",
    4: "overnight_high",
    5: "ib_high",
    6: "prior_week_high",
    11: "swing_low",
    12: "equal_low",
    13: "prior_rth_low",
    14: "overnight_low",
    15: "ib_low",
    16: "prior_week_low",
}

PRE_IB_LONG_CODES = frozenset({13, 14})
PRE_IB_SHORT_CODES = frozenset({3, 4})
POST_IB_LONG_CODES = frozenset({15})
POST_IB_SHORT_CODES = frozenset({5})

SETUP_MIN_SPACING = {
    ICTSetupType.SWEEP_RECLAIM.value: 6,
    ICTSetupType.SWEEP_DISPLACEMENT_FVG.value: 6,
    ICTSetupType.OB_RETEST_AFTER_MSS.value: 8,
    ICTSetupType.IFVG_REVERSAL.value: 8,
    ICTSetupType.PREMIUM_DISCOUNT_CONTINUATION.value: 10,
    ICTSetupType.SESSION_OPEN_MANIPULATION_PRE_IB.value: 60,
    ICTSetupType.SESSION_OPEN_MANIPULATION_POST_IB.value: 60,
    ICTSetupType.DISPLACEMENT_CONTINUATION_AFTER_RAID.value: 16,
}

SETUP_PRIORITY = {
    ICTSetupType.SESSION_OPEN_MANIPULATION_PRE_IB.value: 0,
    ICTSetupType.SESSION_OPEN_MANIPULATION_POST_IB.value: 1,
    ICTSetupType.IFVG_REVERSAL.value: 2,
    ICTSetupType.OB_RETEST_AFTER_MSS.value: 3,
    ICTSetupType.SWEEP_DISPLACEMENT_FVG.value: 4,
    ICTSetupType.DISPLACEMENT_CONTINUATION_AFTER_RAID.value: 5,
    ICTSetupType.PREMIUM_DISCOUNT_CONTINUATION.value: 6,
    ICTSetupType.SWEEP_RECLAIM.value: 7,
}


@dataclass(frozen=True)
class _SetupCandidate:
    setup_type: str
    setup_family: str
    setup_side: int
    confidence: float
    anchor_level: float = np.nan
    entry_price: float = np.nan
    stop_reference: float = np.nan
    target_reference: float = np.nan
    reference_level: float = np.nan
    reference_level_type: str = ""
    sweep_type: str = ""
    htf_context: str = ""
    fvg_id: int | None = None
    ce_price: float = np.nan
    order_block_id: int | None = None
    displacement_id: int | None = None
    displacement_volume_z: float = np.nan
    session_phase: int | None = None
    anchor_key: str = ""


def detect_ict_setups(
    df: pd.DataFrame,
    config: ICTSetupDetectorConfig | None = None,
) -> pd.DataFrame:
    """Detect causal ICT setup events from the Phase 2 detector surface."""

    validate_ict_market_frame(df)
    config = config or ICTSetupDetectorConfig()
    working = _ensure_setup_input_frame(df, config)
    event_time = resolve_event_time(working)

    out = build_empty_setup_frame(df.index, event_time=event_time)
    enabled = {str(value) for value in config.enabled_setup_types}
    previous_active: dict[str, bool] = {}
    last_fired_index: dict[str, int] = {}
    side_session_counts: dict[tuple[object, int], int] = {}
    candidate_records: list[dict[str, object]] = []
    fired_records: list[dict[str, object]] = []

    for position in range(len(working)):
        row = working.iloc[position]
        session_key = row.get("session_date")
        current_candidates: list[_SetupCandidate] = []

        if ICTSetupType.SESSION_OPEN_MANIPULATION_PRE_IB.value in enabled:
            candidate = _candidate_session_open_manipulation(row, config, pre_ib=True)
            if candidate is not None:
                current_candidates.append(candidate)
        if ICTSetupType.SESSION_OPEN_MANIPULATION_POST_IB.value in enabled:
            candidate = _candidate_session_open_manipulation(row, config, pre_ib=False)
            if candidate is not None:
                current_candidates.append(candidate)
        if ICTSetupType.IFVG_REVERSAL.value in enabled:
            candidate = _candidate_ifvg_reversal(row, config, position=position)
            if candidate is not None:
                current_candidates.append(candidate)
        if ICTSetupType.OB_RETEST_AFTER_MSS.value in enabled:
            candidate = _candidate_ob_retest_after_mss(row, config)
            if candidate is not None:
                current_candidates.append(candidate)
        if ICTSetupType.SWEEP_DISPLACEMENT_FVG.value in enabled:
            candidate = _candidate_sweep_displacement_fvg(row, config, position=position)
            if candidate is not None:
                current_candidates.append(candidate)
        if ICTSetupType.DISPLACEMENT_CONTINUATION_AFTER_RAID.value in enabled:
            candidate = _candidate_displacement_continuation_after_raid(row, config)
            if candidate is not None:
                current_candidates.append(candidate)
        if ICTSetupType.PREMIUM_DISCOUNT_CONTINUATION.value in enabled:
            candidate = _candidate_premium_discount_continuation(row, config)
            if candidate is not None:
                current_candidates.append(candidate)
        if ICTSetupType.SWEEP_RECLAIM.value in enabled:
            candidate = _candidate_sweep_reclaim(row, config)
            if candidate is not None:
                current_candidates.append(candidate)

        current_active = {candidate.anchor_key: True for candidate in current_candidates}
        eligible: list[_SetupCandidate] = []

        for candidate in current_candidates:
            active_key = candidate.anchor_key
            session_side_key = (session_key, int(candidate.setup_side))
            already_active = previous_active.get(active_key, False)
            spacing = max(1, SETUP_MIN_SPACING.get(candidate.setup_type, int(config.cooldown_bars)))
            within_spacing = (
                active_key in last_fired_index and (position - last_fired_index[active_key]) < spacing
            )
            side_session_exhausted = (
                side_session_counts.get(session_side_key, 0) >= int(config.max_same_side_fires_per_session)
            )
            is_eligible = (not already_active) and (not within_spacing) and (not side_session_exhausted)
            candidate_records.append(
                _candidate_record(
                    row=row,
                    position=position,
                    candidate=candidate,
                    session_key=session_key,
                    eligible=is_eligible,
                    selected=False,
                )
            )
            if is_eligible:
                eligible.append(candidate)

        selected = _select_candidate(eligible)
        if selected is not None:
            _write_candidate_to_output(out, position, selected)
            last_fired_index[selected.anchor_key] = position
            side_session_key = (session_key, int(selected.setup_side))
            side_session_counts[side_session_key] = side_session_counts.get(side_session_key, 0) + 1
            fired_records.append(
                _candidate_record(
                    row=row,
                    position=position,
                    candidate=selected,
                    session_key=session_key,
                    eligible=True,
                    selected=True,
                )
            )

        previous_active = current_active

    out.attrs["detector_phase"] = "phase3_setup_detector"
    out.attrs["detector_config"] = asdict(config)
    out.attrs["candidate_events"] = pd.DataFrame(candidate_records)
    out.attrs["fired_events"] = pd.DataFrame(fired_records)
    if "session_date" in working.columns:
        out.attrs["session_date"] = pd.Series(working["session_date"].to_numpy(), index=out.index)
    validate_ict_setup_output(out)
    return out


def summarize_setup_fire_rates(setup_output: pd.DataFrame) -> pd.DataFrame:
    """Summarize ICT setup fire rates per session, setup family, and side."""

    validate_ict_setup_output(setup_output)
    session_date = setup_output.attrs.get("session_date")
    if session_date is None:
        event_time = pd.to_datetime(setup_output.get("event_time"), errors="coerce")
        if isinstance(event_time, pd.Series):
            session_date = event_time.dt.tz_localize(None) if getattr(event_time.dt, "tz", None) is not None else event_time
            session_date = session_date.dt.normalize()
        else:
            session_date = pd.Series(pd.Timestamp("1970-01-01"), index=setup_output.index)
    session_series = pd.Series(session_date, index=setup_output.index)
    total_sessions = int(session_series.dropna().nunique()) if not session_series.dropna().empty else 1
    detector_config = setup_output.attrs.get("detector_config", {}) or {}
    target_min = float(detector_config.get("target_daily_trade_count_min", 1) or 1)
    target_max = float(detector_config.get("target_daily_trade_count_max", 3) or 3)

    working = setup_output.copy()
    working["session_date"] = session_series
    fired = working.loc[working["fired"].fillna(False)].copy()
    summary_columns = [
        "setup_type",
        "setup_family",
        "setup_side",
        "total_fires",
        "sessions_with_fire",
        "avg_fires_per_session_side",
        "within_target_band",
    ]
    if fired.empty:
        return pd.DataFrame(columns=summary_columns)

    per_session = (
        fired.groupby(["setup_type", "setup_family", "setup_side", "session_date"], dropna=False)
        .size()
        .reset_index(name="fires")
    )
    grouped = (
        per_session.groupby(["setup_type", "setup_family", "setup_side"], dropna=False)
        .agg(
            total_fires=("fires", "sum"),
            sessions_with_fire=("session_date", "nunique"),
        )
        .reset_index()
    )
    grouped["avg_fires_per_session_side"] = grouped["total_fires"] / float(max(total_sessions, 1))
    grouped["within_target_band"] = grouped["avg_fires_per_session_side"].between(target_min, target_max, inclusive="both")
    return grouped.loc[:, summary_columns]


def _ensure_setup_input_frame(
    df: pd.DataFrame,
    config: ICTSetupDetectorConfig,
) -> pd.DataFrame:
    working = df.reset_index(drop=True).copy()
    context_required = {
        "ict_buy_side_sweep",
        "ict_sell_side_sweep",
        "ict_structure_state",
        "ict_impulse_direction",
        "dist_to_bull_fvg_atr",
        "dist_to_bull_order_block_atr",
    }
    if not context_required.issubset(working.columns):
        context = build_ict_context_features(working, config)
        fresh_columns = [column for column in context.columns if column not in working.columns]
        if fresh_columns:
            working = pd.concat([working, context.loc[:, fresh_columns]], axis=1)

    event_time = resolve_event_time(working)
    derived_columns: dict[str, pd.Series] = {}
    if "session_date" not in working.columns and event_time is not None:
        session = build_ict_equity_session_frame(
            event_time,
            instrument=config.normalized_instrument(),
            source_timezone=getattr(config, "source_timezone", "UTC"),
            canonical_timezone=getattr(config, "canonical_timezone", "UTC"),
        )
        derived_columns["session_date"] = pd.Series(session["session_date"], index=working.index)

    buy_sweep = pd.to_numeric(working.get("ict_buy_side_sweep", 0), errors="coerce").fillna(0).astype(int)
    sell_sweep = pd.to_numeric(working.get("ict_sell_side_sweep", 0), errors="coerce").fillna(0).astype(int)
    sweep_level_code = pd.to_numeric(working.get("ict_sweep_level_code", pd.Series(pd.NA, index=working.index)), errors="coerce")
    sweep_level_value = pd.to_numeric(working.get("ict_sweep_level_value", pd.Series(np.nan, index=working.index)), errors="coerce")
    derived_columns["_last_buy_side_sweep_code"] = sweep_level_code.where(buy_sweep.gt(0)).ffill()
    derived_columns["_last_sell_side_sweep_code"] = sweep_level_code.where(sell_sweep.gt(0)).ffill()
    derived_columns["_last_buy_side_sweep_level"] = sweep_level_value.where(buy_sweep.gt(0)).ffill()
    derived_columns["_last_sell_side_sweep_level"] = sweep_level_value.where(sell_sweep.gt(0)).ffill()
    derived_columns["_last_buy_side_sweep_type"] = derived_columns["_last_buy_side_sweep_code"].map(LEVEL_CODE_TO_NAME).fillna("")
    derived_columns["_last_sell_side_sweep_type"] = derived_columns["_last_sell_side_sweep_code"].map(LEVEL_CODE_TO_NAME).fillna("")
    working = pd.concat([working, pd.DataFrame(derived_columns, index=working.index)], axis=1)

    return working


def _candidate_session_open_manipulation(
    row: pd.Series,
    config: ICTSetupDetectorConfig,
    *,
    pre_ib: bool,
) -> _SetupCandidate | None:
    sweep_direction = _to_int(row.get("ict_sweep_direction"))
    if sweep_direction == 0 or not _as_bool(row.get("ict_is_rth")):
        return None
    if pre_ib and _as_bool(row.get("ict_ib_complete")):
        return None
    if (not pre_ib) and (not _as_bool(row.get("ict_ib_complete"))):
        return None
    if _as_bool(row.get("ict_lunch_lull_flag")) or _as_bool(row.get("ict_close_ramp_flag")):
        return None

    setup_side = 1 if sweep_direction < 0 else -1
    level_code = _to_int(row.get("ict_sweep_level_code"))
    if setup_side > 0 and level_code not in (PRE_IB_LONG_CODES if pre_ib else POST_IB_LONG_CODES):
        return None
    if setup_side < 0 and level_code not in (PRE_IB_SHORT_CODES if pre_ib else POST_IB_SHORT_CODES):
        return None

    reference_level = _to_float(row.get("ict_sweep_level_value"))
    if not np.isfinite(reference_level):
        return None

    setup_type = (
        ICTSetupType.SESSION_OPEN_MANIPULATION_PRE_IB.value
        if pre_ib
        else ICTSetupType.SESSION_OPEN_MANIPULATION_POST_IB.value
    )
    displacement_bonus = 0.10 if _recent_displacement(row, setup_side, 3) else 0.0
    vwap_bonus = 0.05 if _vwap_reaction(row, setup_side) else 0.0
    confidence = _bounded_confidence(0.66 + displacement_bonus + vwap_bonus + _pd_alignment_bonus(row, setup_side))
    anchor_key = _build_anchor_key(
        setup_type,
        setup_side,
        row.get("session_date"),
        LEVEL_CODE_TO_NAME.get(level_code, ""),
        reference_level,
    )
    return _SetupCandidate(
        setup_type=setup_type,
        setup_family=ICTSetupFamily.REVERSAL.value,
        setup_side=setup_side,
        confidence=confidence,
        anchor_level=reference_level,
        entry_price=_to_float(row.get("close")),
        stop_reference=_level_stop(reference_level, setup_side, row, config),
        target_reference=_target_reference_for_side(row, setup_side),
        reference_level=reference_level,
        reference_level_type=LEVEL_CODE_TO_NAME.get(level_code, ""),
        sweep_type="sell_side" if sweep_direction < 0 else "buy_side",
        htf_context=_htf_context_label(row, setup_side),
        displacement_id=_latest_displacement_id(row, setup_side),
        displacement_volume_z=_to_float(row.get("ict_displacement_volume_zscore")),
        session_phase=_to_int(row.get("ict_session_phase_code"), default=0),
        anchor_key=anchor_key,
    )


def _candidate_sweep_reclaim(
    row: pd.Series,
    config: ICTSetupDetectorConfig,
) -> _SetupCandidate | None:
    sweep_direction = _to_int(row.get("ict_sweep_direction"))
    if sweep_direction == 0:
        return None

    setup_side = 1 if sweep_direction < 0 else -1
    level_code = _to_int(row.get("ict_sweep_level_code"))
    reference_level = _to_float(row.get("ict_sweep_level_value"))
    if not np.isfinite(reference_level):
        return None

    confluence = 0.0
    if _recent_displacement(row, setup_side, max(2, int(config.confluence_window_bars))):
        confluence += 0.08
    if _supportive_zone(row, setup_side, include_ifvg=False) is not None:
        confluence += 0.08
    confluence += _pd_alignment_bonus(row, setup_side)
    confidence = _bounded_confidence(
        0.52 + min(max(_to_float(row.get("ict_sweep_penetration_atr")), 0.0), 1.5) / 20.0 + confluence
    )
    return _SetupCandidate(
        setup_type=ICTSetupType.SWEEP_RECLAIM.value,
        setup_family=ICTSetupFamily.REVERSAL.value,
        setup_side=setup_side,
        confidence=confidence,
        anchor_level=reference_level,
        entry_price=_to_float(row.get("close")),
        stop_reference=_level_stop(reference_level, setup_side, row, config),
        target_reference=_target_reference_for_side(row, setup_side),
        reference_level=reference_level,
        reference_level_type=LEVEL_CODE_TO_NAME.get(level_code, ""),
        sweep_type="sell_side" if sweep_direction < 0 else "buy_side",
        htf_context=_htf_context_label(row, setup_side),
        displacement_id=_latest_displacement_id(row, setup_side),
        displacement_volume_z=_to_float(row.get("ict_displacement_volume_zscore")),
        session_phase=_to_int(row.get("ict_session_phase_code"), default=0),
        anchor_key=_build_anchor_key(
            ICTSetupType.SWEEP_RECLAIM.value,
            setup_side,
            LEVEL_CODE_TO_NAME.get(level_code, ""),
            reference_level,
        ),
    )


def _candidate_sweep_displacement_fvg(
    row: pd.Series,
    config: ICTSetupDetectorConfig,
    *,
    position: int,
) -> _SetupCandidate | None:
    for side in (1, -1):
        if not _recent_opposite_side_sweep(row, side, max(int(config.confluence_window_bars), 3)):
            continue
        if not _recent_displacement(row, side, max(int(config.confluence_window_bars), 3)):
            continue
        zone = _supportive_zone(row, side, include_ifvg=False)
        if zone is None or zone["kind"] not in {"fvg", "ifvg"} or zone["is_ifvg"] or not zone["created_by_displacement"]:
            continue
        displacement_index = _latest_displacement_index(row, side)
        if np.isfinite(displacement_index) and position <= int(displacement_index):
            continue
        ce_price = zone["ce"]
        if not np.isfinite(ce_price) or not _bar_touches_level(row, ce_price):
            continue
        if side > 0 and _to_float(row.get("close")) < zone["lower"]:
            continue
        if side < 0 and _to_float(row.get("close")) > zone["upper"]:
            continue

        reference_level, reference_type, sweep_type = _recent_reference_for_side(row, side)
        if not np.isfinite(reference_level):
            continue
        confidence = _bounded_confidence(
            0.67
            + 0.08 * int(_reversal_structure_support(row, side))
            + 0.08 * int(_pd_aligned(row, side))
            + 0.06 * int(_bar_touches_level(row, ce_price))
            + 0.05 * int(zone["dist_atr"] <= 0.25)
        )
        return _SetupCandidate(
            setup_type=ICTSetupType.SWEEP_DISPLACEMENT_FVG.value,
            setup_family=ICTSetupFamily.REVERSAL.value,
            setup_side=side,
            confidence=confidence,
            anchor_level=ce_price,
            entry_price=_to_float(row.get("close")),
            stop_reference=_zone_stop(zone["lower"], zone["upper"], side, row, config),
            target_reference=_target_reference_for_side(row, side),
            reference_level=reference_level,
            reference_level_type=reference_type,
            sweep_type=sweep_type,
            htf_context=_htf_context_label(row, side),
            fvg_id=zone["id"],
            ce_price=ce_price,
            displacement_id=_latest_displacement_id(row, side),
            displacement_volume_z=_to_float(row.get("ict_displacement_volume_zscore")),
            session_phase=_to_int(row.get("ict_session_phase_code"), default=0),
            anchor_key=_build_anchor_key(ICTSetupType.SWEEP_DISPLACEMENT_FVG.value, side, "fvg", zone["id"]),
        )
    return None


def _candidate_ob_retest_after_mss(
    row: pd.Series,
    config: ICTSetupDetectorConfig,
) -> _SetupCandidate | None:
    for side in (1, -1):
        if not _recent_opposite_side_sweep(row, side, max(int(config.confluence_window_bars) + 2, 5)):
            continue
        if not _recent_structure_flip(row, side, max(int(config.confluence_window_bars) + 2, 6)):
            continue
        zone = _order_block_zone(row, side)
        if zone is None or not zone["retest_event"]:
            continue
        if _to_int(row.get("ict_structure_state")) not in {0, side}:
            continue

        reference_level, reference_type, sweep_type = _recent_reference_for_side(row, side)
        confidence = _bounded_confidence(
            0.70
            + 0.08 * int(_recent_mss(row, side, 6))
            + 0.07 * int(_supportive_zone(row, side, include_ifvg=False) is not None)
            + 0.05 * int(_pd_aligned(row, side))
        )
        return _SetupCandidate(
            setup_type=ICTSetupType.OB_RETEST_AFTER_MSS.value,
            setup_family=ICTSetupFamily.REVERSAL.value,
            setup_side=side,
            confidence=confidence,
            anchor_level=_zone_mid(zone["lower"], zone["upper"]),
            entry_price=_to_float(row.get("close")),
            stop_reference=_zone_stop(zone["lower"], zone["upper"], side, row, config),
            target_reference=_target_reference_for_side(row, side),
            reference_level=reference_level,
            reference_level_type=reference_type,
            sweep_type=sweep_type,
            htf_context=_htf_context_label(row, side),
            order_block_id=zone["id"],
            displacement_id=_latest_displacement_id(row, side),
            displacement_volume_z=_to_float(row.get("ict_displacement_volume_zscore")),
            session_phase=_to_int(row.get("ict_session_phase_code"), default=0),
            anchor_key=_build_anchor_key(ICTSetupType.OB_RETEST_AFTER_MSS.value, side, "ob", zone["id"]),
        )
    return None


def _candidate_ifvg_reversal(
    row: pd.Series,
    config: ICTSetupDetectorConfig,
    *,
    position: int,
) -> _SetupCandidate | None:
    for side in (1, -1):
        zone = _supportive_zone(row, side, include_ifvg=True)
        if zone is None or not zone["is_ifvg"]:
            continue
        inversion_index = zone["inversion_index"]
        if not np.isfinite(inversion_index) or position <= int(inversion_index):
            continue
        ce_price = zone["ce"]
        if not np.isfinite(ce_price) or not _bar_touches_level(row, ce_price):
            continue
        close_price = _to_float(row.get("close"))
        if side > 0 and close_price < ce_price:
            continue
        if side < 0 and close_price > ce_price:
            continue
        if not _reversal_structure_support(row, side):
            continue

        confidence = _bounded_confidence(
            0.73 + 0.08 * int(_pd_aligned(row, side)) + 0.06 * int(zone["dist_atr"] <= 0.2)
        )
        return _SetupCandidate(
            setup_type=ICTSetupType.IFVG_REVERSAL.value,
            setup_family=ICTSetupFamily.REVERSAL.value,
            setup_side=side,
            confidence=confidence,
            anchor_level=ce_price,
            entry_price=close_price,
            stop_reference=_zone_stop(zone["lower"], zone["upper"], side, row, config),
            target_reference=_target_reference_for_side(row, side),
            reference_level=ce_price,
            reference_level_type="ifvg_ce",
            sweep_type="",
            htf_context=_htf_context_label(row, side),
            fvg_id=zone["id"],
            ce_price=ce_price,
            displacement_id=_latest_displacement_id(row, side),
            displacement_volume_z=_to_float(row.get("ict_displacement_volume_zscore")),
            session_phase=_to_int(row.get("ict_session_phase_code"), default=0),
            anchor_key=_build_anchor_key(ICTSetupType.IFVG_REVERSAL.value, side, "ifvg", zone["id"]),
        )
    return None


def _candidate_premium_discount_continuation(
    row: pd.Series,
    config: ICTSetupDetectorConfig,
) -> _SetupCandidate | None:
    for side in (1, -1):
        if not _continuation_trend_support(row, side):
            continue
        if not _pd_aligned(row, side):
            continue
        if _recent_structure_flip(row, -side, 3):
            continue
        zone = _supportive_zone(row, side, include_ifvg=False)
        if zone is None or zone["dist_atr"] > 0.75:
            continue

        confidence = _bounded_confidence(
            0.61
            + 0.11 * int(_to_int(row.get("ict_in_ote_band")) == 1)
            + 0.08 * int(zone["dist_atr"] <= 0.25)
            + 0.06 * int(_recent_same_side_sweep(row, side, 8))
        )
        return _SetupCandidate(
            setup_type=ICTSetupType.PREMIUM_DISCOUNT_CONTINUATION.value,
            setup_family=ICTSetupFamily.CONTINUATION.value,
            setup_side=side,
            confidence=confidence,
            anchor_level=zone["ce"] if np.isfinite(zone["ce"]) else _zone_mid(zone["lower"], zone["upper"]),
            entry_price=_to_float(row.get("close")),
            stop_reference=_continuation_stop(row, side, zone, config),
            target_reference=_target_reference_for_side(row, side),
            reference_level=_target_reference_for_side(row, side),
            reference_level_type="dol",
            sweep_type="",
            htf_context=_htf_context_label(row, side),
            fvg_id=zone["id"] if zone["kind"] in {"fvg", "ifvg"} else None,
            ce_price=zone["ce"],
            order_block_id=zone["id"] if zone["kind"] == "order_block" else None,
            displacement_id=_latest_displacement_id(row, side),
            displacement_volume_z=_to_float(row.get("ict_displacement_volume_zscore")),
            session_phase=_to_int(row.get("ict_session_phase_code"), default=0),
            anchor_key=_build_anchor_key(
                ICTSetupType.PREMIUM_DISCOUNT_CONTINUATION.value,
                side,
                zone["kind"],
                zone["id"] if zone["id"] is not None else round(zone["ce"], 6),
            ),
        )
    return None


def _candidate_displacement_continuation_after_raid(
    row: pd.Series,
    config: ICTSetupDetectorConfig,
) -> _SetupCandidate | None:
    for side in (1, -1):
        if not _continuation_trend_support(row, side):
            continue
        if not _recent_opposite_side_sweep(row, side, 8):
            continue
        if not _recent_displacement(row, side, max(int(config.confluence_window_bars), 3)):
            continue
        zone = _supportive_zone(row, side, include_ifvg=False)
        if zone is None:
            continue
        displacement_origin = _latest_displacement_origin(row, side)
        close_price = _to_float(row.get("close"))
        if not np.isfinite(displacement_origin):
            continue
        if side > 0 and close_price <= displacement_origin:
            continue
        if side < 0 and close_price >= displacement_origin:
            continue

        reference_level, reference_type, sweep_type = _recent_reference_for_side(row, side)
        confidence = _bounded_confidence(
            0.64
            + 0.10 * int(zone["dist_atr"] <= 0.35)
            + 0.08 * int(_pd_aligned(row, side))
            + 0.06 * int(_to_float(row.get("ict_displacement_score")) >= 1.0)
        )
        return _SetupCandidate(
            setup_type=ICTSetupType.DISPLACEMENT_CONTINUATION_AFTER_RAID.value,
            setup_family=ICTSetupFamily.CONTINUATION.value,
            setup_side=side,
            confidence=confidence,
            anchor_level=displacement_origin,
            entry_price=close_price,
            stop_reference=_continuation_origin_stop(displacement_origin, side, row, config),
            target_reference=_target_reference_for_side(row, side),
            reference_level=reference_level,
            reference_level_type=reference_type,
            sweep_type=sweep_type,
            htf_context=_htf_context_label(row, side),
            fvg_id=zone["id"] if zone["kind"] in {"fvg", "ifvg"} else None,
            ce_price=zone["ce"],
            order_block_id=zone["id"] if zone["kind"] == "order_block" else None,
            displacement_id=_latest_displacement_id(row, side),
            displacement_volume_z=_to_float(row.get("ict_displacement_volume_zscore")),
            session_phase=_to_int(row.get("ict_session_phase_code"), default=0),
            anchor_key=_build_anchor_key(
                ICTSetupType.DISPLACEMENT_CONTINUATION_AFTER_RAID.value,
                side,
                "displacement",
                _latest_displacement_id(row, side),
            ),
        )
    return None


def _supportive_zone(
    row: pd.Series,
    side: int,
    *,
    include_ifvg: bool,
) -> dict[str, object] | None:
    if side > 0:
        fvg_dist = _to_float(row.get("dist_to_bull_fvg_atr"))
        ob_dist = _to_float(row.get("dist_to_bull_order_block_atr"))
        choose_fvg = np.isfinite(fvg_dist) and (not np.isfinite(ob_dist) or fvg_dist <= ob_dist)
        if choose_fvg:
            is_ifvg = _as_bool(row.get("ict_nearest_bull_fvg_is_ifvg"))
            if is_ifvg and not include_ifvg:
                return None
            return {
                "kind": "ifvg" if is_ifvg else "fvg",
                "id": _nullable_int(row.get("ict_nearest_bull_fvg_id")),
                "lower": _to_float(row.get("ict_nearest_bull_fvg_lower")),
                "upper": _to_float(row.get("ict_nearest_bull_fvg_upper")),
                "ce": _to_float(row.get("ict_nearest_bull_fvg_ce")),
                "dist_atr": fvg_dist,
                "is_ifvg": is_ifvg,
                "created_by_displacement": _as_bool(row.get("ict_nearest_bull_fvg_created_by_displacement")),
                "inversion_index": _to_float(row.get("ict_nearest_bull_fvg_inversion_index")),
            }
        if np.isfinite(ob_dist):
            lower = _to_float(row.get("ict_nearest_bull_order_block_lower"))
            upper = _to_float(row.get("ict_nearest_bull_order_block_upper"))
            return {
                "kind": "order_block",
                "id": _nullable_int(row.get("ict_nearest_bull_order_block_id")),
                "lower": lower,
                "upper": upper,
                "ce": _zone_mid(lower, upper),
                "dist_atr": ob_dist,
                "is_ifvg": False,
                "created_by_displacement": True,
                "inversion_index": np.nan,
            }
        return None

    fvg_dist = _to_float(row.get("dist_to_bear_fvg_atr"))
    ob_dist = _to_float(row.get("dist_to_bear_order_block_atr"))
    choose_fvg = np.isfinite(fvg_dist) and (not np.isfinite(ob_dist) or fvg_dist <= ob_dist)
    if choose_fvg:
        is_ifvg = _as_bool(row.get("ict_nearest_bear_fvg_is_ifvg"))
        if is_ifvg and not include_ifvg:
            return None
        return {
            "kind": "ifvg" if is_ifvg else "fvg",
            "id": _nullable_int(row.get("ict_nearest_bear_fvg_id")),
            "lower": _to_float(row.get("ict_nearest_bear_fvg_lower")),
            "upper": _to_float(row.get("ict_nearest_bear_fvg_upper")),
            "ce": _to_float(row.get("ict_nearest_bear_fvg_ce")),
            "dist_atr": fvg_dist,
            "is_ifvg": is_ifvg,
            "created_by_displacement": _as_bool(row.get("ict_nearest_bear_fvg_created_by_displacement")),
            "inversion_index": _to_float(row.get("ict_nearest_bear_fvg_inversion_index")),
        }
    if np.isfinite(ob_dist):
        lower = _to_float(row.get("ict_nearest_bear_order_block_lower"))
        upper = _to_float(row.get("ict_nearest_bear_order_block_upper"))
        return {
            "kind": "order_block",
            "id": _nullable_int(row.get("ict_nearest_bear_order_block_id")),
            "lower": lower,
            "upper": upper,
            "ce": _zone_mid(lower, upper),
            "dist_atr": ob_dist,
            "is_ifvg": False,
            "created_by_displacement": True,
            "inversion_index": np.nan,
        }
    return None


def _order_block_zone(row: pd.Series, side: int) -> dict[str, object] | None:
    if side > 0 and _as_bool(row.get("ict_bull_order_block_retest_event")):
        return {
            "id": _nullable_int(row.get("ict_nearest_bull_order_block_id")),
            "lower": _to_float(row.get("ict_nearest_bull_order_block_lower")),
            "upper": _to_float(row.get("ict_nearest_bull_order_block_upper")),
            "retest_event": True,
        }
    if side < 0 and _as_bool(row.get("ict_bear_order_block_retest_event")):
        return {
            "id": _nullable_int(row.get("ict_nearest_bear_order_block_id")),
            "lower": _to_float(row.get("ict_nearest_bear_order_block_lower")),
            "upper": _to_float(row.get("ict_nearest_bear_order_block_upper")),
            "retest_event": True,
        }
    return None


def _recent_reference_for_side(row: pd.Series, side: int) -> tuple[float, str, str]:
    if side > 0:
        return (
            _to_float(row.get("_last_sell_side_sweep_level")),
            str(row.get("_last_sell_side_sweep_type", "")),
            "sell_side",
        )
    return (
        _to_float(row.get("_last_buy_side_sweep_level")),
        str(row.get("_last_buy_side_sweep_type", "")),
        "buy_side",
    )


def _recent_opposite_side_sweep(row: pd.Series, side: int, maximum: int) -> bool:
    column = "ict_bars_since_sell_side_sweep" if side > 0 else "ict_bars_since_buy_side_sweep"
    return _within_bars(row.get(column), maximum)


def _recent_same_side_sweep(row: pd.Series, side: int, maximum: int) -> bool:
    column = "ict_bars_since_buy_side_sweep" if side > 0 else "ict_bars_since_sell_side_sweep"
    return _within_bars(row.get(column), maximum)


def _recent_displacement(row: pd.Series, side: int, maximum: int) -> bool:
    column = "ict_bars_since_displacement_bull" if side > 0 else "ict_bars_since_displacement_bear"
    return _within_bars(row.get(column), maximum)


def _recent_structure_flip(row: pd.Series, side: int, maximum: int) -> bool:
    return _recent_choch(row, side, maximum) or _recent_mss(row, side, maximum)


def _recent_choch(row: pd.Series, side: int, maximum: int) -> bool:
    column = "ict_bars_since_choch_bull" if side > 0 else "ict_bars_since_choch_bear"
    return _within_bars(row.get(column), maximum)


def _recent_mss(row: pd.Series, side: int, maximum: int) -> bool:
    column = "ict_bars_since_mss_bull" if side > 0 else "ict_bars_since_mss_bear"
    return _within_bars(row.get(column), maximum)


def _reversal_structure_support(row: pd.Series, side: int) -> bool:
    state = _to_int(row.get("ict_structure_state"), default=0)
    return state == side or _recent_structure_flip(row, side, 8)


def _continuation_trend_support(row: pd.Series, side: int) -> bool:
    structure_state = _to_int(row.get("ict_structure_state"), default=0)
    impulse_direction = _to_int(row.get("ict_impulse_direction"), default=0)
    htf_30m = _to_int(row.get("htf_30m_ema_alignment"), default=0)
    htf_1h = _to_int(row.get("htf_1h_ema_alignment"), default=0)
    htf_score = htf_30m + htf_1h
    if htf_score != 0:
        return htf_score > 0 if side > 0 else htf_score < 0
    return structure_state == side and impulse_direction == side


def _pd_aligned(row: pd.Series, side: int) -> bool:
    if _to_int(row.get("ict_in_ote_band"), default=0) == 1:
        return True
    if side > 0:
        return _to_int(row.get("ict_discount_zone"), default=0) == 1
    return _to_int(row.get("ict_premium_zone"), default=0) == 1


def _pd_alignment_bonus(row: pd.Series, side: int) -> float:
    return 0.07 if _pd_aligned(row, side) else 0.0


def _vwap_reaction(row: pd.Series, side: int) -> bool:
    session_vwap = _to_float(row.get("ict_session_vwap"))
    close_price = _to_float(row.get("close"))
    if not np.isfinite(session_vwap) or not np.isfinite(close_price):
        return False
    return close_price >= session_vwap if side > 0 else close_price <= session_vwap


def _level_stop(level: float, side: int, row: pd.Series, config: ICTSetupDetectorConfig) -> float:
    if not np.isfinite(level):
        return np.nan
    buffer_value = _stop_buffer(row, config)
    return level - buffer_value if side > 0 else level + buffer_value


def _zone_stop(lower: float, upper: float, side: int, row: pd.Series, config: ICTSetupDetectorConfig) -> float:
    if not np.isfinite(lower) or not np.isfinite(upper):
        return np.nan
    buffer_value = _stop_buffer(row, config)
    return lower - buffer_value if side > 0 else upper + buffer_value


def _continuation_stop(
    row: pd.Series,
    side: int,
    zone: dict[str, object],
    config: ICTSetupDetectorConfig,
) -> float:
    origin = _latest_displacement_origin(row, side)
    zone_stop = _zone_stop(float(zone["lower"]), float(zone["upper"]), side, row, config)
    if np.isfinite(origin):
        origin_stop = _continuation_origin_stop(origin, side, row, config)
        if side > 0:
            return min(zone_stop, origin_stop) if np.isfinite(zone_stop) else origin_stop
        return max(zone_stop, origin_stop) if np.isfinite(zone_stop) else origin_stop
    return zone_stop


def _continuation_origin_stop(
    origin: float,
    side: int,
    row: pd.Series,
    config: ICTSetupDetectorConfig,
) -> float:
    if not np.isfinite(origin):
        return np.nan
    buffer_value = _stop_buffer(row, config)
    return origin - buffer_value if side > 0 else origin + buffer_value


def _target_reference_for_side(row: pd.Series, side: int) -> float:
    close_price = _to_float(row.get("close"))
    atr = _to_float(row.get("atr_14"))
    if side > 0:
        for column in ("ict_dol_level_up", "ict_latest_swing_high"):
            level = _to_float(row.get(column))
            if np.isfinite(level) and level > close_price:
                return level
        return close_price + (2.0 * atr) if np.isfinite(atr) else np.nan
    for column in ("ict_dol_level_down", "ict_latest_swing_low"):
        level = _to_float(row.get(column))
        if np.isfinite(level) and level < close_price:
            return level
    return close_price - (2.0 * atr) if np.isfinite(atr) else np.nan


def _latest_displacement_origin(row: pd.Series, side: int) -> float:
    return _to_float(
        row.get("ict_latest_bull_displacement_origin" if side > 0 else "ict_latest_bear_displacement_origin")
    )


def _latest_displacement_index(row: pd.Series, side: int) -> float:
    return _to_float(
        row.get("ict_latest_bull_displacement_index" if side > 0 else "ict_latest_bear_displacement_index")
    )


def _latest_displacement_id(row: pd.Series, side: int) -> int | None:
    return _nullable_int(
        row.get("ict_latest_bull_displacement_id" if side > 0 else "ict_latest_bear_displacement_id")
    )


def _bar_touches_level(row: pd.Series, level: float) -> bool:
    high = _to_float(row.get("high"))
    low = _to_float(row.get("low"))
    return np.isfinite(level) and np.isfinite(high) and np.isfinite(low) and (low <= level <= high)


def _stop_buffer(row: pd.Series, config: ICTSetupDetectorConfig) -> float:
    atr = _to_float(row.get("atr_14"))
    tick_size = float(get_tick_size(config))
    atr_buffer = atr * float(config.invalidation_buffer_atr) if np.isfinite(atr) else 0.0
    atr_floor = atr * 0.25 if np.isfinite(atr) else 0.0
    return max(tick_size, atr_buffer, atr_floor)


def _select_candidate(candidates: list[_SetupCandidate]) -> _SetupCandidate | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            -SETUP_PRIORITY.get(candidate.setup_type, 99),
            candidate.confidence,
            abs(candidate.setup_side),
        ),
    )


def _write_candidate_to_output(out: pd.DataFrame, position: int, candidate: _SetupCandidate) -> None:
    out.iloc[position, out.columns.get_loc("fired")] = True
    out.iloc[position, out.columns.get_loc("setup_type")] = candidate.setup_type
    out.iloc[position, out.columns.get_loc("setup_family")] = candidate.setup_family
    out.iloc[position, out.columns.get_loc("setup_side")] = int(candidate.setup_side)
    out.iloc[position, out.columns.get_loc("confidence")] = float(candidate.confidence)
    out.iloc[position, out.columns.get_loc("anchor_level")] = candidate.anchor_level
    out.iloc[position, out.columns.get_loc("entry_price")] = candidate.entry_price
    out.iloc[position, out.columns.get_loc("stop_reference")] = candidate.stop_reference
    out.iloc[position, out.columns.get_loc("target_reference")] = candidate.target_reference
    out.iloc[position, out.columns.get_loc("reference_level")] = candidate.reference_level
    out.iloc[position, out.columns.get_loc("reference_level_type")] = candidate.reference_level_type
    out.iloc[position, out.columns.get_loc("sweep_type")] = candidate.sweep_type
    out.iloc[position, out.columns.get_loc("htf_context")] = candidate.htf_context
    out.iloc[position, out.columns.get_loc("fvg_id")] = candidate.fvg_id
    out.iloc[position, out.columns.get_loc("ce_price")] = candidate.ce_price
    out.iloc[position, out.columns.get_loc("order_block_id")] = candidate.order_block_id
    out.iloc[position, out.columns.get_loc("displacement_id")] = candidate.displacement_id
    out.iloc[position, out.columns.get_loc("displacement_volume_z")] = candidate.displacement_volume_z
    out.iloc[position, out.columns.get_loc("session_phase")] = candidate.session_phase


def _candidate_record(
    *,
    row: pd.Series,
    position: int,
    candidate: _SetupCandidate,
    session_key: object,
    eligible: bool,
    selected: bool,
) -> dict[str, object]:
    return {
        "bar_index": position,
        "event_time": row.get("datetime", row.get("timestamp")),
        "session_date": session_key,
        "setup_type": candidate.setup_type,
        "setup_family": candidate.setup_family,
        "setup_side": int(candidate.setup_side),
        "confidence": float(candidate.confidence),
        "anchor_level": candidate.anchor_level,
        "reference_level": candidate.reference_level,
        "reference_level_type": candidate.reference_level_type,
        "sweep_type": candidate.sweep_type,
        "fvg_id": candidate.fvg_id,
        "order_block_id": candidate.order_block_id,
        "displacement_id": candidate.displacement_id,
        "eligible": bool(eligible),
        "selected": bool(selected),
        "anchor_key": candidate.anchor_key,
    }


def _build_anchor_key(setup_type: str, side: int, *parts: object) -> str:
    tokens = [setup_type, str(side)]
    for part in parts:
        if part is None or part is pd.NA:
            continue
        if isinstance(part, float) and np.isnan(part):
            continue
        tokens.append(str(part))
    return "|".join(tokens)


def _zone_mid(lower: float, upper: float) -> float:
    if not np.isfinite(lower) or not np.isfinite(upper):
        return np.nan
    return (lower + upper) / 2.0


def _htf_context_label(row: pd.Series, side: int) -> str:
    htf_30m = _to_int(row.get("htf_30m_ema_alignment"), default=0)
    htf_1h = _to_int(row.get("htf_1h_ema_alignment"), default=0)
    if htf_30m != 0 or htf_1h != 0:
        score = htf_30m + htf_1h
        if side > 0:
            return "htf_aligned_long" if score > 0 else "htf_mixed"
        return "htf_aligned_short" if score < 0 else "htf_mixed"
    structure_state = _to_int(row.get("ict_structure_state"), default=0)
    if structure_state == side:
        return "structure_aligned"
    if structure_state == 0:
        return "structure_neutral"
    return "structure_mixed"


def _to_float(value, *, default: float = np.nan) -> float:
    if value is None or value is pd.NA or pd.isna(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, *, default: int = 0) -> int:
    if value is None or value is pd.NA or pd.isna(value):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _nullable_int(value) -> int | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value) -> bool:
    return _to_int(value, default=0) == 1


def _within_bars(value, maximum: int) -> bool:
    numeric = _to_float(value)
    return np.isfinite(numeric) and 0.0 <= numeric <= float(maximum)


def _clamp01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _bounded_confidence(value: float) -> float:
    return _clamp01(value)
