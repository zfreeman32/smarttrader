from __future__ import annotations

import math
from typing import Any

import pandas as pd

from frvp.setups.detector import detect_frvp_setups
from ict.setups.detector import detect_ict_setups
from ote_live.contracts.market_data import MarketBar
from ote_live.storage.db import SQLiteLiveDataStore

DASHBOARD_VIEW_STATE_SCOPE = "dashboard_view_state"
MAX_RECENT_FRVP_SETUPS = 40
MAX_RECENT_ICT_SETUPS = 40


def persist_frvp_dashboard_state(
    store: SQLiteLiveDataStore,
    *,
    group_name: str,
    asset: str,
    timeframe: str,
    data_supplier: str,
    policy_frame: pd.DataFrame,
    bar: MarketBar,
) -> None:
    if policy_frame.empty:
        return

    latest = policy_frame.iloc[-1]
    atr = _coerce_float(latest.get("atr_14"))
    close_price = _coerce_float(latest.get("close"))
    if atr is None or atr <= 0.0 or close_price is None:
        return

    setup_payload = _build_setup_payload(policy_frame, close_price=close_price)
    existing_state = store.get_runtime_state(
        scope=DASHBOARD_VIEW_STATE_SCOPE,
        state_key=group_name,
    ) or {}
    recent_setups = list(existing_state.get("recent_setups") or [])
    if setup_payload.get("fired"):
        setup_event = {
            "timestamp_utc": bar.timestamp.isoformat(),
            "setup_type": setup_payload["setup_type"],
            "setup_side": setup_payload["setup_side"],
            "confidence": setup_payload["confidence"],
            "label": setup_payload["label"],
            "bar_close": close_price,
            "bar_low": _coerce_float(latest.get("low")),
            "bar_high": _coerce_float(latest.get("high")),
        }
        if not recent_setups or recent_setups[-1].get("timestamp_utc") != setup_event["timestamp_utc"]:
            recent_setups.append(setup_event)
            recent_setups = recent_setups[-MAX_RECENT_FRVP_SETUPS:]

    payload = {
        "state_version": 1,
        "group_name": group_name,
        "asset": asset,
        "timeframe": timeframe,
        "data_supplier": data_supplier,
        "latest_bar_timestamp_utc": bar.timestamp.isoformat(),
        "symbol": bar.symbol,
        "contract_symbol": bar.contract_symbol,
        "instrument_id": bar.instrument_id,
        "levels": _build_level_payload(latest, close_price=close_price, atr=atr),
        "latest_setup": setup_payload,
        "recent_setups": recent_setups,
    }
    store.upsert_runtime_state(
        scope=DASHBOARD_VIEW_STATE_SCOPE,
        state_key=group_name,
        payload=payload,
    )


def build_frvp_setup_lines(state: dict[str, Any] | None, *, limit: int = 12) -> list[str]:
    if not state:
        return ["No FRVP setup state recorded yet."]
    setups = list(state.get("recent_setups") or [])
    if not setups:
        return ["No FRVP setups have fired yet."]
    return [
        (
            f"{item.get('timestamp_utc')} | {item.get('label')} | "
            f"conf={_format_float(item.get('confidence'))} | close={_format_float(item.get('bar_close'))}"
        )
        for item in setups[-int(limit) :]
    ]


def persist_ict_dashboard_state(
    store: SQLiteLiveDataStore,
    *,
    group_name: str,
    asset: str,
    timeframe: str,
    data_supplier: str,
    policy_frame: pd.DataFrame,
    bar: MarketBar,
) -> None:
    if policy_frame.empty:
        return

    latest = policy_frame.iloc[-1]
    close_price = _coerce_float(latest.get("close"))
    if close_price is None:
        return

    setup_payload = _build_ict_setup_payload(policy_frame, close_price=close_price)
    existing_state = store.get_runtime_state(
        scope=DASHBOARD_VIEW_STATE_SCOPE,
        state_key=group_name,
    ) or {}
    recent_setups = list(existing_state.get("recent_setups") or [])
    if setup_payload.get("fired"):
        setup_event = {
            "timestamp_utc": bar.timestamp.isoformat(),
            "setup_type": setup_payload["setup_type"],
            "setup_family": setup_payload["setup_family"],
            "setup_side": setup_payload["setup_side"],
            "confidence": setup_payload["confidence"],
            "label": setup_payload["label"],
            "bar_close": close_price,
            "bar_low": _coerce_float(latest.get("low")),
            "bar_high": _coerce_float(latest.get("high")),
            "anchor_level": setup_payload["anchor_level"],
            "entry_price": setup_payload["entry_price"],
            "stop_reference": setup_payload["stop_reference"],
            "target_reference": setup_payload["target_reference"],
            "reference_level": setup_payload["reference_level"],
            "reference_level_type": setup_payload["reference_level_type"],
            "sweep_type": setup_payload["sweep_type"],
            "htf_context": setup_payload["htf_context"],
            "ce_price": setup_payload["ce_price"],
            "order_block_id": setup_payload["order_block_id"],
            "displacement_id": setup_payload["displacement_id"],
            "session_phase": setup_payload["session_phase"],
        }
        recent_setups = _append_recent_setup(
            recent_setups,
            setup_event,
            limit=MAX_RECENT_ICT_SETUPS,
        )

    payload = {
        "state_version": 1,
        "group_name": group_name,
        "asset": asset,
        "timeframe": timeframe,
        "data_supplier": data_supplier,
        "latest_bar_timestamp_utc": bar.timestamp.isoformat(),
        "symbol": bar.symbol,
        "contract_symbol": bar.contract_symbol,
        "instrument_id": bar.instrument_id,
        "levels": _build_ict_level_payload(latest),
        "zones": _build_ict_zone_payload(latest),
        "context": _build_ict_context_payload(latest),
        "latest_setup": setup_payload,
        "recent_setups": recent_setups,
    }
    store.upsert_runtime_state(
        scope=DASHBOARD_VIEW_STATE_SCOPE,
        state_key=group_name,
        payload=payload,
    )


def build_ict_setup_lines(state: dict[str, Any] | None, *, limit: int = 12) -> list[str]:
    if not state:
        return ["No ICT setup state recorded yet."]
    setups = list(state.get("recent_setups") or [])
    if not setups:
        return ["No ICT setups have fired yet."]
    return [
        (
            f"{item.get('timestamp_utc')} | {item.get('label')} | "
            f"conf={_format_float(item.get('confidence'))} | "
            f"anchor={_format_float(item.get('anchor_level'))} | "
            f"target={_format_float(item.get('target_reference'))}"
        )
        for item in setups[-int(limit) :]
    ]


def _build_level_payload(latest: pd.Series, *, close_price: float, atr: float) -> dict[str, float]:
    levels: dict[str, float] = {}

    poc = _resolve_level_below(close_price, latest.get("frvp_dist_poc_session_atr"), atr)
    vah = _resolve_level_above(close_price, latest.get("frvp_dist_vah_atr"), atr)
    val = _resolve_level_below(close_price, latest.get("frvp_dist_val_atr"), atr)
    ib_high = _resolve_level_above(close_price, latest.get("frvp_dist_ib_high_atr"), atr)
    ib_low = _resolve_level_below(close_price, latest.get("frvp_dist_ib_low_atr"), atr)
    naked_above = _resolve_level_above(close_price, latest.get("frvp_naked_vpoc_dist_above_atr"), atr)
    naked_below = _resolve_level_below(close_price, latest.get("frvp_naked_vpoc_dist_below_atr"), atr)

    for name, value in (
        ("poc", poc),
        ("vah", vah),
        ("val", val),
        ("ib_high", ib_high),
        ("ib_low", ib_low),
        ("naked_vpoc_above", naked_above),
        ("naked_vpoc_below", naked_below),
    ):
        if value is not None:
            levels[name] = value
    return levels


def _build_setup_payload(policy_frame: pd.DataFrame, *, close_price: float) -> dict[str, Any]:
    try:
        detected = detect_frvp_setups(policy_frame)
        last_detection = detected.iloc[-1]
        fired = bool(last_detection.get("fired"))
        setup_type = int(last_detection.get("setup_type", 0) or 0)
        setup_side = int(last_detection.get("setup_side", 0) or 0)
        confidence = _coerce_float(last_detection.get("confidence")) or 0.0
    except Exception:
        latest = policy_frame.iloc[-1]
        setup_type = int(_coerce_float(latest.get("frvp_setup_type")) or 0)
        setup_side = int(_coerce_float(latest.get("frvp_setup_side")) or 0)
        confidence = _coerce_float(latest.get("frvp_setup_confidence_rule")) or 0.0
        fired = setup_type > 0 and setup_side != 0

    return {
        "fired": fired,
        "setup_type": setup_type,
        "setup_side": setup_side,
        "confidence": confidence,
        "label": _setup_label(setup_type=setup_type, setup_side=setup_side),
        "close": close_price,
    }


def _build_ict_level_payload(latest: pd.Series) -> dict[str, float]:
    levels: dict[str, float] = {}
    for name, value in (
        ("prior_rth_high", latest.get("ict_prior_rth_high")),
        ("prior_rth_low", latest.get("ict_prior_rth_low")),
        ("overnight_high", latest.get("ict_overnight_high")),
        ("overnight_low", latest.get("ict_overnight_low")),
        ("ib_high", latest.get("ict_ib_high")),
        ("ib_low", latest.get("ict_ib_low")),
        ("prior_week_high", latest.get("ict_prior_week_high")),
        ("prior_week_low", latest.get("ict_prior_week_low")),
        ("midnight_open", latest.get("ict_midnight_open")),
        ("open_0830", latest.get("ict_open_0830")),
        ("session_vwap", latest.get("ict_session_vwap")),
        ("dol_up", latest.get("ict_dol_level_up")),
        ("dol_down", latest.get("ict_dol_level_down")),
        ("latest_swing_high", latest.get("ict_latest_swing_high")),
        ("latest_swing_low", latest.get("ict_latest_swing_low")),
    ):
        resolved = _coerce_price_level(value)
        if resolved is not None:
            levels[name] = resolved
    return levels


def _build_ict_zone_payload(latest: pd.Series) -> dict[str, dict[str, Any]]:
    zones: dict[str, dict[str, Any]] = {}
    for name, lower_key, upper_key, ce_key, id_key, is_ifvg_key, displacement_key in (
        (
            "bull_fvg",
            "ict_nearest_bull_fvg_lower",
            "ict_nearest_bull_fvg_upper",
            "ict_nearest_bull_fvg_ce",
            "ict_nearest_bull_fvg_id",
            "ict_nearest_bull_fvg_is_ifvg",
            "ict_nearest_bull_fvg_created_by_displacement",
        ),
        (
            "bear_fvg",
            "ict_nearest_bear_fvg_lower",
            "ict_nearest_bear_fvg_upper",
            "ict_nearest_bear_fvg_ce",
            "ict_nearest_bear_fvg_id",
            "ict_nearest_bear_fvg_is_ifvg",
            "ict_nearest_bear_fvg_created_by_displacement",
        ),
        (
            "bull_order_block",
            "ict_nearest_bull_order_block_lower",
            "ict_nearest_bull_order_block_upper",
            None,
            "ict_nearest_bull_order_block_id",
            None,
            None,
        ),
        (
            "bear_order_block",
            "ict_nearest_bear_order_block_lower",
            "ict_nearest_bear_order_block_upper",
            None,
            "ict_nearest_bear_order_block_id",
            None,
            None,
        ),
    ):
        lower = _coerce_price_level(latest.get(lower_key))
        upper = _coerce_price_level(latest.get(upper_key))
        zone_id = _coerce_int(latest.get(id_key))
        if (
            lower is None
            or upper is None
            or lower > upper
            or zone_id is None
            or zone_id <= 0
        ):
            continue
        ce_price = _coerce_price_level(latest.get(ce_key)) if ce_key else _zone_mid(lower, upper)
        if ce_price is None or not lower <= ce_price <= upper:
            ce_price = _zone_mid(lower, upper)
        zones[name] = {
            "lower": lower,
            "upper": upper,
            "ce_price": ce_price,
            "id": zone_id,
            "is_ifvg": bool(_coerce_bool(latest.get(is_ifvg_key))) if is_ifvg_key else False,
            "created_by_displacement": bool(_coerce_bool(latest.get(displacement_key))) if displacement_key else False,
        }
    return zones


def _build_ict_context_payload(latest: pd.Series) -> dict[str, Any]:
    return {
        "structure_state": _coerce_int(latest.get("ict_structure_state")),
        "impulse_direction": _coerce_int(latest.get("ict_impulse_direction")),
        "session_phase_code": _coerce_int(latest.get("ict_session_phase_code")),
        "is_rth": _coerce_bool(latest.get("ict_is_rth")),
        "ib_complete": _coerce_bool(latest.get("ict_ib_complete")),
        "in_ote_band": _coerce_int(latest.get("ict_in_ote_band")),
        "premium_zone": _coerce_bool(latest.get("ict_premium_zone")),
        "discount_zone": _coerce_bool(latest.get("ict_discount_zone")),
        "session_vwap": _coerce_float(latest.get("ict_session_vwap")),
    }


def _build_ict_setup_payload(policy_frame: pd.DataFrame, *, close_price: float) -> dict[str, Any]:
    try:
        detected = detect_ict_setups(policy_frame)
        last_detection = detected.iloc[-1]
        fired = bool(last_detection.get("fired"))
        setup_type = str(last_detection.get("setup_type") or "none")
        setup_family = str(last_detection.get("setup_family") or "none")
        setup_side = int(_coerce_float(last_detection.get("setup_side")) or 0)
        confidence = _coerce_float(last_detection.get("confidence")) or 0.0
    except Exception:
        last_detection = pd.Series(dtype=object)
        fired = False
        setup_type = "none"
        setup_family = "none"
        setup_side = 0
        confidence = 0.0
    return {
        "fired": fired,
        "setup_type": setup_type,
        "setup_family": setup_family,
        "setup_side": setup_side,
        "confidence": confidence,
        "label": _ict_setup_label(setup_type=setup_type, setup_side=setup_side),
        "close": close_price,
        "anchor_level": _coerce_float(last_detection.get("anchor_level")),
        "entry_price": _coerce_float(last_detection.get("entry_price")),
        "stop_reference": _coerce_float(last_detection.get("stop_reference")),
        "target_reference": _coerce_float(last_detection.get("target_reference")),
        "reference_level": _coerce_float(last_detection.get("reference_level")),
        "reference_level_type": str(last_detection.get("reference_level_type") or ""),
        "sweep_type": str(last_detection.get("sweep_type") or ""),
        "htf_context": str(last_detection.get("htf_context") or ""),
        "ce_price": _coerce_float(last_detection.get("ce_price")),
        "order_block_id": _coerce_int(last_detection.get("order_block_id")),
        "displacement_id": _coerce_int(last_detection.get("displacement_id")),
        "session_phase": _coerce_int(last_detection.get("session_phase")),
    }


def _resolve_level_above(close_price: float, distance_atr, atr: float) -> float | None:
    distance = _coerce_float(distance_atr)
    if distance is None:
        return None
    return close_price + (distance * atr)


def _resolve_level_below(close_price: float, distance_atr, atr: float) -> float | None:
    distance = _coerce_float(distance_atr)
    if distance is None:
        return None
    return close_price - (distance * atr)


def _setup_label(*, setup_type: int, setup_side: int) -> str:
    if setup_type <= 0 or setup_side == 0:
        return "No setup"
    side_text = "Long" if setup_side > 0 else "Short"
    return f"S{setup_type} {side_text}"


def _ict_setup_label(*, setup_type: str, setup_side: int) -> str:
    if not setup_type or setup_type == "none" or setup_side == 0:
        return "No setup"
    side_text = "Long" if setup_side > 0 else "Short"
    return f"{_titleize_identifier(setup_type)} {side_text}"


def _titleize_identifier(value: str) -> str:
    return " ".join(part.capitalize() for part in str(value).split("_") if part)


def _coerce_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_price_level(value) -> float | None:
    resolved = _coerce_float(value)
    if resolved is None or not math.isfinite(resolved) or resolved <= 0.0:
        return None
    return resolved


def _coerce_int(value) -> int | None:
    resolved = _coerce_float(value)
    if resolved is None:
        return None
    return int(resolved)


def _coerce_bool(value) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def _append_recent_setup(
    recent_setups: list[dict[str, Any]],
    setup_event: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not recent_setups or recent_setups[-1].get("timestamp_utc") != setup_event["timestamp_utc"]:
        recent_setups.append(setup_event)
    return recent_setups[-int(limit) :]


def _zone_mid(lower: float, upper: float) -> float:
    return (float(lower) + float(upper)) / 2.0


def _format_float(value) -> str:
    resolved = _coerce_float(value)
    if resolved is None:
        return "n/a"
    return f"{resolved:.4f}"
