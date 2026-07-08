from __future__ import annotations

from typing import Any

import pandas as pd

from frvp.setups.detector import detect_frvp_setups
from ote_live.contracts.market_data import MarketBar
from ote_live.storage.db import SQLiteLiveDataStore

DASHBOARD_VIEW_STATE_SCOPE = "dashboard_view_state"
MAX_RECENT_FRVP_SETUPS = 40


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


def _coerce_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value) -> str:
    resolved = _coerce_float(value)
    if resolved is None:
        return "n/a"
    return f"{resolved:.4f}"
