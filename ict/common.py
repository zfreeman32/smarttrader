from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from features.transforms import calculate_atr

from .config.instruments import get_ict_instrument_config


def cfg_value(config: object, *names: str, default: Any) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def get_atr_like(df: pd.DataFrame) -> pd.Series:
    if "atr_14" in df.columns:
        return pd.to_numeric(df["atr_14"], errors="coerce")
    return calculate_atr(df)


def resolve_event_time(df: pd.DataFrame) -> pd.Series | None:
    for candidate in ("datetime", "timestamp", "ts_event"):
        if candidate in df.columns:
            return pd.to_datetime(df[candidate], errors="coerce")
    return None


def resolve_instrument(config: object, default: str = "es") -> str:
    raw_value = cfg_value(config, "instrument", default=default)
    if raw_value is None:
        return default
    value = str(raw_value).strip().lower()
    if value in {"", "none", "nan"}:
        return default
    return value


def get_tick_size(config: object, default: float = 0.25) -> float:
    instrument = resolve_instrument(config)
    try:
        return float(get_ict_instrument_config(instrument).tick_size)
    except KeyError:
        return float(default)


def get_tick_value(config: object, default: float = 12.5) -> float:
    instrument = resolve_instrument(config)
    try:
        return float(get_ict_instrument_config(instrument).tick_value)
    except KeyError:
        return float(default)


def sweep_buffer_price(
    atr: pd.Series,
    config: object,
) -> pd.Series:
    tick_size = get_tick_size(config)
    tick_buffer_ticks = float(cfg_value(config, "sweep_buffer_ticks", "ict_sweep_buffer_ticks", default=1.0))
    atr_buffer = float(cfg_value(config, "sweep_buffer_atr", "ict_sweep_buffer_atr", default=0.05))
    tick_component = tick_size * tick_buffer_ticks
    atr_component = atr.fillna(0.0) * atr_buffer
    return np.maximum(atr_component.to_numpy(dtype=float, copy=False), tick_component)


def structure_break_buffer_price(
    atr: pd.Series,
    config: object,
) -> pd.Series:
    tick_size = get_tick_size(config)
    tick_component = tick_size * 1.0
    atr_component = atr.fillna(0.0) * float(
        cfg_value(config, "structure_break_buffer_atr", "ict_break_buffer_atr", default=0.05)
    )
    return np.maximum(atr_component.to_numpy(dtype=float, copy=False), tick_component)


def session_phase_codes_from_frame(session_frame: pd.DataFrame) -> pd.Series:
    if session_frame.empty:
        return pd.Series(dtype="Int64")

    minutes_since_open = pd.to_numeric(session_frame.get("minutes_since_rth_open"), errors="coerce")
    minutes_until_close = pd.to_numeric(session_frame.get("minutes_until_rth_close"), errors="coerce")
    is_rth = session_frame.get("is_rth", pd.Series(False, index=session_frame.index)).fillna(False).astype(bool)
    is_overnight = (
        session_frame.get("is_overnight", pd.Series(False, index=session_frame.index)).fillna(False).astype(bool)
    )
    is_ib = session_frame.get("is_ib", pd.Series(False, index=session_frame.index)).fillna(False).astype(bool)

    phase = pd.Series(0, index=session_frame.index, dtype="Int64")
    phase.loc[is_overnight] = 1
    phase.loc[is_rth & is_ib] = 2
    phase.loc[is_rth & minutes_since_open.ge(60) & minutes_since_open.lt(150)] = 3
    phase.loc[is_rth & minutes_since_open.ge(150) & minutes_since_open.lt(240)] = 4
    phase.loc[is_rth & minutes_until_close.le(90)] = 5
    phase.loc[is_rth & phase.eq(0)] = 3
    return phase


def rolling_group_zscore(
    values: pd.Series,
    groups: pd.Series,
    *,
    window: int,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.empty:
        return pd.Series(dtype=float)

    aligned_groups = pd.Series(groups, index=numeric.index)

    def _zscore(group: pd.Series) -> pd.Series:
        history = group.shift(1)
        mean = history.rolling(window, min_periods=max(5, min(window, 20))).mean()
        std = history.rolling(window, min_periods=max(5, min(window, 20))).std().replace(0, np.nan)
        return (group - mean) / std

    return numeric.groupby(aligned_groups, group_keys=False).apply(_zscore)
