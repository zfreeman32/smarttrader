from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentConfig:
    """Instrument-level FRVP settings shared across continuity, sessions, and profiles."""

    instrument: str
    tick_size: float
    tick_value: float
    market_close_timezone: str
    market_close_hour: int
    market_close_minute: int
    rth_timezone: str
    rth_start_hour: int
    rth_start_minute: int
    rth_end_hour: int
    rth_end_minute: int
    overnight_start_hour: int
    overnight_start_minute: int
    ib_minutes: int
    profile_value_area_pct: float
    composite_sessions: int
    roll_reset_mode: str = "reset"


_CONFIGS = {
    "es": InstrumentConfig(
        instrument="es",
        tick_size=0.25,
        tick_value=12.50,
        market_close_timezone="America/New_York",
        market_close_hour=17,
        market_close_minute=0,
        rth_timezone="America/New_York",
        rth_start_hour=9,
        rth_start_minute=30,
        rth_end_hour=16,
        rth_end_minute=0,
        overnight_start_hour=16,
        overnight_start_minute=0,
        ib_minutes=60,
        profile_value_area_pct=0.70,
        composite_sessions=5,
    ),
    "6e": InstrumentConfig(
        instrument="6e",
        tick_size=0.00005,
        tick_value=6.25,
        market_close_timezone="America/New_York",
        market_close_hour=17,
        market_close_minute=0,
        rth_timezone="America/New_York",
        rth_start_hour=9,
        rth_start_minute=30,
        rth_end_hour=16,
        rth_end_minute=0,
        overnight_start_hour=16,
        overnight_start_minute=0,
        ib_minutes=60,
        profile_value_area_pct=0.70,
        composite_sessions=5,
    ),
}


def get_instrument_config(instrument: str | InstrumentConfig) -> InstrumentConfig:
    """Return FRVP config for a supported instrument or passthrough an existing config."""

    if isinstance(instrument, InstrumentConfig):
        return instrument

    key = str(instrument).strip().lower()
    try:
        return _CONFIGS[key]
    except KeyError as exc:
        raise KeyError(f"Unsupported FRVP instrument '{instrument}'.") from exc
