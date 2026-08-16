from __future__ import annotations

from dataclasses import dataclass

from frvp.config.instruments import InstrumentConfig, get_instrument_config


@dataclass(frozen=True)
class ICTInstrumentConfig:
    """Instrument contract for ICT research and later MES validation."""

    instrument: str
    training_instrument: str
    execution_instrument: str
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
    notes: str = ""


def _from_frvp_config(
    base: InstrumentConfig,
    *,
    instrument: str,
    execution_instrument: str,
    tick_value: float | None = None,
    notes: str = "",
) -> ICTInstrumentConfig:
    return ICTInstrumentConfig(
        instrument=instrument,
        training_instrument=base.instrument,
        execution_instrument=execution_instrument,
        tick_size=float(base.tick_size),
        tick_value=float(base.tick_value if tick_value is None else tick_value),
        market_close_timezone=str(base.market_close_timezone),
        market_close_hour=int(base.market_close_hour),
        market_close_minute=int(base.market_close_minute),
        rth_timezone=str(base.rth_timezone),
        rth_start_hour=int(base.rth_start_hour),
        rth_start_minute=int(base.rth_start_minute),
        rth_end_hour=int(base.rth_end_hour),
        rth_end_minute=int(base.rth_end_minute),
        overnight_start_hour=int(base.overnight_start_hour),
        overnight_start_minute=int(base.overnight_start_minute),
        ib_minutes=int(base.ib_minutes),
        notes=notes,
    )


_ES_BASE = get_instrument_config("es")
_6E_BASE = get_instrument_config("6e")

_CONFIGS: dict[str, ICTInstrumentConfig] = {
    "es": _from_frvp_config(
        _ES_BASE,
        instrument="es",
        execution_instrument="es",
        notes="Primary ES research instrument for the ICT meta-labeling pipeline.",
    ),
    "mes": _from_frvp_config(
        _ES_BASE,
        instrument="mes",
        execution_instrument="mes",
        tick_value=1.25,
        notes="Micro ES alias for paper-trading and small-size validation; shares the ES session model.",
    ),
    "6e": _from_frvp_config(
        _6E_BASE,
        instrument="6e",
        execution_instrument="6e",
        notes="Future FX-style variant retained for later ICT transfer research.",
    ),
}


def normalize_ict_instrument(instrument: str | ICTInstrumentConfig) -> str:
    if isinstance(instrument, ICTInstrumentConfig):
        return instrument.instrument
    return str(instrument).strip().lower()


def get_ict_instrument_config(instrument: str | ICTInstrumentConfig) -> ICTInstrumentConfig:
    if isinstance(instrument, ICTInstrumentConfig):
        return instrument

    key = normalize_ict_instrument(instrument)
    try:
        return _CONFIGS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONFIGS))
        raise KeyError(f"Unsupported ICT instrument '{instrument}'. Supported: {supported}") from exc


def get_ict_base_instrument_config(instrument: str | ICTInstrumentConfig) -> InstrumentConfig:
    config = get_ict_instrument_config(instrument)
    return get_instrument_config(config.training_instrument)
