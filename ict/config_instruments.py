"""Compatibility shim for ICT instrument config imports."""

from .config.instruments import (
    ICTInstrumentConfig,
    get_ict_base_instrument_config,
    get_ict_instrument_config,
    normalize_ict_instrument,
)

__all__ = [
    "ICTInstrumentConfig",
    "get_ict_base_instrument_config",
    "get_ict_instrument_config",
    "normalize_ict_instrument",
]
