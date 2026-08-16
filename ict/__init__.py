"""ES-first ICT / Smart Money Concepts research package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ICTInstrumentConfig",
    "get_ict_base_instrument_config",
    "get_ict_instrument_config",
    "normalize_ict_instrument",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .config.instruments import (
            ICTInstrumentConfig,
            get_ict_base_instrument_config,
            get_ict_instrument_config,
            normalize_ict_instrument,
        )

        exports = {
            "ICTInstrumentConfig": ICTInstrumentConfig,
            "get_ict_base_instrument_config": get_ict_base_instrument_config,
            "get_ict_instrument_config": get_ict_instrument_config,
            "normalize_ict_instrument": normalize_ict_instrument,
        }
        return exports[name]
    raise AttributeError(f"module 'ict' has no attribute {name!r}")
