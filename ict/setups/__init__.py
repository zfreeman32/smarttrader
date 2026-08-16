"""Setup detection surfaces for ICT research."""

from .detector import detect_ict_setups, summarize_setup_fire_rates
from .setup_types import ICTSetupFamily, ICTSetupSide, ICTSetupType

__all__ = [
    "ICTSetupFamily",
    "ICTSetupSide",
    "ICTSetupType",
    "detect_ict_setups",
    "summarize_setup_fire_rates",
]
