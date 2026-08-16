from __future__ import annotations

import pandas as pd

from frvp.sessions.equity import build_equity_market_day_labels, build_equity_session_frame

from ..config.instruments import get_ict_base_instrument_config


def build_ict_market_day_labels(*args, **kwargs) -> pd.Series:
    """Thin ICT alias for the existing futures market-day label helper."""

    return build_equity_market_day_labels(*args, **kwargs)


def build_ict_equity_session_frame(
    datetime_values,
    *,
    instrument: str = "es",
    **kwargs,
) -> pd.DataFrame:
    """Reuse the FRVP equity session model with ICT instrument aliases."""

    base_config = get_ict_base_instrument_config(instrument)
    return build_equity_session_frame(datetime_values, instrument=base_config.instrument, **kwargs)
