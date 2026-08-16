from __future__ import annotations

import pandas as pd

from .setup_types import SETUP_OUTPUT_COLUMNS


REQUIRED_MARKET_COLUMNS = ("open", "high", "low", "close")


def validate_ict_market_frame(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"ICT setup detection requires columns {missing}.")


def validate_ict_setup_output(df: pd.DataFrame) -> None:
    missing = [column for column in SETUP_OUTPUT_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"ICT setup output is missing required columns {missing}.")
