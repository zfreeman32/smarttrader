from __future__ import annotations

import pandas as pd

from ict.setups.detector import detect_ict_setups as _detect_ict_setups


def detect_ict_setups(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for the future ICT setup detector."""

    return _detect_ict_setups(df)
