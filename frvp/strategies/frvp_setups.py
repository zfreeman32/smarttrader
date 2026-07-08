from __future__ import annotations

import pandas as pd

from ..setups.detector import detect_frvp_setups as _detect_frvp_setups


def detect_frvp_setups(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for the Phase 1 FRVP setup detector.

    Design references:
    - Section 3.4 setups
    - Section 7.2 event sampling
    - Phase 1 setup detection milestone
    """

    return _detect_frvp_setups(df)
