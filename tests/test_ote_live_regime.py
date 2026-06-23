from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ote_live.policies.regime import resolve_latest_regime


def test_resolve_latest_regime_accepts_timestamp_only_policy_context() -> None:
    regime = resolve_latest_regime(
        {
            "timestamp": "2024-01-02T07:30:00+00:00",
            "close": 1.20,
            "ema_alignment": 0.25,
            "atr_14": 1.0,
            "range_shock_20": 2.5,
        }
    )

    assert regime is not None
    assert regime.trend_regime == "weak_up"
    assert regime.vol_regime == "high"
    assert regime.session_regime == "london"
    assert regime.stress_regime == "elevated"
    assert regime.composite_regime == "weak_up_high"
