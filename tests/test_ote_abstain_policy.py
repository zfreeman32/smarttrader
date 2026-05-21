from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_testing.ote_abstain_policy import HardAbstainConfig, apply_abstain_policy


def test_apply_abstain_policy_supports_explicit_session_composite_and_quantile_filters() -> None:
    frame = pd.DataFrame(
        {
            "source_row_idx": [0, 1, 2, 3, 4],
            "policy_signal_candidate": [True, True, True, True, True],
            "policy_probability": [0.51, 0.60, 0.72, 0.85, 0.95],
            "session_regime": ["overlap", "london", "london", "asia", "london"],
            "composite_regime": [
                "ranging_low",
                "strong_up_high",
                "strong_up_medium",
                "strong_down_medium",
                "strong_up_medium",
            ],
            "stress_regime": ["normal"] * 5,
        }
    )

    decided = apply_abstain_policy(
        frame,
        config=HardAbstainConfig(
            abstain_high_stress=False,
            abstain_off_hours=False,
            cooldown_bars=0,
            abstain_session_regimes=("overlap",),
            abstain_composite_regimes=("strong_up_high",),
            minimum_probability_quantile=0.20,
            probability_column="policy_probability",
            signal_candidate_column="policy_signal_candidate",
            position_column="source_row_idx",
        ),
    )

    reasons = decided["policy_abstain_reason"].tolist()
    assert reasons[0] == "session_filter"
    assert reasons[1] == "composite_filter"
    assert reasons[2] == "probability_quantile_filter"
    assert bool(decided.loc[3, "policy_emit_signal"]) is True
    assert bool(decided.loc[4, "policy_emit_signal"]) is True


def test_apply_abstain_policy_supports_composite_session_and_stress_pairs() -> None:
    frame = pd.DataFrame(
        {
            "source_row_idx": [0, 1, 2, 3, 4],
            "policy_signal_candidate": [True, True, True, True, True],
            "policy_probability": [0.91, 0.92, 0.93, 0.94, 0.95],
            "session_regime": ["asia", "london", "asia", "overlap", "london"],
            "composite_regime": [
                "strong_up_medium",
                "strong_up_medium",
                "strong_up_high",
                "strong_up_medium",
                "ranging_medium",
            ],
            "stress_regime": ["normal", "elevated", "normal", "elevated", "normal"],
        }
    )

    decided = apply_abstain_policy(
        frame,
        config=HardAbstainConfig(
            abstain_high_stress=False,
            abstain_off_hours=False,
            cooldown_bars=0,
            abstain_composite_session_pairs=(("strong_up_medium", "asia"),),
            abstain_composite_stress_pairs=(("strong_up_medium", "elevated"),),
            probability_column="policy_probability",
            signal_candidate_column="policy_signal_candidate",
            position_column="source_row_idx",
        ),
    )

    reasons = decided["policy_abstain_reason"].tolist()
    assert reasons[0] == "composite_session_filter"
    assert reasons[1] == "composite_stress_filter"
    assert pd.isna(reasons[2])
    assert reasons[3] == "composite_stress_filter"
    assert pd.isna(reasons[4])
    assert bool(decided.loc[2, "policy_emit_signal"]) is True
    assert bool(decided.loc[4, "policy_emit_signal"]) is True
