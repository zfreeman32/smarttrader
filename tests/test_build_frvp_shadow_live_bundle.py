from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_frvp_shadow_live_bundle import _build_policy_contract_audit


def test_build_policy_contract_audit_reports_walk_forward_policy_mix() -> None:
    audit = _build_policy_contract_audit(
        threshold_model_output={
            "selected_policy_name": "global_threshold",
            "selected_policy_reason": "no_non_global_policy_qualified_against_test_baseline",
            "selected_policy_contract": {
                "policy_name": "global_threshold",
                "base_policy_is_hard_pruned": True,
            },
            "qualified_policy_names": [],
        },
        backtest_model_output={
            "selected_policy_counts": {
                "global_threshold": 415,
                "regime_threshold": 109,
            }
        },
    )

    assert audit["static_selected_policy_name"] == "global_threshold"
    assert audit["walk_forward_dominant_policy_name"] == "global_threshold"
    assert audit["walk_forward_policy_mix_is_multicontract"] is True
    assert audit["static_vs_walk_forward_dominant_mismatch"] is False
    assert audit["walk_forward_static_policy_share"] == pytest.approx(415.0 / 524.0)
