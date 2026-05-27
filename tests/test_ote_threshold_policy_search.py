from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ote_threshold_policy_search import _write_registry_policy_updates, build_arg_parser


def test_write_registry_policy_updates_accepts_utf8_bom(tmp_path: Path) -> None:
    payload = {
        "promotion_rules": {
            "min_cv_splits": 3,
            "min_test_event_f05": 0.65,
            "require_regime_robustness": True,
            "require_post_cost_profitability": True,
            "require_paper_trading_confirmation": True,
        },
        "models": [
            {
                "model_id": "demo_model",
                "direction": "long",
                "role": "candidate",
                "backend": "tcn",
                "artifact_path": "models/ote_full_tcn_v2/long_ote",
                "cv_mean_ap": 0.6,
                "cv_mean_event_f05": 0.7,
                "test_ap": 0.7,
                "test_event_f05": 0.8,
                "global_threshold": 0.5,
                "regime_thresholds": None,
                "abstain_policy": None,
                "calibration_method": "platt",
                "promotion_date": "2026-05-13",
                "promotion_reason": "test",
                "status": "candidate",
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8-sig")

    _write_registry_policy_updates(
        registry_path=registry_path,
        updates={
            "demo_model": {
                "global_threshold": 0.81,
                "regime_thresholds": {"strong_up_high": 0.77},
                "abstain_policy": {"policy_name": "global_threshold_plus_abstain"},
            }
        },
    )

    updated = json.loads(registry_path.read_text(encoding="utf-8"))
    record = updated["models"][0]

    assert record["global_threshold"] == 0.81
    assert record["regime_thresholds"] == {"strong_up_high": 0.77}
    assert record["abstain_policy"] == {"policy_name": "global_threshold_plus_abstain"}


def test_build_arg_parser_accepts_targeted_filter_preset() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--targeted-filter-preset",
            "short_reversal_xgb_ranging_medium_london_prune_v1",
        ]
    )

    assert args.targeted_filter_preset == "short_reversal_xgb_ranging_medium_london_prune_v1"


def test_build_arg_parser_accepts_short_reversal_hard_prune_preset() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--regime-report-root",
            "tmp/regime",
            "--targeted-filter-preset",
            "short_reversal_xgb_ranging_medium_london_hard_prune_v1",
        ]
    )

    assert args.targeted_filter_preset == "short_reversal_xgb_ranging_medium_london_hard_prune_v1"
