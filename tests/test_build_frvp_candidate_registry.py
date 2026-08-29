from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_frvp_candidate_registry import build_registry_payload


def test_build_frvp_candidate_registry_generates_meta_ids_and_preserves_existing_metadata() -> None:
    base_dir = ROOT / "tmp" / f"test_build_frvp_candidate_registry_{uuid4().hex}"
    model_root = base_dir / "models" / "frvp_test_xgb"
    meta_dir = model_root / "long_frvp_meta"
    reversal_dir = model_root / "long_frvp_reversal"
    source_registry_path = base_dir / "models" / "source_registry.json"

    meta_dir.mkdir(parents=True, exist_ok=True)
    reversal_dir.mkdir(parents=True, exist_ok=True)

    try:
        meta_dir.joinpath("training_summary.json").write_text(
            json.dumps(
                {
                    "target": "long_frvp_meta",
                    "backend": "xgboost",
                    "threshold": 0.57,
                    "cv_summary": {
                        "mean_average_precision": 0.61,
                        "mean_event_fbeta_0_5": 0.66,
                    },
                    "test_metrics": {
                        "average_precision": 0.63,
                        "event_fbeta_0_5": 0.68,
                    },
                    "calibration": {"resolved_method": "platt"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        reversal_dir.joinpath("training_summary.json").write_text(
            json.dumps(
                {
                    "target": "long_frvp_reversal",
                    "backend": "xgboost",
                    "threshold": 0.64,
                    "cv_summary": {
                        "mean_average_precision": 0.71,
                        "mean_event_fbeta_0_5": 0.77,
                    },
                    "test_metrics": {
                        "average_precision": 0.73,
                        "event_fbeta_0_5": 0.75,
                    },
                    "calibration": {"resolved_method": "platt"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        source_registry_path.write_text(
            json.dumps(
                {
                    "promotion_rules": {
                        "min_cv_splits": 3,
                        "min_test_event_f05": 0.65,
                        "require_regime_robustness": True,
                        "require_post_cost_profitability": True,
                        "require_paper_trading_confirmation": True,
                    },
                    "models": [
                        {
                            "model_id": "frvp_long_reversal_xgb_v1",
                            "direction": "long",
                            "role": "benchmark",
                            "backend": "xgboost",
                            "artifact_path": str(reversal_dir.relative_to(ROOT)),
                            "cv_mean_ap": 0.0,
                            "cv_mean_event_f05": 0.0,
                            "test_ap": 0.0,
                            "test_event_f05": 0.0,
                            "global_threshold": 0.50,
                            "regime_thresholds": {"ranging_low": 0.5},
                            "abstain_policy": None,
                            "calibration_method": "none",
                            "promotion_date": "2026-06-30",
                            "promotion_reason": "keep me",
                            "status": "candidate",
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        payload = build_registry_payload(
            model_roots=[model_root],
            source_registry_path=source_registry_path,
        )

        records = {record["model_id"]: record for record in payload["models"]}
        assert set(records) == {"frvp_long_meta_xgb_v1", "frvp_long_reversal_xgb_v1"}
        assert records["frvp_long_meta_xgb_v1"]["global_threshold"] == 0.57
        assert records["frvp_long_meta_xgb_v1"]["role"] == "candidate"
        assert "pooled meta-model candidate" in records["frvp_long_meta_xgb_v1"]["promotion_reason"].lower()
        assert records["frvp_long_reversal_xgb_v1"]["role"] == "benchmark"
        assert records["frvp_long_reversal_xgb_v1"]["promotion_reason"] == "keep me"
        assert records["frvp_long_reversal_xgb_v1"]["regime_thresholds"] == {"ranging_low": 0.5}
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_build_frvp_candidate_registry_accepts_repo_relative_model_roots() -> None:
    base_dir = ROOT / "tmp" / f"test_build_frvp_candidate_registry_relative_{uuid4().hex}"
    model_root = base_dir / "models" / "frvp_test_xgb"
    meta_dir = model_root / "short_frvp_meta"

    meta_dir.mkdir(parents=True, exist_ok=True)

    try:
        meta_dir.joinpath("training_summary.json").write_text(
            json.dumps(
                {
                    "target": "short_frvp_meta",
                    "backend": "xgboost",
                    "threshold": 0.41,
                    "cv_summary": {
                        "mean_average_precision": 0.58,
                        "mean_event_fbeta_0_5": 0.63,
                    },
                    "test_metrics": {
                        "average_precision": 0.6,
                        "event_fbeta_0_5": 0.64,
                    },
                    "calibration": {"resolved_method": "none"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        payload = build_registry_payload(
            model_roots=[model_root.relative_to(ROOT)],
            source_registry_path=None,
        )

        records = {record["model_id"]: record for record in payload["models"]}
        assert set(records) == {"frvp_short_meta_xgb_v1"}
        assert records["frvp_short_meta_xgb_v1"]["artifact_path"] == str(meta_dir.relative_to(ROOT))
        assert records["frvp_short_meta_xgb_v1"]["global_threshold"] == 0.41
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
