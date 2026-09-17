from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ict.reports.classic_breaker_phase3_audit import _inspect_prepared_features  # noqa: E402


def test_prepared_feature_audit_ignores_report_metadata_tokens(tmp_path: Path) -> None:
    target_dir = tmp_path / "prepared" / "short_ict_classic_breaker"
    target_dir.mkdir(parents=True)
    (target_dir / "features.json").write_text(
        json.dumps({"features": ["ict_total_confluence_1atr", "htf_confluence_short_ict_classic_breaker"]}),
        encoding="utf-8",
    )
    (target_dir / "report.json").write_text(
        json.dumps(
            {
                "target_column": "label_short_ict_classic_breaker",
                "sample_weight_column": "sample_weight_short_ict_classic_breaker",
            }
        ),
        encoding="utf-8",
    )

    summary = _inspect_prepared_features(tmp_path / "prepared")

    assert summary["available"] is True
    assert summary["inspected_feature_count"] == 2
    assert summary["suspicious_reference_count"] == 0


def test_prepared_feature_audit_flags_leaky_feature_names(tmp_path: Path) -> None:
    target_dir = tmp_path / "prepared" / "short_ict_classic_breaker"
    target_dir.mkdir(parents=True)
    (target_dir / "features.json").write_text(
        json.dumps({"features": ["ict_total_confluence_1atr", "tb_outcome_shadow"]}),
        encoding="utf-8",
    )
    pd.DataFrame({"feature": ["label_short_ict_classic_breaker_shadow"]}).to_csv(
        target_dir / "feature_importance.csv",
        index=False,
    )

    summary = _inspect_prepared_features(tmp_path / "prepared")

    assert summary["suspicious_reference_count"] == 2
    assert {item["token"] for item in summary["suspicious_references"]} == {"label_", "tb_"}
