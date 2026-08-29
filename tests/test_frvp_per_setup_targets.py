from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data.labeling.frvp_labeling_engine import (
    _materialize_frvp_targets,
    build_frvp_setup_diagnostic_report,
)
from frvp.target_lanes import (
    FRVP_SETUP_TARGET_COLUMNS,
    FRVP_TARGET_COLUMNS,
    pooled_target_family,
    setup_target_family,
)
from frvp.pipelines.es_primary_phase04 import _setup_type_from_target_name
from model_training.ote_training.ote_xgboost_pipeline import (
    OTETrainingConfig,
    resolve_calibration_method,
)
from preprocessing.config import PreprocessingConfig
from preprocessing.feature_selection import discover_targets
from preprocessing.pipeline import FeaturePreprocessingPipeline
from scripts.build_frvp_candidate_registry import _default_promotion_reason, _infer_model_id


def _event(
    *,
    index: int,
    setup_type: int,
    direction: str,
    outcome: str,
    excluded: bool = False,
    quality: float = 0.8,
    sample_weight: float = 0.6,
    htf_match: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        swing_index=index,
        setup_type=setup_type,
        event_direction=direction,
        target_family=pooled_target_family(setup_type),
        tb_outcome=outcome,
        excluded=excluded,
        label_quality=quality,
        sample_weight=sample_weight,
        htf_match_30m=htf_match,
        htf_match_1h=False,
    )


def test_setup_target_contract_uses_family_rich_names() -> None:
    assert setup_target_family(1) == "frvp_reversal_setup1"
    assert setup_target_family(2) == "frvp_continuation_setup2"
    assert setup_target_family(4) == "frvp_reversal_setup4"
    assert setup_target_family(5) == "frvp_continuation_setup5"
    assert len(FRVP_SETUP_TARGET_COLUMNS) == 12
    assert len(FRVP_TARGET_COLUMNS) == 18
    assert "label_long_frvp_reversal_setup1" in FRVP_TARGET_COLUMNS
    assert "label_short_frvp_continuation_setup5" in FRVP_TARGET_COLUMNS


def test_materialized_setup_targets_are_isolated_and_preserve_pooled_controls() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=4, freq="5min"),
        }
    )
    events = [
        _event(index=0, setup_type=1, direction="long", outcome="tp", sample_weight=0.55, htf_match=True),
        _event(index=1, setup_type=6, direction="long", outcome="sl", sample_weight=0.45),
        _event(index=2, setup_type=4, direction="short", outcome="tp", excluded=True, sample_weight=0.35),
        _event(index=3, setup_type=2, direction="short", outcome="tp", sample_weight=0.65),
    ]
    concurrency = {
        ("long", "frvp_reversal"): np.array([2, 2, 0, 0], dtype=np.int32),
        ("short", "frvp_reversal"): np.array([0, 0, 1, 0], dtype=np.int32),
        ("short", "frvp_continuation"): np.array([0, 0, 0, 1], dtype=np.int32),
    }

    labeled = _materialize_frvp_targets(market, events, concurrency)

    reversal_setup_labels = labeled.loc[
        :, [
            "label_long_frvp_reversal_setup1",
            "label_long_frvp_reversal_setup4",
            "label_long_frvp_reversal_setup6",
        ]
    ].sum(axis=1)
    pd.testing.assert_series_equal(
        labeled["label_long_frvp_reversal"],
        reversal_setup_labels.astype(np.int8),
        check_names=False,
    )
    assert labeled.iloc[0]["label_long_frvp_reversal_setup1"] == 1
    assert labeled.iloc[1]["label_long_frvp_reversal_setup1"] == 0
    assert bool(labeled.iloc[1]["exclude_long_frvp_reversal_setup1"]) is True
    assert bool(labeled.iloc[1]["exclude_long_frvp_reversal_setup6"]) is False
    assert labeled.iloc[0]["sample_weight_long_frvp_reversal_setup1"] == labeled.iloc[0][
        "sample_weight_long_frvp_reversal"
    ]
    assert labeled.iloc[0]["concurrency_long_frvp_reversal_setup1"] == 2
    assert labeled.iloc[0]["htf_confluence_long_frvp_reversal_setup1"] == 1

    assert bool(labeled.iloc[2]["exclude_short_frvp_reversal_setup4"]) is True
    assert labeled.iloc[2]["label_short_frvp_reversal_setup4"] == 0
    assert labeled.iloc[3]["label_short_frvp_continuation_setup2"] == 1
    assert bool(labeled.iloc[3]["exclude_short_frvp_continuation_setup2"]) is False


def test_preprocessing_discovers_setup_target_helpers() -> None:
    target = "label_long_frvp_reversal_setup4"
    frame = pd.DataFrame(
        columns=[
            target,
            "sample_weight_long_frvp_reversal_setup4",
            "label_quality_long_frvp_reversal_setup4",
            "exclude_long_frvp_reversal_setup4",
            "neg_ok_long_frvp_reversal_setup4",
        ]
    )

    specs = discover_targets(frame, PreprocessingConfig(target_columns=[target]))

    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "long_frvp_reversal_setup4"
    assert spec.label_kind == "frvp_reversal_setup4"
    assert spec.sample_weight_column == "sample_weight_long_frvp_reversal_setup4"
    assert spec.quality_column == "label_quality_long_frvp_reversal_setup4"
    assert spec.exclude_column == "exclude_long_frvp_reversal_setup4"
    assert spec.safe_negative_column == "neg_ok_long_frvp_reversal_setup4"


def test_preprocessing_materializes_setup_specific_prepared_directory(tmp_path: Path) -> None:
    rows = 80
    target_name = "long_frvp_reversal_setup1"
    target_column = f"label_{target_name}"
    event_mask = np.arange(rows) % 2 == 0
    positive_mask = np.arange(rows) % 4 == 0
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-02T14:30:00Z", periods=rows, freq="5min"),
            "source_row_idx": np.arange(rows, dtype=np.int64),
            target_column: positive_mask.astype(np.int8),
            f"sample_weight_{target_name}": np.where(event_mask, 1.5, 1.0),
            f"label_quality_{target_name}": np.where(event_mask, 0.75, 0.0),
            f"exclude_{target_name}": ~event_mask,
            f"neg_ok_{target_name}": event_mask,
            f"htf_confluence_{target_name}": (np.arange(rows) % 8 == 0).astype(np.int8),
            "warmup_mask": False,
            "frvp_dist_poc_session_atr": np.sin(np.arange(rows) / 5.0),
            "frvp_day_type": (np.arange(rows) % 3).astype(float),
            "noise_feature": np.cos(np.arange(rows) / 7.0),
        }
    )
    dataset_path = tmp_path / "frvp_setup_targets.csv"
    metadata_path = tmp_path / "frvp_setup_targets.metadata.json"
    frame.to_csv(dataset_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": [
                    "frvp_dist_poc_session_atr",
                    "frvp_day_type",
                    "noise_feature",
                ],
                "timezone_contract": {"canonical_timezone": "UTC"},
                "config": {"drop_warmup_rows": False, "warmup_rows": 0, "fillna_numeric": False},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "prepared"

    summary = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=[target_column],
            min_usable_rows=20,
            min_train_rows=10,
            min_positive_samples=5,
            top_n_features=3,
        )
    ).run(dataset_path, output_root, metadata_path=metadata_path)

    assert set(summary["targets"]) == {target_name}
    report = json.loads((output_root / target_name / "report.json").read_text(encoding="utf-8"))
    assert report["target_construction"]["source_target_column"] == target_column
    assert report["row_counts"]["rows_usable"] == int(event_mask.sum())
    assert (output_root / target_name / "train.csv").exists()
    assert (output_root / target_name / "val.csv").exists()
    assert (output_root / target_name / "test.csv").exists()


def test_setup_reporting_keeps_thin_lanes_advisory() -> None:
    report = build_frvp_setup_diagnostic_report(pd.DataFrame(), [])

    assert len(report) == 12
    assert set(report["setup_type"]) == {1, 2, 3, 4, 5, 6}
    assert set(report["sample_weight_scope"]) == {"pooled_family"}
    assert int(report["events"].sum()) == 0


def test_setup_target_names_preserve_training_and_registry_semantics() -> None:
    config = OTETrainingConfig(calibration_method="platt")

    assert resolve_calibration_method(config, "long_frvp_continuation_setup2") == "none"
    assert resolve_calibration_method(config, "long_frvp_reversal_setup1") == "platt"
    assert _infer_model_id("long_frvp_reversal_setup1", "xgboost") == "frvp_long_reversal_setup1_xgb_v1"
    assert _infer_model_id("short_frvp_continuation_setup5", "tcn") == "frvp_short_continuation_setup5_tcn_v1"
    assert "setup-specific" in _default_promotion_reason("long_frvp_reversal_setup4")
    assert _setup_type_from_target_name("long_frvp_reversal_setup6") == 6
    assert _setup_type_from_target_name("long_frvp_reversal") is None
