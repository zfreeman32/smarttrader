from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from frvp.pipelines.es_primary_phase04 import (
    _default_phase03_labeling_params,
    _load_existing_feature_dataset,
    _make_json_safe,
    _phase02_output_paths,
    _phase01_status,
    _phase02_status,
    _phase03_status,
    _phase04_status,
    _write_json,
)


def test_phase04_pipeline_json_safe_serializes_paths_and_numpy_scalars(tmp_path: Path) -> None:
    payload = {
        "config": {
            "input_path": Path("data/futures_data/ES-5m-tagged.csv"),
            "output_root": tmp_path / "artifacts",
        },
        "rows": np.int64(42),
        "passed": np.bool_(True),
        "nan_metric": np.float64(np.nan),
    }

    report_path = tmp_path / "phase04_gate_report.json"
    _write_json(report_path, payload)

    written = json.loads(report_path.read_text(encoding="utf-8"))
    safe_payload = _make_json_safe(payload)

    assert written == safe_payload
    assert written["config"]["input_path"] == str(Path("data/futures_data/ES-5m-tagged.csv"))
    assert written["config"]["output_root"] == str(tmp_path / "artifacts")
    assert written["rows"] == 42
    assert written["passed"] is True
    assert written["nan_metric"] is None


def test_phase02_status_uses_key_feature_mi_not_only_top_ranked_rows() -> None:
    audit = {
        "duplicate_time_columns": [],
        "roll_lineage_columns_present": [],
        "mi_rankings": {
            key: [{"feature": "frvp_dist_poc_session_atr", "mutual_information": 0.0}]
            for key in (
                "long_frvp_reversal",
                "short_frvp_reversal",
                "long_frvp_continuation",
                "short_frvp_continuation",
            )
        },
        "key_feature_mi": {
            key: {
                "frvp_distance_max_mi": 0.05,
                "frvp_open_type_mi": 0.01,
                "htf_confluence_max_mi": 0.02,
            }
            for key in (
                "long_frvp_reversal",
                "short_frvp_reversal",
                "long_frvp_continuation",
                "short_frvp_continuation",
            )
        },
    }
    preprocessing_summary = {"output_dir": "", "targets": []}

    status, reasons = _phase02_status(audit, preprocessing_summary)

    assert status == "pass"
    assert reasons == ["Merged Phase 2 dataset is clean, target-aware, and passes the correlation-pruning preview."]


def test_phase01_status_treats_setup4_sparsity_as_watch_item() -> None:
    integrity = {
        "missing_required_columns": [],
        "macro_calendar_contract": {"cpi_historical_backfill_complete": True},
    }
    setup_rates = pd.DataFrame(
        [
            {"setup_type": 1, "setup_side": -1, "avg_fires_per_session_side": 0.70, "within_target_band": True},
            {"setup_type": 4, "setup_side": -1, "avg_fires_per_session_side": 0.18, "within_target_band": False},
            {"setup_type": 4, "setup_side": 1, "avg_fires_per_session_side": 0.16, "within_target_band": False},
        ]
    )

    status, reasons, watch_items = _phase01_status(integrity, setup_rates)

    assert status == "pass"
    assert reasons == ["Canonical tagged input decision resolved in code and session calendar overrides are active."]
    assert len(watch_items) == 1
    assert "Setup 4 remains below the historical 0.5-2.0 fires/session band" in watch_items[0]


def test_phase03_status_treats_quality_target_as_watch_item(monkeypatch) -> None:
    monkeypatch.setattr(
        "frvp.pipelines.es_primary_phase04.macro_calendar_contract",
        lambda: {"cpi_historical_backfill_complete": True},
    )
    diagnostics = {
        "todo_macro_flags_unavailable": False,
        "todo_halfday_calendar_overrides_pending": False,
        "halfday_detection_uses_empirical_fallback": False,
    }
    report = pd.DataFrame(
        [
            {
                "direction": "long",
                "family": "frvp_continuation",
                "events_per_year": 600.0,
                "base_rate_pct": 48.0,
                "quality_mean": 0.61,
                "gate_events_per_year": True,
                "gate_base_rate": True,
                "quality_target_mean": 0.65,
                "quality_target_met": False,
                "quality_target_is_blocking": False,
                "gate_roll_spanning_excluded": True,
                "gate_not_excessively_clustered": True,
            }
        ]
    )

    status, reasons, watch_items = _phase03_status(diagnostics, report)

    assert status == "pass"
    assert len(reasons) == 1
    assert "quality target is advisory" in reasons[0].lower()
    assert watch_items == ["long_frvp_continuation: quality_mean=0.610 below advisory target 0.65"]


def test_load_existing_feature_dataset_reads_datetime_and_declared_feature_columns(tmp_path: Path) -> None:
    feature_csv = tmp_path / "features.csv"
    feature_metadata = tmp_path / "features.metadata.json"

    pd.DataFrame(
        {
            "datetime": ["2024-01-02T14:30:00Z", "2024-01-02T14:35:00Z"],
            "frvp_dist_poc_session_atr": [0.1, 0.2],
            "frvp_open_type": [0, 1],
            "ignored_column": [99, 100],
        }
    ).to_csv(feature_csv, index=False)
    feature_metadata.write_text(
        json.dumps({"feature_columns": ["frvp_dist_poc_session_atr", "frvp_open_type"]}, indent=2),
        encoding="utf-8",
    )

    dataset, metadata = _load_existing_feature_dataset(feature_csv, feature_metadata)

    assert list(dataset.columns) == ["datetime", "frvp_dist_poc_session_atr", "frvp_open_type"]
    dtype_text = str(dataset["datetime"].dtype)
    assert dtype_text.startswith("datetime64[")
    assert "UTC" in dtype_text
    assert dataset["frvp_dist_poc_session_atr"].dtype == np.float32
    assert dataset["frvp_open_type"].dtype == np.float32
    assert metadata["feature_columns"] == ["frvp_dist_poc_session_atr", "frvp_open_type"]


def test_phase04_status_reports_skipped_when_pipeline_stops_early() -> None:
    status, reasons = _phase04_status(None, None)

    assert status == "skipped"
    assert reasons == ["Phase 4 was intentionally skipped by the requested stop-after setting."]


def test_phase02_output_paths_support_gzip() -> None:
    feature_path, cleaned_path = _phase02_output_paths(Path("artifacts/example/phase02"), compression="gzip")

    assert feature_path == Path("artifacts/example/phase02/es_primary_frvp_features_full.csv.gz")
    assert cleaned_path == Path("artifacts/example/phase02/es_primary_frvp_phase04_dataset.csv.gz")


def test_default_phase03_labeling_params_include_winning_continuation_overrides() -> None:
    params = _default_phase03_labeling_params(instrument="es")

    assert params.continuation_profit_atr == 1.0
    assert params.continuation_setup3_long_profit_atr == 0.95
    assert params.continuation_setup5_long_profit_atr == 0.90
    assert params.continuation_setup5_short_profit_atr == 0.85
    assert params.enable_failed_auction_labels is True
    assert params.enable_reversal_past_htf_confluence_gate is True
