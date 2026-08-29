from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.builder import FeatureDatasetBuilder  # noqa: E402
from features.config import FeatureBuilderConfig  # noqa: E402
from features.feature_sets.ict_context import build_ict_context  # noqa: E402
from ict.config.instruments import get_ict_base_instrument_config, get_ict_instrument_config  # noqa: E402
from ict.feature_sets.ict_context import build_ict_context_features  # noqa: E402
from ict.labeling.ict_labeling_engine import ICT_LABEL_TARGET_COLUMNS, get_ict_helper_column_names  # noqa: E402
from ict.pipelines.layout import build_ict_artifact_layout  # noqa: E402
from ict.reports.diagnostics import summarize_setup_output  # noqa: E402
from ict.setups.detector import detect_ict_setups, summarize_setup_fire_rates  # noqa: E402


def _sample_market_frame(rows: int = 160) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    timestamps = pd.date_range("2024-01-03 09:30:00", periods=rows, freq="5min", tz="UTC")
    trend = 5000.0 + (index * 0.25)
    cycle = np.sin(index / 9.0) * 1.25
    open_ = trend + cycle
    close = trend + np.cos(index / 7.0) * 1.05
    high = np.maximum(open_, close) + 0.75 + (index % 3) * 0.10
    low = np.minimum(open_, close) - 0.75 - (index % 4) * 0.10
    volume = 1000.0 + ((index.astype(int) % 12) * 40.0)
    return pd.DataFrame(
        {
            "datetime": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_ict_instrument_config_maps_mes_to_es_session_profile() -> None:
    mes = get_ict_instrument_config("mes")
    es_base = get_ict_base_instrument_config("mes")

    assert mes.instrument == "mes"
    assert mes.training_instrument == "es"
    assert mes.execution_instrument == "mes"
    assert mes.tick_size == 0.25
    assert mes.tick_value == 1.25
    assert es_base.instrument == "es"
    assert es_base.ib_minutes == 60


def test_ict_artifact_layout_builds_expected_phase_paths(tmp_path: Path) -> None:
    layout = build_ict_artifact_layout("ict_es_primary_20260711", base_dir=tmp_path, ensure_directories=True)

    assert layout.root == tmp_path / "ict_es_primary_20260711"
    assert layout.phase01_scan.exists()
    assert layout.phase07_backtests.exists()
    assert layout.as_dict()["phase04_prepared"].name == "phase04_prepared"


def test_ict_context_shim_matches_package_native_output() -> None:
    frame = _sample_market_frame()
    config = FeatureBuilderConfig(
        feature_sets=["ict_context"],
        warmup_rows=0,
        drop_warmup_rows=False,
        fillna_numeric=False,
        enable_lags=False,
        enable_rolling_stats=False,
        enable_zscores=False,
        enable_winsorization=False,
        enable_percentile_ranks=False,
        enable_atr_normalization=False,
        enable_sigma_normalization=False,
        enable_interactions=False,
    )

    shim_output = build_ict_context(frame, config)
    package_output = build_ict_context_features(frame, config)

    assert shim_output.columns.tolist() == package_output.columns.tolist()
    np.testing.assert_allclose(
        shim_output.to_numpy(dtype=np.float64),
        package_output.to_numpy(dtype=np.float64),
        equal_nan=True,
    )


def test_ict_interactions_recipe_build_emits_expected_columns() -> None:
    frame = _sample_market_frame()
    config = FeatureBuilderConfig.from_recipe(ROOT / "features" / "recipes" / "ict_es_meta.json")
    config.instrument = "es"
    config.transform_workers = 1
    config.drop_warmup_rows = False

    dataset, metadata = FeatureDatasetBuilder(config).build(frame)

    expected_columns = {
        "dist_to_bull_fvg_atr",
        "ict_bull_sweep_plus_fvg",
        "ict_sweep_plus_choch",
        "ict_bull_zone_in_discount",
    }
    assert expected_columns.issubset(dataset.columns)
    assert "ict_interactions" in metadata["feature_sets"]


def test_detect_ict_setups_phase1_returns_stable_schema_without_fires() -> None:
    frame = _sample_market_frame(rows=24)
    frame["session_date"] = pd.Timestamp("2024-01-03")

    result = detect_ict_setups(frame)

    assert len(result) == len(frame)
    assert result["fired"].sum() == 0
    assert set(
        [
            "event_time",
            "fired",
            "setup_type",
            "setup_family",
            "setup_side",
            "confidence",
            "anchor_level",
            "reference_level",
            "reference_level_type",
            "sweep_type",
        ]
    ).issubset(result.columns)
    assert set(result["setup_type"].dropna().unique()) == {"none"}
    assert set(result["setup_family"].dropna().unique()) == {"none"}
    assert set(result["setup_side"].dropna().astype(int).unique()) == {0}


def test_ict_setup_summaries_and_labeling_contracts_are_exposed() -> None:
    frame = _sample_market_frame(rows=12)
    setup_output = detect_ict_setups(frame)
    fire_rate_summary = summarize_setup_fire_rates(setup_output)
    diagnostics = summarize_setup_output(setup_output)
    helper_columns = get_ict_helper_column_names("long", "ict_reversal")

    assert fire_rate_summary.empty
    assert diagnostics["rows"] == 12
    assert diagnostics["fired_rows"] == 0
    assert ICT_LABEL_TARGET_COLUMNS == (
        "label_long_ict_reversal",
        "label_short_ict_reversal",
        "label_long_ict_continuation",
        "label_short_ict_continuation",
        "label_long_ict_meta",
        "label_short_ict_meta",
    )
    assert helper_columns["sample_weight"] == "sample_weight_long_ict_reversal"
    assert helper_columns["exclude"] == "exclude_long_ict_reversal"
