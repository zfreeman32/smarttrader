from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.labeling.frvp_labeling_engine import (
    FRVP_CONTINUATION_FAMILY,
    FRVP_REVERSAL_FAMILY,
    FRVP_QUALITY_TARGET,
    FRVPLabelingParams,
    build_frvp_diagnostic_report,
    build_frvp_labels,
    build_frvp_setup_diagnostic_report,
    frvp_events_to_frame,
)
from features.builder import FeatureDatasetBuilder
from features.config import FeatureBuilderConfig
from features.io import save_dataset, standardize_market_frame
from frvp.calendars.macro import macro_calendar_contract
from frvp.feature_sets.dataset_audit import summarize_frvp_feature_dataset
from frvp.setups.detector import summarize_setup_fire_rates
from preprocessing.backend_attribution import BackendAttributionConfig, run_backend_attribution
from preprocessing.config import (
    FRVP_DIRECT_TARGET_COLUMNS,
    FRVP_META_TARGET_COLUMNS,
    FRVP_POOLED_DIRECT_TARGET_COLUMNS,
    FRVP_SETUP_TARGET_COLUMNS,
    FRVP_TARGET_COLUMNS,
    PreprocessingConfig,
)
from preprocessing.pipeline import FeaturePreprocessingPipeline


DEFAULT_INPUT = Path("data/futures_data/ES-5m-tagged.csv")
DEFAULT_OUTPUT_ROOT = Path("artifacts/frvp_es_primary")
DEFAULT_RECIPE = Path("features/recipes/frvp_meta.json")
VALID_STOP_PHASES = ("phase01", "phase02", "phase03", "phase04")
FRVP_HELPER_PREFIXES = (
    "label_",
    "label_quality_",
    "sample_weight_",
    "exclude_",
    "neg_ok_",
    "concurrency_",
    "htf_confluence_",
)
PHASE2_FORBIDDEN_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ts_event",
    "symbol",
    "instrument_id",
    "contract_id",
    "contract_symbol",
    "contract_expiration",
    "is_roll_boundary",
    "bars_since_roll",
    "market_day_close",
    "market_day_index",
    "in_roll_bracket",
}
PHASE01_OPTIONAL_SETUP_TYPES = frozenset({4})


@dataclass(frozen=True)
class ESPrimaryPhase04Config:
    input_path: Path = DEFAULT_INPUT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    recipe_path: Path = DEFAULT_RECIPE
    feature_csv_path: Path | None = None
    feature_metadata_path: Path | None = None
    instrument: str = "es"
    transform_workers: int = 1
    backend_max_features: int = 160
    attribution_max_rows: int = 100_000
    top_n_features: int = 25
    min_usable_rows: int = 250
    min_train_rows: int = 100
    min_positive_samples: int = 25
    stop_after: str = "phase04"
    skip_phase02_csvs: bool = False
    phase02_compression: str = "none"


def _default_phase03_labeling_params(*, instrument: str) -> FRVPLabelingParams:
    return FRVPLabelingParams(
        instrument=instrument,
        auto_scale_bar_counts=False,
        continuation_profit_atr=1.0,
        continuation_setup3_long_profit_atr=0.95,
        continuation_setup5_long_profit_atr=0.90,
        continuation_setup5_short_profit_atr=0.85,
        enable_failed_auction_labels=True,
        enable_reversal_past_htf_confluence_gate=True,
        setup6_reversal_cooldown_bars=6,
        setup1_past_htf_30m_bars=48,
        setup1_past_htf_1h_bars=96,
        setup1_short_past_htf_30m_bars=36,
        setup1_short_past_htf_1h_bars=72,
        setup6_past_htf_30m_bars=48,
        setup6_past_htf_1h_bars=96,
    )


def run(config: ESPrimaryPhase04Config) -> dict[str, Any]:
    stop_after = _normalize_stop_phase(config.stop_after)
    if config.skip_phase02_csvs and stop_after == "phase04":
        raise ValueError("--skip-phase02-csvs is only supported when --stop-after is phase03 or earlier.")
    phase02_compression = _normalize_phase02_compression(config.phase02_compression)

    output_root = Path(config.output_root)
    phase01_dir = output_root / "phase01"
    phase02_dir = output_root / "phase02"
    phase03_dir = output_root / "phase03"
    phase04_dir = output_root / "phase04"
    prepared_root = phase04_dir / "prepared"
    for path in (phase01_dir, phase02_dir, phase03_dir, phase04_dir, prepared_root):
        path.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(config.input_path)
    standardized = standardize_market_frame(raw)
    integrity = _phase01_integrity_report(standardized, config=config)
    _write_json(phase01_dir / "integrity_report.json", integrity)
    setup_rates = summarize_setup_fire_rates(
        standardized,
        instrument=config.instrument,
        output_path=phase01_dir / "setup_fire_rates.csv",
    )

    if stop_after == "phase01":
        gate_report = _build_gate_report(
            config=config,
            integrity=integrity,
            setup_rates=setup_rates,
            phase02_audit=None,
            phase03_diag=None,
            phase03_report=None,
            preprocessing_summary=None,
            backend_summary=None,
            artifacts={
                "phase01_integrity": str(phase01_dir / "integrity_report.json"),
                "phase01_setup_fire_rates": str(phase01_dir / "setup_fire_rates.csv"),
            },
        )
        _write_json(output_root / "phase04_gate_report.json", gate_report)
        return _make_json_safe(gate_report)

    feature_csv: Path | None = None
    feature_metadata_path: Path | None = None
    feature_output_path, cleaned_output_path = _phase02_output_paths(
        phase02_dir,
        compression=phase02_compression,
    )
    if config.feature_csv_path is not None and config.feature_metadata_path is not None:
        feature_csv = Path(config.feature_csv_path)
        feature_metadata_path = Path(config.feature_metadata_path)
        feature_dataset, feature_metadata = _load_existing_feature_dataset(
            feature_csv,
            feature_metadata_path,
        )
    else:
        feature_config = FeatureBuilderConfig.from_recipe(config.recipe_path)
        feature_config.instrument = config.instrument
        feature_config.transform_workers = max(int(config.transform_workers), 1)
        feature_config.optimize_feature_dtypes = True
        feature_builder = FeatureDatasetBuilder(feature_config)
        feature_dataset, feature_metadata = feature_builder.build(standardized.copy(), source_path=config.input_path)
        if not config.skip_phase02_csvs:
            feature_csv, feature_metadata_path = save_dataset(
                feature_dataset,
                feature_metadata,
                feature_output_path,
            )

    labeling_input = standardized.copy()
    labeling_input = labeling_input.set_index("datetime", drop=False).sort_index()
    labels, label_diag, events = build_frvp_labels(
        labeling_input,
        params=_default_phase03_labeling_params(instrument=config.instrument),
        verbose=True,
    )
    labels_reset = labels.reset_index()
    if "datetime" not in labels_reset.columns:
        labels_reset = labels_reset.rename(columns={labels_reset.columns[0]: "datetime"})
    labels_reset["datetime"] = pd.to_datetime(labels_reset["datetime"], errors="coerce", utc=True)
    label_csv = phase03_dir / "es_primary_frvp_labels.csv"
    labels_reset.to_csv(label_csv, index=False)
    _write_json(phase03_dir / "labeling_diagnostics.json", label_diag)
    events_frame = frvp_events_to_frame(events)
    events_csv = phase03_dir / "es_primary_frvp_events.csv"
    events_frame.to_csv(events_csv, index=False)
    diagnostic_report = build_frvp_diagnostic_report(labels, events)
    diagnostic_report_csv = phase03_dir / "es_primary_frvp_diagnostic_report.csv"
    diagnostic_report.to_csv(diagnostic_report_csv, index=False)
    setup_diagnostic_report = build_frvp_setup_diagnostic_report(labels, events)
    setup_diagnostic_report_csv = phase03_dir / "es_primary_frvp_setup_diagnostic_report.csv"
    setup_diagnostic_report.to_csv(setup_diagnostic_report_csv, index=False)

    feature_columns = ["datetime", *[str(column) for column in feature_metadata.get("feature_columns", [])]]
    feature_columns = [column for column in feature_columns if column in feature_dataset.columns]
    if list(feature_dataset.columns) == feature_columns:
        feature_frame = feature_dataset
    else:
        feature_frame = feature_dataset.loc[:, feature_columns]

    del raw
    del standardized
    del labels
    del events
    del events_frame
    gc.collect()

    merged = feature_frame.merge(labels_reset, on="datetime", how="inner", validate="one_to_one")
    del feature_frame
    del feature_dataset
    gc.collect()
    cleaned_dataset, cleaned_metadata = _build_clean_phase2_dataset(
        merged,
        feature_metadata=feature_metadata,
        source_path=config.input_path,
        feature_source_csv=feature_csv,
        feature_source_metadata=feature_metadata_path,
    )
    cleaned_csv: Path | None = None
    cleaned_metadata_path: Path | None = None
    if not config.skip_phase02_csvs:
        cleaned_csv, cleaned_metadata_path = save_dataset(
            cleaned_dataset,
            cleaned_metadata,
            cleaned_output_path,
        )

    phase02_audit = summarize_frvp_feature_dataset(
        cleaned_dataset,
        output_path=phase02_dir / "phase2_feature_audit.json",
    )

    if stop_after == "phase03":
        gate_report = _build_gate_report(
            config=config,
            integrity=integrity,
            setup_rates=setup_rates,
            phase02_audit=phase02_audit,
            phase03_diag=label_diag,
            phase03_report=diagnostic_report,
            preprocessing_summary=None,
            backend_summary=None,
            artifacts={
                "phase01_integrity": str(phase01_dir / "integrity_report.json"),
                "phase01_setup_fire_rates": str(phase01_dir / "setup_fire_rates.csv"),
                "phase02_feature_csv": str(feature_csv) if feature_csv is not None else None,
                "phase02_clean_dataset_csv": str(cleaned_csv) if cleaned_csv is not None else None,
                "phase02_audit_json": str(phase02_dir / "phase2_feature_audit.json"),
                "phase03_labels_csv": str(label_csv),
                "phase03_events_csv": str(events_csv),
                "phase03_diag_json": str(phase03_dir / "labeling_diagnostics.json"),
                "phase03_diag_report_csv": str(diagnostic_report_csv),
                "phase03_setup_diag_report_csv": str(setup_diagnostic_report_csv),
            },
        )
        _write_json(output_root / "phase04_gate_report.json", gate_report)
        return _make_json_safe(gate_report)

    preprocessing_summary = FeaturePreprocessingPipeline(
        PreprocessingConfig(
            target_columns=list(FRVP_TARGET_COLUMNS),
            load_time_column=False,
            min_usable_rows=int(config.min_usable_rows),
            min_train_rows=int(config.min_train_rows),
            min_positive_samples=int(config.min_positive_samples),
            top_n_features=int(config.top_n_features),
        )
    ).run(
        cleaned_csv,
        prepared_root,
        metadata_path=cleaned_metadata_path,
    )
    preprocessing_summary["output_dir"] = str(prepared_root)

    backend_summary = run_backend_attribution(
        BackendAttributionConfig(
            prepared_root=str(prepared_root),
            targets=[name.removeprefix("label_") for name in FRVP_TARGET_COLUMNS],
            backends=["xgboost"],
            max_features=int(config.backend_max_features),
            attribution_max_rows=int(config.attribution_max_rows),
            top_n_features=int(config.top_n_features),
        )
    )

    gate_report = _build_gate_report(
        config=config,
        integrity=integrity,
        setup_rates=setup_rates,
        phase02_audit=phase02_audit,
        phase03_diag=label_diag,
        phase03_report=diagnostic_report,
        preprocessing_summary=preprocessing_summary,
        backend_summary=backend_summary,
        artifacts={
            "phase01_integrity": str(phase01_dir / "integrity_report.json"),
            "phase01_setup_fire_rates": str(phase01_dir / "setup_fire_rates.csv"),
            "phase02_feature_csv": str(feature_csv),
            "phase02_clean_dataset_csv": str(cleaned_csv),
            "phase02_audit_json": str(phase02_dir / "phase2_feature_audit.json"),
            "phase03_labels_csv": str(label_csv),
            "phase03_events_csv": str(events_csv),
            "phase03_diag_json": str(phase03_dir / "labeling_diagnostics.json"),
            "phase03_diag_report_csv": str(diagnostic_report_csv),
            "phase03_setup_diag_report_csv": str(setup_diagnostic_report_csv),
            "phase04_prepared_summary": str(prepared_root / "summary.json"),
            "phase04_backend_summary": str(prepared_root / "backend_attribution_summary.json"),
        },
    )
    _write_json(output_root / "phase04_gate_report.json", gate_report)
    return _make_json_safe(gate_report)


def _phase01_integrity_report(
    df: pd.DataFrame,
    *,
    config: ESPrimaryPhase04Config,
) -> dict[str, Any]:
    required_columns = {
        "ts_event",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "instrument_id",
        "contract_symbol",
        "contract_expiration",
        "is_roll_boundary",
        "bars_since_roll",
        "market_day_close",
        "market_day_index",
        "in_roll_bracket",
    }
    return {
        "phase": "0_1",
        "canonical_raw_input": str(config.input_path),
        "canonical_raw_input_decision": "ES-5m-tagged.csv is the canonical FRVP ES-primary upstream source.",
        "decision_rationale": (
            "The tagged continuous file preserves contract lineage, roll context, and canonical event timestamps "
            "needed by continuity-aware FRVP feature building while remaining directly consumable by the existing pipeline."
        ),
        "rows": int(len(df)),
        "datetime_start": _iso_or_none(df["datetime"].min()) if "datetime" in df.columns else None,
        "datetime_end": _iso_or_none(df["datetime"].max()) if "datetime" in df.columns else None,
        "required_columns_present": sorted(required_columns.intersection(df.columns)),
        "missing_required_columns": sorted(required_columns.difference(df.columns)),
        "macro_calendar_contract": macro_calendar_contract(),
        "session_calendar_source": "offline_rule_based_us_equity_calendar",
    }


def _build_clean_phase2_dataset(
    merged: pd.DataFrame,
    *,
    feature_metadata: dict[str, Any],
    source_path: Path,
    feature_source_csv: Path | None,
    feature_source_metadata: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    generated_features = [
        column
        for column in feature_metadata.get("feature_columns", [])
        if column in merged.columns and column not in PHASE2_FORBIDDEN_COLUMNS
    ]
    helper_columns = [
        column
        for column in merged.columns
        if column.startswith(FRVP_HELPER_PREFIXES)
    ]
    carry_through_columns = [
        column for column in helper_columns if column.startswith("htf_confluence_")
    ]
    label_helper_columns = [column for column in helper_columns if column not in carry_through_columns]

    keep = ["datetime"] + generated_features + carry_through_columns + label_helper_columns + ["warmup_mask"]
    keep = [column for column in keep if column in merged.columns]
    cleaned = merged.loc[:, list(dict.fromkeys(keep))].copy()

    metadata = {
        "feature_columns": generated_features,
        "timezone_contract": feature_metadata.get("timezone_contract", {}),
        "source_path": str(feature_source_csv) if feature_source_csv is not None else None,
        "source_metadata_file": str(feature_source_metadata) if feature_source_metadata is not None else None,
        "upstream_source_path": str(source_path),
        "upstream_bar_timestamp_semantics": "bar_open",
        "config": {
            "drop_warmup_rows": False,
            "warmup_rows": 0,
            "fillna_numeric": False,
        },
        "phase02_contract": {
            "datetime_column": "datetime",
            "duplicate_time_columns_removed": ["ts_event", "timestamp"],
            "raw_price_columns_removed": ["open", "high", "low", "close", "volume"],
            "roll_lineage_columns_removed": sorted(PHASE2_FORBIDDEN_COLUMNS.intersection(merged.columns)),
            "target_context_columns": carry_through_columns,
        },
    }
    return cleaned, metadata


def _build_gate_report(
    *,
    config: ESPrimaryPhase04Config,
    integrity: dict[str, Any],
    setup_rates: pd.DataFrame,
    phase02_audit: dict[str, Any] | None,
    phase03_diag: dict[str, Any] | None,
    phase03_report: pd.DataFrame | None,
    preprocessing_summary: dict[str, Any] | None,
    backend_summary: dict[str, Any] | None,
    artifacts: dict[str, str | None],
) -> dict[str, Any]:
    phase01_status, phase01_reasons, phase01_watch_items = _phase01_status(integrity, setup_rates)
    phase02_status, phase02_reasons = _phase02_status(phase02_audit, preprocessing_summary)
    phase03_status, phase03_reasons, phase03_watch_items = _phase03_status(phase03_diag, phase03_report)
    phase04_status, phase04_reasons = _phase04_status(preprocessing_summary, backend_summary)

    return {
        "config": _make_json_safe(asdict(config)),
        "gate_policy": {
            "phase01_optional_setups": sorted(PHASE01_OPTIONAL_SETUP_TYPES),
            "phase01_optional_setup_policy": (
                "Setup 4 remains included and is documented as structurally weaker/sparser; "
                "off-band fire rate alone is not a blocking failure."
            ),
            "phase03_quality_target_mean": float(FRVP_QUALITY_TARGET),
            "phase03_quality_target_is_blocking": False,
            "phase03_quality_target_policy": (
                "Family-level quality below the target is tracked as a watch item, not a blocking "
                "failure, unless future policy explicitly promotes it to a hard gate."
            ),
        },
        "artifacts": artifacts,
        "phase01": {"status": phase01_status, "reasons": phase01_reasons, "watch_items": phase01_watch_items},
        "phase02": {"status": phase02_status, "reasons": phase02_reasons, "watch_items": []},
        "phase03": {"status": phase03_status, "reasons": phase03_reasons, "watch_items": phase03_watch_items},
        "phase04": {"status": phase04_status, "reasons": phase04_reasons, "watch_items": []},
        "phase02_audit": phase02_audit,
        "phase03_diagnostics": phase03_diag,
        "phase03_report": phase03_report.to_dict(orient="records") if phase03_report is not None else None,
        "preprocessing_summary_path": artifacts.get("phase04_prepared_summary"),
        "backend_summary_path": artifacts.get("phase04_backend_summary"),
    }


def _format_setup_fire_rate_rows(rows: pd.DataFrame) -> str:
    return ", ".join(
        f"S{int(row.setup_type)} {'long' if int(row.setup_side) > 0 else 'short'} avg={float(row.avg_fires_per_session_side):.2f}"
        for row in rows.itertuples(index=False)
    )


def _phase01_status(integrity: dict[str, Any], setup_rates: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    reasons: list[str] = []
    watch_items: list[str] = []
    if integrity["missing_required_columns"]:
        reasons.append(f"missing_required_columns={integrity['missing_required_columns']}")
    if not bool(integrity["macro_calendar_contract"].get("cpi_historical_backfill_complete")):
        reasons.append("CPI historical backfill is incomplete for the validated training-window years.")
    off_band = setup_rates.loc[~setup_rates["within_target_band"]].copy()
    required_off_band = off_band.loc[~off_band["setup_type"].isin(PHASE01_OPTIONAL_SETUP_TYPES)].copy()
    optional_off_band = off_band.loc[off_band["setup_type"].isin(PHASE01_OPTIONAL_SETUP_TYPES)].copy()
    if not required_off_band.empty:
        reasons.append(
            "Required setup fire-rate outliers remain for: " + _format_setup_fire_rate_rows(required_off_band)
        )
    if not optional_off_band.empty:
        watch_items.append(
            "Setup 4 remains below the historical 0.5-2.0 fires/session band ("
            + _format_setup_fire_rate_rows(optional_off_band)
            + "). It remains included and is tracked as a documented weaker/rarer setup rather than a gating failure."
        )
    if integrity["missing_required_columns"]:
        return "fail", reasons, watch_items
    if reasons:
        return "partial", reasons, watch_items
    return "pass", ["Canonical tagged input decision resolved in code and session calendar overrides are active."], watch_items


def _phase02_status(
    audit: dict[str, Any] | None,
    preprocessing_summary: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    if audit is None:
        return "skipped", ["Phase 2 was not run because the pipeline stopped after Phase 1."]

    reasons: list[str] = []
    if audit.get("duplicate_time_columns"):
        reasons.append(f"duplicate_time_columns_present={audit['duplicate_time_columns']}")
    if audit.get("roll_lineage_columns_present"):
        reasons.append(f"roll_lineage_columns_present={audit['roll_lineage_columns_present']}")

    mi_rankings = audit.get("mi_rankings", {})
    key_feature_mi = audit.get("key_feature_mi", {})
    for key in ("long_frvp_reversal", "short_frvp_reversal", "long_frvp_continuation", "short_frvp_continuation"):
        rankings = mi_rankings.get(key, [])
        if not rankings:
            reasons.append(f"missing_mi_rankings={key}")
        feature_mi = key_feature_mi.get(key, {})
        if float(feature_mi.get("frvp_distance_max_mi") or 0.0) <= 0.0:
            reasons.append(f"non_trivial_frvp_distance_mi_missing={key}")
        if float(feature_mi.get("frvp_open_type_mi") or 0.0) <= 0.0:
            reasons.append(f"non_trivial_open_type_mi_missing={key}")
        if float(feature_mi.get("htf_confluence_max_mi") or 0.0) <= 0.0:
            reasons.append(f"non_trivial_htf_confluence_mi_missing={key}")

    if preprocessing_summary is not None:
        prepared_reports = _load_prepared_reports(preprocessing_summary)
        for target_name, report in prepared_reports.items():
            validations = {item["check"]: bool(item["passed"]) for item in report.get("validations", [])}
            if not validations.get("max_remaining_correlation", False):
                reasons.append(f"correlation_pruning_gate_failed={target_name}")

    if reasons:
        return ("fail" if any(reason.startswith("duplicate_time_columns_present") or reason.startswith("roll_lineage_columns_present") for reason in reasons) else "partial"), reasons
    if preprocessing_summary is None:
        return "pass", ["Merged Phase 2 dataset is clean and target-aware; correlation-pruning preview was skipped because the pipeline stopped before Phase 4."]
    return "pass", ["Merged Phase 2 dataset is clean, target-aware, and passes the correlation-pruning preview."]


def _phase03_status(
    diagnostics: dict[str, Any] | None,
    report: pd.DataFrame | None,
) -> tuple[str, list[str], list[str]]:
    if diagnostics is None or report is None:
        return "skipped", ["Phase 3 was not run because the pipeline stopped before labeling."], []

    reasons: list[str] = []
    watch_items: list[str] = []
    if bool(diagnostics.get("todo_macro_flags_unavailable")):
        reasons.append("macro flags are still unavailable upstream.")
    if bool(diagnostics.get("todo_halfday_calendar_overrides_pending")):
        reasons.append("half-day / early-close flags are still unavailable upstream.")
    if bool(diagnostics.get("halfday_detection_uses_empirical_fallback")):
        reasons.append("half-day exclusions still relied on the empirical fallback path.")
    if not bool(macro_calendar_contract().get("cpi_historical_backfill_complete")):
        reasons.append("CPI exclusions still lack a complete archived historical monthly schedule.")

    for row in report.itertuples(index=False):
        if not bool(row.gate_events_per_year):
            reasons.append(f"{row.direction}_{row.family}: events/year={row.events_per_year:.1f} outside 250-750 band")
        if not bool(row.gate_base_rate):
            reasons.append(f"{row.direction}_{row.family}: base_rate={row.base_rate_pct:.1f}% outside 45-60 band")
        quality_target = float(getattr(row, "quality_target_mean", FRVP_QUALITY_TARGET) or FRVP_QUALITY_TARGET)
        quality_target_met = getattr(row, "quality_target_met", None)
        if quality_target_met is None:
            quality_target_met = getattr(row, "gate_quality_mean", row.quality_mean >= quality_target)
        quality_target_is_blocking = bool(getattr(row, "quality_target_is_blocking", False))
        if not bool(quality_target_met):
            quality_message = (
                f"{row.direction}_{row.family}: quality_mean={row.quality_mean:.3f} "
                f"below advisory target {quality_target:.2f}"
            )
            if quality_target_is_blocking:
                reasons.append(quality_message)
            else:
                watch_items.append(quality_message)
        if not bool(row.gate_roll_spanning_excluded):
            reasons.append(f"{row.direction}_{row.family}: roll-spanning exclusions failed")
        if not bool(row.gate_not_excessively_clustered):
            reasons.append(f"{row.direction}_{row.family}: event clustering remains excessive")

    if reasons:
        return "partial", reasons, watch_items
    return (
        "pass",
        [
            "Phase 3 event-density, base-rate, roll-span, and clustering gates passed. "
            "The 0.65 quality target is advisory and is tracked in watch_items when missed."
        ],
        watch_items,
    )


def _phase04_status(
    preprocessing_summary: dict[str, Any] | None,
    backend_summary: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    if preprocessing_summary is None or backend_summary is None:
        return "skipped", ["Phase 4 was intentionally skipped by the requested stop-after setting."]

    reasons: list[str] = []
    input_file = str(preprocessing_summary.get("input_file") or "")
    if input_file.endswith("_phase4_audit.csv"):
        reasons.append(
            "Phase 4 used the reduced full-history audit subset derived from Phase 2 MI survivors plus mandatory FRVP/HTF columns because full-width preprocessing remained memory-blocked."
        )
    prepared_reports = _load_prepared_reports(preprocessing_summary)
    for target_name, report in prepared_reports.items():
        features = set(json.loads((Path(preprocessing_summary["output_dir"]) / target_name / "features.json").read_text(encoding="utf-8"))["features"])
        for forbidden in ("open", "high", "low", "close", "volume", "contract_symbol", "instrument_id", "is_roll_boundary"):
            if forbidden in features:
                reasons.append(f"{target_name}: leakage_feature_present={forbidden}")
        if "frvp_day_type" not in features:
            reasons.append(f"{target_name}: frvp_day_type_missing_from_prepared_features")

    for target_name, backend_payload in backend_summary["targets"].items():
        xgb = backend_payload["xgboost"]
        top10 = [row["feature"] for row in xgb.get("top_features_overall", [])[:10]]
        setup_type = _setup_type_from_target_name(target_name)
        if not any(feature.startswith("frvp_dist_") for feature in top10):
            reasons.append(f"{target_name}: no FRVP distance feature reached SHAP top-10")
        # Setup 1 and Setup 6 explicitly require an inside-value open, so
        # frvp_open_type is constant and correctly removed in those lanes.
        if setup_type not in {1, 6} and "frvp_open_type" not in top10:
            reasons.append(f"{target_name}: frvp_open_type missed SHAP top-10")
        if "frvp_day_type" not in top10:
            reasons.append(f"{target_name}: frvp_day_type missed SHAP top-10")
        if not any(feature.startswith("htf_confluence_") for feature in top10):
            reasons.append(f"{target_name}: target HTF confluence missed SHAP top-10")

    if reasons:
        return "partial", reasons
    return "pass", ["Prepared FRVP targets remain zero-special-case downstream and the backend attribution audit is directionally consistent."]


def _setup_type_from_target_name(target_name: str) -> int | None:
    suffix = str(target_name).strip().lower().rsplit("_setup", maxsplit=1)
    if len(suffix) != 2:
        return None
    try:
        setup_type = int(suffix[1])
    except ValueError:
        return None
    return setup_type if setup_type in {1, 2, 3, 4, 5, 6} else None


def _normalize_stop_phase(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in VALID_STOP_PHASES:
        allowed = ", ".join(VALID_STOP_PHASES)
        raise ValueError(f"Unsupported stop_after value '{value}'. Expected one of: {allowed}.")
    return normalized


def _normalize_phase02_compression(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"none", "gzip"}:
        raise ValueError("Unsupported phase02_compression value. Expected one of: none, gzip.")
    return normalized


def _phase02_output_paths(phase02_dir: Path, *, compression: str) -> tuple[Path, Path]:
    normalized = _normalize_phase02_compression(compression)
    suffix = ".csv.gz" if normalized == "gzip" else ".csv"
    return (
        phase02_dir / f"es_primary_frvp_features_full{suffix}",
        phase02_dir / f"es_primary_frvp_phase04_dataset{suffix}",
    )


def _load_prepared_reports(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output_dir = Path(summary["output_dir"])
    reports: dict[str, dict[str, Any]] = {}
    for target_name in summary["targets"]:
        report_path = output_dir / target_name / "report.json"
        reports[target_name] = json.loads(report_path.read_text(encoding="utf-8"))
    return reports


def _load_existing_feature_dataset(
    feature_csv_path: Path,
    feature_metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    feature_columns = [str(column) for column in metadata.get("feature_columns", [])]
    usecols = {"datetime", *feature_columns}
    sample = pd.read_csv(
        feature_csv_path,
        nrows=64,
        usecols=lambda column: str(column) in usecols,
    )
    dtype_map: dict[str, Any] = {}
    for column in feature_columns:
        if column not in sample.columns:
            continue
        if pd.api.types.is_numeric_dtype(sample[column]):
            dtype_map[column] = np.float32
    dataset = pd.read_csv(
        feature_csv_path,
        usecols=lambda column: str(column) in usecols,
        dtype=dtype_map,
    )
    if "datetime" not in dataset.columns:
        raise ValueError(f"Existing feature dataset is missing a datetime column: {feature_csv_path}")
    dataset["datetime"] = pd.to_datetime(dataset["datetime"], errors="coerce", utc=True)
    return dataset, metadata


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_make_json_safe(payload), indent=2), encoding="utf-8")


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    return str(value)


def _iso_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FRVP ES-primary program through Phase 4.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--recipe", default=str(DEFAULT_RECIPE))
    parser.add_argument("--feature-csv", default=None)
    parser.add_argument("--feature-metadata", default=None)
    parser.add_argument("--instrument", default="es")
    parser.add_argument("--transform-workers", type=int, default=1)
    parser.add_argument("--backend-max-features", type=int, default=160)
    parser.add_argument("--attribution-max-rows", type=int, default=100_000)
    parser.add_argument("--top-n-features", type=int, default=25)
    parser.add_argument("--min-usable-rows", type=int, default=250)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-positive-samples", type=int, default=25)
    parser.add_argument("--stop-after", default="phase04", choices=VALID_STOP_PHASES)
    parser.add_argument(
        "--skip-phase02-csvs",
        action="store_true",
        help="Skip writing the large Phase 2 feature and merged dataset CSVs. Supported only with --stop-after phase03 or earlier.",
    )
    parser.add_argument(
        "--phase02-compression",
        default="none",
        choices=("none", "gzip"),
        help="Compression mode for the large Phase 2 feature and merged dataset CSVs.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = run(
        ESPrimaryPhase04Config(
            input_path=Path(args.input),
            output_root=Path(args.output_root),
            recipe_path=Path(args.recipe),
            feature_csv_path=Path(args.feature_csv) if args.feature_csv else None,
            feature_metadata_path=Path(args.feature_metadata) if args.feature_metadata else None,
            instrument=str(args.instrument).strip().lower(),
            transform_workers=int(args.transform_workers),
            backend_max_features=int(args.backend_max_features),
            attribution_max_rows=int(args.attribution_max_rows),
            top_n_features=int(args.top_n_features),
            min_usable_rows=int(args.min_usable_rows),
            min_train_rows=int(args.min_train_rows),
            min_positive_samples=int(args.min_positive_samples),
            stop_after=str(args.stop_after),
            skip_phase02_csvs=bool(args.skip_phase02_csvs),
            phase02_compression=str(args.phase02_compression),
        )
    )
    print(json.dumps(_make_json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
