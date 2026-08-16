from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_TEST_PATHS = (
    "tests/test_continuity.py",
    "tests/test_profiles.py",
    "tests/test_frvp_context.py",
    "tests/test_frvp_setups.py",
    "tests/test_frvp_labeling.py",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble a saved FRVP roll-audit / roll-aware embargo package for a model artifact."
    )
    parser.add_argument(
        "--model-artifact-dir",
        type=Path,
        required=True,
        help="Saved FRVP model artifact directory containing training_summary.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where the audit package files will be written.",
    )
    parser.add_argument(
        "--placebo-summary",
        type=Path,
        default=None,
        help="Optional placebo_readout_summary.json to include in the package.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Build the package without running the FRVP continuity/profile smoke tests.",
    )
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    return (REPO_ROOT / raw_path).resolve()


def _derive_phase03_paths(training_summary: Dict[str, Any]) -> Dict[str, Path]:
    prepared_summary_file = _resolve_repo_path(training_summary.get("prepared_summary_file"))
    if prepared_summary_file is None:
        raise ValueError("training_summary.json is missing prepared_summary_file.")
    phase04_dir = prepared_summary_file.parent.parent
    phase03_dir = phase04_dir.with_name("phase03")
    return {
        "phase03_dir": phase03_dir,
        "events_csv": phase03_dir / "es_primary_frvp_events.csv",
        "labeling_diagnostics_json": phase03_dir / "labeling_diagnostics.json",
    }


def _load_source_tables(training_summary: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    source_lineage = training_summary.get("source_lineage", {})
    if not isinstance(source_lineage, dict):
        raise ValueError("training_summary.json is missing a valid source_lineage payload.")

    feature_csv = _resolve_repo_path(source_lineage.get("feature_csv"))
    upstream_csv = _resolve_repo_path(source_lineage.get("upstream_source_path"))
    if feature_csv is None or upstream_csv is None:
        raise ValueError("Could not resolve feature_csv or upstream_source_path from source_lineage.")

    feature_frame = pd.read_csv(feature_csv, usecols=["datetime"])
    upstream_frame = pd.read_csv(
        upstream_csv,
        usecols=["ts_event", "contract_symbol", "instrument_id", "is_roll_boundary", "in_roll_bracket"],
    )

    if len(feature_frame) != len(upstream_frame):
        raise ValueError(
            "Feature and upstream source row counts differ; cannot safely audit source_row_idx boundaries."
        )
    if str(feature_frame.iloc[0]["datetime"]) != str(upstream_frame.iloc[0]["ts_event"]):
        raise ValueError("Feature and upstream source start timestamps do not align.")
    if str(feature_frame.iloc[-1]["datetime"]) != str(upstream_frame.iloc[-1]["ts_event"]):
        raise ValueError("Feature and upstream source end timestamps do not align.")

    upstream_frame = upstream_frame.rename(columns={"ts_event": "datetime"})
    return {
        "feature_frame": feature_frame,
        "upstream_frame": upstream_frame,
        "feature_csv": feature_csv,
        "upstream_csv": upstream_csv,
    }


def _build_fold_boundary_audit(training_summary: Dict[str, Any], upstream_frame: pd.DataFrame, artifact_dir: Path) -> Dict[str, Any]:
    manifest = pd.DataFrame(training_summary.get("cv_fold_manifest", []))
    if manifest.empty:
        raise ValueError("training_summary.json is missing cv_fold_manifest.")

    oof_predictions = pd.read_csv(
        artifact_dir / "oof_predictions.csv",
        usecols=["row_index_in_dev", "source_row_idx"],
    )
    oof_predictions = oof_predictions.set_index("row_index_in_dev")

    fold_rows: List[Dict[str, Any]] = []
    minimum_required_bars_no_roll = 288
    minimum_required_bars_with_roll = 576

    for record in training_summary["cv_fold_manifest"]:
        train_end_row = int(record["train_end_row"])
        val_start_row = int(record["val_start_row"])
        last_train_dev_row = train_end_row - 1
        first_val_dev_row = val_start_row

        last_train_source_row = int(oof_predictions.loc[last_train_dev_row, "source_row_idx"])
        first_val_source_row = int(oof_predictions.loc[first_val_dev_row, "source_row_idx"])

        last_train_source = upstream_frame.iloc[last_train_source_row]
        first_val_source = upstream_frame.iloc[first_val_source_row]
        between = upstream_frame.iloc[last_train_source_row + 1:first_val_source_row]

        gap_bars = first_val_source_row - last_train_source_row - 1
        last_train_ts = pd.Timestamp(last_train_source["datetime"])
        first_val_ts = pd.Timestamp(first_val_source["datetime"])
        gap_hours = float((first_val_ts - last_train_ts).total_seconds() / 3600.0)
        roll_boundary_rows = between.loc[between["is_roll_boundary"].fillna(False).astype(bool)]
        roll_count_between = int(len(roll_boundary_rows))
        roll_between = bool(roll_count_between > 0)
        required_gap_bars = (
            minimum_required_bars_with_roll if roll_between else minimum_required_bars_no_roll
        )
        passed_gap_requirement = bool(gap_bars >= required_gap_bars)

        fold_rows.append(
            {
                "fold": int(record["fold"]),
                "last_train_dev_row": last_train_dev_row,
                "first_val_dev_row": first_val_dev_row,
                "last_train_source_row_idx": last_train_source_row,
                "first_val_source_row_idx": first_val_source_row,
                "last_train_timestamp": last_train_ts.isoformat(),
                "first_val_timestamp": first_val_ts.isoformat(),
                "last_train_contract_symbol": str(last_train_source["contract_symbol"]),
                "first_val_contract_symbol": str(first_val_source["contract_symbol"]),
                "gap_bars": int(gap_bars),
                "gap_hours": gap_hours,
                "roll_boundary_count_between": roll_count_between,
                "roll_boundary_between": roll_between,
                "roll_dates_between": [pd.Timestamp(value).isoformat() for value in roll_boundary_rows["datetime"].tolist()],
                "required_gap_bars": int(required_gap_bars),
                "passed_gap_requirement": passed_gap_requirement,
            }
        )

    audit_frame = pd.DataFrame(fold_rows)
    return {
        "minimum_required_bars_no_roll": minimum_required_bars_no_roll,
        "minimum_required_bars_with_roll": minimum_required_bars_with_roll,
        "resolved_purge_bars": int(training_summary.get("resolved_purge_bars", -1)),
        "all_folds_pass_gap_requirement": bool(audit_frame["passed_gap_requirement"].all()),
        "folds_with_roll_boundary_between": int(audit_frame["roll_boundary_between"].sum()),
        "minimum_gap_bars": int(audit_frame["gap_bars"].min()),
        "minimum_gap_hours": float(audit_frame["gap_hours"].min()),
        "fold_rows": fold_rows,
    }


def _build_event_roll_audit(events_path: Path, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    events = pd.read_csv(
        events_path,
        usecols=["excluded", "flag_roll_span", "flag_roll_bracket"],
    )
    excluded = events["excluded"].fillna(False).astype(bool)
    roll_span = events["flag_roll_span"].fillna(False).astype(bool)
    roll_bracket = events["flag_roll_bracket"].fillna(False).astype(bool)

    usable_roll_span_events = int((roll_span & ~excluded).sum())
    usable_roll_bracket_events = int((roll_bracket & ~excluded).sum())

    return {
        "total_rows_in_events_csv": int(len(events)),
        "events_excluded_roll_span_reported": int(diagnostics.get("events_excluded_roll_span", 0)),
        "events_roll_bracket_reported": int(diagnostics.get("events_roll_bracket", 0)),
        "roll_span_rows_in_events_csv": int(roll_span.sum()),
        "usable_roll_span_rows_in_events_csv": usable_roll_span_events,
        "roll_bracket_rows_in_events_csv": int(roll_bracket.sum()),
        "usable_roll_bracket_rows_in_events_csv": usable_roll_bracket_events,
        "zero_usable_roll_span_events": bool(usable_roll_span_events == 0),
    }


def _run_pytest_subset(skip_pytest: bool) -> Dict[str, Any]:
    if skip_pytest:
        return {
            "skipped": True,
            "command": [],
            "returncode": None,
            "passed": None,
            "stdout": "",
            "stderr": "",
        }

    command = [sys.executable, "-m", "pytest", *DEFAULT_TEST_PATHS, "-q"]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "skipped": False,
        "command": command,
        "returncode": int(completed.returncode),
        "passed": bool(completed.returncode == 0),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _build_roll_reconstruction_summary(report_path: Path, schedule_path: Path) -> Dict[str, Any]:
    reconstruction = _load_json(report_path)
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    seam_report = reconstruction.get("seam_report", {})
    return {
        "report_path": str(report_path),
        "schedule_path": str(schedule_path),
        "all_checks_passed": bool(reconstruction.get("all_checks_passed", False)),
        "coverage_all_bars_tagged": bool(reconstruction.get("coverage", {}).get("all_bars_tagged", False)),
        "schedule_contiguous": bool(reconstruction.get("coverage", {}).get("schedule_contiguous", False)),
        "schedule_covers_source_range": bool(
            reconstruction.get("coverage", {}).get("schedule_covers_source_range", False)
        ),
        "plausible_full_year_roll_counts": bool(
            reconstruction.get("roll_cadence", {}).get("plausible_full_year_roll_counts", False)
        ),
        "plausible_days_before_expiration": bool(
            reconstruction.get("roll_cadence", {}).get("plausible_days_before_expiration", False)
        ),
        "flagged_seam_boundary_count": int(len(seam_report.get("flagged_boundaries", []))),
        "roll_boundary_count": int(seam_report.get("boundary_count", 0)),
        "schedule_roll_count": int(len(schedule) - 1),
    }


def _build_placebo_summary(placebo_path: Path | None) -> Dict[str, Any] | None:
    if placebo_path is None:
        return None
    summary = _load_json(placebo_path)
    return {
        "summary_path": str(placebo_path),
        "num_shuffles": int(summary.get("num_shuffles", 0)),
        "real_oof_average_precision": float(summary["real_model"]["oof_average_precision"]),
        "shuffled_mean_oof_average_precision": float(summary["placebo_distribution"]["mean_oof_average_precision"]),
        "shuffled_std_oof_average_precision": float(summary["placebo_distribution"]["std_oof_average_precision"]),
        "placebo_gap": float(summary["gate_check"]["placebo_gap"]),
        "passed": bool(summary["gate_check"]["passed"]),
    }


def _write_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    placebo = summary.get("placebo")
    roll = summary["roll_reconstruction"]
    event_roll = summary["event_roll_audit"]
    fold_roll = summary["fold_boundary_audit"]
    pytest_run = summary["pytest"]

    lines = [
        "# FRVP Roll Audit Package",
        "",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        f"- Model artifact: `{summary['model_artifact_dir']}`",
        f"- Target: `{summary['target']}`",
        "",
        "## Roll Gate Summary",
        "",
        f"- Gate 7 overall pass: `{summary['gate_7_roll_audit_passed']}`",
        f"- Roll reconstruction all checks passed: `{roll['all_checks_passed']}`",
        f"- Zero usable roll-span events: `{event_roll['zero_usable_roll_span_events']}`",
        f"- Fold embargo audit pass: `{fold_roll['all_folds_pass_gap_requirement']}`",
        f"- Continuity/profile smoke tests passed: `{pytest_run['passed'] if pytest_run['passed'] is not None else 'skipped'}`",
        "",
        "## Event Audit",
        "",
        f"- Reported roll-span exclusions: `{event_roll['events_excluded_roll_span_reported']}`",
        f"- Usable roll-span rows in saved event file: `{event_roll['usable_roll_span_rows_in_events_csv']}`",
        f"- Reported roll-bracket events: `{event_roll['events_roll_bracket_reported']}`",
        f"- Usable roll-bracket rows in saved event file: `{event_roll['usable_roll_bracket_rows_in_events_csv']}`",
        "",
        "## Fold Boundary Audit",
        "",
        f"- Resolved purge bars in training artifact: `{fold_roll['resolved_purge_bars']}`",
        f"- Minimum raw-source gap across folds: `{fold_roll['minimum_gap_bars']}` bars / `{fold_roll['minimum_gap_hours']:.2f}` hours",
        f"- Folds with a roll boundary inside the train-to-validation gap: `{fold_roll['folds_with_roll_boundary_between']}`",
        f"- Required gap rule used here: `288` bars minimum, or `576` when the gap crosses a roll boundary",
        "",
        "## Roll Reconstruction",
        "",
        f"- Schedule roll count: `{roll['schedule_roll_count']}`",
        f"- Flagged seam boundaries: `{roll['flagged_seam_boundary_count']}`",
        "",
    ]

    if placebo is not None:
        lines.extend(
            [
                "## Linked Placebo Readout",
                "",
                f"- Placebo pass: `{placebo['passed']}`",
                f"- Real OOF AP: `{placebo['real_oof_average_precision']:.6f}`",
                f"- Shuffled mean OOF AP: `{placebo['shuffled_mean_oof_average_precision']:.6f}`",
                f"- Placebo gap: `{placebo['placebo_gap']:.6f}`",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_roll_audit_package(
    *,
    model_artifact_dir: Path,
    output_root: Path,
    placebo_summary_path: Path | None,
    skip_pytest: bool,
) -> Dict[str, Any]:
    training_summary_path = model_artifact_dir / "training_summary.json"
    training_summary = _load_json(training_summary_path)

    phase03_paths = _derive_phase03_paths(training_summary)
    diagnostics = _load_json(phase03_paths["labeling_diagnostics_json"])
    source_tables = _load_source_tables(training_summary)

    roll_report_path = REPO_ROOT / "data/futures_data/es_roll_reconstruction_report.json"
    roll_schedule_path = REPO_ROOT / "data/futures_data/es_roll_schedule.json"

    output_root.mkdir(parents=True, exist_ok=True)

    event_roll_audit = _build_event_roll_audit(phase03_paths["events_csv"], diagnostics)
    fold_boundary_audit = _build_fold_boundary_audit(
        training_summary=training_summary,
        upstream_frame=source_tables["upstream_frame"],
        artifact_dir=model_artifact_dir,
    )
    pytest_summary = _run_pytest_subset(skip_pytest)
    roll_reconstruction = _build_roll_reconstruction_summary(roll_report_path, roll_schedule_path)
    placebo = _build_placebo_summary(placebo_summary_path)

    gate_7_pass = bool(
        roll_reconstruction["all_checks_passed"]
        and event_roll_audit["zero_usable_roll_span_events"]
        and fold_boundary_audit["all_folds_pass_gap_requirement"]
        and (pytest_summary["passed"] is True if pytest_summary["passed"] is not None else True)
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_artifact_dir": str(model_artifact_dir),
        "training_summary_path": str(training_summary_path),
        "target": training_summary.get("target"),
        "phase03_paths": {key: str(value) for key, value in phase03_paths.items()},
        "source_paths": {
            "feature_csv": str(source_tables["feature_csv"]),
            "upstream_csv": str(source_tables["upstream_csv"]),
        },
        "placebo": placebo,
        "labeling_diagnostics": {
            "usable_events": int(diagnostics.get("usable_events", 0)),
            "events_excluded_roll_span": int(diagnostics.get("events_excluded_roll_span", 0)),
            "events_roll_bracket": int(diagnostics.get("events_roll_bracket", 0)),
        },
        "event_roll_audit": event_roll_audit,
        "fold_boundary_audit": fold_boundary_audit,
        "roll_reconstruction": roll_reconstruction,
        "pytest": pytest_summary,
        "gate_7_roll_audit_passed": gate_7_pass,
        "advisory_caveats": {
            "flagged_seam_boundaries_are_not_zero": bool(roll_reconstruction["flagged_seam_boundary_count"] > 0),
            "seam_flags_are_diagnostic_only_for_this_package": True,
        },
    }

    summary_path = output_root / "roll_audit_package_summary.json"
    markdown_path = output_root / "roll_audit_package_summary.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, markdown_path)
    return summary


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_roll_audit_package(
        model_artifact_dir=args.model_artifact_dir,
        output_root=args.output_root,
        placebo_summary_path=args.placebo_summary,
        skip_pytest=args.skip_pytest,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
