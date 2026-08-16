from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ict.labeling.ict_labeling_engine import ICTLabelingConfig
from ict.pipelines.layout import build_ict_artifact_layout
from ict.reports.event_sample_audit import (
    build_ict_event_sample_audit,
    load_ict_phase03_outputs,
    refresh_ict_phase03_labeling_artifacts,
    write_ict_event_sample_audit,
)
from ict.reports.spacing_cooldown_diagnostics import (
    build_ict_spacing_cooldown_diagnostics,
    write_ict_spacing_cooldown_diagnostics,
)
from ict.setups import detector as detector_module

DEFAULT_MARKET_5M = REPO_ROOT / "data" / "futures_data" / "ES-5m-tagged.csv"
DEFAULT_MARKET_1M = REPO_ROOT / "data" / "futures_data" / "ES-1m.csv"
DEFAULT_SETUP_FEATURE_CSV = REPO_ROOT / "artifacts" / "ict_es_primary" / "phase02_features" / "ict_es_features.csv"
DEFAULT_ARTIFACT_BASE_DIR = REPO_ROOT / "artifacts"
DEFAULT_SWEEP_REPORT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ict_spacing_refit_sweeps"
DEFAULT_AUDIT_REPORT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ict_event_sample_audits"
DEFAULT_SPACING_REPORT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ict_spacing_cooldown_diagnostics"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a premium-continuation spacing sweep against a fixed Phase 02 setup surface."
    )
    parser.add_argument("--sweep-id", required=True, help="Top-level identifier for this sweep package.")
    parser.add_argument("--artifact-base-dir", type=Path, default=DEFAULT_ARTIFACT_BASE_DIR)
    parser.add_argument("--market-5m", type=Path, default=DEFAULT_MARKET_5M)
    parser.add_argument("--market-1m", type=Path, default=DEFAULT_MARKET_1M)
    parser.add_argument(
        "--setup-feature-csv",
        type=Path,
        default=DEFAULT_SETUP_FEATURE_CSV,
        help="Required Phase 02 setup surface. This script fails fast if the file is missing.",
    )
    parser.add_argument(
        "--premium-spacing-values",
        type=int,
        nargs="+",
        default=[8, 10, 12],
        help="Premium/discount continuation spacing values to sweep.",
    )
    parser.add_argument(
        "--displacement-spacing",
        type=int,
        default=16,
        help="Fixed spacing for displacement_continuation_after_raid during the sweep.",
    )
    parser.add_argument(
        "--base-run-id-prefix",
        default="ict_es_primary_spacing_sweep",
        help="Artifact run-id prefix. Each spacing point appends its own suffix.",
    )
    parser.add_argument("--instrument", default="es")
    parser.add_argument("--event-audit-report-root", type=Path, default=DEFAULT_AUDIT_REPORT_ROOT)
    parser.add_argument("--spacing-report-root", type=Path, default=DEFAULT_SPACING_REPORT_ROOT)
    parser.add_argument("--sweep-report-root", type=Path, default=DEFAULT_SWEEP_REPORT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.setup_feature_csv.exists():
        raise FileNotFoundError(f"Required setup-feature CSV does not exist: {args.setup_feature_csv}")

    sweep_dir = args.sweep_report_root / args.sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for premium_spacing in args.premium_spacing_values:
        run_id = (
            f"{args.base_run_id_prefix}_{args.instrument}_"
            f"premium{int(premium_spacing)}_disp{int(args.displacement_spacing)}"
        )
        result = _run_single_spacing_point(
            run_id=run_id,
            instrument=args.instrument,
            artifact_base_dir=args.artifact_base_dir,
            market_5m=args.market_5m,
            market_1m=args.market_1m if args.market_1m.exists() else None,
            setup_feature_csv=args.setup_feature_csv,
            premium_spacing=int(premium_spacing),
            displacement_spacing=int(args.displacement_spacing),
            event_audit_report_root=args.event_audit_report_root,
            spacing_report_root=args.spacing_report_root,
        )
        results.append(result)
        print(
            "[ICT sweep] "
            f"premium={premium_spacing} "
            f"usable={result['usable_events']} "
            f"base_rate={result['base_rate_pct']:.2f}% "
            f"continuation={result['continuation_events']} "
            f"premium_overlap={result['premium_overlap_rate_prev_pct']:.2f}% "
            f"premium_next={result['premium_recommended_spacing_bars']}"
        )

    summary_frame = pd.DataFrame(results).sort_values(
        ["premium_spacing_bars", "displacement_spacing_bars"],
        ascending=[True, True],
    )
    manifest = _write_sweep_summary(
        sweep_dir=sweep_dir,
        summary_frame=summary_frame,
        args=args,
    )
    print(f"[ICT sweep] sweep_dir={sweep_dir}")
    print(f"[ICT sweep] summary_csv={manifest['summary_csv']}")
    return 0


def _run_single_spacing_point(
    *,
    run_id: str,
    instrument: str,
    artifact_base_dir: Path,
    market_5m: Path,
    market_1m: Path | None,
    setup_feature_csv: Path,
    premium_spacing: int,
    displacement_spacing: int,
    event_audit_report_root: Path,
    spacing_report_root: Path,
) -> dict[str, Any]:
    layout = build_ict_artifact_layout(
        run_id,
        base_dir=artifact_base_dir,
        ensure_directories=True,
    )
    with _temporary_spacing_override(
        premium_spacing=premium_spacing,
        displacement_spacing=displacement_spacing,
    ):
        refresh_summary = refresh_ict_phase03_labeling_artifacts(
            phase03_dir=layout.phase03_labeling,
            market_5m_path=market_5m,
            market_1m_path=market_1m,
            setup_feature_path=setup_feature_csv,
            config=ICTLabelingConfig(instrument=instrument),
        )

    refresh_summary_path = layout.phase03_labeling / "refresh_summary.json"
    refresh_summary_path.write_text(json.dumps(_json_safe(refresh_summary), indent=2), encoding="utf-8")

    events, diagnostics, phase03_paths = load_ict_phase03_outputs(layout.phase03_labeling)

    audit_report_id = f"{run_id}_event_audit"
    audit_dir = event_audit_report_root / audit_report_id
    audit_summary, review_rows = build_ict_event_sample_audit(events, diagnostics=diagnostics)
    audit_summary["report_id"] = audit_report_id
    audit_summary["artifact_run_id"] = run_id
    audit_summary["source_paths"] = {
        "market_5m": str(market_5m),
        "market_1m": str(market_1m) if market_1m is not None else None,
        "setup_feature_csv": str(setup_feature_csv),
        **phase03_paths,
    }
    audit_summary["refresh_summary"] = refresh_summary
    audit_written = write_ict_event_sample_audit(
        output_dir=audit_dir,
        summary=audit_summary,
        review_rows=review_rows,
    )
    (audit_dir / "manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "report_id": audit_report_id,
                    "report_dir": str(audit_dir),
                    "summary_paths": audit_written,
                    "source_paths": audit_summary["source_paths"],
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    spacing_report_id = f"{run_id}_spacing_followup"
    spacing_dir = spacing_report_root / spacing_report_id
    spacing_summary, spacing_frame = build_ict_spacing_cooldown_diagnostics(
        events,
        diagnostics=diagnostics,
    )
    spacing_summary["report_id"] = spacing_report_id
    spacing_summary["artifact_run_id"] = run_id
    spacing_summary["source_paths"] = {
        "market_5m": str(market_5m),
        "market_1m": str(market_1m) if market_1m is not None else None,
        "setup_feature_csv": str(setup_feature_csv),
        **phase03_paths,
    }
    spacing_written = write_ict_spacing_cooldown_diagnostics(
        output_dir=spacing_dir,
        summary=spacing_summary,
        setup_frame=spacing_frame,
    )
    (spacing_dir / "manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "report_id": spacing_report_id,
                    "report_dir": str(spacing_dir),
                    "summary_paths": spacing_written,
                    "source_paths": spacing_summary["source_paths"],
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    premium_row = _find_setup_row(spacing_frame, "premium_discount_continuation")
    displacement_row = _find_setup_row(spacing_frame, "displacement_continuation_after_raid")
    reference_geometry = audit_summary["reference_geometry"]["overall"]
    continuation = audit_summary["continuation_management"]
    headline = audit_summary["headline"]

    return {
        "run_id": run_id,
        "premium_spacing_bars": int(premium_spacing),
        "displacement_spacing_bars": int(displacement_spacing),
        "total_events_sampled": int(headline["total_events"]),
        "usable_events": int(headline["usable_events"]),
        "positive_events": int(headline["positive_events"]),
        "base_rate_pct": float(headline["base_rate_pct"]),
        "continuation_events": int(continuation["usable_events"]),
        "continuation_target_activations": int(continuation["target_activation_events"]),
        "continuation_target_activation_pct": float(continuation["target_activation_share_pct"]),
        "target_adjusted_pct": float(reference_geometry["target_adjusted_pct"]),
        "premium_event_count": int(premium_row.get("usable_event_count", 0)),
        "premium_overlap_rate_prev_pct": float(premium_row.get("overlap_rate_prev_pct", 0.0)),
        "premium_base_rate_pct": float(premium_row.get("base_rate_pct", 0.0)),
        "premium_recommended_spacing_bars": int(premium_row.get("recommended_spacing_bars", premium_spacing)),
        "displacement_event_count": int(displacement_row.get("usable_event_count", 0)),
        "displacement_overlap_rate_prev_pct": float(displacement_row.get("overlap_rate_prev_pct", 0.0)),
        "displacement_base_rate_pct": float(displacement_row.get("base_rate_pct", 0.0)),
        "artifact_run_id": run_id,
        "phase03_dir": str(layout.phase03_labeling),
        "event_audit_report_id": audit_report_id,
        "spacing_report_id": spacing_report_id,
    }


@contextmanager
def _temporary_spacing_override(
    *,
    premium_spacing: int,
    displacement_spacing: int,
) -> Iterator[None]:
    original = dict(detector_module.SETUP_MIN_SPACING)
    detector_module.SETUP_MIN_SPACING["premium_discount_continuation"] = int(premium_spacing)
    detector_module.SETUP_MIN_SPACING["displacement_continuation_after_raid"] = int(displacement_spacing)
    try:
        yield
    finally:
        detector_module.SETUP_MIN_SPACING.clear()
        detector_module.SETUP_MIN_SPACING.update(original)


def _find_setup_row(frame: pd.DataFrame, setup_type: str) -> dict[str, Any]:
    subset = frame.loc[frame["setup_type"].eq(setup_type)]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _write_sweep_summary(
    *,
    sweep_dir: Path,
    summary_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, str]:
    summary_csv = sweep_dir / "spacing_sweep_summary.csv"
    summary_json = sweep_dir / "spacing_sweep_summary.json"
    summary_md = sweep_dir / "spacing_sweep_summary.md"

    summary_frame.to_csv(summary_csv, index=False)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_id": str(args.sweep_id),
        "setup_feature_csv": str(args.setup_feature_csv),
        "market_5m": str(args.market_5m),
        "market_1m": str(args.market_1m),
        "premium_spacing_values": [int(value) for value in args.premium_spacing_values],
        "displacement_spacing": int(args.displacement_spacing),
        "rows": _json_safe(summary_frame.to_dict(orient="records")),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_md.write_text(_render_sweep_markdown(summary_frame, payload), encoding="utf-8")
    manifest = {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_markdown": str(summary_md),
    }
    (sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _render_sweep_markdown(summary_frame: pd.DataFrame, payload: dict[str, Any]) -> str:
    lines = [
        "# ICT Spacing Refit Sweep",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        f"- Sweep ID: `{payload['sweep_id']}`",
        f"- Setup feature CSV: `{payload['setup_feature_csv']}`",
        f"- Premium spacing values: `{', '.join(str(v) for v in payload['premium_spacing_values'])}`",
        f"- Displacement spacing: `{payload['displacement_spacing']}`",
        "",
        "## Results",
        "",
    ]
    for row in summary_frame.itertuples(index=False):
        lines.append(
            f"- `premium={row.premium_spacing_bars}`: usable=`{row.usable_events}`, "
            f"base_rate=`{row.base_rate_pct:.2f}%`, continuation=`{row.continuation_events}`, "
            f"premium_overlap=`{row.premium_overlap_rate_prev_pct:.2f}%`, "
            f"premium_next=`{row.premium_recommended_spacing_bars}`"
        )
    return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
