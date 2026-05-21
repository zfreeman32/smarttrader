from __future__ import annotations

import json
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from model_testing.ote_regime_slices import build_regime_winner_table

TARGET_MODEL_IDS = (
    "long_ote_tcn_v2_candidate",
    "short_ote_tcn_v2_candidate",
    "long_reversal_tcn_champion",
    "short_reversal_xgb_v1",
    "long_breakout_tcn_champion",
    "short_breakout_tcn_champion",
    "long_ote_meta_tcn_champion",
    "short_ote_meta_tcn_champion",
    "short_reversal_tcn_champion",
    "long_breakout_xgb_v1",
)

SOURCE_ROOTS = {
    "regime": (
        REPO_ROOT / "model_testing" / "reports" / "ote_regime_slices" / "v1_v2_comparison",
        REPO_ROOT / "model_testing" / "reports" / "ote_regime_slices" / "champion_models_20260508_v2",
        REPO_ROOT / "model_testing" / "reports" / "ote_regime_slices" / "breakout_reversal_xgb_20260509",
    ),
    "threshold": (
        REPO_ROOT / "model_testing" / "reports" / "ote_threshold_policies" / "v1_v2_tcn_focus",
        REPO_ROOT / "model_testing" / "reports" / "ote_threshold_policies" / "champion_models_20260508_v2",
        REPO_ROOT / "model_testing" / "reports" / "ote_threshold_policies" / "breakout_reversal_xgb_20260509",
    ),
    "backtest": (
        REPO_ROOT / "model_testing" / "reports" / "ote_policy_backtests" / "v1_v2_tcn_focus",
        REPO_ROOT / "model_testing" / "reports" / "ote_policy_backtests" / "champion_models_20260508_v2_min5",
        REPO_ROOT / "model_testing" / "reports" / "ote_policy_backtests" / "breakout_reversal_xgb_20260509",
    ),
}

TARGET_ROOTS = {
    "regime": REPO_ROOT / "model_testing" / "reports" / "ote_regime_slices" / "multifamily_live_v1",
    "threshold": REPO_ROOT / "model_testing" / "reports" / "ote_threshold_policies" / "multifamily_live_v1",
    "backtest": REPO_ROOT / "model_testing" / "reports" / "ote_policy_backtests" / "multifamily_live_v1",
}

TARGET_REGISTRY_PATH = REPO_ROOT / "models" / "ote_model_registry_multifamily_candidates.json"


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    regime_mapping = _assemble_regime_root(now)
    threshold_mapping = _assemble_threshold_root(now)
    _assemble_backtest_root(now, regime_mapping=regime_mapping, threshold_mapping=threshold_mapping)


def _assemble_regime_root(generated_at_utc: str) -> dict[str, Path]:
    target_root = TARGET_ROOTS["regime"]
    target_root.mkdir(parents=True, exist_ok=True)

    source_entries = _load_model_entries(SOURCE_ROOTS["regime"])
    model_outputs: list[dict[str, Any]] = []
    consolidated_reports: list[pd.DataFrame] = []
    output_dir_by_model: dict[str, Path] = {}

    for model_id in TARGET_MODEL_IDS:
        entry = source_entries[model_id]
        source_root = entry["source_root"]
        source_output_dir = source_root / model_id
        target_output_dir = target_root / model_id
        _copy_model_directory(source_output_dir, target_output_dir)
        output_dir_by_model[model_id] = target_output_dir

        split_report_path = target_output_dir / "slice_report_all_splits.csv"
        if split_report_path.exists():
            consolidated_reports.append(pd.read_csv(split_report_path))

        model_output = _deep_rewrite_paths(
            entry["model_output"],
            source_root=source_root,
            target_root=target_root,
        )
        model_outputs.append(model_output)

    consolidated_report = pd.concat(consolidated_reports, ignore_index=True) if consolidated_reports else pd.DataFrame()
    slice_report_path = target_root / "slice_report.csv"
    consolidated_report.to_csv(slice_report_path, index=False)

    winners = build_regime_winner_table(consolidated_report, regime_family="composite_regime")
    winners_path = target_root / "composite_bucket_winners.csv"
    winners.to_csv(winners_path, index=False)

    summary = {
        "generated_at_utc": generated_at_utc,
        "registry_path": _repo_relative(TARGET_REGISTRY_PATH),
        "output_root": _repo_relative(target_root),
        "model_ids": list(TARGET_MODEL_IDS),
        "statuses": ["active", "candidate"],
        "source_columns": _first_field(source_entries, "source_columns", default=[]),
        "bootstrap_iterations": int(_first_field(source_entries, "bootstrap_iterations", default=200)),
        "min_positive_events_for_threshold": int(
            _first_field(source_entries, "min_positive_events_for_threshold", default=50)
        ),
        "slice_families": _first_field(source_entries, "slice_families", default=[]),
        "slice_report_path": _repo_relative(slice_report_path),
        "composite_winners_path": _repo_relative(winners_path),
        "model_outputs": model_outputs,
    }
    (target_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir_by_model


def _assemble_threshold_root(generated_at_utc: str) -> dict[str, Path]:
    target_root = TARGET_ROOTS["threshold"]
    target_root.mkdir(parents=True, exist_ok=True)

    source_entries = _load_model_entries(SOURCE_ROOTS["threshold"])
    model_outputs: list[dict[str, Any]] = []
    policy_tables: list[pd.DataFrame] = []
    evaluations: list[pd.DataFrame] = []
    output_dir_by_model: dict[str, Path] = {}

    for model_id in TARGET_MODEL_IDS:
        entry = source_entries[model_id]
        source_root = entry["source_root"]
        source_output_dir = source_root / model_id
        target_output_dir = target_root / model_id
        _copy_model_directory(source_output_dir, target_output_dir)
        output_dir_by_model[model_id] = target_output_dir

        policy_table_path = target_output_dir / "policy_table.csv"
        evaluation_path = target_output_dir / "policy_evaluation.csv"
        if policy_table_path.exists():
            policy_tables.append(pd.read_csv(policy_table_path))
        if evaluation_path.exists():
            evaluations.append(pd.read_csv(evaluation_path))

        model_output = _deep_rewrite_paths(
            entry["model_output"],
            source_root=source_root,
            target_root=target_root,
        )
        model_outputs.append(model_output)

    consolidated_policy_table = pd.concat(policy_tables, ignore_index=True) if policy_tables else pd.DataFrame()
    consolidated_evaluation = pd.concat(evaluations, ignore_index=True) if evaluations else pd.DataFrame()
    policy_table_path = target_root / "policy_table.csv"
    policy_evaluation_path = target_root / "policy_evaluation.csv"
    consolidated_policy_table.to_csv(policy_table_path, index=False)
    consolidated_evaluation.to_csv(policy_evaluation_path, index=False)

    summary = {
        "generated_at_utc": generated_at_utc,
        "regime_report_root": _repo_relative(TARGET_ROOTS["regime"]),
        "output_root": _repo_relative(target_root),
        "registry_path": _repo_relative(TARGET_REGISTRY_PATH),
        "model_ids": list(TARGET_MODEL_IDS),
        "include_roles": ["candidate", "challenger", "benchmark"],
        "min_positive_events": int(_first_field(source_entries, "min_positive_events", default=50)),
        "min_events_per_month": float(_first_field(source_entries, "min_events_per_month", default=3.0)),
        "min_trades_per_week": float(_first_field(source_entries, "min_trades_per_week", default=3.0)),
        "policy_table_path": _repo_relative(policy_table_path),
        "policy_evaluation_path": _repo_relative(policy_evaluation_path),
        "model_outputs": model_outputs,
        "registry_updates_written": False,
    }
    (target_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir_by_model


def _assemble_backtest_root(
    generated_at_utc: str,
    *,
    regime_mapping: dict[str, Path],
    threshold_mapping: dict[str, Path],
) -> None:
    target_root = TARGET_ROOTS["backtest"]
    target_root.mkdir(parents=True, exist_ok=True)

    source_entries = _load_model_entries(SOURCE_ROOTS["backtest"])
    model_outputs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_id in TARGET_MODEL_IDS:
        entry = source_entries[model_id]
        source_root = entry["source_root"]
        source_output_dir = source_root / model_id
        target_output_dir = target_root / model_id
        _copy_model_directory(source_output_dir, target_output_dir)

        model_output = _deep_rewrite_paths(
            entry["model_output"],
            source_root=source_root,
            target_root=target_root,
        )
        model_outputs.append(model_output)

        summary_path = target_output_dir / "summary.json"
        if summary_path.exists():
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            overall_test = summary_payload.get("overall_test_metrics", {})
            walk_forward = summary_payload.get("walk_forward_efficiency", {})
            summary_rows.append(
                {
                    "model_id": summary_payload["model_id"],
                    "direction": summary_payload["direction"],
                    "backend": summary_payload["backend"],
                    "fold_count": summary_payload["fold_count"],
                    "selected_test_trades": overall_test.get("trade_count"),
                    "selected_test_net_pnl_pips": overall_test.get("total_net_pnl_pips"),
                    "selected_test_expectancy_pips": overall_test.get("expectancy_pips"),
                    "selected_test_profit_factor": overall_test.get("profit_factor"),
                    "selected_test_sharpe": overall_test.get("monthly_sharpe"),
                    "selected_test_sortino": overall_test.get("monthly_sortino"),
                    "overall_wfe": walk_forward.get("overall_wfe"),
                    "profitable_quarter_share": overall_test.get("profitable_quarter_share"),
                    "positive_composite_expectancy_share": summary_payload.get("positive_composite_expectancy_share"),
                    "accepted_for_paper_trading_gate": all(
                        bool(value) for value in summary_payload.get("acceptance", {}).values()
                    ),
                    "output_dir": _repo_relative(target_output_dir),
                }
            )

    model_summary_frame = pd.DataFrame(summary_rows).sort_values("model_id").reset_index(drop=True)
    model_summary_path = target_root / "model_summary.csv"
    model_summary_frame.to_csv(model_summary_path, index=False)

    summary = {
        "generated_at_utc": generated_at_utc,
        "regime_report_root": _repo_relative(TARGET_ROOTS["regime"]),
        "output_root": _repo_relative(target_root),
        "registry_path": _repo_relative(TARGET_REGISTRY_PATH),
        "model_ids": list(TARGET_MODEL_IDS),
        "include_roles": ["candidate", "challenger", "benchmark"],
        "min_train_years": int(_first_field(source_entries, "min_train_years", default=2)),
        "test_window_months": int(_first_field(source_entries, "test_window_months", default=3)),
        "rolling_step_months": int(_first_field(source_entries, "rolling_step_months", default=3)),
        "min_folds": int(_first_field(source_entries, "min_folds", default=5)),
        "max_folds": _first_field(source_entries, "max_folds", default=None),
        "min_positive_events": int(_first_field(source_entries, "min_positive_events", default=50)),
        "min_events_per_month": float(_first_field(source_entries, "min_events_per_month", default=3.0)),
        "min_trades_per_week": float(_first_field(source_entries, "min_trades_per_week", default=3.0)),
        "fixed_slippage_pips_per_trade": float(
            _first_field(source_entries, "fixed_slippage_pips_per_trade", default=0.3)
        ),
        "commission_pips_per_trade": float(
            _first_field(source_entries, "commission_pips_per_trade", default=0.35)
        ),
        "targeted_filter_preset": None,
        "session_spread_pips": _first_field(source_entries, "session_spread_pips", default={}),
        "model_summary_path": _repo_relative(model_summary_path),
        "model_outputs": model_outputs,
    }
    (target_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _load_model_entries(source_roots: tuple[Path, ...]) -> OrderedDict[str, dict[str, Any]]:
    entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for source_root in source_roots:
        summary_path = source_root / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for model_output in summary.get("model_outputs", []):
            model_id = model_output["model_id"]
            if model_id not in TARGET_MODEL_IDS or model_id in entries:
                continue
            entries[model_id] = {
                "source_root": source_root,
                "summary": summary,
                "model_output": model_output,
            }
    missing = [model_id for model_id in TARGET_MODEL_IDS if model_id not in entries]
    if missing:
        raise KeyError(f"Missing source report outputs for: {missing}")
    return entries


def _copy_model_directory(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _deep_rewrite_paths(value: Any, *, source_root: Path, target_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _deep_rewrite_paths(item, source_root=source_root, target_root=target_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_rewrite_paths(item, source_root=source_root, target_root=target_root) for item in value]
    if isinstance(value, str):
        return _rewrite_path_string(value, source_root=source_root, target_root=target_root)
    return value


def _rewrite_path_string(value: str, *, source_root: Path, target_root: Path) -> str:
    normalized = value.replace("\\", "/")
    source_relative = _repo_relative(source_root)
    target_relative = _repo_relative(target_root)
    if normalized.startswith(source_relative):
        return normalized.replace(source_relative, target_relative, 1)
    return normalized


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _first_field(entries: OrderedDict[str, dict[str, Any]], key: str, *, default: Any) -> Any:
    for entry in entries.values():
        summary = entry["summary"]
        if key in summary:
            return summary[key]
    return default


if __name__ == "__main__":
    main()
