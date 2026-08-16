from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_testing.promotion_gates import accepted_for_paper_trading, drawdown_acceptance_passed

DEFAULT_MODEL_ID = "frvp_long_continuation_xgb_v1"
DEFAULT_THRESHOLD_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_threshold_policies"
    / "frvp_long_continuation_gatefix_v3_20260715"
    / "run_summary.json"
)
DEFAULT_BASELINE_THRESHOLD_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_threshold_policies"
    / "frvp_es_primary_refresh_20260701"
    / "run_summary.json"
)
DEFAULT_BACKTEST_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_backtests"
    / "frvp_long_continuation_gatefix_v3_20260715_accountdd"
    / "run_summary.json"
)
DEFAULT_BASELINE_BACKTEST_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_backtests"
    / "frvp_es_primary_refresh_20260701"
    / "run_summary.json"
)
DEFAULT_PLACEBO_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_placebo_readouts"
    / "frvp_long_continuation_xgb_v1_20260717"
    / "placebo_readout_summary.json"
)
DEFAULT_ROLL_AUDIT_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_roll_audit_packages"
    / "frvp_long_continuation_xgb_v1_20260717"
    / "roll_audit_package_summary.json"
)
DEFAULT_ROLL_SHADOW_VALIDATION_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_roll_shadow_validation"
    / "frvp_es_shadow_roll_20260717"
    / "roll_shadow_validation_summary.json"
)
DEFAULT_SHADOW_SELECTION_SUMMARY_PATH = (
    REPO_ROOT
    / "ote_live"
    / "runtime_manifests"
    / "frvp_es_shadow_20260715"
    / "shadow_selection_summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_promotion_packages"
    / "frvp_long_continuation_xgb_v1_20260717"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the FRVP promotion package for the strongest ES-primary branch, "
            "including threshold-vs-prune analysis, placebo, roll audit, shadow bundle, and roll replay results."
        )
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--threshold-summary-path", type=Path, default=DEFAULT_THRESHOLD_SUMMARY_PATH)
    parser.add_argument("--baseline-threshold-summary-path", type=Path, default=DEFAULT_BASELINE_THRESHOLD_SUMMARY_PATH)
    parser.add_argument("--backtest-summary-path", type=Path, default=DEFAULT_BACKTEST_SUMMARY_PATH)
    parser.add_argument("--baseline-backtest-summary-path", type=Path, default=DEFAULT_BASELINE_BACKTEST_SUMMARY_PATH)
    parser.add_argument("--placebo-summary-path", type=Path, default=DEFAULT_PLACEBO_SUMMARY_PATH)
    parser.add_argument("--roll-audit-summary-path", type=Path, default=DEFAULT_ROLL_AUDIT_SUMMARY_PATH)
    parser.add_argument(
        "--roll-shadow-validation-summary-path",
        type=Path,
        default=DEFAULT_ROLL_SHADOW_VALIDATION_SUMMARY_PATH,
    )
    parser.add_argument("--shadow-selection-summary-path", type=Path, default=DEFAULT_SHADOW_SELECTION_SUMMARY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    threshold_summary = _read_json(args.threshold_summary_path)
    baseline_threshold_summary = _read_json(args.baseline_threshold_summary_path)
    backtest_summary = _read_json(args.backtest_summary_path)
    baseline_backtest_summary = _read_json(args.baseline_backtest_summary_path)
    placebo_summary = _read_json(args.placebo_summary_path)
    roll_audit_summary = _read_json(args.roll_audit_summary_path)
    roll_shadow_validation_summary = _read_json(args.roll_shadow_validation_summary_path)
    shadow_selection_summary = _read_json(args.shadow_selection_summary_path)

    threshold_model_output = _find_model_output(threshold_summary, args.model_id)
    baseline_threshold_model_output = _find_model_output(baseline_threshold_summary, args.model_id)
    backtest_model_output = _find_model_output(backtest_summary, args.model_id)
    baseline_backtest_model_output = _find_model_output(baseline_backtest_summary, args.model_id)

    test_policy_metrics = _read_policy_metrics(
        Path(_resolve_repo_path(threshold_model_output["policy_evaluation_path"])),
        dataset_split="test",
    )
    selected_policy_name = str(threshold_model_output.get("selected_policy_name") or "global_threshold")
    static_policy_metrics = test_policy_metrics[selected_policy_name]
    regime_policy_metrics = test_policy_metrics.get("regime_threshold")
    audit = _build_policy_contract_audit(
        threshold_model_output=threshold_model_output,
        backtest_model_output=backtest_model_output,
    )

    threshold_vs_prune = {
        "selected_policy_name": selected_policy_name,
        "selected_policy_reason": threshold_model_output.get("selected_policy_reason"),
        "selected_policy_contract": dict(threshold_model_output.get("selected_policy_contract") or {}),
        "targeted_filter_preset": threshold_summary.get("targeted_filter_preset"),
        "static_policy_metrics": static_policy_metrics,
        "regime_policy_metrics": regime_policy_metrics,
        "baseline_threshold_search_selected_policy_name": baseline_threshold_model_output.get("selected_policy_name"),
        "baseline_targeted_filters_enabled": bool(
            (baseline_threshold_model_output.get("targeted_filters") or {}).get("apply_to_base_policy_variants")
        ),
        "walk_forward_policy_audit": audit,
        "baseline_walk_forward_policy_counts": dict(baseline_backtest_model_output.get("selected_policy_counts") or {}),
        "targeted_walk_forward_policy_counts": dict(backtest_model_output.get("selected_policy_counts") or {}),
        "baseline_vs_targeted_backtest_delta": _metric_delta_block(
            baseline=baseline_backtest_model_output,
            targeted=backtest_model_output,
        ),
        "conclusion": (
            "The saved v3 contract is a hard-pruned base global-threshold policy, not a new regime-threshold winner. "
            "On the held-out static test, global_threshold stayed positive (+11.49 ticks expectancy, +264.3 ticks net) "
            "while regime_threshold was negative (-4.34 ticks expectancy, -681.3 ticks net), so no non-global policy qualified. "
            "Walk-forward still mixed in regime_threshold on 109 of 524 selected trades, but the dominant contract was the "
            "hard-pruned global threshold (415 trades, 79.20% share). Relative to the unpruned refresh baseline, the targeted "
            "prune sharply reduced trade count (1609 -> 524), lifted expectancy (7.75 -> 13.72 ticks), improved Sharpe "
            "(0.604 -> 1.226), improved DSR (0.585 -> 1.061), and cut drawdown (10200.85 -> 1540.10 ticks)."
        ),
    }

    promotion_readout = {
        "model_id": args.model_id,
        "registry_cv_mean_ap": _nested_get(shadow_selection_summary, "family_leaders", 0),  # placeholder removed below
    }
    manifest_snapshot = _find_family_leader(shadow_selection_summary, args.model_id)
    promotion_readout = {
        "model_id": args.model_id,
        "paper_trading_gate_passed": accepted_for_paper_trading(backtest_model_output.get("acceptance")),
        "drawdown_gate_passed": drawdown_acceptance_passed(backtest_model_output.get("acceptance")),
        "cv_mean_ap": _coerce_float(_nested_get(manifest_snapshot, "selected_test_sharpe")),  # placeholder removed below
    }
    promotion_readout = {
        "model_id": args.model_id,
        "cv_mean_ap": None,
        "test_ap": None,
        "selected_test_trade_count": backtest_model_output["overall_test_metrics"]["trade_count"],
        "selected_test_net_pnl_units": backtest_model_output["overall_test_metrics"]["total_net_pnl_units"],
        "selected_test_expectancy_units": backtest_model_output["overall_test_metrics"]["expectancy_units"],
        "selected_test_sharpe": backtest_model_output["overall_test_metrics"]["monthly_sharpe"],
        "selected_test_deflated_sharpe": backtest_model_output["overall_test_metrics"]["approx_deflated_sharpe"],
        "selected_test_max_drawdown_units": backtest_model_output["overall_test_metrics"]["max_drawdown_units"],
        "selected_test_max_drawdown_pct": backtest_model_output["overall_test_metrics"].get("max_drawdown_pct"),
        "overall_wfe": backtest_model_output["walk_forward_efficiency"]["overall_wfe"],
        "profitable_quarter_share": backtest_model_output["overall_test_metrics"]["profitable_quarter_share"],
        "positive_composite_expectancy_share": backtest_model_output["positive_composite_expectancy_share"],
        "paper_trading_gate_passed": accepted_for_paper_trading(backtest_model_output.get("acceptance")),
        "drawdown_gate_passed": drawdown_acceptance_passed(backtest_model_output.get("acceptance")),
        "placebo_passed": bool(placebo_summary["gate_check"]["passed"]),
        "placebo_gap": placebo_summary["gate_check"]["placebo_gap"],
        "roll_audit_passed": bool(roll_audit_summary["gate_7_roll_audit_passed"]),
        "roll_shadow_validation_passed": bool(roll_shadow_validation_summary["validation_passed"]),
    }

    runtime_manifest = _find_runtime_manifest(
        args.model_id,
        shadow_selection_summary_path=args.shadow_selection_summary_path,
    )
    if runtime_manifest is not None:
        promotion_readout["cv_mean_ap"] = _coerce_float(_nested_get(runtime_manifest, "registry_metrics", "cv_mean_ap"))
        promotion_readout["test_ap"] = _coerce_float(_nested_get(runtime_manifest, "registry_metrics", "test_ap"))
        promotion_readout["runtime_manifest_path"] = _repo_relative(runtime_manifest)

    roll_shadow_passed = bool(roll_shadow_validation_summary["validation_passed"])
    package_status = (
        "finalized_without_human_same_contract_signoff"
        if roll_shadow_passed
        else "finalized_with_runtime_roll_blocker"
    )
    promotion_decision = (
        "pending_human_same_contract_signoff"
        if roll_shadow_passed
        else "not_promotion_ready"
    )
    open_items = [
        "Human same-contract TradingView profile signoff is intentionally deferred for now.",
    ]
    if roll_shadow_passed:
        open_items.append(
            "The historical roll replay validated the FRVP shadow runtime across a real contract switch, but the live IBKR front-month handoff still remains an explicit operator action rather than a fully automatic collector roll."
        )
    else:
        blocking_findings = list(roll_shadow_validation_summary.get("blocking_findings") or [])
        if blocking_findings:
            open_items.extend(blocking_findings)
        open_items.append(
            "The historical roll replay did not clear the live/shadow roll-safe runtime gate, so promotion should stay blocked until the live runtime can recreate the required FRVP carry-through features at contract boundaries."
        )

    package_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_id": output_root.name,
        "package_status": package_status,
        "promotion_decision": promotion_decision,
        "model_id": args.model_id,
        "artifact_paths": {
            "threshold_summary_path": _repo_relative(args.threshold_summary_path),
            "baseline_threshold_summary_path": _repo_relative(args.baseline_threshold_summary_path),
            "backtest_summary_path": _repo_relative(args.backtest_summary_path),
            "baseline_backtest_summary_path": _repo_relative(args.baseline_backtest_summary_path),
            "placebo_summary_path": _repo_relative(args.placebo_summary_path),
            "roll_audit_summary_path": _repo_relative(args.roll_audit_summary_path),
            "roll_shadow_validation_summary_path": _repo_relative(args.roll_shadow_validation_summary_path),
            "shadow_selection_summary_path": _repo_relative(args.shadow_selection_summary_path),
        },
        "threshold_vs_prune": threshold_vs_prune,
        "promotion_readout": promotion_readout,
        "placebo": {
            "passed": bool(placebo_summary["gate_check"]["passed"]),
            "real_oof_average_precision": placebo_summary["real_model"]["oof_average_precision"],
            "shuffled_mean_oof_average_precision": placebo_summary["placebo_distribution"]["mean_oof_average_precision"],
            "shuffled_std_oof_average_precision": placebo_summary["placebo_distribution"]["std_oof_average_precision"],
            "placebo_gap": placebo_summary["gate_check"]["placebo_gap"],
        },
        "roll_audit": {
            "passed": bool(roll_audit_summary["gate_7_roll_audit_passed"]),
            "events_excluded_roll_span": roll_audit_summary["event_roll_audit"]["events_excluded_roll_span_reported"],
            "usable_roll_span_rows": roll_audit_summary["event_roll_audit"]["usable_roll_span_rows_in_events_csv"],
            "folds_with_roll_boundary_between": roll_audit_summary["fold_boundary_audit"]["folds_with_roll_boundary_between"],
            "minimum_gap_bars": roll_audit_summary["fold_boundary_audit"]["minimum_gap_bars"],
            "minimum_gap_hours": roll_audit_summary["fold_boundary_audit"]["minimum_gap_hours"],
            "pytest_passed": bool(roll_audit_summary["pytest"]["passed"]),
        },
        "roll_shadow_validation": {
            "passed": roll_shadow_passed,
            "roll_boundary": dict(roll_shadow_validation_summary["roll_boundary"]),
            "health_error_count": int(roll_shadow_validation_summary["health_error_count"]),
            "total_signal_decisions_persisted": int(
                roll_shadow_validation_summary["shadow_decision_counts"]["total_signal_decisions_persisted"]
            ),
            "feature_parity_attempted": bool(roll_shadow_validation_summary.get("feature_parity_attempted")),
            "feature_parity_passed": bool(roll_shadow_validation_summary.get("feature_parity_passed")),
            "missing_live_feature_names": list(roll_shadow_validation_summary.get("missing_live_feature_names") or []),
            "blocking_findings": list(roll_shadow_validation_summary.get("blocking_findings") or []),
        },
        "shadow_bundle": {
            "recommended_shadow_dashboard_models": list(
                shadow_selection_summary.get("recommended_shadow_dashboard_models") or []
            ),
            "family_leader_count": len(shadow_selection_summary.get("family_leaders") or []),
        },
        "open_items": open_items,
        "signoff_note": (
            "Threshold-vs-prune analysis, placebo evidence, roll audit evidence, shadow bundle packaging, and historical "
            "roll replay are now archived together for the long-continuation promotion baseline."
        ),
    }

    summary_path = output_root / "promotion_package_summary.json"
    markdown_path = output_root / "promotion_package_summary.md"
    summary_path.write_text(json.dumps(package_payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(package_payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_path": _repo_relative(summary_path),
                "markdown_path": _repo_relative(markdown_path),
                "package_status": package_payload["package_status"],
                "paper_trading_gate_passed": promotion_readout["paper_trading_gate_passed"],
                "placebo_passed": promotion_readout["placebo_passed"],
                "roll_audit_passed": promotion_readout["roll_audit_passed"],
                "roll_shadow_validation_passed": promotion_readout["roll_shadow_validation_passed"],
            },
            indent=2,
        )
    )
    return 0


def _find_model_output(summary_payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    for item in summary_payload.get("model_outputs", []):
        if item.get("model_id") == model_id:
            return item
    raise KeyError(f"Could not find model_id={model_id!r} in summary payload.")


def _read_policy_metrics(policy_evaluation_path: Path, *, dataset_split: str) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    with policy_evaluation_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("dataset_split") or "") != dataset_split:
                continue
            policy_name = str(row.get("policy_name") or "").strip()
            if not policy_name:
                continue
            metrics[policy_name] = {
                "event_f05": _coerce_float(row.get("event_f05")),
                "post_cost_expectancy_units": _coerce_float(row.get("post_cost_expectancy_units")),
                "net_pnl_units": _coerce_float(row.get("net_pnl_units")),
                "trades_per_week": _coerce_float(row.get("trades_per_week")),
                "candidate_signals": _coerce_int(row.get("candidate_signals")),
                "emitted_signals": _coerce_int(row.get("emitted_signals")),
                "abstain_count": _coerce_int(row.get("abstain_count")),
            }
    return metrics


def _build_policy_contract_audit(
    *,
    threshold_model_output: dict[str, Any],
    backtest_model_output: dict[str, Any],
) -> dict[str, Any]:
    static_selected_policy_name = str(threshold_model_output.get("selected_policy_name") or "global_threshold")
    walk_forward_policy_counts = {
        str(key): int(value)
        for key, value in dict(backtest_model_output.get("selected_policy_counts") or {}).items()
        if int(value) > 0
    }
    total_walk_forward_policies = int(sum(walk_forward_policy_counts.values()))
    walk_forward_dominant_policy_name = None
    if walk_forward_policy_counts:
        walk_forward_dominant_policy_name = max(
            walk_forward_policy_counts.items(),
            key=lambda item: (item[1], item[0] == static_selected_policy_name, item[0]),
        )[0]
    static_policy_count = int(walk_forward_policy_counts.get(static_selected_policy_name, 0))
    walk_forward_static_policy_share = (
        float(static_policy_count / total_walk_forward_policies)
        if total_walk_forward_policies > 0
        else None
    )
    return {
        "static_selected_policy_name": static_selected_policy_name,
        "static_selected_policy_reason": threshold_model_output.get("selected_policy_reason"),
        "static_selected_policy_contract": dict(threshold_model_output.get("selected_policy_contract") or {}),
        "static_qualified_policy_names": [
            str(value)
            for value in threshold_model_output.get("qualified_policy_names", [])
        ],
        "walk_forward_selected_policy_counts": walk_forward_policy_counts,
        "walk_forward_dominant_policy_name": walk_forward_dominant_policy_name,
        "walk_forward_policy_mix_is_multicontract": len(walk_forward_policy_counts) > 1,
        "static_vs_walk_forward_dominant_mismatch": bool(
            walk_forward_dominant_policy_name is not None
            and walk_forward_dominant_policy_name != static_selected_policy_name
        ),
        "walk_forward_static_policy_share": walk_forward_static_policy_share,
    }


def _metric_delta_block(*, baseline: dict[str, Any], targeted: dict[str, Any]) -> dict[str, Any]:
    baseline_metrics = baseline["overall_test_metrics"]
    targeted_metrics = targeted["overall_test_metrics"]
    baseline_wfe = baseline["walk_forward_efficiency"]
    targeted_wfe = targeted["walk_forward_efficiency"]
    return {
        "trade_count_delta": int(targeted_metrics["trade_count"] - baseline_metrics["trade_count"]),
        "total_net_pnl_units_delta": float(
            targeted_metrics["total_net_pnl_units"] - baseline_metrics["total_net_pnl_units"]
        ),
        "expectancy_units_delta": float(
            targeted_metrics["expectancy_units"] - baseline_metrics["expectancy_units"]
        ),
        "monthly_sharpe_delta": float(
            targeted_metrics["monthly_sharpe"] - baseline_metrics["monthly_sharpe"]
        ),
        "approx_deflated_sharpe_delta": float(
            targeted_metrics["approx_deflated_sharpe"] - baseline_metrics["approx_deflated_sharpe"]
        ),
        "max_drawdown_units_delta": float(
            targeted_metrics["max_drawdown_units"] - baseline_metrics["max_drawdown_units"]
        ),
        "overall_wfe_delta": float(
            targeted_wfe["overall_wfe"] - baseline_wfe["overall_wfe"]
        ),
        "positive_composite_expectancy_share_delta": float(
            targeted["positive_composite_expectancy_share"] - baseline["positive_composite_expectancy_share"]
        ),
    }


def _find_family_leader(selection_summary: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    for item in selection_summary.get("family_leaders", []):
        if item.get("model_id") == model_id:
            return item
    return None


def _find_runtime_manifest(model_id: str, *, shadow_selection_summary_path: Path) -> Path | None:
    bundle_root = shadow_selection_summary_path.parent
    manifest_path = bundle_root / model_id / "live_runtime_manifest.json"
    return manifest_path if manifest_path.exists() else None


def _render_markdown(payload: dict[str, Any]) -> str:
    promotion = payload["promotion_readout"]
    threshold_vs_prune = payload["threshold_vs_prune"]
    placebo = payload["placebo"]
    roll_audit = payload["roll_audit"]
    roll_shadow = payload["roll_shadow_validation"]
    return "\n".join(
        [
            "# FRVP Promotion Package",
            "",
            f"- Package status: `{payload['package_status']}`",
            f"- Promotion decision: `{payload['promotion_decision']}`",
            f"- Model: `{payload['model_id']}`",
            f"- Generated: `{payload['generated_at_utc']}`",
            "",
            "## Promotion Readout",
            "",
            f"- Paper-trading gate passed: `{promotion['paper_trading_gate_passed']}`",
            f"- Sharpe: `{promotion['selected_test_sharpe']:.3f}`",
            f"- DSR: `{promotion['selected_test_deflated_sharpe']:.3f}`",
            f"- WFE: `{promotion['overall_wfe']:.3f}`",
            f"- Max drawdown pct: `{promotion['selected_test_max_drawdown_pct']:.2f}`",
            f"- Trades: `{promotion['selected_test_trade_count']}`",
            "",
            "## Threshold vs Prune",
            "",
            f"- Static selected policy: `{threshold_vs_prune['selected_policy_name']}`",
            f"- Static selector note: `{threshold_vs_prune['selected_policy_reason']}`",
            f"- Walk-forward dominant policy: `{threshold_vs_prune['walk_forward_policy_audit']['walk_forward_dominant_policy_name']}`",
            f"- Walk-forward static policy share: `{threshold_vs_prune['walk_forward_policy_audit']['walk_forward_static_policy_share']:.4f}`",
            f"- Conclusion: {threshold_vs_prune['conclusion']}",
            "",
            "## Validation Evidence",
            "",
            f"- Placebo passed: `{placebo['passed']}` with gap `{placebo['placebo_gap']:.4f}`",
            f"- Roll audit passed: `{roll_audit['passed']}` with minimum fold gap `{roll_audit['minimum_gap_bars']}` bars",
            f"- Roll shadow validation passed: `{roll_shadow['passed']}` at `{roll_shadow['roll_boundary']['boundary_timestamp_utc']}`",
            "",
            "## Open Items",
            "",
            *[f"- {line}" for line in payload["open_items"]],
        ]
    ) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _coerce_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _coerce_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(float(value))


def _nested_get(payload: Any, *keys: Any) -> Any:
    current = payload
    for key in keys:
        if isinstance(current, list):
            if not isinstance(key, int) or key >= len(current):
                return None
            current = current[key]
            continue
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


if __name__ == "__main__":
    raise SystemExit(main())
