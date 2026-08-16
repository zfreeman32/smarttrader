from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ote_registry_loader import OTEModelRecord, load_ote_model_registry
from model_testing.promotion_gates import accepted_for_paper_trading, drawdown_acceptance_passed
from ote_live.features.manifest import (
    AbstainPolicy,
    LivePolicy,
    PolicyCostAssumptions,
    PolicyLineage,
    ThresholdConfig,
)
from ote_live.models.registry import build_direction_runtime_manifests, write_direction_runtime_manifests

DEFAULT_OUTPUT_REGISTRY_PATH = REPO_ROOT / "models" / "frvp_es_shadow_live_registry_20260721.json"
DEFAULT_OUTPUT_POLICY_DIR = REPO_ROOT / "ote_live" / "policy_artifacts" / "frvp_es_shadow_20260721"
DEFAULT_OUTPUT_MANIFEST_DIR = REPO_ROOT / "ote_live" / "runtime_manifests" / "frvp_es_shadow_20260721"
DEFAULT_OUTPUT_REPORT_DIR = (
    REPO_ROOT / "model_testing" / "reports" / "frvp_backtests" / "frvp_es_shadow_live_bundle_20260721"
)
DEFAULT_OUTPUT_SELECTION_SUMMARY_PATH = DEFAULT_OUTPUT_MANIFEST_DIR / "shadow_selection_summary.json"

ES_SESSION_SPREAD_UNITS = {
    "overlap": 1.0,
    "london": 1.0,
    "new_york": 1.0,
    "asia": 1.5,
    "off_hours": 2.0,
}
ES_FIXED_SLIPPAGE_UNITS = 0.25
ES_COMMISSION_UNITS = 0.4


@dataclass(frozen=True)
class ShadowBundleModelSpec:
    family: str
    direction: str
    model_id: str
    source_registry_path: str
    source_backtest_summary_path: str
    source_threshold_summary_path: str
    policy_registry_path: str | None = None
    selected_for_dashboard: bool = False
    dashboard_priority: int = 99
    selection_reason: str = ""


SHADOW_BUNDLE_MODELS: tuple[ShadowBundleModelSpec, ...] = (
    ShadowBundleModelSpec(
        family="continuation",
        direction="long",
        model_id="frvp_long_continuation_xgb_v1",
        source_registry_path="models/frvp_es_primary_model_registry_refresh_20260701.json",
        source_backtest_summary_path=(
            "model_testing/reports/frvp_regime_gated_deployment/frvp_regime_gated_deployment_20260721/backtest/long_continuation_v3_baseline/run_summary.json"
        ),
        source_threshold_summary_path=(
            "model_testing/reports/frvp_regime_gated_deployment/frvp_regime_gated_deployment_20260721/threshold/long_continuation_v3_baseline/run_summary.json"
        ),
        selected_for_dashboard=True,
        dashboard_priority=1,
        selection_reason=(
            "Best current FRVP branch overall and the July 21, 2026 extended-shadow baseline. The frozen v3 "
            "continuation contract stays promotion-near under the current policy stack: +7001.20 ticks, Sharpe "
            "1.187, WFE 2.344, profitable-quarter share 0.625, and accepted paper-trading gate True."
        ),
    ),
    ShadowBundleModelSpec(
        family="continuation",
        direction="short",
        model_id="frvp_short_continuation_tcn_v1",
        source_registry_path="models/frvp_es_primary_model_registry_refresh_20260701.json",
        source_backtest_summary_path=(
            "model_testing/reports/frvp_backtests/frvp_short_family_leaders_refresh_20260715_accountdd/run_summary.json"
        ),
        source_threshold_summary_path=(
            "model_testing/reports/frvp_threshold_policies/frvp_es_primary_refresh_20260701/run_summary.json"
        ),
        selection_reason=(
            "Least-bad short continuation family leader. The refreshed 2026-07-15 family-leader rerun confirms "
            "it stays economics-negative with 109.17% account drawdown, but it is still materially better than "
            "the short continuation XGBoost baseline and remains the right shadow-only sentinel for that branch."
        ),
    ),
    ShadowBundleModelSpec(
        family="meta",
        direction="long",
        model_id="frvp_long_meta_xgb_v1",
        source_registry_path="models/frvp_es_primary_model_registry_refresh_20260701.json",
        source_backtest_summary_path=(
            "model_testing/reports/frvp_backtests/frvp_long_meta_gatefix_v3_20260715_accountdd/run_summary.json"
        ),
        source_threshold_summary_path=(
            "model_testing/reports/frvp_threshold_policies/frvp_long_meta_gatefix_v3_20260703/run_summary.json"
        ),
        policy_registry_path="models/frvp_es_primary_model_registry_long_meta_recency_trial1_20260705.json",
        selection_reason=(
            "Best pooled long-meta checkpoint. The refreshed 2026-07-15 account-drawdown rerun confirms the "
            "saved v3 prune still delivers +9216.05 ticks with WFE 1.897, but account drawdown is 25.93% and "
            "the branch remains below the Sharpe and concentration promotion gates."
        ),
    ),
    ShadowBundleModelSpec(
        family="meta",
        direction="short",
        model_id="frvp_short_meta_xgb_v1",
        source_registry_path="models/frvp_es_primary_model_registry_refresh_20260701.json",
        source_backtest_summary_path=(
            "model_testing/reports/frvp_backtests/frvp_short_family_leaders_refresh_20260715_accountdd/run_summary.json"
        ),
        source_threshold_summary_path=(
            "model_testing/reports/frvp_threshold_policies/frvp_es_primary_refresh_20260701/run_summary.json"
        ),
        selected_for_dashboard=True,
        dashboard_priority=3,
        selection_reason=(
            "Only currently positive short-side FRVP branch. The refreshed 2026-07-15 family-leader rerun keeps "
            "it as the cleanest short sentinel, but Sharpe and DSR stay weak and account drawdown is 54.57%, so "
            "it remains shadow-only rather than promotion-ready."
        ),
    ),
    ShadowBundleModelSpec(
        family="reversal",
        direction="long",
        model_id="frvp_long_reversal_xgb_v1",
        source_registry_path="models/frvp_es_primary_model_registry_long_reversal_recency_trial1_v3_20260704.json",
        source_backtest_summary_path=(
            "model_testing/reports/frvp_regime_gated_deployment/frvp_regime_gated_deployment_20260721/backtest/long_reversal_recent2y_sdh_overlap_prune_v1/run_summary.json"
        ),
        source_threshold_summary_path=(
            "model_testing/reports/frvp_regime_gated_deployment/frvp_regime_gated_deployment_20260721/threshold/long_reversal_recent2y_sdh_overlap_prune_v1/run_summary.json"
        ),
        selected_for_dashboard=True,
        dashboard_priority=2,
        selection_reason=(
            "Operational selective-deployment reversal contract as of July 21, 2026. The saved research path "
            "`long_reversal_recent2y_sdh_overlap_prune_v1` is codified in the repo as "
            "`frvp_long_reversal_xgb_recent_regime_prune_v2`: 63 trades, +3677.05 ticks, Sharpe 1.480, DSR 1.179, "
            "WFE 2.162, profitable-quarter share 0.60, largest-single-trade share 0.099985, and accepted "
            "paper-trading gate True."
        ),
    ),
    ShadowBundleModelSpec(
        family="reversal",
        direction="short",
        model_id="frvp_short_reversal_xgb_v1",
        source_registry_path="models/frvp_es_primary_model_registry_refresh_20260701.json",
        source_backtest_summary_path=(
            "model_testing/reports/frvp_backtests/frvp_short_family_leaders_refresh_20260715_accountdd/run_summary.json"
        ),
        source_threshold_summary_path=(
            "model_testing/reports/frvp_threshold_policies/frvp_es_primary_refresh_20260701/run_summary.json"
        ),
        selection_reason=(
            "Best short reversal family leader on stability grounds. The refreshed 2026-07-15 family-leader rerun "
            "confirms economics are still negative and account drawdown is 48.70%, but the XGBoost branch still "
            "has better WFE, broader positive-composite coverage, and a cleaner live-policy contract than the TCN challenger."
        ),
    ),
)


def main() -> int:
    generated_at_utc = datetime.now(timezone.utc)

    output_registry_path = DEFAULT_OUTPUT_REGISTRY_PATH
    output_policy_dir = DEFAULT_OUTPUT_POLICY_DIR
    output_manifest_dir = DEFAULT_OUTPUT_MANIFEST_DIR
    output_report_dir = DEFAULT_OUTPUT_REPORT_DIR
    output_selection_summary_path = DEFAULT_OUTPUT_SELECTION_SUMMARY_PATH

    output_registry_path.parent.mkdir(parents=True, exist_ok=True)
    output_policy_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    refresh_registry_payload = _read_json(
        _resolve_path("models/frvp_es_primary_model_registry_refresh_20260701.json")
    )
    promotion_rules = dict(refresh_registry_payload["promotion_rules"])

    registry_models: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    unified_model_outputs: list[dict[str, Any]] = []

    for spec in SHADOW_BUNDLE_MODELS:
        source_registry = load_ote_model_registry(_resolve_path(spec.source_registry_path))
        source_record = source_registry.get_model(spec.model_id)

        policy_registry = (
            load_ote_model_registry(_resolve_path(spec.policy_registry_path))
            if spec.policy_registry_path
            else source_registry
        )
        policy_record = policy_registry.get_model(spec.model_id)

        backtest_summary_path = _resolve_path(spec.source_backtest_summary_path)
        backtest_summary = _read_json(backtest_summary_path)
        backtest_model_output = _get_model_output(backtest_summary, spec.model_id)

        threshold_summary_path = _resolve_path(spec.source_threshold_summary_path)
        threshold_summary = _read_json(threshold_summary_path)
        threshold_model_output = _get_model_output(threshold_summary, spec.model_id)

        resolved_thresholds = _resolve_thresholds(
            threshold_model_output=threshold_model_output,
            source_record=source_record,
        )
        abstain_payload = _resolve_abstain_policy_payload(
            policy_record=policy_record,
            selected_policy_name=resolved_thresholds["selected_policy_name"],
        )

        registry_record = _build_registry_record(
            source_record=source_record,
            global_threshold=resolved_thresholds["global_threshold"],
            regime_thresholds=resolved_thresholds["regime_thresholds"],
            abstain_policy=abstain_payload,
            promotion_reason=(
                f"FRVP ES shadow bundle family leader ({spec.family}/{spec.direction}) sourced from "
                f"{Path(spec.source_backtest_summary_path).parent.name}."
            ),
        )
        registry_models.append(registry_record)

        live_policy = _build_live_policy(
            spec=spec,
            registry_record=registry_record,
            threshold_model_output=threshold_model_output,
            abstain_payload=abstain_payload,
            backtest_summary_path=backtest_summary_path,
            threshold_summary_path=threshold_summary_path,
            generated_at_utc=generated_at_utc,
        )
        _write_live_policy_artifacts(
            output_policy_dir=output_policy_dir,
            spec=spec,
            live_policy=live_policy,
            threshold_model_output=threshold_model_output,
            selection_reason=spec.selection_reason,
            backtest_model_output=backtest_model_output,
        )

        unified_model_outputs.append(
            _build_unified_model_output(
                base_model_output=backtest_model_output,
                spec=spec,
                backtest_summary_path=backtest_summary_path,
                threshold_summary_path=threshold_summary_path,
                registry_record=registry_record,
            )
        )

        selection_rows.append(
            _build_selection_row(
                spec=spec,
                source_record=source_record,
                registry_record=registry_record,
                backtest_model_output=backtest_model_output,
                threshold_model_output=threshold_model_output,
                backtest_summary_path=backtest_summary_path,
            )
        )

    registry_payload = {
        "promotion_rules": promotion_rules,
        "models": sorted(registry_models, key=lambda item: (item["direction"], item["model_id"])),
    }
    output_registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")

    model_summary_path = output_report_dir / "model_summary.csv"
    _write_model_summary_csv(model_summary_path, selection_rows)

    unified_run_summary_path = output_report_dir / "run_summary.json"
    unified_run_summary_path.write_text(
        json.dumps(
            _build_unified_run_summary(
                generated_at_utc=generated_at_utc,
                registry_path=output_registry_path,
                output_report_dir=output_report_dir,
                model_summary_path=model_summary_path,
                unified_model_outputs=unified_model_outputs,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    direction_manifests = build_direction_runtime_manifests(
        registry_path=output_registry_path,
        prepared_summary_path="artifacts/frvp_es_primary_refresh_20260701/phase04/prepared/summary.json",
        feature_metadata_path=(
            "artifacts/frvp_es_primary_refresh_20260701/phase02/es_primary_frvp_phase04_dataset.csv.metadata.json"
        ),
        long_feature_path="artifacts/frvp_es_primary_refresh_20260701/phase04/prepared/long_frvp_continuation/features.json",
        short_feature_path="artifacts/frvp_es_primary_refresh_20260701/phase04/prepared/short_frvp_continuation/features.json",
        policy_backtest_summary_path=unified_run_summary_path,
        packaged_policy_dir=output_policy_dir,
        preferred_primary_model_ids={
            "long": "frvp_long_continuation_xgb_v1",
            "short": "frvp_short_meta_xgb_v1",
        },
    )
    write_direction_runtime_manifests(direction_manifests, output_dir=output_manifest_dir)

    selection_summary_payload = _build_selection_summary_payload(
        generated_at_utc=generated_at_utc,
        registry_path=output_registry_path,
        policy_dir=output_policy_dir,
        manifest_dir=output_manifest_dir,
        run_summary_path=unified_run_summary_path,
        selection_rows=selection_rows,
    )
    output_selection_summary_path.write_text(json.dumps(selection_summary_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "registry_path": _repo_relative_str(output_registry_path),
                "policy_dir": _repo_relative_str(output_policy_dir),
                "manifest_dir": _repo_relative_str(output_manifest_dir),
                "run_summary_path": _repo_relative_str(unified_run_summary_path),
                "selection_summary_path": _repo_relative_str(output_selection_summary_path),
                "family_leader_count": len(selection_rows),
                "recommended_shadow_dashboard_models": [
                    row["model_id"]
                    for row in sorted(selection_rows, key=lambda row: row["dashboard_priority"])
                    if row["selected_for_dashboard"]
                ],
            },
            indent=2,
        )
    )
    return 0


def _build_registry_record(
    *,
    source_record: OTEModelRecord,
    global_threshold: float | None,
    regime_thresholds: dict[str, float] | None,
    abstain_policy: dict[str, Any] | None,
    promotion_reason: str,
) -> dict[str, Any]:
    record = source_record.to_dict()
    record["global_threshold"] = global_threshold
    record["regime_thresholds"] = regime_thresholds
    record["abstain_policy"] = abstain_policy
    record["promotion_date"] = datetime.now(timezone.utc).date().isoformat()
    record["promotion_reason"] = promotion_reason
    record["status"] = "candidate"
    return record


def _resolve_thresholds(
    *,
    threshold_model_output: dict[str, Any],
    source_record: OTEModelRecord,
) -> dict[str, Any]:
    selected_policy_name = str(threshold_model_output.get("selected_policy_name") or "global_threshold")
    global_threshold = threshold_model_output.get("global_threshold")
    if global_threshold is None:
        global_threshold = source_record.global_threshold
    regime_thresholds = None
    if selected_policy_name.startswith("regime_threshold"):
        policy_table_path = _resolve_path(threshold_model_output["policy_table_path"])
        regime_thresholds = _load_regime_thresholds(policy_table_path)

    return {
        "selected_policy_name": selected_policy_name,
        "global_threshold": None if global_threshold is None else float(global_threshold),
        "regime_thresholds": regime_thresholds,
    }


def _resolve_abstain_policy_payload(
    *,
    policy_record: OTEModelRecord,
    selected_policy_name: str,
) -> dict[str, Any] | None:
    payload = policy_record.abstain_policy
    if payload is None:
        return None
    # Keep the saved hard-prune contract for FRVP branches that were improved via targeted filters,
    # even when the static threshold search ultimately selected the base global threshold.
    return dict(payload)


def _build_live_policy(
    *,
    spec: ShadowBundleModelSpec,
    registry_record: dict[str, Any],
    threshold_model_output: dict[str, Any],
    abstain_payload: dict[str, Any] | None,
    backtest_summary_path: Path,
    threshold_summary_path: Path,
    generated_at_utc: datetime,
) -> LivePolicy:
    selected_policy_name = str(threshold_model_output.get("selected_policy_name") or "global_threshold")
    qualified_policy_names = [
        str(value)
        for value in threshold_model_output.get("qualified_policy_names", [])
    ]
    selected_policy_reason = threshold_model_output.get("selected_policy_reason")
    targeted_filter_preset = _read_json(backtest_summary_path).get("targeted_filter_preset")

    policy_notes = [
        f"FRVP ES shadow family leader for {spec.family}/{spec.direction}.",
        f"Source backtest contract: {_repo_relative_str(backtest_summary_path)}.",
        f"Source threshold contract: {_repo_relative_str(threshold_summary_path)}.",
        spec.selection_reason,
    ]
    if targeted_filter_preset:
        policy_notes.append(f"Targeted filter preset applied: {targeted_filter_preset}.")
    if selected_policy_reason:
        policy_notes.append(f"Threshold-search selector note: {selected_policy_reason}.")
    if spec.policy_registry_path and spec.policy_registry_path != spec.source_registry_path:
        policy_notes.append(
            "Abstain policy was sourced from a separate research registry so the saved targeted-policy contract "
            "could be paired with the stronger artifact branch."
        )

    policy_status = "complete"
    thresholds = ThresholdConfig(
        global_threshold=registry_record.get("global_threshold"),
        regime_thresholds=registry_record.get("regime_thresholds"),
    )
    abstain_policy = _coerce_live_abstain_policy(abstain_payload)
    return LivePolicy(
        model_id=registry_record["model_id"],
        direction=registry_record["direction"],
        backend=registry_record["backend"],
        calibration_method=registry_record["calibration_method"],
        policy_status=policy_status,
        thresholds=thresholds,
        abstain_policy=abstain_policy,
        cost_assumptions=PolicyCostAssumptions(
            fixed_slippage_pips_per_trade=ES_FIXED_SLIPPAGE_UNITS,
            commission_pips_per_trade=ES_COMMISSION_UNITS,
            session_spread_pips=dict(ES_SESSION_SPREAD_UNITS),
            targeted_filter_preset=targeted_filter_preset,
        ),
        lineage=PolicyLineage(
            threshold_registry_path=_repo_relative_str(DEFAULT_OUTPUT_REGISTRY_PATH),
            policy_source_type="frvp_shadow_bundle_selection",
            policy_backtest_summary_path=_repo_relative_str(DEFAULT_OUTPUT_REPORT_DIR / "run_summary.json"),
            policy_search_summary_path=_repo_relative_str(threshold_summary_path),
            policy_table_path=_repo_relative_str(_resolve_path(threshold_model_output["policy_table_path"])),
            policy_evaluation_path=_repo_relative_str(_resolve_path(threshold_model_output["policy_evaluation_path"])),
            active_registry_path=_repo_relative_str(_resolve_path(spec.source_registry_path)),
            source_model_id=registry_record["model_id"],
            selected_policy_name=selected_policy_name,
            qualified_policy_names=qualified_policy_names,
            source_match_type="artifact_path",
            notes=policy_notes,
        ),
    )


def _write_live_policy_artifacts(
    *,
    output_policy_dir: Path,
    spec: ShadowBundleModelSpec,
    live_policy: LivePolicy,
    threshold_model_output: dict[str, Any],
    selection_reason: str,
    backtest_model_output: dict[str, Any],
) -> None:
    model_policy_dir = output_policy_dir / spec.model_id
    model_policy_dir.mkdir(parents=True, exist_ok=True)
    (model_policy_dir / "live_policy.json").write_text(
        live_policy.model_dump_json(indent=2),
        encoding="utf-8",
    )

    selection_summary = {
        "family": spec.family,
        "direction": spec.direction,
        "model_id": spec.model_id,
        "selected_policy_name": live_policy.lineage.selected_policy_name,
        "qualified_policy_names": live_policy.lineage.qualified_policy_names,
        "selection_reason": selection_reason,
        "thresholds": {
            "global_threshold": live_policy.thresholds.global_threshold,
            "regime_thresholds": live_policy.thresholds.regime_thresholds,
        },
        "walk_forward_snapshot": {
            "selected_test_trades": backtest_model_output["overall_test_metrics"]["trade_count"],
            "selected_test_net_pnl_units": backtest_model_output["overall_test_metrics"]["total_net_pnl_units"],
            "selected_test_expectancy_units": backtest_model_output["overall_test_metrics"]["expectancy_units"],
            "selected_test_sharpe": backtest_model_output["overall_test_metrics"]["monthly_sharpe"],
            "selected_test_approx_deflated_sharpe": backtest_model_output["overall_test_metrics"]["approx_deflated_sharpe"],
            "overall_wfe": backtest_model_output["walk_forward_efficiency"]["overall_wfe"],
            "profitable_quarter_share": backtest_model_output["overall_test_metrics"]["profitable_quarter_share"],
            "positive_composite_expectancy_share": backtest_model_output["positive_composite_expectancy_share"],
            "drawdown_gate_passed": drawdown_acceptance_passed(backtest_model_output["acceptance"]),
            "accepted_for_paper_trading_gate": accepted_for_paper_trading(backtest_model_output["acceptance"]),
        },
        "selected_policy_metrics": threshold_model_output.get("selected_policy_metrics", {}),
        "policy_contract_audit": _build_policy_contract_audit(
            threshold_model_output=threshold_model_output,
            backtest_model_output=backtest_model_output,
        ),
    }
    (model_policy_dir / "policy_selection.json").write_text(
        json.dumps(selection_summary, indent=2),
        encoding="utf-8",
    )


def _build_unified_model_output(
    *,
    base_model_output: dict[str, Any],
    spec: ShadowBundleModelSpec,
    backtest_summary_path: Path,
    threshold_summary_path: Path,
    registry_record: dict[str, Any],
) -> dict[str, Any]:
    model_output = json.loads(json.dumps(base_model_output))
    model_output["source_backtest_summary_path"] = _repo_relative_str(backtest_summary_path)
    model_output["source_threshold_summary_path"] = _repo_relative_str(threshold_summary_path)
    model_output["family"] = spec.family
    model_output["selection_reason"] = spec.selection_reason
    model_output["selected_for_dashboard"] = spec.selected_for_dashboard
    model_output["dashboard_priority"] = spec.dashboard_priority
    model_output["live_registry_thresholds"] = {
        "global_threshold": registry_record.get("global_threshold"),
        "regime_thresholds": registry_record.get("regime_thresholds"),
    }
    return model_output


def _build_selection_row(
    *,
    spec: ShadowBundleModelSpec,
    source_record: OTEModelRecord,
    registry_record: dict[str, Any],
    backtest_model_output: dict[str, Any],
    threshold_model_output: dict[str, Any],
    backtest_summary_path: Path,
) -> dict[str, Any]:
    overall = backtest_model_output["overall_test_metrics"]
    walk_forward = backtest_model_output["walk_forward_efficiency"]
    accepted = accepted_for_paper_trading(backtest_model_output["acceptance"])
    contract_audit = _build_policy_contract_audit(
        threshold_model_output=threshold_model_output,
        backtest_model_output=backtest_model_output,
    )
    return {
        "family": spec.family,
        "direction": spec.direction,
        "model_id": spec.model_id,
        "backend": source_record.backend,
        "artifact_path": registry_record["artifact_path"],
        "source_backtest_run": Path(backtest_summary_path).parent.name,
        "selected_policy_name": str(threshold_model_output.get("selected_policy_name") or "global_threshold"),
        "selected_test_trades": overall["trade_count"],
        "selected_test_net_pnl_units": overall["total_net_pnl_units"],
        "selected_test_expectancy_units": overall["expectancy_units"],
        "selected_test_sharpe": overall["monthly_sharpe"],
        "selected_test_approx_deflated_sharpe": overall["approx_deflated_sharpe"],
        "selected_test_max_drawdown_pct": overall.get("max_drawdown_pct"),
        "selected_test_max_profit_retracement_pct": overall.get("max_profit_retracement_pct"),
        "overall_wfe": walk_forward["overall_wfe"],
        "profitable_quarter_share": overall["profitable_quarter_share"],
        "positive_composite_expectancy_share": backtest_model_output["positive_composite_expectancy_share"],
        "drawdown_gate_passed": drawdown_acceptance_passed(backtest_model_output["acceptance"]),
        "accepted_for_paper_trading_gate": accepted,
        "walk_forward_dominant_policy_name": contract_audit["walk_forward_dominant_policy_name"],
        "walk_forward_policy_mix_is_multicontract": contract_audit["walk_forward_policy_mix_is_multicontract"],
        "static_vs_walk_forward_dominant_mismatch": contract_audit["static_vs_walk_forward_dominant_mismatch"],
        "walk_forward_static_policy_share": contract_audit["walk_forward_static_policy_share"],
        "selected_for_dashboard": spec.selected_for_dashboard,
        "dashboard_priority": spec.dashboard_priority,
        "selection_reason": spec.selection_reason,
    }


def _build_unified_run_summary(
    *,
    generated_at_utc: datetime,
    registry_path: Path,
    output_report_dir: Path,
    model_summary_path: Path,
    unified_model_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at_utc": generated_at_utc.isoformat(),
        "regime_report_root": None,
        "output_root": _repo_relative_str(output_report_dir),
        "registry_path": _repo_relative_str(registry_path),
        "model_ids": [item["model_id"] for item in unified_model_outputs],
        "statuses": ["candidate"],
        "include_roles": [],
        "min_train_years": None,
        "test_window_months": None,
        "rolling_step_months": None,
        "min_scheduled_test_start": None,
        "min_folds": None,
        "max_folds": None,
        "min_positive_events": 50,
        "min_events_per_month": 3.0,
        "min_trades_per_week": 3.0,
        "minimum_dsr": 0.3,
        "dsr_effective_trials_override": None,
        "evaluation_costs": {
            "instrument": "es",
            "unit_label": "ticks",
            "price_increment": 0.25,
            "session_spread_units": dict(ES_SESSION_SPREAD_UNITS),
            "fixed_slippage_units_per_trade": ES_FIXED_SLIPPAGE_UNITS,
            "commission_units_per_trade": ES_COMMISSION_UNITS,
            "tick_size": 0.25,
            "tick_value": 12.5,
        },
        "targeted_filter_preset": "mixed_family_leader_bundle",
        "fixed_slippage_pips_per_trade": ES_FIXED_SLIPPAGE_UNITS,
        "commission_pips_per_trade": ES_COMMISSION_UNITS,
        "session_spread_pips": dict(ES_SESSION_SPREAD_UNITS),
        "model_summary_path": _repo_relative_str(model_summary_path),
        "model_outputs": unified_model_outputs,
    }


def _build_selection_summary_payload(
    *,
    generated_at_utc: datetime,
    registry_path: Path,
    policy_dir: Path,
    manifest_dir: Path,
    run_summary_path: Path,
    selection_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered_rows = sorted(selection_rows, key=lambda row: (row["direction"], row["family"], row["dashboard_priority"]))
    recommended_shadow_dashboard_models = [
        {
            "rank": row["dashboard_priority"],
            "model_id": row["model_id"],
            "family": row["family"],
            "direction": row["direction"],
            "source_backtest_run": row["source_backtest_run"],
            "selected_test_net_pnl_units": row["selected_test_net_pnl_units"],
            "selected_test_sharpe": row["selected_test_sharpe"],
            "overall_wfe": row["overall_wfe"],
            "selection_reason": row["selection_reason"],
        }
        for row in sorted(selection_rows, key=lambda row: row["dashboard_priority"])
        if row["selected_for_dashboard"]
    ]
    return {
        "generated_at_utc": generated_at_utc.isoformat(),
        "bundle_id": "frvp_es_shadow_20260721",
        "registry_path": _repo_relative_str(registry_path),
        "policy_dir": _repo_relative_str(policy_dir),
        "runtime_manifest_dir": _repo_relative_str(manifest_dir),
        "backtest_summary_path": _repo_relative_str(run_summary_path),
        "family_leaders": ordered_rows,
        "recommended_shadow_dashboard_models": recommended_shadow_dashboard_models,
        "notes": [
            "This bundle intentionally keeps all six family leaders in one registry so the live stack can load a complete FRVP menu.",
            "The recommended first dashboard subset is narrower: strongest long continuation, operational selective-deployment long reversal, and the only positive short-side meta sentinel.",
            "This bundle starts the extended FRVP shadow observation window on 2026-07-21 with continuation v3 as the primary baseline.",
        ],
    }


def _write_model_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "family",
        "direction",
        "model_id",
        "backend",
        "artifact_path",
        "source_backtest_run",
        "selected_policy_name",
        "selected_test_trades",
        "selected_test_net_pnl_units",
        "selected_test_expectancy_units",
        "selected_test_sharpe",
        "selected_test_approx_deflated_sharpe",
        "selected_test_max_drawdown_pct",
        "selected_test_max_profit_retracement_pct",
        "overall_wfe",
        "profitable_quarter_share",
        "positive_composite_expectancy_share",
        "drawdown_gate_passed",
        "accepted_for_paper_trading_gate",
        "walk_forward_dominant_policy_name",
        "walk_forward_policy_mix_is_multicontract",
        "static_vs_walk_forward_dominant_mismatch",
        "walk_forward_static_policy_share",
        "selected_for_dashboard",
        "dashboard_priority",
        "selection_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["direction"], item["family"])):
            writer.writerow({field: row.get(field) for field in fieldnames})


def _coerce_live_abstain_policy(payload: dict[str, Any] | None) -> AbstainPolicy:
    if not payload:
        return AbstainPolicy(
            enabled=False,
            abstain_high_stress=False,
            abstain_off_hours=False,
            cooldown_bars=0,
            minimum_expected_move_to_spread=0.0,
            abstain_session_regimes=[],
            abstain_composite_regimes=[],
            abstain_composite_session_pairs=[],
            abstain_composite_stress_pairs=[],
            minimum_probability_quantile=None,
        )

    return AbstainPolicy(
        enabled=bool(payload.get("enabled", True)),
        abstain_high_stress=bool(payload.get("abstain_high_stress", True)),
        abstain_off_hours=bool(payload.get("abstain_off_hours", True)),
        cooldown_bars=int(payload.get("cooldown_bars", 4)),
        minimum_expected_move_to_spread=float(payload.get("minimum_expected_move_to_spread", 2.0)),
        abstain_session_regimes=[str(value) for value in payload.get("abstain_session_regimes", [])],
        abstain_composite_regimes=[str(value) for value in payload.get("abstain_composite_regimes", [])],
        abstain_composite_session_pairs=_coerce_pair_list(payload.get("abstain_composite_session_pairs", [])),
        abstain_composite_stress_pairs=_coerce_pair_list(payload.get("abstain_composite_stress_pairs", [])),
        minimum_probability_quantile=(
            None
            if payload.get("minimum_probability_quantile") is None
            else float(payload["minimum_probability_quantile"])
        ),
    )


def _coerce_pair_list(values: Any) -> list[tuple[str, str]]:
    if not isinstance(values, (list, tuple)):
        return []
    pairs: list[tuple[str, str]] = []
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            pairs.append((str(value[0]), str(value[1])))
    return pairs


def _load_regime_thresholds(policy_table_path: Path) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    with policy_table_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            regime = str(row.get("composite_regime") or "").strip()
            threshold = row.get("threshold")
            if not regime or threshold in {None, ""}:
                continue
            thresholds[regime] = float(threshold)
    return thresholds


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


def _get_model_output(summary_payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    for item in summary_payload.get("model_outputs", []):
        if item.get("model_id") == model_id:
            return item
    raise KeyError(f"Could not find model_id={model_id!r} in summary payload.")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (REPO_ROOT / path_obj).resolve()


def _repo_relative_str(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return path_obj.as_posix()
    return path_obj.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
