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
from ict.taxonomy import (
    ICTTradeType,
    ICT_TRADE_TYPE_TO_LABEL_FAMILY,
    ICT_TRADE_TYPE_TO_SETUP_FAMILIES,
    get_ict_taxonomy_snapshot,
)
from scripts.evaluation_contracts import resolve_saved_paper_trading_gate
from ote_live.features.manifest import (
    AbstainPolicy,
    LivePolicy,
    PolicyCostAssumptions,
    PolicyLineage,
    ThresholdConfig,
)
from ote_live.models.registry import build_direction_runtime_manifests, write_direction_runtime_manifests

SOURCE_REGISTRY_PATH = (
    REPO_ROOT
    / "models"
    / "ict_es_primary_model_registry_bootstrap_20260726_full.json"
)
THRESHOLD_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "ict_threshold_policies"
    / "ict_es_primary_bootstrap_20260726_full"
    / "run_summary.json"
)
BACKTEST_SUMMARY_PATH = (
    REPO_ROOT
    / "model_testing"
    / "reports"
    / "ict_backtests"
    / "ict_es_primary_bootstrap_20260726_full"
    / "run_summary.json"
)

OUTPUT_BUNDLE_ID = "ict_es_shadow_20260730"
OUTPUT_REGISTRY_PATH = REPO_ROOT / "models" / "ict_es_shadow_live_registry_20260730.json"
OUTPUT_POLICY_DIR = REPO_ROOT / "ote_live" / "policy_artifacts" / OUTPUT_BUNDLE_ID
OUTPUT_MANIFEST_DIR = REPO_ROOT / "ote_live" / "runtime_manifests" / OUTPUT_BUNDLE_ID
OUTPUT_REPORT_DIR = REPO_ROOT / "model_testing" / "reports" / "ict_backtests" / OUTPUT_BUNDLE_ID
OUTPUT_SELECTION_SUMMARY_PATH = OUTPUT_MANIFEST_DIR / "shadow_selection_summary.json"

PREPARED_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "ict_es_primary_refresh_20260724_spacing_refit_final_confirm"
    / "phase04_prepared"
)
PREPARED_SUMMARY_PATH = PREPARED_ROOT / "prepared" / "summary.json"
FEATURE_METADATA_PATH = (
    PREPARED_ROOT / "ict_es_phase06_merged_dataset.metadata.json"
)
LONG_FEATURE_PATH = (
    PREPARED_ROOT / "prepared" / "long_ict_continuation" / "features.json"
)
SHORT_FEATURE_PATH = (
    PREPARED_ROOT / "prepared" / "short_ict_continuation" / "features.json"
)
OUTPUT_RUN_SUMMARY_PATH = OUTPUT_REPORT_DIR / "run_summary.json"
OUTPUT_MODEL_SUMMARY_PATH = OUTPUT_REPORT_DIR / "model_summary.csv"

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
    trade_type: str
    direction: str
    model_id: str
    selected_for_dashboard: bool = True
    dashboard_priority: int = 99
    selection_reason: str = ""

    @property
    def label_family(self) -> str:
        return ICT_TRADE_TYPE_TO_LABEL_FAMILY[self.trade_type]

    @property
    def setup_families(self) -> tuple[str, ...]:
        return ICT_TRADE_TYPE_TO_SETUP_FAMILIES[self.trade_type]


SHADOW_BUNDLE_MODELS: tuple[ShadowBundleModelSpec, ...] = (
    ShadowBundleModelSpec(
        trade_type=ICTTradeType.META.value,
        direction="long",
        model_id="ict_long_meta_xgb_v1",
        dashboard_priority=1,
        selection_reason=(
            "Current leakage-safe ICT leader: accepted 8/8 promotion-quality gates "
            "with the strongest Sharpe in the refreshed roster."
        ),
    ),
    ShadowBundleModelSpec(
        trade_type=ICTTradeType.REVERSAL.value,
        direction="long",
        model_id="ict_long_reversal_xgb_v1",
        dashboard_priority=2,
        selection_reason=(
            "Accepted 8/8 promotion-quality gates and retained as the "
            "family-specific long reversal leader."
        ),
    ),
    ShadowBundleModelSpec(
        trade_type=ICTTradeType.REVERSAL.value,
        direction="short",
        model_id="ict_short_reversal_xgb_v1",
        dashboard_priority=3,
        selection_reason=(
            "Best refreshed short-side Sharpe, retained shadow-only because "
            "quarter breadth and single-trade concentration still fail."
        ),
    ),
    ShadowBundleModelSpec(
        trade_type=ICTTradeType.META.value,
        direction="short",
        model_id="ict_short_meta_xgb_v1",
        dashboard_priority=4,
        selection_reason=(
            "Positive pooled short sentinel retained shadow-only because "
            "quarter breadth and single-trade concentration still fail."
        ),
    ),
    ShadowBundleModelSpec(
        trade_type=ICTTradeType.CONTINUATION.value,
        direction="short",
        model_id="ict_short_continuation_xgb_v1",
        dashboard_priority=5,
        selection_reason=(
            "Latest short continuation branch retained for complete family coverage; "
            "it remains shadow-only and concentration-dependent."
        ),
    ),
    ShadowBundleModelSpec(
        trade_type=ICTTradeType.CONTINUATION.value,
        direction="long",
        model_id="ict_long_continuation_xgb_v1",
        dashboard_priority=6,
        selection_reason=(
            "Latest long continuation branch retained for complete family coverage; "
            "it remains shadow-only because post-cost economics are effectively flat."
        ),
    ),
)


def main() -> int:
    generated_at_utc = datetime.now(timezone.utc)
    OUTPUT_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_POLICY_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    source_registry = load_ote_model_registry(SOURCE_REGISTRY_PATH)
    threshold_summary = _read_json(THRESHOLD_SUMMARY_PATH)
    source_registry_payload = _read_json(SOURCE_REGISTRY_PATH)
    backtest_candidates = _load_backtest_candidates()

    registry_models: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    unified_model_outputs: list[dict[str, Any]] = []

    for spec in SHADOW_BUNDLE_MODELS:
        source_record = source_registry.get_model(spec.model_id)
        threshold_model_output = _get_model_output(threshold_summary, spec.model_id)
        backtest_summary_path, backtest_model_output = _select_best_backtest_output(
            model_id=spec.model_id,
            candidates=backtest_candidates,
        )

        resolved_thresholds = _resolve_thresholds(
            threshold_model_output=threshold_model_output,
            source_record=source_record,
        )
        abstain_payload = _build_abstain_policy_payload(threshold_model_output)

        registry_record = _build_registry_record(
            source_record=source_record,
            global_threshold=resolved_thresholds["global_threshold"],
            regime_thresholds=resolved_thresholds["regime_thresholds"],
            abstain_policy=abstain_payload,
            promotion_reason=(
                f"ICT ES shadow bundle candidate ({spec.trade_type}/{spec.direction}) "
                f"sourced from {backtest_summary_path.parent.name}."
            ),
        )
        registry_models.append(registry_record)

        live_policy = _build_live_policy(
            spec=spec,
            registry_record=registry_record,
            source_registry_path=SOURCE_REGISTRY_PATH,
            output_registry_path=OUTPUT_REGISTRY_PATH,
            threshold_model_output=threshold_model_output,
            abstain_payload=abstain_payload,
            backtest_model_output=backtest_model_output,
            backtest_summary_path=backtest_summary_path,
            threshold_summary_path=THRESHOLD_SUMMARY_PATH,
            bundle_run_summary_path=OUTPUT_RUN_SUMMARY_PATH,
        )
        _write_live_policy_artifacts(
            output_policy_dir=OUTPUT_POLICY_DIR,
            spec=spec,
            live_policy=live_policy,
            threshold_model_output=threshold_model_output,
            selection_reason=spec.selection_reason,
            backtest_model_output=backtest_model_output,
            backtest_summary_path=backtest_summary_path,
        )

        unified_model_outputs.append(
            _build_unified_model_output(
                base_model_output=backtest_model_output,
                spec=spec,
                backtest_summary_path=backtest_summary_path,
                threshold_summary_path=THRESHOLD_SUMMARY_PATH,
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
        "promotion_rules": source_registry_payload["promotion_rules"],
        "models": sorted(registry_models, key=lambda item: (item["direction"], item["model_id"])),
    }
    OUTPUT_REGISTRY_PATH.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")

    _write_model_summary_csv(OUTPUT_MODEL_SUMMARY_PATH, selection_rows)
    OUTPUT_RUN_SUMMARY_PATH.write_text(
        json.dumps(
            _build_unified_run_summary(
                generated_at_utc=generated_at_utc,
                registry_path=OUTPUT_REGISTRY_PATH,
                output_report_dir=OUTPUT_REPORT_DIR,
                model_summary_path=OUTPUT_MODEL_SUMMARY_PATH,
                unified_model_outputs=unified_model_outputs,
                source_run_summary_path=BACKTEST_SUMMARY_PATH,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    direction_manifests = build_direction_runtime_manifests(
        registry_path=OUTPUT_REGISTRY_PATH,
        prepared_summary_path=PREPARED_SUMMARY_PATH,
        feature_metadata_path=FEATURE_METADATA_PATH,
        long_feature_path=LONG_FEATURE_PATH,
        short_feature_path=SHORT_FEATURE_PATH,
        policy_backtest_summary_path=OUTPUT_RUN_SUMMARY_PATH,
        packaged_policy_dir=OUTPUT_POLICY_DIR,
        preferred_primary_model_ids={
            "long": "ict_long_meta_xgb_v1",
            "short": "ict_short_reversal_xgb_v1",
        },
    )
    write_direction_runtime_manifests(direction_manifests, output_dir=OUTPUT_MANIFEST_DIR)

    selection_summary_payload = _build_selection_summary_payload(
        generated_at_utc=generated_at_utc,
        registry_path=OUTPUT_REGISTRY_PATH,
        policy_dir=OUTPUT_POLICY_DIR,
        manifest_dir=OUTPUT_MANIFEST_DIR,
        run_summary_path=OUTPUT_RUN_SUMMARY_PATH,
        selection_rows=selection_rows,
    )
    OUTPUT_SELECTION_SUMMARY_PATH.write_text(
        json.dumps(selection_summary_payload, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "registry_path": _repo_relative_str(OUTPUT_REGISTRY_PATH),
                "policy_dir": _repo_relative_str(OUTPUT_POLICY_DIR),
                "manifest_dir": _repo_relative_str(OUTPUT_MANIFEST_DIR),
                "run_summary_path": _repo_relative_str(OUTPUT_RUN_SUMMARY_PATH),
                "selection_summary_path": _repo_relative_str(OUTPUT_SELECTION_SUMMARY_PATH),
                "model_count": len(selection_rows),
                "dashboard_model_order": [
                    row["model_id"]
                    for row in sorted(selection_rows, key=lambda row: row["dashboard_priority"])
                    if row["selected_for_dashboard"]
                ],
            },
            indent=2,
        )
    )
    return 0


def _load_backtest_candidates() -> list[tuple[Path, dict[str, Any]]]:
    if not BACKTEST_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Canonical ICT backtest summary is missing: {BACKTEST_SUMMARY_PATH}."
        )
    return [(BACKTEST_SUMMARY_PATH, _read_json(BACKTEST_SUMMARY_PATH))]


def _select_best_backtest_output(
    *,
    model_id: str,
    candidates: list[tuple[Path, dict[str, Any]]],
) -> tuple[Path, dict[str, Any]]:
    ranked: list[tuple[tuple[float, ...], Path, dict[str, Any]]] = []
    for summary_path, payload in candidates:
        try:
            model_output = _get_model_output(payload, model_id)
        except KeyError:
            continue
        ranked.append((_backtest_sort_key(model_output, run_summary=payload), summary_path, model_output))
    if not ranked:
        raise KeyError(f"No ICT backtest output was found for model_id={model_id!r}.")
    ranked.sort(key=lambda item: item[0], reverse=True)
    _, summary_path, model_output = ranked[0]
    return summary_path, model_output


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


def _build_abstain_policy_payload(threshold_model_output: dict[str, Any]) -> dict[str, Any] | None:
    targeted_filters = dict(threshold_model_output.get("targeted_filters") or {})
    if not targeted_filters:
        return None
    selected_policy_name = str(
        threshold_model_output.get("selected_policy_name") or "global_threshold"
    )
    apply_to_base_policy_variants = bool(
        targeted_filters.get("apply_to_base_policy_variants", False)
    )
    if "abstain" not in selected_policy_name and not apply_to_base_policy_variants:
        return None
    payload = {
        key: value
        for key, value in targeted_filters.items()
        if key in AbstainPolicy.model_fields
    }
    payload["enabled"] = True
    return payload


def _build_live_policy(
    *,
    spec: ShadowBundleModelSpec,
    registry_record: dict[str, Any],
    source_registry_path: Path,
    output_registry_path: Path,
    threshold_model_output: dict[str, Any],
    abstain_payload: dict[str, Any] | None,
    backtest_model_output: dict[str, Any],
    backtest_summary_path: Path,
    threshold_summary_path: Path,
    bundle_run_summary_path: Path,
) -> LivePolicy:
    selected_policy_name = str(threshold_model_output.get("selected_policy_name") or "global_threshold")
    qualified_policy_names = [str(value) for value in threshold_model_output.get("qualified_policy_names", [])]
    selected_policy_reason = threshold_model_output.get("selected_policy_reason")
    backtest_summary = _read_json(backtest_summary_path)
    targeted_filter_preset = backtest_summary.get("targeted_filter_preset")

    policy_notes = [
        f"ICT ES shadow dashboard branch for {spec.trade_type}/{spec.direction}.",
        f"Source backtest contract: {_repo_relative_str(backtest_summary_path)}.",
        f"Source threshold contract: {_repo_relative_str(threshold_summary_path)}.",
        spec.selection_reason,
    ]
    if targeted_filter_preset:
        policy_notes.append(f"Targeted filter preset applied: {targeted_filter_preset}.")
    if selected_policy_reason:
        policy_notes.append(f"Threshold-search selector note: {selected_policy_reason}.")
    walk_forward_dominant_policy_name = _dominant_policy_name(
        backtest_model_output
    )
    if (
        walk_forward_dominant_policy_name
        and walk_forward_dominant_policy_name != selected_policy_name
    ):
        policy_notes.append(
            "Static policy selection differs from the walk-forward dominant "
            f"policy ({selected_policy_name} vs. "
            f"{walk_forward_dominant_policy_name}); keep this model shadow-only "
            "until exact packaged-policy parity is confirmed."
        )

    return LivePolicy(
        model_id=registry_record["model_id"],
        direction=registry_record["direction"],
        backend=registry_record["backend"],
        calibration_method=registry_record["calibration_method"],
        policy_status="complete",
        thresholds=ThresholdConfig(
            global_threshold=registry_record.get("global_threshold"),
            regime_thresholds=registry_record.get("regime_thresholds"),
        ),
        abstain_policy=_coerce_live_abstain_policy(abstain_payload),
        cost_assumptions=PolicyCostAssumptions(
            fixed_slippage_pips_per_trade=ES_FIXED_SLIPPAGE_UNITS,
            commission_pips_per_trade=ES_COMMISSION_UNITS,
            session_spread_pips=dict(ES_SESSION_SPREAD_UNITS),
            targeted_filter_preset=targeted_filter_preset,
        ),
        lineage=PolicyLineage(
            threshold_registry_path=_repo_relative_str(output_registry_path),
            policy_source_type="ict_shadow_bundle_selection",
            policy_backtest_summary_path=_repo_relative_str(bundle_run_summary_path),
            policy_search_summary_path=_repo_relative_str(threshold_summary_path),
            policy_table_path=_repo_relative_str(_resolve_path(threshold_model_output["policy_table_path"])),
            policy_evaluation_path=_repo_relative_str(_resolve_path(threshold_model_output["policy_evaluation_path"])),
            active_registry_path=_repo_relative_str(source_registry_path),
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
    backtest_summary_path: Path,
) -> None:
    model_policy_dir = output_policy_dir / spec.model_id
    model_policy_dir.mkdir(parents=True, exist_ok=True)
    gate = resolve_saved_paper_trading_gate(
        backtest_model_output,
        run_summary=_read_json(backtest_summary_path),
    )
    (model_policy_dir / "live_policy.json").write_text(
        live_policy.model_dump_json(indent=2),
        encoding="utf-8",
    )
    selection_summary = {
        "trade_type": spec.trade_type,
        "label_family": spec.label_family,
        "setup_families": list(spec.setup_families),
        "family": spec.trade_type,
        "direction": spec.direction,
        "model_id": spec.model_id,
        "source_backtest_run": backtest_summary_path.parent.name,
        "selected_policy_name": live_policy.lineage.selected_policy_name,
        "walk_forward_dominant_policy_name": _dominant_policy_name(
            backtest_model_output
        ),
        "static_vs_walk_forward_dominant_mismatch": (
            _dominant_policy_name(backtest_model_output)
            not in {None, live_policy.lineage.selected_policy_name}
        ),
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
            "drawdown_gate_passed": gate["drawdown_gate_passed"],
            "accepted_for_paper_trading_gate": gate["accepted"],
            "raw_accepted_for_paper_trading_gate": gate["accepted_raw"],
            "promotion_quality_gate_eligible": gate["promotion_quality_gate_eligible"],
            "promotion_quality_disqualifiers": gate["promotion_quality_disqualifiers"],
        },
        "selected_policy_metrics": threshold_model_output.get("selected_policy_metrics", {}),
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
    model_output["trade_type"] = spec.trade_type
    model_output["label_family"] = spec.label_family
    model_output["setup_families"] = list(spec.setup_families)
    model_output["family"] = spec.trade_type
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
    gate = resolve_saved_paper_trading_gate(
        backtest_model_output,
        run_summary=_read_json(backtest_summary_path),
    )
    selected_policy_name = str(
        threshold_model_output.get("selected_policy_name") or "global_threshold"
    )
    walk_forward_dominant_policy_name = _dominant_policy_name(
        backtest_model_output
    )
    return {
        "trade_type": spec.trade_type,
        "label_family": spec.label_family,
        "setup_families": list(spec.setup_families),
        "setup_family_scope": ",".join(spec.setup_families),
        "family": spec.trade_type,
        "direction": spec.direction,
        "model_id": spec.model_id,
        "backend": source_record.backend,
        "artifact_path": registry_record["artifact_path"],
        "source_backtest_run": backtest_summary_path.parent.name,
        "selected_policy_name": selected_policy_name,
        "walk_forward_dominant_policy_name": walk_forward_dominant_policy_name,
        "static_vs_walk_forward_dominant_mismatch": (
            walk_forward_dominant_policy_name
            not in {None, selected_policy_name}
        ),
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
        "drawdown_gate_passed": gate["drawdown_gate_passed"],
        "accepted_for_paper_trading_gate": gate["accepted"],
        "raw_accepted_for_paper_trading_gate": gate["accepted_raw"],
        "promotion_quality_gate_eligible": gate["promotion_quality_gate_eligible"],
        "promotion_quality_disqualifiers": ",".join(gate["promotion_quality_disqualifiers"]),
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
    source_run_summary_path: Path,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(_read_json(source_run_summary_path)))
    payload.update(
        {
            "generated_at_utc": generated_at_utc.isoformat(),
            "taxonomy": get_ict_taxonomy_snapshot(),
            "output_root": _repo_relative_str(output_report_dir),
            "registry_path": _repo_relative_str(registry_path),
            "model_ids": [item["model_id"] for item in unified_model_outputs],
            "statuses": ["candidate"],
            "include_roles": [],
            "model_summary_path": _repo_relative_str(model_summary_path),
            "model_outputs": unified_model_outputs,
            "source_backtest_summary_path": _repo_relative_str(
                source_run_summary_path
            ),
        }
    )
    return payload


def _build_selection_summary_payload(
    *,
    generated_at_utc: datetime,
    registry_path: Path,
    policy_dir: Path,
    manifest_dir: Path,
    run_summary_path: Path,
    selection_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered_rows = sorted(selection_rows, key=lambda row: (row["direction"], row["trade_type"], row["dashboard_priority"]))
    recommended_shadow_dashboard_models = [
        {
            "rank": row["dashboard_priority"],
            "model_id": row["model_id"],
            "trade_type": row["trade_type"],
            "label_family": row["label_family"],
            "setup_families": row["setup_families"],
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
        "bundle_id": OUTPUT_BUNDLE_ID,
        "registry_path": _repo_relative_str(registry_path),
        "policy_dir": _repo_relative_str(policy_dir),
        "runtime_manifest_dir": _repo_relative_str(manifest_dir),
        "backtest_summary_path": _repo_relative_str(run_summary_path),
        "taxonomy": get_ict_taxonomy_snapshot(),
        "trade_type_leaders": ordered_rows,
        "family_leaders": ordered_rows,
        "recommended_shadow_dashboard_models": recommended_shadow_dashboard_models,
        "notes": [
            "This bundle intentionally keeps all six ICT direction-plus-trade-type branches in one registry so the shared ES collector can evaluate the full menu from one data feed.",
            "Models come from the July 27 leakage-safe full retrain with target-specific embargoes and ICT sequential bootstrap.",
            "Thresholds come from the matching July 27 policy search. Abstention is enabled only when the selected policy explicitly uses it or applies filters to base variants.",
            "All six records remain candidate/shadow models. The two accepted long branches require exact packaged-policy shadow parity before any active-status promotion.",
            "Terminology note: setup families are reversal/continuation, trade types are reversal/continuation/meta, and `family_leaders` remains as a legacy alias for `trade_type_leaders`.",
        ],
    }


def _write_model_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "trade_type",
        "label_family",
        "setup_family_scope",
        "family",
        "direction",
        "model_id",
        "backend",
        "artifact_path",
        "source_backtest_run",
        "selected_policy_name",
        "walk_forward_dominant_policy_name",
        "static_vs_walk_forward_dominant_mismatch",
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
        "raw_accepted_for_paper_trading_gate",
        "promotion_quality_gate_eligible",
        "promotion_quality_disqualifiers",
        "selected_for_dashboard",
        "dashboard_priority",
        "selection_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["direction"], item["trade_type"])):
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


def _get_model_output(summary_payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    for item in summary_payload.get("model_outputs", []):
        if item.get("model_id") == model_id:
            return item
    raise KeyError(f"Could not find model_id={model_id!r} in summary payload.")


def _dominant_policy_name(model_output: dict[str, Any]) -> str | None:
    counts = model_output.get("selected_policy_counts") or {}
    if not isinstance(counts, dict) or not counts:
        return None
    return str(max(counts.items(), key=lambda item: float(item[1]))[0])


def _backtest_sort_key(
    model_output: dict[str, Any],
    *,
    run_summary: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    overall = model_output.get("overall_test_metrics", {})
    walk_forward = model_output.get("walk_forward_efficiency", {})
    acceptance = model_output.get("acceptance", {})
    gate = resolve_saved_paper_trading_gate(model_output, run_summary=run_summary)
    max_drawdown_units = float(
        overall.get(
            "max_drawdown_units",
            overall.get("max_drawdown_pips", float("inf")),
        )
    )
    return (
        1.0 if gate["accepted"] else 0.0,
        float(sum(bool(value) for value in acceptance.values())),
        float(overall.get("monthly_sharpe", float("-inf"))),
        float(overall.get("approx_deflated_sharpe", float("-inf"))),
        float(overall.get("profit_factor", float("-inf"))),
        float(overall.get("expectancy_units", overall.get("expectancy_pips", float("-inf")))),
        float(walk_forward.get("overall_wfe", float("-inf"))),
        float(model_output.get("positive_composite_expectancy_share", float("-inf"))),
        float(overall.get("trade_count", 0.0)),
        -max_drawdown_units,
    )

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
