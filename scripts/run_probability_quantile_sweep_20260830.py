from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ote_targeted_filter_presets import TARGETED_FILTER_PRESETS
from scripts.run_ote_policy_backtest import run_policy_backtest


OUTPUT_ROOT = REPO_ROOT / "model_testing" / "reports" / "probability_quantile_sweeps" / "frvp_ict_weak_discrimination_20260830"


@dataclass(frozen=True)
class SweepSpec:
    family: str
    model_id: str
    registry_path: Path
    regime_report_root: Path
    note: str


SPECS = (
    SweepSpec(
        family="FRVP",
        model_id="frvp_short_meta_xgb_v1",
        registry_path=REPO_ROOT / "models" / "frvp_es_primary_model_registry_refresh_20260701.json",
        regime_report_root=REPO_ROOT / "model_testing" / "reports" / "frvp_regime_slices" / "frvp_es_primary_refresh_20260701",
        note="Weak AP lift short-meta sentinel.",
    ),
    SweepSpec(
        family="FRVP",
        model_id="frvp_long_meta_xgb_v1",
        registry_path=REPO_ROOT / "models" / "frvp_es_primary_model_registry_refresh_20260701.json",
        regime_report_root=REPO_ROOT / "model_testing" / "reports" / "frvp_regime_slices" / "frvp_es_primary_refresh_20260701",
        note="Weak AP lift long-meta branch.",
    ),
    SweepSpec(
        family="ICT",
        model_id="ict_long_continuation_xgb_v1",
        registry_path=REPO_ROOT / "models" / "ict_es_primary_model_registry_bootstrap_20260726_full.json",
        regime_report_root=REPO_ROOT / "model_testing" / "reports" / "ict_regime_slices" / "ict_es_primary_bootstrap_20260726_full",
        note="Near-flat long-continuation branch.",
    ),
    SweepSpec(
        family="ICT",
        model_id="ict_short_continuation_xgb_v1",
        registry_path=REPO_ROOT / "models" / "ict_es_primary_model_registry_bootstrap_20260726_full.json",
        regime_report_root=REPO_ROOT / "model_testing" / "reports" / "ict_regime_slices" / "ict_es_primary_bootstrap_20260726_full",
        note="Weak AP lift short-continuation branch.",
    ),
)


QUANTILES: tuple[float | None, ...] = (None, 0.10, 0.20, 0.30, 0.40, 0.50)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for spec in SPECS:
        for quantile in QUANTILES:
            label = "baseline" if quantile is None else f"q{int(quantile * 100):02d}"
            preset_name = None
            if quantile is not None:
                preset_name = f"probability_quantile_sweep_{spec.model_id}_{label}"
                TARGETED_FILTER_PRESETS[preset_name] = {
                    spec.model_id: {
                        "minimum_probability_quantile": float(quantile),
                        "apply_to_base_policy_variants": True,
                    }
                }

            run_root = OUTPUT_ROOT / spec.model_id / label
            summary = run_policy_backtest(
                regime_report_root=spec.regime_report_root,
                output_root=run_root,
                registry_path=spec.registry_path,
                model_ids=(spec.model_id,),
                statuses=("active", "candidate"),
                min_train_years=2,
                min_folds=8,
                min_positive_events=50,
                min_events_per_month=3.0,
                min_trades_per_week=3.0,
                instrument="es",
                spread_cost_mode="session_schedule",
                targeted_filter_preset=preset_name,
            )
            model_output = summary["model_outputs"][0]
            metrics = model_output["overall_test_metrics"]
            wfe = model_output["walk_forward_efficiency"]
            acceptance = model_output["acceptance"]
            gate = model_output["paper_trading_gate"]
            rows.append(
                {
                    "family": spec.family,
                    "model_id": spec.model_id,
                    "note": spec.note,
                    "quantile": "" if quantile is None else quantile,
                    "label": label,
                    "output_root": str(run_root.relative_to(REPO_ROOT)),
                    "folds": model_output["fold_count"],
                    "trades": metrics["trade_count"],
                    "net_ticks": metrics["total_net_pnl_units"],
                    "expectancy_ticks": metrics["expectancy_units"],
                    "profit_factor": metrics["profit_factor"],
                    "sharpe": metrics["monthly_sharpe"],
                    "dsr": metrics["approx_deflated_sharpe"],
                    "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "largest_trade_share": metrics.get("largest_single_trade_share_of_total_pnl"),
                    "wfe": wfe["overall_wfe"],
                    "profitable_quarter_share": metrics.get("profitable_quarter_share"),
                    "positive_composite_share": model_output.get("positive_composite_expectancy_share"),
                    "accepted": gate["accepted"],
                    "drawdown_gate": gate["drawdown_gate_passed"],
                    "sharpe_gate": acceptance["annualized_sharpe_above_threshold"],
                    "dsr_gate": acceptance["dsr_above_threshold"],
                    "wfe_gate": acceptance["wfe_above_threshold"],
                    "concentration_gate": acceptance["largest_single_trade_share_below_limit"],
                }
            )

    csv_path = OUTPUT_ROOT / "quantile_sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (OUTPUT_ROOT / "quantile_sweep_results.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(OUTPUT_ROOT), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
