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


MODEL_ID = "ict_long_continuation_xgb_v1"
REGISTRY_PATH = REPO_ROOT / "models" / "ict_es_primary_model_registry_bootstrap_20260726_full.json"
REGIME_REPORT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ict_regime_slices" / "ict_es_primary_bootstrap_20260726_full"
OUTPUT_ROOT = REPO_ROOT / "model_testing" / "reports" / "ict_backtests" / "ict_long_continuation_q40_concentration_controls_20260830"


@dataclass(frozen=True)
class ControlSpec:
    label: str
    filters: dict[str, object]
    note: str


CONTROLS = (
    ControlSpec(
        label="q40_block_largest_pocket",
        filters={
            "minimum_probability_quantile": 0.40,
            "abstain_composite_session_pairs": (("strong_down_high", "new_york"),),
            "apply_to_base_policy_variants": True,
        },
        note="Block the one-off largest winner pocket.",
    ),
    ControlSpec(
        label="q40_no_new_york",
        filters={
            "minimum_probability_quantile": 0.40,
            "abstain_session_regimes": ("new_york",),
            "apply_to_base_policy_variants": True,
        },
        note="Remove New York to test whether Asia/london can carry the branch.",
    ),
    ControlSpec(
        label="q40_asia_only",
        filters={
            "minimum_probability_quantile": 0.40,
            "abstain_session_regimes": ("new_york", "london"),
            "apply_to_base_policy_variants": True,
        },
        note="Keep the main repeatable Asia pocket only.",
    ),
    ControlSpec(
        label="q40_loss_pocket_prune",
        filters={
            "minimum_probability_quantile": 0.40,
            "abstain_composite_session_pairs": (
                ("strong_down_low", "asia"),
                ("ranging_medium", "new_york"),
                ("strong_up_high", "new_york"),
                ("strong_up_medium", "new_york"),
            ),
            "apply_to_base_policy_variants": True,
        },
        note="Remove observed negative composite/session pockets while keeping the largest winner.",
    ),
    ControlSpec(
        label="q40_loss_pocket_prune_no_largest",
        filters={
            "minimum_probability_quantile": 0.40,
            "abstain_composite_session_pairs": (
                ("strong_down_high", "new_york"),
                ("strong_down_low", "asia"),
                ("ranging_medium", "new_york"),
                ("strong_up_high", "new_york"),
                ("strong_up_medium", "new_york"),
            ),
            "apply_to_base_policy_variants": True,
        },
        note="Remove negative pockets and the one-off largest winner pocket.",
    ),
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for spec in CONTROLS:
        preset_name = f"ict_long_continuation_{spec.label}_v1"
        TARGETED_FILTER_PRESETS[preset_name] = {MODEL_ID: spec.filters}
        run_root = OUTPUT_ROOT / spec.label

        summary = run_policy_backtest(
            regime_report_root=REGIME_REPORT_ROOT,
            output_root=run_root,
            registry_path=REGISTRY_PATH,
            model_ids=(MODEL_ID,),
            statuses=("candidate",),
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
        gate = model_output["paper_trading_gate"]
        acceptance = model_output["acceptance"]
        rows.append(
            {
                "label": spec.label,
                "note": spec.note,
                "targeted_filter_preset": preset_name,
                "output_root": str(run_root.relative_to(REPO_ROOT)),
                "trades": metrics["trade_count"],
                "net_ticks": metrics["total_net_pnl_units"],
                "expectancy_ticks": metrics["expectancy_units"],
                "profit_factor": metrics["profit_factor"],
                "sharpe": metrics["monthly_sharpe"],
                "dsr": metrics["approx_deflated_sharpe"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "wfe": wfe["overall_wfe"],
                "profitable_quarter_share": metrics["profitable_quarter_share"],
                "positive_composite_share": model_output["positive_composite_expectancy_share"],
                "largest_trade_share": metrics["largest_single_trade_share_of_total_pnl"],
                "accepted": gate["accepted"],
                "failed_gates": "; ".join(
                    key for key, value in acceptance.items() if value is False
                ),
            }
        )

    aggregate_csv = OUTPUT_ROOT / "concentration_control_summary.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (OUTPUT_ROOT / "concentration_control_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(OUTPUT_ROOT), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
