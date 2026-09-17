from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "model_testing" / "reports" / "probability_quantile_sweeps" / "frvp_ict_weak_discrimination_20260830"

TRADE_FILES = {
    "frvp_short_meta_xgb_v1": REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_backtests"
    / "frvp_es_primary_refresh_20260701"
    / "frvp_short_meta_xgb_v1"
    / "selected_test_trades.csv",
    "frvp_long_meta_xgb_v1": REPO_ROOT
    / "model_testing"
    / "reports"
    / "frvp_backtests"
    / "frvp_es_primary_refresh_20260701"
    / "frvp_long_meta_xgb_v1"
    / "selected_test_trades.csv",
    "ict_long_continuation_xgb_v1": REPO_ROOT
    / "model_testing"
    / "reports"
    / "ict_backtests"
    / "ict_es_primary_bootstrap_20260726_full"
    / "ict_long_continuation_xgb_v1"
    / "selected_test_trades.csv",
    "ict_short_continuation_xgb_v1": REPO_ROOT
    / "model_testing"
    / "reports"
    / "ict_backtests"
    / "ict_es_primary_bootstrap_20260726_full"
    / "ict_short_continuation_xgb_v1"
    / "selected_test_trades.csv",
}

QUANTILES: tuple[float | None, ...] = (None, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_id, path in TRADE_FILES.items():
        trades = pd.read_csv(path)
        trades["policy_probability"] = pd.to_numeric(trades["policy_probability"], errors="coerce")
        trades["net_pnl_units"] = pd.to_numeric(trades["net_pnl_units"], errors="coerce")
        trades["entry_datetime"] = pd.to_datetime(trades["entry_datetime"], utc=True, errors="coerce")
        trades = trades.dropna(subset=["policy_probability", "net_pnl_units", "entry_datetime"]).copy()
        baseline_net = float(trades["net_pnl_units"].sum())
        baseline_trades = int(len(trades))
        for quantile in QUANTILES:
            filtered = _apply_fold_quantile(trades, quantile)
            metrics = _summarize(filtered)
            rows.append(
                {
                    "model_id": model_id,
                    "label": "baseline" if quantile is None else f"q{int(quantile * 100):02d}",
                    "quantile": "" if quantile is None else quantile,
                    "trades": metrics["trades"],
                    "trade_retention_pct": _pct(metrics["trades"], baseline_trades),
                    "net_ticks": metrics["net_ticks"],
                    "net_change_ticks": metrics["net_ticks"] - baseline_net,
                    "expectancy_ticks": metrics["expectancy_ticks"],
                    "profit_factor": metrics["profit_factor"],
                    "hit_rate": metrics["hit_rate"],
                    "max_drawdown_ticks": metrics["max_drawdown_ticks"],
                    "largest_trade_share": metrics["largest_trade_share"],
                }
            )

    csv_path = OUTPUT_ROOT / "trade_level_quantile_screen.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_ROOT / "trade_level_quantile_screen.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(OUTPUT_ROOT), "rows": len(rows)}, indent=2))


def _apply_fold_quantile(trades: pd.DataFrame, quantile: float | None) -> pd.DataFrame:
    if quantile is None:
        return trades.copy()

    kept_parts: list[pd.DataFrame] = []
    for _, fold in trades.groupby("fold_id", sort=False):
        cutoff = float(fold["policy_probability"].quantile(float(quantile)))
        kept_parts.append(fold.loc[fold["policy_probability"] > cutoff].copy())
    if not kept_parts:
        return trades.iloc[0:0].copy()
    return pd.concat(kept_parts, ignore_index=True).sort_values("entry_datetime").reset_index(drop=True)


def _summarize(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "net_ticks": 0.0,
            "expectancy_ticks": 0.0,
            "profit_factor": None,
            "hit_rate": None,
            "max_drawdown_ticks": 0.0,
            "largest_trade_share": None,
        }
    pnl = trades.sort_values("entry_datetime")["net_pnl_units"].astype(float)
    wins = pnl.loc[pnl > 0.0]
    losses = pnl.loc[pnl < 0.0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    net = float(pnl.sum())
    return {
        "trades": int(len(pnl)),
        "net_ticks": net,
        "expectancy_ticks": float(net / len(pnl)),
        "profit_factor": None if gross_loss <= 0.0 else float(gross_win / gross_loss),
        "hit_rate": float(len(wins) / len(pnl)),
        "max_drawdown_ticks": _max_drawdown(pnl),
        "largest_trade_share": None if math.isclose(net, 0.0) else float(pnl.max() / net),
    }


def _max_drawdown(pnl: pd.Series) -> float:
    equity = pnl.cumsum()
    peak = equity.cummax()
    drawdown = peak - equity
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _pct(value: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(100.0 * value / denominator)


if __name__ == "__main__":
    main()
