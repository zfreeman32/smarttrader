"""
Deep analysis for OTE labels and human-review overrides.

Outputs:
- summary.txt
- summary.json
- annual_counts.csv
- hourly_rates.csv
- override_summary.csv
- ote_label_analysis.html
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    go = None
    make_subplots = None
    HAS_PLOTLY = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_ote_labels.csv"
DEFAULT_SWINGS = PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_ote_swings.csv"
DEFAULT_OVERRIDES = PROJECT_ROOT / "data" / "labeling" / "review" / "ote_label_overrides.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "labeling" / "analysis"


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    take_profit_atr: float = 1.0
    stop_loss_atr: float = 2.0
    max_holding_bars: int = 120
    compound_equity: bool = False


@dataclass
class TradeResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    structural_atr: float
    stop_price: float
    take_profit_price: float
    bars_held: int
    exit_reason: str
    pnl_r: float
    pnl_amount: float
    equity_before: float
    equity_after: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep analysis for reviewed OTE labels.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--swings", type=Path, default=DEFAULT_SWINGS)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def apply_overrides(labels: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    reviewed = labels.copy()
    if overrides.empty:
        return reviewed
    latest = overrides.sort_values(["timestamp"]).groupby(["timestamp", "target_label"], as_index=False).tail(1)
    ts_to_idx = pd.Series(reviewed.index.values, index=reviewed["timestamp"])
    for row in latest.itertuples(index=False):
        idx = ts_to_idx.get(row.timestamp)
        if idx is None or row.target_label not in reviewed.columns:
            continue
        reviewed.at[idx, row.target_label] = int(row.value)
    return reviewed


def bars_between_events(mask: pd.Series) -> pd.Series:
    idx = np.flatnonzero(mask.astype(bool).values)
    if len(idx) < 2:
        return pd.Series(dtype=float)
    return pd.Series(np.diff(idx))


def session_bucket(hour: int) -> str:
    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 12:
        return "London"
    if 12 <= hour < 17:
        return "NY AM"
    return "NY PM/Off"


def run_label_backtest(labels: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    usable = labels.loc[~labels["warmup_mask"]].copy() if "warmup_mask" in labels.columns else labels.copy()
    if usable.empty:
        return pd.DataFrame(), {}

    long_mask = usable["label_long_entry"].fillna(0).astype(bool)
    short_mask = usable["label_short_entry"].fillna(0).astype(bool)
    entry_mask = long_mask | short_mask
    entry_candidates = usable.loc[entry_mask].copy()
    entry_candidates["side"] = np.where(
        entry_candidates["label_long_entry"].fillna(0).astype(bool),
        "long",
        "short",
    )
    entry_candidates = entry_candidates.loc[
        entry_candidates["structural_atr"].notna() & (entry_candidates["structural_atr"] > 0)
    ]

    if entry_candidates.empty:
        return pd.DataFrame(), {}

    trades: list[TradeResult] = []
    equity = float(config.initial_capital)
    base_risk_amount = float(config.initial_capital * config.risk_per_trade)
    usable_reset = usable.reset_index(drop=True)
    timestamp_to_pos = pd.Series(np.arange(len(usable_reset)), index=usable_reset["timestamp"])
    next_allowed_pos = 0

    for row in entry_candidates.itertuples(index=False):
        entry_pos = int(timestamp_to_pos[row.timestamp])
        if entry_pos < next_allowed_pos:
            continue

        entry_price = float(row.close)
        atr = float(row.structural_atr)
        if row.side == "long":
            stop_price = entry_price - config.stop_loss_atr * atr
            take_profit_price = entry_price + config.take_profit_atr * atr
        else:
            stop_price = entry_price + config.stop_loss_atr * atr
            take_profit_price = entry_price - config.take_profit_atr * atr

        max_exit_pos = min(entry_pos + config.max_holding_bars, len(usable_reset) - 1)
        exit_pos = max_exit_pos
        exit_reason = "timeout"
        exit_price = float(usable_reset.iloc[exit_pos]["close"])

        for future_pos in range(entry_pos + 1, max_exit_pos + 1):
            future_row = usable_reset.iloc[future_pos]
            high = float(future_row["high"])
            low = float(future_row["low"])

            if row.side == "long":
                hit_stop = low <= stop_price
                hit_target = high >= take_profit_price
                if hit_stop and hit_target:
                    exit_pos = future_pos
                    exit_reason = "stop_and_target_same_bar"
                    exit_price = stop_price
                    break
                if hit_stop:
                    exit_pos = future_pos
                    exit_reason = "stop_loss"
                    exit_price = stop_price
                    break
                if hit_target:
                    exit_pos = future_pos
                    exit_reason = "take_profit"
                    exit_price = take_profit_price
                    break
            else:
                hit_stop = high >= stop_price
                hit_target = low <= take_profit_price
                if hit_stop and hit_target:
                    exit_pos = future_pos
                    exit_reason = "stop_and_target_same_bar"
                    exit_price = stop_price
                    break
                if hit_stop:
                    exit_pos = future_pos
                    exit_reason = "stop_loss"
                    exit_price = stop_price
                    break
                if hit_target:
                    exit_pos = future_pos
                    exit_reason = "take_profit"
                    exit_price = take_profit_price
                    break

        bars_held = int(exit_pos - entry_pos)
        if row.side == "long":
            pnl_r = (exit_price - entry_price) / (config.stop_loss_atr * atr)
        else:
            pnl_r = (entry_price - exit_price) / (config.stop_loss_atr * atr)

        equity_before = equity
        risk_amount = equity_before * config.risk_per_trade if config.compound_equity else base_risk_amount
        pnl_amount = risk_amount * pnl_r
        equity = equity_before + pnl_amount

        trades.append(
            TradeResult(
                entry_time=row.timestamp,
                exit_time=usable_reset.iloc[exit_pos]["timestamp"],
                side=row.side,
                entry_price=entry_price,
                exit_price=float(exit_price),
                structural_atr=atr,
                stop_price=float(stop_price),
                take_profit_price=float(take_profit_price),
                bars_held=bars_held,
                exit_reason=exit_reason,
                pnl_r=float(pnl_r),
                pnl_amount=float(pnl_amount),
                equity_before=float(equity_before),
                equity_after=float(equity),
            )
        )
        next_allowed_pos = exit_pos + 1

    trades_df = pd.DataFrame(asdict(trade) for trade in trades)
    if trades_df.empty:
        return trades_df, {}

    wins = trades_df["pnl_amount"] > 0
    gross_profit = float(trades_df.loc[wins, "pnl_amount"].sum())
    gross_loss = float(-trades_df.loc[~wins, "pnl_amount"].sum())
    equity_curve = trades_df["equity_after"].to_numpy(dtype=float)
    peaks = np.maximum.accumulate(equity_curve)
    drawdown_pct = np.where(peaks > 0, (equity_curve - peaks) / peaks * 100.0, 0.0)
    returns = trades_df["pnl_amount"] / trades_df["equity_before"].replace(0, np.nan)

    stats = {
        "initial_capital": float(config.initial_capital),
        "final_capital": float(trades_df["equity_after"].iloc[-1]),
        "net_pnl": float(trades_df["pnl_amount"].sum()),
        "total_return_pct": float((trades_df["equity_after"].iloc[-1] / config.initial_capital - 1.0) * 100.0),
        "total_trades": int(len(trades_df)),
        "long_trades": int((trades_df["side"] == "long").sum()),
        "short_trades": int((trades_df["side"] == "short").sum()),
        "wins": int(wins.sum()),
        "losses": int((~wins).sum()),
        "win_rate_pct": float(wins.mean() * 100.0),
        "avg_trade_r": float(trades_df["pnl_r"].mean()),
        "avg_win_r": float(trades_df.loc[wins, "pnl_r"].mean() if wins.any() else 0.0),
        "avg_loss_r": float(trades_df.loc[~wins, "pnl_r"].mean() if (~wins).any() else 0.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": float(drawdown_pct.min()),
        "avg_bars_held": float(trades_df["bars_held"].mean()),
        "median_bars_held": float(trades_df["bars_held"].median()),
        "take_profit_exits": int((trades_df["exit_reason"] == "take_profit").sum()),
        "stop_loss_exits": int((trades_df["exit_reason"] == "stop_loss").sum() + (trades_df["exit_reason"] == "stop_and_target_same_bar").sum()),
        "timeout_exits": int((trades_df["exit_reason"] == "timeout").sum()),
        "same_bar_conflicts": int((trades_df["exit_reason"] == "stop_and_target_same_bar").sum()),
        "risk_per_trade_pct": float(config.risk_per_trade * 100.0),
        "risk_model": "compound" if config.compound_equity else "fixed_from_initial_capital",
        "take_profit_atr": float(config.take_profit_atr),
        "stop_loss_atr": float(config.stop_loss_atr),
        "max_holding_bars": int(config.max_holding_bars),
        "entry_execution": "close_of_labeled_bar",
        "overlap_policy": "single_position_skip_overlapping_signals",
        "same_bar_policy": "conservative_stop_first",
        "trade_return_std": float(returns.std(ddof=0)) if len(returns) > 1 else 0.0,
        "sharpe_like": float((returns.mean() / returns.std(ddof=0)) * np.sqrt(len(returns))) if len(returns) > 1 and returns.std(ddof=0) > 0 else 0.0,
    }
    return trades_df, stats


def build_summary(labels: pd.DataFrame, swings: pd.DataFrame, overrides: pd.DataFrame, backtest_stats: dict) -> dict:
    usable = labels.loc[~labels["warmup_mask"]].copy() if "warmup_mask" in labels.columns else labels.copy()
    usable["year"] = usable["timestamp"].dt.year
    usable["month"] = usable["timestamp"].dt.to_period("M").astype(str)
    usable["hour"] = usable["timestamp"].dt.hour
    usable["session"] = usable["hour"].map(session_bucket)

    summary = {
        "rows_total": int(len(labels)),
        "rows_usable": int(len(usable)),
        "long_zone_rate_pct": float(usable["label_long_ote"].mean() * 100),
        "short_zone_rate_pct": float(usable["label_short_ote"].mean() * 100),
        "long_entry_rate_pct": float(usable["label_long_entry"].mean() * 100),
        "short_entry_rate_pct": float(usable["label_short_entry"].mean() * 100),
        "avg_label_quality_long": float(usable.loc[usable["label_quality_long"] > 0, "label_quality_long"].mean() or 0.0),
        "avg_label_quality_short": float(usable.loc[usable["label_quality_short"] > 0, "label_quality_short"].mean() or 0.0),
        "overrides_count": int(len(overrides)),
        "swings_count": int(len(swings)),
    }

    if not swings.empty:
        summary["swing_quality_mean"] = float(swings.get("label_quality", pd.Series(dtype=float)).mean() or 0.0)
        summary["trend_scan_pass_rate_pct"] = float(swings.get("trend_scan_pass", pd.Series(dtype=bool)).mean() * 100 or 0.0)
        summary["long_tier_a_pct"] = float((swings.loc[swings["swing_type"] == "low", "label_tier"] == "A").mean() * 100 or 0.0)
        summary["short_tier_a_pct"] = float((swings.loc[swings["swing_type"] == "high", "label_tier"] == "A").mean() * 100 or 0.0)

    if backtest_stats:
        for key, value in backtest_stats.items():
            summary[f"backtest_{key}"] = value

    return summary


def make_dashboard(labels: pd.DataFrame, swings: pd.DataFrame, overrides: pd.DataFrame, trades: pd.DataFrame, backtest_stats: dict) -> go.Figure:
    if not HAS_PLOTLY:
        raise RuntimeError("Plotly is not installed.")
    usable = labels.loc[~labels["warmup_mask"]].copy() if "warmup_mask" in labels.columns else labels.copy()
    usable["year"] = usable["timestamp"].dt.year
    usable["month"] = usable["timestamp"].dt.to_period("M").astype(str)
    usable["hour"] = usable["timestamp"].dt.hour
    usable["session"] = usable["hour"].map(session_bucket)

    annual = usable.groupby("year")[["label_long_entry", "label_short_entry"]].sum().reset_index()
    hourly = usable.groupby("hour")[["label_long_entry", "label_short_entry"]].mean().reset_index()
    monthly = usable.groupby("month")[["label_long_entry", "label_short_entry"]].sum().tail(36).reset_index()
    spacing_long = bars_between_events(usable["label_long_entry"])
    spacing_short = bars_between_events(usable["label_short_entry"])

    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=[
            "Annual Entry Counts",
            "Hourly Entry Rate",
            "Recent Monthly Entry Counts",
            "Entry Spacing Distribution",
            "Swing Label Quality",
            "Override Activity",
            "Backtest Equity Curve",
            "Backtest Trade R Distribution",
        ],
    )

    fig.add_trace(go.Bar(x=annual["year"], y=annual["label_long_entry"], name="Long Entries", marker_color="#198754"), row=1, col=1)
    fig.add_trace(go.Bar(x=annual["year"], y=annual["label_short_entry"], name="Short Entries", marker_color="#dc3545"), row=1, col=1)

    fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["label_long_entry"] * 100, mode="lines+markers", name="Long %", line=dict(color="#198754")), row=1, col=2)
    fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["label_short_entry"] * 100, mode="lines+markers", name="Short %", line=dict(color="#dc3545")), row=1, col=2)

    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["label_long_entry"], name="Long Monthly", marker_color="#20c997"), row=2, col=1)
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["label_short_entry"], name="Short Monthly", marker_color="#fd7e14"), row=2, col=1)

    if not spacing_long.empty:
        fig.add_trace(go.Histogram(x=spacing_long, name="Bars between long entries", marker_color="#0d6efd", opacity=0.7), row=2, col=2)
    if not spacing_short.empty:
        fig.add_trace(go.Histogram(x=spacing_short, name="Bars between short entries", marker_color="#6f42c1", opacity=0.7), row=2, col=2)

    if not swings.empty:
        fig.add_trace(
            go.Histogram(
                x=swings["label_quality"],
                nbinsx=30,
                name="Swing quality",
                marker_color="#495057",
            ),
            row=3,
            col=1,
        )
    else:
        fig.add_trace(go.Bar(x=["no swings"], y=[0], name="Swing quality", marker_color="#adb5bd"), row=3, col=1)

    if not overrides.empty:
        ov = overrides.groupby(["target_label", "value"]).size().reset_index(name="count")
        ov["name"] = ov["target_label"] + "=" + ov["value"].astype(str)
        fig.add_trace(go.Bar(x=ov["name"], y=ov["count"], name="Overrides", marker_color="#0dcaf0"), row=3, col=2)
    else:
        fig.add_trace(go.Bar(x=["none"], y=[0], name="Overrides", marker_color="#adb5bd"), row=3, col=2)

    if not trades.empty:
        fig.add_trace(
            go.Scatter(
                x=trades["exit_time"],
                y=trades["equity_after"],
                mode="lines+markers",
                name="Equity",
                line=dict(color="#212529", width=2),
                marker=dict(size=5),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=trades["pnl_r"],
                nbinsx=30,
                name="Trade R",
                marker_color="#6610f2",
            ),
            row=4,
            col=2,
        )
    else:
        fig.add_trace(go.Bar(x=["no trades"], y=[0], name="Equity", marker_color="#adb5bd"), row=4, col=1)
        fig.add_trace(go.Bar(x=["no trades"], y=[0], name="Trade R", marker_color="#adb5bd"), row=4, col=2)

    fig.update_layout(
        template="plotly_white",
        height=1550,
        title=(
            "OTE Label Deep Analysis"
            if not backtest_stats
            else (
                "OTE Label Deep Analysis"
                f"<br><sup>Backtest: {backtest_stats['total_trades']} trades | "
                f"WR {backtest_stats['win_rate_pct']:.1f}% | "
                f"Return {backtest_stats['total_return_pct']:.1f}% | "
                f"Max DD {backtest_stats['max_drawdown_pct']:.1f}%</sup>"
            )
        ),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_csv(args.labels, parse_dates=["timestamp"])
    swings = load_csv(args.swings, parse_dates=["swing_time", "confirm_time", "entry_time"])
    overrides = load_csv(args.overrides, parse_dates=["timestamp"])

    if labels.empty:
        raise FileNotFoundError(f"No labels found at {args.labels}")

    reviewed = apply_overrides(labels, overrides)
    backtest_config = BacktestConfig()
    backtest_trades, backtest_stats = run_label_backtest(reviewed, backtest_config)
    summary = build_summary(reviewed, swings, overrides, backtest_stats)

    annual_counts = (
        reviewed.loc[~reviewed["warmup_mask"]]
        .assign(year=lambda d: d["timestamp"].dt.year)
        .groupby("year")[["label_long_entry", "label_short_entry", "label_long_ote", "label_short_ote"]]
        .sum()
        .reset_index()
    )
    hourly_rates = (
        reviewed.loc[~reviewed["warmup_mask"]]
        .assign(hour=lambda d: d["timestamp"].dt.hour)
        .groupby("hour")[["label_long_entry", "label_short_entry", "label_long_ote", "label_short_ote"]]
        .mean()
        .reset_index()
    )
    override_summary = (
        overrides.groupby(["target_label", "value"]).size().reset_index(name="count")
        if not overrides.empty
        else pd.DataFrame(columns=["target_label", "value", "count"])
    )
    backtest_stats_df = pd.DataFrame([backtest_stats]) if backtest_stats else pd.DataFrame()

    annual_counts.to_csv(args.output_dir / "annual_counts.csv", index=False)
    hourly_rates.to_csv(args.output_dir / "hourly_rates.csv", index=False)
    override_summary.to_csv(args.output_dir / "override_summary.csv", index=False)
    backtest_trades.to_csv(args.output_dir / "backtest_trades.csv", index=False)
    backtest_stats_df.to_csv(args.output_dir / "backtest_stats.csv", index=False)

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "OTE Label Deep Analysis",
        "=" * 30,
        f"Rows total: {summary['rows_total']:,}",
        f"Rows usable: {summary['rows_usable']:,}",
        f"Long zone rate: {summary['long_zone_rate_pct']:.3f}%",
        f"Short zone rate: {summary['short_zone_rate_pct']:.3f}%",
        f"Long entry rate: {summary['long_entry_rate_pct']:.3f}%",
        f"Short entry rate: {summary['short_entry_rate_pct']:.3f}%",
        f"Average long label quality: {summary['avg_label_quality_long']:.3f}",
        f"Average short label quality: {summary['avg_label_quality_short']:.3f}",
        f"Override count: {summary['overrides_count']:,}",
        f"Swing count: {summary['swings_count']:,}",
    ]
    if "trend_scan_pass_rate_pct" in summary:
        lines.append(f"Trend-scan pass rate: {summary['trend_scan_pass_rate_pct']:.2f}%")
    if "long_tier_a_pct" in summary:
        lines.append(f"Tier A swing share long/short: {summary['long_tier_a_pct']:.2f}% / {summary['short_tier_a_pct']:.2f}%")
    if backtest_stats:
        lines.extend(
            [
                "",
                "Backtest",
                "-" * 30,
                f"Initial capital: ${backtest_stats['initial_capital']:,.2f}",
                f"Final capital: ${backtest_stats['final_capital']:,.2f}",
                f"Net PnL: ${backtest_stats['net_pnl']:,.2f}",
                f"Total return: {backtest_stats['total_return_pct']:.2f}%",
                f"Total trades: {backtest_stats['total_trades']:,}",
                f"Long / short trades: {backtest_stats['long_trades']:,} / {backtest_stats['short_trades']:,}",
                f"Win rate: {backtest_stats['win_rate_pct']:.2f}%",
                f"Profit factor: {backtest_stats['profit_factor']:.2f}",
                f"Average trade: {backtest_stats['avg_trade_r']:.3f}R",
                f"Max drawdown: {backtest_stats['max_drawdown_pct']:.2f}%",
                f"Average bars held: {backtest_stats['avg_bars_held']:.2f}",
                f"Exit mix TP / SL / timeout: {backtest_stats['take_profit_exits']:,} / {backtest_stats['stop_loss_exits']:,} / {backtest_stats['timeout_exits']:,}",
                f"Same-bar TP+SL conflicts resolved stop-first: {backtest_stats['same_bar_conflicts']:,}",
            ]
        )
    (args.output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    html_path = args.output_dir / "ote_label_analysis.html"
    if HAS_PLOTLY:
        dashboard = make_dashboard(reviewed, swings, overrides, backtest_trades, backtest_stats)
        dashboard.write_html(str(html_path))

    print(f"Saved analysis to: {args.output_dir}")
    print(f"  - {args.output_dir / 'summary.txt'}")
    print(f"  - {args.output_dir / 'summary.json'}")
    print(f"  - {args.output_dir / 'annual_counts.csv'}")
    print(f"  - {args.output_dir / 'hourly_rates.csv'}")
    print(f"  - {args.output_dir / 'override_summary.csv'}")
    print(f"  - {args.output_dir / 'backtest_trades.csv'}")
    print(f"  - {args.output_dir / 'backtest_stats.csv'}")
    if HAS_PLOTLY:
        print(f"  - {html_path}")
    else:
        print("  - HTML dashboard skipped because plotly is not installed.")


if __name__ == "__main__":
    main()
