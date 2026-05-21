"""
Deep analysis for reversal, continuation, and breakout labels plus human-review overrides.

Outputs:
- summary.txt
- summary.json
- family_summary.csv
- annual_counts.csv
- hourly_rates.csv
- override_summary.csv
- backtest_trades.csv
- backtest_stats.csv
- label_analysis.html
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

LABELS_PATH_CANDIDATES = [
    PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_all_labels.csv",
    PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_ote_labels_full.csv",
    PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "labeled_output.csv",
]
EVENTS_PATH_CANDIDATES = [
    PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_all_label_events.csv",
    PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_ote_swings_full.csv",
]
OVERRIDES_PATH_CANDIDATES = [
    PROJECT_ROOT / "data" / "labeling" / "review" / "label_overrides.csv",
    PROJECT_ROOT / "data" / "labeling" / "review" / "reversal_label_overrides.csv",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "labeling" / "analysis"

LEGACY_REVERSAL_COLUMN_MAP = {
    "label_long_ote": "label_long_reversal",
    "label_short_ote": "label_short_reversal",
    "label_long_entry": "label_long_reversal_entry",
    "label_short_entry": "label_short_reversal_entry",
    "label_quality_long": "label_quality_long_reversal",
    "label_quality_short": "label_quality_short_reversal",
    "entry_quality_long": "entry_quality_long_reversal",
    "entry_quality_short": "entry_quality_short_reversal",
}


@dataclass(frozen=True)
class LabelFamilyConfig:
    key: str
    title: str
    short_title: str
    zone_long_col: str
    zone_short_col: str
    entry_long_col: str
    entry_short_col: str
    zone_quality_long_col: str
    zone_quality_short_col: str
    entry_quality_long_col: str
    entry_quality_short_col: str
    structural_atr_candidates: tuple[str, ...]
    long_color: str
    short_color: str

    @property
    def binary_columns(self) -> tuple[str, str, str, str]:
        return (
            self.zone_long_col,
            self.zone_short_col,
            self.entry_long_col,
            self.entry_short_col,
        )

    @property
    def quality_columns(self) -> tuple[str, str, str, str]:
        return (
            self.zone_quality_long_col,
            self.zone_quality_short_col,
            self.entry_quality_long_col,
            self.entry_quality_short_col,
        )

    @property
    def target_columns(self) -> tuple[str, str, str, str]:
        return self.binary_columns


FAMILY_CONFIGS: dict[str, LabelFamilyConfig] = {
    "reversal": LabelFamilyConfig(
        key="reversal",
        title="Reversal",
        short_title="Reversal",
        zone_long_col="label_long_reversal",
        zone_short_col="label_short_reversal",
        entry_long_col="label_long_reversal_entry",
        entry_short_col="label_short_reversal_entry",
        zone_quality_long_col="label_quality_long_reversal",
        zone_quality_short_col="label_quality_short_reversal",
        entry_quality_long_col="entry_quality_long_reversal",
        entry_quality_short_col="entry_quality_short_reversal",
        structural_atr_candidates=("reversal_structural_atr", "structural_atr"),
        long_color="#198754",
        short_color="#dc3545",
    ),
    "continuation_pullback": LabelFamilyConfig(
        key="continuation_pullback",
        title="Continuation Pullback",
        short_title="Continuation",
        zone_long_col="label_long_continuation_pullback",
        zone_short_col="label_short_continuation_pullback",
        entry_long_col="label_long_continuation_entry",
        entry_short_col="label_short_continuation_entry",
        zone_quality_long_col="label_quality_long_continuation",
        zone_quality_short_col="label_quality_short_continuation",
        entry_quality_long_col="entry_quality_long_continuation",
        entry_quality_short_col="entry_quality_short_continuation",
        structural_atr_candidates=("continuation_structural_atr", "structural_atr"),
        long_color="#0f766e",
        short_color="#ea580c",
    ),
    "breakout": LabelFamilyConfig(
        key="breakout",
        title="Breakout",
        short_title="Breakout",
        zone_long_col="label_long_breakout",
        zone_short_col="label_short_breakout",
        entry_long_col="label_long_breakout_entry",
        entry_short_col="label_short_breakout_entry",
        zone_quality_long_col="label_quality_long_breakout",
        zone_quality_short_col="label_quality_short_breakout",
        entry_quality_long_col="entry_quality_long_breakout",
        entry_quality_short_col="entry_quality_short_breakout",
        structural_atr_candidates=("breakout_structural_atr", "structural_atr"),
        long_color="#65a30d",
        short_color="#be123c",
    ),
}


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
    family: str
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


def resolve_default_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep analysis for multi-family reviewed labels.")
    parser.add_argument("--labels", type=Path, default=resolve_default_path(LABELS_PATH_CANDIDATES))
    parser.add_argument(
        "--events",
        "--swings",
        dest="events",
        type=Path,
        default=resolve_default_path(EVENTS_PATH_CANDIDATES),
    )
    parser.add_argument("--overrides", type=Path, default=resolve_default_path(OVERRIDES_PATH_CANDIDATES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--family", choices=["all"] + list(FAMILY_CONFIGS), default="all")
    return parser.parse_args()


def load_csv(path: Path, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    if "timestamp" in df.columns:
        df = normalize_label_columns(df)
    return df


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for legacy, current in LEGACY_REVERSAL_COLUMN_MAP.items():
        if legacy in normalized.columns and current not in normalized.columns:
            normalized = normalized.rename(columns={legacy: current})
    return normalized


def ensure_family_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for config in FAMILY_CONFIGS.values():
        for column in config.binary_columns:
            if column not in prepared.columns:
                prepared[column] = 0
        for column in config.quality_columns:
            if column not in prepared.columns:
                prepared[column] = 0.0
    if "warmup_mask" not in prepared.columns:
        prepared["warmup_mask"] = False
    return prepared


def detect_family_availability(columns: pd.Index) -> dict[str, bool]:
    availability: dict[str, bool] = {}
    column_set = set(columns)
    for key, config in FAMILY_CONFIGS.items():
        tracked = set(config.binary_columns) | set(config.quality_columns)
        availability[key] = any(column in column_set for column in tracked)
    return availability


def apply_overrides(labels: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    reviewed = labels.copy()
    if overrides.empty:
        return reviewed
    overrides = overrides.copy()
    overrides["target_label"] = overrides["target_label"].replace(LEGACY_REVERSAL_COLUMN_MAP)
    latest = overrides.sort_values(["timestamp"]).groupby(["timestamp", "target_label"], as_index=False).tail(1)
    ts_to_idx = pd.Series(reviewed.index.values, index=reviewed["timestamp"])
    for row in latest.itertuples(index=False):
        idx = ts_to_idx.get(row.timestamp)
        if idx is None or row.target_label not in reviewed.columns:
            continue
        reviewed.at[idx, row.target_label] = int(row.value)
    return reviewed


def resolve_structural_atr_column(df: pd.DataFrame, config: LabelFamilyConfig) -> str | None:
    for column in config.structural_atr_candidates:
        if column in df.columns:
            return column
    return None


def bars_between_events(mask: pd.Series) -> pd.Series:
    idx = np.flatnonzero(mask.fillna(0).astype(bool).to_numpy())
    if len(idx) < 2:
        return pd.Series(dtype=float)
    return pd.Series(np.diff(idx), dtype=float)


def session_bucket(hour: int) -> str:
    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 12:
        return "London"
    if 12 <= hour < 17:
        return "NY AM"
    return "NY PM/Off"


def positive_count(series: pd.Series) -> int:
    return int(series.fillna(0).astype(bool).sum())


def positive_rate_pct(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.fillna(0).astype(bool).mean() * 100.0)


def positive_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric[numeric > 0]
    if valid.empty:
        return 0.0
    return float(valid.mean())


def select_requested_families(
    family_arg: str,
    availability: dict[str, bool],
) -> tuple[list[LabelFamilyConfig], list[str]]:
    available_keys = [key for key, is_available in availability.items() if is_available]
    if family_arg == "all":
        return [FAMILY_CONFIGS[key] for key in available_keys], [key for key in FAMILY_CONFIGS if key not in available_keys]
    if availability.get(family_arg, False):
        return [FAMILY_CONFIGS[family_arg]], []
    return [], [family_arg]


def filter_events_for_family(events: pd.DataFrame, family_key: str) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if "label_family" not in events.columns:
        if family_key == "reversal":
            return events.copy()
        return pd.DataFrame(columns=events.columns)
    return events.loc[events["label_family"] == family_key].reset_index(drop=True)


def run_label_backtest(
    labels: pd.DataFrame,
    config: BacktestConfig,
    family_config: LabelFamilyConfig,
) -> tuple[pd.DataFrame, dict]:
    usable = labels.loc[~labels["warmup_mask"].fillna(False).astype(bool)].copy()
    if usable.empty:
        return pd.DataFrame(), {}

    atr_column = resolve_structural_atr_column(usable, family_config)
    if atr_column is None:
        return pd.DataFrame(), {}

    long_mask = usable[family_config.entry_long_col].fillna(0).astype(bool)
    short_mask = usable[family_config.entry_short_col].fillna(0).astype(bool)
    entry_mask = long_mask | short_mask
    entry_candidates = usable.loc[entry_mask].copy()
    entry_candidates["side"] = np.where(long_mask.loc[entry_candidates.index], "long", "short")
    entry_candidates = entry_candidates.loc[
        pd.to_numeric(entry_candidates[atr_column], errors="coerce").notna()
        & (pd.to_numeric(entry_candidates[atr_column], errors="coerce") > 0)
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
        atr = float(getattr(row, atr_column))
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
                family=family_config.key,
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
    returns_std = float(returns.std(ddof=0)) if len(returns) > 1 else 0.0

    stats = {
        "family": family_config.key,
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
        "trade_return_std": returns_std,
        "sharpe_like": float((returns.mean() / returns.std(ddof=0)) * np.sqrt(len(returns))) if len(returns) > 1 and returns.std(ddof=0) > 0 else 0.0,
    }
    return trades_df, stats


def build_family_summary(
    labels: pd.DataFrame,
    events: pd.DataFrame,
    overrides: pd.DataFrame,
    family_config: LabelFamilyConfig,
    backtest_stats: dict,
) -> dict:
    usable = labels.loc[~labels["warmup_mask"].fillna(False).astype(bool)].copy()
    usable["year"] = usable["timestamp"].dt.year
    usable["month"] = usable["timestamp"].dt.to_period("M").astype(str)
    usable["hour"] = usable["timestamp"].dt.hour
    usable["session"] = usable["hour"].map(session_bucket)

    family_targets = set(family_config.target_columns)
    family_overrides = overrides.loc[overrides["target_label"].isin(family_targets)].copy() if not overrides.empty else pd.DataFrame()
    family_events = filter_events_for_family(events, family_config.key)

    summary = {
        "family": family_config.key,
        "title": family_config.title,
        "rows_usable": int(len(usable)),
        "long_zone_count": positive_count(usable[family_config.zone_long_col]),
        "short_zone_count": positive_count(usable[family_config.zone_short_col]),
        "long_entry_count": positive_count(usable[family_config.entry_long_col]),
        "short_entry_count": positive_count(usable[family_config.entry_short_col]),
        "long_zone_rate_pct": positive_rate_pct(usable[family_config.zone_long_col]),
        "short_zone_rate_pct": positive_rate_pct(usable[family_config.zone_short_col]),
        "long_entry_rate_pct": positive_rate_pct(usable[family_config.entry_long_col]),
        "short_entry_rate_pct": positive_rate_pct(usable[family_config.entry_short_col]),
        "avg_zone_quality_long": positive_mean(usable[family_config.zone_quality_long_col]),
        "avg_zone_quality_short": positive_mean(usable[family_config.zone_quality_short_col]),
        "avg_entry_quality_long": positive_mean(usable[family_config.entry_quality_long_col]),
        "avg_entry_quality_short": positive_mean(usable[family_config.entry_quality_short_col]),
        "overrides_count": int(len(family_overrides)),
        "events_count": int(len(family_events)),
    }

    if not family_events.empty:
        summary["event_quality_mean"] = positive_mean(family_events.get("label_quality", pd.Series(dtype=float)))
        summary["trend_scan_pass_rate_pct"] = float(family_events.get("trend_scan_pass", pd.Series(dtype=bool)).fillna(False).mean() * 100.0)
        summary["long_tier_a_pct"] = float((family_events.loc[family_events["swing_type"] == "low", "label_tier"] == "A").mean() * 100.0)
        summary["short_tier_a_pct"] = float((family_events.loc[family_events["swing_type"] == "high", "label_tier"] == "A").mean() * 100.0)

    atr_column = resolve_structural_atr_column(labels, family_config)
    if atr_column is not None:
        atr_values = pd.to_numeric(usable[atr_column], errors="coerce").dropna()
        summary["avg_structural_atr"] = float(atr_values.mean()) if not atr_values.empty else 0.0

    for key, value in backtest_stats.items():
        if key == "family":
            continue
        summary[f"backtest_{key}"] = value

    return summary


def build_summary(
    labels: pd.DataFrame,
    events: pd.DataFrame,
    overrides: pd.DataFrame,
    family_summaries: dict[str, dict],
    skipped_families: list[str],
    labels_path: Path,
    events_path: Path,
    overrides_path: Path,
) -> dict:
    usable = labels.loc[~labels["warmup_mask"].fillna(False).astype(bool)].copy()
    return {
        "labels_path": str(labels_path),
        "events_path": str(events_path),
        "overrides_path": str(overrides_path),
        "rows_total": int(len(labels)),
        "rows_usable": int(len(usable)),
        "override_rows_total": int(len(overrides)),
        "event_rows_total": int(len(events)),
        "families_analyzed": list(family_summaries.keys()),
        "families_skipped": skipped_families,
        "families": family_summaries,
    }


def build_family_summary_frame(family_summaries: dict[str, dict]) -> pd.DataFrame:
    if not family_summaries:
        return pd.DataFrame()
    return pd.DataFrame(list(family_summaries.values()))


def build_annual_counts(labels: pd.DataFrame, family_configs: list[LabelFamilyConfig]) -> pd.DataFrame:
    usable = labels.loc[~labels["warmup_mask"].fillna(False).astype(bool)].copy()
    usable["year"] = usable["timestamp"].dt.year
    rows = []
    for config in family_configs:
        grouped = usable.groupby("year")[list(config.binary_columns)].sum().reset_index()
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "family": config.key,
                    "title": config.title,
                    "year": int(row.year),
                    "long_zone_count": int(getattr(row, config.zone_long_col)),
                    "short_zone_count": int(getattr(row, config.zone_short_col)),
                    "long_entry_count": int(getattr(row, config.entry_long_col)),
                    "short_entry_count": int(getattr(row, config.entry_short_col)),
                }
            )
    return pd.DataFrame(rows)


def build_hourly_rates(labels: pd.DataFrame, family_configs: list[LabelFamilyConfig]) -> pd.DataFrame:
    usable = labels.loc[~labels["warmup_mask"].fillna(False).astype(bool)].copy()
    usable["hour"] = usable["timestamp"].dt.hour
    rows = []
    for config in family_configs:
        grouped = usable.groupby("hour")[list(config.binary_columns)].mean().reset_index()
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "family": config.key,
                    "title": config.title,
                    "hour": int(row.hour),
                    "long_zone_rate_pct": float(getattr(row, config.zone_long_col) * 100.0),
                    "short_zone_rate_pct": float(getattr(row, config.zone_short_col) * 100.0),
                    "long_entry_rate_pct": float(getattr(row, config.entry_long_col) * 100.0),
                    "short_entry_rate_pct": float(getattr(row, config.entry_short_col) * 100.0),
                }
            )
    return pd.DataFrame(rows)


def build_override_summary(overrides: pd.DataFrame, family_configs: list[LabelFamilyConfig]) -> pd.DataFrame:
    if overrides.empty:
        return pd.DataFrame(columns=["family", "title", "target_label", "value", "count"])
    target_to_family = {}
    for config in family_configs:
        for target in config.target_columns:
            target_to_family[target] = config
    summary = overrides.copy()
    summary["family"] = summary["target_label"].map(lambda value: target_to_family.get(value).key if value in target_to_family else "other")
    summary["title"] = summary["target_label"].map(lambda value: target_to_family.get(value).title if value in target_to_family else "Other")
    return (
        summary.groupby(["family", "title", "target_label", "value"])
        .size()
        .reset_index(name="count")
        .sort_values(["family", "target_label", "value"])
        .reset_index(drop=True)
    )


def build_spacing_frame(labels: pd.DataFrame, family_configs: list[LabelFamilyConfig]) -> pd.DataFrame:
    usable = labels.loc[~labels["warmup_mask"].fillna(False).astype(bool)].copy()
    rows = []
    for config in family_configs:
        for side, column in (("Long", config.entry_long_col), ("Short", config.entry_short_col)):
            spacing = bars_between_events(usable[column])
            for value in spacing.tolist():
                rows.append({"family": config.key, "title": config.title, "side": side, "bars_between_entries": float(value)})
    return pd.DataFrame(rows)


def build_backtest_stats_frame(backtest_stats_by_family: dict[str, dict]) -> pd.DataFrame:
    populated = [stats for stats in backtest_stats_by_family.values() if stats]
    if not populated:
        return pd.DataFrame()
    return pd.DataFrame(populated)


def make_dashboard(
    labels: pd.DataFrame,
    events: pd.DataFrame,
    override_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    backtest_trades: pd.DataFrame,
    backtest_stats: pd.DataFrame,
    family_configs: list[LabelFamilyConfig],
    title: str,
) -> go.Figure:
    if not HAS_PLOTLY:
        raise RuntimeError("Plotly is not installed.")

    annual_counts = build_annual_counts(labels, family_configs)
    hourly_rates = build_hourly_rates(labels, family_configs)
    spacing_frame = build_spacing_frame(labels, family_configs)

    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=[
            "Annual Entry Counts",
            "Hourly Entry Rate",
            "Family-Level Label Rates",
            "Entry Spacing Distribution",
            "Event Quality by Family",
            "Override Activity",
            "Backtest Equity Curves",
            "Backtest Trade R Distribution",
        ],
    )

    for config in family_configs:
        family_annual = annual_counts.loc[annual_counts["family"] == config.key]
        if not family_annual.empty:
            fig.add_trace(
                go.Bar(
                    x=family_annual["year"],
                    y=family_annual["long_entry_count"],
                    name=f"{config.short_title} Long Entries",
                    marker_color=config.long_color,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=family_annual["year"],
                    y=family_annual["short_entry_count"],
                    name=f"{config.short_title} Short Entries",
                    marker_color=config.short_color,
                ),
                row=1,
                col=1,
            )

        family_hourly = hourly_rates.loc[hourly_rates["family"] == config.key]
        if not family_hourly.empty:
            fig.add_trace(
                go.Scatter(
                    x=family_hourly["hour"],
                    y=family_hourly["long_entry_rate_pct"],
                    mode="lines+markers",
                    name=f"{config.short_title} Long %",
                    line=dict(color=config.long_color),
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=family_hourly["hour"],
                    y=family_hourly["short_entry_rate_pct"],
                    mode="lines+markers",
                    name=f"{config.short_title} Short %",
                    line=dict(color=config.short_color),
                ),
                row=1,
                col=2,
            )

    if family_summary.empty:
        fig.add_trace(go.Bar(x=["none"], y=[0], name="No family summary", marker_color="#adb5bd"), row=2, col=1)
    else:
        fig.add_trace(go.Bar(x=family_summary["title"], y=family_summary["long_zone_rate_pct"], name="Long Zone %", marker_color="#2dd4bf"), row=2, col=1)
        fig.add_trace(go.Bar(x=family_summary["title"], y=family_summary["short_zone_rate_pct"], name="Short Zone %", marker_color="#fb923c"), row=2, col=1)
        fig.add_trace(go.Bar(x=family_summary["title"], y=family_summary["long_entry_rate_pct"], name="Long Entry %", marker_color="#22c55e"), row=2, col=1)
        fig.add_trace(go.Bar(x=family_summary["title"], y=family_summary["short_entry_rate_pct"], name="Short Entry %", marker_color="#ef4444"), row=2, col=1)

    if spacing_frame.empty:
        fig.add_trace(go.Bar(x=["none"], y=[0], name="No spacing data", marker_color="#adb5bd"), row=2, col=2)
    else:
        for config in family_configs:
            family_spacing = spacing_frame.loc[spacing_frame["family"] == config.key]
            for side, color in (("Long", config.long_color), ("Short", config.short_color)):
                side_spacing = family_spacing.loc[family_spacing["side"] == side]
                if side_spacing.empty:
                    continue
                fig.add_trace(
                    go.Box(
                        y=side_spacing["bars_between_entries"],
                        name=f"{config.short_title} {side}",
                        marker_color=color,
                        boxmean=True,
                    ),
                    row=2,
                    col=2,
                )

    if events.empty:
        fig.add_trace(go.Bar(x=["none"], y=[0], name="No events", marker_color="#adb5bd"), row=3, col=1)
    else:
        for config in family_configs:
            family_events = filter_events_for_family(events, config.key)
            if family_events.empty or "label_quality" not in family_events.columns:
                continue
            fig.add_trace(
                go.Box(
                    y=pd.to_numeric(family_events["label_quality"], errors="coerce").dropna(),
                    name=f"{config.short_title} quality",
                    marker_color=config.long_color,
                    boxmean=True,
                ),
                row=3,
                col=1,
            )

    if override_summary.empty:
        fig.add_trace(go.Bar(x=["none"], y=[0], name="No overrides", marker_color="#adb5bd"), row=3, col=2)
    else:
        for value, color, name in ((1, "#0d6efd", "Override on"), (0, "#6c757d", "Override off")):
            subset = override_summary.loc[override_summary["value"] == value]
            if subset.empty:
                continue
            counts = subset.groupby("title", as_index=False)["count"].sum()
            fig.add_trace(
                go.Bar(
                    x=counts["title"],
                    y=counts["count"],
                    name=name,
                    marker_color=color,
                ),
                row=3,
                col=2,
            )

    if backtest_trades.empty:
        fig.add_trace(go.Bar(x=["none"], y=[0], name="No backtest trades", marker_color="#adb5bd"), row=4, col=1)
        fig.add_trace(go.Bar(x=["none"], y=[0], name="No trade R", marker_color="#adb5bd"), row=4, col=2)
    else:
        for config in family_configs:
            family_trades = backtest_trades.loc[backtest_trades["family"] == config.key]
            if family_trades.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=family_trades["exit_time"],
                    y=family_trades["equity_after"],
                    mode="lines+markers",
                    name=f"{config.short_title} equity",
                    line=dict(color=config.long_color, width=2),
                    marker=dict(size=5),
                ),
                row=4,
                col=1,
            )
            fig.add_trace(
                go.Histogram(
                    x=family_trades["pnl_r"],
                    name=f"{config.short_title} trade R",
                    marker_color=config.short_color,
                    opacity=0.65,
                ),
                row=4,
                col=2,
            )

    subtitle = ""
    if not backtest_stats.empty:
        summary_bits = []
        for row in backtest_stats.itertuples(index=False):
            summary_bits.append(
                f"{row.family}: {int(row.total_trades)} trades, WR {row.win_rate_pct:.1f}%, Return {row.total_return_pct:.1f}%"
            )
        subtitle = "<br><sup>" + " | ".join(summary_bits) + "</sup>"

    fig.update_layout(
        template="plotly_white",
        height=1_650,
        title=title + subtitle,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def summary_lines_from_payload(summary: dict) -> list[str]:
    lines = [
        "Multi-Family Label Deep Analysis",
        "=" * 32,
        f"Labels file: {summary['labels_path']}",
        f"Events file: {summary['events_path']}",
        f"Overrides file: {summary['overrides_path']}",
        f"Rows total: {summary['rows_total']:,}",
        f"Rows usable: {summary['rows_usable']:,}",
        f"Override rows total: {summary['override_rows_total']:,}",
        f"Event rows total: {summary['event_rows_total']:,}",
        f"Families analyzed: {', '.join(summary['families_analyzed']) if summary['families_analyzed'] else 'none'}",
    ]
    if summary["families_skipped"]:
        lines.append(f"Families skipped: {', '.join(summary['families_skipped'])}")

    for family_key, payload in summary["families"].items():
        lines.extend(
            [
                "",
                payload["title"],
                "-" * len(payload["title"]),
                f"Long zone rate: {payload['long_zone_rate_pct']:.3f}%",
                f"Short zone rate: {payload['short_zone_rate_pct']:.3f}%",
                f"Long entry rate: {payload['long_entry_rate_pct']:.3f}%",
                f"Short entry rate: {payload['short_entry_rate_pct']:.3f}%",
                f"Average zone quality L/S: {payload['avg_zone_quality_long']:.3f} / {payload['avg_zone_quality_short']:.3f}",
                f"Average entry quality L/S: {payload['avg_entry_quality_long']:.3f} / {payload['avg_entry_quality_short']:.3f}",
                f"Overrides: {payload['overrides_count']:,}",
                f"Events: {payload['events_count']:,}",
            ]
        )
        if "event_quality_mean" in payload:
            lines.append(f"Event quality mean: {payload['event_quality_mean']:.3f}")
        if "trend_scan_pass_rate_pct" in payload:
            lines.append(f"Trend-scan pass rate: {payload['trend_scan_pass_rate_pct']:.2f}%")
        if "avg_structural_atr" in payload:
            lines.append(f"Average structural ATR: {payload['avg_structural_atr']:.6f}")
        if "backtest_total_trades" in payload:
            lines.extend(
                [
                    "Backtest",
                    "~" * 8,
                    f"Total trades: {int(payload['backtest_total_trades']):,}",
                    f"Win rate: {payload['backtest_win_rate_pct']:.2f}%",
                    f"Profit factor: {payload['backtest_profit_factor']:.2f}",
                    f"Total return: {payload['backtest_total_return_pct']:.2f}%",
                    f"Max drawdown: {payload['backtest_max_drawdown_pct']:.2f}%",
                    f"Average trade: {payload['backtest_avg_trade_r']:.3f}R",
                ]
            )
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_csv(args.labels, date_columns=("timestamp",))
    if labels.empty:
        raise FileNotFoundError(f"No labels found at {args.labels}")
    labels = labels.sort_values("timestamp").reset_index(drop=True)
    family_availability = detect_family_availability(labels.columns)
    labels = ensure_family_columns(labels)

    events = load_csv(
        args.events,
        date_columns=("timestamp", "swing_time", "confirm_time", "entry_time"),
    )
    overrides = load_csv(args.overrides, date_columns=("timestamp",))
    if not overrides.empty:
        overrides = overrides.copy()
        overrides["target_label"] = overrides["target_label"].replace(LEGACY_REVERSAL_COLUMN_MAP)

    requested_families, skipped_families = select_requested_families(args.family, family_availability)
    if not requested_families:
        requested = args.family if args.family != "all" else "any supported family"
        raise ValueError(f"No available label columns found for requested family selection: {requested}")

    reviewed = apply_overrides(labels, overrides)
    backtest_config = BacktestConfig()

    family_summaries: dict[str, dict] = {}
    backtest_trades_frames: list[pd.DataFrame] = []
    backtest_stats_by_family: dict[str, dict] = {}

    for family_config in requested_families:
        trades_df, stats = run_label_backtest(reviewed, backtest_config, family_config)
        if not trades_df.empty:
            backtest_trades_frames.append(trades_df)
        backtest_stats_by_family[family_config.key] = stats
        family_summaries[family_config.key] = build_family_summary(
            reviewed,
            events,
            overrides,
            family_config,
            stats,
        )

    backtest_trades = (
        pd.concat(backtest_trades_frames, ignore_index=True, sort=False)
        if backtest_trades_frames
        else pd.DataFrame()
    )
    backtest_stats = build_backtest_stats_frame(backtest_stats_by_family)
    family_summary = build_family_summary_frame(family_summaries)
    annual_counts = build_annual_counts(reviewed, requested_families)
    hourly_rates = build_hourly_rates(reviewed, requested_families)
    override_summary = build_override_summary(overrides, requested_families)

    summary = build_summary(
        reviewed,
        events,
        overrides,
        family_summaries,
        skipped_families,
        args.labels,
        args.events,
        args.overrides,
    )

    annual_counts.to_csv(args.output_dir / "annual_counts.csv", index=False)
    hourly_rates.to_csv(args.output_dir / "hourly_rates.csv", index=False)
    family_summary.to_csv(args.output_dir / "family_summary.csv", index=False)
    override_summary.to_csv(args.output_dir / "override_summary.csv", index=False)
    backtest_trades.to_csv(args.output_dir / "backtest_trades.csv", index=False)
    backtest_stats.to_csv(args.output_dir / "backtest_stats.csv", index=False)

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    (args.output_dir / "summary.txt").write_text("\n".join(summary_lines_from_payload(summary)), encoding="utf-8")

    html_path = args.output_dir / ("label_analysis.html" if args.family == "all" else f"{args.family}_label_analysis.html")
    if HAS_PLOTLY:
        dashboard = make_dashboard(
            reviewed,
            events,
            override_summary,
            family_summary,
            backtest_trades,
            backtest_stats,
            requested_families,
            title=(
                "Multi-Family Label Deep Analysis"
                if args.family == "all"
                else f"{requested_families[0].title} Label Deep Analysis"
            ),
        )
        dashboard.write_html(str(html_path))

    print(f"Saved analysis to: {args.output_dir}")
    print(f"  - {args.output_dir / 'summary.txt'}")
    print(f"  - {args.output_dir / 'summary.json'}")
    print(f"  - {args.output_dir / 'family_summary.csv'}")
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
