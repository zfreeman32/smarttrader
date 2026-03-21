"""
Auto-label optimal EURUSD swing-entry bars on 5-minute OHLCV data.

This script wraps the causal OTE labeling engine into a reusable workflow:
1. Load EURUSD OHLCV data
2. Build 30-minute and 1-hour structural context
3. Detect swing-entry zones with causal confirmation
4. Save both bar-level labels and swing-level metadata
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from .ote_labeling_engine import OTEParams, build_ote_labels, plot_swings
except ImportError:
    from ote_labeling_engine import OTEParams, build_ote_labels, plot_swings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "currency_data" / "EURUSD_5min.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_ote_labels.csv"
DEFAULT_SWINGS_PATH = PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_ote_swings.csv"


def detect_anomalies(df: pd.DataFrame, atr_period: int = 20, range_atr_mult: float = 10.0) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    invalid_ohlc = (df["high"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])
    huge_range = (atr > 0) & ((df["high"] - df["low"]) > range_atr_mult * atr)
    zero_volume = df["volume"] <= 0
    return invalid_ohlc | huge_range | zero_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label optimal EURUSD 5-minute swing-entry zones."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Source OHLCV CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Bar-level labeled CSV.")
    parser.add_argument("--swings-output", type=Path, default=DEFAULT_SWINGS_PATH, help="Swing metadata CSV.")
    parser.add_argument("--plot", action="store_true", help="Save a diagnostic chart for a representative week.")
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeling" / "plots" / "eurusd_5min_ote_chart.png",
        help="Plot destination when --plot is enabled.",
    )
    return parser.parse_args()


def load_fx_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns_lower = {col.strip().lower(): col for col in df.columns}

    if "date" in columns_lower and "time" in columns_lower:
        df["timestamp"] = pd.to_datetime(
            df[columns_lower["date"]].astype(str) + " " + df[columns_lower["time"]].astype(str),
            format="%Y%m%d %H:%M:%S",
            errors="coerce",
        )
        df = df.drop(columns=[columns_lower["date"], columns_lower["time"]])
    elif "timestamp" in columns_lower:
        df["timestamp"] = pd.to_datetime(df[columns_lower["timestamp"]], errors="coerce")
    elif "datetime" in columns_lower:
        df["timestamp"] = pd.to_datetime(df[columns_lower["datetime"]], errors="coerce")
    else:
        raise ValueError(f"Could not find a timestamp column in {path}. Columns: {df.columns.tolist()}")

    rename_map = {}
    for col in df.columns:
        normalized = col.strip().lower()
        if normalized in {"open", "o"}:
            rename_map[col] = "open"
        elif normalized in {"high", "h"}:
            rename_map[col] = "high"
        elif normalized in {"low", "l"}:
            rename_map[col] = "low"
        elif normalized in {"close", "c", "adj close"}:
            rename_map[col] = "close"
        elif normalized in {"volume", "vol", "v", "tick_volume"}:
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if "volume" not in df.columns:
        df["volume"] = 0

    df = df[["open", "high", "low", "close", "volume"]].copy()
    anomaly_mask = detect_anomalies(df)
    anomaly_count = int(anomaly_mask.sum())
    df["is_anomaly"] = anomaly_mask.astype(bool)
    if anomaly_mask.any():
        df = df.loc[~anomaly_mask].copy()
        df["is_anomaly"] = False
    df.attrs["anomaly_count"] = anomaly_count

    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def build_default_params() -> OTEParams:
    return OTEParams(
        atr_period=14,
        atr_smoothing="ema",
        structural_atr_tf="1hr",
        cusum_atr_mult=0.5,
        cusum_use_dynamic=True,
        confirm_atr_mult=0.8,
        confirm_use_close=True,
        min_swing_atr=1.35,
        min_swing_distance_atr=0.75,
        min_bars_between_swings=18,
        tb_enable=True,
        tb_profit_atr=1.0,
        tb_stop_atr=2.0,
        tb_max_bars=120,
        zone_pre_bars=2,
        zone_post_bars=0,
        entry_lookback_bars=3,
        entry_max_delay_after_swing=4,
        exclusion_pre_bars=10,
        htf_confirm_atr_mult=0.6,
        htf_min_swing_atr=0.8,
        htf_min_bars_between=4,
        htf_confluence_window_minutes=120,
        compute_uniqueness=True,
        warmup_bars=50,
    )


def summarize_labels(df_labeled: pd.DataFrame) -> str:
    usable = ~df_labeled["warmup_mask"]
    quality_long = df_labeled.loc[usable & (df_labeled["label_quality_long"] > 0), "label_quality_long"]
    quality_short = df_labeled.loc[usable & (df_labeled["label_quality_short"] > 0), "label_quality_short"]
    return (
        "\n"
        f"Bars after warmup: {int(usable.sum()):,}\n"
        f"Long positives:    {int(df_labeled.loc[usable, 'label_long_ote'].sum()):,}\n"
        f"Short positives:   {int(df_labeled.loc[usable, 'label_short_ote'].sum()):,}\n"
        f"Long entries:      {int(df_labeled.loc[usable, 'label_long_entry'].sum()):,}\n"
        f"Short entries:     {int(df_labeled.loc[usable, 'label_short_entry'].sum()):,}\n"
        f"Quality L mean:    {float(quality_long.mean()) if not quality_long.empty else 0.0:.3f}\n"
        f"Quality S mean:    {float(quality_short.mean()) if not quality_short.empty else 0.0:.3f}\n"
        f"Safe long negs:    {int(df_labeled.loc[usable, 'neg_ok_long'].sum()):,}\n"
        f"Safe short negs:   {int(df_labeled.loc[usable, 'neg_ok_short'].sum()):,}\n"
        f"30m confluence L:  {int(df_labeled.loc[usable, 'htf_confluence_long'].sum()):,}\n"
        f"30m confluence S:  {int(df_labeled.loc[usable, 'htf_confluence_short'].sum()):,}\n"
    )


def swings_to_frame(swings: list) -> pd.DataFrame:
    rows = []
    for swing in swings:
        rows.append(
            {
                "swing_type": swing.swing_type,
                "source_tf": swing.source_tf,
                "swing_time": swing.swing_time,
                "swing_index": swing.swing_index,
                "swing_price": swing.swing_price,
                "confirm_time": swing.confirm_time,
                "confirm_index": swing.confirm_index,
                "confirm_lag_bars": swing.confirm_lag,
                "atr_at_swing": swing.atr_at_swing,
                "swing_size_atr": swing.swing_size_atr,
                "tb_outcome": swing.tb_outcome,
                "tb_return": swing.tb_return,
                "tb_bars_held": swing.tb_bars_held,
                "entry_time": swing.entry_time,
                "entry_index": swing.entry_index,
                "entry_price": swing.entry_price,
                "entry_score": swing.entry_score,
                "entry_rr": swing.entry_rr,
                "entry_followthrough_atr": swing.entry_followthrough_atr,
                "trend_scan_t": swing.trend_scan_t,
                "trend_scan_window": swing.trend_scan_window,
                "trend_scan_pass": swing.trend_scan_pass,
                "htf_match_30m": swing.htf_match_30m,
                "htf_match_1h": swing.htf_match_1h,
                "bars_since_prev_30m_same": swing.bars_since_prev_30m_same,
                "bars_since_prev_1h_same": swing.bars_since_prev_1h_same,
                "label_quality": swing.label_quality,
                "label_tier": swing.label_tier,
            }
        )
    return pd.DataFrame(rows)


def maybe_save_plot(df_labeled: pd.DataFrame, swings: list, plot_output: Path) -> None:
    if not swings:
        return

    swing_dates = pd.Series([s.swing_time.date() for s in swings])
    swing_weeks = swing_dates.apply(lambda value: value.isocalendar()[1])
    modal_week = swing_weeks.mode().iloc[0]
    modal_year = swing_dates[swing_weeks == modal_week].iloc[0].year
    selected = [
        swing
        for swing in swings
        if swing.swing_time.date().isocalendar()[1] == modal_week and swing.swing_time.year == modal_year
    ]

    if selected:
        start = str(selected[0].swing_time.date() - pd.Timedelta(days=1))
        end = str(selected[-1].swing_time.date() + pd.Timedelta(days=1))
    else:
        midpoint = len(df_labeled) // 2
        start = str(df_labeled.index[max(0, midpoint - 1000)].date())
        end = str(df_labeled.index[min(len(df_labeled) - 1, midpoint + 1000)].date())

    plot_output.parent.mkdir(parents=True, exist_ok=True)
    plot_swings(
        df_labeled,
        swings,
        df_labeled["label_long_ote"].values,
        df_labeled["label_short_ote"].values,
        start_date=start,
        end_date=end,
        save_path=str(plot_output),
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    print(f"Loading source data: {args.input}")
    df_5m = load_fx_csv(args.input)
    df_30m = resample_ohlcv(df_5m, "30min")
    df_1hr = resample_ohlcv(df_5m, "1h")

    print(f"5m rows:  {len(df_5m):,}")
    print(f"30m rows: {len(df_30m):,}")
    print(f"1h rows:  {len(df_1hr):,}")
    print(f"Anomaly bars removed: {int(df_5m.attrs.get('anomaly_count', 0)):,}")
    print(f"Date range: {df_5m.index[0]} -> {df_5m.index[-1]}")

    params = build_default_params()
    df_labeled, diagnostics, swings = build_ote_labels(
        df_5m=df_5m,
        df_30m=df_30m,
        df_1hr=df_1hr,
        params=params,
        verbose=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled_to_save = df_labeled.reset_index().rename(columns={"index": "timestamp"})
    labeled_to_save.to_csv(args.output, index=False)

    swings_df = swings_to_frame(swings)
    args.swings_output.parent.mkdir(parents=True, exist_ok=True)
    swings_df.to_csv(args.swings_output, index=False)

    print(f"\nSaved bar labels to: {args.output}")
    print(f"Saved swing metadata to: {args.swings_output}")
    print(summarize_labels(df_labeled))

    if args.plot:
        maybe_save_plot(df_labeled, swings, args.plot_output)
        print(f"Saved diagnostic plot to: {args.plot_output}")

    if diagnostics.get("sanity_checks"):
        failed = [name for name, passed, _ in diagnostics["sanity_checks"] if not passed]
        if failed:
            print(f"Sanity checks with warnings: {failed}")


if __name__ == "__main__":
    main()
