"""
Interactive reviewer for reversal, continuation, and breakout labels on EURUSD 5-minute data.

Workflow:
1. Load a labeled CSV.
2. Switch between reversal, continuation-pullback, and breakout labels.
3. Inspect the selected slice visually on the chart and statistically in the side panel.
4. Click a bar to inspect it.
5. Add/remove long/short zone or entry labels as overrides.
6. Save overrides separately and optionally export a reviewed label file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
except ImportError as exc:
    raise SystemExit(
        "This review app requires `plotly` and `dash`.\n"
        "Install them in your environment, then rerun:\n"
        "  pip install plotly dash"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABELS_PATH_CANDIDATES = [
    PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_all_labels.csv",
    PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_ote_labels_full.csv",
    PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "labeled_output.csv",
]
OVERRIDES_PATH_CANDIDATES = [
    PROJECT_ROOT / "data" / "labeling" / "review" / "label_overrides.csv",
    PROJECT_ROOT / "data" / "labeling" / "review" / "reversal_label_overrides.csv",
]

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

DEFAULT_CHUNK = 2_000
DEFAULT_EMA_PERIOD = 50


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
    zone_long_color: str
    zone_short_color: str
    entry_long_color: str
    entry_short_color: str

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
    def target_options(self) -> list[tuple[str, str]]:
        return [
            (self.zone_long_col, f"Long {self.short_title} Zone"),
            (self.zone_short_col, f"Short {self.short_title} Zone"),
            (self.entry_long_col, f"Long {self.short_title} Entry"),
            (self.entry_short_col, f"Short {self.short_title} Entry"),
        ]


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
        zone_long_color="#198754",
        zone_short_color="#dc3545",
        entry_long_color="#0d6efd",
        entry_short_color="#6f42c1",
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
        zone_long_color="#0f766e",
        zone_short_color="#ea580c",
        entry_long_color="#0891b2",
        entry_short_color="#92400e",
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
        zone_long_color="#65a30d",
        zone_short_color="#be123c",
        entry_long_color="#0284c7",
        entry_short_color="#c026d3",
    ),
}


def resolve_default_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_reviewed_output_for(labels_path: Path) -> Path:
    return labels_path.with_name(f"{labels_path.stem}_reviewed.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive review app for multi-family OTE labels.")
    parser.add_argument("--labels", type=Path, default=resolve_default_path(LABELS_PATH_CANDIDATES))
    parser.add_argument("--overrides", type=Path, default=resolve_default_path(OVERRIDES_PATH_CANDIDATES))
    parser.add_argument("--reviewed-output", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--prepared-target-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["auto", "all", "train", "val", "test", "dev"], default="auto")
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--initial-family", choices=list(FAMILY_CONFIGS), default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for legacy, current in LEGACY_REVERSAL_COLUMN_MAP.items():
        if legacy in normalized.columns and current not in normalized.columns:
            normalized = normalized.rename(columns={legacy: current})
    return normalized


def detect_family_availability(columns: pd.Index) -> dict[str, bool]:
    availability: dict[str, bool] = {}
    column_set = set(columns)
    for key, config in FAMILY_CONFIGS.items():
        tracked = set(config.binary_columns) | set(config.quality_columns)
        availability[key] = any(column in column_set for column in tracked)
    return availability


def ensure_review_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def load_labels(path: Path) -> tuple[pd.DataFrame, dict[str, bool]]:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Labels file must contain a 'timestamp' column: {path}")
    if "source_row_idx" not in df.columns:
        df["source_row_idx"] = np.arange(len(df), dtype=np.int64)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = normalize_label_columns(df)
    availability = detect_family_availability(df.columns)
    df = ensure_review_columns(df)
    return df, availability


def load_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "target_label", "value", "note", "source"])
    df = pd.read_csv(path)
    expected = {"timestamp", "target_label", "value", "note", "source"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Overrides file missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df["target_label"] = df["target_label"].replace(LEGACY_REVERSAL_COLUMN_MAP)
    return df.sort_values("timestamp").reset_index(drop=True)


def apply_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    reviewed = df.copy()
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


def frame_to_json(df: pd.DataFrame) -> str:
    return df.to_json(date_format="iso", orient="split")


def frame_from_json(payload: str) -> pd.DataFrame:
    df = pd.read_json(StringIO(payload), orient="split")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def resolve_project_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (PROJECT_ROOT / resolved).resolve()


def resolve_prepared_target_dir(
    prepared_target_dir: Path | None,
    model_dir: Path | None,
) -> Path | None:
    explicit_dir = resolve_project_path(prepared_target_dir)
    if explicit_dir is not None:
        return explicit_dir

    resolved_model_dir = resolve_project_path(model_dir)
    if resolved_model_dir is None:
        return None

    candidate_files = [
        resolved_model_dir / "training_summary.json",
        resolved_model_dir / "model_config.json",
    ]
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        prepared_dir = payload.get("prepared_dir")
        if prepared_dir is None:
            prepared_dir = payload.get("config", {}).get("prepared_root")
        if prepared_dir is None:
            continue
        resolved_prepared = resolve_project_path(prepared_dir)
        if resolved_prepared is None:
            continue
        if resolved_prepared.is_dir() and (resolved_prepared / "test.csv").exists():
            return resolved_prepared

        target_name = payload.get("target") or payload.get("report", {}).get("target_name") or resolved_model_dir.name
        if target_name:
            target_dir = resolved_prepared / str(target_name)
            if target_dir.is_dir() and (target_dir / "test.csv").exists():
                return target_dir
    return None


def effective_split_name(split: str, prepared_target_dir: Path | None) -> str:
    if split != "auto":
        return split
    return "test" if prepared_target_dir is not None else "all"


def load_split_source_rows(prepared_target_dir: Path | None, split: str) -> np.ndarray | None:
    if prepared_target_dir is None or split == "all":
        return None

    split_files = {
        "train": ["train.csv"],
        "val": ["val.csv"],
        "test": ["test.csv"],
        "dev": ["train.csv", "val.csv"],
    }
    filenames = split_files[split]
    source_rows: list[np.ndarray] = []
    for filename in filenames:
        path = prepared_target_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prepared split file not found: {path}")
        split_df = pd.read_csv(path, usecols=["source_row_idx"])
        source_rows.append(split_df["source_row_idx"].to_numpy(dtype=np.int64))
    if not source_rows:
        return None
    return np.unique(np.concatenate(source_rows))


def filter_labels_to_source_rows(df: pd.DataFrame, source_rows: np.ndarray | None) -> tuple[pd.DataFrame, int]:
    if source_rows is None:
        return df.copy(), 0
    filtered = df.loc[df["source_row_idx"].isin(source_rows)].copy()
    matched = set(pd.to_numeric(filtered["source_row_idx"], errors="coerce").dropna().astype(np.int64).tolist())
    missing = int(len(source_rows) - len(matched))
    return filtered.reset_index(drop=True), missing


def evenly_sample_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.copy()
    positions = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    positions = np.unique(positions)
    return df.iloc[positions].copy().reset_index(drop=True)


def nearest_row(df: pd.DataFrame, ts) -> tuple[int | None, pd.Timestamp | None]:
    if ts is None or df.empty:
        return None, None
    parsed = pd.to_datetime(ts, errors="coerce")
    if pd.isna(parsed):
        return None, None
    idx = df["timestamp"].searchsorted(parsed)
    candidates = []
    if idx < len(df):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None, None
    best = min(candidates, key=lambda i: abs(df.at[i, "timestamp"] - parsed))
    return int(best), df.at[best, "timestamp"]


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def resolve_structural_atr_column(df: pd.DataFrame, config: LabelFamilyConfig) -> str | None:
    for column in config.structural_atr_candidates:
        if column in df.columns:
            return column
    return None


def positive_count(series: pd.Series) -> int:
    return int(series.fillna(0).astype(bool).sum())


def positive_rate_pct(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.fillna(0).astype(bool).mean() * 100.0)


def positive_mean(series: pd.Series) -> float:
    valid = series[pd.to_numeric(series, errors="coerce").fillna(0) > 0]
    if valid.empty:
        return 0.0
    return float(pd.to_numeric(valid, errors="coerce").mean())


def session_bucket(hour: int) -> str:
    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 12:
        return "London"
    if 12 <= hour < 17:
        return "NY AM"
    return "NY PM/Off"


def build_family_stats_text(
    df: pd.DataFrame,
    config: LabelFamilyConfig,
    *,
    family_available: bool,
    start_idx: int,
    end_idx: int,
) -> str:
    if not family_available:
        return (
            f"{config.title} Statistics\n"
            f"{'=' * (len(config.title) + 11)}\n"
            "This label family is not present in the loaded CSV."
        )

    usable = df.loc[~df["warmup_mask"].fillna(False).astype(bool)].copy()
    slice_df = df.iloc[start_idx:end_idx].copy()
    slice_usable = slice_df.loc[~slice_df["warmup_mask"].fillna(False).astype(bool)].copy()
    atr_column = resolve_structural_atr_column(df, config)

    long_zone_total = positive_count(usable[config.zone_long_col])
    short_zone_total = positive_count(usable[config.zone_short_col])
    long_entry_total = positive_count(usable[config.entry_long_col])
    short_entry_total = positive_count(usable[config.entry_short_col])

    long_zone_slice = positive_count(slice_usable[config.zone_long_col])
    short_zone_slice = positive_count(slice_usable[config.zone_short_col])
    long_entry_slice = positive_count(slice_usable[config.entry_long_col])
    short_entry_slice = positive_count(slice_usable[config.entry_short_col])

    hourly = usable.assign(hour=usable["timestamp"].dt.hour).groupby("hour")[
        [config.entry_long_col, config.entry_short_col]
    ].mean()
    best_long_hour = int(hourly[config.entry_long_col].idxmax()) if not hourly.empty else 0
    best_short_hour = int(hourly[config.entry_short_col].idxmax()) if not hourly.empty else 0

    session_table = usable.assign(
        session=usable["timestamp"].dt.hour.map(session_bucket)
    ).groupby("session")[[config.entry_long_col, config.entry_short_col]].mean()
    best_long_session = (
        str(session_table[config.entry_long_col].idxmax())
        if not session_table.empty
        else "n/a"
    )
    best_short_session = (
        str(session_table[config.entry_short_col].idxmax())
        if not session_table.empty
        else "n/a"
    )

    entry_overlap = positive_count(
        slice_usable[config.entry_long_col].fillna(0).astype(bool)
        & slice_usable[config.entry_short_col].fillna(0).astype(bool)
    )

    lines = [
        f"{config.title} Statistics",
        "=" * (len(config.title) + 11),
        f"Usable rows total / slice: {len(usable):,} / {len(slice_usable):,}",
        f"Long zone total / slice:   {long_zone_total:,} / {long_zone_slice:,}",
        f"Short zone total / slice:  {short_zone_total:,} / {short_zone_slice:,}",
        f"Long entry total / slice:  {long_entry_total:,} / {long_entry_slice:,}",
        f"Short entry total / slice: {short_entry_total:,} / {short_entry_slice:,}",
        f"Long zone rate total / slice:  {positive_rate_pct(usable[config.zone_long_col]):.3f}% / {positive_rate_pct(slice_usable[config.zone_long_col]):.3f}%",
        f"Short zone rate total / slice: {positive_rate_pct(usable[config.zone_short_col]):.3f}% / {positive_rate_pct(slice_usable[config.zone_short_col]):.3f}%",
        f"Long entry rate total / slice:  {positive_rate_pct(usable[config.entry_long_col]):.3f}% / {positive_rate_pct(slice_usable[config.entry_long_col]):.3f}%",
        f"Short entry rate total / slice: {positive_rate_pct(usable[config.entry_short_col]):.3f}% / {positive_rate_pct(slice_usable[config.entry_short_col]):.3f}%",
        f"Zone quality mean L / S:   {positive_mean(usable[config.zone_quality_long_col]):.3f} / {positive_mean(usable[config.zone_quality_short_col]):.3f}",
        f"Entry quality mean L / S:  {positive_mean(usable[config.entry_quality_long_col]):.3f} / {positive_mean(usable[config.entry_quality_short_col]):.3f}",
        f"Peak long / short entry hour: {best_long_hour:02d}:00 / {best_short_hour:02d}:00",
        f"Peak long / short session: {best_long_session} / {best_short_session}",
        f"Slice entry overlap (long+short same bar): {entry_overlap:,}",
    ]
    if atr_column is not None:
        atr_values = pd.to_numeric(usable[atr_column], errors="coerce").dropna()
        if not atr_values.empty:
            lines.append(f"Average structural ATR: {float(atr_values.mean()):.6f}")
    return "\n".join(lines)


def selected_bar_summary(df: pd.DataFrame, ts: pd.Timestamp | None) -> str:
    if ts is None:
        return "Click a bar on the chart to inspect or edit it."
    row = df.loc[df["timestamp"] == ts]
    if row.empty:
        return "Selected bar not found."
    row = row.iloc[0]

    def safe_flag(column: str) -> int:
        return int(bool(row.get(column, 0)))

    def safe_score(column: str) -> float:
        value = pd.to_numeric(pd.Series([row.get(column, 0.0)]), errors="coerce").fillna(0.0).iloc[0]
        return float(value)

    lines = [
        (
            f"{row['timestamp']} | "
            f"O:{row['open']:.5f} H:{row['high']:.5f} L:{row['low']:.5f} C:{row['close']:.5f} "
            f"| warmup={int(bool(row.get('warmup_mask', False)))}"
        )
    ]

    for config in FAMILY_CONFIGS.values():
        lines.append(
            (
                f"{config.short_title}: "
                f"LZ={safe_flag(config.zone_long_col)} "
                f"SZ={safe_flag(config.zone_short_col)} "
                f"LE={safe_flag(config.entry_long_col)} "
                f"SE={safe_flag(config.entry_short_col)} | "
                f"LQ={safe_score(config.zone_quality_long_col):.2f}/"
                f"{safe_score(config.zone_quality_short_col):.2f} | "
                f"EQ={safe_score(config.entry_quality_long_col):.2f}/"
                f"{safe_score(config.entry_quality_short_col):.2f}"
            )
        )
    return "\n".join(lines)


def make_review_figure(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    selected_ts: pd.Timestamp | None,
    ema_period: int,
    show_ema: bool,
    show_zone_labels: bool,
    show_entry_labels: bool,
    family_config: LabelFamilyConfig,
) -> go.Figure:
    view = df.iloc[start_idx:end_idx].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=view["timestamp"],
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="EURUSD 5m",
        )
    )

    if show_ema:
        fig.add_trace(
            go.Scatter(
                x=view["timestamp"],
                y=compute_ema(view["close"], ema_period),
                mode="lines",
                line=dict(color="#ff6b35", width=1.8),
                name=f"EMA({ema_period})",
            )
        )

    if show_zone_labels:
        long_zone = view[family_config.zone_long_col].fillna(0).astype(bool)
        short_zone = view[family_config.zone_short_col].fillna(0).astype(bool)
        if long_zone.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[long_zone, "timestamp"],
                    y=view.loc[long_zone, "low"] * 0.99995,
                    mode="markers",
                    marker=dict(color=family_config.zone_long_color, size=7, symbol="circle"),
                    name=f"Long {family_config.short_title} Zone",
                )
            )
        if short_zone.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[short_zone, "timestamp"],
                    y=view.loc[short_zone, "high"] * 1.00005,
                    mode="markers",
                    marker=dict(color=family_config.zone_short_color, size=7, symbol="circle"),
                    name=f"Short {family_config.short_title} Zone",
                )
            )

    if show_entry_labels:
        long_entry = view[family_config.entry_long_col].fillna(0).astype(bool)
        short_entry = view[family_config.entry_short_col].fillna(0).astype(bool)
        if long_entry.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[long_entry, "timestamp"],
                    y=view.loc[long_entry, "low"] * 0.9998,
                    mode="markers",
                    marker=dict(color=family_config.entry_long_color, size=11, symbol="triangle-up"),
                    name=f"Long {family_config.short_title} Entry",
                )
            )
        if short_entry.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[short_entry, "timestamp"],
                    y=view.loc[short_entry, "high"] * 1.0002,
                    mode="markers",
                    marker=dict(color=family_config.entry_short_color, size=11, symbol="triangle-down"),
                    name=f"Short {family_config.short_title} Entry",
                )
            )

    if selected_ts is not None:
        row = view.loc[view["timestamp"] == selected_ts]
        if not row.empty:
            fig.add_trace(
                go.Scatter(
                    x=[selected_ts],
                    y=[float(row["close"].iloc[0])],
                    mode="markers",
                    marker=dict(color="#111111", size=14, symbol="x"),
                    name="Selected",
                )
            )

    fig.update_layout(
        template="plotly_white",
        height=900,
        dragmode="zoom",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=30, r=20, t=50, b=30),
        title=f"{family_config.title} Label Review | rows {start_idx:,}-{end_idx:,} of {len(df):,}",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def build_override_summary_text(overrides: pd.DataFrame) -> str:
    if overrides.empty:
        return "Overrides: 0"

    family_name_by_target = {
        target: config.short_title
        for config in FAMILY_CONFIGS.values()
        for target, _label in config.target_options
    }
    counted = overrides.copy()
    counted["family"] = counted["target_label"].map(family_name_by_target).fillna("Other")
    family_parts = []
    for family_name, family_df in counted.groupby("family", sort=True):
        family_parts.append(f"{family_name}: {len(family_df):,}")
    detail = (
        counted.groupby(["target_label", "value"]).size().reset_index(name="count")
        .sort_values(["target_label", "value"])
    )
    detail_parts = [
        f"{row.target_label}={int(row.value)} ({int(row.count)})"
        for row in detail.itertuples(index=False)
    ]
    return "Overrides by family: " + " | ".join(family_parts) + "\n" + "Details: " + " | ".join(detail_parts)


def first_available_family(
    family_availability: dict[str, bool],
    requested_family: str | None,
) -> str:
    if requested_family and family_availability.get(requested_family):
        return requested_family
    for key, available in family_availability.items():
        if available:
            return key
    return "reversal"


def main() -> None:
    args = parse_args()
    labels_path = args.labels
    overrides_path = args.overrides
    reviewed_output = args.reviewed_output or default_reviewed_output_for(labels_path)
    chunk_size = max(int(args.chunk_size), 100)

    prepared_target_dir = resolve_prepared_target_dir(args.prepared_target_dir, args.model_dir)
    requested_split = effective_split_name(args.split, prepared_target_dir)

    if requested_split != "all" and prepared_target_dir is None:
        raise ValueError(
            "A split filter was requested but no prepared target directory could be resolved. "
            "Pass --prepared-target-dir directly or --model-dir pointing at a trained model artifact directory."
        )

    df_base, family_availability = load_labels(labels_path)
    filter_notes: list[str] = []
    split_source_rows = load_split_source_rows(prepared_target_dir, requested_split)
    missing_source_rows = 0
    if split_source_rows is not None:
        df_base, missing_source_rows = filter_labels_to_source_rows(df_base, split_source_rows)
        filter_notes.append(
            f"Split filter: {requested_split} from {prepared_target_dir} "
            f"({len(df_base):,} matched rows, {missing_source_rows:,} missing source rows)"
        )
    if args.sample_rows is not None:
        before_rows = len(df_base)
        df_base = evenly_sample_rows(df_base, int(args.sample_rows))
        filter_notes.append(f"Evenly sampled rows: {before_rows:,} -> {len(df_base):,}")

    if df_base.empty:
        raise ValueError("No label rows remain after applying the requested split/sample filters.")

    overrides_df = load_overrides(overrides_path)
    df_reviewed = apply_overrides(df_base, overrides_df)

    initial_family = first_available_family(family_availability, args.initial_family)
    initial_config = FAMILY_CONFIGS[initial_family]
    init_end = min(chunk_size, len(df_reviewed))
    init_fig = make_review_figure(
        df_reviewed,
        0,
        init_end,
        None,
        DEFAULT_EMA_PERIOD,
        True,
        True,
        True,
        initial_config,
    )

    family_options = []
    for key, config in FAMILY_CONFIGS.items():
        available = family_availability.get(key, False)
        label = config.title if available else f"{config.title} (missing in file)"
        family_options.append({"label": label, "value": key, "disabled": not available})

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Multi-Family Label Review"

    app.layout = html.Div(
        style={"maxWidth": "1750px", "margin": "0 auto", "padding": "12px", "fontFamily": "Segoe UI, sans-serif"},
        children=[
            html.H3("EURUSD 5m Multi-Family Label Review", style={"marginBottom": "4px"}),
            html.P(
                "Switch between reversal, continuation-pullback, and breakout labels to inspect how they look visually and statistically.",
                style={"marginTop": "0", "color": "#555"},
            ),
            html.Div(
                style={"padding": "8px", "background": "#eef5fb", "borderRadius": "6px", "marginBottom": "10px", "color": "#445"},
                children=[
                    html.Div(f"Labels file: {labels_path}"),
                    html.Div(f"Overrides file: {overrides_path}"),
                    html.Div(f"Reviewed export path: {reviewed_output}"),
                    html.Div(f"Prepared target dir: {prepared_target_dir if prepared_target_dir is not None else 'none'}"),
                    html.Div(f"Active split filter: {requested_split}"),
                    html.Div("Filter notes: " + (" | ".join(filter_notes) if filter_notes else "none")),
                ],
            ),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "padding": "8px",
                    "background": "#eef5fb",
                    "borderRadius": "6px",
                    "marginBottom": "10px",
                },
                children=[
                    html.Strong("Label family"),
                    dcc.Dropdown(
                        id="family-selector",
                        options=family_options,
                        value=initial_family,
                        clearable=False,
                        style={"width": "320px"},
                    ),
                    html.Strong("Index range:"),
                    html.Span("Start"),
                    dcc.Input(
                        id="idx-start",
                        type="number",
                        value=0,
                        min=0,
                        max=max(len(df_reviewed) - 1, 0),
                        step=1,
                        style={"width": "120px"},
                    ),
                    html.Span("End"),
                    dcc.Input(
                        id="idx-end",
                        type="number",
                        value=init_end,
                        min=1,
                        max=max(len(df_reviewed), 1),
                        step=1,
                        style={"width": "120px"},
                    ),
                    html.Button("Load slice", id="load-slice-btn", n_clicks=0),
                    html.Button("Prev", id="prev-chunk-btn", n_clicks=0),
                    html.Button("Next", id="next-chunk-btn", n_clicks=0),
                    html.Span(f"Rows: {len(df_reviewed):,}", style={"marginLeft": "8px", "color": "#666"}),
                ],
            ),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "14px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "padding": "8px",
                    "background": "#f8f6ef",
                    "borderRadius": "6px",
                    "marginBottom": "10px",
                },
                children=[
                    dcc.Checklist(
                        id="display-toggles",
                        options=[
                            {"label": " EMA", "value": "ema"},
                            {"label": " Zone labels", "value": "zone"},
                            {"label": " Entry labels", "value": "entry"},
                        ],
                        value=["ema", "zone", "entry"],
                        inline=True,
                    ),
                    html.Span("EMA period"),
                    dcc.Input(
                        id="ema-period",
                        type="number",
                        value=DEFAULT_EMA_PERIOD,
                        min=2,
                        max=500,
                        step=1,
                        style={"width": "80px"},
                    ),
                    html.Button("Apply display", id="apply-display-btn", n_clicks=0),
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "12px"},
                children=[
                    dcc.Graph(
                        id="chart",
                        figure=init_fig,
                        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
                        style={"height": "940px"},
                    ),
                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "gap": "10px"},
                        children=[
                            html.Div(
                                style={"padding": "10px", "background": "#f8f9fa", "borderRadius": "6px"},
                                children=[
                                    html.Strong("Selected bar"),
                                    html.Pre(
                                        id="selected-bar-readout",
                                        style={"margin": "8px 0 0 0", "whiteSpace": "pre-wrap", "minHeight": "110px"},
                                        children="Click a bar on the chart to inspect or edit it.",
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"padding": "10px", "background": "#f8f9fa", "borderRadius": "6px"},
                                children=[
                                    html.Strong("Family stats"),
                                    html.Pre(
                                        id="family-stats",
                                        style={"margin": "8px 0 0 0", "whiteSpace": "pre-wrap", "minHeight": "260px"},
                                        children=build_family_stats_text(
                                            df_reviewed,
                                            initial_config,
                                            family_available=family_availability.get(initial_family, False),
                                            start_idx=0,
                                            end_idx=init_end,
                                        ),
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"padding": "10px", "background": "#f8f9fa", "borderRadius": "6px"},
                                children=[
                                    html.Strong("Edit target"),
                                    dcc.RadioItems(
                                        id="edit-target",
                                        options=[
                                            {"label": label, "value": value}
                                            for value, label in initial_config.target_options
                                        ],
                                        value=initial_config.entry_long_col,
                                    ),
                                    html.Strong("Action", style={"marginTop": "8px", "display": "block"}),
                                    dcc.RadioItems(
                                        id="edit-value",
                                        options=[
                                            {"label": "Add / force on", "value": 1},
                                            {"label": "Remove / force off", "value": 0},
                                        ],
                                        value=1,
                                    ),
                                    html.Span("Note"),
                                    dcc.Input(id="override-note", type="text", value="", debounce=True, style={"width": "100%"}),
                                    html.Div(
                                        style={"display": "flex", "gap": "8px", "marginTop": "10px", "flexWrap": "wrap"},
                                        children=[
                                            html.Button(
                                                "Save override",
                                                id="save-override-btn",
                                                n_clicks=0,
                                                style={"background": "#0d6efd", "color": "white", "border": "none", "padding": "6px 12px"},
                                            ),
                                            html.Button(
                                                "Export reviewed labels",
                                                id="export-reviewed-btn",
                                                n_clicks=0,
                                                style={"background": "#198754", "color": "white", "border": "none", "padding": "6px 12px"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                id="status",
                                style={"padding": "10px", "background": "#fff3cd", "borderRadius": "6px", "minHeight": "70px"},
                                children="No edits saved yet.",
                            ),
                            html.Div(
                                style={"padding": "10px", "background": "#f8f9fa", "borderRadius": "6px"},
                                children=[
                                    html.Strong("Override summary"),
                                    html.Pre(
                                        id="override-summary",
                                        style={"margin": "8px 0 0 0", "whiteSpace": "pre-wrap"},
                                        children=build_override_summary_text(overrides_df),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="selected-ts-store"),
            dcc.Store(id="overrides-store", data=frame_to_json(overrides_df)),
        ],
    )

    @app.callback(
        Output("selected-ts-store", "data"),
        Input("chart", "clickData"),
        prevent_initial_call=True,
    )
    def on_click(click_data):
        if not click_data:
            return no_update
        ts = click_data["points"][0]["x"]
        _, snapped = nearest_row(df_base, ts)
        return str(snapped) if snapped is not None else None

    @app.callback(
        Output("selected-bar-readout", "children"),
        Input("selected-ts-store", "data"),
        Input("overrides-store", "data"),
    )
    def update_selected_bar_readout(selected_ts, overrides_json):
        overrides = frame_from_json(overrides_json)
        df = apply_overrides(df_base, overrides)
        selected = pd.to_datetime(selected_ts) if selected_ts else None
        return selected_bar_summary(df, selected)

    @app.callback(
        Output("edit-target", "options"),
        Output("edit-target", "value"),
        Input("family-selector", "value"),
    )
    def update_edit_targets(family_key):
        config = FAMILY_CONFIGS[family_key]
        options = [{"label": label, "value": value} for value, label in config.target_options]
        return options, config.entry_long_col

    @app.callback(
        Output("overrides-store", "data"),
        Output("status", "children"),
        Input("save-override-btn", "n_clicks"),
        Input("export-reviewed-btn", "n_clicks"),
        State("selected-ts-store", "data"),
        State("edit-target", "value"),
        State("edit-value", "value"),
        State("override-note", "value"),
        State("overrides-store", "data"),
        prevent_initial_call=True,
    )
    def save_or_export(save_n, export_n, selected_ts, target_label, edit_value, note, overrides_json):
        trig = ctx.triggered_id
        overrides = frame_from_json(overrides_json)

        if trig == "save-override-btn":
            if not selected_ts:
                return no_update, "Select a bar before saving an override."
            ts = pd.to_datetime(selected_ts)
            new_row = pd.DataFrame(
                [
                    {
                        "timestamp": ts,
                        "target_label": target_label,
                        "value": int(edit_value),
                        "note": (note or "").strip(),
                        "source": "human_review",
                    }
                ]
            )
            overrides = pd.concat([overrides, new_row], ignore_index=True)
            overrides_path.parent.mkdir(parents=True, exist_ok=True)
            overrides.to_csv(overrides_path, index=False)
            return (
                frame_to_json(overrides),
                f"Saved override: {target_label} -> {edit_value} at {ts}. Overrides file: {overrides_path}",
            )

        if trig == "export-reviewed-btn":
            reviewed = apply_overrides(df_base, overrides)
            reviewed_output.parent.mkdir(parents=True, exist_ok=True)
            reviewed.to_csv(reviewed_output, index=False)
            return no_update, f"Exported reviewed labels to: {reviewed_output}"

        return no_update, no_update

    @app.callback(
        Output("chart", "figure"),
        Output("idx-start", "value"),
        Output("idx-end", "value"),
        Output("override-summary", "children"),
        Output("family-stats", "children"),
        Input("load-slice-btn", "n_clicks"),
        Input("next-chunk-btn", "n_clicks"),
        Input("prev-chunk-btn", "n_clicks"),
        Input("apply-display-btn", "n_clicks"),
        Input("overrides-store", "data"),
        Input("family-selector", "value"),
        State("idx-start", "value"),
        State("idx-end", "value"),
        State("selected-ts-store", "data"),
        State("ema-period", "value"),
        State("display-toggles", "value"),
    )
    def refresh_chart(
        load_n,
        next_n,
        prev_n,
        display_n,
        overrides_json,
        family_key,
        start_val,
        end_val,
        selected_ts,
        ema_period,
        display_toggles,
    ):
        overrides = frame_from_json(overrides_json)
        reviewed = apply_overrides(df_base, overrides)
        trig = ctx.triggered_id

        start_val = int(start_val or 0)
        end_val = int(end_val or min(chunk_size, len(reviewed)))
        chunk = max(end_val - start_val, 1)

        if trig == "next-chunk-btn":
            start_val = min(end_val, max(len(reviewed) - 1, 0))
            end_val = min(start_val + chunk, len(reviewed))
        elif trig == "prev-chunk-btn":
            start_val = max(start_val - chunk, 0)
            end_val = min(start_val + chunk, len(reviewed))
        else:
            start_val = max(0, min(start_val, max(len(reviewed) - 1, 0)))
            end_val = max(start_val + 1, min(end_val, len(reviewed)))

        selected = pd.to_datetime(selected_ts) if selected_ts else None
        toggles = set(display_toggles or [])
        family_config = FAMILY_CONFIGS[family_key]
        fig = make_review_figure(
            reviewed,
            start_val,
            end_val,
            selected,
            int(ema_period or DEFAULT_EMA_PERIOD),
            "ema" in toggles,
            "zone" in toggles,
            "entry" in toggles,
            family_config,
        )
        override_summary = build_override_summary_text(overrides)
        family_stats = build_family_stats_text(
            reviewed,
            family_config,
            family_available=family_availability.get(family_key, False),
            start_idx=start_val,
            end_idx=end_val,
        )
        return fig, start_val, end_val, override_summary, family_stats

    print(f"Launching multi-family review app for: {labels_path}")
    print(f"Overrides file: {overrides_path}")
    print(f"Reviewed export: {reviewed_output}")
    print(f"Prepared target dir: {prepared_target_dir}")
    print(f"Active split filter: {requested_split}")
    if filter_notes:
        print("Filter notes:")
        for note in filter_notes:
            print(f"  - {note}")
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
