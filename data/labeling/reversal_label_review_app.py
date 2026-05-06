"""
Interactive reviewer for reversal labels on EURUSD 5-minute data.

Workflow:
1. Load the auto-labeled CSV.
2. Plot price, swing zones, and precise entry labels.
3. Click a bar to select it.
4. Add/remove long/short zone or entry labels as overrides.
5. Save overrides separately and optionally export a reviewed label file.
"""

from __future__ import annotations

import os
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
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_all_labels.csv"
DEFAULT_OVERRIDES_PATH = PROJECT_ROOT / "data" / "labeling" / "review" / "reversal_label_overrides.csv"
DEFAULT_REVIEWED_OUTPUT = PROJECT_ROOT / "data" / "labeling" / "labeled_data" / "eurusd_5min_all_labels_reversal_reviewed.csv"

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

LABEL_TARGETS = [
    ("label_long_reversal", "Long Reversal Zone"),
    ("label_short_reversal", "Short Reversal Zone"),
    ("label_long_reversal_entry", "Long Reversal Entry"),
    ("label_short_reversal_entry", "Short Reversal Entry"),
]

DEFAULT_CHUNK = 2_000
DEFAULT_EMA_PERIOD = 50


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return normalize_reversal_columns(df)


def load_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "target_label", "value", "note", "source"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    expected = {"timestamp", "target_label", "value", "note", "source"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Overrides file missing columns: {missing}")
    df["target_label"] = df["target_label"].replace(LEGACY_REVERSAL_COLUMN_MAP)
    return df.sort_values("timestamp").reset_index(drop=True)


def normalize_reversal_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for legacy, current in LEGACY_REVERSAL_COLUMN_MAP.items():
        if legacy in normalized.columns and current not in normalized.columns:
            normalized = normalized.rename(columns={legacy: current})
    return normalized


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


def make_review_figure(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    selected_ts: pd.Timestamp | None,
    ema_period: int,
    show_ema: bool,
    show_zone_labels: bool,
    show_entry_labels: bool,
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
        long_zone = view["label_long_reversal"].astype(bool)
        short_zone = view["label_short_reversal"].astype(bool)
        if long_zone.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[long_zone, "timestamp"],
                    y=view.loc[long_zone, "low"] * 0.99995,
                    mode="markers",
                    marker=dict(color="#198754", size=7, symbol="circle"),
                    name="Long Reversal Zone",
                )
            )
        if short_zone.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[short_zone, "timestamp"],
                    y=view.loc[short_zone, "high"] * 1.00005,
                    mode="markers",
                    marker=dict(color="#dc3545", size=7, symbol="circle"),
                    name="Short Reversal Zone",
                )
            )

    if show_entry_labels:
        long_entry = view["label_long_reversal_entry"].astype(bool)
        short_entry = view["label_short_reversal_entry"].astype(bool)
        if long_entry.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[long_entry, "timestamp"],
                    y=view.loc[long_entry, "low"] * 0.9998,
                    mode="markers",
                    marker=dict(color="#0d6efd", size=11, symbol="triangle-up"),
                    name="Long Reversal Entry",
                )
            )
        if short_entry.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[short_entry, "timestamp"],
                    y=view.loc[short_entry, "high"] * 1.0002,
                    mode="markers",
                    marker=dict(color="#6f42c1", size=11, symbol="triangle-down"),
                    name="Short Reversal Entry",
                )
            )

    if selected_ts is not None:
        row = view.loc[view["timestamp"] == selected_ts]
        if not row.empty:
            y = float(row["close"].iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=[selected_ts],
                    y=[y],
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
        title=f"Reversal Label Review | rows {start_idx:,}-{end_idx:,} of {len(df):,}",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def selected_bar_summary(df: pd.DataFrame, ts: pd.Timestamp | None) -> str:
    if ts is None:
        return "Click a bar on the chart to inspect or edit it."
    row = df.loc[df["timestamp"] == ts]
    if row.empty:
        return "Selected bar not found."
    row = row.iloc[0]
    return (
        f"{row['timestamp']} | O:{row['open']:.5f} H:{row['high']:.5f} L:{row['low']:.5f} C:{row['close']:.5f} | "
        f"LZ={int(row['label_long_reversal'])} SZ={int(row['label_short_reversal'])} "
        f"LE={int(row['label_long_reversal_entry'])} SE={int(row['label_short_reversal_entry'])} | "
        f"QL={row.get('label_quality_long_reversal', 0):.2f} QS={row.get('label_quality_short_reversal', 0):.2f}"
    )


def main() -> None:
    labels_path = DEFAULT_LABELS_PATH
    overrides_path = DEFAULT_OVERRIDES_PATH
    reviewed_output = DEFAULT_REVIEWED_OUTPUT

    df_base = load_labels(labels_path)
    overrides_df = load_overrides(overrides_path)
    df_reviewed = apply_overrides(df_base, overrides_df)

    init_end = min(DEFAULT_CHUNK, len(df_reviewed))
    init_fig = make_review_figure(
        df_reviewed,
        0,
        init_end,
        None,
        DEFAULT_EMA_PERIOD,
        True,
        True,
        True,
    )

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Reversal Label Review"

    app.layout = html.Div(
        style={"maxWidth": "1700px", "margin": "0 auto", "padding": "12px", "fontFamily": "Segoe UI, sans-serif"},
        children=[
            html.H3("EURUSD 5m Reversal Label Review", style={"marginBottom": "4px"}),
            html.P(
                "Click a bar to inspect it. Save edits as overrides, then export a reviewed label file when you're happy.",
                style={"marginTop": "0", "color": "#555"},
            ),
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap",
                       "padding": "8px", "background": "#eef5fb", "borderRadius": "6px", "marginBottom": "10px"},
                children=[
                    html.Strong("Index range:"),
                    html.Span("Start"),
                    dcc.Input(id="idx-start", type="number", value=0, min=0, max=len(df_reviewed) - 1, step=1, style={"width": "120px"}),
                    html.Span("End"),
                    dcc.Input(id="idx-end", type="number", value=init_end, min=1, max=len(df_reviewed), step=1, style={"width": "120px"}),
                    html.Button("Load slice", id="load-slice-btn", n_clicks=0),
                    html.Button("Prev", id="prev-chunk-btn", n_clicks=0),
                    html.Button("Next", id="next-chunk-btn", n_clicks=0),
                    html.Span(f"Rows: {len(df_reviewed):,}", style={"marginLeft": "8px", "color": "#666"}),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "14px", "alignItems": "center", "flexWrap": "wrap",
                       "padding": "8px", "background": "#f8f6ef", "borderRadius": "6px", "marginBottom": "10px"},
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
                    dcc.Input(id="ema-period", type="number", value=DEFAULT_EMA_PERIOD, min=2, max=500, step=1, style={"width": "80px"}),
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
                                id="selected-bar-readout",
                                style={"padding": "10px", "background": "#f8f9fa", "borderRadius": "6px", "minHeight": "90px"},
                                children="Click a bar on the chart to inspect or edit it.",
                            ),
                            html.Div(
                                style={"padding": "10px", "background": "#f8f9fa", "borderRadius": "6px"},
                                children=[
                                    html.Strong("Edit target"),
                                    dcc.RadioItems(
                                        id="edit-target",
                                        options=[{"label": label, "value": value} for value, label in LABEL_TARGETS],
                                        value="label_long_reversal_entry",
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
                                            html.Button("Save override", id="save-override-btn", n_clicks=0,
                                                        style={"background": "#0d6efd", "color": "white", "border": "none", "padding": "6px 12px"}),
                                            html.Button("Export reviewed labels", id="export-reviewed-btn", n_clicks=0,
                                                        style={"background": "#198754", "color": "white", "border": "none", "padding": "6px 12px"}),
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
                                id="override-summary",
                                style={"padding": "10px", "background": "#f8f9fa", "borderRadius": "6px"},
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="selected-ts-store"),
            dcc.Store(id="reviewed-store", data=df_reviewed.to_json(date_format="iso", orient="split")),
            dcc.Store(id="overrides-store", data=overrides_df.to_json(date_format="iso", orient="split")),
        ],
    )

    @app.callback(
        Output("selected-ts-store", "data"),
        Output("selected-bar-readout", "children"),
        Input("chart", "clickData"),
        State("reviewed-store", "data"),
        prevent_initial_call=True,
    )
    def on_click(click_data, reviewed_json):
        if not click_data:
            return no_update, no_update
        df = pd.read_json(reviewed_json, orient="split")
        ts = click_data["points"][0]["x"]
        _, snapped = nearest_row(df, ts)
        return (str(snapped) if snapped is not None else None), selected_bar_summary(df, snapped)

    @app.callback(
        Output("reviewed-store", "data"),
        Output("overrides-store", "data"),
        Output("status", "children"),
        Input("save-override-btn", "n_clicks"),
        Input("export-reviewed-btn", "n_clicks"),
        State("selected-ts-store", "data"),
        State("edit-target", "value"),
        State("edit-value", "value"),
        State("override-note", "value"),
        State("reviewed-store", "data"),
        State("overrides-store", "data"),
        prevent_initial_call=True,
    )
    def save_or_export(save_n, export_n, selected_ts, target_label, edit_value, note, reviewed_json, overrides_json):
        trig = ctx.triggered_id
        reviewed = pd.read_json(reviewed_json, orient="split")
        overrides = pd.read_json(overrides_json, orient="split")

        if trig == "save-override-btn":
            if not selected_ts:
                return no_update, no_update, "Select a bar before saving an override."
            ts = pd.to_datetime(selected_ts)
            new_row = pd.DataFrame([
                {
                    "timestamp": ts,
                    "target_label": target_label,
                    "value": int(edit_value),
                    "note": (note or "").strip(),
                    "source": "human_review",
                }
            ])
            overrides = pd.concat([overrides, new_row], ignore_index=True)
            reviewed = apply_overrides(df_base, overrides)
            overrides_path.parent.mkdir(parents=True, exist_ok=True)
            overrides.to_csv(overrides_path, index=False)
            return (
                reviewed.to_json(date_format="iso", orient="split"),
                overrides.to_json(date_format="iso", orient="split"),
                f"Saved override: {target_label} -> {edit_value} at {ts}. Overrides file: {overrides_path}",
            )

        if trig == "export-reviewed-btn":
            reviewed_output.parent.mkdir(parents=True, exist_ok=True)
            reviewed.to_csv(reviewed_output, index=False)
            return no_update, no_update, f"Exported reviewed labels to: {reviewed_output}"

        return no_update, no_update, no_update

    @app.callback(
        Output("chart", "figure"),
        Output("idx-start", "value"),
        Output("idx-end", "value"),
        Output("override-summary", "children"),
        Input("load-slice-btn", "n_clicks"),
        Input("next-chunk-btn", "n_clicks"),
        Input("prev-chunk-btn", "n_clicks"),
        Input("apply-display-btn", "n_clicks"),
        Input("reviewed-store", "data"),
        State("idx-start", "value"),
        State("idx-end", "value"),
        State("selected-ts-store", "data"),
        State("ema-period", "value"),
        State("display-toggles", "value"),
        State("overrides-store", "data"),
    )
    def refresh_chart(load_n, next_n, prev_n, display_n, reviewed_json, start_val, end_val, selected_ts, ema_period, display_toggles, overrides_json):
        reviewed = pd.read_json(reviewed_json, orient="split")
        overrides = pd.read_json(overrides_json, orient="split")
        trig = ctx.triggered_id

        start_val = int(start_val or 0)
        end_val = int(end_val or min(DEFAULT_CHUNK, len(reviewed)))
        chunk = max(end_val - start_val, 1)

        if trig == "next-chunk-btn":
            start_val = min(end_val, len(reviewed) - 1)
            end_val = min(start_val + chunk, len(reviewed))
        elif trig == "prev-chunk-btn":
            start_val = max(start_val - chunk, 0)
            end_val = min(start_val + chunk, len(reviewed))
        else:
            start_val = max(0, min(start_val, len(reviewed) - 1))
            end_val = max(start_val + 1, min(end_val, len(reviewed)))

        selected = pd.to_datetime(selected_ts) if selected_ts else None
        toggles = set(display_toggles or [])
        fig = make_review_figure(
            reviewed,
            start_val,
            end_val,
            selected,
            int(ema_period or DEFAULT_EMA_PERIOD),
            "ema" in toggles,
            "zone" in toggles,
            "entry" in toggles,
        )

        if overrides.empty:
            summary = "Overrides: 0"
        else:
            latest = overrides.groupby(["target_label", "value"]).size().reset_index(name="count")
            parts = [f"{row.target_label}={int(row.value)} ({int(row.count)})" for row in latest.itertuples(index=False)]
            summary = "Overrides: " + " | ".join(parts)

        return fig, start_val, end_val, summary

    print(f"Launching reversal review app for: {labels_path}")
    print(f"Overrides file: {overrides_path}")
    print(f"Reviewed export: {reviewed_output}")
    app.run(debug=False)


if __name__ == "__main__":
    main()
