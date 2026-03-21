import os
import sys
import traceback
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, ctx

# ===== CONFIG =====
CSV_PATH = r"C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\data\currency_data\EURUSD_5min.csv"
SYMBOL = "EURUSD"

# EURUSD pip size (0.0001). For JPY pairs use 0.01.
PIP_SIZE = 0.0001

# Optional warning only (doesn't block save)
MIN_PIPS_WARN = 12.0

SWINGS_OUT = "swings_5min_smoothed.csv"

# Default index window
DEFAULT_CHUNK = 20_000

# Smoothing defaults
DEFAULT_EMA_PERIOD = 50       # EMA period in bars (1 bar = 1 min)
DEFAULT_SHOW_RAW = True       # show raw price line
DEFAULT_SHOW_SMOOTH = True    # show smoothed overlay
# ==================


def load_data(path: str) -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_csv(
        path,
        usecols=["Date", "Time", "Close"],
        dtype={"Date": str, "Time": str, "Close": "float32"},
    )
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%Y%m%d %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["Datetime", "Close"]).sort_values("Datetime").reset_index(drop=True)
    return df[["Datetime", "Close"]].copy()


def compute_ema(series: pd.Series, period: int) -> np.ndarray:
    """Exponential moving average — fast pandas impl."""
    return series.ewm(span=period, adjust=False).mean().values


def make_figure(df_slice: pd.DataFrame, start_idx: int, end_idx: int,
                total_rows: int, uirev: int,
                ema_period: int = DEFAULT_EMA_PERIOD,
                show_raw: bool = True,
                show_smooth: bool = True) -> go.Figure:
    fig = go.Figure()
    n = len(df_slice)

    # --- Raw price trace ---
    if show_raw:
        raw_opacity = 0.35 if show_smooth else 1.0
        raw_color = f"rgba(100,149,237,{raw_opacity})"  # cornflower blue

        if n <= 20_000:
            fig.add_trace(go.Scattergl(
                x=df_slice["Datetime"], y=df_slice["Close"],
                mode="lines+markers",
                marker=dict(size=3, color=raw_color),
                line=dict(width=1, color=raw_color),
                name="Raw Close",
            ))
        elif n <= 100_000:
            fig.add_trace(go.Scattergl(
                x=df_slice["Datetime"], y=df_slice["Close"],
                mode="lines+markers",
                marker=dict(size=2, color=raw_color),
                line=dict(width=1, color=raw_color),
                name="Raw Close",
            ))
        else:
            fig.add_trace(go.Scattergl(
                x=df_slice["Datetime"], y=df_slice["Close"],
                mode="lines",
                line=dict(width=1, color=raw_color),
                name="Raw Close",
            ))

    # --- Smoothed EMA overlay ---
    if show_smooth and ema_period > 1:
        ema_vals = compute_ema(df_slice["Close"], ema_period)
        fig.add_trace(go.Scattergl(
            x=df_slice["Datetime"], y=ema_vals,
            mode="lines",
            line=dict(width=2.5, color="rgba(255,69,0,0.9)"),  # orange-red
            name=f"EMA({ema_period})",
        ))

    smooth_label = f" | EMA({ema_period})" if show_smooth else ""
    fig.update_layout(
        title=(f"{SYMBOL} (1min) — rows {start_idx:,}–{end_idx:,} "
               f"({n:,} pts) of {total_rows:,}{smooth_label}"),
        xaxis=dict(type="date", rangeslider=dict(visible=True)),
        yaxis_title="Price",
        template="plotly_white",
        height=900,
        margin=dict(l=40, r=20, t=60, b=40),
        dragmode="drawline",
        newshape=dict(line_color="cyan", line_width=2),
        uirevision=uirev,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(fixedrange=False, uirevision=uirev)
    return fig


def parse_ts(ts):
    try:
        return pd.to_datetime(ts)
    except Exception:
        return None


def nearest_point(df_indexed: pd.DataFrame, ts: pd.Timestamp):
    if ts is None or df_indexed.empty:
        return None, None
    idx = df_indexed.index.get_indexer([ts], method="nearest")
    if idx is None or len(idx) == 0 or idx[0] < 0:
        return None, None
    i = int(idx[0])
    return df_indexed.index[i], float(df_indexed["Close"].iloc[i])


def pips_from_prices(p0: float, p1: float, pip_size: float = PIP_SIZE) -> float:
    return abs(float(p1) - float(p0)) / pip_size


def compute_bars(df_indexed: pd.DataFrame, t_entry: pd.Timestamp, t_exit: pd.Timestamp) -> int:
    if t_exit < t_entry:
        t_entry, t_exit = t_exit, t_entry
    return int(len(df_indexed.loc[t_entry:t_exit]))


def extract_latest_shape_update(relayout_data: dict):
    if not relayout_data:
        return None

    print(f"[DEBUG relayoutData keys]: {list(relayout_data.keys())}")

    # FORMAT 1: Full "shapes" array (Plotly sends this on new shape draw)
    if "shapes" in relayout_data:
        shapes_list = relayout_data["shapes"]
        if isinstance(shapes_list, list) and len(shapes_list) > 0:
            last = shapes_list[-1]
            if isinstance(last, dict):
                x0 = last.get("x0")
                y0 = last.get("y0")
                x1 = last.get("x1")
                y1 = last.get("y1")
                if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
                    print(f"[DEBUG] Shape extracted (full array): {last.get('type')}")
                    return {
                        "shape_type": last.get("type"),
                        "x0": x0, "y0": float(y0),
                        "x1": x1, "y1": float(y1),
                    }

    # FORMAT 2: Indexed keys like shapes[0].x0 (Plotly sends this on shape edit/drag)
    indices = set()
    for k in relayout_data.keys():
        if k.startswith("shapes[") and "]." in k:
            try:
                indices.add(int(k.split("[", 1)[1].split("]", 1)[0]))
            except Exception:
                pass
    if not indices:
        return None
    i = max(indices)
    def get(prop):
        return relayout_data.get(f"shapes[{i}].{prop}")
    x0, y0, x1, y1 = get("x0"), get("y0"), get("x1"), get("y1")
    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None
    print(f"[DEBUG] Shape extracted (indexed keys): idx={i}")
    return {"shape_type": get("type"), "x0": x0, "y0": float(y0), "x1": x1, "y1": float(y1)}


def is_line(shape) -> bool:
    t = shape.get("shape_type")
    return (t == "line") or (t is None)


def save_swing_to_csv(swing: dict, columns: list, path: str) -> str:
    try:
        file_exists = os.path.exists(path)
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if not file_exists:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            print(f"Created new CSV: {os.path.abspath(path)}")

        pd.DataFrame([swing]).to_csv(path, mode="a", header=False, index=False)
        abs_path = os.path.abspath(path)
        print(f"Saved swing: {swing['direction']} {swing['pips']:.1f} pips → {abs_path}")
        return (
            f"✅ Saved: {swing['direction']} {swing['pips']:.1f} pips, "
            f"{swing['bars']} bars → {abs_path}"
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR saving swing: {e}\n{tb}")
        return f"❌ Save failed: {e}"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"File not found: {CSV_PATH}")
        sys.exit(1)

    df_full = load_data(CSV_PATH)
    total_rows = len(df_full)
    print(f"Loaded {total_rows:,} rows total.")

    # Full datetime-indexed df for snapping (always uses RAW prices)
    df_indexed_full = df_full.set_index("Datetime")

    SWING_COLS = [
        "symbol", "direction",
        "entry_time", "entry_price",
        "exit_time", "exit_price",
        "pips", "bars", "note",
    ]

    init_end = min(DEFAULT_CHUNK, total_rows)
    fig = make_figure(df_full.iloc[0:init_end], 0, init_end, total_rows, uirev=0,
                      ema_period=DEFAULT_EMA_PERIOD,
                      show_raw=DEFAULT_SHOW_RAW,
                      show_smooth=DEFAULT_SHOW_SMOOTH)

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = f"{SYMBOL} Swing Label Tool"

    # ================================================================
    # LAYOUT
    # ================================================================
    app.layout = html.Div(
        style={"maxWidth": "1600px", "margin": "0 auto", "padding": "12px",
               "fontFamily": "Segoe UI, sans-serif"},
        children=[
            html.H3(f"{SYMBOL} — Manual Swing Labeler", style={"marginBottom": "4px"}),
            html.P("Draw a line from ENTRY → EXIT on the smoothed curve. "
                    "Labels always snap to exact raw prices.",
                    style={"marginTop": "0", "color": "#555"}),

            # === INDEX CONTROLS ===
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center",
                       "flexWrap": "wrap", "marginBottom": "10px",
                       "padding": "8px", "background": "#f0f4f8", "borderRadius": "6px"},
                children=[
                    html.Strong("Index range:"),
                    html.Span("Start:"),
                    dcc.Input(id="idx-start", type="number", value=0,
                              min=0, max=total_rows - 1, step=1, style={"width": "120px"}),
                    html.Span("End:"),
                    dcc.Input(id="idx-end", type="number", value=init_end,
                              min=1, max=total_rows, step=1, style={"width": "120px"}),
                    html.Button("Load slice", id="load-slice-btn", n_clicks=0,
                                style={"fontWeight": "bold"}),
                    html.Button(f"Next {DEFAULT_CHUNK // 1000}k →", id="next-chunk-btn", n_clicks=0),
                    html.Button(f"← Prev {DEFAULT_CHUNK // 1000}k", id="prev-chunk-btn", n_clicks=0),
                    html.Span(f"Total rows: {total_rows:,}",
                              style={"marginLeft": "16px", "color": "#666"}),
                ],
            ),

            # === SMOOTHING CONTROLS ===
            html.Div(
                style={"display": "flex", "gap": "16px", "alignItems": "center",
                       "flexWrap": "wrap", "marginBottom": "10px",
                       "padding": "8px", "background": "#f8f4f0", "borderRadius": "6px"},
                children=[
                    html.Strong("Display:"),
                    dcc.Checklist(
                        id="trace-toggles",
                        options=[
                            {"label": " Raw price", "value": "raw"},
                            {"label": " Smoothed (EMA)", "value": "smooth"},
                        ],
                        value=["raw", "smooth"] if (DEFAULT_SHOW_RAW and DEFAULT_SHOW_SMOOTH)
                              else ["raw"] if DEFAULT_SHOW_RAW
                              else ["smooth"] if DEFAULT_SHOW_SMOOTH
                              else [],
                        inline=True,
                        style={"display": "flex", "gap": "16px"},
                    ),

                    html.Span("EMA period:", style={"marginLeft": "12px"}),
                    dcc.Input(id="ema-period", type="number", value=DEFAULT_EMA_PERIOD,
                              min=2, max=5000, step=1, style={"width": "80px"}),
                    html.Button("Apply", id="apply-smooth-btn", n_clicks=0),

                    html.Span("|", style={"color": "#ccc", "margin": "0 8px"}),

                    html.Span("Min pips to save:"),
                    dcc.Input(id="min-pips-input", type="number", value=0,
                              min=0, step=1, style={"width": "70px"}),
                    html.Span("(0 = no filter)", style={"fontSize": "12px", "color": "#999"}),
                ],
            ),

            # === SWING READOUT ===
            html.Div(
                id="swing-readout",
                style={"fontSize": "15px", "fontWeight": "600", "color": "#333",
                       "marginBottom": "6px", "minHeight": "24px"},
                children=f"Draw a line from entry to exit. Warning threshold: {MIN_PIPS_WARN:.1f} pips.",
            ),

            # === SWING ACTION BUTTONS ===
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center",
                       "flexWrap": "wrap", "marginBottom": "8px"},
                children=[
                    html.Button("💾 Save swing", id="save-swing-btn", n_clicks=0,
                                style={"background": "#2a7", "color": "white", "border": "none",
                                       "padding": "6px 16px", "borderRadius": "4px", "cursor": "pointer",
                                       "fontSize": "15px"}),
                    html.Button("🗑 Clear drawing", id="clear-drawing-btn", n_clicks=0,
                                style={"padding": "6px 16px", "borderRadius": "4px", "cursor": "pointer",
                                       "fontSize": "15px"}),
                    html.Span("Note (optional):"),
                    dcc.Input(id="note-input", type="text", value="", debounce=True,
                              style={"width": "320px"}),
                ],
            ),

            # === STATUS LINE ===
            html.Div(id="status",
                     style={"fontSize": "14px", "color": "#444", "marginBottom": "4px",
                            "minHeight": "20px"}),

            # === CHART ===
            dcc.Graph(
                id="chart", figure=fig,
                config={"scrollZoom": True, "displaylogo": False, "responsive": True,
                        "modeBarButtonsToAdd": ["drawline", "eraseshape"]},
                style={"height": "920px"},
            ),

            # === STORES ===
            dcc.Store(id="last-swing-store"),
            dcc.Store(id="uirev-store", data=0),
        ],
    )

    # ================================================================
    # CALLBACK 1 — Drawing: measure swing on line draw
    #   SOLE owner of: swing-readout, last-swing-store
    # ================================================================
    @app.callback(
        Output("swing-readout", "children"),
        Output("last-swing-store", "data"),
        Input("chart", "relayoutData"),
        prevent_initial_call=True,
    )
    def on_draw(relayout_data):
        shape = extract_latest_shape_update(relayout_data)
        if not shape:
            print("[DEBUG on_draw] No shape extracted — returning no_update")
            return no_update, no_update
        if not is_line(shape):
            print(f"[DEBUG on_draw] Shape is not a line: {shape}")
            return "Only line drawings are used. Use the line tool.", None

        t0 = parse_ts(shape["x0"])
        t1 = parse_ts(shape["x1"])
        if t0 is None or t1 is None:
            return "Could not parse timestamps from the drawn line.", None

        # ALWAYS snap to RAW data — smoothing is visual only
        snap_t_entry, snap_p_entry = nearest_point(df_indexed_full, t0)
        snap_t_exit, snap_p_exit = nearest_point(df_indexed_full, t1)
        if snap_t_entry is None or snap_t_exit is None:
            return "Could not snap endpoints to nearest minute bars.", None

        direction = ("LONG" if snap_p_exit > snap_p_entry
                     else "SHORT" if snap_p_exit < snap_p_entry
                     else "FLAT")
        pips = pips_from_prices(snap_p_entry, snap_p_exit)
        bars = compute_bars(df_indexed_full, snap_t_entry, snap_t_exit)

        warn = f" ⚠️ < {MIN_PIPS_WARN:.1f} pips" if pips < MIN_PIPS_WARN else ""
        msg = (
            f"{direction}{warn} | "
            f"Entry: {snap_t_entry} @ {snap_p_entry:.5f}  →  "
            f"Exit: {snap_t_exit} @ {snap_p_exit:.5f} | "
            f"Pips: {pips:.1f} | Bars: {bars}"
        )
        swing = {
            "symbol": SYMBOL,
            "direction": direction,
            "entry_time": str(snap_t_entry),
            "entry_price": float(snap_p_entry),
            "exit_time": str(snap_t_exit),
            "exit_price": float(snap_p_exit),
            "pips": float(pips),
            "bars": int(bars),
            "note": "",
        }
        print(f"[DEBUG on_draw] Swing stored: {direction} {pips:.1f} pips")
        return msg, swing

    # ================================================================
    # CALLBACK 2 — ALL buttons: save, clear, load, next, prev, apply-smooth
    #   SOLE owner of: status, chart.figure, uirev-store, idx-start, idx-end
    # ================================================================
    @app.callback(
        Output("status", "children"),
        Output("chart", "figure"),
        Output("uirev-store", "data"),
        Output("idx-start", "value"),
        Output("idx-end", "value"),
        Input("save-swing-btn", "n_clicks"),
        Input("clear-drawing-btn", "n_clicks"),
        Input("load-slice-btn", "n_clicks"),
        Input("next-chunk-btn", "n_clicks"),
        Input("prev-chunk-btn", "n_clicks"),
        Input("apply-smooth-btn", "n_clicks"),
        State("last-swing-store", "data"),
        State("note-input", "value"),
        State("chart", "figure"),
        State("uirev-store", "data"),
        State("idx-start", "value"),
        State("idx-end", "value"),
        State("ema-period", "value"),
        State("trace-toggles", "value"),
        State("min-pips-input", "value"),
        prevent_initial_call=True,
    )
    def handle_buttons(save_n, clear_n, load_n, next_n, prev_n, smooth_n,
                       swing, note, fig_json, uirev,
                       start_val, end_val,
                       ema_period, trace_toggles, min_pips_filter):
        trig = ctx.triggered_id
        if not trig:
            return no_update, no_update, no_update, no_update, no_update

        start_val = int(start_val or 0)
        end_val = int(end_val or DEFAULT_CHUNK)
        uirev = int(uirev or 0)
        ema_period = int(ema_period or DEFAULT_EMA_PERIOD)
        trace_toggles = trace_toggles or []
        show_raw = "raw" in trace_toggles
        show_smooth = "smooth" in trace_toggles
        min_pips_filter = float(min_pips_filter or 0)

        def cleared_figure(new_uirev):
            if not fig_json or not isinstance(fig_json, dict):
                return no_update
            updated = dict(fig_json)
            updated.setdefault("layout", {})
            updated["layout"]["shapes"] = []
            updated["layout"]["uirevision"] = new_uirev
            return updated

        def fresh_figure(s, e, new_uirev):
            s = max(0, min(s, total_rows - 1))
            e = max(s + 1, min(e, total_rows))
            df_slice = df_full.iloc[s:e]
            return make_figure(df_slice, s, e, total_rows, new_uirev,
                               ema_period=ema_period,
                               show_raw=show_raw,
                               show_smooth=show_smooth), s, e

        # ========== SAVE ==========
        if trig == "save-swing-btn":
            print(f"[DEBUG save] Button clicked. swing store = {swing}")
            if not swing:
                print("[DEBUG save] Store is empty/None — nothing to save")
                return ("⚠️ Nothing to save — draw a line first.",
                        no_update, no_update, no_update, no_update)

            swing = dict(swing)
            swing["note"] = (note or "").strip()

            # Min-pips filter: reject if below threshold
            if min_pips_filter > 0 and swing["pips"] < min_pips_filter:
                return (f"⚠️ Rejected: {swing['pips']:.1f} pips is below "
                        f"min filter ({min_pips_filter:.0f} pips). Not saved.",
                        no_update, no_update, no_update, no_update)

            status_msg = save_swing_to_csv(swing, SWING_COLS, SWINGS_OUT)
            new_uirev = uirev + 1
            return status_msg, cleared_figure(new_uirev), new_uirev, no_update, no_update

        # ========== CLEAR ==========
        if trig == "clear-drawing-btn":
            new_uirev = uirev + 1
            return "Cleared.", cleared_figure(new_uirev), new_uirev, no_update, no_update

        # ========== APPLY SMOOTHING (rebuild figure with current slice) ==========
        if trig == "apply-smooth-btn":
            new_uirev = uirev + 1
            new_fig, s, e = fresh_figure(start_val, end_val, new_uirev)
            label = f"EMA({ema_period})" if show_smooth else "no smoothing"
            status_msg = f"Applied: {label}, raw={'on' if show_raw else 'off'}"
            return status_msg, new_fig, new_uirev, s, e

        # ========== LOAD / NEXT / PREV ==========
        if trig == "next-chunk-btn":
            chunk = end_val - start_val
            if chunk <= 0:
                chunk = DEFAULT_CHUNK
            start_val = min(end_val, total_rows - 1)
            end_val = min(start_val + chunk, total_rows)

        elif trig == "prev-chunk-btn":
            chunk = end_val - start_val
            if chunk <= 0:
                chunk = DEFAULT_CHUNK
            start_val = max(start_val - chunk, 0)
            end_val = start_val + chunk

        new_uirev = uirev + 1
        new_fig, s, e = fresh_figure(start_val, end_val, new_uirev)
        status_msg = f"Loaded rows {s:,} – {e:,} ({e - s:,} points)"
        return status_msg, new_fig, new_uirev, s, e

    print(f"\nStarting server — {total_rows:,} rows loaded.")
    print(f"Default view: rows 0 to {min(DEFAULT_CHUNK, total_rows):,}")
    print(f"Swings will save to: {os.path.abspath(SWINGS_OUT)}\n")
    app.run(debug=False)


if __name__ == "__main__":
    main()