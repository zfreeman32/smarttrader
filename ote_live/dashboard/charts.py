from __future__ import annotations

import math
import re

import pandas as pd

from ote_live.storage import SignalAuditTrail

CHART_BACKGROUND_COLOR = "#111a2b"
CHART_PLOT_COLOR = "#0f1726"
CHART_GRID_COLOR = "#23314c"
CHART_TEXT_COLOR = "#e5edf7"
CHART_MUTED_TEXT_COLOR = "#9fb0c5"
CHART_BORDER_COLOR = "#2a3a58"
ICT_FVG_DISPLAY_BAR_CAP = 15


def build_price_signal_figure(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    title: str = "Live Price And Signals",
):
    go = _plotly_go()

    fig = go.Figure()
    if not bars.empty:
        fig.add_trace(
            go.Candlestick(
                x=bars["timestamp"],
                open=bars["open"],
                high=bars["high"],
                low=bars["low"],
                close=bars["close"],
                name="Price",
            )
        )
        _add_contract_roll_markers(fig, bars)

    emitted_signals = _aggregate_emitted_signals(signals)
    if not emitted_signals.empty:
        for direction, group in emitted_signals.groupby("direction", dropna=False):
            resolved_direction = str(direction or "unknown")
            price_column = "bar_low" if resolved_direction == "long" else "bar_high"
            marker_symbol = "triangle-up" if resolved_direction == "long" else "triangle-down"
            marker_color = "#22c55e" if resolved_direction == "long" else "#ef4444"
            scale = 0.9998 if resolved_direction == "long" else 1.0002
            y_values = group[price_column].fillna(group["bar_close"]) * scale
            fig.add_trace(
                go.Scatter(
                    x=group["timestamp"],
                    y=y_values,
                    mode="markers",
                    marker=dict(
                        size=13,
                        symbol=marker_symbol,
                        color=marker_color,
                        opacity=0.95,
                        line=dict(width=1),
                    ),
                    name=f"{resolved_direction} emit",
                    text=[
                        _aggregated_signal_hover_text(row)
                        for row in group.itertuples(index=False)
                    ],
                    hoverinfo="text",
                )
            )

    layout_kwargs = dict(
        template="plotly_dark",
        title=title,
        height=620,
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
        showlegend=False,
        uirevision="ote-live-price",
    )
    layout_kwargs.update(_price_axis_layout_kwargs(bars))
    fig.update_layout(**layout_kwargs)
    _apply_dark_chart_theme(fig)
    return fig


def build_frvp_price_figure(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    runtime_state: dict | None,
    title: str = "FRVP Price And Live State",
):
    go = _plotly_go()

    fig = go.Figure()
    if not bars.empty:
        fig.add_trace(
            go.Candlestick(
                x=bars["timestamp"],
                open=bars["open"],
                high=bars["high"],
                low=bars["low"],
                close=bars["close"],
                name="Price",
            )
        )
        _add_contract_roll_markers(fig, bars)

    runtime_state = _visible_runtime_state(runtime_state, bars)
    _add_frvp_overlay_traces(fig, bars=bars, runtime_state=runtime_state)
    _add_emitted_signal_traces(fig, signals)
    _add_frvp_setup_traces(fig, bars=bars, runtime_state=runtime_state)

    layout_kwargs = dict(
        template="plotly_dark",
        title=title,
        height=640,
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
        showlegend=True,
        uirevision="frvp-live-price",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    layout_kwargs.update(_price_axis_layout_kwargs(bars))
    fig.update_layout(**layout_kwargs)
    _apply_dark_chart_theme(fig)
    return fig


def build_ict_price_figure(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    runtime_state: dict | None,
    title: str = "ICT Price And Live State",
):
    go = _plotly_go()

    fig = go.Figure()
    if not bars.empty:
        fig.add_trace(
            go.Candlestick(
                x=bars["timestamp"],
                open=bars["open"],
                high=bars["high"],
                low=bars["low"],
                close=bars["close"],
                name="Price",
            )
        )
        _add_contract_roll_markers(fig, bars)

    runtime_state = _visible_runtime_state(runtime_state, bars)
    _add_ict_overlay_traces(fig, bars=bars, runtime_state=runtime_state)
    _add_emitted_signal_traces(fig, signals)
    _add_ict_setup_traces(fig, bars=bars, runtime_state=runtime_state)

    layout_kwargs = dict(
        template="plotly_dark",
        title=title,
        height=660,
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
        showlegend=True,
        uirevision="ict-live-price-v2",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    layout_kwargs.update(_price_axis_layout_kwargs(bars))
    fig.update_layout(**layout_kwargs)
    _apply_dark_chart_theme(fig)
    return fig


def _add_contract_roll_markers(fig, bars: pd.DataFrame) -> None:
    if bars.empty:
        return
    identity_column = None
    for candidate in ("instrument_id", "contract_symbol"):
        if candidate in bars.columns and bars[candidate].notna().any():
            identity_column = candidate
            break
    if identity_column is None:
        return
    identity = bars[identity_column].astype("string")
    changed = identity.notna() & identity.shift().notna() & identity.ne(identity.shift())
    if not changed.any():
        return
    shapes = []
    annotations = []
    for index in bars.index[changed]:
        timestamp = bars.loc[index, "timestamp"]
        contract = bars.loc[index, identity_column]
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": timestamp,
                "x1": timestamp,
                "y0": 0,
                "y1": 1,
                "line": {"color": "#f59e0b", "width": 1.5, "dash": "dash"},
            }
        )
        annotations.append(
            {
                "x": timestamp,
                "y": 1,
                "xref": "x",
                "yref": "paper",
                "text": f"Roll → {contract}",
                "showarrow": False,
                "yanchor": "bottom",
                "font": {"color": "#f59e0b", "size": 10},
            }
        )
    fig.update_layout(shapes=shapes, annotations=annotations)


def build_confidence_figure(
    confidence: pd.DataFrame,
    *,
    title: str = "Model Confidence",
    line_color: str = "#0d6efd",
    threshold_color: str = "#fd7e14",
    xaxis_range: tuple[object, object] | None = None,
    fallback_threshold: float | None = None,
    setup_events: list[dict] | pd.DataFrame | None = None,
):
    go = _plotly_go()

    fig = go.Figure()
    threshold_plotted = False
    if not confidence.empty:
        threshold_series = confidence["display_threshold"] if "display_threshold" in confidence.columns else confidence["threshold_applied"]
        fig.add_trace(
            go.Scatter(
                x=confidence["timestamp"],
                y=confidence["calibrated_probability"],
                mode="lines+markers",
                line=dict(color=line_color, width=2),
                marker=dict(size=6),
                name="Calibrated probability",
                text=[
                    _confidence_hover_text(row)
                    for row in confidence.itertuples(index=False)
                ],
                hoverinfo="text",
            )
        )
        if threshold_series.notna().any():
            threshold_plotted = True
            fig.add_trace(
                go.Scatter(
                    x=confidence["timestamp"],
                    y=threshold_series,
                    mode="lines",
                    line=dict(color=threshold_color, width=1.5, dash="dash"),
                    name="Active threshold" if "display_threshold" in confidence.columns else "Threshold",
                    hovertemplate="Threshold: %{y:.4f}<extra></extra>",
                )
            )
        emit_points = confidence.loc[confidence["decision"] == "emit"].copy()
        if not emit_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=emit_points["timestamp"],
                    y=emit_points["calibrated_probability"],
                    mode="markers",
                    marker=dict(size=10, symbol="diamond", color="#198754"),
                    name="Emit",
                    text=[
                        _confidence_hover_text(row)
                        for row in emit_points.itertuples(index=False)
                    ],
                    hoverinfo="text",
                )
            )

    layout_kwargs = dict(
        template="plotly_dark",
        title=title,
        height=320,
        margin=dict(l=30, r=20, t=50, b=30),
        hovermode="x unified",
        yaxis=dict(range=[0.0, 1.0]),
        showlegend=False,
    )
    if xaxis_range is not None:
        layout_kwargs["xaxis"] = {"range": list(xaxis_range)}
    if not threshold_plotted and fallback_threshold is not None:
        resolved_threshold = float(fallback_threshold)
        if not pd.isna(resolved_threshold):
            layout_kwargs["shapes"] = [
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": resolved_threshold,
                    "y1": resolved_threshold,
                    "line": {
                        "color": threshold_color,
                        "width": 1.5,
                        "dash": "dash",
                    },
                }
            ]
            layout_kwargs["annotations"] = [
                {
                    "xref": "paper",
                    "x": 1,
                    "xanchor": "right",
                    "y": resolved_threshold,
                    "yanchor": "bottom",
                    "text": (
                        f"Configured threshold: {resolved_threshold:.4f}"
                        "<br>Awaiting first persisted probability"
                    ),
                    "showarrow": False,
                    "font": {"color": threshold_color, "size": 11},
                }
            ]
    fig.update_layout(**layout_kwargs)
    _add_setup_event_markers(fig, setup_events, xaxis_range=xaxis_range)
    _apply_dark_chart_theme(fig)
    return fig


def build_markout_figure(
    markouts: pd.DataFrame,
    *,
    title: str = "Rolling Paper Markouts",
):
    go = _plotly_go()

    fig = go.Figure()
    if not markouts.empty:
        complete = markouts.loc[markouts["status"] == "complete"].copy()
        if not complete.empty:
            complete["cumulative_markout_pips"] = complete["markout_pips"].cumsum()
            fig.add_trace(
                go.Bar(
                    x=complete["timestamp"],
                    y=complete["markout_pips"],
                    name="Per-signal markout (pips)",
                    marker_color="#20c997",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=complete["timestamp"],
                    y=complete["cumulative_markout_pips"],
                    mode="lines+markers",
                    name="Cumulative markout (pips)",
                    line=dict(color="#0b7285", width=2),
                    yaxis="y2",
                )
            )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=320,
        margin=dict(l=30, r=20, t=50, b=30),
        hovermode="x unified",
        yaxis=dict(title="Per signal"),
        yaxis2=dict(title="Cumulative", overlaying="y", side="right"),
    )
    _apply_dark_chart_theme(fig)
    return fig


def build_health_figure(
    summary,
    *,
    title: str = "System Health",
):
    go = _plotly_go()

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Feed lag (sec)", "Open gaps", "Alerts 24h", "Media 24h"],
                y=[
                    float(summary.heartbeat_lag_seconds or 0.0),
                    int(summary.unresolved_gap_count),
                    int(summary.alerts_sent_last_24h),
                    int(summary.media_artifacts_last_24h),
                ],
                marker_color=["#0d6efd", "#dc3545", "#198754", "#6610f2"],
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=280,
        margin=dict(l=30, r=20, t=50, b=30),
        showlegend=False,
    )
    _apply_dark_chart_theme(fig)
    return fig


def build_audit_trail_figure(
    audit_trail: SignalAuditTrail,
    *,
    title: str | None = None,
):
    bars = pd.DataFrame(
        [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in audit_trail.bars
        ]
    )
    signal = audit_trail.signal
    matching_bar = next((bar for bar in audit_trail.bars if bar.timestamp == signal.timestamp), None)
    signals = pd.DataFrame(
        [
            {
                "model_id": signal.model_id,
                "timestamp": signal.timestamp,
                "direction": signal.direction,
                "decision": signal.decision,
                "probability": signal.probability,
                "raw_score": audit_trail.prediction.raw_score,
                "threshold": signal.threshold,
                "regime": signal.regime,
                "bar_low": matching_bar.low if matching_bar is not None else None,
                "bar_high": matching_bar.high if matching_bar is not None else None,
                "bar_close": matching_bar.close if matching_bar is not None else None,
            }
        ]
    )
    return build_price_signal_figure(
        bars,
        signals,
        title=title or f"{signal.model_id} signal {audit_trail.signal_decision_id}",
    )


def _plotly_go():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Dashboard chart rendering requires the 'plotly' package."
        ) from exc
    return go


def _apply_dark_chart_theme(fig) -> None:
    fig.update_layout(
        paper_bgcolor=CHART_BACKGROUND_COLOR,
        plot_bgcolor=CHART_PLOT_COLOR,
        font=dict(color=CHART_TEXT_COLOR),
        title_font=dict(color=CHART_TEXT_COLOR),
        hoverlabel=dict(
            bgcolor=CHART_BACKGROUND_COLOR,
            bordercolor=CHART_BORDER_COLOR,
            font=dict(color=CHART_TEXT_COLOR),
        ),
        legend=dict(
            bgcolor="rgba(0, 0, 0, 0)",
            font=dict(color=CHART_MUTED_TEXT_COLOR),
        ),
    )
    fig.update_xaxes(
        gridcolor=CHART_GRID_COLOR,
        linecolor=CHART_BORDER_COLOR,
        zerolinecolor=CHART_GRID_COLOR,
        tickfont=dict(color=CHART_MUTED_TEXT_COLOR),
        title_font=dict(color=CHART_MUTED_TEXT_COLOR),
    )
    fig.update_yaxes(
        gridcolor=CHART_GRID_COLOR,
        linecolor=CHART_BORDER_COLOR,
        zerolinecolor=CHART_GRID_COLOR,
        tickfont=dict(color=CHART_MUTED_TEXT_COLOR),
        title_font=dict(color=CHART_MUTED_TEXT_COLOR),
    )


def _signal_hover_text(row) -> str:
    threshold = "n/a" if row.threshold is None else f"{row.threshold:.4f}"
    lines = [
        f"Model: {getattr(row, 'model_id', 'unknown')}",
        f"{row.direction} {row.decision}",
        f"Probability: {_format_probability(getattr(row, 'probability', None))}",
        f"Threshold: {threshold}",
        f"Regime: {row.regime or 'unknown'}",
    ]
    raw_score = getattr(row, "raw_score", None)
    if raw_score is not None:
        lines.insert(3, f"Raw score: {_format_probability(raw_score)}")
    return "<br>".join(lines)


def _setup_hover_text(row) -> str:
    confidence = _format_probability(getattr(row, "confidence", None))
    lines = [
        f"{getattr(row, 'label', 'setup')}",
        f"Rule confidence: {confidence}",
        f"Timestamp: {getattr(row, 'timestamp_utc', 'unknown')}",
    ]
    for label, field_name in (
        ("Anchor", "anchor_level"),
        ("Entry", "entry_price"),
        ("Stop", "stop_reference"),
        ("Target", "target_reference"),
        ("Reference", "reference_level"),
        ("CE", "ce_price"),
    ):
        value = getattr(row, field_name, None)
        if value is None or pd.isna(value):
            continue
        lines.append(f"{label}: {_format_probability(float(value))}")
    for label, field_name in (
        ("Reference Type", "reference_level_type"),
        ("Sweep", "sweep_type"),
        ("HTF Context", "htf_context"),
    ):
        value = getattr(row, field_name, None)
        if value in {None, ""}:
            continue
        lines.append(f"{label}: {value}")
    return "<br>".join(lines)


def _aggregated_signal_hover_text(row) -> str:
    model_ids = [item for item in str(getattr(row, "model_ids", "")).split(",") if item]
    lines = [
        f"{row.direction} emit",
        f"Models: {', '.join(model_ids) if model_ids else 'unknown'}",
        f"Emitted models: {int(getattr(row, 'emitted_model_count', 0) or 0)}",
    ]
    if getattr(row, "regimes", ""):
        lines.append(f"Regimes: {row.regimes}")
    return "<br>".join(lines)


def _confidence_hover_text(row) -> str:
    decision = row.decision or "hold/unpersisted"
    active_threshold = getattr(row, "dashboard_active_threshold", None)
    persisted_threshold = getattr(row, "threshold_applied", None)

    lines = [
        f"Model: {getattr(row, 'model_id', 'unknown')}",
        f"Model probability: {_format_probability(row.calibrated_probability)}",
        f"Raw score: {_format_probability(getattr(row, 'raw_score', None))}",
    ]

    if active_threshold is not None and not pd.isna(active_threshold):
        lines.append(f"Active threshold: {float(active_threshold):.4f}")
    if persisted_threshold is not None and not pd.isna(persisted_threshold):
        if active_threshold is None or pd.isna(active_threshold):
            lines.append(f"Threshold: {float(persisted_threshold):.4f}")
        elif abs(float(active_threshold) - float(persisted_threshold)) > 1e-9:
            lines.append(f"Persisted threshold: {float(persisted_threshold):.4f}")

    if (
        (active_threshold is None or pd.isna(active_threshold))
        and (persisted_threshold is None or pd.isna(persisted_threshold))
    ):
        lines.append("Threshold: n/a")

    lines.append(f"Decision: {decision}")
    lines.append(f"Regime: {row.regime or 'unknown'}")
    recorded = getattr(row, "prediction_recorded_at_utc", None)
    lines.append(f"Prediction recorded: {recorded or 'unknown'}")
    if recorded and getattr(row, "timeframe", None):
        from ote_live.ingestion.base import timeframe_to_timedelta
        delay = (pd.Timestamp(recorded) - pd.Timestamp(row.timestamp)
                 - timeframe_to_timedelta(row.timeframe)).total_seconds()
        lines.append(f"Decision delay after bar close: {delay:.1f}s")
    return "<br>".join(lines)


def _add_setup_event_markers(
    fig,
    setup_events: list[dict] | pd.DataFrame | None,
    *,
    xaxis_range: tuple[object, object] | None,
) -> None:
    events = _visible_setup_event_frame(setup_events, xaxis_range=xaxis_range)
    if events.empty:
        return

    go = _plotly_go()
    shapes = _layout_items(fig, "shapes")
    annotations = _layout_items(fig, "annotations")
    hover_x = []
    hover_y = []
    hover_text = []

    for row in events.itertuples(index=False):
        color = _setup_marker_color(row)
        dash = "solid" if _coerce_setup_side(getattr(row, "setup_side", 0)) > 0 else "dot"
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": row.timestamp,
                "x1": row.timestamp,
                "y0": 0,
                "y1": 1,
                "line": {"color": color, "width": 1.4, "dash": dash},
            }
        )
        annotations.append(
            {
                "x": row.timestamp,
                "y": 1,
                "xref": "x",
                "yref": "paper",
                "text": _setup_marker_annotation(row),
                "showarrow": False,
                "yanchor": "bottom",
                "font": {"color": color, "size": 9},
            }
        )
        hover_x.append(row.timestamp)
        hover_y.append(1.0)
        hover_text.append(_setup_event_hover_text(row))

    fig.update_layout(shapes=shapes, annotations=annotations)
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(size=8, symbol="line-ns-open", color=[_setup_marker_color(row) for row in events.itertuples(index=False)]),
            name="Setup fired",
            text=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )


def _visible_setup_event_frame(
    setup_events: list[dict] | pd.DataFrame | None,
    *,
    xaxis_range: tuple[object, object] | None,
) -> pd.DataFrame:
    if setup_events is None:
        return pd.DataFrame()
    frame = setup_events.copy() if isinstance(setup_events, pd.DataFrame) else pd.DataFrame(setup_events)
    if frame.empty or "timestamp_utc" not in frame.columns:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).copy()
    if frame.empty:
        return frame
    if xaxis_range is not None:
        start = _coerce_timestamp(xaxis_range[0])
        end = _coerce_timestamp(xaxis_range[1])
        if start is not None:
            frame = frame.loc[frame["timestamp"] >= start]
        if end is not None:
            frame = frame.loc[frame["timestamp"] <= end]
    if frame.empty:
        return frame
    if "setup_side" not in frame.columns:
        frame["setup_side"] = 0
    frame["_marker_key"] = [
        _setup_marker_identity(row)
        for row in frame.itertuples(index=False)
    ]
    return (
        frame.sort_values("timestamp")
        .drop_duplicates(subset=["_marker_key"], keep="last")
        .drop(columns=["_marker_key"])
        .reset_index(drop=True)
    )


def _setup_marker_identity(row) -> str:
    return "|".join(
        (
            str(getattr(row, "timestamp_utc", "")),
            str(getattr(row, "label", "") or getattr(row, "setup_type", "")),
            str(_coerce_setup_side(getattr(row, "setup_side", 0))),
        )
    )


def _setup_marker_color(row) -> str:
    key = _setup_marker_family(row)
    palette = (
        "#22c55e",
        "#38bdf8",
        "#f59e0b",
        "#c084fc",
        "#14b8a6",
        "#f97316",
        "#eab308",
        "#60a5fa",
    )
    index = sum(ord(char) for char in key) % len(palette)
    return palette[index]


def _setup_marker_family(row) -> str:
    raw = str(getattr(row, "setup_type", "") or getattr(row, "label", "") or "setup").strip().lower()
    raw = raw.replace(" long", "").replace(" short", "")
    return raw or "setup"


def _setup_marker_annotation(row) -> str:
    label = str(getattr(row, "label", "") or getattr(row, "setup_type", "") or "Setup")
    setup_type = getattr(row, "setup_type", None)
    if setup_type not in {None, ""} and not pd.isna(setup_type):
        setup_text = str(setup_type).strip()
        return setup_text if setup_text.upper().startswith("S") else f"S{setup_text}"
    match = re.match(r"^\s*(S\d+)\b", label, flags=re.IGNORECASE)
    if match is not None:
        return match.group(1).upper()
    return label.split()[0] if label else "Setup"


def _setup_event_hover_text(row) -> str:
    side = _coerce_setup_side(getattr(row, "setup_side", 0))
    direction = "long" if side > 0 else "short" if side < 0 else "unknown"
    lines = [
        f"Setup: {getattr(row, 'label', None) or getattr(row, 'setup_type', 'setup')}",
        f"Direction: {direction}",
        f"Timestamp: {getattr(row, 'timestamp_utc', 'unknown')}",
    ]
    confidence = getattr(row, "confidence", None)
    if confidence is not None and not pd.isna(confidence):
        lines.append(f"Rule confidence: {_format_probability(confidence)}")
    return "<br>".join(lines)


def _layout_items(fig, name: str) -> list:
    layout = getattr(fig, "layout", None)
    if isinstance(layout, dict):
        return list(layout.get(name, []) or [])
    value = getattr(layout, name, None)
    return list(value or [])


def _format_probability(value: float | None) -> str:
    if value is None:
        return "n/a"
    resolved = float(value)
    magnitude = abs(resolved)
    if magnitude >= 0.01:
        return f"{resolved:.4f}"
    if magnitude >= 0.001:
        return f"{resolved:.6f}"
    return f"{resolved:.6e}"


def _aggregate_emitted_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "direction",
                "bar_low",
                "bar_high",
                "bar_close",
                "model_ids",
                "emitted_model_count",
                "regimes",
            ]
        )

    emitted = signals.loc[
        (signals["decision"] == "emit")
        & (signals["direction"].isin(["long", "short"]))
    ].copy()
    if emitted.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "direction",
                "bar_low",
                "bar_high",
                "bar_close",
                "model_ids",
                "emitted_model_count",
                "regimes",
            ]
        )

    aggregated_rows: list[dict[str, object]] = []
    grouped = emitted.groupby(["timestamp", "direction"], dropna=False, sort=True)
    for (timestamp, direction), group in grouped:
        model_ids = sorted(
            {
                str(model_id)
                for model_id in group["model_id"].dropna().astype(str).tolist()
                if str(model_id)
            }
        )
        regimes = sorted(
            {
                str(regime)
                for regime in group["regime"].dropna().astype(str).tolist()
                if str(regime)
            }
        )
        aggregated_rows.append(
            {
                "timestamp": timestamp,
                "direction": direction,
                "bar_low": group["bar_low"].dropna().min() if group["bar_low"].notna().any() else None,
                "bar_high": group["bar_high"].dropna().max() if group["bar_high"].notna().any() else None,
                "bar_close": group["bar_close"].dropna().iloc[-1] if group["bar_close"].notna().any() else None,
                "model_ids": ",".join(model_ids),
                "emitted_model_count": len(model_ids),
                "regimes": ", ".join(regimes),
            }
        )

    return pd.DataFrame(aggregated_rows).sort_values("timestamp").reset_index(drop=True)


def _add_emitted_signal_traces(fig, signals: pd.DataFrame) -> None:
    go = _plotly_go()
    emitted_signals = _aggregate_emitted_signals(signals)
    if emitted_signals.empty:
        return
    for direction, group in emitted_signals.groupby("direction", dropna=False):
        resolved_direction = str(direction or "unknown")
        price_column = "bar_low" if resolved_direction == "long" else "bar_high"
        marker_symbol = "triangle-up" if resolved_direction == "long" else "triangle-down"
        marker_color = "#22c55e" if resolved_direction == "long" else "#ef4444"
        scale = 0.9998 if resolved_direction == "long" else 1.0002
        y_values = group[price_column].fillna(group["bar_close"]) * scale
        fig.add_trace(
            go.Scatter(
                x=group["timestamp"],
                y=y_values,
                mode="markers",
                marker=dict(
                    size=13,
                    symbol=marker_symbol,
                    color=marker_color,
                    opacity=0.95,
                    line=dict(width=1),
                ),
                name=f"{resolved_direction} emit",
                text=[
                    _aggregated_signal_hover_text(row)
                    for row in group.itertuples(index=False)
                ],
                hoverinfo="text",
            )
        )


def _add_frvp_overlay_traces(fig, *, bars: pd.DataFrame, runtime_state: dict | None) -> None:
    if runtime_state is None or not runtime_state.get("levels"):
        return
    if bars.empty:
        return
    go = _plotly_go()
    x_values = [bars["timestamp"].iloc[0], bars["timestamp"].iloc[-1]]
    for name, label, color, dash in (
        ("poc", "POC", "#fbbf24", "solid"),
        ("vah", "VAH", "#38bdf8", "dash"),
        ("val", "VAL", "#38bdf8", "dash"),
        ("ib_high", "IB High", "#34d399", "dot"),
        ("ib_low", "IB Low", "#f87171", "dot"),
        ("naked_vpoc_above", "Naked VPOC +", "#f59e0b", "dashdot"),
        ("naked_vpoc_below", "Naked VPOC -", "#fb923c", "dashdot"),
    ):
        level_value = runtime_state["levels"].get(name)
        if level_value is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[level_value, level_value],
                mode="lines",
                line=dict(color=color, width=1.5, dash=dash),
                name=label,
                hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
            )
        )


def _add_frvp_setup_traces(fig, *, bars: pd.DataFrame, runtime_state: dict | None) -> None:
    if runtime_state is None:
        return
    setups = list(runtime_state.get("recent_setups") or [])
    if not setups or bars.empty:
        return
    go = _plotly_go()
    merged = _visible_setup_frame(bars, setups)
    if merged.empty:
        return
    for side_value, group in merged.groupby("setup_side", dropna=False):
        resolved_side = _coerce_setup_side(side_value)
        if resolved_side == 0:
            continue
        is_long = resolved_side > 0
        price_column = "low" if is_long else "high"
        base_prices = _valid_price_series(group[price_column].fillna(group["close"]))
        plot_group = group.loc[base_prices.notna()]
        if plot_group.empty:
            continue
        y_values = base_prices.loc[plot_group.index] * (0.9996 if is_long else 1.0004)
        fig.add_trace(
            go.Scatter(
                x=plot_group["timestamp"],
                y=y_values,
                mode="markers+text",
                marker=dict(
                    size=12,
                    symbol="triangle-up" if is_long else "triangle-down",
                    color="#22c55e" if is_long else "#ef4444",
                    line=dict(width=1),
                ),
                text=_setup_labels(plot_group),
                textposition="top center" if is_long else "bottom center",
                name="FRVP long setup" if is_long else "FRVP short setup",
                textfont=dict(size=10, color=CHART_TEXT_COLOR),
                hoverinfo="text",
                hovertext=[
                    _setup_hover_text(row)
                    for row in plot_group.itertuples(index=False)
                ],
            )
        )


def _add_ict_overlay_traces(fig, *, bars: pd.DataFrame, runtime_state: dict | None) -> None:
    if runtime_state is None or bars.empty:
        return
    go = _plotly_go()
    x_values = [bars["timestamp"].iloc[0], bars["timestamp"].iloc[-1]]
    levels = dict(runtime_state.get("levels") or {})
    for name, label, color, dash in (
        ("prior_rth_high", "PDH", "#38bdf8", "dash"),
        ("prior_rth_low", "PDL", "#38bdf8", "dash"),
        ("overnight_high", "ONH", "#f59e0b", "dot"),
        ("overnight_low", "ONL", "#f59e0b", "dot"),
        ("ib_high", "IB High", "#34d399", "dashdot"),
        ("ib_low", "IB Low", "#f87171", "dashdot"),
        ("prior_week_high", "PWH", "#8b5cf6", "dot"),
        ("prior_week_low", "PWL", "#8b5cf6", "dot"),
        ("midnight_open", "Midnight Open", "#94a3b8", "dot"),
        ("open_0830", "08:30 Open", "#c084fc", "dash"),
        ("session_vwap", "Session VWAP", "#fbbf24", "solid"),
        ("dol_up", "DOL Up", "#22c55e", "dash"),
        ("dol_down", "DOL Down", "#ef4444", "dash"),
    ):
        level_value = _coerce_price_overlay(levels.get(name))
        if level_value is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[level_value, level_value],
                mode="lines",
                line=dict(color=color, width=1.3, dash=dash),
                name=label,
                hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
            )
        )

    zones = dict(runtime_state.get("zones") or {})
    for name, label, line_color, fill_color in (
        ("bull_order_block", "Bull OB", "#14b8a6", "rgba(20, 184, 166, 0.12)"),
        ("bear_order_block", "Bear OB", "#f97316", "rgba(249, 115, 22, 0.12)"),
    ):
        zone = zones.get(name)
        if not isinstance(zone, dict):
            continue
        lower = _coerce_price_overlay(zone.get("lower"))
        upper = _coerce_price_overlay(zone.get("upper"))
        if lower is None or upper is None or lower > upper:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[lower, lower],
                mode="lines",
                line=dict(color=line_color, width=1, dash="dot"),
                name=f"{label} lower",
                showlegend=False,
                hovertemplate=f"{label} lower: %{{y:.2f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[upper, upper],
                mode="lines",
                line=dict(color=line_color, width=1, dash="dot"),
                name=label,
                fill="tonexty",
                fillcolor=fill_color,
                hovertemplate=f"{label} upper: %{{y:.2f}}<extra></extra>",
            )
        )
        ce_price = _coerce_price_overlay(zone.get("ce_price"))
        if ce_price is not None and lower <= ce_price <= upper:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=[ce_price, ce_price],
                    mode="lines",
                    line=dict(color=line_color, width=1.2, dash="dashdot"),
                    name=f"{label} CE",
                    hovertemplate=f"{label} CE: %{{y:.2f}}<extra></extra>",
            )
        )

    _add_ict_fvg_history_shapes(fig, bars=bars, fvg_history=runtime_state.get("fvg_history"))


def _add_ict_fvg_history_shapes(fig, *, bars: pd.DataFrame, fvg_history) -> None:
    if bars.empty or not fvg_history:
        return

    timestamps = _bar_timestamps(bars)
    if timestamps.empty:
        return
    timestamp_by_position = list(timestamps.sort_values())
    visible_start = timestamp_by_position[0]
    visible_end = timestamp_by_position[-1]
    display_step = _bar_timestamp_step(timestamps) or pd.Timedelta(minutes=5)
    shapes = _layout_items(fig, "shapes")
    annotations = _layout_items(fig, "annotations")

    for zone in fvg_history:
        if not isinstance(zone, dict):
            continue
        lower = _coerce_price_overlay(zone.get("lower"))
        upper = _coerce_price_overlay(zone.get("upper"))
        formed_index = _coerce_int_overlay(zone.get("formed_index"))
        if lower is None or upper is None or lower > upper or formed_index is None:
            continue
        direction = _coerce_int_overlay(zone.get("direction"))
        if direction not in (-1, 1):
            continue
        start_time = _fvg_start_timestamp(
            zone,
            formed_index=formed_index,
            timestamp_by_position=timestamp_by_position,
        )
        if start_time is None:
            continue
        end_time = _fvg_end_timestamp(
            zone,
            formed_index=formed_index,
            start_time=start_time,
            display_step=display_step,
            timestamp_by_position=timestamp_by_position,
        )
        if end_time is None:
            continue
        clipped_start = max(start_time, visible_start)
        clipped_end = min(end_time, visible_end)
        if clipped_end < visible_start or clipped_start > visible_end or clipped_end < clipped_start:
            continue

        line_color = "#22c55e" if direction == 1 else "#ef4444"
        fill_color = "rgba(34, 197, 94, 0.16)" if direction == 1 else "rgba(239, 68, 68, 0.14)"
        label = "Bull FVG" if direction == 1 else "Bear FVG"
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": clipped_start,
                "x1": clipped_end,
                "y0": lower,
                "y1": upper,
                "line": {"color": line_color, "width": 1, "dash": "dot"},
                "fillcolor": fill_color,
                "layer": "below",
            }
        )
        annotations.append(
            {
                "x": clipped_start,
                "y": upper,
                "xref": "x",
                "yref": "y",
                "text": f"{label} {zone.get('source_timeframe') or '5m'}",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "bottom",
                "font": {"color": line_color, "size": 9},
            }
        )
        ce_price = _coerce_price_overlay(zone.get("ce_price"))
        if ce_price is not None and lower <= ce_price <= upper:
            shapes.append(
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "y",
                    "x0": clipped_start,
                    "x1": clipped_end,
                    "y0": ce_price,
                    "y1": ce_price,
                    "line": {"color": line_color, "width": 1, "dash": "dashdot"},
                    "layer": "below",
                }
            )

    if shapes:
        fig.update_layout(shapes=shapes, annotations=annotations)


def _fvg_start_timestamp(
    zone: dict,
    *,
    formed_index: int,
    timestamp_by_position: list[pd.Timestamp],
) -> pd.Timestamp | None:
    timestamp = _coerce_timestamp(zone.get("formed_time"))
    if timestamp is not None:
        return timestamp
    if 0 <= formed_index < len(timestamp_by_position):
        return timestamp_by_position[formed_index]
    return None


def _fvg_end_timestamp(
    zone: dict,
    *,
    formed_index: int,
    start_time: pd.Timestamp,
    display_step: pd.Timedelta,
    timestamp_by_position: list[pd.Timestamp],
) -> pd.Timestamp | None:
    cap_time = start_time + (display_step * ICT_FVG_DISPLAY_BAR_CAP)
    endpoint_candidates = [
        timestamp
        for timestamp in (
            _coerce_timestamp(zone.get("mitigation_time")),
            _coerce_timestamp(zone.get("full_mitigation_time")),
            _coerce_timestamp(zone.get("first_inversion_time")),
            _coerce_timestamp(zone.get("inversion_time")),
            _coerce_timestamp(zone.get("invalidated_time")),
        )
        if timestamp is not None and timestamp >= start_time
    ]
    endpoint_candidates.extend(
        timestamp
        for timestamp in (
            _fvg_index_timestamp(
                zone.get("mitigation_index"),
                formed_index=formed_index,
                start_time=start_time,
                display_step=display_step,
                timestamp_by_position=timestamp_by_position,
            ),
            _fvg_index_timestamp(
                zone.get("full_mitigation_index"),
                formed_index=formed_index,
                start_time=start_time,
                display_step=display_step,
                timestamp_by_position=timestamp_by_position,
            ),
            _fvg_index_timestamp(
                zone.get("first_inversion_index"),
                formed_index=formed_index,
                start_time=start_time,
                display_step=display_step,
                timestamp_by_position=timestamp_by_position,
            ),
            _fvg_index_timestamp(
                zone.get("inversion_index"),
                formed_index=formed_index,
                start_time=start_time,
                display_step=display_step,
                timestamp_by_position=timestamp_by_position,
            ),
            _fvg_index_timestamp(
                zone.get("invalidated_index"),
                formed_index=formed_index,
                start_time=start_time,
                display_step=display_step,
                timestamp_by_position=timestamp_by_position,
            ),
        )
        if timestamp is not None and timestamp >= start_time
    )
    return min([cap_time, *endpoint_candidates])


def _fvg_index_timestamp(
    value,
    *,
    formed_index: int,
    start_time: pd.Timestamp,
    display_step: pd.Timedelta,
    timestamp_by_position: list[pd.Timestamp],
) -> pd.Timestamp | None:
    index = _coerce_int_overlay(value)
    if index is None:
        return None
    offset = index - formed_index
    if offset >= 0:
        return start_time + (display_step * offset)
    if 0 <= index < len(timestamp_by_position):
        return timestamp_by_position[index]
    return None


def _coerce_int_overlay(value) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_price_overlay(value) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved) or resolved <= 0.0:
        return None
    return resolved


def _add_ict_setup_traces(fig, *, bars: pd.DataFrame, runtime_state: dict | None) -> None:
    if runtime_state is None:
        return
    setups = list(runtime_state.get("recent_setups") or [])
    if not setups or bars.empty:
        return
    go = _plotly_go()
    merged = _visible_setup_frame(bars, setups)
    if merged.empty:
        return
    for side_value, group in merged.groupby("setup_side", dropna=False):
        resolved_side = _coerce_setup_side(side_value)
        if resolved_side == 0:
            continue
        is_long = resolved_side > 0
        price_column = "low" if is_long else "high"
        base_prices = _valid_price_series(group[price_column].fillna(group["close"]))
        plot_group = group.loc[base_prices.notna()]
        if plot_group.empty:
            continue
        y_values = base_prices.loc[plot_group.index] * (0.9994 if is_long else 1.0006)
        fig.add_trace(
            go.Scatter(
                x=plot_group["timestamp"],
                y=y_values,
                mode="markers+text",
                marker=dict(
                    size=12,
                    symbol="triangle-up" if is_long else "triangle-down",
                    color="#22c55e" if is_long else "#ef4444",
                    line=dict(width=1),
                ),
                text=_setup_labels(plot_group),
                textposition="top center" if is_long else "bottom center",
                name="ICT long setup" if is_long else "ICT short setup",
                textfont=dict(size=10, color=CHART_TEXT_COLOR),
                hoverinfo="text",
                hovertext=[
                    _setup_hover_text(row)
                    for row in plot_group.itertuples(index=False)
                ],
            )
        )


def _price_axis_layout_kwargs(bars: pd.DataFrame) -> dict[str, dict]:
    xaxis = {"rangeslider": {"visible": False}}
    xaxis_range = _price_xaxis_range(bars)
    if xaxis_range is not None:
        xaxis["range"] = xaxis_range

    kwargs: dict[str, dict] = {"xaxis": xaxis}
    yaxis_range = _price_yaxis_range(bars)
    if yaxis_range is not None:
        kwargs["yaxis"] = {"range": yaxis_range}
    return kwargs


def _price_xaxis_range(bars: pd.DataFrame) -> list[object] | None:
    timestamps = _bar_timestamps(bars)
    if timestamps.empty:
        return None

    first = timestamps.min()
    last = timestamps.max()
    step = _bar_timestamp_step(timestamps) or pd.Timedelta(minutes=5)
    pad = step / 2
    min_pad = pd.Timedelta(minutes=1)
    max_pad = pd.Timedelta(hours=1)
    if pad < min_pad:
        pad = min_pad
    elif pad > max_pad:
        pad = max_pad
    return [first - pad, last + pad]


def _price_yaxis_range(bars: pd.DataFrame) -> list[float] | None:
    if bars.empty or "low" not in bars.columns or "high" not in bars.columns:
        return None

    lows = [
        value
        for value in (_coerce_price_overlay(item) for item in bars["low"])
        if value is not None
    ]
    highs = [
        value
        for value in (_coerce_price_overlay(item) for item in bars["high"])
        if value is not None
    ]
    if not lows or not highs:
        return None

    lower = min(lows)
    upper = max(highs)
    if lower > upper:
        return None

    midpoint = (lower + upper) / 2.0
    span = upper - lower
    padding = max(span * 0.10, abs(midpoint) * 0.0008, 1e-9)
    return [max(0.0, lower - padding), upper + padding]


def _visible_runtime_state(runtime_state: dict | None, bars: pd.DataFrame) -> dict | None:
    if runtime_state is None or bars.empty:
        return runtime_state

    state_timestamp = _coerce_timestamp(runtime_state.get("latest_bar_timestamp_utc"))
    if state_timestamp is None:
        return runtime_state

    timestamps = _bar_timestamps(bars)
    if timestamps.empty:
        return runtime_state

    step = _bar_timestamp_step(timestamps) or pd.Timedelta(minutes=5)
    tolerance = step * 2
    min_tolerance = pd.Timedelta(minutes=10)
    if tolerance < min_tolerance:
        tolerance = min_tolerance

    if timestamps.min() - tolerance <= state_timestamp <= timestamps.max() + tolerance:
        return runtime_state
    return None


def _visible_setup_frame(bars: pd.DataFrame, setups: list[dict]) -> pd.DataFrame:
    if bars.empty or not setups:
        return pd.DataFrame()

    frame = pd.DataFrame(setups)
    if "timestamp_utc" not in frame.columns or "setup_side" not in frame.columns:
        return pd.DataFrame()

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp_utc"],
        errors="coerce",
        utc=True,
    )
    frame = frame.dropna(subset=["timestamp"])
    required_bar_columns = {"timestamp", "low", "high", "close"}
    if frame.empty or not required_bar_columns.issubset(bars.columns):
        return pd.DataFrame()

    latest_bars = bars.loc[:, ["timestamp", "low", "high", "close"]].copy()
    latest_bars["timestamp"] = pd.to_datetime(
        latest_bars["timestamp"],
        errors="coerce",
        utc=True,
    )
    for column in ("low", "high", "close"):
        latest_bars[column] = _valid_price_series(latest_bars[column])
    latest_bars = latest_bars.dropna(subset=["timestamp", "low", "high", "close"])
    if latest_bars.empty:
        return pd.DataFrame()

    latest_bars = (
        latest_bars.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
    )
    return frame.merge(latest_bars, on="timestamp", how="inner")


def _bar_timestamps(bars: pd.DataFrame) -> pd.Series:
    if bars.empty or "timestamp" not in bars.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    timestamps = pd.to_datetime(bars["timestamp"], errors="coerce", utc=True)
    return timestamps.dropna().sort_values().reset_index(drop=True)


def _bar_timestamp_step(timestamps: pd.Series) -> pd.Timedelta | None:
    if len(timestamps) < 2:
        return None
    diffs = timestamps.diff().dropna()
    positive_diffs = diffs.loc[diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        return None
    return positive_diffs.median()


def _coerce_timestamp(value) -> pd.Timestamp | None:
    if value in {None, ""}:
        return None
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp


def _valid_price_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.map(_coerce_price_overlay).notna())


def _coerce_setup_side(value) -> int:
    try:
        resolved = int(float(value))
    except (TypeError, ValueError):
        return 0
    return resolved if resolved in {-1, 1} else 0


def _setup_labels(group: pd.DataFrame) -> list[str]:
    if "label" not in group.columns:
        return ["setup"] * len(group)
    return [str(label or "setup") for label in group["label"]]
