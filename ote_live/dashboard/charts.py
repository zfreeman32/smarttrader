from __future__ import annotations

import pandas as pd

from ote_live.storage import SignalAuditTrail


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

    if not signals.empty:
        palette = (
            "#0d6efd",
            "#dc3545",
            "#198754",
            "#fd7e14",
            "#6f42c1",
            "#0dcaf0",
            "#6c757d",
            "#20c997",
        )
        model_ids = [
            str(model_id)
            for model_id in signals.get("model_id", pd.Series(dtype=str)).dropna().astype(str).unique()
        ]
        color_by_model = {
            model_id: palette[index % len(palette)]
            for index, model_id in enumerate(model_ids)
        }

        grouped = signals.groupby(["model_id", "direction", "decision"], dropna=False)
        for (model_id, direction, decision), group in grouped:
            resolved_direction = str(direction or "unknown")
            resolved_decision = str(decision or "unknown")
            resolved_model_id = str(model_id or "unknown_model")
            price_column = "bar_low" if resolved_direction == "long" else "bar_high"
            fallback_symbol = "circle-open"
            symbol = {
                ("long", "emit"): "triangle-up",
                ("short", "emit"): "triangle-down",
                ("long", "shadow"): "triangle-up-open",
                ("short", "shadow"): "triangle-down-open",
            }.get((resolved_direction, resolved_decision), fallback_symbol)
            scale = 0.9998 if resolved_direction == "long" else 1.0002
            y_values = group[price_column].fillna(group["bar_close"]) * scale
            marker_size = 12 if resolved_decision == "emit" else 8
            marker_opacity = 0.95 if resolved_decision == "emit" else 0.55
            fig.add_trace(
                go.Scatter(
                    x=group["timestamp"],
                    y=y_values,
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        symbol=symbol,
                        color=color_by_model.get(resolved_model_id, "#6c757d"),
                        opacity=marker_opacity,
                        line=dict(width=1),
                    ),
                    name=f"{resolved_model_id} | {resolved_direction} {resolved_decision}",
                    text=[
                        _signal_hover_text(row)
                        for row in group.itertuples(index=False)
                    ],
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=620,
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def build_confidence_figure(
    confidence: pd.DataFrame,
    *,
    title: str = "Model Confidence",
    line_color: str = "#0d6efd",
    threshold_color: str = "#fd7e14",
):
    go = _plotly_go()

    fig = go.Figure()
    if not confidence.empty:
        fig.add_trace(
            go.Scatter(
                x=confidence["timestamp"],
                y=confidence["calibrated_probability"],
                mode="lines+markers",
                line=dict(color=line_color, width=2),
                marker=dict(size=6),
                name="Calibrated probability",
            )
        )
        if confidence["threshold_applied"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=confidence["timestamp"],
                    y=confidence["threshold_applied"],
                    mode="lines",
                    line=dict(color=threshold_color, width=1.5, dash="dash"),
                    name="Threshold",
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

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=320,
        margin=dict(l=30, r=20, t=50, b=30),
        hovermode="x unified",
        yaxis=dict(range=[0.0, 1.0]),
    )
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
        template="plotly_white",
        title=title,
        height=320,
        margin=dict(l=30, r=20, t=50, b=30),
        hovermode="x unified",
        yaxis=dict(title="Per signal"),
        yaxis2=dict(title="Cumulative", overlaying="y", side="right"),
    )
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
        template="plotly_white",
        title=title,
        height=280,
        margin=dict(l=30, r=20, t=50, b=30),
        showlegend=False,
    )
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


def _signal_hover_text(row) -> str:
    threshold = "n/a" if row.threshold is None else f"{row.threshold:.4f}"
    return (
        f"Model: {getattr(row, 'model_id', 'unknown')}<br>"
        f"{row.direction} {row.decision}<br>"
        f"Probability: {row.probability:.4f}<br>"
        f"Threshold: {threshold}<br>"
        f"Regime: {row.regime or 'unknown'}"
    )


def _confidence_hover_text(row) -> str:
    threshold = "n/a" if row.threshold_applied is None else f"{row.threshold_applied:.4f}"
    decision = row.decision or "hold/unpersisted"
    return (
        f"Model: {row.model_id}<br>"
        f"Probability: {row.calibrated_probability:.4f}<br>"
        f"Threshold: {threshold}<br>"
        f"Decision: {decision}<br>"
        f"Regime: {row.regime or 'unknown'}"
    )
