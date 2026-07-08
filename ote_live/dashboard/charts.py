from __future__ import annotations

import pandas as pd

from ote_live.storage import SignalAuditTrail

CHART_BACKGROUND_COLOR = "#111a2b"
CHART_PLOT_COLOR = "#0f1726"
CHART_GRID_COLOR = "#23314c"
CHART_TEXT_COLOR = "#e5edf7"
CHART_MUTED_TEXT_COLOR = "#9fb0c5"
CHART_BORDER_COLOR = "#2a3a58"


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

    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=620,
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
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

    _add_frvp_overlay_traces(fig, bars=bars, runtime_state=runtime_state)
    _add_emitted_signal_traces(fig, signals)
    _add_frvp_setup_traces(fig, bars=bars, runtime_state=runtime_state)

    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=640,
        margin=dict(l=30, r=20, t=60, b=30),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    _apply_dark_chart_theme(fig)
    return fig


def build_confidence_figure(
    confidence: pd.DataFrame,
    *,
    title: str = "Model Confidence",
    line_color: str = "#0d6efd",
    threshold_color: str = "#fd7e14",
    xaxis_range: tuple[object, object] | None = None,
):
    go = _plotly_go()

    fig = go.Figure()
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
    fig.update_layout(**layout_kwargs)
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
    return "<br>".join(
        [
            f"{getattr(row, 'label', 'FRVP setup')}",
            f"Confidence: {confidence}",
            f"Timestamp: {getattr(row, 'timestamp_utc', 'unknown')}",
        ]
    )


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
        f"Probability: {_format_probability(row.calibrated_probability)}",
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
    return "<br>".join(lines)


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
    frame = pd.DataFrame(setups)
    frame["timestamp"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"])
    if frame.empty:
        return
    latest_bars = bars.loc[:, ["timestamp", "low", "high", "close"]].copy()
    merged = frame.merge(latest_bars, on="timestamp", how="left")
    for side_value, group in merged.groupby("setup_side", dropna=False):
        resolved_side = int(side_value or 0)
        if resolved_side == 0:
            continue
        is_long = resolved_side > 0
        price_column = "low" if is_long else "high"
        fallback_close = group["close"]
        y_values = group[price_column].fillna(fallback_close) * (0.9996 if is_long else 1.0004)
        fig.add_trace(
            go.Scatter(
                x=group["timestamp"],
                y=y_values,
                mode="markers+text",
                marker=dict(
                    size=12,
                    symbol="triangle-up" if is_long else "triangle-down",
                    color="#22c55e" if is_long else "#ef4444",
                    line=dict(width=1),
                ),
                text=list(group["label"]),
                textposition="top center" if is_long else "bottom center",
                name="FRVP long setup" if is_long else "FRVP short setup",
                textfont=dict(size=10, color=CHART_TEXT_COLOR),
                hoverinfo="text",
                hovertext=[
                    _setup_hover_text(row)
                    for row in group.itertuples(index=False)
                ],
            )
        )
