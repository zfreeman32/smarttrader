from __future__ import annotations

import base64
from datetime import datetime
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

import pandas as pd

from ote_live.dashboard.charts import (
    build_confidence_figure,
    build_frvp_price_figure,
    build_ict_price_figure,
    build_price_signal_figure,
)
from ote_live.dashboard.queries import (
    build_health_summary,
    compute_signal_markouts,
    fetch_confidence_history,
    fetch_recent_bars,
    fetch_recent_signals,
    summarize_signal_markouts,
)
from ote_live.dashboard.view_registry import DashboardViewConfig
from ote_live.dashboard.view_state import (
    DASHBOARD_VIEW_STATE_SCOPE,
    build_frvp_setup_lines,
    build_ict_setup_lines,
)
from ote_live.models.loaders import load_direction_runtime_manifest
from ote_live.ingestion.ibkr.service import IBKR_RUNTIME_STATE_SCOPE
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LONG_RUNTIME_MANIFEST_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_long.json"
DEFAULT_SHORT_RUNTIME_MANIFEST_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_short.json"

APP_BACKGROUND_COLOR = "#0b1220"
APP_TEXT_COLOR = "#e5edf7"
APP_MUTED_TEXT_COLOR = "#93a4b8"
PANEL_BACKGROUND_COLOR = "#111a2b"
PANEL_BORDER_COLOR = "#22304a"
PANEL_SHADOW = "0 18px 40px rgba(2, 6, 23, 0.38)"
IMAGE_BORDER_COLOR = "#31415f"
FOCUS_BORDER_COLOR = "#4f8cff"


@dataclass(frozen=True)
class DashboardModelSelection:
    configured_model_id: str | None
    resolved_model_id: str | None
    used_fallback: bool


def create_dashboard_app(
    store_or_path: SQLiteLiveDataStore | str | Path,
    *,
    asset: str = "EURUSD",
    timeframe: str = "5m",
    refresh_interval_ms: int = 10_000,
    signal_limit: int = 120,
    model_chart_lookback_hours: int = 24,
    long_runtime_manifest_path: str | Path = DEFAULT_LONG_RUNTIME_MANIFEST_PATH,
    short_runtime_manifest_path: str | Path = DEFAULT_SHORT_RUNTIME_MANIFEST_PATH,
    view_configs: Sequence[DashboardViewConfig] | None = None,
):
    Dash, dcc, html, Input, Output = _dash_modules()

    if isinstance(store_or_path, SQLiteLiveDataStore):
        store = store_or_path
    else:
        store = SQLiteLiveDataStore(store_or_path)
    audit_repository = LiveAuditRepository(store)

    resolved_view_configs = tuple(view_configs or (
        DashboardViewConfig(
            view_id="OTE",
            label="OTE",
            asset=asset,
            timeframe=timeframe,
            data_supplier="FMP",
            long_runtime_manifest_path=Path(long_runtime_manifest_path),
            short_runtime_manifest_path=Path(short_runtime_manifest_path),
            description=f"{asset} {timeframe} live operator view",
            recent_activity_title="Recent Signals",
        ),
    ))
    view_by_id = {view.view_id: view for view in resolved_view_configs}
    default_view = resolved_view_configs[0]

    app = Dash(__name__)
    app.title = "OTE Live Dashboard"
    app.layout = html.Div(
        style={
            "minHeight": "100vh",
            "backgroundColor": APP_BACKGROUND_COLOR,
        },
        children=[
            dcc.Location(id="dashboard-location", refresh=False),
            html.Div(
                style={
                    "maxWidth": "1680px",
                    "margin": "0 auto",
                    "padding": "16px",
                    "fontFamily": "Segoe UI, sans-serif",
                    "color": APP_TEXT_COLOR,
                },
                children=[
                    html.H2("OTE Live Signal Dashboard", style={"marginBottom": "4px", "color": APP_TEXT_COLOR}),
                    html.P(
                        default_view.description or f"{default_view.asset} {default_view.timeframe} live operator view",
                        id="dashboard-subtitle",
                        style={"marginTop": "0", "color": APP_MUTED_TEXT_COLOR},
                    ),
                    dcc.Tabs(
                        id="dashboard-tabs",
                        value=default_view.view_id,
                        colors={
                            "border": PANEL_BORDER_COLOR,
                            "primary": FOCUS_BORDER_COLOR,
                            "background": PANEL_BACKGROUND_COLOR,
                        },
                        children=[
                            dcc.Tab(
                                label=view.label,
                                value=view.view_id,
                                style={
                                    "backgroundColor": PANEL_BACKGROUND_COLOR,
                                    "color": APP_MUTED_TEXT_COLOR,
                                    "border": f"1px solid {PANEL_BORDER_COLOR}",
                                    "padding": "10px 14px",
                                },
                                selected_style={
                                    "backgroundColor": APP_BACKGROUND_COLOR,
                                    "color": APP_TEXT_COLOR,
                                    "border": f"1px solid {FOCUS_BORDER_COLOR}",
                                    "padding": "10px 14px",
                                },
                            )
                            for view in resolved_view_configs
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                            "gap": "12px",
                            "marginTop": "12px",
                        },
                        children=[
                            _summary_card(html, "Latest Bar", "latest-bar"),
                            _summary_card(html, "Latest Signal", "latest-signal"),
                            _summary_card(html, "Heartbeat", "heartbeat"),
                            _summary_card(html, "Paper Markout", "paper-markout"),
                            _summary_card(html, "IBKR ES Feed", "ibkr-feed"),
                        ],
                    ),
                    html.Div(
                        style={"marginTop": "12px"},
                        children=[
                            dcc.Graph(id="price-figure"),
                        ],
                    ),
                    html.Div(
                        style={"marginTop": "12px"},
                        children=[
                            _section_panel(
                                html,
                                title="Long Models",
                                subtitle="Each loaded long model plots probability against its active threshold. Signal decisions are marked on the line.",
                                body_id="long-model-cards",
                            ),
                            _section_panel(
                                html,
                                title="Short Models",
                                subtitle="Each loaded short model plots probability against its active threshold. Signal decisions are marked on the line.",
                                body_id="short-model-cards",
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "12px",
                            "marginTop": "12px",
                        },
                        children=[
                            _text_panel(html, "Recent Health Events", "health-events"),
                            _text_panel(html, "Recent Signals", "recent-signals"),
                        ],
                    ),
                    dcc.Interval(id="refresh-interval", interval=int(refresh_interval_ms), n_intervals=0),
                ],
            ),
        ],
    )

    @app.callback(
        Output("latest-bar", "children"),
        Output("latest-signal", "children"),
        Output("heartbeat", "children"),
        Output("paper-markout", "children"),
        Output("ibkr-feed", "children"),
        Output("dashboard-subtitle", "children"),
        Output("price-figure", "figure"),
        Output("health-events", "children"),
        Output("recent-signals", "children"),
        Output("long-model-cards", "children"),
        Output("short-model-cards", "children"),
        Input("refresh-interval", "n_intervals"),
        Input("dashboard-location", "search"),
        Input("dashboard-tabs", "value"),
    )
    def _refresh(_n_intervals: int, location_search: str | None, active_view_id: str | None):
        view = view_by_id.get(str(active_view_id), default_view)
        manifest_models = _load_manifest_models(
            long_runtime_manifest_path=view.long_runtime_manifest_path,
            short_runtime_manifest_path=view.short_runtime_manifest_path,
        )
        manifest_model_ids = tuple(
            model.model_id
            for direction_models in manifest_models.values()
            for model in direction_models
        )

        bars = fetch_recent_bars(
            store,
            asset=view.asset,
            timeframe=view.timeframe,
            limit=max(signal_limit, 240),
        )
        ibkr_state = (
            store.get_runtime_state(
                scope=IBKR_RUNTIME_STATE_SCOPE,
                state_key=view.asset.upper(),
            )
            if str(view.data_supplier).upper() == "IBKR"
            else None
        )
        bars = _merge_ibkr_live_bars(bars, ibkr_state)
        signals = fetch_recent_signals(
            audit_repository,
            model_ids=manifest_model_ids or None,
            limit=max(signal_limit, 240),
        )
        markouts = compute_signal_markouts(
            store,
            audit_repository,
            model_ids=manifest_model_ids or None,
            limit=signal_limit,
        )
        performance = summarize_signal_markouts(markouts)
        health_summary = build_health_summary(
            store,
            audit_repository,
            asset=view.asset,
            timeframe=view.timeframe,
        )
        query_signal_id = _parse_signal_decision_id_from_search(location_search)
        long_signal_id, short_signal_id = _override_signal_selection_from_query(
            audit_repository,
            query_signal_id=query_signal_id,
            long_signal_id=None,
            short_signal_id=None,
        )
        long_focus_model_id = _resolve_focus_model_id(
            audit_repository,
            signal_decision_id=long_signal_id,
            direction="long",
        )
        short_focus_model_id = _resolve_focus_model_id(
            audit_repository,
            signal_decision_id=short_signal_id,
            direction="short",
        )

        latest_bar_text = (
            health_summary.latest_bar_timestamp.isoformat()
            if health_summary.latest_bar_timestamp is not None
            else "No bars yet"
        )
        latest_signal_text = (
            health_summary.latest_signal_timestamp.isoformat()
            if health_summary.latest_signal_timestamp is not None
            else "No signals yet"
        )
        heartbeat_text = "No heartbeat yet"
        if health_summary.latest_heartbeat_observed_at is not None:
            freshness = "stale" if health_summary.heartbeat_is_stale else "fresh"
            heartbeat_text = (
                f"{health_summary.latest_heartbeat_observed_at.isoformat()} | "
                f"{health_summary.latest_heartbeat_source or 'unknown'} | "
                f"{freshness} | lag {health_summary.heartbeat_lag_seconds or 0:.1f}s"
            )
        if performance.avg_markout_pips is None:
            paper_markout_text = "No completed markouts yet"
        else:
            paper_markout_text = (
                f"avg {performance.avg_markout_pips:.2f} pips | "
                f"cum {performance.cumulative_markout_pips:.2f} pips | "
                f"win {performance.win_rate:.1%}"
            )
        ibkr_feed_text = _format_ibkr_feed_status(
            ibkr_state,
            expected=bool(str(view.data_supplier).upper() == "IBKR"),
        )

        recent_health_lines = [
            (
                f"{event['event_timestamp']} | {event['severity']} | "
                f"{event['component']} | {event['message']}"
            )
            for event in health_summary.recent_health_events
        ] or ["No health events recorded."]

        runtime_state = (
            store.get_runtime_state(
                scope=DASHBOARD_VIEW_STATE_SCOPE,
                state_key=view.resolved_runtime_state_key,
            )
            if view.has_runtime_state_overlays
            else None
        )
        recent_activity_lines = _build_recent_activity_lines(
            signals,
            runtime_state=runtime_state,
            enable_frvp_overlays=view.enable_frvp_overlays,
            enable_ict_overlays=view.enable_ict_overlays,
        )

        return (
            latest_bar_text,
            latest_signal_text,
            heartbeat_text,
            paper_markout_text,
            ibkr_feed_text,
            view.description or f"{view.asset} {view.timeframe} live operator view",
            _build_view_price_figure(
                bars=bars,
                signals=signals,
                runtime_state=runtime_state,
                view=view,
            ),
            "\n".join(recent_health_lines),
            "\n".join(recent_activity_lines),
            _build_model_confidence_panels(
                dcc,
                html,
                audit_repository,
                models=manifest_models["long"],
                history_limit=signal_limit,
                timeframe=view.timeframe,
                lookback_hours=model_chart_lookback_hours,
                focus_model_id=long_focus_model_id,
                preferred_model_order=view.preferred_model_order,
                active_weight_model_ids=view.active_weight_model_ids,
            ),
            _build_model_confidence_panels(
                dcc,
                html,
                audit_repository,
                models=manifest_models["short"],
                history_limit=signal_limit,
                timeframe=view.timeframe,
                lookback_hours=model_chart_lookback_hours,
                focus_model_id=short_focus_model_id,
                preferred_model_order=view.preferred_model_order,
                active_weight_model_ids=view.active_weight_model_ids,
            ),
        )

    return app


def _dash_modules():
    try:
        from dash import Dash, Input, Output, dcc, html
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The live dashboard requires the 'dash' package to be installed."
        ) from exc
    return Dash, dcc, html, Input, Output


def _summary_card(html, title: str, value_id: str):
    return html.Div(
        style=_panel_style(),
        children=[
            html.Div(title, style={"fontSize": "13px", "color": APP_MUTED_TEXT_COLOR, "marginBottom": "6px"}),
            html.Div(id=value_id, style={"fontSize": "15px", "fontWeight": 600, "color": APP_TEXT_COLOR}),
        ],
    )


def _text_panel(html, title: str, value_id: str):
    return html.Div(
        style=_panel_style(),
        children=[
            html.H4(title, style={"marginTop": "0", "marginBottom": "10px", "color": APP_TEXT_COLOR}),
            html.Pre(
                id=value_id,
                style={
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "Consolas, monospace",
                    "fontSize": "12px",
                    "margin": "0",
                    "color": APP_TEXT_COLOR,
                },
            ),
        ],
    )


def _section_panel(html, *, title: str, subtitle: str, body_id: str):
    return html.Div(
        style={**_panel_style(), "marginTop": "12px"},
        children=[
            html.H3(title, style={"margin": "0 0 4px 0", "color": APP_TEXT_COLOR}),
            html.P(
                subtitle,
                style={"margin": "0 0 12px 0", "color": APP_MUTED_TEXT_COLOR, "fontSize": "13px"},
            ),
            html.Div(
                id=body_id,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                    "gap": "12px",
                },
            ),
        ],
    )


def _panel_style() -> dict[str, str]:
    return {
        "backgroundColor": PANEL_BACKGROUND_COLOR,
        "borderRadius": "12px",
        "padding": "14px 16px",
        "boxShadow": PANEL_SHADOW,
        "border": f"1px solid {PANEL_BORDER_COLOR}",
    }


def _load_manifest_models(
    *,
    long_runtime_manifest_path: str | Path,
    short_runtime_manifest_path: str | Path,
) -> dict[str, tuple]:
    return {
        "long": _load_direction_models(long_runtime_manifest_path),
        "short": _load_direction_models(short_runtime_manifest_path),
    }


def _load_direction_models(path: str | Path) -> tuple:
    try:
        manifest = load_direction_runtime_manifest(path)
    except Exception:
        return ()
    return tuple(manifest.models)


def _build_model_confidence_panels(
    dcc,
    html,
    audit_repository: LiveAuditRepository,
    *,
    models: tuple,
    history_limit: int,
    timeframe: str,
    lookback_hours: float | None = 24,
    focus_model_id: str | None = None,
    preferred_model_order: Sequence[str] = (),
    active_weight_model_ids: Sequence[str] = (),
) -> list[object]:
    if not models:
        return [
            html.Div(
                style=_model_panel_style(),
                children=[
                    html.Div("No manifest models loaded.", style={"fontWeight": 600, "color": APP_TEXT_COLOR}),
                    html.Div(
                        "Export the runtime manifests to populate this section.",
                        style={"marginTop": "8px", "color": APP_MUTED_TEXT_COLOR, "fontSize": "13px"},
                    ),
                ],
            )
        ]

    ordered_models = _order_models_for_display(
        models,
        focus_model_id=focus_model_id,
        preferred_model_order=preferred_model_order,
    )
    cards: list[object] = []
    confidence_history_limit = _resolve_confidence_history_fetch_limit(
        timeframe=timeframe,
        history_limit=history_limit,
        lookback_hours=lookback_hours,
    )
    for model_manifest in ordered_models:
        latest_signal = fetch_recent_signals(
            audit_repository,
            model_ids=(model_manifest.model_id,),
            limit=1,
        )
        confidence_history = fetch_confidence_history(
            audit_repository,
            model_id=model_manifest.model_id,
            direction=model_manifest.direction,
            limit=confidence_history_limit,
        )
        full_confidence = _prepare_dashboard_confidence_history(
            confidence_history,
            model_manifest=model_manifest,
        )
        latest_confidence = _trim_confidence_history_to_lookback_window(
            full_confidence,
            lookback_hours=lookback_hours,
        )
        confidence_xaxis_range = _build_confidence_xaxis_range(
            latest_confidence,
            lookback_hours=lookback_hours,
        )
        latest_signal_row = latest_signal.iloc[-1] if not latest_signal.empty else None
        latest_confidence_row = full_confidence.iloc[-1] if not full_confidence.empty else None

        probability = None
        decision = "no signals yet"
        regime = None
        threshold = _resolve_active_threshold(
            model_manifest,
            regime=None,
            persisted_threshold=None,
        )
        last_timestamp = None
        notification_count = 0
        media_artifact_count = 0
        if latest_confidence_row is not None:
            probability = latest_confidence_row["calibrated_probability"]
            decision = str(latest_confidence_row["decision"] or "prediction only")
            regime = latest_confidence_row["regime"]
            latest_threshold = _coerce_optional_float(
                latest_confidence_row.get("display_threshold")
            )
            if latest_threshold is not None:
                threshold = latest_threshold
            last_timestamp = latest_confidence_row["timestamp"]
        if latest_signal_row is not None:
            notification_count = int(latest_signal_row["notification_count"])
            media_artifact_count = int(latest_signal_row["media_artifact_count"])
            if probability is None:
                probability = latest_signal_row["probability"]
            signal_threshold = _coerce_optional_float(latest_signal_row["threshold"])
            if signal_threshold is not None and latest_confidence_row is None:
                threshold = signal_threshold
            if regime is None:
                regime = latest_signal_row["regime"]
            if last_timestamp is None:
                last_timestamp = latest_signal_row["timestamp"]

        selection = DashboardModelSelection(
            configured_model_id=model_manifest.model_id,
            resolved_model_id=model_manifest.model_id,
            used_fallback=False,
        )
        summary_line = (
            f"{_infer_model_family(model_manifest.model_id)} | "
            f"latest {decision} | "
            f"p={_format_probability_value(probability)} | "
            f"thr={_format_probability_value(threshold)} | "
            f"regime={regime or 'unknown'}"
        )
        latest_line = (
            f"{last_timestamp.isoformat()} | notifications={notification_count} | media={media_artifact_count}"
            if last_timestamp is not None
            else "No persisted confidence history yet; configured threshold shown."
        )
        is_focused = bool(focus_model_id) and model_manifest.model_id == focus_model_id
        has_active_weight = model_manifest.model_id in active_weight_model_ids

        cards.append(
            html.Div(
                style=_model_panel_style(focused=is_focused),
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "gap": "8px", "alignItems": "center"},
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        model_manifest.model_id,
                                        style={"fontWeight": 700, "fontSize": "15px", "color": APP_TEXT_COLOR},
                                    ),
                                    html.Div(
                                        f"{model_manifest.direction} | {model_manifest.backend}",
                                        style={"marginTop": "4px", "fontSize": "12px", "color": APP_MUTED_TEXT_COLOR},
                                    ),
                                ]
                            ),
                            _build_model_status_badges(
                                html,
                                status=model_manifest.status,
                                has_active_weight=has_active_weight,
                            ),
                        ],
                    ),
                    html.Div(
                        summary_line,
                        style={
                            "fontSize": "12px",
                            "lineHeight": "1.5",
                            "margin": "12px 0 4px 0",
                            "color": APP_TEXT_COLOR,
                        },
                    ),
                    html.Div(
                        latest_line,
                        style={
                            "fontSize": "12px",
                            "lineHeight": "1.5",
                            "margin": "0 0 10px 0",
                            "color": APP_MUTED_TEXT_COLOR,
                        },
                    ),
                    dcc.Graph(
                        figure=build_confidence_figure(
                            latest_confidence,
                            title=_confidence_title("Probability vs threshold", selection),
                            xaxis_range=confidence_xaxis_range,
                            fallback_threshold=threshold,
                        ),
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "320px"},
                    ),
                    html.Div(
                        (
                            "Focused from alert link."
                            if is_focused
                            else "Signal markers show persisted emit/shadow decisions on the confidence line."
                        ),
                        style={
                            "fontSize": "11px",
                            "marginTop": "8px",
                            "color": APP_TEXT_COLOR,
                        },
                    ),
                ],
            )
        )
    return cards


def _model_panel_style(*, focused: bool = False) -> dict[str, str]:
    style = {
        "backgroundColor": "#0f1726",
        "borderRadius": "12px",
        "padding": "14px",
        "border": f"1px solid {PANEL_BORDER_COLOR}",
        "boxShadow": "0 14px 28px rgba(2, 6, 23, 0.24)",
    }
    if focused:
        style["border"] = f"1px solid {FOCUS_BORDER_COLOR}"
        style["boxShadow"] = "0 0 0 1px rgba(79, 140, 255, 0.35), 0 14px 28px rgba(2, 6, 23, 0.24)"
    return style


def _build_model_status_badges(
    html,
    *,
    status: str,
    has_active_weight: bool,
):
    badges = []
    if has_active_weight:
        badges.append(
            html.Span(
                "ACTIVE WEIGHT",
                style=_status_badge_style("active_weight"),
            )
        )
    badges.append(
        html.Span(
            "SHADOW" if has_active_weight and status == "candidate" else status.upper(),
            style=_status_badge_style(status),
        )
    )
    return html.Div(
        badges,
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "flex-end",
            "gap": "6px",
        },
    )


def _status_badge_style(status: str) -> dict[str, str]:
    palette = {
        "active": {"backgroundColor": "#16351f", "color": "#8ff0a4", "border": "1px solid #266c37"},
        "active_weight": {
            "backgroundColor": "#162d4d",
            "color": "#8fc7ff",
            "border": "1px solid #2f78c4",
        },
        "candidate": {"backgroundColor": "#2c2414", "color": "#f6d365", "border": "1px solid #8d6c1a"},
        "deprecated": {"backgroundColor": "#31161a", "color": "#ff9ca7", "border": "1px solid #7c2630"},
    }.get(status, {"backgroundColor": "#1f2937", "color": APP_TEXT_COLOR, "border": f"1px solid {PANEL_BORDER_COLOR}"})
    return {
        "padding": "4px 8px",
        "borderRadius": "999px",
        "fontSize": "11px",
        "fontWeight": 700,
        **palette,
    }


def _infer_model_family(model_id: str) -> str:
    lowered = model_id.lower()
    if "continuation" in lowered:
        return "Continuation"
    if "breakout" in lowered:
        return "Breakout"
    if "reversal" in lowered:
        return "Reversal"
    if "meta" in lowered:
        return "Meta"
    if lowered.startswith("frvp_"):
        return "FRVP"
    return "OTE"


def _format_threshold_label(model_manifest) -> str:
    thresholds = model_manifest.live_policy.thresholds
    values: list[float] = []
    if thresholds.global_threshold is not None:
        values.append(float(thresholds.global_threshold))
    if thresholds.regime_thresholds:
        values.extend(float(value) for value in thresholds.regime_thresholds.values())
    if not values:
        return "n/a"
    unique_values = {round(value, 10) for value in values}
    if len(unique_values) != 1:
        return "regime-aware"
    return f"{values[0]:.4f}"


def _load_signal_image(audit_trail) -> tuple[str, str]:
    if not audit_trail.media_artifacts:
        return "", "No screenshot stored for this signal."
    artifact = audit_trail.media_artifacts[-1]
    path = Path(artifact.file_path)
    if not path.exists():
        return "", f"Stored screenshot path is missing: {path}"
    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return (
        f"data:image/png;base64,{encoded}",
        f"Showing {artifact.artifact_type}: {path}",
    )


def _prepare_dashboard_confidence_history(
    confidence: pd.DataFrame,
    *,
    model_manifest,
) -> pd.DataFrame:
    if confidence.empty:
        return confidence.copy()

    prepared = confidence.copy().sort_values("timestamp").reset_index(drop=True)
    prepared["dashboard_active_threshold"] = [
        _resolve_active_threshold(
            model_manifest,
            regime=getattr(row, "regime", None),
            persisted_threshold=getattr(row, "threshold_applied", None),
        )
        for row in prepared.itertuples(index=False)
    ]
    prepared["display_threshold"] = prepared["dashboard_active_threshold"].where(
        prepared["dashboard_active_threshold"].notna(),
        prepared["threshold_applied"],
    )
    return prepared


def _trim_confidence_history_to_lookback_window(
    confidence: pd.DataFrame,
    *,
    lookback_hours: float | None,
) -> pd.DataFrame:
    if confidence.empty:
        return confidence.copy()

    resolved_lookback_hours = _coerce_positive_float(lookback_hours)
    if resolved_lookback_hours is None:
        return confidence.copy()

    timestamps = confidence["timestamp"].dropna()
    if timestamps.empty:
        return confidence.copy()

    latest_timestamp = timestamps.max()
    cutoff_timestamp = latest_timestamp - pd.Timedelta(hours=resolved_lookback_hours)
    trimmed = confidence.loc[confidence["timestamp"] >= cutoff_timestamp].copy()
    if trimmed.empty:
        return confidence.tail(1).copy().reset_index(drop=True)
    return trimmed.reset_index(drop=True)


def _build_confidence_xaxis_range(
    confidence: pd.DataFrame,
    *,
    lookback_hours: float | None,
) -> tuple[object, object] | None:
    if confidence.empty:
        return None

    resolved_lookback_hours = _coerce_positive_float(lookback_hours)
    if resolved_lookback_hours is None:
        return None

    timestamps = confidence["timestamp"].dropna()
    if timestamps.empty:
        return None

    latest_timestamp = timestamps.max()
    window_start = latest_timestamp - pd.Timedelta(hours=resolved_lookback_hours)
    if window_start >= latest_timestamp:
        return None
    return window_start, latest_timestamp


def _resolve_confidence_history_fetch_limit(
    *,
    timeframe: str,
    history_limit: int,
    lookback_hours: float | None,
) -> int:
    base_limit = max(60, int(history_limit))
    resolved_lookback_hours = _coerce_positive_float(lookback_hours)
    if resolved_lookback_hours is None:
        return base_limit

    timeframe_minutes = _parse_timeframe_minutes(timeframe)
    if timeframe_minutes is None:
        return base_limit

    estimated_points = int(math.ceil((resolved_lookback_hours * 60.0) / timeframe_minutes)) + 2
    return max(base_limit, estimated_points)


def _parse_timeframe_minutes(timeframe: str | None) -> int | None:
    if timeframe is None:
        return None

    value = str(timeframe).strip().lower()
    if not value:
        return None

    if value.isdigit():
        minutes = int(value)
        return minutes if minutes > 0 else None

    match = re.fullmatch(r"(?P<count>\d+)\s*(?P<unit>[a-z]+)", value)
    if match is None:
        return None

    count = int(match.group("count"))
    if count <= 0:
        return None

    factor = {
        "m": 1,
        "min": 1,
        "mins": 1,
        "minute": 1,
        "minutes": 1,
        "h": 60,
        "hr": 60,
        "hrs": 60,
        "hour": 60,
        "hours": 60,
        "d": 1_440,
        "day": 1_440,
        "days": 1_440,
    }.get(match.group("unit"))
    if factor is None:
        return None
    return count * factor


def _resolve_active_threshold(
    model_manifest,
    *,
    regime: str | None,
    persisted_threshold: float | None,
) -> float | None:
    thresholds = model_manifest.live_policy.thresholds
    regime_thresholds = thresholds.regime_thresholds or {}

    if regime and regime in regime_thresholds:
        return float(regime_thresholds[regime])
    if thresholds.global_threshold is not None:
        return float(thresholds.global_threshold)
    return _coerce_optional_float(persisted_threshold)


def _order_models_for_display(
    models: tuple,
    *,
    focus_model_id: str | None,
    preferred_model_order: Sequence[str] = (),
) -> tuple:
    preferred_rank = {model_id: index for index, model_id in enumerate(preferred_model_order)}
    return tuple(
        sorted(
            models,
            key=lambda model: (
                0 if focus_model_id and model.model_id == focus_model_id else 1,
                preferred_rank.get(model.model_id, len(preferred_rank)),
                model.model_id,
            ),
        )
    )


def _parse_signal_decision_id_from_search(search: str | None) -> int | None:
    if not search:
        return None
    query = parse_qs(search.lstrip("?"), keep_blank_values=True)
    value = query.get("signal_decision_id", [None])[-1]
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_focus_model_id(
    audit_repository: LiveAuditRepository,
    *,
    signal_decision_id: int | None,
    direction: str,
) -> str | None:
    if signal_decision_id is None:
        return None
    signal = audit_repository.get_signal_decision(signal_decision_id)
    if signal is None or signal.direction != direction:
        return None
    return signal.model_id


def _resolve_dashboard_model_selection(
    audit_repository: LiveAuditRepository,
    *,
    configured_model_id: str | None,
    direction: str,
    manifest_model_ids: tuple[str, ...],
) -> DashboardModelSelection:
    resolved_manifest_model_ids = tuple(
        str(model_id)
        for model_id in manifest_model_ids
        if str(model_id)
    )
    if configured_model_id and configured_model_id in resolved_manifest_model_ids:
        return DashboardModelSelection(
            configured_model_id=configured_model_id,
            resolved_model_id=configured_model_id,
            used_fallback=False,
        )

    recent_signals = fetch_recent_signals(
        audit_repository,
        model_ids=resolved_manifest_model_ids or None,
        limit=max(20, len(resolved_manifest_model_ids) * 4),
    )
    if not recent_signals.empty:
        direction_signals = recent_signals.loc[recent_signals["direction"] == direction]
        if not direction_signals.empty:
            resolved_model_id = str(direction_signals.iloc[-1]["model_id"])
            return DashboardModelSelection(
                configured_model_id=configured_model_id,
                resolved_model_id=resolved_model_id,
                used_fallback=True,
            )

    fallback_model_id = resolved_manifest_model_ids[0] if resolved_manifest_model_ids else configured_model_id
    return DashboardModelSelection(
        configured_model_id=configured_model_id,
        resolved_model_id=fallback_model_id,
        used_fallback=bool(fallback_model_id and fallback_model_id != configured_model_id),
    )


def _override_signal_selection_from_query(
    audit_repository: LiveAuditRepository,
    *,
    query_signal_id: int | None,
    long_signal_id: int | None,
    short_signal_id: int | None,
) -> tuple[int | None, int | None]:
    if query_signal_id is None:
        return long_signal_id, short_signal_id
    signal = audit_repository.get_signal_decision(query_signal_id)
    if signal is None:
        return long_signal_id, short_signal_id
    if signal.direction == "long":
        return int(query_signal_id), short_signal_id
    if signal.direction == "short":
        return long_signal_id, int(query_signal_id)
    return long_signal_id, short_signal_id


def _confidence_title(base_title: str, selection: DashboardModelSelection) -> str:
    if not selection.resolved_model_id:
        return base_title
    if selection.used_fallback and selection.configured_model_id and selection.configured_model_id != selection.resolved_model_id:
        return f"{base_title} | {selection.resolved_model_id} (fallback)"
    return f"{base_title} | {selection.resolved_model_id}"


def _coerce_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(resolved):
        return None
    return resolved


def _coerce_positive_float(value: float | None) -> float | None:
    resolved = _coerce_optional_float(value)
    if resolved is None or resolved <= 0:
        return None
    return resolved


def _format_probability_value(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    resolved = float(value)
    magnitude = abs(resolved)
    if magnitude >= 0.01:
        return f"{resolved:.4f}"
    if magnitude >= 0.001:
        return f"{resolved:.6f}"
    return f"{resolved:.6e}"


def _build_recent_activity_lines(
    signals: pd.DataFrame,
    *,
    runtime_state: dict | None,
    enable_frvp_overlays: bool,
    enable_ict_overlays: bool = False,
) -> list[str]:
    if not enable_frvp_overlays and not enable_ict_overlays:
        if signals.empty:
            return ["No signals recorded."]
        return [
            (
                f"{row.timestamp.isoformat()} | {row.model_id} | "
                f"{row.direction} {row.decision} | "
                f"p={_format_probability_value(row.probability)} | "
                f"raw={_format_probability_value(getattr(row, 'raw_score', None))} | "
                f"notif={row.notification_count} | media={row.media_artifact_count}"
            )
            for row in signals.tail(12).itertuples(index=False)
        ]

    if enable_frvp_overlays:
        setup_lines = build_frvp_setup_lines(runtime_state, limit=8)
    else:
        setup_lines = build_ict_setup_lines(runtime_state, limit=8)
    if signals.empty:
        return setup_lines
    signal_lines = [
        (
            f"{row.timestamp.isoformat()} | {row.model_id} | "
            f"{row.direction} {row.decision} | "
            f"p={_format_probability_value(row.probability)} | "
            f"thr={_format_probability_value(getattr(row, 'threshold', None))}"
        )
        for row in signals.tail(6).itertuples(index=False)
    ]
    return [*setup_lines, "", "Latest model decisions:", *signal_lines]


def _merge_ibkr_live_bars(
    persisted_bars: pd.DataFrame,
    ibkr_state: dict | None,
) -> pd.DataFrame:
    payload = list((ibkr_state or {}).get("bars") or [])
    if not payload:
        return persisted_bars
    live = pd.DataFrame(payload)
    required = {"timestamp_utc", "open", "high", "low", "close", "conid"}
    if live.empty or not required.issubset(live.columns):
        return persisted_bars
    live["timestamp"] = pd.to_datetime(
        live["timestamp_utc"],
        errors="coerce",
        utc=True,
    )
    live = live.dropna(subset=["timestamp", "open", "high", "low", "close"])
    status = dict((ibkr_state or {}).get("status") or {})
    if (
        status.get("connection_state") == "rolling"
        and status.get("active_conId") is not None
    ):
        live_conids = pd.to_numeric(live["conid"], errors="coerce")
        live = live.loc[
            live_conids == int(status["active_conId"])
        ].copy()
    live = _filter_live_frame_to_roll_segments(
        live,
        list((ibkr_state or {}).get("roll_events") or []),
    )
    if live.empty:
        return persisted_bars
    live = live.rename(
        columns={
            "conid": "instrument_id",
            "local_symbol": "contract_symbol",
        }
    )
    live["asset"] = "ES"
    live["timeframe"] = "5m"
    live["symbol"] = "ES"
    live["bid"] = None
    live["ask"] = None
    live["spread"] = None
    columns = list(
        dict.fromkeys(
            [
                *persisted_bars.columns.tolist(),
                "asset",
                "timeframe",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "bid",
                "ask",
                "spread",
                "source",
                "symbol",
                "contract_symbol",
                "instrument_id",
                "is_complete",
            ]
        )
    )
    combined = pd.concat(
        [
            persisted_bars.reindex(columns=columns),
            live.reindex(columns=columns),
        ],
        ignore_index=True,
        sort=False,
    )
    combined["_contract_key"] = combined["instrument_id"].fillna(-1).astype(str)
    combined = combined.drop_duplicates(
        subset=["_contract_key", "timestamp"],
        keep="last",
    )
    return (
        combined.drop(columns=["_contract_key"])
        .sort_values(["timestamp", "instrument_id"], kind="stable")
        .reset_index(drop=True)
    )


def _filter_live_frame_to_roll_segments(
    live: pd.DataFrame,
    roll_events: list[dict],
) -> pd.DataFrame:
    filtered = live
    for event in roll_events:
        try:
            effective_at = pd.to_datetime(
                event["effective_at_utc"],
                errors="raise",
                utc=True,
            )
            from_conid = int(event["from_conId"])
            to_conid = int(event["to_conId"])
        except (KeyError, TypeError, ValueError):
            continue
        conids = pd.to_numeric(filtered["conid"], errors="coerce")
        keep = ~(
            ((conids == to_conid) & (filtered["timestamp"] < effective_at))
            | ((conids == from_conid) & (filtered["timestamp"] >= effective_at))
        )
        filtered = filtered.loc[keep].copy()
    return filtered


def _format_ibkr_feed_status(
    ibkr_state: dict | None,
    *,
    expected: bool,
) -> str:
    if not expected:
        return "Not used by this view"
    if not ibkr_state:
        return "Disabled or collector not started"
    status = dict(ibkr_state.get("status") or {})
    quote = dict(ibkr_state.get("quote") or {})
    state = str(status.get("connection_state") or "unknown")
    contract = str(status.get("active_local_symbol") or "no contract")
    expiration = str(status.get("active_expiration") or "unknown expiry")
    mode = str(
        status.get("live_or_delayed")
        or status.get("market_data_type_requested")
        or "unknown"
    )
    bid = _format_optional_market_value(quote.get("bid"))
    ask = _format_optional_market_value(quote.get("ask"))
    last = _format_optional_market_value(quote.get("last"))
    spread_value = None
    try:
        if quote.get("bid") is not None and quote.get("ask") is not None:
            spread_value = float(quote["ask"]) - float(quote["bid"])
    except (TypeError, ValueError):
        spread_value = None
    spread = _format_optional_market_value(spread_value)
    last_update = _format_new_york_timestamp(
        quote.get("last_update_timestamp_utc")
        or status.get("last_bar_at")
    )
    next_roll = _format_new_york_timestamp(status.get("next_roll_at"))
    warnings: list[str] = []
    if status.get("stale_quote") or status.get("stale_bar"):
        warnings.append("STALE")
    if state == "rolling":
        warnings.append("ROLLING")
    if status.get("last_error_message"):
        warnings.append(str(status["last_error_message"]))
    suffix = f" | {'; '.join(warnings)}" if warnings else ""
    return (
        f"{state} / {mode} | {contract} exp {expiration} | "
        f"bid {bid} ask {ask} last {last} spread {spread} | "
        f"updated {last_update} | next roll {next_roll}{suffix}"
    )


def _format_optional_market_value(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _format_new_york_timestamp(value: object) -> str:
    if value in {None, ""}:
        return "—"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(ZoneInfo("America/New_York")).strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
    except (TypeError, ValueError):
        return str(value)


def _build_view_price_figure(
    *,
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    runtime_state: dict | None,
    view: DashboardViewConfig,
):
    if view.enable_frvp_overlays:
        return build_frvp_price_figure(
            bars,
            signals,
            runtime_state=runtime_state,
            title=f"{view.asset} {view.timeframe} price with FRVP live overlays",
        )
    if view.enable_ict_overlays:
        return build_ict_price_figure(
            bars,
            signals,
            runtime_state=runtime_state,
            title=f"{view.asset} {view.timeframe} price with ICT live overlays",
        )
    return build_price_signal_figure(
        bars,
        signals,
        title=f"{view.asset} {view.timeframe} price with multifamily live-model signals",
    )
