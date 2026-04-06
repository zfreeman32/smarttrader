from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import parse_qs

import pandas as pd

from ote_live.dashboard.charts import (
    build_audit_trail_figure,
    build_confidence_figure,
    build_health_figure,
    build_markout_figure,
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
from ote_live.models.loaders import load_direction_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LONG_RUNTIME_MANIFEST_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_long.json"
DEFAULT_SHORT_RUNTIME_MANIFEST_PATH = REPO_ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_short.json"


def create_dashboard_app(
    store_or_path: SQLiteLiveDataStore | str | Path,
    *,
    asset: str = "EURUSD",
    timeframe: str = "5m",
    refresh_interval_ms: int = 10_000,
    signal_limit: int = 120,
    long_runtime_manifest_path: str | Path = DEFAULT_LONG_RUNTIME_MANIFEST_PATH,
    short_runtime_manifest_path: str | Path = DEFAULT_SHORT_RUNTIME_MANIFEST_PATH,
):
    Dash, dcc, html, Input, Output = _dash_modules()

    if isinstance(store_or_path, SQLiteLiveDataStore):
        store = store_or_path
    else:
        store = SQLiteLiveDataStore(store_or_path)
    audit_repository = LiveAuditRepository(store)

    long_manifest_path = Path(long_runtime_manifest_path)
    short_manifest_path = Path(short_runtime_manifest_path)

    app = Dash(__name__)
    app.title = "OTE Live Dashboard"
    app.layout = html.Div(
        style={
            "maxWidth": "1680px",
            "margin": "0 auto",
            "padding": "16px",
            "fontFamily": "Segoe UI, sans-serif",
            "backgroundColor": "#f7f9fc",
        },
        children=[
            dcc.Location(id="location", refresh=False),
            html.H2("OTE Live Signal Dashboard", style={"marginBottom": "4px"}),
            html.P(
                f"{asset} {timeframe} live operator view",
                style={"marginTop": "0", "color": "#55606d"},
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                    "gap": "12px",
                },
                children=[
                    _summary_card(html, "Latest Bar", "latest-bar"),
                    _summary_card(html, "Latest Signal", "latest-signal"),
                    _summary_card(html, "Heartbeat", "heartbeat"),
                    _summary_card(html, "Paper Markout", "paper-markout"),
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "12px", "marginTop": "12px"},
                children=[
                    dcc.Graph(id="price-figure"),
                    dcc.Graph(id="health-figure"),
                ],
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                    "gap": "12px",
                    "marginTop": "12px",
                },
                children=[
                    dcc.Graph(id="long-primary-confidence-figure"),
                    dcc.Graph(id="short-primary-confidence-figure"),
                    dcc.Graph(id="markout-figure"),
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
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "12px",
                    "marginTop": "12px",
                },
                children=[
                    _inspection_panel(
                        html,
                        dcc,
                        title="Long Primary Signal Inspection",
                        subtitle="Defaults to the latest persisted signal from the current long primary model.",
                        value_prefix="long-primary-signal",
                    ),
                    _inspection_panel(
                        html,
                        dcc,
                        title="Short Primary Signal Inspection",
                        subtitle="Defaults to the latest persisted signal from the current short primary model.",
                        value_prefix="short-primary-signal",
                    ),
                ],
            ),
            dcc.Interval(id="refresh-interval", interval=int(refresh_interval_ms), n_intervals=0),
        ],
    )

    @app.callback(
        Output("latest-bar", "children"),
        Output("latest-signal", "children"),
        Output("heartbeat", "children"),
        Output("paper-markout", "children"),
        Output("price-figure", "figure"),
        Output("health-figure", "figure"),
        Output("long-primary-confidence-figure", "figure"),
        Output("short-primary-confidence-figure", "figure"),
        Output("markout-figure", "figure"),
        Output("health-events", "children"),
        Output("recent-signals", "children"),
        Output("long-primary-signal-figure", "figure"),
        Output("long-primary-signal-details", "children"),
        Output("long-primary-signal-image", "src"),
        Output("long-primary-signal-image-note", "children"),
        Output("short-primary-signal-figure", "figure"),
        Output("short-primary-signal-details", "children"),
        Output("short-primary-signal-image", "src"),
        Output("short-primary-signal-image-note", "children"),
        Input("refresh-interval", "n_intervals"),
        Input("location", "search"),
    )
    def _refresh(_n_intervals: int, location_search: str | None):
        primary_model_ids = _load_primary_model_ids(
            long_runtime_manifest_path=long_manifest_path,
            short_runtime_manifest_path=short_manifest_path,
        )
        long_primary_model_id = primary_model_ids["long"]
        short_primary_model_id = primary_model_ids["short"]

        bars = fetch_recent_bars(
            store,
            asset=asset,
            timeframe=timeframe,
            limit=max(signal_limit, 240),
        )
        signals = fetch_recent_signals(
            audit_repository,
            limit=signal_limit,
        )
        long_primary_confidence = _fetch_primary_confidence(
            audit_repository,
            model_id=long_primary_model_id,
            direction="long",
            limit=max(signal_limit, 240),
        )
        short_primary_confidence = _fetch_primary_confidence(
            audit_repository,
            model_id=short_primary_model_id,
            direction="short",
            limit=max(signal_limit, 240),
        )
        long_primary_signals = _fetch_primary_signals(
            audit_repository,
            model_id=long_primary_model_id,
            limit=signal_limit,
        )
        short_primary_signals = _fetch_primary_signals(
            audit_repository,
            model_id=short_primary_model_id,
            limit=signal_limit,
        )
        markouts = compute_signal_markouts(
            store,
            audit_repository,
            limit=signal_limit,
        )
        performance = summarize_signal_markouts(markouts)
        health_summary = build_health_summary(
            store,
            audit_repository,
            asset=asset,
            timeframe=timeframe,
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

        recent_health_lines = [
            (
                f"{event['event_timestamp']} | {event['severity']} | "
                f"{event['component']} | {event['message']}"
            )
            for event in health_summary.recent_health_events
        ] or ["No health events recorded."]

        if signals.empty:
            recent_signal_lines = ["No signals recorded."]
        else:
            recent_signal_lines = [
                (
                    f"{row.timestamp.isoformat()} | {row.model_id} | "
                    f"{row.direction} {row.decision} | p={row.probability:.4f} | "
                    f"notif={row.notification_count} | media={row.media_artifact_count}"
                )
                for row in signals.tail(12).itertuples(index=False)
            ]

        query_signal_id = _signal_decision_id_from_query(location_search)
        long_primary_signal_id = _resolve_latest_signal_id(long_primary_signals)
        short_primary_signal_id = _resolve_latest_signal_id(short_primary_signals)

        if query_signal_id is not None:
            try:
                linked_signal = audit_repository.reconstruct_signal(int(query_signal_id), include_bars=False)
            except Exception:
                linked_signal = None
            if linked_signal is not None:
                linked_model_id = linked_signal.signal.model_id
                if linked_model_id == long_primary_model_id:
                    long_primary_signal_id = int(query_signal_id)
                if linked_model_id == short_primary_model_id:
                    short_primary_signal_id = int(query_signal_id)

        long_panel = _build_signal_inspection_payload(
            audit_repository,
            signal_decision_id=long_primary_signal_id,
            model_id=long_primary_model_id,
            direction_label="long",
        )
        short_panel = _build_signal_inspection_payload(
            audit_repository,
            signal_decision_id=short_primary_signal_id,
            model_id=short_primary_model_id,
            direction_label="short",
        )

        return (
            latest_bar_text,
            latest_signal_text,
            heartbeat_text,
            paper_markout_text,
            build_price_signal_figure(
                bars,
                signals,
                title=f"{asset} {timeframe} price with all stored model signals",
            ),
            build_health_figure(health_summary),
            build_confidence_figure(
                long_primary_confidence,
                title=_confidence_title("Long Primary Confidence", long_primary_model_id),
                line_color="#198754",
                threshold_color="#14532d",
            ),
            build_confidence_figure(
                short_primary_confidence,
                title=_confidence_title("Short Primary Confidence", short_primary_model_id),
                line_color="#dc3545",
                threshold_color="#7f1d1d",
            ),
            build_markout_figure(markouts),
            "\n".join(recent_health_lines),
            "\n".join(recent_signal_lines),
            long_panel[0],
            long_panel[1],
            long_panel[2],
            long_panel[3],
            short_panel[0],
            short_panel[1],
            short_panel[2],
            short_panel[3],
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
        style={
            "backgroundColor": "white",
            "borderRadius": "12px",
            "padding": "14px 16px",
            "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.08)",
        },
        children=[
            html.Div(title, style={"fontSize": "13px", "color": "#5b6673", "marginBottom": "6px"}),
            html.Div(id=value_id, style={"fontSize": "15px", "fontWeight": 600}),
        ],
    )


def _text_panel(html, title: str, value_id: str):
    return html.Div(
        style={
            "backgroundColor": "white",
            "borderRadius": "12px",
            "padding": "14px 16px",
            "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.08)",
        },
        children=[
            html.H4(title, style={"marginTop": "0", "marginBottom": "10px"}),
            html.Pre(
                id=value_id,
                style={
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "Consolas, monospace",
                    "fontSize": "12px",
                    "margin": "0",
                },
            ),
        ],
    )


def _inspection_panel(html, dcc, *, title: str, subtitle: str, value_prefix: str):
    return html.Div(
        style={
            "backgroundColor": "white",
            "borderRadius": "12px",
            "padding": "14px 16px",
            "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.08)",
        },
        children=[
            html.H4(title, style={"margin": "0 0 4px 0"}),
            html.P(
                subtitle,
                style={"margin": "0 0 12px 0", "color": "#5b6673", "fontSize": "13px"},
            ),
            html.Pre(
                id=f"{value_prefix}-details",
                style={
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "Consolas, monospace",
                    "fontSize": "12px",
                    "margin": "0 0 12px 0",
                },
            ),
            dcc.Graph(id=f"{value_prefix}-figure"),
            html.Div(
                id=f"{value_prefix}-image-note",
                style={"fontSize": "12px", "color": "#5b6673", "marginBottom": "8px"},
            ),
            html.Img(
                id=f"{value_prefix}-image",
                style={"width": "100%", "borderRadius": "10px", "border": "1px solid #dbe3ec"},
            ),
        ],
    )


def _load_primary_model_ids(
    *,
    long_runtime_manifest_path: str | Path,
    short_runtime_manifest_path: str | Path,
) -> dict[str, str | None]:
    return {
        "long": _load_primary_model_id(long_runtime_manifest_path),
        "short": _load_primary_model_id(short_runtime_manifest_path),
    }


def _load_primary_model_id(path: str | Path) -> str | None:
    try:
        manifest = load_direction_runtime_manifest(path)
    except Exception:
        return None
    if manifest.recommendations.recommended_primary_model_id:
        return manifest.recommendations.recommended_primary_model_id
    if manifest.models:
        return manifest.models[0].model_id
    return None


def _fetch_primary_confidence(
    audit_repository: LiveAuditRepository,
    *,
    model_id: str | None,
    direction: str,
    limit: int,
) -> pd.DataFrame:
    if model_id is None:
        return pd.DataFrame()
    return fetch_confidence_history(
        audit_repository,
        model_id=model_id,
        direction=direction,
        limit=limit,
    )


def _fetch_primary_signals(
    audit_repository: LiveAuditRepository,
    *,
    model_id: str | None,
    limit: int,
) -> pd.DataFrame:
    if model_id is None:
        return pd.DataFrame()
    return fetch_recent_signals(
        audit_repository,
        model_ids=(model_id,),
        limit=limit,
    )


def _resolve_latest_signal_id(signals: pd.DataFrame) -> int | None:
    if signals.empty:
        return None
    try:
        return int(signals.iloc[-1]["signal_decision_id"])
    except Exception:
        return None


def _signal_decision_id_from_query(location_search: str | None) -> int | None:
    if not location_search:
        return None
    parsed = parse_qs(location_search.lstrip("?"))
    raw = parsed.get("signal_decision_id", [None])[0]
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _confidence_title(prefix: str, model_id: str | None) -> str:
    if model_id is None:
        return f"{prefix} | unresolved"
    return f"{prefix} | {model_id}"


def _build_signal_inspection_payload(
    audit_repository: LiveAuditRepository,
    *,
    signal_decision_id: int | None,
    model_id: str | None,
    direction_label: str,
) -> tuple[object, str, str, str]:
    if model_id is None:
        return (
            build_price_signal_figure(pd.DataFrame(), pd.DataFrame(), title=f"No {direction_label} primary model configured"),
            f"No {direction_label} primary model could be resolved from the runtime manifest.",
            "",
            "No screenshot available.",
        )
    if signal_decision_id is None:
        return (
            build_price_signal_figure(pd.DataFrame(), pd.DataFrame(), title=f"No persisted signals yet for {model_id}"),
            (
                f"model_id: {model_id}\n"
                f"direction: {direction_label}\n"
                "status: no persisted signals yet for this primary model\n"
                "note: confidence history above can still update before the first emitted signal is stored"
            ),
            "",
            "No screenshot available.",
        )
    try:
        audit_trail = audit_repository.reconstruct_signal(
            int(signal_decision_id),
            include_bars=True,
            lookback_bars=120,
        )
        image_src, image_note = _load_signal_image(audit_trail)
        return (
            build_audit_trail_figure(
                audit_trail,
                title=f"{direction_label.title()} primary signal | {audit_trail.signal.model_id}",
            ),
            _format_signal_details(audit_trail),
            image_src,
            image_note,
        )
    except Exception as exc:
        return (
            build_price_signal_figure(pd.DataFrame(), pd.DataFrame(), title=f"Could not load signal {signal_decision_id}"),
            (
                f"model_id: {model_id}\n"
                f"direction: {direction_label}\n"
                f"signal_decision_id: {signal_decision_id}\n"
                f"error: {type(exc).__name__}: {exc}"
            ),
            "",
            "Screenshot unavailable.",
        )


def _format_signal_details(audit_trail) -> str:
    signal = audit_trail.signal
    prediction = audit_trail.prediction
    notifications = ", ".join(
        f"{item.channel}:{item.status}" for item in audit_trail.notifications
    ) or "none"
    media_paths = ", ".join(item.file_path for item in audit_trail.media_artifacts) or "none"
    return "\n".join(
        [
            f"signal_decision_id: {audit_trail.signal_decision_id}",
            f"timestamp_utc: {signal.timestamp.isoformat()}",
            f"model_id: {signal.model_id}",
            f"direction: {signal.direction}",
            f"decision: {signal.decision}",
            f"probability: {signal.probability:.4f}",
            f"threshold: {signal.threshold if signal.threshold is not None else 'n/a'}",
            f"regime: {signal.regime or 'unknown'}",
            f"backend: {prediction.backend}",
            f"reasons: {', '.join(signal.reasons) if signal.reasons else 'none'}",
            f"notifications: {notifications}",
            f"media_artifacts: {media_paths}",
        ]
    )


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
