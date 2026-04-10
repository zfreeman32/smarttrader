from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs

import pandas as pd

from ote_live.dashboard.charts import (
    build_audit_trail_figure,
    build_confidence_figure,
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

APP_BACKGROUND_COLOR = "#0b1220"
APP_TEXT_COLOR = "#e5edf7"
APP_MUTED_TEXT_COLOR = "#93a4b8"
PANEL_BACKGROUND_COLOR = "#111a2b"
PANEL_BORDER_COLOR = "#22304a"
PANEL_SHADOW = "0 18px 40px rgba(2, 6, 23, 0.38)"
IMAGE_BORDER_COLOR = "#31415f"


@dataclass(frozen=True)
class DashboardModelSelection:
    configured_model_id: str | None
    resolved_model_id: str | None
    direction: str
    used_fallback: bool = False


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
            "minHeight": "100vh",
            "backgroundColor": APP_BACKGROUND_COLOR,
        },
        children=[
            html.Div(
                style={
                    "maxWidth": "1680px",
                    "margin": "0 auto",
                    "padding": "16px",
                    "fontFamily": "Segoe UI, sans-serif",
                    "color": APP_TEXT_COLOR,
                },
                children=[
                    dcc.Location(id="location", refresh=False),
                    html.H2("OTE Live Signal Dashboard", style={"marginBottom": "4px", "color": APP_TEXT_COLOR}),
                    html.P(
                        f"{asset} {timeframe} live operator view",
                        style={"marginTop": "0", "color": APP_MUTED_TEXT_COLOR},
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
                        style={"marginTop": "12px"},
                        children=[
                            dcc.Graph(id="price-figure"),
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                            "gap": "12px",
                            "marginTop": "12px",
                        },
                        children=[
                            dcc.Graph(id="long-primary-confidence-figure"),
                            dcc.Graph(id="short-primary-confidence-figure"),
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
                                subtitle="Defaults to the latest persisted signal for the configured long primary model.",
                                value_prefix="long-primary-signal",
                            ),
                            _inspection_panel(
                                html,
                                dcc,
                                title="Short Primary Signal Inspection",
                                subtitle="Defaults to the latest persisted signal for the configured short primary model.",
                                value_prefix="short-primary-signal",
                            ),
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
        Output("price-figure", "figure"),
        Output("long-primary-confidence-figure", "figure"),
        Output("short-primary-confidence-figure", "figure"),
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
        manifest_model_ids = _load_manifest_model_ids(
            long_runtime_manifest_path=long_manifest_path,
            short_runtime_manifest_path=short_manifest_path,
        )
        long_model_selection = _resolve_dashboard_model_selection(
            audit_repository,
            configured_model_id=primary_model_ids["long"],
            direction="long",
            manifest_model_ids=manifest_model_ids["long"],
        )
        short_model_selection = _resolve_dashboard_model_selection(
            audit_repository,
            configured_model_id=primary_model_ids["short"],
            direction="short",
            manifest_model_ids=manifest_model_ids["short"],
        )

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
            model_id=long_model_selection.resolved_model_id,
            direction="long",
            limit=max(signal_limit, 240),
        )
        short_primary_confidence = _fetch_primary_confidence(
            audit_repository,
            model_id=short_model_selection.resolved_model_id,
            direction="short",
            limit=max(signal_limit, 240),
        )
        long_primary_signals = _fetch_primary_signals(
            audit_repository,
            model_id=long_model_selection.resolved_model_id,
            limit=signal_limit,
        )
        short_primary_signals = _fetch_primary_signals(
            audit_repository,
            model_id=short_model_selection.resolved_model_id,
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
                    f"{row.direction} {row.decision} | "
                    f"p={_format_probability_value(row.probability)} | "
                    f"raw={_format_probability_value(getattr(row, 'raw_score', None))} | "
                    f"notif={row.notification_count} | media={row.media_artifact_count}"
                )
                for row in signals.tail(12).itertuples(index=False)
            ]

        query_signal_id = _signal_decision_id_from_query(location_search)
        long_primary_signal_id = _resolve_latest_signal_id(long_primary_signals)
        short_primary_signal_id = _resolve_latest_signal_id(short_primary_signals)
        long_primary_signal_id, short_primary_signal_id = _override_signal_selection_from_query(
            audit_repository,
            query_signal_id=query_signal_id,
            long_signal_id=long_primary_signal_id,
            short_signal_id=short_primary_signal_id,
        )

        long_panel = _build_signal_inspection_payload(
            audit_repository,
            signal_decision_id=long_primary_signal_id,
            model_selection=long_model_selection,
            direction_label="long",
        )
        short_panel = _build_signal_inspection_payload(
            audit_repository,
            signal_decision_id=short_primary_signal_id,
            model_selection=short_model_selection,
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
            build_confidence_figure(
                long_primary_confidence,
                title=_confidence_title("Long Primary Confidence", long_model_selection),
                line_color="#198754",
                threshold_color="#14532d",
            ),
            build_confidence_figure(
                short_primary_confidence,
                title=_confidence_title("Short Primary Confidence", short_model_selection),
                line_color="#dc3545",
                threshold_color="#7f1d1d",
            ),
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


def _inspection_panel(html, dcc, *, title: str, subtitle: str, value_prefix: str):
    return html.Div(
        style=_panel_style(),
        children=[
            html.H4(title, style={"margin": "0 0 4px 0", "color": APP_TEXT_COLOR}),
            html.P(
                subtitle,
                style={"margin": "0 0 12px 0", "color": APP_MUTED_TEXT_COLOR, "fontSize": "13px"},
            ),
            html.Pre(
                id=f"{value_prefix}-details",
                style={
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "Consolas, monospace",
                    "fontSize": "12px",
                    "margin": "0 0 12px 0",
                    "color": APP_TEXT_COLOR,
                },
            ),
            dcc.Graph(id=f"{value_prefix}-figure"),
            html.Div(
                id=f"{value_prefix}-image-note",
                style={"fontSize": "12px", "color": APP_MUTED_TEXT_COLOR, "marginBottom": "8px"},
            ),
            html.Img(
                id=f"{value_prefix}-image",
                style={"width": "100%", "borderRadius": "10px", "border": f"1px solid {IMAGE_BORDER_COLOR}"},
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


def _load_primary_model_ids(
    *,
    long_runtime_manifest_path: str | Path,
    short_runtime_manifest_path: str | Path,
) -> dict[str, str | None]:
    return {
        "long": _load_primary_model_id(long_runtime_manifest_path),
        "short": _load_primary_model_id(short_runtime_manifest_path),
    }


def _load_manifest_model_ids(
    *,
    long_runtime_manifest_path: str | Path,
    short_runtime_manifest_path: str | Path,
) -> dict[str, tuple[str, ...]]:
    return {
        "long": _load_direction_model_ids(long_runtime_manifest_path),
        "short": _load_direction_model_ids(short_runtime_manifest_path),
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


def _load_direction_model_ids(path: str | Path) -> tuple[str, ...]:
    try:
        manifest = load_direction_runtime_manifest(path)
    except Exception:
        return ()
    return tuple(model.model_id for model in manifest.models)


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


def _resolve_dashboard_model_selection(
    audit_repository: LiveAuditRepository,
    *,
    configured_model_id: str | None,
    direction: str,
    manifest_model_ids: tuple[str, ...],
) -> DashboardModelSelection:
    if configured_model_id is not None:
        return DashboardModelSelection(
            configured_model_id=configured_model_id,
            resolved_model_id=configured_model_id,
            direction=direction,
            used_fallback=False,
        )
    candidate_model_ids = _ordered_unique(
        *manifest_model_ids,
        *_fetch_live_model_ids_for_direction(audit_repository, direction=direction),
    )
    return DashboardModelSelection(
        configured_model_id=None,
        resolved_model_id=candidate_model_ids[0] if candidate_model_ids else None,
        direction=direction,
        used_fallback=False,
    )


def _fetch_live_model_ids_for_direction(
    audit_repository: LiveAuditRepository,
    *,
    direction: str,
    limit: int = 20,
) -> tuple[str, ...]:
    rows = audit_repository.store.connection.execute(
        """
        SELECT model_id, MAX(timestamp_utc) AS latest_timestamp_utc, MAX(id) AS latest_prediction_id
        FROM model_predictions
        WHERE direction = ?
        GROUP BY model_id
        ORDER BY latest_timestamp_utc DESC, latest_prediction_id DESC
        LIMIT ?
        """,
        (direction, int(limit)),
    ).fetchall()
    return tuple(str(row["model_id"]) for row in rows if row["model_id"])


def _ordered_unique(*values: str | None) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return tuple(ordered)


def _override_signal_selection_from_query(
    audit_repository: LiveAuditRepository,
    *,
    query_signal_id: int | None,
    long_signal_id: int | None,
    short_signal_id: int | None,
) -> tuple[int | None, int | None]:
    if query_signal_id is None:
        return long_signal_id, short_signal_id
    try:
        linked_signal = audit_repository.reconstruct_signal(int(query_signal_id), include_bars=False)
    except Exception:
        return long_signal_id, short_signal_id
    if linked_signal.signal.direction == "long":
        return int(query_signal_id), short_signal_id
    if linked_signal.signal.direction == "short":
        return long_signal_id, int(query_signal_id)
    return long_signal_id, short_signal_id


def _confidence_title(prefix: str, model_selection: DashboardModelSelection) -> str:
    model_id = model_selection.resolved_model_id
    if model_id is None:
        return f"{prefix} | unresolved"
    if model_selection.used_fallback and model_selection.configured_model_id:
        return f"{prefix} | {model_id} (fallback from {model_selection.configured_model_id})"
    return f"{prefix} | {model_id}"


def _build_signal_inspection_payload(
    audit_repository: LiveAuditRepository,
    *,
    signal_decision_id: int | None,
    model_selection: DashboardModelSelection,
    direction_label: str,
) -> tuple[object, str, str, str]:
    model_id = model_selection.resolved_model_id
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
            _build_missing_signal_details(model_selection, direction_label=direction_label),
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


def _build_missing_signal_details(
    model_selection: DashboardModelSelection,
    *,
    direction_label: str,
) -> str:
    lines = [
        f"model_id: {model_selection.resolved_model_id or 'unresolved'}",
        f"direction: {direction_label}",
        "status: no persisted signals yet for this primary model",
        "note: confidence history above can still update before the first emitted signal is stored",
    ]
    if model_selection.used_fallback and model_selection.configured_model_id:
        lines.insert(
            1,
            f"configured_primary_model_id: {model_selection.configured_model_id}",
        )
        lines.append(
            "fallback: configured primary has no stored live predictions yet, so the dashboard selected the nearest live model with data"
        )
    return "\n".join(lines)


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
            f"probability: {_format_probability_value(signal.probability)}",
            f"raw_score: {_format_probability_value(prediction.raw_score)}",
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


def _format_probability_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    resolved = float(value)
    magnitude = abs(resolved)
    if magnitude >= 0.01:
        return f"{resolved:.4f}"
    if magnitude >= 0.001:
        return f"{resolved:.6f}"
    return f"{resolved:.6e}"
