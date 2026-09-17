from __future__ import annotations

import base64
from datetime import datetime
from html import unescape
import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
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
from ote_live.dashboard.es_chart import (
    build_es_price_figure, fetch_es_chart_data, matching_qualified_decisions,
    setup_layer_options, visible_setups, setup_hover, decision_hover,
)
from ote_live.dashboard.queries import (
    build_health_summary,
    compute_signal_markouts,
    fetch_confidence_history,
    fetch_frvp_paper_signal_markouts,
    fetch_recent_bars,
    fetch_recent_signals,
    summarize_signal_markouts,
    summarize_frvp_paper_signal_markouts,
)
from ote_live.dashboard.view_registry import (
    FRVP_PAPER_SIGNAL_BUNDLE_ID,
    DashboardViewConfig,
    is_frvp_paper_signal_bundle_path_set,
    validate_frvp_paper_signal_bundle_contents,
)
from ote_live.dashboard.view_state import (
    DASHBOARD_VIEW_STATE_SCOPE,
    build_frvp_setup_lines,
    build_ict_setup_lines,
)
from ote_live.models.loaders import load_direction_runtime_manifest
from ote_live.models.setup_family import infer_setup_model_route
from ote_live.models.frvp_research import (
    FRVP_RESEARCH_PRIORITIES, foreground_frvp_model, frvp_research_priority,
)
from ote_live.models.ict_research import ICT_RESEARCH_PRIORITIES, ict_research_priority
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

FRVP_SETUP_REFERENCE_LINES = (
    "S1 Value Edge Fade: Balanced profile/open auction; fade VAH for shorts or VAL for longs when the touch/re-entry is not real displacement.",
    "S2 Breakout Retest: After a real VAH/VAL breakout, continue with the break when price retests that edge within 15 bars and holds.",
    "S3 Value-Area Breakout: High-volume displacement closes outside VAH/VAL; follow the breakout direction.",
    "S4 Failed Auction: Brief push outside value plus sweep, then quiet re-entry back into value; fade the failed break.",
    "S5 LVN Burst: Displacement fires near the nearest LVN, within 0.30 ATR; follow the move through thin volume.",
    "S6 HVN Magnet: Balanced, quiet, inside-value session with one HVN clearly closer than the other; trade toward the nearer HVN.",
)
FRVP_LEVEL_REFERENCE_LINES = (
    "POC: Session volume point of control; main acceptance/magnet price.",
    "VAH: Value area high; upper accepted-volume boundary and breakout/fade reference.",
    "VAL: Value area low; lower accepted-volume boundary and breakout/fade reference.",
    "IB High: Initial balance high; early-session upper range reference.",
    "IB Low: Initial balance low; early-session lower range reference.",
    "Naked VPOC +: Untested prior VPOC above price; possible upside magnet/target.",
    "Naked VPOC -: Untested prior VPOC below price; possible downside magnet/target.",
    "Setup markers: FRVP fired bars; green triangle is long, red triangle is short.",
)
ICT_SETUP_REFERENCE_LINES = (
    "Session Open Manipulation Pre IB: RTH sweep before IB completes; long after PDL/ONL raid, short after PDH/ONH raid.",
    "Session Open Manipulation Post IB: RTH sweep after IB completes; long after IB-low raid, short after IB-high raid.",
    "IFVG Reversal: Inverted FVG retest at CE, close back on the reversal side, with structure support.",
    "OB Retest After MSS: Opposite-side liquidity sweep plus MSS/CHoCH, then retest of the supportive order block.",
    "Sweep Displacement FVG: Liquidity sweep, same-side displacement-created FVG, then CE touch and zone hold.",
    "Displacement Continuation After Raid: Trend-aligned raid plus recent displacement; price stays beyond the displacement origin near support.",
    "Premium Discount Continuation: Trend-aligned pullback into OTE/premium-discount near a supportive FVG or order block.",
    "Sweep Reclaim: Liquidity level is swept and reclaimed; confidence improves with displacement, support zone, and PD alignment.",
)
ICT_LEVEL_REFERENCE_LINES = (
    "PDH: Prior RTH high; buy-side liquidity/reference.",
    "PDL: Prior RTH low; sell-side liquidity/reference.",
    "ONH: Overnight high; overnight buy-side liquidity.",
    "ONL: Overnight low; overnight sell-side liquidity.",
    "IB High: Initial balance high; post-open range high.",
    "IB Low: Initial balance low; post-open range low.",
    "PWH: Prior week high; higher-timeframe buy-side liquidity.",
    "PWL: Prior week low; higher-timeframe sell-side liquidity.",
    "Midnight Open: Midnight session open; intraday bias/reference line.",
    "08:30 Open: US data/cash-session reference open.",
    "Session VWAP: Session fair-value line.",
    "DOL Up: Nearest draw-on-liquidity target above price.",
    "DOL Down: Nearest draw-on-liquidity target below price.",
    "Bull/Bear FVG: Fair value gap zone; lower/upper bracket the gap, CE marks the midpoint.",
    "Bull/Bear OB: Order block zone; lower/upper bracket the block, CE marks the midpoint.",
    "Setup markers: ICT fired bars; green triangle is long, red triangle is short.",
)


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
                            html.Div(id="es-chart-controls", style={"display": "block" if default_view.asset.upper() == "ES" and default_view.has_runtime_state_overlays else "none"}, children=[
                                dcc.RadioItems(id="es-chart-mode", value="setups", inline=True,
                                               labelStyle={"color": APP_TEXT_COLOR, "marginRight": "18px"}, options=[
                                    {"label": "Setups and qualified decisions", "value": "setups"},
                                    {"label": "Research: thresholds and probability streams", "value": "research"},
                                ]),
                                html.Label("Setup types"),
                                dcc.Dropdown(id="es-setup-types", options=setup_layer_options(), value=["all"], multi=True,
                                             style={"color": "#111827", "backgroundColor": "#ffffff"}),
                                dcc.Checklist(id="es-chart-layers", value=[], inline=True,
                                              labelStyle={"color": APP_TEXT_COLOR, "marginRight": "18px"}, options=[
                                    {"label": "Rejected and research setups", "value": "rejected"},
                                    {"label": "All revisions and invalidations", "value": "revisions"},
                                    {"label": "Strategy levels and zones", "value": "levels"},
                                ]),
                                html.Label("Collection"),
                                dcc.Dropdown(id="es-chart-collection", value="latest", clearable=False,
                                             style={"color": "#111827", "backgroundColor": "#ffffff"},
                                             options=[{"label": "Latest collection in chart window", "value": "latest"}]),
                                html.P(id="es-chart-status"),
                                html.Div(id="frvp-research-priorities"),
                            ]),
                            dcc.Graph(id="price-figure"),
                        ],
                    ),
                    html.Div(
                        id="model-research-panels",
                        style={"marginTop": "12px"},
                        children=[
                            _section_panel(
                                html,
                                title="Long Models",
                                subtitle="Research diagnostics: model probability and threshold history. Threshold crossings are not qualified trades.",
                                body_id="long-model-cards",
                            ),
                            _section_panel(
                                html,
                                title="Short Models",
                                subtitle="Research diagnostics: model probability and threshold history. Threshold crossings are not qualified trades.",
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
                    html.Div(
                        id="quick-reference-row",
                        style=_quick_reference_row_style(visible=default_view.has_runtime_state_overlays),
                        children=[
                            _text_panel(html, "Setup Quick Reference", "setup-reference"),
                            _text_panel(html, "Plotted Levels Quick Reference", "plotted-levels-reference"),
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
        Output("setup-reference", "children"),
        Output("plotted-levels-reference", "children"),
        Output("quick-reference-row", "style"),
        Output("es-chart-controls", "style"),
        Output("model-research-panels", "style"),
        Output("es-chart-collection", "options"),
        Output("es-chart-status", "children"),
        Output("frvp-research-priorities", "children"),
        Input("refresh-interval", "n_intervals"),
        Input("dashboard-location", "search"),
        Input("dashboard-tabs", "value"),
        Input("es-chart-mode", "value"),
        Input("es-setup-types", "value"),
        Input("es-chart-layers", "value"),
        Input("es-chart-collection", "value"),
    )
    def _refresh(_n_intervals: int, location_search: str | None, active_view_id: str | None,
                 chart_mode="setups", setup_types=None, chart_layers=None, chart_collection="latest"):
        view = view_by_id.get(str(active_view_id), default_view)
        es_chart = view.asset.upper() == "ES" and view.has_runtime_state_overlays
        show_model_research = not es_chart or chart_mode == "research"
        chart_layers = chart_layers or []
        controlled_frvp_requested = bool(
            view.view_id == "FRVP"
            and view.registry_path is not None
            and is_frvp_paper_signal_bundle_path_set(
                view.long_runtime_manifest_path,
                view.short_runtime_manifest_path,
                view.registry_path,
            )
        )
        controlled_frvp_valid = bool(
            controlled_frvp_requested
            and validate_frvp_paper_signal_bundle_contents(
                view.long_runtime_manifest_path,
                view.short_runtime_manifest_path,
                view.registry_path,
            )
        )
        manifest_models = (
            _load_manifest_models(
                long_runtime_manifest_path=view.long_runtime_manifest_path,
                short_runtime_manifest_path=view.short_runtime_manifest_path,
            )
            if not controlled_frvp_requested or controlled_frvp_valid
            else {"long": (), "short": ()}
        )
        manifest_model_ids = tuple(
            model.model_id
            for direction_models in manifest_models.values()
            for model in direction_models
        )
        manifest_hashes = tuple(
            _runtime_manifest_hash(model)
            for direction_models in manifest_models.values()
            for model in direction_models
        )
        active_manifest_hashes = tuple(
            _runtime_manifest_hash(model)
            for direction_models in manifest_models.values()
            for model in direction_models
            if model.status == "active"
        )
        history_manifest_hashes = (
            manifest_hashes
            if controlled_frvp_valid
            else () if controlled_frvp_requested else None
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
            runtime_manifest_hashes=history_manifest_hashes,
            limit=max(signal_limit, 240),
        )
        if controlled_frvp_valid:
            frvp_markouts = fetch_frvp_paper_signal_markouts(
                store,
                bundle_id=FRVP_PAPER_SIGNAL_BUNDLE_ID,
                runtime_manifest_hashes=active_manifest_hashes,
            )
            paper_markout_text = _format_frvp_paper_markout_summary(
                summarize_frvp_paper_signal_markouts(frvp_markouts)
            )
        elif controlled_frvp_requested:
            paper_markout_text = (
                "Unavailable: controlled FRVP bundle content verification failed"
            )
        else:
            from ote_live.storage.collection import MixedCollectionError

            try:
                markouts = compute_signal_markouts(
                    store,
                    audit_repository,
                    model_ids=manifest_model_ids or None,
                    limit=signal_limit,
                )
                performance = summarize_signal_markouts(markouts)
                if performance.avg_markout_pips is None:
                    paper_markout_text = "No completed markouts yet"
                else:
                    version = markouts.iloc[0]["collection_version"]
                    paper_markout_text = (
                        f"avg {performance.avg_markout_pips:.2f} pips | "
                        f"cum {performance.cumulative_markout_pips:.2f} pips | "
                        f"win {performance.win_rate:.1%} | collection {version}"
                    )
            except MixedCollectionError:
                paper_markout_text = "Unavailable: mixed collection versions; select a collection in the research report"
        health_summary = build_health_summary(
            store,
            audit_repository,
            asset=view.asset,
            timeframe=view.timeframe,
            runtime_manifest_hashes=history_manifest_hashes,
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
        if controlled_frvp_valid:
            recent_activity_lines = _build_controlled_frvp_signal_history_lines(
                signals
            )
        elif controlled_frvp_requested:
            recent_activity_lines = [
                "Controlled FRVP signal history unavailable: bundle verification failed."
            ]
        else:
            recent_activity_lines = _build_recent_activity_lines(
                signals,
                runtime_state=runtime_state,
                enable_frvp_overlays=view.enable_frvp_overlays,
                enable_ict_overlays=view.enable_ict_overlays,
            )
        setup_reference_text, levels_reference_text, show_quick_reference = (
            _build_quick_reference_text(view)
        )

        chart_data = None
        chart_status = ""
        if es_chart:
            strategy = "FRVP" if view.enable_frvp_overlays else "ICT"
            chart_data = fetch_es_chart_data(
                store, bars=bars, asset=view.asset, timeframe=view.timeframe, strategy=strategy,
                model_ids=manifest_model_ids, collection_version=chart_collection,
                runtime_manifest_hashes=history_manifest_hashes,
            )
            if strategy == "FRVP" and chart_mode != "research":
                chart_data = replace(chart_data, decisions=[
                    row for row in chart_data.decisions if foreground_frvp_model(row["model_id"])
                ])
            if strategy == "ICT" and chart_mode != "research":
                chart_data = replace(chart_data, decisions=[
                    row for row in chart_data.decisions
                    if (priority := ict_research_priority(row["model_id"])) and priority.foreground
                ])
            price_figure = build_es_price_figure(
                bars, data=chart_data, strategy=strategy, timeframe=view.timeframe,
                runtime_state=runtime_state, mode=chart_mode, setup_types=setup_types, layers=chart_layers,
            )
            chart_status = (
                f"Collection: {chart_data.collection_version or 'none in window'}. "
                "Triangles: setups; open marks: provisional/rejected/research; diamonds: revised; X: invalidated; stars: qualified decisions. "
                "Hover for rule confidence, model probability and actual observation delay. "
                "Qualified collection remains gated by the existing prerequisites."
            )
            selected_setups = visible_setups(chart_data, strategy=strategy, setup_types=setup_types, layers=chart_layers)
            qualified = matching_qualified_decisions(chart_data, selected_setups)
            latest_signal_text = (qualified[-1]["prediction"]["prediction_recorded_at_utc"] if qualified
                                  else "No matching qualified decisions in chart window")
            # Full persisted diagnostics belong to Research; no off-setup scores in the default activity panel.
            recent_activity_lines = [unescape(setup_hover(r, view.timeframe).replace("<br>", " | "))
                                     for r in selected_setups[-8:]]
            activity_decisions = chart_data.decisions if chart_mode == "research" else qualified
            recent_activity_lines += [unescape(decision_hover(r, view.timeframe).replace("<br>", " | "))
                                      for r in activity_decisions[-8:]]
            if not recent_activity_lines:
                recent_activity_lines = ["No setup observations or matching qualified decisions in this collection/window."]
            if controlled_frvp_requested:
                chart_status += (" Setup observations are strategy-wide; model decisions use the verified bundle scope."
                                 if controlled_frvp_valid else " Setup observations are strategy-wide; bundle decisions are unavailable.")
            else:
                paper_markout_text = "Event markouts are diagnostics; simulated outcomes remain in the research ledger."
        else:
            price_figure = _build_view_price_figure(bars=bars, signals=signals, runtime_state=runtime_state, view=view)

        return (
            latest_bar_text,
            latest_signal_text,
            heartbeat_text,
            paper_markout_text,
            ibkr_feed_text,
            (
                f"{view.asset} {view.timeframe} FRVP INVALID controlled-bundle "
                "content (controlled label refused)"
                if controlled_frvp_requested and not controlled_frvp_valid
                else "ES ICT research-only; required feature contracts and B2 artifact-lineage review pending"
                if es_chart and view.enable_ict_overlays
                else view.description
                or f"{view.asset} {view.timeframe} live operator view"
            ),
            price_figure,
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
                scope_to_runtime_manifest=controlled_frvp_valid,
                setup_events=() if es_chart else _runtime_setup_events(runtime_state),
                collection_version=(chart_data.collection_version or "__no_collection_in_window__") if chart_data else None,
                asset=view.asset if es_chart else None,
            ) if show_model_research else [],
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
                scope_to_runtime_manifest=controlled_frvp_valid,
                setup_events=() if es_chart else _runtime_setup_events(runtime_state),
                collection_version=(chart_data.collection_version or "__no_collection_in_window__") if chart_data else None,
                asset=view.asset if es_chart else None,
            ) if show_model_research else [],
            setup_reference_text,
            levels_reference_text,
            _quick_reference_row_style(visible=show_quick_reference),
            {"display": "block" if es_chart else "none"},
            {"marginTop": "12px", "display": "block" if show_model_research else "none"},
            chart_data.collection_options if chart_data else [{"label": "Latest collection in chart window", "value": "latest"}],
            chart_status,
            _build_frvp_research_priorities(html, manifest_models, chart_data)
            if es_chart and view.enable_frvp_overlays else
            _build_ict_research_priorities(html, manifest_models)
            if es_chart and view.enable_ict_overlays else [],
        )

    return app


def _build_frvp_research_priorities(html, manifest_models, chart_data):
    """Default-view roster without raw scores or inferred trade opportunities."""
    models = {m.model_id: m for group in manifest_models.values() for m in group}
    rows = []
    for model_id, priority in FRVP_RESEARCH_PRIORITIES.items():
        if not priority.foreground or model_id not in models:
            continue
        # Count immutable selected opportunities, never short setups for a long specialist.
        route = infer_setup_model_route(model_id)
        coverage = ""
        if route is not None and chart_data is not None:
            matches = [row for row in visible_setups(chart_data, strategy="FRVP")
                       if str(row["setup_type"]) == route.setup_type
                       and row["setup_side"] == route.expected_side
                       and row["event_kind"] != "invalidated"]
            coverage = f" Selected matching long setups in this collection/window: {len(matches)} (not resolved outcomes)."
        rows.append(html.Div([
            html.Strong(f"{model_id} — {priority.label}"),
            html.Div(priority.note + coverage),
        ], style={"marginBottom": "10px"}))
    return [html.H4("FRVP shadow research priorities"), *rows, html.P(
        "Research priority grants no promotion or execution authority. Input, policy and baseline gates still apply. "
        "Background models and paused S4 diagnostics are available in Research; short reversal remains retired."
    )]


def _build_ict_research_priorities(html, manifest_models):
    models = {m.model_id for group in manifest_models.values() for m in group}
    rows = [html.Div([
        html.Strong(f"{model_id} — {priority.label}"), html.Div(priority.note),
    ], style={"marginBottom": "10px"})
        for model_id, priority in ICT_RESEARCH_PRIORITIES.items()
        if priority.tier == "priority_pending_review" and model_id in models]
    return [html.H4("ICT research-only roster"), html.P(
        "All ICT models remain diagnostic-only pending required feature contracts and B2 artifact-lineage review. "
        "A corrected live feature path alone does not verify trained artifacts."
    ), *rows, html.P(
        "After review, long continuation and the short premium/discount specialist have focused comparison priority. "
        "Other families, meta models, IFVG and sweep specialists remain background comparisons in Research. "
        "Existing concentration, promotion, baseline and paper-trial restrictions remain; no activation is authorized."
    )]


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


def _quick_reference_row_style(*, visible: bool) -> dict[str, str]:
    style = {
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "12px",
        "marginTop": "12px",
    }
    if not visible:
        style["display"] = "none"
    return style


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


def _runtime_manifest_hash(manifest) -> str:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _format_frvp_paper_markout_summary(summary) -> str:
    if summary.avg_net_ticks is None:
        if summary.event_count:
            return (
                f"120-bar ES net ticks after friction | {summary.open_count} open | "
                "no completed markouts yet"
            )
        return "No controlled FRVP 120-bar ES markouts yet"
    return (
        f"120-bar ES | avg {summary.avg_net_ticks:.2f} net ticks after friction | "
        f"cum {summary.cumulative_net_ticks:.2f} net ticks | "
        f"win {summary.win_rate:.1%} | {summary.completed_count} complete / "
        f"{summary.open_count} open"
    )


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
    scope_to_runtime_manifest: bool = False,
    setup_events: Sequence[dict] = (),
    collection_version: str | None = None,
    asset: str | None = None,
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
    if asset == "ES":
        ranks = {model_id: rank for rank, model_id in enumerate((*FRVP_RESEARCH_PRIORITIES, *ICT_RESEARCH_PRIORITIES))}
        ordered_models = sorted(ordered_models, key=lambda m: ranks.get(m.model_id, len(ranks)))
    cards: list[object] = []
    confidence_history_limit = _resolve_confidence_history_fetch_limit(
        timeframe=timeframe,
        history_limit=history_limit,
        lookback_hours=lookback_hours,
    )
    for model_manifest in ordered_models:
        runtime_manifest_hashes = (
            (_runtime_manifest_hash(model_manifest),)
            if scope_to_runtime_manifest
            else None
        )
        latest_signal = fetch_recent_signals(
            audit_repository,
            model_ids=(model_manifest.model_id,),
            runtime_manifest_hashes=runtime_manifest_hashes,
            limit=1,
            collection_version=collection_version,
            asset=asset,
            timeframe=timeframe if asset else None,
        )
        confidence_history = fetch_confidence_history(
            audit_repository,
            model_id=model_manifest.model_id,
            direction=model_manifest.direction,
            runtime_manifest_hashes=runtime_manifest_hashes,
            limit=confidence_history_limit,
            collection_version=collection_version,
            asset=asset,
            timeframe=timeframe if asset else None,
        )
        full_confidence = confidence_history.copy() if asset == "ES" else _prepare_dashboard_confidence_history(
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
        ict_priority = ict_research_priority(model_manifest.model_id) if asset == "ES" else None
        if ict_priority is not None:
            has_active_weight = False

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
                                        _format_model_descriptor(model_manifest),
                                        style={"marginTop": "4px", "fontSize": "12px", "color": APP_MUTED_TEXT_COLOR},
                                    ),
                                    html.Div(
                                        f"{priority.label}: {priority.note}",
                                        style={"marginTop": "6px", "fontSize": "12px", "color": APP_TEXT_COLOR},
                                    ) if asset == "ES" and (priority := frvp_research_priority(model_manifest.model_id) or ict_priority) else None,
                                    html.Div(
                                        _format_model_metadata_line(model_manifest),
                                        style={
                                            "marginTop": "4px",
                                            "fontSize": "11px",
                                            "lineHeight": "1.4",
                                            "color": APP_MUTED_TEXT_COLOR,
                                            "wordBreak": "break-word",
                                        },
                                    ),
                                ]
                            ),
                            _build_model_status_badges(
                                html,
                                status="research-only" if ict_priority is not None else model_manifest.status,
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
                            setup_events=list(setup_events),
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


def _runtime_setup_events(runtime_state: dict | None) -> tuple[dict, ...]:
    if runtime_state is None:
        return ()
    return tuple(
        dict(item)
        for item in runtime_state.get("recent_setups") or ()
        if isinstance(item, dict)
    )


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
    setup_route = infer_setup_model_route(model_id)
    if setup_route is not None:
        return setup_route.display_label
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


def _format_model_descriptor(model_manifest) -> str:
    setup_route = infer_setup_model_route(model_manifest.model_id)
    direction = str(getattr(model_manifest, "direction", "unknown"))
    backend = str(getattr(model_manifest, "backend", "unknown"))
    calibration = str(getattr(model_manifest, "calibration_method", "unknown"))
    if setup_route is not None:
        return (
            f"{setup_route.display_label} | {direction} | {backend} "
            f"v{setup_route.version} | calib={calibration}"
        )
    version = _extract_model_version(model_manifest.model_id)
    version_text = f" v{version}" if version is not None else ""
    return f"{direction} | {backend}{version_text} | calib={calibration}"


def _format_model_metadata_line(model_manifest) -> str:
    live_policy = getattr(model_manifest, "live_policy", None)
    lineage = getattr(live_policy, "lineage", None)
    policy_status = str(getattr(live_policy, "policy_status", "unknown"))
    match_type = str(getattr(lineage, "source_match_type", "unknown"))
    artifact_refs = getattr(model_manifest, "artifact_references", None)
    artifact_dir = str(getattr(artifact_refs, "artifact_dir", "unknown"))
    threshold = _format_threshold_label(model_manifest)
    return (
        f"policy={policy_status}/{match_type} | threshold={threshold} | "
        f"artifact={artifact_dir}"
    )


def _extract_model_version(model_id: str) -> int | None:
    match = re.search(r"_v(?P<version>\d+)\b", str(model_id))
    if match is None:
        return None
    return int(match.group("version"))


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


def _build_quick_reference_text(view: DashboardViewConfig) -> tuple[str, str, bool]:
    if view.enable_frvp_overlays:
        return (
            "\n".join(FRVP_SETUP_REFERENCE_LINES),
            "\n".join(FRVP_LEVEL_REFERENCE_LINES),
            True,
        )
    if view.enable_ict_overlays:
        return (
            "\n".join(ICT_SETUP_REFERENCE_LINES),
            "\n".join(ICT_LEVEL_REFERENCE_LINES),
            True,
        )
    return "", "", False


def _build_controlled_frvp_signal_history_lines(signals: pd.DataFrame) -> list[str]:
    """Render only exact-bundle decisions; generic FRVP setup state is unscoped."""

    if signals.empty:
        return ["No controlled FRVP model decisions yet."]
    return [
        (
            f"{row.timestamp.isoformat()} | {row.model_id} | "
            f"{row.direction} {row.decision} | "
            f"p={_format_probability_value(row.probability)} | "
            f"thr={_format_probability_value(getattr(row, 'threshold', None))}"
        )
        for row in signals.tail(12).itertuples(index=False)
    ]


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
