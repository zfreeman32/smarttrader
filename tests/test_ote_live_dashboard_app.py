from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.dashboard.app import (
    _build_model_confidence_panels,
    _build_quick_reference_text,
    _build_recent_activity_lines,
    _confidence_title,
    _override_signal_selection_from_query,
    _resolve_confidence_history_fetch_limit,
    _resolve_dashboard_model_selection,
    create_dashboard_app,
)
from ote_live.dashboard.charts import build_confidence_figure, build_price_signal_figure
from ote_live.dashboard.view_registry import DashboardViewConfig
from ote_live.models.loaders import load_direction_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

LONG_RUNTIME_MANIFEST_PATH = ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_long.json"
FRVP_LONG_RUNTIME_MANIFEST_PATH = (
    ROOT
    / "ote_live"
    / "runtime_manifests"
    / "frvp_es_shadow_20260721"
    / "live_runtime_manifest_long.json"
)


def test_resolve_dashboard_model_selection_stays_pinned_to_configured_primary() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        _record_signal(
            audit,
            model_id="long_ote_union_tcn_candidate_20260523",
            direction="long",
            timestamp=datetime(2026, 4, 7, 3, 50, tzinfo=UTC),
        )

        selection = _resolve_dashboard_model_selection(
            audit,
            configured_model_id="long_reversal_tcn_v2_20260525_narrow48",
            direction="long",
            manifest_model_ids=(
                "long_reversal_tcn_v2_20260525_narrow48",
                "long_ote_union_tcn_candidate_20260523",
                "long_ote_meta_tcn_champion",
            ),
        )

    assert selection.configured_model_id == "long_reversal_tcn_v2_20260525_narrow48"
    assert selection.resolved_model_id == "long_reversal_tcn_v2_20260525_narrow48"
    assert selection.used_fallback is False
    assert (
        _confidence_title("Long Primary Confidence", selection)
        == "Long Primary Confidence | long_reversal_tcn_v2_20260525_narrow48"
    )


def test_override_signal_selection_from_query_routes_by_signal_direction() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        long_signal_id = _record_signal(
            audit,
            model_id="long_ote_tcn_v1_candidate",
            direction="long",
            timestamp=datetime(2026, 4, 7, 3, 50, tzinfo=UTC),
        )

        resolved_long, resolved_short = _override_signal_selection_from_query(
            audit,
            query_signal_id=long_signal_id,
            long_signal_id=None,
            short_signal_id=None,
        )

    assert resolved_long == long_signal_id
    assert resolved_short is None


def test_build_model_confidence_panels_renders_probability_and_threshold_graph(monkeypatch) -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    direction_manifest = load_direction_runtime_manifest(LONG_RUNTIME_MANIFEST_PATH)
    model_manifest = next(
        item
        for item in direction_manifest.models
        if item.model_id == "long_reversal_tcn_v2_20260525_narrow48"
    )
    monkeypatch.setattr(
        "ote_live.dashboard.app.build_confidence_figure",
        lambda confidence, **kwargs: _FakeFigure(
            trace_names=("Calibrated probability", "Active threshold"),
            confidence=confidence,
            title=kwargs.get("title"),
        ),
    )

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        _record_signal(
            audit,
            model_id=model_manifest.model_id,
            direction="long",
            timestamp=datetime(2026, 4, 7, 3, 50, tzinfo=UTC),
        )

        panels = _build_model_confidence_panels(
            _FakeComponentFactory(),
            _FakeComponentFactory(),
            audit,
            models=(model_manifest,),
            history_limit=20,
            timeframe="5m",
            focus_model_id=model_manifest.model_id,
        )

    assert len(panels) == 1
    graph = next(
        child
        for child in panels[0]["children"]
        if child["tag"] == "Graph"
    )
    trace_names = [str(trace.name) for trace in graph["figure"].data]
    assert "Calibrated probability" in trace_names
    assert any("threshold" in name.lower() for name in trace_names)


def test_build_confidence_figure_shows_configured_threshold_while_awaiting_prediction() -> None:
    figure = build_confidence_figure(
        pd.DataFrame(),
        fallback_threshold=0.6,
    )

    assert len(figure.data) == 0
    assert len(figure.layout.shapes) == 1
    assert float(figure.layout.shapes[0].y0) == pytest.approx(0.6)
    assert "Configured threshold: 0.6000" in figure.layout.annotations[0].text
    assert "Awaiting first persisted probability" in figure.layout.annotations[0].text


def test_build_confidence_figure_adds_setup_markers_at_exact_timestamps() -> None:
    setup_time = datetime(2026, 4, 7, 3, 50, tzinfo=UTC)
    confidence = pd.DataFrame(
        [
            _confidence_row(datetime(2026, 4, 7, 3, 45, tzinfo=UTC), 0.42),
            _confidence_row(setup_time, 0.68),
        ]
    )

    figure = build_confidence_figure(
        confidence,
        setup_events=[
            {
                "timestamp_utc": setup_time.isoformat(),
                "setup_side": 1,
                "label": "S4 Failed Auction Long",
                "confidence": 0.81,
            }
        ],
    )

    setup_shapes = [
        shape for shape in figure.layout.shapes if shape.type == "line" and shape.yref == "paper"
    ]
    assert len(setup_shapes) == 1
    assert setup_shapes[0].x0 == setup_time
    assert setup_shapes[0].x1 == setup_time
    assert setup_shapes[0].y0 == 0
    assert setup_shapes[0].y1 == 1
    assert figure.layout.annotations[0].text == "S4"
    assert "S4 Failed Auction Long" in figure.data[-1].text[0]
    assert setup_time in list(figure.data[-1].x)


def test_build_confidence_figure_retains_multiple_visible_setup_markers() -> None:
    first = datetime(2026, 4, 7, 3, 45, tzinfo=UTC)
    second = datetime(2026, 4, 7, 3, 50, tzinfo=UTC)
    confidence = pd.DataFrame(
        [
            _confidence_row(first, 0.51),
            _confidence_row(second, 0.73),
        ]
    )

    figure = build_confidence_figure(
        confidence,
        xaxis_range=(first, second),
        setup_events=[
            {"timestamp_utc": first.isoformat(), "setup_side": 1, "label": "S1 Value Edge Fade Long"},
            {"timestamp_utc": second.isoformat(), "setup_side": -1, "label": "S4 Failed Auction Short"},
            {
                "timestamp_utc": datetime(2026, 4, 7, 2, 0, tzinfo=UTC).isoformat(),
                "setup_side": 1,
                "label": "Outside Window Long",
            },
        ],
    )

    marker_shapes = [
        shape for shape in figure.layout.shapes if shape.type == "line" and shape.yref == "paper"
    ]
    assert [shape.x0 for shape in marker_shapes] == [first, second]
    assert list(figure.data[-1].x) == [first, second]


def test_setup_marker_colors_are_deterministic_and_side_dash_distinguishes_direction() -> None:
    timestamp = datetime(2026, 4, 7, 3, 50, tzinfo=UTC)
    confidence = pd.DataFrame([_confidence_row(timestamp, 0.68)])
    events = [
        {
            "timestamp_utc": timestamp.isoformat(),
            "setup_side": 1,
            "label": "Sweep Reclaim Long",
        },
        {
            "timestamp_utc": (timestamp + pd.Timedelta(minutes=5)).isoformat(),
            "setup_side": -1,
            "label": "Sweep Reclaim Short",
        },
    ]

    first_figure = build_confidence_figure(confidence, setup_events=events)
    second_figure = build_confidence_figure(confidence, setup_events=events)

    first_shapes = list(first_figure.layout.shapes)
    second_shapes = list(second_figure.layout.shapes)
    assert [shape.line.color for shape in first_shapes] == [
        shape.line.color for shape in second_shapes
    ]
    assert first_shapes[0].line.color == first_shapes[1].line.color
    assert first_shapes[0].line.dash == "solid"
    assert first_shapes[1].line.dash == "dot"


def test_repeated_refresh_payload_does_not_duplicate_setup_markers() -> None:
    timestamp = datetime(2026, 4, 7, 3, 50, tzinfo=UTC)
    confidence = pd.DataFrame([_confidence_row(timestamp, 0.68)])
    duplicated_event = {
        "timestamp_utc": timestamp.isoformat(),
        "setup_side": 1,
        "label": "S2 Breakout Retest Long",
    }

    figure = build_confidence_figure(
        confidence,
        setup_events=[duplicated_event, duplicated_event.copy()],
    )

    marker_shapes = [
        shape for shape in figure.layout.shapes if shape.type == "line" and shape.yref == "paper"
    ]
    assert len(marker_shapes) == 1
    assert list(figure.data[-1].x) == [timestamp]


def test_frvp_model_card_shows_manifest_threshold_before_first_prediction() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    direction_manifest = load_direction_runtime_manifest(
        FRVP_LONG_RUNTIME_MANIFEST_PATH
    )
    model_manifest = next(
        item
        for item in direction_manifest.models
        if item.model_id == "frvp_long_reversal_xgb_v1"
    )

    with SQLiteLiveDataStore(db_path) as store:
        panels = _build_model_confidence_panels(
            _FakeComponentFactory(),
            _FakeComponentFactory(),
            LiveAuditRepository(store),
            models=(model_manifest,),
            history_limit=20,
            timeframe="5m",
        )

    text_children = [
        child["children"]
        for child in panels[0]["children"]
        if child["tag"] == "Div" and isinstance(child["children"], str)
    ]
    assert any("p=n/a | thr=0.6000" in text for text in text_children)
    assert any(
        "configured threshold shown" in text.lower()
        for text in text_children
    )
    graph = next(
        child for child in panels[0]["children"] if child["tag"] == "Graph"
    )
    assert float(graph["figure"].layout.shapes[0].y0) == pytest.approx(0.6)


def test_active_weight_model_card_is_visually_distinct_but_remains_shadow() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    direction_manifest = load_direction_runtime_manifest(
        FRVP_LONG_RUNTIME_MANIFEST_PATH
    )
    model_manifest = next(
        item
        for item in direction_manifest.models
        if item.model_id == "frvp_long_reversal_xgb_v1"
    )

    with SQLiteLiveDataStore(db_path) as store:
        panels = _build_model_confidence_panels(
            _FakeComponentFactory(),
            _FakeComponentFactory(),
            LiveAuditRepository(store),
            models=(model_manifest,),
            history_limit=20,
            timeframe="5m",
            active_weight_model_ids=(model_manifest.model_id,),
        )

    badge_labels = _flatten_span_labels(panels)
    assert "ACTIVE WEIGHT" in badge_labels
    assert "SHADOW" in badge_labels
    assert model_manifest.status == "candidate"


def test_build_model_confidence_panels_trims_chart_to_recent_lookback_window(monkeypatch) -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"
    direction_manifest = load_direction_runtime_manifest(LONG_RUNTIME_MANIFEST_PATH)
    model_manifest = next(
        item
        for item in direction_manifest.models
        if item.model_id == "long_reversal_tcn_v2_20260525_narrow48"
    )
    monkeypatch.setattr(
        "ote_live.dashboard.app.build_confidence_figure",
        lambda confidence, **kwargs: _FakeFigure(
            trace_names=("Calibrated probability", "Active threshold"),
            confidence=confidence,
            title=kwargs.get("title"),
            xaxis_range=kwargs.get("xaxis_range"),
        ),
    )

    oldest = datetime(2026, 4, 5, 3, 50, tzinfo=UTC)
    near_latest = datetime(2026, 4, 6, 4, 0, tzinfo=UTC)
    latest = datetime(2026, 4, 7, 3, 50, tzinfo=UTC)

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        _record_signal(
            audit,
            model_id=model_manifest.model_id,
            direction="long",
            timestamp=oldest,
        )
        _record_signal(
            audit,
            model_id=model_manifest.model_id,
            direction="long",
            timestamp=near_latest,
        )
        _record_signal(
            audit,
            model_id=model_manifest.model_id,
            direction="long",
            timestamp=latest,
        )

        panels = _build_model_confidence_panels(
            _FakeComponentFactory(),
            _FakeComponentFactory(),
            audit,
            models=(model_manifest,),
            history_limit=20,
            timeframe="5m",
            lookback_hours=24,
            focus_model_id=model_manifest.model_id,
        )

    graph = next(
        child
        for child in panels[0]["children"]
        if child["tag"] == "Graph"
    )
    visible_timestamps = list(graph["figure"].confidence["timestamp"])
    assert visible_timestamps == [near_latest, latest]
    assert graph["figure"].xaxis_range == (
        datetime(2026, 4, 6, 3, 50, tzinfo=UTC),
        latest,
    )


def test_resolve_confidence_history_fetch_limit_scales_to_lookback_window() -> None:
    assert _resolve_confidence_history_fetch_limit(
        timeframe="5m",
        history_limit=120,
        lookback_hours=24,
    ) == 290


def test_build_price_signal_figure_only_plots_emitted_arrows_and_hides_legend(monkeypatch) -> None:
    monkeypatch.setattr("ote_live.dashboard.charts._plotly_go", lambda: _FakePlotlyGO())
    bars = pd.DataFrame(
        [
            {
                "timestamp": datetime(2026, 4, 7, 3, 45, tzinfo=UTC),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
            },
            {
                "timestamp": datetime(2026, 4, 7, 3, 50, tzinfo=UTC),
                "open": 1.105,
                "high": 1.115,
                "low": 1.095,
                "close": 1.101,
            },
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "model_id": "model_a",
                "timestamp": datetime(2026, 4, 7, 3, 45, tzinfo=UTC),
                "direction": "long",
                "decision": "emit",
                "regime": "trend",
                "bar_low": 1.09,
                "bar_high": 1.11,
                "bar_close": 1.105,
            },
            {
                "model_id": "model_b",
                "timestamp": datetime(2026, 4, 7, 3, 45, tzinfo=UTC),
                "direction": "long",
                "decision": "emit",
                "regime": "trend",
                "bar_low": 1.09,
                "bar_high": 1.11,
                "bar_close": 1.105,
            },
            {
                "model_id": "model_c",
                "timestamp": datetime(2026, 4, 7, 3, 50, tzinfo=UTC),
                "direction": "short",
                "decision": "shadow",
                "regime": "range",
                "bar_low": 1.095,
                "bar_high": 1.115,
                "bar_close": 1.101,
            },
            {
                "model_id": "model_d",
                "timestamp": datetime(2026, 4, 7, 3, 50, tzinfo=UTC),
                "direction": "short",
                "decision": "emit",
                "regime": "range",
                "bar_low": 1.095,
                "bar_high": 1.115,
                "bar_close": 1.101,
            },
        ]
    )

    figure = build_price_signal_figure(bars, signals)

    assert len(figure.data) == 3
    assert figure.layout["showlegend"] is False
    assert [trace.name for trace in figure.data[1:]] == ["long emit", "short emit"]
    assert list(figure.data[1].x) == [datetime(2026, 4, 7, 3, 45, tzinfo=UTC)]
    assert "Emitted models: 2" in figure.data[1].text[0]


def test_create_dashboard_app_places_health_and_signal_panels_below_model_sections(monkeypatch) -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    monkeypatch.setattr(
        "ote_live.dashboard.app._dash_modules",
        lambda: (
            _FakeDashApp,
            _FakeComponentFactory(),
            _FakeComponentFactory(),
            _FakeIoFactory("Input"),
            _FakeIoFactory("Output"),
        ),
    )

    app = create_dashboard_app(db_path)
    titles = _flatten_titles(app.layout)

    assert titles.index("Long Models") < titles.index("Recent Health Events")
    assert titles.index("Short Models") < titles.index("Recent Signals")


def test_create_dashboard_app_places_quick_reference_below_recent_panels(monkeypatch) -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    monkeypatch.setattr(
        "ote_live.dashboard.app._dash_modules",
        lambda: (
            _FakeDashApp,
            _FakeComponentFactory(),
            _FakeComponentFactory(),
            _FakeIoFactory("Input"),
            _FakeIoFactory("Output"),
        ),
    )

    app = create_dashboard_app(db_path)
    titles = _flatten_titles(app.layout)

    assert titles.index("Recent Health Events") < titles.index("Setup Quick Reference")
    assert titles.index("Recent Signals") < titles.index("Plotted Levels Quick Reference")


def test_build_quick_reference_text_is_view_specific() -> None:
    ote_view = DashboardViewConfig(
        view_id="OTE",
        label="OTE",
        asset="EURUSD",
        timeframe="5m",
        data_supplier="FMP",
        long_runtime_manifest_path=LONG_RUNTIME_MANIFEST_PATH,
        short_runtime_manifest_path=LONG_RUNTIME_MANIFEST_PATH,
    )
    frvp_view = DashboardViewConfig(
        view_id="FRVP",
        label="FRVP",
        asset="ES",
        timeframe="5m",
        data_supplier="IBKR",
        long_runtime_manifest_path=LONG_RUNTIME_MANIFEST_PATH,
        short_runtime_manifest_path=LONG_RUNTIME_MANIFEST_PATH,
        enable_frvp_overlays=True,
    )
    ict_view = DashboardViewConfig(
        view_id="ICT",
        label="ICT",
        asset="ES",
        timeframe="5m",
        data_supplier="IBKR",
        long_runtime_manifest_path=LONG_RUNTIME_MANIFEST_PATH,
        short_runtime_manifest_path=LONG_RUNTIME_MANIFEST_PATH,
        enable_ict_overlays=True,
    )

    setup_text, levels_text, visible = _build_quick_reference_text(ote_view)
    assert setup_text == ""
    assert levels_text == ""
    assert visible is False

    setup_text, levels_text, visible = _build_quick_reference_text(frvp_view)
    assert "S4 Failed Auction" in setup_text
    assert "Naked VPOC +" in levels_text
    assert visible is True

    setup_text, levels_text, visible = _build_quick_reference_text(ict_view)
    assert "Sweep Reclaim" in setup_text
    assert "PDH" in levels_text
    assert visible is True


def test_create_dashboard_app_renders_tab_strip_for_multiple_views(monkeypatch) -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    monkeypatch.setattr(
        "ote_live.dashboard.app._dash_modules",
        lambda: (
            _FakeDashApp,
            _FakeComponentFactory(),
            _FakeComponentFactory(),
            _FakeIoFactory("Input"),
            _FakeIoFactory("Output"),
        ),
    )

    app = create_dashboard_app(
        db_path,
        view_configs=(
            DashboardViewConfig(
                view_id="OTE",
                label="OTE",
                asset="EURUSD",
                timeframe="5m",
                data_supplier="FMP",
                long_runtime_manifest_path=LONG_RUNTIME_MANIFEST_PATH,
                short_runtime_manifest_path=LONG_RUNTIME_MANIFEST_PATH,
            ),
            DashboardViewConfig(
                view_id="FRVP",
                label="FRVP",
                asset="ES",
                timeframe="5m",
                data_supplier="IBKR",
                long_runtime_manifest_path=LONG_RUNTIME_MANIFEST_PATH,
                short_runtime_manifest_path=LONG_RUNTIME_MANIFEST_PATH,
                enable_frvp_overlays=True,
            ),
        ),
    )

    assert _flatten_tab_labels(app.layout) == ["OTE", "FRVP"]


def test_build_recent_activity_lines_uses_frvp_setup_state_when_enabled() -> None:
    lines = _build_recent_activity_lines(
        pd.DataFrame(),
        runtime_state={
            "recent_setups": [
                {
                    "timestamp_utc": "2026-07-06T15:30:00+00:00",
                    "label": "S4 Long",
                    "confidence": 0.82,
                    "bar_close": 6281.5,
                }
            ]
        },
        enable_frvp_overlays=True,
    )

    assert len(lines) == 1
    assert "S4 Long" in lines[0]


def _record_signal(
    audit: LiveAuditRepository,
    *,
    model_id: str,
    direction: str,
    timestamp: datetime,
) -> int:
    snapshot = FeatureSnapshot(
        asset="EURUSD",
        timeframe="5m",
        direction=direction,
        timestamp=timestamp,
        source_row_idx=1,
        feature_values={"feature_a": 0.42},
        valid_feature_count=1,
    )
    feature_snapshot_id = audit.record_feature_snapshot(snapshot)
    prediction = ModelPrediction(
        model_id=model_id,
        direction=direction,
        backend="xgboost",
        timestamp=timestamp,
        source_row_idx=1,
        regime="test_regime",
        raw_score=0.42,
        calibrated_probability=0.73,
        threshold_applied=0.55,
        threshold_source="global",
    )
    prediction_id = audit.record_prediction(
        prediction,
        feature_snapshot_id=feature_snapshot_id,
    )
    return audit.record_signal_decision(
        SignalDecision(
            model_id=model_id,
            direction=direction,
            timestamp=timestamp,
            source_row_idx=1,
            decision="shadow",
            probability=prediction.calibrated_probability,
            threshold=prediction.threshold_applied,
            regime=prediction.regime,
            reasons=["test_seed"],
            cooldown_bars_remaining=None,
        ),
        prediction_id=prediction_id,
    )


def _confidence_row(timestamp: datetime, probability: float) -> dict:
    return {
        "prediction_id": 1,
        "signal_decision_id": None,
        "model_id": "model_a",
        "direction": "long",
        "timestamp": timestamp,
        "regime": "test_regime",
        "raw_score": probability,
        "calibrated_probability": probability,
        "threshold_applied": 0.55,
        "threshold_source": "global",
        "decision": "shadow",
    }


class _FakeComponentFactory:
    def __getattr__(self, tag: str):
        def _component(*args, **kwargs):
            children = kwargs.pop("children", None)
            if children is None:
                if not args:
                    children = []
                elif len(args) == 1:
                    children = args[0]
                else:
                    children = list(args)
            return {
                "tag": tag,
                "children": children,
                **kwargs,
            }

        return _component


class _FakePlotlyTrace:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _FakePlotlyFigure:
    def __init__(self) -> None:
        self.data = []
        self.layout = {}

    def add_trace(self, trace) -> None:
        self.data.append(trace)

    def update_layout(self, **kwargs) -> None:
        self.layout.update(kwargs)

    def update_xaxes(self, **kwargs) -> None:
        return None

    def update_yaxes(self, **kwargs) -> None:
        return None


class _FakePlotlyGO:
    Figure = _FakePlotlyFigure
    Candlestick = _FakePlotlyTrace
    Scatter = _FakePlotlyTrace


class _FakeDashApp:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.title = ""
        self.layout = None

    def callback(self, *args, **kwargs):
        def _decorator(func):
            self._callback = func
            return func

        return _decorator


class _FakeIoFactory:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __call__(self, *args, **kwargs):
        return {
            "tag": self.tag,
            "args": args,
            "kwargs": kwargs,
        }


class _FakeTrace:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeFigure:
    def __init__(
        self,
        *,
        trace_names: tuple[str, ...],
        confidence,
        title: str | None,
        xaxis_range=None,
    ) -> None:
        self.data = tuple(_FakeTrace(name) for name in trace_names)
        self.confidence = confidence
        self.title = title
        self.xaxis_range = xaxis_range


def _flatten_titles(node) -> list[str]:
    titles: list[str] = []
    if isinstance(node, dict):
        tag = node.get("tag")
        children = node.get("children")
        if tag in {"H3", "H4"} and isinstance(children, str):
            titles.append(children)
        elif isinstance(children, list):
            for child in children:
                titles.extend(_flatten_titles(child))
        elif children is not None:
            titles.extend(_flatten_titles(children))
    elif isinstance(node, list):
        for child in node:
            titles.extend(_flatten_titles(child))
    return titles


def _flatten_tab_labels(node) -> list[str]:
    labels: list[str] = []
    if isinstance(node, dict):
        if node.get("tag") == "Tab" and "label" in node:
            labels.append(str(node["label"]))
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                labels.extend(_flatten_tab_labels(child))
        elif children is not None:
            labels.extend(_flatten_tab_labels(children))
    elif isinstance(node, list):
        for child in node:
            labels.extend(_flatten_tab_labels(child))
    return labels


def _flatten_span_labels(node) -> list[str]:
    labels: list[str] = []
    if isinstance(node, dict):
        if node.get("tag") == "Span" and isinstance(node.get("children"), str):
            labels.append(node["children"])
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                labels.extend(_flatten_span_labels(child))
        elif children is not None:
            labels.extend(_flatten_span_labels(children))
    elif isinstance(node, list):
        for child in node:
            labels.extend(_flatten_span_labels(child))
    return labels
