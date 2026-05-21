from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.dashboard.app import (
    _build_model_confidence_panels,
    _confidence_title,
    _override_signal_selection_from_query,
    _resolve_dashboard_model_selection,
    create_dashboard_app,
)
from ote_live.dashboard.charts import build_price_signal_figure
from ote_live.models.loaders import load_direction_runtime_manifest
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore

LONG_RUNTIME_MANIFEST_PATH = ROOT / "ote_live" / "runtime_manifests" / "live_runtime_manifest_long.json"


def test_resolve_dashboard_model_selection_stays_pinned_to_configured_primary() -> None:
    tmp_root = ROOT / "tmp" / "ote_live_dashboard_app_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    with SQLiteLiveDataStore(db_path) as store:
        audit = LiveAuditRepository(store)
        _record_signal(
            audit,
            model_id="long_ote_tcn_v1_candidate",
            direction="long",
            timestamp=datetime(2026, 4, 7, 3, 50, tzinfo=UTC),
        )

        selection = _resolve_dashboard_model_selection(
            audit,
            configured_model_id="long_ote_tcn_v2_candidate",
            direction="long",
            manifest_model_ids=(
                "long_ote_tcn_v2_candidate",
                "long_ote_tcn_v1_candidate",
                "long_ote_xgb_v2_candidate",
            ),
        )

    assert selection.configured_model_id == "long_ote_tcn_v2_candidate"
    assert selection.resolved_model_id == "long_ote_tcn_v2_candidate"
    assert selection.used_fallback is False
    assert _confidence_title("Long Primary Confidence", selection) == "Long Primary Confidence | long_ote_tcn_v2_candidate"


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
        if item.model_id == "long_ote_tcn_v2_candidate"
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
    def __init__(self, *, trace_names: tuple[str, ...], confidence, title: str | None) -> None:
        self.data = tuple(_FakeTrace(name) for name in trace_names)
        self.confidence = confidence
        self.title = title


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
