from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.builder import FeatureDatasetBuilder
from features.config import FeatureBuilderConfig
from ote_live.contracts.market_data import MarketBar
from ote_live.dashboard.app import _build_recent_activity_lines
from ote_live.dashboard.charts import build_ict_price_figure
from ote_live.dashboard.view_registry import (
    FRVP_PAPER_SIGNAL_LONG_RUNTIME_MANIFEST_PATH,
    FRVP_PAPER_SIGNAL_REGISTRY_PATH,
    FRVP_PAPER_SIGNAL_SHORT_RUNTIME_MANIFEST_PATH,
    build_default_dashboard_views,
)
from ote_live.dashboard.view_state import (
    DASHBOARD_VIEW_STATE_SCOPE,
    _build_ict_fvg_history_payload,
    _build_ict_level_payload,
    _build_ict_zone_payload,
    persist_ict_dashboard_state,
)
from ote_live.ingestion.signals import LiveSignalProcessor
from ote_live.storage import SQLiteLiveDataStore


def test_build_default_dashboard_views_uses_controlled_ict_paper_signal_bundle() -> None:
    view_by_id = {view.view_id: view for view in build_default_dashboard_views()}

    assert "ICT" in view_by_id
    ict_view = view_by_id["ICT"]
    assert ict_view.enable_ict_overlays is True
    assert ict_view.resolved_runtime_state_key == "ICT"
    assert ict_view.asset == "ES"
    assert ict_view.timeframe == "5m"
    assert ict_view.long_runtime_manifest_path.parent.name == (
        "ict_es_paper_signal_20260813"
    )
    assert ict_view.short_runtime_manifest_path.parent.name == (
        "ict_es_paper_signal_20260813"
    )
    assert ict_view.registry_path is not None
    assert ict_view.registry_path.name == "ict_es_paper_signal_registry_20260813.json"
    assert ict_view.active_weight_model_ids == ("ict_long_meta_xgb_v1",)
    assert "paper-signal" in ict_view.description


def test_frvp_dashboard_matches_reversal_only_controlled_bundle(monkeypatch) -> None:
    monkeypatch.setenv(
        "FRVP_LIVE_LONG_RUNTIME_MANIFEST_PATH",
        str(FRVP_PAPER_SIGNAL_LONG_RUNTIME_MANIFEST_PATH),
    )
    monkeypatch.setenv(
        "FRVP_LIVE_SHORT_RUNTIME_MANIFEST_PATH",
        str(FRVP_PAPER_SIGNAL_SHORT_RUNTIME_MANIFEST_PATH),
    )
    monkeypatch.setenv(
        "FRVP_LIVE_REGISTRY_PATH",
        str(FRVP_PAPER_SIGNAL_REGISTRY_PATH),
    )

    view_by_id = {view.view_id: view for view in build_default_dashboard_views()}
    frvp_view = view_by_id["FRVP"]

    assert frvp_view.active_weight_model_ids == ("frvp_long_reversal_xgb_v1",)
    assert "controlled paper-signal" in frvp_view.description


def test_build_recent_activity_lines_uses_ict_setup_state_when_enabled() -> None:
    lines = _build_recent_activity_lines(
        pd.DataFrame(),
        runtime_state={
            "recent_setups": [
                {
                    "timestamp_utc": "2026-07-13T14:35:00+00:00",
                    "label": "Sweep Reclaim Long",
                    "confidence": 0.78,
                    "anchor_level": 6284.25,
                    "target_reference": 6291.5,
                }
            ]
        },
        enable_frvp_overlays=False,
        enable_ict_overlays=True,
    )

    assert len(lines) == 1
    assert "Sweep Reclaim Long" in lines[0]
    assert "anchor=6284.2500" in lines[0]


def test_persist_ict_dashboard_state_records_levels_zones_and_recent_setup(monkeypatch) -> None:
    tmp_root = ROOT / "tmp" / "ote_live_ict_dashboard_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    monkeypatch.setattr(
        "ote_live.dashboard.view_state.detect_ict_setups",
        lambda frame: pd.DataFrame(
            [
                {
                    "fired": True,
                    "setup_type": "sweep_reclaim",
                    "setup_family": "reversal",
                    "setup_side": 1,
                    "confidence": 0.81,
                    "anchor_level": 6284.25,
                    "entry_price": 6286.0,
                    "stop_reference": 6281.75,
                    "target_reference": 6291.5,
                    "reference_level": 6284.25,
                    "reference_level_type": "prior_rth_low",
                    "sweep_type": "sell_side",
                    "htf_context": "aligned_bull",
                    "ce_price": 6285.0,
                    "order_block_id": 17,
                    "displacement_id": 44,
                    "session_phase": 2,
                }
            ]
        ),
    )

    policy_frame = pd.DataFrame(
        [
            {
                "close": 6286.0,
                "low": 6283.75,
                "high": 6287.25,
                "ict_prior_rth_high": 6301.0,
                "ict_prior_rth_low": 6279.75,
                "ict_overnight_high": 6295.25,
                "ict_overnight_low": 6276.5,
                "ict_ib_high": 6289.0,
                "ict_ib_low": 6281.0,
                "ict_prior_week_high": 6310.5,
                "ict_prior_week_low": 6225.25,
                "ict_midnight_open": 6280.0,
                "ict_open_0830": 6282.25,
                "ict_session_vwap": 6285.5,
                "ict_dol_level_up": 6291.5,
                "ict_dol_level_down": 6271.0,
                "ict_latest_swing_high": 6293.0,
                "ict_latest_swing_low": 6278.0,
                "ict_nearest_bull_fvg_lower": 6283.5,
                "ict_nearest_bull_fvg_upper": 6286.5,
                "ict_nearest_bull_fvg_ce": 6285.0,
                "ict_nearest_bull_fvg_id": 11,
                "ict_nearest_bull_fvg_is_ifvg": 0,
                "ict_nearest_bull_fvg_created_by_displacement": 1,
                "ict_nearest_bear_order_block_lower": 6290.0,
                "ict_nearest_bear_order_block_upper": 6294.0,
                "ict_nearest_bear_order_block_id": 22,
                "ict_structure_state": 1,
                "ict_impulse_direction": 1,
                "ict_session_phase_code": 2,
                "ict_is_rth": 1,
                "ict_ib_complete": 1,
                "ict_in_ote_band": 1,
                "ict_premium_zone": 0,
                "ict_discount_zone": 1,
            }
        ]
    )
    bar = MarketBar(
        asset="ES",
        timeframe="5m",
        timestamp=datetime(2026, 7, 13, 14, 35, tzinfo=UTC),
        open=6284.5,
        high=6287.25,
        low=6283.75,
        close=6286.0,
        volume=1200.0,
        source="unit-test",
        symbol="ES",
        contract_symbol="ESU6",
        instrument_id=9001,
    )

    with SQLiteLiveDataStore(db_path) as store:
        persist_ict_dashboard_state(
            store,
            group_name="ICT",
            asset="ES",
            timeframe="5m",
            data_supplier="IBKR",
            policy_frame=policy_frame,
            bar=bar,
        )
        runtime_state = store.get_runtime_state(
            scope=DASHBOARD_VIEW_STATE_SCOPE,
            state_key="ICT",
        )

    assert runtime_state is not None
    assert runtime_state["levels"]["prior_rth_high"] == 6301.0
    assert runtime_state["levels"]["session_vwap"] == 6285.5
    assert runtime_state["zones"]["bull_fvg"]["ce_price"] == 6285.0
    assert runtime_state["zones"]["bear_order_block"]["upper"] == 6294.0
    assert runtime_state["recent_setups"][0]["label"] == "Sweep Reclaim Long"
    assert runtime_state["context"]["discount_zone"] is True


def test_persist_ict_dashboard_state_sanitizes_setup_price_sentinels(monkeypatch) -> None:
    tmp_root = ROOT / "tmp" / "ote_live_ict_dashboard_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / f"{uuid.uuid4().hex}.sqlite"

    monkeypatch.setattr(
        "ote_live.dashboard.view_state.detect_ict_setups",
        lambda frame: pd.DataFrame(
            [
                {
                    "fired": True,
                    "setup_type": "premium_discount_continuation",
                    "setup_family": "continuation",
                    "setup_side": 1,
                    "confidence": 0.86,
                    "anchor_level": 0.0,
                    "entry_price": 7633.75,
                    "stop_reference": -0.5669642857142857,
                    "target_reference": 7635.5,
                    "reference_level": 7635.5,
                    "reference_level_type": "dol",
                    "ce_price": 0.0,
                    "order_block_id": 0,
                    "displacement_id": 1033,
                }
            ]
        ),
    )
    policy_frame = pd.DataFrame(
        [
            {
                "close": 7633.75,
                "low": 7632.5,
                "high": 7635.0,
            }
        ]
    )
    bar = MarketBar(
        asset="ES",
        timeframe="5m",
        timestamp=datetime(2026, 8, 3, 22, 15, tzinfo=UTC),
        open=7633.5,
        high=7635.0,
        low=7632.5,
        close=7633.75,
        volume=1200.0,
        source="unit-test",
        symbol="ES",
        contract_symbol="ESU6",
        instrument_id=9001,
    )

    with SQLiteLiveDataStore(db_path) as store:
        persist_ict_dashboard_state(
            store,
            group_name="ICT",
            asset="ES",
            timeframe="5m",
            data_supplier="IBKR",
            policy_frame=policy_frame,
            bar=bar,
        )
        runtime_state = store.get_runtime_state(
            scope=DASHBOARD_VIEW_STATE_SCOPE,
            state_key="ICT",
        )

    setup = runtime_state["recent_setups"][0]
    assert setup["anchor_level"] is None
    assert setup["ce_price"] is None
    assert setup["stop_reference"] is None
    assert setup["entry_price"] == 7633.75
    assert setup["order_block_id"] is None
    assert setup["displacement_id"] == 1033


def test_ict_dashboard_state_omits_zero_price_sentinels() -> None:
    latest = pd.Series(
        {
            "ict_prior_rth_high": 7482.75,
            "ict_prior_week_high": 0.0,
            "ict_prior_week_low": 0.0,
            "ict_nearest_bull_order_block_lower": 0.0,
            "ict_nearest_bull_order_block_upper": 0.0,
            "ict_nearest_bull_order_block_id": 0,
            "ict_nearest_bear_order_block_lower": 7476.5,
            "ict_nearest_bear_order_block_upper": 7483.0,
            "ict_nearest_bear_order_block_id": 84,
        }
    )

    levels = _build_ict_level_payload(latest)
    zones = _build_ict_zone_payload(latest)

    assert levels == {"prior_rth_high": 7482.75}
    assert "bull_order_block" not in zones
    assert zones["bear_order_block"] == {
        "lower": 7476.5,
        "upper": 7483.0,
        "ce_price": 7479.75,
        "id": 84,
        "is_ifvg": False,
        "created_by_displacement": False,
    }


def test_build_ict_price_figure_renders_ict_levels_zones_and_setup_markers(monkeypatch) -> None:
    monkeypatch.setattr("ote_live.dashboard.charts._plotly_go", lambda: _FakePlotlyGO())
    bars = pd.DataFrame(
        [
            {
                "timestamp": datetime(2026, 7, 13, 14, 30, tzinfo=UTC),
                "open": 6282.0,
                "high": 6285.0,
                "low": 6281.5,
                "close": 6284.0,
            },
            {
                "timestamp": datetime(2026, 7, 13, 14, 35, tzinfo=UTC),
                "open": 6284.0,
                "high": 6287.25,
                "low": 6283.75,
                "close": 6286.0,
            },
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "model_id": "ict_long_reversal_xgb_v1",
                "timestamp": datetime(2026, 7, 13, 14, 35, tzinfo=UTC),
                "direction": "long",
                "decision": "emit",
                "regime": "strong_down_high",
                "bar_low": 6283.75,
                "bar_high": 6287.25,
                "bar_close": 6286.0,
            }
        ]
    )
    runtime_state = {
        "levels": {
            "prior_rth_high": 6301.0,
            "session_vwap": 6285.5,
        },
        "zones": {
            "bull_fvg": {
                "lower": 6283.5,
                "upper": 6286.5,
                "ce_price": 6285.0,
            },
        },
        "recent_setups": [
            {
                "timestamp_utc": "2026-07-13T14:35:00+00:00",
                "setup_side": 1,
                "label": "Sweep Reclaim Long",
                "confidence": 0.81,
                "anchor_level": 6284.25,
                "target_reference": 6291.5,
                "reference_level_type": "prior_rth_low",
                "htf_context": "aligned_bull",
            }
        ],
    }

    figure = build_ict_price_figure(
        bars,
        signals,
        runtime_state=runtime_state,
    )

    trace_names = [trace.name for trace in figure.data]
    assert "Price" in trace_names
    assert "PDH" in trace_names
    assert "Session VWAP" in trace_names
    assert "long emit" in trace_names
    assert "ICT long setup" in trace_names


def test_build_ict_price_figure_renders_retained_bounded_fvg_history(monkeypatch) -> None:
    monkeypatch.setattr("ote_live.dashboard.charts._plotly_go", lambda: _FakePlotlyGO())
    base_time = pd.Timestamp("2026-07-13T14:00:00Z")
    bars = pd.DataFrame(
        [
            {
                "timestamp": base_time + pd.Timedelta(minutes=5 * index),
                "open": 100.0 + index * 0.1,
                "high": 101.0 + index * 0.1,
                "low": 99.0 + index * 0.1,
                "close": 100.5 + index * 0.1,
            }
            for index in range(24)
        ]
    )
    runtime_state = {
        "latest_bar_timestamp_utc": bars["timestamp"].iloc[-1].isoformat(),
        "levels": {},
        "zones": {},
        "recent_setups": [],
        "fvg_history": [
            {
                "id": 1,
                "direction": 1,
                "lower": 100.0,
                "upper": 101.0,
                "ce_price": 100.5,
                "formed_index": 4,
                "formed_time": bars["timestamp"].iloc[4].isoformat(),
                "source_timeframe": "5m",
                "inversion_index": 8,
            },
            {
                "id": 2,
                "direction": -1,
                "lower": 103.0,
                "upper": 104.0,
                "ce_price": 103.5,
                "formed_index": 9,
                "formed_time": bars["timestamp"].iloc[9].isoformat(),
                "source_timeframe": "5m",
            },
            {
                "id": 3,
                "direction": 1,
                "lower": 105.0,
                "upper": 106.0,
                "ce_price": 105.5,
                "formed_index": 12,
                "formed_time": bars["timestamp"].iloc[12].isoformat(),
                "source_timeframe": "5m",
            },
        ],
    }

    figure = build_ict_price_figure(bars, pd.DataFrame(), runtime_state=runtime_state)

    rects = [shape for shape in figure.layout["shapes"] if shape["type"] == "rect"]
    assert len(rects) == 3
    assert rects[0]["x0"] == bars["timestamp"].iloc[4]
    assert rects[0]["x1"] == bars["timestamp"].iloc[8]
    assert rects[0]["y0"] == 100.0
    assert rects[0]["y1"] == 101.0
    assert rects[1]["x1"] <= bars["timestamp"].iloc[23]
    assert rects[1]["x1"] == bars["timestamp"].iloc[23]
    assert rects[2]["x1"] == bars["timestamp"].iloc[23]


def test_build_ict_price_figure_caps_fvg_history_at_exact_15_display_bars(monkeypatch) -> None:
    monkeypatch.setattr("ote_live.dashboard.charts._plotly_go", lambda: _FakePlotlyGO())
    base_time = pd.Timestamp("2026-07-13T14:00:00Z")
    bars = pd.DataFrame(
        [
            {
                "timestamp": base_time + pd.Timedelta(minutes=5 * index),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
            }
            for index in range(32)
        ]
    )
    runtime_state = {
        "latest_bar_timestamp_utc": bars["timestamp"].iloc[-1].isoformat(),
        "fvg_history": [
            {
                "id": 1,
                "direction": 1,
                "lower": 100.0,
                "upper": 101.0,
                "ce_price": 100.5,
                "formed_index": 4,
                "formed_time": bars["timestamp"].iloc[4].isoformat(),
                "source_timeframe": "5m",
            }
        ],
    }

    first = build_ict_price_figure(bars, pd.DataFrame(), runtime_state=runtime_state)
    second = build_ict_price_figure(bars, pd.DataFrame(), runtime_state=runtime_state)

    rect = next(shape for shape in first.layout["shapes"] if shape["type"] == "rect")
    assert rect["x0"] == bars["timestamp"].iloc[4]
    assert rect["x1"] == bars["timestamp"].iloc[19]
    assert [shape for shape in first.layout["shapes"] if shape["type"] == "rect"] == [
        shape for shape in second.layout["shapes"] if shape["type"] == "rect"
    ]


def test_build_ict_price_figure_shortens_fvg_history_at_first_mitigation(monkeypatch) -> None:
    monkeypatch.setattr("ote_live.dashboard.charts._plotly_go", lambda: _FakePlotlyGO())
    base_time = pd.Timestamp("2026-07-13T14:00:00Z")
    bars = pd.DataFrame(
        [
            {
                "timestamp": base_time + pd.Timedelta(minutes=5 * index),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
            }
            for index in range(24)
        ]
    )
    runtime_state = {
        "latest_bar_timestamp_utc": bars["timestamp"].iloc[-1].isoformat(),
        "fvg_history": [
            {
                "id": 1,
                "direction": 1,
                "lower": 100.0,
                "upper": 101.0,
                "ce_price": 100.5,
                "formed_index": 4,
                "formed_time": bars["timestamp"].iloc[4].isoformat(),
                "source_timeframe": "5m",
                "mitigation_index": 6,
                "mitigation_time": bars["timestamp"].iloc[6].isoformat(),
                "inversion_index": 10,
                "inversion_time": bars["timestamp"].iloc[10].isoformat(),
            }
        ],
    }

    figure = build_ict_price_figure(bars, pd.DataFrame(), runtime_state=runtime_state)

    rect = next(shape for shape in figure.layout["shapes"] if shape["type"] == "rect")
    ce_line = [shape for shape in figure.layout["shapes"] if shape["type"] == "line"][-1]
    assert rect["x0"] == bars["timestamp"].iloc[4]
    assert rect["x1"] == bars["timestamp"].iloc[6]
    assert ce_line["x1"] == bars["timestamp"].iloc[6]


def test_build_ict_price_figure_clips_fvg_that_started_before_visible_window(monkeypatch) -> None:
    monkeypatch.setattr("ote_live.dashboard.charts._plotly_go", lambda: _FakePlotlyGO())
    base_time = pd.Timestamp("2026-07-13T14:00:00Z")
    all_timestamps = [base_time + pd.Timedelta(minutes=5 * index) for index in range(30)]
    bars = pd.DataFrame(
        [
            {
                "timestamp": all_timestamps[index],
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
            }
            for index in range(10, 23)
        ]
    )
    runtime_state = {
        "latest_bar_timestamp_utc": bars["timestamp"].iloc[-1].isoformat(),
        "fvg_history": [
            {
                "id": 1,
                "direction": 1,
                "lower": 100.0,
                "upper": 101.0,
                "ce_price": 100.5,
                "formed_index": 4,
                "formed_time": all_timestamps[4].isoformat(),
                "source_timeframe": "5m",
            }
        ],
    }

    figure = build_ict_price_figure(bars, pd.DataFrame(), runtime_state=runtime_state)

    rect = next(shape for shape in figure.layout["shapes"] if shape["type"] == "rect")
    assert rect["x0"] == bars["timestamp"].iloc[0]
    assert rect["x1"] == all_timestamps[19]


def test_ict_fvg_history_payload_preserves_detector_history_without_duplicates() -> None:
    policy_frame = pd.DataFrame([{"close": 100.0}])
    fvg_zones = pd.DataFrame(
        [
            {
                "fvg_id": 7,
                "direction": 1,
                "original_direction": 1,
                "lower": 99.5,
                "upper": 100.5,
                "ce": 100.0,
                "formed_index": 3,
                "formed_time": "2026-07-13T14:15:00Z",
                "source_timeframe": "5m",
                "created_by_displacement": True,
                "inverted": True,
                "inversion_index": 6,
                "inversion_time": "2026-07-13T14:30:00Z",
                "first_inversion_index": 6,
                "first_inversion_time": "2026-07-13T14:30:00Z",
                "ce_tapped": True,
                "mitigated_pct": 1.0,
                "mitigation_index": 5,
                "mitigation_time": "2026-07-13T14:25:00Z",
                "full_mitigation_index": 5,
                "full_mitigation_time": "2026-07-13T14:25:00Z",
                "invalidated": False,
            },
            {
                "fvg_id": 7,
                "direction": 1,
                "original_direction": 1,
                "lower": 99.5,
                "upper": 100.5,
                "ce": 100.0,
                "formed_index": 3,
                "formed_time": "2026-07-13T14:15:00Z",
                "source_timeframe": "5m",
            },
        ]
    )
    policy_frame.attrs["fvg_zones"] = fvg_zones

    history = _build_ict_fvg_history_payload(policy_frame)

    assert len(history) == 1
    assert history[0]["id"] == 7
    assert history[0]["direction"] == 1
    assert history[0]["lower"] == 99.5
    assert history[0]["upper"] == 100.5
    assert history[0]["inversion_index"] == 6
    assert history[0]["first_inversion_time"] == "2026-07-13T14:30:00+00:00"
    assert history[0]["mitigation_index"] == 5
    assert history[0]["mitigation_time"] == "2026-07-13T14:25:00+00:00"


def test_live_policy_frame_preserves_fvg_history_attrs_from_feature_build() -> None:
    rows = 5
    market = pd.DataFrame(
        {
            "asset": ["ES"] * rows,
            "timeframe": ["5m"] * rows,
            "timestamp": pd.date_range("2026-07-13T14:00:00Z", periods=rows, freq="5min"),
            "open": [99.5, 99.8, 101.2, 101.1, 101.6],
            "high": [100.0, 100.2, 102.0, 101.5, 102.1],
            "low": [99.0, 99.5, 101.0, 100.4, 101.3],
            "close": [99.8, 100.0, 101.4, 101.2, 101.8],
            "volume": [100.0] * rows,
        }
    )
    feature_frame, _ = FeatureDatasetBuilder(
        FeatureBuilderConfig(
            feature_sets=["ict_context"],
            instrument="es",
            drop_warmup_rows=False,
            fillna_numeric=False,
        )
    ).build(market)
    assert not feature_frame.attrs["fvg_zones"].empty

    class _FakeState:
        def to_frame(self) -> pd.DataFrame:
            return market.copy()

    class _FakeFeatureEngine:
        state = _FakeState()

    processor = object.__new__(LiveSignalProcessor)
    processor.feature_engine = _FakeFeatureEngine()

    policy_frame = LiveSignalProcessor._build_policy_frame(processor, feature_frame)
    history = _build_ict_fvg_history_payload(policy_frame)

    assert not policy_frame.attrs["fvg_zones"].empty
    assert len(history) == 1
    assert history[0]["formed_time"] == market.loc[2, "timestamp"].isoformat()
    assert history[0]["mitigation_time"] == market.loc[3, "timestamp"].isoformat()


def test_build_ict_price_figure_rejects_stale_zero_price_overlays(monkeypatch) -> None:
    monkeypatch.setattr("ote_live.dashboard.charts._plotly_go", lambda: _FakePlotlyGO())
    bars = pd.DataFrame(
        [
            {
                "timestamp": datetime(2026, 7, 30, 20, 50, tzinfo=UTC),
                "open": 7475.0,
                "high": 7484.0,
                "low": 7468.0,
                "close": 7480.0,
            },
            {
                "timestamp": datetime(2026, 7, 30, 20, 55, tzinfo=UTC),
                "open": 7480.0,
                "high": 7498.0,
                "low": 7472.0,
                "close": 7488.0,
            },
        ]
    )
    runtime_state = {
        "levels": {
            "prior_rth_high": 7482.75,
            "prior_week_high": 0.0,
            "prior_week_low": 0.0,
        },
        "zones": {
            "bull_order_block": {
                "lower": 0.0,
                "upper": 0.0,
                "ce_price": 0.0,
            },
            "bear_fvg": {
                "lower": 7476.5,
                "upper": 7483.0,
                "ce_price": 7479.75,
            },
        },
        "recent_setups": [],
    }

    figure = build_ict_price_figure(
        bars,
        pd.DataFrame(),
        runtime_state=runtime_state,
    )

    trace_names = [trace.name for trace in figure.data]
    assert "PDH" in trace_names
    assert "PWH" not in trace_names
    assert "PWL" not in trace_names
    assert "Bull OB" not in trace_names
    overlay_y_values = [
        float(value)
        for trace in figure.data
        for value in (getattr(trace, "y", None) or [])
    ]
    assert overlay_y_values
    assert min(overlay_y_values) > 0.0


def test_build_ict_price_figure_ignores_stale_runtime_state_for_price_scale(monkeypatch) -> None:
    monkeypatch.setattr("ote_live.dashboard.charts._plotly_go", lambda: _FakePlotlyGO())
    bars = pd.DataFrame(
        [
            {
                "timestamp": datetime(2026, 8, 31, 20, 25, tzinfo=UTC),
                "open": 7701.25,
                "high": 7702.75,
                "low": 7701.0,
                "close": 7702.5,
            },
            {
                "timestamp": datetime(2026, 8, 31, 20, 30, tzinfo=UTC),
                "open": 7702.5,
                "high": 7702.5,
                "low": 7701.0,
                "close": 7701.75,
            },
        ]
    )
    runtime_state = {
        "latest_bar_timestamp_utc": "2026-08-04T03:30:00+00:00",
        "levels": {
            "prior_week_low": 7345.75,
            "session_vwap": 7633.116449486526,
        },
        "recent_setups": [
            {
                "timestamp_utc": "2026-08-03T22:15:00+00:00",
                "setup_side": 1,
                "label": "Premium Discount Continuation Long",
                "anchor_level": 0.0,
                "bar_close": 7633.75,
                "bar_low": 7632.5,
                "bar_high": 7635.0,
            }
        ],
    }

    figure = build_ict_price_figure(
        bars,
        pd.DataFrame(),
        runtime_state=runtime_state,
    )

    assert [trace.name for trace in figure.data] == ["Price"]
    assert figure.layout["yaxis"]["range"][0] > 7600.0
    assert figure.layout["xaxis"]["range"][0] > pd.Timestamp(
        "2026-08-31T20:00:00+00:00"
    )


def test_build_ict_price_figure_uses_visible_candle_for_zero_anchor_setup_marker(monkeypatch) -> None:
    monkeypatch.setattr("ote_live.dashboard.charts._plotly_go", lambda: _FakePlotlyGO())
    timestamp = datetime(2026, 8, 31, 20, 30, tzinfo=UTC)
    bars = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "open": 7702.5,
                "high": 7702.5,
                "low": 7701.0,
                "close": 7701.75,
            },
        ]
    )
    runtime_state = {
        "latest_bar_timestamp_utc": "2026-08-31T20:30:00+00:00",
        "recent_setups": [
            {
                "timestamp_utc": "2026-08-31T20:30:00+00:00",
                "setup_side": 1,
                "label": "Premium Discount Continuation Long",
                "anchor_level": 0.0,
                "ce_price": 0.0,
            }
        ],
    }

    figure = build_ict_price_figure(
        bars,
        pd.DataFrame(),
        runtime_state=runtime_state,
    )

    setup_trace = next(trace for trace in figure.data if trace.name == "ICT long setup")
    assert min(float(value) for value in setup_trace.y) > 7600.0


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
