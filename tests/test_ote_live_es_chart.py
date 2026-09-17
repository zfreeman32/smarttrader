"""A8 UI contracts exercised through actual SQLite records, Plotly and Dash."""
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.contracts.feature_snapshot import FeatureSnapshot
from ote_live.contracts.market_data import MarketBar
from ote_live.contracts.prediction import ModelPrediction
from ote_live.contracts.signal import SignalDecision
from ote_live.dashboard.app import create_dashboard_app
from ote_live.dashboard.es_chart import (
    build_es_price_figure, fetch_es_chart_data, matching_qualified_decisions,
    setup_hover, setup_layer_options, visible_setups,
)
from ote_live.dashboard.queries import fetch_confidence_history, fetch_recent_bars
from ote_live.dashboard.view_registry import DashboardViewConfig
from ote_live.policies.shadow import SHADOW_POLICY_CONTRACT, SETUP_MATCH_CONTRACT
from ote_live.storage import LiveAuditRepository, SQLiteLiveDataStore
from ote_live.storage.setup_events import SetupEventRepository

START = datetime(2026, 9, 11, 14, tzinfo=UTC)
MODEL = "frvp_long_continuation_xgb_v1"


def bar(at=START, **changes):
    return MarketBar(asset="ES", timeframe="5m", timestamp=at, source_timestamp=at,
                     open=5000, high=5002, low=4999, close=5001, volume=100,
                     is_complete=True, bar_version="v1", feed_type="live", observation_kind="live",
                     first_observed_at=at + timedelta(minutes=5, seconds=2),
                     last_observed_at=at + timedelta(minutes=5, seconds=2)).model_copy(update=changes)


def record_setup(repo, source, *, strategy="FRVP", collection="a8-test", selected=True, **changes):
    observation = {"setup_type": "3" if strategy == "FRVP" else "premium_discount_continuation",
                   "setup_family": "continuation", "setup_side": 1, "confidence": .73,
                   "selected": selected, "geometry": {"stop_reference": 4999}, **changes}
    event, = repo.record_observations([observation], bar=source, strategy=strategy,
                                     collection_version=collection,
                                     observed_at=source.timestamp + timedelta(minutes=5, seconds=3))
    return event


def record_decision(audit, source, *, event=None, qualified=False, crossed=True, collection="a8-test",
                    model=MODEL, recorded_at=None, metadata_changes=None, evaluation_changes=None):
    recorded = recorded_at or source.timestamp + timedelta(minutes=5, seconds=17)
    metadata = {"source_bar": source.model_dump(mode="json"),
                "bar_eligibility": {"eligible": True, "evaluation_kind": "live"}, **(metadata_changes or {})}
    snapshot = FeatureSnapshot(asset=source.asset, timeframe=source.timeframe, timestamp=source.timestamp,
                               collection_version=collection, observation_metadata=metadata, feature_values={"x": 1.0})
    snapshot_id = audit.record_feature_snapshot(snapshot)
    prediction = ModelPrediction(model_id=model, direction="long", timestamp=source.timestamp,
                                 backend="xgboost", collection_version=collection,
                                 prediction_recorded_at_utc=recorded, raw_score=.62,
                                 calibrated_probability=.82 if crossed else .2, threshold_applied=.5)
    prediction_id = audit.record_prediction(prediction, feature_snapshot_id=snapshot_id)
    evaluation = {"contract_version": SHADOW_POLICY_CONTRACT, "threshold_crossed": crossed,
                  "setup_match": {"contract_version": SETUP_MATCH_CONTRACT, "matched": event is not None,
                                  "matched_event_ids": [event.event_id] if event else []},
                  "policy_candidate_eligible": event is not None,
                  "qualified_shadow_entry": qualified,
                  "prerequisite_reasons": [] if qualified else ["outcome_contract_B1_pending"],
                  **(evaluation_changes or {})}
    signal = SignalDecision(model_id=model, direction="long", timestamp=source.timestamp,
                             collection_version=collection, decision="shadow", probability=prediction.calibrated_probability,
                             threshold=.5, shadow_evaluation=evaluation)
    audit.record_signal_decision(signal, prediction_id=prediction_id)
    return prediction_id


def load(store, *, strategy="FRVP", collection="latest", model_ids=(MODEL,), **kwargs):
    return fetch_es_chart_data(store, bars=fetch_recent_bars(store, asset="ES", timeframe="5m"),
                               asset="ES", timeframe="5m", strategy=strategy, model_ids=model_ids,
                               collection_version=collection, **kwargs)


@pytest.mark.parametrize("strategy", ["FRVP", "ICT"])
def test_full_window_history_optional_types_and_read_only_collection_scope(tmp_path, strategy):
    path = tmp_path / "a8.sqlite"
    with SQLiteLiveDataStore(path) as store:
        repo = SetupEventRepository(store)
        for offset in range(48):
            source = bar(START + timedelta(minutes=5 * offset))
            store.upsert_bar(source)
            record_setup(repo, source, strategy=strategy)
        record_setup(repo, source, strategy=strategy, collection="old", selected=False)
        record_setup(repo, source, strategy="ICT" if strategy == "FRVP" else "FRVP")
        counts = store.connection.total_changes
        data = load(store, strategy=strategy, collection="a8-test")
        assert len(data.setups) == 48
        assert len(visible_setups(data, strategy=strategy)) == 48
        assert visible_setups(data, strategy=strategy, setup_types=[]) == []
        wrong_type = "FRVP:1" if strategy == "FRVP" else "ICT:ifvg_reversal"
        assert visible_setups(data, strategy=strategy, setup_types=[wrong_type]) == []
        bars = fetch_recent_bars(store, asset="ES", timeframe="5m")
        figure = build_es_price_figure(bars, data=data, strategy=strategy, timeframe="5m")
        assert sum(len(t.x) for t in figure.data if t.type == "scatter") == 48
        assert "Rule confidence: 0.7300" in figure.data[1].hovertext[0]
        assert "Setup observation delay: 3.0s after bar close" in figure.data[1].hovertext[0]
        assert store.connection.total_changes == counts
    with SQLiteLiveDataStore(path) as reopened:
        assert len(load(reopened, strategy=strategy, collection="a8-test").setups) == 48


def test_default_rejects_off_setup_thresholds_and_pending_candidates_research_keeps_them(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "a8.sqlite") as store:
        source = bar()
        store.upsert_bar(source)
        repo, audit = SetupEventRepository(store), LiveAuditRepository(store)
        event = record_setup(repo, source)
        prediction_id = record_decision(audit, source, event=event)
        repo.link_prediction(event.event_id, prediction_id, model_id=MODEL, decision="shadow")
        record_decision(audit, source)
        record_decision(audit, source, crossed=False)
        data, bars = load(store), fetch_recent_bars(store, asset="ES", timeframe="5m")
        default = build_es_price_figure(bars, data=data, strategy="FRVP", timeframe="5m")
        assert len(default.data) == 2  # Candle and selected rule setup only.
        assert matching_qualified_decisions(data, visible_setups(data, strategy="FRVP")) == []
        assert f"Associated prediction IDs (including diagnostics): [{prediction_id}]" in setup_hover(data.setups[0], "5m")
        research = build_es_price_figure(bars, data=data, strategy="FRVP", timeframe="5m", mode="research")
        diagnostics = [t for t in research.data if "diagnostic" in t.name.lower()]
        assert len(diagnostics) == 2
        assert sum(len(t.x) for t in diagnostics) == 2
        text = diagnostics[0].hovertext[0]
        assert "Model probability: 0.8200" in text and "Decision delay: 17.0s after bar close" in text
        assert "outcome_contract_B1_pending" in text


def test_qualified_decision_uses_actual_time_and_exact_setup_link(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "a8.sqlite") as store:
        source = bar()
        store.upsert_bar(source)
        event = record_setup(SetupEventRepository(store), source)
        audit = LiveAuditRepository(store)
        # Synthetic future qualified record tests presentation only; production qualification remains gated.
        record_decision(audit, source, event=event, qualified=True)
        record_decision(audit, source, event=event, qualified=True,
                        metadata_changes={"source_bar": source.model_copy(update={"bar_version": "wrong"}).model_dump(mode="json")})
        record_decision(audit, source, event=event, qualified=True,
                        metadata_changes={"bar_eligibility": {"eligible": False}})
        record_decision(audit, source, event=event, qualified=True, recorded_at=START + timedelta(seconds=1))
        record_decision(audit, source, qualified=True)
        data = load(store)
        selected = visible_setups(data, strategy="FRVP")
        qualified = matching_qualified_decisions(data, selected)
        assert len(qualified) == 1
        assert qualified[0]["matching_setup_ids"] == [event.event_id]
        figure = build_es_price_figure(fetch_recent_bars(store, asset="ES", timeframe="5m"),
                                      data=data, strategy="FRVP", timeframe="5m")
        star = next(t for t in figure.data if "Qualified" in t.name)
        assert pd.Timestamp(star.x[0]) == START + timedelta(minutes=5, seconds=17)
        assert "reference price" in star.name
        assert pd.Timestamp(figure.layout.xaxis.range[-1]) > pd.Timestamp(star.x[0])
        assert not matching_qualified_decisions(data, [])


def test_revisions_invalidations_and_provisional_marks_remain_distinct(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "a8.sqlite") as store:
        repo = SetupEventRepository(store)
        original = bar(is_complete=False)
        store.upsert_bar(original)
        first = record_setup(repo, original)
        record_setup(repo, bar(bar_version="v2"), confidence=.6)
        repo.record_observations([], bar=bar(bar_version="v3"), strategy="FRVP",
                                 collection_version="a8-test", observed_at=START + timedelta(minutes=8))
        other = bar(START + timedelta(minutes=5))
        store.upsert_bar(other)
        record_setup(repo, other, selected=False)
        data = load(store)
        default = visible_setups(data, strategy="FRVP")
        assert len(default) == 1 and default[0]["event_kind"] == "invalidated"
        all_events = visible_setups(data, strategy="FRVP", layers=["revisions", "rejected"])
        assert len(all_events) == 4 and all_events[0]["id"] == first.event_id
        figure = build_es_price_figure(fetch_recent_bars(store, asset="ES", timeframe="5m"), data=data,
                                      strategy="FRVP", timeframe="5m", layers=["revisions", "rejected"])
        symbols = {t.marker.symbol for t in figure.data if t.type == "scatter"}
        assert {"triangle-up-open", "diamond", "x", "square-open"}.issubset(symbols)
        assert len(repo.history(first.event_key)) == 3


def test_collections_assets_timeframes_and_manifest_scope_do_not_mix(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "a8.sqlite") as store:
        source = bar()
        store.upsert_bar(source)
        repo, audit = SetupEventRepository(store), LiveAuditRepository(store)
        record_setup(repo, source, collection="old")
        record_setup(repo, source, collection="new")
        record_decision(audit, source, collection="old")
        record_decision(audit, source, collection="new", recorded_at=START + timedelta(minutes=6))
        record_decision(audit, source.model_copy(update={"asset": "NQ"}), collection="other")
        record_decision(audit, source.model_copy(update={"timeframe": "1m"}), collection="other")
        latest = load(store)
        assert latest.collection_version == "new"
        assert len(latest.decisions) == 1 and len(latest.setups) == 1
        assert {o["value"] for o in latest.collection_options} == {"latest", "old", "new"}
        assert len(load(store, collection="old").decisions) == 1
        assert load(store, runtime_manifest_hashes=()).decisions == []
        assert load(store, model_ids=()).decisions == []
        assert load(store, collection="not-present").decisions == []
        confidence = fetch_confidence_history(audit, model_id=MODEL, collection_version="new", asset="ES", timeframe="5m")
        assert len(confidence) == 1
        assert confidence.iloc[0]["prediction_recorded_at_utc"] == (START + timedelta(minutes=6)).isoformat().replace("+00:00", "Z")


def test_missing_history_does_not_fabricate_legacy_setups_or_create_tables(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "a8.sqlite") as store:
        # Simulate a pre-history database after ordinary store initialization.
        store.connection.execute("DROP TABLE setup_events")
        store.upsert_bar(bar())
        data = load(store)
        assert not data.history_available
        state = {"recent_setups": [{"setup_type": "3", "setup_side": 1, "confidence": .8}]}
        figure = build_es_price_figure(fetch_recent_bars(store, asset="ES", timeframe="5m"), data=data,
                                      strategy="FRVP", timeframe="5m", runtime_state=state)
        assert len(figure.data) == 1
        assert "No immutable setup history" in figure.layout.annotations[-1].text
        assert not store.connection.execute("SELECT 1 FROM sqlite_master WHERE name='setup_events'").fetchone()


def test_later_revision_keeps_original_qualified_link_and_labels_latest_state(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "a8.sqlite") as store:
        source = bar()
        store.upsert_bar(source)
        repo = SetupEventRepository(store)
        original = record_setup(repo, source)
        record_decision(LiveAuditRepository(store), source, event=original, qualified=True)
        repo.record_observations([], bar=bar(bar_version="v2"), strategy="FRVP", collection_version="a8-test",
                                 observed_at=START + timedelta(minutes=10))
        data = load(store)
        selected = visible_setups(data, strategy="FRVP")
        assert selected[0]["event_kind"] == "invalidated"
        qualified = matching_qualified_decisions(data, selected)
        assert len(qualified) == 1
        assert qualified[0]["matching_setup_ids"] == [original.event_id]
        figure = build_es_price_figure(fetch_recent_bars(store, asset="ES", timeframe="5m"), data=data,
                                      strategy="FRVP", timeframe="5m")
        star = next(t for t in figure.data if "Qualified" in t.name)
        assert "latest state: invalidated revision 2" in star.hovertext[0]
        assert "rule confidence: 0.7300" in star.hovertext[0]


def test_real_dash_refresh_defaults_to_setups_and_research_is_opt_in(tmp_path):
    with SQLiteLiveDataStore(tmp_path / "a8.sqlite") as store:
        source = bar()
        store.upsert_bar(source)
        record_setup(SetupEventRepository(store), source)
        record_decision(LiveAuditRepository(store), source)
        manifests = ROOT / "ote_live/runtime_manifests/frvp_es_shadow_20260721"
        view = DashboardViewConfig(view_id="FRVP", label="FRVP", asset="ES", timeframe="5m", data_supplier="IBKR",
                                    enable_frvp_overlays=True, long_runtime_manifest_path=manifests / "live_runtime_manifest_long.json",
                                    short_runtime_manifest_path=manifests / "live_runtime_manifest_short.json")
        app = create_dashboard_app(store, view_configs=[view])
        callback = next(iter(app.callback_map.values()))["callback"].__wrapped__
        default = callback(0, None, "FRVP")
        assert len(default[6].data) == 2
        assert default[9] == [] and default[10] == []
        assert "Model probability" not in default[8]
        assert default[15]["display"] == "none"
        research = callback(0, None, "FRVP", "research", ["all"], [], "latest")
        assert any("threshold diagnostic" in t.name for t in research[6].data)
        assert research[9] and research[10]
        assert "Model probability" in research[8]
        assert research[15]["display"] == "block"
        client = app.server.test_client()
        assert client.get("/").status_code == 200
        assert client.get("/_dash-layout").status_code == 200
        assert client.get("/_dash-dependencies").status_code == 200


def test_optional_layers_include_all_current_types_and_classic_breaker():
    values = {o["value"] for o in setup_layer_options()}
    assert {f"FRVP:{i}" for i in range(1, 7)}.issubset(values)
    assert "ICT:classic_breaker" in values
    assert len([v for v in values if v.startswith("ICT:")]) == 9


@pytest.mark.parametrize("setup_tab", [False, True])
def test_a11_ict_pending_priorities_and_background_diagnostics(tmp_path, setup_tab):
    import json
    from plotly.utils import PlotlyJSONEncoder
    from ote_live.dashboard.view_registry import build_default_dashboard_views, DEFAULT_ICT_MODEL_ORDER, DEFAULT_ICT_SETUP_MODEL_ORDER
    from ote_live.models.ict_research import ICT_RESEARCH_PRIORITIES, ict_research_priority

    assert set(ICT_RESEARCH_PRIORITIES) == set(DEFAULT_ICT_MODEL_ORDER + DEFAULT_ICT_SETUP_MODEL_ORDER)
    assert {model for model, priority in ICT_RESEARCH_PRIORITIES.items() if priority.tier == "priority_pending_review"} == {
        "ict_long_continuation_xgb_v1", "ict_short_continuation_premium_discount_continuation_xgb_v1"}
    assert all(not priority.foreground for priority in ICT_RESEARCH_PRIORITIES.values())
    view_id = "ICT_SETUP" if setup_tab else "ICT"
    views = {v.view_id: v for v in build_default_dashboard_views()}
    focused = "ict_short_continuation_premium_discount_continuation_xgb_v1" if setup_tab else "ict_long_continuation_xgb_v1"
    background = "ict_short_reversal_ifvg_reversal_xgb_v1" if setup_tab else "ict_long_meta_xgb_v1"
    with SQLiteLiveDataStore(tmp_path / "ict.sqlite") as store:
        audit, repo = LiveAuditRepository(store), SetupEventRepository(store)
        source = bar()
        store.upsert_bar(source)
        event = record_setup(repo, source, strategy="ICT", setup_side=-1 if setup_tab else 1)
        for model in (focused, background):
            record_decision(audit, source, event=event, model=model, evaluation_changes={
                "policy_candidate_eligible": False,
                "research_priority": ict_research_priority(model).payload()})
        app = create_dashboard_app(store, view_configs=[views[view_id]])
        callback = next(iter(app.callback_map.values()))["callback"].__wrapped__
        count = store.connection.total_changes
        default = callback(0, None, view_id)
        assert "research-only" in default[5] and "B2" in default[5]
        assert default[9] == [] and default[10] == []
        assert not any("Qualified" in trace.name for trace in default[6].data)
        assert "Model probability" not in default[8]
        text = json.dumps(default[18], cls=PlotlyJSONEncoder)
        assert focused in text and background not in text
        assert "Focused comparison pending review" in text
        assert "concentration" in text and "no activation is authorized" in text
        research = callback(0, None, view_id, "research", ["all"], [], "latest")
        cards = json.dumps(research[10] if setup_tab else research[9], cls=PlotlyJSONEncoder)
        assert cards.index(focused) < cards.index(background)
        assert "Background" in cards and "RESEARCH-ONLY" in cards and "ACTIVE WEIGHT" not in cards
        assert focused in research[8] and background in research[8]
        assert any("threshold diagnostic" in trace.name for trace in research[6].data)
        assert store.connection.total_changes == count
        assert store.connection.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0] == 2
        assert app.server.test_client().get("/_dash-layout").status_code == 200


@pytest.mark.parametrize("setup_tab", [False, True])
def test_a9_a10_default_priorities_and_background_research_keep_all_records(tmp_path, setup_tab):
    import json
    from plotly.utils import PlotlyJSONEncoder
    from ote_live.dashboard.view_registry import build_default_dashboard_views

    def rendered(value):
        return json.dumps(value, cls=PlotlyJSONEncoder)

    view_id = "FRVP_SETUP" if setup_tab else "FRVP"
    views = {v.view_id: v for v in build_default_dashboard_views()}
    foreground = "frvp_long_continuation_setup3_xgb_v1" if setup_tab else MODEL
    background = "frvp_long_continuation_setup2_xgb_v1" if setup_tab else "frvp_long_meta_xgb_v1"
    with SQLiteLiveDataStore(tmp_path / "roster.sqlite") as store:
        audit, repo = LiveAuditRepository(store), SetupEventRepository(store)
        source = bar()
        store.upsert_bar(source)
        event = record_setup(repo, source)
        record_decision(audit, source, event=event, model=foreground, qualified=True)
        if setup_tab:
            source = bar(START + timedelta(minutes=5))
            store.upsert_bar(source)
            event = record_setup(repo, source, setup_type="2")
        record_decision(audit, source, event=event, model=background, qualified=True)
        # Three short S5 opportunities and one long S5 must show one long match.
        for offset, side in enumerate([-1, -1, -1, 1], start=2):
            source = bar(START + timedelta(minutes=5 * offset))
            store.upsert_bar(source)
            record_setup(repo, source, setup_type="5", setup_side=side)
        app = create_dashboard_app(store, view_configs=[views[view_id]])
        callback = next(iter(app.callback_map.values()))["callback"].__wrapped__
        count = store.connection.total_changes
        default = callback(0, None, view_id)
        stars = [t for t in default[6].data if "Qualified" in t.name]
        assert sum(len(t.x) for t in stars) == 1
        assert background not in default[8]
        priority_text = rendered(default[18])
        assert foreground in priority_text and background not in priority_text
        assert "no promotion or execution authority" in priority_text
        assert default[9] == [] and default[10] == []  # A8 probability streams stay opt-in.
        if setup_tab:
            assert "Exploratory long S3" in priority_text
            assert "Collect matching long S5" in priority_text
            assert "Selected matching long setups in this collection/window: 1" in priority_text
            assert "predominantly short-S5 results do not establish long-model quality" in priority_text
        else:
            assert "frvp_short_continuation_tcn_v1" in priority_text
            assert "economics and drawdown restrictions remain" in priority_text
        research = callback(0, None, view_id, "research", ["all"], [], "latest")
        assert background in research[8]
        research_cards = rendered(research[9])
        assert research_cards.index(foreground) < research_cards.index(background)
        if setup_tab:
            assert "S4 qualified candidacy paused" in research_cards
            assert "insufficient evidence, not model failure" in research_cards
        else:
            assert "ACTIVE WEIGHT" not in rendered(research[9:11])
            assert "short reversal remains retired" in priority_text
        assert store.connection.total_changes == count
        assert store.connection.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0] == 2
