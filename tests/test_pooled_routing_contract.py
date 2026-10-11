from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from model_testing.outcome_diagnostics import (
    ModelSetupContext,
    POOLED_ROUTING_CONTRACT,
    SetupObservation,
    resolve_pooled_routing,
)
from ote_live.policies.shadow import match_shadow_setups


UTC = timezone.utc
DECISION = datetime(2026, 9, 29, 14, 30, tzinfo=UTC)


def _context(*setups):
    return ModelSetupContext(
        collection_id="collection-v1", opportunity_id="bar-1", prediction_id="prediction-1",
        available_at_ts=DECISION - timedelta(seconds=1), coverage="all_model_observations",
        setup_coverage="complete", context_source_hash="a" * 64, setups=tuple(setups),
    )


def _setup(event_id, family, *, selected=False, known_at=DECISION, side="long"):
    return SetupObservation(
        setup_event_id=event_id, strategy="frvp", setup_family=family, side=side,
        known_at_ts=known_at, selected=selected, detector_version="detector-v1",
    )


def test_pooled_routing_keeps_other_same_side_markouts_and_excludes_later_setup():
    routing = resolve_pooled_routing(
        _context(
            _setup("primary", "setup3", selected=True),
            _setup("other", "setup2"),
            _setup("later", "setup3", selected=True, known_at=DECISION + timedelta(minutes=5)),
        ),
        strategy="frvp", setup_family="setup3", side="long", decision_at=DECISION,
    )

    assert routing.contract == POOLED_ROUTING_CONTRACT
    assert routing.primary_setup_event_id == "primary"
    assert routing.counterfactual_setup_event_ids == ("other",)
    assert routing.later_same_side_setup_event_ids == ("later",)
    assert routing.executable


def test_pooled_routing_fails_closed_when_detector_selects_two_candidates():
    routing = resolve_pooled_routing(
        _context(_setup("one", "setup3", selected=True), _setup("two", "setup3", selected=True)),
        strategy="frvp", setup_family="setup3", side="long", decision_at=DECISION,
    )

    assert routing.status == "ambiguous_selected_setup"
    assert routing.primary_setup_event_id is None
    assert routing.counterfactual_setup_event_ids == ("one", "two")


def test_live_pooled_match_exposes_primary_counterfactual_and_later_roles():
    source = DECISION - timedelta(minutes=5)
    prediction = SimpleNamespace(
        model_id="frvp_long_continuation_xgb_v1", direction="long", timestamp=source,
        collection_version="collection-v1", prediction_recorded_at_utc=DECISION,
    )
    snapshot = SimpleNamespace(
        collection_version="collection-v1", asset="ES", timeframe="5m",
        observation_metadata={"source_bar": {"bar_version": "bar-v1"}},
    )

    def event(event_id, setup_type, family="continuation", *, selected=False, observed_at=DECISION):
        return SimpleNamespace(
            event_id=event_id, strategy="FRVP", setup_type=setup_type, setup_side=1,
            setup_family=family, selected=selected, event_kind="observed",
            collection_version="collection-v1", source_timestamp=source,
            source_bar_version="bar-v1", observed_at=observed_at,
            payload={"source_bar": {"asset": "ES", "timeframe": "5m"}},
        )

    result = match_shadow_setups(prediction, snapshot, [
        event(1, "3", selected=True), event(2, "2"),
        event(3, "1", family="reversal"),
        event(4, "5", selected=True, observed_at=DECISION + timedelta(minutes=5)),
    ])

    assert result["primary_event_id"] == 1
    assert result["matched_event_ids"] == [1]
    assert result["counterfactual_event_ids"] == [2, 3]
    assert result["later_event_ids"] == [4]
    assert result["hindsight_selection_forbidden"] is True
