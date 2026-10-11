"""Model/setup relationships and gross response summaries, never P&L.

The routing decision in this module is deliberately made from the immutable
setup census and the model decision clock only.  Forward prices are never an
input to setup selection.
"""
from datetime import datetime
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import Field, field_validator

from model_testing.outcome_contracts import FrozenModel


POOLED_ROUTING_CONTRACT = "pooled_detector_selected_same_side_v1"


class SetupObservation(FrozenModel):
    setup_event_id: str
    strategy: str
    setup_family: str
    side: Literal["long", "short"]
    known_at_ts: datetime
    selected: bool
    detector_version: str

    @field_validator("known_at_ts")
    @classmethod
    def timezone_required(cls, value):
        if value.tzinfo is None:
            raise ValueError("Setup audit timestamps require timezone")
        return value


class ModelSetupContext(FrozenModel):
    """Complete same-opportunity setup census, captured before threshold filtering."""
    collection_id: str
    opportunity_id: str
    prediction_id: str
    available_at_ts: datetime
    coverage: Literal["all_model_observations", "selected_signals_only", "unknown"]
    setup_coverage: Literal["complete", "unknown"]
    expected_model_ids: tuple[str, ...] = ()
    setups: tuple[SetupObservation, ...] = ()
    model_artifact_hash: str | None = None
    calibration_hash: str | None = None
    policy_hash: str | None = None
    raw_score: float | None = None
    calibrated_probability: float | None = Field(default=None, ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)
    threshold_source: str | None = None
    threshold_passed: bool | None = None
    trend_regime: str | None = None
    volatility_regime: str | None = None
    composite_regime: str | None = None
    session_regime: str | None = None
    stress_regime: str | None = None
    rejection_reasons: tuple[str, ...] = ()
    context_source_hash: str

    @field_validator("available_at_ts")
    @classmethod
    def timezone_required(cls, value):
        if value.tzinfo is None:
            raise ValueError("Context timestamp requires timezone")
        return value


@dataclass(frozen=True)
class PooledRoutingDecision:
    """Causal routing result for one model/setup opportunity.

    ``primary_setup_event_id`` is the only setup that can support an
    executable pooled outcome.  ``counterfactual_setup_event_ids`` are still
    valid same-opportunity, same-side markout rows, but can never create a
    position.  A later setup is kept separate and is never promoted into
    either list.
    """

    contract: str
    status: Literal[
        "primary_selected",
        "counterfactual_only",
        "no_causal_setup",
        "no_same_side_setup",
        "ambiguous_selected_setup",
    ]
    primary_setup_event_id: str | None
    contemporaneous_same_side_setup_event_ids: tuple[str, ...]
    counterfactual_setup_event_ids: tuple[str, ...]
    later_same_side_setup_event_ids: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return self.status == "primary_selected" and self.primary_setup_event_id is not None


def resolve_pooled_routing(
    context: ModelSetupContext,
    *,
    strategy: str,
    setup_family: str,
    side: Literal["long", "short"],
    decision_at: datetime,
) -> PooledRoutingDecision:
    """Resolve the predeclared selected setup without outcome-dependent choice.

    The context is already scoped to one model opportunity.  Matching the
    declared strategy/family/side plus ``known_at_ts <= decision_at`` is the
    contemporaneous contract.  Multiple selected candidates are ambiguous and
    therefore fail closed instead of being ordered by confidence, recency, or
    eventual outcome.
    """

    decision_ts = pd.Timestamp(decision_at)
    if decision_ts.tzinfo is None:
        raise ValueError("Routing decision timestamp requires timezone")
    decision_ts = decision_ts.tz_convert("UTC")
    expected_strategy = str(strategy).strip().lower()
    expected_family = str(setup_family).strip().lower()
    expected_side = str(side).strip().lower()

    same_side = [
        setup for setup in context.setups
        if str(setup.strategy).strip().lower() == expected_strategy
        and str(setup.side).strip().lower() == expected_side
    ]
    # The selected route must belong to the model's declared setup scope,
    # while the diagnostic census deliberately retains every other
    # contemporaneous same-side setup for a model-by-setup markout.
    route_candidates = [
        setup for setup in same_side
        if str(setup.setup_family).strip().lower() == expected_family
    ]
    causal = [setup for setup in same_side if pd.Timestamp(setup.known_at_ts).tz_convert("UTC") <= decision_ts]
    later = [setup for setup in same_side if pd.Timestamp(setup.known_at_ts).tz_convert("UTC") > decision_ts]
    selected = [setup for setup in causal if setup in route_candidates and setup.selected]

    causal_ids = tuple(setup.setup_event_id for setup in causal)
    later_ids = tuple(setup.setup_event_id for setup in later)
    if len(selected) > 1:
        return PooledRoutingDecision(
            contract=POOLED_ROUTING_CONTRACT,
            status="ambiguous_selected_setup",
            primary_setup_event_id=None,
            contemporaneous_same_side_setup_event_ids=causal_ids,
            counterfactual_setup_event_ids=causal_ids,
            later_same_side_setup_event_ids=later_ids,
        )
    if len(selected) == 1:
        primary = selected[0].setup_event_id
        return PooledRoutingDecision(
            contract=POOLED_ROUTING_CONTRACT,
            status="primary_selected",
            primary_setup_event_id=primary,
            contemporaneous_same_side_setup_event_ids=causal_ids,
            counterfactual_setup_event_ids=tuple(event_id for event_id in causal_ids if event_id != primary),
            later_same_side_setup_event_ids=later_ids,
        )
    if causal:
        return PooledRoutingDecision(
            contract=POOLED_ROUTING_CONTRACT,
            status="counterfactual_only",
            primary_setup_event_id=None,
            contemporaneous_same_side_setup_event_ids=causal_ids,
            counterfactual_setup_event_ids=causal_ids,
            later_same_side_setup_event_ids=later_ids,
        )
    return PooledRoutingDecision(
        contract=POOLED_ROUTING_CONTRACT,
        status="no_causal_setup" if later else "no_same_side_setup",
        primary_setup_event_id=None,
        contemporaneous_same_side_setup_event_ids=(),
        counterfactual_setup_event_ids=(),
        later_same_side_setup_event_ids=later_ids,
    )


def build_model_setup_audit(signals, outcomes):
    from model_testing.outcome_engine import session_at, utc

    rows, coverage = [], []
    by_event = {row["event_id"]: row for row in outcomes}
    groups = {}
    for signal in signals:
        result, context = by_event[signal.event_id], signal.audit_context
        known = utc(signal.known_at_ts)
        valid_context = context is not None and context.available_at_ts <= signal.known_at_ts
        base = {key: result[key] for key in (
            "event_id", "model_id", "model_version", "model_family", "instrument", "side", "known_at_ts",
            "entry_ts", "entry_price", "evaluation_contract_id", "evaluation_contract_hash",
            "baseline_scope", "baseline_scope_id", "baseline_scope_hash", "baseline_scope_reason",
            "atr_price", "atr_definition", "atr_status", "atr_source_close_ts", "atr_source_hash")}
        base.update({key: value for key, value in result.items() if key.startswith("forward_")})
        base.update(observation_strategy=signal.strategy, evaluation_setup_family=signal.setup_family,
            decision_session=session_at(known), decision_hour_new_york=known.tz_convert("America/New_York").hour,
            decision_date_new_york=known.tz_convert("America/New_York").date().isoformat(),
            context_status="OK" if valid_context else "context_missing_or_unavailable",
            coverage=context.coverage if valid_context else "unknown",
            setup_coverage=context.setup_coverage if valid_context else "unknown",
            collection_id=context.collection_id if valid_context else "UNKNOWN",
            opportunity_id=context.opportunity_id if valid_context else signal.event_id,
            prediction_id=context.prediction_id if valid_context else None,
            policy_eligible=signal.policy_eligible, policy_rejection_reason=signal.policy_rejection_reason)
        for key in ("model_artifact_hash", "calibration_hash", "policy_hash", "raw_score", "calibrated_probability",
                    "threshold", "threshold_source", "threshold_passed", "trend_regime", "volatility_regime",
                    "composite_regime", "session_regime", "stress_regime", "context_source_hash"):
            base[key] = getattr(context, key) if valid_context else None
        base["rejection_reasons"] = "|".join(context.rejection_reasons) if valid_context else "context_unverified"
        if valid_context:
            group = groups.setdefault((context.collection_id, context.opportunity_id), {"expected": set(), "observed": set()})
            group["expected"].update(context.expected_model_ids)
            group["observed"].add(signal.model_id)
        candidates = context.setups if valid_context else ()
        routing = (resolve_pooled_routing(
            context, strategy=signal.strategy, setup_family=signal.setup_family,
            side=signal.side, decision_at=signal.known_at_ts
        ) if valid_context else None)
        if not candidates:
            relationship = "no_setup" if valid_context and context.setup_coverage == "complete" else "unknown_setup_coverage"
            rows.append(dict(base, observed_setup_id=None, observed_setup_strategy=None,
                             observed_setup_family=None, observed_setup_side=None,
                             setup_selected=None, relationship=relationship, setup_detector_version=None,
                             routing_contract=routing.contract if routing else None,
                             routing_status=routing.status if routing else None,
                             routing_role="no_setup", primary_executable_setup_id=(
                                 routing.primary_setup_event_id if routing else None),
                             counterfactual_markout_only=False, executable_route=False,
                             markout_eligible=False))
        for setup in candidates:
            relationship = ("not_known_at_decision" if setup.known_at_ts > signal.known_at_ts else
                            "opposite_side" if setup.side != signal.side else
                            "selected_same_side" if setup.selected else "other_same_side")
            same_side = setup.side == signal.side
            causal = setup.known_at_ts <= signal.known_at_ts
            primary = bool(routing and routing.primary_setup_event_id == setup.setup_event_id)
            counterfactual = bool(routing and setup.setup_event_id in routing.counterfactual_setup_event_ids)
            routing_role = (
                "primary_executable" if primary else
                "counterfactual_same_side" if counterfactual else
                "not_known_at_decision" if same_side and not causal else
                "opposite_side" if not same_side else
                "out_of_declared_route"
            )
            rows.append(dict(base, observed_setup_id=setup.setup_event_id, observed_setup_strategy=setup.strategy,
                observed_setup_family=setup.setup_family, observed_setup_side=setup.side,
                setup_selected=setup.selected, relationship=relationship, setup_detector_version=setup.detector_version,
                routing_contract=routing.contract if routing else None,
                routing_status=routing.status if routing else None,
                routing_role=routing_role,
                primary_executable_setup_id=(routing.primary_setup_event_id if routing else None),
                counterfactual_markout_only=counterfactual,
                executable_route=primary,
                markout_eligible=bool(same_side and causal)))
    for (collection, opportunity), group in sorted(groups.items()):
        for model in sorted(group["expected"] | group["observed"]):
            coverage.append(dict(collection_id=collection, opportunity_id=opportunity, model_id=model,
                                 expected=model in group["expected"], observed=model in group["observed"]))
    observations = pd.DataFrame(rows)
    return observations, pd.DataFrame(coverage, columns=["collection_id", "opportunity_id", "model_id", "expected", "observed"])


def summarize_model_setup_audit(observations: pd.DataFrame) -> pd.DataFrame:
    """Predeclared slices retain sample counts and incomplete-horizon denominators.

    Repeated candidates of one family cannot multiply a prediction's contribution.
    Across setup families rows overlap and must never be summed as independent trades.
    """
    keys = ["collection_id", "model_id", "model_version", "model_artifact_hash", "side", "instrument",
            "observed_setup_strategy", "observed_setup_family", "observed_setup_side", "relationship",
            "routing_role", "evaluation_contract_hash", "baseline_scope_hash", "context_status", "coverage"]
    rows = []
    for facet in ("all", "decision_session", "decision_hour_new_york", "trend_regime", "volatility_regime"):
        frame = observations.copy()
        frame["slice_value"] = "all" if facet == "all" else frame[facet].fillna("UNKNOWN").astype(str)
        frame = frame.drop_duplicates(["event_id", *keys, "slice_value"])
        for values, group in frame.groupby([*keys, "slice_value"], dropna=False, sort=False):
            row = dict(zip([*keys, "slice_value"], values))
            row.update(slice_dimension=facet, observation_count=len(group),
                       opportunity_count=group.opportunity_id.nunique(),
                       threshold_pass_count=int(group.threshold_passed.eq(True).sum()),
                       threshold_unknown_count=int(group.threshold_passed.isna().sum()),
                       exploratory_only=True)
            for horizon in (1, 3, 6, 12, 24, 48):
                column = f"forward_return_{horizon}"
                valid = group[column + "_status"].eq("OK") & group.relationship.ne("not_known_at_decision")
                moves = pd.to_numeric(group.loc[valid, column + "_units"], errors="coerce").dropna()
                normalized = pd.to_numeric(group.loc[valid, column + "_atr"], errors="coerce").dropna()
                row.update({column + "_valid_count": len(moves), column + "_missing_count": len(group) - len(moves),
                    column + "_positive_count": int(moves.gt(0).sum()),
                    column + "_mean_units": moves.mean(), column + "_median_units": moves.median(),
                    column + "_atr_count": len(normalized), column + "_mean_atr": normalized.mean()})
            rows.append(row)
    return pd.DataFrame(rows)
