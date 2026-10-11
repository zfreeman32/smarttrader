"""Frozen research-scope membership, separate from setup economics and live policies."""
from pathlib import Path
import json
from typing import Literal

from model_testing.outcome_contracts import ROOT, FrozenModel
from model_testing.outcome_diagnostics import (
    ModelSetupContext,
    POOLED_ROUTING_CONTRACT,
    resolve_pooled_routing,
)

SCOPE_PATH = ROOT / "configs/outcome_contracts/baseline_scope_v1.json"
RESEARCH_SCOPE_PATH = ROOT / "configs/outcome_contracts/b2_research_scope_v1.json"


class BaselineScope(FrozenModel):
    scope_id: str
    version: int
    excluded_setups: tuple[str, ...]
    diagnostic_setups: tuple[str, ...]
    pooled_families: tuple[tuple[str, tuple[str, ...]], ...]
    decision_source: str

    def reason(self, *, strategy: str, model_family: str, setup_family: str) -> str | None:
        if setup_family in self.excluded_setups:
            return "classic_breaker_excluded_from_baseline"
        if setup_family in self.diagnostic_setups:
            return "unmatched_prediction_diagnostics_only"
        if model_family in dict(self.pooled_families).get(strategy, ()):
            return "pooled_model_diagnostics_only"
        return None

    def signal_reason(self, signal) -> str | None:
        """Return the scope decision for one signal.

        The baseline scope has no conditional admission rule.  Research scopes
        override this method because pooled admission depends on the immutable
        setup context captured at the signal decision clock.
        """
        return self.reason(strategy=signal.strategy, model_family=signal.model_family,
                           setup_family=signal.setup_family)


class ResearchScope(BaselineScope):
    """Conditional pooled-model scope for simulated research outcomes.

    Pooled predictions are executable only for the detector-selected,
    contemporaneous, same-side setup that was known at the prediction's
    decision clock.  Other same-side setup associations remain descriptive
    counterfactual markouts and cannot create a simulated position.
    """

    scope_kind: Literal["research"] = "research"
    pooled_admission: Literal["selected_same_side_causal_setup"]
    pooled_counterfactual: Literal["other_same_side_markout_only"]
    pooled_routing_contract: Literal["pooled_detector_selected_same_side_v1"]
    pooled_primary_route: Literal["detector_selected_contemporaneous_same_side"]
    pooled_counterfactual_route: Literal["all_other_contemporaneous_same_side_model_by_setup_markouts"]
    pooled_hindsight_rule: Literal["never_attach_later_or_select_after_outcome"]
    required_context_coverage: Literal["all_model_observations"]
    required_setup_coverage: Literal["complete"]
    causal_clock: Literal["setup_known_at_decision_time"]
    simulated_outcomes_only: Literal[True] = True
    broker_order_authority: Literal[False] = False

    def reason(self, *, strategy: str, model_family: str, setup_family: str) -> str | None:
        """Return unconditional exclusions; pooled routes are conditional."""
        base_reason = BaselineScope.reason(self, strategy=strategy, model_family=model_family,
                                           setup_family=setup_family)
        return None if base_reason == "pooled_model_diagnostics_only" else base_reason

    def signal_reason(self, signal) -> str | None:
        base_reason = BaselineScope.reason(self, strategy=signal.strategy,
                                           model_family=signal.model_family,
                                           setup_family=signal.setup_family)
        if base_reason != "pooled_model_diagnostics_only":
            return base_reason

        context: ModelSetupContext | None = getattr(signal, "audit_context", None)
        if context is None:
            return "pooled_setup_context_missing"
        if context.available_at_ts > signal.known_at_ts:
            return "pooled_setup_context_not_available_at_decision"
        if context.coverage != self.required_context_coverage:
            return "pooled_setup_coverage_unverified"
        if context.setup_coverage != self.required_setup_coverage:
            return "pooled_setup_coverage_unverified"
        if self.pooled_routing_contract != POOLED_ROUTING_CONTRACT:
            return "pooled_routing_contract_unverified"

        routing = resolve_pooled_routing(
            context,
            strategy=signal.strategy,
            setup_family=signal.setup_family,
            side=signal.side,
            decision_at=signal.known_at_ts,
        )
        if routing.status == "primary_selected":
            return None
        if routing.status == "ambiguous_selected_setup":
            return "pooled_setup_selection_ambiguous"
        if routing.status == "counterfactual_only":
            return "pooled_counterfactual_setup_non_executable"
        return "pooled_setup_not_known_at_decision"


def load_scope(path: str | Path = SCOPE_PATH) -> BaselineScope | ResearchScope:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    scope_type = ResearchScope if payload.get("scope_kind") == "research" else BaselineScope
    scope = scope_type.model_validate(payload)
    lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    if lock != {scope.scope_id: scope.sha256} or not scope.scope_id.endswith(f"_v{scope.version}"):
        label = "research" if isinstance(scope, ResearchScope) else "baseline"
        raise ValueError(f"Frozen {label} scope changed; publish a new version")
    return scope


def load_research_scope(path: str | Path = RESEARCH_SCOPE_PATH) -> ResearchScope:
    """Load and verify the conditional pooled-model research scope."""
    scope = load_scope(path)
    if not isinstance(scope, ResearchScope):
        raise ValueError("Expected a locked research scope")
    return scope
