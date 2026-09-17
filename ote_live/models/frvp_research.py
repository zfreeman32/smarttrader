"""A9/A10 prospective research priorities, independent of frozen model policies."""
from dataclasses import asdict, dataclass


FRVP_RESEARCH_ROSTER_VERSION = "es-frvp-research-priorities-v1"


@dataclass(frozen=True)
class ResearchPriority:
    tier: str
    label: str
    note: str
    candidate_block_reason: str | None = None

    @property
    def foreground(self) -> bool:
        return self.tier in {"priority", "exploratory", "collect_matches"}

    def payload(self) -> dict:
        return {"contract_version": FRVP_RESEARCH_ROSTER_VERSION, **asdict(self),
                "research_only": True, "promotion_authorized": False}


FRVP_RESEARCH_PRIORITIES = {
    "frvp_short_continuation_tcn_v1": ResearchPriority(
        "priority", "Priority shadow comparison",
        "Short-side economics and drawdown restrictions remain; no promotion or paper-trial authorization."),
    "frvp_long_continuation_xgb_v1": ResearchPriority(
        "priority", "Priority shadow comparison",
        "Qualification requires verified inputs, full policy and the existing baseline prerequisites."),
    "frvp_long_continuation_setup3_xgb_v1": ResearchPriority(
        "exploratory", "Exploratory long S3",
        "Exploratory specialist; matching observations are research evidence only."),
    "frvp_long_continuation_setup5_xgb_v1": ResearchPriority(
        "collect_matches", "Collect matching long S5",
        "Assess long S5 using matching long observations; predominantly short-S5 results do not establish long-model quality."),
    "frvp_long_reversal_xgb_v1": ResearchPriority(
        "background", "Background research", "Retain the frozen long-reversal research control and diagnostic scores."),
    "frvp_long_meta_xgb_v1": ResearchPriority(
        "background", "Background research", "Research checkpoint; existing economics and promotion restrictions remain."),
    "frvp_short_meta_xgb_v1": ResearchPriority(
        "background", "Background research", "Short sentinel; existing short-side economics and drawdown restrictions remain."),
    "frvp_long_reversal_setup1_xgb_v1": ResearchPriority(
        "background", "Background long S1", "Insufficient matching examples are insufficient evidence, not model failure."),
    "frvp_long_continuation_setup2_xgb_v1": ResearchPriority(
        "background", "Background long S2", "Insufficient matching examples are insufficient evidence, not model failure."),
    "frvp_long_reversal_setup6_xgb_v1": ResearchPriority(
        "background", "Background long S6", "Retain diagnostic scores and matching/rejected observations for research."),
    "frvp_long_reversal_setup4_xgb_v1": ResearchPriority(
        "paused", "S4 qualified candidacy paused",
        "Retain artifacts and diagnostic scores. Resume only after a model-specific policy review and roster version change.",
        "frvp_s4_model_specific_policy_pending"),
    "frvp_short_reversal_xgb_v1": ResearchPriority(
        "retired", "Retired", "Retained artifacts and historical diagnostics; retirement remains in force.",
        "frvp_short_reversal_retired"),
}


def frvp_research_priority(model_id: str) -> ResearchPriority | None:
    return FRVP_RESEARCH_PRIORITIES.get(model_id)


def foreground_frvp_model(model_id: str) -> bool:
    priority = frvp_research_priority(model_id)
    return priority is not None and priority.foreground
