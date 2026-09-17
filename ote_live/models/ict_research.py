"""A11 research roster; B2 must review artifacts before any candidacy release."""
from dataclasses import asdict, dataclass


ICT_RESEARCH_ROSTER_VERSION = "es-ict-research-priorities-v1"
ICT_LINEAGE_BLOCK_REASON = "ict_feature_contract_and_B2_artifact_review_pending"


@dataclass(frozen=True)
class ICTResearchPriority:
    tier: str
    label: str
    note: str
    candidate_block_reason: str = ICT_LINEAGE_BLOCK_REASON

    @property
    def foreground(self) -> bool:
        return self.tier == "priority_pending_review" and not self.candidate_block_reason

    def payload(self) -> dict:
        return {"contract_version": ICT_RESEARCH_ROSTER_VERSION, **asdict(self),
                "research_only": True, "promotion_authorized": False,
                "artifact_review_status": "unverified",
                "post_review_tier": "priority" if self.tier == "priority_pending_review" else "background"}


ICT_RESEARCH_PRIORITIES = {
    "ict_long_continuation_xgb_v1": ICTResearchPriority(
        "priority_pending_review", "Focused comparison pending review",
        "Long continuation: prioritize only after required inputs and B2 artifact lineage pass. "
        "The q40 concentration failure remains binding; no promotion or further simple pocket-pruning."),
    "ict_short_continuation_premium_discount_continuation_xgb_v1": ICTResearchPriority(
        "priority_pending_review", "Focused comparison pending review",
        "Short premium/discount specialist: prioritize only after required inputs and B2 artifact lineage pass. "
        "Existing concentration and promotion restrictions remain."),
    "ict_long_reversal_xgb_v1": ICTResearchPriority(
        "background", "Background research", "Retain long-reversal diagnostics and matching/rejected observations."),
    "ict_short_reversal_xgb_v1": ICTResearchPriority(
        "background", "Background research", "Single-trade concentration, profitable-quarter breadth and shock-day exposure remain unresolved."),
    "ict_short_continuation_xgb_v1": ICTResearchPriority(
        "background", "Background research", "Weak q50/q60 results remain; retain as a background comparison."),
    "ict_long_meta_xgb_v1": ICTResearchPriority(
        "background", "Background research", "B2 must also verify the missing live confluence-helper producers and their timing/parity."),
    "ict_short_meta_xgb_v1": ICTResearchPriority(
        "background", "Background research", "Confluence-helper lineage, concentration, profitable-quarter breadth and shock-day exposure remain unresolved."),
    "ict_short_reversal_ifvg_reversal_xgb_v1": ICTResearchPriority(
        "background", "Background IFVG comparison", "Retain IFVG specialist diagnostics and matching/rejected observations."),
    "ict_short_reversal_sweep_reclaim_xgb_v1": ICTResearchPriority(
        "background", "Background sweep comparison", "Retain sweep specialist diagnostics and matching/rejected observations."),
}

_UNREVIEWED_ICT = ICTResearchPriority(
    "background", "Unreviewed ICT research", "Models outside the reviewed roster remain quarantined pending explicit B2 review.")


def ict_research_priority(model_id: str) -> ICTResearchPriority | None:
    if not model_id.lower().startswith("ict_"):
        return None
    # New model names and generic input-contract stamps cannot authorize release.
    # B2 must introduce an evidence-backed, versioned roster change per artifact.
    return ICT_RESEARCH_PRIORITIES.get(model_id, _UNREVIEWED_ICT)
