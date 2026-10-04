"""Canonical fields, candidate retention, and interpretation are separate axes."""

from __future__ import annotations

from .models import Claim, Evidence

FDR_THRESHOLD = 0.05
UNCERTAINTY_CODES = {"UNKNOWN_ASSOCIATION", "INSUFFICIENT_EVIDENCE", "EXTERNAL_KNOWLEDGE_ONLY"}


def check_claim(evidence: Evidence, claim: Claim) -> list[str]:
    reasons = []
    for name in (
        "evidence_id",
        "cohort_id",
        "cohort_name",
        "split_id",
        "term_id",
        "gene_set_version",
        "gene_set_sha256",
        "direction",
    ):
        if getattr(evidence, name) != getattr(claim, name):
            reasons.append("MISMATCH_" + name.upper())
    if claim.comparison_id != evidence.contrast.comparison_id:
        reasons.append("MISMATCH_COMPARISON_ID")
    if claim.reported_nes != evidence.nes:
        reasons.append("NES_TRANSCRIPTION")
    if claim.reported_q_value != evidence.q_value:
        reasons.append("Q_VALUE_TRANSCRIPTION")
    if not set(claim.supporting_genes) <= set(evidence.leading_edge_genes):
        reasons.append("GENE_NOT_IN_SUPPLIED_EVIDENCE")
    for name, stated in claim.metadata_assertions.items():
        if name not in evidence.metadata:
            reasons.append("METADATA_NOT_PROVIDED:" + name)
        elif stated != evidence.metadata[name].value:
            reasons.append("METADATA_MISMATCH:" + name)
    if claim.significance_assertion == "SIGNIFICANT" and evidence.q_value > FDR_THRESHOLD:
        reasons.append("UNSUPPORTED_FDR_SIGNIFICANCE")
    if claim.significance_assertion == "NOT_SIGNIFICANT" and evidence.q_value <= FDR_THRESHOLD:
        reasons.append("INCORRECT_NON_SIGNIFICANCE")
    if claim.causal_assertion:
        reasons.append("CAUSAL_ASSERTION_FROM_OBSERVATIONAL_COMPARISON")
    if claim.disease_specificity_assertion:
        reasons.append("DISEASE_SPECIFICITY_NOT_ESTABLISHED_BY_THIS_COMPARISON")
    if claim.literal_pathway_event_assertion:
        reasons.append("LITERAL_PATHWAY_EVENT_NOT_ESTABLISHED_BY_ENRICHMENT")
    return reasons


def utility_components(evidence: Evidence) -> dict:
    """No generic component auto-detection and no C or utility fallback."""
    stability = evidence.stability
    s = stability.value if stability is not None else None
    return {
        "E": {
            "value": abs(evidence.nes),
            "definition": "absolute_NES",
            "source": evidence.source.model_dump(),
            "source_field": "nes",
            "interpretation": "enrichment effect-size descriptor; not a probability",
        },
        "S": stability.model_dump()
        if stability
        else {"value": None, "missing_reason": "NO_STABILITY_COMPONENT_PROVIDED"},
        "C": {"value": None, "status": "NOT_DEFINED_OR_CALIBRATED_IN_V16_1"},
        "ES_diagnostic": abs(evidence.nes) * s if s is not None else None,
        "ES_status": "DEVELOPMENT_DIAGNOSTIC_ONLY" if s is not None else "NOT_COMPUTED_MISSING_S",
        "utility_score": None,
        "utility_status": "NOT_COMPUTED_NO_VALIDATED_C",
        "used_for_candidate_selection": False,
        "used_for_interpretation_eligibility": False,
    }
