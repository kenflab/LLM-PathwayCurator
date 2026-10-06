"""Keep canonical candidates, deterministic statements and free-text review separate."""

from collections import Counter
from copy import deepcopy

from pydantic import ValidationError

from . import CONTRACT_VERSION, METHOD_ID
from .models import ASPECTS, Claim, Evidence, Review
from .structured import check_claim, utility_components
from .text_checks import explicit_numeric_checks, factual_statement

ISSUES = {
    "numerics_and_direction": "NUMERICAL_OR_DIRECTION_MISMATCH",
    "metadata": "METADATA_MISMATCH",
    "causality": "UNSUPPORTED_CAUSAL_LANGUAGE",
    "disease_specificity": "UNSUPPORTED_DISEASE_SPECIFICITY",
    "literal_pathway_event": "UNSUPPORTED_LITERAL_PATHWAY_EVENT",
    "evidence_scope": "OTHER_CONCERN",
}


def prepare(evidence_data, claim_data):
    result = {
        "contract_version": CONTRACT_VERSION,
        "method_id": METHOD_ID,
        "raw_evidence_preserved": deepcopy(evidence_data),
        "submitted_claim_preserved": deepcopy(claim_data),
        "statistical_candidate_retained": False,
        "statistical_fdr_0_05": None,
        "canonical_status": "INVALID",
        "contract_status": "NOT_EVALUATED",
        "deterministic_reason_codes": [],
        "text_numeric_checks": [],
        "numeric_text_coverage": "LIMITED_EXPLICIT_SYNTAX_NOT_ALL_PROSE",
        "execution_status": "INPUT_INVALID",
        "semantic_status": "NOT_RUN",
        "interpretation_status": "INCOMPLETE",
        "interpretation_eligible": False,
        "automatic_free_text_publication_allowed": False,
        "biological_truth": "NOT_ESTABLISHED",
        "source_artifact_content_independently_authenticated": False,
        "review": None,
        "issue_codes": [],
        "technical_error": None,
    }
    try:
        evidence = Evidence.model_validate(evidence_data)
    except ValidationError as error:
        result["technical_error"] = {"code": "CANONICAL_INPUT_INVALID", "detail": str(error)}
        return result, None, None
    result.update(
        canonical_status="VALID",
        statistical_candidate_retained=True,
        statistical_fdr_0_05=evidence.q_value <= 0.05,
        factual_enrichment_statement=factual_statement(evidence),
        components=utility_components(evidence),
    )
    try:
        claim = Claim.model_validate(claim_data)
    except ValidationError as error:
        result["technical_error"] = {"code": "CLAIM_SCHEMA_INVALID", "detail": str(error)}
        return result, evidence, None
    structured = check_claim(evidence, claim)
    numeric = explicit_numeric_checks(claim.text, evidence)
    text_codes = sorted({item["code"] for item in numeric if not item["satisfied"]})
    result.update(
        contract_status="VIOLATION"
        if structured or text_codes
        else "NO_STRUCTURED_VIOLATION_DETECTED",
        deterministic_reason_codes=structured + text_codes,
        text_numeric_checks=numeric,
        execution_status="COMPLETE" if structured or text_codes else "NOT_RUN",
        semantic_status="NOT_RUN_CONTRACT_VIOLATION"
        if structured
        else "NOT_RUN_TEXT_VIOLATION"
        if text_codes
        else "NOT_RUN",
        interpretation_status="WITHHELD_CONTRACT_VIOLATION"
        if structured
        else "WITHHELD_TEXT_VIOLATION"
        if text_codes
        else "NOT_REVIEWED",
        issue_codes=["NUMERICAL_OR_DIRECTION_MISMATCH"] if text_codes else [],
    )
    return result, evidence, claim


def apply_review(result, review: Review):
    if result["contract_status"] != "NO_STRUCTURED_VIOLATION_DETECTED":
        raise ValueError("An LLM cannot override a deterministic violation")
    result = deepcopy(result)
    concerns = [name for name in ASPECTS if getattr(review, name).verdict == "CONCERN"]
    unresolved = [name for name in ASPECTS if getattr(review, name).verdict == "UNRESOLVED"]
    if concerns:
        semantic, interpretation = "CONCERN_REQUIRES_REVIEW", "WITHHELD_REVIEW_REQUIRED"
    elif unresolved:
        semantic, interpretation = "UNRESOLVED", "WITHHELD_UNRESOLVED"
    else:
        semantic, interpretation = "NO_CONCERN_DETECTED", "ELIGIBLE_FOR_REVIEW"
    result.update(
        execution_status="COMPLETE",
        semantic_status=semantic,
        interpretation_status=interpretation,
        interpretation_eligible=semantic == "NO_CONCERN_DETECTED",
        review=review.model_dump(),
        technical_error=None,
        issue_codes=[ISSUES[name] for name in concerns] + ["UNRESOLVED:" + x for x in unresolved],
    )
    return result


def incomplete(result, code, detail):
    result = deepcopy(result)
    result.update(
        execution_status="INCOMPLETE",
        semantic_status="NOT_COMPLETED",
        interpretation_status="INCOMPLETE",
        interpretation_eligible=False,
        technical_error={"code": code, "detail": detail},
        review=None,
        issue_codes=[],
    )
    return result


def summarize(results):
    if not results:
        raise ValueError("Require a nonempty candidate census")
    complete = all(r["execution_status"] == "COMPLETE" for r in results)
    return {
        "method_id": METHOD_ID,
        "scope": "development_integration_only",
        "records": len(results),
        "technical_or_input_errors": sum(r["technical_error"] is not None for r in results),
        "retained_statistical_candidates": sum(
            r["statistical_candidate_retained"] for r in results
        ),
        "execution_status_counts": dict(Counter(r["execution_status"] for r in results)),
        "semantic_status_counts": dict(Counter(r["semantic_status"] for r in results)),
        "interpretation_eligible_count": sum(r["interpretation_eligible"] for r in results),
        "interpretation_eligible_fraction": sum(r["interpretation_eligible"] for r in results)
        / len(results)
        if complete
        else None,
        "all_requested_reviews_complete": complete,
        "biological_accuracy_or_replication_estimated": False,
        "membership_exported": False,
        "automatic_free_text_publication_allowed": False,
    }
