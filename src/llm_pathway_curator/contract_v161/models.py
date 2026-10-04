"""Strict wire contracts. Unknown fields and implicit type coercions are rejected."""

from __future__ import annotations

from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .registry import TCGA_NAMES

Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Direction = Literal["UP", "DOWN", "ZERO"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class SourceRef(StrictModel):
    artifact: Text
    sha256: SHA256
    locator: Text


class MetadataValue(StrictModel):
    value: Text
    source: SourceRef


class Contrast(StrictModel):
    comparison_id: Literal["TP53_MUTANT_vs_TP53_WILD_TYPE"]
    positive_group: Literal["TP53_MUTANT"]
    reference_group: Literal["TP53_WILD_TYPE"]
    study_design: Literal["observational"]


class Stability(StrictModel):
    value: Fraction | None
    kind: Literal["empirical_sample_resampling", "synthetic_gene_perturbation"]
    definition: Text
    protocol_id: Text
    source: SourceRef
    source_column: Text
    missing_reason: Text | None

    @model_validator(mode="after")
    def explicit_missingness(self):
        if (self.value is None) != (self.missing_reason is not None):
            raise ValueError("Missing S requires a reason; present S requires missing_reason=null")
        return self


def unique_strings(items: list[str]) -> list[str]:
    if any(not x or x != x.strip() for x in items) or len(set(items)) != len(items):
        raise ValueError("Identifiers must be nonblank, unpadded, and unique")
    return items


class Evidence(StrictModel):
    schema_version: Literal["CRM_R1_EVIDENCE_v16"]
    evidence_id: Text
    cohort_id: Text
    cohort_name: Text
    split_id: Text
    contrast: Contrast
    term_id: Text
    term_name: Text
    gene_set_version: Text
    gene_set_sha256: SHA256
    nes: Finite
    q_value: Fraction
    direction: Direction
    leading_edge_genes: list[Text]
    metadata: dict[str, MetadataValue]
    source: SourceRef
    stability: Stability | None

    _unique_genes = field_validator("leading_edge_genes")(unique_strings)

    @model_validator(mode="after")
    def canonical_consistency(self):
        if TCGA_NAMES.get(self.cohort_id) != self.cohort_name:
            raise ValueError("TCGA code/name must match the frozen project-name registry")
        expected = "UP" if self.nes > 0 else "DOWN" if self.nes < 0 else "ZERO"
        if self.direction != expected:
            raise ValueError("Canonical direction disagrees with NES under the declared contrast")
        allowed_metadata = {"specimen", "tissue", "histological_subtype", "disease_subtype"}
        if set(self.metadata) - allowed_metadata:
            raise ValueError("Only declared source metadata fields may be passed to the reviewer")
        return self


class Claim(StrictModel):
    schema_version: Literal["CRM_R1_CLAIM_v16"]
    claim_id: Text
    evidence_id: Text
    text: Text
    cohort_id: Text
    cohort_name: Text
    split_id: Text
    comparison_id: Text
    term_id: Text
    gene_set_version: Text
    gene_set_sha256: SHA256
    reported_nes: Finite
    reported_q_value: Fraction
    direction: Direction
    supporting_genes: list[Text]
    metadata_assertions: dict[str, Text]
    significance_assertion: Literal["SIGNIFICANT", "NOT_SIGNIFICANT", "NOT_STATED"]
    causal_assertion: bool
    disease_specificity_assertion: bool
    literal_pathway_event_assertion: bool

    _unique_genes = field_validator("supporting_genes")(unique_strings)


class GenerationOptions(StrictModel):
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 0.0
    seed: int = 42
    num_ctx: Annotated[int, Field(ge=4096)] = 16384
    num_predict: Annotated[int, Field(ge=128)] = 2048
    top_k: Annotated[int, Field(ge=1)] = 40
    top_p: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.9
    repeat_penalty: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.1


class ModelConfig(StrictModel):
    host: Text
    model: Text
    model_digest: SHA256
    server_version: Text
    options: GenerationOptions = Field(default_factory=GenerationOptions)
    timeout_seconds: Annotated[float, Field(gt=0, le=600, allow_inf_nan=False)] = 600.0
    keep_alive: Literal["5m"] = "5m"

    @field_validator("host")
    @classmethod
    def local_plain_url(cls, value):
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or any(x in value for x in "[]() ")
            and parsed.hostname != "::1"
        ):
            raise ValueError("Expected a plain local HTTP URL; Markdown links are invalid")
        _ = parsed.port
        return value.rstrip("/")


class PromptPolicy(StrictModel):
    max_evidence_genes: Annotated[int, Field(ge=1)] | None = None
    ordering: Literal["claim_genes_then_lexicographic_remaining"] = (
        "claim_genes_then_lexicographic_remaining"
    )


class TechnicalIssue(StrictModel):
    code: Text
    detail: str


ASPECTS = (
    "numerics_and_direction",
    "metadata",
    "causality",
    "disease_specificity",
    "literal_pathway_event",
    "evidence_scope",
)


class AspectReview(StrictModel):
    verdict: Literal["CLEAR", "CONCERN", "UNRESOLVED"]
    reason: Text
    claim_quotes: list[Text]
    evidence_pointers: list[Text]

    @model_validator(mode="after")
    def concern_needs_text(self):
        if self.verdict == "CONCERN" and not self.claim_quotes:
            raise ValueError("A concern requires an exact excerpt from the actual claim text")
        if self.verdict == "CONCERN" and not self.evidence_pointers:
            raise ValueError("A concern requires a pointer to supplied evidence or its metadata")
        return self


class Review(StrictModel):
    numerics_and_direction: AspectReview
    metadata: AspectReview
    causality: AspectReview
    disease_specificity: AspectReview
    literal_pathway_event: AspectReview
    evidence_scope: AspectReview


class AuditResult(StrictModel):
    contract_version: Literal["CRM_R1_CONTRACT_v16.1"]
    method_id: Literal["contract_separated_v16_1"]
    raw_evidence_preserved: dict
    submitted_claim_preserved: dict
    statistical_candidate_retained: bool
    statistical_fdr_0_05: bool | None
    canonical_status: Literal["VALID", "INVALID"]
    contract_status: Literal["VIOLATION", "NO_STRUCTURED_VIOLATION_DETECTED", "NOT_EVALUATED"]
    deterministic_reason_codes: list[str]
    text_numeric_checks: list[dict]
    numeric_text_coverage: Literal["LIMITED_EXPLICIT_SYNTAX_NOT_ALL_PROSE"]
    execution_status: Literal["COMPLETE", "INCOMPLETE", "INPUT_INVALID", "NOT_RUN"]
    semantic_status: Literal[
        "NOT_RUN",
        "NOT_RUN_CONTRACT_VIOLATION",
        "NOT_RUN_TEXT_VIOLATION",
        "NOT_COMPLETED",
        "UNRESOLVED",
        "CONCERN_REQUIRES_REVIEW",
        "NO_CONCERN_DETECTED",
    ]
    interpretation_status: Literal[
        "INCOMPLETE",
        "NOT_REVIEWED",
        "WITHHELD_CONTRACT_VIOLATION",
        "WITHHELD_TEXT_VIOLATION",
        "WITHHELD_UNRESOLVED",
        "WITHHELD_REVIEW_REQUIRED",
        "ELIGIBLE_FOR_REVIEW",
    ]
    interpretation_eligible: bool
    automatic_free_text_publication_allowed: Literal[False]
    biological_truth: Literal["NOT_ESTABLISHED"]
    source_artifact_content_independently_authenticated: Literal[False]
    review: Review | None
    issue_codes: list[str]
    technical_error: TechnicalIssue | None
    factual_enrichment_statement: str | None = None
    components: dict | None = None
    request_key: SHA256 | None = None
    cache_hit: bool | None = None
    successful_attempt: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def enforce_separation(self):
        if self.statistical_candidate_retained != (self.canonical_status == "VALID"):
            raise ValueError("Canonical validity alone determines candidate retention")
        eligible = (
            self.execution_status == "COMPLETE"
            and self.contract_status == "NO_STRUCTURED_VIOLATION_DETECTED"
            and self.semantic_status == "NO_CONCERN_DETECTED"
            and not self.deterministic_reason_codes
            and self.review is not None
        )
        if self.interpretation_eligible != eligible:
            raise ValueError("Interpretation eligibility is inconsistent")
        if self.execution_status == "INCOMPLETE" and self.technical_error is None:
            raise ValueError("Technical incompletion needs an error record")
        if self.components and (
            self.components["C"]["value"] is not None
            or self.components["utility_score"] is not None
            or self.components["used_for_candidate_selection"]
        ):
            raise ValueError("No proxy or confidence-based utility is allowed")
        return self
