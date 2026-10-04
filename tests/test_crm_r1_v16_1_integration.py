"""Actual package API, simulated transport; these are not live LLM performance tests."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from llm_pathway_curator.contract_pipeline import ContractRunConfig, run_contract_pipeline
from llm_pathway_curator.contract_v161.checks import ISSUES, apply_review, prepare
from llm_pathway_curator.contract_v161.models import ASPECTS, AuditResult, ModelConfig, Review
from llm_pathway_curator.contract_v161.prompt import make_request, strict_json
from llm_pathway_curator.contract_v161.runtime import (
    AttemptStore,
    TechnicalError,
    WireResponse,
    run_review,
)
from llm_pathway_curator.contract_v161.text_checks import explicit_numeric_checks

FIXTURE = Path(__file__).parent / "fixtures/crm_r1_v16_1/cases_v161.json"
CASES = json.loads(FIXTURE.read_text())["cases"]


@pytest.fixture
def model():
    return ModelConfig(
        host="http://127.0.0.1:11434",
        model="test-model",
        model_digest="a" * 64,
        server_version="synthetic",
    )


def review_dict(case=None, concern=None, unresolved=None):
    out = {}
    for name in ASPECTS:
        out[name] = {
            "verdict": "CONCERN"
            if name == concern
            else "UNRESOLVED"
            if name == unresolved
            else "CLEAR",
            "reason": "Simulated response for software tests, not model performance.",
            "claim_quotes": [case["claim"]["text"]] if name == concern else [],
            "evidence_pointers": ["/evidence/metadata"] if name == concern else [],
        }
    return out


def wire(model, review):
    return WireResponse(
        json.dumps(
            {
                "model": model.model,
                "done": True,
                "done_reason": "stop",
                "response": json.dumps(review),
            }
        ).encode()
    )


def predicted(case):
    issue = case["expected_v161"]["required_issue_code"]
    aspect = next((k for k, v in ISSUES.items() if v == issue), None)
    return review_dict(case, concern=aspect)


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_fixed_inputs_and_routing_with_simulated_responses(case, model, tmp_path):
    calls = []

    def transport(request):
        calls.append(request)
        assert set(request["envelope"]["payload"]["claim"]) == {"text"}
        return wire(model, predicted(case))

    result = run_review(case["evidence"], case["claim"], model, AttemptStore(tmp_path), transport)
    AuditResult.model_validate(result)
    expected = case["expected_v161"]
    for key in (
        "execution_status",
        "semantic_status",
        "interpretation_eligible",
        "statistical_candidate_retained",
        "deterministic_reason_codes",
    ):
        assert result[key] == expected[key]
    assert len(calls) == (not bool(expected["deterministic_reason_codes"]))
    assert result["components"]["C"]["value"] is None
    assert result["components"]["utility_score"] is None
    assert not result["automatic_free_text_publication_allowed"]


def test_unknown_fields_and_nonblank_reasons():
    for modifier in (
        lambda d: d.pop("metadata"),
        lambda d: d["causality"].update(reason=" "),
        lambda d: d.update(confidence=1.0),
    ):
        value = review_dict()
        modifier(value)
        with pytest.raises(ValueError):
            Review.model_validate(value)


def test_concern_has_own_issue_even_with_another_unresolved_aspect():
    case = next(c for c in CASES if c["case_id"] == "N03")
    result, _, _ = prepare(case["evidence"], case["claim"])
    applied = apply_review(result, Review(**review_dict(case, "causality", "evidence_scope")))
    assert applied["semantic_status"] == "CONCERN_REQUIRES_REVIEW"
    assert "UNSUPPORTED_CAUSAL_LANGUAGE" in applied["issue_codes"]
    assert "UNRESOLVED:evidence_scope" in applied["issue_codes"]


@pytest.mark.parametrize("bad", ["quote", "pointer", "old_schema", "empty_reason"])
def test_invalid_wire_is_incomplete_and_preserves_raw_candidate(bad, model, tmp_path):
    case = next(c for c in CASES if c["case_id"] == "N04")
    response = review_dict(case, "metadata")
    if bad == "quote":
        response["metadata"]["claim_quotes"] = ["not present in the input"]
    elif bad == "pointer":
        response["metadata"]["evidence_pointers"] = ["/evidence/metadata/invented"]
    elif bad == "old_schema":
        response = {"assessment": "NO_CONCERN_DETECTED", "reason": "Old format"}
    else:
        response["metadata"]["reason"] = ""
    result = run_review(
        case["evidence"],
        case["claim"],
        model,
        AttemptStore(tmp_path),
        lambda _: wire(model, response),
    )
    assert result["execution_status"] == "INCOMPLETE"
    assert result["statistical_candidate_retained"]
    assert result["technical_error"]["code"] == "RESPONSE_SCHEMA_INVALID"
    assert not list(tmp_path.glob("*/SUCCESS.json"))
    assert len(list(tmp_path.glob("*/attempt-*.result.json"))) == 1


def test_technical_retry_does_not_resample_valid_unresolved(model, tmp_path):
    case = CASES[0]
    calls = []

    def transport(_):
        calls.append(True)
        if len(calls) == 1:
            raise TechnicalError("CONNECTION_ERROR", "simulated", retryable=True)
        return wire(model, review_dict(unresolved="evidence_scope"))

    store = AttemptStore(tmp_path)
    result = run_review(
        case["evidence"], case["claim"], model, store, transport, sleep=lambda _: None
    )
    before = {p: p.read_bytes() for p in tmp_path.glob("*/*.json")}
    again = run_review(case["evidence"], case["claim"], model, store, transport)
    assert len(calls) == 2
    assert result["semantic_status"] == again["semantic_status"] == "UNRESOLVED"
    assert again["cache_hit"]
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_full_input_hash_and_explicit_gene_omission(model):
    from llm_pathway_curator.contract_v161.models import PromptPolicy

    case = CASES[0]
    _, ev, cl = prepare(case["evidence"], case["claim"])
    request = make_request(ev, cl, model, PromptPolicy(max_evidence_genes=1))
    assert request["envelope"]["payload"]["gene_display"]["omitted"] == 2
    changed = ev.model_copy(deep=True)
    changed.stability.value = 0.7
    other = make_request(changed, cl, model, PromptPolicy(max_evidence_genes=1))
    assert request["envelope"]["payload"] == other["envelope"]["payload"]
    assert request["key"] != other["key"]
    bad = model.model_copy(deep=True)
    bad.options.num_ctx = 4096
    with pytest.raises(ValueError, match="context budget"):
        make_request(ev, cl, bad)


@pytest.mark.parametrize(
    "text,matched",
    [
        ("NES=1.75; BH q=2e-2", True),
        ("NES=3.50; BH q=0.02", False),
        ("BH q < 0.05", True),
        ("BH q > 0.05", False),
        ("statistically significant at FDR 0.05", True),
        ("not statistically significant at FDR 0.05", False),
    ],
)
def test_explicit_numeric_relations_and_negation(text, matched):
    _, evidence, _ = prepare(CASES[0]["evidence"], CASES[0]["claim"])
    checks = explicit_numeric_checks(text, evidence)
    assert checks
    assert all(r["satisfied"] for r in checks) == matched
    assert all(text[r["start"] : r["end"]] == r["quote"] for r in checks)


def test_standard_statement_uses_canonical_not_mutated_claim():
    case = next(c for c in CASES if c["case_id"] == "N01")
    result, _, _ = prepare(case["evidence"], case["claim"])
    assert "NES=1.75" in result["factual_enrichment_statement"]
    assert "NES=3.50" in result["submitted_claim_preserved"]["text"]
    broken = deepcopy(case["evidence"])
    broken["cohort_name"] = "invented"
    result, _, _ = prepare(broken, case["claim"])
    assert not result["statistical_candidate_retained"]
    assert result["raw_evidence_preserved"] == broken


def test_package_api_writes_separate_ledgers_and_never_calls_legacy(tmp_path, model, monkeypatch):
    import llm_pathway_curator.pipeline as legacy

    def forbidden(*args, **kwargs):
        pytest.fail("Legacy selection/gating was called")

    monkeypatch.setattr(legacy, "run_pipeline", forbidden)
    cases = [CASES[0], next(c for c in CASES if c["case_id"] == "N01")]
    for name, value in (
        ("evidence", [c["evidence"] for c in cases]),
        ("claims", [c["claim"] for c in cases]),
        ("model", model.model_dump()),
    ):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    cfg = ContractRunConfig(
        str(tmp_path / "evidence.json"),
        str(tmp_path / "claims.json"),
        str(tmp_path / "out"),
        "ollama",
        str(tmp_path / "model.json"),
        str(tmp_path / "cache"),
    )
    output = run_contract_pipeline(cfg, transport=lambda _: wire(model, review_dict()))
    assert output.summary["retained_statistical_candidates"] == 2
    assert output.summary["interpretation_eligible_count"] == 1
    assert (
        len(strict_json(Path(output.artifacts["standard_statements.private.json"]).read_bytes()))
        == 2
    )
    assert "audit_log.tsv" not in output.artifacts
    with pytest.raises(ValueError, match="new output"):
        run_contract_pipeline(cfg, transport=lambda _: wire(model, review_dict()))
