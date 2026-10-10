"""Controlled parser regressions, not estimates of natural-language accuracy."""

import pytest

from llm_pathway_curator.grounding import inspect_text


def check(text, q=0.2):
    return inspect_text(
        text, {"stat": -1.2, "stat_kind": "NES", "qval": q, "direction": "down"}, 0.05
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "significantly downregulated",
        "significantly down-regulated",
        "significantly up regulated",
        "significant down-regulation",
        "significant upregulation",
        "significant up regulation",
    ],
)
@pytest.mark.parametrize("q,expect_error", [(0.2, True), (0.05, False), (0.01, False)])
def test_regulation_wording_uses_declared_cutoff_and_preserves_spans(phrase, q, expect_error):
    text = f"The result shows {phrase} in the treated group."
    findings = check(text, q)["findings"]
    assert any(f["code"] == "SIGNIFICANCE_MISMATCH" for f in findings) is expect_error
    for finding in findings:
        assert text[finding["start"] : finding["end"]] == finding["quote"]


@pytest.mark.parametrize(
    "text",
    [
        "The term is not significantly downregulated.",
        "There is no significant down-regulation.",
        "The nominally significant down-regulation requires review.",
        "The term is significantly downregulated at the unadjusted level (p=0.01).",
        "The result shows biologically significant down-regulation.",
        "The result shows clinically significant enrichment.",
        "Significant clinical benefit has not been established.",
        "The term is downregulated, with moderate statistical support.",
    ],
)
def test_negation_and_alternative_meanings_do_not_create_false_contradictions(text):
    assert not any(f["severity"] == "ERROR" for f in check(text)["findings"])


def test_non_significant_regulation_at_significant_q_is_a_declared_mismatch():
    result = check("The term is not significantly downregulated.", q=0.01)
    assert any(f["code"] == "SIGNIFICANCE_MISMATCH" for f in result["findings"])


def test_clause_qualification_does_not_hide_separate_adjusted_claim():
    text = (
        "There is significant down-regulation at the unadjusted level, "
        "but significant down-regulation after adjustment (q=0.2)."
    )
    result = check(text)
    codes = [f["code"] for f in result["findings"]]
    assert "SIGNIFICANCE_BASIS_REQUIRES_REVIEW" in codes
    assert codes.count("SIGNIFICANCE_MISMATCH") == 1
    assert result["automatic_prose_acceptance"] is False
