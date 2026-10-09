"""Controlled regression cases; these do not estimate biological accuracy."""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from llm_pathway_curator import ReviewConfig, review_enrichment
from llm_pathway_curator.adapters.fgsea import (
    FgseaAdapterConfig,
    fgsea_to_evidence_table,
)
from llm_pathway_curator.grounding import inspect_text


@pytest.mark.parametrize("kind", ["ES", "NES"])
@pytest.mark.parametrize("source", ["fgsea", "custom_collection"])
def test_adapter_preserves_statistic_identity_through_public_report(tmp_path, kind, source):
    raw = pd.DataFrame([{"pathway": "SET_A", kind: -0.6, "padj": 0.01, "leadingEdge": "A;B"}])
    ev = fgsea_to_evidence_table(raw, config=FgseaAdapterConfig(source_name=source))
    ev.to_csv(tmp_path / "evidence.tsv", sep="\t", index=False)
    (tmp_path / "card.json").write_text('{"comparison":"treated versus control"}')
    result = review_enrichment(
        ReviewConfig(
            str(tmp_path / "evidence.tsv"), str(tmp_path / "card.json"), str(tmp_path / "out")
        )
    )
    record = json.loads(Path(result.artifacts["report_jsonl"]).read_text())
    assert record["evidence"]["stat_kind"] == kind
    assert f"({kind}=-0.6;" in record["source_statement"]
    assert "negative enrichment" in record["source_statement"]
    checked = inspect_text(f"{kind}=-0.6; q=0.01", record["evidence"], 0.05)
    assert checked["explicit_numeric_fields"] == ["qval", "stat"]
    assert not checked["findings"]


def test_nes_still_takes_precedence_over_es():
    ev = fgsea_to_evidence_table(
        pd.DataFrame([{"pathway": "A", "ES": 0.6, "NES": 1.8, "padj": 0.01, "leadingEdge": "A"}])
    )
    assert ev.iloc[0]["stat"] == 1.8
    assert ev.iloc[0]["stat_kind"] == "NES"


@pytest.mark.parametrize("reported,source_kind", [("ES", "NES"), ("NES", "ES")])
def test_different_statistic_types_are_not_compared_as_the_same_number(reported, source_kind):
    result = inspect_text(
        f"{reported}=1.5",
        {"stat": 1.5, "stat_kind": source_kind, "qval": 0.01, "direction": "up"},
        0.05,
    )
    assert result["explicit_numeric_fields"] == []
    assert result["findings"][0]["code"] == "STATISTIC_TYPE_UNVERIFIED"
    assert result["findings"][0]["severity"] == "REVIEW"


@pytest.mark.parametrize("value", [None, "", "  "])
def test_missing_card_attribute_is_unverified_not_a_contradiction(tmp_path, value):
    (tmp_path / "evidence.tsv").write_text(
        "term_id\tterm_name\tsource\tstat\tqval\tdirection\tevidence_genes\n"
        "A\tPath A\tfgsea\t1.5\t0.01\tup\tG1;G2\n"
    )
    card = {"comparison": "treated versus control", "tissue": value}
    (tmp_path / "card.json").write_text(json.dumps(card))
    with (tmp_path / "claims.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["term_id", "text", "tissue"])
        writer.writerow(["A", "Enrichment was observed.", "lung"])
    result = review_enrichment(
        ReviewConfig(
            str(tmp_path / "evidence.tsv"),
            str(tmp_path / "card.json"),
            str(tmp_path / "out"),
            str(tmp_path / "claims.tsv"),
        )
    )
    record = json.loads(Path(result.artifacts["report_jsonl"]).read_text())
    review = record["submitted_reviews"][0]
    assert review["prose_disposition"] == "ABSTAIN"
    assert review["findings"][0]["code"] == "CONTEXT_ATTRIBUTE_UNVERIFIED"
    assert review["findings"][0]["severity"] == "REVIEW"


@pytest.mark.parametrize("qval", [0.01, 0.2])
@pytest.mark.parametrize(
    "text",
    [
        "The result is statistically significant at the unadjusted level (p=0.001).",
        "The nominally statistically significant result requires adjusted-value review.",
        "The result is not statistically significant at the uncorrected level.",
    ],
)
def test_explicit_unadjusted_significance_cannot_be_judged_using_q_alone(qval, text):
    result = inspect_text(
        text, {"stat": 1.5, "stat_kind": "NES", "qval": qval, "direction": "up"}, 0.05
    )
    assert not any(f["severity"] == "ERROR" for f in result["findings"])
    assert result["findings"][0]["code"] == "SIGNIFICANCE_BASIS_REQUIRES_REVIEW"
    assert result["automatic_prose_acceptance"] is False


def test_unadjusted_qualification_does_not_hide_a_separate_adjusted_contradiction():
    text = (
        "The result is statistically significant at the unadjusted level, "
        "but statistically significant after adjustment (q=0.2)."
    )
    result = inspect_text(
        text, {"stat": 1.5, "stat_kind": "NES", "qval": 0.2, "direction": "up"}, 0.05
    )
    codes = [f["code"] for f in result["findings"]]
    assert codes == ["SIGNIFICANCE_BASIS_REQUIRES_REVIEW", "SIGNIFICANCE_MISMATCH"]
    for finding in result["findings"]:
        assert text[finding["start"] : finding["end"]] == finding["quote"]
