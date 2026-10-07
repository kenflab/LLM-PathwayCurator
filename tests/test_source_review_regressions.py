"""Regressions for incorrect source binding and ordinary numeric wording.

These controlled examples exercise software behavior, not biological accuracy.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_pathway_curator import ReviewConfig, review_enrichment
from llm_pathway_curator.grounding import inspect_text


@pytest.mark.parametrize(
    ("text", "qval", "mismatch"),
    [
        ("FDR=5%.", 0.05, False),
        ("q=5.0%", 0.0504, False),
        ("q=5.0%", 0.0506, True),
        ("q=5.0%", 0.055, True),
        ("q<5%", 0.1, True),
        ("q<5%", 0.05, True),
        ("q<=5%", 0.05, False),
        ("q≤5%", 0.05, False),
        ("q>5%", 0.04, True),
        ("q≥5%", 0.05, False),
        ("q=0.5%", 0.005, False),
        ("q=0%", 0.0, False),
        ("q=0%", 1e-8, True),
    ],
)
def test_adjusted_values_in_percent_keep_scale_precision_and_bounds(text, qval, mismatch):
    source = {"stat": 1.5, "stat_kind": "NES", "qval": qval, "direction": "up"}
    result = inspect_text(text, source, 0.05)
    errors = [f for f in result["findings"] if f["severity"] == "ERROR"]
    assert bool(errors) is mismatch
    assert all(f["code"] == "NUMERIC_MISMATCH" for f in errors)
    assert result["explicit_numeric_fields"] == ["qval"]
    for f in result["findings"]:
        assert text[f["start"] : f["end"]] == f["quote"]
        assert "%" in f["quote"]
    assert result["automatic_prose_acceptance"] is False


def test_percent_unit_is_not_silently_assumed_for_nes():
    result = inspect_text(
        "NES=5%", {"stat": 5, "stat_kind": "NES", "qval": 0.01, "direction": "up"}, 0.05
    )
    assert result["limited_checks_status"] == "REVIEW_FLAGGED"
    assert result["findings"][0]["code"] == "NUMERIC_UNIT_UNVERIFIED"
    assert result["explicit_numeric_fields"] == []


def test_not_only_is_not_a_negated_significance_assertion():
    result = inspect_text(
        "The result is not only statistically significant but also reproducible.",
        {"stat": 1.5, "stat_kind": "NES", "qval": 0.01, "direction": "up"},
        0.05,
    )
    assert result["findings"] == []
    assert result["automatic_prose_acceptance"] is False


def test_negation_of_causality_does_not_become_significance_failure():
    text = (
        "Although it does not establish causality, "
        "statistically significant enrichment was observed."
    )
    result = inspect_text(
        text, {"stat": 1.5, "stat_kind": "NES", "qval": 0.01, "direction": "up"}, 0.05
    )
    assert not any(f["severity"] == "ERROR" for f in result["findings"])
    assert result["limited_checks_status"] == "REVIEW_FLAGGED"
    assert result["findings"][0]["code"] == "SIGNIFICANCE_SCOPE_REQUIRES_REVIEW"


@pytest.mark.parametrize(
    ("text", "qval"),
    [
        ("The result is not statistically significant.", 0.01),
        ("The result is statistically significant.", 0.2),
    ],
)
def test_direct_significance_contradictions_still_fail(text, qval):
    result = inspect_text(
        text, {"stat": 1.5, "stat_kind": "NES", "qval": qval, "direction": "up"}, 0.05
    )
    assert any(
        f["code"] == "SIGNIFICANCE_MISMATCH" and f["severity"] == "ERROR"
        for f in result["findings"]
    )


def _inputs(tmp_path: Path, *, source="fgsea", stat=1.5, direction="up"):
    evidence = tmp_path / "evidence.tsv"
    with evidence.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["term_id", "term_name", "source", "stat", "qval", "direction", "evidence_genes"]
        )
        writer.writerow(["SET_A", "Response A", source, stat, 0.01, direction, "GENE1"])
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"comparison": "treated versus control"}), encoding="utf-8")
    return evidence, card


@pytest.mark.parametrize("source", [" fgsea", "fgsea ", " fgsea ", " FGSEA "])
def test_source_padding_cannot_bypass_nes_sign_validation(tmp_path, source):
    evidence, card = _inputs(tmp_path, source=source, stat=-1.5, direction="up")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="NES direction mismatch"):
        review_enrichment(ReviewConfig(str(evidence), str(card), str(out)))
    assert not out.exists()


def test_source_padding_with_valid_direction_is_normalized(tmp_path):
    evidence, card = _inputs(tmp_path, source=" fgsea ", stat=-1.5, direction="down")
    result = review_enrichment(ReviewConfig(str(evidence), str(card), str(tmp_path / "out")))
    record = json.loads(Path(result.artifacts["report_jsonl"]).read_text())
    assert record["evidence"]["source"] == "fgsea"
    assert record["evidence"]["stat_kind"] == "NES"
    assert "negative enrichment" in record["source_statement"]


@pytest.mark.parametrize(
    ("term_id", "source"), [("SET_B", "fgsea"), ("SET_A", "other"), ("", "other")]
)
def test_conflicting_declared_evidence_identity_stops_before_output(tmp_path, term_id, source):
    evidence, card = _inputs(tmp_path)
    claims = tmp_path / "claims.tsv"
    with claims.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["term_uid", "term_id", "source", "text"])
        writer.writerow(["fgsea:SET_A", term_id, source, "NES=1.5; q=0.01."])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Conflicting evidence identity"):
        review_enrichment(ReviewConfig(str(evidence), str(card), str(out), str(claims)))
    assert not out.exists()


def test_consistent_redundant_identity_preserves_text(tmp_path):
    evidence, card = _inputs(tmp_path)
    claims = tmp_path / "claims.tsv"
    text = "NES=1.5; q=1%."
    with claims.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["term_uid", "term_id", "source", "text"])
        writer.writerow(["fgsea:SET_A", " SET_A ", " fgsea ", text])
    result = review_enrichment(
        ReviewConfig(str(evidence), str(card), str(tmp_path / "out"), str(claims))
    )
    record = json.loads(Path(result.artifacts["report_jsonl"]).read_text())
    checked = record["submitted_reviews"][0]
    assert checked["text"] == text
    assert checked["limited_checks_status"] == "NO_LIMITED_FLAG"
    assert checked["prose_disposition"] == "ABSTAIN"
