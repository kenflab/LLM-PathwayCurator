"""Software conformance checks, including source linkage and cautious wording.

Synthetic examples here are not an independent biological performance dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from llm_pathway_curator import ReviewConfig, RunConfig, review_enrichment, run_pipeline
from llm_pathway_curator.grounding import inspect_text


@pytest.fixture
def files(tmp_path):
    evidence = tmp_path / "source.tsv"
    with evidence.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["term_id", "term_name", "source", "stat", "qval", "direction", "evidence_genes"]
        )
        writer.writerow(
            [
                "SET_A",
                "Response A",
                "fgsea",
                -2.1340243231909,
                1.15266177813956e-8,
                "down",
                "GENE1;GENE2",
            ]
        )
        writer.writerow(["SET_B", "Response B", "fgsea", 1.28, 0.6644, "up", "GENE3"])
        writer.writerow(["SET_C", "Response C", "ora", 8, "NA", "na", "GENE4"])
    card = tmp_path / "study.json"
    card.write_text(
        json.dumps(
            {
                "comparison": "drug-treated versus vehicle",
                "tissue": "airway",
                "condition": "inflammation",
                "study_design": "interventional",
            }
        )
    )
    return evidence, card


def cfg(files, out, claims=None, **options):
    evidence, card = files
    return ReviewConfig(
        str(evidence), str(card), str(out), str(claims) if claims else None, **options
    )


def drafts(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_default_pipeline_does_not_use_models_or_hash_context(files, tmp_path, monkeypatch):
    import llm_pathway_curator.pipeline as pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("Default source reporting must not invoke the legacy/model path")

    monkeypatch.setattr(pipeline, "get_backend_from_env", forbidden)
    monkeypatch.setattr(pipeline, "_proxy_context_review", forbidden)
    monkeypatch.setenv("LLMPATH_BACKEND", "gemini")
    monkeypatch.setenv("LLMPATH_CONTEXT_GATE_MODE", "hard")
    result = run_pipeline(RunConfig(str(files[0]), str(files[1]), str(tmp_path / "out")))
    meta = json.loads(Path(result.meta_path).read_text())
    assert meta["model_calls"] == 0 and meta["context_proxy_used"] is False
    assert meta["selected_count"] == 1 and meta["candidate_retained_count"] == 3
    rows = [
        json.loads(line) for line in Path(result.artifacts["report_jsonl"]).read_text().splitlines()
    ]
    assert [r["decision_status"] for r in rows] == ["PASS", "ABSTAIN", "ABSTAIN"]
    assert all(r["decision_scope"] == "SOURCE_STATISTICAL_STATEMENT" for r in rows)
    assert "drug-treated versus vehicle" in rows[0]["source_statement"]
    assert "causes" not in rows[0]["source_statement"]


def test_changed_text_and_source_are_retained_with_specific_findings(files, tmp_path):
    path = tmp_path / "claims.tsv"
    text = (
        "Response A has NES=−2.134 and q=1.153e-8.\n"
        "Loss of function leads to checkpoint suppression."
    )
    drafts(
        path,
        [
            {"term_id": "SET_A", "text": text},
            {"term_id": "SET_B", "text": "Response B is statistically significant."},
        ],
    )
    result = review_enrichment(cfg(files, tmp_path / "out", path))
    records = [
        json.loads(line) for line in Path(result.artifacts["report_jsonl"]).read_text().splitlines()
    ]
    first = records[0]["submitted_reviews"][0]
    assert first["text"] == text
    assert first["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert first["numeric_coverage"] == "COMPLETE_EXPLICIT_STAT_Q"
    assert {f["code"] for f in first["findings"]} == {"INFERENTIAL_LANGUAGE_REQUIRES_REVIEW"}
    flag = first["findings"][0]
    assert text[flag["start"] : flag["end"]] == flag["quote"] == "leads to"
    assert first["prose_disposition"] == "ABSTAIN"
    second = records[1]["submitted_reviews"][0]
    assert second["prose_disposition"] == "FAIL"
    assert second["findings"][0]["code"] == "SIGNIFICANCE_MISMATCH"
    assert len(records) == 3


@pytest.mark.parametrize(
    "text",
    [
        "NES=-2.134 and q=1.153e-8.",
        "A normalized enrichment score (NES) of -2.134 and q-value of 1.153e-8.",
        "NES=-2.134; adjusted value=1.153e-8.",
        "IL-2/STAT5 and IL6/JAK/STAT3 refer to pathway names, not source statistics.",
        "NES is -2.13; FDR=1.2×10^-8.",
        "NES<0 and q≤0.05.",
        "No evidence that the treatment causes this response.",
        "This does not prove a mechanism.",
        "We cannot conclude that this leads to a clinical benefit.",
    ],
)
def test_rounding_bounds_and_negative_inferential_wording(text):
    source = {
        "stat": -2.1340243231909,
        "stat_kind": "NES",
        "qval": 1.15266177813956e-8,
        "direction": "down",
    }
    result = inspect_text(text, source, 0.05)
    assert result["findings"] == []
    assert result["automatic_prose_acceptance"] is False


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("NES=+2.13", "NUMERIC_MISMATCH"),
        ("q>0.05", "NUMERIC_MISMATCH"),
        ("The result is not statistically significant.", "SIGNIFICANCE_MISMATCH"),
        ("The pathway is positively enriched.", "DIRECTION_MISMATCH"),
        ("It drives checkpoint loss.", "INFERENTIAL_LANGUAGE_REQUIRES_REVIEW"),
        ("The intervention is clinically effective.", "INFERENTIAL_LANGUAGE_REQUIRES_REVIEW"),
        (
            "It is not significant and drives checkpoint loss.",
            "INFERENTIAL_LANGUAGE_REQUIRES_REVIEW",
        ),
    ],
)
def test_explicit_mismatches_and_positive_inferential_wording(text, code):
    source = {
        "stat": -2.1340243231909,
        "stat_kind": "NES",
        "qval": 1.15266177813956e-8,
        "direction": "down",
    }
    assert code in {f["code"] for f in inspect_text(text, source, 0.05)["findings"]}


def test_missing_numbers_do_not_become_error_labels(files, tmp_path):
    path = tmp_path / "claims.tsv"
    drafts(path, [{"term_id": "SET_A", "text": "This pathway may merit follow-up."}])
    result = review_enrichment(cfg(files, tmp_path / "out", path))
    record = json.loads(Path(result.artifacts["report_jsonl"]).read_text().splitlines()[0])
    review = record["submitted_reviews"][0]
    assert review["limited_checks_status"] == "NO_LIMITED_FLAG"
    assert review["numeric_coverage"] == "PARTIAL_OR_ABSENT"
    assert review["prose_disposition"] == "ABSTAIN"
    assert review["semantic_correctness"] == "NOT_ESTABLISHED"


def test_metadata_and_stale_evidence_binding_fail(files, tmp_path):
    path = tmp_path / "claims.tsv"
    drafts(
        path,
        [
            {
                "term_id": "SET_A",
                "text": "NES=-2.134",
                "tissue": "brain",
                "evidence_sha256": "a" * 64,
            }
        ],
    )
    result = review_enrichment(cfg(files, tmp_path / "out", path))
    r = json.loads(Path(result.artifacts["report_jsonl"]).read_text().splitlines()[0])[
        "submitted_reviews"
    ][0]
    assert r["prose_disposition"] == "FAIL"
    assert {x["code"] for x in r["findings"]} == {
        "CONTEXT_ATTRIBUTE_MISMATCH",
        "EVIDENCE_IDENTITY_MISMATCH",
    }


def test_ambiguous_ids_and_unknown_links_stop_before_output(files, tmp_path):
    path = tmp_path / "claims.tsv"
    drafts(path, [{"term_id": "UNRESOLVED", "text": "Draft"}])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Ambiguous or absent"):
        review_enrichment(cfg(files, out, path))
    assert not out.exists()


def test_source_validation_and_prior_output_are_preserved(files, tmp_path):
    out = tmp_path / "out"
    review_enrichment(cfg(files, out))
    marker = out / "coauthor_feedback.txt"
    marker.write_text("retain")
    with pytest.raises(ValueError, match="never overwrite"):
        review_enrichment(cfg(files, out))
    assert marker.read_text() == "retain"
    bad = tmp_path / "bad.tsv"
    bad.write_text(files[0].read_text().replace("0.6644", "-0.5"))
    with pytest.raises(ValueError, match="qval"):
        review_enrichment(ReviewConfig(str(bad), str(files[1]), str(tmp_path / "invalid")))
    assert not (tmp_path / "invalid").exists()


def test_html_escapes_original_prose_and_output_hashes_match(files, tmp_path):
    path = tmp_path / "claims.tsv"
    drafts(path, [{"term_id": "SET_A", "text": "<script>alert('draft')</script>"}])
    result = review_enrichment(cfg(files, tmp_path / "out", path))
    rendered = Path(result.artifacts["report_html"]).read_text()
    assert "&lt;script&gt;" in rendered
    assert "<script>alert('draft')</script>" not in rendered
    meta = json.loads(Path(result.meta_path).read_text())
    for item in meta["artifacts"].values():
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_public_cli_and_explicit_legacy_requirement(files, tmp_path):
    command = [
        sys.executable,
        "-m",
        "llm_pathway_curator.cli",
        "run",
        "--evidence-table",
        str(files[0]),
        "--sample-card",
        str(files[1]),
        "--outdir",
        str(tmp_path / "cli"),
    ]
    subprocess.run(command, check=True, capture_output=True)
    assert (tmp_path / "cli/report.html").is_file()
    with pytest.raises(ValueError, match="workflow='legacy'"):
        run_pipeline(RunConfig(str(files[0]), str(files[1]), str(tmp_path / "bad"), tau=0.8))
    assert not (tmp_path / "bad").exists()
