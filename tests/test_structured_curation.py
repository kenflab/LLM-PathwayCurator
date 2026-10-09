"""Software contracts, not tests of interpretive utility or biological truth."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from llm_pathway_curator import ReviewConfig, review_enrichment
from llm_pathway_curator.curation import source_anchor, support_modules


@pytest.fixture
def inputs(tmp_path):
    # A--B--C is transitive; A and C have no shared gene. D has no support.
    rows = [
        ["A", "Term A", "fgsea", 1.5, "NES", 0.01, "up", "G1;G2"],
        ["B", "Term B", "fgsea", -1.8, "NES", 0.02, "down", "G1;G2;G3;G4"],
        ["C", "Term <C>", "fgsea", 1.7, "NES", 0.20, "up", "G3;G4"],
        ["D", "Term D", "fgsea", "", "NES", "", "na", ""],
    ]
    path = tmp_path / "evidence.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "term_id",
                "term_name",
                "source",
                "stat",
                "stat_kind",
                "qval",
                "direction",
                "evidence_genes",
            ]
        )
        writer.writerows(rows)
    card = tmp_path / "card.json"
    card.write_text('{"comparison":"treated versus control"}')
    return path, card


def run(inputs, tmp_path, name="out", **kwargs):
    return review_enrichment(
        ReviewConfig(str(inputs[0]), str(inputs[1]), str(tmp_path / name), **kwargs)
    )


def read_json(result, key):
    return json.loads(Path(result.artifacts[key]).read_text())


def read_jsonl(result, key):
    return [json.loads(line) for line in Path(result.artifacts[key]).read_text().splitlines()]


def proposal_for(record, **kwargs):
    e = record["evidence"]
    return {
        "claim_id": "P1",
        "claim_type": "statistical_observation",
        "text": "Positive enrichment was observed (NES=1.5; q=0.01).",
        "comparison": "treated versus control",
        "evidence_refs": [{"term_uid": e["term_uid"], "evidence_sha256": e["evidence_sha256"]}],
        "supporting_genes": ["G1"],
        "generator": {"name": "synthetic test fixture"},
        **kwargs,
    }


def test_transitive_modules_keep_all_candidates_and_do_not_assert_common_genes(inputs, tmp_path):
    result = run(inputs, tmp_path, modules=True, module_min_shared_genes=2, k_claims=1)
    grouping = read_json(result, "modules")
    connected, singleton = grouping["modules"]
    assert connected["member_term_uids"] == ["fgsea:A", "fgsea:B", "fgsea:C"]
    assert connected["common_to_all_genes"] == []
    assert connected["shared_at_least_two_genes"] == ["G1", "G2", "G3", "G4"]
    assert connected["direction_counts"] == {"up": 2, "down": 1}
    assert "MIXED_ENRICHMENT_DIRECTIONS" in connected["review_flags"]
    assert "NO_GENE_SHARED_BY_ALL_MEMBERS" in connected["review_flags"]
    assert singleton["member_term_uids"] == ["fgsea:D"]
    assert singleton["review_flags"] == ["SUPPORT_GENES_UNAVAILABLE"]
    assert len(grouping["edges"]) == 2
    assert len(read_jsonl(result, "report_jsonl")) == 4
    plain = run(inputs, tmp_path, "plain", k_claims=1)
    assert read_jsonl(plain, "report_jsonl") == read_jsonl(result, "report_jsonl")
    assert (
        sum(
            c["selected"]
            for c in read_jsonl(result, "structured_claims")
            if c["claim_type"] == "statistical_observation"
        )
        == 1
    )


def test_module_identity_ignores_row_order_but_references_track_exact_input(inputs, tmp_path):
    result = run(inputs, tmp_path)
    evidence = [r["evidence"] for r in read_jsonl(result, "report_jsonl")]
    a = support_modules(evidence, min_shared=2)
    b = support_modules(list(reversed(evidence)), min_shared=2)
    assert a == b
    lines = inputs[0].read_text().splitlines()
    inputs[0].write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    second = run(inputs, tmp_path, "reordered", modules=True, module_min_shared_genes=2)
    modules = read_json(second, "modules")["modules"]
    assert [m["module_id"] for m in modules] == [m["module_id"] for m in a["modules"]]
    assert modules[0]["evidence_refs"] != a["modules"][0]["evidence_refs"]


@pytest.mark.parametrize(
    "change,code",
    [
        ({"comparison": "control versus treated"}, "COMPARISON_MISMATCH"),
        ({"supporting_genes": ["NOT_PRESENT"]}, "GENE_NOT_IN_LINKED_SUPPORT"),
        ({"text": "NES=-1.5; q=0.2"}, "NUMERIC_MISMATCH"),
        (
            {"evidence_refs": [{"term_uid": "fgsea:A", "evidence_sha256": "0" * 64}]},
            "EVIDENCE_IDENTITY_MISMATCH",
        ),
        (
            {"evidence_refs": [{"term_uid": "fgsea:UNKNOWN", "evidence_sha256": "0" * 64}]},
            "UNKNOWN_EVIDENCE_LINK",
        ),
    ],
)
def test_explicit_proposal_violations_are_preserved(inputs, tmp_path, change, code):
    first = run(inputs, tmp_path, "first")
    proposal = proposal_for(read_jsonl(first, "report_jsonl")[0], **change)
    raw = (json.dumps(proposal, indent=None) + "\n").encode()
    path = tmp_path / "proposals.jsonl"
    path.write_bytes(raw)
    result = run(inputs, tmp_path, proposals_file=str(path))
    checked = read_jsonl(result, "proposals_checked")[0]
    assert checked["proposal"] == proposal
    assert checked["prose_disposition"] == "FAIL"
    assert code in [f["code"] for f in checked["findings"]]
    assert Path(result.artifacts["proposals_submitted"]).read_bytes() == raw
    assert checked["automatic_prose_acceptance"] is False


def test_valid_hypothesis_and_multi_source_text_are_never_automatically_accepted(inputs, tmp_path):
    first = run(inputs, tmp_path, "first")
    records = read_jsonl(first, "report_jsonl")
    correct = proposal_for(records[0])
    hypothesis = proposal_for(
        records[0],
        claim_id="P2",
        claim_type="hypothesis",
        text="<script>alert('x')</script> If a future study found NES=9, review this hypothesis.",
    )
    multi = proposal_for(
        records[0],
        claim_id="P3",
        claim_type="support_summary",
        text="A has NES=1.5 and B has NES=-1.8.",
    )
    multi["evidence_refs"].extend(proposal_for(records[1])["evidence_refs"])
    path = tmp_path / "proposals.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in [correct, hypothesis, multi]))
    result = run(inputs, tmp_path, proposals_file=str(path), modules=True)
    checked = read_jsonl(result, "proposals_checked")
    assert all(p["prose_disposition"] == "ABSTAIN" for p in checked)
    assert checked[2]["numeric_coverage"] == "NOT_CHECKED"
    assert checked[1]["numeric_coverage"] == "NOT_CHECKED"
    assert "MULTI_SOURCE_TEXT_REQUIRES_REVIEW" in [f["code"] for f in checked[2]["findings"]]
    html = Path(result.artifacts["report_html"]).read_text()
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "Term &lt;C&gt;" in html
    assert f"href='#{source_anchor('fgsea:A')}'" in html
    assert f"id='{source_anchor('fgsea:A')}'" in html
    meta = json.loads(Path(result.meta_path).read_text())
    assert meta["model_calls"] == 0
    for artifact in meta["artifacts"].values():
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


@pytest.mark.parametrize("invalid", ["{", "{}", "[]", "null"])
def test_invalid_proposal_schema_fails_before_creating_outputs(inputs, tmp_path, invalid):
    path = tmp_path / "proposals.jsonl"
    path.write_text(invalid + "\n")
    with pytest.raises(ValueError, match="proposal"):
        run(inputs, tmp_path, proposals_file=str(path))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"modules": True, "module_min_shared_genes": 0},
        {"modules": True, "module_jaccard_min": float("nan")},
        {"module_jaccard_min": 0.2},
    ],
)
def test_invalid_module_options_fail_before_creating_outputs(inputs, tmp_path, options):
    with pytest.raises(ValueError):
        run(inputs, tmp_path, **options)
    assert not (tmp_path / "out").exists()


def test_cli_routes_curation_and_rejects_legacy_mix(inputs, tmp_path):
    command = [
        sys.executable,
        "-m",
        "llm_pathway_curator.cli",
        "run",
        "--evidence-table",
        str(inputs[0]),
        "--sample-card",
        str(inputs[1]),
        "--outdir",
        str(tmp_path / "cli"),
        "--modules",
        "--module-min-shared-genes",
        "2",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli" / "modules.json").is_file()
    rejected = subprocess.run(command + ["--workflow", "legacy"], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "source workflow" in rejected.stderr
