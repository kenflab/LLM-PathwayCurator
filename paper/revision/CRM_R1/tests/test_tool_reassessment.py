"""Integrity boundaries for reassessment; synthetic fixtures are not validation data."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[4] / "paper/revision/CRM_R1/experiments/reassessment/run.py"
)
spec = importlib.util.spec_from_file_location("tool_reassessment", SCRIPT)
reassessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reassessment)


def source_archive(tmp_path, *, corrupt=False):
    evidence, texts = [], []
    for i in range(50):
        evidence.append(
            {
                "evidence_id": f"fixture_{i}",
                "term_id": f"SET_{i}",
                "term_name": f"Set {i}",
                "cohort_name": "NON_SCIENTIFIC_FIXTURE",
                "nes": 1.25,
                "q_value": 0.2,
                "direction": "UP",
                "leading_edge_genes": ["FAKEGENE"],
                "contrast": {
                    "positive_group": "treated",
                    "reference_group": "control",
                    "study_design": "experimental",
                },
                "metadata": {"tissue": {"value": "synthetic"}},
                "source": {"artifact": "fixture.tsv", "sha256": "a" * 64, "locator": f"row={i}"},
            }
        )
        texts.append(
            {
                "evidence_id": f"fixture_{i}",
                "term_id": f"SET_{i}",
                "canonical_q_value": 0.2,
                "canonical_nes": 1.25,
                "unaudited_text": "NES=1.25; q=0.2.\nIt is statistically significant.",
                "numeric_gate_status": "SYNTHETIC_OLD_STATUS",
            }
        )
    payload = {
        "results/summary.json": json.dumps({"schema": "CRM_R1_REVISION_v17_R07"}).encode(),
        "results/canonical_evidence.private.json": json.dumps(evidence).encode(),
        "results/comparisons.private.json": json.dumps(texts).encode(),
    }
    manifest = {p: hashlib.sha256(raw).hexdigest() for p, raw in payload.items()}
    if corrupt:
        payload["results/comparisons.private.json"] += b" "
    payload["EXPORT_SHA256.json"] = json.dumps(manifest).encode()
    source = tmp_path / "fixture.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for path, raw in payload.items():
            archive.writestr(path, raw)
    return source


def test_source_hash_mismatch_stops_before_preparation(tmp_path):
    source = source_archive(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="Export hash mismatch"):
        reassessment.prepare(tmp_path, source)
    assert not (tmp_path / "output").exists()


def test_reassessment_keeps_all_original_texts_and_no_accuracy_labels(tmp_path):
    source = source_archive(tmp_path)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    design = reassessment.prepare(tmp_path, source)
    summary = reassessment.run(tmp_path, design)
    assert summary["candidate_retained_count"] == summary["candidate_count"] == 50
    assert summary["prose_disposition_counts"] == {"FAIL": 50}
    assert summary["model_calls"] == summary["new_expert_ratings"] == 0
    assert summary["independent_validation"] is False
    assert summary["semantic_accuracy_estimated"] is False
    assert summary["biological_accuracy_estimated"] is False
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    records = [
        json.loads(s)
        for s in Path(summary["outdir"])
        .joinpath("source_report/report.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(
        r["submitted_reviews"][0]["text"] == "NES=1.25; q=0.2.\nIt is statistically significant."
        for r in records
    )
    second = reassessment.run(tmp_path, design)
    assert second["outdir"] != summary["outdir"]


def test_prepared_text_change_stops_before_run_output(tmp_path):
    design = reassessment.prepare(tmp_path, source_archive(tmp_path))
    (design / "drafts.private.tsv").write_text("changed")
    with pytest.raises(ValueError, match="Prepared input changed"):
        reassessment.run(tmp_path, design)
    assert list((tmp_path / "output/revision_v17").iterdir()) == [design]


def test_implementation_change_requires_new_development_design(tmp_path):
    design = reassessment.prepare(tmp_path, source_archive(tmp_path))
    manifest = json.loads((design / "DESIGN.json").read_text())
    manifest["implementation_sha256"]["src/llm_pathway_curator/review.py"] = "0" * 64
    (design / "DESIGN.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Implementation changed"):
        reassessment.run(tmp_path, design)
