"""Regression checks against the supplied manuscript records and archived inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "paper/revision/CRM_R1"


def load(name):
    spec = importlib.util.spec_from_file_location(name, RESEARCH / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manuscript_records_reproduce(tmp_path):
    out = tmp_path / "reproduction"
    subprocess.run(
        [sys.executable, str(RESEARCH / "reproduce.py"), "--outdir", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads((out / "reproduction_manifest.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["verified_inputs"] == 64
    assert receipt["model_calls"] == 0
    assert not receipt["expression_models_refitted"]
    assert not receipt["human_labels_changed"]
    assert receipt["dex"]["counts_50"] == {
        "VIOLATION_DETECTED": 12,
        "REVIEW_FLAGGED": 30,
        "NO_LIMITED_FLAG": 8,
    }
    assert [e["agree"] for e in receipt["dex"]["endpoints"]] == [9, 13]
    assert len(receipt["source_reports"]) == 7
    assert sum(r["candidates"] for r in receipt["source_reports"]) == 350
    assert sum(r["selected"] for r in receipt["source_reports"]) == 159
    assert receipt["split_replication"]["intervals_reproduced"] == 6
    ratings = pd.read_csv(out / "recomputed_tables/HNSC_interrater_agreement.tsv", sep="\t")
    assert ratings.fleiss_kappa.tolist() == pytest.approx(
        [-0.31399897941826843, -0.04487732708667242, -0.3290592133698314]
    )
    assert ratings.kappa_ci_low.tolist() == pytest.approx(
        [-0.3735775248933144, -0.1574074074074074, -0.3539577225736769]
    )
    assert len(list(out.rglob("*.png"))) == 6
    examples = pd.read_csv(out / "supplement/tables/Table_S6_worked_examples.tsv", sep="\t")
    assert examples.reference_id.tolist() == ["SRC003", "SRC019"]
    assert examples.limited_checks_status.tolist() == ["VIOLATION_DETECTED", "NO_LIMITED_FLAG"]
    figures = json.loads((out / "supplement/render_manifest.json").read_text())
    assert "tables/Table_S6_worked_examples.tsv" in figures["outputs_sha256"]
    for name, expected in receipt["outputs_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected


def test_modified_source_text_stops_before_outputs(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(RESEARCH / "data", data)
    text = data / "Supplementary_Data_2_DEX/DEX_text_checks_50.tsv"
    before = text.read_text()
    after = before.replace("significantly downregulated", "downregulated", 1)
    assert before != after
    text.write_text(after)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Source checksum differs"):
        load("reproduce").main(["--data-dir", str(data), "--outdir", str(out)])
    assert not out.exists()


def test_reproduction_preserves_existing_output(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "report.txt"
    sentinel.write_text("previous result")
    with pytest.raises(ValueError, match="new output directory"):
        load("reproduce").main(["--outdir", str(out)])
    assert sentinel.read_text() == "previous result"


def test_tcga_snapshot_matches_original_manifest():
    archive = ROOT / "paper/source_data/CRM_R1_TCGA_inputs.zip"
    expected = archive.with_suffix(".sha256").read_text().split()[0]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
    with zipfile.ZipFile(archive) as z:
        manifest_bytes = z.read("INPUT_MANIFEST.json")
        assert (
            manifest_bytes == (ROOT / "paper/scripts/tcga_rebuild/INPUT_MANIFEST.json").read_bytes()
        )
        manifest = json.loads(manifest_bytes)
        assert len(manifest["public_files"]) == 11
        assert set(z.namelist()) == {"INPUT_MANIFEST.json"} | {
            record["path"] for record in manifest["public_files"]
        }
        for record in manifest["public_files"]:
            assert hashlib.sha256(z.read(record["path"])).hexdigest() == record["sha256"]


def test_provenance_collection_is_bounded_and_read_only(tmp_path):
    data = tmp_path / "data"
    folder = data / "output/priority2b/design_lock_v14_1"
    folder.mkdir(parents=True)
    (folder / "design_manifest.json").write_text('{"fixture": true}\n')
    (folder / "discovery.R").write_text("# original code\n")
    (folder / "samples.tsv").write_text("sample\tvalue\nprivate\t1\n")
    (folder / "labels.private.json").write_text('{"private": true}\n')
    projects = tmp_path / "projects"
    old = projects / "LLM-PathwayCurator-v16-backup"
    old.mkdir(parents=True)
    (old / "28_run_priority2b.R").write_text("# archived fit code\n")
    (old / "ratings.tsv").write_text("do not collect\n")
    (projects / "LLM-scCurator").mkdir()
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    out = tmp_path / "packet"
    load("collect_split_provenance").main(
        [
            "--data-root",
            str(data),
            "--repo",
            str(tmp_path / "empty_repo"),
            "--projects-root",
            str(projects),
            "--outdir",
            str(out),
        ]
    )
    result = json.loads((out / "COLLECTION.json").read_text())
    assert result["model_calls"] == 0
    assert not result["study_relocked"] and not result["inputs_changed"]
    assert len(result["files"]) == 3
    assert result["known_lock_checks"][0]["status"] == "DIFFERENT_BYTES"
    assert [r["name"] for r in result["project_inventory"]] == [old.name]
    assert result["project_inventory"][0]["code_files"] == 1
    assert all(p.endswith((".R", ".json")) for p in result["files"])
    assert not any(".private." in p or p.endswith(".tsv") for p in result["files"])
    assert out.with_suffix(".zip").is_file()
    for name, original in before.items():
        assert Path(name).read_bytes() == original
