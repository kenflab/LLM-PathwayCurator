"""Synthetic controls for source alignment, freeze integrity, and output safety."""

import importlib.util
import json
import math
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "paper/revision/CRM_R1/scripts/v17_revision.py"
SPEC = importlib.util.spec_from_file_location("crm_v17_preflight", SCRIPT)
revision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(revision)


@pytest.fixture
def census():
    rows = []
    for term, nes, q, gene in (("T1", "1.5", "0.05", "G1"), ("T2", "-2", "1e-20", "G2")):
        rows.append(
            {
                "cohort_id": "ACC",
                "split_id": "S001",
                "term_id": term,
                "term_name": term,
                "stat": nes,
                "qval": q,
                "direction": "up" if float(nes) > 0 else "down",
                "evidence_genes": gene,
                "source": "synthetic",
                "n_discovery_mutant": "9",
                "n_discovery_wild_type": "38",
            }
        )
    return pd.DataFrame(rows)


def checked(table):
    return revision.validate_stats(
        table, {"T1": {"G1"}, "T2": {"G2"}}, {"ACC": "test"}, expected_terms=2
    )


def test_census_checks_boundary_and_sign(census):
    result = checked(census)
    assert len(result) == 1
    assert result[0]["fdr_le_0_05"] == 2
    assert result[0]["up"] == result[0]["down"] == 1


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("qval", "NaN", "Nonfinite"),
        ("qval", "1.01", "outside"),
        ("direction", "down", "disagrees"),
        ("evidence_genes", "G2", "outside frozen"),
        ("evidence_genes", "G1,G1", "Duplicate leading"),
        ("n_discovery_mutant", "10", "Inconsistent group"),
        ("cohort_id", "UNKNOWN", "Unknown TCGA"),
    ],
)
def test_invalid_discovery_data_stop(census, column, value, message):
    if column == "cohort_id":
        census[column] = value
    else:
        census.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        checked(census)


def test_duplicate_and_incomplete_census_stop(census):
    with pytest.raises(ValueError, match="Duplicate term"):
        checked(pd.concat([census, census.iloc[[0]]]))
    with pytest.raises(ValueError, match="Incomplete terms"):
        checked(census.iloc[[0]])


def test_one_step_reserialization_is_explicit_and_not_statistical_tolerance():
    x = 1.2713188521139301e-09
    y = math.nextafter(x, 0)
    assert revision.saved_number_alignment(repr(x), repr(y), "qval") == "ONE_ULP_RESERIALIZATION"
    assert revision.saved_number_alignment(repr(x), repr(x), "qval") == "EXACT_FLOAT"
    with pytest.raises(ValueError, match="qval mismatch"):
        revision.saved_number_alignment(repr(x), repr(math.nextafter(y, 0)), "qval")
    with pytest.raises(ValueError, match="qval mismatch"):
        revision.saved_number_alignment("1e-20", "2e-20", "qval")
    with pytest.raises(ValueError, match="FDR decision"):
        revision.saved_number_alignment("0.05", repr(math.nextafter(0.05, 1)), "qval")
    with pytest.raises(ValueError, match="NES sign"):
        revision.saved_number_alignment("0.0", "5e-324", "stat")


def test_frozen_manifest_rebases_full_path_and_detects_tampering(tmp_path):
    root = tmp_path / "CRM_R1"
    source = root / "input/subdir/stats.tsv"
    source.parent.mkdir(parents=True)
    source.write_text("original\n")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "inputs": {},
                "outputs": {
                    "stats": {
                        "path": "/Users/example/CRM_R1/input/subdir/stats.tsv",
                        "sha256": revision.sha(source),
                    }
                },
            }
        )
    )
    manifest.with_suffix(".sha256").write_text(revision.sha(manifest) + "  manifest.json\n")
    inputs = revision.Inputs()
    _, checks = revision.verify_manifest(root, Path("manifest.json"), inputs)
    assert checks[0]["relative_path"] == "input/subdir/stats.tsv"
    assert checks[0]["status"] == "MATCH"
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="Input changed"):
        inputs.verify()
    with pytest.raises(ValueError, match="digest mismatch"):
        revision.verify_manifest(root, Path("manifest.json"), revision.Inputs())


def test_rebasing_never_falls_back_to_basename_or_escaping_symlink(tmp_path):
    root = tmp_path / "CRM_R1"
    root.mkdir()
    with pytest.raises(ValueError, match="Cannot rebase"):
        revision.declared_path(root, "/other/stats.tsv")
    with pytest.raises(ValueError, match="escapes"):
        revision.data_path(root, Path("../stats.tsv"))
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        revision.data_path(root, Path("outside/stats.tsv"))


def test_output_is_new_and_external_to_git(tmp_path):
    root = tmp_path / "CRM_R1"
    (root / "input").mkdir(parents=True)
    (root / "output").mkdir()
    _, out = revision.output_directory(root, repo=tmp_path / "repo")
    assert out.parent == root / "output/revision_v17"
    assert not out.exists()
    old = root / "output/priority2b/final_v14_1_3"
    with pytest.raises(ValueError, match="new child"):
        revision.output_directory(root, old, repo=tmp_path / "repo")
    with pytest.raises(ValueError, match="inside Git"):
        revision.output_directory(root, repo=tmp_path)
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        revision.output_directory(root, out, repo=tmp_path / "repo")


def test_duplicate_manifest_keys_are_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"path":"one","path":"two"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        revision.read_json(path)
