from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tcga_groups", ROOT / "paper/scripts/fig2_make_groups.py"
)
GROUPS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GROUPS)


def sample(n):
    return f"TCGA-AA-{n:04d}-01"


def test_absence_from_mutation_rows_never_establishes_wild_type():
    cohort = pd.DataFrame({"sample": [sample(1), sample(2), sample(3)]})
    without_manifest = GROUPS.assign_tp53_groups(cohort, {sample(1)}, None)
    assert without_manifest.group.tolist() == ["TP53_mut", "TP53_unknown", "TP53_unknown"]
    assessment = pd.DataFrame({"sample": [sample(2)], "tp53_assessed": ["true"]})
    with_manifest = GROUPS.assign_tp53_groups(cohort, {sample(1)}, assessment)
    assert with_manifest.group.tolist() == ["TP53_mut", "TP53_wt", "TP53_unknown"]
    assert with_manifest.group_basis.tolist() == [
        "protein_altering_call",
        "assessed_no_protein_altering_call",
        "no_TP53_assessment",
    ]


def test_assay_manifest_can_establish_zero_variant_sample_as_comparator():
    cohort = pd.DataFrame({"sample": [sample(1)]})
    assessment = pd.DataFrame({"sample": [sample(1)], "tp53_assessed": [1]})
    assert GROUPS.assign_tp53_groups(cohort, set(), assessment).group.iloc[0] == "TP53_wt"


def test_aliquots_match_the_sample_without_collapsing_sample_types():
    assert GROUPS._normalize_barcode(sample(1) + "A-01D-1234-01") == sample(1)
    assert GROUPS._normalize_barcode(sample(1)[:-2] + "11A") != sample(1)
    cohort = pd.DataFrame({"sample": [sample(1)]})
    assert GROUPS.assign_tp53_groups(cohort, {sample(1) + "A"}, None).group.iloc[0] == "TP53_mut"


@pytest.mark.parametrize("value", ["", "unknown", "yes", None])
def test_unclear_assessment_is_rejected(value):
    assessment = pd.DataFrame({"sample": [sample(1)], "tp53_assessed": [value]})
    with pytest.raises(ValueError, match="tp53_assessed"):
        GROUPS.assign_tp53_groups(pd.DataFrame({"sample": [sample(1)]}), set(), assessment)


def test_conflicting_aliquot_assessments_and_positive_calls_are_rejected():
    cohort = pd.DataFrame({"sample": [sample(1)]})
    manifest = pd.DataFrame(
        {"sample": [sample(1), sample(1) + "A"], "tp53_assessed": [True, False]}
    )
    with pytest.raises(ValueError, match="Conflicting"):
        GROUPS.assign_tp53_groups(cohort, set(), manifest)
    with pytest.raises(ValueError, match="conflicts"):
        GROUPS.assign_tp53_groups(cohort, {sample(1)}, manifest.iloc[1:])


def test_duplicate_cohort_samples_and_invalid_barcodes_are_rejected():
    with pytest.raises(ValueError, match="Duplicate sample"):
        GROUPS.assign_tp53_groups(
            pd.DataFrame({"sample": [sample(1), sample(1) + "A"]}), set(), None
        )
    with pytest.raises(ValueError, match="Invalid TCGA"):
        GROUPS._normalize_barcode("TCGA-not-a-sample")


def test_group_cli_preserves_unknowns_records_provenance_and_refuses_overwrite(tmp_path):
    mc3 = tmp_path / "mc3.tsv.gz"
    phenotype = tmp_path / "phenotype.tsv.gz"
    assessed = tmp_path / "assessed.tsv"
    pd.DataFrame(
        {
            "sample": [sample(1) + "A-01D-1234-01", sample(2)],
            "gene": ["TP53", "TTN"],
            "effect": ["Missense_Mutation", "Missense_Mutation"],
        }
    ).to_csv(mc3, sep="\t", index=False)
    pd.DataFrame(
        {
            "sampleID": [sample(1), sample(2), sample(3)],
            "sample_type_id": ["01"] * 3,
            "_primary_disease": ["head & neck squamous cell carcinoma"] * 3,
        }
    ).to_csv(phenotype, sep="\t", index=False)
    pd.DataFrame({"sample": [sample(3)], "tp53_assessed": [True]}).to_csv(
        assessed, sep="\t", index=False
    )
    out = tmp_path / "groups"
    command = [
        sys.executable,
        str(ROOT / "paper/scripts/fig2_make_groups.py"),
        "--mc3",
        str(mc3),
        "--phenotype",
        str(phenotype),
        "--assessed-samples",
        str(assessed),
        "--outdir",
        str(out),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    table = pd.read_csv(out / "HNSC.groups.tsv", sep="\t").set_index("sample")
    assert table.loc[sample(1), "group"] == "TP53_mut"
    assert table.loc[sample(2), "group"] == "TP53_unknown"  # TTN alone is not TP53 assessment.
    assert (
        table.loc[sample(3), "group"] == "TP53_wt"
    )  # Explicitly assessed, even with zero variants.
    provenance = json.loads((out / "groups.provenance.json").read_text())
    assert len(provenance["assessment_sha256"]) == 64
    assert provenance["mutation_filter"] == "no_FILTER_column"
    before = (out / "HNSC.groups.tsv").read_bytes()
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode != 0 and "already exist" in rerun.stderr
    assert (out / "HNSC.groups.tsv").read_bytes() == before
