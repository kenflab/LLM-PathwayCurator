#!/usr/bin/env python3
"""Test exclusion rules with independent synthetic assay metadata (no network)."""

import gzip
import importlib.util
import json
import tempfile
from pathlib import Path

import pandas as pd


def test_native_assessment():
    code_dir = Path(__file__).resolve().parents[1]
    path = code_dir / "paper/scripts/tcga_rebuild/RUN.py"
    spec = importlib.util.spec_from_file_location("runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        src = root / "public_inputs"
        src.mkdir()
        out = root / "out"
        out.mkdir()
        runner.BUNDLE = root
        runner.INPUTS = root
        samples = [f"TCGA-AA-{i:04d}-01" for i in range(1, 10)]

        def write(name, records):
            pd.DataFrame(records).to_csv(src / name, sep="\t", index=False)

        write(
            "TCGA_phenotype_dense.tsv.gz",
            [
                {
                    "sample": s,
                    "sample_type_id": "01",
                    "_primary_disease": "breast invasive carcinoma",
                }
                for s in samples
            ],
        )
        write(
            "pairing_of_best_bams_2016-04-05.txt",
            [
                {
                    "broad_file_pair_id": s[:12] + "_tumor_normal",
                    "dnanexus_pair_id": "BRCA/pair_01_10",
                    "cohort": "BRCA",
                }
                for s in samples
                if s != samples[2]
            ],
        )
        write(
            "merged_sample_quality_annotations.tsv",
            [
                {
                    "aliquot_barcode": s + "A-01R-0000-01",
                    "platform": "IlluminaHiSeq_RNASeqV2",
                    "Do_not_use": "True" if s == samples[6] else "False",
                    "AWG_excluded_because_of_pathology": "0.0",
                }
                for s in samples
            ],
        )
        inventory = {}
        for s in samples[:-1]:
            inventory[s] = {
                "filters": {"PASS": 10},
                "tumor_aliquots": [s + "A-01D-0000-01"],
                "normal_aliquots": [s[:12] + "-10A-01D-0000-01"],
            }
        inventory[samples[3]]["filters"]["wga"] = 10
        inventory[samples[3]]["tumor_aliquots"] = [samples[3] + "A-01W-0000-01"]
        (src / "mc3_sample_filter_inventory.json").write_text(json.dumps({"samples": inventory}))
        write(
            "mc3.TP53.all_filters.tsv.gz",
            [
                {
                    "Hugo_Symbol": "TP53",
                    "Gene": "ENSG00000141510",
                    "Tumor_Sample_Barcode": samples[i] + "A-01D-0000-01",
                    "Variant_Classification": "Missense_Mutation",
                    "FILTER": "oxog" if i == 4 else "PASS",
                }
                for i in [0, 3, 4, 6, 7]
            ],
        )
        (root / "INPUT_MANIFEST.json").write_text(
            json.dumps({"full_mc3": {"sha256": "synthetic_fixture"}})
        )
        expression = root / "expression.tsv.gz"
        with gzip.open(expression, "wt") as f:
            f.write("gene\t" + "\t".join(samples + [samples[5]]) + "\n")
            f.write("TP53\t" + "\t".join(["1"] * 9 + ["2"]) + "\n")
        ledger, summary = runner.build_assessment(code_dir, expression, out, {"cohorts": ["BRCA"]})
        expected = [
            "TP53_mut",
            "TP53_wt",
            "TP53_unknown",
            "TP53_unknown",
            "TP53_unknown",
            "TP53_unknown",
            "TP53_unknown",
            "TP53_mut",
            "TP53_unknown",
        ]
        assert ledger.set_index("sample").loc[samples, "group"].tolist() == expected
        reasons = ledger.set_index("sample").exclusion_reasons
        assert "no_official_MC3_pair" in reasons[samples[2]]
        assert "WGA_flag" in reasons[samples[3]]
        assert "ambiguous_TP53_call" in reasons[samples[4]]
        assert "duplicate_expression_columns" in reasons[samples[5]]
        assert "RNA_QC_missing_or_excluded" in reasons[samples[6]]
        assert "native_tumor_normal_DNA_unconfirmed" in reasons[samples[8]]
        canonical = pd.read_csv(out / "TP53.eligible.tsv.gz", sep="\t")
        assert canonical.gene.eq("TP53").all() and "Gene" not in canonical
        assert not bool(summary.iloc[0].fit_eligible)
    print(
        "PASS: assay membership, WGA, filtered TP53, duplicate RNA, RNA "
        "QC, missing native evidence, and MAF symbol normalization"
    )
