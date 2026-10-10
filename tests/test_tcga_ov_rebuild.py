#!/usr/bin/env python3
"""Regression: OV wga-only calls are retained, compound filters are not rescued."""

import gzip
import importlib.util
import json
import tempfile
from pathlib import Path

import pandas as pd


def test_ov_policy():
    code_dir = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "runner", code_dir / "paper/scripts/tcga_rebuild/RUN.py"
    )
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "public_inputs"
        src.mkdir()
        run.BUNDLE = root
        run.INPUTS = root
        samples = [f"TCGA-AA-{i:04d}-01" for i in range(1, 10)]

        def write(name, rows):
            pd.DataFrame(rows).to_csv(src / name, sep="\t", index=False)

        write(
            "TCGA_phenotype_dense.tsv.gz",
            [
                {
                    "sample": s,
                    "sample_type_id": "01",
                    "_primary_disease": "ovarian serous cystadenocarcinoma",
                }
                for s in samples
            ],
        )
        write(
            "pairing_of_best_bams_2016-04-05.txt",
            [
                {
                    "broad_file_pair_id": s[:12] + "_tumor_normal",
                    "dnanexus_pair_id": "OV/pair_01_10",
                    "cohort": "OV",
                }
                for s in samples
                if s != samples[5]
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
        inv = {
            s: {
                "filters": {"wga": 10},
                "tumor_aliquots": [s + "A-01W-0000-01"],
                "normal_aliquots": [s[:12] + "-10A-01W-0000-01"],
            }
            for s in samples
        }
        inv[samples[7]]["filters"]["native_wga_mix"] = 1
        (src / "mc3_sample_filter_inventory.json").write_text(json.dumps({"samples": inv}))
        write(
            "mc3.TP53.all_filters.tsv.gz",
            [
                {
                    "Hugo_Symbol": "TP53",
                    "Gene": "ENSG00000141510",
                    "Tumor_Sample_Barcode": samples[i] + "A-01W-0000-01",
                    "Variant_Classification": "Missense_Mutation",
                    "FILTER": "wga,oxog" if i == 1 else "wga",
                }
                for i in [0, 1, 5, 6, 7, 8]
            ],
        )
        (src / "gdc_OV_TP53_occurrences.json").write_text(
            json.dumps(
                {
                    "data": {
                        "pagination": {"total": 1},
                        "hits": [
                            {
                                "case": {"submitter_id": samples[3][:12]},
                                "ssm": {
                                    "consequence": [
                                        {
                                            "transcript": {
                                                "gene": {"gene_id": "ENSG00000141510"},
                                                "consequence_type": "frameshift_variant",
                                            }
                                        }
                                    ]
                                },
                            }
                        ],
                    }
                }
            )
        )
        (root / "INPUT_MANIFEST.json").write_text(json.dumps({"full_mc3": {"sha256": "fixture"}}))
        expression = root / "expression.tsv.gz"
        with gzip.open(expression, "wt") as f:
            f.write(
                "gene\t"
                + "\t".join(samples + [samples[8]])
                + "\nTP53\t"
                + "\t".join(["1"] * 10)
                + "\n"
            )
        policy = {"cohorts": ["OV"], "OV_wga_only": True, "OV_GDC_positive_guard": True}
        out = root / "out"
        out.mkdir()
        (out / "ANALYSIS_POLICY.json").write_text(json.dumps(policy))
        ledger, _ = run.build_assessment(code_dir, expression, out, policy)
        actual = ledger.set_index("sample")
        expected = [
            "TP53_mut",
            "TP53_unknown",
            "TP53_wt",
            "TP53_unknown",
            "TP53_wt",
            "TP53_unknown",
            "TP53_unknown",
            "TP53_unknown",
            "TP53_unknown",
        ]
        assert actual.loc[samples, "group"].tolist() == expected
        assert actual.loc[samples[6], "TP53_call_status_before_RNA_QC"] == "MUT"
        assert "MC3_negative_GDC_positive_case" in actual.loc[samples[3], "exclusion_reasons"]
        variants = pd.read_csv(out / "TP53.eligible.tsv.gz", sep="\t")
        assert set(variants.FILTER) == {"wga"}, "Do not rewrite source wga to PASS"
        run.write_groups(code_dir, out, policy)
        g = pd.read_csv(out / "groups/OV.groups.tsv", sep="\t").set_index("sample")
        assert g.loc[samples, "group"].tolist() == expected
        print(
            "PASS: OV wga-only; compound-filter rejection; GDC discordance; "
            "assay/QC/duplicate exclusions; original FILTER preservation; "
            "merged public assignment"
        )
