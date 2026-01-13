from __future__ import annotations

import csv
import subprocess
import sys

import pandas as pd


def _write_demo_evidence(path):
    rows = [
        ["HALLMARK_P53_PATHWAY", "p53 pathway", "fgsea", 2.1, 0.01, "up", "TP53,CDKN1A,MDM2,BAX"],
        [
            "HALLMARK_G2M_CHECKPOINT",
            "G2M checkpoint",
            "fgsea",
            1.8,
            0.03,
            "up",
            "MKI67,TOP2A,CCNB1,CDK1",
        ],
        ["GO:0006977", "DNA damage response", "metascape", 15.0, 1e-6, "na", "ATM,ATR,TP53,BRCA1"],
    ]
    header = ["term_id", "term_name", "source", "stat", "qval", "direction", "evidence_genes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def test_demo_cli_runs(tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    ev = tmp_path / "de_table.tsv"
    _write_demo_evidence(ev)

    cmd = [
        sys.executable,
        "-m",
        "llm_pathway_curator.cli",
        "run",
        "--evidence-table",
        str(ev),
        "--sample-card",
        "examples/demo/sample_card.json",
        "--outdir",
        str(outdir),
    ]
    subprocess.check_call(cmd)

    audit = pd.read_csv(outdir / "audit_log.tsv", sep="\t")
    assert "status" in audit.columns
    assert (outdir / "report.md").exists()
