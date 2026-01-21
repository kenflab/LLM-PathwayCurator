# LLM-PathwayCurator/tests/test_demo_cli.py
from __future__ import annotations

import csv
import json
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


def _write_sample_card(path):
    obj = {
        "disease": "TP53-mut disease",
        "tissue": "tumor",
        "perturbation": "NA",
        "comparison": "mut vs wt",
        "notes": "demo",
        # optional: keep schema contract explicit for tests
        "extra": {"schema_version": "v1"},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _run_cli(tmp_path, *, tau: float | None = None):
    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    ev = tmp_path / "de_table.tsv"
    _write_demo_evidence(ev)

    sc = tmp_path / "sample_card.json"
    _write_sample_card(sc)

    cmd = [
        sys.executable,
        "-m",
        "llm_pathway_curator.cli",
        "run",
        "--evidence-table",
        str(ev),
        "--sample-card",
        str(sc),
        "--outdir",
        str(outdir),
    ]
    if tau is not None:
        cmd += ["--tau", str(tau)]

    subprocess.check_call(cmd)
    return outdir


def test_demo_cli_runs_and_writes_jsonl(tmp_path):
    outdir = _run_cli(tmp_path)

    # core artifacts
    audit = pd.read_csv(outdir / "audit_log.tsv", sep="\t")
    assert "status" in audit.columns
    assert (outdir / "report.md").exists()
    assert (outdir / "distilled.tsv").exists()

    # paper artifact contract: report.jsonl
    jsonl_path = outdir / "report.jsonl"
    assert jsonl_path.exists()

    first = jsonl_path.read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(first)

    # minimal v1 contract keys (add-only, but must exist)
    assert "schema_version" in rec
    assert "claim" in rec and isinstance(rec["claim"], dict)
    assert "metrics" in rec and isinstance(rec["metrics"], dict)

    # decision aliases must be consistent
    assert "decision_status" in rec
    assert "decision" in rec
    assert rec["decision"] == rec["decision_status"]

    # survival metric must exist (may be null per-row, but demo should not be all-null)
    assert "term_survival_agg" in rec["metrics"]


def test_cli_tau_override_propagates_to_audit_and_jsonl(tmp_path):
    tau = 0.99
    outdir = _run_cli(tmp_path, tau=tau)

    audit = pd.read_csv(outdir / "audit_log.tsv", sep="\t")
    assert "tau_used" in audit.columns
    # At least one row should carry the override value
    assert (pd.to_numeric(audit["tau_used"], errors="coerce") == tau).any()

    first = (outdir / "report.jsonl").read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(first)
    assert float(rec["tau"]) == tau
