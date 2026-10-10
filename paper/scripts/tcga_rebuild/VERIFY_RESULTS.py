#!/usr/bin/env python3
"""Check result identity, memberships and the current source report independently."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    args = ap.parse_args()
    root = args.run_dir
    reference = pd.read_csv(
        args.inputs / "public_inputs/hallmark.2026_1_Hs.memberships.tsv", sep="\t"
    )
    memberships = set(zip(reference.gs_name, reference.gene_symbol, strict=True))
    summary = pd.read_csv(root / "cohort_summary.tsv", sep="\t")
    ledger = pd.read_csv(root / "sample_ledger.tsv", sep="\t")
    require(not ledger["sample"].duplicated().any(), "Duplicate sample in ledger")
    checks = {}
    for row in summary.itertuples(index=False):
        c = row.cancer
        sample_counts = ledger.loc[ledger.cancer.eq(c), "group"].value_counts()
        require(int(sample_counts.get("TP53_mut", 0)) == row.n_mut, c + ": MUT count mismatch")
        require(int(sample_counts.get("TP53_wt", 0)) == row.n_wt, c + ": comparator count mismatch")
        if not row.fit_eligible:
            require(
                pd.isna(row.n_q_le_0_05), c + ": ineligible cohort reported as zero discoveries"
            )
            continue
        rank = pd.read_csv(root / "rankings" / f"{c}.deg_ranking.tsv", sep="\t")
        require(not rank.gene.duplicated().any(), c + ": duplicate ranking gene")
        require(
            not rank.gene.astype(str).str.fullmatch(r"\d+").any(), c + ": numeric gene identity"
        )
        require(
            np.isfinite(rank.score).all() and rank.gene_id_type.eq("symbol").all(),
            c + ": invalid ranking",
        )
        path = root / "evidence_tables" / f"{c}.evidence_table.tsv"
        table = pd.read_csv(path, sep="\t")
        require(
            len(table) == 50 and not table.term_id.duplicated().any(),
            c + ": incomplete Hallmark table",
        )
        require(
            np.isfinite(table.stat).all() and table.qval.between(0, 1).all(),
            c + ": invalid statistics",
        )
        require(table.term_id.isin(reference.gs_name).all(), c + ": unexpected pathway identity")
        require(int(table.qval.le(0.05).sum()) == row.n_q_le_0_05, c + ": q count mismatch")
        require(
            (
                (table.stat.gt(0) & table.direction.eq("up"))
                | (table.stat.lt(0) & table.direction.eq("down"))
            ).all(),
            c + ": direction mismatch",
        )
        current = pd.read_csv(str(path) + ".memberships.tsv", sep="\t")
        require(
            set(zip(current.gs_name, current.gene_symbol, strict=True)) == memberships,
            c + ": membership snapshot differs",
        )
        genes = set(rank.gene)
        for term in table.itertuples(index=False):
            edge = set(str(term.evidence_genes).split(","))
            require(edge <= genes, c + ": leading edge outside ranking")
            require(
                all((term.term_id, g) in memberships for g in edge),
                c + ": leading edge outside gene set",
            )
        report = root / "source_reports" / c
        meta = json.loads((report / "run_meta.json").read_text())
        require(
            meta["workflow"] == "source" and meta["status"] == "COMPLETE",
            c + ": incomplete source workflow",
        )
        require(meta["model_calls"] == 0, c + ": unexpected model call")
        require(
            meta["selected_count"] == row.n_q_le_0_05 and meta["candidate_count"] == 50,
            c + ": source selection mismatch",
        )
        require(
            hashlib.sha256(path.read_bytes()).hexdigest()
            == meta["inputs"]["evidence_table"]["sha256"],
            c + ": source input hash mismatch",
        )
        for artifact in meta["artifacts"].values():
            local = report / Path(artifact["path"]).name
            require(
                local.exists()
                and hashlib.sha256(local.read_bytes()).hexdigest() == artifact["sha256"],
                c + ": report artifact hash mismatch",
            )
        checks[c] = {
            "ranking_genes": len(rank),
            "tested_sets": len(table),
            "q_le_0_05": int(row.n_q_le_0_05),
            "memberships_match": True,
            "leading_edges_match_ranking_and_sets": True,
            "report_artifact_hashes_match": True,
            "source_selection_matches": True,
            "model_calls": 0,
        }
    (root / "result_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"status": "passed", "cohorts": list(checks)}, indent=2))


if __name__ == "__main__":
    main()
