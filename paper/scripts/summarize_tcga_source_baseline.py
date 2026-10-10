#!/usr/bin/env python3
"""Compare a completed TCGA source report with its unmodified q-value baseline.

This is a saved-output audit, not an evaluation of LLM or biological accuracy.
It requires all 50 Hallmark candidates, checks source identities, and writes a
new directory. Input reports and enrichment results are never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(run_dir: Path, outdir: Path):
    status = json.loads((run_dir / "RUN_STATUS.json").read_text())
    if status["status"] != "completed":
        raise ValueError("A completed rebuild is required")
    cohort_table = pd.read_csv(run_dir / "cohort_summary.tsv", sep="\t")
    outdir.mkdir(parents=True, exist_ok=False)
    comparisons, term_rows, module_rows = [], [], []
    provenance = {
        "run_status_sha256": sha256(run_dir / "RUN_STATUS.json"),
        "code_commit": status["code_commit"],
        "model_calls": 0,
        "inputs": {},
    }
    for cohort in cohort_table.itertuples(index=False):
        c = cohort.cancer
        if not cohort.fit_eligible:
            continue
        path = run_dir / "evidence_tables" / f"{c}.evidence_table.tsv"
        source = run_dir / "source_reports" / c
        meta = json.loads((source / "run_meta.json").read_text())
        table = pd.read_csv(path, sep="\t")
        reports = [json.loads(line) for line in (source / "report.jsonl").read_text().splitlines()]
        if (
            len(table) != 50
            or table.term_id.duplicated().any()
            or len(reports) != 50
            or meta["config"]["q_threshold"] != 0.05
        ):
            raise ValueError(f"{c}: expected 50 distinct Hallmark candidates at q=0.05")
        if sha256(path) != meta["inputs"]["evidence_table"]["sha256"]:
            raise ValueError(f"{c}: source hash mismatch")
        for artifact in meta["artifacts"].values():
            saved = source / Path(artifact["path"]).name
            if sha256(saved) != artifact["sha256"]:
                raise ValueError(f"{c}: report artifact hash mismatch: {saved.name}")
        if meta["candidate_count"] != 50 or meta["config"]["k_claims"] < 50:
            raise ValueError(f"{c}: capped or incomplete comparison")
        keyed = {r["evidence"]["term_id"]: r for r in reports}
        if set(keyed) != set(table.term_id):
            raise ValueError(f"{c}: source term identities disagree")
        q_selected = set(table.loc[table.qval.le(0.05), "term_id"])
        source_selected = {term for term, row in keyed.items() if row["selected"]}
        modules = json.loads((source / "modules.json").read_text())
        selected_uids = {row["evidence"]["term_uid"] for row in reports if row["selected"]}
        for m in modules["modules"]:
            module_rows.append(
                {
                    "cancer": c,
                    "module_id": m["module_id"],
                    "n_terms": len(m["member_term_uids"]),
                    "n_selected_terms": len(selected_uids & set(m["member_term_uids"])),
                    "n_union_genes": len(m["union_genes"]),
                    "n_common_genes": len(m["common_to_all_genes"]),
                    "review_flags": ";".join(m["review_flags"]),
                    "members": ";".join(m["member_term_uids"]),
                }
            )
        for row in table.itertuples(index=False):
            current = keyed[row.term_id]
            evidence = current["evidence"]
            if not math.isclose(
                evidence["stat"], row.stat, rel_tol=1e-12, abs_tol=1e-12
            ) or not math.isclose(evidence["qval"], row.qval, rel_tol=1e-12, abs_tol=0):
                raise ValueError(f"{c}: source statistic differs for {row.term_id}")
            term_rows.append(
                {
                    "cancer": c,
                    "term_id": row.term_id,
                    "NES": row.stat,
                    "qval": row.qval,
                    "q_baseline_selected": row.term_id in q_selected,
                    "source_selected": current["selected"],
                    "source_status": current["decision_status"],
                    "source_scope": current["decision_scope"],
                    "source_statement": current["source_statement"],
                    "evidence_sha256": evidence["evidence_sha256"],
                }
            )
        comparisons.append(
            {
                "cancer": c,
                "n_mut": cohort.n_mut,
                "n_call_negative": cohort.n_wt,
                "n_candidates": len(table),
                "q_baseline_selected": len(q_selected),
                "source_selected": len(source_selected),
                "selection_difference_count": len(q_selected ^ source_selected),
                "all_source_candidates_retained": len(reports) == len(table),
                "n_modules_all_candidates": len(modules["modules"]),
                "n_overlap_edges": len(modules["edges"]),
                "submitted_text_count": meta["submitted_text_count"],
                "biological_accuracy_evaluated": False,
            }
        )
        provenance["inputs"][c] = {
            "evidence_sha256": sha256(path),
            "report_sha256": sha256(source / "report.jsonl"),
            "modules_sha256": sha256(source / "modules.json"),
        }
    pd.DataFrame(comparisons).to_csv(outdir / "q_baseline_comparison.tsv", sep="\t", index=False)
    pd.DataFrame(term_rows).to_csv(outdir / "source_term_comparison.tsv", sep="\t", index=False)
    pd.DataFrame(module_rows).to_csv(outdir / "source_modules.tsv", sep="\t", index=False)
    provenance["interpretation"] = (
        "Source selection at q<=0.05 is compared with the same q-value rule. "
        "Equal selection is expected and is not biological or LLM superiority. "
        "Modules describe supporting-gene overlap across all candidates. "
        "No submitted prose was independently evaluated by this audit."
    )
    (outdir / "comparison_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    print(pd.DataFrame(summarize(args.run_dir, args.outdir)).to_string(index=False))


if __name__ == "__main__":
    main()
