# LLM-PathwayCurator/src/llm_pathway_curator/cli.py
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

from .audit import audit_claims
from .distill import distill_evidence
from .modules import attach_module_ids, factorize_modules_connected_components
from .report import write_report
from .sample_card import SampleCard
from .schema import EvidenceTable
from .select import select_claims


@dataclass(frozen=True)
class RunConfig:
    evidence_table: str
    sample_card: str
    outdir: str


def cmd_run(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)

    ev = EvidenceTable.read_tsv(args.evidence_table).df
    card = SampleCard.from_json(args.sample_card)

    # A) distill (evidence hygiene)
    distilled = distill_evidence(ev, card)

    # B) modules (evidence factorization)
    mod_out = factorize_modules_connected_components(distilled)
    distilled = attach_module_ids(distilled, mod_out.term_modules_df)

    # C) claims -> audit -> report
    proposed = select_claims(distilled, card)
    audited = audit_claims(proposed, distilled, card)

    write_report(audited, distilled, card, args.outdir)

    # Persist resolved SampleCard
    out_sc = os.path.join(args.outdir, "sample_card.resolved.json")
    with open(out_sc, "w", encoding="utf-8") as f:
        json.dump(card.model_dump(), f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Optional: also persist modules/edges for debugging (cheap + useful)
    mod_out.modules_df.to_csv(os.path.join(args.outdir, "modules.tsv"), sep="\t", index=False)
    mod_out.term_modules_df.to_csv(
        os.path.join(args.outdir, "term_modules.tsv"), sep="\t", index=False
    )
    mod_out.edges_df.to_csv(os.path.join(args.outdir, "term_gene_edges.tsv"), sep="\t", index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-pathway-curator")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run distill → modules → claims → audit → report")
    p_run.add_argument("--evidence-table", required=True, help="TSV EvidenceTable (term×gene)")
    p_run.add_argument("--sample-card", required=True, help="sample_card.json")
    p_run.add_argument("--outdir", required=True, help="output directory")
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        # Minimal, reproducible failure (nice for CI)
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
