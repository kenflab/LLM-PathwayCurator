from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

from .audit import audit_claims
from .distill import distill_evidence
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

    distilled = distill_evidence(ev, card)
    proposed = select_claims(distilled, card)
    audited = audit_claims(proposed, distilled, card)

    write_report(audited, distilled, card, args.outdir)

    audited.to_csv(os.path.join(args.outdir, "audit_log.tsv"), sep="\t", index=False)
    with open(os.path.join(args.outdir, "sample_card.resolved.json"), "w") as f:
        json.dump(card.model_dump(), f, indent=2)


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
    args.func(args)


if __name__ == "__main__":
    main()
