#!/usr/bin/env python3
# LLM-PathwayCurator/examples/demo/make_evidence_table.py
from __future__ import annotations

import argparse
from pathlib import Path

from llm_pathway_curator.adapters.fgsea import (
    FgseaAdapterConfig,
    convert_fgsea_table_to_evidence_tsv,
)
from llm_pathway_curator.schema import EvidenceTable


def main() -> None:
    ap = argparse.ArgumentParser(description="demo: fgsea results -> EvidenceTable TSV")
    ap.add_argument(
        "--fgsea", required=True, help="Input fgsea results TSV (must include pathway, leadingEdge)"
    )
    ap.add_argument("--out", required=True, help="Output EvidenceTable TSV")
    ap.add_argument("--source-name", default="fgsea_demo", help="EvidenceTable.source tag")
    args = ap.parse_args()

    inp = Path(args.fgsea).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = FgseaAdapterConfig(source_name=str(args.source_name))
    convert_fgsea_table_to_evidence_tsv(str(inp), str(out), config=cfg)

    # Fail-fast contract check
    EvidenceTable.read_tsv(str(out))

    print("[OK] wrote EvidenceTable:", out)


if __name__ == "__main__":
    main()
