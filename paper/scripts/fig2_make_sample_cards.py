#!/usr/bin/env python3
# paper/scripts/fig2_make_sample_cards.py
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # paper/
SD = ROOT / "source_data" / "PANCAN_TP53_v1"
DER = SD / "derived"
GROUPS = DER / "groups"
OUT = SD / "sample_cards"


def _die(msg: str) -> None:
    raise SystemExit(msg)


def make_card(cancer: str) -> dict:
    # minimal v1 sample card for Fig2
    return {
        "schema_version": "v1",
        "benchmark_id": "PANCAN_TP53_v1",
        "disease": cancer,
        "tissue": "tumor",
        "perturbation": "genotype",
        "comparison": "TP53_mut_vs_TP53_wt",
        "goal": "context-conditioned pathway claim selection with calibrated abstention",
        "notes": "TCGA Pan-Cancer; groups defined from MC3 TP53 PASS mutations.",
    }


def main() -> None:
    if not GROUPS.exists():
        _die(f"[make_sample_cards] missing dir: {GROUPS}")

    OUT.mkdir(parents=True, exist_ok=True)

    cancer_files = sorted(GROUPS.glob("*.groups.tsv"))
    cancer_files = [p for p in cancer_files if p.name != "PANCAN.groups.tsv"]
    if not cancer_files:
        _die("[make_sample_cards] no per-cancer groups found (run fig2_make_groups.py first)")

    for p in cancer_files:
        cancer = p.stem.split(".", 1)[0].upper()
        out_path = OUT / f"{cancer}.sample_card.json"
        card = make_card(cancer)
        out_path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[make_sample_cards] OK")
    print(f"  wrote: {OUT} ({len(cancer_files)} cards)")


if __name__ == "__main__":
    main()
