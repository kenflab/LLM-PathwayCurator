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


def make_card(cancer: str, *, context_gate_mode: str) -> dict:
    # minimal v1 sample card for Fig2 (audit knobs live under extra)
    return {
        "disease": cancer,
        "tissue": "tumor",
        "perturbation": "genotype",
        "comparison": "TP53_mut_vs_TP53_wt",
        "k_claims": 100,
        "notes": "TCGA Pan-Cancer; groups defined from MC3 TP53 PASS mutations.",
        "extra": {
            "schema_version": "v1",
            "benchmark_id": "PANCAN_TP53_v1",
            "goal": "context-conditioned pathway claim selection with calibrated abstention",
            # knobs (Fig2 needs these)
            "audit_tau": 0.8,
            # gate behavior
            #   note  -> PASS + annotate context nonspecificity
            #   hard  -> ABSTAIN_CONTEXT_NONSPECIFIC
            "context_gate_mode": context_gate_mode,
            "gene_id_map_tsv": "resources/gene_id_maps/id_map.tsv.gz",
            # "gene_id_map_tsv": "resources/gene_id_maps/ensembl_id_map.tsv.gz"
        },
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

        # A) Fig2-normal: do NOT gate on context nonspecificity
        card_note = make_card(cancer, context_gate_mode="note")
        out_note = OUT / f"{cancer}.note.sample_card.json"
        out_note.write_text(
            json.dumps(card_note, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # B) Fig2-hard: ABSTAIN on context nonspecificity
        card_hard = make_card(cancer, context_gate_mode="hard")
        out_hard = OUT / f"{cancer}.hard.sample_card.json"
        out_hard.write_text(
            json.dumps(card_hard, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print("[make_sample_cards] OK")
    print(f"  wrote: {OUT} ({len(cancer_files)} cancers × 2 cards)")


if __name__ == "__main__":
    main()
