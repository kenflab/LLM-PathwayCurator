#!/usr/bin/env python3
# paper/scripts/figS_beataml_make_sample_cards.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # paper/
SD = ROOT / "source_data" / "BEATAML_TP53_v1"
DER = SD / "derived"
GROUPS = DER / "groups" / "BEATAML.groups.tsv"
OUT = SD / "sample_cards"


def _die(msg: str) -> None:
    raise SystemExit(msg)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_card(*, card_id: str, context_gate_mode: str, k_claims: int, tau: float) -> dict:
    """
    card_id:
      - "BEATAML"      : true BeatAML context
      - "BEATAML_SWAP" : counterfactual context for context_swap ablation
    """
    if card_id not in {"BEATAML", "BEATAML_SWAP"}:
        _die(f"[beataml_sample_cards] unknown card_id: {card_id}")

    if card_id == "BEATAML":
        condition = "BEATAML_AML"
        tissue = "blood"
        comparison = "TP53_mut_wes_vs_TP53_wt_wes"
        notes = (
            "BeatAML (waves 1–4). Groups from WES (TP53 protein-altering) mapped to "
            "RNA-seq sample IDs; clinical TP53 column used only for cohort QC and "
            "optional annotation."
        )
    else:
        # Counterfactual context: deliberately wrong disease/tissue to stress context gating.
        # EvidenceTable / groups remain BeatAML; only sample_card context keys are
        # swapped by runner.
        condition = "LUAD"
        tissue = "tumor"
        comparison = "TP53_mut_vs_TP53_wt"
        notes = (
            "COUNTERFACTUAL sample card for context_swap ablation (Supplement). "
            "Do not use for biological interpretation; used only to swap context keys "
            "against BeatAML evidence."
        )

    return {
        "condition": condition,
        "tissue": tissue,
        "perturbation": "genotype",
        "comparison": comparison,
        "k_claims": int(k_claims),
        "notes": notes,
        "extra": {
            "benchmark_id": "BEATAML_TP53_v1",
            "goal": "context-conditioned pathway claim selection with calibrated abstention",
            "audit_tau": float(tau),
            # gate behavior: note -> annotate; hard -> ABSTAIN_CONTEXT_NONSPECIFIC
            "context_gate_mode": str(context_gate_mode),
            # Where groups live (pipeline/scripts can use this)
            "groups_tsv": "derived/groups/BEATAML.groups.tsv",
            # Gene ID map
            "gene_id_map_tsv": "resources/gene_id_maps/ensembl_id_map.tsv.gz",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BeatAML (Supplement): write sample cards (note/hard)."
    )
    ap.add_argument("--groups", default=str(GROUPS), help="derived/groups/BEATAML.groups.tsv")
    ap.add_argument("--outdir", default=str(OUT), help="output dir for sample cards")
    ap.add_argument("--k-claims", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.8)
    args = ap.parse_args()

    groups_path = Path(args.groups)
    if not groups_path.exists():
        _die(f"[beataml_sample_cards] missing groups.tsv: {groups_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 2 IDs × 2 gates
    for card_id in ("BEATAML", "BEATAML_SWAP"):
        for gm in ("note", "hard"):
            card = make_card(
                card_id=card_id,
                context_gate_mode=gm,
                k_claims=int(args.k_claims),
                tau=float(args.tau),
            )
            out_path = outdir / f"{card_id}.{gm}.sample_card.json"
            _write_json(out_path, card)

    print("[beataml_sample_cards] OK")
    print(f"  groups: {groups_path}")
    print(f"  wrote: {outdir}/(BEATAML|BEATAML_SWAP).(note|hard).sample_card.json")


if __name__ == "__main__":
    main()
