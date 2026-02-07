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
    """Abort execution with a message.

    Parameters
    ----------
    msg : str
        Human-readable error message.

    Raises
    ------
    SystemExit
        Always raised with ``msg`` as the exit message.
    """
    raise SystemExit(msg)


def make_card(cancer: str, *, context_gate_mode: str) -> dict:
    """Create a Sample Card dictionary for a single cancer cohort.

    This returns a JSON-serializable dict that encodes the study context
    and analysis knobs used by the Fig. 2 pipelines.

    Parameters
    ----------
    cancer : str
        Cancer/condition label (e.g., ``"HNSC"``).
    context_gate_mode : str
        Context gate behavior. Expected values are ``"note"`` or
        ``"hard"`` (written under ``extra.context_gate_mode``).

    Returns
    -------
    dict
        Sample Card object to be written as ``*.sample_card.json``.
        The top-level keys include ``condition``, ``tissue``,
        ``perturbation``, ``comparison``, ``k_claims``, ``notes``,
        and an ``extra`` mapping for benchmark-specific knobs.

    Notes
    -----
    - The caller controls gating via ``context_gate_mode``; this function
      does not run any audits and does not validate existence of files.
    """
    return {
        "condition": cancer,
        "tissue": "tumor",
        "perturbation": "genotype",
        "comparison": "TP53_mut_vs_TP53_wt",
        "k_claims": 100,
        "notes": "TCGA Pan-Cancer; groups defined from MC3 TP53 PASS mutations.",
        "extra": {
            "benchmark_id": "PANCAN_TP53_v1",
            "goal": "context-conditioned pathway claim selection with calibrated abstention",
            # knobs (Fig2 needs these)
            "audit_tau": 0.8,
            # gate behavior
            #   note  -> PASS + annotate context nonspecificity
            #   hard  -> ABSTAIN_CONTEXT_NONSPECIFIC
            "context_gate_mode": context_gate_mode,
            # "select_context_mode": "proxy",
            "gene_id_map_tsv": "resources/gene_id_maps/id_map.tsv.gz",
            # "gene_id_map_tsv": "resources/gene_id_maps/ensembl_id_map.tsv.gz"
        },
    }


def main() -> None:
    """Generate per-cancer Sample Card JSON files for Fig. 2.

    This script scans ``paper/source_data/PANCAN_TP53_v1/derived/groups``
    for ``*.groups.tsv`` files (excluding ``PANCAN.groups.tsv``) and writes
    two cards per cancer into ``paper/source_data/PANCAN_TP53_v1/sample_cards``:

    - ``{CANCER}.note.sample_card.json`` (no hard gate on nonspecificity)
    - ``{CANCER}.hard.sample_card.json`` (ABSTAIN on nonspecificity)

    Raises
    ------
    SystemExit
        If the expected groups directory is missing or no per-cancer group
        files are found.

    Notes
    -----
    - Cancer labels are inferred from the filename prefix before the first
      dot (e.g., ``BRCA.groups.tsv`` -> ``BRCA``).
    - Output JSON is pretty-printed with sorted keys for reproducibility.
    """
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
