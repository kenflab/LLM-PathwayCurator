# LLM-PathwayCurator/src/llm_pathway_curator/masking.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .noise_lists import NOISE_LISTS, NOISE_PATTERNS

RESCUE_MODULES_DEFAULT = ("LINC_Noise", "Hemo_Contam")


@dataclass(frozen=True)
class MaskResult:
    """
    masked_distilled:
      - evidence_genes をマスク済み（list[str] のまま）
      - evidence_genes_str を再生成（任意）
      - masked_genes_count を追加（任意）
    gene_reasons:
      - gene -> reason（辞書）
    """

    masked_distilled: pd.DataFrame
    gene_reasons: dict[str, str]


def _as_gene_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(g).strip() for g in x if str(g).strip()]
    s = str(x).strip().replace(";", ",")
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    return [g.strip() for g in s.split(",") if g.strip()]


def build_noise_gene_reasons(
    *,
    rescue_modules: tuple[str, ...] | None = None,
    whitelist: list[str] | None = None,
) -> dict[str, str]:
    """
    Build a global noise-gene reason map from NOISE_LISTS / NOISE_PATTERNS.

    Note: regex-based patterns cannot enumerate all possible symbols without a universe;
    therefore, this function only returns list-based reasons.
    Regex masking is applied per-row using the observed genes.
    """
    if rescue_modules is None:
        rescue_modules = RESCUE_MODULES_DEFAULT
    if whitelist is None:
        whitelist = []

    reasons: dict[str, str] = {}
    for module_name, genes in NOISE_LISTS.items():
        for g in genes:
            reasons[g] = f"Module_{module_name}"

    # whitelist rescue
    for g in whitelist:
        reasons.pop(g, None)

    return reasons


def apply_gene_masking(
    distilled: pd.DataFrame,
    *,
    genes_col: str = "evidence_genes",
    rescue_modules: tuple[str, ...] | None = None,
    whitelist: list[str] | None = None,
) -> MaskResult:
    """
    Apply biological-noise masking to EvidenceTable-like df.

    Inputs:
      distilled[genes_col] should be list[str] (preferred) or string.

    Behavior:
      - list-based masking: remove genes appearing in NOISE_LISTS
      - regex-based masking: remove genes matching NOISE_PATTERNS
      - rescue policy: for modules in rescue_modules, keep ONE sentinel gene per row
        if multiple genes match that module (deterministic: lexicographically smallest)

    Returns:
      MaskResult with:
        - masked_distilled (same shape, genes masked)
        - gene_reasons dict (gene -> reason code)
    """
    if rescue_modules is None:
        rescue_modules = RESCUE_MODULES_DEFAULT
    if whitelist is None:
        whitelist = []

    if genes_col not in distilled.columns:
        raise ValueError(f"apply_gene_masking: missing column {genes_col}")

    # Precompute list-based reasons
    list_reasons = build_noise_gene_reasons(rescue_modules=rescue_modules, whitelist=whitelist)

    # Regex patterns (module -> compiled pattern)
    compiled = {k: v for k, v in NOISE_PATTERNS.items()}  # assume already regex strings
    # compile lazily to avoid importing re at module import time (optional)
    import re

    compiled = {k: re.compile(pat) for k, pat in compiled.items()}

    gene_reasons: dict[str, str] = {}
    out = distilled.copy()

    masked_lists: list[list[str]] = []
    masked_counts: list[int] = []

    for _, row in out.iterrows():
        genes = _as_gene_list(row[genes_col])
        if not genes:
            masked_lists.append([])
            masked_counts.append(0)
            continue

        # Whitelist: never mask these
        wl = set(whitelist)

        # 1) list-based removals
        kept = []
        removed = []
        for g in genes:
            if g in wl:
                kept.append(g)
                continue
            if g in list_reasons:
                removed.append((g, list_reasons[g]))
            else:
                kept.append(g)

        # 2) regex-based removals (+ rescue)
        # For each module, check which of the *current* kept genes match.
        kept2 = kept[:]
        for module_name, rx in compiled.items():
            matches = [g for g in kept2 if (g not in wl) and bool(rx.match(g))]
            if not matches:
                continue

            if module_name in rescue_modules and len(matches) >= 2:
                # deterministic sentinel: keep lexicographically smallest
                sentinel = sorted(matches)[0]
                for g in matches:
                    if g == sentinel:
                        continue
                    removed.append((g, f"Module_{module_name} (Redundant)"))
                kept2 = [g for g in kept2 if (g == sentinel) or (g not in matches)]
            else:
                # mask all matches
                for g in matches:
                    removed.append((g, f"Module_{module_name}"))
                kept2 = [g for g in kept2 if g not in matches]

        # record reasons (global dict; last write wins is fine for v0)
        for g, reason in removed:
            gene_reasons[g] = reason

        masked_lists.append(kept2)
        masked_counts.append(len(removed))

    out[genes_col] = masked_lists
    out["evidence_genes_str"] = out[genes_col].map(lambda xs: ",".join(xs))
    out["masked_genes_count"] = masked_counts

    return MaskResult(masked_distilled=out, gene_reasons=gene_reasons)
