# LLM-PathwayCurator/src/llm_pathway_curator/masking.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .noise_lists import NOISE_LISTS, NOISE_PATTERNS

RESCUE_MODULES_DEFAULT = ("LINC_Noise", "Hemo_Contam")


@dataclass(frozen=True)
class MaskResult:
    masked_distilled: pd.DataFrame
    gene_reasons: dict[str, str]


def _dedup_preserve_order(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        x = str(x).strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _as_gene_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        genes = [str(g).strip() for g in x if str(g).strip()]
        return _dedup_preserve_order(genes)
    s = str(x).strip().replace(";", ",")
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    genes = [g.strip() for g in s.split(",") if g.strip()]
    return _dedup_preserve_order(genes)


def build_noise_gene_reasons(
    *,
    whitelist: list[str] | None = None,
) -> dict[str, str]:
    if whitelist is None:
        whitelist = []

    reasons: dict[str, str] = {}
    for module_name, genes in NOISE_LISTS.items():
        for g in genes:
            reasons[g] = f"Module_{module_name}"

    for g in whitelist:
        reasons.pop(g, None)

    return reasons


def apply_gene_masking(
    distilled: pd.DataFrame,
    *,
    genes_col: str = "evidence_genes",
    rescue_modules: tuple[str, ...] | None = None,
    whitelist: list[str] | None = None,
    seed: int | None = None,
) -> MaskResult:
    """
    NOTE(v0): `seed` is accepted for CLI-level reproducibility plumbing.
    Masking in v0 is deterministic, so seed is intentionally unused.
    """
    _ = seed  # keep v0 deterministic, but don't crash if CLI passes seed

    if rescue_modules is None:
        rescue_modules = RESCUE_MODULES_DEFAULT
    if whitelist is None:
        whitelist = []

    if genes_col not in distilled.columns:
        raise ValueError(f"apply_gene_masking: missing column {genes_col}")

    list_reasons = build_noise_gene_reasons(whitelist=whitelist)

    import re

    compiled = {k: re.compile(pat) for k, pat in NOISE_PATTERNS.items()}

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

        wl = set(whitelist)

        kept: list[str] = []
        removed: list[tuple[str, str]] = []
        for g in genes:
            if g in wl:
                kept.append(g)
            elif g in list_reasons:
                removed.append((g, list_reasons[g]))
            else:
                kept.append(g)

        kept2 = kept[:]
        for module_name, rx in compiled.items():
            matches = [g for g in kept2 if (g not in wl) and bool(rx.match(g))]
            if not matches:
                continue

            if module_name in rescue_modules and len(matches) >= 2:
                sentinel = sorted(matches)[0]
                for g in matches:
                    if g != sentinel:
                        removed.append((g, f"Module_{module_name} (Redundant)"))
                kept2 = [g for g in kept2 if (g == sentinel) or (g not in matches)]
            else:
                for g in matches:
                    removed.append((g, f"Module_{module_name}"))
                kept2 = [g for g in kept2 if g not in matches]

        for g, reason in removed:
            gene_reasons[g] = reason

        masked_lists.append(kept2)
        masked_counts.append(len(removed))

    out[genes_col] = masked_lists
    out["evidence_genes_str"] = out[genes_col].map(lambda xs: ",".join(xs))
    out["masked_genes_count"] = masked_counts

    return MaskResult(masked_distilled=out, gene_reasons=gene_reasons)
