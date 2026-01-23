# LLM-PathwayCurator/src/llm_pathway_curator/masking.py
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .noise_lists import NOISE_LISTS, NOISE_PATTERNS

RESCUE_MODULES_DEFAULT = ("LINC_Noise", "Hemo_Contam")


# -------------------------
# Results
# -------------------------
@dataclass(frozen=True)
class MaskResult:
    masked_distilled: pd.DataFrame
    gene_reasons: dict[str, str]
    term_events: pd.DataFrame  # audit-grade per-term masking/stress log


# -------------------------
# Helpers
# -------------------------
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
    s = str(x).strip().replace(";", ",").replace("|", ",")
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    genes = [g.strip() for g in s.split(",") if g.strip()]
    return _dedup_preserve_order(genes)


def _sha256_short(payload: object, n: int = 12) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def _validate_frac(name: str, x: float) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise ValueError(f"{name} must be a float in [0,1] (got {x!r})") from e
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be in [0,1] (got {v})")
    return v


def build_noise_gene_reasons(*, whitelist: list[str] | None = None) -> dict[str, str]:
    if whitelist is None:
        whitelist = []

    reasons: dict[str, str] = {}
    for module_name, genes in NOISE_LISTS.items():
        for g in genes:
            reasons[g] = f"Module_{module_name}"

    for g in whitelist:
        reasons.pop(g, None)

    return reasons


def _build_noise_pool(*, whitelist: set[str]) -> list[str]:
    """
    Pool for noise injection.
    Keep it simple + deterministic:
      - all genes from NOISE_LISTS
      - exclude whitelist
    """
    pool: list[str] = []
    for _module, genes in NOISE_LISTS.items():
        for g in genes:
            gg = str(g).strip()
            if gg and gg not in whitelist:
                pool.append(gg)
    return _dedup_preserve_order(pool)


def _pick_k(rng: random.Random, xs: list[str], k: int) -> list[str]:
    if k <= 0 or not xs:
        return []
    if k >= len(xs):
        ys = xs[:]
        rng.shuffle(ys)
        return ys
    return rng.sample(xs, k)


# -------------------------
# v0: deterministic masking (kept for backward compatibility)
# -------------------------
def apply_gene_masking(
    distilled: pd.DataFrame,
    *,
    genes_col: str = "evidence_genes",
    rescue_modules: tuple[str, ...] | None = None,
    whitelist: list[str] | None = None,
    seed: int | None = None,
) -> MaskResult:
    """
    v0 behavior: deterministic "noise gene" masking by lists/patterns.

    NOTE:
      - seed is accepted for CLI plumbing; v0 masking is deterministic (seed ignored).
      - For Fig2 v4 stress (dropout/noise/contradiction), use apply_evidence_stress().
    """
    _ = seed  # intentionally unused in v0

    if rescue_modules is None:
        rescue_modules = RESCUE_MODULES_DEFAULT
    if whitelist is None:
        whitelist = []

    if genes_col not in distilled.columns:
        raise ValueError(f"apply_gene_masking: missing column {genes_col}")

    list_reasons = build_noise_gene_reasons(whitelist=whitelist)

    import re

    compiled = {k: re.compile(pat) for k, pat in NOISE_PATTERNS.items()}

    wl = set([str(g).strip() for g in whitelist if str(g).strip()])

    gene_reasons: dict[str, str] = {}
    out = distilled.copy()

    masked_lists: list[list[str]] = []
    masked_counts: list[int] = []

    # term-level audit log
    events: list[dict[str, Any]] = []

    for idx, row in out.iterrows():
        genes = _as_gene_list(row[genes_col])
        before = genes[:]

        if not genes:
            masked_lists.append([])
            masked_counts.append(0)
            events.append(
                {
                    "row_index": int(idx),
                    "mode": "clean",
                    "before_n": 0,
                    "after_n": 0,
                    "dropped_n": 0,
                    "dropped_hash": "",
                    "injected_n": 0,
                    "injected_hash": "",
                    "notes": "empty evidence_genes",
                }
            )
            continue

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
            # v0: global map (best-effort). Keep first reason to avoid overwriting.
            if g not in gene_reasons:
                gene_reasons[g] = reason

        after = kept2[:]
        masked_lists.append(after)
        masked_counts.append(len(removed))

        dropped = [g for g, _ in removed]
        events.append(
            {
                "row_index": int(idx),
                "mode": "clean",
                "before_n": int(len(before)),
                "after_n": int(len(after)),
                "dropped_n": int(len(dropped)),
                "dropped_hash": _sha256_short(dropped) if dropped else "",
                "injected_n": 0,
                "injected_hash": "",
                "notes": "",
            }
        )

    out[genes_col] = masked_lists
    out["evidence_genes_str"] = out[genes_col].map(lambda xs: ",".join(xs))
    out["masked_genes_count"] = masked_counts

    term_events = pd.DataFrame(events)

    # attach minimal provenance for downstream report/meta
    out.attrs["masking"] = {
        "mode": "clean",
        "rescue_modules": list(rescue_modules),
        "whitelist_n": int(len(whitelist)),
        "notes": "deterministic masking by NOISE_LISTS/NOISE_PATTERNS",
    }

    return MaskResult(masked_distilled=out, gene_reasons=gene_reasons, term_events=term_events)


# -------------------------
# v4: evidence identity stress (dropout/noise/contradiction tags)
# -------------------------
def apply_evidence_stress(
    distilled: pd.DataFrame,
    *,
    genes_col: str = "evidence_genes",
    term_uid_col: str = "term_uid",
    direction_col: str = "direction",
    seed: int = 0,
    # identity collapse knobs
    dropout_frac: float = 0.0,
    noise_frac: float = 0.0,
    # contradiction knob (tag only; audit/claim layer can consume)
    contradiction_frac: float = 0.0,
    # safety knobs
    min_genes_after: int = 1,
    whitelist: list[str] | None = None,
) -> MaskResult:
    """
    Fig2 v4: stress that targets evidence identity.

    What it does:
      1) gene dropout: remove a fraction of evidence genes per term (seeded; per-term RNG)
      2) gene noise injection: add a fraction of "noise" genes per term (seeded; per-term RNG)
      3) contradiction tag: mark some terms for direction flip downstream (seeded)
         - does NOT modify claim_json here (keep layers separated)
         - emits stress_tag/contradiction_flip columns + term_events log

    Outputs:
      - masked_distilled: updated evidence_genes (+ evidence_genes_str)
      - gene_reasons: global reasons (best-effort)
      - term_events: per-term audit log (drop/inject/flip tags, hashes)
    """
    if genes_col not in distilled.columns:
        raise ValueError(f"apply_evidence_stress: missing column {genes_col}")

    if int(min_genes_after) < 0:
        raise ValueError(f"min_genes_after must be >= 0 (got {min_genes_after})")

    dropout_frac = _validate_frac("dropout_frac", dropout_frac)
    noise_frac = _validate_frac("noise_frac", noise_frac)
    contradiction_frac = _validate_frac("contradiction_frac", contradiction_frac)

    out = distilled.copy()

    # Ensure term_uid exists for stable per-term stress logging
    if term_uid_col not in out.columns:
        if {"source", "term_id"}.issubset(set(out.columns)):
            out[term_uid_col] = (
                out["source"].astype(str).str.strip() + ":" + out["term_id"].astype(str).str.strip()
            )
        elif "term_id" in out.columns:
            out[term_uid_col] = out["term_id"].astype(str).str.strip()
        else:
            raise ValueError(
                f"apply_evidence_stress: need {term_uid_col} or (source, term_id) or term_id"
            )

    if direction_col not in out.columns:
        out[direction_col] = "na"

    wl = set([str(g).strip() for g in (whitelist or []) if str(g).strip()])

    # Deterministic per-term RNG seed is derived from (seed, term_uid, row_index fallback).
    seed = int(seed)

    noise_pool = _build_noise_pool(whitelist=wl)

    gene_reasons: dict[str, str] = {}
    events: list[dict[str, Any]] = []

    stressed_lists: list[list[str]] = []
    stress_tags: list[str] = []
    contradiction_tags: list[bool] = []

    for idx, row in out.iterrows():
        term_uid = str(row.get(term_uid_col, "")).strip()
        genes0 = _as_gene_list(row[genes_col])
        genes0 = [g for g in genes0 if g]  # safety
        before = genes0[:]

        # If term_uid is empty, include row_index as a fallback
        # to avoid identical per-term RNG streams.
        seed_payload = {
            "seed": seed,
            "term_uid": term_uid,
            "row_index": int(idx) if not term_uid else None,
        }
        term_seed = int(_sha256_short(seed_payload, n=12), 16) % (2**31 - 1)
        trng = random.Random(term_seed)

        # 1) dropout (floor-based)
        genes = before[:]
        dropped: list[str] = []
        if dropout_frac > 0.0 and len(genes) > int(min_genes_after):
            k = int(len(genes) * float(dropout_frac))
            # never drop below min_genes_after
            k = min(k, max(0, len(genes) - int(min_genes_after)))
            to_drop = set(_pick_k(trng, genes, k))
            if to_drop:
                genes = [g for g in genes if g not in to_drop]
                dropped = [g for g in before if g in to_drop]
                for g in dropped:
                    if g not in gene_reasons:
                        gene_reasons[g] = "stress_dropout"

        # 2) noise injection (floor-based)
        injected: list[str] = []
        if noise_frac > 0.0 and noise_pool:
            base = max(1, len(genes))
            k_inj = int(base * float(noise_frac))
            k_inj = max(0, k_inj)
            if k_inj > 0:
                cand = [g for g in noise_pool if g not in wl and g not in set(genes)]
                inj = _pick_k(trng, cand, k_inj)
                if inj:
                    genes = genes + inj
                    injected = inj[:]
                    for g in injected:
                        if g not in gene_reasons:
                            gene_reasons[g] = "stress_noise_inject"

        genes = _dedup_preserve_order(genes)

        # 3) contradiction tagging (direction flip downstream)
        flip = False
        if contradiction_frac > 0.0:
            flip = trng.random() < float(contradiction_frac)

        tags: list[str] = []
        if dropped:
            tags.append("dropout")
        if injected:
            tags.append("noise")
        if flip:
            tags.append("contradiction_flip")

        tag = ",".join(tags) if tags else ""

        after = genes[:]
        stressed_lists.append(after)
        stress_tags.append(tag)
        contradiction_tags.append(bool(flip))

        events.append(
            {
                "row_index": int(idx),
                "term_uid": term_uid,
                "mode": "stress",
                "seed": int(seed),
                "term_seed": int(term_seed),
                "before_n": int(len(before)),
                "after_n": int(len(after)),
                "dropped_n": int(len(dropped)),
                "dropped_hash": _sha256_short(dropped) if dropped else "",
                "injected_n": int(len(injected)),
                "injected_hash": _sha256_short(injected) if injected else "",
                "contradiction_flip": bool(flip),
                "stress_tag": tag,
            }
        )

    out[genes_col] = stressed_lists
    out["evidence_genes_str"] = out[genes_col].map(lambda xs: ",".join(xs))
    out["stress_tag"] = stress_tags
    out["contradiction_flip"] = contradiction_tags

    term_events = pd.DataFrame(events)

    out.attrs["masking"] = {
        "mode": "stress",
        "seed": int(seed),
        "dropout_frac": float(dropout_frac),
        "noise_frac": float(noise_frac),
        "contradiction_frac": float(contradiction_frac),
        "min_genes_after": int(min_genes_after),
        "whitelist_n": int(len(wl)),
        "noise_pool_n": int(len(noise_pool)),
        "notes": "evidence identity stress for Fig2 v4",
    }

    return MaskResult(masked_distilled=out, gene_reasons=gene_reasons, term_events=term_events)
