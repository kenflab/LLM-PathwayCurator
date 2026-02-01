# LLM-PathwayCurator/src/llm_pathway_curator/masking.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import _shared
from .noise_lists import NOISE_LISTS, NOISE_PATTERNS
from .utils import (
    build_id_to_symbol_from_distilled,
    load_id_map_tsv,
    map_ids_to_symbols,
)

RESCUE_MODULES_DEFAULT = ("LINC_Noise", "LOC_Locus", "Mouse_Predicted_Gm", "Mouse_Rik")


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
def _as_gene_list(x: Any) -> list[str]:
    """
    Single source of truth: _shared.parse_genes()

    Policy:
      - conservative splitting (aligns with distill/modules/select/audit)
      - NA handling via _shared.NA_TOKENS
      - deterministic dedup preserving order
    """
    genes = _shared.parse_genes(x)
    genes = [str(g).strip() for g in genes if str(g).strip()]
    return _shared.dedup_preserve_order(genes)


def _validate_frac(name: str, x: float) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise ValueError(f"{name} must be a float in [0,1] (got {x!r})") from e
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be in [0,1] (got {v})")
    return v


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _load_gene_id_map_from_env() -> tuple[dict[str, str], str]:
    """
    Reproducible, no network.

    Priority:
      1) env LLMPATH_GENE_ID_MAP_TSV (explicit user-supplied)
      2) repo-local defaults (union):
           - resources/gene_id_maps/id_map.tsv(.gz)
           - resources/gene_id_maps/ensembl_id_map.tsv(.gz)

    Returns:
      (map, source_label)
    """
    # 1) explicit env (single file)
    try:
        p = (os.environ.get("LLMPATH_GENE_ID_MAP_TSV", "") or "").strip()
        if p:
            m = load_id_map_tsv(p)
            if isinstance(m, dict) and m:
                return m, f"env:{p}"
    except Exception:
        pass

    # 2) repo-local default union
    merged: dict[str, str] = {}
    used: list[str] = []

    try:
        here = Path(__file__).resolve()
        repo_root = here.parents[2]
        cand = [
            repo_root / "resources" / "gene_id_maps" / "id_map.tsv.gz",
            repo_root / "resources" / "gene_id_maps" / "id_map.tsv",
            repo_root / "resources" / "gene_id_maps" / "ensembl_id_map.tsv.gz",
            repo_root / "resources" / "gene_id_maps" / "ensembl_id_map.tsv",
        ]
        for p in cand:
            if not p.exists():
                continue
            try:
                m = load_id_map_tsv(str(p))
                if isinstance(m, dict) and m:
                    # deterministic merge: keep first mapping for a given key
                    for k, v in m.items():
                        merged.setdefault(str(k).strip(), str(v).strip())
                    used.append(str(p))
            except Exception:
                continue
    except Exception:
        pass

    if merged:
        # keep src short but informative
        if len(used) == 1:
            return merged, f"default:{used[0]}"
        return merged, "default:entrez+ensembl"

    return {}, "none"


def _resolve_id2sym(
    distilled: pd.DataFrame,
    *,
    user_id2sym: dict[str, str] | None = None,
) -> tuple[dict[str, str], str]:
    """
    Best-effort, reproducible, no network.

    Priority (same spirit as audit_log/report):
      1) distilled-derived mapping (if columns exist)
      2) env LLMPATH_GENE_ID_MAP_TSV OR repo-local default (entrez+ensembl union)
      3) user-provided override (function arg)  <-- wins

    Returns:
      (id2sym, source_label)
    """
    base: dict[str, str] = {}
    try:
        base = build_id_to_symbol_from_distilled(distilled) or {}
    except Exception:
        base = {}

    extra, extra_src = _load_gene_id_map_from_env()

    merged: dict[str, str] = {}
    # deterministic: base first, then extra (extra can fill gaps)
    for k, v in (base or {}).items():
        kk = str(k).strip()
        vv = str(v).strip()
        if kk and vv:
            merged.setdefault(kk, vv)
    for k, v in (extra or {}).items():
        kk = str(k).strip()
        vv = str(v).strip()
        if kk and vv:
            merged.setdefault(kk, vv)

    # user overrides win (explicit > implicit)
    if isinstance(user_id2sym, dict) and user_id2sym:
        for k, v in user_id2sym.items():
            kk = str(k).strip()
            vv = str(v).strip()
            if kk and vv:
                merged[kk] = vv
        src = "user_override"
    else:
        if extra:
            src = extra_src
        elif base:
            src = "distilled"
        else:
            src = "none"

    return merged, src


def _expand_whitelist(
    *, whitelist: list[str] | None, id2sym: dict[str, str], sym2id: dict[str, str]
) -> set[str]:
    """
    Allow whitelist entries to be either IDs or symbols.
    If user passes a symbol, we also allow its mapped ID when known (and vice versa).
    """
    wl_raw = [str(g).strip() for g in (whitelist or []) if str(g).strip()]
    wl: set[str] = set(wl_raw)

    # Add mapped symbol for whitelisted IDs
    for g in wl_raw:
        s = id2sym.get(g, "")
        if s:
            wl.add(s)

    # Add mapped ID for whitelisted symbols (best-effort)
    for g in wl_raw:
        i = sym2id.get(g, "")
        if i:
            wl.add(i)

    return wl


def _build_sym2id(id2sym: dict[str, str]) -> dict[str, str]:
    """
    Best-effort reverse map.
    If multiple IDs map to the same symbol, keep the first encountered
    (deterministic by insertion order).
    """
    sym2id: dict[str, str] = {}
    for i, s in id2sym.items():
        if s and s not in sym2id:
            sym2id[s] = i
    return sym2id


def build_noise_gene_reasons(*, whitelist: list[str] | None = None) -> dict[str, str]:
    """
    Noise dictionaries are symbol-centric.
    Keep them as-is (trim-only), and do symbol-view matching upstream.
    """
    if whitelist is None:
        whitelist = []

    reasons: dict[str, str] = {}
    for module_name, genes in NOISE_LISTS.items():
        for g in genes:
            reasons[str(g).strip()] = f"Module_{module_name}"

    for g in whitelist:
        reasons.pop(str(g).strip(), None)

    return reasons


def _build_noise_pool(*, whitelist: set[str]) -> list[str]:
    """
    Pool for noise injection.
    Keep it simple + deterministic:
      - all genes from NOISE_LISTS (symbol tokens)
      - exclude whitelist (accepts IDs/symbols)
    """
    pool: list[str] = []
    for _module, genes in NOISE_LISTS.items():
        for g in genes:
            gg = str(g).strip()
            if gg and gg not in whitelist:
                pool.append(gg)
    return _shared.dedup_preserve_order(pool)


def _pick_k(rng: random.Random, xs: list[str], k: int) -> list[str]:
    if k <= 0 or not xs:
        return []
    if k >= len(xs):
        ys = xs[:]
        rng.shuffle(ys)
        return ys
    return rng.sample(xs, k)


def _row_id_for_events(row: pd.Series, idx: Any) -> int:
    """
    Prefer stable provenance if upstream provided it.
    - distill.py creates raw_index; keep that if present.
    Fallback: current DataFrame index.
    """
    if "raw_index" in row.index:
        try:
            return int(row["raw_index"])
        except Exception:
            pass
    try:
        return int(idx)
    except Exception:
        return 0


def _match_key(token: str, *, id2sym: dict[str, str], use_symbol_view: bool) -> str:
    """
    Key used for matching against NOISE_LISTS/NOISE_PATTERNS.

    Policy (audit-safe):
      - Matching is performed in "symbol-view" when enabled (best-effort ID->symbol).
      - Removal always targets the ORIGINAL token (ID) to preserve evidence identity.

    Implementation:
      - Use utils.map_ids_to_symbols() as the single source of truth
        (same behavior as report/audit_log symbol rendering).
    """
    t = str(token).strip()
    if not t:
        return ""
    if not use_symbol_view:
        return t
    try:
        xs = map_ids_to_symbols(t, id2sym)
        if isinstance(xs, list) and xs:
            s = str(xs[0]).strip()
            return s or t
    except Exception:
        pass
    return t


# -------------------------
# deterministic masking (backward compatible)
# -------------------------
def apply_gene_masking(
    distilled: pd.DataFrame,
    *,
    genes_col: str = "evidence_genes",
    rescue_modules: tuple[str, ...] | None = None,
    whitelist: list[str] | None = None,
    seed: int | None = None,
    # advanced knobs (default keep v0 behavior)
    pattern_mode: str = "match",  # "match" (prefix) or "search" (substring)
    # NEW: symbol-view matching for mixed ID/symbol inputs
    id2sym: dict[str, str] | None = None,
    use_symbol_view: bool = True,
) -> MaskResult:
    """
    behavior: deterministic "noise gene" masking by lists/patterns.

    NOTE:
      - seed is accepted for CLI plumbing; v0 masking is deterministic (seed ignored).
      - For Fig2 v4 stress (dropout/noise/contradiction), use apply_evidence_stress().

    Mixed token policy (important):
      - NOISE_LISTS/NOISE_PATTERNS are symbol-centric.
      - If use_symbol_view=True, masking matches on symbol-view keys (best-effort mapping),
        but always removes the ORIGINAL token from evidence_genes (audit identity preserved).
    """
    _ = seed  # intentionally unused in v0

    if rescue_modules is None:
        rescue_modules = RESCUE_MODULES_DEFAULT
    if whitelist is None:
        whitelist = []

    if genes_col not in distilled.columns:
        raise ValueError(f"apply_gene_masking: missing column {genes_col}")

    # Resolve mapping (reproducible, no network)
    if use_symbol_view:
        resolved_id2sym, id2sym_src = _resolve_id2sym(distilled, user_id2sym=id2sym)
    else:
        resolved_id2sym, id2sym_src = {}, "disabled"

    sym2id = _build_sym2id(resolved_id2sym) if use_symbol_view else {}

    # Whitelist supports both token and symbol forms.
    # IMPORTANT: keep consistent with report/audit_log mapping behavior.
    if use_symbol_view:
        wl_raw = [str(g).strip() for g in (whitelist or []) if str(g).strip()]
        wl: set[str] = set(wl_raw)

        # Add symbol-view for each whitelist token (ID -> symbol if possible)
        for g in wl_raw:
            try:
                syms = map_ids_to_symbols(g, resolved_id2sym)
                if isinstance(syms, list):
                    for s in syms:
                        ss = str(s).strip()
                        if ss:
                            wl.add(ss)
            except Exception:
                pass

        # Add reverse map (symbol -> ID) best-effort
        for g in wl_raw:
            i = sym2id.get(g, "")
            if i:
                wl.add(str(i).strip())
    else:
        wl = set([str(g).strip() for g in (whitelist or []) if str(g).strip()])

    # Noise dictionaries are symbol-centric; remove whitelisted tokens in either space.
    list_reasons = build_noise_gene_reasons(whitelist=list(wl))

    import re

    compiled = {k: re.compile(pat) for k, pat in NOISE_PATTERNS.items()}
    pattern_mode = str(pattern_mode or "match").strip().lower()
    if pattern_mode not in {"match", "search"}:
        raise ValueError("pattern_mode must be 'match' or 'search'")

    gene_reasons: dict[str, str] = {}
    out = distilled.copy()

    masked_lists: list[list[str]] = []
    masked_counts: list[int] = []

    # term-level audit log
    events: list[dict[str, Any]] = []

    for idx, row in out.iterrows():
        genes = _as_gene_list(row[genes_col])
        before = genes[:]
        row_id = _row_id_for_events(row, idx)

        if not genes:
            masked_lists.append([])
            masked_counts.append(0)
            events.append(
                {
                    "row_index": int(row_id),
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

        # list-based masking (symbol-centric match key)
        for token in genes:
            t = str(token).strip()
            if not t:
                continue
            key = _match_key(t, id2sym=resolved_id2sym, use_symbol_view=use_symbol_view)

            # whitelist applies to both token and symbol view
            if t in wl or key in wl:
                kept.append(t)
            elif key in list_reasons:
                removed.append((t, list_reasons[key]))
            else:
                kept.append(t)

        kept2 = kept[:]

        # pattern-based masking (match on key, remove original token)
        for module_name, rx in compiled.items():
            if pattern_mode == "search":
                matches = [
                    t
                    for t in kept2
                    if (t not in wl)
                    and (
                        _match_key(t, id2sym=resolved_id2sym, use_symbol_view=use_symbol_view)
                        not in wl
                    )
                    and bool(
                        rx.search(
                            _match_key(t, id2sym=resolved_id2sym, use_symbol_view=use_symbol_view)
                        )
                    )
                ]
            else:
                matches = [
                    t
                    for t in kept2
                    if (t not in wl)
                    and (
                        _match_key(t, id2sym=resolved_id2sym, use_symbol_view=use_symbol_view)
                        not in wl
                    )
                    and bool(
                        rx.match(
                            _match_key(t, id2sym=resolved_id2sym, use_symbol_view=use_symbol_view)
                        )
                    )
                ]

            if not matches:
                continue

            if module_name in rescue_modules and len(matches) >= 2:
                sentinel = sorted(matches)[0]
                for t in matches:
                    if t != sentinel:
                        removed.append((t, f"Module_{module_name} (Redundant)"))
                kept2 = [t for t in kept2 if (t == sentinel) or (t not in matches)]
            else:
                for t in matches:
                    removed.append((t, f"Module_{module_name}"))
                kept2 = [t for t in kept2 if t not in matches]

        for t, reason in removed:
            # v0: global map (best-effort). Keep first reason to avoid overwriting.
            if t not in gene_reasons:
                gene_reasons[t] = reason

        after = _shared.dedup_preserve_order(kept2[:])
        masked_lists.append(after)
        masked_counts.append(len(removed))

        dropped = [t for t, _ in removed]

        # Human-readable: also emit symbol-view dropped list (does NOT affect masking identity).
        dropped_syms: list[str] = []
        if dropped and use_symbol_view and resolved_id2sym:
            for tok in dropped:
                try:
                    xs = map_ids_to_symbols(tok, resolved_id2sym)
                    if isinstance(xs, list) and xs:
                        dropped_syms.append(str(xs[0]).strip())
                    else:
                        dropped_syms.append(str(tok).strip())
                except Exception:
                    dropped_syms.append(str(tok).strip())
        notes = ""
        if use_symbol_view and not resolved_id2sym:
            notes = "symbol_view_requested_but_id2sym_empty"

        events.append(
            {
                "row_index": int(row_id),
                "mode": "clean",
                "before_n": int(len(before)),
                "after_n": int(len(after)),
                "dropped_n": int(len(dropped)),
                "dropped_genes_str": _shared.join_genes_tsv(dropped) if dropped else "",
                "dropped_genes_symbols_str": _shared.join_genes_tsv(dropped_syms)
                if dropped_syms
                else "",
                "dropped_hash": _shared.sha256_short(dropped) if dropped else "",
                "injected_n": 0,
                "injected_hash": "",
                "notes": notes,
            }
        )

    out[genes_col] = masked_lists
    # IMPORTANT: keep delimiter consistent across layers for reproducibility.
    out["evidence_genes_str"] = out[genes_col].map(_shared.join_genes_tsv)
    out["masked_genes_count"] = masked_counts

    term_events = pd.DataFrame(events)

    # attach minimal provenance for downstream report/meta
    out.attrs["masking"] = {
        "mode": "clean",
        "rescue_modules": list(rescue_modules),
        "whitelist_n_expanded": int(len(wl)),  # <- expanded wl size
        "pattern_mode": pattern_mode,
        "gene_join_delim": _shared.GENE_JOIN_DELIM,
        "use_symbol_view": bool(use_symbol_view),
        "id2sym_src": str(id2sym_src),
        "id2sym_n": int(len(resolved_id2sym)) if use_symbol_view else 0,  # <- no recompute
        "notes": (
            "deterministic masking by NOISE_LISTS/NOISE_PATTERNS "
            "(symbol-view matching when enabled)"
        ),
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
    # NEW (defaults preserve current behavior: floor-based k can be 0)
    dropout_min_k: int = 0,
    noise_min_k: int = 0,
    max_inject_k: int | None = None,
) -> MaskResult:
    """
    Fig2 v4: stress that targets evidence identity.

    What it does:
      1) gene dropout: remove a fraction of evidence genes per term (seeded; per-term RNG)
      2) gene noise injection: add a fraction of "noise" genes per term (seeded; per-term RNG)
      3) contradiction tag: mark some terms for direction flip downstream (seeded)
         - does NOT modify claim_json here (keep layers separated)
         - emits stress_tag/contradiction_flip columns + term_events log

    Notes:
      - noise injection pool is symbol-centric (NOISE_LISTS). This is acceptable
        because stress is an adversarial perturbation; it is not a biological claim.
    """
    if genes_col not in distilled.columns:
        raise ValueError(f"apply_evidence_stress: missing column {genes_col}")

    if int(min_genes_after) < 0:
        raise ValueError(f"min_genes_after must be >= 0 (got {min_genes_after})")

    dropout_frac = _validate_frac("dropout_frac", dropout_frac)
    noise_frac = _validate_frac("noise_frac", noise_frac)
    contradiction_frac = _validate_frac("contradiction_frac", contradiction_frac)

    dropout_min_k = max(0, _safe_int(dropout_min_k, 0))
    noise_min_k = max(0, _safe_int(noise_min_k, 0))
    if max_inject_k is not None:
        max_inject_k = max(0, _safe_int(max_inject_k, 0))

    out = distilled.copy()

    # Ensure term_uid exists for stable per-term stress logging
    if term_uid_col not in out.columns:
        if {"source", "term_id"}.issubset(set(out.columns)):
            out[term_uid_col] = (
                out.apply(
                    lambda r: _shared.make_term_uid(r.get("source"), r.get("term_id")),
                    axis=1,
                )
                .astype(str)
                .str.strip()
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
        row_id = _row_id_for_events(row, idx)

        # Deterministic per-term RNG stream (spec-level).
        # Use (seed, term_uid, row_id) to avoid collisions when term_uid duplicates exist.
        term_seed = _shared.seed_for_term(
            seed=int(seed),
            term_uid=term_uid,
            term_row_id=int(row_id),
        )
        trng = random.Random(int(term_seed))

        # 1) dropout (floor-based + optional min k)
        genes = before[:]
        dropped: list[str] = []
        if dropout_frac > 0.0 and len(genes) > int(min_genes_after):
            k = int(len(genes) * float(dropout_frac))
            k = max(k, dropout_min_k)
            # never drop below min_genes_after
            k = min(k, max(0, len(genes) - int(min_genes_after)))
            to_drop = set(_pick_k(trng, genes, k))
            if to_drop:
                genes = [g for g in genes if g not in to_drop]
                dropped = [g for g in before if g in to_drop]
                for g in dropped:
                    if g not in gene_reasons:
                        gene_reasons[g] = "stress_dropout"

        # 2) noise injection (floor-based + optional min k + optional max clamp)
        injected: list[str] = []
        if noise_frac > 0.0 and noise_pool:
            base = max(1, len(genes))
            k_inj = int(base * float(noise_frac))
            k_inj = max(k_inj, noise_min_k)
            if max_inject_k is not None:
                k_inj = min(k_inj, int(max_inject_k))
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

        genes = _shared.dedup_preserve_order(genes)

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

        tag = _shared.join_tags(tags) if tags else ""

        after = genes[:]
        stressed_lists.append(after)
        stress_tags.append(tag)
        contradiction_tags.append(bool(flip))

        events.append(
            {
                "row_index": int(row_id),
                "df_index": str(idx),
                "term_uid": term_uid,
                "mode": "stress",
                "seed": int(seed),
                "term_seed": int(term_seed),
                "before_n": int(len(before)),
                "after_n": int(len(after)),
                "dropped_n": int(len(dropped)),
                "dropped_hash": _shared.sha256_short(dropped) if dropped else "",
                "injected_n": int(len(injected)),
                "injected_hash": _shared.sha256_short(injected) if injected else "",
                "contradiction_flip": bool(flip),
                "stress_tag": tag,
            }
        )

    out[genes_col] = stressed_lists
    # IMPORTANT: keep delimiter consistent across layers for reproducibility.
    out["evidence_genes_str"] = out[genes_col].map(_shared.join_genes_tsv)
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
        "dropout_min_k": int(dropout_min_k),
        "noise_min_k": int(noise_min_k),
        "max_inject_k": (None if max_inject_k is None else int(max_inject_k)),
        "whitelist_n": int(len(wl)),
        "noise_pool_n": int(len(noise_pool)),
        "gene_join_delim": _shared.GENE_JOIN_DELIM,
        "notes": "evidence identity stress for Fig2 v4",
    }

    return MaskResult(masked_distilled=out, gene_reasons=gene_reasons, term_events=term_events)


def write_masking_artifacts(
    masked: MaskResult,
    *,
    outdir: str,
    prefix: str = "masking",
) -> None:
    """
    Persist masking artifacts as TSV.

    - Explicit call only (no hidden side effects in apply_gene_masking).
    - Token space only (symbols/IDs); no mapping; no network.
    - Minimal outputs: events + compact summary.
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    events = masked.term_events.copy()
    events.to_csv(out_path / f"{prefix}_events.tsv", sep="\t", index=False)

    # Compact per-term summary (easy to read)
    cols = [
        "row_index",
        "mode",
        "before_n",
        "after_n",
        "dropped_n",
        "dropped_hash",
        "injected_n",
        "injected_hash",
        "contradiction_flip",
        "stress_tag",
    ]
    keep = [c for c in cols if c in events.columns]
    summary = events[keep].copy() if keep else pd.DataFrame()
    if (not summary.empty) and ("row_index" in summary.columns):
        summary = summary.drop_duplicates(subset=["row_index"])
    summary.to_csv(out_path / f"{prefix}_summary.tsv", sep="\t", index=False)
