# LLM-PathwayCurator/src/llm_pathway_curator/masking.py

"""
Result container for gene masking / evidence stress.

Attributes
----------
masked_distilled : pandas.DataFrame
    Copy of the input `distilled` with `genes_col` updated to the new
    evidence gene list per term. Also includes helper columns such as
    `evidence_genes_str`, `masked_genes_count`, `stress_tag`, etc.,
    depending on the operation.
gene_reasons : dict[str, str]
    Best-effort mapping from a gene token to a short reason string.
    Reasons are token-space labels (no network), e.g. "Module_X",
    "stress_dropout", "stress_noise_inject".
term_events : pandas.DataFrame
    Audit-grade per-term event log (drop/inject counts, hashes, tags).
"""

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
    Parse a gene field into a deterministic list of tokens.

    Parameters
    ----------
    x : Any
        Input scalar or list-like gene field.

    Returns
    -------
    list[str]
        Parsed gene tokens with:
        - conservative splitting (spec-aligned),
        - NA handling via `_shared` vocabulary,
        - deterministic de-duplication preserving order.

    Notes
    -----
    This function delegates parsing policy to `_shared.parse_genes()` and then
    applies trim + order-preserving de-duplication to keep behavior consistent
    across distill/modules/select/audit layers.
    """
    genes = _shared.parse_genes(x)
    genes = [str(g).strip() for g in genes if str(g).strip()]
    return _shared.dedup_preserve_order(genes)


def _validate_frac(name: str, x: float) -> float:
    """
    Validate a fraction parameter in the inclusive range [0, 1].

    Parameters
    ----------
    name : str
        Parameter name (for error messages).
    x : float
        Candidate value.

    Returns
    -------
    float
        Validated float in [0, 1].

    Raises
    ------
    ValueError
        If `x` is not convertible to float or is outside [0, 1].
    """
    try:
        v = float(x)
    except Exception as e:
        raise ValueError(f"{name} must be a float in [0,1] (got {x!r})") from e
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be in [0,1] (got {v})")
    return v


def _safe_int(x: Any, default: int) -> int:
    """
    Convert a value to int with a default fallback.

    Parameters
    ----------
    x : Any
        Candidate value.
    default : int
        Value returned if conversion fails.

    Returns
    -------
    int
        Parsed integer or `default` on failure.
    """
    try:
        return int(x)
    except Exception:
        return int(default)


def _load_gene_id_map_from_env() -> tuple[dict[str, str], str]:
    """
    Load a gene ID -> symbol mapping from local resources (no network).

    Returns
    -------
    (tuple[dict[str, str], str])
        (id2sym, source_label)
        - id2sym : dict[str, str]
          Mapping from gene IDs to symbols.
        - source_label : str
          Short provenance label ("env:...", "default:...", "none").

    Notes
    -----
    Priority
    1) Environment variable LLMPATH_GENE_ID_MAP_TSV (single user-supplied file)
    2) Repo-local defaults (union, first-hit wins per key):
       - resources/gene_id_maps/id_map.tsv(.gz)
       - resources/gene_id_maps/ensembl_id_map.tsv(.gz)

    The merge is deterministic: the first mapping for a given key is kept.
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
    Resolve an ID->symbol mapping for symbol-view matching (reproducible, no network).

    Parameters
    ----------
    distilled : pandas.DataFrame
        Distilled evidence table. If recognizable columns exist, an ID->symbol
        mapping can be inferred from the DataFrame itself.
    user_id2sym : dict[str, str] or None, optional
        Explicit caller-supplied mapping that overrides all other sources.

    Returns
    -------
    (tuple[dict[str, str], str])
        (id2sym, source_label) where `source_label` describes the provenance.

    Notes
    -----
    This function builds a best-effort mapping with deterministic precedence:
    - base mapping from `distilled` (when possible),
    - extra mapping from env/repo-local maps (fills gaps),
    - optional `user_id2sym` overrides all keys (explicit wins).
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
        if base and extra:
            src = f"distilled+{extra_src}"
        elif base:
            src = "distilled"
        elif extra:
            src = extra_src
        else:
            src = "none"

    return merged, src


def _expand_whitelist(
    *, whitelist: list[str] | None, id2sym: dict[str, str], sym2id: dict[str, str]
) -> set[str]:
    """
    Expand a whitelist so entries may be treated as IDs or symbols.

    Parameters
    ----------
    whitelist : list[str] or None
        Raw whitelist entries (IDs and/or symbols).
    id2sym : dict[str, str]
        Mapping from IDs to symbols.
    sym2id : dict[str, str]
        Mapping from symbols to representative IDs.

    Returns
    -------
    set[str]
        Expanded whitelist tokens including:
        - the original entries,
        - mapped symbols for whitelisted IDs (if known),
        - mapped IDs for whitelisted symbols (if known).

    Notes
    -----
    This is a best-effort convenience for mixed token spaces.
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
    Build a best-effort reverse mapping (symbol -> representative ID).

    Parameters
    ----------
    id2sym : dict[str, str]
        ID -> symbol mapping.

    Returns
    -------
    dict[str, str]
        Symbol -> ID mapping.

    Notes
    -----
    If multiple IDs map to the same symbol, the first encountered mapping is kept.
    In Python 3.7+, dict insertion order is stable, so this is deterministic given
    a deterministic `id2sym` construction.
    """
    sym2id: dict[str, str] = {}
    for i, s in id2sym.items():
        if s and s not in sym2id:
            sym2id[s] = i
    return sym2id


def build_noise_gene_reasons(*, whitelist: list[str] | None = None) -> dict[str, str]:
    """
    Construct a symbol-centric noise gene reason dictionary.

    Parameters
    ----------
    whitelist : list[str] or None, optional
        Tokens to exclude from the noise dictionary. Entries are treated as
        literal keys to remove from the symbol-centric reason map.

    Returns
    -------
    dict[str, str]
        Mapping {gene_symbol: reason} where reason is "Module_<module_name>".

    Notes
    -----
    `NOISE_LISTS` are symbol-centric by design. Mixed ID/symbol handling is done
    upstream (e.g., by symbol-view matching) rather than rewriting the noise lists.
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
    Build a deterministic pool of noise genes for injection stress.

    Parameters
    ----------
    whitelist : set[str]
        Tokens to exclude from the pool (IDs and/or symbols).

    Returns
    -------
    list[str]
        De-duplicated list of symbol tokens from `NOISE_LISTS` excluding whitelist.

    Notes
    -----
    The pool is used for adversarial perturbation (stress), not for biological
    interpretation.
    """
    pool: list[str] = []
    for _module, genes in NOISE_LISTS.items():
        for g in genes:
            gg = str(g).strip()
            if gg and gg not in whitelist:
                pool.append(gg)
    return _shared.dedup_preserve_order(pool)


def _pick_k(rng: random.Random, xs: list[str], k: int) -> list[str]:
    """
    Pick k items from a list using a provided RNG.

    Parameters
    ----------
    rng : random.Random
        RNG instance (caller-owned).
    xs : list[str]
        Candidate list.
    k : int
        Number of items to pick.

    Returns
    -------
    list[str]
        If k >= len(xs), returns a shuffled copy of xs.
        Otherwise returns rng.sample(xs, k).
    """
    if k <= 0 or not xs:
        return []
    if k >= len(xs):
        ys = xs[:]
        rng.shuffle(ys)
        return ys
    return rng.sample(xs, k)


def _row_id_for_events(row: pd.Series, idx: Any) -> int:
    """
    Determine a stable row identifier for event logging.

    Parameters
    ----------
    row : pandas.Series
        Current row.
    idx : Any
        DataFrame index value.

    Returns
    -------
    int
        Preferred stable row id for audit logs.

    Notes
    -----
    If `raw_index` exists (often produced by distill.py), it is preferred.
    Otherwise, falls back to the current DataFrame index when it can be cast to int.
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
    Compute the match key for NOISE_LISTS/NOISE_PATTERNS.

    Parameters
    ----------
    token : str
        Original gene token from evidence (often an ID).
    id2sym : dict[str, str]
        ID -> symbol mapping used for symbol-view matching.
    use_symbol_view : bool
        If True, attempt to map the token to a symbol for matching.

    Returns
    -------
    str
        Match key used for list/pattern matching.

    Notes
    -----
    Audit-safe policy:
    - Matching may occur in "symbol-view" when enabled (best-effort ID->symbol).
    - Removal always targets the ORIGINAL token to preserve evidence identity.
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
    Deterministic masking of noise genes using lists and regex patterns.

    Parameters
    ----------
    distilled : pandas.DataFrame
        Distilled evidence table.
    genes_col : str, optional
        Column name containing per-term evidence genes (list-like or scalar).
    rescue_modules : tuple[str, ...] or None, optional
        Module names for "rescue" behavior in pattern masking:
        if multiple matches are found, keep one sentinel and drop redundants.
    whitelist : list[str] or None, optional
        Tokens to protect from masking (supports IDs and symbols when symbol-view
        matching is enabled).
    seed : int or None, optional
        Accepted for CLI plumbing; masking is deterministic and seed is ignored.
    pattern_mode : str, optional
        Regex mode for patterns:
        - "match" : prefix-style match (rx.match)
        - "search": substring match (rx.search)
    id2sym : dict[str, str] or None, optional
        Optional ID -> symbol mapping override for symbol-view matching.
    use_symbol_view : bool, optional
        If True, perform matching against noise lists/patterns on symbol-view keys
        (best-effort ID->symbol mapping) while removing ORIGINAL tokens.

    Returns
    -------
    MaskResult
        Result container including:
        - updated DataFrame,
        - global gene->reason map,
        - per-term audit event table.

    Raises
    ------
    ValueError
        If `genes_col` is missing or `pattern_mode` is invalid.

    Notes
    -----
    Mixed token policy
    - `NOISE_LISTS` and `NOISE_PATTERNS` are symbol-centric.
    - When `use_symbol_view=True`, this function matches on symbol-view keys,
      but removes the original token to preserve evidence identity and hashes.

    Side effects
    ------------
    None. This function does not write files; use `write_masking_artifacts()`.
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
# Evidence identity stress (dropout/noise/contradiction tags)
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
    Apply seeded evidence-identity stress (dropout / noise / contradiction tags).

    Parameters
    ----------
    distilled : pandas.DataFrame
        Distilled evidence table.
    genes_col : str, optional
        Column name containing per-term evidence genes.
    term_uid_col : str, optional
        Column name for term UID. If missing, it will be derived from
        (source, term_id) when available, otherwise from term_id.
    direction_col : str, optional
        Direction column name. If missing, it is created as "na".
    seed : int, optional
        Global stress seed. A deterministic per-term seed is derived from
        (seed, term_uid, row_id) for stable, per-term perturbations.
    dropout_frac : float, optional
        Fraction of genes to drop per term in [0, 1].
    noise_frac : float, optional
        Fraction of genes to inject per term in [0, 1].
    contradiction_frac : float, optional
        Fraction of terms to tag for downstream direction flip (tag only).
    min_genes_after : int, optional
        Lower bound on genes remaining after dropout (clamped by term size).
    whitelist : list[str] or None, optional
        Tokens to exclude from injection candidates and preserve when possible.
    dropout_min_k : int, optional
        Minimum number of genes to drop when dropout is enabled (default 0).
    noise_min_k : int, optional
        Minimum number of genes to inject when noise is enabled (default 0).
    max_inject_k : int or None, optional
        Optional cap on injected genes per term.

    Returns
    -------
    MaskResult
        Updated DataFrame plus audit-grade term event logs.

    Raises
    ------
    ValueError
        If required columns cannot be derived or if parameters are invalid.

    Notes
    -----
    - Noise injection uses a symbol-centric pool (`NOISE_LISTS`). This is acceptable
      because stress is adversarial perturbation, not biological interpretation.
    - Contradiction is emitted as tags/flags only; it does not mutate claim JSON here.
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
    Write masking/stress artifacts to disk as TSV files.

    Parameters
    ----------
    masked : MaskResult
        Output from `apply_gene_masking()` or `apply_evidence_stress()`.
    outdir : str
        Output directory path. Created if missing.
    prefix : str, optional
        Filename prefix (default "masking").

    Returns
    -------
    None

    Notes
    -----
    - This function is an explicit I/O step (no hidden file writes in masking).
    - Outputs:
      - <prefix>_events.tsv  : full audit-grade per-term event log
      - <prefix>_summary.tsv : compact per-term summary (subset of columns)
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
