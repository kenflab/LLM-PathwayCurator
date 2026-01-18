# LLM-PathwayCurator/src/llm_pathway_curator/distill.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .masking import apply_gene_masking
from .sample_card import SampleCard

_NA_TOKENS_L = {"", "na", "nan", "none"}


def _is_na_scalar(x: Any) -> bool:
    """pd.isna is unsafe for list-like; only treat scalars here."""
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict)):
        return False
    try:
        v = pd.isna(x)
        return bool(v) if isinstance(v, bool) else False
    except Exception:
        return False


def _normalize_direction_loose(x: Any) -> str:
    if _is_na_scalar(x):
        return "na"
    s = str(x).strip().lower()
    if s in {
        "up",
        "upregulated",
        "increase",
        "increased",
        "activated",
        "+",
        "pos",
        "positive",
        "1",
    }:
        return "up"
    if s in {
        "down",
        "downregulated",
        "decrease",
        "decreased",
        "suppressed",
        "-",
        "neg",
        "negative",
        "-1",
    }:
        return "down"
    return "na"


def _clean_gene_symbol(g: str) -> str:
    s = g.strip().strip('"').strip("'")
    s = " ".join(s.split())
    s = s.strip(",;|")
    return s


def _split_genes_loose(x: Any) -> list[str]:
    """
    Tolerant parsing for evidence_genes:
      - list/tuple/set -> unique, preserve order
      - string -> split on , ; | (fallback: whitespace) -> unique, preserve order
      - NA/empty -> []
    """
    if _is_na_scalar(x):
        return []

    if isinstance(x, (list, tuple, set)):
        parts = [str(g) for g in x]
    else:
        s = str(x).strip()
        if not s or s.lower() in _NA_TOKENS_L:
            return []
        s = s.replace(";", ",").replace("|", ",")
        if "," not in s and " " in s:
            parts = s.split()
        else:
            parts = s.split(",")

    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        g = _clean_gene_symbol(str(p))
        if not g:
            continue
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _ensure_float64_na_series(n: int) -> pd.Series:
    """Float64 NA-capable series of length n."""
    return pd.Series([pd.NA] * n, dtype="Float64")


def _get_card_param(card: SampleCard, key: str, default: Any) -> Any:
    """
    Best-effort parameter extraction from SampleCard without depending on its schema.
    Supports:
      - attribute access: card.<key>
      - dict-like via model_dump(): card.model_dump().get(key)
      - nested distill config: card.distill.<key> or card.model_dump().get("distill", {}).get(key)
    """
    # attr direct
    if hasattr(card, key):
        v = getattr(card, key)
        if v is not None:
            return v

    # nested attr (distill)
    if hasattr(card, "distill"):
        d = card.distill
        if d is not None and hasattr(d, key):
            v = getattr(d, key)
            if v is not None:
                return v

    # dict-like
    if hasattr(card, "model_dump"):
        try:
            dd = card.model_dump()
            if key in dd and dd[key] is not None:
                return dd[key]
            if isinstance(dd.get("distill", None), dict) and dd["distill"].get(key) is not None:
                return dd["distill"][key]
        except Exception:
            pass

    return default


def _perturb_genes(
    genes: list[str],
    gene_pool: np.ndarray,
    rng: np.random.Generator,
    *,
    p_drop: float,
    p_add: float,
    min_genes: int,
) -> set[str]:
    """
    Evidence perturbation:
      - dropout: remove each gene with prob p_drop
      - add/jitter: add ~p_add * |genes| random genes from global pool
    """
    if not genes:
        return set()

    g_arr = np.array(genes, dtype=object)

    # dropout
    keep_mask = rng.random(len(g_arr)) >= p_drop
    kept = g_arr[keep_mask].tolist()

    # if we dropped too much, force-keep a few (deterministic under RNG)
    if len(kept) < min_genes and len(g_arr) > 0:
        # sample additional from the originals
        need = min_genes - len(kept)
        # avoid duplicates
        remaining = [g for g in genes if g not in set(kept)]
        if remaining:
            add_back = rng.choice(
                np.array(remaining, dtype=object), size=min(need, len(remaining)), replace=False
            )
            kept.extend(add_back.tolist())

    out = set(kept)

    # jitter/add
    if p_add > 0.0 and gene_pool.size > 0:
        k_add = int(round(p_add * max(1, len(genes))))
        if k_add > 0:
            add_genes = rng.choice(gene_pool, size=min(k_add, gene_pool.size), replace=False)
            out.update([str(g) for g in add_genes.tolist()])

    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def distill_evidence(
    evidence: pd.DataFrame,
    card: SampleCard,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    A) Evidence hygiene (v1, deterministic)

    What we do in v1:
      1) Normalize EvidenceTable-like inputs into a stable, joinable table.
      2) Compute term-level survival under *evidence perturbations*:
         - gene dropout (missing evidence)
         - gene jitter/add (evidence drift)
         Survival is the fraction of perturbations where evidence stays "close enough"
         to the original (Jaccard >= threshold).
      3) Preserve survival columns if already provided; otherwise fill computed values.

    Note:
      - True patient-level LOO/jackknife requires re-running enrichment per left-out sample.
        v1 implements evidence-level stress tests that are fully reproducible from EvidenceTable.
    """
    required = {
        "term_id",
        "term_name",
        "source",
        "stat",
        "qval",
        "direction",
        "evidence_genes",
    }
    missing = sorted(required - set(evidence.columns))
    if missing:
        raise ValueError(f"distill_evidence: missing columns: {missing}")

    out = evidence.copy()

    # Minimal normalization for "EvidenceTable-like" inputs
    out["term_id"] = out["term_id"].astype(str).str.strip()
    out["term_name"] = out["term_name"].astype(str).str.strip()
    out["source"] = out["source"].astype(str).str.strip()
    out["direction"] = out["direction"].map(_normalize_direction_loose)

    out["stat"] = pd.to_numeric(out["stat"], errors="coerce")
    out["qval"] = pd.to_numeric(out["qval"], errors="coerce")

    if out["stat"].isna().any():
        i = int(out.index[out["stat"].isna()][0])
        raise ValueError(
            f"distill_evidence: non-numeric stat at row index={i}. "
            "Upstream must supply numeric stat (NES / -log10(q) / LogP)."
        )

    # Normalize evidence_genes (list[str]) and validate non-empty
    out["evidence_genes"] = out["evidence_genes"].map(_split_genes_loose)
    out["n_evidence_genes"] = out["evidence_genes"].map(len)

    empty = out["n_evidence_genes"].eq(0)
    if empty.any():
        i = int(out.index[empty][0])
        raise ValueError(
            f"distill_evidence: empty evidence_genes at row index={i}. "
            "Upstream adapter must supply overlap genes / leadingEdge."
        )

    # Optional masking (deterministic if masking is deterministic for a given seed)
    masked = apply_gene_masking(out, genes_col="evidence_genes", seed=seed)
    out = masked.masked_distilled.copy()

    # Stable IDs for joins/reports
    out = out.reset_index(drop=True)
    out["term_row_id"] = range(len(out))
    out["term_uid"] = (
        out["source"].astype(str).str.strip() + ":" + out["term_id"].astype(str).str.strip()
    )

    # TSV-friendly genes
    out["evidence_genes_str"] = out["evidence_genes"].map(lambda xs: ",".join(xs))

    # ---- v1 survival computation (evidence perturbation) ----
    # Pull parameters from card if present (best-effort), else defaults.
    n_reps = int(_get_card_param(card, "n_perturb", 64))
    p_drop = float(_get_card_param(card, "gene_dropout_p", 0.10))
    p_add = float(_get_card_param(card, "gene_jitter_p", 0.05))
    j_min = float(_get_card_param(card, "evidence_jaccard_min", 0.60))
    min_genes = int(_get_card_param(card, "min_evidence_genes", 3))

    # Clamp for safety
    n_reps = max(1, min(n_reps, 512))
    p_drop = min(max(p_drop, 0.0), 0.95)
    p_add = min(max(p_add, 0.0), 0.95)
    j_min = min(max(j_min, 0.0), 1.0)
    min_genes = max(1, min_genes)

    # Build global gene pool for jitter/add
    all_genes: list[str] = []
    for xs in out["evidence_genes"].tolist():
        all_genes.extend(xs)
    gene_pool = np.array(sorted(set(all_genes)), dtype=object)

    # Deterministic RNG
    rng = np.random.default_rng(seed if seed is not None else 0)

    term_surv: list[float] = []
    for xs in out["evidence_genes"].tolist():
        orig = set(xs)
        ok = 0
        for _ in range(n_reps):
            pert = _perturb_genes(
                xs,
                gene_pool=gene_pool,
                rng=rng,
                p_drop=p_drop,
                p_add=p_add,
                min_genes=min_genes,
            )
            if _jaccard(orig, pert) >= j_min:
                ok += 1
        term_surv.append(ok / float(n_reps))

    # Attach survival (preserve if already provided)
    n = len(out)

    if "term_survival" in out.columns:
        out["term_survival"] = pd.to_numeric(out["term_survival"], errors="coerce").astype(
            "Float64"
        )
    else:
        out["term_survival"] = pd.Series(term_surv, dtype="Float64")

    # v1: gene_survival is a per-term aggregate of evidence stability (use term_survival as proxy)
    # (True gene-level survival needs sample-level LOO or per-gene re-scoring; add later.)
    if "gene_survival" in out.columns:
        out["gene_survival"] = pd.to_numeric(out["gene_survival"], errors="coerce").astype(
            "Float64"
        )
    else:
        out["gene_survival"] = out["term_survival"].astype("Float64")

    # module_survival is computed after modules are created; keep placeholder if absent.
    if "module_survival" in out.columns:
        out["module_survival"] = pd.to_numeric(out["module_survival"], errors="coerce").astype(
            "Float64"
        )
    else:
        out["module_survival"] = _ensure_float64_na_series(n)

    # Record distill parameters for provenance/debug (minimal)
    out["distill_n_perturb"] = n_reps
    out["distill_gene_dropout_p"] = p_drop
    out["distill_gene_jitter_p"] = p_add
    out["distill_evidence_jaccard_min"] = j_min
    out["distill_min_evidence_genes"] = min_genes

    # Gates for downstream (now meaningful): keep_term can be used by select/audit
    if "keep_term" not in out.columns:
        out["keep_term"] = True
    if "keep_reason" not in out.columns:
        out["keep_reason"] = "ok"

    # Optional gate: drop clearly unstable terms early (still "hygiene", not audit)
    tau = _get_card_param(card, "tau", None)
    if tau is not None:
        try:
            tau_f = float(tau)
            # Do not delete rows here (keep artifacts stable); just mark.
            unstable = out["term_survival"].astype(float) < tau_f
            out.loc[unstable, "keep_term"] = False
            out.loc[unstable, "keep_reason"] = "low_term_survival"
        except Exception:
            pass

    return out
