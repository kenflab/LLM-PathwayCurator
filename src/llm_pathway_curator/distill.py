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

    # jitter/add: add genes NOT already in the original set, to actually model drift
    if p_add > 0.0 and gene_pool.size > 0:
        k_add = int(round(p_add * max(1, len(genes))))
        if k_add > 0:
            orig = set([str(g) for g in genes if str(g).strip()])
            # exclude orig genes from add candidates
            candidates = np.array(
                [g for g in gene_pool.tolist() if str(g) not in orig], dtype=object
            )
            if candidates.size > 0:
                add_genes = rng.choice(candidates, size=min(k_add, candidates.size), replace=False)
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


# --- add near top (helpers) ---
def _compute_similarity_metrics(orig: set[str], pert: set[str]) -> tuple[float, float, float]:
    """Return (jaccard, recall, precision)."""
    if not orig and not pert:
        return (1.0, 1.0, 1.0)
    if not orig:
        # no original evidence -> treat as fully recalled but precision depends
        prec = 0.0 if pert else 1.0
        return (0.0, 1.0, prec)
    inter = len(orig & pert)
    union = len(orig | pert)
    j = float(inter / union) if union else 0.0
    recall = float(inter / len(orig)) if orig else 1.0
    precision = float(inter / len(pert)) if pert else 0.0
    return (j, recall, precision)


def _loo_survival_from_replicates(
    df: pd.DataFrame,
    *,
    baseline_id: str,
    direction_match: bool,
    j_min: float,
    r_min: float,
    p_min: float,
) -> pd.DataFrame:
    """
    Compute patient-level survival given stacked EvidenceTable with replicate_id column.

    Survival definition (per term_uid, anchored to baseline replicate):
      survival = fraction of non-baseline replicates where:
        - term exists (same term_uid), and
        - direction matches baseline (optional), and
        - evidence_genes similarity to baseline passes gates (jaccard/recall/precision).

    Returns a per-term_uid summary table with:
      term_uid, term_survival_loo, n_replicates, n_ok, n_total
    """
    if "replicate_id" not in df.columns:
        raise ValueError("replicate_id column is required for LOO/jackknife survival")

    # baseline rows
    base = df[df["replicate_id"].astype(str) == str(baseline_id)].copy()
    if base.empty:
        raise ValueError(f"baseline replicate_id='{baseline_id}' not found in evidence table")

    # ensure needed cols exist
    if "term_uid" not in df.columns:
        raise ValueError("term_uid column missing (should be constructed in distill_evidence)")
    if "evidence_genes" not in df.columns:
        raise ValueError("evidence_genes column missing")
    if "direction" not in df.columns:
        raise ValueError("direction column missing")

    # build baseline maps
    base_map_genes: dict[str, set[str]] = {}
    base_map_dir: dict[str, str] = {}

    for _, r in base.iterrows():
        tu = str(r["term_uid"]).strip()
        if not tu:
            continue

        genes = r["evidence_genes"]
        gset = set(genes) if isinstance(genes, list) else set(_split_genes_loose(genes))
        g_clean = set([_clean_gene_symbol(g) for g in gset if str(g).strip()])

        base_map_genes.setdefault(tu, set()).update(g_clean)

        if tu not in base_map_dir:
            base_map_dir[tu] = _normalize_direction_loose(r.get("direction"))

    # list of non-baseline replicate IDs
    reps = sorted(set(df["replicate_id"].astype(str).tolist()))
    reps_nb = [rid for rid in reps if rid != str(baseline_id)]
    if not reps_nb:
        # no LOO folds provided
        out = pd.DataFrame(
            {
                "term_uid": list(base_map_genes.keys()),
                "term_survival_loo": [pd.NA] * len(base_map_genes),
                "loo_n_total": [0] * len(base_map_genes),
                "loo_n_ok": [0] * len(base_map_genes),
            }
        )
        return out

    # For each replicate, we need quick lookup by term_uid
    # Build dict: replicate_id -> {term_uid -> (dir, gene_set)}
    rep_lookup: dict[str, dict[str, tuple[str, set[str]]]] = {}
    for rid in reps_nb:
        sub = df[df["replicate_id"].astype(str) == rid]
        m: dict[str, tuple[str, set[str]]] = {}
        for _, r in sub.iterrows():
            tu = str(r["term_uid"]).strip()
            if not tu:
                continue
            d = _normalize_direction_loose(r.get("direction"))
            genes = r["evidence_genes"]
            gset = set(genes) if isinstance(genes, list) else set(_split_genes_loose(genes))
            m[tu] = (d, set([_clean_gene_symbol(g) for g in gset if str(g).strip()]))
        rep_lookup[rid] = m

    rows = []
    for tu, base_genes in base_map_genes.items():
        base_dir = base_map_dir.get(tu, "na")
        ok = 0
        total = 0

        for rid in reps_nb:
            total += 1
            m = rep_lookup.get(rid, {})
            if tu not in m:
                continue  # term disappears in this fold -> fail this fold
            d_rep, g_rep = m[tu]

            if direction_match and (base_dir in {"up", "down"}) and (d_rep in {"up", "down"}):
                if d_rep != base_dir:
                    continue

            j, recall, prec = _compute_similarity_metrics(base_genes, g_rep)
            if (j >= j_min) and (recall >= r_min) and (prec >= p_min):
                ok += 1

        surv = ok / float(total) if total > 0 else pd.NA
        rows.append(
            {"term_uid": tu, "term_survival_loo": surv, "loo_n_total": total, "loo_n_ok": ok}
        )

    return pd.DataFrame(rows)


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
    r_min = float(_get_card_param(card, "evidence_recall_min", 0.80))
    p_min = float(_get_card_param(card, "evidence_precision_min", 0.80))

    # default min_genes: make it less “helpful”
    min_genes = int(_get_card_param(card, "min_evidence_genes", 1))

    # Clamp for safety
    n_reps = max(1, min(n_reps, 512))
    p_drop = min(max(p_drop, 0.0), 0.95)
    p_add = min(max(p_add, 0.0), 0.95)
    j_min = min(max(j_min, 0.0), 1.0)
    r_min = min(max(r_min, 0.0), 1.0)
    p_min = min(max(p_min, 0.0), 1.0)
    min_genes = max(1, min_genes)

    # ===========================
    # PATIENT-LEVEL LOO/JACKKNIFE (NEW)
    # ===========================
    if "replicate_id" in out.columns:
        baseline_id = str(_get_card_param(card, "loo_baseline_id", "full")).strip() or "full"
        direction_match = bool(_get_card_param(card, "loo_direction_match", True))

        loo_tbl = _loo_survival_from_replicates(
            out,
            baseline_id=baseline_id,
            direction_match=direction_match,
            j_min=j_min,
            r_min=r_min,
            p_min=p_min,
        )

        # Attach per-term_uid survival to all rows
        out = out.merge(loo_tbl, on="term_uid", how="left")

        # If multiple replicates are stacked, keep the baseline replicate rows
        # as the canonical output rows.
        out = out[out["replicate_id"].astype(str) == baseline_id].copy()
        out = out.reset_index(drop=True)

        # Patient-level survival becomes primary term_survival
        out["term_survival"] = pd.to_numeric(out["term_survival_loo"], errors="coerce").astype(
            "Float64"
        )

        # v1 compatibility: keep these columns present even in patient_loo mode
        # gene_survival: proxy == term_survival (same as evidence-perturb branch)
        out["gene_survival"] = out["term_survival"].astype("Float64")

        # module_survival is computed after modules; keep NA placeholder
        out["module_survival"] = _ensure_float64_na_series(len(out))

        # provenance (minimal, stable keys)
        out["distill_mode"] = "patient_loo"
        out["distill_loo_baseline_id"] = baseline_id
        out["distill_loo_direction_match"] = bool(direction_match)

        # reuse the same thresholds as the evidence-perturb branch (so Fig2 metadata is comparable)
        out["distill_evidence_jaccard_min"] = j_min
        out["distill_evidence_recall_min"] = r_min
        out["distill_evidence_precision_min"] = p_min

        # also record min genes (even though LOO does not perturb)
        out["distill_min_evidence_genes"] = min_genes

    else:
        # ===========================
        # EVIDENCE-LEVEL PERTURBATION (existing)
        # ===========================

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

                # keep the original jaccard gate
                j = _jaccard(orig, pert)

                # NEW: also require recall/precision to prevent “too-easy” passes
                inter = len(orig & pert)
                recall = (inter / len(orig)) if orig else 1.0
                precision = (inter / len(pert)) if pert else 0.0

                if (j >= j_min) and (recall >= r_min) and (precision >= p_min):
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

        # v1: gene_survival is a per-term aggregate of evidence stability
        # (use term_survival as proxy)
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
        out["distill_evidence_recall_min"] = r_min
        out["distill_evidence_precision_min"] = p_min
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
