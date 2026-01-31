# LLM-PathwayCurator/src/llm_pathway_curator/distill.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import _shared
from .masking import apply_gene_masking
from .sample_card import SampleCard


def _is_na_scalar(x: Any) -> bool:
    """Single source of truth: _shared.is_na_scalar."""
    return _shared.is_na_scalar(x)


def _get_distill_knob(card: SampleCard, key: str, default: Any) -> Any:
    """
    Single source of truth: SampleCard.extra only.
    Distill knobs are namespaced to avoid collisions: "distill_*".

    We also tolerate a legacy fallback (un-namespaced) if already shipped.
    """
    v = None
    if hasattr(card, "get_extra"):
        v = card.get_extra(f"distill_{key}", None)
    if v is None:
        ex = getattr(card, "extra", {}) or {}
        if isinstance(ex, dict):
            v = ex.get(f"distill_{key}", None)
            if v is None:
                v = ex.get(key, None)  # legacy fallback
    return default if v is None else v


def _normalize_direction(x: Any) -> str:
    """Single source of truth: _shared.normalize_direction()."""
    return _shared.normalize_direction(x)


def _clean_gene_token(g: str) -> str:
    """
    Single source of truth: _shared.clean_gene_token (NO forced uppercasing).
    """
    return _shared.clean_gene_token(g)


def _split_genes_loose(x: Any) -> list[str]:
    """
    Single source of truth: _shared.parse_genes().

    Rationale:
      - Align tokenization across distill/audit/modules/select.
      - Conservative split (avoid destructive whitespace splitting).
      - NO forced uppercasing.
    """
    return _shared.parse_genes(x)


def _ensure_float64_na_series(n: int) -> pd.Series:
    """Float64 NA-capable series of length n."""
    return pd.Series([pd.NA] * n, dtype="Float64")


def _ensure_int64_na_series(n: int) -> pd.Series:
    """Int64 (nullable) NA-capable series of length n."""
    return pd.Series([pd.NA] * n, dtype="Int64")


def _seed_for_term(seed: int | None, term_uid: str, term_row_id: int | None = None) -> int:
    """Single source of truth: _shared.seed_for_term()."""
    return _shared.seed_for_term(seed, term_uid, term_row_id=term_row_id)


def _hash_gene_set_short12(genes: list[str]) -> str:
    """
    Single source of truth: _shared.hash_gene_set_12hex (NO forced uppercasing).
    Kept as a thin wrapper for backward readability inside distill.py.
    """
    return _shared.hash_gene_set_12hex(list(genes or []))


def _get_distill_mode(card: SampleCard) -> str:
    """
    Explicit mode switch. Default keeps v1 semantics (evidence perturbation).

    Allowed:
      - "evidence_perturb" (default): perturb evidence genes deterministically
        (no re-running enrichment)
      - "replicates_proxy": proxy LOO-like survival using replicate_id
        (NOT true patient-level LOO enrichment)

    IMPORTANT:
      Do NOT accept aliases like "loo"/"patient_loo" here: those imply true re-run LOO enrichment.
      Keep vocabulary unambiguous for reproducibility and paper claims.
    """
    m = _get_distill_knob(card, "mode", None)
    if m is None:
        return "evidence_perturb"
    s = str(m).strip().lower()
    if s in {"evidence", "perturb", "evidence_perturb"}:
        return "evidence_perturb"
    if s in {"replicates_proxy"}:
        return "replicates_proxy"
    return "evidence_perturb"


def _get_distill_pre_gate_mode(card: SampleCard, default: str = "off") -> str:
    """
    Distill is measurement, audit is decision.
    pre_gate_mode controls whether distill marks keep_term early.

      - "off" (default): do not change keep_term based on tau
      - "note": keep_term stays True, but keep_reason annotates low survival
      - "hard": keep_term=False when term_survival_agg < tau
    """
    v = _get_distill_knob(card, "pre_gate_mode", None)
    if v is None:
        return str(default).strip().lower()

    s = str(v).strip().lower()
    if s in {"off", "disable", "disabled", "none"}:
        return "off"
    if s in {"note", "warn", "warning"}:
        return "note"
    if s in {"hard", "abstain"}:
        return "hard"
    return str(default).strip().lower()


def _compute_similarity_metrics(orig: set[str], pert: set[str]) -> tuple[float, float, float]:
    """Return (jaccard, recall, precision)."""
    if not orig and not pert:
        return (1.0, 1.0, 1.0)
    if not orig:
        prec = 0.0 if pert else 1.0
        return (0.0, 1.0, prec)
    inter = len(orig & pert)
    union = len(orig | pert)
    j = float(inter / union) if union else 0.0
    recall = float(inter / len(orig)) if orig else 1.0
    precision = float(inter / len(pert)) if pert else 0.0
    return (j, recall, precision)


def _perturb_genes(
    genes: list[str],
    gene_pool: np.ndarray,
    rng: np.random.Generator,
    *,
    p_drop: float,
    p_add: float,
    min_genes: int,
    rescue: bool,
) -> set[str]:
    """
    Evidence perturbation:
      - dropout: remove each gene with prob p_drop
      - add/jitter: add ~p_add * |genes| random genes from global pool

    min_genes:
      - always enforce a hard floor by adding back from originals when needed
    rescue:
      - retained for backward compatibility; no longer controls the hard floor
    """
    if not genes:
        return set()

    g_arr = np.array(genes, dtype=object)

    keep_mask = rng.random(len(g_arr)) >= p_drop
    kept = g_arr[keep_mask].tolist()

    # Enforce a hard floor when min_genes requests it.
    # This makes min_keep_frac/min_evidence_genes actually work even if rescue=False.
    if len(kept) < min_genes and len(g_arr) > 0:
        need = min_genes - len(kept)
        kept_set = set([str(g) for g in kept])
        remaining = [g for g in genes if str(g) not in kept_set]
        if remaining and need > 0:
            add_back = rng.choice(
                np.array(remaining, dtype=object),
                size=min(need, len(remaining)),
                replace=False,
            )
            kept.extend(add_back.tolist())

    out = set([str(g) for g in kept if str(g).strip()])

    if p_add > 0.0 and gene_pool.size > 0:
        k_add = int(round(p_add * max(1, len(genes))))
        if k_add > 0:
            orig = set([str(g) for g in genes if str(g).strip()])
            candidates = np.array(
                [g for g in gene_pool.tolist() if str(g) not in orig], dtype=object
            )
            if candidates.size > 0:
                add_genes = rng.choice(candidates, size=min(k_add, candidates.size), replace=False)
                out.update([str(g) for g in add_genes.tolist()])

    out = set([_clean_gene_token(g) for g in out if str(g).strip()])
    return out


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
    Compute proxy survival given stacked EvidenceTable with replicate_id column.

    Proxy survival definition (per term_uid, anchored to baseline replicate):
      survival = fraction of non-baseline replicates where:
        - term exists (same term_uid), and
        - direction matches baseline (optional), and
        - evidence_genes similarity to baseline passes gates (jaccard/recall/precision).

    NOTE:
      This is NOT true patient-level LOO enrichment re-run. It is a proxy
      based on replicate tables already computed upstream.
    """
    if "replicate_id" not in df.columns:
        raise ValueError("replicate_id column is required for replicates_proxy survival")
    if "term_uid" not in df.columns:
        raise ValueError("term_uid column missing (should be constructed in distill_evidence)")
    if "evidence_genes" not in df.columns:
        raise ValueError("evidence_genes column missing")
    if "direction" not in df.columns:
        raise ValueError("direction column missing")

    base = df[df["replicate_id"].astype(str) == str(baseline_id)].copy()
    if base.empty:
        raise ValueError(f"baseline replicate_id='{baseline_id}' not found in evidence table")

    # Baseline map: require 1 row per term_uid (avoid silent union bias).
    base_map_genes: dict[str, set[str]] = {}
    base_map_dir: dict[str, str] = {}

    for _, r in base.iterrows():
        tu = str(r["term_uid"]).strip()
        if not tu:
            continue

        if tu in base_map_genes:
            raise ValueError(
                "replicates_proxy baseline has duplicate term_uid rows. "
                f"term_uid={tu!r} baseline_id={baseline_id!r}. "
                "Upstream EvidenceTable should have unique (replicate_id, source, term_id)."
            )

        genes = r["evidence_genes"]
        gset = set(genes) if isinstance(genes, list) else set(_split_genes_loose(genes))
        g_clean = set([_clean_gene_token(g) for g in gset if str(g).strip()])
        base_map_genes[tu] = set(g_clean)
        base_map_dir[tu] = _normalize_direction(r.get("direction"))

    reps = sorted(set(df["replicate_id"].astype(str).tolist()))
    reps_nb = [rid for rid in reps if rid != str(baseline_id)]
    if not reps_nb:
        return pd.DataFrame(
            {
                "term_uid": list(base_map_genes.keys()),
                "term_survival_loo": [pd.NA] * len(base_map_genes),
                "loo_n_total": [0] * len(base_map_genes),
                "loo_n_ok": [0] * len(base_map_genes),
            }
        )

    # For each non-baseline replicate, build tu -> (dir, geneset).
    # Decision-grade policy: require unique term_uid per replicate.
    # (Union would bias survival upward and hide upstream duplication bugs.)
    rep_lookup: dict[str, dict[str, tuple[str, set[str]]]] = {}
    for rid in reps_nb:
        sub = df[df["replicate_id"].astype(str) == rid]
        m: dict[str, tuple[str, set[str]]] = {}
        for _, r in sub.iterrows():
            tu = str(r["term_uid"]).strip()
            if not tu:
                continue

            if tu in m:
                raise ValueError(
                    "replicates_proxy replicate has duplicate term_uid rows. "
                    f"term_uid={tu!r} replicate_id={rid!r}. "
                    "Upstream EvidenceTable should have unique (replicate_id, source, term_id)."
                )

            d = _normalize_direction(r.get("direction"))
            genes = r["evidence_genes"]
            gset = set(genes) if isinstance(genes, list) else set(_split_genes_loose(genes))
            g_clean = set([_clean_gene_token(g) for g in gset if str(g).strip()])

            m[tu] = (d, set(g_clean))

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
                continue
            d_rep, g_rep = m[tu]

            if direction_match and (base_dir in {"up", "down"}) and (d_rep in {"up", "down"}):
                if d_rep != base_dir:
                    continue
            j, recall, precision = _compute_similarity_metrics(base_genes, g_rep)
            n0 = len(base_genes)
            j_thr, r_thr, p_thr = (
                (0.30, 0.60, 0.60)
                if n0 <= 5
                else (0.45, 0.70, 0.70)
                if n0 <= 10
                else (j_min, r_min, p_min)
            )
            if (j >= j_thr) and (recall >= r_thr) and (precision >= p_thr):
                ok += 1

        surv = ok / float(total) if total > 0 else pd.NA
        rows.append(
            {"term_uid": tu, "term_survival_loo": surv, "loo_n_total": total, "loo_n_ok": ok}
        )

    tbl = pd.DataFrame(rows)

    # Explicitly mark as proxy survival to avoid "LOO" ambiguity in papers.
    # Keep legacy column name for backward compatibility; add proxy-named alias.
    if "term_survival_loo" in tbl.columns:
        tbl["term_survival_proxy"] = pd.to_numeric(
            tbl["term_survival_loo"], errors="coerce"
        ).astype("Float64")

    return tbl


def distill_evidence(
    evidence: pd.DataFrame,
    card: SampleCard,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    A) Evidence hygiene (v1, deterministic)

    v1 semantics:
      - Evidence-level stress tests reproducible from EvidenceTable (no re-running enrichment).
      - Optional replicates_proxy mode if replicate_id exists (NOT true re-run LOO enrichment).

    Contract:
      - Input must preserve term×gene (evidence_genes non-empty) for rows used downstream.
      - Output is stable/joinable and carries term_uid + survival fields.
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
    # Provenance (minimal, decision-grade reproducibility)
    out["distill_seed"] = pd.NA if seed is None else int(seed)

    # Preserve raw index for error messages / provenance
    if "raw_index" not in out.columns:
        out = out.reset_index(drop=False).rename(columns={"index": "raw_index"})

    # If schema.py provided validity flags, respect them.
    if "is_valid" in out.columns:
        out = out[out["is_valid"].astype(bool)].copy()

    # Normalize required fields
    out["term_id"] = out["term_id"].astype(str).str.strip()
    out["term_name"] = out["term_name"].astype(str).str.strip()
    out["source"] = out["source"].astype(str).str.strip()
    out.loc[out["source"].eq(""), "source"] = "unknown"
    out["direction"] = out["direction"].map(_normalize_direction)

    out["stat"] = pd.to_numeric(out["stat"], errors="coerce")
    out["qval"] = pd.to_numeric(out["qval"], errors="coerce")

    # Robust evidence_genes: prefer list-like, tolerate scalar strings
    out["evidence_genes_was_list"] = out["evidence_genes"].map(
        lambda x: isinstance(x, (list, tuple, set))
    )
    out["evidence_genes"] = out["evidence_genes"].map(_split_genes_loose)
    out["n_evidence_genes"] = out["evidence_genes"].map(len)

    # Validate core (distill should be strict by default)
    if out["stat"].isna().any():
        i = int(out.index[out["stat"].isna()][0])
        bad = out.loc[i, ["term_id", "term_name", "source", "raw_index"]].to_dict()
        raise ValueError(
            f"distill_evidence: non-numeric stat at row index={i} (row={bad}). "
            "Upstream must supply numeric stat (NES / -log10(q) / LogP)."
        )

    empty = out["n_evidence_genes"].eq(0)
    if empty.any():
        i = int(out.index[empty][0])
        bad = out.loc[i, ["term_id", "term_name", "source", "raw_index"]].to_dict()
        raise ValueError(
            f"distill_evidence: empty evidence_genes at row index={i} (row={bad}). "
            "Upstream adapter/schema must supply overlap genes / leadingEdge, or drop invalid rows."
        )

    # Optional masking
    masked = apply_gene_masking(out, genes_col="evidence_genes", seed=seed)
    out = masked.masked_distilled.copy()

    # Stable IDs for joins/reports
    out = out.reset_index(drop=True)
    out["term_row_id"] = range(len(out))
    out["term_uid"] = out.apply(
        lambda r: _shared.make_term_uid(r.get("source"), r.get("term_id")),
        axis=1,
    )

    # TSV-friendly genes (spec-level contract lives in _shared)
    out["evidence_genes_str"] = out["evidence_genes"].map(
        lambda xs: _shared.join_genes_tsv(xs) if isinstance(xs, list) else ""
    )

    # ===========================
    # Distill knobs (single read)
    # ===========================
    distill_mode = _get_distill_mode(card)

    n_reps = int(_get_distill_knob(card, "n_perturb", 64))
    p_drop = float(_get_distill_knob(card, "gene_dropout_p", 0.10))
    p_add = float(_get_distill_knob(card, "gene_jitter_p", 0.05))

    j_min = float(_get_distill_knob(card, "evidence_jaccard_min", 0.60))
    r_min = float(_get_distill_knob(card, "evidence_recall_min", 0.80))
    p_min = float(_get_distill_knob(card, "evidence_precision_min", 0.80))

    drift_hash = bool(_get_distill_knob(card, "track_hash_drift", True))

    min_keep_frac = float(_get_distill_knob(card, "min_keep_frac", 0.0))
    min_keep_frac = min(max(min_keep_frac, 0.0), 1.0)

    user_min_genes_raw = _get_distill_knob(card, "min_evidence_genes", None)
    rescue = bool(_get_distill_knob(card, "perturb_rescue", False))

    # Clamp knobs for safety / reproducibility
    n_reps = max(1, min(int(n_reps), 512))
    p_drop = min(max(float(p_drop), 0.0), 0.95)
    p_add = min(max(float(p_add), 0.0), 0.95)
    j_min = min(max(float(j_min), 0.0), 1.0)
    r_min = min(max(float(r_min), 0.0), 1.0)
    p_min = min(max(float(p_min), 0.0), 1.0)

    # Always export knob provenance (report/audit can cite this)
    out["distill_semantics"] = distill_mode
    out["distill_track_hash_drift"] = bool(drift_hash)

    # ===========================
    # REPLICATES PROXY
    # ===========================
    if distill_mode == "replicates_proxy":
        if "replicate_id" not in out.columns:
            raise ValueError("distill_mode=replicates_proxy requires replicate_id column")

        baseline_id = str(_get_distill_knob(card, "loo_baseline_id", "full")).strip() or "full"
        direction_match = bool(_get_distill_knob(card, "loo_direction_match", True))

        loo_tbl = _loo_survival_from_replicates(
            out,
            baseline_id=baseline_id,
            direction_match=direction_match,
            j_min=j_min,
            r_min=r_min,
            p_min=p_min,
        )

        out = out.merge(loo_tbl, on="term_uid", how="left")
        out = out[out["replicate_id"].astype(str) == baseline_id].copy()
        out = out.reset_index(drop=True)

        trust_input_survival = bool(_get_distill_knob(card, "trust_input_survival", False))
        out["distill_trust_input_survival"] = bool(trust_input_survival)

        out["term_survival_input"] = (
            pd.to_numeric(out.get("term_survival", pd.NA), errors="coerce").astype("Float64")
            if "term_survival" in out.columns
            else _ensure_float64_na_series(len(out))
        )
        out["term_survival_computed"] = pd.to_numeric(
            out["term_survival_loo"], errors="coerce"
        ).astype("Float64")

        out["term_survival_source"] = "computed"
        if trust_input_survival and "term_survival" in out.columns:
            out["term_survival_source"] = "input"

        out["term_survival"] = (
            out["term_survival_input"] if trust_input_survival else out["term_survival_computed"]
        ).astype("Float64")

        out["term_gene_set_hash"] = out["evidence_genes"].map(_hash_gene_set_short12)
        out["term_survival_agg"] = out["term_survival"].astype("Float64")
        out["term_hash_drift_rate"] = _ensure_float64_na_series(len(out))

        # convenience fields (downstream expects these columns)
        out["gene_survival"] = out["term_survival"].astype("Float64")
        # out["module_survival"] = _ensure_float64_na_series(len(out))

        out["distill_loo_is_proxy"] = True
        out["distill_loo_baseline_id"] = baseline_id
        out["distill_loo_direction_match"] = bool(direction_match)

        # use nullable ints/floats to avoid downstream type traps
        out["distill_n_perturb"] = _ensure_int64_na_series(len(out))
        out["distill_gene_dropout_p"] = _ensure_float64_na_series(len(out))
        out["distill_gene_jitter_p"] = _ensure_float64_na_series(len(out))

        out["distill_evidence_jaccard_min"] = float(j_min)
        out["distill_evidence_recall_min"] = float(r_min)
        out["distill_evidence_precision_min"] = float(p_min)

    else:
        # ===========================
        # EVIDENCE-LEVEL PERTURBATION
        # ===========================
        # Build pool from MASKED evidence genes (post apply_gene_masking).
        # Normalize tokens via spec-level cleaning to keep jitter pool consistent.
        all_genes: list[str] = []
        for xs in out["evidence_genes"].tolist():
            all_genes.extend([_clean_gene_token(g) for g in xs if str(g).strip()])

        gene_pool = np.array(sorted({g for g in all_genes if g and str(g).strip()}), dtype=object)
        out["distill_gene_pool_n"] = int(gene_pool.size)
        out["distill_gene_pool_is_masked"] = True

        user_min_genes: int | None = None
        if user_min_genes_raw is not None:
            try:
                user_min_genes = max(1, int(user_min_genes_raw))
            except Exception:
                user_min_genes = None

        term_surv: list[float] = []
        mean_j: list[float] = []
        mean_r: list[float] = []
        mean_p: list[float] = []
        n_ok_list: list[int] = []
        n_total_list: list[int] = []
        base_hashes: list[str] = []
        drift_rate_list: list[float] = []
        min_genes_eff_list: list[int] = []

        # Avoid zip(strict=...) portability concerns; iterate by index explicitly.
        for i in range(len(out)):
            tu = str(out.loc[i, "term_uid"])
            rid = int(out.loc[i, "term_row_id"])
            xs = out.loc[i, "evidence_genes"]

            orig = set([_clean_gene_token(g) for g in xs if str(g).strip()])

            base_hash = _hash_gene_set_short12(xs)
            base_hashes.append(base_hash)

            drift_n = 0

            if user_min_genes is None:
                min_genes_eff = max(1, int(np.ceil(min_keep_frac * max(1, len(xs)))))
            else:
                min_genes_eff = user_min_genes
            min_genes_eff_list.append(int(min_genes_eff))
            rng_term = np.random.default_rng(_seed_for_term(seed, tu, term_row_id=rid))

            ok = 0
            js: list[float] = []
            rs: list[float] = []
            ps: list[float] = []

            n0 = len(orig)
            j_thr, r_thr, p_thr = (
                (0.30, 0.60, 0.60)
                if n0 <= 5
                else (0.45, 0.70, 0.70)
                if n0 <= 10
                else (j_min, r_min, p_min)
            )

            for _ in range(n_reps):
                pert = _perturb_genes(
                    xs,
                    gene_pool=gene_pool,
                    rng=rng_term,
                    p_drop=p_drop,
                    p_add=p_add,
                    min_genes=min_genes_eff,
                    rescue=rescue,
                )

                if drift_hash:
                    pert_hash = _hash_gene_set_short12(list(pert))
                    if pert_hash != base_hash:
                        drift_n += 1

                j, rec, prec = _compute_similarity_metrics(orig, pert)
                js.append(j)
                rs.append(rec)
                ps.append(prec)
                if (j >= j_thr) and (rec >= r_thr) and (prec >= p_thr):
                    ok += 1

            term_surv.append(ok / float(n_reps))
            drift_rate_list.append(drift_n / float(n_reps) if n_reps > 0 else float("nan"))
            mean_j.append(float(np.mean(js)) if js else float("nan"))
            mean_r.append(float(np.mean(rs)) if rs else float("nan"))
            mean_p.append(float(np.mean(ps)) if ps else float("nan"))
            n_ok_list.append(int(ok))
            n_total_list.append(int(n_reps))

        trust_input_survival = bool(_get_distill_knob(card, "trust_input_survival", False))
        out["distill_trust_input_survival"] = bool(trust_input_survival)
        out["distill_min_genes_effective_per_term"] = pd.Series(min_genes_eff_list, dtype="Int64")

        if "term_survival" in out.columns:
            out["term_survival_input"] = pd.to_numeric(
                out["term_survival"], errors="coerce"
            ).astype("Float64")
        else:
            out["term_survival_input"] = _ensure_float64_na_series(len(out))

        out["term_survival_computed"] = pd.Series(term_surv, dtype="Float64")
        out["term_survival_n_ok"] = pd.Series(n_ok_list, dtype="Int64")
        out["term_survival_n_total"] = pd.Series(n_total_list, dtype="Int64")

        out["term_survival_source"] = "computed"
        if trust_input_survival and "term_survival" in out.columns:
            out["term_survival_source"] = "input"

        out["term_survival"] = (
            out["term_survival_input"] if trust_input_survival else out["term_survival_computed"]
        ).astype("Float64")

        out["term_gene_set_hash"] = pd.Series(base_hashes, dtype="string")
        out["term_survival_agg"] = out["term_survival"].astype("Float64")

        if drift_hash:
            out["term_hash_drift_rate"] = pd.Series(drift_rate_list, dtype="Float64")
        else:
            out["term_hash_drift_rate"] = _ensure_float64_na_series(len(out))

        out["distill_loo_is_proxy"] = False
        out["distill_n_perturb"] = pd.Series([n_reps] * len(out), dtype="Int64")
        out["distill_gene_dropout_p"] = pd.Series([p_drop] * len(out), dtype="Float64")
        out["distill_gene_jitter_p"] = pd.Series([p_add] * len(out), dtype="Float64")
        out["distill_evidence_jaccard_min"] = float(j_min)
        out["distill_evidence_recall_min"] = float(r_min)
        out["distill_evidence_precision_min"] = float(p_min)
        out["distill_min_keep_frac"] = float(min_keep_frac)
        out["distill_min_evidence_genes_effective"] = (
            pd.NA if user_min_genes is None else int(user_min_genes)
        )
        out["distill_perturb_rescue"] = bool(rescue)

        out["term_survival_mean_jaccard"] = pd.Series(mean_j, dtype="Float64")
        out["term_survival_mean_recall"] = pd.Series(mean_r, dtype="Float64")
        out["term_survival_mean_precision"] = pd.Series(mean_p, dtype="Float64")

        # convenience fields (downstream expects these columns)
        out["gene_survival"] = out["term_survival"].astype("Float64")
        # out["module_survival"] = _ensure_float64_na_series(len(out))

    # Gates for downstream
    if "keep_term" not in out.columns:
        out["keep_term"] = True
    if "keep_reason" not in out.columns:
        out["keep_reason"] = "ok"

    # Optional pre-gate (explicit): distill measures; audit decides.
    pre_gate_mode = _get_distill_pre_gate_mode(card, default="off")

    # IMPORTANT:
    #  - do NOT reuse audit_tau here.
    #  - pre-gate is a convenience filter, so we namespace its threshold explicitly.
    pre_gate_tau = _get_distill_knob(card, "pre_gate_tau", None)
    if pre_gate_tau is None:
        # legacy fallback: tolerate distill_tau / tau if previously shipped
        pre_gate_tau = _get_distill_knob(card, "tau", None)

    if pre_gate_tau is not None:
        try:
            tau_f = float(pre_gate_tau)
            surv = pd.to_numeric(out["term_survival_agg"], errors="coerce")
            unstable = surv < tau_f

            if pre_gate_mode == "hard":
                out.loc[unstable, "keep_term"] = False
                out.loc[unstable, "keep_reason"] = "low_term_survival_agg"
            elif pre_gate_mode == "note":
                out.loc[unstable, "keep_reason"] = "low_term_survival_agg(note)"
        except Exception:
            pass

    # Export pre-gate provenance (distill convenience; audit decides)
    out["distill_pre_gate_mode"] = str(pre_gate_mode)
    out["distill_pre_gate_tau"] = pd.NA if pre_gate_tau is None else float(pre_gate_tau)

    return out
