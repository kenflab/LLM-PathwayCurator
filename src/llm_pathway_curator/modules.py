# LLM-PathwayCurator/src/llm_pathway_curator/modules.py
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from . import _shared


@dataclass(frozen=True)
class ModuleOutputs:
    """
    Container for module factorization outputs.

    Attributes
    ----------
    modules_df
        Per-module summary table. One row per module_id. Contains stable hashes
        (terms/genes/content) and representative genes, plus optional survival
        fields if computed upstream.
    term_modules_df
        Term-to-module assignment table. Contract: one module_id per term_uid.
    edges_df
        Filtered term-by-gene edge table used for module construction.
        Columns: term_uid, gene_id, weight. Additional debug/provenance lives in
        ``edges_df.attrs``.
    """

    modules_df: pd.DataFrame
    term_modules_df: pd.DataFrame
    edges_df: pd.DataFrame


# -------------------------
# Normalization (align with schema/distill: trim-only; NO forced uppercasing)
# -------------------------
def _clean_gene_id(g: object) -> str:
    """
    Clean a gene token using the project-wide canonical normalizer.

    Parameters
    ----------
    g
        Raw gene token (string-like, numeric, NA, etc.).

    Returns
    -------
    str
        Canonical gene token after trim-only cleaning. Returns an empty string
        for invalid/NA-like inputs.

    Notes
    -----
    This is a thin wrapper around ``_shared.clean_gene_token`` to keep
    modules.py readable.
    """
    return _shared.clean_gene_token(g)


def _norm_gene_id(g: object) -> str:
    """
    Normalize gene tokens for graph/module construction.

    Parameters
    ----------
    g
        Raw gene token.

    Returns
    -------
    str
        Canonical gene token (trim-only). Returns an empty string for NA/invalid.

    Notes
    -----
    - This function intentionally does NOT force uppercasing.
    - Case conventions can be species-dependent (e.g., mouse genes) and should
      be handled upstream if required.
    """
    return _clean_gene_id(g)


def _hash_set_short12(items: list[str]) -> str:
    """
    Compute a deterministic short hash for an unordered set of strings.

    Parameters
    ----------
    items
        List of string tokens. Order is ignored by the hash.

    Returns
    -------
    str
        12-hex-character stable hash.

    Notes
    -----
    This uses the spec-owned implementation ``_shared.hash_set_12hex``.
    """
    return _shared.hash_set_12hex(items)


def _hash_gene_set_short12(genes: list[str]) -> str:
    """
    Compute a deterministic short hash for a gene set.

    Parameters
    ----------
    genes
        List of gene tokens. Order is ignored.

    Returns
    -------
    str
        12-hex-character stable hash.

    Notes
    -----
    - Genes are normalized with ``_norm_gene_id`` before hashing.
    - Hashing is delegated to ``_shared.hash_gene_set_12hex``.
    """
    return _shared.hash_gene_set_12hex([_norm_gene_id(x) for x in (genes or [])])


def _module_hash_content12(terms: list[str], genes: list[str]) -> str:
    """
    Compute a deterministic short hash for module content (terms + genes).

    Parameters
    ----------
    terms
        Term identifiers included in the module.
    genes
        Gene identifiers included in the module.

    Returns
    -------
    str
        12-hex-character content hash.

    Notes
    -----
    - Terms are trim-normalized.
    - Genes are normalized with ``_norm_gene_id``.
    - Hashing is delegated to ``_shared.module_hash_content12``.
    """
    return _shared.module_hash_content12(
        [str(x).strip() for x in (terms or [])],
        [_norm_gene_id(x) for x in (genes or [])],
    )


# -------------------------
# Edges: term x gene (bipartite)
# -------------------------
def build_term_gene_edges(
    evidence_df: pd.DataFrame,
    *,
    term_id_col: str = "term_uid",
    genes_col: str = "evidence_genes",
) -> pd.DataFrame:
    """
    Build term-by-gene bipartite edges from an evidence table.

    Parameters
    ----------
    evidence_df
        Evidence table containing at least a term identifier column and a gene
        evidence column.
    term_id_col
        Column name for the term identifier in ``evidence_df``.
    genes_col
        Column name for evidence genes in ``evidence_df``. Values can be
        list-like (preferred) or legacy scalar strings.

    Returns
    -------
    pandas.DataFrame
        Edge table with columns:
        - term_uid : str
        - gene_id  : str
        - weight   : float

        The returned DataFrame also stores a small provenance dict under
        ``out.attrs["edges"]``.

    Raises
    ------
    ValueError
        If required columns are missing.

    Notes
    -----
    - Empty/invalid gene lists produce no edges and are dropped.
    - List-like gene inputs are processed via vectorized explode.
    - Scalar/string inputs are parsed via ``_shared.parse_genes``.
    - Duplicate (term_uid, gene_id) edges are summed into a single row with
      weight equal to the multiplicity.
    """
    if term_id_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {term_id_col} (hint: run distill first to add term_uid)")
    if genes_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {genes_col}")

    df = evidence_df[[term_id_col, genes_col]].copy()
    df[term_id_col] = df[term_id_col].astype(str).str.strip()
    df = df[df[term_id_col].ne("")].copy()
    if df.empty:
        out = pd.DataFrame(columns=["term_uid", "gene_id", "weight"])
        out.attrs["edges"] = {"term_id_col": term_id_col, "genes_col": genes_col, "n_edges": 0}
        return out

    def _normalize_list_like(xs: object) -> list[str]:
        # Fast path: list/tuple/set is already tokenized upstream (schema/distill).
        if isinstance(xs, (list, tuple, set)):
            genes = [_norm_gene_id(g) for g in xs if str(g).strip()]
            genes = [g for g in genes if g]
            return _shared.dedup_preserve_order(genes)

        # Slow path: scalar/string legacy inputs
        genes = _shared.parse_genes(xs)
        genes = [_norm_gene_id(g) for g in genes if str(g).strip()]
        genes = [g for g in genes if g]
        return _shared.dedup_preserve_order(genes)

    df["_genes_list"] = df[genes_col].map(_normalize_list_like)
    df = df[df["_genes_list"].map(len).gt(0)].copy()
    if df.empty:
        out = pd.DataFrame(columns=["term_uid", "gene_id", "weight"])
        out.attrs["edges"] = {"term_id_col": term_id_col, "genes_col": genes_col, "n_edges": 0}
        return out

    edges = df[[term_id_col, "_genes_list"]].explode("_genes_list", ignore_index=True)

    edges = edges.rename(columns={term_id_col: "term_uid", "_genes_list": "gene_id"})
    edges["gene_id"] = edges["gene_id"].astype(str).map(_norm_gene_id)
    edges = edges[edges["gene_id"].ne("")].copy()
    # Term UID hygiene (spec-level): ensure trimming is applied AFTER explode/rename too.
    edges["term_uid"] = edges["term_uid"].astype(str).str.strip()
    edges = edges[edges["term_uid"].ne("")].copy()
    if edges.empty:
        out = pd.DataFrame(columns=["term_uid", "gene_id", "weight"])
        out.attrs["edges"] = {"term_id_col": term_id_col, "genes_col": genes_col, "n_edges": 0}
        return out

    edges["weight"] = 1.0

    edges = (
        edges.groupby(["term_uid", "gene_id"], as_index=False)["weight"]
        .sum()
        .sort_values(["term_uid", "gene_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    edges.attrs["edges"] = {
        "term_id_col": term_id_col,
        "genes_col": genes_col,
        "n_terms": int(edges["term_uid"].nunique()) if not edges.empty else 0,
        "n_genes": int(edges["gene_id"].nunique()) if not edges.empty else 0,
        "n_edges": int(len(edges)),
    }
    return edges


def filter_hub_genes(
    edges: pd.DataFrame,
    *,
    max_gene_term_degree: int | None = 200,
    # backward-compat alias (deprecated)
    max_term_degree: int | None = None,
) -> pd.DataFrame:
    """
    Remove hub genes that connect too many terms (high gene term-degree).

    Parameters
    ----------
    edges
        Edge table with columns ``term_uid`` and ``gene_id``.
    max_gene_term_degree
        Hub threshold. Genes with term-degree strictly greater than this value
        are removed. If None, no hub filtering is applied.
    max_term_degree
        Deprecated alias for ``max_gene_term_degree``. If provided and
        ``max_gene_term_degree`` is None, it is used as the threshold.

    Returns
    -------
    pandas.DataFrame
        Filtered edge table. Hub filter metadata is recorded in
        ``out.attrs["hub_filter"]``.

    Raises
    ------
    ValueError
        If ``edges`` does not have the required columns.

    Notes
    -----
    The filter uses a strict condition: degree > threshold (not >=).
    """
    if max_gene_term_degree is None and max_term_degree is not None:
        max_gene_term_degree = max_term_degree

    if edges.empty or max_gene_term_degree is None:
        out = edges.reset_index(drop=True)
        out.attrs.setdefault(
            "hub_filter", {"max_gene_term_degree": max_gene_term_degree, "n_hubs": 0}
        )
        return out

    if "gene_id" not in edges.columns or "term_uid" not in edges.columns:
        raise ValueError("filter_hub_genes: edges must have columns term_uid, gene_id")

    deg = edges.groupby("gene_id")["term_uid"].nunique()
    hubs = sorted(deg[deg > int(max_gene_term_degree)].index.astype(str).tolist())
    if not hubs:
        out = edges.reset_index(drop=True)
        out.attrs["hub_filter"] = {"max_gene_term_degree": int(max_gene_term_degree), "n_hubs": 0}
        return out

    out = edges[~edges["gene_id"].isin(set(hubs))].reset_index(drop=True)
    out.attrs["hub_filter"] = {
        "max_gene_term_degree": int(max_gene_term_degree),
        "n_hubs": len(hubs),
        "hubs": hubs[:200],
    }
    return out


# -------------------------
# Connected components on bipartite graph (legacy)
# -------------------------
def _connected_components_from_bipartite_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Compute connected components on a bipartite term-gene graph.

    Parameters
    ----------
    edges
        Edge table with columns ``term_uid`` and ``gene_id``.

    Returns
    -------
    pandas.DataFrame
        Node table with columns:
        - node      : str  (prefixed with "T:" or "G:")
        - kind      : {"term", "gene"}
        - term_uid  : str or None
        - gene_id   : str or None
        - component : int

    Notes
    -----
    This is a legacy component finder used as a fallback when term-term
    pairwise construction would be too expensive.
    """
    if edges.empty:
        return pd.DataFrame(columns=["node", "kind", "term_uid", "gene_id", "component"])

    adj: dict[str, set[str]] = {}

    def _add_edge(a: str, b: str) -> None:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for t, g in edges[["term_uid", "gene_id"]].itertuples(index=False):
        _add_edge(f"T:{t}", f"G:{g}")

    seen: set[str] = set()
    comp_map: dict[str, int] = {}
    comp = 0

    for start in sorted(adj.keys()):
        if start in seen:
            continue
        q: deque[str] = deque([start])
        seen.add(start)
        comp_map[start] = comp

        while q:
            cur = q.popleft()
            for nb in sorted(adj.get(cur, ())):
                if nb in seen:
                    continue
                seen.add(nb)
                comp_map[nb] = comp
                q.append(nb)

        comp += 1

    nodes: list[tuple[str, str, str | None, str | None, int]] = []
    for node, cid in comp_map.items():
        if node.startswith("T:"):
            nodes.append((node, "term", node[2:], None, cid))
        else:
            nodes.append((node, "gene", None, node[2:], cid))

    out = pd.DataFrame(nodes, columns=["node", "kind", "term_uid", "gene_id", "component"])
    kind_order = pd.Categorical(out["kind"], categories=["term", "gene"], ordered=True)
    out = out.assign(kind=kind_order).sort_values(
        ["component", "kind", "term_uid", "gene_id"], kind="mergesort"
    )
    return out.reset_index(drop=True)


# -------------------------
# Term-term graph by shared genes
# -------------------------
def _estimate_pair_stats(
    term_to_genes: dict[str, set[str]],
    *,
    pair_sample_max: int = 200000,
    seed: int = 0,
) -> dict[str, object]:
    """
    Estimate pairwise sparsity/connectivity statistics for term-term graphs.

    Parameters
    ----------
    term_to_genes
        Mapping from term identifier to a set of genes.
    pair_sample_max
        Maximum number of term pairs to sample. If the total number of pairs is
        below this cap, all pairs are evaluated deterministically.
    seed
        Random seed used when sampling a subset of pairs.

    Returns
    -------
    dict
        Summary statistics for pairwise overlap among sampled pairs. Keys include
        ``n_terms``, ``n_pairs_total``, ``n_pairs_sampled``, and overlap
        percentiles for shared-gene counts and Jaccard similarity.

    Notes
    -----
    The returned stats are intended for tuning heuristics and debugging, not for
    scientific claims.
    """
    terms = sorted(term_to_genes.keys())
    n_terms = len(terms)
    n_pairs_total = n_terms * (n_terms - 1) // 2

    if n_terms < 2 or n_pairs_total == 0:
        return {
            "n_terms": int(n_terms),
            "n_pairs_total": int(n_pairs_total),
            "n_pairs_sampled": 0,
            "shared_pos_rate": 0.0,
        }

    rng = random.Random(int(seed))
    max_s = int(pair_sample_max) if pair_sample_max is not None else 0
    n_pairs_sampled = min(int(n_pairs_total), max_s) if max_s > 0 else int(n_pairs_total)

    shared_vals: list[int] = []
    jac_vals: list[float] = []
    shared_pos = 0

    # If sampling all pairs is feasible, do it deterministically.
    do_all = n_pairs_sampled == int(n_pairs_total)

    if do_all:
        for i, a in enumerate(terms):
            ga = term_to_genes.get(a, set())
            for b in terms[i + 1 :]:
                gb = term_to_genes.get(b, set())

                if not ga or not gb:
                    inter = 0
                    union = len(ga | gb) if (ga or gb) else 0
                else:
                    inter = len(ga & gb)
                    union = len(ga | gb)

                if inter > 0:
                    shared_pos += 1
                jac = float(inter / union) if union else 0.0

                shared_vals.append(int(inter))
                jac_vals.append(float(jac))
    else:
        seen: set[tuple[int, int]] = set()
        while len(seen) < n_pairs_sampled:
            i = rng.randrange(0, n_terms - 1)
            j = rng.randrange(i + 1, n_terms)
            seen.add((i, j))

        for i, j in seen:
            a = terms[i]
            b = terms[j]
            ga = term_to_genes.get(a, set())
            gb = term_to_genes.get(b, set())
            if not ga or not gb:
                inter = 0
                union = len(ga | gb) if (ga or gb) else 0
            else:
                inter = len(ga & gb)
                union = len(ga | gb)
            if inter > 0:
                shared_pos += 1
            jac = float(inter / union) if union else 0.0
            shared_vals.append(int(inter))
            jac_vals.append(float(jac))

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        if len(ys) == 1:
            return float(ys[0])
        k = (len(ys) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(ys) - 1)
        if f == c:
            return float(ys[f])
        d0 = ys[f] * (c - k)
        d1 = ys[c] * (k - f)
        return float(d0 + d1)

    shared_pos_rate = float(shared_pos / len(shared_vals)) if shared_vals else 0.0

    return {
        "n_terms": int(n_terms),
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_sampled": int(len(shared_vals)),
        "shared_pos_rate": float(shared_pos_rate),
        "shared_p50": float(_pct([float(x) for x in shared_vals], 50)),
        "shared_p75": float(_pct([float(x) for x in shared_vals], 75)),
        "shared_p90": float(_pct([float(x) for x in shared_vals], 90)),
        "shared_p95": float(_pct([float(x) for x in shared_vals], 95)),
        "shared_p99": float(_pct([float(x) for x in shared_vals], 99)),
        "jaccard_p50": float(_pct(jac_vals, 50)),
        "jaccard_p75": float(_pct(jac_vals, 75)),
        "jaccard_p90": float(_pct(jac_vals, 90)),
        "jaccard_p95": float(_pct(jac_vals, 95)),
        "jaccard_p99": float(_pct(jac_vals, 99)),
    }


def _build_term_gene_sets(edges: pd.DataFrame) -> dict[str, set[str]]:
    """
    Convert an edge table into a term-to-genes mapping.

    Parameters
    ----------
    edges
        Edge table with columns ``term_uid`` and ``gene_id``.

    Returns
    -------
    dict[str, set[str]]
        Mapping from term_uid to a set of gene_id values.
    """
    term_to_genes: dict[str, set[str]] = {}
    if edges.empty:
        return term_to_genes
    for t, g in edges[["term_uid", "gene_id"]].itertuples(index=False):
        term_to_genes.setdefault(str(t), set()).add(str(g))
    return term_to_genes


def _term_term_components_shared_genes(
    term_to_genes: dict[str, set[str]],
    *,
    min_shared_genes: int = 2,
    jaccard_min: float = 0.0,
) -> dict[str, int]:
    """
    Compute connected components on a term-term graph defined by shared genes.

    Parameters
    ----------
    term_to_genes
        Mapping from term identifier to a set of genes.
    min_shared_genes
        Minimum number of shared genes required to create an edge.
    jaccard_min
        Minimum Jaccard similarity required to create an edge.

    Returns
    -------
    dict[str, int]
        Mapping from term identifier to component id.

    Notes
    -----
    This is O(n_terms^2) in the number of terms and should be guarded by a
    max-terms threshold in the caller.
    """
    terms = sorted(term_to_genes.keys())
    if not terms:
        return {}

    adj: dict[str, list[str]] = {t: [] for t in terms}

    for i, a in enumerate(terms):
        ga = term_to_genes.get(a, set())
        if not ga:
            continue
        for b in terms[i + 1 :]:
            gb = term_to_genes.get(b, set())
            if not gb:
                continue
            inter = len(ga & gb)
            if inter < int(min_shared_genes):
                continue
            union = len(ga | gb)
            jac = float(inter / union) if union else 0.0
            if jac < float(jaccard_min):
                continue
            adj[a].append(b)
            adj[b].append(a)

    comp_map: dict[str, int] = {}
    seen: set[str] = set()
    comp = 0

    for start in terms:
        if start in seen:
            continue
        q: deque[str] = deque([start])
        seen.add(start)
        comp_map[start] = comp

        while q:
            cur = q.popleft()
            for nb in sorted(adj.get(cur, [])):
                if nb in seen:
                    continue
                seen.add(nb)
                comp_map[nb] = comp
                q.append(nb)

        comp += 1

    return comp_map


def _pick_rep_genes_by_degree(
    edges_f: pd.DataFrame, genes: list[str], *, topk: int = 10
) -> list[str]:
    """
    Pick representative genes for a module using term-degree in the edge graph.

    Parameters
    ----------
    edges_f
        Filtered edge table with columns ``gene_id`` and ``term_uid``.
    genes
        Candidate genes for a module.
    topk
        Number of representative genes to return.

    Returns
    -------
    list[str]
        Representative genes sorted by descending term-degree, with
        lexicographic tie-breaking.

    Notes
    -----
    If degree information is unavailable, this falls back to the first ``topk``
    genes in the input list.
    """
    if not genes:
        return []
    if edges_f.empty or "gene_id" not in edges_f.columns or "term_uid" not in edges_f.columns:
        return list(genes)[:topk]

    deg = edges_f.groupby("gene_id")["term_uid"].nunique()
    scored = []
    for g in genes:
        try:
            d = int(deg.get(g, 0))
        except Exception:
            d = 0
        scored.append((d, str(g)))
    scored = sorted(scored, key=lambda x: (-x[0], x[1]))
    return [g for _, g in scored[:topk]]


# -------------------------
# Public API
# -------------------------
ModuleMethod = Literal["bipartite_cc", "term_jaccard_cc"]


def factorize_modules_connected_components(
    evidence_df: pd.DataFrame,
    *,
    method: ModuleMethod = "term_jaccard_cc",
    module_prefix: str = "M",
    max_gene_term_degree: int | None = None,
    max_term_degree: int | None = None,  # deprecated alias
    hub_degree_quantile: float | None = 0.995,
    min_shared_genes: int = 3,
    jaccard_min: float = 0.10,
    term_id_col: str = "term_uid",
    genes_col: str = "evidence_genes",
    sparsity_mode: Literal["auto", "off"] = "auto",
    shared_pos_target: float = 0.10,
    sparse_relax_min_shared_genes: int = 2,
    sparse_relax_jaccard_min: float = 0.02,
    pair_sample_max: int = 200000,
    seed: int = 42,
) -> ModuleOutputs:
    """
    Factorize enrichment evidence into stable "evidence modules".

    This constructs a term-by-gene bipartite graph from an evidence table and
    groups related terms into modules. Module identity is stable: module_id is
    derived from a content hash of (terms, genes).

    Parameters
    ----------
    evidence_df
        Evidence table containing term identifiers and evidence genes.
    method
        Module construction method.
        - "term_jaccard_cc": connected components on a term-term graph derived
          from shared genes (recommended).
        - "bipartite_cc": connected components on the bipartite graph (legacy).
    module_prefix
        Prefix prepended to the module_id (default "M").
    max_gene_term_degree
        If set, removes genes whose term-degree is strictly greater than this
        threshold before module construction.
    max_term_degree
        Deprecated alias for ``max_gene_term_degree``.
    hub_degree_quantile
        If not None and explicit thresholds are not given, infer the hub degree
        threshold from the specified quantile of gene term-degree.
    min_shared_genes
        Minimum shared genes for term-term edges (term_jaccard_cc).
    jaccard_min
        Minimum Jaccard similarity for term-term edges (term_jaccard_cc).
    term_id_col
        Column name in ``evidence_df`` holding the term identifier. The pipeline
        convention is "term_uid".
    genes_col
        Column name in ``evidence_df`` holding evidence genes.
    sparsity_mode
        If "auto", relaxes thresholds for sparse graphs and may tighten
        thresholds to avoid giant-component collapse.
    shared_pos_target
        Target lower bound for P(shared_genes > 0) under auto sparsity tuning.
    sparse_relax_min_shared_genes
        Relaxed min_shared_genes used when sparsity is detected.
    sparse_relax_jaccard_min
        Relaxed jaccard_min used when sparsity is detected.
    pair_sample_max
        Maximum number of term pairs sampled for sparsity diagnostics.
    seed
        Random seed for sampling-based diagnostics.

    Returns
    -------
    ModuleOutputs
        Object containing:
        - modules_df: per-module summary table
        - term_modules_df: term_uid -> module_id assignments (one per term)
        - edges_df: filtered edge table used to build modules

    Raises
    ------
    ValueError
        If an unknown method is requested, required columns are missing, or the
        term->module contract is violated.

    Notes
    -----
    - Hub filtering and sparsity/giant-component heuristics are recorded in
      ``edges_df.attrs["modules"]`` for reproducibility and debugging.
    - module_id is stable and derived from module content, not from component
      numbering.
    """
    if module_prefix is None:
        module_prefix = "M"
    module_prefix = str(module_prefix).strip() or "M"

    requested_method: str = str(method)

    edges = build_term_gene_edges(evidence_df, term_id_col=term_id_col, genes_col=genes_col)

    # --- Pairwise sparsity stats (RAW; pre hub-filter) ---
    # Rationale:
    #   - We must not change HALLMARK/GO behavior due to hub filtering.
    #   - Use raw edges to decide whether sparsity tuning should trigger.
    term_to_genes_raw = _build_term_gene_sets(edges)
    pair_stats_raw = _estimate_pair_stats(
        term_to_genes_raw, pair_sample_max=pair_sample_max, seed=seed
    )

    # Data-driven hub threshold
    # Only used when caller explicitly sets max_gene_term_degree=None and provides
    # hub_degree_quantile.
    if hub_degree_quantile is not None and max_gene_term_degree is None and max_term_degree is None:
        try:
            deg0 = edges.groupby("gene_id")["term_uid"].nunique()
            q = float(hub_degree_quantile)
            q = min(max(q, 0.0), 1.0)
            thr = int(deg0.quantile(q))
            thr = max(1, thr)
            max_gene_term_degree = thr
        except Exception:
            # Contract-safe fallback
            max_gene_term_degree = 200

    edges_f = filter_hub_genes(
        edges, max_gene_term_degree=max_gene_term_degree, max_term_degree=max_term_degree
    )

    # recompute degrees on FILTERED edges for consistency/debuggability
    if not edges_f.empty:
        edges_f = edges_f.copy()
        edges_f["gene_term_degree"] = edges_f.groupby("gene_id")["term_uid"].transform("nunique")
        edges_f["term_gene_degree"] = edges_f.groupby("term_uid")["gene_id"].transform("nunique")

    hub_meta = edges_f.attrs.get("hub_filter", {}) if hasattr(edges_f, "attrs") else {}
    hub_max = hub_meta.get("max_gene_term_degree", max_gene_term_degree)
    hub_n = hub_meta.get("n_hubs", 0)

    # initialize provenance early
    edges_f.attrs.setdefault("modules", {})
    edges_f.attrs["modules"].update(
        {
            "requested_method": requested_method,
            "effective_method": None,
            "min_shared_genes": int(min_shared_genes),
            "jaccard_min": float(jaccard_min),
            "term_id_col": str(term_id_col),
            "genes_col": str(genes_col),
        }
    )

    if edges_f.empty:
        modules_df = pd.DataFrame(
            columns=[
                "module_id",
                "module_rank",
                "n_terms",
                "n_genes",
                "rep_gene_ids",
                "term_ids",
                "rep_gene_ids_str",
                "term_ids_str",
                "module_method",
                "module_min_shared_genes",
                "module_requested_method",
                "module_effective_method",
                "module_jaccard_min",
                "hub_filter_max_gene_term_degree",
                "hub_filter_n_hubs",
                "module_terms_hash12",
                "module_genes_hash12",
                "module_content_hash12",
                # optional survival (only filled when term_survival exists upstream)
                "module_survival",
                "module_survival_n_ok",
                "module_survival_n_total",
                "module_survival_source",
            ]
        )
        term_modules_df = pd.DataFrame(columns=["term_uid", "module_id"])
        edges_f.attrs["modules"]["effective_method"] = "none_empty_edges"
        return ModuleOutputs(
            modules_df=modules_df, term_modules_df=term_modules_df, edges_df=edges_f
        )

    term_to_genes = _build_term_gene_sets(edges_f)

    # Pairwise sparsity stats:
    #   - raw (pre hub-filter): used for tuning decision (protects HALLMARK/GO)
    #   - filtered (post hub-filter): recorded for audit/debug only
    pair_stats_filtered = _estimate_pair_stats(
        term_to_genes, pair_sample_max=pair_sample_max, seed=seed
    )
    edges_f.attrs["modules"].update(
        {
            "pair_stats_raw": pair_stats_raw,
            "pair_stats_filtered": pair_stats_filtered,
            "pair_stats_for_tuning": "raw_pre_hub_filter",
        }
    )

    tuned_min_shared = int(min_shared_genes)
    tuned_jaccard = float(jaccard_min)
    did_tune = False

    if sparsity_mode == "auto" and method == "term_jaccard_cc":
        spr = float(pair_stats_raw.get("shared_pos_rate", 0.0) or 0.0)
        edges_f.attrs["modules"]["shared_pos_rate_for_tuning"] = float(spr)

        if spr < float(shared_pos_target):
            tuned_min_shared = int(sparse_relax_min_shared_genes)
            tuned_jaccard = float(sparse_relax_jaccard_min)
            did_tune = True
            edges_f.attrs["modules"].update(
                {
                    "sparsity_detected": True,
                    "tuned_min_shared_genes": tuned_min_shared,
                    "tuned_jaccard_min": tuned_jaccard,
                    "tune_reason": f"shared_pos_rate<{shared_pos_target}",
                }
            )
        else:
            edges_f.attrs["modules"]["sparsity_detected"] = False

    # record pairwise safety info (even if we don't use it)
    n_terms = int(len(term_to_genes))
    n_pairs_est = int(n_terms * (n_terms - 1) / 2)
    edges_f.attrs["modules"].update({"n_terms": n_terms, "n_pairs_est": n_pairs_est})

    effective_method: str = requested_method
    buckets: list[tuple[int, list[str], list[str]]] = []

    if method == "bipartite_cc":
        nodes = _connected_components_from_bipartite_edges(edges_f)
        comp_ids = sorted(nodes["component"].unique().tolist())

        term_lists: dict[int, list[str]] = {c: [] for c in comp_ids}
        gene_lists: dict[int, list[str]] = {c: [] for c in comp_ids}

        tdf = nodes[nodes["kind"].eq("term")].copy()
        for cid, g in tdf.groupby("component"):
            term_lists[int(cid)] = sorted(
                [str(x) for x in g["term_uid"].dropna().astype(str).tolist() if str(x).strip()]
            )

        gdf = nodes[nodes["kind"].eq("gene")].copy()
        for cid, g in gdf.groupby("component"):
            gene_lists[int(cid)] = sorted(
                [str(x) for x in g["gene_id"].dropna().astype(str).tolist() if str(x).strip()]
            )

        buckets = [(cid, term_lists.get(cid, []), gene_lists.get(cid, [])) for cid in comp_ids]
        effective_method = "bipartite_cc"

    elif method == "term_jaccard_cc":
        max_terms_for_pairwise = 3500
        try:
            # Best-effort (attrs can be lost across merges/copies); default is explicit.
            max_terms_for_pairwise = int(
                getattr(evidence_df, "attrs", {}).get("max_terms_for_pairwise", 3500)
            )
        except Exception:
            max_terms_for_pairwise = 3500

        edges_f.attrs["modules"]["max_terms_for_pairwise"] = int(max_terms_for_pairwise)

        if n_terms > int(max_terms_for_pairwise):
            # fallback to bipartite CC and RECORD it truthfully
            nodes = _connected_components_from_bipartite_edges(edges_f)
            comp_ids = sorted(nodes["component"].unique().tolist())

            term_lists = {c: [] for c in comp_ids}
            gene_lists = {c: [] for c in comp_ids}

            tdf = nodes[nodes["kind"].eq("term")].copy()
            for cid, g in tdf.groupby("component"):
                term_lists[int(cid)] = sorted(
                    [str(x) for x in g["term_uid"].dropna().astype(str).tolist() if str(x).strip()]
                )

            gdf = nodes[nodes["kind"].eq("gene")].copy()
            for cid, g in gdf.groupby("component"):
                gene_lists[int(cid)] = sorted(
                    [str(x) for x in g["gene_id"].dropna().astype(str).tolist() if str(x).strip()]
                )

            buckets = [(cid, term_lists.get(cid, []), gene_lists.get(cid, [])) for cid in comp_ids]
            effective_method = "bipartite_cc"

            edges_f.attrs["modules"].update(
                {
                    "effective_method": "bipartite_cc",
                    "fallback_from": "term_jaccard_cc",
                    "fallback_reason": f"n_terms>{max_terms_for_pairwise}",
                }
            )
        else:
            comp_map = _term_term_components_shared_genes(
                term_to_genes,
                min_shared_genes=tuned_min_shared,
                jaccard_min=tuned_jaccard,
            )

            # Decide whether to fallback BEFORE constructing jaccard buckets.
            # ---- Fragmentation diagnostics (once) ----
            if sparsity_mode == "auto":
                comp_sizes: dict[int, int] = {}
                for _, cid in comp_map.items():
                    comp_sizes[int(cid)] = comp_sizes.get(int(cid), 0) + 1

                n_comps = len(comp_sizes)
                n_single = sum(1 for s in comp_sizes.values() if s == 1)
                singleton_frac = (n_single / n_comps) if n_comps > 0 else 1.0

                edges_f.attrs["modules"].update(
                    {
                        "singleton_frac": float(singleton_frac),
                        "did_tune": bool(did_tune),
                        "effective_min_shared_genes": int(tuned_min_shared),
                        "effective_jaccard_min": float(tuned_jaccard),
                    }
                )

            # ---- Giant-component diagnostics & tighten (auto; prevents anchor collapse) ----
            def _comp_sizes_from_map(cm: dict[str, int]) -> dict[int, int]:
                sizes: dict[int, int] = {}
                for _, cid in cm.items():
                    sizes[int(cid)] = sizes.get(int(cid), 0) + 1
                return sizes

            def _largest_frac(sizes: dict[int, int], n_total: int) -> float:
                if n_total <= 0 or not sizes:
                    return 0.0
                return float(max(sizes.values()) / n_total)

            giant_frac_threshold = 0.60
            tighten_max_iter = 2

            if sparsity_mode == "auto":
                comp_sizes0 = _comp_sizes_from_map(comp_map)
                giant_frac0 = _largest_frac(comp_sizes0, n_terms)
                edges_f.attrs["modules"].update(
                    {
                        "largest_component_frac": float(giant_frac0),
                        "giant_frac_threshold": float(giant_frac_threshold),
                    }
                )

                # If giant component dominates, tighten thresholds (opposite of relax)
                if giant_frac0 >= giant_frac_threshold:
                    tight_min_shared = int(tuned_min_shared)
                    tight_jaccard = float(tuned_jaccard)
                    did_tighten = False

                    for it in range(1, tighten_max_iter + 1):
                        # tighten schedule: raise shared first, then jaccard
                        tight_min_shared = max(tight_min_shared, 2) + 1
                        tight_jaccard = min(0.30, tight_jaccard + 0.02)

                        cm2 = _term_term_components_shared_genes(
                            term_to_genes,
                            min_shared_genes=tight_min_shared,
                            jaccard_min=tight_jaccard,
                        )
                        sizes2 = _comp_sizes_from_map(cm2)
                        giant_frac2 = _largest_frac(sizes2, n_terms)

                        edges_f.attrs["modules"].update(
                            {
                                f"tighten_iter_{it}": {
                                    "min_shared_genes": int(tight_min_shared),
                                    "jaccard_min": float(tight_jaccard),
                                    "largest_component_frac": float(giant_frac2),
                                    "n_components": int(len(sizes2)),
                                }
                            }
                        )

                        if giant_frac2 < giant_frac_threshold:
                            comp_map = cm2
                            did_tighten = True
                            edges_f.attrs["modules"].update(
                                {
                                    "giant_detected": True,
                                    "giant_tighten_applied": True,
                                    "effective_min_shared_genes": int(tight_min_shared),
                                    "effective_jaccard_min": float(tight_jaccard),
                                }
                            )
                            break

                    if not did_tighten:
                        edges_f.attrs["modules"].update(
                            {
                                "giant_detected": True,
                                "giant_tighten_applied": False,
                                "giant_tighten_reason": (
                                    f"still giant after {tighten_max_iter} iterations"
                                ),
                            }
                        )

            # ---- Build buckets from comp_map (term -> component id) ----
            comp_to_terms: dict[int, list[str]] = {}
            for t, cid in comp_map.items():
                comp_to_terms.setdefault(int(cid), []).append(str(t))

            comp_ids = sorted(comp_to_terms.keys())
            buckets = []
            for cid in comp_ids:
                terms = sorted([x for x in comp_to_terms[cid] if str(x).strip()])
                genes_set: set[str] = set()
                for t in terms:
                    genes_set |= term_to_genes.get(t, set())
                genes = sorted([_norm_gene_id(g) for g in genes_set if str(g).strip()])
                genes = [g for g in genes if g]
                buckets.append((int(cid), terms, genes))

            effective_method = "term_jaccard_cc"
            edges_f.attrs["modules"].update({"effective_method": "term_jaccard_cc"})
    else:
        raise ValueError(f"Unknown method: {method}")

    # If we didn't set effective_method above (non-fallback path), set it now.
    if edges_f.attrs["modules"].get("effective_method") is None:
        edges_f.attrs["modules"]["effective_method"] = str(effective_method)

    # --- ensure every term gets a module (singleton for isolated terms) ---
    all_terms = sorted(set(edges_f["term_uid"].astype(str).tolist()))
    assigned_terms: set[str] = set()
    for _, terms, _ in buckets:
        for t in terms:
            assigned_terms.add(str(t))

    missing_terms = [t for t in all_terms if t not in assigned_terms]
    if missing_terms:
        next_cid = max([int(x[0]) for x in buckets], default=-1) + 1
        for t in sorted(missing_terms):
            genes = sorted(list(term_to_genes.get(t, set())))
            buckets.append((next_cid, [t], genes))
            next_cid += 1

    # --- stable module_id by content hash ONLY (no component numbering) ---
    hashed: list[tuple[str, list[str], list[str], str, str]] = []
    for _, terms, genes in buckets:
        terms_clean = [str(x).strip() for x in terms if str(x).strip()]
        genes_clean = [_norm_gene_id(x) for x in genes if str(x).strip()]
        genes_clean = [g for g in genes_clean if g]

        terms_hash = _hash_set_short12(terms_clean)
        genes_hash = _hash_gene_set_short12(genes_clean)
        content_hash = _module_hash_content12(terms_clean, genes_clean)
        hashed.append((content_hash, terms_clean, genes_clean, terms_hash, genes_hash))

    hashed = sorted(
        hashed,
        key=lambda x: (x[0], len(x[1]), len(x[2]), (x[1][0] if x[1] else "")),
    )

    term_modules_rows: list[tuple[str, str]] = []
    modules_rows: list[dict[str, object]] = []

    for rank, (content_hash, terms, genes, terms_hash, genes_hash) in enumerate(hashed, start=1):
        mid = f"{module_prefix}{content_hash}"

        for t in terms:
            term_modules_rows.append((str(t), mid))

        rep = _pick_rep_genes_by_degree(edges_f, genes, topk=10)

        modules_rows.append(
            {
                "module_id": mid,
                "module_rank": int(rank),
                "n_terms": int(len(terms)),
                "n_genes": int(len(genes)),
                "rep_gene_ids": rep,
                "term_ids": terms,
                "rep_gene_ids_str": _shared.join_genes_tsv(rep),
                "term_ids_str": _shared.join_id_list_tsv(terms),
                "module_method": str(
                    edges_f.attrs["modules"].get("effective_method", effective_method)
                ),
                "module_requested_method": str(requested_method),
                "module_effective_method": str(
                    edges_f.attrs["modules"].get("effective_method", effective_method)
                ),
                "module_min_shared_genes": int(
                    edges_f.attrs.get("modules", {}).get(
                        "effective_min_shared_genes", min_shared_genes
                    )
                ),
                "module_jaccard_min": float(
                    edges_f.attrs.get("modules", {}).get("effective_jaccard_min", jaccard_min)
                ),
                "hub_filter_max_gene_term_degree": int(hub_max) if hub_max is not None else pd.NA,
                "hub_filter_n_hubs": int(hub_n),
                "module_terms_hash12": terms_hash,
                "module_genes_hash12": genes_hash,
                "module_content_hash12": content_hash,
            }
        )

    term_modules_df = (
        pd.DataFrame(term_modules_rows, columns=["term_uid", "module_id"])
        .sort_values(["term_uid", "module_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    # Contract: 1 term_uid -> exactly 1 module_id
    dup_term = term_modules_df.groupby("term_uid")["module_id"].nunique()
    bad = dup_term[dup_term > 1]
    if not bad.empty:
        ex = bad.index.astype(str).tolist()[:5]
        raise ValueError(
            "term_modules_df violates contract: a term_uid maps to multiple module_id. "
            f"examples={ex}"
        )

    modules_df = (
        pd.DataFrame(modules_rows)
        .sort_values(["module_rank", "module_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    # -------------------------
    # Optional: module_survival (computed from term_survival if available)
    # -------------------------
    if "term_survival" in evidence_df.columns and (term_id_col in evidence_df.columns):
        try:
            tmp = evidence_df[[term_id_col, "term_survival"]].copy()
            tmp[term_id_col] = tmp[term_id_col].astype(str).str.strip()
            tmp["term_survival"] = pd.to_numeric(tmp["term_survival"], errors="coerce")

            # Join term -> module, then aggregate
            # term_modules_df is contractually keyed by "term_uid" regardless of term_id_col
            j = tmp.merge(
                term_modules_df[["term_uid", "module_id"]],
                left_on=term_id_col,
                right_on="term_uid",
                how="inner",
                validate="m:1",
            )
            # Drop helper join key to avoid confusion downstream
            if "term_uid" in j.columns:
                j = j.drop(columns=["term_uid"])

            j["module_id"] = j["module_id"].astype(str).str.strip()

            g = j.groupby("module_id")["term_survival"]
            ms = g.min()
            n_ok = g.apply(lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum()))
            n_total = g.size().astype(int)

            # Attach to modules_df (left join; preserve module ordering)
            modules_df = modules_df.merge(
                ms.rename("module_survival").reset_index(),
                on="module_id",
                how="left",
                validate="1:1",
            )
            modules_df = modules_df.merge(
                n_ok.rename("module_survival_n_ok").reset_index(),
                on="module_id",
                how="left",
                validate="1:1",
            )
            modules_df = modules_df.merge(
                n_total.rename("module_survival_n_total").reset_index(),
                on="module_id",
                how="left",
                validate="1:1",
            )
            modules_df["module_survival_source"] = "term_survival:min"
        except Exception:
            # Contract-safe: never break module factorization due to survival attach
            if "module_survival" not in modules_df.columns:
                modules_df["module_survival"] = pd.NA
            if "module_survival_n_ok" not in modules_df.columns:
                modules_df["module_survival_n_ok"] = pd.NA
            if "module_survival_n_total" not in modules_df.columns:
                modules_df["module_survival_n_total"] = pd.NA
            if "module_survival_source" not in modules_df.columns:
                modules_df["module_survival_source"] = ""

    return ModuleOutputs(modules_df=modules_df, term_modules_df=term_modules_df, edges_df=edges_f)


def attach_module_ids(
    evidence_df: pd.DataFrame,
    term_modules_df: pd.DataFrame,
    *,
    term_id_col: str = "term_uid",
    modules_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Attach module identifiers to an evidence table by term_uid.

    Parameters
    ----------
    evidence_df
        Evidence table that includes ``term_id_col`` (typically "term_uid").
    term_modules_df
        Term-to-module table with columns ``term_id_col`` and ``module_id``.
    term_id_col
        Join key column name for term identifiers.
    modules_df
        Optional per-module table. If provided, module-level survival fields are
        joined onto each term row.

    Returns
    -------
    pandas.DataFrame
        Copy of ``evidence_df`` with:
        - module_id
        - module_id_missing (bool)
        and, optionally, module survival columns if ``modules_df`` was provided.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if term_id_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {term_id_col} (hint: run distill first)")
    if "module_id" not in term_modules_df.columns:
        raise ValueError("term_modules_df must have column: module_id")

    # Backward/forward compatible join key:
    # - Prefer the caller-specified term_id_col if present
    # - Otherwise accept canonical 'term_uid'
    if term_id_col in term_modules_df.columns:
        tm_key = term_id_col
    elif "term_uid" in term_modules_df.columns:
        tm_key = "term_uid"
    else:
        raise ValueError(
            f"term_modules_df must have a term id column. Expected '{term_id_col}' or 'term_uid'."
        )

    out = evidence_df.merge(
        term_modules_df[[tm_key, "module_id"]],
        left_on=term_id_col,
        right_on=tm_key,
        how="left",
        validate="m:1",
    )
    if tm_key != term_id_col and tm_key in out.columns:
        out = out.drop(columns=[tm_key])

    out["module_id_missing"] = out["module_id"].isna()

    # Optional attach: module_survival (and related columns) onto each term row
    if modules_df is not None and (not modules_df.empty) and ("module_id" in modules_df.columns):
        cols = ["module_id"]
        for c in [
            "module_survival",
            "module_survival_n_ok",
            "module_survival_n_total",
            "module_survival_source",
        ]:
            if c in modules_df.columns:
                cols.append(c)

        if len(cols) > 1:
            md = modules_df[cols].copy()
            md["module_id"] = md["module_id"].astype(str).str.strip()
            out = out.merge(md, on="module_id", how="left", validate="m:1")

    return out


# -------------------------
# Stress helpers (evidence identity collapse)
# -------------------------
def compute_term_module_drift(
    baseline_term_modules_df: pd.DataFrame,
    stressed_term_modules_df: pd.DataFrame,
    *,
    term_id_col: str = "term_uid",
) -> pd.DataFrame:
    """
    Compute per-term drift of module assignment under stress.

    Parameters
    ----------
    baseline_term_modules_df
        Baseline term-to-module assignments.
    stressed_term_modules_df
        Stressed term-to-module assignments.
    term_id_col
        Term identifier column name (default "term_uid").

    Returns
    -------
    pandas.DataFrame
        Drift table with columns:
        - term_uid
        - module_id_base
        - module_id_stress
        - module_drift (bool)

    Raises
    ------
    ValueError
        If inputs do not have required columns or violate the one-term-one-module
        contract.
    """
    for df, name in [
        (baseline_term_modules_df, "baseline"),
        (stressed_term_modules_df, "stressed"),
    ]:
        if term_id_col not in df.columns or "module_id" not in df.columns:
            raise ValueError(f"{name} term_modules_df must have columns: {term_id_col}, module_id")

    b = baseline_term_modules_df[[term_id_col, "module_id"]].copy()
    s = stressed_term_modules_df[[term_id_col, "module_id"]].copy()

    # Canonicalize output key to 'term_uid' to keep downstream stable.
    if term_id_col != "term_uid":
        b = b.rename(columns={term_id_col: "term_uid"})
        s = s.rename(columns={term_id_col: "term_uid"})
        join_col = "term_uid"
    else:
        join_col = term_id_col

    if b.groupby(join_col)["module_id"].nunique().max() > 1:
        raise ValueError(
            "baseline_term_modules_df violates contract (term maps to multiple modules)"
        )
    if s.groupby(join_col)["module_id"].nunique().max() > 1:
        raise ValueError(
            "stressed_term_modules_df violates contract (term maps to multiple modules)"
        )

    out = b.merge(s, on=join_col, how="outer", suffixes=("_base", "_stress"))
    out["module_id_base"] = out["module_id_base"].astype("string")
    out["module_id_stress"] = out["module_id_stress"].astype("string")

    out["module_drift"] = out["module_id_base"].fillna("") != out["module_id_stress"].fillna("")
    out = out.sort_values([join_col], kind="mergesort").reset_index(drop=True)
    return out


def summarize_module_drift(drift_df: pd.DataFrame) -> dict[str, object]:
    """
    Summarize module drift statistics.

    Parameters
    ----------
    drift_df
        Output of ``compute_term_module_drift`` with required columns:
        term_uid, module_id_base, module_id_stress, module_drift.

    Returns
    -------
    dict
        Summary metrics including:
        - n_terms_total, n_terms_drift, term_drift_rate
        - n_modules_base, n_modules_stress, n_modules_shared
        - module_churn_rate
    """
    required = {"term_uid", "module_id_base", "module_id_stress", "module_drift"}
    if not required.issubset(set(drift_df.columns)):
        raise ValueError(f"drift_df missing columns: {sorted(required - set(drift_df.columns))}")

    n_terms = int(len(drift_df))
    n_drift = int(drift_df["module_drift"].astype(bool).sum())

    base_mods = set(
        [str(x) for x in drift_df["module_id_base"].dropna().astype(str).tolist() if str(x)]
    )
    stress_mods = set(
        [str(x) for x in drift_df["module_id_stress"].dropna().astype(str).tolist() if str(x)]
    )

    shared = base_mods & stress_mods

    return {
        "n_terms_total": n_terms,
        "n_terms_drift": n_drift,
        "term_drift_rate": (n_drift / n_terms) if n_terms > 0 else 0.0,
        "n_modules_base": int(len(base_mods)),
        "n_modules_stress": int(len(stress_mods)),
        "n_modules_shared": int(len(shared)),
        "module_churn_rate": (1.0 - (len(shared) / len(base_mods))) if base_mods else 0.0,
    }


def attach_module_drift_stress_tag(
    distilled_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    *,
    term_id_col: str = "term_uid",
    stress_col: str = "stress_tag",
    tag: str = "module_drift",
) -> pd.DataFrame:
    """
    Annotate terms with a stress tag when module assignment drifted.

    Parameters
    ----------
    distilled_df
        Distilled evidence table with ``term_id_col`` and an optional stress tag
        column.
    drift_df
        Drift table containing ``term_id_col`` and ``module_drift`` (bool).
    term_id_col
        Term identifier column name (default "term_uid").
    stress_col
        Column name used to store stress tags (default "stress_tag").
    tag
        Tag value to append when drift is detected (default "module_drift").

    Returns
    -------
    pandas.DataFrame
        Copy of ``distilled_df`` with updated ``stress_col``. Existing tags are
        preserved and the new tag is appended if missing.

    Raises
    ------
    ValueError
        If required columns are missing.

    Notes
    -----
    - Does not overwrite non-empty tags; it appends.
    - Tag splitting/joining is delegated to ``_shared.split_tags`` and
      ``_shared.join_tags``.
    """
    if term_id_col not in distilled_df.columns:
        raise ValueError(f"distilled_df missing column: {term_id_col}")
    if "module_drift" not in drift_df.columns:
        raise ValueError("drift_df must have column: module_drift")

    # Backward/forward compatible join key for drift_df:
    if term_id_col in drift_df.columns:
        d_key = term_id_col
    elif "term_uid" in drift_df.columns:
        d_key = "term_uid"
    else:
        raise ValueError(
            f"drift_df must have a term id column. Expected '{term_id_col}' or 'term_uid'."
        )

    d = drift_df[[d_key, "module_drift"]].copy()
    d["module_drift"] = d["module_drift"].astype(bool)

    out = distilled_df.copy()
    out = out.merge(
        d,
        left_on=term_id_col,
        right_on=d_key,
        how="left",
        validate="m:1",
    )
    if d_key != term_id_col and d_key in out.columns:
        out = out.drop(columns=[d_key])
    out["module_drift"] = out["module_drift"].fillna(False).astype(bool)

    if stress_col not in out.columns:
        out[stress_col] = ""

    def _append(old: object, add: str) -> str:
        tags0 = _shared.split_tags("" if old is None else str(old))
        if add not in tags0:
            tags0.append(add)
        return _shared.join_tags(tags0)

    mask = out["module_drift"].astype(bool)
    out.loc[mask, stress_col] = out.loc[mask, stress_col].map(lambda x: _append(x, tag))

    return out.drop(columns=["module_drift"])
