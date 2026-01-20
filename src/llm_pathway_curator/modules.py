# LLM-PathwayCurator/src/llm_pathway_curator/modules.py
from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class ModuleOutputs:
    modules_df: pd.DataFrame
    term_modules_df: pd.DataFrame
    edges_df: pd.DataFrame


# -------------------------
# Normalization (align with audit/select)
# -------------------------
def _clean_gene_id(g: str) -> str:
    s = str(g).strip().strip('"').strip("'")
    s = " ".join(s.split())
    s = s.strip(",;|")
    return s


def _norm_gene_id(g: str) -> str:
    # match audit.py (_norm_gene_id) to avoid spurious drift
    return _clean_gene_id(g).upper()


def _module_hash(terms: list[str], genes: list[str]) -> str:
    payload = "T:" + "|".join(sorted(terms)) + "\n" + "G:" + "|".join(sorted(genes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


# -------------------------
# Edges: term x gene (bipartite)
# -------------------------
def build_term_gene_edges(
    evidence_df: pd.DataFrame,
    *,
    term_id_col: str = "term_uid",
    genes_col: str = "evidence_genes",
) -> pd.DataFrame:
    if term_id_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {term_id_col} (hint: run distill first to add term_uid)")
    if genes_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {genes_col}")

    rows: list[tuple[str, str, float]] = []

    for _, r in evidence_df[[term_id_col, genes_col]].iterrows():
        term_uid = str(r[term_id_col]).strip()
        if not term_uid:
            continue

        genes = r[genes_col]

        if genes is None:
            genes_list: list[str] = []
        elif isinstance(genes, (list, tuple, set)):
            genes_list = [_norm_gene_id(g) for g in genes]
            genes_list = [g for g in genes_list if g]
        else:
            try:
                if isinstance(genes, float) and math.isnan(genes):
                    genes_list = []
                elif isinstance(pd.isna(genes), bool) and bool(pd.isna(genes)):
                    genes_list = []
                else:
                    s = str(genes).strip().replace(";", ",").replace("|", ",")
                    genes_list = [_norm_gene_id(g) for g in s.split(",")]
                    genes_list = [g for g in genes_list if g]
            except Exception:
                s = str(genes).strip().replace(";", ",").replace("|", ",")
                genes_list = [_norm_gene_id(g) for g in s.split(",")]
                genes_list = [g for g in genes_list if g]

        for g in genes_list:
            rows.append((term_uid, g, 1.0))

    edges = pd.DataFrame(rows, columns=["term_uid", "gene_id", "weight"])
    if edges.empty:
        return pd.DataFrame(columns=["term_uid", "gene_id", "weight"])

    edges = (
        edges.groupby(["term_uid", "gene_id"], as_index=False)["weight"]
        .sum()
        .sort_values(["term_uid", "gene_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return edges


def filter_hub_genes(
    edges: pd.DataFrame,
    *,
    max_gene_term_degree: int | None = 200,
    # backward-compat alias (deprecated)
    max_term_degree: int | None = None,
) -> pd.DataFrame:
    """
    Remove hub genes that connect too many terms (gene term-degree).

    NOTE: max_term_degree is a deprecated alias kept for backward compatibility.
    """
    if max_gene_term_degree is None and max_term_degree is not None:
        max_gene_term_degree = max_term_degree

    if edges.empty or max_gene_term_degree is None:
        return edges

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
# Connected components on bipartite graph (old behavior)
# -------------------------
def _connected_components_from_bipartite_edges(edges: pd.DataFrame) -> pd.DataFrame:
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
# Term-term graph by shared genes (new, paper-friendly)
# -------------------------
def _build_term_gene_sets(edges: pd.DataFrame) -> dict[str, set[str]]:
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
    Deterministic CC on term-term graph where edge exists if:
      shared_genes >= min_shared_genes AND jaccard >= jaccard_min
    """
    terms = sorted(term_to_genes.keys())
    if not terms:
        return {}

    # Build adjacency lists deterministically (O(T^2) but T is typically manageable for Fig2)
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

    # BFS components
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


# -------------------------
# Public API
# -------------------------
ModuleMethod = Literal["bipartite_cc", "term_jaccard_cc"]


def factorize_modules_connected_components(
    evidence_df: pd.DataFrame,
    *,
    method: ModuleMethod = "term_jaccard_cc",  # <-- was "bipartite_cc"
    module_prefix: str = "M",
    # hub filter (gene term-degree)
    max_gene_term_degree: int | None = 200,
    # backward compat alias
    max_term_degree: int | None = None,
    # term-term method knobs (tighter defaults to avoid giant components)
    min_shared_genes: int = 3,
    jaccard_min: float = 0.10,
    term_id_col: str = "term_uid",
    genes_col: str = "evidence_genes",
) -> ModuleOutputs:
    """
    Build modules from evidence.

    Methods:
      - bipartite_cc: connected components on term-gene bipartite graph (legacy)
      - term_jaccard_cc: connected components on term-term graph built from shared genes
        (recommended for paper figures; avoids giant-component collapse)

    Returns:
      modules_df, term_modules_df, edges_df (filtered)
    """
    edges = build_term_gene_edges(evidence_df, term_id_col=term_id_col, genes_col=genes_col)

    if not edges.empty:
        edges = edges.copy()
        edges["gene_term_degree"] = edges.groupby("gene_id")["term_uid"].transform("nunique")
        edges["term_gene_degree"] = edges.groupby("term_uid")["gene_id"].transform("nunique")

    edges_f = filter_hub_genes(
        edges, max_gene_term_degree=max_gene_term_degree, max_term_degree=max_term_degree
    )

    if edges_f.empty:
        modules_df = pd.DataFrame(
            columns=[
                "module_id",
                "n_terms",
                "n_genes",
                "rep_gene_ids",
                "term_ids",
                "rep_gene_ids_str",
                "term_ids_str",
            ]
        )
        term_modules_df = pd.DataFrame(columns=["term_uid", "module_id"])
        return ModuleOutputs(
            modules_df=modules_df, term_modules_df=term_modules_df, edges_df=edges_f
        )

    if method == "bipartite_cc":
        nodes = _connected_components_from_bipartite_edges(edges_f)

        # derive term lists / gene lists per component (base module)
        comp_ids = sorted(nodes["component"].unique().tolist())
        comp_to_mid_base = {c: f"{module_prefix}{i:04d}" for i, c in enumerate(comp_ids, start=1)}
        nodes = nodes.copy()
        nodes["module_id_base"] = nodes["component"].map(comp_to_mid_base)

        gene_lists = (
            nodes[nodes["kind"].eq("gene")]
            .groupby("module_id_base")["gene_id"]
            .apply(lambda s: sorted([g for g in s.dropna().astype(str).tolist() if g]))
        )
        term_lists = (
            nodes[nodes["kind"].eq("term")]
            .groupby("module_id_base")["term_uid"]
            .apply(lambda s: sorted([t for t in s.dropna().astype(str).tolist() if t]))
        )

    elif method == "term_jaccard_cc":
        term_to_genes = _build_term_gene_sets(edges_f)

        # Guardrail: term-term CC is O(T^2). If too large, fall back deterministically.
        max_terms_for_pairwise = 2500
        try:
            max_terms_for_pairwise = int(evidence_df.attrs.get("max_terms_for_pairwise", 2500))  # type: ignore[attr-defined]
        except Exception:
            pass

        if len(term_to_genes) > max_terms_for_pairwise:
            # Deterministic fallback: bipartite_cc (legacy) to avoid quadratic blowup.
            nodes = _connected_components_from_bipartite_edges(edges_f)

            comp_ids = sorted(nodes["component"].unique().tolist())
            comp_to_mid_base = {
                c: f"{module_prefix}{i:04d}" for i, c in enumerate(comp_ids, start=1)
            }
            nodes = nodes.copy()
            nodes["module_id_base"] = nodes["component"].map(comp_to_mid_base)

            gene_lists = (
                nodes[nodes["kind"].eq("gene")]
                .groupby("module_id_base")["gene_id"]
                .apply(lambda s: sorted([g for g in s.dropna().astype(str).tolist() if g]))
            )
            term_lists = (
                nodes[nodes["kind"].eq("term")]
                .groupby("module_id_base")["term_uid"]
                .apply(lambda s: sorted([t for t in s.dropna().astype(str).tolist() if t]))
            )

            # record fallback provenance (in-memory)
            edges_f.attrs["modules"] = {
                "method": "term_jaccard_cc_fallback_bipartite_cc",
                "reason": f"n_terms>{max_terms_for_pairwise}",
                "min_shared_genes": int(min_shared_genes),
                "jaccard_min": float(jaccard_min),
            }
        else:
            comp_map = _term_term_components_shared_genes(
                term_to_genes,
                min_shared_genes=min_shared_genes,
                jaccard_min=jaccard_min,
            )

            # stable base ids
            comp_ids = sorted(set(comp_map.values()))
            comp_to_mid_base = {
                c: f"{module_prefix}{i:04d}" for i, c in enumerate(comp_ids, start=1)
            }

            # build lists directly
            term_lists = {}
            gene_lists = {}
            for t, cid in comp_map.items():
                base = comp_to_mid_base[cid]
                term_lists.setdefault(base, []).append(t)
            for base in list(term_lists.keys()):
                terms = sorted(term_lists[base])
                term_lists[base] = terms
                genes_union: set[str] = set()
                for t in terms:
                    genes_union |= set(term_to_genes.get(t, set()))
                gene_lists[base] = sorted([g for g in genes_union if g])

            gene_lists = pd.Series(gene_lists)
            term_lists = pd.Series(term_lists)

    else:
        raise ValueError(f"Unknown method: {method}")

    # stabilize module_id by content hash
    base_to_mid: dict[str, str] = {}
    for base in sorted(term_lists.index.tolist()):
        genes = gene_lists.get(base, [])
        terms = term_lists.get(base, [])
        h = _module_hash(list(terms), list(genes))
        base_to_mid[str(base)] = f"{base}__{h}"

    # term_modules_df
    term_modules_rows: list[tuple[str, str]] = []
    for base in sorted(base_to_mid.keys()):
        mid = base_to_mid[base]
        for t in term_lists.get(base, []):
            term_modules_rows.append((str(t), mid))

    term_modules_df = (
        pd.DataFrame(term_modules_rows, columns=["term_uid", "module_id"])
        .drop_duplicates()
        .sort_values(["module_id", "term_uid"], kind="mergesort")
        .reset_index(drop=True)
    )

    # modules_df
    modules_rows: list[dict[str, object]] = []
    for base in sorted(base_to_mid.keys()):
        mid = base_to_mid[base]
        genes = list(gene_lists.get(base, []))
        terms = list(term_lists.get(base, []))
        rep = genes[:10]
        modules_rows.append(
            {
                "module_id": mid,
                "n_terms": int(len(terms)),
                "n_genes": int(len(genes)),
                "rep_gene_ids": rep,
                "term_ids": terms,
                "rep_gene_ids_str": ",".join(rep),
                "term_ids_str": ",".join(terms),
                "module_method": method,
                "module_min_shared_genes": int(min_shared_genes),
                "module_jaccard_min": float(jaccard_min),
            }
        )

    modules_df = (
        pd.DataFrame(modules_rows).sort_values("module_id", kind="mergesort").reset_index(drop=True)
    )

    # record provenance on edges_f (kept in-memory; TSV writers can optionally serialize)
    edges_f.attrs["modules"] = {
        "method": method,
        "min_shared_genes": int(min_shared_genes),
        "jaccard_min": float(jaccard_min),
    }
    return ModuleOutputs(modules_df=modules_df, term_modules_df=term_modules_df, edges_df=edges_f)


def attach_module_ids(
    evidence_df: pd.DataFrame,
    term_modules_df: pd.DataFrame,
    *,
    term_id_col: str = "term_uid",
) -> pd.DataFrame:
    if term_id_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {term_id_col} (hint: run distill first)")
    if term_id_col not in term_modules_df.columns or "module_id" not in term_modules_df.columns:
        raise ValueError(f"term_modules_df must have columns: {term_id_col}, module_id")

    out = evidence_df.merge(
        term_modules_df,
        on=term_id_col,
        how="left",
        validate="m:1",
    )
    return out
