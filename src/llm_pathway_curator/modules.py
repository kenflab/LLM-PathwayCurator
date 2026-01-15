# LLM-PathwayCurator/src/llm_pathway_curator/modules.py

# LLM-PathwayCurator/src/llm_pathway_curator/modules.py
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModuleOutputs:
    modules_df: pd.DataFrame
    term_modules_df: pd.DataFrame
    edges_df: pd.DataFrame


def build_term_gene_edges(
    evidence_df: pd.DataFrame,
    *,
    term_id_col: str = "term_uid",
    genes_col: str = "evidence_genes",
) -> pd.DataFrame:
    """
    Build a term-gene edge list from EvidenceTable-like df.

    Expects:
      - term_id_col: term identifier column (default: term_uid)
      - genes_col: list[str] per row OR comma-separated string (accepted)

    Returns edges_df with columns:
      - term_uid
      - gene_id
      - weight (currently 1.0)
    """
    if term_id_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {term_id_col}")
    if genes_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {genes_col}")

    rows: list[tuple[str, str, float]] = []

    for _, r in evidence_df[[term_id_col, genes_col]].iterrows():
        term_uid = str(r[term_id_col]).strip()
        if not term_uid:
            continue

        genes = r[genes_col]

        # IMPORTANT: never call pd.isna() on list-like; it returns an array.
        if genes is None:
            genes_list: list[str] = []
        elif isinstance(genes, (list, tuple, set)):
            genes_list = [str(g).strip() for g in genes if str(g).strip()]
        else:
            # scalar-ish NA check
            try:
                if isinstance(genes, float) and math.isnan(genes):
                    genes_list = []
                elif isinstance(pd.isna(genes), bool) and bool(pd.isna(genes)):
                    genes_list = []
                else:
                    s = str(genes).strip().replace(";", ",").replace("|", ",")
                    genes_list = [g.strip() for g in s.split(",") if g.strip()]
            except Exception:
                s = str(genes).strip().replace(";", ",").replace("|", ",")
                genes_list = [g.strip() for g in s.split(",") if g.strip()]

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


def filter_hub_genes(edges: pd.DataFrame, *, max_term_degree: int | None = 200) -> pd.DataFrame:
    """
    Remove hub genes that connect too many terms (prevents giant components).
    max_term_degree:
      - None: no filtering
      - int: drop genes appearing in > max_term_degree unique terms
    """
    if edges.empty or max_term_degree is None:
        return edges

    if "gene_id" not in edges.columns or "term_uid" not in edges.columns:
        raise ValueError("filter_hub_genes: edges must have columns term_uid, gene_id")

    deg = edges.groupby("gene_id")["term_uid"].nunique()
    hubs = set(deg[deg > int(max_term_degree)].index.tolist())
    if not hubs:
        return edges

    return edges[~edges["gene_id"].isin(hubs)].reset_index(drop=True)


def _connected_components_from_edges(edges: pd.DataFrame) -> pd.DataFrame:
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

    # stable ordering
    kind_order = pd.Categorical(out["kind"], categories=["term", "gene"], ordered=True)
    out = out.assign(kind=kind_order).sort_values(
        ["component", "kind", "term_uid", "gene_id"], kind="mergesort"
    )
    return out.reset_index(drop=True)


def factorize_modules_connected_components(
    evidence_df: pd.DataFrame,
    *,
    module_prefix: str = "M",
    max_term_degree: int | None = 200,
    term_id_col: str = "term_uid",
    genes_col: str = "evidence_genes",
) -> ModuleOutputs:
    """
    Deterministic v0 factorization:
      EvidenceTable(term->genes) -> edges -> (optional hub filtering) -> connected components.

    NOTE(v0):
      - module membership is keyed by term_uid (stable join key).
      - modules_df 'term_ids' contains term_uid strings for reproducible joins.
    """
    edges = build_term_gene_edges(evidence_df, term_id_col=term_id_col, genes_col=genes_col)

    # Add degrees for audit/debug (cheap, useful)
    if not edges.empty:
        edges = edges.copy()
        edges["gene_term_degree"] = edges.groupby("gene_id")["term_uid"].transform("nunique")
        edges["term_gene_degree"] = edges.groupby("term_uid")["gene_id"].transform("nunique")

    edges_f = filter_hub_genes(edges, max_term_degree=max_term_degree)

    nodes = _connected_components_from_edges(edges_f)

    if nodes.empty:
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

    comp_ids = sorted(nodes["component"].unique().tolist())
    comp_to_mid = {c: f"{module_prefix}{i:04d}" for i, c in enumerate(comp_ids, start=1)}

    nodes = nodes.copy()
    nodes["module_id"] = nodes["component"].map(comp_to_mid)

    term_modules_df = (
        nodes[nodes["kind"].eq("term")][["term_uid", "module_id"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["module_id", "term_uid"], kind="mergesort")
        .reset_index(drop=True)
    )

    gene_lists = (
        nodes[nodes["kind"].eq("gene")]
        .groupby("module_id")["gene_id"]
        .apply(lambda s: sorted([g for g in s.dropna().astype(str).tolist() if g]))
    )
    term_lists = (
        nodes[nodes["kind"].eq("term")]
        .groupby("module_id")["term_uid"]
        .apply(lambda s: sorted([t for t in s.dropna().astype(str).tolist() if t]))
    )

    modules_rows: list[dict[str, object]] = []
    for mid in sorted(comp_to_mid.values()):
        genes = gene_lists.get(mid, [])
        terms = term_lists.get(mid, [])
        rep = genes[:10]
        modules_rows.append(
            {
                "module_id": mid,
                "n_terms": int(len(terms)),
                "n_genes": int(len(genes)),
                "rep_gene_ids": rep,
                "term_ids": terms,  # v0: term_uid list (stable join key)
                # TSV-friendly
                "rep_gene_ids_str": ",".join(rep),
                "term_ids_str": ",".join(terms),
            }
        )

    modules_df = (
        pd.DataFrame(modules_rows).sort_values("module_id", kind="mergesort").reset_index(drop=True)
    )

    return ModuleOutputs(modules_df=modules_df, term_modules_df=term_modules_df, edges_df=edges_f)


def attach_module_ids(
    evidence_df: pd.DataFrame,
    term_modules_df: pd.DataFrame,
    *,
    term_id_col: str = "term_uid",
) -> pd.DataFrame:
    if term_id_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {term_id_col}")
    if term_id_col not in term_modules_df.columns or "module_id" not in term_modules_df.columns:
        raise ValueError(f"term_modules_df must have columns: {term_id_col}, module_id")

    out = evidence_df.merge(
        term_modules_df,
        on=term_id_col,
        how="left",
        validate="m:1",  # v0 assumption: each term maps to one module
    )
    return out
