# LLM-PathwayCurator/src/llm_pathway_curator/modules.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ----------------------------
# Public outputs (minimal)
# ----------------------------


@dataclass(frozen=True)
class ModuleOutputs:
    """
    modules_df: one row per module (summary)
    term_modules_df: term -> module mapping (many-to-many allowed; we start 1-to-1)
    edges_df: term-gene edges used to derive modules (for audit/debug)
    """

    modules_df: pd.DataFrame
    term_modules_df: pd.DataFrame
    edges_df: pd.DataFrame


# ----------------------------
# Core: build term-gene edges
# ----------------------------


def build_term_gene_edges(
    evidence_df: pd.DataFrame,
    *,
    term_id_col: str = "term_id",
    genes_col: str = "evidence_genes",
) -> pd.DataFrame:
    """
    Build a term-gene edge list from EvidenceTable-like df.

    Expects:
      - term_id_col: str
      - genes_col: list[str] in each row (canonical), OR comma-separated string (accepted)

    Returns edges_df with columns:
      - term_id
      - gene_id
      - weight (currently 1.0)
    """
    if term_id_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {term_id_col}")
    if genes_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {genes_col}")

    rows = []
    for _, r in evidence_df[[term_id_col, genes_col]].iterrows():
        term_id = str(r[term_id_col]).strip()
        if not term_id:
            continue

        genes = r[genes_col]
        if genes is None:
            genes_list: list[str] = []
        elif isinstance(genes, list):
            genes_list = [str(g).strip() for g in genes if str(g).strip()]
        else:
            # accept strings (comma/semicolon)
            s = str(genes).strip().replace(";", ",")
            genes_list = [g.strip() for g in s.split(",") if g.strip()]

        for g in genes_list:
            rows.append((term_id, g, 1.0))

    edges = pd.DataFrame(rows, columns=["term_id", "gene_id", "weight"])
    if edges.empty:
        # keep schema stable
        return pd.DataFrame(columns=["term_id", "gene_id", "weight"])

    # de-dup edges (term,gene)
    edges = (
        edges.groupby(["term_id", "gene_id"], as_index=False)["weight"]
        .sum()
        .sort_values(["term_id", "gene_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return edges


# ----------------------------
# Core: connected-components modules (deterministic stub)
# ----------------------------


def _connected_components_from_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic connected components on a bipartite graph.
    Nodes are prefixed to avoid collisions: "T:<term_id>", "G:<gene_id>"

    Returns node->component mapping with columns:
      - node
      - kind ("term"|"gene")
      - term_id (nullable)
      - gene_id (nullable)
      - component (int >= 0)
    """
    if edges.empty:
        return pd.DataFrame(columns=["node", "kind", "term_id", "gene_id", "component"])

    # adjacency list
    adj: dict[str, set[str]] = {}

    def _add_edge(a: str, b: str) -> None:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for t, g in edges[["term_id", "gene_id"]].itertuples(index=False):
        tn = f"T:{t}"
        gn = f"G:{g}"
        _add_edge(tn, gn)

    # deterministic traversal: sorted nodes, BFS with sorted neighbors
    seen: set[str] = set()
    comp_map: dict[str, int] = {}
    comp = 0

    for start in sorted(adj.keys()):
        if start in seen:
            continue
        # BFS
        queue = [start]
        seen.add(start)
        comp_map[start] = comp
        while queue:
            cur = queue.pop(0)
            for nb in sorted(adj.get(cur, ())):
                if nb in seen:
                    continue
                seen.add(nb)
                comp_map[nb] = comp
                queue.append(nb)
        comp += 1

    # build dataframe
    nodes = []
    for node, cid in comp_map.items():
        if node.startswith("T:"):
            nodes.append((node, "term", node[2:], None, cid))
        else:
            nodes.append((node, "gene", None, node[2:], cid))

    out = pd.DataFrame(nodes, columns=["node", "kind", "term_id", "gene_id", "component"])
    return out.sort_values(
        ["component", "kind", "term_id", "gene_id"], kind="mergesort"
    ).reset_index(drop=True)


def factorize_modules_connected_components(
    evidence_df: pd.DataFrame,
    *,
    module_prefix: str = "M",
) -> ModuleOutputs:
    """
    Minimal, deterministic module factorization:
      EvidenceTable(term->genes) -> bipartite edges -> connected components as modules.

    Returns:
      - modules_df: per-module summary
      - term_modules_df: per-term module assignment
      - edges_df: term-gene edges used

    Notes:
      - This is a STUB for v0: replace with a fixed community algorithm later.
      - Connected components are deterministic and easy to audit.
    """
    edges = build_term_gene_edges(evidence_df)

    nodes = _connected_components_from_edges(edges)

    if nodes.empty:
        modules_df = pd.DataFrame(
            columns=[
                "module_id",
                "n_terms",
                "n_genes",
                "rep_gene_ids",
                "term_ids",
            ]
        )
        term_modules_df = pd.DataFrame(columns=["term_id", "module_id"])
        return ModuleOutputs(modules_df=modules_df, term_modules_df=term_modules_df, edges_df=edges)

    # component -> module_id (stable ordering by component integer)
    comp_ids = sorted(nodes["component"].unique().tolist())
    comp_to_mid = {c: f"{module_prefix}{i:04d}" for i, c in enumerate(comp_ids, start=1)}

    nodes = nodes.copy()
    nodes["module_id"] = nodes["component"].map(comp_to_mid)

    # term -> module mapping (1-to-1 under connected components)
    term_modules = (
        nodes[nodes["kind"].eq("term")][["term_id", "module_id"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["module_id", "term_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    # per-module gene lists and term lists
    gene_lists = (
        nodes[nodes["kind"].eq("gene")]
        .groupby("module_id")["gene_id"]
        .apply(lambda s: sorted([g for g in s.dropna().astype(str).tolist() if g]))
    )
    term_lists = (
        nodes[nodes["kind"].eq("term")]
        .groupby("module_id")["term_id"]
        .apply(lambda s: sorted([t for t in s.dropna().astype(str).tolist() if t]))
    )

    modules_rows = []
    for mid in sorted(comp_to_mid.values()):
        genes = gene_lists.get(mid, [])
        terms = term_lists.get(mid, [])
        # representative genes (deterministic): first K genes sorted
        rep = genes[:10]
        modules_rows.append(
            {
                "module_id": mid,
                "n_terms": int(len(terms)),
                "n_genes": int(len(genes)),
                "rep_gene_ids": rep,  # list[str]
                "term_ids": terms,  # list[str]
            }
        )

    modules_df = (
        pd.DataFrame(modules_rows).sort_values("module_id", kind="mergesort").reset_index(drop=True)
    )

    return ModuleOutputs(modules_df=modules_df, term_modules_df=term_modules, edges_df=edges)


# ----------------------------
# Optional: attach module_id back onto evidence_df
# ----------------------------


def attach_module_ids(
    evidence_df: pd.DataFrame,
    term_modules_df: pd.DataFrame,
    *,
    term_id_col: str = "term_id",
) -> pd.DataFrame:
    """
    Add module_id to each term row in evidence_df via term_id join.
    """
    if term_id_col not in evidence_df.columns:
        raise ValueError(f"Missing column: {term_id_col}")
    if "term_id" not in term_modules_df.columns or "module_id" not in term_modules_df.columns:
        raise ValueError("term_modules_df must have columns: term_id, module_id")

    out = evidence_df.merge(
        term_modules_df.rename(columns={"term_id": term_id_col}),
        on=term_id_col,
        how="left",
        validate="m:1",
    )
    return out
