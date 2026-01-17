# LLM-PathwayCurator/src/llm_pathway_curator/modules.py
from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModuleOutputs:
    modules_df: pd.DataFrame
    term_modules_df: pd.DataFrame
    edges_df: pd.DataFrame


def _clean_gene_id(g: str) -> str:
    s = str(g).strip().strip('"').strip("'")
    s = " ".join(s.split())
    s = s.strip(",;|")
    return s


def _module_hash(terms: list[str], genes: list[str]) -> str:
    # stable content hash (order-independent via sorted lists)
    payload = "T:" + "|".join(sorted(terms)) + "\n" + "G:" + "|".join(sorted(genes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


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
            genes_list = [_clean_gene_id(g) for g in genes]
            genes_list = [g for g in genes_list if g]
        else:
            try:
                if isinstance(genes, float) and math.isnan(genes):
                    genes_list = []
                elif isinstance(pd.isna(genes), bool) and bool(pd.isna(genes)):
                    genes_list = []
                else:
                    s = str(genes).strip().replace(";", ",").replace("|", ",")
                    genes_list = [_clean_gene_id(g) for g in s.split(",")]
                    genes_list = [g for g in genes_list if g]
            except Exception:
                s = str(genes).strip().replace(";", ",").replace("|", ",")
                genes_list = [_clean_gene_id(g) for g in s.split(",")]
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


def filter_hub_genes(edges: pd.DataFrame, *, max_term_degree: int | None = 200) -> pd.DataFrame:
    if edges.empty or max_term_degree is None:
        return edges

    if "gene_id" not in edges.columns or "term_uid" not in edges.columns:
        raise ValueError("filter_hub_genes: edges must have columns term_uid, gene_id")

    deg = edges.groupby("gene_id")["term_uid"].nunique()
    hubs = sorted(deg[deg > int(max_term_degree)].index.astype(str).tolist())
    if not hubs:
        return edges

    out = edges[~edges["gene_id"].isin(set(hubs))].reset_index(drop=True)
    # provenance for audit/report
    out.attrs["hub_filter"] = {
        "max_term_degree": int(max_term_degree),
        "n_hubs": len(hubs),
        "hubs": hubs[:200],
    }
    return out


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
    edges = build_term_gene_edges(evidence_df, term_id_col=term_id_col, genes_col=genes_col)

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
    comp_to_mid_base = {c: f"{module_prefix}{i:04d}" for i, c in enumerate(comp_ids, start=1)}

    nodes = nodes.copy()
    nodes["module_id_base"] = nodes["component"].map(comp_to_mid_base)

    # lists per base module
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

    # stabilize module_id by content hash
    base_to_mid: dict[str, str] = {}
    for base in sorted(comp_to_mid_base.values()):
        genes = gene_lists.get(base, [])
        terms = term_lists.get(base, [])
        h = _module_hash(terms, genes)
        base_to_mid[base] = f"{base}__{h}"

    nodes["module_id"] = nodes["module_id_base"].map(base_to_mid)

    term_modules_df = (
        nodes[nodes["kind"].eq("term")][["term_uid", "module_id"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["module_id", "term_uid"], kind="mergesort")
        .reset_index(drop=True)
    )

    modules_rows: list[dict[str, object]] = []
    for base in sorted(base_to_mid.keys()):
        mid = base_to_mid[base]
        genes = gene_lists.get(base, [])
        terms = term_lists.get(base, [])
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
