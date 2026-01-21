# LLM-PathwayCurator/src/llm_pathway_curator/modules.py
from __future__ import annotations

import hashlib
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


def _hash_set_short12(items: list[str]) -> str:
    """
    Set-stable short hash (sha256[:12]).
    Order invariant, strips whitespace, drops empties.
    """
    uniq = sorted({str(x).strip() for x in items if str(x).strip()})
    payload = ",".join(uniq)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hash_gene_set_short12(genes: list[str]) -> str:
    """
    Gene-set stable short hash (sha256[:12]), normalized (upper).
    This prevents spurious drift from casing differences.
    """
    uniq = sorted({_norm_gene_id(x) for x in genes if str(x).strip()})
    payload = ",".join(uniq)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _module_hash_content12(terms: list[str], genes: list[str]) -> str:
    t = sorted([str(x).strip() for x in terms if str(x).strip()])
    g = sorted([_norm_gene_id(x) for x in genes if str(x).strip()])
    payload = "T:" + "|".join(t) + "\n" + "G:" + "|".join(g)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# -------------------------
# Edges: term x gene (bipartite)
# -------------------------
def _parse_genes_fallback(x: object) -> list[str]:
    """
    Last-resort parser for legacy string inputs.
    Matches schema/distill tolerant behavior:
      - delimiters: , ; |
      - whitespace fallback if no commas present
    """
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []

    s = s.replace(";", ",").replace("|", ",")
    s = s.replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split()).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []

    if "," in s:
        parts = s.split(",")
    else:
        parts = s.split()

    genes = [_norm_gene_id(p) for p in parts]
    genes = [g for g in genes if g]
    # de-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


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

        # Preferred contract: list-like (post-schema/post-distill)
        if isinstance(genes, (list, tuple, set)):
            genes_list = [_norm_gene_id(g) for g in genes]
            genes_list = [g for g in genes_list if g]
        else:
            genes_list = _parse_genes_fallback(genes)

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

    Policy:
      - Remove genes with degree STRICTLY greater than threshold (> max_gene_term_degree).
      - Threshold is recorded in edges.attrs and should also be exported by caller.

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
# Connected components on bipartite graph (legacy)
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
# Term-term graph by shared genes (paper-friendly)
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

    NOTE: O(n^2) in n_terms. Keep max_terms_for_pairwise guard in caller.
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
    Deterministic representative genes:
      - prefer genes with high term-degree in the filtered edge graph
      - tie-break lexicographically
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
    max_gene_term_degree: int | None = 200,
    max_term_degree: int | None = None,  # deprecated alias
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

    modules_df includes identity hashes:
      - module_terms_hash12, module_genes_hash12, module_content_hash12
      - module_id uses module_content_hash12 (prefix + hash)
    """
    edges = build_term_gene_edges(evidence_df, term_id_col=term_id_col, genes_col=genes_col)

    if not edges.empty:
        edges = edges.copy()
        edges["gene_term_degree"] = edges.groupby("gene_id")["term_uid"].transform("nunique")
        edges["term_gene_degree"] = edges.groupby("term_uid")["gene_id"].transform("nunique")

    edges_f = filter_hub_genes(
        edges, max_gene_term_degree=max_gene_term_degree, max_term_degree=max_term_degree
    )

    hub_meta = edges_f.attrs.get("hub_filter", {}) if hasattr(edges_f, "attrs") else {}
    hub_max = hub_meta.get("max_gene_term_degree", max_gene_term_degree)
    hub_n = hub_meta.get("n_hubs", 0)

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
                "module_jaccard_min",
                "hub_filter_max_gene_term_degree",
                "hub_filter_n_hubs",
                "module_terms_hash12",
                "module_genes_hash12",
                "module_content_hash12",
            ]
        )
        term_modules_df = pd.DataFrame(columns=["term_uid", "module_id"])
        return ModuleOutputs(
            modules_df=modules_df, term_modules_df=term_modules_df, edges_df=edges_f
        )

    term_to_genes = _build_term_gene_sets(edges_f)

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

    elif method == "term_jaccard_cc":
        # Conservative default guard (paper+tool safety).
        # 1200 -> ~720k pairs, still heavy but less likely to timeout.
        max_terms_for_pairwise = 1200
        try:
            max_terms_for_pairwise = int(
                evidence_df.attrs.get("max_terms_for_pairwise", 1200)  # type: ignore[attr-defined]
            )
        except Exception:
            pass

        if len(term_to_genes) > max_terms_for_pairwise:
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

            edges_f.attrs.setdefault("modules", {})
            edges_f.attrs["modules"].update(
                {
                    "method": "term_jaccard_cc_fallback_bipartite_cc",
                    "reason": f"n_terms>{max_terms_for_pairwise}",
                    "min_shared_genes": int(min_shared_genes),
                    "jaccard_min": float(jaccard_min),
                    "max_terms_for_pairwise": int(max_terms_for_pairwise),
                }
            )
        else:
            comp_map = _term_term_components_shared_genes(
                term_to_genes,
                min_shared_genes=min_shared_genes,
                jaccard_min=jaccard_min,
            )
            comp_ids = sorted(set(comp_map.values()))

            buckets = []
            for cid in comp_ids:
                terms = sorted([t for t, c in comp_map.items() if c == cid])
                genes_union: set[str] = set()
                for t in terms:
                    genes_union |= set(term_to_genes.get(t, set()))
                genes = sorted([g for g in genes_union if str(g).strip()])
                buckets.append((cid, terms, genes))
    else:
        raise ValueError(f"Unknown method: {method}")

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
        # normalize genes for stable identity
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
                "rep_gene_ids_str": ",".join(rep),
                "term_ids_str": ",".join(terms),
                "module_method": method,
                "module_min_shared_genes": int(min_shared_genes),
                "module_jaccard_min": float(jaccard_min),
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

    # record provenance (do not clobber fallback provenance)
    edges_f.attrs.setdefault("modules", {})
    edges_f.attrs["modules"].update(
        {
            "method": method,
            "min_shared_genes": int(min_shared_genes),
            "jaccard_min": float(jaccard_min),
        }
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
    out["module_id_missing"] = out["module_id"].isna()
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
    Compare baseline vs stressed term->module assignments.

    Returns per-term drift table:
      term_uid, module_id_base, module_id_stress, module_drift (bool)
    """
    for df, name in [
        (baseline_term_modules_df, "baseline"),
        (stressed_term_modules_df, "stressed"),
    ]:
        if term_id_col not in df.columns or "module_id" not in df.columns:
            raise ValueError(f"{name} term_modules_df must have columns: {term_id_col}, module_id")

    b = baseline_term_modules_df[[term_id_col, "module_id"]].copy()
    s = stressed_term_modules_df[[term_id_col, "module_id"]].copy()

    if b.groupby(term_id_col)["module_id"].nunique().max() > 1:
        raise ValueError(
            "baseline_term_modules_df violates contract (term maps to multiple modules)"
        )
    if s.groupby(term_id_col)["module_id"].nunique().max() > 1:
        raise ValueError(
            "stressed_term_modules_df violates contract (term maps to multiple modules)"
        )

    out = b.merge(s, on=term_id_col, how="outer", suffixes=("_base", "_stress"))
    out["module_id_base"] = out["module_id_base"].astype("string")
    out["module_id_stress"] = out["module_id_stress"].astype("string")

    out["module_drift"] = out["module_id_base"].fillna("") != out["module_id_stress"].fillna("")
    out = out.sort_values([term_id_col], kind="mergesort").reset_index(drop=True)
    return out


def summarize_module_drift(drift_df: pd.DataFrame) -> dict[str, object]:
    """
    Summarize drift_df from compute_term_module_drift().
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
    Add stress_tag=module_drift to distilled_df for terms whose module assignment drifted.

    Contract:
      - distilled_df must have term_uid
      - drift_df must have term_uid and module_drift (bool)
      - does NOT overwrite existing non-empty stress_tag (append with '+')
    """
    if term_id_col not in distilled_df.columns:
        raise ValueError(f"distilled_df missing column: {term_id_col}")
    if term_id_col not in drift_df.columns or "module_drift" not in drift_df.columns:
        raise ValueError("drift_df must have columns: term_uid, module_drift")

    d = drift_df[[term_id_col, "module_drift"]].copy()
    d["module_drift"] = d["module_drift"].astype(bool)

    out = distilled_df.copy()
    out = out.merge(d, on=term_id_col, how="left", validate="m:1")
    out["module_drift"] = out["module_drift"].fillna(False).astype(bool)

    if stress_col not in out.columns:
        out[stress_col] = ""

    def _append(old: object, add: str) -> str:
        s = "" if old is None else str(old).strip()
        if not s:
            return add
        if add in s.split("+"):
            return s
        return s + "+" + add

    mask = out["module_drift"].astype(bool)
    out.loc[mask, stress_col] = out.loc[mask, stress_col].map(lambda x: _append(x, tag))

    return out.drop(columns=["module_drift"])
