# LLM-PathwayCurator/src/llm_pathway_curator/utils.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

_NA_TOKENS = {"", "na", "nan", "none", "NA"}
_ENSG_RE = re.compile(r"^(ENSG\d+)(\.\d+)?$", re.IGNORECASE)


def norm_ensembl_id(x: str) -> str:
    s = str(x).strip()
    if not s:
        return s
    m = _ENSG_RE.match(s)
    if not m:
        return s
    return m.group(1).upper()  # drop version


def _is_na_scalar(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict)):
        return False
    try:
        v = pd.isna(x)
        return bool(v) if isinstance(v, bool) else False
    except Exception:
        return False


def parse_id_list(x: Any) -> list[str]:
    if _is_na_scalar(x):
        return []
    if isinstance(x, (list, tuple, set)):
        out = []
        for t in x:
            s = str(t).strip()
            if not s:
                continue
            # Excel text prefix
            if s.startswith("'") and len(s) > 1:
                s = s[1:].strip()
            if s:
                out.append(s)
        return out

    s = str(x).strip()
    if not s or s.lower() in {t.lower() for t in _NA_TOKENS}:
        return []

    s = s.replace(";", ",").replace("|", ",")
    parts = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        if p.startswith("'") and len(p) > 1:
            p = p[1:].strip()
        if p:
            parts.append(p)
    return parts


def build_id_to_symbol_from_distilled(distilled: pd.DataFrame) -> dict[str, str]:
    """
    Build mapping from IDs to symbol if columns exist in distilled.
    Supports both Entrez-like and Ensembl-like IDs.
    """
    if distilled is None or distilled.empty:
        return {}

    df = distilled.copy()
    sym_cols = [c for c in ["gene_symbol", "symbol", "hgnc_symbol"] if c in df.columns]
    if not sym_cols:
        return {}
    symc = sym_cols[0]

    out: dict[str, str] = {}

    entrez_cols = [c for c in ["gene_id", "entrez_id", "entrez", "geneid"] if c in df.columns]
    if entrez_cols:
        idc = entrez_cols[0]
        for gid, sym in zip(
            df[idc].astype(str).str.strip().tolist(),
            df[symc].astype(str).str.strip().tolist(),
            strict=False,
        ):
            if not gid or gid.lower() in _NA_TOKENS:
                continue
            if not sym or sym.lower() in _NA_TOKENS:
                continue
            out.setdefault(gid, sym)

    ensg_cols = [c for c in ["ensembl_gene_id", "ensembl_id", "ensg"] if c in df.columns]
    if ensg_cols:
        idc = ensg_cols[0]
        for gid, sym in zip(
            df[idc].astype(str).str.strip().tolist(),
            df[symc].astype(str).str.strip().tolist(),
            strict=False,
        ):
            if not gid or gid.lower() in _NA_TOKENS:
                continue
            if not sym or sym.lower() in _NA_TOKENS:
                continue
            out.setdefault(norm_ensembl_id(gid), sym)

    return out


def load_id_map_tsv(path: str | Path) -> dict[str, str]:
    """
    Load user-supplied mapping TSV.
    Expected columns (any works):
      - symbol + (entrez_id|gene_id) and/or (ensembl_gene_id|ensembl_id|ensg)
    Returns dict mapping from ID -> symbol.
    """
    p = Path(path)
    if not p.exists():
        return {}

    compression: str | None = "gzip" if p.suffix == ".gz" else None
    df = pd.read_csv(p, sep="\t", dtype=str, compression=compression).fillna("")
    sym_cols = [c for c in ["symbol", "gene_symbol", "hgnc_symbol"] if c in df.columns]
    if not sym_cols:
        return {}
    symc = sym_cols[0]

    out: dict[str, str] = {}
    # Entrez-like
    for idc in ["entrez_id", "gene_id", "entrez", "geneid"]:
        if idc in df.columns:
            for gid, sym in zip(df[idc].astype(str), df[symc].astype(str), strict=False):
                gid = gid.strip()
                sym = sym.strip()
                if gid and sym:
                    out.setdefault(gid, sym)
    # Ensembl-like
    for idc in ["ensembl_gene_id", "ensembl_id", "ensg"]:
        if idc in df.columns:
            for gid, sym in zip(df[idc].astype(str), df[symc].astype(str), strict=False):
                gid = norm_ensembl_id(gid)
                sym = sym.strip()
                if gid and sym:
                    out.setdefault(gid, sym)
    return out


def map_ids_to_symbols(ids: Any, id2sym: dict[str, str]) -> list[str]:
    gids = parse_id_list(ids)
    if not id2sym:
        return gids

    out: list[str] = []
    for g in gids:
        g0 = g[1:].strip() if (g.startswith("'") and len(g) > 1) else g

        # 1) raw
        if g0 in id2sym:
            out.append(id2sym[g0])
            continue

        # 2) normalize Ensembl (drop version)
        g2 = norm_ensembl_id(g0)
        if g2 in id2sym:
            out.append(id2sym[g2])
            continue

        out.append(g0)
    return out
