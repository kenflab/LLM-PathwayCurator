# LLM-PathwayCurator/src/llm_pathway_curator/utils.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from . import _shared

_ENSG_RE = re.compile(r"^(ENSG\d+)(\.\d+)?$", re.IGNORECASE)


def norm_ensembl_id(x: str) -> str:
    """
    Normalize Ensembl gene IDs by dropping version suffix (e.g., ENSG.. .15 -> ENSG..)
    This is intentionally a "convenience" helper (mapping/display), not a spec-level hash policy.
    """
    s = str(x).strip()
    if not s:
        return s
    m = _ENSG_RE.match(s)
    if not m:
        return s
    return m.group(1).upper()


def build_id_to_symbol_from_distilled(distilled: pd.DataFrame) -> dict[str, str]:
    """
    Build mapping from gene IDs -> symbol if columns exist in distilled.

    Supported ID types:
      - Entrez-like: gene_id / entrez_id / entrez / geneid
      - Ensembl-like: ensembl_gene_id / ensembl_id / ensg (version tolerated)

    Contract:
      - NA/empty handling uses _shared.is_na_token (single source of truth)
      - Leading Excel text prefix (') is stripped via _shared.strip_excel_text_prefix
      - For Ensembl keys, we apply norm_ensembl_id() (drops version) before storing.
    """
    if distilled is None or distilled.empty:
        return {}

    df = distilled.copy()

    sym_cols = [c for c in ("gene_symbol", "symbol", "hgnc_symbol") if c in df.columns]
    if not sym_cols:
        return {}
    symc = sym_cols[0]

    out: dict[str, str] = {}

    def _iter_pairs(id_col: str) -> list[tuple[str, str]]:
        ids = df[id_col].astype(str).map(str.strip).tolist()
        syms = df[symc].astype(str).map(str.strip).tolist()
        return list(zip(ids, syms, strict=False))

    # Entrez-like
    entrez_cols = [c for c in ("gene_id", "entrez_id", "entrez", "geneid") if c in df.columns]
    if entrez_cols:
        idc = entrez_cols[0]
        for gid, sym in _iter_pairs(idc):
            gid = _shared.strip_excel_text_prefix(gid).strip()
            sym = _shared.strip_excel_text_prefix(sym).strip()
            if not gid or _shared.is_na_token(gid):
                continue
            if not sym or _shared.is_na_token(sym):
                continue
            out.setdefault(gid, sym)

    # Ensembl-like
    ensg_cols = [c for c in ("ensembl_gene_id", "ensembl_id", "ensg") if c in df.columns]
    if ensg_cols:
        idc = ensg_cols[0]
        for gid, sym in _iter_pairs(idc):
            gid = _shared.strip_excel_text_prefix(gid).strip()
            sym = _shared.strip_excel_text_prefix(sym).strip()
            if not gid or _shared.is_na_token(gid):
                continue
            if not sym or _shared.is_na_token(sym):
                continue
            out.setdefault(norm_ensembl_id(gid), sym)

    return out


def load_id_map_tsv(path: str | Path) -> dict[str, str]:
    """
    Load user-supplied mapping TSV.
    Expected columns (any works):
      - symbol + (entrez_id|gene_id|entrez|geneid) and/or (ensembl_gene_id|ensembl_id|ensg)

    Returns:
      dict mapping from ID -> symbol
    """
    p = Path(path)
    if not p.exists():
        return {}

    compression: str | None = "gzip" if p.suffix == ".gz" else None
    df = pd.read_csv(p, sep="\t", dtype=str, compression=compression).fillna("")

    sym_cols = [c for c in ("symbol", "gene_symbol", "hgnc_symbol") if c in df.columns]
    if not sym_cols:
        return {}
    symc = sym_cols[0]

    out: dict[str, str] = {}

    def _put(id_raw: str, sym_raw: str, *, ensembl: bool = False) -> None:
        gid = _shared.strip_excel_text_prefix(id_raw).strip()
        sym = _shared.strip_excel_text_prefix(sym_raw).strip()
        if not gid or _shared.is_na_token(gid):
            return
        if not sym or _shared.is_na_token(sym):
            return
        if ensembl:
            gid = norm_ensembl_id(gid)
        out.setdefault(gid, sym)

    # Entrez-like
    for idc in ("entrez_id", "gene_id", "entrez", "geneid"):
        if idc in df.columns:
            for gid, sym in zip(df[idc].astype(str), df[symc].astype(str), strict=False):
                _put(gid, sym, ensembl=False)

    # Ensembl-like
    for idc in ("ensembl_gene_id", "ensembl_id", "ensg"):
        if idc in df.columns:
            for gid, sym in zip(df[idc].astype(str), df[symc].astype(str), strict=False):
                _put(gid, sym, ensembl=True)

    return out


def map_ids_to_symbols(ids: Any, id2sym: dict[str, str]) -> list[str]:
    """
    Map a gene ID list (scalar/list/TSV-style string) to symbols using id2sym.

    Contract:
      - Parsing is delegated to _shared.parse_id_list (single source of truth).
      - Excel prefix stripping is already handled by parse_id_list.
    """
    gids = _shared.parse_id_list(ids)
    if not id2sym:
        return gids

    out: list[str] = []
    for g in gids:
        g0 = str(g).strip()

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
