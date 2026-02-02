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
    Normalize an Ensembl gene ID by dropping an optional version suffix.

    Examples
    --------
    "ENSG00000141510.15" -> "ENSG00000141510"
    "ensg00000141510" -> "ENSG00000141510"

    Parameters
    ----------
    x : str
        Ensembl-like gene identifier, optionally with a version suffix.

    Returns
    -------
    str
        Normalized Ensembl ID (uppercased) if the input matches the ENSG
        pattern. Otherwise, returns the trimmed input unchanged.

    Notes
    -----
    This is a convenience helper for mapping/display, not a spec-level policy.
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
    Build an ID-to-symbol mapping from a distilled evidence table.

    The function searches for a symbol column and one of several supported
    ID columns. It then builds a mapping from gene ID to gene symbol.

    Parameters
    ----------
    distilled : pandas.DataFrame
        Distilled table that may contain gene ID and symbol columns.

    Returns
    -------
    dict[str, str]
        Mapping from gene ID to gene symbol. If required columns are missing
        or `distilled` is empty, returns an empty dict.

    Notes
    -----
    Supported symbol columns (first match is used):
    - "gene_symbol", "symbol", "hgnc_symbol"

    Supported ID columns:
    - Entrez-like: "gene_id", "entrez_id", "entrez", "geneid"
    - Ensembl-like: "ensembl_gene_id", "ensembl_id", "ensg"

    Contract
    --------
    - NA/empty filtering uses `_shared.is_na_token`.
    - Leading Excel text prefix (') is stripped via
      `_shared.strip_excel_text_prefix`.
    - Ensembl IDs are normalized with `norm_ensembl_id()` (drops version).
    - First-seen mapping wins (`dict.setdefault`).
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
    Load a user-supplied ID-to-symbol mapping from a TSV file.

    The file is expected to have a symbol column and one or more ID columns.
    Gzip-compressed TSV (".gz") is supported.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a TSV (optionally gzipped).

    Returns
    -------
    dict[str, str]
        Mapping from ID to symbol. Returns an empty dict if the file does
        not exist or required columns are missing.

    Notes
    -----
    Recognized symbol columns (first match is used):
    - "symbol", "gene_symbol", "hgnc_symbol"

    Recognized ID columns:
    - Entrez-like: "entrez_id", "gene_id", "entrez", "geneid"
    - Ensembl-like: "ensembl_gene_id", "ensembl_id", "ensg"

    Contract
    --------
    - Excel prefix stripping uses `_shared.strip_excel_text_prefix`.
    - NA filtering uses `_shared.is_na_token`.
    - Ensembl IDs are normalized via `norm_ensembl_id()`.
    - First-seen mapping wins (`dict.setdefault`).
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
    Map gene IDs to symbols using a provided ID-to-symbol mapping.

    Parameters
    ----------
    ids : Any
        Gene IDs as a scalar, list-like, or a delimiter-separated string.
        Parsing is delegated to `_shared.parse_id_list()`.
    id2sym : dict[str, str]
        Mapping from gene ID to gene symbol.

    Returns
    -------
    list[str]
        A list of symbols when mappings exist, otherwise original tokens.
        Tokens not found in `id2sym` are returned unchanged.

    Notes
    -----
    Lookup order per token:
    1) raw token
    2) Ensembl-normalized token via `norm_ensembl_id()` (drops version)
    """
    gids = _shared.parse_id_list(ids)
    if not id2sym:
        return gids

    out: list[str] = []
    for g in gids:
        g0 = _shared.strip_excel_text_prefix(str(g)).strip()

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
