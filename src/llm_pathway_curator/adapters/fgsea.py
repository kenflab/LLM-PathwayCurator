# LLM-PathwayCurator/src/llm_pathway_curator/adapters/fgsea.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .. import _shared

ALIASES: dict[str, str] = {
    "pathway": "pathway",
    "term": "pathway",
    "termname": "pathway",
    "term_name": "pathway",
    "name": "pathway",
    "geneset": "pathway",
    "set": "pathway",
    # NOTE: DO NOT alias a generic "id" to pathway (too error-prone in public tools)
    "nes": "NES",
    "es": "ES",
    "padj": "padj",
    "fdr": "padj",
    "qval": "padj",
    "pval": "pval",
    "pvalue": "pval",
    "leadingedge": "leadingEdge",
    "leading_edge": "leadingEdge",
    "leadingedgegenes": "leadingEdge",
    "leadingedgegene": "leadingEdge",
    "le": "leadingEdge",
}


REQUIRED_CORE = ["pathway", "leadingEdge"]
PREFERRED_STAT = ["NES", "ES"]


def _norm_colname(c: Any) -> str:
    """Normalize a column name for alias matching.

    Parameters
    ----------
    c : Any
        Raw column label.

    Returns
    -------
    str
        Normalized key used to look up ``ALIASES``.
        Strips BOM, removes spaces/underscores, lowercases.
    """
    s = str(c).strip().lstrip("\ufeff")
    s = s.replace(" ", "").replace("_", "")
    return s.lower()


def _is_na(x: Any) -> bool:
    """Return True if a value is treated as NA by the tool contract.

    Parameters
    ----------
    x : Any
        Input scalar.

    Returns
    -------
    bool
        True if NA-like per ``_shared`` (scalar NA or NA tokens).
    """
    return _shared.is_na_scalar(x) or _shared.is_na_token(x)


def _clean_str(x: Any) -> str:
    """Convert a scalar to a stripped string with NA mapped to empty.

    Parameters
    ----------
    x : Any
        Input scalar.

    Returns
    -------
    str
        Stripped string, or ``""`` if NA-like.
    """
    return "" if _is_na(x) else str(x).strip()


def _to_float(x: Any) -> float | None:
    """Convert a scalar to a finite float.

    Parameters
    ----------
    x : Any
        Input scalar. NA-like, non-numeric, or non-finite values yield ``None``.

    Returns
    -------
    float or None
        Finite float value, otherwise ``None``.
    """
    if _is_na(x):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _clean_gene_symbol(g: str) -> str:
    """Clean a single gene symbol token.

    This is a lightweight normalizer that trims whitespace/quotes and strips
    common trailing separators.

    Parameters
    ----------
    g : str
        Raw gene token.

    Returns
    -------
    str
        Cleaned gene token.
    """
    s = g.strip().strip('"').strip("'")
    s = " ".join(s.split())
    s = s.strip(",;|")
    return s


_R_C_WRAPPER_RE = re.compile(r"^\s*c\s*\((.*)\)\s*$", flags=re.IGNORECASE)


def _split_genes(x: Any) -> list[str]:
    """Parse evidence genes from an fgsea leading-edge cell.

    Delegates to ``_shared.parse_genes`` for the spec-level parsing
    (supports separators like ',', ';', '|', bracketed lists, etc.).
    Additionally tolerates common R vector syntax such as ``c(a, b, c)``.

    Parameters
    ----------
    x : Any
        Raw leading-edge cell value.

    Returns
    -------
    list of str
        Parsed gene symbols/IDs (may be empty if input is missing).
    """
    if isinstance(x, str):
        m = _R_C_WRAPPER_RE.match(x)
        if m:
            x = m.group(1)
    return _shared.parse_genes(x)


def _term_slug(s: str) -> str:
    """Create a short, filesystem-safe slug for a term.

    Parameters
    ----------
    s : str
        Input term string.

    Returns
    -------
    str
        Slug containing only alnum and ``_-.`` characters, with other
        characters replaced by ``_``. Output is capped to 80 chars.
    """
    out = []
    for ch in s.strip():
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    # collapse multiple underscores
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:80] or "term"


def _term_hash(s: str) -> str:
    """Compute a stable short hash for a term.

    Parameters
    ----------
    s : str
        Input term string.

    Returns
    -------
    str
        Short 12-hex hash (sha256-derived) via ``_shared.sha256_12hex``.
    """
    return _shared.sha256_12hex(s)


@dataclass(frozen=True)
class FgseaAdapterConfig:
    """Configuration for converting an fgsea result table to EvidenceTable.

    Attributes
    ----------
    source_name : str
        Value to populate the EvidenceTable ``source`` column.
    require_genes : bool
        If True, raise an error when ``leadingEdge`` yields no genes.
    keep_pval : bool
        If True and ``pval`` exists, store it separately (does not replace qval).
    term_id_mode : str
        Term identifier policy.

        - ``"raw"``: ``term_id == pathway`` (recommended; paper-aligned)
        - ``"prefixed_hashed"``: ``term_id == "FGSEA:<slug>|<hash>"`` (legacy)
    drop_na_qval : bool
        If True, drop rows where qval (padj) is missing.
    sort_output : bool
        If True, sort output deterministically by ``qval`` asc then
        ``abs(stat)`` desc.

    Notes
    -----
    Defaults are chosen to match the paper-side EvidenceTable behavior:
    human-readable term IDs, stable ordering, and dropping NA q-values.
    """

    source_name: str = "fgsea"
    require_genes: bool = True
    keep_pval: bool = True  # store pval separately (does not replace qval)

    # term_id policy:
    # - "raw": term_id == pathway (recommended; matches paper EvidenceTable)
    # - "prefixed_hashed": term_id == "FGSEA:<slug>|<hash>" (legacy)
    term_id_mode: str = "raw"

    # qval and output determinism
    drop_na_qval: bool = True
    sort_output: bool = True


def read_fgsea_table(path: str) -> pd.DataFrame:
    """Read an fgsea result table from disk.

    Supports TSV by default and falls back to delimiter sniffing or
    whitespace parsing (best-effort).

    Parameters
    ----------
    path : str
        Path to an fgsea result file.

    Returns
    -------
    pandas.DataFrame
        Parsed fgsea table.
    """
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] > 1:
        return df
    df2 = pd.read_csv(path, sep=None, engine="python")
    if df2.shape[1] > 1:
        return df2
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _rename_with_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the adapter's canonical schema using ``ALIASES``.

    The first encountered column that maps to a canonical name wins.
    Additional columns mapping to the same canonical name are recorded
    in ``out.attrs["alias_conflicts"]`` for debugging.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with renamed columns (and possible ``.attrs`` metadata).
    """
    used: set[str] = set()
    rename: dict[str, str] = {}
    conflicts: dict[str, list[str]] = {}
    for orig in df.columns:
        key = _norm_colname(orig)
        std = ALIASES.get(key)
        if std is None:
            continue
        if std in used:
            conflicts.setdefault(std, []).append(str(orig))
            continue
        rename[orig] = std
        used.add(std)

    out = df.rename(columns=rename)
    if conflicts:
        out.attrs["alias_conflicts"] = conflicts
    return out


def fgsea_to_evidence_table(
    fgsea_df: pd.DataFrame,
    *,
    config: FgseaAdapterConfig | None = None,
) -> pd.DataFrame:
    """Convert an fgsea result table to the EvidenceTable contract.

    Parameters
    ----------
    fgsea_df : pandas.DataFrame
        fgsea results table. Must contain (after aliasing) ``pathway`` and
        ``leadingEdge`` plus at least one statistic column among ``NES``/``ES``.
    config : FgseaAdapterConfig or None, optional
        Conversion configuration. If None, defaults are used.

    Returns
    -------
    pandas.DataFrame
        EvidenceTable with core columns:

        - ``term_id`` : str
        - ``term_name`` : str
        - ``source`` : str
        - ``stat`` : float
        - ``qval`` : float or NA (from padj only)
        - ``direction`` : {"up", "down", "na"}
        - ``evidence_genes`` : list[str]

        Plus minimal provenance fields (e.g., ``pval``, ``term_id_h``).

    Raises
    ------
    ValueError
        If required columns are missing, if no stat column is present,
        if ``pathway`` is empty, if the stat column is non-numeric, or if
        ``require_genes=True`` and evidence genes are empty.

    Notes
    -----
    - Only ``padj`` is treated as q-value (FDR) and mapped to ``qval``.
      ``pval`` is stored separately when present and enabled.
    - Output ordering can be stabilized via ``sort_output``.
    """
    if config is None:
        config = FgseaAdapterConfig()

    df = _rename_with_aliases(fgsea_df.copy())

    missing = [c for c in REQUIRED_CORE if c not in df.columns]
    if missing:
        raise ValueError(
            "fgsea_to_evidence_table: missing required columns: "
            f"{missing}. Found={list(df.columns)}"
        )

    stat_col: str | None = None
    for c in PREFERRED_STAT:
        if c in df.columns:
            stat_col = c
            break
    if stat_col is None:
        raise ValueError(
            f"fgsea_to_evidence_table: need one of {PREFERRED_STAT} for stat. "
            f"Found={list(df.columns)}"
        )

    df["__pathway"] = df["pathway"].map(_clean_str)
    empty_pw = df["__pathway"].eq("")
    if empty_pw.any():
        idx = df.index[empty_pw][0]
        raise ValueError(f"fgsea_to_evidence_table: empty pathway at row index={idx!r}")

    df["evidence_genes"] = df["leadingEdge"].map(_split_genes)
    if config.require_genes:
        empty = df["evidence_genes"].map(len).eq(0)
        if empty.any():
            idx = df.index[empty][0]
            pw = df.at[idx, "__pathway"]
            raise ValueError(
                "fgsea_to_evidence_table: empty leadingEdge/evidence_genes at row "
                f"index={idx!r} (pathway={pw})"
            )

    df["__stat"] = df[stat_col].map(_to_float)
    if df["__stat"].isna().any():
        idx = df.index[df["__stat"].isna()][0]
        raise ValueError(f"fgsea_to_evidence_table: non-numeric {stat_col} at row index={idx!r}")

    # qval: ONLY padj maps to qval (FDR). pval stored separately if present.
    if "padj" in df.columns:
        qval = df["padj"].map(_to_float)
    else:
        qval = pd.Series([pd.NA] * len(df), index=df.index)

    if ("pval" in df.columns) and config.keep_pval:
        pval = df["pval"].map(_to_float)
    else:
        pval = pd.Series([pd.NA] * len(df), index=df.index)

    def _dir_from(v: Any) -> str:
        vv = _to_float(v)
        if vv is None:
            return "na"
        return "up" if vv > 0 else ("down" if vv < 0 else "na")

    # direction: prefer NES sign, else fall back to chosen stat_col sign
    if "NES" in df.columns:
        df["direction"] = df["NES"].map(_dir_from)
    else:
        df["direction"] = df[stat_col].map(_dir_from)

    # term_id: default "raw" matches paper EvidenceTable
    term_hash = df["__pathway"].map(_term_hash)
    term_slug = df["__pathway"].map(_term_slug)
    if config.term_id_mode == "prefixed_hashed":
        term_id = term_slug.combine(term_hash, lambda a, b: f"FGSEA:{a}|{b}")
    else:
        term_id = df["__pathway"]

    out = pd.DataFrame(
        {
            "term_id": term_id,
            "term_name": df["__pathway"],
            "source": config.source_name,
            "stat": df["__stat"].astype(float),
            "qval": qval,
            "direction": df["direction"],
            "evidence_genes": df["evidence_genes"],
            # optional provenance fields (kept minimal)
            "q_kind": "padj" if "padj" in df.columns else "na",
            "pval": pval,
            "term_id_h": term_hash,
        }
    ).reset_index(drop=True)

    # Paper-aligned filtering and stable ordering
    if config.drop_na_qval and "qval" in out.columns:
        out = out[out["qval"].notna()].reset_index(drop=True)

    if config.sort_output and ("qval" in out.columns) and (len(out) > 0):
        out["__abs_stat"] = out["stat"].abs()
        out = out.sort_values(by=["qval", "__abs_stat"], ascending=[True, False])
        out = out.drop(columns="__abs_stat").reset_index(drop=True)

    return out


def convert_fgsea_table_to_evidence_tsv(
    in_path: str,
    out_path: str,
    *,
    config: FgseaAdapterConfig | None = None,
) -> pd.DataFrame:
    """Read an fgsea table, convert it, and write an EvidenceTable TSV.

    This is a convenience wrapper around:
    ``read_fgsea_table`` -> ``fgsea_to_evidence_table`` -> TSV write.

    Parameters
    ----------
    in_path : str
        Path to the fgsea result file.
    out_path : str
        Destination path for the EvidenceTable TSV.
    config : FgseaAdapterConfig or None, optional
        Conversion configuration. If None, defaults are used.

    Returns
    -------
    pandas.DataFrame
        EvidenceTable as written, with ``evidence_genes`` serialized for TSV.

    Raises
    ------
    ValueError
        Propagated from ``fgsea_to_evidence_table`` on invalid inputs.
    """
    raw = read_fgsea_table(in_path)
    ev = fgsea_to_evidence_table(raw, config=config)

    ev_out = ev.copy()
    ev_out["evidence_genes"] = ev_out["evidence_genes"].map(_shared.join_genes_tsv)
    ev_out.to_csv(out_path, sep="\t", index=False)
    return ev_out
