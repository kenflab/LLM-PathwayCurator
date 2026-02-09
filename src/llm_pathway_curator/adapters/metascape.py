# LLM-PathwayCurator/src/llm_pathway_curator/adapters/metascape.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .. import _shared

# Columns commonly present in Metascape "Enrichment" sheet.
# NOTE: Genes/Symbols are not both guaranteed; we accept either.
REQUIRED_CORE_COLS = [
    "GroupID",
    "Category",
    "Term",
    "Description",
    "LogP",
    "Log(q-value)",
    "InTerm_InList",
]
EVIDENCE_COLS = ["Symbols", "Genes"]


def _to_float(x: Any) -> float | None:
    if _shared.is_na_scalar(x) or _shared.is_na_token(x):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _clean_str(x: Any) -> str:
    if _shared.is_na_scalar(x) or _shared.is_na_token(x):
        return ""
    return str(x).strip()


def _parse_interm_inlist(x: Any) -> tuple[int | None, int | None]:
    s = _clean_str(x)
    if not s or "/" not in s:
        return (None, None)
    a, b = s.split("/", 1)
    a = a.strip()
    b = b.strip()
    try:
        return (int(a), int(b))
    except Exception:
        return (None, None)


def _is_summary_groupid(gid: str) -> bool:
    s = (gid or "").strip()
    return s.endswith("_Summary")


def _normalize_term_id(term: Any) -> str:
    s = _clean_str(term)
    if not s:
        return ""
    u = s.upper()
    if u.startswith("GO:"):
        return s
    if u.startswith("GO") and len(u) > 2 and u[2:].isdigit():
        return "GO:" + s[2:]
    return s


def _infer_q_from_logq(logq_val: Any) -> float | None:
    """
    Metascape 'Log(q-value)' is observed in the wild as either:
      - log10(q)    (negative for q<1), or
      - -log10(q)   (positive for q<1)
    We infer convention by sign and reconstruct q in (0, 1].
    """
    vv = _to_float(logq_val)
    if vv is None:
        return None

    # If vv <= 0, likely log10(q): q = 10^(vv)
    # If vv > 0, likely -log10(q): q = 10^(-vv)
    if vv <= 0:
        q = float(10**vv)
    else:
        q = float(10 ** (-vv))

    if not (q > 0.0 and q <= 1.0):
        return None
    return q


@dataclass(frozen=True)
class MetascapeAdapterConfig:
    source_name: str = "metascape"
    sheet_name: str = "Enrichment"  # use Enrichment sheet by default
    include_summary: bool = False  # IMPORTANT default: exclude *_Summary rows
    prefer_symbols: bool = True  # prefer Symbols over numeric Genes
    strict_qval: bool = False  # if True: error when Log(q-value) exists but cannot reconstruct
    drop_na_qval: bool = True  # if True: drop rows where qval is NA


def read_metascape_table(path: str, *, sheet_name: str = "Enrichment") -> pd.DataFrame:
    """
    Read Metascape export as either Excel (.xlsx/.xls) or delimited text.
    For Excel exports, the 'Enrichment' sheet is the canonical input.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

    # Fallback: TSV/CSV/whitespace (best-effort)
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] > 1:
        return df
    df2 = pd.read_csv(path, sep=None, engine="python")
    if df2.shape[1] > 1:
        return df2
    return pd.read_csv(path, sep=r"\s+", engine="python")


def metascape_to_evidence_table(
    metascape_df: pd.DataFrame,
    *,
    config: MetascapeAdapterConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = MetascapeAdapterConfig()

    df = metascape_df.copy()

    missing = [c for c in REQUIRED_CORE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"metascape_to_evidence_table: missing required columns: {missing}. "
            f"Found={list(df.columns)}"
        )

    have_evidence = [c for c in EVIDENCE_COLS if c in df.columns]
    if not have_evidence:
        raise ValueError(
            "metascape_to_evidence_table: need at least one of {Symbols, Genes}. "
            f"Found={list(df.columns)}"
        )

    df["group_id"] = df["GroupID"].map(_clean_str)
    df["is_summary"] = df["group_id"].map(_is_summary_groupid)

    if not config.include_summary:
        df = df[~df["is_summary"]].copy()

    df["term_id_raw"] = df["Term"].map(_clean_str)
    df["term_name"] = df["Description"].map(_clean_str)

    bad = df["term_id_raw"].eq("") | df["term_name"].eq("")
    if bad.any():
        i = int(df.index[bad][0])
        raise ValueError(f"metascape_to_evidence_table: empty Term/Description at row index={i}")

    # qval reconstruction from Log(q-value)
    qval = df["Log(q-value)"].map(_infer_q_from_logq)

    if config.strict_qval:
        any_logq = df["Log(q-value)"].map(_to_float).notna().any()
        if any_logq and qval.isna().all():
            raise ValueError(
                "metascape_to_evidence_table: could not reconstruct any valid qval "
                "from Log(q-value). Check column semantics."
            )

    if config.drop_na_qval:
        df = df[qval.notna()].copy()
        qval = qval.loc[df.index]

    # stat: make it monotone-positive for ranking (paper-friendly)
    # Use abs(Log(q-value)) if numeric else abs(LogP) if numeric.
    logq_raw = df["Log(q-value)"].map(_to_float)
    logp_raw = df["LogP"].map(_to_float)

    stat = logq_raw.where(logq_raw.notna(), logp_raw)
    if stat.isna().any():
        i = int(df.index[stat.isna()][0])
        raise ValueError(
            f"metascape_to_evidence_table: non-numeric Log(q-value)/LogP at row index={i}"
        )
    stat_kind = pd.Series("logq", index=df.index).where(logq_raw.notna(), "logp")
    stat = stat.abs().astype(float)

    # evidence genes: prefer Symbols then fall back to Genes (or vice versa)
    if config.prefer_symbols and "Symbols" in df.columns:
        primary = df["Symbols"]
        secondary = df["Genes"] if "Genes" in df.columns else None
    else:
        primary = df["Genes"] if "Genes" in df.columns else df["Symbols"]
        secondary = df["Symbols"] if "Symbols" in df.columns else None

    genes = primary.map(_shared.parse_genes)
    empty = genes.map(len).eq(0)

    if secondary is not None and empty.any():
        genes2 = secondary.map(_shared.parse_genes)
        genes = genes.where(~empty, genes2)

    empty2 = genes.map(len).eq(0)
    if empty2.any():
        i = int(df.index[empty2][0])
        tid = df.loc[i, "term_id_raw"]
        raise ValueError(
            f"metascape_to_evidence_table: empty evidence genes at row index={i} (Term={tid})"
        )

    # parse InTerm/InList
    n_in_term: list[int | None] = []
    n_in_list: list[int | None] = []
    for x in df["InTerm_InList"].tolist():
        a, b = _parse_interm_inlist(x)
        n_in_term.append(a)
        n_in_list.append(b)

    out = pd.DataFrame(
        {
            "term_id": df["term_id_raw"].map(_normalize_term_id),
            "term_name": df["term_name"],
            "source": config.source_name,
            "stat": stat,
            "qval": qval.astype(float),
            "direction": "na",
            "evidence_genes": genes,
            # provenance (optional)
            "stat_kind": stat_kind.astype(str),
            "group_id": df["group_id"],
            "is_summary": df["is_summary"].astype(bool),
            "n_in_term": n_in_term,
            "n_in_list": n_in_list,
            "category": df["Category"].map(_clean_str),
        }
    ).reset_index(drop=True)

    return out


def convert_metascape_table_to_evidence_tsv(
    in_path: str,
    out_path: str,
    *,
    config: MetascapeAdapterConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = MetascapeAdapterConfig()

    raw = read_metascape_table(in_path, sheet_name=config.sheet_name)
    ev = metascape_to_evidence_table(raw, config=config)

    ev_out = ev.copy()
    ev_out["evidence_genes"] = ev_out["evidence_genes"].map(_shared.join_genes_tsv)
    ev_out.to_csv(out_path, sep="\t", index=False)
    return ev_out
