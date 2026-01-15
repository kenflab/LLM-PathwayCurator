# LLM-PathwayCurator/src/llm_pathway_curator/adapters/fgsea.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

# -----------------------------
# Column normalization / aliases
# -----------------------------
# Keys are normalized: strip, remove BOM, remove spaces/underscores, lower.
ALIASES: dict[str, str] = {
    # pathway / term name
    "pathway": "pathway",
    "term": "pathway",
    "termname": "pathway",
    "term_name": "pathway",
    "name": "pathway",
    "geneset": "pathway",
    "set": "pathway",
    "id": "pathway",
    # statistics
    "nes": "NES",
    "es": "ES",
    # q-values
    "padj": "padj",
    "fdr": "padj",
    "qval": "padj",
    "pval": "pval",
    "pvalue": "pval",
    # leading edge genes
    "leadingedge": "leadingEdge",
    "leading_edge": "leadingEdge",
    "leadingedgegenes": "leadingEdge",
    "leadingedgegene": "leadingEdge",
    "le": "leadingEdge",
}

REQUIRED_CORE = ["pathway", "leadingEdge"]  # must exist
PREFERRED_STAT = ["NES", "ES"]  # choose first available
PREFERRED_Q = ["padj", "pval"]  # choose first available


def _norm_colname(c: Any) -> str:
    s = str(c).strip().lstrip("\ufeff")
    s = s.replace(" ", "").replace("_", "")
    return s.lower()


def _is_na(x: Any) -> bool:
    if x is None:
        return True
    try:
        if isinstance(x, float) and math.isnan(x):
            return True
    except Exception:
        pass
    s = str(x).strip().lower()
    return s in {"", "na", "nan", "none"}


def _clean_str(x: Any) -> str:
    return "" if _is_na(x) else str(x).strip()


def _to_float(x: Any) -> float | None:
    if _is_na(x):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _split_genes(x: Any) -> list[str]:
    """
    fgsea leadingEdge can be:
      - list-like (already list)
      - "A,B,C" / "A;B;C" / "A|B|C"
      - R-style c("A","B")
      - whitespace separated
    Return stable sorted unique symbols.
    """
    if _is_na(x):
        return []
    if isinstance(x, (list, tuple, set)):
        genes = [str(g).strip() for g in x if str(g).strip()]
        return sorted(set(genes))

    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []

    # tolerate R c("A","B") style
    s = s.replace("c(", "").replace(")", "").replace('"', "").replace("'", "")

    # separators: comma/semicolon/pipe/tab
    for sep in [";", "|", "\t"]:
        s = s.replace(sep, ",")

    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]

    return sorted(set(parts))


@dataclass(frozen=True)
class FgseaAdapterConfig:
    source_name: str = "fgsea"
    require_genes: bool = True  # recommended for your pipeline


def read_fgsea_table(path: str) -> pd.DataFrame:
    """
    Read fgsea output table flexibly (TSV/CSV/auto/whitespace).
    """
    # Try TSV first
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] > 1:
        return df

    # auto-sep
    df2 = pd.read_csv(path, sep=None, engine="python")
    if df2.shape[1] > 1:
        return df2

    # whitespace
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _rename_with_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns using ALIASES with a safe deterministic mapping.
    - Multiple original columns may map to the same standardized name.
      In that case, we keep the first occurrence and ignore later ones.
    """
    used: set[str] = set()
    rename: dict[str, str] = {}
    for orig in df.columns:
        key = _norm_colname(orig)
        std = ALIASES.get(key)
        if std is None:
            continue
        if std in used:
            continue
        rename[orig] = std
        used.add(std)
    return df.rename(columns=rename)


def fgsea_to_evidence_table(
    fgsea_df: pd.DataFrame,
    *,
    config: FgseaAdapterConfig | None = None,
) -> pd.DataFrame:
    """
    Convert fgsea results into EvidenceTable-like dataframe with columns:
      - term_id
      - term_name
      - source
      - stat          (NES preferred, else ES)
      - qval          (padj preferred, else pval; may be NA)
      - direction     (from NES sign if available)
      - evidence_genes (from leadingEdge; list[str])
    """
    if config is None:
        config = FgseaAdapterConfig()

    df = _rename_with_aliases(fgsea_df.copy())

    # required columns
    missing = [c for c in REQUIRED_CORE if c not in df.columns]
    if missing:
        found = list(df.columns)
        raise ValueError(
            f"fgsea_to_evidence_table: missing required columns: {missing}. Found={found}"
        )

    # choose stat column
    stat_col: str | None = None
    for c in PREFERRED_STAT:
        if c in df.columns:
            stat_col = c
            break
    if stat_col is None:
        found = list(df.columns)
        raise ValueError(
            f"fgsea_to_evidence_table: need one of {PREFERRED_STAT} for stat. Found={found}"
        )

    # choose q column (optional)
    q_col: str | None = None
    for c in PREFERRED_Q:
        if c in df.columns:
            q_col = c
            break

    # pathway name
    df["__pathway"] = df["pathway"].map(_clean_str)
    empty_pw = df["__pathway"].eq("")
    if empty_pw.any():
        i = int(df.index[empty_pw][0])
        raise ValueError(f"fgsea_to_evidence_table: empty pathway at row index={i}")

    # evidence genes
    df["evidence_genes"] = df["leadingEdge"].map(_split_genes)
    if config.require_genes:
        empty = df["evidence_genes"].map(len).eq(0)
        if empty.any():
            i = int(df.index[empty][0])
            pw = df.loc[i, "__pathway"]
            raise ValueError(
                "fgsea_to_evidence_table: empty leadingEdge/evidence_genes at row "
                f"index={i} (pathway={pw})"
            )

    # stat numeric
    df["__stat"] = df[stat_col].map(_to_float)
    if df["__stat"].isna().any():
        i = int(df.index[df["__stat"].isna()][0])
        raise ValueError(f"fgsea_to_evidence_table: non-numeric {stat_col} at row index={i}")

    # qval numeric (allowed NA)
    if q_col is not None:
        df["__qval"] = df[q_col].map(_to_float)
    else:
        df["__qval"] = pd.NA

    # direction: prefer NES sign if NES exists + numeric
    if "NES" in df.columns:
        nes = df["NES"].map(_to_float)

        def _dir(v: Any) -> str:
            vv = _to_float(v)
            if vv is None:
                return "na"
            if vv > 0:
                return "up"
            if vv < 0:
                return "down"
            return "na"

        df["direction"] = nes.map(_dir)
    else:
        df["direction"] = "na"

    out = pd.DataFrame(
        {
            "term_id": df["__pathway"].map(lambda s: f"FGSEA:{s}"),
            "term_name": df["__pathway"],
            "source": config.source_name,
            "stat": df["__stat"].astype(float),
            "qval": df["__qval"],
            "direction": df["direction"],
            "evidence_genes": df["evidence_genes"],
        }
    ).reset_index(drop=True)

    return out


def convert_fgsea_table_to_evidence_tsv(
    in_path: str,
    out_path: str,
    *,
    config: FgseaAdapterConfig | None = None,
) -> pd.DataFrame:
    """
    Convenience: file -> Evidence TSV.
    Writes evidence_genes as comma-joined string for EvidenceTable.read_tsv().
    """
    raw = read_fgsea_table(in_path)
    ev = fgsea_to_evidence_table(raw, config=config)

    ev_out = ev.copy()
    ev_out["evidence_genes"] = ev_out["evidence_genes"].map(lambda xs: ",".join(xs))
    ev_out.to_csv(out_path, sep="\t", index=False)
    return ev
