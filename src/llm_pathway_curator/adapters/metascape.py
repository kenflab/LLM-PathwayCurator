# LLM-PathwayCurator/src/llm_pathway_curator/adapters/metascape.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

METASCAPE_COLS = [
    "GroupID",
    "Category",
    "Term",
    "Description",
    "LogP",
    "Log(q-value)",
    "Genes",
    "Symbols",
    "InTerm_InList",
]


def _is_na(x: Any) -> bool:
    try:
        if x is None:
            return True
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
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _split_genes(x: Any) -> list[str]:
    if _is_na(x):
        return []
    if isinstance(x, (list, tuple, set)):
        genes = [str(g).strip() for g in x if str(g).strip()]
        return sorted(set(genes))
    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    s = s.replace(";", ",").replace("|", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return sorted(set(parts))


def _parse_interm_inlist(x: Any) -> tuple[int | None, int | None]:
    # "123/456" -> (123,456)
    s = _clean_str(x)
    if not s or "/" not in s:
        return (None, None)
    a, b = s.split("/", 1)
    try:
        return (int(a), int(b))
    except Exception:
        return (None, None)


def _is_summary_groupid(gid: str) -> bool:
    # examples: "1_Summary", "2_Member"
    s = (gid or "").strip()
    return s.endswith("_Summary")


@dataclass(frozen=True)
class MetascapeAdapterConfig:
    source_name: str = "metascape"
    include_summary: bool = False  # IMPORTANT default
    prefer_symbols: bool = True  # Symbols preferred over Genes


def read_metascape_table(path: str) -> pd.DataFrame:
    # Try TSV first
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

    missing = [c for c in METASCAPE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"metascape_to_evidence_table: missing columns: {missing}. Found={list(df.columns)}"
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

    # stat: prefer Log(q-value) if numeric else LogP
    logq = df["Log(q-value)"].map(_to_float)
    logp = df["LogP"].map(_to_float)
    stat = logq.where(logq.notna(), logp)

    if stat.isna().any():
        i = int(df.index[stat.isna()][0])
        raise ValueError(
            f"metascape_to_evidence_table: non-numeric Log(q-value)/LogP at row index={i}"
        )

    # qval: if Log(q-value) exists numeric => 10^(-logq) else NA
    def _q_from_logq(v: Any) -> float | None:
        vv = _to_float(v)
        if vv is None:
            return None
        # Metascape typically uses log10(q-value)
        return float(10 ** (-vv))

    qval = df["Log(q-value)"].map(_q_from_logq)

    # evidence genes: prefer Symbols
    if config.prefer_symbols:
        genes_src = df["Symbols"]
        fallback = df["Genes"]
    else:
        genes_src = df["Genes"]
        fallback = df["Symbols"]

    genes = genes_src.map(_split_genes)
    empty = genes.map(len).eq(0)
    if empty.any():
        # fallback to the other column
        genes2 = fallback.map(_split_genes)
        genes = genes.where(~empty, genes2)

    empty2 = genes.map(len).eq(0)
    if empty2.any():
        i = int(df.index[empty2][0])
        tid = df.loc[i, "term_id_raw"]
        raise ValueError(
            f"metascape_to_evidence_table: empty Symbols/Genes at row index={i} (Term={tid})"
        )

    n_in_term, n_in_list = zip(*df["InTerm_InList"].map(_parse_interm_inlist), strict=False)

    out = pd.DataFrame(
        {
            "term_id": df["term_id_raw"].map(lambda s: f"GO:{s}" if s.startswith("GO") else s),
            "term_name": df["term_name"],
            "source": config.source_name,
            "stat": stat.astype(float),
            "qval": qval,
            "direction": "na",
            "evidence_genes": genes,
            "group_id": df["group_id"],
            "is_summary": df["is_summary"].astype(bool),
            "n_in_term": list(n_in_term),
            "n_in_list": list(n_in_list),
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
    raw = read_metascape_table(in_path)
    ev = metascape_to_evidence_table(raw, config=config)

    ev_out = ev.copy()
    ev_out["evidence_genes"] = ev_out["evidence_genes"].map(lambda xs: ",".join(xs))
    ev_out.to_csv(out_path, sep="\t", index=False)
    return ev
