# LLM-PathwayCurator/src/llm_pathway_curator/adapters/fgsea.py
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

ALIASES: dict[str, str] = {
    "pathway": "pathway",
    "term": "pathway",
    "termname": "pathway",
    "term_name": "pathway",
    "name": "pathway",
    "geneset": "pathway",
    "set": "pathway",
    "id": "pathway",
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


def _clean_gene_symbol(g: str) -> str:
    s = g.strip().strip('"').strip("'")
    s = " ".join(s.split())
    s = s.strip(",;|")
    return s


def _split_genes(x: Any) -> list[str]:
    if _is_na(x):
        return []
    if isinstance(x, (list, tuple, set)):
        parts = [_clean_gene_symbol(str(g)) for g in x]
        parts = [p for p in parts if p]
        return sorted(set(parts))

    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []

    s = s.replace("c(", "").replace(")", "").replace('"', "").replace("'", "")

    for sep in [";", "|", "\t"]:
        s = s.replace(sep, ",")

    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]

    parts = [_clean_gene_symbol(p) for p in parts]
    parts = [p for p in parts if p]
    return sorted(set(parts))


def _term_slug(s: str) -> str:
    # minimal slug: keep alnum + _-. replace others with _
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
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]


@dataclass(frozen=True)
class FgseaAdapterConfig:
    source_name: str = "fgsea"
    require_genes: bool = True
    keep_pval: bool = True  # store pval separately (does not replace qval)


def read_fgsea_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] > 1:
        return df
    df2 = pd.read_csv(path, sep=None, engine="python")
    if df2.shape[1] > 1:
        return df2
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _rename_with_aliases(df: pd.DataFrame) -> pd.DataFrame:
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
    if config is None:
        config = FgseaAdapterConfig()

    df = _rename_with_aliases(fgsea_df.copy())

    missing = [c for c in REQUIRED_CORE if c not in df.columns]
    if missing:
        raise ValueError(
            f"fgsea_to_evidence_table: missing required columns: "
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
        i = int(df.index[empty_pw][0])
        raise ValueError(f"fgsea_to_evidence_table: empty pathway at row index={i}")

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

    df["__stat"] = df[stat_col].map(_to_float)
    if df["__stat"].isna().any():
        i = int(df.index[df["__stat"].isna()][0])
        raise ValueError(f"fgsea_to_evidence_table: non-numeric {stat_col} at row index={i}")

    # qval: ONLY padj maps to qval (FDR). pval stored separately if present.
    df["__qval"] = df["padj"].map(_to_float) if "padj" in df.columns else pd.NA
    pval = df["pval"].map(_to_float) if ("pval" in df.columns and config.keep_pval) else pd.NA

    # direction: prefer NES sign if NES exists
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

    # stable term_id: FGSEA:<slug>|<hash>
    term_hash = df["__pathway"].map(_term_hash)
    term_slug = df["__pathway"].map(_term_slug)

    out = pd.DataFrame(
        {
            "term_id": term_slug.combine(term_hash, lambda a, b: f"FGSEA:{a}|{b}"),
            "term_name": df["__pathway"],
            "source": config.source_name,
            "stat": df["__stat"].astype(float),
            "qval": df["__qval"],
            "direction": df["direction"],
            "evidence_genes": df["evidence_genes"],
            # optional provenance fields
            "q_kind": "padj" if "padj" in df.columns else "na",
            "pval": pval,
            "term_id_h": term_hash,
        }
    ).reset_index(drop=True)

    return out


def convert_fgsea_table_to_evidence_tsv(
    in_path: str,
    out_path: str,
    *,
    config: FgseaAdapterConfig | None = None,
) -> pd.DataFrame:
    raw = read_fgsea_table(in_path)
    ev = fgsea_to_evidence_table(raw, config=config)

    ev_out = ev.copy()
    ev_out["evidence_genes"] = ev_out["evidence_genes"].map(lambda xs: ",".join(xs))
    ev_out.to_csv(out_path, sep="\t", index=False)
    return ev
