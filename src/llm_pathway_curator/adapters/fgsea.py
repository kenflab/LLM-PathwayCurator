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
    s = str(c).strip().lstrip("\ufeff")
    s = s.replace(" ", "").replace("_", "")
    return s.lower()


def _is_na(x: Any) -> bool:
    # Spec-level NA policy lives in _shared.
    return _shared.is_na_scalar(x) or _shared.is_na_token(x)


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


_R_C_WRAPPER_RE = re.compile(r"^\s*c\s*\((.*)\)\s*$", flags=re.IGNORECASE)


def _split_genes(x: Any) -> list[str]:
    # Spec-level gene parsing lives in _shared (handles ',', ';', '|', bracketed lists, etc.).
    # Also tolerate common R vector syntax like: c(10, 20, 30)
    if isinstance(x, str):
        m = _R_C_WRAPPER_RE.match(x)
        if m:
            x = m.group(1)
    return _shared.parse_genes(x)


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
    # Spec-level short hash (12-hex) to keep identity rules consistent across layers.
    return _shared.sha256_12hex(s)


@dataclass(frozen=True)
class FgseaAdapterConfig:
    """
    Adapter configuration for converting an fgsea result table into EvidenceTable.

    Defaults are chosen to match the paper's R-side EvidenceTable behavior:
    - term_id is human-readable (raw pathway name)
    - rows are sorted by qval asc, abs(stat) desc
    - qval NA rows are dropped
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
    raw = read_fgsea_table(in_path)
    ev = fgsea_to_evidence_table(raw, config=config)

    ev_out = ev.copy()
    ev_out["evidence_genes"] = ev_out["evidence_genes"].map(_shared.join_genes_tsv)
    ev_out.to_csv(out_path, sep="\t", index=False)
    return ev_out
