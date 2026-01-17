# LLM-PathwayCurator/src/llm_pathway_curator/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

REQUIRED_EVIDENCE_COLS = [
    "term_id",
    "term_name",
    "source",
    "stat",
    "qval",
    "direction",
    "evidence_genes",
]

ALIASES = {
    "term": "term_id",
    "id": "term_id",
    "description": "term_name",
    "name": "term_name",
    "nes": "stat",
    "padj": "qval",
    "fdr": "qval",
    "leadingedge": "evidence_genes",
    "leading_edge": "evidence_genes",
    "genes": "evidence_genes",
}

ReadMode = Literal["tsv", "sniff", "whitespace"]


@dataclass(frozen=True)
class EvidenceReadResult:
    df: pd.DataFrame
    read_mode: ReadMode


@dataclass(frozen=True)
class EvidenceTable:
    df: pd.DataFrame

    @staticmethod
    def _normalize_col(c: str) -> str:
        c = c.strip().lstrip("\ufeff")
        c = c.replace(" ", "_")
        return c.lower()

    @staticmethod
    def _is_na_scalar(x: object) -> bool:
        """pd.isna is unsafe for list-like; only treat scalars here."""
        if x is None:
            return True
        if isinstance(x, (list, tuple, set, dict)):
            return False
        try:
            return bool(pd.isna(x))
        except Exception:
            return False

    @staticmethod
    def _clean_required_str(x: object) -> str:
        if EvidenceTable._is_na_scalar(x):
            return ""
        s = str(x).strip()
        if s.lower() in {"na", "nan", "none"}:
            return ""
        return s

    @staticmethod
    def _normalize_direction(x: object) -> str:
        if EvidenceTable._is_na_scalar(x):
            return "na"
        s = str(x).strip().lower()
        if s in {
            "up",
            "upregulated",
            "increase",
            "increased",
            "activated",
            "+",
            "pos",
            "positive",
            "1",
        }:
            return "up"
        if s in {
            "down",
            "downregulated",
            "decrease",
            "decreased",
            "suppressed",
            "-",
            "neg",
            "negative",
            "-1",
        }:
            return "down"
        if s in {"na", "none", "", "nan"}:
            return "na"
        return "na"

    @staticmethod
    def _clean_gene_symbol(g: str) -> str:
        # minimal, conservative cleaning
        s = g.strip().strip('"').strip("'")
        # collapse internal whitespace
        s = " ".join(s.split())
        # common artifacts: trailing commas or semicolons already split, but be safe
        s = s.strip(",;|")
        return s

    @classmethod
    def _parse_genes(cls, x: object) -> list[str]:
        if cls._is_na_scalar(x):
            return []
        if isinstance(x, (list, tuple, set)):
            genes = [cls._clean_gene_symbol(str(g)) for g in x]
            genes = [g for g in genes if g]
        else:
            s = str(x).strip()
            if not s or s.lower() in {"na", "nan", "none"}:
                return []
            s = s.replace(";", ",").replace("|", ",")
            genes = [cls._clean_gene_symbol(g) for g in s.split(",")]
            genes = [g for g in genes if g]

        seen: set[str] = set()
        out: list[str] = []
        for g in genes:
            if g not in seen:
                seen.add(g)
                out.append(g)
        return out

    @classmethod
    def _read_flexible(cls, path: str) -> EvidenceReadResult:
        # 1) TSV first (expected)
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] > 1:
            return EvidenceReadResult(df=df, read_mode="tsv")

        # 2) sniff delimiter (csv/tsv/others)
        df2 = pd.read_csv(path, sep=None, engine="python")
        if df2.shape[1] > 1:
            return EvidenceReadResult(df=df2, read_mode="sniff")

        # 3) whitespace fallback
        df3 = pd.read_csv(path, sep=r"\s+", engine="python")
        return EvidenceReadResult(df=df3, read_mode="whitespace")

    @classmethod
    def read_tsv(cls, path: str) -> EvidenceTable:
        rr = cls._read_flexible(path)
        df_raw = rr.df

        cols_norm = [cls._normalize_col(c) for c in df_raw.columns]
        cols_mapped = [ALIASES.get(c, c) for c in cols_norm]

        df = df_raw.copy()
        df.columns = cols_mapped

        dup = df.columns[df.columns.duplicated()].tolist()
        if dup:
            raise ValueError(
                f"EvidenceTable has duplicate columns after aliasing: {sorted(set(dup))}. "
                f"Columns={list(df.columns)}"
            )

        missing = [c for c in REQUIRED_EVIDENCE_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"EvidenceTable missing columns: {missing}. Found columns: {list(df.columns)}"
            )

        df["term_id"] = df["term_id"].map(cls._clean_required_str)
        df["term_name"] = df["term_name"].map(cls._clean_required_str)
        df["source"] = df["source"].map(cls._clean_required_str)

        df["direction"] = df["direction"].map(cls._normalize_direction)
        df["evidence_genes"] = df["evidence_genes"].map(cls._parse_genes)
        df["evidence_genes_str"] = df["evidence_genes"].map(lambda xs: ",".join(xs))

        df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
        df["qval"] = pd.to_numeric(df["qval"], errors="coerce")

        bad_required = df["term_id"].eq("") | df["term_name"].eq("") | df["source"].eq("")
        if bad_required.any():
            i = int(df.index[bad_required][0])
            raise ValueError(
                f"EvidenceTable has empty required fields at row index={i}. "
                f"Fix: ensure term_id/term_name/source are non-empty."
            )

        if df["stat"].isna().any():
            i = int(df.index[df["stat"].isna()][0])
            bad = df.loc[i, ["term_id", "term_name", "source"]].to_dict()
            raise ValueError(
                f"EvidenceTable has non-numeric stat at row index={i} (row={bad}). "
                f"Fix: provide a numeric stat (e.g., NES, -log10(q), LogP)."
            )

        empty_ev = df["evidence_genes"].map(len).eq(0)
        if empty_ev.any():
            i = int(df.index[empty_ev][0])
            raise ValueError(
                f"EvidenceTable has empty evidence_genes at row index={i} "
                f"(term_id={df.loc[i, 'term_id']}). "
                f"Fix: provide overlap genes (ORA) or leadingEdge (GSEA/fgsea)."
            )

        # Attach read_mode for provenance (non-breaking: stored as attribute on df)
        df.attrs["read_mode"] = rr.read_mode

        return cls(df=df)

    def summarize(self) -> dict[str, object]:
        df = self.df
        genes_n = df["evidence_genes"].map(len)
        return {
            "n_terms": int(df.shape[0]),
            "n_sources": int(df["source"].nunique()),
            "sources": sorted(df["source"].astype(str).unique().tolist()),
            "direction_counts": df["direction"].value_counts(dropna=False).to_dict(),
            "genes_per_term_median": float(genes_n.median()) if len(genes_n) else 0.0,
            "genes_per_term_p90": float(genes_n.quantile(0.9)) if len(genes_n) else 0.0,
            "read_mode": df.attrs.get("read_mode", "unknown"),
        }

    def write_tsv(self, path: str) -> None:
        out = self.df.copy()
        if "evidence_genes_str" not in out.columns:
            out["evidence_genes_str"] = out["evidence_genes"].map(lambda xs: ",".join(xs))
        out = out.drop(columns=["evidence_genes"], errors="ignore")
        out = out.rename(columns={"evidence_genes_str": "evidence_genes"})
        out.to_csv(path, sep="\t", index=False)
