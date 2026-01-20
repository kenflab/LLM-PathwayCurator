# LLM-PathwayCurator/src/llm_pathway_curator/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

# v1 contract (tool-facing): we MUST preserve term×gene.
# Practical reality:
# - ORA often has no direction
# - some exports have no qval (or only p-value)
# We therefore enforce a "core required" set and auto-fill the rest.
CORE_REQUIRED_EVIDENCE_COLS = [
    "term_id",
    "term_name",
    "stat",
    "evidence_genes",
]

# Columns that should exist after normalization (auto-filled if missing).
NORMALIZED_COLS = [
    "term_id",
    "term_name",
    "source",
    "stat",
    "qval",
    "direction",
    "evidence_genes",
]

# Keep aliases conservative but cover common real-world headers.
ALIASES = {
    # -------------------------
    # term id/name
    # -------------------------
    "term": "term_id",
    "termid": "term_id",
    "term_id": "term_id",
    "id": "term_id",
    "pathway": "term_id",
    "geneset": "term_id",
    "gene_set": "term_id",
    "set": "term_id",
    "term_identifier": "term_id",
    "termid_": "term_id",
    # names/descriptions
    "description": "term_name",
    "name": "term_name",
    "term_name": "term_name",
    "desc": "term_name",
    "term_description": "term_name",
    "termname": "term_name",
    "gene_set_name": "term_name",
    # -------------------------
    # source / database
    # -------------------------
    "source": "source",
    "database": "source",
    "db": "source",
    "category": "source",
    "collection": "source",
    # -------------------------
    # stat (effect/strength)
    # -------------------------
    "nes": "stat",
    "score": "stat",
    "stat": "stat",
    "logp": "stat",
    "log_p": "stat",
    "log(p)": "stat",
    "-log10p": "stat",
    "-log10_p": "stat",
    "-log10(p)": "stat",
    "-log10q": "stat",
    "-log10_q": "stat",
    "-log10(q)": "stat",
    "logq": "stat",
    "log(q)": "stat",
    "log_q": "stat",
    # -------------------------
    # qval / pval (allow both; we normalize to qval)
    # -------------------------
    "qval": "qval",
    "q_value": "qval",
    "q-value": "qval",
    "qvalue": "qval",
    "fdr": "qval",
    "fdr_q": "qval",
    "padj": "qval",
    "p.adjust": "qval",
    "adj_p": "qval",
    "adj_pval": "qval",
    "adj_p_value": "qval",
    # some tools only export p-value; we map to qval as "best available"
    "pval": "pval",
    "p_value": "pval",
    "p-value": "pval",
    "p": "pval",
    # -------------------------
    # direction
    # -------------------------
    "direction": "direction",
    "dir": "direction",
    "sign": "direction",
    # -------------------------
    # evidence genes
    # -------------------------
    "leadingedge": "evidence_genes",
    "leading_edge": "evidence_genes",
    "leading edge": "evidence_genes",
    "leadingedgegenes": "evidence_genes",
    "core_enrichment": "evidence_genes",
    "core enrichment": "evidence_genes",
    "genes": "evidence_genes",
    "gene": "evidence_genes",
    "symbols": "evidence_genes",
    "overlap": "evidence_genes",
    "overlap_genes": "evidence_genes",
    "overlap genes": "evidence_genes",
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
        c = c.replace("-", "_")
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
        return "na"

    @staticmethod
    def _clean_gene_symbol(g: str) -> str:
        s = g.strip().strip('"').strip("'")
        s = " ".join(s.split())
        s = s.strip(",;|")
        return s.upper()

    @classmethod
    def _parse_genes(cls, x: object) -> list[str]:
        if cls._is_na_scalar(x):
            return []

        # Already list-like
        if isinstance(x, (list, tuple, set)):
            genes = [cls._clean_gene_symbol(str(g)) for g in x]
            genes = [g for g in genes if g]
        else:
            s = str(x).strip()
            if not s or s.lower() in {"na", "nan", "none"}:
                return []
            # normalize common delimiters
            s = s.replace(";", ",").replace("|", ",")
            # sometimes exported as space-separated tokens without commas
            if "," not in s and " " in s:
                parts = s.split()
            else:
                parts = s.split(",")
            genes = [cls._clean_gene_symbol(g) for g in parts]
            genes = [g for g in genes if g]

        # de-duplicate while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for g in genes:
            if g not in seen:
                seen.add(g)
                out.append(g)
        return out

    @classmethod
    def _read_flexible(cls, path: str) -> EvidenceReadResult:
        # robust defaults: BOM, comments, gzip, do not auto-convert NA tokens
        common_kwargs = dict(
            encoding="utf-8-sig",
            comment="#",
            compression="infer",
            keep_default_na=False,
            na_values=[],
            low_memory=False,
        )

        # 1) TSV first (expected)
        df = pd.read_csv(path, sep="\t", **common_kwargs)
        if df.shape[1] > 1:
            return EvidenceReadResult(df=df, read_mode="tsv")

        # 2) sniff delimiter (csv/tsv/others)
        df2 = pd.read_csv(path, sep=None, engine="python", **common_kwargs)
        if df2.shape[1] > 1:
            return EvidenceReadResult(df=df2, read_mode="sniff")

        # 3) whitespace fallback
        df3 = pd.read_csv(path, sep=r"\s+", engine="python", **common_kwargs)
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
                "EvidenceTable has duplicate columns after aliasing: "
                f"{sorted(set(dup))}. "
                f"read_mode={rr.read_mode}. "
                f"raw_columns={list(df_raw.columns)} "
                f"mapped_columns={list(df.columns)}"
            )

        # ---- auto-fill optional normalized columns ----
        # source
        if "source" not in df.columns:
            df["source"] = "unknown"

        if "qval" not in df.columns:
            df["qval"] = pd.NA

        if "pval" not in df.columns:
            df["pval"] = pd.NA

        # provenance flag: whether qval was truly provided
        df["qval_provided"] = "qval" in cols_mapped

        # direction: ORA often lacks it
        if "direction" not in df.columns:
            df["direction"] = "na"

        # evidence_genes: must exist
        # term_id / term_name / stat: must exist
        missing_core = [c for c in CORE_REQUIRED_EVIDENCE_COLS if c not in df.columns]
        if missing_core:
            raise ValueError(
                f"EvidenceTable missing core columns: {missing_core}. "
                f"read_mode={rr.read_mode}. "
                f"raw_columns={list(df_raw.columns)} "
                f"mapped_columns={list(df.columns)}"
            )

        # ensure all normalized columns exist (even if not used downstream)
        for c in NORMALIZED_COLS:
            if c not in df.columns:
                df[c] = pd.NA

        # ---- clean required string fields ----
        df["term_id"] = df["term_id"].map(cls._clean_required_str)
        df["term_name"] = df["term_name"].map(cls._clean_required_str)
        df["source"] = df["source"].map(cls._clean_required_str)
        if (df["source"] == "").any():
            df.loc[df["source"] == "", "source"] = "unknown"

        # ---- direction normalization ----
        df["direction"] = df["direction"].map(cls._normalize_direction)

        # ---- evidence genes parsing ----
        df["evidence_genes"] = df["evidence_genes"].map(cls._parse_genes)
        df["evidence_genes_str"] = df["evidence_genes"].map(lambda xs: ",".join(xs))

        # ---- numeric normalization ----
        df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
        df["qval"] = pd.to_numeric(df["qval"], errors="coerce")

        bad_required = df["term_id"].eq("") | df["term_name"].eq("")
        if bad_required.any():
            i = int(df.index[bad_required][0])
            raise ValueError(
                f"EvidenceTable has empty required fields at row index={i}. "
                f"read_mode={rr.read_mode}. "
                "Fix: ensure term_id/term_name are non-empty."
            )

        if df["stat"].isna().any():
            i = int(df.index[df["stat"].isna()][0])
            bad = df.loc[i, ["term_id", "term_name"]].to_dict()
            raise ValueError(
                f"EvidenceTable has non-numeric stat at row index={i} (row={bad}). "
                f"read_mode={rr.read_mode}. "
                "Fix: provide a numeric stat (e.g., NES, -log10(q), LogP)."
            )

        empty_ev = df["evidence_genes"].map(len).eq(0)
        if empty_ev.any():
            i = int(df.index[empty_ev][0])
            raise ValueError(
                f"EvidenceTable has empty evidence_genes at row index={i} "
                f"(term_id={df.loc[i, 'term_id']}). "
                f"read_mode={rr.read_mode}. "
                "Fix: provide overlap genes (ORA) or leadingEdge/core_enrichment (GSEA/fgsea)."
            )

        # provenance
        df.attrs["read_mode"] = rr.read_mode

        # provenance / health hints (paper-facing)
        genes_n = df["evidence_genes"].map(len)
        df.attrs["health"] = {
            "n_terms": int(df.shape[0]),
            "n_terms_genes_le1": int((genes_n <= 1).sum()),
            "genes_per_term_median": float(genes_n.median()) if len(genes_n) else 0.0,
        }

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
            "genes_per_term_p10": float(genes_n.quantile(0.1)) if len(genes_n) else 0.0,
            "n_terms_genes_le1": int((genes_n <= 1).sum()),
        }

    def write_tsv(self, path: str) -> None:
        out = self.df.copy()

        # robust stringify: handle both list-like and scalar strings
        if "evidence_genes_str" in out.columns:
            ev_str = out["evidence_genes_str"]
        else:

            def _stringify(x: object) -> str:
                if self._is_na_scalar(x):
                    return ""
                if isinstance(x, (list, tuple, set)):
                    return ",".join([str(g) for g in x])
                return str(x)

            ev_str = (
                out["evidence_genes"].map(_stringify) if "evidence_genes" in out.columns else ""
            )

        out = out.drop(columns=["evidence_genes"], errors="ignore")
        out["evidence_genes"] = ev_str
        out.to_csv(path, sep="\t", index=False)
