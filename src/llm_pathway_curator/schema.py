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
    # "pathway": "term_id",
    # "geneset": "term_id",
    # "gene_set": "term_id",
    # "set": "term_id",
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
    # some tools only export p-value; we map to pval then (optionally) to qval
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
    # -------------------------
    # optional flags (not required; used to tolerate summary rows)
    # -------------------------
    "is_summary": "is_summary",
    "summary": "is_summary",
    "group_id": "group_id",
    "groupid": "group_id",
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
        # Normalize common messiness while keeping it conservative.
        c = c.strip().lstrip("\ufeff")
        c = c.replace(" ", "_").replace("-", "_")
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
            # tolerate tabs/newlines
            s = s.replace("\n", " ").replace("\t", " ")
            s = " ".join(s.split()).strip()

            if not s or s.lower() in {"na", "nan", "none"}:
                return []

            # If comma-delimited, prefer commas.
            if "," in s:
                parts = s.split(",")
            else:
                # Space-separated fallback ONLY if all tokens look gene-like.
                parts0 = s.split(" ")
                import re

                gene_tok = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
                if parts0 and all(bool(gene_tok.match(tok)) for tok in parts0):
                    parts = parts0
                else:
                    # treat as a single field (avoid destructive split)
                    parts = [s]

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

    @staticmethod
    def _normalize_bool(x: object) -> bool:
        if EvidenceTable._is_na_scalar(x):
            return False
        s = str(x).strip().lower()
        return s in {"1", "true", "t", "yes", "y"}

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
    def read_tsv(
        cls, path: str, *, strict: bool = False, drop_invalid: bool = True
    ) -> EvidenceTable:
        """
        Read an evidence table and normalize it to the tool-facing contract.

        strict:
          - if True, any invalid row triggers ValueError (old behavior).
          - if False (default), invalid rows are optionally dropped with provenance.

        drop_invalid:
          - if True (default), drop rows that violate term×gene (empty evidence_genes),
            unless strict=True (then we raise).
          - if False, keep rows but mark them invalid; downstream should ignore them.
        """
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

        # qval / pval (best-effort)
        qval_in_input = "qval" in cols_mapped
        pval_in_input = "pval" in cols_mapped

        if "qval" not in df.columns:
            df["qval"] = pd.NA
        if "pval" not in df.columns:
            df["pval"] = pd.NA

        # direction: ORA often lacks it
        if "direction" not in df.columns:
            df["direction"] = "na"

        # optional flags
        if "is_summary" in df.columns:
            df["is_summary"] = df["is_summary"].map(cls._normalize_bool)
        else:
            df["is_summary"] = False

        if "group_id" not in df.columns:
            df["group_id"] = pd.NA

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
        df["pval"] = pd.to_numeric(df["pval"], errors="coerce")

        # ---- qval fallback policy ----
        # If qval is missing but pval exists, compute q-values by Benjamini–Hochberg (BH)
        # within conservative groups (source × direction) to preserve interpretability.
        #
        # Provenance:
        #   - "qval": provided in input
        #   - "bh(pval)": computed from p-values (BH)
        #   - "missing": neither qval nor pval available
        df["qval_source"] = "missing"

        # mark provided qvals
        if qval_in_input:
            df.loc[~df["qval"].isna(), "qval_source"] = "qval"

        # helper: BH q-value (no external deps)
        def _bh_qvalues(p: pd.Series) -> pd.Series:
            p = pd.to_numeric(p, errors="coerce")
            m = int(p.notna().sum())
            if m == 0:
                return pd.Series([pd.NA] * len(p), index=p.index, dtype="float64")

            # ranks among non-NA
            p_non = p.dropna()
            order = p_non.sort_values().index
            ranks = pd.Series(range(1, len(order) + 1), index=order, dtype="float64")

            q_non = p_non.loc[order] * (m / ranks.loc[order])
            # enforce monotonicity
            q_non = q_non.iloc[::-1].cummin().iloc[::-1]
            q_non = q_non.clip(lower=0.0, upper=1.0)

            q = pd.Series(pd.NA, index=p.index, dtype="float64")
            q.loc[q_non.index] = q_non
            return q

        # compute BH qvals only where qval is missing but pval exists
        needs_q = df["qval"].isna() & (~df["pval"].isna())

        # Safety: do NOT BH-correct rows with unknown source (mixed DB risk).
        needs_q = needs_q & (df["source"].astype(str).str.strip() != "unknown")

        if needs_q.any():
            # conservative grouping: source × direction
            group_cols = ["source", "direction"]
            for _, idx in df.loc[needs_q].groupby(group_cols).groups.items():
                idx = list(idx)
                q_calc = _bh_qvalues(df.loc[idx, "pval"])
                df.loc[idx, "qval"] = q_calc
                df.loc[idx, "qval_source"] = "bh(pval)"

        # provenance flags (stable, downstream-friendly)
        df["qval_provided"] = df["qval_source"].eq("qval")
        df["pval_provided"] = pval_in_input

        # ---- required sanity checks ----
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

        # ---- enforce term×gene contract (tolerate + drop/mark invalid) ----
        df["is_valid"] = True
        df["invalid_reason"] = ""

        empty_ev = df["evidence_genes"].map(len).eq(0)

        # Summary rows are common in ORA exports; they often have no gene list.
        # We never pass empty gene evidence downstream; we either drop or mark invalid.
        invalid_mask = empty_ev
        if invalid_mask.any():
            df.loc[invalid_mask, "is_valid"] = False
            df.loc[invalid_mask, "invalid_reason"] = "EMPTY_EVIDENCE_GENES"

            if strict:
                i = int(df.index[invalid_mask][0])
                raise ValueError(
                    f"EvidenceTable has empty evidence_genes at row index={i} "
                    f"(term_id={df.loc[i, 'term_id']}). "
                    f"read_mode={rr.read_mode}. "
                    "Fix: provide overlap genes (ORA) or leadingEdge/core_enrichment (GSEA/fgsea), "
                    "or set strict=False to drop/mark summary rows."
                )

            if drop_invalid:
                df = df.loc[df["is_valid"]].copy()

        # provenance
        df.attrs["read_mode"] = rr.read_mode

        # provenance / health hints (paper-facing)
        genes_n = (
            df["evidence_genes"].map(len)
            if "evidence_genes" in df.columns
            else pd.Series([], dtype=int)
        )
        df.attrs["health"] = {
            "n_terms": int(df.shape[0]),
            "n_terms_genes_le1": int((genes_n <= 1).sum()) if len(genes_n) else 0,
            "genes_per_term_median": float(genes_n.median()) if len(genes_n) else 0.0,
            "qval_source_counts": df["qval_source"].value_counts(dropna=False).to_dict()
            if "qval_source" in df.columns
            else {},
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
            "genes_per_term_p10": float(genes_n.quantile(0.1)) if len(genes_n) else 0.0,
            "n_terms_genes_le1": int((genes_n <= 1).sum()),
            "read_mode": df.attrs.get("read_mode", "unknown"),
            "qval_source_counts": df["qval_source"].value_counts(dropna=False).to_dict()
            if "qval_source" in df.columns
            else {},
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
