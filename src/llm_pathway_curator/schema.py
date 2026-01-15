# LLM-PathwayCurator/src/llm_pathway_curator/schema.py
from __future__ import annotations

from dataclasses import dataclass

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
        # pandas considers strings as scalars; lists/tuples/sets are not
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
    def _parse_genes(x: object) -> list[str]:
        # IMPORTANT: avoid pd.isna(x) on list-like (can raise)
        if EvidenceTable._is_na_scalar(x):
            return []
        if isinstance(x, (list, tuple, set)):
            genes = [str(g).strip() for g in x if str(g).strip()]
        else:
            s = str(x).strip()
            if not s or s.lower() in {"na", "nan", "none"}:
                return []
            s = s.replace(";", ",").replace("|", ",")
            genes = [g.strip() for g in s.split(",") if g.strip()]

        seen: set[str] = set()
        out: list[str] = []
        for g in genes:
            if g not in seen:
                seen.add(g)
                out.append(g)
        return out

    @classmethod
    def _read_flexible(cls, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] > 1:
            return df
        df2 = pd.read_csv(path, sep=None, engine="python")
        if df2.shape[1] > 1:
            return df2
        return pd.read_csv(path, sep=r"\s+", engine="python")

    @classmethod
    def read_tsv(cls, path: str) -> EvidenceTable:
        df_raw = cls._read_flexible(path)

        cols_norm = [cls._normalize_col(c) for c in df_raw.columns]
        cols_mapped = [ALIASES.get(c, c) for c in cols_norm]

        df = df_raw.copy()
        df.columns = cols_mapped

        # NEW: detect duplicate columns after aliasing (v0 safest)
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

        # NEW: always regenerate to keep consistent with parsed genes
        df["evidence_genes_str"] = df["evidence_genes"].map(lambda xs: ",".join(xs))

        df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
        df["qval"] = pd.to_numeric(df["qval"], errors="coerce")

        bad_required = df["term_id"].eq("") | df["term_name"].eq("") | df["source"].eq("")
        if bad_required.any():
            i = int(df.index[bad_required][0])
            raise ValueError(f"EvidenceTable has empty required fields at row index={i}")

        if df["stat"].isna().any():
            i = int(df.index[df["stat"].isna()][0])
            raise ValueError(f"EvidenceTable has non-numeric stat at row index={i}")

        empty_ev = df["evidence_genes"].map(len).eq(0)
        if empty_ev.any():
            i = int(df.index[empty_ev][0])
            raise ValueError(
                f"EvidenceTable has empty evidence_genes at row index={i} "
                f"(term_id={df.loc[i, 'term_id']})"
            )

        # Optional (recommended): qval sanity (do not hard-fail; just clip or warn later)
        # if ((df["qval"].notna()) & ((df["qval"] < 0) | (df["qval"] > 1))).any():
        #     ...

        return cls(df=df)

    def write_tsv(self, path: str) -> None:
        out = self.df.copy()
        if "evidence_genes_str" not in out.columns:
            out["evidence_genes_str"] = out["evidence_genes"].map(lambda xs: ",".join(xs))
        out = out.drop(columns=["evidence_genes"], errors="ignore")
        out = out.rename(columns={"evidence_genes_str": "evidence_genes"})
        out.to_csv(path, sep="\t", index=False)
