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

# Minimal alias support (optional, can expand later)
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
        # strip whitespace + BOM, normalize to lower, replace spaces with underscore
        c = c.strip().lstrip("\ufeff")
        c = c.replace(" ", "_")
        return c.lower()

    @staticmethod
    def _normalize_direction(x: object) -> str:
        s = str(x).strip().lower()
        if s in {"up", "down", "activated", "suppressed"}:
            return s
        if s in {"na", "none", "", "nan"}:
            return "na"
        return "na"

    @staticmethod
    def _parse_genes(x: object) -> list[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(g).strip() for g in x if str(g).strip()]
        s = str(x).strip()
        if not s or s.lower() in {"na", "nan", "none"}:
            return []
        return [g.strip() for g in s.split(",") if g.strip()]

    @classmethod
    def _read_flexible(cls, path: str) -> pd.DataFrame:
        """
        Read TSV/CSV/whitespace-separated tables robustly.
        Handles common failure mode: everything becomes 1 column because sep mismatch.
        """
        # Try strict TSV first
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            # Maybe CSV or whitespace-delimited; try auto-sep
            df2 = pd.read_csv(path, sep=None, engine="python")
            if df2.shape[1] > 1:
                df = df2
            else:
                # last resort: whitespace
                df3 = pd.read_csv(path, sep=r"\s+", engine="python")
                if df3.shape[1] > 1:
                    df = df3
        return df

    @classmethod
    def read_tsv(cls, path: str) -> EvidenceTable:
        df = cls._read_flexible(path)

        # normalize column names + alias mapping
        cols_norm = [cls._normalize_col(c) for c in df.columns]
        cols_mapped = [ALIASES.get(c, c) for c in cols_norm]
        df = df.copy()
        df.columns = cols_mapped

        missing = [c for c in REQUIRED_EVIDENCE_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"EvidenceTable missing columns: {missing}. Found columns: {list(df.columns)}"
            )

        # basic hygiene
        df["term_id"] = df["term_id"].astype(str).str.strip()
        df["term_name"] = df["term_name"].astype(str).str.strip()
        df["source"] = df["source"].astype(str).str.strip()

        df["direction"] = df["direction"].map(cls._normalize_direction)

        df["evidence_genes_list"] = df["evidence_genes"].map(cls._parse_genes)
        df["evidence_genes"] = df["evidence_genes_list"].map(lambda xs: ",".join(xs))

        df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
        df["qval"] = pd.to_numeric(df["qval"], errors="coerce")

        bad = df["term_id"].eq("") | df["term_name"].eq("") | df["source"].eq("")
        if bad.any():
            i = df.index[bad][0]
            raise ValueError(f"EvidenceTable has empty required fields at row index={i}")

        if df["stat"].isna().any():
            i = df.index[df["stat"].isna()][0]
            raise ValueError(f"EvidenceTable has non-numeric stat at row index={i}")

        # qval may be NA; allowed
        return cls(df=df)
