# LLM-PathwayCurator/src/llm_pathway_curator/_shared.py
from __future__ import annotations

import hashlib
import re

import pandas as pd

# -----------------------------------------------------------------------------
# _shared.py (minimal)
# -----------------------------------------------------------------------------
# Purpose:
#   - Centralize "spec-level" utilities that affect contract/boundary/reproducibility.
#   - If these change, outputs (IDs/hashes/edges/modules/audits) may change.
#
# Policy:
#   - Keep conservative parsing to avoid destructive splits.
#   - Do NOT force uppercasing (species- / ID-system-dependent).
#   - Prefer explicit delimiters; whitespace split only if all tokens look gene-like.
# -----------------------------------------------------------------------------


# NA tokens used across layers for gene-list parsing (string forms).
NA_TOKENS: set[str] = {"", "na", "nan", "none", "NA"}

# Conservative gene-like token heuristic (used for whitespace-separated fallback).
GENE_TOKEN_RE_STR = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
GENE_TOKEN_RE = re.compile(GENE_TOKEN_RE_STR)

# Recognize bracketed/list-like strings to strip wrappers (no eval).
BRACKETED_LIST_RE = re.compile(r"^\s*[\[\(\{].*[\]\)\}]\s*$")


def is_na_scalar(x: object) -> bool:
    """
    pd.isna is unsafe for list-like; only treat scalars as NA here.
    """
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def dedup_preserve_order(items: list[str]) -> list[str]:
    """
    Deterministic de-duplication while preserving first occurrence order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def clean_gene_token(g: object) -> str:
    """
    Clean a single gene-like token conservatively.

    NOTE: Do NOT force uppercase.
      - Mouse gene symbols can be case-sensitive (e.g., Trp53).
      - Ensembl/other IDs may appear in mixed case.
    Canonicalization beyond trimming should be an explicit, opt-in mapping step.
    """
    s = str(g).strip().strip('"').strip("'")
    s = " ".join(s.split())

    # Strip common wrapping punctuation from list-like exports.
    s = s.strip(",;|")
    s = s.strip("[](){}")
    # Some exports keep trailing commas/brackets mixed.
    s = s.strip(",;|").strip("[](){}")
    return s


def split_gene_string(s: str) -> list[str]:
    """
    Split a gene string into tokens with conservative rules.

    Supports common real-world formats:
      - "A,B,C"
      - "A;B;C"
      - "A|B|C"
      - "A/B/C" (last-resort, only when no other delimiter present)
      - "['A', 'B']" / '["A","B"]' / "{A,B}"
      - whitespace separated (ONLY if all tokens look gene-like)
    """
    s = str(s).strip()
    if not s:
        return []

    # Remove a single leading quote that sometimes appears (TSV quoting issues)
    if s.startswith("'") and len(s) > 1:
        s = s[1:].strip()

    # Tolerate explicit list-like wrappers: [..], (..), {..}
    # We do NOT eval; just strip wrappers if present.
    if BRACKETED_LIST_RE.match(s):
        s = s.strip().lstrip("[({").rstrip("])}").strip()

    # Normalize whitespace
    s = s.replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split()).strip()
    if not s:
        return []

    # Prefer explicit delimiters first
    if any(d in s for d in [",", ";", "|"]):
        s = s.replace(";", ",").replace("|", ",")
    elif "/" in s:
        # Use "/" only as a last-resort delimiter (reduces accidental splits)
        s = s.replace("/", ",")
    else:
        # no delimiter
        pass

    if "," in s:
        parts = [p for p in s.split(",")]
    else:
        parts0 = [p for p in s.split(" ") if p]
        # Space-separated fallback ONLY if all tokens look gene-like.
        if parts0 and all(bool(GENE_TOKEN_RE.match(tok.strip())) for tok in parts0):
            parts = parts0
        else:
            # treat as a single field (avoid destructive split)
            parts = [s]

    return parts


def parse_genes(x: object) -> list[str]:
    """
    Parse evidence genes from messy inputs into a list[str].

    Rules:
      - NA scalars -> []
      - list/tuple/set -> cleaned tokens
      - string -> split conservatively -> cleaned tokens
      - de-duplicate while preserving order
    """
    if is_na_scalar(x):
        return []

    # Already list-like
    if isinstance(x, (list, tuple, set)):
        genes = [clean_gene_token(g) for g in x]
        genes = [g for g in genes if g and str(g).strip()]
        return dedup_preserve_order([str(g) for g in genes if str(g).strip()])

    s = str(x).strip()
    if not s or s.lower() in NA_TOKENS:
        return []

    parts = split_gene_string(s)
    genes = [clean_gene_token(g) for g in parts]
    genes = [g for g in genes if g and str(g).strip()]
    return dedup_preserve_order([str(g) for g in genes if str(g).strip()])


# -----------------------------------------------------------------------------
# Identity / hashing (spec-level)
# -----------------------------------------------------------------------------

_HEX12_RE = re.compile(r"^[0-9a-f]{12}$")


def looks_like_12hex(x: object) -> bool:
    """
    True iff x is exactly 12 lowercase hex chars.
    Used for tool-owned IDs/hashes (sha256[:12]).
    """
    if x is None:
        return False
    s = str(x).strip().lower()
    return bool(_HEX12_RE.match(s))


def sha256_12hex(payload: str) -> str:
    """
    Deterministic short hash used across layers (sha256[:12]).
    Payload must be a stable string representation.
    """
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()[:12]


def hash_gene_set_12hex(genes: list[object]) -> str:
    """
    Gene-set fingerprint (12-hex), set-stable and trim-only.

    Policy:
      - order-invariant
      - dedup
      - trim/unwrap via clean_gene_token
      - NO forced uppercasing (species/ID-system dependent)

    NOTE:
      This is the "evidence identity" hash and should match distill/modules intent.
    """
    uniq = sorted({clean_gene_token(g) for g in genes if g is not None and str(g).strip()})
    payload = ",".join([g for g in uniq if g])
    return sha256_12hex(payload)


def canonical_sorted_unique(xs: list[object]) -> list[str]:
    """
    Stable canonicalization for ID payloads:
      - strip
      - drop empty/NA tokens
      - dedup
      - sort
    """
    out = []
    for x in xs:
        if x is None:
            continue
        s = str(x).strip()
        if not s or s.lower() in NA_TOKENS:
            continue
        out.append(s)
    return sorted(set(out))


def make_term_uid(source: object, term_id: object) -> str:
    """
    Single source of truth for term_uid construction.

    Contract:
      term_uid := "<source>:<term_id>" with conservative trimming.
      - empty source -> "unknown"
      - term_id is required (caller should ensure non-empty)
    """
    s = "" if source is None else str(source).strip()
    if not s or s.lower() in NA_TOKENS:
        s = "unknown"
    t = "" if term_id is None else str(term_id).strip()
    return f"{s}:{t}" if t else f"{s}:"


def hash_set_12hex(items: list[object]) -> str:
    """
    Generic set fingerprint (12-hex), set-stable and trim-only.

    Policy:
      - order-invariant
      - dedup
      - strip
      - drop empty/NA-like tokens (shared NA_TOKENS)
    """
    uniq = canonical_sorted_unique(items)
    payload = ",".join(uniq)
    return sha256_12hex(payload)


def module_hash_content12(terms: list[object], genes: list[object]) -> str:
    """
    Module content hash binds module identity to BOTH term set and gene set (12-hex).

    Spec:
      - terms: canonical_sorted_unique (no uppercasing)
      - genes: hash_gene_set_12hex's canonicalization (clean_gene_token + sort/dedup)
      - payload format is stable and explicit
    """
    t = canonical_sorted_unique(terms)
    g = canonical_sorted_unique(
        [clean_gene_token(x) for x in genes if x is not None and str(x).strip()]
    )
    payload = "T:" + "|".join(t) + "\n" + "G:" + "|".join(g)
    return sha256_12hex(payload)
