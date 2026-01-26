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
NA_TOKENS_L: set[str] = {t.lower() for t in NA_TOKENS}

# Single source of truth for TSV-friendly gene join delimiter across layers.
# Use ';' (rare in gene symbols; safer than ',' which is often CSV delimiter).
GENE_JOIN_DELIM = ";"


def is_na_token(s: object) -> bool:
    """
    Spec-level NA token check (case-insensitive).
    Keep this centralized to avoid contract drift.
    """
    if s is None:
        return True
    return str(s).strip().lower() in NA_TOKENS_L


# -----------------------------------------------------------------------------
# TSV-friendly joins (spec-level)
# -----------------------------------------------------------------------------

# Generic ID join delimiter for TSV exports (term_uids, module_ids, etc.).
# Keep this stable: changing it can break downstream parsing.
ID_JOIN_DELIM = ";"


def join_id_list_tsv(ids: list[object], *, delim: str = ID_JOIN_DELIM) -> str:
    """
    Join generic identifiers into a TSV-friendly string.

    Policy:
      - strip whitespace
      - drop empty/NA tokens
      - preserve order (do NOT sort)
      - do NOT apply gene-specific cleaning (clean_gene_token), to avoid over-normalization
    """
    xs: list[str] = []
    for x in ids or []:
        if x is None:
            continue
        s = str(x).strip()
        if not s or is_na_token(s):
            continue
        xs.append(s)
    return delim.join(xs)


def strip_excel_text_prefix(s: object) -> str:
    """
    Excel-safe text fields sometimes start with a single quote.
    Strip it for downstream parsing (e.g., json.loads).
    """
    ss = str(s or "").strip()
    if ss.startswith("'") and len(ss) > 1:
        return ss[1:].strip()
    return ss


def excel_force_text(s: object) -> str:
    """
    Force Excel to treat a field as Text by prefixing a single quote.
    """
    txt = str(s or "")
    if not txt:
        return ""
    return txt if txt.startswith("'") else ("'" + txt)


def excel_safe_ids(x: object, *, list_sep: str = ID_JOIN_DELIM) -> str:
    """
    Make an ID field safe for Excel:
      - Accept list-like or scalar.
      - Parse with parse_id_list (spec-level).
      - Join with list_sep.
      - Prefix with a single quote to force Text in Excel.
    """
    if is_na_scalar(x):
        return ""

    ids = parse_id_list(x)
    if not ids:
        return ""

    joined = join_id_list_tsv(ids, delim=list_sep)
    if not joined:
        return ""
    return excel_force_text(joined)


# -----------------------------------------------------------------------------
# Decision status (spec-level)
# -----------------------------------------------------------------------------
# Single source of truth for decision vocabulary across audit/report/calibrate.
ALLOWED_STATUSES = {"PASS", "ABSTAIN", "FAIL"}


def normalize_status_str(x: object) -> str:
    """
    Normalize a status scalar to canonical uppercase string.
    This is spec-level: changing this changes denominators/metrics downstream.
    """
    s = str(x or "").strip().upper()
    return s


def normalize_status_series(s: pd.Series) -> pd.Series:
    """
    Normalize a status column to uppercase strings (vectorized).
    NOTE: pd.NA/NaN become strings after astype(str); validation must catch them.
    """
    return s.astype(str).str.strip().str.upper()


def validate_status_values(s_norm: pd.Series) -> None:
    """
    Strict validation: refuse unknown status values (auditable denominators).
    """
    bad = sorted(set(s_norm.unique().tolist()) - ALLOWED_STATUSES)
    if bad:
        raise ValueError(f"invalid status values: {bad} (allowed={sorted(ALLOWED_STATUSES)})")


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


def parse_id_list(x: object) -> list[str]:
    """
    Parse generic ID fields (term_ids, gene_ids, etc.) into list[str].

    Policy:
      - NA scalars -> []
      - list/tuple/set -> preserve order (dedup)
      - string -> split on strong delimiters first: ',', ';', '|', '\\t', '\\n'
      - whitespace split ONLY if all tokens look identifier-like (to avoid destructive splits)
      - normalize whitespace/newlines/tabs
      - drop NA tokens and empties
      - deterministic de-duplication (preserve first-seen order)

    NOTE:
      This is intentionally separate from parse_genes():
        - parse_genes() is conservative and gene-token-aware
        - parse_id_list() is generic/tolerant for ID-like fields
    """
    if is_na_scalar(x):
        return []

    if isinstance(x, (list, tuple, set)):
        items = [str(t).strip() for t in x if str(t).strip()]
    else:
        s = str(x).strip()
        if s.startswith("'") and len(s) > 1:
            s = s[1:].strip()
        if not s or is_na_token(s):
            return []

        # normalize whitespace
        s = s.replace("\n", " ").replace("\t", " ")
        s = " ".join(s.split()).strip()
        if not s or is_na_token(s):
            return []

        # strong separators first
        if any(sep in s for sep in ("|", ";", ",")):
            s2 = s.replace("|", ",").replace(";", ",")
            items = [t.strip() for t in s2.split(",") if t.strip()]
        else:
            # whitespace split only if all tokens look identifier-like
            if any(ch.isspace() for ch in s):
                parts0 = [p for p in s.split() if p.strip()]
                _ID_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]+$")
                if parts0 and all(bool(_ID_TOKEN.match(tok)) for tok in parts0):
                    items = [p.strip() for p in parts0 if p.strip()]
                else:
                    items = [s]
            else:
                items = [s]

    out: list[str] = []
    seen: set[str] = set()
    for t in items:
        tt = str(t).strip()
        if not tt or is_na_token(tt):
            continue
        if tt not in seen:
            seen.add(tt)
            out.append(tt)

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
    if not s or is_na_token(s):
        return []

    parts = split_gene_string(s)
    genes = [clean_gene_token(g) for g in parts]
    genes = [g for g in genes if g and str(g).strip() and (not is_na_token(g))]
    return dedup_preserve_order([str(g) for g in genes if str(g).strip()])


def normalize_direction(x: object) -> str:
    """
    Normalize direction vocabulary across schema/distill/audit/select.

    Output:
      - "up" / "down" / "na"
    """
    if is_na_scalar(x):
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


def seed_for_term(seed: int | None, term_uid: str, term_row_id: int | None = None) -> int:
    """
    Order-invariant deterministic seed derived from (seed, term_uid, term_row_id).

    NOTE:
      - Using blake2b for stable cross-platform hashing.
      - term_row_id avoids collisions when term_uid duplicates exist.
    """
    base = 0 if seed is None else int(seed)
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode("utf-8"))
    h.update(b"|")
    h.update(str(term_uid).encode("utf-8"))
    if term_row_id is not None:
        h.update(b"|")
        h.update(str(int(term_row_id)).encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


# -----------------------------------------------------------------------------
# Stress tags (spec-level)
# -----------------------------------------------------------------------------
# Canonical delimiter is comma. We also tolerate legacy '+' in inputs.
STRESS_TAG_DELIM = ","


def split_tags(s: object, *, delim: str = STRESS_TAG_DELIM) -> list[str]:
    """
    Split a stress_tag string into normalized tags.

    Spec:
      - canonical delimiter is comma
      - tolerate legacy '+' as an additional delimiter
      - trim whitespace
      - drop empty tokens
      - de-duplicate while preserving order
    """
    if s is None:
        return []
    txt = str(s).strip()
    if not txt:
        return []

    # tolerate legacy '+'
    txt = txt.replace("+", delim)

    parts = [p.strip() for p in txt.split(delim)]
    parts = [p for p in parts if p]
    return dedup_preserve_order(parts)


def join_tags(tags: list[object], *, delim: str = STRESS_TAG_DELIM) -> str:
    """
    Join tags into canonical stress_tag string.

    Spec:
      - trim
      - drop empties
      - preserve first-seen order
      - join with canonical delimiter
    """
    cleaned: list[str] = []
    for t in tags or []:
        if t is None:
            continue
        s = str(t).strip()
        if not s:
            continue
        cleaned.append(s)
    cleaned = dedup_preserve_order(cleaned)
    return delim.join(cleaned)


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


def norm_gene_id_upper(g: object) -> str:
    """
    Explicit uppercase normalization for gene IDs/symbols.

    IMPORTANT:
      - This is an opt-in policy for legacy compatibility in layers that already case-fold.
      - Default parsing/hash helpers in _shared remain "preserve-case" by default.
    """
    return clean_gene_token(g).upper()


def hash_gene_set_12hex_upper(genes: list[object]) -> str:
    """
    Legacy-compatible gene-set fingerprint (12-hex), set-stable and UPPERCASE-normalized.

    Use this ONLY when you must match older outputs that case-folded gene IDs.
    """
    uniq = sorted({norm_gene_id_upper(g) for g in genes if g is not None and str(g).strip()})
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
        if not s or is_na_token(s):
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
    if not s or is_na_token(s):
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
      - genes: same canonicalization policy as hash_gene_set_12hex
              (clean_gene_token + drop empties/NA + sort/dedup; NO uppercasing)
      - payload format is stable and explicit
    """
    t = canonical_sorted_unique(terms)

    # Align gene canonicalization with hash_gene_set_12hex (single policy).
    g_clean = []
    for x in genes or []:
        if x is None:
            continue
        s = clean_gene_token(x)
        if not s or is_na_token(s):
            continue
        g_clean.append(s)
    g = sorted(set(g_clean))

    payload = "T:" + "|".join(t) + "\n" + "G:" + "|".join(g)
    return sha256_12hex(payload)


# -------------------------
# TSV / hashing shared contracts (spec-level)
# -------------------------


def stable_json_dumps(obj: object) -> str:
    """
    Deterministic JSON serialization for hashing/provenance.
    - sort_keys=True for stable dict ordering
    - separators to avoid whitespace instability
    """
    import json

    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_short(obj: object, n: int = 12) -> str:
    """
    Deterministic short hash for arbitrary payloads (dict/list/str).
    Uses stable_json_dumps + sha256_12hex.
    """
    s = stable_json_dumps(obj)
    h12 = sha256_12hex(s)
    nn = 12 if n is None else int(n)
    return h12 if nn == 12 else h12[:nn]


def seed_int_from_payload(payload: object, *, mod: int = 2**31 - 1) -> int:
    """
    Deterministic integer seed from an arbitrary payload.
    Useful for per-term RNG streams that must be stable across runs.
    """
    h = sha256_short(payload, n=12)
    return int(h, 16) % int(mod)


def join_genes_tsv(genes: list[object]) -> str:
    """
    Join genes into a TSV-friendly string using GENE_JOIN_DELIM.
    Applies clean_gene_token() per token and drops empties.
    """
    xs = [
        str(clean_gene_token(g)).strip() for g in (genes or []) if g is not None and str(g).strip()
    ]
    return GENE_JOIN_DELIM.join([x for x in xs if x])


# Public spec API (do not change lightly)
__all__ = [
    # NA / parsing
    "NA_TOKENS",
    "NA_TOKENS_L",
    "GENE_JOIN_DELIM",
    "is_na_token",
    "is_na_scalar",
    "parse_id_list",
    "parse_genes",
    "join_genes_tsv",
    "clean_gene_token",
    "split_gene_string",
    # vocab
    "ALLOWED_STATUSES",
    "normalize_status_str",
    "normalize_status_series",
    "validate_status_values",
    "normalize_direction",
    # tags
    "STRESS_TAG_DELIM",
    "split_tags",
    "join_tags",
    # identity / hashing / ids / seeds
    "looks_like_12hex",
    "sha256_12hex",
    "sha256_short",
    "stable_json_dumps",
    "hash_gene_set_12hex",
    "hash_set_12hex",
    "module_hash_content12",
    "canonical_sorted_unique",
    "make_term_uid",
    "seed_for_term",
    "seed_int_from_payload",
    "dedup_preserve_order",
    "norm_gene_id_upper",
    "hash_gene_set_12hex_upper",
    # TSV joins (spec)
    "ID_JOIN_DELIM",
    "join_id_list_tsv",
    "strip_excel_text_prefix",
    "excel_force_text",
    "excel_safe_ids",
]
