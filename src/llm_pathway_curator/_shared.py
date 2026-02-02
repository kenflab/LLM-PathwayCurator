# LLM-PathwayCurator/src/llm_pathway_curator/_shared.py
from __future__ import annotations

# -----------------------------------------------------------------------------
# SPEC POLICY (DO NOT CHANGE LIGHTLY)
# -----------------------------------------------------------------------------
# This module defines "spec-level" helpers that can affect:
#   - IDs / hashes
#   - module factorization outputs
#   - audit outcomes / denominators
#   - TSV round-trips (parse <-> join)
#
# Rule of thumb:
#   - If a change can alter report.jsonl content or evidence identity, it belongs here.
#   - Convenience helpers that only affect UI/formatting should live in utils.py/report.py.
# -----------------------------------------------------------------------------
import hashlib
import json
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
    Check whether a value represents an NA token (case-insensitive).

    This is a spec-level helper used across parsing and TSV round-trips.
    The NA vocabulary is centralized to prevent contract drift.

    Parameters
    ----------
    s : object
        Input value.

    Returns
    -------
    bool
        True if `s` is None or its trimmed lowercase string form is in the
        NA token set.

    Notes
    -----
    This function treats empty strings as NA.
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

    The join is stable and order-preserving. This function is intentionally
    *not* gene-aware to avoid over-normalization at the spec boundary.

    Parameters
    ----------
    ids : list of object
        Identifiers to join. None/empty/NA-like tokens are dropped.
    delim : str, optional
        Delimiter for joining. Default is `ID_JOIN_DELIM`.

    Returns
    -------
    str
        Joined identifier string.

    Notes
    -----
    - Preserves input order (no sorting).
    - Does not apply `clean_gene_token()`.
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
    Strip the Excel "force text" prefix from a value.

    Excel-safe exports sometimes prefix values with a single quote (').
    This helper removes one leading quote to support downstream parsing.

    Parameters
    ----------
    s : object
        Input value.

    Returns
    -------
    str
        Cleaned string without a single leading quote.
    """

    ss = str(s or "").strip()
    if ss.startswith("'") and len(ss) > 1:
        return ss[1:].strip()
    return ss


def excel_force_text(s: object) -> str:
    """
    Prefix a value with a single quote to force Excel to treat it as text.

    Parameters
    ----------
    s : object
        Input value.

    Returns
    -------
    str
        Excel-safe text representation. Empty input returns "".
    """

    txt = str(s or "")
    if not txt:
        return ""
    return txt if txt.startswith("'") else ("'" + txt)


def excel_safe_ids(x: object, *, list_sep: str = ID_JOIN_DELIM) -> str:
    """
    Convert an ID field into an Excel-safe, TSV-friendly text string.

    This helper accepts either scalar or list-like inputs, parses them via
    `parse_id_list()`, joins the IDs with `list_sep`, and prefixes a single
    quote to force Excel "Text" interpretation.

    Parameters
    ----------
    x : object
        Scalar or list-like ID field.
    list_sep : str, optional
        Join delimiter for the ID list. Default is `ID_JOIN_DELIM`.

    Returns
    -------
    str
        Excel-safe text value. Returns "" if the input is NA-like or empty.
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
    Normalize a status value into canonical uppercase text.

    Parameters
    ----------
    x : object
        Input scalar.

    Returns
    -------
    str
        Uppercased, trimmed string.

    Notes
    -----
    This function does not validate membership in `ALLOWED_STATUSES`.
    Use `validate_status_values()` for strict checking.
    """

    s = str(x or "").strip().upper()
    return s


def normalize_status_series(s: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series of statuses to uppercase strings.

    Parameters
    ----------
    s : pandas.Series
        Input series.

    Returns
    -------
    pandas.Series
        Series with string dtype, trimmed and uppercased.

    Notes
    -----
    NA values may become strings (e.g., "nan") after `astype(str)`.
    Always validate with `validate_status_values()` when needed.
    """

    return s.astype(str).str.strip().str.upper()


def validate_status_values(s_norm: pd.Series) -> None:
    """
    Strict validation: refuse unknown status values (auditable denominators).
    """
    bad = sorted(set(s_norm.unique().tolist()) - ALLOWED_STATUSES)
    if bad:
        raise ValueError(f"invalid status values: {bad} (allowed={sorted(ALLOWED_STATUSES)})")


# -----------------------------------------------------------------------------
# Gate mode vocabulary (spec-level)
# -----------------------------------------------------------------------------
# Canonical gate modes used across pipeline/select/audit/sample_card.
# NOTE: We ACCEPT many legacy/synonym inputs, but we EMIT only canonical values.
ALLOWED_GATE_MODES = {"off", "note", "hard"}


def normalize_gate_mode(x: object, *, default: str = "note") -> str:
    """
    Normalize a gate mode to canonical vocabulary: {"off", "note", "hard"}.

    Parameters
    ----------
    x : object
        Input value (canonical, synonym, or legacy form).
    default : str, optional
        Default to use when `x` is empty. If invalid, falls back to "note".

    Returns
    -------
    str
        Canonical gate mode: "off", "note", or "hard".

    Notes
    -----
    Accepted synonyms include:
    - off: off, none, disable, disabled
    - note: note, warn, warning, soft
    - hard: hard, strict, abstain, on, enable, enabled
    """

    s = ("" if x is None else str(x)).strip().lower()
    if not s:
        s = str(default).strip().lower()

    if s in {"off", "none", "disable", "disabled"}:
        return "off"
    if s in {"note", "warn", "warning", "soft"}:
        return "note"
    if s in {"hard", "strict", "abstain", "on", "enable", "enabled"}:
        return "hard"

    d = str(default).strip().lower()
    if d in ALLOWED_GATE_MODES:
        return d
    return "note"


# Conservative gene-like token heuristic (used for whitespace-separated fallback).
GENE_TOKEN_RE_STR = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
GENE_TOKEN_RE = re.compile(GENE_TOKEN_RE_STR)

# Recognize bracketed/list-like strings to strip wrappers (no eval).
BRACKETED_LIST_RE = re.compile(r"^\s*[\[\(\{].*[\]\)\}]\s*$")


# Generic ID-like token (whitespace split fallback for parse_id_list)
ID_TOKEN_RE_STR = r"^[A-Za-z0-9_.:-]+$"
ID_TOKEN_RE = re.compile(ID_TOKEN_RE_STR)


def is_na_scalar(x: object) -> bool:
    """
    Determine whether a value should be treated as NA *as a scalar*.

    This function avoids calling `pandas.isna` on list-like containers
    because it can return array-like results and break boolean contexts.

    Parameters
    ----------
    x : object
        Input value.

    Returns
    -------
    bool
        True if `x` is a scalar NA value (or None). Containers return False.

    Notes
    -----
    Strings like "na"/"nan" are not treated as scalar NA here; use
    `is_na_token()` for token-level NA checks.
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
    De-duplicate strings while preserving first occurrence order.

    Parameters
    ----------
    items : list of str
        Input tokens.

    Returns
    -------
    list of str
        Deduplicated tokens in first-seen order.

    Notes
    -----
    Empty strings are ignored.
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
    Parse a generic ID field into a list of strings.

    This is a tolerant parser for ID-like fields (term IDs, module IDs,
    gene IDs when treated as IDs, etc.). It is intentionally separate from
    `parse_genes()`, which is more gene-token-aware.

    Parameters
    ----------
    x : object
        Scalar or list-like input.

    Returns
    -------
    list of str
        Parsed IDs in deterministic order.

    Notes
    -----
    Policy:
    - NA scalars -> []
    - list/tuple -> preserve order (dedup)
    - set -> sorted for determinism (dedup)
    - string -> split on strong delimiters first: ',', ';', '|'
    - whitespace split only if all tokens look identifier-like
    - drop NA tokens and empties
    """

    if is_na_scalar(x):
        return []

    if isinstance(x, (list, tuple, set)):
        # Determinism: sets are unordered; sort to avoid non-reproducible outputs.
        it = list(x)
        if isinstance(x, set):
            it = sorted([str(t) for t in it], key=lambda z: z)
        items = [str(t).strip() for t in it if str(t).strip()]
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
                if parts0 and all(bool(ID_TOKEN_RE.match(tok)) for tok in parts0):
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

    Parameters
    ----------
    g : object
        Gene-like token.

    Returns
    -------
    str
        Cleaned token.

    Notes
    -----
    - Trims whitespace and strips simple quote wrappers.
    - Removes common list/export wrappers (brackets, trailing separators).
    - Does NOT force uppercase (species/ID-system dependent).
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
    Split a gene string into candidate tokens using conservative rules.

    Parameters
    ----------
    s : str
        Input gene string.

    Returns
    -------
    list of str
        Token candidates (not yet fully cleaned).

    Notes
    -----
    Supported formats:
    - Comma/semicolon/pipe separated: "A,B", "A;B", "A|B"
    - Bracketed lists: "['A','B']", '["A","B"]', "{A,B}"
    - Slash-separated as a last resort: "A/B/C"
    - Whitespace-separated only if all tokens look gene-like
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
    Parse evidence genes from messy inputs into a list of cleaned tokens.

    Parameters
    ----------
    x : object
        Scalar or list-like gene field.

    Returns
    -------
    list of str
        Cleaned gene tokens, deduplicated in first-seen order.

    Notes
    -----
    Rules:
    - NA scalars -> []
    - list/tuple -> cleaned per-token
    - set -> sorted for determinism, then cleaned
    - string -> split conservatively via `split_gene_string()`
    """
    if is_na_scalar(x):
        return []

    if isinstance(x, (list, tuple, set)):
        # Determinism: sets are unordered; sort to avoid non-reproducible outputs.
        it = list(x)
        if isinstance(x, set):
            it = sorted([str(t) for t in it], key=lambda z: z)

        genes = [clean_gene_token(g) for g in it]
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

    Parameters
    ----------
    x : object
        Input scalar.

    Returns
    -------
    str
        One of {"up", "down", "na"}.

    Notes
    -----
    This is a lightweight normalizer. Unrecognized values map to "na".
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
    Create a deterministic per-term integer seed.

    The seed is derived from `(seed, term_uid, term_row_id)` using a stable
    hash to keep RNG streams reproducible across platforms.

    Parameters
    ----------
    seed : int or None
        Optional base seed. None maps to 0.
    term_uid : str
        Stable term identifier (e.g., "<source>:<term_id>").
    term_row_id : int or None, optional
        Optional row identifier to avoid collisions for duplicate term_uids.

    Returns
    -------
    int
        Deterministic unsigned integer seed.

    Raises
    ------
    ValueError
        If `term_row_id` cannot be converted to int (when provided).
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
    Split a stress tag string into normalized tags.

    Parameters
    ----------
    s : object
        Input scalar tag string.
    delim : str, optional
        Canonical delimiter. Default is `STRESS_TAG_DELIM` (comma).

    Returns
    -------
    list of str
        Tags in first-seen order.

    Notes
    -----
    - Canonical delimiter is comma.
    - Legacy '+' is tolerated as an additional delimiter.
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
    Join tags into a canonical stress tag string.

    Parameters
    ----------
    tags : list of object
        Tag tokens.
    delim : str, optional
        Join delimiter. Default is `STRESS_TAG_DELIM` (comma).

    Returns
    -------
    str
        Canonical tag string.

    Notes
    -----
    Trims whitespace, drops empties, and de-duplicates in first-seen order.
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
    Check whether a value is exactly 12 lowercase hex characters.

    Parameters
    ----------
    x : object
        Input value.

    Returns
    -------
    bool
        True if `x` matches the 12-hex pattern (lowercase).
    """
    if x is None:
        return False
    s = str(x).strip().lower()
    return bool(_HEX12_RE.match(s))


def sha256_12hex(payload: str) -> str:
    """
    Compute a deterministic short SHA-256 hash (first 12 hex chars).

    Parameters
    ----------
    payload : str
        Stable string payload.

    Returns
    -------
    str
        12-character lowercase hex digest.
    """
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()[:12]


def hash_gene_set_12hex(genes: list[object]) -> str:
    """
    Compute a set-stable gene-set fingerprint (12-hex), preserving case.

    Parameters
    ----------
    genes : list of object
        Gene tokens.

    Returns
    -------
    str
        12-character lowercase hex fingerprint.

    Notes
    -----
    Policy:
    - order-invariant (set-stable)
    - `clean_gene_token()` per token
    - no forced uppercasing (species/ID dependent)
    """
    uniq = sorted({clean_gene_token(g) for g in genes if g is not None and str(g).strip()})
    payload = ",".join([g for g in uniq if g])
    return sha256_12hex(payload)


def norm_gene_id_upper(g: object) -> str:
    """
    Normalize a gene token by applying conservative cleaning and uppercasing.

    Parameters
    ----------
    g : object
        Gene token.

    Returns
    -------
    str
        Cleaned and uppercased token.

    Notes
    -----
    This is opt-in for legacy compatibility. The default spec policy in this
    module is to preserve case.
    """
    return clean_gene_token(g).upper()


def hash_gene_set_12hex_upper(genes: list[object]) -> str:
    """
    Compute a legacy-compatible gene-set fingerprint (12-hex), uppercasing IDs.

    Parameters
    ----------
    genes : list of object
        Gene tokens.

    Returns
    -------
    str
        12-character lowercase hex fingerprint.

    Notes
    -----
    Use only when you must match older outputs that case-folded gene IDs.
    """
    uniq = sorted({norm_gene_id_upper(g) for g in genes if g is not None and str(g).strip()})
    payload = ",".join([g for g in uniq if g])
    return sha256_12hex(payload)


def canonical_sorted_unique(xs: list[object]) -> list[str]:
    """
    Canonicalize a list of values into sorted unique strings.

    Parameters
    ----------
    xs : list of object
        Input values.

    Returns
    -------
    list of str
        Sorted unique tokens after trimming and NA filtering.
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
    Construct a stable `term_uid` from `(source, term_id)`.

    Parameters
    ----------
    source : object
        Term source (e.g., "fgsea", "metascape"). Empty maps to "unknown".
    term_id : object
        Term identifier. Caller should ensure it is non-empty.

    Returns
    -------
    str
        Term UID formatted as "<source>:<term_id>".
    """
    s = "" if source is None else str(source).strip()
    if not s or is_na_token(s):
        s = "unknown"
    t = "" if term_id is None else str(term_id).strip()
    return f"{s}:{t}" if t else f"{s}:"


def hash_set_12hex(items: list[object]) -> str:
    """
    Compute a generic set-stable fingerprint (12-hex) from a list of items.

    Parameters
    ----------
    items : list of object
        Input items.

    Returns
    -------
    str
        12-character lowercase hex fingerprint.

    Notes
    -----
    Trims tokens, drops NA-like values, de-duplicates, sorts, then hashes.
    """
    uniq = canonical_sorted_unique(items)
    payload = ",".join(uniq)
    return sha256_12hex(payload)


def module_hash_content12(terms: list[object], genes: list[object]) -> str:
    """
    Compute a module content hash binding both term set and gene set (12-hex).

    Parameters
    ----------
    terms : list of object
        Term identifiers.
    genes : list of object
        Gene tokens.

    Returns
    -------
    str
        12-character lowercase hex fingerprint.

    Notes
    -----
    - Terms: `canonical_sorted_unique()` (no uppercasing)
    - Genes: `clean_gene_token()` + drop NA/empty + sort/dedup (no uppercasing)
    - Payload format is stable and explicit to prevent ambiguity.
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
    Serialize an object to deterministic JSON for hashing/provenance.

    Parameters
    ----------
    obj : object
        JSON-serializable object.

    Returns
    -------
    str
        Deterministic JSON string.

    Notes
    -----
    Uses:
    - sort_keys=True
    - separators=(",", ":")
    - ensure_ascii=False
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_short(obj: object, n: int = 12) -> str:
    """
    Compute a deterministic SHA-256 short hash from an arbitrary payload.

    Parameters
    ----------
    obj : object
        Payload to hash. It is serialized via `stable_json_dumps()`.
    n : int, optional
        Number of hex characters to return. Default is 12.

    Returns
    -------
    str
        Lowercase hex digest prefix.

    Raises
    ------
    ValueError
        If `n` is not positive.

    Notes
    -----
    - For n == 12, this matches the legacy behavior (`sha256_12hex`).
    - SHA-256 hex digests have length 64; if n > 64, the output length is
      effectively capped at 64 by Python slicing.
    """
    nn = 12 if n is None else int(n)
    if nn <= 0:
        raise ValueError(f"sha256_short n must be positive; got {nn}")

    s = stable_json_dumps(obj)

    # Legacy fast path (exactly matches previous outputs)
    if nn == 12:
        return sha256_12hex(s)

    # Full sha256 for general n
    full = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
    return full[:nn]


def seed_int_from_payload(payload: object, *, mod: int = 2**31 - 1) -> int:
    """
    Derive a deterministic integer seed from an arbitrary payload.

    Parameters
    ----------
    payload : object
        Any JSON-serializable payload.
    mod : int, optional
        Modulus for the resulting seed. Default is 2**31 - 1.

    Returns
    -------
    int
        Deterministic integer seed in [0, mod).

    Notes
    -----
    Uses `sha256_short(..., n=12)` to keep stability aligned with other IDs.
    """
    h = sha256_short(payload, n=12)
    return int(h, 16) % int(mod)


def join_genes_tsv(genes: list[object]) -> str:
    """
    Join gene tokens into a TSV-friendly string.

    Parameters
    ----------
    genes : list of object
        Gene tokens.

    Returns
    -------
    str
        Genes joined by `GENE_JOIN_DELIM`.

    Notes
    -----
    Applies `clean_gene_token()` and drops empty/NA tokens. Does not sort;
    preserves input order.
    """
    xs: list[str] = []
    for g in genes or []:
        if g is None:
            continue
        raw = str(g).strip()
        if not raw:
            continue
        s = str(clean_gene_token(raw)).strip()
        if not s or is_na_token(s):
            continue
        xs.append(s)
    return GENE_JOIN_DELIM.join(xs)


# Public spec API (do not change lightly)
__all__ = [
    # -------------------------
    # NA / parsing (spec)
    # -------------------------
    "NA_TOKENS",
    "NA_TOKENS_L",
    "is_na_token",
    "is_na_scalar",
    "dedup_preserve_order",
    "parse_id_list",
    "BRACKETED_LIST_RE",
    "ID_TOKEN_RE",
    "GENE_JOIN_DELIM",
    "GENE_TOKEN_RE",
    "clean_gene_token",
    "split_gene_string",
    "parse_genes",
    "join_genes_tsv",
    # -------------------------
    # Vocab (spec)
    # -------------------------
    "ALLOWED_STATUSES",
    "normalize_status_str",
    "normalize_status_series",
    "validate_status_values",
    "normalize_direction",
    "ALLOWED_GATE_MODES",
    "normalize_gate_mode",
    # -------------------------
    # Tags (spec)
    # -------------------------
    "STRESS_TAG_DELIM",
    "split_tags",
    "join_tags",
    # -------------------------
    # Identity / hashing / IDs / seeds (spec)
    # -------------------------
    "looks_like_12hex",
    "sha256_12hex",
    "stable_json_dumps",
    "sha256_short",
    "hash_gene_set_12hex",
    "norm_gene_id_upper",
    "hash_gene_set_12hex_upper",
    "canonical_sorted_unique",
    "hash_set_12hex",
    "make_term_uid",
    "module_hash_content12",
    "seed_for_term",
    "seed_int_from_payload",
    # -------------------------
    # TSV / Excel-safe joins (spec)
    # -------------------------
    "ID_JOIN_DELIM",
    "join_id_list_tsv",
    "strip_excel_text_prefix",
    "excel_force_text",
    "excel_safe_ids",
]
