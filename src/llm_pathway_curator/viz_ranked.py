# LLM-PathwayCurator/src/llm_pathway_curator/viz_ranked.py
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_pub_style(fontsize: int = 16) -> None:
    """Apply publication-style matplotlib rcParams.

    Parameters
    ----------
    fontsize : int, optional
        Base font size used for axes/labels/legend (default 16).

    Notes
    -----
    This mutates global matplotlib rcParams for the current Python process.
    Call once near the start of plotting.
    """
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize + 1,
            "axes.labelsize": fontsize + 2,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "figure.titlesize": fontsize + 4,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "axes.linewidth": 1.1,
        }
    )


def _die(msg: str) -> None:
    """
    Raise a fatal, user-facing error.

    Parameters
    ----------
    msg : str
        Error message to display.

    Returns
    -------
    None
        This function does not return; it raises SystemExit.

    Notes
    -----
    Kept as a small helper to preserve CLI-like behavior (fail fast with a clear
    single-line prefix) across plotting utilities.
    """
    raise SystemExit(msg)


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first candidate column name that exists in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    candidates : list of str
        Column names to check in order.

    Returns
    -------
    str or None
        First existing column name, or None if none exist.

    Notes
    -----
    Used to tolerate minor schema variations between `claims_ranked.tsv`,
    `audit_log.tsv`, and legacy outputs.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_str_series(s: pd.Series) -> pd.Series:
    """
    Convert a Series to string dtype, mapping missing values to empty strings.

    Parameters
    ----------
    s : pandas.Series
        Input series.

    Returns
    -------
    pandas.Series
        String-valued series with NaN/NA converted to "" (not the literal "nan").

    Notes
    -----
    This avoids accidental propagation of "nan" into labels, join keys, or IDs.
    """
    return s.where(pd.notna(s), "").astype(str)


def _col_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    """
    Get `df[col]` as a string Series aligned to `df.index`, or a default-filled Series.

    Parameters
    ----------
    df : pandas.DataFrame
        Source table.
    col : str
        Column name to retrieve. If empty or missing, a default series is returned.
    default : str, optional
        Value used to fill the returned series when `col` is missing.

    Returns
    -------
    pandas.Series
        A string series of length `len(df)` aligned to `df.index`.

    Notes
    -----
    - Never returns a scalar.
    - Uses `_safe_str_series` when the column exists.
    """
    if col and col in df.columns:
        return _safe_str_series(df[col])
    return pd.Series([default] * len(df), index=df.index, dtype=object)


def _get_cmap(name: str):
    """
    Resolve a Matplotlib colormap by name.

    Parameters
    ----------
    name : str
        Matplotlib colormap name (e.g., "tab20", "viridis", "coolwarm").

    Returns
    -------
    matplotlib.colors.Colormap
        Colormap instance.

    Notes
    -----
    This is a thin wrapper around `matplotlib.colormaps.get_cmap`.
    """
    return matplotlib.colormaps.get_cmap(name)


def _stable_color(key: str, cmap_name: str = "tab20") -> tuple[float, float, float, float]:
    """
    Map an arbitrary key to a deterministic categorical color.

    Parameters
    ----------
    key : str
        Hash key (typically module_id).
    cmap_name : str, optional
        Name of a categorical colormap. Only the first 20 entries are used.

    Returns
    -------
    tuple of float
        RGBA color in [0, 1], length 4.

    Notes
    -----
    - Determinism is achieved by hashing `key` (md5) and mapping to an index mod 20.
    - Empty/NA-like keys return a neutral gray.
    """
    cmap = _get_cmap(cmap_name)
    if key in {"", "NA", "nan", "NaN"}:
        return (0.75, 0.75, 0.75, 1.0)
    h = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % 20
    return cmap(idx)


def _mix_rgb(
    a: tuple[float, float, float], b: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    """
    Linearly interpolate between two RGB colors.

    Parameters
    ----------
    a : tuple of float
        RGB triple (r, g, b) for the start color, each in [0, 1].
    b : tuple of float
        RGB triple (r, g, b) for the end color, each in [0, 1].
    t : float
        Interpolation factor in [0, 1]. Values are clamped to [0, 1].

    Returns
    -------
    tuple of float
        Interpolated RGB triple.

    Notes
    -----
    Uses simple linear blending: (1 - t) * a + t * b.
    """
    t = max(0.0, min(1.0, float(t)))
    return (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t, a[2] * (1 - t) + b[2] * t)


def _dir_base_rgb(d: str) -> tuple[float, float, float]:
    """
    Convert a direction token to a base RGB color (up/down/na).

    Parameters
    ----------
    d : str
        Direction token (e.g., 'up', 'down', '+', '-', 'positive', 'negative').

    Returns
    -------
    tuple of float
        Base RGB triple (r, g, b) in [0, 1].

    Notes
    -----
    The palette is intentionally simple:
    - up   -> red-ish
    - down -> blue-ish
    - na   -> gray
    """
    d = str(d).strip().lower()
    if d in {"up", "+", "pos", "positive", "1"}:
        return (0.85, 0.20, 0.20)  # red
    if d in {"down", "-", "neg", "negative", "-1"}:
        return (0.20, 0.35, 0.85)  # blue
    return (0.55, 0.55, 0.55)  # gray


def _direction_shaded_color(
    d: str,
    score: float,
    vmin: float,
    vmax: float,
    *,
    bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    gamma: float = 1.0,
    min_t: float = 0.15,
) -> tuple[float, float, float, float]:
    """
    Build a direction-colored RGBA with score-dependent shading.

    Parameters
    ----------
    d : str
        Direction token (up/down/na-like).
    score : float
        Score to map into [vmin, vmax] for shading intensity.
    vmin : float
        Lower bound for score normalization.
    vmax : float
        Upper bound for score normalization.
    bg_rgb : tuple of float, optional
        Background RGB used as the "low intensity" anchor color.
    gamma : float, optional
        Power transform applied to normalized score to control contrast.
    min_t : float, optional
        Minimum blend factor to prevent colors becoming too faint.

    Returns
    -------
    tuple of float
        RGBA color (r, g, b, a) in [0, 1].

    Notes
    -----
    - Computes t = clip((score - vmin) / (vmax - vmin), 0, 1), then t := max(min_t, t**gamma).
    - Mixes bg_rgb -> base(direction) using t and returns alpha=1.0.
    - If score is non-finite, uses t=min_t.
    """
    base = _dir_base_rgb(d)
    if not np.isfinite(score):
        t = min_t
    elif vmax <= vmin:
        t = 1.0
    else:
        t = (float(score) - float(vmin)) / (float(vmax) - float(vmin))
        t = max(0.0, min(1.0, t))
        t = t ** max(1e-9, float(gamma))
        t = max(float(min_t), t)
    rgb = _mix_rgb(bg_rgb, base, t)
    return (rgb[0], rgb[1], rgb[2], 1.0)


def _simplify_term_label(raw: str, drop_hallmark_prefix: bool = True) -> str:
    """
    Normalize a term label for plotting readability.

    Parameters
    ----------
    raw : str
        Raw label, potentially of the form "ID: NAME" or an underscore-separated token.
    drop_hallmark_prefix : bool, optional
        If True, drop a leading "HALLMARK_" prefix.

    Returns
    -------
    str
        Cleaned label (human-readable, whitespace-normalized).

    Notes
    -----
    Rules:
    - If "A: B" is present, keep "B" unless A==B (then keep A).
    - Optionally drop "HALLMARK_" prefix.
    - Replace underscores with spaces; collapse repeated whitespace.
    """
    s = str(raw).strip()
    if not s:
        return s

    if ": " in s:
        a, b = s.split(": ", 1)
        a = a.strip()
        b = b.strip()
        s = a if a == b else b

    if drop_hallmark_prefix and s.startswith("HALLMARK_"):
        s = s[len("HALLMARK_") :]

    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def _shorten(s: str, n: int) -> str:
    """
    Truncate a string to at most `n` characters, using an ellipsis.

    Parameters
    ----------
    s : str
        Input string.
    n : int
        Maximum length in characters.

    Returns
    -------
    str
        Original string if `len(s) <= n`, else a truncated string ending with "…".

    Notes
    -----
    Uses a single Unicode ellipsis and keeps total length <= n.
    """
    s = str(s).strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def _wrap_words(s: str, width: int = 14) -> str:
    """
    Greedy word wrap for labels (no hyphenation).

    Parameters
    ----------
    s : str
        Input string.
    width : int, optional
        Maximum line length (characters) for each wrapped line.

    Returns
    -------
    str
        Wrapped string with newline separators.

    Notes
    -----
    This is intentionally simple and deterministic; it does not split long words.
    """
    words = str(s).split()
    if not words:
        return ""
    lines = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return "\n".join(lines)


def _find_input_tsv(run_dir: Path) -> Path:
    """
    Locate an input TSV for plotting under a run directory.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory to search.

    Returns
    -------
    pathlib.Path
        Path to the first found non-empty TSV in priority order.

    Notes
    -----
    Priority:
    1) `run_dir/claims_ranked.tsv`
    2) `run_dir/audit_log.tsv`
    If not found at the top level, recursively searches for either filename and
    returns the lexicographically smallest match. Raises on failure.
    """
    for name in ["claims_ranked.tsv", "audit_log.tsv"]:
        p = run_dir / name
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p
    hits = []
    for name in ["claims_ranked.tsv", "audit_log.tsv"]:
        hits.extend(run_dir.rglob(name))
    hits = [p for p in hits if p.is_file() and p.stat().st_size > 0]
    if hits:
        return sorted(hits)[0]
    _die(f"[plot-ranked] Could not find claims_ranked.tsv or audit_log.tsv under: {run_dir}")


def _normalize_claims_ranked(
    df: pd.DataFrame, drop_hallmark_prefix: bool
) -> tuple[pd.DataFrame, str]:
    """
    Normalize a `claims_ranked.tsv`-like table into a common plotting schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table read from claims_ranked.tsv (or compatible).
    drop_hallmark_prefix : bool
        Whether to drop "HALLMARK_" prefix in labels.

    Returns
    -------
    tbl : pandas.DataFrame
        Normalized table with columns:
        - module_id, term_label_raw, term_label, score, decision, direction
        - optionally module_rank if present in the input
    score_label : str
        Name of the input column used as the score source (e.g., "utility_score").

    Notes
    -----
    - Score column is required and auto-detected from: utility_score/evidence_strength/score.
    - Missing/empty IDs are normalized to "NA".
    - Negative scores are clipped to 0 to keep size/colormap stable.
    """
    module_col = _first_existing_col(df, ["module_id"]) or "module_id"
    label_col = _first_existing_col(df, ["term_label", "term_name", "term_id"]) or ""
    score_col = _first_existing_col(df, ["utility_score", "evidence_strength", "score"]) or ""
    dec_col = _first_existing_col(df, ["decision", "status"]) or ""
    dir_col = _first_existing_col(df, ["direction", "direction_norm", "dir"]) or ""

    if not score_col:
        _die("[plot-ranked] claims_ranked.tsv needs utility_score/evidence_strength/score.")

    out = pd.DataFrame(index=df.index)

    out["module_id"] = _col_series(df, module_col, default="NA").replace({"": "NA"})
    if label_col:
        out["term_label_raw"] = _col_series(df, label_col, default="")
    else:
        out["term_label_raw"] = _safe_str_series(df.index.to_series())

    out["term_label"] = out["term_label_raw"].map(
        lambda x: _simplify_term_label(x, drop_hallmark_prefix)
    )

    out["score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    if dec_col:
        out["decision"] = _col_series(df, dec_col, default="NA").replace({"": "NA"})
    else:
        out["decision"] = "NA"

    if dir_col:
        out["direction"] = _col_series(df, dir_col, default="na").replace({"": "na"})
    else:
        out["direction"] = "na"

    if "module_rank" in df.columns:
        out["module_rank"] = pd.to_numeric(df["module_rank"], errors="coerce")

    return out, score_col


def _normalize_audit_log(df: pd.DataFrame, drop_hallmark_prefix: bool) -> tuple[pd.DataFrame, str]:
    """
    Normalize an `audit_log.tsv`-like table into a common plotting schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table read from audit_log.tsv (or compatible).
    drop_hallmark_prefix : bool
        Whether to drop "HALLMARK_" prefix in labels.

    Returns
    -------
    tbl : pandas.DataFrame
        Normalized table with columns:
        - module_id, term_label_raw, term_label, score, decision, direction
    score_label : str
        Human-readable label describing the computed score ("stability×context").

    Notes
    -----
    - Score is computed as `stability * context_fit` using auto-detected columns.
    - If stability/context columns are missing, the corresponding factor defaults to 1.0.
    - Direction is not inferred from audit_log here and defaults to 'na' (plotting mode
      can still color by module or score).
    """
    module_col = (
        _first_existing_col(df, ["module_id_effective", "module_id", "module_uid"]) or "module_id"
    )
    term_id_col = _first_existing_col(df, ["term_id", "termId"]) or ""
    term_name_col = _first_existing_col(df, ["term_name", "termName"]) or ""
    dec_col = _first_existing_col(df, ["status", "decision", "final_decision"]) or "status"

    stab_col = _first_existing_col(df, ["term_survival_agg", "term_survival", "stability"]) or ""
    ctx_col = (
        _first_existing_col(
            df,
            [
                "context_score_proxy_u01_norm",
                "context_score_proxy_u01",
                "context_score_u01_norm",
                "context_score_u01",
                "context_score",
            ],
        )
        or ""
    )

    if not stab_col:
        df["_stab"] = 1.0
        stab_col = "_stab"
    if not ctx_col:
        df["_ctx"] = 1.0
        ctx_col = "_ctx"

    out = pd.DataFrame(index=df.index)
    out["module_id"] = _col_series(df, module_col, default="NA").replace({"": "NA"})

    if term_id_col and term_name_col:
        raw = (
            _col_series(df, term_id_col, default="")
            + ": "
            + _col_series(df, term_name_col, default="")
        )
    elif term_name_col:
        raw = _col_series(df, term_name_col, default="")
    elif term_id_col:
        raw = _col_series(df, term_id_col, default="")
    else:
        raw = _safe_str_series(df.index.to_series())

    out["term_label_raw"] = raw
    out["term_label"] = out["term_label_raw"].map(
        lambda x: _simplify_term_label(x, drop_hallmark_prefix)
    )

    stab = pd.to_numeric(df[stab_col], errors="coerce").fillna(1.0).clip(lower=0.0)
    ctx = pd.to_numeric(df[ctx_col], errors="coerce").fillna(1.0).clip(lower=0.0)
    out["score"] = (stab * ctx).clip(lower=0.0)

    out["decision"] = _col_series(df, dec_col, default="NA").replace({"": "NA"})
    out["direction"] = "na"
    return out, "stability×context"


def _load_tbl(in_tsv: Path, drop_hallmark_prefix: bool) -> tuple[pd.DataFrame, str, str]:
    """
    Load and normalize a TSV into the common plotting table.

    Parameters
    ----------
    in_tsv : pathlib.Path
        Path to input TSV. Typically `claims_ranked.tsv` or `audit_log.tsv`.
    drop_hallmark_prefix : bool
        Whether to drop "HALLMARK_" prefix in term labels.

    Returns
    -------
    tbl : pandas.DataFrame
        Normalized plotting table.
    src_kind : str
        Either "claims_ranked" or "audit_log" indicating which parser was used.
    score_label : str
        Label describing the score source (column name or derived label).

    Notes
    -----
    Heuristic detection:
    - Uses filename and a set of "audit-like" columns to decide whether the TSV
      is an audit log. This is robust to renamed audit log files.
    """
    df = pd.read_csv(in_tsv, sep="\t")
    if df.empty:
        _die(f"[plot-ranked] Empty TSV: {in_tsv}")

    name = in_tsv.name.lower()
    audit_name = name in {"audit_log.tsv", "audit.tsv", "audit_results.tsv"}

    audit_cols = {
        "claim_json",
        "final_decision",
        "audit_reason",
        "reason_code",
        "module_id_effective",
        "term_survival",
        "term_survival_agg",
        "context_score_proxy_u01_norm",
        "context_score_u01_norm",
    }
    is_audit = audit_name or any(c in df.columns for c in audit_cols)

    if is_audit:
        tbl, score_label = _normalize_audit_log(df, drop_hallmark_prefix=drop_hallmark_prefix)
        return tbl, "audit_log", score_label

    tbl, score_label = _normalize_claims_ranked(df, drop_hallmark_prefix=drop_hallmark_prefix)
    return tbl, "claims_ranked", score_label


def _size_transform(scores: np.ndarray, gamma: float, min_norm: float) -> np.ndarray:
    """
    Convert raw scores to normalized positive sizes for packed-circle areas.

    Parameters
    ----------
    scores : numpy.ndarray
        Raw scores (any shape). Negative values are treated as 0.
    gamma : float
        Power transform exponent applied after normalization (u := u**gamma).
    min_norm : float
        Minimum normalized value after transform to keep small circles visible.

    Returns
    -------
    numpy.ndarray
        Array of normalized sizes in (0, 1], same shape as `scores`.

    Notes
    -----
    - Normalization is performed by dividing by the maximum finite score.
    - If max <= 0, returns an array of ones to avoid degenerate circles.
    """
    s = np.asarray(scores, dtype=float)
    s[s < 0] = 0.0
    mx = float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else 0.0
    if mx <= 0:
        return np.full_like(s, 1.0)
    u = s / mx
    u = np.power(u, gamma)
    u = np.clip(u, min_norm, None)
    return u


def _infer_module_rank_map(tbl: pd.DataFrame) -> dict[str, int]:
    """
    Build a deterministic module_id -> display rank map (1-based; shown as M##).

    Parameters
    ----------
    tbl : pandas.DataFrame
        Normalized plotting table containing at least `module_id` and `score`.
        If present, `module_rank` is preferred as an external single source of truth.

    Returns
    -------
    dict of str to int
        Mapping from module_id to 1-based rank.

    Notes
    -----
    Priority:
    1) If `module_rank` exists and provides non-null values, use the per-module minimum rank,
       normalize 0-based ranks to 1-based, and require uniqueness (decision-grade).
    2) Otherwise, rank modules by sum(score) descending with deterministic tie-break by module_id.
    """
    if tbl is None or tbl.empty:
        return {}

    # 1) Prefer precomputed module_rank if available
    if "module_rank" in tbl.columns and tbl["module_rank"].notna().any():
        s = pd.to_numeric(tbl["module_rank"], errors="coerce")
        df = pd.DataFrame({"module_id": tbl["module_id"].astype(str), "_r": s}).dropna(
            subset=["_r"]
        )
        if not df.empty:
            per = df.groupby("module_id")["_r"].min()  # robust to repeated per-row ranks
            per_i = per.round().astype(int)

            # 0-based -> 1-based shift
            if int(per_i.min()) == 0:
                per_i = per_i + 1

            # basic sanity
            if (per_i <= 0).any():
                _die("[plot-ranked] Invalid module_rank detected (<=0 after normalization).")

            # If duplicates exist, it's ambiguous; fail loudly (decision-grade).
            if per_i.duplicated().any():
                dup = per_i[per_i.duplicated(keep=False)].sort_values()
                _die(
                    "[plot-ranked] Duplicate module_rank detected after normalization; "
                    "cannot assign unique M## labels.\n"
                    f"Duplicates:\n{dup.to_string()}"
                )

            return {str(k): int(v) for k, v in per_i.to_dict().items()}

    # 2) Fallback: sum(score) desc, deterministic tie-break by module_id
    mod_sum = tbl.groupby("module_id")["score"].sum()
    order = sorted(mod_sum.index.tolist(), key=lambda m: (-float(mod_sum.loc[m]), str(m)))
    return {str(m): i + 1 for i, m in enumerate(order)}


def _module_rank_map_1based(tbl: pd.DataFrame) -> dict[str, int]:
    """
    Convert an existing `module_rank` column into a 1-based map.

    Parameters
    ----------
    tbl : pandas.DataFrame
        Table containing `module_id` and optionally `module_rank`.

    Returns
    -------
    dict of str to int
        Mapping from module_id to 1-based module rank. Returns {} if unavailable.

    Notes
    -----
    - If ranks look 0-based (min==0), shifts by +1.
    - If ranks look 1-based (min>=1), keeps as-is.
    - Uses per-module minimum rank as the canonical value.
    """
    if tbl is None or tbl.empty:
        return {}
    if "module_rank" not in tbl.columns:
        return {}

    t = tbl[["module_id", "module_rank"]].copy()
    t["module_id"] = t["module_id"].fillna("NA").astype(str)
    t["module_rank"] = pd.to_numeric(t["module_rank"], errors="coerce")
    t = t.dropna(subset=["module_rank"])
    if t.empty:
        return {}

    # per-module canonical rank (min; robust to repeated rows)
    g = t.groupby("module_id", sort=False)["module_rank"].min()

    mn = float(g.min())
    shift = 1 if np.isfinite(mn) and mn == 0.0 else 0

    out: dict[str, int] = {}
    for mid, r in g.items():
        try:
            out[str(mid)] = int(float(r)) + int(shift)
        except Exception:
            continue
    return out


def _build_hierarchy(
    tbl: pd.DataFrame,
    top_modules: int,
    top_terms_per_module: int,
    score_gamma: float,
    score_min_norm: float,
    max_term_chars: int,
    *,
    mod_rank_map: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """
    Build a circlify-compatible hierarchy from a normalized plotting table.

    Parameters
    ----------
    tbl : pandas.DataFrame
        Normalized plotting table containing:
        - module_id, term_label, score, and optionally direction.
    top_modules : int
        If > 0, keep only the top modules by sum(score).
    top_terms_per_module : int
        If > 0, keep only the top terms per module by score.
    score_gamma : float
        Power transform exponent for term sizes (see `_size_transform`).
    score_min_norm : float
        Minimum normalized term size (see `_size_transform`).
    max_term_chars : int
        Maximum characters to keep per term label prior to wrapping.
    mod_rank_map : dict of str to int, optional
        Optional module_id -> 1-based rank mapping to force shared M## labels
        across plot variants.

    Returns
    -------
    hier : list of dict
        Hierarchical structure for `circlify.circlify`, where:
        - level 1 nodes represent modules
        - level 2 nodes represent terms
    mod_rank : dict of str to int
        Mapping for module_id -> 1-based display rank used in labels.

    Notes
    -----
    - Module ordering is deterministic: sum(score) descending, tie-break by module_id.
    - Term ordering within module is deterministic: score descending, tie-break by term_label.
    - Child IDs encode module_id, direction, score, and label for later parsing.
    """
    # deterministic module ordering: sum(score) desc, tie-break by module_id
    mod_sum = tbl.groupby("module_id")["score"].sum()
    mods = sorted(mod_sum.index.tolist(), key=lambda m: (-float(mod_sum.loc[m]), str(m)))

    if top_modules and top_modules > 0:
        mods = mods[:top_modules]

    keep_mods = set(mods)
    sub = tbl[tbl["module_id"].isin(keep_mods)].copy()

    # rank labels (M##) must be shared across plots
    if mod_rank_map is None:
        mod_rank = {str(m): i + 1 for i, m in enumerate(mods)}
    else:
        mod_rank = {str(m): int(mod_rank_map.get(str(m), 99)) for m in mods}

    sub["_term_datum"] = _size_transform(
        sub["score"].to_numpy(), gamma=score_gamma, min_norm=score_min_norm
    )

    data: list[dict] = []

    # iterate modules in `mods` order for deterministic circlify inputs
    for module_id in mods:
        g = sub[sub["module_id"] == module_id].copy()
        if g.empty:
            continue

        g = g.sort_values(["score", "term_label"], ascending=[False, True])
        if top_terms_per_module and top_terms_per_module > 0:
            g = g.head(top_terms_per_module)

        children = []
        for _, r in g.iterrows():
            label = _wrap_words(_shorten(r["term_label"], max_term_chars), width=14)
            d = str(r.get("direction", "na")).strip().lower()
            sc = float(r.get("score", 0.0))
            children.append(
                {
                    "id": f"{module_id}||{d}||{sc:.6g}||{label}",
                    "datum": float(max(r["_term_datum"], 1e-9)),
                }
            )

        parent_datum = float(max(sum(ch["datum"] for ch in children), 1e-9))
        mlabel = f"M{mod_rank.get(str(module_id), 99):02d}"
        data.append({"id": f"{module_id}||{mlabel}", "datum": parent_datum, "children": children})

    return data, mod_rank


def _parse_id(s: str) -> tuple[str, str]:
    """
    Parse a packed-circle node ID for module-level circles.

    Parameters
    ----------
    s : str
        Encoded ID string in the form "{module_id}||{mlabel}".

    Returns
    -------
    module_id : str
        Parsed module identifier (or "NA" if missing).
    mlabel : str
        Parsed module label (e.g., "M01").

    Notes
    -----
    This is the inverse of the encoding used in `_build_hierarchy`.
    """
    s = str(s)
    if "||" in s:
        a, b = s.split("||", 1)
        return a, b
    return "NA", s


def _parse_term_id(s: str) -> tuple[str, str, float, str]:
    """
    Parse a packed-circle node ID for term-level circles.

    Parameters
    ----------
    s : str
        Encoded ID string in the form "{module_id}||{dir}||{score}||{label}".

    Returns
    -------
    module_id : str
        Parsed module identifier.
    direction : str
        Parsed direction token (e.g., 'up', 'down', 'na').
    score : float
        Parsed score value; NaN if parsing fails.
    label : str
        Parsed wrapped label text.

    Notes
    -----
    Robust to partially encoded IDs; returns 'na'/NaN fallbacks when needed.
    """
    s = str(s)
    parts = s.split("||", 3)
    if len(parts) == 4:
        mod, d, sc, label = parts
        try:
            return mod, d, float(sc), label
        except Exception:
            return mod, d, float("nan"), label
    if len(parts) == 2:
        mod, label = parts
        return mod, "na", float("nan"), label
    return "NA", "na", float("nan"), s


def _clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp a scalar to [lo, hi].

    Parameters
    ----------
    x : float
        Value to clamp.
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Returns
    -------
    float
        Clamped value.

    Notes
    -----
    Used primarily for font size and stroke width stabilization.
    """
    return max(lo, min(hi, x))


def _runlength_blocks(mods: list[str]) -> list[tuple[int, int, str]]:
    """
    Compute run-length encoded blocks for consecutive identical module IDs.

    Parameters
    ----------
    mods : list of str
        Sequence of module IDs aligned to plotted rows (top-to-bottom ordering).

    Returns
    -------
    list of tuple
        List of (start, end, module_id) blocks, where the block covers indices
        [start, end) and module_id is constant.

    Notes
    -----
    Used to draw grouped left-strip annotations when bars are ordered by module.
    """
    if not mods:
        return []
    out: list[tuple[int, int, str]] = []
    start = 0
    cur = mods[0]
    for i in range(1, len(mods)):
        if mods[i] != cur:
            out.append((start, i, cur))
            start = i
            cur = mods[i]
    out.append((start, len(mods), cur))
    return out


def _require_circlify() -> None:
    """
    Ensure the optional `circlify` dependency is available for packed-circle plots.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - If missing, raises a SystemExit with a clear install hint.
    - Users can still produce bar plots without circlify.
    """
    try:
        import circlify  # noqa: F401
    except Exception as e:
        _die(
            "[plot-ranked] Missing dependency: circlify\n"
            "Install: python -m pip install circlify\n"
            "Or use: llm-pathway-curator plot-ranked --mode bars ...\n"
            f"Original error: {e}"
        )


@dataclass(frozen=True)
class PlotRankedConfig:
    """
    Configuration for ranked-claim plotting.

    Notes
    -----
    This dataclass is an immutable "config contract" used by both CLI and library
    entrypoints to keep plotting behavior stable and reproducible. Values are
    intentionally explicit (rather than nested dicts) to simplify provenance capture.
    """

    mode: str = "packed"  # packed|bars
    in_tsv: str = ""
    run_dir: str = ""
    out_png: str = ""

    decision: str = "PASS"
    drop_hallmark_prefix: bool = False
    dpi: int = 220
    annotate: bool = False
    fontsize: int = 16

    # packed
    top_modules: int = 12
    top_terms_per_module: int = 8
    size_gamma: float = 3.5
    size_min_norm: float = 0.03
    min_term_label_r: float = 0.09
    dark: bool = False
    module_cmap: str = "tab20"
    term_color_mode: str = "module"  # score|module|direction
    term_cmap: str = "coolwarm"
    term_vmin: float = float("nan")
    term_vmax: float = float("nan")
    term_font_scale: float = 60.0
    term_font_min: float = 7.0
    term_font_max: float = 18.0
    module_font_scale: float = 34.0
    module_font_min: float = 10.0
    module_font_max: float = 20.0

    # bars
    top_n: int = 50
    group_by_module: bool = False
    left_strip: bool = False
    strip_labels: bool = False
    xlabel: str = ""
    bar_color_mode: str = "score"  # score|module|direction
    bar_cmap: str = "coolwarm"
    bar_vmin: float = float("nan")
    bar_vmax: float = float("nan")


def plot_ranked(cfg: PlotRankedConfig) -> Path:
    """
    Plot ranked claims as either packed circles or a horizontal bar chart.

    Parameters
    ----------
    cfg : PlotRankedConfig
        Plot configuration specifying input location, output path, and styling options.

    Returns
    -------
    pathlib.Path
        Output PNG path that was written.

    Notes
    -----
    Input resolution:
    - If cfg.in_tsv is given, it is used directly.
    - Otherwise, cfg.run_dir is searched for `claims_ranked.tsv` or `audit_log.tsv`.

    Filtering:
    - If cfg.decision is non-empty, rows are filtered by decision/status (case-insensitive).

    Reproducibility:
    - Module labels (M##) are derived from a single module-rank map computed once per plot call,
      so packed and bar variants remain consistent for the same filtered table.
    """

    # Apply global matplotlib style once per plot call (single source of truth).
    apply_pub_style(fontsize=int(cfg.fontsize))
    mode = str(cfg.mode).strip().lower()
    if mode not in {"packed", "bars"}:
        _die(f"[plot-ranked] invalid mode={cfg.mode!r} (use packed|bars)")

    if not cfg.out_png:
        _die("[plot-ranked] missing out_png")

    in_tsv = Path(cfg.in_tsv) if cfg.in_tsv else None
    if in_tsv is None:
        if not cfg.run_dir:
            _die("[plot-ranked] Provide in_tsv or run_dir")
        in_tsv = _find_input_tsv(Path(cfg.run_dir))

    tbl, src_kind, score_label = _load_tbl(
        in_tsv, drop_hallmark_prefix=bool(cfg.drop_hallmark_prefix)
    )

    if str(cfg.decision).strip():
        allowed = {x.strip().upper() for x in str(cfg.decision).split(",") if x.strip()}
        dec_norm = tbl["decision"].astype(str).str.strip().str.upper()
        tbl = tbl[dec_norm.isin(allowed)].copy()
        if tbl.empty:
            _die(f"[plot-ranked] No rows after decision filter: {sorted(allowed)}")

    # Single source of truth: module_id -> display rank (M##)
    mod_rank_map = _infer_module_rank_map(tbl)

    out_png = Path(cfg.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if mode == "bars":
        _plot_bars(
            tbl,
            out_png,
            src_kind=src_kind,
            score_label=score_label,
            cfg=cfg,
            mod_rank_map=mod_rank_map,
        )
        return out_png

    _require_circlify()
    _plot_packed(tbl, out_png, src_kind=src_kind, cfg=cfg, mod_rank_map=mod_rank_map)
    return out_png


def _plot_packed(
    tbl: pd.DataFrame,
    out_png: Path,
    *,
    src_kind: str,
    cfg: PlotRankedConfig,
    mod_rank_map: dict[str, int] | None = None,
) -> None:
    """
    Render a hierarchical packed-circle plot (modules outer ring; terms inner circles).

    Parameters
    ----------
    tbl : pandas.DataFrame
        Normalized plotting table (from `_load_tbl`), containing module_id, term_label, score,
        and optionally direction.
    out_png : pathlib.Path
        Output PNG path to write.
    src_kind : str
        Source kind label ("claims_ranked" or "audit_log") used only for annotation.
    cfg : PlotRankedConfig
        Plot configuration (packed-specific fields are used).
    mod_rank_map : dict of str to int or None, optional
        Optional precomputed module rank map (module_id -> 1-based M##) to enforce
        consistent labels across plot types.

    Returns
    -------
    None
        Writes the figure to `out_png`.

    Notes
    -----
    - Requires `circlify`.
    - Circle areas are proportional to a transformed score (see `_size_transform`).
    - Term colors can reflect module identity, direction (shaded by score), or score gradient.
    """
    import circlify

    hier, _mod_rank = _build_hierarchy(
        tbl,
        top_modules=cfg.top_modules,
        top_terms_per_module=cfg.top_terms_per_module,
        score_gamma=cfg.size_gamma,
        score_min_norm=cfg.size_min_norm,
        max_term_chars=28,
        mod_rank_map=mod_rank_map,
    )

    circles = circlify.circlify(
        hier,
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1),
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    if cfg.dark:
        bg = "#0b0f14"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        text_color = "white"
        edge = "white"
        stroke_fg = "black"
    else:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        text_color = "#111111"
        edge = "white"
        stroke_fg = "white"

    def _outline_for_fs(fs: float):
        lw = _clamp(0.20 * fs, 1.2, 3.2)
        return [pe.withStroke(linewidth=lw, foreground=stroke_fg)]

    term_cmap = _get_cmap(cfg.term_cmap)
    s = tbl["score"].to_numpy(dtype=float)
    vmin = float(np.nanmin(s)) if len(s) else 0.0
    vmax = float(np.nanmax(s)) if len(s) else 1.0
    if not np.isnan(cfg.term_vmin):
        vmin = float(cfg.term_vmin)
    if not np.isnan(cfg.term_vmax):
        vmax = float(cfg.term_vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-9
    term_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for c in circles:
        if c.level == 1:
            module_id, mlabel = _parse_id(c.ex.get("id", "NA||M??"))
            color = _stable_color(module_id, cmap_name=cfg.module_cmap)

            ax.add_patch(
                plt.Circle(
                    (c.x, c.y),
                    c.r,
                    facecolor=color,
                    alpha=0.06 if not cfg.dark else 0.10,
                    edgecolor=color,
                    linewidth=2.4 if not cfg.dark else 3.0,
                )
            )

            fs = _clamp(c.r * cfg.module_font_scale, cfg.module_font_min, cfg.module_font_max)

            # Place module label well INSIDE the circle to reduce clipping.
            inset_frac = 0.72
            x = c.x - inset_frac * c.r
            y = c.y + inset_frac * c.r
            ha = "left"
            va = "top"

            if c.r < 0.16:
                x = c.x
                y = c.y
                ha = "center"
                va = "center"

            t = ax.text(
                x,
                y,
                mlabel,
                ha=ha,
                va=va,
                fontsize=fs,
                fontweight="bold",
                color=color if cfg.dark else "#111111",
                zorder=10,
            )
            t.set_path_effects(_outline_for_fs(fs))

        elif c.level == 2:
            module_id, d, sc, label = _parse_term_id(c.ex.get("id", "NA||na||nan||"))
            mcolor = _stable_color(module_id, cmap_name=cfg.module_cmap)

            if cfg.term_color_mode == "module":
                fill = mcolor
            elif cfg.term_color_mode == "direction":
                bg_rgb = (0.043, 0.059, 0.078) if cfg.dark else (1.0, 1.0, 1.0)
                fill = _direction_shaded_color(
                    d, sc, vmin, vmax, bg_rgb=bg_rgb, gamma=1.0, min_t=0.15
                )
            else:
                fill = term_cmap(term_norm(float(sc)))

            ax.add_patch(
                plt.Circle(
                    (c.x, c.y),
                    c.r,
                    facecolor=fill,
                    alpha=0.82 if not cfg.dark else 0.85,
                    edgecolor=edge,
                    linewidth=1.1 if not cfg.dark else 1.2,
                )
            )

            if c.r >= cfg.min_term_label_r:
                fs = _clamp(c.r * cfg.term_font_scale, cfg.term_font_min, cfg.term_font_max)
                t = ax.text(
                    c.x,
                    c.y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=text_color,
                )
                t.set_path_effects(_outline_for_fs(fs))

    if cfg.annotate:
        ax.text(
            0.01,
            0.99,
            f"{src_kind} | decision={cfg.decision} | outer=modules (M##), inner=terms",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=text_color if not cfg.dark else "white",
        )

    lim = max(max(abs(c.x) + c.r, abs(c.y) + c.r) for c in circles)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    fig.tight_layout()
    fig.savefig(out_png, dpi=int(cfg.dpi))
    plt.close(fig)


def _plot_bars(
    tbl: pd.DataFrame,
    out_png: Path,
    *,
    src_kind: str,
    score_label: str,
    cfg: PlotRankedConfig,
    mod_rank_map: dict[str, int] | None = None,
) -> None:
    """
    Render a horizontal bar plot of ranked terms (optionally grouped by module).

    Parameters
    ----------
    tbl : pandas.DataFrame
        Normalized plotting table (from `_load_tbl`), containing:
        - score, term_label, module_id, and optionally direction.
    out_png : pathlib.Path
        Output PNG path to write.
    src_kind : str
        Source kind label ("claims_ranked" or "audit_log") used only for annotation.
    score_label : str
        X-axis label derived from the score source.
    cfg : PlotRankedConfig
        Plot configuration (bars-specific fields are used).
    mod_rank_map : dict of str to int or None, optional
        Optional module rank map for consistent grouping/strip labels.

    Returns
    -------
    None
        Writes the figure to `out_png`.

    Notes
    -----
    - If cfg.group_by_module is enabled, ordering is by module rank then score.
    - Optional left strip can show module blocks (and M## tags if strip_labels).
    - Bar colors can reflect score gradient, module identity, or direction shading.
    """
    df = tbl.copy()
    df["_score"] = pd.to_numeric(df["score"], errors="coerce")
    df["_module"] = df["module_id"].fillna("NA").astype(str)
    df["_label"] = df["term_label"].astype(str)
    df["_dir"] = df.get("direction", "na").astype(str)

    df = df[~df["_score"].isna()].copy()
    if df.empty:
        _die("[plot-ranked] No numeric scores after parsing.")

    # Use a shared module rank map (single source of truth)
    rank_map = mod_rank_map or _infer_module_rank_map(df)

    if cfg.group_by_module:
        df["_modrank"] = df["_module"].map(rank_map).fillna(1e9)
        df = (
            df.sort_values(["_modrank", "_score"], ascending=[True, False])
            .head(int(cfg.top_n))
            .copy()
        )
    else:
        df = df.sort_values("_score", ascending=False).head(int(cfg.top_n)).copy()

    labels = df["_label"].tolist()
    scores = df["_score"].to_numpy(dtype=float)
    modules = df["_module"].tolist()
    dirs = df["_dir"].tolist()

    if cfg.bar_color_mode == "direction":
        vmin = float(np.nanmin(scores)) if np.isfinite(np.nanmin(scores)) else 0.0
        vmax = float(np.nanmax(scores)) if np.isfinite(np.nanmax(scores)) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-9
        bar_colors = [
            _direction_shaded_color(
                d, float(s), vmin, vmax, bg_rgb=(1.0, 1.0, 1.0), gamma=1.0, min_t=0.15
            )
            for d, s in zip(dirs, scores, strict=False)
        ]
    elif cfg.bar_color_mode == "module":
        bar_colors = [_stable_color(m, cmap_name=cfg.module_cmap) for m in modules]
    else:
        vmin = float(np.nanmin(scores)) if np.isfinite(np.nanmin(scores)) else 0.0
        vmax = float(np.nanmax(scores)) if np.isfinite(np.nanmax(scores)) else 1.0
        if not np.isnan(cfg.bar_vmin):
            vmin = float(cfg.bar_vmin)
        if not np.isnan(cfg.bar_vmax):
            vmax = float(cfg.bar_vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        bar_cmap = _get_cmap(cfg.bar_cmap)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        bar_colors = [bar_cmap(norm(float(x))) for x in scores]

    n = len(df)
    fig_h = max(5.0, 0.33 * n)

    if cfg.left_strip:
        fig, (ax_strip, ax) = plt.subplots(
            ncols=2,
            figsize=(13.0, fig_h),
            sharey=True,
            gridspec_kw={"width_ratios": [0.10, 0.90]},
        )
        ax_strip.set_xlim(0, 1)
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])
        for spine in ax_strip.spines.values():
            spine.set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(12.5, fig_h))
        ax_strip = None

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    if ax_strip is not None:
        ax_strip.set_facecolor("white")

    ax.barh(range(n), scores, color=bar_colors, height=0.82)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    if ax_strip is not None:
        ax_strip.set_ylim(ax.get_ylim())

    xlabel = cfg.xlabel.strip() or score_label
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.20)

    if ax_strip is not None:
        if cfg.group_by_module:
            blocks = _runlength_blocks(modules)

            for s0, e0, m in blocks:
                y_mid = (s0 + e0 - 1) / 2.0
                height = (e0 - s0) * 0.82
                color = _stable_color(m, cmap_name=cfg.module_cmap)
                ax_strip.barh([y_mid], [1.0], height=height, color=color)

                if cfg.strip_labels:
                    r = int(rank_map.get(m, 0)) if m in rank_map else 0
                    tag = f"M{r:02d}" if r > 0 else "M"
                    ax_strip.text(
                        0.5, y_mid, tag, ha="center", va="center", fontsize=10, color="white"
                    )

            for i in range(1, n):
                if modules[i] != modules[i - 1]:
                    ax.axhline(i - 0.5, linewidth=1.0, alpha=0.25)
                    ax_strip.axhline(i - 0.5, linewidth=1.0, alpha=0.20)
        else:
            strip_colors = [_stable_color(m, cmap_name=cfg.module_cmap) for m in modules]
            ax_strip.barh(range(n), [1.0] * n, color=strip_colors, height=0.82)

    if cfg.annotate:
        ax.text(
            0.01,
            0.99,
            f"{src_kind} | decision={cfg.decision or 'ALL'} | top_n={n}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#111111",
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=int(cfg.dpi))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for `llm-pathway-curator plot-ranked`.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured parser that matches `PlotRankedConfig` fields.

    Notes
    -----
    The parser is kept in a function so that:
    - it can be reused by unit tests,
    - it stays synchronized with PlotRankedConfig defaults,
    - it avoids side effects at import time.
    """
    p = argparse.ArgumentParser(prog="llm-pathway-curator plot-ranked", add_help=True)

    p.add_argument("--mode", type=str, default="packed", choices=["packed", "bars"])
    p.add_argument(
        "--in-tsv", type=str, default="", help="claims_ranked.tsv (recommended) or audit_log.tsv"
    )
    p.add_argument("--run-dir", type=str, default="", help="Auto-detect TSV under this directory")
    p.add_argument("--out-png", type=str, required=True)

    p.add_argument(
        "--decision",
        type=str,
        default="PASS",
        help="Filter decision/status (e.g., PASS or PASS,ABSTAIN)",
    )
    p.add_argument("--drop-hallmark-prefix", action="store_true")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--annotate", action="store_true")

    # packed
    p.add_argument("--top-modules", type=int, default=12)
    p.add_argument("--top-terms-per-module", type=int, default=8)
    p.add_argument("--size-gamma", type=float, default=3.5)
    p.add_argument("--size-min-norm", type=float, default=0.03)
    p.add_argument("--min-term-label-r", type=float, default=0.09)
    p.add_argument("--dark", action="store_true")
    p.add_argument("--module-cmap", type=str, default="tab20")
    p.add_argument(
        "--term-color-mode", type=str, default="module", choices=["score", "module", "direction"]
    )
    p.add_argument("--term-cmap", type=str, default="coolwarm")
    p.add_argument("--term-vmin", type=float, default=float("nan"))
    p.add_argument("--term-vmax", type=float, default=float("nan"))

    # bars
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--group-by-module", action="store_true")
    p.add_argument("--left-strip", action="store_true")
    p.add_argument("--strip-labels", action="store_true")
    p.add_argument("--xlabel", type=str, default="")
    p.add_argument(
        "--bar-color-mode", type=str, default="score", choices=["score", "module", "direction"]
    )
    p.add_argument("--bar-cmap", type=str, default="coolwarm")
    p.add_argument("--bar-vmin", type=float, default=float("nan"))
    p.add_argument("--bar-vmax", type=float, default=float("nan"))
    p.add_argument("--fontsize", type=int, default=16, help="base font size (default 16)")
    return p


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint for ranked-claim plotting.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments (excluding program name). If None, argparse reads from sys.argv.

    Returns
    -------
    int
        Exit code (0 for success).

    Notes
    -----
    This constructs `PlotRankedConfig` from parsed args, runs `plot_ranked`, and prints
    the output path. Exceptions are intentionally surfaced as SystemExit via `_die`
    for consistent CLI behavior.
    """
    p = build_parser()
    a = p.parse_args(argv)

    cfg = PlotRankedConfig(
        mode=a.mode,
        in_tsv=a.in_tsv,
        run_dir=a.run_dir,
        out_png=a.out_png,
        decision=a.decision,
        drop_hallmark_prefix=bool(a.drop_hallmark_prefix),
        dpi=int(a.dpi),
        annotate=bool(a.annotate),
        fontsize=int(a.fontsize),
        top_modules=int(a.top_modules),
        top_terms_per_module=int(a.top_terms_per_module),
        size_gamma=float(a.size_gamma),
        size_min_norm=float(a.size_min_norm),
        min_term_label_r=float(a.min_term_label_r),
        dark=bool(a.dark),
        module_cmap=str(a.module_cmap),
        term_color_mode=str(a.term_color_mode),
        term_cmap=str(a.term_cmap),
        term_vmin=float(a.term_vmin),
        term_vmax=float(a.term_vmax),
        top_n=int(a.top_n),
        group_by_module=bool(a.group_by_module),
        left_strip=bool(a.left_strip),
        strip_labels=bool(a.strip_labels),
        xlabel=str(a.xlabel),
        bar_color_mode=str(a.bar_color_mode),
        bar_cmap=str(a.bar_cmap),
        bar_vmin=float(a.bar_vmin),
        bar_vmax=float(a.bar_vmax),
    )

    out = plot_ranked(cfg)
    print(f"[OK] wrote plot: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
