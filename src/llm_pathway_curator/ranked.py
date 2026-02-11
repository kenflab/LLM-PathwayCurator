# LLM-PathwayCurator/src/llm_pathway_curator/ranked.py
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


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
    This is intentionally a thin wrapper over SystemExit to keep CLI-style error
    behavior consistent across scripts and library entrypoints.
    """
    raise SystemExit(msg)


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first column name that exists in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table to search.
    candidates : list of str
        Candidate column names, checked in order.

    Returns
    -------
    str or None
        The first matching column name, or None if none exist.

    Notes
    -----
    This supports backwards/forwards compatibility across slightly different
    TSV schemas (e.g., paper outputs vs. library outputs).
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_str_series(s: pd.Series) -> pd.Series:
    """
    Convert a Series to string dtype, mapping missing values to empty string.

    Parameters
    ----------
    s : pandas.Series
        Input series (any dtype).

    Returns
    -------
    pandas.Series
        String series where NaN/NA values become "" (not the literal "nan").

    Notes
    -----
    This is used to prevent accidental "nan" strings from propagating into IDs,
    join keys, or labels.
    """
    return s.where(pd.notna(s), "").astype(str)


def _find_one(run_dir: Path, names: list[str]) -> Path | None:
    """
    Locate the first existing non-empty file by name under a directory.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory to search.
    names : list of str
        Candidate filenames to look for.

    Returns
    -------
    pathlib.Path or None
        Path to the first match (deterministic by sorted order for recursive hits),
        or None if no matches are found.

    Notes
    -----
    Search strategy:
    1) Check `run_dir / name` for each name (fast path).
    2) If none found, recursively glob for each name and choose the lexicographically
       smallest path among non-empty files.
    """
    for name in names:
        p = run_dir / name
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p
    hits: list[Path] = []
    for name in names:
        hits.extend(run_dir.rglob(name))
    hits = [p for p in hits if p.is_file() and p.stat().st_size > 0]
    return sorted(hits)[0] if hits else None


def _neglog10_series(s: pd.Series) -> pd.Series:
    """
    Compute -log10(x) for a numeric-like Series with safe handling of zeros.

    Parameters
    ----------
    s : pandas.Series
        Input values (e.g., q-values). Non-numeric values are coerced to NaN.

    Returns
    -------
    pandas.Series
        Series of floats with -log10(max(x, 1e-300)) applied elementwise.

    Notes
    -----
    - Values <= 0 are clamped to 1e-300 before log to avoid inf/-inf.
    - Missing values remain NaN.
    """
    v = pd.to_numeric(s, errors="coerce")
    return v.map(lambda x: (-math.log10(max(float(x), 1e-300))) if pd.notna(x) else float("nan"))


def _norm_dir_token(x: str) -> str:
    """
    Normalize direction tokens to {'up','down','na'}.

    Parameters
    ----------
    x : str
        Direction-like token (e.g., 'up', '+', 'negative', '-1').

    Returns
    -------
    str
        Normalized direction: 'up', 'down', or 'na'.

    Notes
    -----
    This function is intentionally permissive to support heterogeneous upstream
    sources and legacy outputs.
    """
    s = str(x).strip().lower()
    if s in {"up", "+", "pos", "positive", "1"}:
        return "up"
    if s in {"down", "-", "neg", "negative", "-1"}:
        return "down"
    return "na"


def _infer_dir_from_signed(x) -> str:
    """
    Infer direction from a signed numeric value.

    Parameters
    ----------
    x : any
        Numeric-like value (e.g., NES). Coerced with pandas.to_numeric.

    Returns
    -------
    str
        'up' if x > 0, 'down' if x < 0, otherwise 'na'.

    Notes
    -----
    This is used as a fallback when explicit direction columns are absent.
    """
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return "na"
    vv = float(v)
    if vv > 0:
        return "up"
    if vv < 0:
        return "down"
    return "na"


def build_claims_ranked_df(
    *,
    audit_log: Path,
    evidence_tsv: Path | None = None,
    prefer_q: bool = False,
    evidence_score_col: str = "",
    default_direction: str = "na",
    fill_direction_if_na: bool = False,
) -> pd.DataFrame:
    """
    Build a figure-friendly ranked claims table from an audit log (+ optional evidence table).

    Parameters
    ----------
    audit_log : pathlib.Path
        Path to an audit log TSV (e.g., `audit_log.tsv`). Must be non-empty.
        Expected to contain, at minimum, claim identifiers and final decisions, and
        ideally term/module identifiers and stability/context proxies.
    evidence_tsv : pathlib.Path or None, optional
        Optional path to an evidence TSV (e.g., `evidence.normalized.tsv`).
        When provided, this is used to compute an evidence strength proxy and, when
        possible, to fill direction for undirected sources.
    prefer_q : bool, optional
        If True and a q-value column is available in `evidence_tsv`, use -log10(q)
        as the evidence strength proxy in preference to NES/statistics.
    evidence_score_col : str, optional
        If non-empty and present in `evidence_tsv`, use this column (absolute value)
        as the evidence strength proxy.
    default_direction : str, optional
        Default direction token used only when `fill_direction_if_na=True` and the
        direction remains 'na' after audit/evidence inference. Must be one of
        {'up','down','na'} (extra tokens are normalized).
    fill_direction_if_na : bool, optional
        If True, fill remaining 'na' directions using `default_direction`.

    Returns
    -------
    pandas.DataFrame
        Ranked table with one row per claim, suitable for plotting. Key columns include:
        - claim_id, decision, module_id
        - term_uid, term_id, term_name, term_label
        - direction, direction_source
        - stability, context_fit
        - evidence_strength, score_source
        - utility_score, rank_global
        - module_utility_sum, module_rank, rank_in_module

    Notes
    -----
    Ranking heuristic
        `utility_score = evidence_strength_filled * stability * context_fit`,
        where `evidence_strength_filled` defaults to 1.0 when missing.

    Join behavior (audit -> evidence)
        The merge key is chosen conservatively:
        - Prefer term_uid if present in BOTH tables and audit has any non-empty term_uid.
        - Else use term_id (or an evidence 'term' column as fallback).
        When evidence contains duplicate join keys, the row with maximum evidence
        strength is kept (deterministic after sorting).

    Compatibility
        Column names are auto-detected to tolerate schema drift between paper scripts
        and library outputs, while keeping output semantics stable.
    """
    if not audit_log.exists():
        _die(f"[ranked] missing audit log: {audit_log}")

    audit = pd.read_csv(audit_log, sep="\t")
    if audit.empty:
        _die(f"[ranked] empty audit log: {audit_log}")

    # --- audit columns (robust) ---
    claim_id_col = _first_existing_col(audit, ["claim_id", "claim_uid", "id"]) or "claim_id"
    decision_col = _first_existing_col(audit, ["status", "decision", "final_decision"]) or "status"

    term_uid_col = _first_existing_col(audit, ["term_uid", "termUid"]) or ""
    term_id_col = _first_existing_col(audit, ["term_id", "termId"]) or ""
    term_name_col = _first_existing_col(audit, ["term_name", "termName"]) or ""
    module_col = (
        _first_existing_col(audit, ["module_id_effective", "module_id", "module_uid", "module"])
        or "module_id"
    )
    audit_dir_col = _first_existing_col(audit, ["direction_norm", "direction"]) or ""

    stability_col = (
        _first_existing_col(audit, ["term_survival_agg", "term_survival", "stability"]) or ""
    )
    if not stability_col:
        audit["_stability"] = 1.0
        stability_col = "_stability"

    context_fit_col = (
        _first_existing_col(
            audit,
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
    if not context_fit_col:
        audit["_context_fit"] = 1.0
        context_fit_col = "_context_fit"

    out = pd.DataFrame()
    out["claim_id"] = _safe_str_series(audit[claim_id_col])
    out["decision"] = _safe_str_series(audit[decision_col]).replace({"": "NA"})
    out["module_id"] = _safe_str_series(audit[module_col]).replace({"": "NA"})

    out["term_uid"] = _safe_str_series(audit[term_uid_col]) if term_uid_col else ""
    out["term_id"] = _safe_str_series(audit[term_id_col]) if term_id_col else ""
    out["term_name"] = _safe_str_series(audit[term_name_col]) if term_name_col else ""

    if audit_dir_col:
        out["direction"] = audit[audit_dir_col].fillna("na").astype(str).map(_norm_dir_token)
        out["direction_source"] = "audit"
    else:
        out["direction"] = "na"
        out["direction_source"] = "none"

    out["stability"] = (
        pd.to_numeric(audit[stability_col], errors="coerce").fillna(1.0).clip(lower=0.0)
    )
    out["context_fit"] = (
        pd.to_numeric(audit[context_fit_col], errors="coerce").fillna(1.0).clip(lower=0.0)
    )

    # --- evidence strength via evidence.normalized.tsv (optional) ---
    out["evidence_strength"] = float("nan")
    out["score_source"] = "none"

    if evidence_tsv is not None and evidence_tsv.exists():
        ev = pd.read_csv(evidence_tsv, sep="\t")
        if not ev.empty:
            # Determine best join key.
            # Prefer term_uid if present in BOTH tables, else term_id, else heuristics.
            ev_term_uid_col = _first_existing_col(ev, ["term_uid", "termUid"]) or ""
            ev_term_id_col = _first_existing_col(ev, ["term_id", "termId"]) or ""
            ev_term_col = _first_existing_col(ev, ["term"]) or ""

            has_out_uid = out["term_uid"].astype(bool).any()
            has_out_id = out["term_id"].astype(bool).any()

            join_col = ""
            join_out = None

            if has_out_uid and ev_term_uid_col:
                join_col = ev_term_uid_col
                join_out = out["term_uid"].astype(str)
            elif has_out_id and (ev_term_id_col or ev_term_col):
                join_col = ev_term_id_col or ev_term_col
                join_out = out["term_id"].astype(str)
            elif ev_term_uid_col and has_out_uid:
                join_col = ev_term_uid_col
                join_out = out["term_uid"].astype(str)
            elif ev_term_id_col and has_out_id:
                join_col = ev_term_id_col
                join_out = out["term_id"].astype(str)
            elif ev_term_col and has_out_id:
                join_col = ev_term_col
                join_out = out["term_id"].astype(str)

            if not join_col:
                _die(
                    f"[ranked] evidence TSV missing join key term_uid/term_id/term: {evidence_tsv}"
                )

            # Candidate score columns (used both for evidence strength and direction heuristics)
            q_col = _first_existing_col(ev, ["qval", "padj", "fdr", "q_value", "q"]) or ""
            nes_col = _first_existing_col(ev, ["NES", "nes"]) or ""
            stat_col = _first_existing_col(ev, ["stat", "score"]) or ""

            # evidence strength
            if evidence_score_col and evidence_score_col in ev.columns:
                ev["_evidence_strength"] = pd.to_numeric(
                    ev[evidence_score_col], errors="coerce"
                ).abs()
                ev["_src"] = evidence_score_col
            else:
                if prefer_q and q_col:
                    ev["_evidence_strength"] = _neglog10_series(ev[q_col])
                    ev["_src"] = f"-log10({q_col})"
                elif nes_col:
                    ev["_evidence_strength"] = pd.to_numeric(ev[nes_col], errors="coerce").abs()
                    ev["_src"] = f"abs({nes_col})"
                elif q_col:
                    ev["_evidence_strength"] = _neglog10_series(ev[q_col])
                    ev["_src"] = f"-log10({q_col})"
                elif stat_col:
                    ev["_evidence_strength"] = pd.to_numeric(ev[stat_col], errors="coerce").abs()
                    ev["_src"] = f"abs({stat_col})"
                else:
                    ev["_evidence_strength"] = float("nan")
                    ev["_src"] = "none"

            # direction from evidence (optional; ALWAYS define these columns)
            ev_dir_col = _first_existing_col(ev, ["direction_norm", "direction"]) or ""
            if ev_dir_col:
                ev["_direction"] = ev[ev_dir_col].fillna("na").astype(str).map(_norm_dir_token)
                ev["_direction_src"] = ev_dir_col
            else:
                signed_col = nes_col or stat_col
                if signed_col:
                    ev["_direction"] = ev[signed_col].map(_infer_dir_from_signed)
                    ev["_direction_src"] = f"sign({signed_col})"
                else:
                    ev["_direction"] = "na"
                    ev["_direction_src"] = "none"

            ev_small = ev[
                [join_col, "_evidence_strength", "_src", "_direction", "_direction_src"]
            ].copy()
            ev_small[join_col] = _safe_str_series(ev_small[join_col])

            # For deterministic merging when duplicates exist:
            # keep the row with max evidence strength.
            ev_small["_evidence_strength_num"] = pd.to_numeric(
                ev_small["_evidence_strength"], errors="coerce"
            )
            ev_small = (
                ev_small.sort_values(["_evidence_strength_num"], ascending=False)
                .drop_duplicates(subset=[join_col], keep="first")
                .drop(columns=["_evidence_strength_num"])
                .rename(columns={join_col: "_join"})
            )

            out["_join"] = join_out if join_out is not None else ""

            out = out.merge(ev_small, on="_join", how="left")

            need = out["direction"].astype(str).map(_norm_dir_token).eq("na")
            out.loc[need, "direction"] = out.loc[need, "_direction"].fillna("na").astype(str)
            out.loc[need & out["_direction"].notna(), "direction_source"] = "evidence:" + out.loc[
                need & out["_direction"].notna(), "_direction_src"
            ].fillna("unknown").astype(str)
            out["direction"] = out["direction"].astype(str).map(_norm_dir_token)

            out["evidence_strength"] = pd.to_numeric(out["_evidence_strength"], errors="coerce")
            out["score_source"] = out["_src"].fillna("none").astype(str)
            out = out.drop(columns=["_evidence_strength", "_src", "_direction", "_direction_src"])

    def _mk_label(r) -> str:
        tid = str(r.get("term_id", "")).strip()
        tnm = str(r.get("term_name", "")).strip()
        if tid and tnm:
            return f"{tid}: {tnm}"
        if tnm:
            return tnm
        if tid:
            return tid
        return str(r.get("term_uid", "")).strip()

    out["term_label"] = out.apply(_mk_label, axis=1)

    if fill_direction_if_na:
        dd = _norm_dir_token(default_direction)
        if dd in {"up", "down"}:
            need = out["direction"].astype(str).map(_norm_dir_token).eq("na")
            out.loc[need, "direction"] = dd
            out.loc[need, "direction_source"] = "assumed"

    # decision-grade utility score (mirrors script)
    out["evidence_strength_filled"] = pd.to_numeric(
        out["evidence_strength"], errors="coerce"
    ).fillna(1.0)
    out["utility_score"] = out["evidence_strength_filled"] * out["stability"] * out["context_fit"]

    out = out.sort_values("utility_score", ascending=False).reset_index(drop=True)
    out["rank_global"] = range(1, len(out) + 1)

    mod_sum = out.groupby("module_id")["utility_score"].sum().sort_values(ascending=False)
    mod_rank = {m: i + 1 for i, m in enumerate(mod_sum.index.tolist())}
    out["module_utility_sum"] = out["module_id"].map(mod_sum).fillna(0.0)
    out["module_rank"] = out["module_id"].map(mod_rank).fillna(10**9).astype(int)

    out["rank_in_module"] = (
        out.sort_values(["module_rank", "utility_score"], ascending=[True, False])
        .groupby("module_id")
        .cumcount()
        + 1
    )
    out = out.sort_values("utility_score", ascending=False).reset_index(drop=True)
    return out


def write_claims_ranked_tsv(
    *,
    out_tsv: Path,
    run_dir: Path | None = None,
    audit_log: Path | None = None,
    evidence_tsv: Path | None = None,
    top_n: int = 0,
    prefer_q: bool = False,
    evidence_score_col: str = "",
    default_direction: str = "na",
    fill_direction_if_na: bool = False,
) -> Path:
    """
    Build and write a ranked claims TSV to disk.

    Parameters
    ----------
    out_tsv : pathlib.Path
        Output path for the ranked TSV (e.g., `claims_ranked.tsv`).
        Parent directories are created if needed.
    run_dir : pathlib.Path or None, optional
        If provided, attempt to auto-locate inputs under this directory using
        common filenames (audit_log.tsv, evidence.normalized.tsv, etc.).
    audit_log : pathlib.Path or None, optional
        Path to `audit_log.tsv`. If None and `run_dir` is provided, it will be
        auto-detected. Must exist (after detection).
    evidence_tsv : pathlib.Path or None, optional
        Optional path to evidence TSV. If None and `run_dir` is provided, it will
        be auto-detected if present.
    top_n : int, optional
        If > 0, keep only the top-N rows after ranking (global).
    prefer_q : bool, optional
        Passed through to `build_claims_ranked_df`.
    evidence_score_col : str, optional
        Passed through to `build_claims_ranked_df`.
    default_direction : str, optional
        Passed through to `build_claims_ranked_df`.
    fill_direction_if_na : bool, optional
        Passed through to `build_claims_ranked_df`.

    Returns
    -------
    pathlib.Path
        The resolved output path (`out_tsv`) after writing.

    Notes
    -----
    This is a convenience wrapper intended for paper scripts and CLIs. It keeps
    the “single source of truth” logic in `build_claims_ranked_df` while providing
    deterministic input discovery under a run directory.
    """
    if run_dir is not None:
        if audit_log is None:
            audit_log = _find_one(run_dir, ["audit_log.tsv", "audit.tsv", "audit_results.tsv"])
        if evidence_tsv is None:
            evidence_tsv = _find_one(
                run_dir, ["evidence.normalized.tsv", "evidence_normalized.tsv", "evidence.tsv"]
            )

    if audit_log is None or not audit_log.exists():
        _die("[ranked] Provide audit_log or run_dir containing audit_log.tsv")

    df = build_claims_ranked_df(
        audit_log=audit_log,
        evidence_tsv=evidence_tsv,
        prefer_q=prefer_q,
        evidence_score_col=evidence_score_col,
        default_direction=default_direction,
        fill_direction_if_na=fill_direction_if_na,
    )

    if top_n and top_n > 0:
        df = df.head(top_n).copy()

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    return out_tsv
