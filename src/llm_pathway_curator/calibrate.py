# LLM-PathwayCurator/src/llm_pathway_curator/calibrate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from . import _shared

CalibMethod = Literal["none", "temperature", "isotonic"]


# -----------------------------
# Core definitions: Risk/Coverage
# -----------------------------
def compute_counts(status: pd.Series) -> dict[str, int]:
    """
    Count PASS/FAIL/ABSTAIN/TOTAL from a status series (strict validation).

    Parameters
    ----------
    status : pandas.Series
        Status values. Must normalize into {"PASS", "ABSTAIN", "FAIL"}.

    Returns
    -------
    dict[str, int]
        Counts with keys: {"PASS", "FAIL", "ABSTAIN", "TOTAL"}.

    Raises
    ------
    ValueError
        If unknown status values are present (strict spec validation).
    """
    s = _shared.normalize_status_series(status)
    _shared.validate_status_values(s)

    n_pass = int((s == "PASS").sum())
    n_fail = int((s == "FAIL").sum())
    n_abs = int((s == "ABSTAIN").sum())
    n_total = int(len(s))
    return {"PASS": n_pass, "FAIL": n_fail, "ABSTAIN": n_abs, "TOTAL": n_total}


def risk_coverage_from_status(status: pd.Series) -> dict[str, float]:
    """
    Compute spec-safe Risk/Coverage metrics from a status series.

    Parameters
    ----------
    status : pandas.Series
        Status values in {"PASS", "ABSTAIN", "FAIL"}.

    Returns
    -------
    dict[str, float]
        Metrics with explicit denominators:

        - coverage_pass_total
            PASS / TOTAL
        - coverage_decided_total
            (PASS + FAIL) / TOTAL
        - risk_fail_given_decided
            FAIL / (PASS + FAIL)
        - risk_fail_total
            FAIL / TOTAL
        - fail_rate_total
            Alias of FAIL / TOTAL (kept for backward compatibility)

        Also includes count fields as floats:
        n_pass, n_fail, n_abstain, n_decided, n_total

    Notes
    -----
    "decided" = PASS ∪ FAIL (ABSTAIN excluded).
    FAIL is a negative decision produced by mechanical audits.
    """
    c = compute_counts(status)
    total = c["TOTAL"]
    decided = c["PASS"] + c["FAIL"]

    coverage_pass_total = (c["PASS"] / total) if total > 0 else float("nan")
    coverage_decided_total = (decided / total) if total > 0 else float("nan")
    risk_fail_total = (c["FAIL"] / total) if total > 0 else float("nan")
    risk_fail_given_decided = (c["FAIL"] / decided) if decided > 0 else float("nan")

    return {
        "coverage_pass_total": float(coverage_pass_total),
        "coverage_decided_total": float(coverage_decided_total),
        "risk_fail_given_decided": float(risk_fail_given_decided),
        "risk_fail_total": float(risk_fail_total),
        "fail_rate_total": float(risk_fail_total),  # keep old key stable
        "n_pass": float(c["PASS"]),
        "n_fail": float(c["FAIL"]),
        "n_abstain": float(c["ABSTAIN"]),
        "n_decided": float(decided),
        "n_total": float(c["TOTAL"]),
    }


def _validate_numeric_series(x: pd.Series, *, name: str) -> pd.Series:
    """
    Validate that a pandas Series is fully numeric and finite.

    Parameters
    ----------
    x : pandas.Series
        Input series.
    name : str
        Name used in error messages.

    Returns
    -------
    pandas.Series
        Numeric series (float-coercible) with the original index.

    Raises
    ------
    ValueError
        If non-numeric or non-finite values are detected.
    """
    v = pd.to_numeric(x, errors="coerce")
    if v.isna().any():
        bad_idx = v.index[v.isna()].tolist()
        head = bad_idx[:10]
        raise ValueError(
            f"{name}: non-numeric values detected (n_bad={len(bad_idx)}). example_indices={head}"
        )
    arr = v.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))[0]
        head = bad[:10].tolist()
        raise ValueError(
            f"{name}: non-finite values detected (n_bad={len(bad)}). example_positions={head}"
        )
    return v


def _default_thresholds_from_scores(
    scores: np.ndarray,
    *,
    max_thresholds: int = 200,
) -> list[float]:
    """
    Build a stable set of thresholds from a score distribution.

    Parameters
    ----------
    scores : numpy.ndarray
        1D score array.
    max_thresholds : int, optional
        Maximum number of thresholds to return.

    Returns
    -------
    list of float
        Sorted unique thresholds.

    Notes
    -----
    Policy:
    - If the number of unique score values is small, return all unique values.
    - Otherwise, return quantile-based thresholds over the full distribution.
    """
    if scores.size == 0:
        raise ValueError("risk_coverage_curve: empty score array")

    uniq = np.unique(scores)
    uniq = np.sort(uniq)

    if uniq.size <= max_thresholds:
        thrs = [float(x) for x in uniq.tolist()]
        return sorted(set(thrs))

    qs = np.linspace(0.0, 1.0, max_thresholds)
    thrs = np.quantile(scores, qs).astype(float).tolist()
    return sorted(set(float(x) for x in thrs))


def risk_coverage_curve(
    df: pd.DataFrame,
    *,
    score_col: str,
    status_col: str = "status",
    decision_thresholds: list[float] | None = None,
    pass_if_score_ge: bool = True,
    promote_abstain: bool = True,
    fail_on_degenerate: bool = False,
    max_thresholds: int = 200,
) -> pd.DataFrame:
    """
    Build a Risk–Coverage curve by sweeping a PASS threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing score and status columns.
    score_col : str
        Column name of probability-like or score values.
    status_col : str, optional
        Column name of base status. Default is "status".
    decision_thresholds : list of float or None, optional
        Thresholds to sweep. If None, thresholds are derived from scores.
    pass_if_score_ge : bool, optional
        If True, PASS when score >= threshold; else PASS when score <= threshold.
    promote_abstain : bool, optional
        If True, among non-FAIL items reassign:
        PASS if threshold satisfied else ABSTAIN.
        If False, gate only existing PASS -> ABSTAIN below threshold.
    fail_on_degenerate : bool, optional
        If True, raise on degenerate score distributions (<=1 unique value).
    max_thresholds : int, optional
        Max thresholds when auto-deriving. Must be >= 10.

    Returns
    -------
    pandas.DataFrame
        One row per threshold with risk/coverage metrics and metadata fields:
        threshold, score_col, status_col, pass_if_score_ge, promote_abstain.

    Raises
    ------
    ValueError
        If required columns are missing, scores are invalid, statuses are invalid,
        or thresholds are empty/invalid.

    Notes
    -----
    Safety semantics:
    - FAIL is never changed.
    - ABSTAIN never enters the risk denominator.
    """
    if score_col not in df.columns:
        raise ValueError(f"risk_coverage_curve: missing score_col={score_col}")
    if status_col not in df.columns:
        raise ValueError(f"risk_coverage_curve: missing status_col={status_col}")
    if max_thresholds < 10:
        raise ValueError("risk_coverage_curve: max_thresholds must be >= 10")

    scores_s = _validate_numeric_series(
        df[score_col], name=f"risk_coverage_curve.score_col={score_col}"
    )
    scores = scores_s.to_numpy(dtype=float)

    base_status = _shared.normalize_status_series(df[status_col])
    _shared.validate_status_values(base_status)

    # thresholds
    if decision_thresholds is not None:
        if len(decision_thresholds) == 0:
            raise ValueError("risk_coverage_curve: decision_thresholds is empty")
        thrs = sorted(set(float(x) for x in decision_thresholds))
    else:
        thrs = _default_thresholds_from_scores(scores, max_thresholds=int(max_thresholds))

    # Degenerate curve guardrail (paper-facing)
    if len(thrs) <= 1:
        msg = (
            "risk_coverage_curve: degenerate thresholds (score has <=1 unique value). "
            f"score_col={score_col} unique={len(np.unique(scores))}"
        )
        if fail_on_degenerate:
            raise ValueError(msg)
        # Keep at least one threshold so downstream doesn't crash
        if len(thrs) == 0:
            thrs = [float(scores[0])]

    out_rows: list[dict[str, Any]] = []

    # SAFETY: never change FAIL
    is_fail = base_status == "FAIL"
    not_fail = ~is_fail

    # Precompute "was_pass" for promote_abstain=False gating mode
    was_pass = base_status == "PASS"

    for thr in thrs:
        s = base_status.copy()

        if pass_if_score_ge:
            pass_mask = not_fail & (scores >= float(thr))
        else:
            pass_mask = not_fail & (scores <= float(thr))

        if promote_abstain:
            # Among non-FAIL: PASS if threshold satisfied else ABSTAIN
            s = s.where(is_fail, other="ABSTAIN")
            s = s.where(~pass_mask, other="PASS")
        else:
            # Only allow gating of already-PASS items
            to_abstain = was_pass & not_fail & (~pass_mask)
            s = s.where(~to_abstain, other="ABSTAIN")

        m = risk_coverage_from_status(s)
        m["threshold"] = float(thr)
        m["score_col"] = str(score_col)
        m["status_col"] = str(status_col)
        m["pass_if_score_ge"] = bool(pass_if_score_ge)
        m["promote_abstain"] = bool(promote_abstain)
        out_rows.append(m)

    return pd.DataFrame(out_rows)


# -----------------------------
# Calibration (Stage 2)
# -----------------------------
def _clip01(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Clip probabilities into (0, 1) with an epsilon margin.

    Parameters
    ----------
    p : numpy.ndarray
        Probability array.
    eps : float, optional
        Clipping epsilon. Default is 1e-6.

    Returns
    -------
    numpy.ndarray
        Clipped array.
    """
    return np.clip(p, eps, 1.0 - eps)


def _is_binary_labels(y: np.ndarray) -> bool:
    """
    Check whether labels are binary {0, 1}.

    Parameters
    ----------
    y : numpy.ndarray
        Label array.

    Returns
    -------
    bool
        True if unique values are a subset of {0, 1}.
    """
    uniq = np.unique(y)
    return set(uniq.tolist()).issubset({0, 1})


def _validate_calibration_inputs(
    probs: np.ndarray,
    y: np.ndarray | None,
    *,
    allow_unlabeled: bool,
) -> None:
    """
    Validate calibration inputs (probabilities and optional binary labels).

    Parameters
    ----------
    probs : numpy.ndarray
        1D probability-like array in [0, 1].
    y : numpy.ndarray or None
        Optional label array.
    allow_unlabeled : bool
        Whether to allow y=None for non-fitting modes.

    Raises
    ------
    ValueError
        If shapes/values are invalid, or if fitting is requested without labels,
        or if labels are not binary {0,1}, or if only one class is present.
    """
    if probs.ndim != 1:
        raise ValueError("calibrate: probs must be 1D array")
    if not np.isfinite(probs).all():
        raise ValueError("calibrate: probs contains non-finite values")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("calibrate: probs must be within [0, 1]")

    if y is None:
        if allow_unlabeled:
            return
        raise ValueError(
            "calibrate: y_true is required for fitting (refusing to learn without labels)"
        )

    if y.ndim != 1 or len(y) != len(probs):
        raise ValueError("calibrate: y_true must be 1D and same length as probs")

    if not np.isfinite(y).all():
        raise ValueError("calibrate: y_true contains non-finite values")

    if not _is_binary_labels(y):
        raise ValueError("calibrate: y_true must be binary {0,1}")

    if len(np.unique(y)) < 2:
        raise ValueError("calibrate: y_true has only one class; cannot fit calibration")


def _logit(p: np.ndarray) -> np.ndarray:
    """
    Compute logit(p) with internal clipping.

    Parameters
    ----------
    p : numpy.ndarray
        Probability array.

    Returns
    -------
    numpy.ndarray
        Logit-transformed array.
    """
    p = _clip01(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid(z).

    Parameters
    ----------
    z : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Sigmoid-transformed array.
    """
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature_scaling(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    grid: tuple[float, float, int] = (0.25, 10.0, 80),
) -> float:
    """
    Fit a single temperature T > 0 by minimizing NLL (binary labels).

    Model
    -----
    p' = sigmoid(logit(p) / T)

    Parameters
    ----------
    probs : numpy.ndarray
        1D probability-like array in [0, 1].
    y_true : numpy.ndarray
        1D binary labels in {0, 1}.
    grid : tuple[float, float, int], optional
        (t_min, t_max, n_grid). Search is performed in log-space.

    Returns
    -------
    float
        Best temperature T, clipped to a conservative range [0.25, 10.0].

    Raises
    ------
    ValueError
        If inputs are invalid or the grid is invalid.

    Notes
    -----
    No scipy dependency: uses deterministic grid search.
    """
    _validate_calibration_inputs(probs, y_true, allow_unlabeled=False)
    p = _clip01(np.asarray(probs, dtype=float))
    y = np.asarray(y_true, dtype=float)
    z = _logit(p)

    def nll(T: float) -> float:
        T = float(T)
        p2 = _sigmoid(z / T)
        p2 = _clip01(p2)
        return float(-(y * np.log(p2) + (1.0 - y) * np.log(1.0 - p2)).mean())

    t_min, t_max, n = grid
    t_min = float(t_min)
    t_max = float(t_max)
    n = int(n)
    if t_min <= 0 or t_max <= 0 or t_min >= t_max or n < 10:
        raise ValueError("fit_temperature_scaling: invalid grid")

    Ts = np.exp(np.linspace(np.log(t_min), np.log(t_max), n))
    losses = np.array([nll(T) for T in Ts], dtype=float)
    best_i = int(np.argmin(losses))
    T_best = float(Ts[best_i])

    # keep within a conservative, interpretable range
    return float(np.clip(T_best, 0.25, 10.0))


def apply_temperature_scaling(probs: np.ndarray, T: float) -> np.ndarray:
    """
    Apply temperature scaling to probability-like scores in [0, 1].

    Parameters
    ----------
    probs : numpy.ndarray
        1D probability-like array.
    T : float
        Temperature parameter (must be finite and > 0).

    Returns
    -------
    numpy.ndarray
        Calibrated probabilities clipped to (0, 1).

    Raises
    ------
    ValueError
        If T is invalid.
    """
    if not np.isfinite(T) or T <= 0:
        raise ValueError("apply_temperature_scaling: T must be finite and > 0")
    z = _logit(_clip01(np.asarray(probs, dtype=float)))
    return _clip01(_sigmoid(z / float(T)))


def fit_isotonic_regression(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> Any:
    """
    Fit isotonic regression mapping probs -> calibrated probs.

    Parameters
    ----------
    probs : numpy.ndarray
        1D probability-like array in [0, 1].
    y_true : numpy.ndarray
        1D binary labels in {0, 1}.

    Returns
    -------
    Any
        Fitted isotonic regression model (scikit-learn object).

    Raises
    ------
    ImportError
        If scikit-learn is not available.
    ValueError
        If inputs are invalid.
    """
    _validate_calibration_inputs(probs, y_true, allow_unlabeled=False)
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore
    except Exception as e:
        raise ImportError(
            "fit_isotonic_regression requires scikit-learn (sklearn). "
            "Install scikit-learn or use temperature scaling."
        ) from e

    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(np.asarray(probs, dtype=float), np.asarray(y_true, dtype=float))
    return ir


def apply_isotonic(model: Any, probs: np.ndarray) -> np.ndarray:
    """
    Apply a fitted isotonic regression model to probabilities.

    Parameters
    ----------
    model : Any
        Fitted isotonic regression model with `predict`.
    probs : numpy.ndarray
        Probability array.

    Returns
    -------
    numpy.ndarray
        Calibrated probabilities (float array).
    """
    return np.asarray(model.predict(np.asarray(probs, dtype=float)), dtype=float)


@dataclass(frozen=True)
class CalibrationResult:
    """
    Calibration result object.

    Attributes
    ----------
    method : {"none", "temperature", "isotonic"}
        Calibration method identifier.
    params : dict[str, Any]
        Method parameters:
        - temperature: {"T": float}
        - isotonic: {"model": fitted_model}
        - none: {}

    Notes
    -----
    This object is serializable only when params are JSON-safe.
    (isotonic model objects are not JSON-serializable by default.)
    """

    method: CalibMethod
    params: dict[str, Any]

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply the calibration mapping to probability-like scores.

        Parameters
        ----------
        probs : numpy.ndarray
            Probability array.

        Returns
        -------
        numpy.ndarray
            Calibrated probabilities clipped to (0, 1).

        Raises
        ------
        ValueError
            If `method` is unknown or required params are missing.
        """
        p = np.asarray(probs, dtype=float)
        if self.method == "none":
            return _clip01(p)
        if self.method == "temperature":
            T = float(self.params["T"])
            return apply_temperature_scaling(p, T)
        if self.method == "isotonic":
            model = self.params["model"]
            return _clip01(apply_isotonic(model, p))
        raise ValueError(f"Unknown method: {self.method}")


def calibrate_probs(
    probs: np.ndarray,
    y_true: np.ndarray | None,
    *,
    method: CalibMethod = "temperature",
    allow_unlabeled: bool = False,
) -> CalibrationResult:
    """
    Stage-2 calibration entry point.

    Parameters
    ----------
    probs : numpy.ndarray
        1D probability-like array in [0, 1].
    y_true : numpy.ndarray or None
        Optional binary labels in {0, 1}.
    method : {"none", "temperature", "isotonic"}, optional
        Calibration method. Default is "temperature".
    allow_unlabeled : bool, optional
        If True and y_true is None, returns a no-op calibration ("none").
        If False and y_true is None, refuses to fit.

    Returns
    -------
    CalibrationResult
        Calibration mapping object.

    Raises
    ------
    ValueError
        If inputs are invalid or fitting is requested without labels.

    Notes
    -----
    Design intent:
    - Keep dependencies optional (no scipy).
    - Temperature scaling uses deterministic grid search.
    """
    p = np.asarray(probs, dtype=float)

    if y_true is None:
        if allow_unlabeled:
            _validate_calibration_inputs(p, None, allow_unlabeled=True)
            return CalibrationResult(method="none", params={})
        _validate_calibration_inputs(p, None, allow_unlabeled=False)  # raises

    y = np.asarray(y_true, dtype=float)

    if method == "temperature":
        T = fit_temperature_scaling(p, y)
        return CalibrationResult(method="temperature", params={"T": T})

    if method == "isotonic":
        model = fit_isotonic_regression(p, y)
        return CalibrationResult(method="isotonic", params={"model": model})

    if method == "none":
        return CalibrationResult(method="none", params={})

    raise ValueError(f"calibrate_probs: unknown method={method}")


# -----------------------------
# Convenience: from audit_log
# -----------------------------
def extract_probs_and_labels(
    audit_log: pd.DataFrame,
    *,
    prob_col: str,
    label_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Extract probability-like scores and optional strict binary labels.

    Parameters
    ----------
    audit_log : pandas.DataFrame
        Audit log table.
    prob_col : str
        Column name containing probabilities/scores.
    label_col : str or None, optional
        Column name containing labels. Only exact {0,1} accepted.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None]
        (probs, labels). Labels are returned as int array when provided.

    Raises
    ------
    ValueError
        If columns are missing or values are non-numeric/non-finite, or labels
        are not exactly binary {0,1}.
    """
    if prob_col not in audit_log.columns:
        raise ValueError(f"extract_probs_and_labels: missing prob_col={prob_col}")

    probs_s = pd.to_numeric(audit_log[prob_col], errors="coerce")
    if probs_s.isna().any():
        bad_idx = probs_s.index[probs_s.isna()].tolist()
        head = bad_idx[:10]
        raise ValueError(
            f"extract_probs_and_labels: non-numeric prob detected (n_bad={len(bad_idx)}). "
            f"example_indices={head}"
        )
    probs = probs_s.to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        bad = np.where(~np.isfinite(probs))[0]
        head = bad[:10].tolist()
        raise ValueError(
            f"extract_probs_and_labels: prob contains non-finite values (n_bad={len(bad)}). "
            f"example_positions={head}"
        )

    y = None
    if label_col is not None:
        if label_col not in audit_log.columns:
            raise ValueError(f"extract_probs_and_labels: missing label_col={label_col}")

        yv = pd.to_numeric(audit_log[label_col], errors="coerce")
        if yv.isna().any():
            bad_idx = yv.index[yv.isna()].tolist()
            head = bad_idx[:10]
            raise ValueError(
                f"extract_probs_and_labels: non-numeric label detected (n_bad={len(bad_idx)}). "
                f"example_indices={head}"
            )

        y_arr = yv.to_numpy(dtype=float)
        if not np.isfinite(y_arr).all():
            bad = np.where(~np.isfinite(y_arr))[0]
            head = bad[:10].tolist()
            raise ValueError(
                f"extract_probs_and_labels: label contains non-finite values (n_bad={len(bad)}). "
                f"example_positions={head}"
            )

        uniq = set(np.unique(y_arr).tolist())
        if uniq - {0.0, 1.0}:
            raise ValueError(
                f"extract_probs_and_labels: labels must be binary {{0,1}}, got {sorted(uniq)}"
            )
        y = y_arr.astype(int)

    return probs.astype(float), (None if y is None else y)
