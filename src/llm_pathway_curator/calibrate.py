# LLM-PathwayCurator/src/llm_pathway_curator/calibrate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

Status = Literal["PASS", "ABSTAIN", "FAIL"]
CalibMethod = Literal["none", "temperature", "isotonic"]


# -----------------------------
# Core definitions: Risk/Coverage
# -----------------------------
def compute_counts(status: pd.Series) -> dict[str, int]:
    s = status.astype(str).str.strip().str.upper()
    n_pass = int((s == "PASS").sum())
    n_fail = int((s == "FAIL").sum())
    n_abs = int((s == "ABSTAIN").sum())
    n_total = int(len(s))
    return {"PASS": n_pass, "FAIL": n_fail, "ABSTAIN": n_abs, "TOTAL": n_total}


def risk_coverage_from_status(status: pd.Series) -> dict[str, float]:
    """
    Spec-safe metrics with explicit denominators.

    Coverage:
      - coverage_pass_total = PASS / TOTAL   (PASS率; ABSTAINは分母に含まれる)

    Risk (answered-subset risk; ABSTAIN excluded):
      - risk_fail_given_decided = FAIL / (PASS + FAIL)

    Also provide "audit failure rate" over all:
      - risk_fail_total = FAIL / TOTAL
      - fail_rate_total = FAIL / TOTAL  (alias; kept for backward compatibility)
    """
    c = compute_counts(status)
    total = c["TOTAL"]
    decided = c["PASS"] + c["FAIL"]

    coverage_pass_total = (c["PASS"] / total) if total > 0 else float("nan")
    risk_fail_total = (c["FAIL"] / total) if total > 0 else float("nan")

    # IMPORTANT: do NOT include ABSTAIN in denominator here
    risk_fail_given_decided = (c["FAIL"] / decided) if decided > 0 else float("nan")

    return {
        "coverage_pass_total": float(coverage_pass_total),
        "risk_fail_given_decided": float(risk_fail_given_decided),
        "risk_fail_total": float(risk_fail_total),
        "fail_rate_total": float(risk_fail_total),  # keep old key stable
        "n_pass": float(c["PASS"]),
        "n_fail": float(c["FAIL"]),
        "n_abstain": float(c["ABSTAIN"]),
        "n_total": float(c["TOTAL"]),
    }


def risk_coverage_curve(
    df: pd.DataFrame,
    *,
    score_col: str,
    status_col: str = "status",
    decision_thresholds: list[float] | None = None,
    pass_if_score_ge: bool = True,
    promote_abstain: bool = True,
    fail_on_degenerate: bool = False,
) -> pd.DataFrame:
    """
    Build a Risk–Coverage curve by sweeping a PASS threshold.

    Intended semantics (selective prediction):
      - FAIL stays FAIL always (safety)
      - Among non-FAIL items:
          PASS if score passes threshold, else ABSTAIN
        (This avoids the pitfall: "only already-PASS can be gated".)

    Guardrails:
      - ABSTAIN must NOT enter risk denominator (handled by risk_coverage_from_status).
      - score must be finite numeric.
    """
    if score_col not in df.columns:
        raise ValueError(f"risk_coverage_curve: missing score_col={score_col}")
    if status_col not in df.columns:
        raise ValueError(f"risk_coverage_curve: missing status_col={status_col}")

    scores = pd.to_numeric(df[score_col], errors="coerce")
    if scores.isna().any():
        i = int(scores.index[scores.isna()][0])
        raise ValueError(f"risk_coverage_curve: non-numeric score at row index={i}")
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("risk_coverage_curve: score contains non-finite values")

    if decision_thresholds is None:
        uniq = np.unique(scores.to_numpy())
        uniq = np.sort(uniq)
        if len(uniq) == 0:
            raise ValueError("risk_coverage_curve: empty score array")
        if len(uniq) <= 400:
            decision_thresholds = [float(x) for x in uniq.tolist()]
        else:
            decision_thresholds = [float(x) for x in np.quantile(uniq, np.linspace(0, 1, 200))]

    # Degenerate curve guardrail (paper-facing)
    if len(decision_thresholds) <= 1:
        msg = (
            "risk_coverage_curve: degenerate thresholds (score has <=1 unique value). "
            f"score_col={score_col}"
        )
        if fail_on_degenerate:
            raise ValueError(msg)
        # keep behavior but make it explicit
        decision_thresholds = list(decision_thresholds) if decision_thresholds is not None else []

    base_status = df[status_col].astype(str).str.strip().str.upper()
    out_rows: list[dict[str, Any]] = []

    for thr in decision_thresholds:
        s = base_status.copy()

        # SAFETY: never change FAIL
        is_fail = s == "FAIL"
        not_fail = ~is_fail

        if pass_if_score_ge:
            pass_mask = not_fail & (scores >= float(thr))
        else:
            pass_mask = not_fail & (scores <= float(thr))

        if promote_abstain:
            # Among non-FAIL: PASS if threshold satisfied else ABSTAIN
            s = s.where(~not_fail, other="ABSTAIN")
            s = s.where(~pass_mask, other="PASS")
        else:
            # Only allow gating of already-PASS items (more conservative)
            # - PASS can drop to ABSTAIN if below threshold
            # - ABSTAIN stays ABSTAIN
            # - FAIL stays FAIL
            was_pass = s == "PASS"
            to_abstain = was_pass & not_fail & (~pass_mask)
            s = s.where(~to_abstain, other="ABSTAIN")
            # keep PASS where pass_mask, keep ABSTAIN where already ABSTAIN

        m = risk_coverage_from_status(s)
        m["threshold"] = float(thr)
        m["promote_abstain"] = bool(promote_abstain)
        out_rows.append(m)

    return pd.DataFrame(out_rows)


# -----------------------------
# Calibration (Stage 2)
# -----------------------------
def _clip01(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _is_binary_labels(y: np.ndarray) -> bool:
    uniq = np.unique(y)
    return set(uniq.tolist()).issubset({0, 1})


def _validate_calibration_inputs(
    probs: np.ndarray,
    y: np.ndarray | None,
    *,
    allow_unlabeled: bool,
) -> None:
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
    p = _clip01(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature_scaling(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    grid: tuple[float, float, int] = (0.25, 10.0, 80),
) -> float:
    """
    Fit a single temperature T > 0 on probabilities (binary) using NLL.
    Model: p' = sigmoid(logit(p) / T)

    Stable v0 approach (no scipy):
      - coarse log-space grid search over T
    """
    _validate_calibration_inputs(probs, y_true, allow_unlabeled=False)
    p = _clip01(probs)
    y = y_true.astype(float)
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
    T_best = float(Ts[int(np.argmin(losses))])

    # constrain extreme temps (avoid pathological output)
    return float(np.clip(T_best, 0.25, 10.0))


def apply_temperature_scaling(probs: np.ndarray, T: float) -> np.ndarray:
    if not np.isfinite(T) or T <= 0:
        raise ValueError("apply_temperature_scaling: T must be finite and > 0")
    z = _logit(_clip01(np.asarray(probs, dtype=float)))
    return _sigmoid(z / float(T))


def fit_isotonic_regression(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> Any:
    """
    Fit isotonic regression mapping probs -> calibrated probs.
    Uses scikit-learn if available.
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
    ir.fit(probs.astype(float), y_true.astype(float))
    return ir


def apply_isotonic(model: Any, probs: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(np.asarray(probs, dtype=float)), dtype=float)


@dataclass(frozen=True)
class CalibrationResult:
    method: CalibMethod
    params: dict[str, Any]

    def apply(self, probs: np.ndarray) -> np.ndarray:
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
    Stage 2 calibration:
      - If y_true is None and allow_unlabeled is False => refuse to fit
      - If y_true is None and allow_unlabeled is True => return method="none"
      - Otherwise fit requested method.
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
    Extract probability-like scores + optional binary labels.
    """
    if prob_col not in audit_log.columns:
        raise ValueError(f"extract_probs_and_labels: missing prob_col={prob_col}")
    probs = pd.to_numeric(audit_log[prob_col], errors="coerce")
    if probs.isna().any():
        i = int(probs.index[probs.isna()][0])
        raise ValueError(f"extract_probs_and_labels: non-numeric prob at row index={i}")
    if not np.isfinite(probs.to_numpy()).all():
        raise ValueError("extract_probs_and_labels: prob contains non-finite values")

    y = None
    if label_col is not None:
        if label_col not in audit_log.columns:
            raise ValueError(f"extract_probs_and_labels: missing label_col={label_col}")
        yv = pd.to_numeric(audit_log[label_col], errors="coerce")
        if yv.isna().any():
            i = int(yv.index[yv.isna()][0])
            raise ValueError(f"extract_probs_and_labels: non-numeric label at row index={i}")
        if not np.isfinite(yv.to_numpy()).all():
            raise ValueError("extract_probs_and_labels: label contains non-finite values")

        y_int = yv.to_numpy().astype(int)
        if set(np.unique(y_int).tolist()) - {0, 1}:
            raise ValueError("extract_probs_and_labels: labels must be binary {0,1}")
        y = y_int

    return probs.to_numpy().astype(float), (None if y is None else y)
