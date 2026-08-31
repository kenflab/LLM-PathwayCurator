#!/usr/bin/env python3
"""Small shared helpers for immutable CRM R1 V11 result locks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def normalized(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.loc[:, list(columns)].copy()
    for column in result:
        result[column] = result[column].fillna("").astype(str).str.strip()
    return result


def as_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    mapped = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    require(mapped.notna().all(), f"Invalid Boolean values in {name}")
    return mapped.astype(bool)


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    require(total > 0, "Wilson interval requires a positive denominator")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def fleiss_kappa(ratings: pd.DataFrame, categories: list[str]) -> float:
    """Return Fleiss' kappa for a complete subjects-by-raters categorical table."""

    require(not ratings.empty, "Fleiss kappa requires ratings")
    require(not ratings.isna().any().any(), "Fleiss kappa requires complete ratings")
    n_raters = ratings.shape[1]
    require(n_raters >= 2, "Fleiss kappa requires at least two raters")
    counts = pd.DataFrame(
        {category: ratings.eq(category).sum(axis=1) for category in categories},
        index=ratings.index,
    )
    require(counts.sum(axis=1).eq(n_raters).all(), "Unknown category in Fleiss input")
    subject_agreement = (
        (counts.pow(2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    ).mean()
    marginal = counts.sum(axis=0) / (len(counts) * n_raters)
    expected = float(marginal.pow(2).sum())
    if math.isclose(expected, 1.0):
        return float("nan")
    return float((subject_agreement - expected) / (1 - expected))


def overlap_aware_exact(
    outcomes: pd.Series,
    selected_a: pd.Series,
    selected_b: pd.Series,
    *,
    alternative: str,
) -> dict[str, float | int | str]:
    """Conditional exact reference for two equal-K, overlapping fixed selections.

    Membership is exchangeable only within the symmetric difference. Common claims
    cancel. The reference is descriptive because pathway outcomes are dependent.
    """

    require(alternative in {"greater", "less"}, "Unsupported exact-test alternative")
    y = as_bool(outcomes, name="outcomes")
    a = as_bool(selected_a, name="selected_a")
    b = as_bool(selected_b, name="selected_b")
    require(int(a.sum()) == int(b.sum()), "Exact comparison requires equal K")
    a_only = a & ~b
    b_only = b & ~a
    require(int(a_only.sum()) == int(b_only.sum()), "Symmetric-difference sizes differ")
    n_each = int(a_only.sum())
    common = int((a & b).sum())
    k = int(a.sum())
    successes_union = int(y.loc[a_only | b_only].sum())
    observed_a_only = int(y.loc[a_only].sum())
    observed_difference = float(y.loc[a].mean() - y.loc[b].mean())
    denominator = math.comb(2 * n_each, n_each)
    support: list[tuple[int, float, float]] = []
    for x in range(max(0, successes_union - n_each), min(n_each, successes_union) + 1):
        probability = (
            math.comb(successes_union, x)
            * math.comb(2 * n_each - successes_union, n_each - x)
            / denominator
        )
        difference = (2 * x - successes_union) / k
        support.append((x, probability, difference))
    tolerance = 1e-12
    if alternative == "greater":
        one_sided = sum(
            probability
            for _, probability, diff in support
            if diff >= observed_difference - tolerance
        )
    else:
        one_sided = sum(
            probability
            for _, probability, diff in support
            if diff <= observed_difference + tolerance
        )
    two_sided = sum(
        probability
        for _, probability, diff in support
        if abs(diff) >= abs(observed_difference) - tolerance
    )
    return {
        "common_claims": common,
        "a_only_claims": n_each,
        "b_only_claims": n_each,
        "exclusive_successes": successes_union,
        "observed_a_only_successes": observed_a_only,
        "observed_fraction_difference_a_minus_b": observed_difference,
        "alternative": alternative,
        "p_one_sided": min(1.0, float(one_sided)),
        "p_two_sided": min(1.0, float(two_sided)),
        "assignments": denominator,
        "interpretation": (
            "descriptive conditional exact reference; pathways are biologically dependent"
        ),
    }
