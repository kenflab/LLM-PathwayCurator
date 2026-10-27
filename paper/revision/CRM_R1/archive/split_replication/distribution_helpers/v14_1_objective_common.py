#!/usr/bin/env python3
"""Shared helpers for immutable CRM R1 V14.1 objective-validation locks."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def software_versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("pandas", "numpy", "matplotlib"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
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


def normalized_strings(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.loc[:, columns].copy()
    for column in columns:
        result[column] = result[column].fillna("").astype(str).str.strip()
    return result


def require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = required - set(frame.columns)
    require(not missing, f"{label} missing columns: {sorted(missing)}")


def validate_manifest_outputs(manifest_path: Path, schema: str) -> dict[str, Path]:
    require(manifest_path.is_file(), f"Missing manifest: {manifest_path}")
    digest_path = manifest_path.with_suffix(".sha256")
    require(digest_path.is_file(), f"Missing manifest digest: {digest_path}")
    tokens = digest_path.read_text(encoding="utf-8").split()
    require(len(tokens) >= 1, f"Malformed manifest digest: {digest_path}")
    require(tokens[0] == sha256_file(manifest_path), f"Manifest drift: {manifest_path}")
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == schema, f"Schema drift: {manifest_path}")
    paths: dict[str, Path] = {}
    for label, record in manifest.get("outputs", {}).items():
        path = Path(record["path"])
        require(path.is_file(), f"Missing locked output: {label}: {path}")
        require(
            sha256_file(path) == record["sha256"],
            f"Locked output drift: {label}: {path}",
        )
        paths[label] = path
    return paths


def write_manifest_digest(manifest_path: Path) -> Path:
    digest_path = manifest_path.with_suffix(".sha256")
    digest_path.write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    return digest_path


def bootstrap_mean(
    values: np.ndarray, *, draws: int, seed: int
) -> tuple[float, float, float]:
    require(values.ndim == 1 and len(values) > 0, "Bootstrap requires a vector")
    require(np.isfinite(values).all(), "Bootstrap values must be finite")
    require(draws > 0, "Bootstrap draws must be positive")
    observed = float(values.mean())
    if len(values) == 1:
        return observed, observed, observed
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return observed, float(low), float(high)
