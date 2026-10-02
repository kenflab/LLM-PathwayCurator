"""Independent, read-only V15 diagnostics; never import or run the frozen pipeline."""

from __future__ import annotations

import hashlib
import json
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["cohort_id", "split_id", "claim_uid"]
PAIR = KEY[:2]
METHODS = ["raw_pool", "q_value_matched", "stability_matched", "full_audit"]
CONTEXT_COLUMNS = [
    "context_score_proxy_u01_norm",
    "context_score_proxy_u01",
    "context_score_u01_norm",
    "context_score_u01",
    "context_score",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def read_tsv(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)


def boolean(values):
    result = (
        values.astype(str)
        .str.lower()
        .str.strip()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    require(result.notna().all(), f"Invalid boolean in {values.name}")
    return result.astype(bool)


def numeric(values, bounded=False):
    result = pd.to_numeric(values, errors="coerce")
    require(np.isfinite(result).all(), f"Non-finite value in {values.name}")
    if bounded:
        require(result.between(0, 1).all(), f"Out of [0,1]: {values.name}")
    return result


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_tsv(path, frame):
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


@contextmanager
def new_output(path):
    path = Path(path).resolve()
    require(not path.exists(), f"Output already exists: {path}")
    require(
        not any("_lock" in p or p.startswith("final_v14") for p in path.parts),
        "Diagnostic output cannot be inside a frozen bundle",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v15-", dir=path.parent) as temp:
        root = Path(temp) / "result"
        root.mkdir()
        yield root
        root.rename(path)


def finish(root, inputs, details):
    for item in inputs:
        require(sha256(item["path"]) == item["sha256"], "Input changed during diagnosis")
    write_json(
        root / "run_manifest.json",
        {
            "schema": "CRM_R1_V15_READ_ONLY_DIAGNOSTIC_1",
            "created_utc": datetime.now(UTC).isoformat(),
            "post_hoc": True,
            "model_calls": 0,
            "frozen_inputs_modified": False,
            "inputs": inputs,
            "outputs": [
                {**record(p), "path": p.name} for p in sorted(root.iterdir()) if p.is_file()
            ],
            "details": details,
        },
    )
