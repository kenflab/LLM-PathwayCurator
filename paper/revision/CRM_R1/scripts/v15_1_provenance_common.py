"""Independent V15.1 development tools; never import or rewrite frozen runners."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "CRM_R1_PROVENANCE_CORRECTION_v15_1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def read_json(path):
    def checked_object(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, f"Duplicate JSON key: {key}")
            obj[key] = value
        return obj

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=checked_object)


def read_table(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_table(path, table):
    table.to_csv(path, sep="\t", index=False, lineterminator="\n")


def numeric(values, *, lower=None, upper=None):
    require(not values.map(lambda x: isinstance(x, bool)).any(), "Boolean numeric value")
    result = pd.to_numeric(values, errors="coerce")
    require(np.isfinite(result).all(), "Missing or nonfinite numeric value")
    if lower is not None:
        require(result.ge(lower).all(), f"Numeric value below {lower}")
    if upper is not None:
        require(result.le(upper).all(), f"Numeric value above {upper}")
    return result


def unique(table, key):
    require(key in table, f"Missing column: {key}")
    values = table[key].astype(str)
    require(values.str.strip().ne("").all(), f"Blank {key}")
    require(values.eq(values.str.strip()).all(), f"Whitespace in {key}")
    require(not values.duplicated().any(), f"Duplicate {key}")


@contextmanager
def fresh_output(path):
    """Leave failure records in a new directory; never overwrite or delete an old run."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    except BaseException as error:
        write_json(path / "FAILED.json", {"error": str(error), "completed": False})
        raise


def finish(path, inputs, summary):
    for item in inputs:
        require(sha(item["path"]) == item["sha256"], "Input changed during execution")
    write_json(path / "summary.json", summary)
    outputs = {
        p.name: {"sha256": sha(p), "size_bytes": p.stat().st_size}
        for p in sorted(path.iterdir())
        if p.is_file()
    }
    write_json(
        path / "manifest.json",
        {
            "schema": VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "inputs": inputs,
            "outputs": outputs,
            "frozen_inputs_modified": False,
            "model_calls": 0,
            "role": "posthoc_development_not_confirmatory_validation",
        },
    )


def code_records(script):
    return [record(script), record(Path(__file__))]


def proxy_v2(ctx_id, context_keys, term_uid):
    """Exact historical pipeline payload, including json.dumps default separators."""
    payload = json.dumps(
        {
            "ctx_id": ctx_id.strip(),
            "context_keys": [key.strip() for key in context_keys if key.strip()],
            "term_uid": term_uid.strip(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:13], 16) / float(1 << 52)


def stored_es(table, evidence_column, stability_column):
    """A declared E*S diagnostic, NOT E*S*C with a missing C silently filled to one."""
    unique(table, "claim_id")
    unique(table, "term_uid")
    require(len(table) > 0, "Empty candidate table")
    require(evidence_column != stability_column, "E and S need distinct columns")
    for name in (evidence_column, stability_column):
        require(name in table, f"Explicit column missing: {name}")
        require(
            not any(token in name.lower() for token in ("context", "confidence", "proxy", "hash")),
            "Context/proxy fields cannot supply E or S",
        )
    evidence = numeric(table[evidence_column], lower=0)
    stability = numeric(table[stability_column], lower=0, upper=1)
    out = table[["claim_id", "term_uid"]].copy()
    out["evidence_component"] = evidence
    out["stability_component"] = stability
    out["stored_es_diagnostic"] = evidence * stability
    require(np.isfinite(out.stored_es_diagnostic).all(), "Score overflow")
    out = out.sort_values(
        ["stored_es_diagnostic", "term_uid", "claim_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    out["rank_stored_es"] = range(1, len(out) + 1)
    out["candidate_scope"] = "ALL_INPUT_ROWS_NO_DECISION_FILTER"
    out["context_used_for_score_or_filter"] = False
    return out
