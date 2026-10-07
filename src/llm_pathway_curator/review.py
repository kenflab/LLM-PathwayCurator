"""Public enrichment reporting and draft inspection, with no model calls."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import _shared
from .grounding import RULESET, factual_statement, inspect_text


@dataclass(frozen=True)
class ReviewConfig:
    """Source workflow inputs and the declared statistical reporting rule.

    Parameters
    ----------
    evidence_table
        UTF-8 enrichment TSV with source identities, statistics and supporting genes.
    sample_card
        JSON study description requiring an explicit comparison direction.
    outdir
        New or empty directory; prior outputs are never overwritten.
    claims_file
        Optional TSV of unchanged draft text linked to the source table.
    q_threshold
        Adjusted-value cutoff for source statistical statement eligibility.
    k_claims
        Optional cap on eligible source statements. All candidates remain exported.
    """

    evidence_table: str
    sample_card: str
    outdir: str
    claims_file: str | None = None
    q_threshold: float = 0.05
    k_claims: int | None = None


@dataclass(frozen=True)
class ReviewResult:
    """Paths to the complete source report, checked drafts and run metadata."""

    run_id: str
    outdir: str
    artifacts: dict[str, str]
    meta_path: str


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _number(value: str, field: str) -> float | None:
    if value.strip().lower() in {"", "na", "nan", "none", "null"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {field}")
    if field == "qval" and not 0 <= result <= 1:
        raise ValueError("qval must be in [0,1]")
    return result


def _read_table(path: Path) -> tuple[bytes, list[dict[str, str]]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    names = reader.fieldnames or []
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Missing or duplicate column names in {path.name}")
    rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Malformed TSV row in {path.name}")
    return raw, rows


def _evidence(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw, rows = _read_table(path)
    required = {"term_id", "term_name", "source", "stat", "qval", "direction", "evidence_genes"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(
            "Require a nonempty EvidenceTable with term_id, term_name, source, "
            "stat, qval, direction and evidence_genes"
        )
    seen: set[str] = set()
    result = []
    for index, row in enumerate(rows, 2):
        if any(not row[k].strip() for k in ("term_id", "term_name", "source")):
            raise ValueError(f"Blank evidence identity at TSV line {index}")
        source = row["source"].strip()
        uid = f"{source}:{row['term_id'].strip()}"
        if row.get("term_uid", uid).strip() != uid:
            raise ValueError(f"term_uid does not match source:term_id at line {index}")
        if uid in seen:
            raise ValueError(
                f"Duplicate term_uid: {uid}. Use distinct source identifiers or separate contrasts."
            )
        seen.add(uid)
        direction = row["direction"].strip().lower()
        if direction not in {"up", "down", "na"}:
            raise ValueError(f"Direction must be up/down/na at line {index}")
        stat = _number(row["stat"], "stat")
        kind = row.get("stat_kind", "").strip() or (
            "NES" if source.lower().startswith("fgsea") else "source statistic"
        )
        if kind.upper() == "NES" and stat is not None and direction in {"up", "down"}:
            if (stat > 0 and direction != "up") or (stat < 0 and direction != "down") or stat == 0:
                raise ValueError(f"NES direction mismatch at line {index}")
        genes = sorted(set(_shared.parse_genes(row["evidence_genes"])))
        item = {
            "term_uid": uid,
            "term_id": row["term_id"].strip(),
            "term_name": row["term_name"].strip(),
            "source": source,
            "stat": stat,
            "stat_kind": kind,
            "qval": _number(row["qval"], "qval"),
            "direction": direction,
            "evidence_genes": genes,
            "source_locator": f"TSV data record {index - 1}",
            "source_file_sha256": _sha(raw),
        }
        item["evidence_sha256"] = _sha(_json(item).encode())
        item["gene_set_sha256"] = _sha(_json(genes).encode())
        result.append(item)
    return raw, result


def _claims(path: Path | None, evidence: list[dict]) -> tuple[bytes | None, dict[str, list[dict]]]:
    if path is None:
        return None, {}
    raw, rows = _read_table(path)
    if not rows or "text" not in rows[0]:
        raise ValueError("Claims TSV requires a text column and term_uid or term_id")
    by_uid = {item["term_uid"]: item for item in evidence}
    by_id: dict[str, list[str]] = {}
    for item in evidence:
        by_id.setdefault(item["term_id"], []).append(item["term_uid"])
    result: dict[str, list[dict]] = {}
    ids: set[str] = set()
    for index, row in enumerate(rows, 2):
        uid = row.get("term_uid", "").strip()
        if not uid:
            tid = row.get("term_id", "").strip()
            source = row.get("source", "").strip()
            matches = [f"{source}:{tid}"] if source else by_id.get(tid, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Ambiguous or absent term identity at claims line {index}; provide term_uid"
                )
            uid = matches[0]
        if uid not in by_uid:
            raise ValueError(f"Unknown evidence link at claims line {index}: {uid}")
        for field in ("term_id", "source"):
            declared = row.get(field, "").strip()
            if declared and declared != by_uid[uid][field]:
                raise ValueError(
                    f"Conflicting evidence identity at claims line {index}: "
                    f"{field} disagrees with term_uid {uid}"
                )
        cid = row.get("claim_id", "").strip() or f"submitted_{index - 1}"
        if cid in ids:
            raise ValueError(f"Duplicate claim_id: {cid}")
        ids.add(cid)
        result.setdefault(uid, []).append({**row, "claim_id": cid, "term_uid": uid})
    return raw, result


def _write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _json(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key)
                    for key in fields
                }
            )


def _html(records: list[dict], comparison: str) -> str:
    esc = html.escape
    cards = []
    for record in records:
        source = record["evidence"]
        texts = []
        for review in record["submitted_reviews"]:
            flags = "".join(
                f"<li><strong>{esc(x['quote'])}</strong>: {esc(x['message'])}</li>"
                for x in review["findings"]
            )
            texts.append(
                "<div class='draft'><h4>Submitted draft · "
                f"{esc(review['limited_checks_status'])}</h4>"
                f"<p class='verbatim'>{esc(review['text'])}</p><ul>{flags}</ul>"
                "<p class='small'>Explicit numeric coverage: "
                f"{esc(review['numeric_coverage'])}. Free prose requires human review.</p></div>"
            )
        search = esc((source["term_name"] + " " + source["term_uid"]).lower(), quote=True)
        chosen = str(record["selected"]).lower()
        cards.append(
            f"<article data-search='{search}' data-selected='{chosen}'><header>"
            f"<h3>{esc(source['term_name'])}</h3>"
            f"<span class='badge'>{record['decision_status']}</span></header>"
            f"<p>{esc(record['source_statement'])}</p><p class='small'>"
            f"{esc(source['term_uid'])} · {esc(source['source_locator'])}</p>"
            "<details><summary>Supporting genes and evidence identity</summary>"
            f"<p>{esc('; '.join(source['evidence_genes']))}</p>"
            f"<p class='small'>Evidence SHA-256: {esc(source['evidence_sha256'])}</p>"
            f"</details>{''.join(texts)}</article>"
        )
    head = """<!doctype html>
<html lang='en'>
<head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Pathway source report</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;background:#f4f5f7;color:#182230;
max-width:1050px;margin:32px auto;padding:0 22px}
h1{font-size:30px}
article{background:white;border:1px solid #d9e0e8;border-radius:10px;
padding:20px;margin:18px 0}
header{display:flex;justify-content:space-between;gap:18px;align-items:start}
h3{margin:0;font-size:19px}
.badge{border-radius:6px;background:#edf0f8;padding:3px 9px;font-size:13px}
.draft{border-top:1px solid #d9e0e8;margin-top:18px}
.small{font-size:13px;color:#536276;overflow-wrap:anywhere}
input[type=search]{padding:10px;width:min(450px,85%);font:inherit;
border:1px solid #aab6c4;border-radius:6px}
li{margin:8px 0}p{overflow-wrap:anywhere}.verbatim{white-space:pre-wrap}
</style></head><body><h1>Pathway source report</h1><p>"""
    note = """</p><p>Source statements use the supplied statistics.
PASS refers to statistical reporting eligibility under the declared cutoff.
Submitted prose is inspected by limited rules and requires human review.</p>
<input id='search' type='search' placeholder='Find a pathway or identifier'
aria-label='Search pathways'>
<label><input id='selected' type='checkbox'> Selected only</label>"""
    script = """<script>
function filter(){
  const q=document.querySelector('#search').value.toLowerCase();
  const only=document.querySelector('#selected').checked;
  document.querySelectorAll('article').forEach(x=>{
    x.hidden=!x.dataset.search.includes(q)||(only&&x.dataset.selected!=='true');
  });
}
document.querySelector('#search').addEventListener('input',filter);
document.querySelector('#selected').addEventListener('change',filter);
</script></body></html>"""
    return head + esc(comparison) + note + "".join(cards) + script


def review_enrichment(cfg: ReviewConfig, *, run_id: str | None = None) -> ReviewResult:
    """Keep the complete census, write source statements, inspect supplied prose.

    Validation precedes output creation. Existing output artifacts are never
    overwritten. No environment variable selects a model or a proxy gate.
    """
    if not math.isfinite(cfg.q_threshold) or not 0 < cfg.q_threshold < 1:
        raise ValueError("q_threshold must be finite and between 0 and 1")
    if cfg.k_claims is not None and (type(cfg.k_claims) is not int or cfg.k_claims < 1):
        raise ValueError("k_claims must be a positive integer")
    ev_path, card_path = Path(cfg.evidence_table), Path(cfg.sample_card)
    raw_ev, evidence = _evidence(ev_path)
    raw_card = card_path.read_bytes()
    card = json.loads(raw_card)
    if (
        not isinstance(card, dict)
        or not isinstance(card.get("comparison"), str)
        or not card["comparison"].strip()
    ):
        raise ValueError(
            "Sample Card must contain a nonblank comparison string, "
            "with its direction explicitly described"
        )
    raw_claims, claims = _claims(Path(cfg.claims_file) if cfg.claims_file else None, evidence)
    ranked = sorted(
        (e for e in evidence if e["qval"] is not None and e["qval"] <= cfg.q_threshold),
        key=lambda e: (e["qval"], e["term_uid"]),
    )
    selected = {e["term_uid"] for e in ranked[: cfg.k_claims]}
    records, flat_reviews = [], []
    for item in evidence:
        reviews = []
        for submitted in claims.get(item["term_uid"], []):
            check = inspect_text(submitted["text"], item, cfg.q_threshold)
            for field in ("comparison", "condition", "tissue", "perturbation"):
                asserted = submitted.get(field, "").strip()
                if asserted and asserted != str(card.get(field, "")):
                    check["findings"].append(
                        {
                            "code": "CONTEXT_ATTRIBUTE_MISMATCH",
                            "severity": "ERROR",
                            "start": None,
                            "end": None,
                            "quote": asserted,
                            "source_value": card.get(field),
                            "message": f"The declared {field} does not match the Sample Card.",
                        }
                    )
            expected_hash = submitted.get("evidence_sha256", "").strip()
            if expected_hash and expected_hash != item["evidence_sha256"]:
                check["findings"].append(
                    {
                        "code": "EVIDENCE_IDENTITY_MISMATCH",
                        "severity": "ERROR",
                        "start": None,
                        "end": None,
                        "quote": expected_hash,
                        "source_value": item["evidence_sha256"],
                        "message": "The draft refers to a different source evidence record.",
                    }
                )
            if not submitted["text"].strip():
                check["findings"].append(
                    {
                        "code": "EMPTY_DRAFT",
                        "severity": "REVIEW",
                        "start": None,
                        "end": None,
                        "quote": "",
                        "source_value": None,
                        "message": "No submitted text is available.",
                    }
                )
            errors = any(f["severity"] == "ERROR" for f in check["findings"])
            check["limited_checks_status"] = (
                "VIOLATION_DETECTED"
                if errors
                else "REVIEW_FLAGGED"
                if check["findings"]
                else "NO_LIMITED_FLAG"
            )
            reviewed = {
                **check,
                "claim_id": submitted["claim_id"],
                "text": submitted["text"],
                "text_sha256": _sha(submitted["text"].encode()),
                "prose_disposition": "FAIL" if errors else "ABSTAIN",
                "submitted_fields_preserved": submitted,
            }
            reviews.append(reviewed)
            flat_reviews.append({"term_uid": item["term_uid"], **reviewed})
        chosen = item["term_uid"] in selected
        reason = (
            "ADJUSTED_VALUE_ELIGIBLE"
            if chosen
            else "ADJUSTED_VALUE_UNAVAILABLE"
            if item["qval"] is None
            else "ADJUSTED_VALUE_ABOVE_CUTOFF"
            if item["qval"] > cfg.q_threshold
            else "OUTSIDE_DECLARED_TOP_K"
        )
        records.append(
            {
                "schema_version": "pathway-source-report/1",
                "evidence": item,
                "selected": chosen,
                "decision_status": "PASS" if chosen else "ABSTAIN",
                "decision_scope": "SOURCE_STATISTICAL_STATEMENT",
                "reason_code": reason,
                "source_statement": factual_statement(item, card["comparison"], cfg.q_threshold),
                "submitted_reviews": reviews,
            }
        )

    out = Path(cfg.outdir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(
            "Use a new or empty output directory; source reports never overwrite prior work"
        )
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        key: str(out / name)
        for key, name in {
            "evidence": "evidence.source.tsv",
            "audit_log": "audit_log.tsv",
            "report_jsonl": "report.jsonl",
            "report_md": "report.md",
            "report_html": "report.html",
            "claims_checked": "claims.checked.tsv",
        }.items()
    }
    _write_tsv(Path(artifacts["evidence"]), evidence, list(evidence[0]))
    audit = [
        {
            "claim_id": f"source:{r['evidence']['term_uid']}",
            "term_uid": r["evidence"]["term_uid"],
            "term_name": r["evidence"]["term_name"],
            "stat": r["evidence"]["stat"],
            "qval": r["evidence"]["qval"],
            "status": r["decision_status"],
            "reason_code": r["reason_code"],
            "decision_scope": r["decision_scope"],
            "evidence_sha256": r["evidence"]["evidence_sha256"],
            "source_statement": r["source_statement"],
        }
        for r in records
    ]
    _write_tsv(Path(artifacts["audit_log"]), audit, list(audit[0]))
    review_fields = [
        "term_uid",
        "claim_id",
        "text",
        "text_sha256",
        "prose_disposition",
        "limited_checks_status",
        "numeric_coverage",
        "findings",
        "automatic_prose_acceptance",
        "semantic_correctness",
    ]
    _write_tsv(Path(artifacts["claims_checked"]), flat_reviews, review_fields)
    Path(artifacts["report_jsonl"]).write_text(
        "".join(_json(r) + "\n" for r in records), encoding="utf-8"
    )
    lines = [
        "# Pathway source report",
        "",
        card["comparison"],
        "",
        f"Selected {len(selected)} of {len(evidence)} source records "
        f"under adjusted-value cutoff {cfg.q_threshold:g}.",
        "",
        "PASS applies to the source statistical statement. "
        "Submitted free prose requires human review.",
        "",
    ]
    for r in sorted(
        records,
        key=lambda r: (
            not r["selected"],
            r["evidence"]["qval"] if r["evidence"]["qval"] is not None else 2,
            r["evidence"]["term_uid"],
        ),
    ):
        lines.extend(
            [
                f"## {r['evidence']['term_name']}",
                "",
                r["source_statement"],
                "",
                f"Disposition: {r['decision_status']} · {r['reason_code']}",
                "",
            ]
        )
        for checked in r["submitted_reviews"]:
            lines.extend(
                [
                    "Submitted draft:",
                    "",
                    checked["text"],
                    "",
                    f"Limited check: {checked['limited_checks_status']}; human review required.",
                    "",
                ]
            )
            lines.extend(
                f"- {f['code']}: {f['quote']} — {f['message']}" for f in checked["findings"]
            )
            lines.append("")
    Path(artifacts["report_md"]).write_text("\n".join(lines), encoding="utf-8")
    Path(artifacts["report_html"]).write_text(_html(records, card["comparison"]), encoding="utf-8")
    inputs = {
        "evidence_table": {"path": str(ev_path.resolve()), "sha256": _sha(raw_ev)},
        "sample_card": {"path": str(card_path.resolve()), "sha256": _sha(raw_card)},
    }
    if raw_claims is not None:
        inputs["claims_file"] = {
            "path": str(Path(cfg.claims_file).resolve()),
            "sha256": _sha(raw_claims),
        }
    rid = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    meta = {
        "tool": "llm-pathway-curator",
        "workflow": "source",
        "ruleset": RULESET,
        "run_id": rid,
        "status": "COMPLETE",
        "config": asdict(cfg),
        "inputs": inputs,
        "candidate_count": len(evidence),
        "candidate_retained_count": len(evidence),
        "selected_count": len(selected),
        "submitted_text_count": len(flat_reviews),
        "limited_text_check_counts": dict(
            Counter(r["limited_checks_status"] for r in flat_reviews)
        ),
        "model_calls": 0,
        "context_proxy_used": False,
        "synthetic_stability_used": False,
        "semantic_accuracy_estimated": False,
        "biological_accuracy_estimated": False,
        "automatic_prose_acceptance": False,
        "runtime": {"python": platform.python_version()},
        "implementation_sha256": {
            Path(__file__).name: _sha(Path(__file__).read_bytes()),
            "grounding.py": _sha(Path(__file__).with_name("grounding.py").read_bytes()),
        },
        "artifacts": {
            key: {"path": path, "sha256": _sha(Path(path).read_bytes())}
            for key, path in artifacts.items()
        },
    }
    meta_path = out / "run_meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return ReviewResult(rid, str(out), artifacts, str(meta_path))
