"""Opt-in V16.1 package API; canonical JSON input, separate output ledgers.

Run with python -m llm_pathway_curator.contract_pipeline. This entry point does
not invoke the legacy selection/gating path or replace its method membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .contract_v161 import CONTRACT_VERSION, METHOD_ID
from .contract_v161.checks import prepare, summarize
from .contract_v161.models import AuditResult, ModelConfig, PromptPolicy
from .contract_v161.prompt import implementation_digest, strict_json
from .contract_v161.runtime import AttemptStore, OllamaTransport, now, run_review, write_new


@dataclass(frozen=True)
class ContractRunConfig:
    evidence_json: str
    claims_json: str
    outdir: str
    mode: str = "deterministic"
    model_config_json: str | None = None
    cache_root: str | None = None
    max_evidence_genes: int | None = None


@dataclass(frozen=True)
class ContractRunResult:
    method_id: str
    outdir: str
    artifacts: dict[str, str]
    summary: dict


def safe_output(path):
    path = Path(path).resolve()
    if any("lock_" in item or item.startswith("final_v") for item in path.parts):
        raise ValueError("Use a development directory outside historical frozen results")
    return path


def index_records(raw, key):
    records = strict_json(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("Input must be a nonempty JSON array")
    if any(not isinstance(x, dict) or not isinstance(x.get(key), str) for x in records):
        raise ValueError(f"Every input must have a string {key}")
    indexed = {x[key]: x for x in records}
    if len(indexed) != len(records):
        raise ValueError(f"Duplicate {key}")
    return records, indexed


def run_contract_pipeline(cfg: ContractRunConfig, *, transport=None) -> ContractRunResult:
    if cfg.mode not in {"deterministic", "ollama"}:
        raise ValueError("Explicit mode must be deterministic or ollama")
    paths = {"evidence": Path(cfg.evidence_json), "claims": Path(cfg.claims_json)}
    if cfg.model_config_json:
        paths["model_config"] = Path(cfg.model_config_json)
    raw = {k: path.read_bytes() for k, path in paths.items()}
    evidence, by_evidence = index_records(raw["evidence"], "evidence_id")
    claims, _ = index_records(raw["claims"], "claim_id")
    if any(not isinstance(x.get("evidence_id"), str) for x in claims):
        raise ValueError("Every claim must identify its canonical evidence")
    by_claim = {x["evidence_id"]: x for x in claims}
    if len(by_claim) != len(claims) or set(by_claim) != set(by_evidence):
        raise ValueError("Require one claim per evidence record with identical evidence IDs")
    policy = PromptPolicy(max_evidence_genes=cfg.max_evidence_genes)
    model = None
    if cfg.mode == "ollama":
        if "model_config" not in raw:
            raise ValueError("Ollama mode requires a pinned model configuration")
        model = ModelConfig.model_validate(strict_json(raw["model_config"]))
        transport = transport if transport is not None else OllamaTransport(model)
    elif cfg.model_config_json or cfg.cache_root or transport is not None:
        raise ValueError("Model, cache and transport apply only to ollama mode")
    out = safe_output(cfg.outdir)
    if out.exists():
        raise ValueError("Use a new output directory; reuse cache_root to resume")
    cache = safe_output(cfg.cache_root or out / "attempts")
    if any(p.resolve().is_relative_to(out) for p in paths.values()):
        raise ValueError("Output must not contain input files")
    out.mkdir(parents=True)
    write_new(
        out / "STARTED.json",
        {
            "method_id": METHOD_ID,
            "contract_version": CONTRACT_VERSION,
            "mode": cfg.mode,
            "started_utc": now(),
            "prompt_policy": policy.model_dump(),
            "implementation_sha256": implementation_digest(),
            "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {
                k: {"path": str(p.resolve()), "sha256": hashlib.sha256(raw[k]).hexdigest()}
                for k, p in paths.items()
            },
            "legacy_memberships_modified": False,
            "performance_evaluation": False,
        },
    )
    for name, data in raw.items():
        (out / f"input_{name}.private.json").write_bytes(data)
    store = AttemptStore(cache) if cfg.mode == "ollama" else None
    results = []
    with (out / "audit_results.private.jsonl").open("x", encoding="utf-8") as handle:
        for i, item in enumerate(evidence, 1):
            claim = by_claim[item["evidence_id"]]
            if cfg.mode == "ollama":
                result = run_review(item, claim, model, store, transport, policy=policy)
            else:
                result, _, _ = prepare(item, claim)
            result = AuditResult.model_validate(result).model_dump()
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(
                f"[RECORD] {i}/{len(evidence)} {claim['claim_id']} "
                f"{result['execution_status']} {result['semantic_status']}",
                flush=True,
            )
    write_new(
        out / "candidate_ledger.private.json",
        [
            {
                "evidence": r["raw_evidence_preserved"],
                "statistical_candidate_retained": r["statistical_candidate_retained"],
                "statistical_fdr_0_05": r["statistical_fdr_0_05"],
                "canonical_status": r["canonical_status"],
            }
            for r in results
        ],
    )
    write_new(
        out / "standard_statements.private.json",
        [
            {
                "evidence_id": r["raw_evidence_preserved"]["evidence_id"],
                "text": r["factual_enrichment_statement"],
                "canonical_status": r["canonical_status"],
                "scope": "deterministic_description_of_supplied_record_not_biological_truth",
            }
            for r in results
        ],
    )
    write_new(
        out / "interpretation_ledger.private.json",
        [
            {
                k: r[k]
                for k in (
                    "submitted_claim_preserved",
                    "execution_status",
                    "semantic_status",
                    "interpretation_status",
                    "interpretation_eligible",
                    "technical_error",
                    "automatic_free_text_publication_allowed",
                    "issue_codes",
                    "review",
                )
            }
            for r in results
        ],
    )
    summary = summarize(results) | {"finished_utc": now()}
    write_new(out / "summary.json", summary)
    artifacts = {p.name: str(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_new(
        out / "OUTPUT_MANIFEST.json",
        {
            "outputs": {
                k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in artifacts.items()
            },
            "finished_utc": now(),
        },
    )
    return ContractRunResult(METHOD_ID, str(out), artifacts, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("deterministic", "ollama"), default="deterministic")
    parser.add_argument("--model-config")
    parser.add_argument("--cache-root")
    parser.add_argument("--max-evidence-genes", type=int)
    args = parser.parse_args()
    result = run_contract_pipeline(
        ContractRunConfig(
            args.evidence,
            args.claims,
            args.out,
            args.mode,
            args.model_config,
            args.cache_root,
            args.max_evidence_genes,
        )
    )
    print(json.dumps(result.summary, indent=2))
    return 2 if result.summary["technical_or_input_errors"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"[STOP] {error}", file=sys.stderr)
        raise SystemExit(2) from error
