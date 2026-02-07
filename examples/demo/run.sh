#!/usr/bin/env bash
# LLM-PathwayCurator/examples/demo/run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEMO="${ROOT}/examples/demo"
OUTDIR="${1:-${ROOT}/out/demo}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${OUTDIR}"

EVID="${DEMO}/evidence_table.tsv"
FGSEA_IN="${DEMO}/inputs/fgsea_results.tsv"

# 1) Input -> EvidenceTable (optional; demo includes it)
if [[ ! -s "${EVID}" ]]; then
  if [[ ! -s "${FGSEA_IN}" ]]; then
    echo "[demo] missing: ${EVID} and ${FGSEA_IN}" >&2
    echo "[demo] provide EvidenceTable OR provide inputs/fgsea_results.tsv" >&2
    exit 2
  fi
  python "${DEMO}/make_evidence_table.py" --fgsea "${FGSEA_IN}" --out "${EVID}"
fi

# 2) EvidenceTable -> distill/modules/claims/audit/report (core)
python - <<'PY'
from pathlib import Path
from llm_pathway_curator.pipeline import RunConfig, run_pipeline

root = Path(".").resolve()
demo = root / "examples/demo"
outdir = root / "out/demo"

cfg = RunConfig(
    evidence_table=str(demo / "evidence_table.tsv"),
    sample_card=str(demo / "sample_card.json"),
    outdir=str(outdir),
    force=True,
    seed=42,
    run_meta_name="run_meta.json",
    tau=0.80,
    k_claims=20,
    stress_evidence_dropout_p=0.0,
    stress_evidence_dropout_min_keep=1,
    stress_contradictory_p=0.0,
    stress_contradictory_max_extra=0,
)
run_pipeline(cfg)

rp = outdir / "report.jsonl"
assert rp.exists() and rp.stat().st_size > 0, "report.jsonl missing/empty"
print("[OK] demo pipeline complete:", outdir)
print("[OK] report:", rp)
PY
