"""Source/caching/failure/coverage tests; no actual model or biological results."""

import base64
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "paper/revision/CRM_R1/scripts"))
SPEC = importlib.util.spec_from_file_location(
    "r07_tested", REPO / "paper/revision/CRM_R1/scripts/v17_r07.py"
)
r07 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r07)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def root(tmp_path):
    data = tmp_path / "CRM_R1"
    (data / "input").mkdir(parents=True)
    (data / "output").mkdir()
    stats = [
        {
            "cohort_id": "HNSC",
            "split_id": "S001",
            "term_id": f"T{i:02}",
            "term_name": f"T{i:02}",
            "stat": "1.5" if i % 2 else "-2.0",
            "qval": str((i + 1) / 100),
            "direction": "up" if i % 2 else "down",
            "evidence_genes": f"G{i:02}",
            "source": "synthetic",
            "n_discovery_mutant": "10",
            "n_discovery_wild_type": "12",
        }
        for i in range(50)
    ]
    stats_path = data / r07.export_adapter.__globals__["STATS"]
    tsv(stats_path, stats)
    dump(
        stats_path.parent / "discovery_statistics_manifest.json",
        {
            "inputs": {},
            "outputs": {"statistics": {"path": str(stats_path), "sha256": r07.sha(stats_path)}},
            "rows": 50,
            "cohort_split_pairs": 1,
        },
    )
    genes_path = data / r07.export_adapter.__globals__["HALLMARK"]
    tsv(genes_path, [{"term_id": f"T{i:02}", "gene_id": f"G{i:02}"} for i in range(50)])
    dump(
        genes_path.parent / "hallmark_manifest.json",
        {
            "inputs": {},
            "outputs": {"gene_sets": {"path": str(genes_path), "sha256": r07.sha(genes_path)}},
            "term_gene_pairs": 50,
        },
    )
    tsv(
        genes_path.parent / "hallmark_export_metadata.tsv",
        [
            {"field": "gene_identifier", "value": "HGNC gene symbol (msigdbr gene_symbol)"},
            {"field": "msigdbr_version", "value": "synthetic"},
        ],
    )
    job = data / r07.export_adapter.__globals__["JOB_ROOT"] / "HNSC/S001"
    tsv(job / "discovery_evidence.tsv", stats)
    dump(
        job / "sample_card.json",
        {
            "condition": "HNSC",
            "comparison": "TP53_mut_vs_TP53_wt",
            "perturbation": "genotype",
            "tissue": "tumor",
        },
    )
    r07.prepare_design(data)
    return data


@pytest.fixture
def source(root):
    return r07.load_design(root)[3][0]


def response(request, text, done_reason="stop", wrong_model=False):
    config = request["envelope"]["model_config"]
    identity = {
        "model": config["model"],
        "digest": config["model_digest"],
        "version": config["server_version"],
    }
    raw = json.dumps(
        {
            "model": "wrong" if wrong_model else config["model"],
            "done": True,
            "done_reason": done_reason,
            "response": text,
        }
    ).encode()
    return r07.WireResponse(raw, {"identity_before": identity, "identity_after": identity})


def test_offline_preparation_reuses_exact_freeze_without_model(root, monkeypatch):
    monkeypatch.setattr(
        r07, "OllamaTransport", lambda *a: pytest.fail("Offline network/model access")
    )
    folder = root / r07.DESIGN_REL
    original = {str(p): r07.sha(p) for p in folder.rglob("*") if p.is_file()}
    assert r07.prepare_design(root)["status"] == "DESIGN_ALREADY_FROZEN_UNCHANGED"
    assert original == {str(p): r07.sha(p) for p in folder.rglob("*") if p.is_file()}


def test_modified_source_and_design_block_before_generation(root):
    source = r07.load_design(root)[3][0]
    Path(source["source"]["artifact"]).write_text("changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        r07.run_live(root, transport=lambda request: pytest.fail("Call despite changed source"))


def test_prose_preserves_whitespace_and_reuses_first_success(tmp_path, source):
    policy = r07.strict_json(r07.POLICY.read_bytes())
    request = r07.make_request(source, policy, r07.model_config(policy), "a" * 64)
    store = r07.ProseStore(tmp_path)
    text = " \nNES=-2.0; BH q=0.01.\n "
    record, new = store.run(request, lambda req: response(req, text), True)
    assert record["text"] == text and new
    reused, new = store.run(request, lambda req: pytest.fail("Second generation"), True)
    assert reused == record and not new


@pytest.mark.parametrize(
    "kind", ["truncated", "wrong_model", "empty", "malformed", "wrong_identity"]
)
def test_failed_first_outcome_is_not_retried(tmp_path, source, kind):
    policy = r07.strict_json(r07.POLICY.read_bytes())
    request = r07.make_request(source, policy, r07.model_config(policy), "a" * 64)

    def wire(req):
        result = response(
            req,
            "" if kind == "empty" else "NES=-2; BH q=0.01",
            "length" if kind == "truncated" else "stop",
            kind == "wrong_model",
        )
        if kind == "malformed":
            return r07.WireResponse(b"bad json", result.observations)
        if kind == "wrong_identity":
            result.observations["identity_after"] = {}
        return result

    store = r07.ProseStore(tmp_path)
    first, new = store.run(request, wire, True)
    assert first["status"] == "TECHNICAL_ERROR" and first["text"] is None and new
    second, new = store.run(request, lambda req: pytest.fail("Retry of failure"), True)
    assert second == first and not new


def test_interruption_is_preserved_without_retry(tmp_path, source):
    policy = r07.strict_json(r07.POLICY.read_bytes())
    request = r07.make_request(source, policy, r07.model_config(policy), "a" * 64)
    store = r07.ProseStore(tmp_path)
    with store.locked(request) as folder:
        r07.write_new(folder / "STARTED.json", {"request_key": request["key"]})
    result, new = store.run(request, lambda req: pytest.fail("Retry of interruption"), True)
    assert result["status"] == "INTERRUPTED" and not new


def test_tampered_raw_or_seal_is_rejected(tmp_path, source):
    policy = r07.strict_json(r07.POLICY.read_bytes())
    request = r07.make_request(source, policy, r07.model_config(policy), "a" * 64)
    store = r07.ProseStore(tmp_path)
    store.run(request, lambda req: response(req, "NES=-2; BH q=0.01"), True)
    path = tmp_path / request["key"] / "FIRST_RESULT.json"
    value = json.loads(path.read_text())
    value["raw_response_base64"] = base64.b64encode(b"modified").decode()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="seal differs"):
        store.run(request, lambda req: pytest.fail("Regeneration"), True)


def test_limited_numeric_gate_covers_boundaries_missing_syntax_and_rounding(source):
    assert r07.numeric_gate("NES=-2.0; BH q=0.01", source)["retained"]
    assert r07.numeric_gate("NES=-2.0", source)["status"] == "EXPLICIT_NUMERIC_COVERAGE_INCOMPLETE"
    assert not r07.numeric_gate("NES=-2.01; BH q=0.01", source)["retained"]
    assert not r07.numeric_gate(None, source)["retained"]
    # This retained text deliberately contains an unexamined direction error.
    result = r07.numeric_gate("NES=-2.0; BH q=0.01; enrichment toward TP53-mutant samples.", source)
    assert result["retained"] and result["semantic_safety_assessed"] is False


def test_budget_resume_keeps_all_50_denominators_and_previous_failures(root):
    calls = []

    def wire(request):
        supplied = json.loads(request["envelope"]["body"]["prompt"].split("EVIDENCE_RECORD:\n")[1])
        calls.append(supplied["evidence_id"])
        return response(request, f"NES={supplied['nes']}; BH q={supplied['q_value']}")

    first = r07.run_live(root, calls=2, transport=wire)
    assert (
        first["new_local_model_requests"] == 2
        and first["generation_status_counts"]["NOT_RUN_BUDGET"] == 48
    )
    assert first["candidate_retained_count"] == 50 and not first["model_backend_authenticated"]
    second = r07.run_live(root, calls=1, transport=wire)
    assert (
        second["new_local_model_requests"] == 1
        and second["generation_status_counts"]["SUCCEEDED"] == 3
    )
    assert len(set(calls)) == len(calls) == 3
    rows = json.loads((Path(second["outdir"]) / "comparisons.private.json").read_text())
    assert (
        sum(r["numeric_gate_retained"] for r in rows)
        == sum(r["matched_q_selected"] for r in rows)
        == 3
    )
    packet = list(
        csv.DictReader(
            (Path(second["outdir"]) / "source_fact_check.optional.private.tsv").open(),
            delimiter="\t",
        )
    )
    assert len(packet) == 3 and all(not r["source_faithfulness"] for r in packet)
    assert not any("gate" in key or "selected" in key for key in packet[0])


def test_zero_coverage_is_visible_and_template_does_not_get_accuracy(root):
    result = r07.run_live(root, calls=0, transport=lambda request: pytest.fail("Zero-budget call"))
    table = list(
        csv.DictReader((Path(result["outdir"]) / "method_coverage.tsv").open(), delimiter="\t")
    )
    assert table[0]["coverage_all_candidates"] == "0.0"
    assert table[3]["coverage_all_candidates"] == "1.0"
    assert all(r["source_faithfulness_accuracy"].startswith("NOT_ESTIMATED") for r in table)


def test_injected_and_real_request_identities_are_distinct(source):
    policy = r07.strict_json(r07.POLICY.read_bytes())
    model = r07.model_config(policy)
    assert (
        r07.make_request(source, policy, model, "a" * 64)["key"]
        != r07.make_request(source, policy, model, "a" * 64, software_test=True)["key"]
    )


def test_annotations_are_bound_to_exact_text_and_unknowns_remain_unknown(root):
    run = r07.run_live(root, calls=2, transport=lambda req: response(req, "NES=-2.0; BH q=0.01"))
    packet_path = Path(run["outdir"]) / "source_fact_check.optional.private.tsv"
    packet = list(csv.DictReader(packet_path.open(), delimiter="\t"))
    packet[0].update(source_faithfulness="CHECKED_FACTS_MATCH", annotator_id="synthetic_author")
    annotations = root / "input/source_facts.tsv"
    tsv(annotations, packet)
    result = r07.evaluate_archive(root, run["results_archive"], annotations)
    assert result["annotation_labels"] == {"CHECKED_FACTS_MATCH": 1}
    table = list(
        csv.DictReader(
            (Path(result["outdir"]) / "source_fact_comparison.tsv").open(), delimiter="\t"
        )
    )
    assert (
        table[0]["unknown_offered"] == "1" and table[0]["source_fact_error_fraction_upper"] == "0.5"
    )
    assert result["status"] == "SOFTWARE_TRANSPORT_TEST_ONLY"
    packet[0]["text_sha256"] = hashlib.sha256(b"different text").hexdigest()
    tsv(annotations, packet)
    with pytest.raises(ValueError, match="different text"):
        r07.evaluate_archive(root, run["results_archive"], annotations)
