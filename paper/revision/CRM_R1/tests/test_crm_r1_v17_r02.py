"""Safety and research-unit controls; synthetic metadata, no external requests."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "paper/revision/CRM_R1/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("crm_r02", SCRIPTS / "v17_r02.py")
r02 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r02)


def store_record(snapshot, acc, fields):
    kind = "SERIES" if acc.startswith("GSE") else "SAMPLE"
    body = [f"^{kind} = {acc}", f"!{kind.title()}_geo_accession = {acc}"]
    body.extend(f"!{k} = {v}" for k, v in fields)
    path = snapshot / f"{acc}.metadata.soft.txt"
    path.write_text("\n".join(body) + "\n")
    return {
        "accession": acc,
        "path": path.name,
        "sha256": r02.sha(path),
        "scope": "metadata_prefix_only_no_expression_rows",
        "source_url": "https://example.invalid",
        "retrieved_utc": "2026-10-04T00:00:00Z",
    }


@pytest.fixture
def metadata(tmp_path):
    snapshot = tmp_path / "metadata"
    snapshot.mkdir()
    records, census = [], {}
    # Identifiers are synthetic. Four donors have paired discovery treatments.
    for series, roles in (
        ("GSE52778", ("untreated", "Dex", "Alb", "Alb_Dex")),
        ("GSE34313", ("nodex", "dex4hr", "dex24hr")),
    ):
        census[series] = []
        for role in roles:
            count = 4 if series == "GSE52778" or role == "nodex" else 3
            for i in range(1, count + 1):
                acc = f"GSM{len(records) + 100}"
                census[series].append(acc)
                if series == "GSE52778":
                    title = f"N{i}_{role}"
                    treatment = {
                        "untreated": "Untreated",
                        "Dex": "Dexamethasone",
                        "Alb": "Albuterol",
                        "Alb_Dex": "Albuterol_Dexamethasone",
                    }[role]
                    chars = [
                        ("Sample_characteristics_ch1", f"cell line: N{i}"),
                        ("Sample_characteristics_ch1", f"treatment: {treatment}"),
                    ]
                else:
                    title = f"{role}_{i}"
                    treatment = {
                        "nodex": "none",
                        "dex4hr": "dexamethasone for 4 hr",
                        "dex24hr": "dexamethasone for 24 hr",
                    }[role]
                    chars = [("Sample_characteristics_ch1", f"treatment: {treatment}")]
                records.append(
                    store_record(
                        snapshot,
                        acc,
                        [
                            ("Sample_title", title),
                            ("Sample_organism_ch1", "Homo sapiens"),
                            ("Sample_series_id", series),
                            (
                                "Sample_platform_id",
                                "GPL11154" if series == "GSE52778" else "GPL6480",
                            ),
                            ("Sample_treatment_protocol_ch1", "synthetic protocol 18 h"),
                            *chars,
                        ],
                    )
                )
        records.append(
            store_record(snapshot, series, [("Series_sample_id", a) for a in census[series]])
        )
    manifest = snapshot / "METADATA_MANIFEST.json"
    manifest.write_text(json.dumps({"expression_rows_saved": 0, "files": records}))
    config = {"metadata_manifest_sha256": r02.sha(manifest), "metadata_sample_census": census}
    return snapshot, config


def test_replicate_suffix_does_not_create_donor_pairing(metadata):
    snapshot, config = metadata
    samples, pairs, _, result = r02.metadata_preflight(snapshot, config, r02.Inputs())
    assert len(samples) == 26 and len(pairs) == 4
    assert result["validation"]["public_24h_dex"] == 3
    assert result["validation"]["public_controls"] == 4
    assert not result["validation"]["donor_pairing_verified"]
    assert not result["donor_disjointness_between_studies_verified"]
    assert not result["gene_universe_and_probe_mapping_checked"]
    assert not result["biological_validation_performed"]


def test_metadata_tampering_fails_even_without_expression(metadata):
    snapshot, config = metadata
    file = next(snapshot.glob("GSM*.txt"))
    file.write_text(file.read_text() + "!Sample_description = tampered\n")
    with pytest.raises(ValueError, match="Metadata snapshot changed"):
        r02.metadata_preflight(snapshot, config, r02.Inputs())


def test_missing_sample_does_not_become_a_smaller_eligible_census(metadata):
    snapshot, config = metadata
    m = snapshot / "METADATA_MANIFEST.json"
    data = json.loads(m.read_text())
    data["files"] = data["files"][1:]
    m.write_text(json.dumps(data))
    config["metadata_manifest_sha256"] = r02.sha(m)
    with pytest.raises(ValueError, match="Sample metadata missing"):
        r02.metadata_preflight(snapshot, config, r02.Inputs())


def test_sample_title_and_characteristic_must_agree(metadata):
    snapshot, config = metadata
    m = snapshot / "METADATA_MANIFEST.json"
    data = json.loads(m.read_text())
    row = data["files"][0]
    file = snapshot / row["path"]
    file.write_text(file.read_text().replace("cell line: N1", "cell line: N9"))
    row["sha256"] = r02.sha(file)
    m.write_text(json.dumps(data))
    config["metadata_manifest_sha256"] = r02.sha(m)
    with pytest.raises(ValueError, match="Title/cell-line mismatch"):
        r02.metadata_preflight(snapshot, config, r02.Inputs())


@pytest.mark.parametrize("extra", ["!sample_table_begin\nG1\t12\n", "G1\t12\n"])
def test_expression_rows_are_rejected(tmp_path, extra):
    row = store_record(tmp_path, "GSM1", [])
    path = tmp_path / row["path"]
    path.write_text(path.read_text() + extra)
    with pytest.raises(ValueError, match="table|Non-metadata"):
        r02.parse_soft(path, "GSM1")


def test_source_lock_detects_changes_and_new_run_census(tmp_path):
    file = tmp_path / "source.txt"
    file.write_text("source")
    config = {"repository_sources": {"source.txt": r02.sha(file)}, "run_metadata_paths": []}
    assert len(r02.check_source_lock(tmp_path, config, r02.Inputs())) == 1
    file.write_text("changed")
    with pytest.raises(ValueError, match="Reviewed source changed"):
        r02.check_source_lock(tmp_path, config, r02.Inputs())
    file.write_text("source")
    run = tmp_path / "paper/source_data/PANCAN_TP53_v1/out_fig2/new/run_meta.json"
    run.parent.mkdir(parents=True)
    run.write_text("{}")
    with pytest.raises(ValueError, match="run census changed"):
        r02.check_source_lock(tmp_path, config, r02.Inputs())


def test_aborted_metadata_with_retained_proxy_artifacts_is_not_success(tmp_path):
    path = tmp_path / "run_meta.json"
    path.write_text(json.dumps({"status": "aborted", "step": "distill", "inputs": {}}))
    value = r02.proxy_hash("synthetic", ["condition"], "T1")
    pd.DataFrame(
        [
            {
                "claim_id": "C1",
                "term_uid": "T1",
                "status": "PASS",
                "context_ctx_id": "synthetic",
                "context_keys": "condition",
                "context_score_proxy_u01_norm": value,
            }
        ]
    ).to_csv(tmp_path / "audit_log.tsv", sep="\t", index=False)
    result = r02.inspect_run(path, tmp_path, r02.Inputs())
    assert result["proxy_hash_matches"] == 1
    assert result["final_pass"] == 1
    assert not result["execution_lineage_resolved"]
    assert "ABORTED" in result["interpretation"]


def test_path_traversal_and_snapshot_outside_data_root_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsafe source"):
        r02.local_path(tmp_path, "../source")
    root, external = tmp_path / "CRM_R1", tmp_path / "metadata"
    root.mkdir()
    external.mkdir()
    with pytest.raises(ValueError, match="under CRM_R1/input"):
        r02.run_r02(root, external, repo=tmp_path / "repo")
    assert not (root / "output").exists()
