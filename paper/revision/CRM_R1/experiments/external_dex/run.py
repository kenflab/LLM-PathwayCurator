#!/usr/bin/env python3
"""Freeze and run a public-data dex case. No model or expert-rating calls."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGES = ("airway", "SummarizedExperiment", "edgeR", "limma", "fgsea", "jsonlite")
TERMS = tuple("HALLMARK_" + x for x in (
    "ADIPOGENESIS ALLOGRAFT_REJECTION ANDROGEN_RESPONSE ANGIOGENESIS APICAL_JUNCTION "
    "APICAL_SURFACE APOPTOSIS BILE_ACID_METABOLISM CHOLESTEROL_HOMEOSTASIS COAGULATION "
    "COMPLEMENT DNA_REPAIR E2F_TARGETS EPITHELIAL_MESENCHYMAL_TRANSITION "
    "ESTROGEN_RESPONSE_EARLY ESTROGEN_RESPONSE_LATE FATTY_ACID_METABOLISM "
    "G2M_CHECKPOINT GLYCOLYSIS HEDGEHOG_SIGNALING HEME_METABOLISM HYPOXIA "
    "IL2_STAT5_SIGNALING IL6_JAK_STAT3_SIGNALING INFLAMMATORY_RESPONSE "
    "INTERFERON_ALPHA_RESPONSE INTERFERON_GAMMA_RESPONSE KRAS_SIGNALING_DN "
    "KRAS_SIGNALING_UP MITOTIC_SPINDLE MTORC1_SIGNALING MYC_TARGETS_V1 MYC_TARGETS_V2 "
    "MYOGENESIS NOTCH_SIGNALING OXIDATIVE_PHOSPHORYLATION P53_PATHWAY "
    "PANCREAS_BETA_CELLS PEROXISOME PI3K_AKT_MTOR_SIGNALING PROTEIN_SECRETION "
    "REACTIVE_OXYGEN_SPECIES_PATHWAY SPERMATOGENESIS TGF_BETA_SIGNALING "
    "TNFA_SIGNALING_VIA_NFKB UNFOLDED_PROTEIN_RESPONSE UV_RESPONSE_DN UV_RESPONSE_UP "
    "WNT_BETA_CATENIN_SIGNALING XENOBIOTIC_METABOLISM"
).split())
DOWNLOADS = {
    "GSE34313_RAW.tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE34nnn/GSE34313/suppl/GSE34313_RAW.tar",
    "GSE34313_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE34nnn/GSE34313/matrix/GSE34313_series_matrix.txt.gz",
    "GPL6480.soft.txt": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL6480&targ=self&form=text&view=full",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_tsv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def data_path(root, relative):
    result = root / relative
    require(result.resolve().is_relative_to(root), f"Path escapes data root: {relative}")
    require(not any(p.is_symlink() for p in (result, *result.parents)), f"Symlink: {relative}")
    return result


def load_memberships(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        if fields == {"term_id", "gene_id"}:
            term_col, gene_col = "term_id", "gene_id"
        else:
            require({"gs_name", "gene_symbol"}.issubset(fields), "Hallmark columns must be term_id/gene_id (symbols) or gs_name/gene_symbol")
            term_col, gene_col = "gs_name", "gene_symbol"
        pairs = set()
        for row in reader:
            term, gene = row[term_col].strip(), row[gene_col].strip()
            require(term in TERMS and gene and not gene.isdigit(), "Wrong Hallmark census or non-symbol gene identifier")
            require(not any(x.isspace() for x in gene) and not any(x in gene for x in (";", "/", "|")), "Ambiguous Hallmark symbol")
            require((term, gene) not in pairs, "Duplicate Hallmark membership")
            pairs.add((term, gene))
    require({x[0] for x in pairs} == set(TERMS), "All 50 canonical Hallmark terms are required")
    return [{"term_id": t, "gene_symbol": g} for t, g in sorted(pairs)]


def code_hashes():
    return {name: sha(HERE / name) for name in ("run.py", "analysis.R", "protocol.json")}


def freeze(root, hallmark=None):
    lock_dir = data_path(root, "output/revision_v17/external_dex_design_v1")
    lock_path = lock_dir / "DESIGN_LOCK.json"
    if lock_dir.exists():
        require(lock_path.is_file(), "Incomplete design directory preserved; inspect before proceeding")
        lock = read_json(lock_path)
        require(lock["code_sha256"] == code_hashes(), "Frozen code/protocol changed; preserve the lock and document a technical amendment")
        for name, digest in lock["frozen_file_sha256"].items():
            require(sha(lock_dir / name) == digest, f"Frozen file changed: {name}")
        source = data_path(root, lock["hallmark_source_relative_path"])
        require(sha(source) == lock["hallmark_source_sha256"], "Original Hallmark snapshot changed")
        if hallmark is not None:
            require(hallmark.resolve() == source.resolve(), "A different Hallmark source cannot replace the frozen snapshot")
        return lock_dir, lock
    source = hallmark or data_path(root, "output/priority2b/hallmark_lock_v14_1_2/hallmark_gene_sets.tsv")
    require(source.resolve().is_relative_to(root), "Hallmark source must be in the existing CRM_R1 data root")
    require(source.is_file(), f"Missing existing Hallmark snapshot: {source}")
    rows = load_memberships(source)
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".external_dex_design_", dir=lock_dir.parent))
    try:
        shutil.copyfile(HERE / "protocol.json", staging / "protocol.json")
        write_tsv(staging / "hallmark_gene_symbols.tsv", rows, ["term_id", "gene_symbol"])
        write_tsv(staging / "candidate_census.tsv", [{"term_id": t} for t in TERMS], ["term_id"])
        lock = {
            "schema": "CRM_R1_EXTERNAL_DEX_DESIGN_LOCK_v1", "created_utc": stamp(),
            "code_sha256": code_hashes(), "candidate_count": 50,
            "hallmark_source_relative_path": str(source.resolve().relative_to(root)),
            "hallmark_source_sha256": sha(source),
            "frozen_file_sha256": {p.name: sha(p) for p in sorted(staging.iterdir())},
            "model_calls": 0, "expression_outcomes_loaded_by_freeze": False,
            "published_result_exposure": read_json(HERE / "protocol.json")["outcome_exposure"],
        }
        write_json(staging / "DESIGN_LOCK.json", lock)
        require(not lock_dir.exists(), "Concurrent freeze detected")
        staging.rename(lock_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return lock_dir, lock


def runtime_identity():
    executable = shutil.which("Rscript")
    if not executable:
        return {"ready": False, "missing": ["Rscript"]}
    code = 'p<-c(' + ",".join(json.dumps(p) for p in PACKAGES) + '); cat("R_VERSION\\t",R.version.string,"\\n",sep=""); for(x in p) cat(x,"\\t",if(requireNamespace(x,quietly=TRUE)) as.character(packageVersion(x)) else "MISSING","\\n",sep=""); if(requireNamespace("airway",quietly=TRUE))cat("AIRWAY_DATA_DIR\\t",system.file("data",package="airway"),"\\n",sep="")'
    process = subprocess.run([executable, "--vanilla", "-e", code], capture_output=True, text=True, check=True)
    records = dict(line.split("\t", 1) for line in process.stdout.splitlines() if "\t" in line)
    missing = [p for p in PACKAGES if records.get(p) == "MISSING"]
    identity = {"R": records["R_VERSION"], "packages": {p: records.get(p) for p in PACKAGES}, "ready": not missing, "missing": missing}
    if not missing:
        directory = Path(records["AIRWAY_DATA_DIR"])
        require(directory.is_dir(), "airway package has no data directory")
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".rda", ".rdata", ".rdb", ".rdx"})
        require(files, "No airway data payload found for hashing")
        identity["airway_data_sha256"] = {p.name: sha(p) for p in files}
    return identity


def install_missing(identity):
    require(shutil.which("Rscript"), "Install R first; Rscript is unavailable")
    missing = [p for p in identity["missing"] if p in PACKAGES]
    if not missing:
        return
    code = 'if(!requireNamespace("BiocManager",quietly=TRUE))install.packages("BiocManager",repos="https://cloud.r-project.org"); BiocManager::install(c(' + ",".join(json.dumps(p) for p in missing) + '),ask=FALSE,update=FALSE)'
    subprocess.run([shutil.which("Rscript"), "--vanilla", "-e", code], check=True)


def download(path, url, design_sha):
    receipt_path = path.with_name(path.name + ".receipt.json")
    if path.exists():
        require(receipt_path.is_file(), f"Unrecorded local download preserved: {path}")
        receipt = read_json(receipt_path)
        require(receipt["url"] == url and receipt["design_sha256"] == design_sha and receipt["sha256"] == sha(path), f"Downloaded input/receipt changed: {path.name}")
        return receipt
    require(not receipt_path.exists(), "Download receipt exists without input; preserved")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial_" + stamp())
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "CRM-R1-public-data-analysis/1.0"})
        total = 0
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                require(total < 150 * 1024 * 1024, "Unexpected download larger than 150 MiB")
                handle.write(block)
        require(total > 0, "Empty public-data download")
        receipt = {"url": url, "created_utc": stamp(), "design_sha256": design_sha, "sha256": sha(temporary), "bytes": total}
        require(not path.exists(), "Concurrent download detected")
        temporary.rename(path)
        write_json(receipt_path, receipt)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def input_manifest(root, lock, *, fetch=False):
    directory = data_path(root, "input/external_dex_r08_v1")
    records = {}
    for name, url in DOWNLOADS.items():
        path = directory / name
        if fetch:
            records[name] = download(path, url, object_sha(lock))
        else:
            require(path.is_file() and path.with_name(name + ".receipt.json").is_file(), "Missing public input; run with --fetch")
            records[name] = read_json(path.with_name(name + ".receipt.json"))
            require(records[name]["url"] == url and records[name]["sha256"] == sha(path) and records[name]["design_sha256"] == object_sha(lock), f"Public input changed: {name}")
    manifest = {"schema": "CRM_R1_EXTERNAL_DEX_INPUTS_v1", "design_sha256": object_sha(lock), "files": records}
    seal = directory / "INPUT_MANIFEST.json"
    if seal.exists():
        require(read_json(seal) == manifest, "Input manifest changed")
    else:
        write_json(seal, manifest)
    return directory, manifest


def check_sample_metadata(matrix_path, protocol):
    metadata = {}
    with gzip.open(matrix_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith(("!Sample_geo_accession", "!Sample_title", "!Sample_platform_id")):
                row = next(csv.reader([line], delimiter="\t"))
                metadata[row[0]] = row[1:]
    ids = metadata.get("!Sample_geo_accession", [])
    expected = {s["geo_accession"]: s["title"] for s in protocol["validation"]["samples"]}
    require(len(ids) == 10 and set(ids) == set(expected) and len(set(ids)) == 10, "Public validation sample census changed")
    titles = metadata.get("!Sample_title", [])
    require(len(titles) == 10 and dict(zip(ids, titles)) == expected, "Validation sample titles changed")
    require(metadata.get("!Sample_platform_id") == ["GPL6480"] * 10, "Unexpected validation platform")


def platform_mapping(path):
    active, lines = False, []
    platform_ok = False
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.startswith("^PLATFORM") and "GPL6480" in line:
                platform_ok = True
            if line.startswith("!platform_table_begin"):
                active = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if active:
                lines.append(line)
    require(platform_ok and lines, "GPL6480 SOFT annotation unavailable (HTML/sign-in pages are not annotation)")
    reader = csv.DictReader(lines, delimiter="\t")
    require({"ID", "GENE_SYMBOL", "CONTROL_TYPE"}.issubset(reader.fieldnames or []), "GPL6480 required mapping fields missing")
    rows, seen = [], set()
    for row in reader:
        require(row["ID"] and row["ID"] not in seen, "Duplicate/blank GPL6480 probe ID")
        seen.add(row["ID"])
        rows.append({"probe_id": row["ID"], "gene_symbol": row["GENE_SYMBOL"].strip(), "control_type": row["CONTROL_TYPE"]})
    return rows


def extract_arrays(archive, target, protocol):
    expected = {s["geo_accession"] for s in protocol["validation"]["samples"]}
    selected = {}
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            name = Path(member.name)
            require(not name.is_absolute() and ".." not in name.parts and not member.issym() and not member.islnk(), "Unsafe public archive member")
            if member.isdir():
                continue
            require(member.isfile(), "Unexpected archive entry type")
            sample = name.name.split("_", 1)[0]
            require(sample in expected and name.name.endswith((".txt", ".txt.gz")), "Unexpected raw-array file/sample")
            require(sample not in selected and member.size < 50 * 1024 * 1024, "Duplicate/oversized raw-array sample")
            raw = handle.extractfile(member)
            require(raw is not None, "Cannot read raw-array member")
            with raw:
                data = raw.read()
            if name.name.endswith(".gz"):
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                    data = stream.read(50 * 1024 * 1024 + 1)
                require(len(data) <= 50 * 1024 * 1024, "Oversized decompressed array")
            require(b"gMedianSignal" in data and b"gBGMedianSignal" in data and b"gIsWellAboveBG" in data, "Required raw Agilent signal/background/detection fields missing")
            destination = target / (sample + ".txt")
            with destination.open("xb") as out:
                out.write(data)
            selected[sample] = str(destination)
    require(set(selected) == expected, "Raw archive does not contain exactly the ten public arrays")
    return selected


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def load_enrichment(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    require(len(records) == 50 and {r["term_id"] for r in records} == set(TERMS), "Enrichment table must retain the complete 50-term census")
    for row in records:
        p, q, nes = number(row["pval"]), number(row["q"]), number(row["NES"])
        if row["status"] == "ESTIMABLE":
            require(p is not None and 0 <= p <= 1 and q is not None and 0 <= q <= 1 and nes is not None, "Invalid estimable enrichment result")
        else:
            require(row["status"] in {"OUTSIDE_SIZE_LIMITS", "NUMERICAL_FAILURE", "INCOMPLETE_FIXED_UNIVERSE"}, "Unexpected enrichment status")
    return {r["term_id"]: r for r in records}


def direction(row):
    n = number(row["NES"])
    return 1 if n is not None and n > 0 else -1 if n is not None and n < 0 else 0


def same_direction(left, right):
    return left["status"] == right["status"] == "ESTIMABLE" and direction(left) != 0 and direction(left) == direction(right)


def compare(discovery, validation, folds):
    require(len(folds) == 4, "Exactly four donor folds are required")
    hits = {t: sum(same_direction(discovery[t], f[t]) and number(f[t]["q"]) <= .05 for f in folds) for t in TERMS}
    q_pool = [t for t in TERMS if discovery[t]["status"] == "ESTIMABLE" and number(discovery[t]["q"]) <= .05]
    stable = [t for t in q_pool if hits[t] >= 3]
    ranked = sorted((t for t in TERMS if discovery[t]["status"] == "ESTIMABLE"), key=lambda t: (number(discovery[t]["q"]), t))
    selections = {"ALL_50_CANDIDATES": list(TERMS), "Q_VALUE_0_05": q_pool, "Q_PLUS_DONOR_LOO_3_OF_4": stable, "MATCHED_Q_K": ranked[:len(stable)]}
    outcomes = {t: same_direction(discovery[t], validation[t]) and number(validation[t]["q"]) <= .05 for t in TERMS}
    records = []
    for method, ids in selections.items():
        n = len(ids)
        records.append({"method": method, "selected_count": n, "coverage": n / 50, "validation_estimable_count": sum(validation[t]["status"] == "ESTIMABLE" for t in ids), "same_direction_count": sum(same_direction(discovery[t], validation[t]) for t in ids), "replicated_count": sum(outcomes[t] for t in ids), "replication_rate": sum(outcomes[t] for t in ids) / n if n else None, "selected_ids": ";".join(ids)})
    by_method = {r["method"]: r for r in records}
    a, b = (by_method[m]["replication_rate"] for m in ("Q_PLUS_DONOR_LOO_3_OF_4", "MATCHED_Q_K"))
    return records, selections, hits, outcomes, (a - b if a is not None and b is not None else None)


def summarize(out, protocol, job):
    discovery = load_enrichment(out / "discovery.tsv")
    validation = load_enrichment(out / "validation_24h.tsv")
    secondary = load_enrichment(out / "validation_4h_secondary.tsv")
    donors = sorted({s["donor"] for s in protocol["discovery"]["samples"]})
    folds = [load_enrichment(out / ("donor_loo_" + d + ".tsv")) for d in donors]
    comparisons, selections, hits, outcomes, delta = compare(discovery, validation, folds)
    secondary_comparisons, _, _, secondary_outcomes, _ = compare(discovery, secondary, folds)
    for row in comparisons:
        row["contrast"] = "24h_PRIMARY"
    for row in secondary_comparisons:
        row["contrast"] = "4h_SECONDARY_SHARED_CONTROLS"
    fields = ["contrast", "method", "selected_count", "coverage", "validation_estimable_count", "same_direction_count", "replicated_count", "replication_rate", "selected_ids"]
    write_tsv(out / "METHOD_COMPARISON.tsv", comparisons + secondary_comparisons, fields)
    ledger, templates = [], []
    source_sha = sha(out / "discovery.tsv")
    for t in TERMS:
        d, v = discovery[t], validation[t]
        ledger.append({"term_id": t, "discovery_status": d["status"], "discovery_NES": d["NES"], "discovery_q": d["q"], "donor_loo_hits_of_4": hits[t], "validation_status": v["status"], "validation_NES": v["NES"], "validation_q": v["q"], "replicated_24h": outcomes[t], "replicated_4h_secondary": secondary_outcomes[t], **{m: t in ids for m, ids in selections.items()}})
        text = (f"In GSE52778, dexamethasone versus control at 18 h has NES={d['NES']} and BH q={d['q']} for {t}. This is a gene-set enrichment result in four paired donor-derived cell lines; it does not establish pathway activation or clinical effects." if d["status"] == "ESTIMABLE" else f"For {t}, enrichment is non-estimable under the fixed protocol ({d['status']}).")
        templates.append({"term_id": t, "statement": text, "baseline": "SOURCE_TEMPLATE", "source_file": "discovery.tsv", "source_sha256": source_sha, "semantic_or_biological_accuracy_assessed": False})
    write_tsv(out / "TERM_LEDGER.tsv", ledger, list(ledger[0]))
    write_json(out / "SOURCE_TEMPLATES.json", templates)
    write_json(out / "SAMPLE_CARDS.json", {"schema": "GENERIC_STUDY_CONTRAST_v1", "discovery": {"study_id": "GSE52778", "tissue": "human airway smooth muscle", "positive_group": "dexamethasone", "reference_group": "control vehicle", "hours": 18, "paired_donors": 4}, "validation": {"study_id": "GSE34313", "primary_hours": 24, "treated_cultures": 3, "control_cultures": 4, "cell_line": "HASM1", "pairing_verified": False, "cross_study_donor_disjointness_verified": False}})
    summary = {"schema": "CRM_R1_EXTERNAL_DEX_RESULTS_R08", "status": "COMPLETE_LIMITED_CROSS_STUDY_CASE", "scope": protocol["scope"], "candidate_count": 50, "candidate_retained_count": 50, "discovery_donor_count": 4, "validation_cell_line_count": 1, "validation_primary_cultures": {"dex24": 3, "control": 4}, "model_calls": 0, "new_expert_ratings": 0, "full_audit_performance_estimated": False, "semantic_accuracy_estimated": False, "cross_study_donor_disjointness_verified": False, "descriptive_replication_rate_difference_stability_minus_matched_q": delta, "comparisons": comparisons, "design_sha256": job["design_sha256"], "input_manifest_sha256": job["input_manifest_sha256"], "published_result_exposure": protocol["outcome_exposure"], "interpretation": "Statistical pathway replication in a limited worked case; overlapping pathways do not supply independent trials, and no generalization/superiority CI is computed."}
    write_json(out / "SUMMARY.json", summary)
    return summary


def verify_export(out):
    for name, digest in read_json(out / "EXPORT_SHA256.json")["files"].items():
        p = Path(name)
        require(not p.is_absolute() and ".." not in p.parts and sha(out / p) == digest, "Existing analysis export changed")


def analyze(root, lock_dir, lock, directory, inputs):
    identity = runtime_identity()
    require(identity["ready"], "Missing R dependencies: " + ", ".join(identity["missing"]) + ". Use --install-missing with --analyze after installing R if needed.")
    runtime_seal = lock_dir / "RUNTIME_IDENTITY.json"
    if runtime_seal.exists():
        require(read_json(runtime_seal) == identity, "Frozen R/airway runtime changed")
    else:
        write_json(runtime_seal, identity)
    receipt = lock_dir / "COMPLETED_ANALYSIS.json"
    if receipt.exists():
        record = read_json(receipt)
        require(record["input_manifest_sha256"] == object_sha(inputs), "Analysis inputs changed")
        out = data_path(root, record["outdir_relative_path"])
        verify_export(out)
        require(sha(Path(str(out) + ".zip")) == record["archive_sha256"], "Existing results archive changed")
        print(json.dumps({"status": "EXISTING_COMPLETED_ANALYSIS_REUSED", "outdir": str(out), "results_archive": str(out) + ".zip"}, indent=2))
        return
    protocol = read_json(lock_dir / "protocol.json")
    check_sample_metadata(directory / "GSE34313_series_matrix.txt.gz", protocol)
    mapping = platform_mapping(directory / "GPL6480.soft.txt")
    out = data_path(root, "output/revision_v17/external_dex_" + stamp())
    out.mkdir(parents=True)
    job = {"schema": "CRM_R1_EXTERNAL_DEX_JOB_R08", "outdir": str(out), "protocol": str(lock_dir / "protocol.json"), "hallmark": str(lock_dir / "hallmark_gene_symbols.tsv"), "design_sha256": object_sha(lock), "input_manifest_sha256": object_sha(inputs), "R_runtime": identity}
    try:
        arrays = out / "raw_arrays"
        arrays.mkdir()
        job["raw_files"] = extract_arrays(directory / "GSE34313_RAW.tar", arrays, protocol)
        write_tsv(out / "GPL6480.mapping.tsv", mapping, ["probe_id", "gene_symbol", "control_type"])
        job["platform_mapping"] = str(out / "GPL6480.mapping.tsv")
        write_json(out / "JOB.private.json", job)
        shutil.copyfile(lock_dir / "protocol.json", out / "protocol.json")
        shutil.copyfile(lock_dir / "DESIGN_LOCK.json", out / "DESIGN_LOCK.json")
        shutil.copyfile(lock_dir / "hallmark_gene_symbols.tsv", out / "hallmark_gene_symbols.tsv")
        write_json(out / "INPUT_MANIFEST.json", inputs)
        write_json(out / "RUNTIME_IDENTITY.json", identity)
        with (out / "analysis.log.txt").open("x", encoding="utf-8") as log:
            print("[R08] Running paired discovery, four donor folds, and culture-level validation", flush=True)
            result = subprocess.run([shutil.which("Rscript"), "--vanilla", str(HERE / "analysis.R"), str(out / "JOB.private.json")], stdout=log, stderr=subprocess.STDOUT, check=False)
        require(result.returncode == 0, f"R analysis stopped; inspect {out / 'analysis.log.txt'}")
        marker = read_json(out / "R_STATISTICS_COMPLETE.json")
        require(marker.get("schema") == "CRM_R1_EXTERNAL_DEX_R_STATS_R08" and marker.get("donor_folds") == 4 and marker.get("model_calls") == 0 and marker.get("expression_outcomes_loaded") is True, "R completion marker is missing or inconsistent")
        with (out / "common_gene_universe.tsv").open(encoding="utf-8", newline="") as handle:
            genes = [r["gene_symbol"] for r in csv.DictReader(handle, delimiter="\t")]
        require(len(genes) >= 1000 and len(set(genes)) == len(genes) and len(genes) == marker.get("common_genes"), "Common measured universe is inconsistent")
        require(runtime_identity() == identity, "R/airway runtime changed during analysis")
        require(input_manifest(root, lock)[1] == inputs, "Public inputs changed during analysis")
        freeze(root)
        summary = summarize(out, protocol, job)
        # Public raw files remain as immutable inputs; the compact results ZIP contains their hashes.
        write_json(out / "RAW_ARRAY_SHA256.json", {p.name: sha(p) for p in sorted(arrays.iterdir())})
        export = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and not p.is_relative_to(arrays)}
        write_json(out / "EXPORT_SHA256.json", {"schema": "CRM_R1_EXTERNAL_DEX_EXPORT_R08", "files": export, "excluded_public_raw_arrays": "raw_arrays/; see RAW_ARRAY_SHA256.json and original input archive"})
        archive = Path(str(out) + ".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
            for name in [*export, "EXPORT_SHA256.json"]:
                handle.write(out / name, name)
        write_json(receipt, {"outdir_relative_path": str(out.relative_to(root)), "input_manifest_sha256": object_sha(inputs), "archive_sha256": sha(archive)})
        print(json.dumps({**summary, "outdir": str(out), "results_archive": str(archive)}, indent=2))
    except Exception as error:
        if not (out / "STOPPED.private.json").exists():
            write_json(out / "STOPPED.private.json", {"status": "STOPPED_PRESERVED", "reason": str(error), "design_sha256": object_sha(lock), "model_calls": 0})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--hallmark", type=Path, help="Existing human gene-symbol Hallmark snapshot inside CRM_R1")
    parser.add_argument("--freeze", action="store_true", help="Freeze protocol only (also performed before any other action)")
    parser.add_argument("--fetch", action="store_true", help="Download the fixed public GEO files; no model calls")
    parser.add_argument("--analyze", action="store_true", help="Run the fixed statistical analysis")
    parser.add_argument("--install-missing", action="store_true", help="Install only missing free R packages; never update existing packages")
    args = parser.parse_args(argv)
    try:
        require(not args.install_missing or args.analyze, "--install-missing requires --analyze")
        root = args.data_root.expanduser().resolve()
        require((root / "input").is_dir() and (root / "output").is_dir(), "Use the existing CRM_R1 with input/ and output/")
        repo = HERE.parents[4]
        require(not root.is_relative_to(repo), "Data root must be outside the repository")
        lock_dir, lock = freeze(root, args.hallmark.expanduser() if args.hallmark else None)
        identity = runtime_identity()
        print(json.dumps({"status": "PROTOCOL_FROZEN_BEFORE_ANALYSIS", "design_dir": str(lock_dir), "design_sha256": object_sha(lock), "candidate_count": 50, "R_dependencies": identity, "model_calls": 0, "new_expert_ratings": 0}, indent=2), flush=True)
        if args.install_missing:
            install_missing(identity)
        if args.fetch or args.analyze:
            directory, inputs = input_manifest(root, lock, fetch=args.fetch)
            print("[R08] Fixed public input hashes recorded", flush=True)
        if args.analyze:
            analyze(root, lock_dir, lock, directory, inputs)
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"[STOP] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
