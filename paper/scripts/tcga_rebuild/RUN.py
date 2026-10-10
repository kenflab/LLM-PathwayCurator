#!/usr/bin/env python3
"""Rebuild TCGA diagnostic inputs using the merged code and public MC3 metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import uuid
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

BUNDLE = Path(__file__).resolve().parent
INPUTS = BUNDLE


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def command(args, cwd=None, env=None, log=None):
    args = [str(x) for x in args]
    if log:
        with Path(log).open("w") as handle:
            completed = subprocess.run(
                args, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
        if completed.returncode:
            raise RuntimeError(f"Command failed; see {log}")
        return ""
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def verify_bundle():
    manifest = json.loads((BUNDLE / "INPUT_MANIFEST.json").read_text())
    for item in manifest["public_files"]:
        p = INPUTS / item["path"]
        if not p.is_file() or digest(p) != item["sha256"]:
            raise RuntimeError(f"Public input checksum mismatch: {p}")
    return manifest


def export_code(repo, out, commit):
    # Export committed code; do not use or change the author's working files.
    try:
        command(["git", "-C", repo, "cat-file", "-e", commit + "^{commit}"])
    except subprocess.CalledProcessError:
        command(["git", "-C", repo, "fetch", "origin", "main"])
        command(["git", "-C", repo, "cat-file", "-e", commit + "^{commit}"])
    out.mkdir()
    archive = out.parent / "code.tar"
    with archive.open("wb") as f:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                commit,
                "--",
                "src/llm_pathway_curator",
                "paper/scripts",
                "resources/gene_id_maps",
                "pyproject.toml",
            ],
            stdout=f,
            check=True,
        )
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            target = (out / member.name).resolve()
            if out.resolve() not in target.parents or not (member.isfile() or member.isdir()):
                raise RuntimeError("Unexpected entry in committed code archive")
        for member in tf.getmembers():
            target = out / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    archive.unlink()
    save_json(
        out.parent / "code_manifest.json",
        {
            "commit": commit,
            "files": {
                str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()
            },
        },
    )


def expression_path(args, manifest):
    expected = manifest["expression"]
    if args.expression:
        choices = [args.expression.expanduser().resolve()]
    else:
        raw = args.repo / "paper/source_data/PANCAN_TP53_v1/raw"
        cache = args.data_root / "input/tcga_public_inputs_20261009"
        choices = [
            raw / "expression.xena.gz",
            raw / "expression.tsv.gz",
            raw / "expression.xena.tsv.gz",
            cache / "expression.xena.gz",
        ]
    for p in choices:
        if p.exists():
            print(f"Checking expression checksum: {p}", flush=True)
            if digest(p) != expected["sha256"]:
                raise RuntimeError(f"Expression differs from the reviewed public input: {p}")
            return p
    if args.expression:
        raise FileNotFoundError(args.expression)
    p = choices[-1]
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + "." + uuid.uuid4().hex + ".part")
    print("Downloading reviewed Xena expression (331 MB)...", flush=True)
    try:
        with urllib.request.urlopen(expected["url"], timeout=90) as r, temporary.open("wb") as w:
            shutil.copyfileobj(r, w)
        if temporary.stat().st_size != expected["bytes"] or digest(temporary) != expected["sha256"]:
            raise RuntimeError("Downloaded expression checksum mismatch")
        if p.exists():
            raise RuntimeError("Expression destination appeared during download; inspect it")
        temporary.rename(p)
    finally:
        if temporary.exists():
            temporary.unlink()
    save_json(p.with_suffix(".gz.receipt.json"), expected)
    return p


def load_group_module(code):
    path = code / "paper/scripts/fig2_make_groups.py"
    spec = importlib.util.spec_from_file_location("current_tcga_group_code", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_assessment(code, expression, out, policy):
    module = load_group_module(code)
    src = INPUTS / "public_inputs"
    ph = pd.read_csv(src / "TCGA_phenotype_dense.tsv.gz", sep="\t", dtype=str).fillna("")
    ph["sample"] = ph["sample"].map(module._normalize_barcode)
    ph["cancer"] = ph["_primary_disease"].map(module._norm_disease).map(module.DISEASE_TO_TCGA)
    ph = ph[ph.sample_type_id.eq("01") & ph.cancer.isin(policy["cohorts"])].copy()
    if ph.groupby("sample").cancer.nunique().gt(1).any():
        raise RuntimeError("Conflicting phenotype labels")
    ph = ph.drop_duplicates("sample")
    with gzip.open(expression, "rt") as f:
        expression_samples = next(f).rstrip("\r\n").split("\t")[1:]
    duplicated_expression = {s for s, n in Counter(expression_samples).items() if n > 1}
    save_json(
        out / "expression_sample_audit.json",
        {
            "n_columns": len(expression_samples),
            "n_unique_samples": len(set(expression_samples)),
            "duplicate_samples_excluded": sorted(duplicated_expression),
            "policy": (
                "Exclude all columns for ambiguous duplicate sample identifiers; "
                "do not choose a replicate."
            ),
        },
    )
    ph = ph[ph["sample"].isin(expression_samples)].copy()

    pairs = pd.read_csv(src / "pairing_of_best_bams_2016-04-05.txt", sep="\t", dtype=str).fillna("")
    types = pairs.dnanexus_pair_id.str.extract(r"_([0-9]{2})_[0-9]{2}$")[0]
    if types.isna().any():
        raise RuntimeError("Unrecognized sample types in official pairing table")
    pairs["sample"] = (pairs.broad_file_pair_id.str.split("_").str[0] + "-" + types).map(
        module._normalize_barcode
    )
    paired = set(pairs["sample"])
    pair_cohorts = pairs.groupby("sample").cohort.agg(set).to_dict()

    qc = pd.read_csv(src / "merged_sample_quality_annotations.tsv", sep="\t", dtype=str).fillna("")
    qc = qc[qc.platform.isin(["IlluminaHiSeq_RNASeqV2", "IlluminaGA_RNASeqV2"])].copy()
    qc["sample"] = qc.aliquot_barcode.map(module._normalize_barcode)
    good_row = qc.Do_not_use.eq("False") & qc.AWG_excluded_because_of_pathology.eq("0.0")
    qc_good = set(qc.loc[good_row, "sample"]) - set(qc.loc[~good_row, "sample"])

    inventory = json.loads((src / "mc3_sample_filter_inventory.json").read_text())["samples"]
    tp = pd.read_csv(src / "mc3.TP53.all_filters.tsv.gz", sep="\t", dtype=str).fillna("")
    tp["sample"] = tp.Tumor_Sample_Barcode.map(module._normalize_barcode)
    cancer_by_sample = ph.set_index("sample").cancer.to_dict()
    pa = module.PROTEIN_ALTERING
    extended_pa = pa | {"Translation_Start_Site", "Nonstop_Mutation"}
    pass_only = tp.FILTER.eq("PASS") & tp.Variant_Classification.isin(pa)
    ov_wga = (
        tp["sample"].map(cancer_by_sample).eq("OV")
        & tp.FILTER.eq("wga")
        & bool(policy.get("OV_wga_only", False))
    )
    passing = (tp.FILTER.eq("PASS") | ov_wga) & tp.Variant_Classification.isin(pa)
    mutation_samples = set(tp.loc[passing, "sample"])
    pass_samples = set(tp.loc[pass_only, "sample"])
    ambiguous = (
        set(tp.loc[tp.Variant_Classification.isin(extended_pa) & ~passing, "sample"])
        - mutation_samples
    )
    wga_tp53 = set(
        tp.loc[tp.FILTER.eq("wga") & tp.Variant_Classification.isin(extended_pa), "sample"]
    )
    gdc_positive_cases = set()
    if policy.get("OV_GDC_positive_guard", False):
        snapshot = json.loads((src / "gdc_OV_TP53_occurrences.json").read_text())["data"]
        if snapshot["pagination"]["total"] != len(snapshot["hits"]):
            raise RuntimeError("Incomplete GDC occurrence snapshot")
        protein_terms = {
            "missense_variant",
            "frameshift_variant",
            "stop_gained",
            "splice_acceptor_variant",
            "splice_donor_variant",
            "inframe_insertion",
            "inframe_deletion",
            "stop_lost",
            "start_lost",
        }
        for hit in snapshot["hits"]:
            relevant = [
                x["transcript"]
                for x in hit["ssm"].get("consequence", [])
                if x.get("transcript", {}).get("gene", {}).get("gene_id") == "ENSG00000141510"
            ]
            if any(set(t.get("consequence_type", "").split("&")) & protein_terms for t in relevant):
                gdc_positive_cases.add(hit["case"]["submitter_id"])
    records = []
    for row in ph.itertuples(index=False):
        s = row.sample
        inv = inventory.get(s)
        flags = {flag for group in inv["filters"] for flag in group.split(",")} if inv else set()
        aliquots = inv["tumor_aliquots"] + inv["normal_aliquots"] if inv else []
        native = bool(aliquots) and all(a.split("-")[4][-1] == "D" for a in aliquots)
        dna_confirmed = bool(aliquots) and all(
            a.split("-")[4][-1] in {"D", "W", "X", "G"} for a in aliquots
        )
        allow_ov_wga = row.cancer == "OV" and policy.get("OV_wga_only", False)
        preferred = bool(inv) and any(
            "nonpreferredpair" not in v.split(",") for v in inv["filters"]
        )
        reasons = []
        if s in duplicated_expression:
            reasons.append("duplicate_expression_columns")
        if s not in paired:
            reasons.append("no_official_MC3_pair")
        elif pair_cohorts[s] != {row.cancer}:
            reasons.append("pair_cohort_conflict")
        if s not in qc_good:
            reasons.append("RNA_QC_missing_or_excluded")
        if allow_ov_wga:
            if not dna_confirmed:
                reasons.append("tumor_normal_DNA_unconfirmed")
            if "native_wga_mix" in flags:
                reasons.append("mixed_native_WGA")
        else:
            if not native:
                reasons.append("native_tumor_normal_DNA_unconfirmed")
            if flags & {"wga", "native_wga_mix"}:
                reasons.append("WGA_flag")
        if flags & {"badseq", "contest"}:
            reasons.append("sample_filter")
        if not preferred:
            reasons.append("preferred_pair_unconfirmed")
        if s in ambiguous:
            reasons.append("ambiguous_TP53_call")
        eligible_before_gdc_guard = not reasons
        dna_reasons = [
            r
            for r in reasons
            if r not in {"RNA_QC_missing_or_excluded", "duplicate_expression_columns"}
        ]
        gdc_discordant = (
            row.cancer == "OV" and s[:12] in gdc_positive_cases and s not in mutation_samples
        )
        if gdc_discordant:
            reasons.append("MC3_negative_GDC_positive_case")
        group = (
            ("TP53_mut" if s in mutation_samples else "TP53_wt") if not reasons else "TP53_unknown"
        )
        records.append(
            {
                "sample": s,
                "cancer": row.cancer,
                "group": group,
                "exclusion_reasons": ";".join(reasons),
                "MC3_best_pair": s in paired,
                "RNA_QC_pass": s in qc_good,
                "native_DNA": native,
                "WGA_flag": bool(flags & {"wga", "native_wga_mix"}),
                "TP53_PASS_protein_altering": s in pass_samples,
                "TP53_wga_protein_altering": s in wga_tp53,
                "TP53_accepted_MC3_call": s in mutation_samples,
                "TP53_call_status_before_RNA_QC": (
                    "MUT" if s in mutation_samples else "MC3_call_negative"
                )
                if not dna_reasons
                else "unknown",
                "eligible_before_GDC_guard": eligible_before_gdc_guard,
                "GDC_TP53_positive_case": s[:12] in gdc_positive_cases
                if row.cancer == "OV"
                else "not_queried",
                "MC3_GDC_discordant": gdc_discordant,
                "TP53_locus_callable": "not_established_by_public_metadata",
            }
        )
    ledger = pd.DataFrame(records).sort_values(["cancer", "sample"])
    ledger.to_csv(out / "sample_ledger.tsv", sep="\t", index=False)
    assessed = ledger.group.ne("TP53_unknown")
    assessment = ledger[["sample", "cancer"]].copy()
    assessment["tp53_assessed"] = assessed.map({True: "true", False: "false"})
    assessment["assessment_basis"] = "MC3_exome_metadata_cohort_specific_variant_policy_and_RNA_QC"
    assessment["locus_callability"] = "not_established"
    assessment.to_csv(out / "assessment.tsv", sep="\t", index=False)
    eligible = set(ledger.loc[assessed, "sample"])
    selected = tp.loc[tp["sample"].isin(eligible)]
    # In the original MAF, Gene is an Ensembl ID and Hugo_Symbol is the symbol.
    # Export explicit Xena-style columns to avoid alias-dependent interpretation.
    canonical = pd.DataFrame(
        {
            "sample": selected.Tumor_Sample_Barcode,
            "cancer": selected["sample"].map(cancer_by_sample),
            "gene": selected.Hugo_Symbol,
            "effect": selected.Variant_Classification,
            "FILTER": selected.FILTER,
        }
    )
    canonical.to_csv(out / "TP53.eligible.tsv.gz", sep="\t", index=False, compression="gzip")
    ph.drop(columns="cancer").to_csv(
        out / "phenotype.expression_matched.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    summary = []
    for cancer in policy["cohorts"]:
        table = ledger[ledger.cancer.eq(cancer)]
        n = table.group.value_counts()
        mut, wt = int(n.get("TP53_mut", 0)), int(n.get("TP53_wt", 0))
        eligible_fit = min(mut, wt) >= 2 and mut + wt >= 10
        summary.append(
            {
                "cancer": cancer,
                "n_expression": len(table),
                "n_mut": mut,
                "n_wt": wt,
                "n_unknown": int(n.get("TP53_unknown", 0)),
                "n_wga_flag": int(table.WGA_flag.sum()),
                "n_TP53_wga_calls": int(table.TP53_wga_protein_altering.sum()),
                "n_GDC_discordant_excluded_after_QC": int(
                    (table.MC3_GDC_discordant & table.eligible_before_GDC_guard).sum()
                ),
                "fit_eligible": eligible_fit,
                "small_comparator_arm": min(mut, wt) < 20,
                "enrichment_status": "pending" if eligible_fit else "not_estimable",
            }
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "cohort_summary.tsv", sep="\t", index=False)
    save_json(
        out / "assessment_provenance.json",
        {
            "policy": policy,
            "raw_MC3_sha256": json.loads((BUNDLE / "INPUT_MANIFEST.json").read_text())["full_mc3"][
                "sha256"
            ],
            "assessment_sha256": digest(out / "assessment.tsv"),
            "note": (
                "WT is operationally call-negative among eligible exome-profiled "
                "samples; per-base TP53 coverage is unavailable."
            ),
        },
    )
    return ledger, summary


def write_groups(code, out, policy):
    """Use the merged public group-assignment function, preserving source FILTER tags."""
    module = load_group_module(code)
    variants = pd.read_csv(out / "TP53.eligible.tsv.gz", sep="\t", dtype=str).fillna("")
    accepted = variants.FILTER.eq("PASS")
    if policy.get("OV_wga_only", False):
        accepted |= variants.cancer.eq("OV") & variants.FILTER.eq("wga")
    accepted &= variants.gene.eq("TP53") & variants.effect.isin(module.PROTEIN_ALTERING)
    mutants = set(variants.loc[accepted, "sample"].map(module._normalize_barcode))
    assessment = pd.read_csv(out / "assessment.tsv", sep="\t", dtype=str)
    df = module.assign_tp53_groups(assessment[["sample", "cancer"]], mutants, assessment)
    directory = out / "groups"
    directory.mkdir()
    for c, g in df.groupby("cancer"):
        g[["sample", "group", "group_basis"]].to_csv(
            directory / f"{c}.groups.tsv", sep="\t", index=False
        )
    df.to_csv(directory / "PANCAN.groups.tsv", sep="\t", index=False)
    save_json(
        directory / "groups.provenance.json",
        {
            "assignment_function": "merged fig2_make_groups.assign_tp53_groups",
            "policy_sha256": digest(out / "ANALYSIS_POLICY.json"),
            "input_sha256": digest(out / "TP53.eligible.tsv.gz"),
            "assessment_sha256": digest(out / "assessment.tsv"),
            "mutation_filter": "PASS in all cohorts; exact wga-only additionally accepted for OV",
            "original_FILTER_tags_preserved": True,
            "OV_GDC_positive_guard": policy.get("OV_GDC_positive_guard", False),
            "barcode_unit": "sample_15_characters; GDC discordance guard at case_12_characters",
            "negative_label": "MC3 call-negative, not confirmed biological WT",
        },
    )


def split_expression(expression, ledger, summary, directory):
    directory.mkdir()
    cancers = summary.loc[summary.fit_eligible, "cancer"].tolist()
    handles = {}
    try:
        with gzip.open(expression, "rt") as source:
            header = next(source).rstrip("\r\n").split("\t")
            counts = Counter(header[1:])
            indexes = {}
            for c in cancers:
                wanted = set(ledger.loc[ledger.cancer.eq(c), "sample"])
                indexes[c] = [0] + [
                    i for i, s in enumerate(header) if s in wanted and counts[s] == 1
                ]
                handles[c] = gzip.open(directory / f"{c}.expression.tsv.gz", "wt")
                handles[c].write("\t".join(header[i] for i in indexes[c]) + "\n")
            n = 0
            for line in source:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != len(header):
                    raise RuntimeError("Expression row width mismatch")
                for c, handle in handles.items():
                    handle.write("\t".join(fields[i] for i in indexes[c]) + "\n")
                n += 1
            if n != 20531:
                raise RuntimeError(f"Unexpected expression gene count: {n}")
    finally:
        for handle in handles.values():
            handle.close()


def collect_results(out, summary):
    all_terms = []
    for index, row in summary.iterrows():
        path = out / "evidence_tables" / f"{row.cancer}.evidence_table.tsv"
        if not path.exists():
            continue
        table = pd.read_csv(path, sep="\t")
        if table.term_id.duplicated().any() or not table.qval.between(0, 1).all():
            raise RuntimeError(f"Invalid enrichment result: {path}")
        summary.loc[index, "n_tested"] = len(table)
        summary.loc[index, "n_q_le_0_05"] = int(table.qval.le(0.05).sum())
        summary.loc[index, "enrichment_status"] = (
            "complete" if len(table) == 50 else "incomplete_Hallmark_results"
        )
        table.insert(0, "cancer", row.cancer)
        all_terms.append(table)
    summary.to_csv(out / "cohort_summary.tsv", sep="\t", index=False)
    if all_terms:
        pd.concat(all_terms).to_csv(out / "hallmark_results.tsv", sep="\t", index=False)
    return summary


def return_zip(out):
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(out)
            if rel.parts[0] in {"code", "expression_by_cohort"} or "__pycache__" in rel.parts:
                continue
            z.write(p, arcname=out.name + "/" + str(rel))
    return archive


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    global INPUTS
    ap.add_argument(
        "--inputs",
        required=True,
        type=Path,
        help="Directory containing the checksum-verified public_inputs snapshot",
    )
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--expression", type=Path)
    ap.add_argument("--outdir", type=Path)
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument(
        "--run",
        action="store_true",
        help="Also run ranking, Hallmark fgsea and current source reports",
    )
    args = ap.parse_args()
    args.repo, args.data_root = (
        args.repo.expanduser().resolve(),
        args.data_root.expanduser().resolve(),
    )
    INPUTS = args.inputs.expanduser().resolve()
    manifest = verify_bundle()
    policy = json.loads((BUNDLE / "ANALYSIS_POLICY.json").read_text())
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = (
        args.outdir or args.data_root / "output/revision_v17" / ("tcga_rebuild_OVFIX1_" + stamp)
    ).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    shutil.copyfile(BUNDLE / "ANALYSIS_POLICY.json", out / "ANALYSIS_POLICY.json")
    shutil.copyfile(BUNDLE / "INPUT_MANIFEST.json", out / "INPUT_MANIFEST.json")
    print(f"Output: {out}", flush=True)
    status = {
        "code_commit": manifest["code_commit"],
        "status": "started",
        "model_calls": 0,
        "analysis_policy_sha256": digest(out / "ANALYSIS_POLICY.json"),
        "python": platform.python_version(),
        "wrapper_sha256": {p.name: digest(p) for p in sorted(BUNDLE.glob("*.py"))},
        "R_preflight_sha256": digest(BUNDLE / "CHECK_R.R"),
        "public_script_port": "OVFIX1; selection and analysis rules unchanged",
    }
    summary = None
    try:
        expression = expression_path(args, manifest)
        status["expression_sha256"] = manifest["expression"]["sha256"]
        code = out / "code"
        export_code(args.repo, code, manifest["code_commit"])
        ledger, summary = build_assessment(code, expression, out, policy)
        print(summary.to_string(index=False), flush=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLMPATH_")}
        env.update(
            {
                "PYTHONPATH": str(code / "src"),
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "LLMPATH_BACKEND": "none",
            }
        )
        write_groups(code, out, policy)
        groups = pd.read_csv(out / "groups/PANCAN.groups.tsv", sep="\t")
        merged = ledger.merge(
            groups,
            on=["sample", "cancer"],
            suffixes=("_expected", "_actual"),
            validate="one_to_one",
        )
        if len(merged) != len(ledger) or not merged.group_expected.eq(merged.group_actual).all():
            raise RuntimeError("Group builder disagrees with the independent sample ledger")
        if args.run:
            command(
                [
                    args.rscript,
                    BUNDLE / "CHECK_R.R",
                    code,
                    INPUTS / "public_inputs/hallmark.2026_1_Hs.memberships.tsv",
                    out / "R_package_versions.tsv",
                ],
                env=env,
                log=out / "logs/R_environment.log",
            )
            print(
                "Splitting expression by cohort without changing expression values...", flush=True
            )
            split_expression(expression, ledger, summary, out / "expression_by_cohort")
            (out / "sample_cards").mkdir()
            for row in summary.itertuples(index=False):
                if not row.fit_eligible:
                    continue
                c = row.cancer
                print(f"Running {c}: MUT={row.n_mut}, assay-negative={row.n_wt}", flush=True)
                command(
                    [
                        args.rscript,
                        code / "paper/scripts/fig2_deg_rank.R",
                        c,
                        "--expression",
                        out / "expression_by_cohort" / f"{c}.expression.tsv.gz",
                        "--groups",
                        out / "groups" / f"{c}.groups.tsv",
                        "--outdir",
                        out / "rankings",
                    ],
                    cwd=code,
                    env=env,
                    log=out / "logs" / f"{c}.ranking.log",
                )
                command(
                    [
                        args.rscript,
                        code / "paper/scripts/fig2_fgsea_to_evidence_table.R",
                        c,
                        "--rank",
                        out / "rankings" / f"{c}.deg_ranking.tsv",
                        "--outdir",
                        out / "evidence_tables",
                    ],
                    cwd=code,
                    env=env,
                    log=out / "logs" / f"{c}.fgsea.log",
                )
                card = {
                    "condition": c,
                    "tissue": "tumor",
                    "perturbation": "genotype",
                    "comparison": "TP53_MUT_vs_MC3_call_negative",
                    "k_claims": 50,
                    "notes": (
                        "Diagnostic MC3 comparison; OV additionally accepts exact "
                        "wga-only calls and excludes known GDC-positive/MC3-negative "
                        "cases. WT is operationally call-negative, not biological WT."
                    ),
                    "extra": {
                        "benchmark_id": policy["analysis_id"],
                        "n_mut": row.n_mut,
                        "n_wt": row.n_wt,
                    },
                }
                card_path = out / "sample_cards" / f"{c}.json"
                save_json(card_path, card)
                command(
                    [
                        sys.executable,
                        "-m",
                        "llm_pathway_curator.cli",
                        "run",
                        "--workflow",
                        "source",
                        "--evidence-table",
                        out / "evidence_tables" / f"{c}.evidence_table.tsv",
                        "--sample-card",
                        card_path,
                        "--outdir",
                        out / "source_reports" / c,
                        "--q-threshold",
                        "0.05",
                        "--modules",
                        "--k-claims",
                        "50",
                    ],
                    cwd=code,
                    env=env,
                    log=out / "logs" / f"{c}.curator.log",
                )
                if not (out / "source_reports" / c / "report.html").exists():
                    raise RuntimeError(f"Current source workflow did not produce a report for {c}")
            summary = collect_results(out, summary)
            status["status"] = (
                "completed"
                if all(summary.loc[summary.fit_eligible, "enrichment_status"].eq("complete"))
                else "incomplete_results"
            )
            if status["status"] == "completed":
                for script in [
                    "AUDIT_OV.py",
                    "VERIFY_RESULTS.py",
                    "MAKE_FIGURES.py",
                    "MAKE_REPORT.py",
                ]:
                    command(
                        [sys.executable, BUNDLE / script, "--run-dir", out]
                        + (
                            ["--inputs", INPUTS]
                            if script in {"AUDIT_OV.py", "VERIFY_RESULTS.py"}
                            else []
                        ),
                        env=env,
                        log=out / "logs" / (script + ".log"),
                    )
        else:
            status["status"] = "assessment_and_groups_complete"
        status["small_arm_cohorts"] = summary.loc[summary.small_comparator_arm, "cancer"].tolist()
        status["not_estimable_cohorts"] = summary.loc[~summary.fit_eligible, "cancer"].tolist()
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        if summary is not None:
            collect_results(out, summary)
        raise
    finally:
        save_json(out / "RUN_STATUS.json", status)
        archive = return_zip(out)
        print(
            json.dumps(
                {"status": status["status"], "outdir": str(out), "return_zip": str(archive)},
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
