#!/usr/bin/env python3
# paper/scripts/figS4_beataml_make_groups.py
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # paper/
SD = ROOT / "source_data" / "BEATAML_TP53_v1"
RAW = SD / "raw"
DER = SD / "derived"
OUT_GROUPS = DER / "groups"


def _die(msg: str) -> None:
    raise SystemExit(msg)


def _warn(msg: str) -> None:
    print(f"[beataml_groups] WARNING: {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[beataml_groups] {msg}", file=sys.stderr)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        _die(f"[beataml_groups] missing file: {path}")
    suf = path.suffix.lower()
    try:
        if suf == ".xlsx":
            return pd.read_excel(path).fillna("")
        try:
            return pd.read_csv(path, sep="\t", dtype=str, low_memory=False).fillna("")
        except Exception:
            return pd.read_csv(path, sep=",", dtype=str, low_memory=False).fillna("")
    except Exception as e:
        _die(f"[beataml_groups] failed to read {path}\n{e}")


def _canon(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _pick_first_present(df: pd.DataFrame, candidates: list[list[str]]) -> str | None:
    for group in candidates:
        c = _pick_col(df, group)
        if c:
            return c
    return None


def _clean_id(x: str) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", "", s)
    return s


def _score_overlap(a: set[str], b: set[str]) -> int:
    return len(a & b)


def _extract_sample_ids_from_counts(counts_path: Path, *, max_cols: int = 200000) -> list[str]:
    if not counts_path.exists():
        _die(f"[beataml_groups] missing counts file: {counts_path}")
    with counts_path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n")
    cols = header.split("\t")
    if len(cols) <= 1:
        cols = header.split(",")
    if len(cols) <= 1:
        _die("[beataml_groups] cannot parse counts header as TSV/CSV (need sample columns)")
    sample_cols = cols[1:]
    if len(sample_cols) > max_cols:
        _warn(
            f"counts header has {len(sample_cols)} sample columns; truncating to first {max_cols}"
        )
        sample_cols = sample_cols[:max_cols]
    out = []
    for x in sample_cols:
        s = _clean_id(x)
        if s:
            out.append(s)
    return out


def _canon_effect(x: str) -> str:
    s = _canon(x)
    return s.replace("-", "_").replace(" ", "_")


def _is_protein_altering(effect: str) -> bool:
    s = _canon_effect(effect)
    if any(
        tok in s
        for tok in [
            "missense",
            "nonsense",
            "frameshift",
            "frame_shift",
            "splice",
            "inframe",
            "in_frame",
        ]
    ):
        return True
    if s in {
        "stop_gained",
        "stop_lost",
        "start_lost",
        "start_gained",
        "nonstop",
        "protein_altering_variant",
    }:
        return True
    if "nonsyn" in s or "non_syn" in s:
        return True
    return False


def _truthy(x: str) -> bool:
    s = _canon(x)
    return s in {"1", "true", "t", "yes", "y"} or s.startswith("true")


def _clinical_tp53_positive(x: str) -> bool:
    s = str(x or "").strip()
    if not s:
        return False
    # broad: any TP53 token counts as positive; avoid explicit "WT"
    su = s.upper()
    if "TP53" in su and "WT" not in su:
        return True
    return False


@dataclass(frozen=True)
class ClinicalBridge:
    subj_col: str
    rnaseq_col: str | None
    dnaseq_col: str | None
    tp53_col: str | None
    flag_used_col: str | None
    flag_rna_col: str | None
    flag_exome_col: str | None


def _detect_clinical_bridge(clin: pd.DataFrame) -> ClinicalBridge:
    subj_col = _pick_first_present(
        clin,
        [
            ["dbgap_subject_id", "subject_id", "participant_id", "patient_id", "case_id"],
            ["subject", "participant", "patient", "case"],
        ],
    )
    if not subj_col:
        _die("[beataml_groups] clinical: cannot detect subject/patient column")

    rnaseq_col = _pick_first_present(
        clin,
        [
            ["dbgap_rnaseq_sample", "rnaseq_sample", "rna_sample_id", "rnaseq_sample_id"],
            ["dbgap_rnaseq", "rnaseq"],
        ],
    )
    dnaseq_col = _pick_first_present(
        clin,
        [
            [
                "dbgap_dnaseq_sample",
                "dnaseq_sample",
                "exomeseq_sample",
                "dna_sample_id",
                "dnaseq_sample_id",
            ],
            ["dbgap_dnaseq", "dnaseq", "exomeseq"],
        ],
    )

    tp53_col = _pick_first_present(
        clin,
        [
            ["tp53"],
            ["variantSummary", "variant_summary"],
        ],
    )

    flag_used_col = _pick_first_present(
        clin,
        [
            ["used_manuscript_analyses"],
            ["manuscript_dnaseq", "manuscript_rnaseq"],
        ],
    )
    flag_rna_col = _pick_first_present(
        clin,
        [
            ["analysisRnaSeq", "analysis_rnaseq", "rnaSeq", "manuscript_rnaseq"],
        ],
    )
    flag_exome_col = _pick_first_present(
        clin,
        [
            ["analysisExomeSeq", "analysis_exomeseq", "exomeSeq", "manuscript_dnaseq"],
        ],
    )

    return ClinicalBridge(
        subj_col=subj_col,
        rnaseq_col=rnaseq_col,
        dnaseq_col=dnaseq_col,
        tp53_col=tp53_col,
        flag_used_col=flag_used_col,
        flag_rna_col=flag_rna_col,
        flag_exome_col=flag_exome_col,
    )


def _choose_counts_key(
    counts_ids: set[str], clin: pd.DataFrame, br: ClinicalBridge
) -> tuple[str, dict[str, str], dict[str, str]]:
    """
    Returns:
      key_level: "rnaseq" or "subject"
      rnaseq_to_subj: mapping for rnaseq sample -> subject
      dnaseq_to_subj: mapping for dnaseq sample -> subject
    """
    rnaseq_to_subj: dict[str, str] = {}
    dnaseq_to_subj: dict[str, str] = {}

    subj_ids = {_clean_id(x) for x in clin[br.subj_col].astype(str).tolist() if _clean_id(x)}

    if br.rnaseq_col:
        for r, s in zip(
            clin[br.rnaseq_col].astype(str), clin[br.subj_col].astype(str), strict=False
        ):
            rr = _clean_id(r)
            ss = _clean_id(s)
            if rr and ss:
                rnaseq_to_subj[rr] = ss

    if br.dnaseq_col:
        for d, s in zip(
            clin[br.dnaseq_col].astype(str), clin[br.subj_col].astype(str), strict=False
        ):
            dd = _clean_id(d)
            ss = _clean_id(s)
            if dd and ss:
                dnaseq_to_subj[dd] = ss

    ov_rna = _score_overlap(set(rnaseq_to_subj.keys()), counts_ids) if rnaseq_to_subj else 0
    ov_subj = _score_overlap(subj_ids, counts_ids)

    key_level = "rnaseq" if ov_rna >= ov_subj and ov_rna > 0 else "subject"
    _info(f"counts key choice: level={key_level} (overlap rnaseq={ov_rna}, subject={ov_subj})")

    return key_level, rnaseq_to_subj, dnaseq_to_subj


def _eligible_subjects(clin: pd.DataFrame, br: ClinicalBridge) -> set[str]:
    subj = clin[br.subj_col].astype(str).map(_clean_id)
    mask = pd.Series([True] * len(clin))

    # apply only if column exists; otherwise no-op
    if br.flag_used_col and br.flag_used_col in clin.columns:
        mask &= clin[br.flag_used_col].astype(str).map(_truthy)
    if br.flag_rna_col and br.flag_rna_col in clin.columns:
        mask &= clin[br.flag_rna_col].astype(str).map(_truthy)
    # exome flag is NOT required for inclusion in expression-based comparisons by default
    # (we still use exome IDs for mapping if available)

    out = set(subj[mask].tolist())
    out.discard("")
    return out


def _detect_mut_id_candidates(mut: pd.DataFrame) -> list[str]:
    cols = list(mut.columns)
    mut_cols_l = {c.lower(): c for c in cols}

    cand: list[str] = []
    for c in cols:
        cl = c.lower()
        if any(
            k in cl
            for k in [
                "sample",
                "specimen",
                "barcode",
                "rna",
                "participant",
                "patient",
                "case",
                "subject",
                "dbgap",
            ]
        ):
            cand.append(c)

    for k in ["dbgap_sample_id", "sample_id", "patient_id", "participant_id", "subject_id"]:
        if k in mut_cols_l:
            cand.insert(0, mut_cols_l[k])

    # uniq preserve order
    return list(dict.fromkeys(cand))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build TP53 groups for BeatAML (Supplement).")
    ap.add_argument("--counts", default=str(RAW / "beataml_waves1to4_counts_dbgap.txt"))
    ap.add_argument("--mutations", default=str(RAW / "beataml_wes_wv1to4_mutations_dbgap.txt"))
    ap.add_argument("--clinical", default=str(RAW / "beataml_wv1to4_clinical.xlsx"))
    ap.add_argument("--out", default=str(OUT_GROUPS / "BEATAML.groups.tsv"))

    ap.add_argument("--tp53-gene-col", default="", help="Mutations gene column (auto if empty).")
    ap.add_argument(
        "--effect-col", default="", help="Mutations effect/class column (auto if empty)."
    )
    ap.add_argument("--id-col-mut", default="", help="Mutations ID column (auto if empty).")

    ap.add_argument(
        "--min-overlap", type=int, default=50, help="Fail if overlap < this (default: 50)."
    )
    ap.add_argument(
        "--no-clinical-filter",
        action="store_true",
        help="Do not filter to clinical analysis cohort.",
    )
    args = ap.parse_args()

    counts_path = Path(args.counts)
    muts_path = Path(args.mutations)
    clin_path = Path(args.clinical)
    out_path = Path(args.out)

    if not counts_path.exists():
        _die(f"[beataml_groups] counts missing: {counts_path}")
    if not muts_path.exists():
        _die(f"[beataml_groups] mutations missing: {muts_path}")

    # counts IDs
    counts_ids_list = _extract_sample_ids_from_counts(counts_path)
    counts_ids = set(counts_ids_list)
    _info(f"counts: n_samples={len(counts_ids)} head={sorted(list(counts_ids))[:5]}")

    # load mutations/clinical
    mut = _read_table(muts_path)
    _info(f"mutations: shape={mut.shape}")

    clin = None
    br = None
    key_level = "subject"
    rnaseq_to_subj: dict[str, str] = {}
    dnaseq_to_subj: dict[str, str] = {}
    eligible_subj: set[str] = set()

    if clin_path.exists():
        clin = _read_table(clin_path)
        _info(f"clinical: shape={clin.shape}")
        br = _detect_clinical_bridge(clin)
        _info(
            "clinical bridge: "
            f"subject={br.subj_col!r}, rnaseq={br.rnaseq_col!r}, "
            f"dnaseq={br.dnaseq_col!r}, tp53={br.tp53_col!r}, "
            f"used={br.flag_used_col!r}, rna={br.flag_rna_col!r}, "
            f"exome={br.flag_exome_col!r}"
        )
        key_level, rnaseq_to_subj, dnaseq_to_subj = _choose_counts_key(counts_ids, clin, br)
        eligible_subj = _eligible_subjects(clin, br) if (not args.no_clinical_filter) else set()
        if eligible_subj:
            _info(f"clinical cohort: eligible_subjects={len(eligible_subj)} (filters applied)")
        else:
            _info("clinical cohort: no filters applied (using all subjects)")
    else:
        _warn(f"clinical not found (OK): {clin_path}")

    # gene/effect columns
    gene_col = str(args.tp53_gene_col).strip()
    if not gene_col:
        # Prefer symbol-like columns first (BeatAML is often like this)
        gene_col = _pick_col(mut, ["symbol", "hugo_symbol", "Hugo_Symbol", "Gene", "gene"]) or ""

    if not gene_col:
        for c in mut.columns:
            cl = c.lower()
            if "symbol" in cl or "hugo" in cl or cl == "gene":
                gene_col = c
                break
    if not gene_col:
        _die("[beataml_groups] cannot detect gene column in mutations table (use --tp53-gene-col)")

    # If chosen gene_col yields no TP53 rows, try a fallback column if present.
    def _tp53_count(col: str) -> int:
        s = mut[col].astype(str).str.strip().str.upper()
        return int((s == "TP53").sum())

    tp53_n = _tp53_count(gene_col)
    if tp53_n == 0:
        for alt in ["symbol", "hugo_symbol", "Hugo_Symbol", "Gene", "gene"]:
            if alt in mut.columns and alt != gene_col:
                n2 = _tp53_count(alt)
                if n2 > 0:
                    _warn(f"gene_col {gene_col!r} has TP53=0; switching to {alt!r} (TP53={n2})")
                    gene_col = alt
                    break

    eff_col = str(args.effect_col).strip()
    if not eff_col:
        eff_col = (
            _pick_col(
                mut,
                [
                    "variant_classification",
                    "effect",
                    "consequence",
                    "annotation",
                    "Variant_Classification",
                ],
            )
            or ""
        )
    if not eff_col:
        for c in mut.columns:
            cl = c.lower()
            if any(k in cl for k in ["class", "effect", "conseq", "annotation"]):
                eff_col = c
                break
    if not eff_col:
        _die("[beataml_groups] cannot detect effect/consequence column (use --effect-col)")

    _info(f"mutations: gene_col={gene_col!r} effect_col={eff_col!r}")

    # pick mutation ID column (best overlap to counts OR bridge-able via clinical)
    mut_id_col = str(args.id_col_mut).strip()
    if mut_id_col:
        if mut_id_col not in mut.columns:
            _die(f"[beataml_groups] --id-col-mut not found: {mut_id_col}")
    else:
        cands = _detect_mut_id_candidates(mut)
        if not cands:
            _die("[beataml_groups] cannot find any plausible ID column in mutations table")
        best = ("", -1)
        for c in cands:
            ids = {_clean_id(x) for x in mut[c].astype(str).tolist() if _clean_id(x)}
            ov = _score_overlap(ids, counts_ids)
            # bonus if can bridge via clinical dnaseq->subject or subject->rnaseq
            if clin is not None and dnaseq_to_subj:
                ov2 = _score_overlap(ids, set(dnaseq_to_subj.keys()))
                ov = max(ov, ov2)
            best = (c, ov) if ov > best[1] else best
        mut_id_col = best[0]
    if not mut_id_col:
        _die("[beataml_groups] failed to choose mutations ID column")

    _info(f"mutations: id_col={mut_id_col!r}")

    # TP53 protein-altering set (raw IDs at whatever level)
    gene_up = mut[gene_col].astype(str).str.strip().str.upper()
    eff_raw = mut[eff_col].astype(str)
    mask_tp53 = gene_up.eq("TP53")
    mask_pa = eff_raw.map(_is_protein_altering)

    n_tp53 = int(mask_tp53.sum())
    n_tp53_pa = int((mask_tp53 & mask_pa).sum())
    _info(f"TP53 rows={n_tp53} protein_altering={n_tp53_pa}")
    if n_tp53 == 0:
        _die("[beataml_groups] no TP53 rows found in mutations")
    if n_tp53_pa == 0:
        tp53_eff = (
            eff_raw[mask_tp53].astype(str).map(_canon_effect).value_counts().head(20).to_dict()
        )
        _die(
            "[beataml_groups] TP53 present but none protein-altering by rules. "
            f"TP53 effects top20={tp53_eff}"
        )

    tp53_mut_ids_raw = {
        _clean_id(x)
        for x in mut.loc[mask_tp53 & mask_pa, mut_id_col].astype(str).tolist()
        if _clean_id(x)
    }

    # Map mutated set onto counts key level
    tp53_mut_counts: set[str] = set()
    tp53_mut_subjects: set[str] = set()

    if clin is None:
        # must be directly comparable
        tp53_mut_counts = tp53_mut_ids_raw & counts_ids
        if not tp53_mut_counts:
            _die(
                "[beataml_groups] no TP53_mut IDs overlap counts IDs "
                "(no clinical bridge available). "
                "Provide clinical file or specify a matching --id-col-mut."
            )

        key_level = "rnaseq"  # treat counts IDs as output samples
    else:
        assert br is not None

        if key_level == "rnaseq":
            # case 1: raw IDs already match rnaseq counts IDs
            direct = tp53_mut_ids_raw & counts_ids
            if direct:
                tp53_mut_counts = direct
            else:
                # case 2: raw IDs are dnaseq sample IDs -> subject -> rnaseq sample
                subj_from_dna = {dnaseq_to_subj.get(x, "") for x in tp53_mut_ids_raw}
                subj_from_dna.discard("")
                tp53_mut_subjects = subj_from_dna

                if tp53_mut_subjects and rnaseq_to_subj:
                    inv = {}
                    for rna_id, sid in rnaseq_to_subj.items():
                        inv.setdefault(sid, set()).add(rna_id)
                    for sid in tp53_mut_subjects:
                        for rna_id in inv.get(sid, set()):
                            if rna_id in counts_ids:
                                tp53_mut_counts.add(rna_id)

            if len(tp53_mut_counts) < int(args.min_overlap):
                _warn(
                    f"TP53_mut counts overlap is small (n={len(tp53_mut_counts)}). "
                    "If this seems wrong, inspect ID columns / clinical bridge."
                )

        else:  # key_level == "subject"
            # raw may be subject IDs directly
            subj_direct = set(tp53_mut_ids_raw) & {
                _clean_id(x) for x in clin[br.subj_col].astype(str).tolist()
            }
            if subj_direct:
                tp53_mut_subjects = subj_direct
            else:
                # raw may be dnaseq sample IDs -> subject
                tp53_mut_subjects = {dnaseq_to_subj.get(x, "") for x in tp53_mut_ids_raw}
                tp53_mut_subjects.discard("")

            if not tp53_mut_subjects:
                _die("[beataml_groups] could not map TP53 mutations to clinical subjects")

    # clinical TP53 positivity by subject (QC / fallback)
    clin_tp53_pos: set[str] = set()
    if clin is not None and br is not None and br.tp53_col and br.tp53_col in clin.columns:
        subj = clin[br.subj_col].astype(str).map(_clean_id)
        tp53s = clin[br.tp53_col].astype(str)
        for sid, val in zip(subj, tp53s, strict=False):
            if sid and _clinical_tp53_positive(val):
                clin_tp53_pos.add(sid)
        _info(f"clinical TP53+: n_subjects={len(clin_tp53_pos)}")
    elif clin is not None:
        _warn("clinical TP53 column not found; clinical-only group will be empty")

    # define eligible output IDs
    out_samples: list[str] = []
    sample_to_subj: dict[str, str] = {}

    if clin is None:
        out_samples = counts_ids_list
    else:
        assert br is not None
        if key_level == "rnaseq":
            out_samples = counts_ids_list
            if rnaseq_to_subj:
                sample_to_subj = {s: rnaseq_to_subj.get(s, "") for s in out_samples}
        else:
            # counts IDs are subjects
            out_samples = counts_ids_list
            sample_to_subj = {s: s for s in out_samples}

    # apply clinical cohort filter (subject-based) if available
    if clin is not None and eligible_subj:
        kept = []
        for s in out_samples:
            sid = sample_to_subj.get(s, "")
            if sid and sid in eligible_subj:
                kept.append(s)
        _info(f"cohort filter: kept {len(kept)}/{len(out_samples)} samples")
        out_samples = kept

    if not out_samples:
        _die("[beataml_groups] no samples left after filtering")

    # assign groups (4-level)
    rows = []
    for s in out_samples:
        sid = sample_to_subj.get(s, "")
        if clin is None:
            grp = "TP53_mut_wes" if s in tp53_mut_counts else "TP53_wt_wes"
        else:
            if key_level == "rnaseq":
                wes_mut = s in tp53_mut_counts
            else:
                wes_mut = s in tp53_mut_subjects

            clin_pos = bool(sid and sid in clin_tp53_pos)

            if wes_mut:
                grp = "TP53_mut_wes"
            elif clin_pos:
                grp = "TP53_clinical_only"
            elif sid:
                grp = "TP53_wt_wes"
            else:
                grp = "TP53_missing"

        rows.append({"sample": s, "subject": sid, "group": grp})

    df = pd.DataFrame(rows)

    # summary
    vc = df["group"].value_counts().to_dict()
    _info(f"group counts: {vc}")

    # sanity overlap checks
    if clin is None:
        if len(tp53_mut_counts) < int(args.min_overlap):
            _warn(f"TP53_mut overlap with counts is small: n={len(tp53_mut_counts)}")
    else:
        if key_level == "rnaseq":
            _info(f"TP53_mut_wes (rnaseq samples) n={len(tp53_mut_counts)}")
        else:
            _info(f"TP53_mut_wes (subjects) n={len(tp53_mut_subjects)}")

    _ensure_dir(out_path.parent)
    df.to_csv(out_path, sep="\t", index=False)

    dbg = out_path.with_suffix(".debug.txt")
    dbg_lines = [
        "BeatAML TP53 groups debug",
        f"counts_path={counts_path}",
        f"mutations_path={muts_path}",
        f"clinical_path={clin_path if clin_path.exists() else ''}",
        f"counts_n={len(counts_ids)}",
        f"mutations_shape={mut.shape}",
        f"gene_col={gene_col}",
        f"effect_col={eff_col}",
        f"mut_id_col={mut_id_col}",
        f"counts_key_level={key_level}",
        f"clinical_filter_applied={bool(clin is not None and eligible_subj)}",
        f"TP53_rows={n_tp53}",
        f"TP53_protein_altering_rows={n_tp53_pa}",
        f"TP53_mut_raw_ids={len(tp53_mut_ids_raw)}",
    ]
    if clin is not None:
        dbg_lines += [
            f"clinical_subject_col={br.subj_col if br else ''}",
            f"clinical_rnaseq_col={br.rnaseq_col if br else ''}",
            f"clinical_dnaseq_col={br.dnaseq_col if br else ''}",
            f"clinical_tp53_col={br.tp53_col if br else ''}",
            f"clinical_tp53_pos_subjects={len(clin_tp53_pos)}",
        ]
    dbg_lines += [
        "group_counts=" + str(vc),
        "",
    ]
    dbg.write_text("\n".join(dbg_lines), encoding="utf-8")

    print("[beataml_groups] OK")
    print(f"  wrote: {out_path}")
    print(f"  debug: {dbg}")


if __name__ == "__main__":
    main()
