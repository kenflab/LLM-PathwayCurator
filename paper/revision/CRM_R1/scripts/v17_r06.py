"""Source-linked, rater-specific P4 reanalysis independent of P3 grading.

This evaluates the unchanged, structured candidate wording rated in P4.
It is not a benchmark of new LLM prose, a corrected audit, or biological truth.
No provider SDK, HTTP client, LLM pipeline, or P3 grading code is imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

SCHEMA = "CRM_R1_REVISION_v17_R06"
TABLE_READER_VERSION = "CRM_R1_P4_TSV_READER_R06_1"
REPO = Path(__file__).resolve().parents[4]
DEFAULT_POLICY = Path(__file__).resolve().parents[1] / "config/r06_p4_reuse_policy.json"
DEFAULT_BENCHMARK = "PANCAN_TP53_v1_HNSC_R1"
RATING_FIELDS = [
    "q1_statistical_support",
    "q2_external_evidence",
    "q3_overstatement",
    "confidence_1_to_5",
    "concise_rationale",
]
METHOD_LABELS = {
    "raw_pool": "All frozen candidates",
    "q_value_matched": "q-value matched",
    "stability_matched": "Stability matched",
    "legacy_full_audit": "Legacy full audit",
}


class ReuseBlocked(ValueError):
    """An input/provenance problem, never an unfavorable scientific result."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise ReuseBlocked(code, message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "DUPLICATE_JSON_KEY", f"Duplicate key: {key}")
            result[key] = value
        return result

    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    require(isinstance(value, dict), "BAD_JSON_ROOT", f"Expected JSON object: {path}")
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_table(path, frame):
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.17g")


class Inputs:
    def __init__(self, root, repo):
        self.root, self.repo = Path(root).resolve(), Path(repo).resolve()
        self.files = {}
        self.checks = []
        self.table_checks = []

    def add(self, path):
        path = Path(path).resolve()
        running_sources = {
            Path(__file__).resolve(),
            Path(__file__).with_name("65_revision_r06.py").resolve(),
            DEFAULT_POLICY.resolve(),
        }
        require(
            path.is_relative_to(self.root)
            or path.is_relative_to(self.repo)
            or path in running_sources,
            "PATH_OUTSIDE_ROOTS",
            f"Input is outside the data root and repository: {path}",
        )
        require(path.is_file(), "MISSING_INPUT", f"Missing input: {path}")
        digest = sha(path)
        previous = self.files.get(str(path))
        require(
            previous is None or previous["sha256"] == digest,
            "INPUT_CHANGED",
            f"Input changed during this run: {path}",
        )
        self.files[str(path)] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
        return digest

    def resolve(self, declared):
        """Rebase complete historical paths, never search by basename."""
        require(isinstance(declared, str) and declared, "BAD_PATH", "Empty declared path")
        path = Path(declared)
        require(".." not in path.parts, "BAD_PATH", f"Parent traversal: {declared}")
        if not path.is_absolute():
            target = self.repo / path if path.parts[0] == "paper" else self.root / path
        elif path.is_relative_to(self.root) or path.is_relative_to(self.repo):
            target = path
        elif "LLM-PathwayCurator" in path.parts and "paper" in path.parts:
            indexes = [i for i, part in enumerate(path.parts) if part == "LLM-PathwayCurator"]
            require(len(indexes) == 1, "AMBIGUOUS_PATH", f"Ambiguous repo path: {declared}")
            suffix = Path(*path.parts[indexes[0] + 1 :])
            require(
                suffix.parts and suffix.parts[0] == "paper",
                "BAD_PATH",
                f"Not a revision repository path: {declared}",
            )
            target = self.repo / suffix
        else:
            indexes = [i for i, part in enumerate(path.parts) if part == "CRM_R1"]
            require(len(indexes) == 1, "AMBIGUOUS_PATH", f"Cannot rebase: {declared}")
            target = self.root / Path(*path.parts[indexes[0] + 1 :])
        target = target.resolve()
        require(
            target.is_relative_to(self.root) or target.is_relative_to(self.repo),
            "PATH_OUTSIDE_ROOTS",
            f"Rebased path escapes its root: {declared}",
        )
        return target

    def record(self, manifest, section, label):
        item = manifest.get(section, {}).get(label)
        require(isinstance(item, dict), "MISSING_RECORD", f"Missing {section}/{label}")
        expected = item.get("sha256")
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            "BAD_DIGEST",
            f"Invalid SHA256: {section}/{label}",
        )
        path = self.resolve(item.get("path"))
        observed = self.add(path)
        self.checks.append(
            {
                "section": section,
                "record": label,
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matches": observed == expected,
            }
        )
        require(observed == expected, "HASH_MISMATCH", f"Frozen hash mismatch: {path}")
        return path

    def manifest(self, path):
        digest = self.add(path)
        companion = Path(path).with_suffix(".sha256")
        self.add(companion)
        lines = companion.read_text(encoding="utf-8").strip().splitlines()
        require(
            len(lines) == 1 and lines[0].split()[0] == digest,
            "MANIFEST_DIGEST_MISMATCH",
            f"Manifest digest mismatch: {path}",
        )
        return read_json(path)

    def table(self, path):
        """Reproduce the P4 pandas import without rewriting spreadsheet exports.

        Blank supplementary headers become pandas' positional Unnamed columns.
        All columns are retained. Named duplicates are rejected before pandas
        can choose/mangle them; ratings still have to reproduce the frozen lock.
        """
        digest = self.add(path)
        check = {
            "path": str(Path(path).resolve()),
            "sha256": digest,
            "reader_version": TABLE_READER_VERSION,
            "input_bytes_rewritten": False,
            "leading_blank_lines": 0,
            "raw_header": [],
            "unnamed_header_positions_1_based": [],
            "duplicate_named_headers": [],
            "pandas_columns": [],
            "status": "STARTED",
        }
        self.table_checks.append(check)
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            # pandas skips empty/space-only lines, but a tab-only line is a row.
            for line in handle:
                if line.strip() or "\t" in line:
                    break
                check["leading_blank_lines"] += 1
            else:
                line = ""
            try:
                header = next(csv.reader(chain([line], handle), delimiter="\t"), [])
            except csv.Error as error:
                check["status"] = "TABLE_PARSE_FAILED"
                raise ReuseBlocked(
                    "TABLE_PARSE_FAILED", f"Cannot read TSV header: {path}"
                ) from error
        check["raw_header"] = header
        check["unnamed_header_positions_1_based"] = [
            i + 1 for i, name in enumerate(header) if name == ""
        ]
        duplicates = sorted(name for name, count in Counter(header).items() if name and count > 1)
        check["duplicate_named_headers"] = duplicates
        check["status"] = (
            "EMPTY_TABLE" if not any(name.strip() for name in header) else "HEADER_READ"
        )
        require(
            any(name.strip() for name in header), "EMPTY_TABLE", f"Missing named TSV header: {path}"
        )
        if duplicates:
            check["status"] = "DUPLICATE_COLUMNS"
        require(
            not duplicates,
            "DUPLICATE_COLUMNS",
            f"Duplicate named TSV headers {duplicates}: {path}; no column selected automatically",
        )
        try:
            frame = pd.read_csv(
                path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig"
            )
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
            check["status"] = "TABLE_PARSE_FAILED"
            raise ReuseBlocked("TABLE_PARSE_FAILED", f"Cannot parse frozen TSV: {path}") from error
        check["pandas_columns"] = list(frame.columns)
        require(frame.columns.is_unique, "DUPLICATE_COLUMNS", f"Duplicate columns: {path}")
        matches = len(header) == len(frame.columns) and all(
            name == "" or name == frame.columns[i] for i, name in enumerate(header)
        )
        if not matches:
            check["status"] = "HEADER_PARSE_MISMATCH"
        require(
            matches, "HEADER_PARSE_MISMATCH", f"Named TSV headers changed during parsing: {path}"
        )
        check["rows"] = len(frame)
        check["unnamed_column_nonempty_counts"] = {
            str(frame.columns[i]): int(frame.iloc[:, i].ne("").sum())
            for i, name in enumerate(header)
            if name == ""
        }
        check["status"] = "PARSED_WITH_UNNAMED_COLUMNS" if "" in header else "PARSED"
        return frame

    def verify(self):
        for item in self.files.values():
            require(
                Path(item["path"]).is_file() and sha(item["path"]) == item["sha256"],
                "INPUT_CHANGED",
                f"Input changed during this run: {item['path']}",
            )


def fields(frame, required, name):
    require(set(required) <= set(frame), "BAD_TABLE_SCHEMA", f"{name} lacks {sorted(required)}")


def unique_ids(frame, column, name):
    fields(frame, [column], name)
    require(
        frame[column].ne("").all() and frame[column].is_unique,
        "DUPLICATE_OR_EMPTY_ID",
        f"{name}: empty or duplicate {column}",
    )


def booleans(values, name):
    parsed = values.astype(str).str.lower()
    require(
        parsed.isin({"true", "false", "1", "0"}).all(),
        "INVALID_MEMBERSHIP",
        f"Invalid boolean in {name}",
    )
    return parsed.isin({"true", "1"}).to_numpy()


def integers(values, name):
    numeric = pd.to_numeric(values, errors="raise")
    require(
        np.isfinite(numeric).all() and np.equal(numeric, np.floor(numeric)).all(),
        "INVALID_INTEGER",
        f"Noninteger {name}",
    )
    return numeric.astype(int)


def normalize_returned(frame, questions):
    """Only the whitespace/integer conversions used by the original P4 importer."""
    frame = frame.copy()
    for column in [*questions, "concise_rationale"]:
        frame[column] = frame[column].str.strip()
    frame["packet_order"] = integers(frame["packet_order"], "packet_order")
    frame["confidence_1_to_5"] = integers(frame["confidence_1_to_5"], "confidence")
    return frame


def validate_ratings(long, templates, returned, policy, packet):
    questions = policy["questions"]
    columns = ["rater_id", "review_id", "packet_order", *RATING_FIELDS]
    fields(long, columns, "Locked ratings")
    require(
        not long.duplicated(["rater_id", "review_id"]).any(),
        "DUPLICATE_RATING",
        "Duplicate rater/review pair",
    )
    raters = sorted(templates)
    require(
        len(raters) == policy["expected_raters"] and set(long.rater_id) == set(raters),
        "RATER_CENSUS_CHANGED",
        "Locked ratings and assigned raters differ",
    )
    require(set(returned) == set(raters), "RETURNED_CENSUS_CHANGED", "Missing original returns")
    long = long.copy()
    long["packet_order"] = integers(long.packet_order, "locked packet_order")
    long["confidence_1_to_5"] = integers(long.confidence_1_to_5, "locked confidence")
    for question, allowed in questions.items():
        require(
            long[question].isin(allowed).all(),
            "INVALID_LOCKED_CATEGORY",
            f"Invalid or missing locked {question}; no automatic recoding",
        )
    require(
        long.confidence_1_to_5.isin([1, 2, 3, 4, 5]).all()
        and long.concise_rationale.str.strip().ne("").all(),
        "INVALID_LOCKED_RATING",
        "Missing rationale or invalid confidence in locked ratings",
    )
    for rater in raters:
        template = templates[rater].copy()
        raw = returned[rater]
        fields(template, columns, f"Template {rater}")
        fields(raw, columns, f"Return {rater}")
        require(
            not (
                {
                    *policy["methods"].values(),
                    "claim_uid",
                    "claim_id",
                    "audit_status",
                    "status_full",
                    "method_membership",
                    "term_survival",
                    "context_status",
                }
                & set(raw)
            ),
            "METHOD_FIELDS_IN_RETURN",
            f"Method membership appeared in the return for {rater}",
        )
        require(
            template[RATING_FIELDS].eq("").all().all(),
            "TEMPLATE_NOT_BLANK",
            f"Original template is not blank: {rater}",
        )
        template["packet_order"] = integers(template.packet_order, "template packet_order")
        require(
            set(template.rater_id) == {rater} and set(raw.rater_id) == {rater},
            "RATER_ASSIGNMENT_CHANGED",
            f"Wrong rater assignment: {rater}",
        )
        for table, name in ((template, "template"), (raw, "return")):
            unique_ids(table, "review_id", f"{rater} {name}")
            require(
                set(table.review_id) == set(packet.review_id)
                and len(table) == policy["expected_claims"],
                "RATING_CENSUS_CHANGED",
                f"{rater}: {name} does not cover every frozen candidate",
            )
        raw = normalize_returned(raw, questions)
        observed = long.loc[long.rater_id.eq(rater), columns].sort_values("review_id")
        expected = raw[columns].sort_values("review_id")
        require(
            observed.reset_index(drop=True).equals(expected.reset_index(drop=True)),
            "ORIGINAL_RETURN_DIFFERS",
            f"Locked ratings no longer reproduce the original returned fields: {rater}",
        )
        assignment = ["rater_id", "review_id", "packet_order"]
        require(
            observed[assignment]
            .reset_index(drop=True)
            .equals(template[assignment].sort_values("review_id").reset_index(drop=True)),
            "TEMPLATE_ASSIGNMENT_CHANGED",
            f"Original assignment differs: {rater}",
        )
        mapped = observed.merge(packet[["review_id", "packet_order"]], on="review_id")
        require(
            np.array_equal(
                mapped.packet_order_x.to_numpy(),
                integers(mapped.packet_order_y, "packet order").to_numpy(),
            ),
            "PACKET_ORDER_CHANGED",
            f"Frozen packet order differs: {rater}",
        )
    return long, raters


def join_candidates(claims, membership, sampling, packet, policy):
    n = policy["expected_claims"]
    for frame, name in ((claims, "claims"), (membership, "membership"), (sampling, "sampling")):
        unique_ids(frame, "claim_uid", name)
        require(len(frame) == n, "CANDIDATE_CENSUS_CHANGED", f"{name}: expected {n} rows")
    unique_ids(sampling, "review_id", "sampling")
    unique_ids(packet, "review_id", "packet")
    require(len(packet) == n, "PACKET_CENSUS_CHANGED", f"Packet must have {n} rows")
    require(
        set(claims.claim_uid) == set(membership.claim_uid) == set(sampling.claim_uid)
        and set(sampling.review_id) == set(packet.review_id),
        "ID_MAPPING_FAILED",
        "Frozen candidate and review ID sets differ; no inner-join exclusions are allowed",
    )
    shared = ["claim_text", "pathway_label", "direction", "statistic", "q_value"]
    fields(claims, shared + ["status_full"], "Frozen claims")
    fields(packet, shared + ["packet_order"], "Frozen packet")
    fields(membership, policy["methods"].values(), "Frozen memberships")
    fields(sampling, ["packet_order"], "Frozen sampling")
    selected_columns = ["claim_uid", *policy["methods"].values()]
    joined = sampling.merge(claims, on="claim_uid", validate="one_to_one").merge(
        membership[selected_columns], on="claim_uid", validate="one_to_one"
    )
    packet_copy = packet[["review_id", "packet_order", *shared]].rename(
        columns={column: "packet_" + column for column in ["packet_order", *shared]}
    )
    joined = joined.merge(packet_copy, on="review_id", validate="one_to_one")
    joined = joined.sort_values("review_id").reset_index(drop=True)
    checks = pd.DataFrame({"review_id": joined.review_id, "claim_uid": joined.claim_uid})
    for column in ["claim_text", "pathway_label", "direction"]:
        checks[column + "_matches"] = joined[column].eq(joined["packet_" + column])
    require(joined.claim_text.ne("").all(), "EMPTY_CLAIM_TEXT", "Empty original wording")
    for column in ["statistic", "q_value"]:
        first = pd.to_numeric(joined[column], errors="raise").to_numpy(float)
        second = pd.to_numeric(joined["packet_" + column], errors="raise").to_numpy(float)
        require(
            np.isfinite(first).all() and np.isfinite(second).all(),
            "INVALID_STATISTIC",
            f"Nonfinite {column}",
        )
        checks[column + "_matches"] = first == second
        joined[column] = first
    checks["packet_order_matches"] = integers(joined.packet_order, "sampling order") == integers(
        joined.packet_packet_order, "packet order"
    )
    checks["claim_text_sha256"] = joined.claim_text.map(text_sha)
    checks["packet_text_sha256"] = joined.packet_claim_text.map(text_sha)
    matches = [column for column in checks if column.endswith("_matches")]
    require(
        checks[matches].all().all(),
        "FROZEN_PACKET_DIFFERS",
        "Original packet wording, evidence fields, or order differ from the frozen P2 source",
    )
    require(
        joined.q_value.between(0, 1).all(),
        "INVALID_Q_VALUE",
        "q-values must lie in [0,1]",
    )
    for column in policy["methods"].values():
        joined[column] = booleans(joined[column], column)
    require(joined.raw_pool_selected.all(), "RAW_POOL_CHANGED", "Raw pool excludes candidates")
    k = int(joined.full_audit_selected.sum())
    require(
        int(joined.q_value_matched_selected.sum())
        == int(joined.stability_matched_selected.sum())
        == k,
        "MATCHED_COVERAGE_CHANGED",
        "Frozen comparator sizes do not match legacy full audit",
    )
    require(
        joined.status_full.isin(["PASS", "ABSTAIN", "FAIL"]).all()
        and np.array_equal(
            joined.status_full.eq("PASS").to_numpy(), joined.full_audit_selected.to_numpy()
        ),
        "LEGACY_STATUS_DIFFERS",
        "Legacy status and frozen full-audit membership differ",
    )
    return joined, checks


def load_frozen(inputs, benchmark, policy):
    p2 = inputs.root / "output/priority2" / benchmark
    p4 = inputs.root / "output/priority4" / benchmark
    p2_path = p2 / "metrics/priority2_freeze_manifest.json"
    packet_path = p4 / "packet_v1/priority4_packet_manifest.json"
    ratings_path = p4 / "ratings_lock_v1/priority4_ratings_lock_manifest.json"
    p2_manifest = inputs.manifest(p2_path)
    packet_manifest = inputs.manifest(packet_path)
    ratings_manifest = inputs.manifest(ratings_path)
    require(p2_manifest.get("status") == "FROZEN", "P2_NOT_FROZEN", "P2 freeze is absent")
    require(
        packet_manifest.get("status") == "PACKETS_FROZEN_RATINGS_NOT_STARTED"
        and packet_manifest.get("method_membership_disclosed") is False
        and packet_manifest.get("ratings_inspected") is False,
        "PACKET_LOCK_FLAGS_CHANGED",
        "Original packet lock/blinding flags differ",
    )
    require(
        ratings_manifest.get("status") == "P4_RATINGS_LOCKED_BEFORE_METHOD_UNBLINDING"
        and ratings_manifest.get("method_membership_read") is False,
        "RATINGS_LOCK_FLAGS_CHANGED",
        "Original ratings lock/blinding flags differ",
    )
    for manifest in (p2_manifest, packet_manifest, ratings_manifest):
        require(
            manifest.get("benchmark_id") == benchmark,
            "BENCHMARK_CHANGED",
            "Manifest benchmark IDs differ",
        )
    p2_paths = {
        label: inputs.record(p2_manifest, "outputs", label)
        for label in p2_manifest.get("outputs", {})
    }
    packet_paths = {
        label: inputs.record(packet_manifest, "outputs", label)
        for label in packet_manifest.get("outputs", {})
    }
    ratings_paths = {
        label: inputs.record(ratings_manifest, "outputs", label)
        for label in ratings_manifest.get("outputs", {})
    }
    for label, target in (
        ("p2_manifest", p2_path),
        ("p2_claims", p2_paths.get("claims")),
        ("p2_sampling_frame", p2_paths.get("sampling_frame")),
    ):
        require(target is not None, "MISSING_P2_OUTPUT", f"Missing P2 output for {label}")
        require(
            inputs.record(packet_manifest, "inputs", label) == target,
            "PACKET_SOURCE_CHANGED",
            f"Packet's original {label} reference differs",
        )
    require(
        inputs.record(ratings_manifest, "inputs", "packet_manifest") == packet_path,
        "RATINGS_PACKET_CHANGED",
        "Ratings refer to a different packet manifest",
    )
    frozen_protocol = read_json(inputs.record(ratings_manifest, "inputs", "protocol"))
    require(
        frozen_protocol.get("minimum_independent_raters") == policy["expected_raters"]
        and {key: set(value) for key, value in frozen_protocol.get("questions", {}).items()}
        == {key: set(value) for key, value in policy["questions"].items()},
        "RATING_PROTOCOL_CHANGED",
        "Original question categories or rater census differ",
    )
    templates, returned = {}, {}
    for label, path in packet_paths.items():
        if label.startswith("ratings_R"):
            frame = inputs.table(path)
            fields(frame, ["rater_id"], label)
            identifiers = set(frame.rater_id)
            require(len(identifiers) == 1, "BAD_TEMPLATE_RATER", f"Bad template: {label}")
            rater = next(iter(identifiers))
            require(rater not in templates, "DUPLICATE_TEMPLATE", f"Duplicate template: {rater}")
            templates[rater] = frame
            original = inputs.record(ratings_manifest, "inputs", "blank_template_" + rater)
            require(original == path, "TEMPLATE_REFERENCE_CHANGED", f"Template differs: {rater}")
    for label in ratings_manifest.get("inputs", {}):
        if label.startswith("returned_ratings_"):
            frame = inputs.table(inputs.record(ratings_manifest, "inputs", label))
            fields(frame, ["rater_id"], label)
            identifiers = set(frame.rater_id)
            require(len(identifiers) == 1, "BAD_RETURN_RATER", f"Bad return: {label}")
            rater = next(iter(identifiers))
            require(rater not in returned, "DUPLICATE_RETURN", f"Duplicate return: {rater}")
            returned[rater] = frame
    require(
        {"claims", "membership", "sampling_frame"} <= set(p2_paths)
        and "claims_packet" in packet_paths
        and "ratings_long_private" in ratings_paths,
        "MISSING_CORE_OUTPUT",
        "Missing claims, memberships, packet, sampling, or original long ratings",
    )
    packet = inputs.table(packet_paths["claims_packet"])
    candidates, checks = join_candidates(
        inputs.table(p2_paths["claims"]),
        inputs.table(p2_paths["membership"]),
        inputs.table(p2_paths["sampling_frame"]),
        packet,
        policy,
    )
    long, raters = validate_ratings(
        inputs.table(ratings_paths["ratings_long_private"]),
        templates,
        returned,
        policy,
        packet,
    )
    return candidates, checks, long, raters


def wilson(events, n, confidence=0.95):
    if not n:
        return None, None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = events / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_interval(values, confidence=0.95):
    valid = np.asarray(values)[np.isfinite(values)]
    if not len(valid):
        return None, None, 0, "UNDEFINED"
    if len(valid) < 0.95 * len(values):
        return None, None, len(valid), "TOO_MANY_UNDEFINED_RESAMPLES"
    if np.ptp(valid) <= 1e-12:
        return None, None, len(valid), "DEGENERATE_EMPIRICAL_BOOTSTRAP"
    alpha = (1 - confidence) / 2
    low, high = np.quantile(valid, [alpha, 1 - alpha])
    return float(low), float(high), len(valid), "AVAILABLE_DESCRIPTIVE"


def fleiss(values):
    values = np.asarray(values)
    n, raters = values.shape
    categories = sorted(set(values.ravel()))
    counts = np.stack([(values == category).sum(axis=1) for category in categories], axis=1)
    observed_by_item = (np.square(counts).sum(axis=1) - raters) / (raters * (raters - 1))
    marginal = counts.sum(axis=0) / (n * raters)
    expected = float(np.square(marginal).sum())
    point = (
        None
        if np.isclose(expected, 1.0)
        else (float(observed_by_item.mean()) - expected) / (1 - expected)
    )
    return point, counts, observed_by_item


def agreement_tables(long, ids, raters, policy, draws):
    summary, category_rows, pairs, confusion = [], [], [], []
    b, n = draws.shape
    for question, categories in policy["questions"].items():
        wide = long.pivot(index="review_id", columns="rater_id", values=question).reindex(
            index=ids, columns=raters
        )
        values = wide.to_numpy()
        point, counts, observed = fleiss(values)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_b = np.square(draws @ counts / (n * len(raters))).sum(axis=1)
            kappa_b = (draws @ observed / n - expected_b) / (1 - expected_b)
        kappa_b[expected_b >= 1 - 1e-12] = np.nan
        low, high, valid, status = bootstrap_interval(kappa_b, policy["confidence_level"])
        unanimous = np.all(values == values[:, :1], axis=1)
        summary.append(
            {
                "question": question,
                "n_claims": n,
                "n_raters": len(raters),
                "fleiss_kappa": point,
                "kappa_ci_low": low,
                "kappa_ci_high": high,
                "kappa_ci_status": status,
                "valid_bootstrap_replicates": valid,
                "bootstrap_replicates": b,
                "exact_unanimous_fraction": float(unanimous.mean()),
            }
        )
        for i, rater in enumerate(raters):
            for category in categories:
                category_rows.append(
                    {
                        "rater_id": rater,
                        "question": question,
                        "category": category,
                        "n_claims": n,
                        "count": int((values[:, i] == category).sum()),
                    }
                )
            for j in range(i + 1, len(raters)):
                second = raters[j]
                equal = values[:, i] == values[:, j]
                p_low, p_high, p_valid, p_status = bootstrap_interval(
                    draws @ equal.astype(float) / n, policy["confidence_level"]
                )
                pairs.append(
                    {
                        "question": question,
                        "rater_a": rater,
                        "rater_b": second,
                        "n_claims": n,
                        "exact_agreement": float(equal.mean()),
                        "ci_low": p_low,
                        "ci_high": p_high,
                        "ci_status": p_status,
                        "valid_bootstrap_replicates": p_valid,
                    }
                )
                for first_category in categories:
                    for second_category in categories:
                        confusion.append(
                            {
                                "question": question,
                                "rater_a": rater,
                                "rater_b": second,
                                "category_a": first_category,
                                "category_b": second_category,
                                "count": int(
                                    (
                                        (values[:, i] == first_category)
                                        & (values[:, j] == second_category)
                                    ).sum()
                                ),
                            }
                        )
    return {
        "interrater_agreement.tsv": pd.DataFrame(summary),
        "rater_category_counts.tsv": pd.DataFrame(category_rows),
        "pairwise_rater_agreement.tsv": pd.DataFrame(pairs),
        "pairwise_confusion.tsv": pd.DataFrame(confusion),
    }


def evaluate(candidates, long, raters, policy):
    n = len(candidates)
    rng = np.random.default_rng(policy["bootstrap_seed"])
    draws = rng.multinomial(n, np.full(n, 1 / n), size=policy["bootstrap_replicates"])
    ids = candidates.review_id.tolist()
    results = agreement_tables(long, ids, raters, policy, draws)
    summaries, differences, overlap = [], [], []
    masks = {
        method: candidates[column].to_numpy(bool) for method, column in policy["methods"].items()
    }
    for comparator in ("raw_pool", "q_value_matched", "stability_matched"):
        a, c = masks["legacy_full_audit"], masks[comparator]
        overlap.append(
            {
                "method_a": "legacy_full_audit",
                "method_b": comparator,
                "n_a": int(a.sum()),
                "n_b": int(c.sum()),
                "n_common": int((a & c).sum()),
                "n_a_only": int((a & ~c).sum()),
                "n_b_only": int((c & ~a).sum()),
                "coverage_matched": int(a.sum()) == int(c.sum()),
            }
        )
    for endpoint, specification in policy["endpoints"].items():
        wide = long.pivot(
            index="review_id", columns="rater_id", values=specification["question"]
        ).reindex(index=ids, columns=raters)
        event = wide.isin(specification["event_categories"]).to_numpy(float)
        unknown = wide.eq("UNCERTAIN").to_numpy(float)
        events = np.column_stack([event, event.mean(axis=1)])
        unknowns = np.column_stack([unknown, unknown.mean(axis=1)])
        estimates, replicates = {}, {}
        for method, mask in masks.items():
            selected = int(mask.sum())
            numerator = events[mask].sum(axis=0)
            unknown_numerator = unknowns[mask].sum(axis=0)
            denominator_b = draws @ mask.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                lower_b = (draws @ (events * mask[:, None])) / denominator_b[:, None]
            replicates[method] = lower_b
            estimates[method] = (
                np.full(events.shape[1], np.nan) if not selected else numerator / selected,
                np.full(events.shape[1], np.nan)
                if not selected
                else (numerator + unknown_numerator) / selected,
            )
            for i, rater in enumerate([*raters, "FIXED_RATER_MEAN"]):
                mean_row = i == len(raters)
                if mean_row:
                    low, high, valid, ci_status = bootstrap_interval(
                        lower_b[:, i], policy["confidence_level"]
                    )
                    ci_method = "joint_claim_bootstrap_fixed_raters_descriptive"
                else:
                    low, high = wilson(int(numerator[i]), selected, policy["confidence_level"])
                    valid, ci_status = 0, "AVAILABLE_DESCRIPTIVE" if selected else "UNDEFINED"
                    ci_method = "wilson_descriptive"
                multiplier = len(raters) if mean_row else 1
                fraction, upper = estimates[method][0][i], estimates[method][1][i]
                known = selected - unknown_numerator[: len(raters)]
                with np.errstate(divide="ignore", invalid="ignore"):
                    known_fractions = numerator[: len(raters)] / known
                known_fraction = (
                    float(known_fractions.mean())
                    if mean_row and np.isfinite(known_fractions).all()
                    else None
                    if mean_row
                    else float(known_fractions[i])
                    if np.isfinite(known_fractions[i])
                    else None
                )
                summaries.append(
                    {
                        "endpoint": endpoint,
                        "question": specification["question"],
                        "rater_id": rater,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "n_candidate_claims": n,
                        "n_selected_claims": selected,
                        "n_selected_ratings": selected * multiplier,
                        "n_event_ratings": int(round(numerator[i] * multiplier)),
                        "n_uncertain_ratings": int(round(unknown_numerator[i] * multiplier)),
                        "confirmed_fraction": float(fraction) if np.isfinite(fraction) else None,
                        "uncertainty_bound_low": float(fraction) if np.isfinite(fraction) else None,
                        "uncertainty_bound_high": float(upper) if np.isfinite(upper) else None,
                        "known_rating_fraction": known_fraction,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_method": ci_method,
                        "ci_status": ci_status,
                        "valid_bootstrap_replicates": valid,
                        "coverage": selected / n,
                        "comparison_role": "descriptive_full_census"
                        if method == "raw_pool"
                        else "coverage_matched",
                    }
                )
        for comparator in ("raw_pool", "q_value_matched", "stability_matched"):
            a, c = "legacy_full_audit", comparator
            delta_b = replicates[a] - replicates[c]
            for i, rater in enumerate([*raters, "FIXED_RATER_MEAN"]):
                point = estimates[a][0][i] - estimates[c][0][i]
                low, high, valid, status = bootstrap_interval(
                    delta_b[:, i], policy["confidence_level"]
                )
                bounds_low = estimates[a][0][i] - estimates[c][1][i]
                bounds_high = estimates[a][1][i] - estimates[c][0][i]
                differences.append(
                    {
                        "endpoint": endpoint,
                        "rater_id": rater,
                        "method_a": a,
                        "method_b": c,
                        "difference_confirmed_fraction": float(point)
                        if np.isfinite(point)
                        else None,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_status": status,
                        "valid_bootstrap_replicates": valid,
                        "uncertainty_identification_bound_low": float(bounds_low)
                        if np.isfinite(bounds_low)
                        else None,
                        "uncertainty_identification_bound_high": float(bounds_high)
                        if np.isfinite(bounds_high)
                        else None,
                        "ci_method": "joint_claim_bootstrap_shared_memberships_fixed_raters",
                        "coverage_matched": int(masks[a].sum()) == int(masks[c].sum()),
                        "analysis_role": policy["analysis_role"],
                    }
                )
    results["rater_method_endpoints.tsv"] = pd.DataFrame(summaries)
    results["paired_method_differences.tsv"] = pd.DataFrame(differences)
    results["method_overlap.tsv"] = pd.DataFrame(overlap)
    q_rows = []
    for method, mask in masks.items():
        selected = int(mask.sum())
        significant = int(
            (mask & candidates.q_value.le(policy["q_value_descriptive_threshold"])).sum()
        )
        low, high = wilson(significant, selected, policy["confidence_level"])
        q_rows.append(
            {
                "method": method,
                "n_selected": selected,
                "n_q_le_threshold": significant,
                "q_threshold": policy["q_value_descriptive_threshold"],
                "fraction": significant / selected if selected else None,
                "ci_low": low,
                "ci_high": high,
                "endpoint_scope": "enrichment_statistic_only_not_biological_correctness",
            }
        )
    results["descriptive_q_value_summary.tsv"] = pd.DataFrame(q_rows)
    return results


def render(outdir, table, raters):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, len(raters), figsize=(12, 4.7), sharey=True, layout="constrained")
    methods = list(METHOD_LABELS)
    colors = ["#777777", "#D55E00", "#009E73", "#0072B2"]
    for ax, rater in zip(np.atleast_1d(axes), raters, strict=True):
        rows = table.loc[
            table.endpoint.eq("confirmed_major_overstatement") & table.rater_id.eq(rater)
        ].set_index("method")
        for position, (method, color) in enumerate(zip(methods, colors, strict=True)):
            row = rows.loc[method]
            value = row.confirmed_fraction
            if pd.isna(value):
                ax.text(position, 0.04, "undefined\nK=0", ha="center", fontsize=8)
                continue
            ax.bar(position, value, color=color, width=0.65)
            upper = row.uncertainty_bound_high
            if upper > value:
                ax.bar(
                    position,
                    upper - value,
                    bottom=value,
                    color="none",
                    edgecolor=color,
                    hatch="////",
                    width=0.65,
                )
            if pd.notna(row.ci_low) and pd.notna(row.ci_high):
                ax.plot([position, position], [row.ci_low, row.ci_high], color="black", lw=1)
                ax.plot([position], [value], "o", color="black", ms=3)
            ax.text(
                position,
                min(1.13, max(upper, row.ci_high) + 0.025),
                f"n={int(row.n_selected_claims)}",
                ha="center",
                fontsize=8,
            )
        ax.set_title(rater)
        ax.set_xticks(range(4), ["All\ncandidates", "q-value", "Stability", "Legacy\nfull audit"])
        ax.set_ylim(0, 1.18)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction explicitly rated MAJOR_OVERSTATEMENT")
    fig.suptitle("Frozen P4 ratings: unchanged candidate wording; exploratory reanalysis")
    fig.legend(
        handles=[
            Patch(color="#777777", label="Confirmed major overstatement"),
            Patch(
                facecolor="none",
                edgecolor="#777777",
                hatch="////",
                label="UNCERTAIN range; all selected items retained",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        fontsize=8,
    )
    stem = outdir / "Fig_R06_P4_major_overstatement_by_rater"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)


def report(outdir, summary, tables=None):
    lines = [
        "# R06：既存P4評価と凍結済み比較法の再集計",
        "",
        f"状態：**{summary['status']}**",
        "",
        "モデル呼び出し0、追加専門家評価0。P3の文献採点は前提にしていません。",
        "",
        "対象はP4で評価された元の構造化候補文と、保存されたP2選択集合です。",
        "原文と評価を変更せず、事後的・探索的な比較として集計します。",
        "全候補は選択しない比較法であり、新規の未監査LLM自由文ではありません。",
        "旧full auditを修正版の監査に置き換えたり、同じ名称で混同したりしません。",
        "",
        "R06.1のTSV読込は、旧P4と同じpandasの無名補足列処理を使用します。",
        "原本は書き換えず、採点8列は元の名前で固定台帳と照合します。",
        "見出しのある列が重複した場合は、列を自動選択せず停止します。",
        "table_read_checks.private.jsonに実際のヘッダーと読込状態を記録します。",
        "",
        "評価者は別々に表示します。等重み平均も、この3名を固定した記述値です。",
        "150評価を150の独立候補として扱いません。多数決は正解に使いません。",
        "UNCERTAINは選択候補の分母に残し、該当数と部分情報による上下限を出します。",
        "confirmed_fractionは、明示的にそのカテゴリーを付けられた割合です。",
        "UNCERTAINを正しい・安全と分類した値ではありません。",
        "",
        "個別割合のWilson CIは、観測されたカテゴリー割合の記述的な区間です。",
        "方法間差と一致度は、同じ候補IDを全評価者・比較法で一緒に再抽出します。",
        "候補単位のbootstrapでは経路間の重なりをモデル化していません。",
        "新しい患者・コホート・評価者への一般化を示す区間ではありません。",
        "bootstrap分布が一様な場合はCI未定義とし、不確実性0とは表示しません。",
        "有意差検定、優越性宣言、生物学的正確性の推定は行いません。",
        "",
    ]
    if tables is None:
        lines.extend(
            [
                f"阻害要因：{summary.get('blocker_code')}",
                "",
                summary.get("reason", ""),
                "",
                "性能表・図は推定していません。原本はそのまま保存してください。",
                "先にmanifestの参照先・ハッシュ・割当・原文の不一致を確認します。",
                "この状態を理由に、再評価依頼やカテゴリーの付け替えは行いません。",
            ]
        )
    else:
        lines.extend(
            [
                f"対応確認済み：{summary['candidate_claims']}候補、"
                f"{summary['raters']}評価者、{summary['ratings']}評価。",
                f"旧full auditと比較法の選択数K：{summary['legacy_full_audit_selected']}。",
                "",
                "Fleiss κ：",
            ]
        )
        for row in tables["interrater_agreement.tsv"].to_dict("records"):
            value = row["fleiss_kappa"]
            display = "未定義" if pd.isna(value) else f"{value:.4f}"
            lines.append(f"- {row['question']}: {display}; CI: {row['kappa_ci_status']}.")
        lines.extend(
            [
                "",
                "主な出力：",
                "- linkage_checks.private.tsv：原文ハッシュ、IDと根拠数値の対応。",
                "- ratings_linked.private.tsv：元の個別評価と保存済み選択集合。",
                "- rater_method_endpoints.tsv：カテゴリー割合、UNCERTAINの数・上下限・CI。",
                "- paired_method_differences.tsv：同じ候補を共有する旧full auditと比較法の差。",
                "- rater_category_counts.tsv / pairwise_confusion.tsv：カテゴリーの使用分布。",
                "- descriptive_q_value_summary.tsv：統計的有意性のみ。生物学的正解ではありません。",
                "",
                "保存されたlock情報・packet・返却評価の対応を確認した解析です。",
                "評価者が保存資料以外で何を読んだかや、評価内容の真偽は認証していません。",
                "全ての上流実行履歴やP3の採点・retrieval入力を再検証したわけではありません。",
                "",
                "低いP4一致度と、不利だったP2Bのreplicationは、そのまま所見として残します。",
                "この結果は修正版の監査の独立検証ではありません。",
                "査読対応には、評価者別比較の実際の結果と残る証拠不足を確認して使います。",
            ]
        )
    (outdir / "READOUT_JA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    data_root,
    repo=REPO,
    benchmark=DEFAULT_BENCHMARK,
    policy_path=DEFAULT_POLICY,
    outdir=None,
    skip_plot=False,
):
    root, repo = Path(data_root).expanduser().resolve(), Path(repo).expanduser().resolve()
    require(
        (root / "input").is_dir() and (root / "output").is_dir(),
        "BAD_DATA_ROOT",
        "Use the existing CRM_R1 directory containing input/ and output/",
    )
    require(not root.is_relative_to(repo), "DATA_INSIDE_GIT", "Data root must stay outside Git")
    require(
        re.fullmatch(r"[A-Za-z0-9_.-]+", benchmark) is not None,
        "BAD_BENCHMARK",
        "Invalid benchmark identifier",
    )
    parent = root / "output/revision_v17"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = parent / f"r06_{stamp}" if outdir is None else Path(outdir).expanduser().resolve()
    require(
        output.resolve().is_relative_to(parent.resolve()) and output.resolve() != parent.resolve(),
        "BAD_OUTPUT_ROOT",
        "New output must be a child of CRM_R1/output/revision_v17/",
    )
    require(not output.exists(), "OUTPUT_EXISTS", f"Output is immutable: {output}")
    output.mkdir(parents=True)
    inputs = Inputs(root, repo)
    summary = {
        "schema": SCHEMA,
        "status": "STARTED",
        "outdir": str(output),
        "benchmark": benchmark,
        "model_calls": 0,
        "new_expert_ratings": 0,
        "p3_grading_required": False,
        "candidate_wording_changed": False,
        "biological_accuracy_estimated": False,
        "revised_audit_performance_estimated": False,
        "independent_validation_performed": False,
        "new_model_comparison_performed": False,
        "table_reader_version": TABLE_READER_VERSION,
    }
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    summary["git_head"] = result.stdout.strip() if result.returncode == 0 else None
    write_json(output / "STARTED.json", summary)
    tables = None
    try:
        policy_path = Path(policy_path).resolve()
        inputs.add(policy_path)
        policy = read_json(policy_path)
        require(
            policy.get("schema") == "CRM_R1_P4_REUSE_POLICY_R06"
            and policy.get("model_requests_allowed") == 0
            and policy.get("majority_vote_used") is False
            and policy.get("candidate_wording_changed") is False,
            "POLICY_CHANGED",
            "Use the fixed R06 zero-model, original-wording policy",
        )
        sources = [
            Path(__file__),
            Path(__file__).with_name("65_revision_r06.py"),
            policy_path,
        ]
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for path in sources:
            inputs.add(path)
            shutil.copy2(path, snapshot / path.name)
        candidates, checks, long, raters = load_frozen(inputs, benchmark, policy)
        inputs.verify()
        tables = evaluate(candidates, long, raters, policy)
        inputs.verify()
        write_table(output / "linkage_checks.private.tsv", checks)
        linked = long.merge(
            candidates[["review_id", "claim_uid", "claim_text", *policy["methods"].values()]],
            on="review_id",
            validate="many_to_one",
        )
        write_table(output / "ratings_linked.private.tsv", linked)
        for name, table in tables.items():
            write_table(output / name, table)
        if not skip_plot:
            render(output, tables["rater_method_endpoints.tsv"], raters)
        inputs.verify()
        summary.update(
            status="COMPLETE_EXPLORATORY_P4_COMPARISON",
            candidate_claims=len(candidates),
            raters=len(raters),
            ratings=len(long),
            legacy_full_audit_selected=int(candidates.full_audit_selected.sum()),
            input_hashes_unchanged=True,
            source_linkage_verified=True,
            descriptive_p4_comparisons_estimated=True,
            bootstrap_replicates=policy["bootstrap_replicates"],
            bootstrap_seed=policy["bootstrap_seed"],
            analysis_role=policy["analysis_role"],
            text_origin="unchanged_frozen_structured_P2_wording_in_original_P4_packet",
            naive_llm_prose_comparison_performed=False,
            next_step="review_rater_specific_comparisons_and_remaining_reviewer_evidence_gaps",
        )
    except ReuseBlocked as error:
        try:
            inputs.verify()
            require(tables is None, "INPUT_CHANGED", "Input/provenance changed after estimation")
        except ReuseBlocked as changed:
            write_json(
                output / "FAILED.json",
                summary
                | {
                    "status": "FAILED_INPUT_CHANGED",
                    "error": str(changed),
                },
            )
            raise
        summary.update(
            status="P4_REUSE_BLOCKED",
            blocker_code=error.code,
            reason=str(error),
            source_linkage_verified=False,
            input_hashes_unchanged=True,
            performance_estimated=False,
            descriptive_p4_comparisons_estimated=False,
            next_step="inspect_frozen_manifest_and_original_packet_mapping",
        )
    except Exception as error:
        write_json(output / "FAILED.json", summary | {"error": str(error)})
        raise
    inputs.verify()
    write_json(output / "INPUT_MANIFEST.private.json", {"files": list(inputs.files.values())})
    write_table(output / "manifest_checks.private.tsv", pd.DataFrame(inputs.checks))
    write_json(
        output / "table_read_checks.private.json",
        {"reader_version": TABLE_READER_VERSION, "tables": inputs.table_checks},
    )
    report(output, summary, tables)
    archive = output.with_suffix(".zip")
    summary["results_archive"] = str(archive)
    write_json(output / "summary.json", summary)
    records = {
        str(path.relative_to(output)): {"sha256": sha(path), "size_bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write_json(output / "OUTPUT_MANIFEST.json", {"schema": SCHEMA, "files": records})
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                handle.write(path, arcname=str(path.relative_to(output.parent)))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CRM_R1_DATA_ROOT"))
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.data_root is None:
        parser.error("Set CRM_R1_DATA_ROOT or pass --data-root")
    try:
        summary = run(
            args.data_root,
            repo=args.repo,
            benchmark=args.benchmark,
            outdir=args.outdir,
            skip_plot=args.skip_plot,
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"[R06 STOP] {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    # Zero means the diagnostic completed, not that reuse or publication is certified.
    # A recorded input block still allows the independently tested code to be published.
    return 0
