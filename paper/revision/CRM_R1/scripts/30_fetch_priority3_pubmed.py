#!/usr/bin/env python3
"""Freeze PubMed searches and records for all Priority 2 claims before grading."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
DEFAULT_P2_CONFIG = CRM_DIR / "config" / "priorities2_5_protocol.json"
DEFAULT_P3_CONFIG = CRM_DIR / "config" / "priority3_protocol.json"
DEFAULT_P2_CHECKER = Path(__file__).resolve().with_name("21_check_priority2_freeze.py")
RECORD_COLUMNS = [
    "pmid",
    "title",
    "abstract",
    "abstract_available",
    "journal",
    "publication_year",
    "authors",
    "publication_types",
    "doi",
    "pmcid",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def require_clean_tracked_worktree() -> None:
    status = git_value("status", "--porcelain", "--untracked-files=no")
    require(not status, f"Commit V9.1 and leave tracked files clean before retrieval: {status}")


def validate_iso_date(value: str, *, name: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be YYYY-MM-DD: {value}") from error
    require(parsed <= date.today(), f"{name} cannot be in the future: {value}")
    return parsed.isoformat()


def run_p2_gate(*, checker: Path, data_root: Path, config: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(checker), "--data-root", str(data_root), "--config", str(config)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    print(output, end="")
    require(result.returncode == 0, "Priority 2 freeze gate failed")
    require("[GO] P3 evidence retrieval" in output, "Priority 2 checker did not release P3")


def xml_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def tagged_clause(terms: Iterable[str]) -> str:
    cleaned = []
    for term in terms:
        value = normalize_space(term).replace('"', "")
        if value:
            cleaned.append(f'"{value}"[Title/Abstract]')
    require(bool(cleaned), "Query term list is empty")
    return "(" + " OR ".join(dict.fromkeys(cleaned)) + ")"


def default_pathway_term(entity: str) -> str:
    value = str(entity).strip().removeprefix("HALLMARK_")
    value = value.replace("_", " ")
    value = re.sub(r"\b(?:V[12]|UP|DN)\b", "", value)
    return normalize_space(value).lower()


def pathway_terms(entity: str, protocol: dict[str, Any]) -> list[str]:
    mapped = protocol.get("pathway_synonyms", {}).get(str(entity), [])
    terms = [default_pathway_term(entity), *[str(value) for value in mapped]]
    return list(dict.fromkeys(normalize_space(value) for value in terms if normalize_space(value)))


def exclusion_clause(protocol: dict[str, Any]) -> str:
    types = [str(value).strip() for value in protocol["exclude_publication_types"]]
    return "NOT (" + " OR ".join(f'"{value}"[Publication Type]' for value in types) + ")"


def build_queries(entity: str, protocol: dict[str, Any]) -> dict[str, str]:
    hnsc = tagged_clause(protocol["hnsc_terms"])
    tp53 = tagged_clause(protocol["tp53_terms"])
    pathway = tagged_clause(pathway_terms(entity, protocol))
    exclude = exclusion_clause(protocol)
    return {
        "direct_hnsc_tp53_pathway": f"{hnsc} AND {tp53} AND {pathway} AND {exclude}",
        "context_hnsc_pathway": f"{hnsc} AND {pathway} AND {exclude}",
        "perturbation_tp53_pathway": f"{tp53} AND {pathway} AND {exclude}",
    }


class NCBIClient:
    """Small rate-limited E-utilities client that never persists an API key."""

    def __init__(
        self,
        *,
        base_url: str,
        tool: str,
        email: str,
        api_key: str = "",
        ssl_context: Any | None = None,
        requester: Callable[[urllib.request.Request, float], bytes] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tool = tool
        self.email = email
        self.api_key = api_key
        self.ssl_context = ssl_context
        self.interval_seconds = 0.11 if api_key else 0.36
        self.last_request = 0.0
        self.requester = requester or self._default_requester

    def _default_requester(self, request: urllib.request.Request, timeout: float) -> bytes:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=self.ssl_context,
        ) as response:
            return response.read()

    def request(self, endpoint: str, parameters: dict[str, Any]) -> bytes:
        payload = {
            **parameters,
            "tool": self.tool,
            "email": self.email,
        }
        if self.api_key:
            payload["api_key"] = self.api_key
        url = f"{self.base_url}/{endpoint}?{urllib.parse.urlencode(payload, doseq=True)}"
        wait = self.interval_seconds - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": f"{self.tool}/1.0 ({self.email})"},
            method="GET",
        )
        error: Exception | None = None
        for attempt in range(4):
            try:
                body = self.requester(request, 120.0)
                self.last_request = time.monotonic()
                return body
            except (urllib.error.URLError, TimeoutError) as caught:
                error = caught
                reason = getattr(caught, "reason", None)
                if isinstance(reason, ssl.SSLCertVerificationError):
                    raise RuntimeError(
                        "NCBI TLS certificate verification failed. On a managed macOS host, "
                        "install truststore and rerun with --use-system-trust; do not disable "
                        "certificate verification."
                    ) from caught
                if attempt == 3:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"NCBI request failed after retries: {endpoint}") from error

    def search(
        self,
        *,
        query: str,
        retmax: int,
        sort: str,
        publication_cutoff: str,
    ) -> dict[str, Any]:
        cutoff = publication_cutoff.replace("-", "/")
        body = self.request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": retmax,
                "sort": sort,
                "datetype": "pdat",
                "maxdate": cutoff,
            },
        )
        value = json.loads(body.decode("utf-8"))
        require("esearchresult" in value, "Malformed ESearch response")
        return value

    def fetch(self, pmids: list[str]) -> bytes:
        require(bool(pmids), "EFetch PMID list is empty")
        return self.request(
            "efetch.fcgi",
            {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        )


def build_tls_context(*, use_system_trust: bool) -> tuple[Any | None, dict[str, Any]]:
    """Return an explicit native trust context only when the caller requests it."""
    if not use_system_trust:
        verify_paths = ssl.get_default_verify_paths()
        return None, {
            "mode": "python_default",
            "openssl_cafile": verify_paths.cafile or "",
            "ssl_cert_file_env_supplied": bool(os.environ.get("SSL_CERT_FILE", "").strip()),
        }
    try:
        import truststore
    except ImportError as error:
        raise RuntimeError(
            "--use-system-trust requires: python -m pip install 'truststore==0.10.4'"
        ) from error
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    version = importlib.metadata.version("truststore")
    return context, {
        "mode": "macos_native_system_trust",
        "implementation": "truststore",
        "truststore_version": version,
    }


def parse_pubmed_xml(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    rows: list[dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        article_node = citation.find("Article") if citation is not None else None
        pmid = xml_text(citation.find("PMID") if citation is not None else None)
        if not pmid or article_node is None:
            continue
        title = normalize_space(xml_text(article_node.find("ArticleTitle")))
        abstract_parts = []
        for abstract_node in article_node.findall("Abstract/AbstractText"):
            text = normalize_space(xml_text(abstract_node))
            label = normalize_space(abstract_node.attrib.get("Label", ""))
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts)
        journal = normalize_space(xml_text(article_node.find("Journal/Title")))
        pub_date = article_node.find("Journal/JournalIssue/PubDate")
        year = xml_text(pub_date.find("Year") if pub_date is not None else None)
        if not year and pub_date is not None:
            match = re.search(r"\b(19|20)\d{2}\b", xml_text(pub_date.find("MedlineDate")))
            year = match.group(0) if match else ""
        authors = []
        for author in article_node.findall("AuthorList/Author"):
            collective = xml_text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
                continue
            last = xml_text(author.find("LastName"))
            initials = xml_text(author.find("Initials"))
            name = normalize_space(f"{last} {initials}")
            if name:
                authors.append(name)
        publication_types = [
            normalize_space(xml_text(node))
            for node in article_node.findall("PublicationTypeList/PublicationType")
            if normalize_space(xml_text(node))
        ]
        doi = ""
        pmcid = ""
        for article_id in article.findall("PubmedData/ArticleIdList/ArticleId"):
            kind = str(article_id.attrib.get("IdType", "")).lower()
            if kind == "doi":
                doi = normalize_space(xml_text(article_id))
            if kind == "pmc":
                pmcid = normalize_space(xml_text(article_id))
        rows.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "abstract_available": bool(abstract),
                "journal": journal,
                "publication_year": year,
                "authors": "; ".join(authors),
                "publication_types": "; ".join(publication_types),
                "doi": doi,
                "pmcid": pmcid,
            }
        )
    frame = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates("pmid").sort_values("pmid").reset_index(drop=True)
    return frame


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def make_screening_template(
    *,
    links: pd.DataFrame,
    records: pd.DataFrame,
    claims_blinded: pd.DataFrame,
) -> pd.DataFrame:
    if links.empty:
        base = pd.DataFrame(columns=["review_id", "pmid", "query_families", "best_query_rank"])
    else:
        base = (
            links.groupby(["review_id", "pmid"], as_index=False)
            .agg(
                query_families=("query_family", lambda x: ";".join(sorted(set(x)))),
                best_query_rank=("query_rank", "min"),
            )
            .sort_values(["review_id", "best_query_rank", "pmid"])
        )
    template = base.merge(records, on="pmid", how="left", validate="many_to_one")
    template = template.merge(
        claims_blinded[["review_id", "claim_text", "pathway_label", "direction"]],
        on="review_id",
        how="left",
        validate="many_to_one",
    )
    template.insert(0, "screening_id", [f"P3S{index:05d}" for index in range(1, len(template) + 1)])
    template["eligible"] = ""
    template["exclusion_reason"] = ""
    template["evidence_grade"] = ""
    template["direction_match"] = ""
    template["context_match"] = ""
    template["study_design"] = ""
    template["data_overlap"] = ""
    template["contradiction"] = ""
    template["supporting_note"] = ""
    template["curator_id"] = ""
    ordered = [
        "screening_id",
        "review_id",
        "claim_text",
        "pathway_label",
        "direction",
        "pmid",
        "query_families",
        "best_query_rank",
        "title",
        "publication_year",
        "journal",
        "publication_types",
        "doi",
        "pmcid",
        "abstract_available",
        "abstract",
        "eligible",
        "exclusion_reason",
        "evidence_grade",
        "direction_match",
        "context_match",
        "study_design",
        "data_overlap",
        "contradiction",
        "supporting_note",
        "curator_id",
    ]
    for column in ordered:
        if column not in template:
            template[column] = ""
    return template[ordered]


def grading_instructions(protocol: dict[str, Any]) -> str:
    definitions = protocol["grade_definitions"]
    fields = protocol["record_grading_fields"]
    grade_lines = "\n".join(f"- `{key}`: {value}" for key, value in definitions.items())
    field_lines = "\n".join(
        f"- `{key}`: {', '.join(f'`{choice}`' for choice in choices)}"
        for key, choices in fields.items()
    )
    return f"""# Priority 3 blinded evidence-grading instructions

Grade the frozen record rows without consulting audit status, method membership, empirical
stability, or context-review outputs. Do not run additional searches and do not replace, add, or
delete PubMed records. Work from `record_screening_template.private.tsv` in `screening_id` order.
Abstract text is private and must not be redistributed.

For every row, first decide eligibility. Ineligible rows receive `evidence_grade=EXCLUDE` and one
exclusion reason. Eligible rows receive one record-level grade from E1 through E4. E0 is never a
record-level grade; it is derived at claim level only when the frozen retrieval contains no eligible
supporting record. A direction-mismatched E3/E4 record may document contradiction but cannot count
as primary independent support.

## Evidence grades

{grade_lines}

## Allowed record-level values

{field_lines}

The primary independent-support endpoint requires a claim-level maximum of E3 or E4, direction
`MATCH`, and data overlap `INDEPENDENT`. `SAME_TCGA` and `POSSIBLE_TCGA_OVERLAP` do not qualify.
Write a concise evidence-linked note and a non-identifying assigned curator code for each eligible
record. Do not infer that E0 means the claim is false.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--email", required=True)
    parser.add_argument("--search-date", required=True)
    parser.add_argument("--publication-cutoff", required=True)
    parser.add_argument("--api-key-env", default="NCBI_API_KEY")
    parser.add_argument(
        "--use-system-trust",
        action="store_true",
        help="Use native macOS trust with verified TLS through truststore.",
    )
    parser.add_argument("--p2-config", type=Path, default=DEFAULT_P2_CONFIG)
    parser.add_argument("--p3-config", type=Path, default=DEFAULT_P3_CONFIG)
    parser.add_argument("--p2-checker", type=Path, default=DEFAULT_P2_CHECKER)
    args = parser.parse_args()

    require_clean_tracked_worktree()
    search_date = validate_iso_date(args.search_date, name="search_date")
    publication_cutoff = validate_iso_date(args.publication_cutoff, name="publication_cutoff")
    require(publication_cutoff <= search_date, "publication_cutoff must not follow search_date")
    email = str(args.email).strip()
    require("@" in email and " " not in email, "--email must be a valid contact email")
    p3_protocol = read_json(args.p3_config)
    require(
        p3_protocol["status"] == "QUERY_DESIGN_FROZEN_BEFORE_RETRIEVAL",
        "P3 query design is not frozen",
    )
    run_p2_gate(checker=args.p2_checker, data_root=args.data_root.resolve(), config=args.p2_config)

    benchmark_id = str(p3_protocol["parent_priority2_benchmark_id"])
    p2_root = args.data_root.resolve() / "output" / "priority2" / benchmark_id
    p3_root = args.data_root.resolve() / "output" / "priority3" / benchmark_id
    p2_manifest_path = p2_root / "metrics" / "priority2_freeze_manifest.json"
    p2_manifest = read_json(p2_manifest_path)
    p2_outputs = {key: Path(value["path"]) for key, value in p2_manifest["outputs"].items()}
    claims = pd.read_csv(p2_outputs["claims"], sep="\t")
    sampling = pd.read_csv(p2_outputs["sampling_frame"], sep="\t")
    require(len(claims) == len(sampling) == 50, "P3 requires the frozen 50-claim census")
    claims_blinded = sampling.merge(
        claims[["claim_uid", "claim_text", "pathway_label", "entity", "direction"]],
        on="claim_uid",
        how="left",
        validate="one_to_one",
    ).sort_values("packet_order")
    require(claims_blinded["claim_text"].notna().all(), "P2 claim mapping failed")

    paths = {
        "query_manifest": p3_root / "retrieval" / "query_manifest.tsv",
        "search_responses": p3_root / "retrieval" / "esearch_responses.jsonl",
        "links": p3_root / "retrieval" / "claim_record_links.tsv",
        "records": p3_root / "retrieval" / "records_unique.private.tsv",
        "claims_blinded": p3_root / "grading" / "claims_for_grading.tsv",
        "screening": p3_root / "grading" / "record_screening_template.private.tsv",
        "grading_instructions": p3_root / "grading" / "P3_GRADING_INSTRUCTIONS.md",
        "manifest": p3_root / "retrieval" / "priority3_retrieval_manifest.json",
        "manifest_sha256": p3_root / "retrieval" / "priority3_retrieval_manifest.sha256",
    }
    raw_xml_dir = p3_root / "retrieval" / "efetch_xml.private"
    collisions = [str(path) for path in paths.values() if path.exists()]
    if raw_xml_dir.exists():
        collisions.append(str(raw_xml_dir))
    require(not collisions, f"P3 retrieval is immutable; output collisions: {collisions}")

    api_key = os.environ.get(str(args.api_key_env), "").strip()
    ssl_context, tls_metadata = build_tls_context(use_system_trust=bool(args.use_system_trust))
    client = NCBIClient(
        base_url=str(p3_protocol["base_url"]),
        tool=str(p3_protocol["tool_name"]),
        email=email,
        api_key=api_key,
        ssl_context=ssl_context,
    )
    query_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    families = list(p3_protocol["query_families"])
    for claim in claims_blinded.itertuples(index=False):
        queries = build_queries(claim.entity, p3_protocol)
        require(set(queries) == set(families), "Query-family implementation drift")
        terms = pathway_terms(claim.entity, p3_protocol)
        for family in families:
            response = client.search(
                query=queries[family],
                retmax=int(p3_protocol["retmax_per_query"]),
                sort=str(p3_protocol["sort"]),
                publication_cutoff=publication_cutoff,
            )
            result = response["esearchresult"]
            pmids = [str(value) for value in result.get("idlist", [])]
            query_id = f"{claim.review_id}__{family}"
            query_rows.append(
                {
                    "query_id": query_id,
                    "review_id": claim.review_id,
                    "packet_order": int(claim.packet_order),
                    "entity": claim.entity,
                    "query_family": family,
                    "pathway_terms": "; ".join(terms),
                    "query": queries[family],
                    "sort": p3_protocol["sort"],
                    "retmax": int(p3_protocol["retmax_per_query"]),
                    "total_hits": int(result.get("count", 0)),
                    "returned_pmids": len(pmids),
                    "search_date": search_date,
                    "publication_cutoff": publication_cutoff,
                    "query_translation": normalize_space(result.get("querytranslation", "")),
                }
            )
            raw_responses.append({"query_id": query_id, "response": response})
            for rank, pmid in enumerate(pmids, start=1):
                link_rows.append(
                    {
                        "query_id": query_id,
                        "review_id": claim.review_id,
                        "claim_uid": claim.claim_uid,
                        "query_family": family,
                        "query_rank": rank,
                        "pmid": pmid,
                    }
                )
            print(
                f"[P3] {int(claim.packet_order):02d}/50 {family}: "
                f"hits={int(result.get('count', 0))} returned={len(pmids)}"
            )

    query_manifest = pd.DataFrame(query_rows).sort_values(["packet_order", "query_family"])
    links = pd.DataFrame(link_rows)
    if links.empty:
        links = pd.DataFrame(
            columns=["query_id", "review_id", "claim_uid", "query_family", "query_rank", "pmid"]
        )
    unique_pmids = sorted(set(links["pmid"].astype(str)))
    raw_xml_dir.mkdir(parents=True, exist_ok=True)
    record_frames = []
    raw_xml_paths = []
    for index, batch in enumerate(chunks(unique_pmids, 200), start=1):
        xml_bytes = client.fetch(batch)
        xml_path = raw_xml_dir / f"batch_{index:04d}.xml"
        xml_path.write_bytes(xml_bytes)
        raw_xml_paths.append(xml_path)
        record_frames.append(parse_pubmed_xml(xml_bytes))
        print(f"[P3] EFetch batch {index}: requested={len(batch)}")
    records = (
        pd.concat(record_frames, ignore_index=True).drop_duplicates("pmid")
        if record_frames
        else pd.DataFrame(columns=RECORD_COLUMNS)
    )
    fetched_pmids = set(records["pmid"].astype(str))
    missing_pmids = sorted(set(unique_pmids) - fetched_pmids)

    blinded_columns = ["review_id", "packet_order", "claim_text", "pathway_label", "direction"]
    claims_for_grading = claims_blinded[blinded_columns].copy()
    screening = make_screening_template(
        links=links,
        records=records,
        claims_blinded=claims_for_grading,
    )
    table_outputs = {
        "query_manifest": query_manifest,
        "links": links,
        "records": records,
        "claims_blinded": claims_for_grading,
        "screening": screening,
    }
    for label, table in table_outputs.items():
        path = paths[label]
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, sep="\t", index=False, lineterminator="\n")
    paths["search_responses"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_responses),
        encoding="utf-8",
    )
    paths["grading_instructions"].write_text(grading_instructions(p3_protocol), encoding="utf-8")

    input_paths = {
        "p2_manifest": p2_manifest_path,
        "p2_manifest_sha256": p2_root / "metrics" / "priority2_freeze_manifest.sha256",
        "p2_claims": p2_outputs["claims"],
        "p2_sampling_frame": p2_outputs["sampling_frame"],
        "p2_protocol": args.p2_config,
        "p3_protocol": args.p3_config,
    }
    output_inventory = {
        label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for label, path in paths.items()
        if label not in {"manifest", "manifest_sha256"}
    }
    output_inventory["efetch_xml_batches"] = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in raw_xml_paths
    ]
    manifest = {
        "protocol_version": p3_protocol["protocol_version"],
        "status": "RETRIEVAL_FROZEN_GRADING_NOT_STARTED",
        "benchmark_id": benchmark_id,
        "search_date": search_date,
        "publication_cutoff": publication_cutoff,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "provider": p3_protocol["provider"],
        "base_url": p3_protocol["base_url"],
        "sort": p3_protocol["sort"],
        "retmax_per_query": int(p3_protocol["retmax_per_query"]),
        "api_key_used": bool(api_key),
        "contact_email_supplied": True,
        "tls_verification": tls_metadata,
        "candidate_claims": len(claims_for_grading),
        "queries": len(query_manifest),
        "claim_record_links": len(links),
        "unique_pmids_requested": len(unique_pmids),
        "unique_records_fetched": len(records),
        "missing_pmids": missing_pmids,
        "grading_outcomes_inspected": False,
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for label, path in input_paths.items()
        },
        "outputs": output_inventory,
        "private_outputs_not_for_public_redistribution": [
            "records_unique.private.tsv",
            "record_screening_template.private.tsv",
            "efetch_xml.private",
        ],
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["manifest_sha256"].write_text(
        f"{sha256_file(paths['manifest'])}  {paths['manifest'].name}\n", encoding="utf-8"
    )
    print("[PASS] Priority 3 PubMed retrieval frozen before grading")
    print(f"[INFO] Claims: {len(claims_for_grading)}; queries: {len(query_manifest)}")
    print(f"[INFO] Unique PMIDs requested/fetched: {len(unique_pmids)}/{len(records)}")
    print(f"[INFO] Record-screening rows: {len(screening)}")
    print("[INFO] Audit status and method membership are absent from grading files")
    print(f"[INFO] Wrote: {paths['manifest']}")


if __name__ == "__main__":
    main()
