from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CRM = ROOT / "paper" / "revision" / "CRM_R1"


def load_script(name: str):
    path = CRM / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_priority3_protocol_applies_the_same_fixed_queries_to_every_claim() -> None:
    module = load_script("30_fetch_priority3_pubmed.py")
    protocol = json.loads((CRM / "config" / "priority3_protocol.json").read_text())
    assert protocol["status"] == "QUERY_DESIGN_FROZEN_BEFORE_RETRIEVAL"
    assert protocol["population"] == "all_50_frozen_priority2_claims"
    assert protocol["apply_every_family_to_every_claim"] is True
    assert protocol["sort"] == "relevance"
    assert protocol["retmax_per_query"] == 10
    assert protocol["record_grading_fields"]["evidence_grade"] == [
        "EXCLUDE",
        "E1",
        "E2",
        "E3",
        "E4",
    ]
    queries = module.build_queries("HALLMARK_P53_PATHWAY", protocol)
    assert list(queries) == protocol["query_families"]
    assert all('"Review"[Publication Type]' in query for query in queries.values())
    assert all("NOT (" in query for query in queries.values())
    assert '"p53 pathway"[Title/Abstract]' in queries["direct_hnsc_tp53_pathway"]


def test_priority3_client_uses_required_contact_fields_and_does_not_mutate_response() -> None:
    module = load_script("30_fetch_priority3_pubmed.py")
    observed = {}

    def requester(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return b'{"esearchresult":{"count":"1","idlist":["123"]}}'

    client = module.NCBIClient(
        base_url="https://example.test/eutils",
        tool="crm_test",
        email="analyst@example.org",
        requester=requester,
    )
    client.interval_seconds = 0
    result = client.search(
        query="TP53",
        retmax=10,
        sort="relevance",
        publication_cutoff="2026-08-09",
    )
    parameters = parse_qs(urlparse(observed["url"]).query)
    assert parameters["tool"] == ["crm_test"]
    assert parameters["email"] == ["analyst@example.org"]
    assert parameters["sort"] == ["relevance"]
    assert parameters["maxdate"] == ["2026/08/09"]
    assert observed["timeout"] == 120.0
    assert result["esearchresult"]["idlist"] == ["123"]


def test_priority3_tls_modes_never_disable_certificate_verification() -> None:
    module = load_script("30_fetch_priority3_pubmed.py")
    context, metadata = module.build_tls_context(use_system_trust=False)
    assert context is None
    assert metadata["mode"] == "python_default"
    source = (CRM / "scripts" / "30_fetch_priority3_pubmed.py").read_text()
    assert "--use-system-trust" in source
    assert "truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)" in source
    assert "_create_unverified_context" not in source
    assert "CERT_NONE" not in source


def test_priority3_xml_parser_and_blank_grading_template() -> None:
    module = load_script("30_fetch_priority3_pubmed.py")
    xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>123</PMID><Article><Journal><JournalIssue><PubDate><Year>2025</Year>
      </PubDate></JournalIssue><Title>Journal X</Title></Journal>
      <ArticleTitle>TP53 and <i>EMT</i> in HNSC</ArticleTitle>
      <Abstract><AbstractText Label="RESULTS">Direction matched.</AbstractText></Abstract>
      <AuthorList><Author><LastName>Khan</LastName><Initials>A</Initials></Author></AuthorList>
      <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
      </Article></MedlineCitation><PubmedData><ArticleIdList>
      <ArticleId IdType="doi">10.1/test</ArticleId><ArticleId IdType="pmc">PMC1</ArticleId>
      </ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""
    records = module.parse_pubmed_xml(xml)
    assert records.loc[0, "pmid"] == "123"
    assert records.loc[0, "title"] == "TP53 and EMT in HNSC"
    assert records.loc[0, "abstract"] == "RESULTS: Direction matched."
    assert records.loc[0, "publication_year"] == "2025"
    links = pd.DataFrame(
        {
            "review_id": ["R001"],
            "pmid": ["123"],
            "query_family": ["direct_hnsc_tp53_pathway"],
            "query_rank": [1],
        }
    )
    claims = pd.DataFrame(
        {
            "review_id": ["R001"],
            "claim_text": ["Fixed claim."],
            "pathway_label": ["EMT"],
            "direction": ["up"],
        }
    )
    template = module.make_screening_template(links=links, records=records, claims_blinded=claims)
    grading = {
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
    }
    assert template[list(grading)].eq("").all().all()
    assert "claim_uid" not in template
    assert "audit_status" not in template
    instructions = module.grading_instructions(
        json.loads((CRM / "config" / "priority3_protocol.json").read_text())
    )
    assert "E0 is never a" in instructions
    assert "do not qualify" in instructions


def test_priority4_packets_mask_method_and_apply_literature_limit() -> None:
    module = load_script("40_make_blinded_packets.py")
    count = 50
    claims = pd.DataFrame(
        {
            "claim_uid": [f"C{index:03d}" for index in range(count)],
            "claim_id_mechanical": [f"M{index:03d}" for index in range(count)],
            "claim_text": [f"Claim {index}" for index in range(count)],
            "pathway_label": [f"Pathway {index}" for index in range(count)],
            "direction": ["up" if index % 2 else "down" for index in range(count)],
            "statistic": [float(index) for index in range(count)],
            "q_value": [0.001 * (index + 1) for index in range(count)],
            "status_full": ["PASS"] * count,
        }
    )
    sampling = pd.DataFrame(
        {
            "review_id": [f"R{index:03d}" for index in range(count)],
            "packet_order": list(range(1, count + 1)),
            "claim_uid": claims["claim_uid"],
        }
    )
    mechanical = pd.DataFrame(
        {
            "claim_id": claims["claim_id_mechanical"],
            "gene_symbols_str": ["TP53;CDKN1A;MDM2"] * count,
            "status": ["PASS"] * count,
        }
    )
    packet = module.make_claim_packet(claims=claims, sampling=sampling, mechanical_audit=mechanical)
    assert len(packet) == 50
    assert packet.loc[0, "leading_edge_gene_count"] == 3
    assert "claim_uid" not in packet
    assert "status_full" not in packet

    links = pd.DataFrame(
        {
            "review_id": ["R000"] * 7,
            "query_family": ["context_hnsc_pathway"] * 7,
            "query_rank": list(range(1, 8)),
            "pmid": [str(index) for index in range(1, 8)],
            "claim_uid": ["C000"] * 7,
        }
    )
    records = pd.DataFrame(
        {
            "pmid": [str(index) for index in range(1, 8)],
            "title": [f"Paper {index}" for index in range(1, 8)],
            "publication_year": [2025] * 7,
            "journal": ["Journal"] * 7,
            "publication_types": ["Journal Article"] * 7,
            "doi": [""] * 7,
            "pmcid": [""] * 7,
            "abstract_available": [True] * 7,
            "abstract": ["Abstract"] * 7,
        }
    )
    literature = module.make_literature_packet(
        links=links, records=records, claim_packet=packet, limit_per_family=5
    )
    assert len(literature) == 5
    assert literature["query_rank"].max() == 5
    assert "claim_uid" not in literature
    ratings = module.make_rating_template(packet, rater_id="P4_R1")
    assert len(ratings) == 50
    assert ratings["q3_overstatement"].eq("").all()


def test_priority4_protocol_is_narrow_and_three_rater() -> None:
    protocol = json.loads((CRM / "config" / "priority4_review_protocol.json").read_text())
    assert protocol["status"] == "PACKET_SCHEMA_FROZEN_BEFORE_RATING"
    assert protocol["minimum_independent_raters"] == 3
    assert protocol["questions"]["q3_overstatement"] == [
        "NO_OVERSTATEMENT",
        "MINOR_OVERSTATEMENT",
        "MAJOR_OVERSTATEMENT",
        "UNCERTAIN",
    ]
    assert protocol["primary_human_outcome"] == "MAJOR_OVERSTATEMENT by majority vote"


def test_v9_scripts_refuse_overwrite_and_keep_private_abstracts_out_of_public_source() -> None:
    fetch = (CRM / "scripts" / "30_fetch_priority3_pubmed.py").read_text()
    packet = (CRM / "scripts" / "40_make_blinded_packets.py").read_text()
    checker = (CRM / "scripts" / "31_check_priority3_retrieval.py").read_text()
    assert "P3 retrieval is immutable; output collisions" in fetch
    assert "P4 packets are immutable; output collisions" in packet
    assert "records_unique.private.tsv" in fetch
    assert "blinded_literature.private.tsv" in packet
    assert "grading_is_blank" in checker
    assert '"contact_email": email' not in fetch
    assert '"tls_verification": tls_metadata' in fetch
    assert "require_clean_tracked_worktree()" in fetch
    assert "require_clean_tracked_worktree()" in packet
