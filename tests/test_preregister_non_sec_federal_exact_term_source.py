from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_non_sec_federal_exact_term_source as nfet


def _rehash(payload: dict[str, object]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = nfet.canonical_hash(core)


def _valid_positive_records() -> tuple[dict[str, object], dict[str, object]]:
    publication_date = "2020-01-27"
    document_number = "2020-01065"
    urls = nfet.govinfo_document_urls(publication_date, document_number)
    mods: dict[str, object] = {
        "collection_code": "FR",
        "access_id": document_number,
        "fr_doc_number": document_number,
        "identifier_fr_doc_number": document_number,
        "granule_class": "RULE",
        "host_date_issued": publication_date,
        "host_uri": f"https://www.govinfo.gov/app/details/FR-{publication_date}",
        "agency_names": ("COMMODITY FUTURES TRADING COMMISSION",),
        "other_format_urls": tuple(sorted((urls["html"], urls["pdf"]))),
    }
    detail: dict[str, object] = {
        "document_number": document_number,
        "publication_date": publication_date,
        "pdf_url": urls["pdf"],
        "agencies": [{"slug": "commodity-futures-trading-commission"}],
        "correction_of": None,
        "corrections": [],
    }
    return mods, detail


def test_manifest_is_source_only_and_incidence_blind() -> None:
    payload = nfet.build_manifest()
    nfet.validate_manifest(payload)
    assert payload["source_id"] == "NFET"
    assert payload["outcomes_opened"] is False
    assert payload["market_clocks_opened"] is False
    assert payload["candidate_envelope_probe_opened"] is True
    assert payload["final_exact_member_incidence_opened"] is False
    assert payload["candidate_envelope_probe"] == {
        "scope": "broad FederalRegister API feasibility union only",
        "years": [2020, 2021, 2022, 2023],
        "unique_document_numbers": 486,
        "unique_publication_days": 348,
        "documents_with_sec_agency": 218,
        "exact_govinfo_membership_evaluated": False,
        "market_or_outcomes_opened": False,
    }
    assert payload["semantic_model_opened"] is False
    assert payload["later_model_boundary"]["authorized_now"] is False


def test_authority_hierarchy_rejects_bulk_xml_membership() -> None:
    source = nfet.build_manifest()["source_contract"]
    assert source["bulk_xml"]["used"] is False
    assert source["official_document"]["html_establishes_membership"] is True
    assert source["official_document"]["pdf_adds_or_repairs_membership"] is False
    assert (
        source["candidate_discovery"]["search_rank_or_snippet_establishes_membership"]
        is False
    )


def test_official_html_parser_is_visible_ordered_and_control_stable() -> None:
    raw = (
        b"<html><head><style>bitcoin</style><script>blockchain</script></head>"
        b"<body><p>Virtual&nbsp;Currency</p>\x00<table><tr><td>Distributed</td>"
        b"<td>Ledger</td></tr></table></body></html>"
    )
    view = nfet.official_html_membership_view(raw)
    assert view == "virtual currency distributed ledger"
    assert [item["pattern_id"] for item in nfet.exact_term_matches(view)] == [
        "virtual_currency",
        "distributed_ledger",
    ]


def test_exact_terms_accept_frozen_plural_and_hyphen_forms() -> None:
    view = nfet.normalize_membership_text(
        "Bitcoin; cryptocurrencies; virtual-currencies; BLOCKCHAIN; distributed-ledger."
    )
    assert [item["pattern_id"] for item in nfet.exact_term_matches(view)] == [
        "bitcoin",
        "cryptocurrency",
        "virtual_currency",
        "blockchain",
        "distributed_ledger",
    ]


def test_exact_terms_reject_substrings_and_nonadjacent_phrases() -> None:
    view = nfet.normalize_membership_text(
        "bitcoinized xcryptocurrency virtual reserve currency "
        "blockchains distributed public ledger"
    )
    assert nfet.exact_term_matches(view) == []


def test_agency_routing_sends_any_sec_coagency_to_comparator() -> None:
    names = nfet.canonical_agency_names(
        [
            {"name": "Treasury\u0000 Department"},
            {"name": "Securities and Exchange Commission"},
        ]
    )
    assert names == (
        "SECURITIES AND EXCHANGE COMMISSION",
        "TREASURY DEPARTMENT",
    )
    assert nfet.route_stratum(names) == "sec_comparator"
    assert nfet.route_stratum(("COMMODITY FUTURES TRADING COMMISSION",)) == (
        "primary_non_sec"
    )


@pytest.mark.parametrize(
    "agencies, match",
    [
        ([], "no agencies"),
        ([{"slug": "missing-name"}], "malformed"),
        ([{"name": "treasury"}, {"name": "TREASURY"}], "repeats"),
    ],
)
def test_agency_contract_fails_closed(
    agencies: list[dict[str, str]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        nfet.canonical_agency_names(agencies)


def test_detail_agency_cross_check_freezes_slugs() -> None:
    assert nfet.canonical_detail_agency_slugs(
        [{"slug": "securities-and-exchange-commission"}]
    ) == ("securities-and-exchange-commission",)
    with pytest.raises(ValueError, match="malformed"):
        nfet.canonical_detail_agency_slugs([{"slug": "SEC Commission"}])


def test_issue_inventory_parser_sorts_and_preserves_lastmod() -> None:
    xml = b"""<?xml version='1.0' encoding='UTF-8'?>
    <urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
      <url><loc>https://www.govinfo.gov/app/details/FR-2020-01-03</loc>
      <lastmod>2025-09-24T16:23:01.273Z</lastmod></url>
      <url><loc>https://www.govinfo.gov/app/details/FR-2020-01-02</loc>
      <lastmod>2025-09-24T16:23:01.357Z</lastmod></url>
    </urlset>"""
    records = nfet.parse_issue_inventory(xml, 2020)
    assert [record["package_id"] for record in records] == [
        "FR-2020-01-02",
        "FR-2020-01-03",
    ]
    assert records[0]["lastmod"] == "2025-09-24T16:23:01.357Z"


def test_issue_inventory_rejects_duplicates_and_out_of_year() -> None:
    duplicate = b"""<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
      <url><loc>https://www.govinfo.gov/app/details/FR-2020-01-02</loc><lastmod>x</lastmod></url>
      <url><loc>https://www.govinfo.gov/app/details/FR-2020-01-02</loc><lastmod>y</lastmod></url>
    </urlset>"""
    with pytest.raises(ValueError, match="repeats"):
        nfet.parse_issue_inventory(duplicate, 2020)

    out_of_year = b"""<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
      <url><loc>https://www.govinfo.gov/app/details/FR-2021-01-04</loc><lastmod>x</lastmod></url>
    </urlset>"""
    with pytest.raises(ValueError, match="out-of-year"):
        nfet.parse_issue_inventory(out_of_year, 2020)

    unexpected = b"""<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
      <unexpected/>
    </urlset>"""
    with pytest.raises(ValueError, match="unexpected root child"):
        nfet.parse_issue_inventory(unexpected, 2020)


def test_govinfo_mods_parser_freezes_identity_agency_and_renditions() -> None:
    xml = b"""<mods xmlns='http://www.loc.gov/mods/v3'
      xmlns:xlink='http://www.w3.org/1999/xlink'>
      <identifier type='FR Doc No.'>2020-01065</identifier>
      <extension>
        <collectionCode>FR</collectionCode><granuleClass>RULE</granuleClass>
        <accessId>2020-01065</accessId><frDocNumber>2020-01065</frDocNumber>
        <agency order='1'>SECURITIES AND EXCHANGE COMMISSION</agency>
      </extension>
      <relatedItem type='otherFormat'
        xlink:href='https://www.govinfo.gov/content/pkg/FR-2020-01-27/html/2020-01065.htm'/>
      <relatedItem type='otherFormat'
        xlink:href='https://www.govinfo.gov/content/pkg/FR-2020-01-27/pdf/2020-01065.pdf'/>
      <relatedItem type='host'>
        <originInfo><dateIssued>2020-01-27</dateIssued></originInfo>
        <identifier type='uri'>https://www.govinfo.gov/app/details/FR-2020-01-27</identifier>
      </relatedItem>
    </mods>"""
    parsed = nfet.parse_govinfo_mods(xml)
    assert parsed["collection_code"] == "FR"
    assert parsed["access_id"] == "2020-01065"
    assert parsed["fr_doc_number"] == "2020-01065"
    assert parsed["identifier_fr_doc_number"] == "2020-01065"
    assert parsed["granule_class"] == "RULE"
    assert parsed["host_date_issued"] == "2020-01-27"
    assert parsed["agency_names"] == ("SECURITIES AND EXCHANGE COMMISSION",)
    assert parsed["other_format_urls"] == (
        "https://www.govinfo.gov/content/pkg/FR-2020-01-27/html/2020-01065.htm",
        "https://www.govinfo.gov/content/pkg/FR-2020-01-27/pdf/2020-01065.pdf",
    )


def test_positive_identity_reconciliation_accepts_only_exact_sources() -> None:
    mods, detail = _valid_positive_records()
    result = nfet.reconcile_positive_identity("2020-01065", "2020-01-27", mods, detail)
    assert result["stratum"] == "primary_non_sec"
    assert result["govinfo_agency_names"] == ("COMMODITY FUTURES TRADING COMMISSION",)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("collection_code", "CFR"),
        ("access_id", "2020-99999"),
        ("fr_doc_number", "2020-99999"),
        ("identifier_fr_doc_number", "2020-99999"),
        ("host_date_issued", "2020-01-28"),
        ("host_uri", "https://example.invalid/issue"),
        ("other_format_urls", ("https://example.invalid/document",)),
    ],
)
def test_positive_identity_rejects_each_mods_mismatch(
    field: str, bad_value: object
) -> None:
    mods, detail = _valid_positive_records()
    mods[field] = bad_value
    with pytest.raises(ValueError, match="identity mismatch"):
        nfet.reconcile_positive_identity("2020-01065", "2020-01-27", mods, detail)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("document_number", "2020-99999"),
        ("publication_date", "2020-01-28"),
        ("pdf_url", "https://example.invalid/document.pdf"),
    ],
)
def test_positive_identity_rejects_each_detail_mismatch(
    field: str, bad_value: object
) -> None:
    mods, detail = _valid_positive_records()
    detail[field] = bad_value
    with pytest.raises(ValueError, match="detail .* mismatch"):
        nfet.reconcile_positive_identity("2020-01065", "2020-01-27", mods, detail)


def test_positive_identity_requires_correction_metadata() -> None:
    mods, detail = _valid_positive_records()
    del detail["correction_of"]
    with pytest.raises(ValueError, match="correction metadata"):
        nfet.reconcile_positive_identity("2020-01065", "2020-01-27", mods, detail)


def test_agency_reconciliation_rejects_both_sec_disagreement_directions() -> None:
    with pytest.raises(ValueError, match="routing disagree"):
        nfet.reconcile_agency_routing(
            ("SECURITIES AND EXCHANGE COMMISSION",),
            [{"slug": "commodity-futures-trading-commission"}],
        )
    with pytest.raises(ValueError, match="routing disagree"):
        nfet.reconcile_agency_routing(
            ("COMMODITY FUTURES TRADING COMMISSION",),
            [{"slug": "securities-and-exchange-commission"}],
        )


def test_predictable_govinfo_document_urls_are_range_and_identity_bound() -> None:
    urls = nfet.govinfo_document_urls("2020-01-02", "2019-27801")
    assert urls == {
        "mods": (
            "https://www.govinfo.gov/metadata/granule/FR-2020-01-02/2019-27801/mods.xml"
        ),
        "html": (
            "https://www.govinfo.gov/content/pkg/FR-2020-01-02/html/2019-27801.htm"
        ),
        "pdf": ("https://www.govinfo.gov/content/pkg/FR-2020-01-02/pdf/2019-27801.pdf"),
    }
    with pytest.raises(ValueError, match="escaped"):
        nfet.govinfo_document_urls("2024-01-01", "2024-00001")
    with pytest.raises(ValueError, match="malformed"):
        nfet.govinfo_document_urls("2020-01-02", "../escape")


def test_quality_contract_requires_full_reconciliation_without_quarantine() -> None:
    integrity = nfet.build_manifest()["source_quality_gates"]["integrity"]
    assert integrity["candidate_page_reconciliation_fraction"] == 1.0
    assert integrity["candidate_issue_inventory_reconciliation_fraction"] == 1.0
    assert integrity["positive_mods_html_pdf_detail_reconciliation_fraction"] == 1.0
    assert integrity["quarantine_or_imputation_allowed"] is False


def test_novelty_contract_freezes_metrics_comparators_and_thresholds() -> None:
    novelty = nfet.build_manifest()["novelty_gates"]
    assert novelty["required_comparators"] == [
        "NFET_SEC_same_source",
        "GDELT_GNRC",
        "SEC_EDGAR",
        "Wikimedia",
        "BitMEX_Trollbox",
        "executable_live_portfolio",
    ]
    assert novelty["exact_entry_jaccard_max"] == 0.20
    assert novelty["tolerant_window_hours"] == 24
    assert novelty["tolerant_one_to_one_jaccard_max"] == 0.35
    assert novelty["primary_containment_max"] == 0.50
    assert novelty["all_metrics_all_comparators_must_pass"] is True


def test_manifest_hash_detects_mutation() -> None:
    payload = nfet.build_manifest()
    payload["source_contract"]["candidate_discovery"]["literal_term_queries"] = [
        "bitcoin"
    ]
    with pytest.raises(RuntimeError, match="hash mismatch"):
        nfet.validate_manifest(payload)


def test_recomputed_hash_cannot_change_frozen_contract() -> None:
    payload = nfet.build_manifest()
    payload["source_quality_gates"]["primary_non_sec"]["minimum_documents"] = 1
    _rehash(payload)
    with pytest.raises(RuntimeError, match="differs from code"):
        nfet.validate_manifest(payload)


def test_builds_do_not_share_mutable_state() -> None:
    first = nfet.build_manifest()
    first["novelty_gates"]["required_comparators"][0] = "MUTATED"
    second = nfet.build_manifest()
    assert second["novelty_gates"]["required_comparators"][0] == "NFET_SEC_same_source"


def test_repository_bindings_match_bytes() -> None:
    payload = nfet.build_manifest()
    assert (
        nfet.sha256_file(payload["decision_binding"]["path"])
        == payload["decision_binding"]["sha256"]
    )
    assert (
        nfet.sha256_file(payload["implementation_binding"]["path"])
        == payload["implementation_binding"]["sha256"]
    )


def test_write_once_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "nfet.json"
    payload = nfet.build_manifest()
    assert nfet.write_manifest_once(path, payload) == "created"
    assert nfet.write_manifest_once(path, nfet.build_manifest()) == "verified_existing"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == payload


def test_write_once_rejects_mutated_first_write(tmp_path: Path) -> None:
    path = tmp_path / "nfet-mutated.json"
    payload = nfet.build_manifest()
    payload["final_exact_member_incidence_opened"] = True
    _rehash(payload)
    with pytest.raises(RuntimeError, match="must keep .*false"):
        nfet.write_manifest_once(path, payload)
    assert not path.exists()


def test_repository_artifact_matches_code() -> None:
    artifact = json.loads((nfet.REPO_ROOT / nfet.DEFAULT_OUTPUT).read_text())
    nfet.validate_manifest(artifact)
    assert artifact == nfet.build_manifest()
