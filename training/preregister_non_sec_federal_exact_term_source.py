"""Freeze NFET source access before exact-term incidence or outcomes are opened."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "results/non_sec_federal_exact_term_source_protocol_2026-07-20.json"
)
DECISION_PATH = Path(
    "docs/non-sec-federal-exact-term-source-axis-decision-2026-07-20.md"
)
DECISION_SHA256 = "d95dc74bad967f2fc4b3a64ccb7bbaa2ee71a5586053e74a27fcfbf01da7b436"
SCRIPT_PATH = Path("training/preregister_non_sec_federal_exact_term_source.py")
SOURCE_START = "2020-01-01"
SOURCE_END_EXCLUSIVE = "2024-01-01"
UNICODE_DATABASE_VERSION = "13.0.0"
SEC_AGENCY_SLUG = "securities-and-exchange-commission"
SEC_AGENCY_NAME = "SECURITIES AND EXCHANGE COMMISSION"
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
MODS_NAMESPACE = "http://www.loc.gov/mods/v3"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"

TERM_QUERIES = (
    "bitcoin",
    "cryptocurrency",
    "cryptocurrencies",
    "virtual currency",
    "virtual currencies",
    "virtual-currency",
    "virtual-currencies",
    "blockchain",
    "distributed ledger",
    "distributed-ledger",
)

TERM_PATTERN_SPECS = (
    ("bitcoin", r"(?<![a-z0-9_])bitcoin(?![a-z0-9_])"),
    (
        "cryptocurrency",
        r"(?<![a-z0-9_])cryptocurrenc(?:y|ies)(?![a-z0-9_])",
    ),
    (
        "virtual_currency",
        r"(?<![a-z0-9_])virtual[ -]+currenc(?:y|ies)(?![a-z0-9_])",
    ),
    ("blockchain", r"(?<![a-z0-9_])blockchain(?![a-z0-9_])"),
    (
        "distributed_ledger",
        r"(?<![a-z0-9_])distributed[ -]+ledger(?![a-z0-9_])",
    ),
)
TERM_PATTERNS = tuple(
    (pattern_id, re.compile(pattern, re.ASCII))
    for pattern_id, pattern in TERM_PATTERN_SPECS
)
AGENCY_SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.ASCII)
DOCUMENT_NUMBER_PATTERN = re.compile(r"(?=.{3,64}\Z)[A-Z0-9]+(?:-[A-Z0-9]+)+", re.ASCII)
ISSUE_URL_PATTERN = re.compile(
    r"https://www\.govinfo\.gov/app/details/FR-(\d{4}-\d{2}-\d{2})",
    re.ASCII,
)


class _VisibleTextParser(HTMLParser):
    """Collect visible HTML data in document order under the frozen contract."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.fragments: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.fragments.append(data)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return sha256_bytes(candidate.read_bytes())


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def normalize_membership_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    without_controls = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in normalized
    )
    return re.sub(r"\s+", " ", without_controls).strip()


def official_html_membership_view(raw_html: bytes) -> str:
    parser = _VisibleTextParser()
    parser.feed(raw_html.decode("utf-8", errors="strict"))
    parser.close()
    return normalize_membership_text(" ".join(parser.fragments))


def exact_term_matches(normalized_text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for pattern_id, pattern in TERM_PATTERNS:
        for match in pattern.finditer(normalized_text):
            matches.append(
                {
                    "pattern_id": pattern_id,
                    "substring": match.group(0),
                    "span_start": match.start(),
                    "span_end_exclusive": match.end(),
                }
            )
    return sorted(
        matches,
        key=lambda item: (
            item["span_start"],
            item["span_end_exclusive"],
            item["pattern_id"],
        ),
    )


def canonical_agency_names(agencies: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if not agencies:
        raise ValueError("NFET GovInfo summary has no agencies")
    names: list[str] = []
    for agency in agencies:
        raw_name = agency.get("name")
        if not isinstance(raw_name, str):
            raise ValueError("NFET GovInfo summary has a malformed agency name")
        normalized = unicodedata.normalize("NFKC", raw_name).upper()
        without_controls = "".join(
            " " if unicodedata.category(char).startswith("C") else char
            for char in normalized
        )
        name = re.sub(r"\s+", " ", without_controls).strip()
        if not name:
            raise ValueError("NFET GovInfo summary has an empty agency name")
        if name in names:
            raise ValueError("NFET GovInfo summary repeats an agency name")
        names.append(name)
    return tuple(sorted(names))


def canonical_detail_agency_slugs(
    agencies: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    if not agencies:
        raise ValueError("NFET detail JSON has no agencies")
    slugs: list[str] = []
    for agency in agencies:
        slug = agency.get("slug")
        if not isinstance(slug, str) or AGENCY_SLUG_PATTERN.fullmatch(slug) is None:
            raise ValueError("NFET detail JSON has a malformed agency slug")
        if slug in slugs:
            raise ValueError("NFET detail JSON repeats an agency slug")
        slugs.append(slug)
    return tuple(sorted(slugs))


def route_stratum(agency_names: Sequence[str]) -> str:
    if not agency_names:
        raise ValueError("NFET agency routing requires a nonempty agency set")
    return "sec_comparator" if SEC_AGENCY_NAME in agency_names else "primary_non_sec"


def reconcile_agency_routing(
    govinfo_agency_names: Sequence[str],
    detail_agencies: Sequence[Mapping[str, Any]],
) -> str:
    if not govinfo_agency_names:
        raise ValueError("NFET agency reconciliation requires GovInfo agencies")
    detail_slugs = canonical_detail_agency_slugs(detail_agencies)
    govinfo_is_sec = SEC_AGENCY_NAME in govinfo_agency_names
    detail_is_sec = SEC_AGENCY_SLUG in detail_slugs
    if govinfo_is_sec != detail_is_sec:
        raise ValueError("NFET GovInfo and detail SEC agency routing disagree")
    return route_stratum(govinfo_agency_names)


def parse_issue_inventory(raw_xml: bytes, expected_year: int) -> list[dict[str, str]]:
    if expected_year not in {2020, 2021, 2022, 2023}:
        raise ValueError("NFET issue inventory year escaped the frozen range")
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as exc:
        raise ValueError("NFET GovInfo issue sitemap is malformed") from exc
    if root.tag != f"{{{SITEMAP_NAMESPACE}}}urlset":
        raise ValueError("NFET GovInfo issue sitemap has the wrong root")

    url_tag = f"{{{SITEMAP_NAMESPACE}}}url"
    if any(child.tag != url_tag for child in root):
        raise ValueError("NFET issue sitemap has an unexpected root child")

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    allowed_entry_tags = {
        f"{{{SITEMAP_NAMESPACE}}}{name}"
        for name in ("loc", "lastmod", "changefreq", "priority")
    }
    for url_node in root.findall(url_tag):
        if any(child.tag not in allowed_entry_tags for child in url_node):
            raise ValueError("NFET issue sitemap entry has an unexpected child")
        loc_nodes = url_node.findall(f"{{{SITEMAP_NAMESPACE}}}loc")
        lastmod_nodes = url_node.findall(f"{{{SITEMAP_NAMESPACE}}}lastmod")
        if len(loc_nodes) != 1 or len(lastmod_nodes) != 1:
            raise ValueError("NFET issue sitemap entry is not singular")
        loc = loc_nodes[0].text or ""
        lastmod = lastmod_nodes[0].text or ""
        match = ISSUE_URL_PATTERN.fullmatch(loc)
        if match is None or not lastmod:
            raise ValueError("NFET issue sitemap entry is malformed")
        publication_date = match.group(1)
        parsed_date = date.fromisoformat(publication_date)
        if parsed_date.year != expected_year:
            raise ValueError("NFET issue sitemap contains an out-of-year package")
        package_id = f"FR-{publication_date}"
        if package_id in seen:
            raise ValueError("NFET issue sitemap repeats a package")
        seen.add(package_id)
        records.append(
            {
                "package_id": package_id,
                "publication_date": publication_date,
                "lastmod": lastmod,
            }
        )
    if not records:
        raise ValueError("NFET issue sitemap is empty")
    return sorted(records, key=lambda item: item["publication_date"])


def _singular_text(root: ET.Element, path: str, label: str) -> str:
    nodes = root.findall(path)
    values = [node.text.strip() for node in nodes if node.text and node.text.strip()]
    if len(nodes) != 1 or len(values) != 1:
        raise ValueError(f"NFET GovInfo MODS {label} is not singular and nonempty")
    return values[0]


def parse_govinfo_mods(raw_xml: bytes) -> dict[str, Any]:
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as exc:
        raise ValueError("NFET GovInfo MODS is malformed") from exc
    namespace = f"{{{MODS_NAMESPACE}}}"
    if root.tag != f"{namespace}mods":
        raise ValueError("NFET GovInfo MODS has the wrong root")

    granule_extensions = [
        node
        for node in root.findall(f"{namespace}extension")
        if node.find(f"{namespace}granuleClass") is not None
    ]
    if len(granule_extensions) != 1:
        raise ValueError("NFET GovInfo MODS granule extension is not singular")
    granule_extension = granule_extensions[0]

    identifiers = [
        (node.text or "").strip()
        for node in root.findall(f"{namespace}identifier")
        if node.get("type") == "FR Doc No."
    ]
    if len(identifiers) != 1 or not identifiers[0]:
        raise ValueError("NFET GovInfo MODS FR Doc No. is not singular")

    host_nodes = [
        node
        for node in root.findall(f"{namespace}relatedItem")
        if node.get("type") == "host"
    ]
    if len(host_nodes) != 1:
        raise ValueError("NFET GovInfo MODS host is not singular")
    host = host_nodes[0]
    host_date = _singular_text(
        host, f"{namespace}originInfo/{namespace}dateIssued", "host dateIssued"
    )
    host_uris = [
        (node.text or "").strip()
        for node in host.findall(f"{namespace}identifier")
        if node.get("type") == "uri"
    ]
    if len(host_uris) != 1 or not host_uris[0]:
        raise ValueError("NFET GovInfo MODS host URI is not singular")

    other_formats: list[str] = []
    for node in root.findall(f"{namespace}relatedItem"):
        if node.get("type") == "otherFormat":
            href = node.get(f"{{{XLINK_NAMESPACE}}}href")
            if not href:
                raise ValueError("NFET GovInfo MODS otherFormat has no href")
            other_formats.append(href)
    if len(other_formats) != 2 or len(set(other_formats)) != 2:
        raise ValueError("NFET GovInfo MODS otherFormat links are not exactly two")

    agency_values = [
        (node.text or "").strip()
        for node in granule_extension.findall(f"{namespace}agency")
    ]
    agencies = canonical_agency_names(
        [{"name": value} for value in agency_values if value]
    )
    return {
        "collection_code": _singular_text(
            root, f".//{namespace}collectionCode", "collectionCode"
        ),
        "access_id": _singular_text(
            granule_extension, f"{namespace}accessId", "granule accessId"
        ),
        "fr_doc_number": _singular_text(
            granule_extension, f"{namespace}frDocNumber", "frDocNumber"
        ),
        "identifier_fr_doc_number": identifiers[0],
        "granule_class": _singular_text(
            granule_extension, f"{namespace}granuleClass", "granuleClass"
        ),
        "host_date_issued": host_date,
        "host_uri": host_uris[0],
        "agency_names": agencies,
        "other_format_urls": tuple(sorted(other_formats)),
    }


def govinfo_document_urls(
    publication_date: str, document_number: str
) -> dict[str, str]:
    parsed_date = date.fromisoformat(publication_date)
    if (
        not date.fromisoformat(SOURCE_START)
        <= parsed_date
        < date.fromisoformat(SOURCE_END_EXCLUSIVE)
    ):
        raise ValueError("NFET document date escaped the frozen range")
    if DOCUMENT_NUMBER_PATTERN.fullmatch(document_number) is None:
        raise ValueError("NFET document number is malformed")
    package_id = f"FR-{publication_date}"
    content_base = f"https://www.govinfo.gov/content/pkg/{package_id}"
    return {
        "mods": (
            f"https://www.govinfo.gov/metadata/granule/{package_id}/"
            f"{document_number}/mods.xml"
        ),
        "html": f"{content_base}/html/{document_number}.htm",
        "pdf": f"{content_base}/pdf/{document_number}.pdf",
    }


def reconcile_positive_identity(
    document_number: str,
    publication_date: str,
    mods: Mapping[str, Any],
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    urls = govinfo_document_urls(publication_date, document_number)
    expected_mods = {
        "collection_code": "FR",
        "access_id": document_number,
        "fr_doc_number": document_number,
        "identifier_fr_doc_number": document_number,
        "host_date_issued": publication_date,
        "host_uri": f"https://www.govinfo.gov/app/details/FR-{publication_date}",
        "other_format_urls": tuple(sorted((urls["html"], urls["pdf"]))),
    }
    for field, expected in expected_mods.items():
        if mods.get(field) != expected:
            raise ValueError(f"NFET GovInfo MODS {field} identity mismatch")
    granule_class = mods.get("granule_class")
    if not isinstance(granule_class, str) or not granule_class.strip():
        raise ValueError("NFET GovInfo MODS granule_class is empty")
    agency_names = mods.get("agency_names")
    if not isinstance(agency_names, (list, tuple)) or not all(
        isinstance(name, str) and name for name in agency_names
    ):
        raise ValueError("NFET GovInfo MODS agency_names are malformed")

    required_detail = {
        "document_number": document_number,
        "publication_date": publication_date,
        "pdf_url": urls["pdf"],
    }
    for field, expected in required_detail.items():
        if detail.get(field) != expected:
            raise ValueError(f"NFET FederalRegister detail {field} mismatch")
    if "correction_of" not in detail or "corrections" not in detail:
        raise ValueError("NFET FederalRegister correction metadata is missing")
    detail_agencies = detail.get("agencies")
    if not isinstance(detail_agencies, list):
        raise ValueError("NFET FederalRegister detail agencies are malformed")
    stratum = reconcile_agency_routing(agency_names, detail_agencies)
    return {
        "stratum": stratum,
        "govinfo_agency_names": tuple(agency_names),
        "detail_agency_slugs": canonical_detail_agency_slugs(detail_agencies),
        "urls": urls,
    }


def _source_contract() -> dict[str, Any]:
    return {
        "source_id": "NFET",
        "source_name": "Non-SEC Federal Exact-Term publication salience",
        "claim_scope": (
            "complete only under the frozen FederalRegister API search envelope; "
            "not a full-corpus semantic census or recall claim"
        ),
        "range": [SOURCE_START, SOURCE_END_EXCLUSIVE],
        "end_is_exclusive": True,
        "authority_hierarchy": [
            "GovInfo collection sitemap for expected issue packages",
            "FederalRegister.gov API for candidate document discovery and metadata reconciliation only",
            "official GovInfo granule MODS and HTML for metadata and exact membership",
            "official GovInfo granule PDF retained as archival/legal authority",
        ],
        "bulk_xml": {
            "used": False,
            "reason": "derived nonofficial rendition with observed parse anomalies",
            "guide": "https://www.govinfo.gov/bulkdata/FR/resources/FDsys_OFR-XML_User-Guide-v1.pdf",
        },
        "issue_inventory": {
            "url_template": "https://www.govinfo.gov/sitemap/FR_{YYYY}_sitemap.xml",
            "years": [2020, 2021, 2022, 2023],
            "root": f"{{{SITEMAP_NAMESPACE}}}urlset",
            "entry_fields": ["loc", "lastmod"],
            "allowed_ignored_entry_fields": ["changefreq", "priority"],
            "loc_fullmatch": ISSUE_URL_PATTERN.pattern,
            "ordering": "publication_date ascending",
            "weekend_holiday_rule": "only sealed sitemap packages exist; never infer calendar dates",
            "failure_effect": "REJECT_NO_REPAIR",
        },
        "candidate_discovery": {
            "endpoint": "https://www.federalregister.gov/api/v1/documents.json",
            "literal_term_queries": list(TERM_QUERIES),
            "fixed_parameters": {
                "conditions[publication_date][gte]": SOURCE_START,
                "conditions[publication_date][lte]": "2023-12-31",
                "order": "oldest",
                "per_page": 1000,
            },
            "pagination": "reconstruct integer page=1..total_pages for every query",
            "candidate_union_key": "document_number",
            "duplicate_must_agree_fields": [
                "document_number",
                "publication_date",
                "type",
                "title",
                "pdf_url",
            ],
            "search_rank_or_snippet_establishes_membership": False,
            "api_false_negative_scope": (
                "a GovInfo exact-term document not proposed by the frozen query "
                "union is outside NFET-v1 and cannot support a full-corpus claim"
            ),
            "conflicting_duplicate_effect": "REJECT_NO_REPAIR",
        },
        "official_document": {
            "mods_template": (
                "https://www.govinfo.gov/metadata/granule/FR-{publication_date}/"
                "{document_number}/mods.xml"
            ),
            "html_template": (
                "https://www.govinfo.gov/content/pkg/FR-{publication_date}/html/"
                "{document_number}.htm"
            ),
            "pdf_template": (
                "https://www.govinfo.gov/content/pkg/FR-{publication_date}/pdf/"
                "{document_number}.pdf"
            ),
            "api_key_required": False,
            "detail_template": (
                "https://www.federalregister.gov/api/v1/documents/"
                "{document_number}.json"
            ),
            "html_establishes_membership": True,
            "pdf_adds_or_repairs_membership": False,
            "positive_pdf_magic": "%PDF-",
            "all_candidate_retention": (
                "content-addressed deterministic compressed search page, MODS, raw "
                "HTML, canonical visible text, and detail JSON"
            ),
            "positive_additional_retention": (
                "content-addressed GovInfo PDF plus exact match records"
            ),
            "html_raw_sha_role": "transport audit only",
            "html_identity": "canonical normalized visible-text SHA-256",
            "reason_raw_html_is_not_identity": (
                "edge email-obfuscation attributes can vary while visible text is stable"
            ),
        },
        "disk": {
            "used_abort_gib": 300,
            "check": "before every response download on the output filesystem",
            "raw_archives_persisted": True,
            "storage": "content-addressed deterministic gzip with mtime=0",
        },
        "revision_rule": (
            "changed previously sealed sitemap, search, MODS, canonical text, or "
            "pdf bytes halt NFET; never rebuild history in place"
        ),
    }


def _parser_contract() -> dict[str, Any]:
    return {
        "mods": {
            "namespace": MODS_NAMESPACE,
            "xlink_namespace": XLINK_NAMESPACE,
            "root": f"{{{MODS_NAMESPACE}}}mods",
            "singular_fields": [
                "collectionCode",
                "accessId",
                "frDocNumber",
                "identifier[type=FR Doc No.]",
                "granuleClass",
                "relatedItem[type=host] originInfo/dateIssued",
                "relatedItem[type=host] identifier[type=uri]",
            ],
            "agency_selector": "all MODS extension agency values",
            "other_format_rule": "exactly two unique xlink href values: HTML and PDF",
        },
        "html_engine": "Python stdlib html.parser.HTMLParser",
        "convert_charrefs": True,
        "decode": "UTF-8 strict",
        "ignored_elements": ["script", "style"],
        "text_order": "all other handle_data fragments in document order",
        "fragment_joiner": "one ASCII space",
        "unicode_normalization": "NFKC",
        "unicode_database_version": UNICODE_DATABASE_VERSION,
        "case_normalization": "str.lower",
        "control_rule": "replace code points whose Unicode category starts C with ASCII space",
        "whitespace_rule": "Python Unicode \\s+ collapsed to one ASCII space, then strip",
        "regex_engine": "Python re",
        "regex_flags": ["re.ASCII"],
        "patterns": [
            {"pattern_id": pattern_id, "regex": pattern}
            for pattern_id, pattern in TERM_PATTERN_SPECS
        ],
        "support_record": [
            "pattern_id",
            "substring",
            "span_start",
            "span_end_exclusive",
        ],
        "stemming_fuzzy_semantic_or_llm_membership": False,
    }


def _quality_contract() -> dict[str, Any]:
    return {
        "primary_non_sec": {
            "minimum_documents": 140,
            "minimum_documents_each_year": 25,
            "minimum_unique_publication_days": 120,
            "minimum_unique_publication_days_each_year": 20,
            "minimum_documents_each_quarter": 5,
            "maximum_month_share": 0.15,
            "maximum_fractional_agency_share": 0.45,
        },
        "sec_comparator": {
            "minimum_documents": 100,
            "minimum_documents_each_year": 15,
            "minimum_unique_publication_days": 80,
        },
        "integrity": {
            "candidate_page_reconciliation_fraction": 1.0,
            "candidate_issue_inventory_reconciliation_fraction": 1.0,
            "positive_mods_html_pdf_detail_reconciliation_fraction": 1.0,
            "quarantine_or_imputation_allowed": False,
            "deterministic_rebuild_required": True,
        },
        "failure_effect": "REJECT_NO_REPAIR",
    }


def _novelty_contract() -> dict[str, Any]:
    return {
        "outcomes_opened": False,
        "source_clock": "one event per unique primary publication date at D+1 12:00 UTC",
        "required_comparators": [
            "NFET_SEC_same_source",
            "GDELT_GNRC",
            "SEC_EDGAR",
            "Wikimedia",
            "BitMEX_Trollbox",
            "executable_live_portfolio",
        ],
        "comparator_bindings": (
            "paths, schemas, common coverage, and SHA-256 values must be committed "
            "in a later outcome-blind novelty access seal"
        ),
        "matching": (
            "within common coverage sort unique UTC entries; visit primary entries "
            "chronologically and match the unused comparator entry with minimum "
            "absolute distance, ties toward the earlier comparator"
        ),
        "exact_entry_jaccard_max": 0.20,
        "tolerant_window_hours": 24,
        "tolerant_one_to_one_jaccard_max": 0.35,
        "primary_containment_max": 0.50,
        "all_metrics_all_comparators_must_pass": True,
        "missing_malformed_unhashable_or_empty_common_coverage": "fail closed",
    }


def build_manifest() -> dict[str, Any]:
    if unicodedata.unidata_version != UNICODE_DATABASE_VERSION:
        raise RuntimeError("NFET Unicode database version differs from the freeze")
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise RuntimeError("NFET source-axis decision hash differs from the freeze")
    core: dict[str, Any] = {
        "protocol_version": "non_sec_federal_exact_term_source_v1",
        "source_id": "NFET",
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "candidate_envelope_probe_opened": True,
        "final_exact_member_incidence_opened": False,
        "semantic_model_opened": False,
        "candidate_envelope_probe": {
            "scope": "broad FederalRegister API feasibility union only",
            "years": [2020, 2021, 2022, 2023],
            "unique_document_numbers": 486,
            "unique_publication_days": 348,
            "documents_with_sec_agency": 218,
            "exact_govinfo_membership_evaluated": False,
            "market_or_outcomes_opened": False,
        },
        "decision_binding": {
            "path": str(DECISION_PATH),
            "sha256": DECISION_SHA256,
        },
        "implementation_binding": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "source_contract": _source_contract(),
        "parser_contract": _parser_contract(),
        "agency_contract": {
            "source": "GovInfo MODS extension agency values",
            "normalization": (
                "Unicode NFKC; str.upper; category-C to space; collapse whitespace"
            ),
            "set_rule": "sorted unique nonempty set; duplicate name is fatal",
            "sec_name": SEC_AGENCY_NAME,
            "mixed_sec_document": "sec_comparator only",
            "primary": "all positive documents without the SEC name",
            "concentration": (
                "each k-agency primary document contributes 1/k to each name"
            ),
            "detail_cross_check": {
                "source": "FederalRegister detail agencies[].slug",
                "slug_fullmatch": AGENCY_SLUG_PATTERN.pattern,
                "sec_slug": SEC_AGENCY_SLUG,
                "sec_membership_must_agree_with_govinfo": True,
            },
            "manual_alias_or_parent_rollup": False,
        },
        "identity_and_reconciliation": {
            "stable_identity": "document_number",
            "implementation_functions": [
                "parse_govinfo_mods",
                "reconcile_agency_routing",
                "reconcile_positive_identity",
            ],
            "document_number_fullmatch": DOCUMENT_NUMBER_PATTERN.pattern,
            "required_equalities": [
                "candidate document_number == GovInfo MODS accessId",
                "candidate document_number == GovInfo MODS frDocNumber",
                "candidate document_number == GovInfo MODS FR Doc No. identifier",
                "candidate publication_date == GovInfo MODS host dateIssued",
                "GovInfo MODS host URI == predictable issue details URL",
                "GovInfo MODS collectionCode == FR",
                "GovInfo MODS otherFormat links == predictable HTML and PDF URLs",
                "candidate document_number == detail document_number",
                "candidate publication_date == detail publication_date",
                "detail pdf_url == predictable GovInfo PDF URL",
            ],
            "positive_required_objects": [
                "candidate search record",
                "GovInfo granule MODS",
                "official GovInfo HTML",
                "official GovInfo PDF",
                "FederalRegister detail JSON",
            ],
            "correction_rule": (
                "record correction_of and corrections; correction remains a new "
                "document at its own availability and never mutates the original"
            ),
            "mismatch_effect": "REJECT_NO_REPAIR",
        },
        "availability_contract": {
            "historical": "publication_date + 1 calendar day at 12:00 UTC",
            "advance_display_or_public_inspection_used": False,
            "live": (
                "max(historical floor, durable local receipt + parse + hash + "
                "reconciliation + manifest commit)"
            ),
            "event_deduplication": "one source event per primary publication date",
        },
        "source_quality_gates": _quality_contract(),
        "novelty_gates": _novelty_contract(),
        "later_model_boundary": {
            "authorized_now": False,
            "allowed_later": "single quote-grounded Gemma extractor/classifier then train-only RLLM gate",
            "forbidden": [
                "LLM membership or agency routing",
                "source event creation deletion or retiming",
                "eval reward for prompt adapter or checkpoint selection",
                "future prices or post-entry information",
                "analyzer/trader two-model split",
            ],
        },
        "next_stage_artifacts": {
            "issue_inventory_and_candidate_response_seal": (
                "results/non_sec_federal_exact_term_source_access_seal_2026-07-20.json"
            ),
            "selected_source": (
                "data/non_sec_federal_exact_term_source_2020_2023.jsonl.gz"
            ),
            "source_manifest": (
                "results/non_sec_federal_exact_term_source_manifest_2026-07-20.json"
            ),
            "source_support_result": (
                "results/non_sec_federal_exact_term_source_support_2026-07-20.json"
            ),
            "write_once": True,
            "deterministic_gzip_mtime": 0,
        },
        "rejection_contract": (
            "any source hierarchy, inventory, pagination, official-text, agency, "
            "identity, reconciliation, disk, quality, or novelty failure retires "
            "NFET without changing queries, terms, dates, parser, SEC routing, "
            "availability, thresholds, or comparators"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("NFET source protocol hash mismatch")
    for field in (
        "outcomes_opened",
        "market_clocks_opened",
        "final_exact_member_incidence_opened",
        "semantic_model_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"NFET source protocol must keep {field}=false")
    if payload.get("candidate_envelope_probe_opened") is not True:
        raise RuntimeError("NFET source protocol must disclose its candidate probe")
    expected = build_manifest()
    expected_core = {
        key: value for key, value in expected.items() if key != "manifest_hash"
    }
    if core != expected_core:
        raise RuntimeError("NFET frozen source contract differs from code")


def write_manifest_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    validate_manifest(payload)
    output = Path(path)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise RuntimeError("NFET existing source protocol is not an object")
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen NFET source protocol")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = build_manifest()
    status = write_manifest_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "source_id": payload["source_id"],
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": False,
                "candidate_envelope_probe_opened": True,
                "final_exact_member_incidence_opened": False,
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
