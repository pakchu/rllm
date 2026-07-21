from __future__ import annotations

import gzip
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from training import audit_sec_edgar_bitcoin_filing_source as audit


YEARS = tuple(str(year) for year in range(2018, 2024))


def _accession(index: int) -> str:
    return f"0000000001-{18 + index:02d}-{index + 1:06d}"


def _timestamp(index: int) -> str:
    return f"{YEARS[index]}-01-02T17:00:00.000Z"


def _search_payload() -> bytes:
    hits = []
    for index, year in enumerate(YEARS):
        accession = _accession(index)
        hits.append(
            {
                "_id": f"{accession}:event-{year}.htm",
                "_source": {
                    "adsh": accession,
                    "ciks": [f"{index + 1:010d}"],
                    "file_date": f"{year}-01-02",
                    "form": "8-K" if index % 2 == 0 else "6-K",
                    "sequence": "1",
                    "file_description": "Bitcoin event",
                },
            }
        )
    return json.dumps(
        {
            "query": {
                "sort": [
                    {"file_date": {"order": "asc"}},
                    {"_id": {"order": "asc"}},
                ]
            },
            "hits": {
                "total": {"value": len(hits), "relation": "eq"},
                "hits": hits,
            }
        }
    ).encode()


def _submissions(index: int) -> bytes:
    return json.dumps(
        {
            "filings": {
                "recent": {
                    "accessionNumber": [_accession(index)],
                    "acceptanceDateTime": [_timestamp(index)],
                    "form": ["8-K" if index % 2 == 0 else "6-K"],
                },
                "files": [],
            }
        }
    ).encode()


def _header(index: int) -> bytes:
    return (
        f"<ACCEPTANCE-DATETIME>{YEARS[index]}0102120000\n"
        f"ACCESSION NUMBER: {_accession(index)}"
    ).encode()


def _fetch(url: str) -> bytes:
    if url in audit.OFFICIAL_DOCS.values():
        if url == audit.OFFICIAL_DOCS["api"]:
            return b"submissions history by filer"
        if url == audit.OFFICIAL_DOCS["access"]:
            return b"10 requests/second"
        if url == audit.OFFICIAL_DOCS["submit"]:
            return b"The SEC cannot rescind dissemination"
        if url == audit.OFFICIAL_DOCS["adjust"]:
            return b"An acceptance time adjustment request will be denied"
        return b"full text since 2001"
    if url.startswith(audit.EFTS_ENDPOINT):
        offset = int(parse_qs(urlparse(url).query)["from"][0])
        if offset != 0:
            raise AssertionError("small fixture should fit one page")
        return _search_payload()
    if url.startswith(audit.SUBMISSIONS_BASE):
        cik = int(url.rsplit("CIK", 1)[1].split(".json", 1)[0])
        return _submissions(cik - 1)
    if url.endswith("-index-headers.html"):
        accession = url.rsplit("/", 1)[1].split("-index", 1)[0]
        index = int(accession.rsplit("-", 1)[1]) - 1
        return _header(index)
    if url.startswith(audit.ARCHIVES_BASE):
        return b"<html><body>Bitcoin treasury disclosure.</body></html>"
    raise AssertionError(f"unexpected URL: {url}")


SMALL_THRESHOLDS = {
    "source_train_min_accessions": 1,
    "source_train_min_days": 1,
    "source_test_min_accessions": 1,
    "source_test_min_days": 1,
    "selection_min_accessions": 1,
    "selection_min_days": 1,
    "all_min_documents": 6,
    "all_min_ciks": 6,
    "period_max_top1_share": 1.0,
    "period_max_top5_share": 1.0,
    "period_max_hhi": 1.0,
}


def test_source_audit_passes_immutable_source_before_semantic_model(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl.gz"
    report = audit.build_report(_fetch, source, SMALL_THRESHOLDS)

    assert report["metrics"]["documents"] == 6
    assert report["metrics"]["accessions"] == 6
    assert report["metrics"]["event_days"] == 6
    assert report["decision"]["status"] == "passed_for_candidate_preregistration"
    assert report["decision"]["candidate_preregistration_authorized"] is True
    assert report["decision"]["semantic_model_execution_authorized"] is False
    assert report["decision"]["economic_evaluation_authorized"] is False
    assert report["outcome_boundary"]["economic_outcomes_opened"] is False
    assert all(
        sample["header_matches_submissions"]
        for sample in report["transport"]["samples"]
    )
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert [row["acceptance_datetime"] for row in rows] == sorted(
        row["acceptance_datetime"] for row in rows
    )


def test_source_gzip_is_deterministic(tmp_path: Path) -> None:
    rows = [
        {
            "accession": _accession(0),
            "document": "event.htm",
            "acceptance_datetime": _timestamp(0),
        }
    ]
    first = tmp_path / "first.gz"
    second = tmp_path / "second.gz"
    audit.write_source(rows, first)
    audit.write_source(rows, second)
    assert first.read_bytes() == second.read_bytes()


def test_search_parser_rejects_path_traversal() -> None:
    payload = json.loads(_search_payload())
    payload["hits"]["hits"][0]["_id"] = f"{_accession(0)}:../event.htm"
    with pytest.raises(ValueError, match="schema drift"):
        audit.parse_search_page(json.dumps(payload).encode())


def test_rate_limiter_requires_declared_contact() -> None:
    with pytest.raises(ValueError, match="contact address"):
        audit.RateLimitedFetcher("anonymous-bot", 0.11)
    with pytest.raises(ValueError, match="10 requests/second"):
        audit.RateLimitedFetcher("research contact@example.com", 0.01)
