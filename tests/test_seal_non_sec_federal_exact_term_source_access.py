from __future__ import annotations

import copy
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from training import preregister_non_sec_federal_exact_term_source as nfet
from training import seal_non_sec_federal_exact_term_source_access as access


def _issue_xml(year: int) -> bytes:
    return f"""<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
      <url><loc>https://www.govinfo.gov/app/details/FR-{year}-01-02</loc>
      <lastmod>2026-01-01T00:00:00Z</lastmod><changefreq>monthly</changefreq>
      <priority>1.0</priority></url>
    </urlset>""".encode()


def _candidate(query_index: int) -> dict[str, Any]:
    year = 2020 + query_index % 4
    publication_date = f"{year}-01-02"
    document_number = f"{year}-{query_index + 1:05d}"
    return {
        "document_number": document_number,
        "publication_date": publication_date,
        "type": "Rule",
        "title": f"Frozen candidate {query_index}",
        "pdf_url": nfet.govinfo_document_urls(publication_date, document_number)["pdf"],
        "agencies": [
            {
                "slug": "commodity-futures-trading-commission",
                "name": "Commodity Futures Trading Commission",
            }
        ],
        "excerpts": "Search-index text is retained raw but not promoted.",
    }


def _search_page(rows: list[dict[str, Any]], *, count: int | None = None) -> bytes:
    total = len(rows) if count is None else count
    return json.dumps(
        {
            "description": "fixture",
            "count": total,
            "total_pages": (total + 999) // 1000,
            "results": rows,
            "next_page_url": "https://example.invalid/must-not-follow",
        }
    ).encode()


def _response_map() -> dict[str, bytes]:
    responses = {
        access.issue_inventory_url(year): _issue_xml(year)
        for year in (2020, 2021, 2022, 2023)
    }
    for index, query in enumerate(nfet.TERM_QUERIES):
        responses[access.candidate_query_url(query, 1)] = _search_page(
            [_candidate(index)]
        )
    return responses


def _config(tmp_path: Path) -> access.Config:
    root = tmp_path / "nfet"
    return access.Config(
        archive_root=root,
        candidate_index=root / "candidate_index.jsonl.gz",
        resume_state=root / "resume_state.json",
        access_seal=tmp_path / "access_seal.json",
        request_pause_seconds=0.0,
    )


def _fixed_now() -> datetime:
    return datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def _fetch_result(url: str, body: bytes) -> access.FetchResult:
    return access.FetchResult(
        body=body,
        status=200,
        final_url=url,
        etag=None,
        last_modified=None,
    )


def test_query_url_freezes_parameter_order_and_integer_page() -> None:
    url = access.candidate_query_url("bitcoin", 2)
    assert url == (
        "https://www.federalregister.gov/api/v1/documents.json?"
        "conditions%5Bterm%5D=bitcoin&"
        "conditions%5Bpublication_date%5D%5Bgte%5D=2020-01-01&"
        "conditions%5Bpublication_date%5D%5Blte%5D=2023-12-31&"
        "order=oldest&per_page=1000&page=2"
    )
    with pytest.raises(ValueError, match="escaped"):
        access.candidate_query_url("not-frozen", 1)


def test_search_page_ignores_arbitrary_next_url_and_reconciles_counts() -> None:
    rows = [_candidate(0)]
    count, pages, parsed = access.parse_search_page(_search_page(rows))
    assert (count, pages, parsed) == (1, 1, rows)
    with pytest.raises(ValueError, match="positive count"):
        access.parse_search_page(
            json.dumps({"count": 0, "total_pages": 0, "results": []}).encode()
        )
    with pytest.raises(ValueError, match="total_pages"):
        access.parse_search_page(
            json.dumps({"count": 1, "total_pages": 2, "results": rows}).encode()
        )


def test_reconcile_query_pages_requires_every_integer_page() -> None:
    first = [{"document_number": f"2020-{index:05d}"} for index in range(1000)]
    second = [{"document_number": "2020-01000"}]
    count, rows = access.reconcile_query_pages([(1001, 2, first), (1001, 2, second)])
    assert count == 1001
    assert len(rows) == 1001
    with pytest.raises(ValueError, match="incomplete"):
        access.reconcile_query_pages([(1001, 2, first)])
    with pytest.raises(ValueError, match="row count"):
        access.reconcile_query_pages([(1001, 2, first), (1001, 2, [])])


def test_reconcile_query_pages_rejects_a_repeated_full_page() -> None:
    first = [{"document_number": f"2020-{index:05d}"} for index in range(1000)]
    with pytest.raises(ValueError, match="repeats a document number"):
        access.reconcile_query_pages([(2000, 2, first), (2000, 2, first)])


def test_candidate_union_rejects_conflicting_duplicate() -> None:
    issue_dates = frozenset({"2020-01-02"})
    first = access.normalize_candidate(
        _candidate(0), query="bitcoin", issue_dates=issue_dates
    )
    changed = dict(first)
    changed["queries"] = ["blockchain"]
    changed["title"] = "Conflicting title"
    candidates: dict[str, dict[str, Any]] = {}
    access.merge_candidate(candidates, first)
    with pytest.raises(ValueError, match="conflicts on title"):
        access.merge_candidate(candidates, changed)


def test_candidate_date_must_exist_in_issue_inventory() -> None:
    with pytest.raises(ValueError, match="identity or metadata"):
        access.normalize_candidate(
            _candidate(0), query="bitcoin", issue_dates=frozenset()
        )


def test_content_addressed_gzip_is_deterministic_and_tamper_evident(
    tmp_path: Path,
) -> None:
    raw = b'{"source":"official"}\n'
    first = access.store_content_addressed(tmp_path, raw, kind="search_json")
    second = access.store_content_addressed(tmp_path, raw, kind="search_json")
    assert first == second
    encoded = Path(first["object_path"]).read_bytes()
    assert encoded[4:8] == b"\x00\x00\x00\x00"
    assert gzip.decompress(encoded) == raw
    assert access.load_content_addressed(first) == raw
    Path(first["object_path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="gzip hash"):
        access.load_content_addressed(first)


def test_builder_seals_candidate_envelope_without_membership_or_outcomes(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    responses = _response_map()
    calls: list[str] = []

    def fetch(url: str) -> access.FetchResult:
        calls.append(url)
        return _fetch_result(url, responses[url])

    result = access.build(cfg, fetch=fetch, now=_fixed_now)
    access.validate_access_seal(result)
    assert result["candidate_count"] == len(nfet.TERM_QUERIES)
    assert result["candidate_envelope_opened"] is True
    assert result["exact_official_membership_evaluated"] is False
    assert result["market_clocks_opened"] is False
    assert result["outcomes_opened"] is False
    assert len(calls) == 4 + len(nfet.TERM_QUERIES)
    assert all(receipt["http"]["status"] == 200 for receipt in result["responses"])
    assert all(
        set(receipt["http"]) == {"status", "final_url", "etag", "last_modified"}
        for receipt in result["responses"]
    )
    assert all(receipt["http"]["etag"] is None for receipt in result["responses"])
    assert not cfg.resume_state.exists()
    rows = [
        json.loads(line)
        for line in gzip.decompress(cfg.candidate_index.read_bytes()).splitlines()
    ]
    assert len(rows) == len(nfet.TERM_QUERIES)
    assert all("exact_matches" not in row for row in rows)

    def fail_fetch(url: str) -> access.FetchResult:
        raise AssertionError(f"sealed rerun fetched {url}")

    cfg.resume_state.write_text("stale post-seal state")
    assert access.build(cfg, fetch=fail_fetch, now=_fixed_now) == result
    assert not cfg.resume_state.exists()


def test_existing_seal_rejects_rehashed_semantic_contamination(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    responses = _response_map()
    result = access.build(
        cfg,
        fetch=lambda url: _fetch_result(url, responses[url]),
        now=_fixed_now,
    )
    drifted = copy.deepcopy(result)
    drifted["source_contract"]["queries"] = []
    drifted_core = {
        key: value for key, value in drifted.items() if key != "manifest_hash"
    }
    drifted["manifest_hash"] = access.canonical_hash(drifted_core)
    with pytest.raises(RuntimeError, match="source contract"):
        access.validate_access_seal(drifted, cfg=cfg)

    contaminated = copy.deepcopy(result)
    contaminated["forbidden_sources_opened"] = ["BTC future returns"]
    core = {key: value for key, value in contaminated.items() if key != "manifest_hash"}
    contaminated["manifest_hash"] = access.canonical_hash(core)
    cfg.access_seal.write_text(json.dumps(contaminated))
    with pytest.raises(RuntimeError, match="crossed the source boundary"):
        access.build(cfg, fetch=lambda url: _fetch_result(url, responses[url]))


def test_rehashed_off_envelope_receipt_is_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    responses = _response_map()
    result = access.build(
        cfg,
        fetch=lambda url: _fetch_result(url, responses[url]),
        now=_fixed_now,
    )
    changed = copy.deepcopy(result)
    changed["responses"][0]["url"] = "https://www.govinfo.gov/off-envelope"
    changed["responses"][0]["http"]["final_url"] = changed["responses"][0]["url"]
    core = {key: value for key, value in changed.items() if key != "manifest_hash"}
    changed["manifest_hash"] = access.canonical_hash(core)
    with pytest.raises(RuntimeError, match="response envelope"):
        access.validate_access_seal(changed, cfg=cfg)


def test_http_fetch_rejects_same_host_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResponse:
        status = 200
        headers = {
            "ETag": '"fixture"',
            "Last-Modified": "Mon, 20 Jul 2026 12:00:00 GMT",
        }

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://www.govinfo.gov/same-host-but-off-envelope"

        def read(self) -> bytes:
            return b"official bytes"

    monkeypatch.setattr(
        access.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(),
    )
    cfg = access.Config(
        archive_root=tmp_path,
        maximum_retries=0,
        request_pause_seconds=0.0,
    )
    url = access.issue_inventory_url(2020)
    with pytest.raises(RuntimeError, match="failed after retries") as caught:
        access._http_fetch(cfg, url)
    assert "redirected outside the frozen URL" in str(caught.value.__cause__)


def test_http_fetch_records_explicit_null_when_validators_are_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = access.candidate_query_url("bitcoin", 1)

    class FakeResponse:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def geturl(self) -> str:
            return url

        def read(self) -> bytes:
            return b'{"count":1}'

    monkeypatch.setattr(
        access.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(),
    )
    cfg = access.Config(
        archive_root=tmp_path,
        maximum_retries=0,
        request_pause_seconds=0.0,
    )
    response = access._http_fetch(cfg, url)
    assert response.final_url == url
    assert response.etag is None
    assert response.last_modified is None


def test_interrupted_build_resumes_only_after_hash_verifying_receipt(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    responses = _response_map()
    ordered = [access.issue_inventory_url(year) for year in (2020, 2021, 2022, 2023)]
    first_calls: list[str] = []

    def interrupt(url: str) -> access.FetchResult:
        first_calls.append(url)
        if len(first_calls) == 2:
            raise RuntimeError("simulated interruption")
        return _fetch_result(url, responses[url])

    with pytest.raises(RuntimeError, match="simulated interruption"):
        access.build(cfg, fetch=interrupt, now=_fixed_now)
    assert cfg.resume_state.exists()
    assert first_calls[:2] == ordered[:2]

    resumed_calls: list[str] = []

    def resume(url: str) -> access.FetchResult:
        resumed_calls.append(url)
        return _fetch_result(url, responses[url])

    result = access.build(cfg, fetch=resume, now=_fixed_now)
    assert result["candidate_count"] == len(nfet.TERM_QUERIES)
    assert ordered[0] not in resumed_calls
    assert ordered[1] in resumed_calls


def test_corrupted_resume_blob_halts_before_refetch(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    responses = _response_map()
    calls = 0

    def interrupt(url: str) -> access.FetchResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stop")
        return _fetch_result(url, responses[url])

    with pytest.raises(RuntimeError, match="stop"):
        access.build(cfg, fetch=interrupt, now=_fixed_now)
    state = json.loads(cfg.resume_state.read_text())
    first_receipt = next(iter(state["responses"].values()))
    Path(first_receipt["object_path"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="gzip hash"):
        access.build(
            cfg,
            fetch=lambda url: _fetch_result(url, responses[url]),
            now=_fixed_now,
        )


def test_disk_guard_aborts_at_frozen_limit_before_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Usage:
        used = 300 * 1024**3

    monkeypatch.setattr(access.shutil, "disk_usage", lambda path: Usage())
    with pytest.raises(RuntimeError, match="300 GiB"):
        access.ensure_disk_budget(tmp_path, abort_gib=300)
