from __future__ import annotations

import copy
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from training import build_non_sec_federal_exact_term_source as source
from training import preregister_non_sec_federal_exact_term_source as nfet
from training import seal_non_sec_federal_exact_term_source_access as access


PRIMARY_AGENCY = "COMMODITY FUTURES TRADING COMMISSION"
PRIMARY_SLUG = "commodity-futures-trading-commission"
SEC_AGENCY = nfet.SEC_AGENCY_NAME
SEC_SLUG = nfet.SEC_AGENCY_SLUG


def _cfg(tmp_path: Path) -> source.Config:
    root = tmp_path / "nfet-source"
    return source.Config(
        archive_root=root,
        decisions=root / "candidate_decisions.jsonl.gz",
        sec_events=root / "sec_events.jsonl.gz",
        selected_source=root / "selected_primary.jsonl.gz",
        resume_state=root / "resume_state.json",
        source_manifest=tmp_path / "source_manifest.json",
        support_result=tmp_path / "support_result.json",
        request_pause_seconds=0.0,
    )


def _now() -> datetime:
    return datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)


def _candidate(
    document_number: str,
    publication_date: str,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    return {
        "document_number": document_number,
        "publication_date": publication_date,
        "type": "Notice",
        "title": title or f"Synthetic candidate {document_number}",
        "pdf_url": nfet.govinfo_document_urls(publication_date, document_number)["pdf"],
        "queries": ["bitcoin"],
    }


def _mods(document_number: str, publication_date: str, agency_name: str) -> bytes:
    urls = nfet.govinfo_document_urls(publication_date, document_number)
    return f"""<mods xmlns='{nfet.MODS_NAMESPACE}' xmlns:xlink='{nfet.XLINK_NAMESPACE}'>
      <identifier type='FR Doc No.'>{document_number}</identifier>
      <relatedItem type='host'>
        <originInfo><dateIssued>{publication_date}</dateIssued></originInfo>
        <identifier type='uri'>https://www.govinfo.gov/app/details/FR-{publication_date}</identifier>
      </relatedItem>
      <relatedItem type='otherFormat' xlink:href='{urls["html"]}' />
      <relatedItem type='otherFormat' xlink:href='{urls["pdf"]}' />
      <extension>
        <granuleClass>NOTICE</granuleClass>
        <accessId>{document_number}</accessId>
        <frDocNumber>{document_number}</frDocNumber>
        <agency>{agency_name}</agency>
      </extension>
      <classification><classificationIdentifier><collectionCode>FR</collectionCode></classificationIdentifier></classification>
    </mods>""".encode()


def _detail(
    document_number: str,
    publication_date: str,
    agency_slug: str,
    *,
    correction_of: str | None = None,
    corrections: list[str] | None = None,
) -> bytes:
    urls = nfet.govinfo_document_urls(publication_date, document_number)
    return json.dumps(
        {
            "document_number": document_number,
            "publication_date": publication_date,
            "pdf_url": urls["pdf"],
            "agencies": [{"slug": agency_slug, "name": agency_slug.replace("-", " ")}],
            "correction_of": (
                None
                if correction_of is None
                else source.correction_relationship_url(correction_of)
            ),
            "corrections": (
                []
                if corrections is None
                else [
                    source.correction_relationship_url(value) for value in corrections
                ]
            ),
        }
    ).encode()


def _html(text: str) -> bytes:
    return f"<html><head><style>.x{{}}</style></head><body><p>{text}</p></body></html>".encode()


def _pdf(text: bytes = b"archival bitcoin pdf") -> bytes:
    return b"%PDF-1.7\n" + text


def _fetch_result(url: str, body: bytes) -> access.FetchResult:
    return access.FetchResult(
        body=body,
        status=200,
        final_url=url,
        etag=None,
        last_modified=None,
    )


def _responses_for(
    candidate: dict[str, Any],
    *,
    agency_name: str = PRIMARY_AGENCY,
    agency_slug: str = PRIMARY_SLUG,
    html: bytes | None = None,
    pdf: bytes | None = None,
    correction_of: str | None = None,
    corrections: list[str] | None = None,
) -> dict[str, bytes]:
    document_number = candidate["document_number"]
    publication_date = candidate["publication_date"]
    urls = nfet.govinfo_document_urls(publication_date, document_number)
    return {
        urls["mods"]: _mods(document_number, publication_date, agency_name),
        urls["html"]: html
        if html is not None
        else _html("bitcoin official membership"),
        source.detail_url(document_number): _detail(
            document_number,
            publication_date,
            agency_slug,
            correction_of=correction_of,
            corrections=corrections,
        ),
        urls["pdf"]: pdf if pdf is not None else _pdf(),
    }


def _fetcher(responses: dict[str, bytes], calls: list[str] | None = None):
    def fetch(url: str) -> access.FetchResult:
        if calls is not None:
            calls.append(url)
        return _fetch_result(url, responses[url])

    return fetch


def _decision(
    document_number: str,
    publication_date: str,
    *,
    stratum: str = "primary_non_sec",
    agencies: list[str] | None = None,
    member: bool = True,
) -> dict[str, Any]:
    candidate = _candidate(document_number, publication_date)
    raw_sha = f"{document_number.replace('-', ''):0<64}"[:64]
    return {
        "candidate": candidate,
        "member": member,
        "identity_stratum": stratum,
        "member_stratum": stratum if member else None,
        "govinfo_agency_names": agencies or [f"AGENCY {document_number}"],
        "detail_agency_slugs": [
            SEC_SLUG if stratum == "sec_comparator" else PRIMARY_SLUG
        ],
        "exact_matches": [
            {
                "pattern_id": "bitcoin",
                "substring": "bitcoin",
                "span_start": 0,
                "span_end_exclusive": 7,
            }
        ]
        if member
        else [],
        "historical_available_at": source.historical_available_at(publication_date),
        "receipts": {
            "canonical_text": {"object_path": "unused", "raw_sha256": raw_sha},
            "html": {"raw_sha256": raw_sha},
            "mods": {"raw_sha256": raw_sha},
            "detail": {"raw_sha256": raw_sha},
            "pdf": {"raw_sha256": raw_sha, "raw_bytes": 12} if member else None,
            "matches": {"object_path": "unused", "raw_sha256": raw_sha}
            if member
            else None,
        },
    }


def _quality_contract(**primary_overrides: Any) -> dict[str, Any]:
    primary = {
        "minimum_documents": 16,
        "minimum_documents_each_year": 4,
        "minimum_unique_publication_days": 16,
        "minimum_unique_publication_days_each_year": 4,
        "minimum_documents_each_quarter": 1,
        "maximum_month_share": 0.10,
        "maximum_fractional_agency_share": 0.20,
    }
    primary.update(primary_overrides)
    return {
        "primary_non_sec": primary,
        "sec_comparator": {
            "minimum_documents": 4,
            "minimum_documents_each_year": 1,
            "minimum_unique_publication_days": 4,
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


def _quality_decisions() -> list[dict[str, Any]]:
    primary_dates = [
        "2020-01-02",
        "2020-04-02",
        "2020-07-02",
        "2020-10-02",
        "2021-01-02",
        "2021-04-02",
        "2021-07-02",
        "2021-10-02",
        "2022-01-02",
        "2022-04-02",
        "2022-07-02",
        "2022-10-02",
        "2023-01-02",
        "2023-04-02",
        "2023-07-02",
        "2023-10-02",
    ]
    decisions = [
        _decision(f"{date[:4]}-{index:05d}", date, agencies=[f"AGENCY {index}"])
        for index, date in enumerate(primary_dates, start=1)
    ]
    decisions.extend(
        _decision(f"{year}-9{offset:04d}", f"{year}-02-02", stratum="sec_comparator")
        for offset, year in enumerate((2020, 2021, 2022, 2023), start=1)
    )
    return decisions


def test_positive_primary_full_evaluation_retains_identity_matches_and_receipts(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    candidate = _candidate("2020-00001", "2020-01-02")
    decision = source._process_candidate(
        cfg,
        source._state_core(cfg),
        candidate,
        fetch=_fetcher(_responses_for(candidate)),
        now=_now,
    )
    assert decision["member"] is True
    assert decision["identity_stratum"] == "primary_non_sec"
    assert decision["member_stratum"] == "primary_non_sec"
    assert decision["govinfo_agency_names"] == [PRIMARY_AGENCY]
    assert decision["detail_agency_slugs"] == [PRIMARY_SLUG]
    assert decision["exact_matches"] == [
        {
            "pattern_id": "bitcoin",
            "substring": "bitcoin",
            "span_start": 0,
            "span_end_exclusive": 7,
        }
    ]
    assert decision["historical_available_at"] == "2020-01-03T12:00:00+00:00"
    receipts = decision["receipts"]
    assert receipts["pdf"]["kind"] == "pdf"
    assert receipts["matches"]["kind"] == "match_json"
    assert (
        source._validate_decision(cfg, decision, expected_candidate=candidate)
        == decision
    )


def test_sec_agency_routes_to_comparator_not_primary(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    candidate = _candidate("2020-00002", "2020-01-02")
    decision = source._process_candidate(
        cfg,
        source._state_core(cfg),
        candidate,
        fetch=_fetcher(
            _responses_for(candidate, agency_name=SEC_AGENCY, agency_slug=SEC_SLUG)
        ),
        now=_now,
    )
    assert decision["member"] is True
    assert decision["identity_stratum"] == "sec_comparator"
    assert decision["member_stratum"] == "sec_comparator"
    assert source.build_events([decision], stratum="primary_non_sec") == []
    assert (
        source.build_events([decision], stratum="sec_comparator")[0]["source_id"]
        == "NFET_SEC"
    )


def test_negative_html_stays_negative_even_when_unrequested_pdf_contains_term(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    candidate = _candidate("2020-00003", "2020-01-02")
    calls: list[str] = []
    responses = _responses_for(
        candidate, html=_html("no frozen term here"), pdf=_pdf(b"bitcoin only in pdf")
    )
    decision = source._process_candidate(
        cfg,
        source._state_core(cfg),
        candidate,
        fetch=_fetcher(responses, calls),
        now=_now,
    )
    assert decision["member"] is False
    assert decision["member_stratum"] is None
    assert decision["exact_matches"] == []
    assert decision["receipts"]["pdf"] is None
    assert nfet.govinfo_document_urls("2020-01-02", "2020-00003")["pdf"] not in calls


def test_bad_positive_pdf_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    candidate = _candidate("2020-00004", "2020-01-02")
    with pytest.raises(ValueError, match="PDF is missing or malformed"):
        source._process_candidate(
            cfg,
            source._state_core(cfg),
            candidate,
            fetch=_fetcher(_responses_for(candidate, pdf=b"not a pdf with bitcoin")),
            now=_now,
        )


def test_all_candidates_retain_identity_but_relationships_stay_in_raw_audit(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    positive = _candidate("2021-00001", "2021-04-05")
    negative = _candidate("2021-00002", "2021-04-06")
    state = source._state_core(cfg)
    positive_decision = source._process_candidate(
        cfg,
        state,
        positive,
        fetch=_fetcher(
            _responses_for(
                positive, correction_of="2020-99999", corrections=["2021-11111"]
            )
        ),
        now=_now,
    )
    negative_decision = source._process_candidate(
        cfg,
        state,
        negative,
        fetch=_fetcher(
            _responses_for(
                negative,
                html=_html("ordinary agency notice"),
                corrections=["2021-22222"],
            )
        ),
        now=_now,
    )
    assert positive_decision["historical_available_at"] == "2021-04-06T12:00:00+00:00"
    assert negative_decision["historical_available_at"] == "2021-04-07T12:00:00+00:00"
    assert "correction_of" not in positive_decision
    assert "corrections" not in positive_decision
    assert "correction_of" not in negative_decision
    assert "corrections" not in negative_decision
    positive_detail = json.loads(
        source.load_object(
            positive_decision["receipts"]["detail"], archive_root=cfg.archive_root
        )
    )
    assert positive_detail["correction_of"] == source.correction_relationship_url(
        "2020-99999"
    )
    assert positive_detail["corrections"] == [
        source.correction_relationship_url("2021-11111")
    ]
    assert (
        positive_decision["identity_stratum"]
        == negative_decision["identity_stratum"]
        == "primary_non_sec"
    )


@pytest.mark.parametrize(
    "correction_of,corrections,match",
    [
        (["not-a-url"], [], "correction_of"),
        ("https://example.invalid/document", [], "correction_of"),
        (None, {"future": "2029-99999"}, "not a list"),
        (None, [123], "non-string"),
        (
            None,
            [
                source.correction_relationship_url("2021-00001"),
                source.correction_relationship_url("2021-00001"),
            ],
            "repeats",
        ),
    ],
)
def test_correction_metadata_rejects_schema_drift(
    correction_of: object, corrections: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        source.normalize_correction_metadata(
            {"correction_of": correction_of, "corrections": corrections},
            current_document_number="2021-99999",
        )


def test_official_correction_relationship_schema_accepts_c_prefix_without_json() -> (
    None
):
    correction_of, corrections = source.normalize_correction_metadata(
        {
            "correction_of": source.correction_relationship_url("2023-25026"),
            "corrections": [source.correction_relationship_url("C1-2023-25026")],
        },
        current_document_number="2023-99999",
    )
    assert correction_of == (
        "https://www.federalregister.gov/api/v1/documents/2023-25026"
    )
    assert corrections == [
        "https://www.federalregister.gov/api/v1/documents/C1-2023-25026"
    ]


def test_forward_corrections_remain_only_in_raw_detail_audit_object(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    candidate = _candidate("2020-00001", "2020-01-02")
    decision = source._process_candidate(
        cfg,
        source._state_core(cfg),
        candidate,
        fetch=_fetcher(_responses_for(candidate, corrections=["2029-99999"])),
        now=_now,
    )
    assert "corrections" not in decision
    detail_raw = source.load_object(
        decision["receipts"]["detail"], archive_root=cfg.archive_root
    )
    assert json.loads(detail_raw)["corrections"] == [
        source.correction_relationship_url("2029-99999")
    ]
    assert (
        "corrections"
        not in source.build_events([decision], stratum="primary_non_sec")[0][
            "documents"
        ][0]
    )


def test_event_same_date_dedup_groups_documents_and_keeps_sec_separate() -> None:
    primary_a = _decision("2020-00001", "2020-01-02")
    primary_b = _decision("2020-00002", "2020-01-02")
    sec = _decision("2020-00003", "2020-01-02", stratum="sec_comparator")
    primary_events = source.build_events(
        [primary_b, sec, primary_a], stratum="primary_non_sec"
    )
    sec_events = source.build_events(
        [primary_b, sec, primary_a], stratum="sec_comparator"
    )
    assert len(primary_events) == 1
    assert primary_events[0]["event_id"] == "primary_non_sec:2020-01-02"
    assert primary_events[0]["document_count"] == 2
    assert [doc["document_number"] for doc in primary_events[0]["documents"]] == [
        "2020-00001",
        "2020-00002",
    ]
    assert len(sec_events) == 1
    assert sec_events[0]["event_id"] == "sec_comparator:2020-01-02"
    assert sec_events[0]["source_id"] == "NFET_SEC"
    assert sec_events[0]["document_count"] == 1


def test_quality_gate_pass_and_specific_count_month_agency_failures() -> None:
    passing = source.evaluate_source_quality(_quality_decisions(), _quality_contract())
    assert passing["status"] == "PASS"
    assert passing["failed_gates"] == []
    count_failure = source.evaluate_source_quality(
        _quality_decisions()[:-1], _quality_contract()
    )
    assert count_failure["status"] == "REJECT"
    assert "sec.minimum_documents" in count_failure["failed_gates"]
    month_clustered = [
        _decision(f"2020-{index:05d}", "2020-01-02", agencies=[f"AGENCY {index}"])
        for index in range(1, 17)
    ] + [
        _decision(f"{year}-9{offset:04d}", f"{year}-02-02", stratum="sec_comparator")
        for offset, year in enumerate((2020, 2021, 2022, 2023), start=1)
    ]
    month_failure = source.evaluate_source_quality(
        month_clustered,
        _quality_contract(
            minimum_documents_each_year=0,
            minimum_unique_publication_days=1,
            minimum_unique_publication_days_each_year=0,
            minimum_documents_each_quarter=0,
        ),
    )
    assert "primary.maximum_month_share" in month_failure["failed_gates"]
    one_agency = copy.deepcopy(_quality_decisions())
    for row in one_agency:
        if row["member_stratum"] == "primary_non_sec":
            row["govinfo_agency_names"] = ["DOMINANT AGENCY"]
    agency_failure = source.evaluate_source_quality(one_agency, _quality_contract())
    assert "primary.maximum_fractional_agency_share" in agency_failure["failed_gates"]


def test_content_addressed_gzip_is_deterministic_and_tamper_evident(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    raw = b"official retained bytes"
    first = source.store_object(cfg.archive_root, raw, kind="html_raw")
    second = source.store_object(cfg.archive_root, raw, kind="html_raw")
    assert first == second
    encoded = Path(first["object_path"]).read_bytes()
    assert encoded[4:8] == b"\x00\x00\x00\x00"
    assert gzip.decompress(encoded) == raw
    assert source.load_object(first, archive_root=cfg.archive_root) == raw
    Path(first["object_path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="gzip hash mismatch"):
        source.load_object(first, archive_root=cfg.archive_root)


def test_derived_receipt_binds_parser_kind_and_source_hash(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    upstream = source.store_object(
        cfg.archive_root, b"<html>bitcoin</html>", kind="html_raw"
    )
    receipt = source._derived_receipt(
        cfg,
        b"bitcoin",
        kind="canonical_text",
        derived_from_raw_sha256=upstream["raw_sha256"],
    )
    assert (
        source._validate_derived_receipt(
            receipt,
            expected_kind="canonical_text",
            expected_source_sha256=upstream["raw_sha256"],
            archive_root=cfg.archive_root,
        )
        == b"bitcoin"
    )
    tampered = dict(receipt, derived_from_raw_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="derived receipt binding changed"):
        source._validate_derived_receipt(
            tampered,
            expected_kind="canonical_text",
            expected_source_sha256=upstream["raw_sha256"],
            archive_root=cfg.archive_root,
        )
    tampered = dict(receipt, parser_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="derived receipt binding changed"):
        source._validate_derived_receipt(
            tampered,
            expected_kind="canonical_text",
            expected_source_sha256=upstream["raw_sha256"],
            archive_root=cfg.archive_root,
        )


def test_network_receipt_resume_skips_fetch_and_detects_tamper(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    state = source._load_state(cfg)
    url = nfet.govinfo_document_urls("2020-01-02", "2020-00001")["mods"]
    raw = b"official MODS bytes"
    loaded, receipt = source._load_or_fetch(
        cfg,
        state,
        url=url,
        kind="mods_xml",
        fetch=lambda request_url: _fetch_result(request_url, raw),
        now=_now,
    )
    assert loaded == raw
    resumed = source._load_state(cfg, allow_existing=True)
    assert (
        source._load_or_fetch(
            cfg,
            resumed,
            url=url,
            kind="mods_xml",
            fetch=lambda request_url: pytest.fail(f"unexpected fetch {request_url}"),
            now=_now,
        )[0]
        == raw
    )
    Path(receipt["object_path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="gzip hash mismatch"):
        source._load_or_fetch(
            cfg,
            source._load_state(cfg, allow_existing=True),
            url=url,
            kind="mods_xml",
            fetch=lambda request_url: pytest.fail(f"unexpected fetch {request_url}"),
            now=_now,
        )


def test_preexisting_resume_requires_explicit_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    source._write_state(cfg, source._state_core(cfg))
    candidate = _candidate("2020-00001", "2020-01-02")
    monkeypatch.setattr(
        source,
        "_load_inputs",
        lambda: ({"source_quality_gates": _quality_contract()}, [candidate]),
    )
    with pytest.raises(RuntimeError, match="explicit resume authorization"):
        source.build(
            cfg,
            fetch=lambda url: pytest.fail(f"unexpected fetch {url}"),
            now=_now,
        )


def test_resume_rejects_receipt_from_the_future(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    state = source._load_state(cfg)
    url = nfet.govinfo_document_urls("2020-01-02", "2020-00001")["mods"]
    source._load_or_fetch(
        cfg,
        state,
        url=url,
        kind="mods_xml",
        fetch=lambda request_url: _fetch_result(request_url, b"official MODS bytes"),
        now=_now,
    )
    stored = json.loads(cfg.resume_state.read_text())
    stored["responses"][url]["fetched_at"] = "2027-01-01T00:00:00+00:00"
    core = {key: value for key, value in stored.items() if key != "state_hash"}
    stored["state_hash"] = source.canonical_hash(core)
    cfg.resume_state.write_text(json.dumps(stored))
    resumed = source._load_state(cfg, allow_existing=True)
    with pytest.raises(RuntimeError, match="after the resume clock"):
        source._load_or_fetch(
            cfg,
            resumed,
            url=url,
            kind="mods_xml",
            fetch=lambda request_url: pytest.fail(f"unexpected fetch {request_url}"),
            now=_now,
        )


def test_negative_runtime_never_requests_pdf_when_html_membership_is_feasible(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    candidate = _candidate("2022-00001", "2022-05-06")
    calls: list[str] = []
    urls = nfet.govinfo_document_urls(
        candidate["publication_date"], candidate["document_number"]
    )
    responses = _responses_for(
        candidate, html=_html("visible text lacks the frozen terms")
    )
    del responses[urls["pdf"]]
    decision = source._process_candidate(
        cfg,
        source._state_core(cfg),
        candidate,
        fetch=_fetcher(responses, calls),
        now=_now,
    )
    assert decision["member"] is False
    assert calls == [
        urls["mods"],
        urls["html"],
        source.detail_url(candidate["document_number"]),
    ]


def test_build_with_synthetic_inputs_writes_manifest_support_and_avoids_actual_source_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    candidates = [
        _candidate("2023-00001", "2023-01-03"),
        _candidate("2023-00002", "2023-01-03"),
        _candidate("2023-00003", "2023-01-04"),
    ]
    protocol = {
        "source_quality_gates": _quality_contract(
            minimum_documents=1,
            minimum_documents_each_year=0,
            minimum_unique_publication_days=1,
            minimum_unique_publication_days_each_year=0,
            minimum_documents_each_quarter=0,
            maximum_month_share=1.0,
            maximum_fractional_agency_share=1.0,
        )
    }
    protocol["source_quality_gates"]["sec_comparator"].update(
        {
            "minimum_documents": 1,
            "minimum_documents_each_year": 0,
            "minimum_unique_publication_days": 1,
        }
    )
    monkeypatch.setattr(source, "_load_inputs", lambda: (protocol, candidates))
    responses: dict[str, bytes] = {}
    responses.update(_responses_for(candidates[0]))
    responses.update(
        _responses_for(candidates[1], agency_name=SEC_AGENCY, agency_slug=SEC_SLUG)
    )
    responses.update(
        _responses_for(
            candidates[2],
            html=_html("not an exact member"),
            pdf=_pdf(b"bitcoin ignored"),
        )
    )
    calls: list[str] = []
    manifest, support = source.build(cfg, fetch=_fetcher(responses, calls), now=_now)
    assert manifest["candidate_count"] == 3
    assert manifest["incidence"] == {
        "exact_members": 2,
        "primary_non_sec_documents": 1,
        "sec_comparator_documents": 1,
        "nonmembers": 1,
    }
    assert support["status"] == "PASS"
    assert manifest["retention_contract"]["negative_pdf_requests"] == 0
    assert nfet.govinfo_document_urls("2023-01-04", "2023-00003")["pdf"] not in calls
    decision_rows = [
        json.loads(line)
        for line in gzip.decompress(cfg.decisions.read_bytes()).splitlines()
    ]
    assert [row["candidate"] for row in decision_rows] == candidates
    assert decision_rows[2]["receipts"]["pdf"] is None
    assert source.build(
        cfg, fetch=lambda url: pytest.fail(f"unexpected fetch {url}"), now=_now
    ) == (manifest, support)
