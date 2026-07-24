from __future__ import annotations

import ast
from collections import namedtuple
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pytest

from training import (
    audit_federal_reserve_deliberation_communication_source as audit,
)


ROOT = Path(__file__).resolve().parents[1]


def _page(article: str, last_update: str = "January 25, 2012") -> bytes:
    return (
        "<!doctype html><html><body>"
        f"{article}"
        f'<div id="lastUpdate">Last Update: {last_update}</div>'
        "</body></html>"
    ).encode("utf-8")


def _article(body: str) -> str:
    return f'<div id="article">{body}</div>'


def _payload(
    intent: audit.RequestIntent,
    *,
    status: int = 200,
    raw: bytes = b"<html></html>",
) -> audit.HttpPayload:
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    return audit.HttpPayload(
        intent=intent,
        status=status,
        headers=(
            ("Content-Type", "text/html; charset=UTF-8"),
            ("Content-Length", str(len(raw))),
        ),
        raw=raw,
        request_started_at_utc=now,
        response_completed_at_utc=now,
    )


def _paths(tmp_path: Path) -> audit.AuditPaths:
    return audit.AuditPaths(
        sentinel=tmp_path / "attempt.started",
        manifest=tmp_path / "attempt.manifest.ndjson",
        raw_dir=tmp_path / "raw-artifacts",
        report=tmp_path / "report.json",
    )


def test_frozen_protocol_files_and_hashes_match() -> None:
    boundary = ROOT / audit.BOUNDARY_PATH
    ledger = ROOT / audit.LEDGER_PATH
    assert audit.sha256_file(boundary) == audit.BOUNDARY_SHA256
    assert audit.sha256_file(ledger) == audit.LEDGER_SHA256
    assert ledger.stat().st_size == audit.LEDGER_BYTES
    assert audit.INDEX_YEARS == tuple(range(2011, 2021))
    assert len(audit.INDEX_URLS) == 10
    assert all("fomccalendars" not in url for url in audit.INDEX_URLS)


def test_committed_ledger_replays_exact_contract() -> None:
    rows = audit.load_ledger()
    assert len(rows) == 147
    assert sum(row.source_eligible for row in rows) == 145
    assert rows[0].meeting_date == date(2011, 12, 13)
    assert rows[0].release_date == date(2012, 1, 3)
    assert rows[-1].release_date == date(2020, 12, 16)
    assert all(row.release_date.year <= 2020 for row in rows)
    assert all("beige" not in row.official_url.lower() for row in rows)


def test_causal_clock_is_next_day_0005_new_york_across_dst() -> None:
    assert audit.available_at_utc(date(2020, 1, 29)) == datetime(
        2020, 1, 30, 5, 5, tzinfo=timezone.utc
    )
    assert audit.available_at_utc(date(2020, 7, 29)) == datetime(
        2020, 7, 30, 4, 5, tzinfo=timezone.utc
    )


def test_historical_index_parser_extracts_only_frozen_classes_and_window() -> None:
    raw = b"""
    <html><body>
      <p><a href="/newsevents/pressreleases/monetary20120125a.htm">
        <span>Statement</span></a></p>
      <p><a href="/monetarypolicy/fomcminutes20120125.htm">Minutes</a><br>
        (Released February 15, 2012)</p>
      <p><a href="/newsevents/pressreleases/monetary20120125a1.htm">
        Implementation Note</a></p>
      <p><a href="/monetarypolicy/fomcminutes20121212.htm">Minutes</a>
        (Released January 3, 2013)</p>
    </body></html>
    """
    rows = audit.parse_historical_index(
        raw,
        index_url=(
            "https://www.federalreserve.gov/monetarypolicy/"
            "fomchistorical2012.htm"
        ),
    )
    assert [
        (
            row.document_class,
            row.meeting_date.isoformat(),
            row.release_date.isoformat(),
        )
        for row in rows
    ] == [
        ("fomc_statement", "2012-01-25", "2012-01-25"),
        ("fomc_minutes", "2012-01-25", "2012-02-15"),
        ("fomc_minutes", "2012-12-12", "2013-01-03"),
    ]


def test_historical_index_parser_uses_release_window_not_meeting_window() -> None:
    raw = b"""
    <html><body>
      <p><a href="/newsevents/pressreleases/monetary20111213a.htm">
        Statement</a></p>
      <p><a href="/monetarypolicy/fomcminutes20111213.htm">Minutes</a>
        (Released January 3, 2012)</p>
    </body></html>
    """
    rows = audit.parse_historical_index(
        raw,
        index_url=(
            "https://www.federalreserve.gov/monetarypolicy/"
            "fomchistorical2011.htm"
        ),
    )
    assert len(rows) == 1
    assert rows[0].document_class == "fomc_minutes"
    assert rows[0].meeting_date == date(2011, 12, 13)
    assert rows[0].release_date == date(2012, 1, 3)


@pytest.mark.parametrize(
    "raw, message",
    [
        (
            b'<p><a href="/monetarypolicy/fomcminutes20120125.htm">'
            b"Minutes</a></p>",
            "release date",
        ),
        (
            b'<p><a href="/monetarypolicy/fomcminutes20120125.htm">'
            b"Minute</a> (Released February 15, 2012)</p>",
            "label",
        ),
        (
            b'<p><a href="/newsevents/pressreleases/monetary20120125a.htm">'
            b"Press Release</a></p>",
            "label",
        ),
        (
            b'<a href="/monetarypolicy/fomcminutes20120125.htm">'
            b"Minutes</a>",
            "outside",
        ),
    ],
)
def test_historical_index_parser_fails_closed(
    raw: bytes,
    message: str,
) -> None:
    with pytest.raises(audit.IndexError, match=message):
        audit.parse_historical_index(
            raw,
            index_url=(
                "https://www.federalreserve.gov/monetarypolicy/"
                "fomchistorical2012.htm"
            ),
        )


def test_page_metadata_requires_one_article_and_one_exact_last_update() -> None:
    metadata = audit.parse_page_metadata(
        _page(_article("<p>Opaque body</p>")).decode()
    )
    assert metadata == audit.PageMetadata(
        article_count=1,
        last_update_date=date(2012, 1, 25),
    )
    with pytest.raises(audit.PageMetadataError, match="one closed lastUpdate"):
        audit.parse_page_metadata(_article("<p>x</p>"))
    with pytest.raises(audit.PageMetadataError, match="exactly one article"):
        audit.parse_page_metadata(
            '<div id="lastUpdate">Last Update: January 25, 2012</div>'
        )
    with pytest.raises(audit.PageMetadataError, match="one closed lastUpdate"):
        audit.parse_page_metadata(
            _page(_article("<p>x</p>")).decode()
            + '<div id="lastUpdate">Last Update: January 25, 2012</div>'
        )


def test_article_canonicalization_is_exact_and_hash_frozen() -> None:
    text = _article(
        "<h2>Cafe\u0301&nbsp;Policy</h2>"
        "<p>Maintain <strong>rates</strong>.\u200b<br/>"
        'Second <a href="/x"><em>line</em></a>.</p>'
        "<nav><p>DROP-NAV</p></nav>"
        "<table>"
        "<tr><th>A</th><th>B</th></tr>"
        "<tr><td>1</td><td>2</td></tr>"
        "</table>"
        '<div id="lastUpdate"><p>DROP-FOOTER</p></div>'
    )
    canonical, blocks = audit.canonicalize_article(text)
    expected = (
        "Caf\u00e9 Policy\n"
        "Maintain rates.\n"
        "Second line.\n"
        "A\n"
        "B\n"
        "1\n"
        "2\n"
    )
    assert canonical == expected
    assert blocks == 7
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        "31e7cb0bcc4d79f3b60a3acd2a3b0fc127b8639d400346c77ecd5127d834aee1"
    )


def test_article_nested_blocks_preserve_dom_order() -> None:
    canonical, blocks = audit.canonicalize_article(
        _article("<ul><li>before<p>inside</p>after</li></ul>")
    )
    assert canonical == "before\ninside\nafter\n"
    assert blocks == 3


@pytest.mark.parametrize(
    "text, message",
    [
        (_article("<p>x</div>"), "nesting"),
        (_article("<mark>x</mark>"), "unknown"),
        (_article("direct text<p>x</p>"), "outside"),
        (_article("<table><td>x</td></table>"), "outside a row"),
        (
            _article("<table><tr><td><table></table></td></tr></table>"),
            "nested table",
        ),
        (
            '<div id="article"><div id="article"><p>x</p></div></div>',
            "repeats",
        ),
        (
            '<div id="article"><p>x</p></div>'
            '<div id="article"><p>y</p></div>',
            "exactly one",
        ),
    ],
)
def test_article_parser_rejects_grammar_drift(
    text: str,
    message: str,
) -> None:
    with pytest.raises(audit.ArticleError, match=message):
        audit.canonicalize_article(text)


def test_quarantine_never_calls_article_canonicalizer() -> None:
    row = next(row for row in audit.load_ledger() if not row.source_eligible)
    raw = _page(
        _article("<p>THIS MUST STAY OPAQUE</p>"),
        last_update=row.last_update_date.strftime("%B %-d, %Y"),
    )
    calls: list[str] = []

    def forbidden(_: str) -> tuple[str, int]:
        calls.append("called")
        raise AssertionError("quarantine canonicalizer was called")

    summary, canonical = audit.summarize_document(
        row,
        raw,
        canonicalizer=forbidden,
    )
    assert calls == []
    assert canonical is None
    assert summary.source_eligible is False
    assert summary.canonical_sha256 is None


def test_strict_frozen_decoding_has_no_fallback_or_replacement() -> None:
    assert audit.decode_document(b"policy \x96 restraint", encoding="windows-1252") == (
        "policy \u2013 restraint"
    )
    with pytest.raises(audit.PageMetadataError, match="strict"):
        audit.decode_document(b"policy \x96 restraint", encoding="utf-8")
    with pytest.raises(audit.PageMetadataError, match="unauthorized"):
        audit.decode_document(b"policy", encoding="latin-1")


@pytest.mark.parametrize(
    "url",
    [
        "http://www.federalreserve.gov/monetarypolicy/fomchistorical2012.htm",
        "https://federalreserve.gov/monetarypolicy/fomchistorical2012.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomchistorical2021.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20210127.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20201216.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20200129.htm?x=1",
        "https://www.federalreserve.gov/monetarypolicy/%66omcminutes20200129.htm",
    ],
)
def test_get_intent_rejects_nonfrozen_urls(url: str) -> None:
    with pytest.raises(audit.TransportError):
        audit.make_get_intent(url)


def test_document_intent_is_exactly_ledger_bound() -> None:
    ledger = audit.load_ledger()
    row = ledger[0]
    intent = audit.make_document_intent(row)
    assert intent.url == row.official_url
    postwindow = audit.LedgerRow(
        document_class="fomc_minutes",
        meeting_date=date(2020, 12, 16),
        release_date=date(2021, 1, 6),
        last_update_date=date(2021, 1, 6),
        encoding="utf-8",
        source_eligible=True,
        official_url=(
            "https://www.federalreserve.gov/monetarypolicy/"
            "fomcminutes20201216.htm"
        ),
        index_url=(
            "https://www.federalreserve.gov/monetarypolicy/"
            "fomchistorical2020.htm"
        ),
    )
    with pytest.raises(audit.TransportError, match="absent"):
        audit.make_document_intent(postwindow)


class _RetryTransport:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.calls = 0

    def request(self, intent: audit.RequestIntent) -> audit.HttpPayload:
        status = self.statuses[self.calls]
        self.calls += 1
        return _payload(intent, status=status)


def test_retry_scope_and_fixed_delays_are_exact() -> None:
    intent = audit.make_get_intent(audit.INDEX_URLS[0])
    transport = _RetryTransport([503, 429, 200])
    sleeps: list[float] = []
    responses: list[tuple[int, int]] = []
    retries: list[tuple[str, int, float]] = []
    payload = audit.request_with_retries(
        transport,
        intent,
        sleep=sleeps.append,
        on_response=lambda response, attempt: responses.append(
            (response.status, attempt)
        ),
        on_retry=lambda reason, attempt, delay: retries.append(
            (reason, attempt, delay)
        ),
    )
    assert payload.status == 200
    assert transport.calls == 3
    assert sleeps == [1.0, 2.0]
    assert responses == [(503, 1), (429, 2), (200, 3)]
    assert retries == [("http_503", 1, 1.0), ("http_429", 2, 2.0)]

    forbidden = _RetryTransport([404, 200])
    with pytest.raises(audit.TransportError, match="status"):
        audit.request_with_retries(forbidden, intent, sleep=sleeps.append)
    assert forbidden.calls == 1


def test_oversized_retry_response_is_terminal_before_persistence() -> None:
    intent = audit.make_get_intent(audit.INDEX_URLS[0])

    class Oversized:
        calls = 0

        def request(self, request: audit.RequestIntent) -> audit.HttpPayload:
            self.calls += 1
            return _payload(
                request,
                status=503,
                raw=b"x" * (audit.HTML_BODY_CAP + 1),
            )

    transport = Oversized()
    observed: list[int] = []
    with pytest.raises(audit.TransportError, match="body exceeds"):
        audit.request_with_retries(
            transport,
            intent,
            sleep=lambda _: None,
            on_response=lambda payload, _: observed.append(len(payload.raw)),
        )
    assert transport.calls == 1
    assert observed == []


def test_evaluate_support_requires_exact_panel_and_unique_text() -> None:
    ledger = audit.load_ledger()
    summaries: list[audit.DocumentSummary] = []
    statement_index = 0
    minutes_index = 0
    for row in ledger:
        if not row.source_eligible:
            summaries.append(
                audit.DocumentSummary(
                    document_class=row.document_class,
                    source_eligible=False,
                    raw_bytes=100,
                    raw_sha256=f"{len(summaries) + 1:064x}",
                    article_count=1,
                    canonical_bytes=None,
                    canonical_characters=None,
                    canonical_blocks=None,
                    canonical_sha256=None,
                )
            )
            continue
        if row.document_class == "fomc_statement":
            statement_index += 1
            canonical_characters = 1_000
            identity = 10_000 + statement_index
        else:
            minutes_index += 1
            canonical_characters = 20_000
            identity = 20_000 + minutes_index
        summaries.append(
            audit.DocumentSummary(
                document_class=row.document_class,
                source_eligible=True,
                raw_bytes=1_000,
                raw_sha256=f"{30_000 + identity:064x}",
                article_count=1,
                canonical_bytes=canonical_characters,
                canonical_characters=canonical_characters,
                canonical_blocks=10,
                canonical_sha256=f"{identity:064x}",
            )
        )
    support = audit.evaluate_support(ledger, summaries)
    assert support["decision"] == "SOURCE_SUPPORT_PASS"
    assert support["candidate_count"] == 147
    assert support["eligible_count"] == 145
    assert support["same_class_transition_count"] == 143

    duplicate = list(summaries)
    first = next(summary for summary in duplicate if summary.source_eligible)
    later = next(
        index
        for index, summary in enumerate(duplicate)
        if summary.source_eligible
        and summary.document_class == first.document_class
        and summary is not first
    )
    duplicate[later] = audit.DocumentSummary(
        document_class=duplicate[later].document_class,
        source_eligible=True,
        raw_bytes=duplicate[later].raw_bytes,
        raw_sha256=duplicate[later].raw_sha256,
        article_count=1,
        canonical_bytes=duplicate[later].canonical_bytes,
        canonical_characters=duplicate[later].canonical_characters,
        canonical_blocks=duplicate[later].canonical_blocks,
        canonical_sha256=first.canonical_sha256,
    )
    with pytest.raises(audit.SupportError, match="repeats"):
        audit.evaluate_support(ledger, duplicate)


def test_manifest_chain_binds_raw_and_canonical_artifacts(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    guard = audit.reserve_attempt(
        paths=paths,
        verifier_commit="1" * 40,
        runner_blob="2" * 40,
    )
    intent = audit.make_get_intent(audit.INDEX_URLS[0])
    raw_path = audit._persist_response(
        guard,
        _payload(intent, raw=b"<html>index</html>"),
        kind="index",
        key="2011",
        attempt=1,
    )
    canonical = b"frozen canonical\n"
    canonical_path = audit._persist_canonical(guard, canonical, key="001")
    guard.append(
        "document_parsed",
        {
            "canonical_bytes": len(canonical),
            "canonical_path": canonical_path.relative_to(paths.raw_dir).as_posix(),
            "canonical_sha256": audit.sha256_bytes(canonical),
            "source_eligible": True,
        },
    )
    guard.append(
        "source_support_complete",
        {"decision": "SOURCE_SUPPORT_PASS"},
    )
    replay = audit.validate_manifest_chain(
        paths.manifest,
        raw_dir=paths.raw_dir,
        require_complete=True,
    )
    assert replay == {
        "final_record_hash": guard.previous_hash,
        "record_count": guard.sequence,
        "raw_artifact_count": 1,
        "canonical_artifact_count": 1,
    }
    raw_path.write_bytes(b"tampered")
    with pytest.raises(audit.ProtocolError, match="binding"):
        audit.validate_manifest_chain(
            paths.manifest,
            raw_dir=paths.raw_dir,
            require_complete=True,
        )


def test_aggregate_raw_cap_rejects_before_artifact_or_manifest_write(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    guard = audit.reserve_attempt(
        paths=paths,
        verifier_commit="1" * 40,
        runner_blob="2" * 40,
    )
    guard.raw_bytes_stored = audit.TOTAL_RAW_SOURCE_CAP - 5
    intent = audit.make_get_intent(audit.INDEX_URLS[0])
    with pytest.raises(audit.DiskGuardError, match="256 MiB"):
        audit._persist_response(
            guard,
            _payload(intent, raw=b"123456"),
            kind="index",
            key="2011",
            attempt=1,
        )
    assert list((paths.raw_dir / "raw").iterdir()) == []
    assert paths.manifest.read_bytes() == b""
    assert guard.response_count == 0


def test_one_shot_reservation_and_atomic_report_do_not_clobber(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    audit.reserve_attempt(
        paths=paths,
        verifier_commit="1" * 40,
        runner_blob="2" * 40,
    )
    assert paths.sentinel.stat().st_mode & 0o777 == 0o600
    with pytest.raises(audit.ProtocolError, match="consumed"):
        audit.reserve_attempt(
            paths=paths,
            verifier_commit="1" * 40,
            runner_blob="2" * 40,
        )
    audit._atomic_publish(paths.report, b"{}\n")
    with pytest.raises(audit.PublicationError, match="exists"):
        audit._atomic_publish(paths.report, b'{"changed":true}\n')
    assert paths.report.read_bytes() == b"{}\n"


def test_disk_guard_uses_reported_used_and_free_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Usage = namedtuple("Usage", "total used free")
    used = 299 * 1024**3
    free = 9 * 1024**3
    monkeypatch.setattr(
        audit.shutil,
        "disk_usage",
        lambda _: Usage(total=400 * 1024**3, used=used, free=free),
    )
    assert audit._disk_guard() == (used, free)
    monkeypatch.setattr(
        audit.shutil,
        "disk_usage",
        lambda _: Usage(
            total=400 * 1024**3,
            used=300 * 1024**3,
            free=100 * 1024**3,
        ),
    )
    with pytest.raises(audit.DiskGuardError, match="300 GiB"):
        audit._disk_guard()


def test_aggregate_report_is_deterministic_and_text_free() -> None:
    canary = "NEVER_EMIT_THIS_ARTICLE_TEXT"
    support = {
        "candidate_count": 147,
        "decision": "SOURCE_SUPPORT_PASS",
        "eligible_count": 145,
    }
    kwargs = {
        "execution_authority": "fixture",
        "source_audit_authoritative": False,
        "verifier_commit": "1" * 40,
        "runner_blob": "2" * 40,
        "manifest_sha256": "3" * 64,
        "sentinel_sha256": "4" * 64,
        "index_fingerprint": "5" * 64,
        "document_fingerprint": "6" * 64,
        "canonical_fingerprint": "7" * 64,
        "support": support,
    }
    left = audit.build_report(**kwargs)
    right = audit.build_report(**kwargs)
    assert left == right
    encoded = audit.canonical_json_bytes(left)
    assert canary.encode() not in encoded
    assert left["mechanism_preregistration_authorized"] is False
    assert left["outcome_boundary"] == {
        "article_text_emitted": False,
        "database_opened": False,
        "market_price_return_or_funding_opened": False,
        "model_tokenizer_adapter_prompt_or_checkpoint_opened": False,
        "portfolio_reward_or_performance_opened": False,
        "post_2020_document_body_opened": False,
        "semantic_label_embedding_or_inference_called": False,
    }
    assert json.loads(encoded)["decision"] == "SOURCE_SUPPORT_PASS"


def test_module_import_surface_is_standard_library_only() -> None:
    source = (ROOT / audit.SCRIPT_PATH).read_text()
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    allowed = set(sys.stdlib_module_names) | {"__future__"}
    assert roots <= allowed
    assert not roots & {
        "numpy",
        "pandas",
        "requests",
        "sqlalchemy",
        "torch",
        "transformers",
    }
