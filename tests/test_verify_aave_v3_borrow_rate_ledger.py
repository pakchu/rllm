from __future__ import annotations

import hashlib
import json
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from training import verify_aave_v3_borrow_rate_ledger as verifier


def _hash(value: int) -> str:
    return "0x" + f"{value:064x}"


def _header(
    number: int,
    *,
    block_hash: str | None = None,
    parent_hash: str | None = None,
    timestamp: int | None = None,
) -> dict[str, str]:
    return {
        "number": hex(number),
        "hash": block_hash or _hash(number),
        "parentHash": parent_hash or _hash(max(0, number - 1)),
        "timestamp": hex(timestamp if timestamp is not None else number + 1_000),
    }


def _log(
    block_number: int,
    *,
    reserve_index: int = 0,
    transaction_index: int = 0,
    log_index: int = 0,
    data_words: Sequence[int] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    transaction_identity = (
        block_number * 10_000 + transaction_index * 100 + log_index + 1
    )
    return {
        "address": verifier.POOL_ADDRESS.upper().replace("0X", "0x"),
        "topics": [
            verifier.EVENT_TOPIC.upper().replace("0X", "0x"),
            verifier.RESERVE_TOPICS[reserve_index].upper().replace("0X", "0x"),
        ],
        "data": "0x" + "".join(f"{value:064x}" for value in data_words),
        "blockNumber": hex(block_number),
        "blockHash": _hash(block_number).upper().replace("0X", "0x"),
        "transactionHash": _hash(transaction_identity),
        "transactionIndex": hex(transaction_index),
        "logIndex": hex(log_index),
        "removed": False,
        "providerSpecificExtra": "ignored",
    }


def _event(
    block_number: int,
    *,
    reserve_index: int,
    transaction_index: int = 0,
    log_index: int = 0,
) -> verifier.CanonicalEvent:
    return verifier.parse_event(
        _log(
            block_number,
            reserve_index=reserve_index,
            transaction_index=transaction_index,
            log_index=log_index,
        ),
        first_block=block_number,
        last_block=block_number,
    )


class MappingRpc:
    def __init__(self, responses: dict[tuple[str, int], tuple[Any, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, tuple[Any, ...], int]] = []

    def call_pair(
        self,
        method: str,
        params: Sequence[Any],
        request_id: int,
        *,
        capability: object | None = None,
    ) -> tuple[Any, Any]:
        if method == "eth_getLogs":
            if capability is None:
                raise verifier.ProtocolError("missing test capability")
            verifier._require_source_capability(capability)
        self.calls.append((method, tuple(params), request_id))
        return self.responses[(method, request_id)]


def _chunk_rpc(
    chunk: verifier.Chunk,
    logs: Sequence[dict[str, Any]],
) -> MappingRpc:
    responses: dict[tuple[str, int], tuple[Any, Any]] = {
        (
            "eth_getLogs",
            chunk.request_id,
        ): (list(reversed(logs)), list(logs)),
    }
    for subdivision in chunk.subdivisions():
        subset = [
            row
            for row in logs
            if subdivision.start <= int(row["blockNumber"], 16) <= subdivision.end
        ]
        responses[("eth_getLogs", subdivision.request_id)] = (
            list(reversed(subset)),
            list(subset),
        )
    return MappingRpc(responses)


def _all_true_gates() -> dict[str, bool]:
    return {name: True for name in verifier.STAGE_A_GATE_NAMES}


def _source_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    sentinel = tmp_path / "capability.started"
    monkeypatch.setattr(verifier, "DEFAULT_SENTINEL", sentinel)
    monkeypatch.setattr(verifier, "assert_protocol_committed", lambda: "1" * 40)
    monkeypatch.setattr(verifier, "assert_disk_guard", lambda: (1, 2))
    verifier.reserve_one_shot(
        sentinel_path=sentinel,
        report_path=tmp_path / "unused-report.json",
        temp_report_path=tmp_path / "unused-report.tmp",
        verifier_commit="1" * 40,
        started_at_utc="2026-07-24T00:00:00Z",
    )
    return verifier._mint_source_capability(sentinel, "1" * 40)


def _patch_run_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    sentinel = tmp_path / "run.started"
    temporary = tmp_path / "report.tmp"
    report = tmp_path / "report.json"
    monkeypatch.setattr(verifier, "DEFAULT_SENTINEL", sentinel)
    monkeypatch.setattr(verifier, "DEFAULT_TEMP_REPORT", temporary)
    monkeypatch.setattr(verifier, "DEFAULT_REPORT", report)
    return sentinel, temporary, report


def test_frozen_constants_and_schedules_are_self_consistent() -> None:
    assert hashlib.sha256(verifier.SCHEMA_PREIMAGE).hexdigest() == (
        verifier.SCHEMA_SHA256
    )
    assert verifier.computed_event_topic() == verifier.EVENT_TOPIC
    assert verifier.RESERVE_TOPICS == tuple(
        "0x" + "0" * 24 + address[2:]
        for address in verifier.RESERVE_ADDRESSES
    )

    historical = verifier.chunks_for_window(verifier.HISTORICAL_WINDOW)
    recent = verifier.chunks_for_window(verifier.RECENT_WINDOW)
    assert len(historical) == 14
    assert len(recent) == 15
    assert [row.ordinal for row in historical] == list(range(14))
    assert [row.ordinal for row in recent] == list(range(14, 29))
    assert historical[0].start == 17_382_266
    assert historical[-1].end == 17_389_364
    assert recent[0].start == 25_594_813
    assert recent[-1].end == 25_602_012
    assert len(historical[-1].subdivisions()) == 4
    assert len(recent[-1].subdivisions()) == 1
    assert recent[-1].subdivisions()[0].request_id == (
        300_000_000 + 4 * recent[-1].ordinal
    )
    assert (
        (verifier.FULL_HISTORY_LAST_BLOCK - verifier.FULL_HISTORY_FIRST_BLOCK + 1)
        + verifier.CHUNK_SIZE
        - 1
    ) // verifier.CHUNK_SIZE == verifier.FULL_HISTORY_CHUNK_COUNT


@pytest.mark.parametrize(
    "value",
    ["0x00", "0x01", "0X1", "", None, 1, True, "-0x1"],
)
def test_quantity_parser_rejects_nonminimal_values(value: Any) -> None:
    with pytest.raises(verifier.SchemaError):
        verifier.parse_quantity(value, "quantity")


def test_json_parser_rejects_duplicate_keys_batches_and_nonfinite_values() -> None:
    with pytest.raises(verifier.SchemaError):
        verifier.parse_json_object(b'{"id":1,"id":2}')
    with pytest.raises(verifier.SchemaError):
        verifier.parse_json_object(b"[1,2]")
    with pytest.raises(verifier.SchemaError):
        verifier.parse_json_object(b'{"value":NaN}')


def test_parse_event_freezes_shape_and_canonical_json() -> None:
    row = _log(
        123,
        reserve_index=2,
        transaction_index=4,
        log_index=7,
        data_words=(0, 2, 3, 4, 2**255),
    )
    event = verifier.parse_event(row, first_block=100, last_block=200)
    assert event.address == verifier.POOL_ADDRESS
    assert event.reserve == verifier.RESERVE_ADDRESSES[2]
    assert event.data_words == (0, 2, 3, 4, 2**255)
    assert event.order_key == (123, 4, 7)
    decoded = json.loads(event.canonical_json_bytes())
    assert decoded["data_words"] == [0, 2, 3, 4, 2**255]
    assert decoded["topics"] == list(event.topics)
    assert "providerSpecificExtra" not in decoded
    assert list(decoded) == sorted(decoded)


def test_parse_event_rejects_removed_wrong_length_and_leading_zero() -> None:
    removed = _log(123)
    removed["removed"] = 0
    with pytest.raises(verifier.SchemaError):
        verifier.parse_event(removed, first_block=100, last_block=200)

    short = _log(123)
    short["data"] = short["data"][:-2]
    with pytest.raises(verifier.SchemaError):
        verifier.parse_event(short, first_block=100, last_block=200)

    nonminimal = _log(123)
    nonminimal["logIndex"] = "0x00"
    with pytest.raises(verifier.SchemaError):
        verifier.parse_event(nonminimal, first_block=100, last_block=200)


def test_canonicalize_events_rejects_duplicate_identity() -> None:
    row = _log(123)
    with pytest.raises(verifier.ParityError):
        verifier.canonicalize_events(
            [row, dict(row)],
            first_block=100,
            last_block=200,
        )


def test_canonicalize_events_rejects_duplicate_order_position() -> None:
    first = _log(123, reserve_index=0, transaction_index=1, log_index=2)
    second = _log(123, reserve_index=1, transaction_index=1, log_index=2)
    second["transactionHash"] = _hash(999_999)
    with pytest.raises(verifier.ParityError):
        verifier.canonicalize_events(
            [first, second],
            first_block=100,
            last_block=200,
        )


def test_parent_subdivision_redundancy_accepts_only_exact_union(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = verifier.Chunk(ordinal=0, start=100, end=611)
    logs = [
        _log(100, reserve_index=0),
        _log(227, reserve_index=1),
        _log(228, reserve_index=2),
        _log(356, reserve_index=3),
        _log(611, reserve_index=0),
    ]
    rpc = _chunk_rpc(chunk, logs)
    disk_calls: list[int] = []

    events = verifier._fetch_chunk_with_redundancy(
        _source_capability(tmp_path, monkeypatch),
        rpc,
        chunk,
        disk_guard=lambda: (disk_calls.append(1) or (1, 2)),
    )

    assert [row.block_number for row in events] == [100, 227, 228, 356, 611]
    assert len(disk_calls) == 5
    assert [call[2] for call in rpc.calls] == [
        chunk.request_id,
        *[row.request_id for row in chunk.subdivisions()],
    ]
    expected_filter = verifier.get_logs_filter(100, 611)
    assert rpc.calls[0][1] == (expected_filter,)


def test_parent_subdivision_redundancy_rejects_common_omission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = verifier.Chunk(ordinal=0, start=100, end=611)
    logs = [_log(100), _log(300, reserve_index=1)]
    rpc = _chunk_rpc(chunk, logs)
    target = chunk.subdivisions()[1]
    rpc.responses[("eth_getLogs", target.request_id)] = ([], [])
    with pytest.raises(verifier.ParityError):
        verifier._fetch_chunk_with_redundancy(
            _source_capability(tmp_path, monkeypatch),
            rpc,
            chunk,
            disk_guard=lambda: (1, 2),
        )


def test_parent_subdivision_redundancy_rejects_provider_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = verifier.Chunk(ordinal=0, start=100, end=611)
    logs = [_log(100), _log(300, reserve_index=1)]
    rpc = _chunk_rpc(chunk, logs)
    target = chunk.subdivisions()[1]
    primary, _ = rpc.responses[("eth_getLogs", target.request_id)]
    rpc.responses[("eth_getLogs", target.request_id)] = (primary, [])
    with pytest.raises(verifier.ParityError):
        verifier._fetch_chunk_with_redundancy(
            _source_capability(tmp_path, monkeypatch),
            rpc,
            chunk,
            disk_guard=lambda: (1, 2),
        )


def test_chain_and_boundary_headers_allow_different_finalized_heights() -> None:
    responses: dict[tuple[str, int], tuple[Any, Any]] = {
        ("eth_chainId", 1): ("0x1", "0x1"),
        (
            "eth_getBlockByNumber",
            2,
        ): (
            _header(verifier.FULL_HISTORY_LAST_BLOCK + 100),
            _header(verifier.FULL_HISTORY_LAST_BLOCK + 99),
        ),
    }
    for expected in verifier.BOUNDARY_HEADERS:
        header = _header(
            expected.number,
            block_hash=expected.block_hash,
            timestamp=expected.timestamp,
        )
        responses[("eth_getBlockByNumber", 100_000_000 + expected.number)] = (
            dict(header),
            dict(header),
        )
    verifier._verify_chain_and_boundaries(MappingRpc(responses))


def test_chain_and_boundary_headers_reject_hash_drift() -> None:
    responses: dict[tuple[str, int], tuple[Any, Any]] = {
        ("eth_chainId", 1): ("0x1", "0x1"),
        (
            "eth_getBlockByNumber",
            2,
        ): (
            _header(verifier.FULL_HISTORY_LAST_BLOCK),
            _header(verifier.FULL_HISTORY_LAST_BLOCK),
        ),
    }
    for expected in verifier.BOUNDARY_HEADERS:
        header = _header(
            expected.number,
            block_hash=expected.block_hash,
            timestamp=expected.timestamp,
        )
        responses[("eth_getBlockByNumber", 100_000_000 + expected.number)] = (
            dict(header),
            dict(header),
        )
    last = verifier.BOUNDARY_HEADERS[-1]
    bad = _header(last.number, block_hash=_hash(999), timestamp=last.timestamp)
    responses[("eth_getBlockByNumber", 100_000_000 + last.number)] = (
        bad,
        dict(bad),
    )
    with pytest.raises(verifier.HeaderError):
        verifier._verify_chain_and_boundaries(MappingRpc(responses))


def test_exact_block_header_requests_are_deduplicated() -> None:
    number = 123
    request_id = 100_000_000 + number
    header = _header(number)
    delegate = MappingRpc(
        {
            ("eth_getBlockByNumber", request_id): (
                dict(header),
                dict(header),
            )
        }
    )
    cached = verifier.HeaderCachingPairRpc(delegate)
    params = (hex(number), False)
    first = cached.call_pair("eth_getBlockByNumber", params, request_id)
    second = cached.call_pair("eth_getBlockByNumber", params, request_id)
    assert first == second
    assert len(delegate.calls) == 1
    with pytest.raises(verifier.ProtocolError):
        cached.call_pair(
            "eth_getBlockByNumber",
            (hex(number + 1), False),
            request_id,
        )


def test_window_support_and_header_audit_are_cross_provider_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = tuple(
        _event(1_000 + index, reserve_index=index) for index in range(4)
    )
    recent = tuple(
        _event(2_000 + index, reserve_index=index) for index in range(4)
    )
    streams = {"historical": historical, "recent": recent}
    verifier.require_window_support(streams)

    expected = verifier.header_audit_block_hashes(streams)
    responses: dict[tuple[str, int], tuple[Any, Any]] = {}
    for number, block_hash in expected.items():
        header = _header(number, block_hash=block_hash, timestamp=number + 10_000)
        responses[("eth_getBlockByNumber", 100_000_000 + number)] = (
            dict(header),
            dict(header),
        )
    rpc = MappingRpc(responses)
    capability = _source_capability(tmp_path, monkeypatch)
    verifier._verify_event_header_audit(capability, rpc, streams)
    assert len(rpc.calls) == len(expected)

    first = next(iter(expected))
    primary, parity = responses[("eth_getBlockByNumber", 100_000_000 + first)]
    parity = dict(parity)
    parity["parentHash"] = _hash(777)
    responses[("eth_getBlockByNumber", 100_000_000 + first)] = (
        primary,
        parity,
    )
    with pytest.raises(verifier.HeaderError):
        verifier._verify_event_header_audit(
            capability,
            MappingRpc(responses),
            streams,
        )


def test_window_hash_is_order_canonical_and_domain_separated() -> None:
    events = (
        _event(100, reserve_index=0),
        _event(101, reserve_index=1),
    )
    historical = verifier.window_stream_sha256("historical", events)
    assert historical == verifier.window_stream_sha256("historical", events)
    assert historical != verifier.window_stream_sha256("recent", events)
    assert historical != verifier.window_stream_sha256(
        "historical", tuple(reversed(events))
    )


def test_pin_verification_is_hash_and_schema_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    event = (
        b"// comment\n"
        + verifier.NORMALIZED_EVENT_DECLARATION.encode()
    )
    payloads = {
        "address_book": (
            "\n".join(
                [verifier.POOL_ADDRESS, *verifier.RESERVE_ADDRESSES]
            ).encode()
        ),
        "address_book_license": b"MIT\n",
        "current_ipool": event,
        "archived_ipool": event,
        "aave_utilities_readme": verifier.PRIMARY_RPC.encode(),
    }
    specs = tuple(
        verifier.PinSpec(name, f"https://example.test/{name}", hashlib.sha256(data).hexdigest())
        for name, data in payloads.items()
    )
    monkeypatch.setattr(verifier, "PIN_SPECS", specs)
    by_url = {spec.url: payloads[spec.name] for spec in specs}
    verifier._verify_pins(lambda url: by_url[url])

    by_url[specs[0].url] += b"drift"
    with pytest.raises(verifier.ProtocolError):
        verifier._verify_pins(lambda url: by_url[url])


def test_terminal_report_is_exact_and_manifest_bound() -> None:
    evidence = verifier.SourceEvidence(
        historical_sha256="a" * 64,
        recent_sha256="b" * 64,
    )
    report = verifier.build_terminal_report(
        status="PASS",
        verifier_commit="1" * 40,
        started_at_utc="2026-07-24T00:00:00Z",
        terminal_at_utc="2026-07-24T01:00:00Z",
        gates=_all_true_gates(),
        evidence=evidence,
    )
    assert set(report) == {
        "protocol_version",
        "status",
        "state",
        "source_boundary_sha256",
        "verifier_commit",
        "started_at_utc",
        "terminal_at_utc",
        "pins",
        "bindings",
        "range",
        "providers",
        "schema_sha256",
        "window_streams",
        "full_history_sha256",
        "ledger_gzip_sha256",
        "ledger_gzip_bytes",
        "gates",
        "failure",
        "forbidden_access",
        "manifest_sha256",
    }
    assert set(report["gates"]) == set(verifier.STAGE_A_GATE_NAMES)
    assert all(report["gates"].values())
    assert report["full_history_sha256"] is None
    assert report["ledger_gzip_sha256"] is None
    assert report["ledger_gzip_bytes"] is None
    assert report["bindings"] == {
        "source_parity_report_path": None,
        "source_parity_report_sha256": None,
        "mechanism_document_path": None,
        "mechanism_document_sha256": None,
        "preregistration_document_path": None,
        "preregistration_document_sha256": None,
    }
    preimage = dict(report)
    manifest = preimage.pop("manifest_sha256")
    expected = hashlib.sha256(
        b"AV3BRL-v1\0terminal-report\0"
        + verifier.canonical_json_bytes(preimage)
    ).hexdigest()
    assert manifest == expected
    assert verifier.terminal_report_bytes(report).endswith(b"\n")


def test_pass_report_rejects_noncanonical_commit_hash_and_stream_hash() -> None:
    with pytest.raises(verifier.ProtocolError):
        verifier.build_terminal_report(
            status="PASS",
            verifier_commit="A" * 40,
            started_at_utc="2026-07-24T00:00:00Z",
            terminal_at_utc="2026-07-24T01:00:00Z",
            gates=_all_true_gates(),
            evidence=verifier.SourceEvidence("a" * 64, "b" * 64),
        )
    with pytest.raises(verifier.ProtocolError):
        verifier.build_terminal_report(
            status="PASS",
            verifier_commit="1" * 40,
            started_at_utc="2026-07-24T00:00:00Z",
            terminal_at_utc="2026-07-24T01:00:00Z",
            gates=_all_true_gates(),
            evidence=verifier.SourceEvidence("A" * 64, "b" * 64),
        )


def test_reject_report_never_serializes_sensitive_failure_details() -> None:
    gates = _all_true_gates()
    for name in (
        "subdivision_redundancy",
        "historical_window",
        "recent_window",
        "event_header_audit",
    ):
        gates[name] = False
    report = verifier.build_terminal_report(
        status="REJECT",
        verifier_commit="1" * 40,
        started_at_utc="2026-07-24T00:00:00Z",
        terminal_at_utc="2026-07-24T01:00:00Z",
        gates=gates,
        error=verifier.SupportError(
            "WBTC historical 2023-06 block 123 had 12 events at 9%"
        ),
    )
    encoded = verifier.terminal_report_bytes(report)
    assert report["window_streams"] is None
    assert report["providers"]["canonical_streams_equal"] is False
    assert report["failure"] == {
        "stage": "support",
        "exception_type": "SupportError",
        "message": "support gate failed",
    }
    for forbidden in (b"WBTC", b"historical", b"2023-06", b"123", b"12", b"9%"):
        assert forbidden not in json.dumps(report["failure"]).encode()
    assert b"support gate failed" in encoded


def test_one_shot_pass_publishes_once_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel, temporary, report_path = _patch_run_paths(monkeypatch, tmp_path)
    calls = 0
    fixed_now = datetime(2026, 7, 24, 0, 0, tzinfo=timezone.utc)

    def source_probe(
        capability: object,
        rpc: object,
        **_: Any,
    ) -> verifier.SourceEvidence:
        nonlocal calls
        del rpc
        verifier._require_source_capability(capability)
        calls += 1
        return verifier.SourceEvidence("a" * 64, "b" * 64)

    monkeypatch.setattr(verifier, "assert_protocol_committed", lambda: "1" * 40)
    monkeypatch.setattr(
        verifier, "assert_atomic_publication_supported", lambda _: None
    )
    monkeypatch.setattr(verifier, "assert_disk_guard", lambda: (1, 2))
    monkeypatch.setattr(verifier, "_verify_source", source_probe)
    monkeypatch.setattr(verifier, "DualRpc", lambda: object())
    monkeypatch.setattr(verifier, "utc_now", lambda: fixed_now)

    report = verifier.run_stage_a()
    assert report["status"] == "PASS"
    assert calls == 1
    assert sentinel.exists()
    assert report_path.exists()
    assert not temporary.exists()
    assert json.loads(report_path.read_text()) == report

    with pytest.raises(verifier.ProtocolError):
        verifier.run_stage_a()
    assert calls == 1


def test_one_shot_failure_publishes_only_generic_reject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, report_path = _patch_run_paths(monkeypatch, tmp_path)

    def rejected_probe(
        capability: object,
        rpc: object,
        **_: Any,
    ) -> verifier.SourceEvidence:
        del rpc
        verifier._require_source_capability(capability)
        raise verifier.ParityError("USDC recent block 999 rate 123")

    monkeypatch.setattr(verifier, "assert_protocol_committed", lambda: "1" * 40)
    monkeypatch.setattr(
        verifier, "assert_atomic_publication_supported", lambda _: None
    )
    monkeypatch.setattr(verifier, "assert_disk_guard", lambda: (1, 2))
    monkeypatch.setattr(verifier, "_verify_source", rejected_probe)
    monkeypatch.setattr(verifier, "DualRpc", lambda: object())
    monkeypatch.setattr(
        verifier,
        "utc_now",
        lambda: datetime(2026, 7, 24, 0, 0, tzinfo=timezone.utc),
    )

    report = verifier.run_stage_a()
    assert report["status"] == "REJECT"
    assert report["failure"] == {
        "stage": "canonicalization",
        "exception_type": "ParityError",
        "message": "canonical parity failed",
    }
    assert report["window_streams"] is None
    assert all(
        report["gates"][name] is False
        for name in (
            "pinned_sources",
            "chain_identity",
            "boundary_headers",
            "subdivision_redundancy",
            "historical_window",
            "recent_window",
            "event_header_audit",
        )
    )
    assert "USDC" not in report_path.read_text()


def test_protocol_failure_cannot_create_sentinel_or_open_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel, _, _ = _patch_run_paths(monkeypatch, tmp_path)
    source_calls = 0

    def source_probe(*_: Any, **__: Any) -> verifier.SourceEvidence:
        nonlocal source_calls
        source_calls += 1
        return verifier.SourceEvidence("a" * 64, "b" * 64)

    def rejected_protocol() -> str:
        raise verifier.ProtocolError("dirty worktree")

    monkeypatch.setattr(verifier, "assert_protocol_committed", rejected_protocol)
    monkeypatch.setattr(verifier, "_verify_source", source_probe)
    with pytest.raises(verifier.ProtocolError):
        verifier.run_stage_a()
    assert source_calls == 0
    assert not sentinel.exists()


def test_disk_failure_consumes_identity_without_opening_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel, _, report_path = _patch_run_paths(monkeypatch, tmp_path)
    source_calls = 0

    def source_probe(*_: Any, **__: Any) -> verifier.SourceEvidence:
        nonlocal source_calls
        source_calls += 1
        return verifier.SourceEvidence("a" * 64, "b" * 64)

    def rejected_disk() -> tuple[int, int]:
        raise verifier.DiskGuardError("full")

    monkeypatch.setattr(verifier, "assert_protocol_committed", lambda: "1" * 40)
    monkeypatch.setattr(
        verifier, "assert_atomic_publication_supported", lambda _: None
    )
    monkeypatch.setattr(verifier, "assert_disk_guard", rejected_disk)
    monkeypatch.setattr(verifier, "_verify_source", source_probe)
    monkeypatch.setattr(
        verifier,
        "utc_now",
        lambda: datetime(2026, 7, 24, 0, 0, tzinfo=timezone.utc),
    )

    report = verifier.run_stage_a()
    assert source_calls == 0
    assert sentinel.exists()
    assert report_path.exists()
    assert report["status"] == "REJECT"
    assert report["failure"] == {
        "stage": "preflight",
        "exception_type": "DiskGuardError",
        "message": "disk guard failed",
    }


def test_atomic_publication_is_non_overwriting(tmp_path: Path) -> None:
    verifier.assert_atomic_publication_supported(tmp_path)
    temporary = tmp_path / "report.tmp"
    destination = tmp_path / "report.json"
    verifier.atomic_publish_report(
        temporary=temporary,
        destination=destination,
        payload=b"first\n",
    )
    assert destination.read_bytes() == b"first\n"
    with pytest.raises(verifier.PublicationError):
        verifier.atomic_publish_report(
            temporary=temporary,
            destination=destination,
            payload=b"second\n",
        )
    assert destination.read_bytes() == b"first\n"


def test_disk_guard_uses_exact_used_and_free_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        verifier.shutil,
        "disk_usage",
        lambda _: Usage(
            verifier.DISK_LIMIT_BYTES + verifier.DISK_MINIMUM_FREE_BYTES,
            verifier.DISK_LIMIT_BYTES - 1,
            verifier.DISK_MINIMUM_FREE_BYTES,
        ),
    )
    assert verifier.assert_disk_guard(Path(".")) == (
        verifier.DISK_LIMIT_BYTES - 1,
        verifier.DISK_MINIMUM_FREE_BYTES,
    )

    monkeypatch.setattr(
        verifier.shutil,
        "disk_usage",
        lambda _: Usage(
            verifier.DISK_LIMIT_BYTES + verifier.DISK_MINIMUM_FREE_BYTES,
            verifier.DISK_LIMIT_BYTES,
            verifier.DISK_MINIMUM_FREE_BYTES,
        ),
    )
    with pytest.raises(verifier.DiskGuardError):
        verifier.assert_disk_guard(Path("."))

    monkeypatch.setattr(
        verifier.shutil,
        "disk_usage",
        lambda _: Usage(
            verifier.DISK_LIMIT_BYTES + verifier.DISK_MINIMUM_FREE_BYTES,
            verifier.DISK_LIMIT_BYTES - 1,
            verifier.DISK_MINIMUM_FREE_BYTES - 1,
        ),
    )
    with pytest.raises(verifier.DiskGuardError):
        verifier.assert_disk_guard(Path("."))


class StubJsonRpcClient(verifier.JsonRpcClient):
    def __init__(
        self,
        replies: list[
            tuple[int, dict[str, str], bytes] | BaseException
        ],
    ) -> None:
        super().__init__(
            "https://example.test",
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )
        self.replies = replies
        self.bodies: list[bytes] = []

    def _request_once(self, body: bytes) -> tuple[int, dict[str, str], bytes]:
        self.bodies.append(body)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def test_json_rpc_request_is_canonical_and_retries_only_frozen_status() -> None:
    result = {"number": "0x1"}
    client = StubJsonRpcClient(
        [
            (429, {}, b""),
            (
                200,
                {},
                verifier.canonical_json_bytes(
                    {"jsonrpc": "2.0", "id": 7, "result": result}
                ),
            ),
        ]
    )
    assert client.call("eth_test", ("x",), 7) == result
    expected = verifier.canonical_json_bytes(
        {
            "id": 7,
            "jsonrpc": "2.0",
            "method": "eth_test",
            "params": ["x"],
        }
    )
    assert client.bodies == [expected, expected]


def test_json_rpc_response_rejects_boolean_id() -> None:
    client = StubJsonRpcClient(
        [
            (
                200,
                {},
                verifier.canonical_json_bytes(
                    {"jsonrpc": "2.0", "id": True, "result": "0x1"}
                ),
            )
        ]
    )
    with pytest.raises(verifier.SchemaError):
        client.call("eth_chainId", (), 1)


def test_json_rpc_getlogs_requires_minted_capability() -> None:
    client = StubJsonRpcClient([])
    with pytest.raises(verifier.ProtocolError):
        client.call("eth_getLogs", ({},), 200_000_000)
    assert client.bodies == []


def test_source_capability_rejects_sentinel_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = tmp_path / "capability.started"
    monkeypatch.setattr(verifier, "DEFAULT_SENTINEL", sentinel)
    monkeypatch.setattr(verifier, "assert_protocol_committed", lambda: "1" * 40)
    monkeypatch.setattr(verifier, "assert_disk_guard", lambda: (1, 2))
    verifier.reserve_one_shot(
        sentinel_path=sentinel,
        report_path=tmp_path / "report.json",
        temp_report_path=tmp_path / "report.tmp",
        verifier_commit="1" * 40,
        started_at_utc="2026-07-24T00:00:00Z",
    )
    capability = verifier._mint_source_capability(sentinel, "1" * 40)
    sentinel.write_bytes(sentinel.read_bytes() + b" ")
    with pytest.raises(verifier.ProtocolError):
        verifier._require_source_capability(capability)


def test_json_rpc_retries_pre_body_failure_but_not_schema_or_rpc_error() -> None:
    success = (
        200,
        {},
        verifier.canonical_json_bytes(
            {"jsonrpc": "2.0", "id": 1, "result": "0x1"}
        ),
    )
    pre_body = StubJsonRpcClient([ConnectionError("connect"), success])
    assert pre_body.call("eth_chainId", (), 1) == "0x1"
    assert len(pre_body.bodies) == 2

    malformed = StubJsonRpcClient(
        [(200, {}, b'{"jsonrpc":'), success]
    )
    with pytest.raises(verifier.SchemaError):
        malformed.call("eth_chainId", (), 1)
    assert len(malformed.bodies) == 1

    rpc_error = StubJsonRpcClient(
        [
            (
                200,
                {},
                verifier.canonical_json_bytes(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "error": {"code": -32000},
                    }
                ),
            ),
            success,
        ]
    )
    with pytest.raises(verifier.TransportError):
        rpc_error.call("eth_chainId", (), 1)
    assert len(rpc_error.bodies) == 1

    body_failure = StubJsonRpcClient(
        [verifier.TransportError("body failed"), success]
    )
    with pytest.raises(verifier.TransportError):
        body_failure.call("eth_chainId", (), 1)
    assert len(body_failure.bodies) == 1


def test_json_rpc_retries_5xx_exactly_twice_then_rejects() -> None:
    client = StubJsonRpcClient(
        [
            (500, {}, b""),
            (502, {}, b""),
            (503, {}, b""),
        ]
    )
    with pytest.raises(verifier.TransportError):
        client.call("eth_chainId", (), 1)
    assert len(client.bodies) == 3


def test_http_body_validation_rejects_partial_invalid_and_oversized() -> None:
    with pytest.raises(verifier.TransportError):
        verifier.validate_http_body(b"abc", {"Content-Length": "4"})
    with pytest.raises(verifier.TransportError):
        verifier.validate_http_body(b"abc", {"content-length": "invalid"})
    with pytest.raises(verifier.TransportError):
        verifier.validate_http_body(
            b"x" * (verifier.MAXIMUM_HTTP_BODY_BYTES + 1),
            {},
        )
