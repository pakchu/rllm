from __future__ import annotations

import asyncio
import base64
import inspect
import json
import threading
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pytest

from training import capture_bybit_public_trade_live_parity as capture


def trade(identifier: int, *, time_ms: int | None = None) -> capture.NormalizedTrade:
    return capture.NormalizedTrade(
        source_id=f"trade-{identifier}",
        symbol="BTCUSDT",
        side="Buy" if identifier % 2 == 0 else "Sell",
        price=str(100_000 + identifier),
        size="0.001",
        time_ms=1_700_000_000_000 + (identifier if time_ms is None else time_ms),
        seq=identifier // 2,
        block_trade=False,
        rpi_trade=False,
    )


def rest_raw(rows: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {"retCode": 0, "retMsg": "OK", "result": {"category": "linear", "list": rows}}
    ).encode()


def rest_row(identifier: int) -> dict[str, object]:
    return {
        "execId": f"trade-{identifier}",
        "symbol": "BTCUSDT",
        "price": "100.00" if identifier == 0 else str(100_000 + identifier),
        "size": "0.0010",
        "side": "Buy" if identifier % 2 == 0 else "Sell",
        "time": str(1_700_000_000_000 + identifier),
        "isBlockTrade": False,
        "isRPITrade": False,
        "seq": str(identifier // 2),
    }


def ws_raw(rows: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {
            "topic": capture.WS_TOPIC,
            "type": "snapshot",
            "ts": 1_700_000_001_000,
            "data": rows,
        }
    ).encode()


def ws_row(identifier: int, *, trade_time: int | None = None) -> dict[str, object]:
    return {
        "T": 1_700_000_000_000 + (identifier if trade_time is None else trade_time),
        "s": "BTCUSDT",
        "S": "Buy" if identifier % 2 == 0 else "Sell",
        "v": "0.001",
        "p": "100.0" if identifier == 0 else str(100_000 + identifier),
        "L": "PlusTick",
        "i": f"trade-{identifier}",
        "BT": False,
        "RPI": False,
        "seq": identifier // 2,
    }


def passing_inputs() -> tuple[list[capture.ObservedWsTrade], list[capture.RestSnapshot]]:
    records = [trade(index) for index in range(1_200)]
    ws = [
        capture.ObservedWsTrade(record, 1_100 + index)
        for index, record in enumerate(records)
    ]
    snapshots: list[capture.RestSnapshot] = []
    for ordinal in range(10):
        start = min(ordinal * 25, 200)
        window = tuple(reversed(records[start : start + 1_000]))
        snapshots.append(
            capture.RestSnapshot(
                ordinal=ordinal + 1,
                request_start_monotonic_ns=5_000 if ordinal == 9 else 900 + ordinal,
                response_end_monotonic_ns=1_000 + ordinal,
                final=ordinal == 9,
                trades=window,
            )
        )
    return ws, snapshots


def test_frozen_transport_and_bindings_are_exact() -> None:
    capture.validate_bindings()
    assert capture.PROTOCOL_VERSION == "bybit_public_trade_live_parity_capture_v2"
    assert capture.REST_URL == (
        "https://api.bybit.com/v5/market/recent-trade?"
        "category=linear&symbol=BTCUSDT&limit=1000"
    )
    assert capture.WS_URL == "wss://stream.bybit.com/v5/public/linear"
    assert capture.WS_TOPIC == "publicTrade.BTCUSDT"
    assert capture.CAPTURE_SECONDS == 600
    assert capture.REST_INTERVAL_SECONDS == 1


def test_local_clock_sample_uses_bracketed_monotonic_midpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic = iter((100, 120))
    monkeypatch.setattr(capture.time, "monotonic_ns", lambda: next(monotonic))
    monkeypatch.setattr(capture.time, "time_ns", lambda: 1_000)
    sample = capture._local_clock_sample("fixture", 7)
    assert sample == capture.ClockSample(
        source="fixture",
        ordinal=7,
        monotonic_ns=110,
        utc_ns=1_000,
        uncertainty_ns=20,
    )


def test_rest_parser_normalizes_exact_fields_without_float_rounding() -> None:
    parsed = capture.parse_rest_response(rest_raw([rest_row(0), rest_row(1)]))
    assert parsed[0] == capture.NormalizedTrade(
        source_id="trade-0",
        symbol="BTCUSDT",
        side="Buy",
        price="100",
        size="0.001",
        time_ms=1_700_000_000_000,
        seq=0,
        block_trade=False,
        rpi_trade=False,
    )
    assert parsed[1].price == "100001"


def test_rest_parser_rejects_bad_flags_ids_and_limits() -> None:
    bad = rest_row(1)
    bad["isRPITrade"] = "false"
    with pytest.raises(capture.ParityCaptureError, match="boolean"):
        capture.parse_rest_response(rest_raw([bad]))
    bad = rest_row(1)
    bad["execId"] = ""
    with pytest.raises(capture.ParityCaptureError, match="nonempty"):
        capture.parse_rest_response(rest_raw([bad]))
    with pytest.raises(capture.ParityCaptureError, match="count"):
        capture.parse_rest_response(rest_raw([]))


def test_ws_parser_accepts_ack_pong_and_shared_sequence() -> None:
    assert capture.parse_ws_payload(b'{"op":"subscribe","success":true}') == (
        "subscribe",
        (),
    )
    assert capture.parse_ws_payload(b'{"op":"pong"}') == ("pong", ())
    left = ws_row(0)
    right = ws_row(1)
    right["seq"] = left["seq"]
    kind, rows = capture.parse_ws_payload(ws_raw([left, right]))
    assert kind == "trade"
    assert [row.source_id for row in rows] == ["trade-0", "trade-1"]
    assert rows[0].price == "100"
    assert rows[0].seq == rows[1].seq == 0


def test_ws_parser_rejects_decreasing_message_time_and_seq_as_bool() -> None:
    with pytest.raises(capture.ParityCaptureError, match="decreasing"):
        capture.parse_ws_payload(
            ws_raw([ws_row(0, trade_time=2), ws_row(1, trade_time=1)])
        )
    row = ws_row(1)
    row["seq"] = True
    with pytest.raises(capture.ParityCaptureError, match="nonnegative"):
        capture.parse_ws_payload(ws_raw([row]))

    row = ws_row(1)
    row.pop("L")
    with pytest.raises(capture.ParityCaptureError, match="tick direction"):
        capture.parse_ws_payload(ws_raw([row]))


def test_malformed_ws_bytes_are_audited_before_parser_rejects() -> None:
    state = capture.CaptureState(
        capture_day="2026-07-23",
        started_at_utc="2026-07-23T00:00:00Z",
        output_dir=Path("data/fixture"),
    )
    handle = BytesIO()
    utc_ns = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp() * 1e9)
    raw = b"\xffnot-json"
    with pytest.raises(capture.ParityCaptureError, match="UTF-8 JSON"):
        capture._record_ws_message(state, handle, raw, utc_ns, 100)
    envelope = json.loads(handle.getvalue())
    assert base64.b64decode(envelope["raw_frame_base64"]) == raw
    assert envelope["raw_frame_sha256"] == capture.sha256_bytes(raw)


def test_binary_ws_frame_is_clocked_and_audited_before_rejection() -> None:
    state = capture.CaptureState(
        capture_day="2026-07-23",
        started_at_utc="2026-07-23T00:00:00Z",
        output_dir=Path("data/fixture"),
    )
    handle = BytesIO()
    utc_ns = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp() * 1e9)
    raw = b"\x00\xffbinary"
    with pytest.raises(capture.ParityCaptureError, match="binary"):
        capture._record_ws_message(
            state,
            handle,
            raw,
            utc_ns,
            100,
            3,
            "binary",
        )
    envelope = json.loads(handle.getvalue())
    assert envelope["frame_type"] == "binary"
    assert base64.b64decode(envelope["raw_frame_base64"]) == raw
    assert state.clock_samples == [
        capture.ClockSample("websocket_receipt", 1, 100, utc_ns, 3)
    ]


def test_ws_capture_bound_rejects_only_after_raw_frame_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = capture.CaptureState(
        capture_day="2026-07-23",
        started_at_utc="2026-07-23T00:00:00Z",
        output_dir=Path("data/fixture"),
    )
    handle = BytesIO()
    raw = b'{"op":"pong"}'
    utc_ns = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp() * 1e9)
    monkeypatch.setattr(capture, "MAX_CAPTURE_RAW_BYTES", len(raw) - 1)
    with pytest.raises(capture.ParityCaptureError, match="raw bytes"):
        capture._record_ws_message(state, handle, raw, utc_ns, 100)
    envelope = json.loads(handle.getvalue())
    assert base64.b64decode(envelope["raw_frame_base64"]) == raw
    assert state.raw_bytes == len(raw)


def test_http_error_body_and_status_are_audited_before_rejection() -> None:
    state = capture.CaptureState(
        capture_day="2026-07-23",
        started_at_utc="2026-07-23T00:00:00Z",
        output_dir=Path("data/fixture"),
    )
    handle = BytesIO()
    utc_ns = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp() * 1e9)
    raw = b'{"retCode":10006,"retMsg":"Too many visits"}'
    response = capture.RestTransportResponse(
        status=429,
        final_url=capture.REST_URL,
        content_type="application/json",
        location=None,
        raw=raw,
    )
    start = capture.ClockSample("rest_request_start", 1, 100, utc_ns, 1)
    end = capture.ClockSample("rest_response_end", 1, 101, utc_ns + 1, 1)
    capture._begin_rest_attempt(state, handle, start)
    with pytest.raises(capture.ParityCaptureError, match="status"):
        capture._record_rest_response(
            state,
            handle,
            response,
            ordinal=1,
            start=start,
            end=end,
            final=False,
        )
    lines = [json.loads(line) for line in handle.getvalue().splitlines()]
    assert lines[0]["record_type"] == "request_start"
    envelope = lines[1]
    assert envelope["record_type"] == "response"
    assert envelope["http_status"] == 429
    assert base64.b64decode(envelope["raw_json_base64"]) == raw
    assert envelope["raw_json_sha256"] == capture.sha256_bytes(raw)


def test_rest_size_bound_rejects_only_after_body_and_clock_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = capture.CaptureState(
        capture_day="2026-07-23",
        started_at_utc="2026-07-23T00:00:00Z",
        output_dir=Path("data/fixture"),
    )
    handle = BytesIO()
    utc_ns = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp() * 1e9)
    start = capture.ClockSample("rest_request_start", 1, 100, utc_ns, 1)
    end = capture.ClockSample("rest_response_end", 1, 200, utc_ns + 100, 1)
    response = capture.RestTransportResponse(
        status=200,
        final_url=capture.REST_URL,
        content_type="application/json",
        location=None,
        raw=b"oversized",
    )
    monkeypatch.setattr(capture, "MAX_REST_RESPONSE_BYTES", 4)
    capture._begin_rest_attempt(state, handle, start)
    with pytest.raises(capture.ParityCaptureError, match="response exceeds"):
        capture._record_rest_response(
            state,
            handle,
            response,
            ordinal=1,
            start=start,
            end=end,
            final=False,
        )
    envelope = json.loads(handle.getvalue().splitlines()[1])
    assert envelope["raw_body_complete"] is False
    assert base64.b64decode(envelope["raw_json_base64"]) == response.raw
    assert state.rest_attempts_completed == state.rest_responses_audited == 1


def test_rest_clock_identity_is_audited_before_rejection() -> None:
    state = capture.CaptureState(
        capture_day="2026-07-23",
        started_at_utc="2026-07-23T00:00:00Z",
        output_dir=Path("data/fixture"),
    )
    handle = BytesIO()
    utc_ns = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp() * 1e9)
    malformed = capture.ClockSample("wrong_source", 7, 100, utc_ns, 1)
    with pytest.raises(capture.ParityCaptureError, match="identity"):
        capture._begin_rest_attempt(state, handle, malformed)
    envelope = json.loads(handle.getvalue())
    assert envelope["clock_source"] == "wrong_source"
    assert envelope["clock_ordinal"] == 7
    assert state.clock_samples == [malformed]


def test_redirect_response_is_returned_for_audit_without_target_request() -> None:
    handler = capture._AuditNoRedirectHandler()
    original = object()
    returned = handler.http_error_302(
        capture.urllib.request.Request(capture.REST_URL),
        original,
        302,
        "Found",
        {"Location": "https://example.com/forbidden"},
    )
    assert returned is original

    state = capture.CaptureState(
        capture_day="2026-07-23",
        started_at_utc="2026-07-23T00:00:00Z",
        output_dir=Path("data/fixture"),
    )
    handle = BytesIO()
    utc_ns = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp() * 1e9)
    raw = b"redirect"
    response = capture.RestTransportResponse(
        status=302,
        final_url=capture.REST_URL,
        content_type="text/plain",
        location="https://example.com/forbidden",
        raw=raw,
    )
    start = capture.ClockSample("rest_request_start", 1, 100, utc_ns, 1)
    end = capture.ClockSample("rest_response_end", 1, 101, utc_ns + 1, 1)
    capture._begin_rest_attempt(state, handle, start)
    with pytest.raises(capture.ParityCaptureError, match="status"):
        capture._record_rest_response(
            state,
            handle,
            response,
            ordinal=1,
            start=start,
            end=end,
            final=False,
        )
    envelope = json.loads(handle.getvalue().splitlines()[1])
    assert envelope["http_status"] == 302
    assert envelope["location"] == "https://example.com/forbidden"
    assert base64.b64decode(envelope["raw_json_base64"]) == raw


def test_rest_transport_failure_persists_start_and_end_attempt_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utc_ns = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp() * 1e9)
    samples = iter(
        (
            capture.ClockSample("rest_request_start", 1, 100, utc_ns, 1),
            capture.ClockSample("rest_response_end", 1, 200, utc_ns + 100, 1),
        )
    )
    monkeypatch.setattr(capture, "enforce_disk_guard", lambda: 250)
    monkeypatch.setattr(capture, "_local_clock_sample", lambda *_: next(samples))
    state = capture.CaptureState(
        capture_day="2026-07-23",
        started_at_utc="2026-07-23T00:00:00Z",
        output_dir=Path("data/fixture"),
    )
    handle = BytesIO()

    def fail() -> capture.RestTransportResponse:
        raise capture.ParityCaptureError("transport fixture")

    with pytest.raises(capture.ParityCaptureError, match="fixture"):
        asyncio.run(
            capture._fetch_and_record_rest(
                state,
                handle,
                final=False,
                fetch=fail,
            )
        )
    lines = [json.loads(line) for line in handle.getvalue().splitlines()]
    assert [line["record_type"] for line in lines] == [
        "request_start",
        "transport_error",
    ]
    assert state.rest_attempts_started == state.rest_attempts_completed == 1
    assert state.rest_responses_audited == 0
    assert [sample.source for sample in state.clock_samples] == [
        "rest_request_start",
        "rest_response_end",
    ]


def test_rest_poller_stop_waits_for_inflight_response_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utc_ns = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp() * 1e9)
    samples = iter(
        (
            capture.ClockSample("rest_request_start", 1, 100, utc_ns, 1),
            capture.ClockSample("rest_response_end", 1, 200, utc_ns + 100, 1),
        )
    )
    monkeypatch.setattr(capture, "enforce_disk_guard", lambda: 250)
    monkeypatch.setattr(capture, "_local_clock_sample", lambda *_: next(samples))
    state = capture.CaptureState(
        capture_day="2026-07-23",
        started_at_utc="2026-07-23T00:00:00Z",
        output_dir=Path("data/fixture"),
    )
    handle = BytesIO()
    fetch_started = threading.Event()
    release_fetch = threading.Event()

    def fetch() -> capture.RestTransportResponse:
        fetch_started.set()
        assert release_fetch.wait(timeout=2)
        return capture.RestTransportResponse(
            status=200,
            final_url=capture.REST_URL,
            content_type="application/json",
            location=None,
            raw=rest_raw([rest_row(1)]),
        )

    async def exercise() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(
            capture._rest_poller(state, handle, 10**30, stop, fetch=fetch)
        )
        while not fetch_started.is_set():
            await asyncio.sleep(0)
        stop.set()
        release_fetch.set()
        await task

    asyncio.run(exercise())
    lines = [json.loads(line) for line in handle.getvalue().splitlines()]
    assert [line["record_type"] for line in lines] == ["request_start", "response"]
    assert state.rest_attempts_started == state.rest_attempts_completed == 1
    assert state.rest_responses_audited == 1


def test_parity_passes_with_unordered_overlapping_rest_windows() -> None:
    ws, snapshots = passing_inputs()
    result = capture.evaluate_rest_ws_parity(ws, snapshots)
    assert result["decision"] == "REST_WS_PARITY_PASS_PENDING_ARCHIVE"
    assert all(result["checks"].values())
    assert result["counts"]["ws_unique_ids"] == 1_200
    assert result["counts"]["common_ids"] == 1_200
    assert result["counts"]["minimum_adjacent_rest_overlap"] >= 975
    assert result["counts"]["common_field_mismatches"] == 0


def test_parity_rejects_field_conflict_and_missing_interior_trade() -> None:
    ws, snapshots = passing_inputs()
    changed = list(snapshots[-1].trades)
    changed[0] = replace(changed[0], price="1")
    snapshots[-1] = replace(snapshots[-1], trades=tuple(changed))
    result = capture.evaluate_rest_ws_parity(ws, snapshots)
    assert result["decision"] == "REJECT_NO_REPAIR"
    assert result["checks"]["no_conflicting_duplicate_ids"] is False

    ws, snapshots = passing_inputs()
    missing_id = "trade-600"
    snapshots = [
        replace(
            snapshot,
            trades=tuple(row for row in snapshot.trades if row.source_id != missing_id),
        )
        for snapshot in snapshots
    ]
    result = capture.evaluate_rest_ws_parity(ws, snapshots)
    assert result["decision"] == "REJECT_NO_REPAIR"
    assert result["checks"]["eligible_ws_complete_in_rest"] is False
    assert result["counts"]["missing_eligible_ws_ids_in_rest"] == 1


def test_parity_rejects_nonoverlapping_rest_windows_and_missing_final_marker() -> None:
    ws, snapshots = passing_inputs()
    snapshots[1] = replace(snapshots[1], trades=tuple(trade(10_000 + i) for i in range(1_000)))
    result = capture.evaluate_rest_ws_parity(ws, snapshots)
    assert result["checks"]["adjacent_rest_windows_overlap"] is False

    ws, snapshots = passing_inputs()
    snapshots[-1] = replace(snapshots[-1], final=False)
    result = capture.evaluate_rest_ws_parity(ws, snapshots)
    assert result["checks"]["one_final_rest_snapshot_last"] is False


def test_clock_integrity_sorts_cross_task_samples_before_validation() -> None:
    samples = [
        capture.ClockSample("capture_end", 1, 500, 1_500, 2),
        capture.ClockSample("websocket_receipt", 1, 200, 1_200, 2),
        capture.ClockSample("rest_response_end", 1, 400, 1_400, 3),
        capture.ClockSample("capture_start", 1, 100, 1_100, 4),
        capture.ClockSample("rest_request_start", 1, 300, 1_300, 2),
    ]
    audit = capture.evaluate_clock_integrity(
        samples,
        capture.ClockLedgerExpectation(1, 1, 1, 1),
    )
    assert audit["clock_contract_passed"] is True
    assert audit["utc_reversal_count"] == 0
    assert audit["nonincreasing_monotonic_count"] == 0
    assert audit["ledger_complete"] is True
    assert audit["monotonic_elapsed_ns"] == audit["utc_elapsed_ns"] == 400
    assert audit["maximum_sampling_uncertainty_ns"] == 4


def test_clock_reversal_forces_a_previously_passing_parity_to_reject() -> None:
    ws, snapshots = passing_inputs()
    parity = capture.evaluate_rest_ws_parity(ws, snapshots)
    audit = capture.evaluate_clock_integrity(
        [
            capture.ClockSample("capture_start", 1, 100, 1_000, 1),
            capture.ClockSample("capture_end", 1, 200, 900, 1),
        ],
        capture.ClockLedgerExpectation(0, 0, 0, 0),
    )
    assert audit["utc_reversal_count"] == 1
    assert audit["clock_contract_passed"] is False
    gated = capture.apply_clock_gate(parity, audit)
    assert gated["decision"] == "REJECT_NO_REPAIR"
    assert gated["checks"]["local_utc_nonreversing"] is False
    assert "local_utc_nonreversing" in gated["failures"]


def test_missing_or_duplicate_clock_samples_fail_closed() -> None:
    empty = capture.ClockLedgerExpectation(0, 0, 0, 0)
    assert capture.evaluate_clock_integrity([], empty)["clock_contract_passed"] is False
    duplicate = capture.evaluate_clock_integrity(
        [
            capture.ClockSample("capture_start", 1, 100, 1_000, 0),
            capture.ClockSample("capture_start", 1, 100, 1_001, 0),
            capture.ClockSample("capture_end", 1, 200, 1_100, 0),
        ],
        empty,
    )
    assert duplicate["nonincreasing_monotonic_count"] == 1
    assert duplicate["clock_contract_passed"] is False

    missing_ws = capture.evaluate_clock_integrity(
        [
            capture.ClockSample("capture_start", 1, 100, 1_000, 0),
            capture.ClockSample("capture_end", 1, 200, 1_100, 0),
        ],
        capture.ClockLedgerExpectation(1, 0, 0, 0),
    )
    assert missing_ws["ledger_complete"] is False
    assert missing_ws["missing_ledger_entries"] == 1

    misplaced_boundaries = capture.evaluate_clock_integrity(
        [
            capture.ClockSample("capture_end", 1, 100, 1_000, 0),
            capture.ClockSample("websocket_receipt", 1, 200, 1_100, 0),
            capture.ClockSample("capture_start", 1, 300, 1_200, 0),
        ],
        capture.ClockLedgerExpectation(1, 0, 0, 0),
    )
    assert misplaced_boundaries["boundary_ledger_complete"] is False
    assert misplaced_boundaries["ledger_complete"] is False


def test_manifest_is_hash_bound_immutable_and_outcome_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture, "REPO_ROOT", tmp_path)
    ws_path = tmp_path / "ws.gz"
    rest_path = tmp_path / "rest.gz"
    ws_path.write_bytes(b"ws")
    rest_path.write_bytes(b"rest")
    state = capture.CaptureState(
        capture_day="2026-07-23",
        started_at_utc="2026-07-23T00:00:00Z",
        ended_at_utc="2026-07-23T00:10:00Z",
        output_dir=tmp_path,
        ws_messages=10,
        ws_subscription_acks=1,
    )
    parity = {"decision": "REJECT_NO_REPAIR", "checks": {}, "failures": ["x"]}
    manifest = capture.build_manifest(
        state,
        parity,
        disk_used_gib_before_capture=250,
        ws_path=ws_path,
        rest_path=rest_path,
    )
    core = dict(manifest)
    observed = core.pop("manifest_hash_without_self")
    assert observed == capture.canonical_hash(core)
    assert manifest["outcome_boundary"] == {
        "bsea_clock_built": False,
        "candidate_incidence_opened": False,
        "binance_comparator_opened": False,
        "market_outcomes_opened": False,
        "returns_or_pnl_opened": False,
    }
    output = tmp_path / "manifest.json"
    capture.write_manifest(output, manifest)
    assert json.loads(output.read_text()) == manifest
    with pytest.raises(FileExistsError, match="immutable"):
        capture.write_manifest(output, manifest)


def test_production_entrypoint_has_no_duration_endpoint_or_disk_bypass() -> None:
    assert not inspect.signature(capture.run_capture).parameters
    with pytest.raises(SystemExit):
        capture.parse_args(["--duration", "1"])


def test_disk_guard_uses_repo_filesystem_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []

    def fake_used(path: Path) -> int:
        seen.append(path)
        return 300

    monkeypatch.setattr(capture, "used_gib", fake_used)
    with pytest.raises(capture.ParityCaptureError, match="disk guard"):
        capture.enforce_disk_guard()
    assert seen == [capture.REPO_ROOT]
