from __future__ import annotations

import base64
import inspect
import json
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
    assert capture.REST_URL == (
        "https://api.bybit.com/v5/market/recent-trade?"
        "category=linear&symbol=BTCUSDT&limit=1000"
    )
    assert capture.WS_URL == "wss://stream.bybit.com/v5/public/linear"
    assert capture.WS_TOPIC == "publicTrade.BTCUSDT"
    assert capture.CAPTURE_SECONDS == 600
    assert capture.REST_INTERVAL_SECONDS == 1


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
    assert base64.b64decode(envelope["raw_json_base64"]) == raw
    assert envelope["raw_json_sha256"] == capture.sha256_bytes(raw)


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
    with pytest.raises(capture.ParityCaptureError, match="status"):
        capture._record_rest_response(
            state,
            handle,
            response,
            request_start_utc_ns=utc_ns,
            request_start_monotonic_ns=100,
            response_end_utc_ns=utc_ns + 1,
            response_end_monotonic_ns=101,
            final=False,
        )
    envelope = json.loads(handle.getvalue())
    assert envelope["http_status"] == 429
    assert base64.b64decode(envelope["raw_json_base64"]) == raw
    assert envelope["raw_json_sha256"] == capture.sha256_bytes(raw)


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
    with pytest.raises(capture.ParityCaptureError, match="status"):
        capture._record_rest_response(
            state,
            handle,
            response,
            request_start_utc_ns=utc_ns,
            request_start_monotonic_ns=100,
            response_end_utc_ns=utc_ns + 1,
            response_end_monotonic_ns=101,
            final=False,
        )
    envelope = json.loads(handle.getvalue())
    assert envelope["http_status"] == 302
    assert envelope["location"] == "https://example.com/forbidden"
    assert base64.b64decode(envelope["raw_json_base64"]) == raw


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
