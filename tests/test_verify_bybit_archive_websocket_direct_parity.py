from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from training import verify_bybit_archive_websocket_direct_parity as parity


def make_trade(identifier: int, time_ms: int) -> parity.Trade:
    return parity.Trade(
        source_id=f"trade-{identifier}",
        symbol="BTCUSDT",
        side="Buy" if identifier % 2 == 0 else "Sell",
        price=Decimal(f"{100000 + identifier}.00"),
        size=Decimal("0.0010"),
        time_ms=time_ms,
    )


def ws_payload(rows: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {
            "topic": "publicTrade.BTCUSDT",
            "type": "snapshot",
            "ts": rows[-1]["T"],
            "data": rows,
        },
        separators=(",", ":"),
    ).encode()


def ws_row(identifier: int, time_ms: int) -> dict[str, object]:
    return {
        "T": time_ms,
        "s": "BTCUSDT",
        "S": "Buy" if identifier % 2 == 0 else "Sell",
        "v": "0.001",
        "p": str(100000 + identifier),
        "L": "PlusTick",
        "i": f"trade-{identifier}",
        "BT": False,
        "RPI": False,
        "seq": identifier,
    }


def capture_line(ordinal: int, raw: bytes) -> bytes:
    return (
        json.dumps(
            {
                "ordinal": ordinal,
                "receipt_utc_ns": 1_784_790_000_000_000_000 + ordinal,
                "receipt_monotonic_ns": 700_000_000 + ordinal,
                "receipt_clock_uncertainty_ns": 1,
                "frame_type": "text",
                "raw_frame_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_frame_base64": base64.b64encode(raw).decode(),
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def synthetic_capture(path: Path, *, duplicate: bool = False) -> None:
    start = parity.ARCHIVE_DAY_START_MS + 10_000
    lines: list[bytes] = []
    subscribe = b'{"op":"subscribe","success":true}'
    lines.append(capture_line(1, subscribe))
    ordinal = 2
    rows: list[dict[str, object]] = []
    for identifier in range(1_100):
        row = ws_row(identifier, start + identifier * 300)
        rows.append(row)
        if duplicate and identifier == 500:
            rows.append(dict(row))
        if len(rows) >= 100:
            raw = ws_payload(rows)
            lines.append(capture_line(ordinal, raw))
            ordinal += 1
            rows = []
    if rows:
        raw = ws_payload(rows)
        lines.append(capture_line(ordinal, raw))
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as out:
            for line in lines:
                out.write(line)


def archive_bytes(records: list[parity.Trade]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=parity.ARCHIVE_HEADER,
        lineterminator="\n",
    )
    writer.writeheader()
    for record in records:
        row = {
            "timestamp": str(Decimal(record.time_ms) / Decimal(1_000)),
            "symbol": record.symbol,
            "side": record.side,
            "size": str(record.size),
            "price": str(record.price),
            "tickDirection": "PlusTick",
            "trdMatchID": record.source_id,
            "grossValue": "1",
            "homeNotional": "1",
            "foreignNotional": "1",
            "RPI": "false",
        }
        writer.writerow(row)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as zipped:
        zipped.write(text.getvalue().encode())
    return output.getvalue()


def metadata_for(payload: bytes) -> parity.ArchiveMetadata:
    return parity.ArchiveMetadata(
        status=200,
        final_url=parity.ARCHIVE_URL,
        content_type="text/csv",
        content_length=len(payload),
        etag=parity.ARCHIVE_ETAG,
        last_modified="fixture",
        response_date="fixture",
    )


def test_protocol_constants_exclude_bsea_repair() -> None:
    assert parity.PROTOCOL_VERSION == "bybit_archive_websocket_direct_parity_v1"
    assert parity.ARCHIVE_URL.endswith("BTCUSDT2026-07-23.csv.gz")
    assert parity.EDGE_EXCLUSION_MS == 5_000
    assert parity.MINIMUM_INTERVAL_MS == 300_000
    assert parity.MINIMUM_INTERVAL_IDS == 1_000
    assert "rest_responses.ndjson.gz" not in Path(parity.__file__).read_text()


def test_validate_frozen_bindings_and_outcome_boundary() -> None:
    capture, rejection = parity.validate_frozen_bindings()
    assert capture["decision"] == "REJECT_NO_REPAIR"
    assert rejection["decision"] == "SOURCE_PARITY_REJECT_REST_WINDOW_OVERFLOW"
    assert not any(capture["outcome_boundary"].values())
    assert not any(rejection["outcome_boundary"].values())


def test_load_websocket_capture_uses_fixed_edges(tmp_path: Path) -> None:
    path = tmp_path / "capture.ndjson.gz"
    synthetic_capture(path)
    audit = parity.load_websocket_capture(path)
    assert audit.subscription_acks == 1
    assert audit.trade_frames == 11
    assert len(audit.records) == 1_100
    assert audit.start_ms == audit.first_ws_ms + 5_000
    assert audit.end_ms == audit.last_ws_ms - 5_000
    assert audit.end_ms - audit.start_ms >= 300_000
    assert len(audit.interval_records) >= 1_000
    assert len({row.source_id for row in audit.interval_records}) == len(
        audit.interval_records
    )


def test_load_websocket_capture_rejects_duplicate_interval_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.ndjson.gz"
    synthetic_capture(path, duplicate=True)
    with pytest.raises(
        parity.TerminalSourceFailure,
        match="duplicate IDs",
    ):
        parity.load_websocket_capture(path)


def test_archive_stream_hashes_full_payload_and_filters_interval() -> None:
    start = parity.ARCHIVE_DAY_START_MS + 10_000
    records = [make_trade(index, start + index * 1_000) for index in range(20)]
    payload = archive_bytes(records)
    audit = parity._read_archive(
        io.BytesIO(payload),
        metadata=metadata_for(payload),
        start_ms=start + 5_000,
        end_ms=start + 15_000,
    )
    assert audit.total_rows == 20
    assert audit.interval_rows == 10
    assert audit.compressed_bytes == len(payload)
    assert audit.compressed_sha256 == hashlib.sha256(payload).hexdigest()
    assert [row.source_id for row in audit.records] == [
        f"trade-{index}" for index in range(5, 15)
    ]


def test_archive_rejects_fractional_millisecond() -> None:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=parity.ARCHIVE_HEADER,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(
        {
            "timestamp": "1784764800.0001",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "size": "1",
            "price": "1",
            "tickDirection": "PlusTick",
            "trdMatchID": "trade",
            "grossValue": "1",
            "homeNotional": "1",
            "foreignNotional": "1",
            "RPI": "false",
        }
    )
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as zipped:
        zipped.write(text.getvalue().encode())
    payload = output.getvalue()
    with pytest.raises(
        parity.TerminalSourceFailure,
        match="exact millisecond",
    ):
        parity._read_archive(
            io.BytesIO(payload),
            metadata=metadata_for(payload),
            start_ms=parity.ARCHIVE_DAY_START_MS,
            end_ms=parity.ARCHIVE_DAY_END_MS,
        )


def test_archive_rejects_header_drift() -> None:
    payload_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=payload_buffer, mode="wb", mtime=0) as zipped:
        zipped.write(b"timestamp,symbol,side,size,price,trdMatchID\n")
    payload = payload_buffer.getvalue()
    with pytest.raises(parity.TerminalSourceFailure, match="header differs"):
        parity._read_archive(
            io.BytesIO(payload),
            metadata=metadata_for(payload),
            start_ms=parity.ARCHIVE_DAY_START_MS,
            end_ms=parity.ARCHIVE_DAY_END_MS,
        )


def test_exact_parity_pass_and_field_mismatch_fail() -> None:
    start = parity.ARCHIVE_DAY_START_MS + 10_000
    websocket = [make_trade(index, start + index) for index in range(10)]
    failures, diagnostics = parity.evaluate_parity(websocket, list(websocket))
    assert failures == ()
    assert diagnostics["common_ids"] == 10

    changed = list(websocket)
    changed[3] = parity.Trade(
        source_id=changed[3].source_id,
        symbol=changed[3].symbol,
        side=changed[3].side,
        price=changed[3].price + Decimal("1"),
        size=changed[3].size,
        time_ms=changed[3].time_ms,
    )
    failures, diagnostics = parity.evaluate_parity(websocket, changed)
    assert failures == ("parity:canonical_field_mismatch",)
    assert diagnostics["canonical_field_mismatches"] == 1


def test_exact_parity_rejects_missing_ids_and_time_reversal() -> None:
    start = parity.ARCHIVE_DAY_START_MS + 10_000
    websocket = [make_trade(index, start + index) for index in range(4)]
    archive = [websocket[0], websocket[2], websocket[1]]
    failures, diagnostics = parity.evaluate_parity(websocket, archive)
    assert "parity:websocket_ids_missing_from_archive" in failures
    assert "parity:archive_time_order" in failures
    assert diagnostics["websocket_ids_missing_from_archive"] == 1


def test_interval_gzip_is_deterministic_and_canonical() -> None:
    start = parity.ARCHIVE_DAY_START_MS + 10_000
    records = [
        make_trade(2, start + 2),
        make_trade(1, start + 1),
    ]
    first = parity.deterministic_interval_gzip(records)
    second = parity.deterministic_interval_gzip(records)
    assert first == second
    with gzip.open(io.BytesIO(first), "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["trade_id"] for row in rows] == ["trade-1", "trade-2"]
    assert rows[0]["size"] == "0.001"


def test_clean_guard_requires_tracked_clean_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "deadbeef\n", ""),
        ]
    )
    monkeypatch.setattr(parity, "_git", lambda *args: next(responses))
    assert parity.assert_protocol_committed() == "deadbeef"

    dirty = iter(
        [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "?? dirty\n", ""),
        ]
    )
    monkeypatch.setattr(parity, "_git", lambda *args: next(dirty))
    with pytest.raises(parity.DirectParityError, match="not HEAD-clean"):
        parity.assert_protocol_committed()


def test_run_checks_commit_before_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fetcher(**_: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("must not fetch")

    monkeypatch.setattr(
        parity,
        "assert_protocol_committed",
        lambda: (_ for _ in ()).throw(
            parity.DirectParityError("not committed")
        ),
    )
    with pytest.raises(parity.DirectParityError, match="not committed"):
        parity.run(fetcher=fetcher)
    assert called is False


def test_run_persists_terminal_source_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = parity.ARCHIVE_DAY_START_MS + 10_000
    records = tuple(make_trade(index, start + index) for index in range(1_001))
    websocket = parity.WebSocketAudit(
        records=records,
        total_frames=2,
        trade_frames=1,
        subscription_acks=1,
        pongs=0,
        first_ws_ms=start,
        last_ws_ms=start + 310_000,
        start_ms=start + 5_000,
        end_ms=start + 305_000,
        interval_records=records,
    )
    writes: list[tuple[Path, bytes]] = []
    monkeypatch.setattr(parity, "assert_protocol_committed", lambda: "deadbeef")
    monkeypatch.setattr(
        parity,
        "validate_frozen_bindings",
        lambda: (
            {"manifest_hash_without_self": "b" * 64},
            {"decision": "SOURCE_PARITY_REJECT_REST_WINDOW_OVERFLOW"},
        ),
    )
    monkeypatch.setattr(parity, "load_websocket_capture", lambda: websocket)
    monkeypatch.setattr(parity, "_disk_used_gib", lambda: 289)
    monkeypatch.setattr(
        parity,
        "atomic_write",
        lambda path, payload: (
            writes.append((path, payload)) or "written"
        ),
    )

    def failed_fetcher(**_: object) -> None:
        raise parity.TerminalSourceFailure("partial archive")

    report, statuses = parity.run(fetcher=failed_fetcher)
    assert report["decision"] == "RETIRE_BAWDP_V1_NO_REPAIR"
    assert report["passed"] is False
    assert report["failures"] == [
        "source:archive_transport_or_stream:partial archive"
    ]
    assert not any(report["outcome_boundary"].values())
    assert statuses == {"report": "written"}
    assert len(writes) == 1


def test_main_reports_terminal_rejection_without_archive_hash(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "transport_id": "BAWDP-v1",
        "decision": "RETIRE_BAWDP_V1_NO_REPAIR",
        "passed": False,
        "manifest_hash": "a" * 64,
        "archive": {"complete_archive_hash_available": False},
    }
    monkeypatch.setattr(
        parity,
        "run",
        lambda **_: (report, {"report": "written"}),
    )
    assert parity.main([]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["archive_sha256"] is None
    assert output["interval_rows"] == 0


def test_build_report_keeps_every_outcome_closed() -> None:
    start = parity.ARCHIVE_DAY_START_MS + 10_000
    records = tuple(make_trade(index, start + index) for index in range(1_001))
    websocket = parity.WebSocketAudit(
        records=records,
        total_frames=2,
        trade_frames=1,
        subscription_acks=1,
        pongs=0,
        first_ws_ms=start,
        last_ws_ms=start + 310_000,
        start_ms=start + 5_000,
        end_ms=start + 305_000,
        interval_records=records,
    )
    metadata = parity.ArchiveMetadata(
        200,
        parity.ARCHIVE_URL,
        "text/csv",
        parity.ARCHIVE_CONTENT_LENGTH,
        parity.ARCHIVE_ETAG,
        "fixture",
        "fixture",
    )
    archive = parity.ArchiveAudit(
        records,
        1_001,
        1_001,
        parity.ARCHIVE_CONTENT_LENGTH,
        "a" * 64,
        metadata,
    )
    report = parity.build_report(
        protocol_commit="deadbeef",
        capture_manifest={"manifest_hash_without_self": "b" * 64},
        websocket=websocket,
        archive=archive,
        failures=(),
        diagnostics={"common_ids": 1_001},
        interval_path=parity.DEFAULT_INTERVAL,
        interval_sha256="c" * 64,
        disk_used_gib=289,
    )
    assert report["passed"] is True
    assert report["decision"] == (
        "PASS_AUTHORIZES_ORTHOGONAL_CANDIDATE_BOUNDARY"
    )
    assert not any(report["outcome_boundary"].values())
    core = {
        key: value
        for key, value in report.items()
        if key != "manifest_hash"
    }
    assert report["manifest_hash"] == parity.canonical_hash(core)


def test_atomic_write_is_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parity, "REPOSITORY_ROOT", tmp_path)
    path = Path("results/output.json")
    assert parity.atomic_write(path, b"one") == "written"
    assert parity.atomic_write(path, b"one") == "verified"
    with pytest.raises(parity.DirectParityError, match="differs"):
        parity.atomic_write(path, b"two")
