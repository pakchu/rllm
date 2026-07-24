from __future__ import annotations

import asyncio
import json
import subprocess
import urllib.error
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from training import verify_deribit_options_trade_ledger as ledger


def trade_row(
    identifier: int,
    *,
    sequence: int | None = None,
    timestamp: int = ledger.HISTORY_START_MS + 1_000,
    instrument: str = "BTC-29JAN21-32000-C",
    direction: str = "buy",
    price: object = "0.01",
    amount: object = "1.0",
    iv: object | None = "50.0",
) -> dict[str, object]:
    row: dict[str, object] = {
        "trade_id": str(identifier),
        "trade_seq": identifier if sequence is None else sequence,
        "instrument_name": instrument,
        "timestamp": timestamp,
        "direction": direction,
        "price": price,
        "amount": amount,
        "tick_direction": identifier % 4,
        "index_price": "32000.00",
        "mark_price": "0.0100",
    }
    if iv is not None:
        row["iv"] = iv
    return row


def page_bytes(rows: list[dict[str, object]], has_more: bool) -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "result": {"trades": rows, "has_more": has_more},
        },
        separators=(",", ":"),
    ).encode()


def payload(
    request: ledger.PageRequest,
    rows: list[dict[str, object]],
    has_more: bool,
) -> ledger.HttpPayload:
    raw = page_bytes(rows, has_more)
    return ledger.HttpPayload(
        final_url=request.url,
        status=200,
        headers=(),
        raw=raw,
    )


def parsed_trade(
    identifier: int,
    *,
    timestamp: int = ledger.HISTORY_START_MS + 1_000,
    instrument: str = "BTC-29JAN21-32000-C",
    auxiliary_iv: str = "50",
) -> ledger.Trade:
    return ledger.parse_trade(
        trade_row(
            identifier,
            timestamp=timestamp,
            instrument=instrument,
            iv=auxiliary_iv,
        ),
        start_ms=timestamp - 1,
        end_ms=timestamp + 1,
    )


def ws_ack(identifier: int, method: str) -> bytes:
    del method
    return json.dumps(
        {"jsonrpc": "2.0", "id": identifier, "result": [ledger.WS_CHANNEL]},
        separators=(",", ":"),
    ).encode()


def ws_trades(rows: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "subscription",
            "params": {"channel": ledger.WS_CHANNEL, "data": rows},
        },
        separators=(",", ":"),
    ).encode()


def test_protocol_constants_freeze_absolute_window() -> None:
    assert ledger.PROTOCOL_VERSION.endswith("_v1")
    assert ledger.LIVE_END_MS - ledger.LIVE_START_MS == 20 * 60 * 1_000
    assert ledger.LIVE_DRAIN_END_MS == ledger.LIVE_END_MS + 5_000
    assert (
        ledger.LIVE_UNSUBSCRIBE_DEADLINE_MS
        == ledger.LIVE_END_MS + 10_000
    )
    assert ledger.HISTORY_END_MS - ledger.HISTORY_START_MS == 86_400_000
    assert ledger.WS_CHANNEL == "trades.option.BTC.100ms"


def test_parse_trade_preserves_decimal_and_null_auxiliary() -> None:
    row = trade_row(1, price="1E-8", amount="1.2300", iv=None)
    trade = ledger.parse_trade(
        row,
        start_ms=ledger.HISTORY_START_MS,
        end_ms=ledger.HISTORY_END_MS,
    )
    assert trade.hard_row()["price"] == "0.00000001"
    assert trade.hard_row()["amount"] == "1.23"
    assert trade.auxiliary_map()["iv"] is None
    assert trade.auxiliary_map()["block_trade_id"] is None


@pytest.mark.parametrize("field", ledger.HARD_FIELDS)
def test_parse_trade_requires_every_hard_field(field: str) -> None:
    row = trade_row(1)
    del row[field]
    with pytest.raises(ledger.TerminalSourceFailure, match="hard fields"):
        ledger.parse_trade(
            row,
            start_ms=ledger.HISTORY_START_MS,
            end_ms=ledger.HISTORY_END_MS,
        )


@pytest.mark.parametrize(
    ("instrument", "match"),
    [
        ("ETH-29JAN21-32000-C", "BTC inverse option"),
        ("BTC_USDC-29JAN21-32000-C", "BTC inverse option"),
        ("BTC-31FEB21-32000-C", "expiry date"),
        ("BTC-29JAN21-32000-X", "BTC inverse option"),
    ],
)
def test_parse_trade_rejects_nonfrozen_symbol(
    instrument: str,
    match: str,
) -> None:
    with pytest.raises(ledger.TerminalSourceFailure, match=match):
        ledger.parse_trade(
            trade_row(1, instrument=instrument),
            start_ms=ledger.HISTORY_START_MS,
            end_ms=ledger.HISTORY_END_MS,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("trade_id", "abc", "decimal BTC ID"),
        ("price", 0, "positive"),
        ("amount", -1, "positive"),
        ("price", 1.5, "decimal literal"),
        ("iv", -1, "nonnegative"),
    ],
)
def test_parse_trade_rejects_invalid_identity(
    field: str,
    value: object,
    match: str,
) -> None:
    row = trade_row(1)
    row[field] = value
    with pytest.raises(ledger.TerminalSourceFailure, match=match):
        ledger.parse_trade(
            row,
            start_ms=ledger.HISTORY_START_MS,
            end_ms=ledger.HISTORY_END_MS,
        )


def test_parse_trade_rejects_unknown_schema_field() -> None:
    row = trade_row(1)
    row["future_label"] = 1
    with pytest.raises(ledger.TerminalSourceFailure, match="unknown fields"):
        ledger.parse_trade(
            row,
            start_ms=ledger.HISTORY_START_MS,
            end_ms=ledger.HISTORY_END_MS,
        )


def test_page_request_uses_exact_initial_and_inclusive_cursor() -> None:
    initial = ledger.page_request(
        ledger.HISTORY_ENDPOINT,
        start_ms=ledger.HISTORY_START_MS,
        end_ms=ledger.HISTORY_END_MS,
        start_id=None,
    )
    assert dict(initial.params) == {
        "count": "1000",
        "currency": "BTC",
        "end_timestamp": str(ledger.HISTORY_END_MS - 1),
        "kind": "option",
        "sorting": "asc",
        "start_timestamp": str(ledger.HISTORY_START_MS),
    }
    continuation = ledger.page_request(
        ledger.HISTORY_ENDPOINT,
        start_ms=None,
        end_ms=ledger.HISTORY_END_MS,
        start_id="105",
    )
    assert dict(continuation.params)["start_id"] == "105"
    assert "start_timestamp" not in dict(continuation.params)


def test_paginate_requires_and_discards_exact_overlap() -> None:
    calls: list[ledger.PageRequest] = []

    def fetch(request: ledger.PageRequest) -> ledger.HttpPayload:
        calls.append(request)
        if len(calls) == 1:
            return payload(
                request,
                [trade_row(100), trade_row(105)],
                True,
            )
        return payload(
            request,
            [trade_row(105), trade_row(110)],
            False,
        )

    audit = ledger.paginate(
        endpoint=ledger.HISTORY_ENDPOINT,
        start_ms=ledger.HISTORY_START_MS,
        end_ms=ledger.HISTORY_END_MS,
        fetcher=fetch,
        sleep=lambda _: None,
    )
    assert [trade.trade_id for trade in audit.accepted] == [
        "100",
        "105",
        "110",
    ]
    assert dict(calls[1].params)["start_id"] == "105"


def test_paginate_rejects_missing_or_conflicting_overlap() -> None:
    def missing(request: ledger.PageRequest) -> ledger.HttpPayload:
        if "start_timestamp" in dict(request.params):
            return payload(request, [trade_row(100), trade_row(105)], True)
        return payload(request, [trade_row(110)], False)

    with pytest.raises(ledger.TerminalSourceFailure, match="boundary ID"):
        ledger.paginate(
            endpoint=ledger.HISTORY_ENDPOINT,
            start_ms=ledger.HISTORY_START_MS,
            end_ms=ledger.HISTORY_END_MS,
            fetcher=missing,
            sleep=lambda _: None,
        )

    def conflict(request: ledger.PageRequest) -> ledger.HttpPayload:
        if "start_timestamp" in dict(request.params):
            return payload(request, [trade_row(100), trade_row(105)], True)
        changed = trade_row(105)
        changed["price"] = "0.02"
        return payload(request, [changed, trade_row(110)], False)

    with pytest.raises(ledger.TerminalSourceFailure, match="record differs"):
        ledger.paginate(
            endpoint=ledger.HISTORY_ENDPOINT,
            start_ms=ledger.HISTORY_START_MS,
            end_ms=ledger.HISTORY_END_MS,
            fetcher=conflict,
            sleep=lambda _: None,
        )


def test_paginate_rejects_no_advancement_and_duplicate_accepted_id() -> None:
    def no_advance(request: ledger.PageRequest) -> ledger.HttpPayload:
        if "start_timestamp" in dict(request.params):
            return payload(request, [trade_row(100)], True)
        return payload(request, [trade_row(100)], True)

    with pytest.raises(ledger.TerminalSourceFailure, match="does not advance"):
        ledger.paginate(
            endpoint=ledger.HISTORY_ENDPOINT,
            start_ms=ledger.HISTORY_START_MS,
            end_ms=ledger.HISTORY_END_MS,
            fetcher=no_advance,
            sleep=lambda _: None,
        )

    calls = 0

    def final_overlap_only(request: ledger.PageRequest) -> ledger.HttpPayload:
        nonlocal calls
        calls += 1
        if calls == 1:
            return payload(request, [trade_row(100)], True)
        return payload(request, [trade_row(100)], False)

    with pytest.raises(ledger.TerminalSourceFailure, match="does not advance"):
        ledger.paginate(
            endpoint=ledger.HISTORY_ENDPOINT,
            start_ms=ledger.HISTORY_START_MS,
            end_ms=ledger.HISTORY_END_MS,
            fetcher=final_overlap_only,
            sleep=lambda _: None,
        )

    calls = 0

    def duplicate(request: ledger.PageRequest) -> ledger.HttpPayload:
        nonlocal calls
        calls += 1
        if calls == 1:
            return payload(
                request,
                [trade_row(100), trade_row(105), trade_row(110)],
                True,
            )
        return payload(
            request,
            [trade_row(110), trade_row(115), trade_row(105)],
            False,
        )

    with pytest.raises(
        ledger.TerminalSourceFailure,
        match="strictly increase|repeats",
    ):
        ledger.paginate(
            endpoint=ledger.HISTORY_ENDPOINT,
            start_ms=ledger.HISTORY_START_MS,
            end_ms=ledger.HISTORY_END_MS,
            fetcher=duplicate,
            sleep=lambda _: None,
        )


def test_replay_pages_rejects_hard_drift() -> None:
    def original(request: ledger.PageRequest) -> ledger.HttpPayload:
        return payload(request, [trade_row(100), trade_row(105)], False)

    audit = ledger.paginate(
        endpoint=ledger.HISTORY_ENDPOINT,
        start_ms=ledger.HISTORY_START_MS,
        end_ms=ledger.HISTORY_END_MS,
        fetcher=original,
        sleep=lambda _: None,
    )

    def drift(request: ledger.PageRequest) -> ledger.HttpPayload:
        changed = trade_row(105)
        changed["direction"] = "sell"
        return payload(request, [trade_row(100), changed], False)

    with pytest.raises(ledger.TerminalSourceFailure, match="replay differs"):
        ledger.replay_pages(
            audit,
            start_ms=ledger.HISTORY_START_MS,
            end_ms=ledger.HISTORY_END_MS,
            fetcher=drift,
            sleep=lambda _: None,
        )


def test_http_retries_prebody_but_never_partial_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ledger.page_request(
        ledger.HISTORY_ENDPOINT,
        start_ms=ledger.HISTORY_START_MS,
        end_ms=ledger.HISTORY_END_MS,
        start_id=None,
    )

    class Response:
        status = 200
        headers: dict[str, str] = {}

        def __init__(self, *, fail_read: bool) -> None:
            self.fail_read = fail_read

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, _: int) -> bytes:
            if self.fail_read:
                raise TimeoutError("partial")
            return b"{}"

        def geturl(self) -> str:
            return request.url

    class PrebodyOpener:
        calls = 0

        def open(self, *_: object, **__: object) -> Response:
            self.calls += 1
            if self.calls == 1:
                raise urllib.error.URLError("connect")
            return Response(fail_read=False)

    prebody = PrebodyOpener()
    sleeps: list[float] = []
    monkeypatch.setattr(
        ledger.urllib.request, "build_opener", lambda *args: prebody
    )
    monkeypatch.setattr(ledger.time, "sleep", sleeps.append)
    fetched = ledger._default_fetch(request)
    assert fetched.raw == b"{}"
    assert prebody.calls == 2
    assert sleeps == [5.0]

    class PartialOpener:
        calls = 0

        def open(self, *_: object, **__: object) -> Response:
            self.calls += 1
            return Response(fail_read=True)

    partial = PartialOpener()
    monkeypatch.setattr(
        ledger.urllib.request, "build_opener", lambda *args: partial
    )
    with pytest.raises(
        ledger.TerminalSourceFailure,
        match="after transfer began",
    ):
        ledger._default_fetch(request)
    assert partial.calls == 1


def test_websocket_audit_filters_fixed_window_and_requires_acks() -> None:
    before = trade_row(
        1,
        timestamp=ledger.LIVE_START_MS - 1,
        instrument="BTC-31JUL26-100000-C",
    )
    inside = trade_row(
        2,
        timestamp=ledger.LIVE_START_MS,
        instrument="BTC-31JUL26-100000-C",
    )
    at_end = trade_row(
        3,
        timestamp=ledger.LIVE_END_MS,
        instrument="BTC-31JUL26-100000-P",
    )
    audit = ledger.audit_websocket_messages(
        [
            ws_ack(1, "subscribe"),
            ws_trades([before, inside, at_end]),
            ws_ack(2, "unsubscribe"),
        ]
    )
    assert [trade.trade_id for trade in audit.trades] == ["2"]
    with pytest.raises(ledger.TerminalSourceFailure, match="incomplete"):
        ledger.audit_websocket_messages(
            [ws_ack(1, "subscribe"), ws_trades([inside])]
        )


def test_websocket_rejects_duplicate_and_trade_after_unsubscribe() -> None:
    row = trade_row(
        2,
        timestamp=ledger.LIVE_START_MS,
        instrument="BTC-31JUL26-100000-C",
    )
    with pytest.raises(ledger.TerminalSourceFailure, match="repeats trade ID"):
        ledger.audit_websocket_messages(
            [
                ws_ack(1, "subscribe"),
                ws_trades([row]),
                ws_trades([row]),
                ws_ack(2, "unsubscribe"),
            ]
        )
    with pytest.raises(
        ledger.TerminalSourceFailure, match="after unsubscribe"
    ):
        ledger.audit_websocket_messages(
            [
                ws_ack(1, "subscribe"),
                ws_ack(2, "unsubscribe"),
                ws_trades([row]),
            ]
        )


def test_hard_parity_ignores_auxiliary_but_not_hard_drift() -> None:
    left = parsed_trade(1)
    changed_aux = ledger.Trade(
        trade_id=left.trade_id,
        trade_seq=left.trade_seq,
        instrument_name=left.instrument_name,
        timestamp=left.timestamp,
        direction=left.direction,
        price=left.price,
        amount=left.amount,
        auxiliary=tuple(
            (field, "60" if field == "iv" else value)
            for field, value in left.auxiliary
        ),
    )
    failures, diagnostics = ledger.evaluate_hard_parity(
        [left], [changed_aux]
    )
    assert failures == ()
    assert diagnostics["auxiliary_value_mismatches"]["iv"] == 1

    hard_drift = ledger.Trade(
        trade_id=left.trade_id,
        trade_seq=left.trade_seq,
        instrument_name=left.instrument_name,
        timestamp=left.timestamp,
        direction="sell",
        price=left.price,
        amount=left.amount,
        auxiliary=left.auxiliary,
    )
    failures, diagnostics = ledger.evaluate_hard_parity(
        [left], [hard_drift]
    )
    assert failures == ("parity:hard_field_mismatch",)
    assert diagnostics["hard_field_mismatches"] == 1


def test_validate_instrument_sequences_rejects_duplicate_or_time_reverse() -> None:
    first = parsed_trade(1)
    duplicate = ledger.Trade(
        trade_id="2",
        trade_seq=first.trade_seq,
        instrument_name=first.instrument_name,
        timestamp=first.timestamp + 1,
        direction=first.direction,
        price=first.price,
        amount=first.amount,
        auxiliary=first.auxiliary,
    )
    with pytest.raises(ledger.TerminalSourceFailure, match="repeats"):
        ledger.validate_instrument_sequences([first, duplicate], "fixture")

    reverse = ledger.Trade(
        trade_id="3",
        trade_seq=first.trade_seq + 1,
        instrument_name=first.instrument_name,
        timestamp=first.timestamp - 1,
        direction=first.direction,
        price=first.price,
        amount=first.amount,
        auxiliary=first.auxiliary,
    )
    with pytest.raises(ledger.TerminalSourceFailure, match="time reverses"):
        ledger.validate_instrument_sequences([first, reverse], "fixture")


def test_clock_samples_enforce_brackets_and_monotonicity() -> None:
    before = ledger.ClockSample(
        request_start_utc_ns=1_000_000_000,
        response_end_utc_ns=1_100_000_000,
        request_start_monotonic_ns=10,
        response_end_monotonic_ns=110,
        server_ms=1_050,
    )
    after = ledger.ClockSample(
        request_start_utc_ns=2_000_000_000,
        response_end_utc_ns=2_100_000_000,
        request_start_monotonic_ns=210,
        response_end_monotonic_ns=310,
        server_ms=2_050,
    )
    ledger.validate_clock_samples(before, after)
    bad = ledger.ClockSample(
        request_start_utc_ns=900_000_000,
        response_end_utc_ns=950_000_000,
        request_start_monotonic_ns=210,
        response_end_monotonic_ns=310,
        server_ms=925,
    )
    with pytest.raises(ledger.TerminalSourceFailure, match="UTC reverses"):
        ledger.validate_clock_samples(before, bad)


def test_clean_guard_requires_committed_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ledger,
        "sha256_file",
        lambda path: (
            ledger.BOUNDARY_SHA256
            if path == ledger.BOUNDARY_PATH
            else "fixture"
        ),
    )
    responses = [
        subprocess.CompletedProcess([], 0, "ok\n", ""),
        subprocess.CompletedProcess([], 0, "ok\n", ""),
        subprocess.CompletedProcess([], 0, "ok\n", ""),
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 0, "deadbeef\n", ""),
    ]
    monkeypatch.setattr(ledger, "_git", lambda *args: responses.pop(0))
    assert ledger.assert_protocol_committed() == "deadbeef"


def test_disk_guard_uses_reported_used_not_reserved_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    monkeypatch.setattr(
        ledger.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(
            total=1_000 * gib,
            used=298 * gib,
            free=650 * gib,
        ),
    )
    assert ledger.assert_disk_guard() == 298
    monkeypatch.setattr(
        ledger.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(
            total=1_000 * gib,
            used=299 * gib + 1,
            free=649 * gib,
        ),
    )
    with pytest.raises(ledger.LedgerError, match="headroom"):
        ledger.assert_disk_guard()


def _bulk_rows(
    *,
    start_ms: int,
    count: int,
    instruments: int,
    live: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expiry = "31JUL26" if live else "29JAN21"
    for index in range(count):
        side = "C" if index % 2 == 0 else "P"
        rows.append(
            trade_row(
                10_000 + index,
                sequence=index + 1,
                timestamp=start_ms + index,
                instrument=(
                    f"BTC-{expiry}-{30000 + (index % instruments) * 1000}-{side}"
                ),
            )
        )
    return rows


def test_run_mocked_pass_and_report_excludes_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(ledger, "assert_protocol_committed", lambda: "deadbeef")
    monkeypatch.setattr(ledger, "assert_disk_guard", lambda: 10)

    history_rows = _bulk_rows(
        start_ms=ledger.HISTORY_START_MS + 1_000,
        count=56,
        instruments=8,
        live=False,
    )
    live_rows = _bulk_rows(
        start_ms=ledger.LIVE_START_MS,
        count=100,
        instruments=4,
        live=True,
    )

    def fetch(request: ledger.PageRequest) -> ledger.HttpPayload:
        if request.endpoint == ledger.HISTORY_ENDPOINT:
            return payload(request, history_rows, False)
        if request.endpoint == ledger.RECENT_ENDPOINT:
            return payload(request, live_rows, False)
        raise AssertionError(request)

    websocket_trades = tuple(
        ledger.parse_trade(
            row,
            start_ms=ledger.LIVE_START_MS,
            end_ms=ledger.LIVE_END_MS,
        )
        for row in live_rows
    )
    ws_audit = ledger.WebSocketAudit(
        trades=websocket_trades,
        raw_messages=(ws_ack(1, "subscribe"), ws_ack(2, "unsubscribe")),
        subscription_ack=True,
        unsubscribe_ack=True,
        messages=2,
        bytes_read=10,
    )
    before = ledger.ClockSample(
        1_000_000_000,
        1_100_000_000,
        10,
        110,
        1_050,
    )
    after = ledger.ClockSample(
        2_000_000_000,
        2_100_000_000,
        210,
        310,
        2_050,
    )

    async def live_capture() -> tuple[
        ledger.WebSocketAudit,
        ledger.ClockSample,
        ledger.ClockSample,
    ]:
        return ws_audit, before, after

    report = asyncio.run(
        ledger.run(
            report_path=tmp_path / "report.json",
            data_dir=tmp_path / "data",
            sentinel_path=tmp_path / "sentinel",
            fetcher=fetch,
            live_capture=live_capture,
            sleep=lambda _: None,
        )
    )
    assert report["decision"] == "SOURCE_PARITY_PASS"
    assert not any(report["outcome_boundary"].values())

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                nested
                for child in value.values()
                for nested in keys(child)
            }
        if isinstance(value, list):
            return {
                nested for child in value for nested in keys(child)
            }
        return set()

    assert not (
        {"profit", "pnl", "return", "cagr", "mdd", "drawdown", "reward"}
        & {key.lower() for key in keys(report)}
    )


def test_run_checks_clean_guard_before_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def fetch(_: ledger.PageRequest) -> ledger.HttpPayload:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(
        ledger,
        "assert_protocol_committed",
        lambda: (_ for _ in ()).throw(ledger.LedgerError("dirty")),
    )
    with pytest.raises(ledger.LedgerError, match="dirty"):
        asyncio.run(
            ledger.run(
                report_path=tmp_path / "report.json",
                data_dir=tmp_path / "data",
                sentinel_path=tmp_path / "sentinel",
                fetcher=fetch,
            )
        )
    assert not called


def test_run_persists_terminal_source_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(ledger, "assert_protocol_committed", lambda: "deadbeef")
    monkeypatch.setattr(ledger, "assert_disk_guard", lambda: 10)

    def fetch(_: ledger.PageRequest) -> ledger.HttpPayload:
        raise ledger.TerminalSourceFailure("fixture failure")

    report_path = tmp_path / "reject.json"
    with pytest.raises(ledger.TerminalSourceFailure, match="fixture failure"):
        asyncio.run(
            ledger.run(
                report_path=report_path,
                data_dir=tmp_path / "data",
                sentinel_path=tmp_path / "sentinel",
                fetcher=fetch,
                sleep=lambda _: None,
            )
        )
    report = json.loads(report_path.read_text())
    assert report["decision"] == "SOURCE_PARITY_REJECT"
    assert report["failures"] == [
        "source:TerminalSourceFailure:fixture failure"
    ]
    assert not any(report["outcome_boundary"].values())


def test_reserve_one_shot_never_overwrites(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "sentinel"
    digest = ledger.reserve_one_shot(
        sentinel_path=sentinel,
        report_path=tmp_path / "report.json",
        data_dir=tmp_path / "data",
        commit="deadbeef",
    )
    original = sentinel.read_bytes()
    assert digest == ledger.sha256_bytes(original)
    with pytest.raises(ledger.LedgerError, match="sentinel already exists"):
        ledger.reserve_one_shot(
            sentinel_path=sentinel,
            report_path=tmp_path / "report.json",
            data_dir=tmp_path / "data",
            commit="deadbeef",
        )
    assert sentinel.read_bytes() == original


def test_verifier_source_has_no_outcome_loader() -> None:
    source = Path(ledger.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "read_parquet",
        "pandas",
        "btcusdt_5m",
        "funding_history",
        "strict_mdd",
        "backtest",
    ):
        assert forbidden not in source
