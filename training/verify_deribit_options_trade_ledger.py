"""Verify Deribit BTC option public-trade source identity without outcomes."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse


PROTOCOL_VERSION = "deribit_options_trade_ledger_source_parity_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_PATH = Path(
    "docs/deribit-options-trade-ledger-source-boundary-2026-07-24.md"
)
BOUNDARY_SHA256 = (
    "82532408ae44dac0cdc907e181fcd130879f6ec67479e443f471a2d984ca72ee"
)
SCRIPT_PATH = Path("training/verify_deribit_options_trade_ledger.py")
TEST_PATH = Path("tests/test_verify_deribit_options_trade_ledger.py")

HISTORY_ENDPOINT = (
    "https://history.deribit.com/api/v2/"
    "public/get_last_trades_by_currency"
)
RECENT_ENDPOINT = (
    "https://www.deribit.com/api/v2/"
    "public/get_last_trades_by_currency"
)
TIME_ENDPOINT = "https://www.deribit.com/api/v2/public/get_time"
WS_ENDPOINT = "wss://www.deribit.com/ws/api/v2"
WS_CHANNEL = "trades.option.BTC.100ms"

LIVE_START_MS = int(
    datetime(2026, 7, 24, 13, 0, tzinfo=timezone.utc).timestamp() * 1_000
)
LIVE_END_MS = int(
    datetime(2026, 7, 24, 13, 20, tzinfo=timezone.utc).timestamp() * 1_000
)
LIVE_DRAIN_END_MS = LIVE_END_MS + 5_000
LIVE_UNSUBSCRIBE_DEADLINE_MS = LIVE_END_MS + 10_000

HISTORY_START_MS = int(
    datetime(2021, 1, 4, tzinfo=timezone.utc).timestamp() * 1_000
)
HISTORY_END_MS = int(
    datetime(2021, 1, 5, tzinfo=timezone.utc).timestamp() * 1_000
)

PAGE_SIZE = 1_000
MAXIMUM_PAGES = 10_000
MINIMUM_LIVE_TRADES = 100
MINIMUM_LIVE_INSTRUMENTS = 4
MINIMUM_HISTORY_TRADES = 50
MINIMUM_HISTORY_INSTRUMENTS = 8
MAXIMUM_PAGE_BYTES = 32 * 1024 * 1024
MAXIMUM_WS_MESSAGE_BYTES = 32 * 1024 * 1024
MAXIMUM_WS_CAPTURE_BYTES = 256 * 1024 * 1024
HTTP_TIMEOUT_SECONDS = 30
HTTP_PAGE_INTERVAL_SECONDS = 0.25
RETRY_DELAYS_SECONDS = (5.0, 15.0)
DISK_LIMIT_GIB = 300
DISK_HEADROOM_BYTES = 1 * 1024**3
USER_AGENT = "rllm-deribit-option-ledger/1.0"

DEFAULT_REPORT = Path(
    "results/deribit_options_trade_ledger_source_parity_2026-07-24.json"
)
DEFAULT_SENTINEL = Path(
    "results/.deribit_options_trade_ledger_source_parity_2026-07-24.started"
)
DEFAULT_DATA_DIR = Path(
    "data/deribit_options_trade_ledger_source_parity_2026-07-24"
)

OPTION_RE = re.compile(
    r"^BTC-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-"
    r"(?P<strike>\d+(?:\.\d+)?)-(?P<option_type>[CP])$"
)
MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

HARD_FIELDS = (
    "trade_id",
    "trade_seq",
    "instrument_name",
    "timestamp",
    "direction",
    "price",
    "amount",
)
AUXILIARY_FIELDS = (
    "tick_direction",
    "index_price",
    "mark_price",
    "iv",
    "block_trade_id",
    "block_trade_leg_count",
    "combo_id",
    "combo_trade_id",
    "block_rfq_id",
)
IGNORED_DOCUMENTED_FIELDS = frozenset(
    {
        "contracts",
        "liquidation",
        "starbase_timestamp",
        "starbase_match_id",
    }
)
ALLOWED_TRADE_FIELDS = frozenset(
    {*HARD_FIELDS, *AUXILIARY_FIELDS, *IGNORED_DOCUMENTED_FIELDS}
)


class LedgerError(RuntimeError):
    """The frozen DOTL protocol cannot be evaluated."""


class TerminalSourceFailure(LedgerError):
    """A no-repair source identity failure."""


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: str
    trade_seq: int
    instrument_name: str
    timestamp: int
    direction: str
    price: Decimal
    amount: Decimal
    auxiliary: tuple[tuple[str, str | None], ...]

    def hard_row(self) -> dict[str, str]:
        return {
            "trade_id": self.trade_id,
            "trade_seq": str(self.trade_seq),
            "instrument_name": self.instrument_name,
            "timestamp": str(self.timestamp),
            "direction": self.direction,
            "price": canonical_positive_decimal(self.price),
            "amount": canonical_positive_decimal(self.amount),
        }

    def auxiliary_map(self) -> dict[str, str | None]:
        return dict(self.auxiliary)


@dataclass(frozen=True, slots=True)
class PageRequest:
    endpoint: str
    params: tuple[tuple[str, str], ...]

    @property
    def url(self) -> str:
        return f"{self.endpoint}?{urllib.parse.urlencode(self.params)}"

    def as_dict(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint, "params": dict(self.params), "url": self.url}


@dataclass(frozen=True, slots=True)
class HttpPayload:
    final_url: str
    status: int
    headers: tuple[tuple[str, str], ...]
    raw: bytes


@dataclass(frozen=True, slots=True)
class PageCommitment:
    request: PageRequest
    raw_sha256: str
    raw_bytes: int
    hard_hash: str
    rows: int
    has_more: bool
    first_id: str
    last_id: str
    response_id: str | int | None
    response_status: int
    response_final_url: str
    response_headers: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.as_dict(),
            "raw_sha256": self.raw_sha256,
            "raw_bytes": self.raw_bytes,
            "hard_hash": self.hard_hash,
            "rows": self.rows,
            "has_more": self.has_more,
            "first_id": self.first_id,
            "last_id": self.last_id,
            "response_id": self.response_id,
            "response_status": self.response_status,
            "response_final_url": self.response_final_url,
            "response_headers": dict(self.response_headers),
        }


@dataclass(frozen=True, slots=True)
class PaginationAudit:
    accepted: tuple[Trade, ...]
    pages: tuple[PageCommitment, ...]
    auxiliary_presence: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ClockSample:
    request_start_utc_ns: int
    response_end_utc_ns: int
    request_start_monotonic_ns: int
    response_end_monotonic_ns: int
    server_ms: int

    def as_dict(self) -> dict[str, int]:
        return {
            "request_start_utc_ns": self.request_start_utc_ns,
            "response_end_utc_ns": self.response_end_utc_ns,
            "request_start_monotonic_ns": self.request_start_monotonic_ns,
            "response_end_monotonic_ns": self.response_end_monotonic_ns,
            "server_ms": self.server_ms,
            "round_trip_ns": (
                self.response_end_monotonic_ns
                - self.request_start_monotonic_ns
            ),
        }


@dataclass(frozen=True, slots=True)
class WebSocketAudit:
    trades: tuple[Trade, ...]
    raw_messages: tuple[bytes, ...]
    subscription_ack: bool
    unsubscribe_ack: bool
    messages: int
    bytes_read: int


Fetcher = Callable[[PageRequest], HttpPayload]
Sleep = Callable[[float], None]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def repository_path(path: Path) -> Path:
    return REPOSITORY_ROOT / path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload).rstrip(b"\n"))


def canonical_positive_decimal(value: Decimal) -> str:
    if not value.is_finite() or value <= 0:
        raise LedgerError("decimal must be finite and positive")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def canonical_nonnegative_decimal(value: Decimal) -> str:
    if not value.is_finite() or value < 0:
        raise LedgerError("decimal must be finite and nonnegative")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TerminalSourceFailure(f"{field} must be an integer")
    if positive and value <= 0:
        raise TerminalSourceFailure(f"{field} must be positive")
    if not positive and value < 0:
        raise TerminalSourceFailure(f"{field} must be nonnegative")
    return value


def _decimal(
    value: Any,
    field: str,
    *,
    positive: bool,
) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal, str)):
        raise TerminalSourceFailure(f"{field} must be a decimal literal")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise TerminalSourceFailure(f"{field} is not a decimal") from exc
    if not parsed.is_finite():
        raise TerminalSourceFailure(f"{field} must be finite")
    if positive and parsed <= 0:
        raise TerminalSourceFailure(f"{field} must be positive")
    if not positive and parsed < 0:
        raise TerminalSourceFailure(f"{field} must be nonnegative")
    return parsed


def _optional_identifier(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TerminalSourceFailure(f"{field} must be an identifier")
    rendered = str(value)
    if not rendered or len(rendered) > 256:
        raise TerminalSourceFailure(f"{field} must be a bounded identifier")
    return rendered


def _expiry_ms(instrument_name: str) -> int:
    match = OPTION_RE.fullmatch(instrument_name)
    if match is None:
        raise TerminalSourceFailure("instrument_name is not a BTC inverse option")
    expiry = match.group("expiry")
    day = int(expiry[:-5])
    month_code = expiry[-5:-2]
    year = 2000 + int(expiry[-2:])
    month = MONTHS.get(month_code)
    if month is None:
        raise TerminalSourceFailure("instrument expiry month is invalid")
    try:
        timestamp = datetime(
            year, month, day, 8, 0, tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise TerminalSourceFailure("instrument expiry date is invalid") from exc
    return int(timestamp.timestamp() * 1_000)


def parse_trade(
    row: Mapping[str, Any],
    *,
    start_ms: int,
    end_ms: int,
) -> Trade:
    unknown = set(row) - ALLOWED_TRADE_FIELDS
    if unknown:
        raise TerminalSourceFailure(
            f"trade schema has unknown fields: {sorted(unknown)!r}"
        )
    missing = set(HARD_FIELDS) - set(row)
    if missing:
        raise TerminalSourceFailure(
            f"trade schema misses hard fields: {sorted(missing)!r}"
        )

    trade_id = row["trade_id"]
    if not isinstance(trade_id, str) or not trade_id.isdigit():
        raise TerminalSourceFailure("trade_id must be a decimal BTC ID string")
    trade_seq = _integer(row["trade_seq"], "trade_seq")
    instrument = row["instrument_name"]
    if not isinstance(instrument, str):
        raise TerminalSourceFailure("instrument_name must be a string")
    timestamp = _integer(row["timestamp"], "timestamp", positive=True)
    if not start_ms <= timestamp < end_ms:
        raise TerminalSourceFailure("trade timestamp is outside the request")
    if _expiry_ms(instrument) <= timestamp:
        raise TerminalSourceFailure("option trade is not before expiry")
    direction = row["direction"]
    if direction not in {"buy", "sell"}:
        raise TerminalSourceFailure("direction must be buy or sell")
    price = _decimal(row["price"], "price", positive=True)
    amount = _decimal(row["amount"], "amount", positive=True)

    auxiliary: dict[str, str | None] = {
        field: None for field in AUXILIARY_FIELDS
    }
    if row.get("tick_direction") is not None:
        tick = _integer(row["tick_direction"], "tick_direction")
        if tick not in {0, 1, 2, 3}:
            raise TerminalSourceFailure("tick_direction is outside 0..3")
        auxiliary["tick_direction"] = str(tick)
    for field in ("index_price", "mark_price"):
        if row.get(field) is not None:
            auxiliary[field] = canonical_positive_decimal(
                _decimal(row[field], field, positive=True)
            )
    if row.get("iv") is not None:
        auxiliary["iv"] = canonical_nonnegative_decimal(
            _decimal(row["iv"], "iv", positive=False)
        )
    if row.get("block_trade_leg_count") is not None:
        auxiliary["block_trade_leg_count"] = str(
            _integer(
                row["block_trade_leg_count"],
                "block_trade_leg_count",
                positive=True,
            )
        )
    for field in (
        "block_trade_id",
        "combo_id",
        "combo_trade_id",
        "block_rfq_id",
    ):
        auxiliary[field] = _optional_identifier(row.get(field), field)

    return Trade(
        trade_id=trade_id,
        trade_seq=trade_seq,
        instrument_name=instrument,
        timestamp=timestamp,
        direction=str(direction),
        price=price,
        amount=amount,
        auxiliary=tuple((field, auxiliary[field]) for field in AUXILIARY_FIELDS),
    )


def parse_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_float=Decimal,
            parse_int=int,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalSourceFailure("response is not valid UTF-8 JSON") from exc


def page_request(
    endpoint: str,
    *,
    start_ms: int | None,
    end_ms: int,
    start_id: str | None,
) -> PageRequest:
    if (start_ms is None) == (start_id is None):
        raise LedgerError("exactly one initial or continuation cursor is required")
    params: dict[str, str] = {
        "count": str(PAGE_SIZE),
        "currency": "BTC",
        "end_timestamp": str(end_ms - 1),
        "kind": "option",
        "sorting": "asc",
    }
    if start_ms is not None:
        params["start_timestamp"] = str(start_ms)
    else:
        assert start_id is not None
        params["start_id"] = start_id
    return PageRequest(endpoint=endpoint, params=tuple(sorted(params.items())))


def _validate_http_payload(
    payload: HttpPayload,
    request: PageRequest,
) -> None:
    parsed = urlparse(payload.final_url)
    expected = urlparse(request.endpoint)
    if parsed.scheme != "https" or parsed.hostname != expected.hostname:
        raise TerminalSourceFailure("HTTP response left the frozen Deribit host")
    if parsed.path != expected.path:
        raise TerminalSourceFailure("HTTP response path differs")
    if payload.status != 200:
        raise TerminalSourceFailure(f"HTTP status is {payload.status}")
    if len(payload.raw) > MAXIMUM_PAGE_BYTES:
        raise TerminalSourceFailure("HTTP page exceeds the frozen byte cap")


def parse_page(
    payload: HttpPayload,
    request: PageRequest,
    *,
    start_ms: int,
    end_ms: int,
) -> tuple[list[Trade], bool, str | int | None]:
    _validate_http_payload(payload, request)
    decoded = parse_json(payload.raw)
    if not isinstance(decoded, dict):
        raise TerminalSourceFailure("response envelope is not an object")
    if decoded.get("jsonrpc") != "2.0":
        raise TerminalSourceFailure("response JSON-RPC version differs")
    if decoded.get("error") is not None:
        raise TerminalSourceFailure("response contains an application error")
    response_id = decoded.get("id")
    if (
        response_id is not None
        and (
            isinstance(response_id, bool)
            or not isinstance(response_id, (str, int))
        )
    ):
        raise TerminalSourceFailure("response id has an invalid type")
    result = decoded.get("result")
    if not isinstance(result, dict):
        raise TerminalSourceFailure("response result is not an object")
    rows = result.get("trades")
    has_more = result.get("has_more")
    if not isinstance(rows, list):
        raise TerminalSourceFailure("result.trades is not a list")
    if not isinstance(has_more, bool):
        raise TerminalSourceFailure("result.has_more is not a boolean")
    if len(rows) > PAGE_SIZE:
        raise TerminalSourceFailure("page exceeds the frozen row count")
    if not rows:
        raise TerminalSourceFailure("trade page is empty")
    trades: list[Trade] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TerminalSourceFailure("trade row is not an object")
        trades.append(parse_trade(row, start_ms=start_ms, end_ms=end_ms))
    ids = [int(row.trade_id) for row in trades]
    if any(right <= left for left, right in zip(ids, ids[1:])):
        raise TerminalSourceFailure("trade IDs do not strictly increase in page")
    return trades, has_more, response_id


def _page_commitment(
    request: PageRequest,
    payload: HttpPayload,
    trades: Sequence[Trade],
    has_more: bool,
    response_id: str | int | None,
) -> PageCommitment:
    hard_rows = [trade.hard_row() for trade in trades]
    return PageCommitment(
        request=request,
        raw_sha256=sha256_bytes(payload.raw),
        raw_bytes=len(payload.raw),
        hard_hash=canonical_hash(hard_rows),
        rows=len(trades),
        has_more=has_more,
        first_id=trades[0].trade_id,
        last_id=trades[-1].trade_id,
        response_id=response_id,
        response_status=payload.status,
        response_final_url=payload.final_url,
        response_headers=payload.headers,
    )


def paginate(
    *,
    endpoint: str,
    start_ms: int,
    end_ms: int,
    fetcher: Fetcher,
    sleep: Sleep = time.sleep,
) -> PaginationAudit:
    accepted: list[Trade] = []
    pages: list[PageCommitment] = []
    accepted_by_id: dict[str, Trade] = {}
    previous_last: Trade | None = None
    request = page_request(
        endpoint, start_ms=start_ms, end_ms=end_ms, start_id=None
    )

    for ordinal in range(1, MAXIMUM_PAGES + 1):
        if ordinal > 1:
            sleep(HTTP_PAGE_INTERVAL_SECONDS)
        payload = fetcher(request)
        trades, has_more, response_id = parse_page(
            payload, request, start_ms=start_ms, end_ms=end_ms
        )
        pages.append(
            _page_commitment(
                request, payload, trades, has_more, response_id
            )
        )

        new_rows = trades
        if previous_last is not None:
            overlap = trades[0]
            if overlap.trade_id != previous_last.trade_id:
                raise TerminalSourceFailure(
                    "continuation page lacks the exact boundary ID"
                )
            if overlap.hard_row() != previous_last.hard_row():
                raise TerminalSourceFailure(
                    "continuation boundary hard record differs"
                )
            new_rows = trades[1:]
            if not new_rows:
                raise TerminalSourceFailure(
                    "continuation page does not advance"
                )

        for trade in new_rows:
            if trade.trade_id in accepted_by_id:
                raise TerminalSourceFailure("accepted trade ID repeats")
            accepted_by_id[trade.trade_id] = trade
            accepted.append(trade)

        previous_last = trades[-1]
        if not has_more:
            break
        request = page_request(
            endpoint,
            start_ms=None,
            end_ms=end_ms,
            start_id=previous_last.trade_id,
        )
    else:
        raise TerminalSourceFailure("pagination exceeds the frozen page cap")

    if not accepted:
        raise TerminalSourceFailure("pagination accepted no trades")
    presence = {
        field: sum(
            trade.auxiliary_map()[field] is not None for trade in accepted
        )
        for field in AUXILIARY_FIELDS
    }
    return PaginationAudit(
        accepted=tuple(accepted),
        pages=tuple(pages),
        auxiliary_presence=tuple(sorted(presence.items())),
    )


def replay_pages(
    audit: PaginationAudit,
    *,
    start_ms: int,
    end_ms: int,
    fetcher: Fetcher,
    sleep: Sleep = time.sleep,
) -> tuple[PageCommitment, ...]:
    replayed: list[PageCommitment] = []
    for ordinal, expected in enumerate(audit.pages, start=1):
        if ordinal > 1:
            sleep(HTTP_PAGE_INTERVAL_SECONDS)
        payload = fetcher(expected.request)
        trades, has_more, response_id = parse_page(
            payload,
            expected.request,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        observed = _page_commitment(
            expected.request, payload, trades, has_more, response_id
        )
        if (
            observed.hard_hash != expected.hard_hash
            or observed.has_more != expected.has_more
            or observed.first_id != expected.first_id
            or observed.last_id != expected.last_id
            or observed.response_id != expected.response_id
        ):
            raise TerminalSourceFailure("historical exact-page replay differs")
        replayed.append(observed)
    return tuple(replayed)


def evaluate_hard_parity(
    websocket: Sequence[Trade],
    recent: Sequence[Trade],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    ws_by_id = {trade.trade_id: trade for trade in websocket}
    rest_by_id = {trade.trade_id: trade for trade in recent}
    failures: list[str] = []
    missing_rest = sorted(set(ws_by_id) - set(rest_by_id), key=int)
    missing_ws = sorted(set(rest_by_id) - set(ws_by_id), key=int)
    if missing_rest:
        failures.append("parity:websocket_ids_missing_from_rest")
    if missing_ws:
        failures.append("parity:rest_ids_missing_from_websocket")

    hard_mismatches = 0
    auxiliary_presence_mismatches = {
        field: 0 for field in AUXILIARY_FIELDS
    }
    auxiliary_value_mismatches = {
        field: 0 for field in AUXILIARY_FIELDS
    }
    for identifier in set(ws_by_id) & set(rest_by_id):
        left = ws_by_id[identifier]
        right = rest_by_id[identifier]
        if left.hard_row() != right.hard_row():
            hard_mismatches += 1
        left_aux = left.auxiliary_map()
        right_aux = right.auxiliary_map()
        for field in AUXILIARY_FIELDS:
            if (left_aux[field] is None) != (right_aux[field] is None):
                auxiliary_presence_mismatches[field] += 1
            elif (
                left_aux[field] is not None
                and left_aux[field] != right_aux[field]
            ):
                auxiliary_value_mismatches[field] += 1
    if hard_mismatches:
        failures.append("parity:hard_field_mismatch")

    return tuple(failures), {
        "websocket_ids": len(ws_by_id),
        "rest_ids": len(rest_by_id),
        "common_ids": len(set(ws_by_id) & set(rest_by_id)),
        "websocket_ids_missing_from_rest": len(missing_rest),
        "rest_ids_missing_from_websocket": len(missing_ws),
        "hard_field_mismatches": hard_mismatches,
        "auxiliary_presence_mismatches": auxiliary_presence_mismatches,
        "auxiliary_value_mismatches": auxiliary_value_mismatches,
    }


def validate_instrument_sequences(trades: Sequence[Trade], source: str) -> None:
    grouped: dict[str, list[Trade]] = {}
    for trade in trades:
        grouped.setdefault(trade.instrument_name, []).append(trade)
    for instrument, rows in grouped.items():
        ordered = sorted(rows, key=lambda trade: trade.trade_seq)
        sequences = [trade.trade_seq for trade in ordered]
        if len(sequences) != len(set(sequences)):
            raise TerminalSourceFailure(
                f"{source} repeats trade_seq for {instrument}"
            )
        timestamps = [trade.timestamp for trade in ordered]
        if any(
            right < left for left, right in zip(timestamps, timestamps[1:])
        ):
            raise TerminalSourceFailure(
                f"{source} trade_seq time reverses for {instrument}"
            )


def websocket_message(
    raw: bytes,
    *,
    start_ms: int,
    end_ms: int,
) -> tuple[str, list[Trade]]:
    decoded = parse_json(raw)
    if not isinstance(decoded, dict):
        raise TerminalSourceFailure("WebSocket message is not an object")
    if decoded.get("id") == 1:
        if decoded.get("result") != [WS_CHANNEL]:
            raise TerminalSourceFailure("subscription acknowledgement differs")
        return "subscribe", []
    if decoded.get("id") == 2:
        if decoded.get("result") != [WS_CHANNEL]:
            raise TerminalSourceFailure("unsubscribe acknowledgement differs")
        return "unsubscribe", []
    if decoded.get("method") != "subscription":
        if decoded.get("error") is not None:
            raise TerminalSourceFailure("WebSocket returned an error")
        raise TerminalSourceFailure("unexpected WebSocket control message")
    params = decoded.get("params")
    if not isinstance(params, dict) or params.get("channel") != WS_CHANNEL:
        raise TerminalSourceFailure("WebSocket subscription channel differs")
    rows = params.get("data")
    if not isinstance(rows, list):
        raise TerminalSourceFailure("WebSocket trade data is not a list")
    trades: list[Trade] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TerminalSourceFailure("WebSocket trade row is not an object")
        timestamp = row.get("timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise TerminalSourceFailure("WebSocket timestamp must be integer")
        if start_ms <= timestamp < end_ms:
            trades.append(
                parse_trade(row, start_ms=start_ms, end_ms=end_ms)
            )
        else:
            # Validate the row under an unbounded positive interval before
            # discarding it, so schema drift outside the retained window fails.
            parse_trade(row, start_ms=1, end_ms=10**16)
    return "trades", trades


def audit_websocket_messages(
    messages: Iterable[bytes],
    *,
    start_ms: int = LIVE_START_MS,
    end_ms: int = LIVE_END_MS,
) -> WebSocketAudit:
    retained: list[Trade] = []
    by_id: dict[str, Trade] = {}
    raw_messages: list[bytes] = []
    subscription_ack = False
    unsubscribe_ack = False
    bytes_read = 0
    message_count = 0
    for raw in messages:
        if len(raw) > MAXIMUM_WS_MESSAGE_BYTES:
            raise TerminalSourceFailure("WebSocket message exceeds byte cap")
        bytes_read += len(raw)
        if bytes_read > MAXIMUM_WS_CAPTURE_BYTES:
            raise TerminalSourceFailure("WebSocket capture exceeds byte cap")
        raw_messages.append(raw)
        message_count += 1
        kind, trades = websocket_message(
            raw, start_ms=start_ms, end_ms=end_ms
        )
        if kind == "subscribe":
            if subscription_ack:
                raise TerminalSourceFailure("duplicate subscription ack")
            subscription_ack = True
        elif kind == "unsubscribe":
            if unsubscribe_ack:
                raise TerminalSourceFailure("duplicate unsubscribe ack")
            unsubscribe_ack = True
        else:
            if not subscription_ack:
                raise TerminalSourceFailure("trade arrived before subscribe ack")
            if unsubscribe_ack:
                raise TerminalSourceFailure("trade arrived after unsubscribe ack")
            for trade in trades:
                if trade.trade_id in by_id:
                    raise TerminalSourceFailure("WebSocket repeats trade ID")
                by_id[trade.trade_id] = trade
                retained.append(trade)
    if not subscription_ack or not unsubscribe_ack:
        raise TerminalSourceFailure("WebSocket acknowledgement is incomplete")
    validate_instrument_sequences(retained, "websocket")
    return WebSocketAudit(
        trades=tuple(retained),
        raw_messages=tuple(raw_messages),
        subscription_ack=subscription_ack,
        unsubscribe_ack=unsubscribe_ack,
        messages=message_count,
        bytes_read=bytes_read,
    )


def _monotonic_raw_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def validate_clock_samples(
    before: ClockSample,
    after: ClockSample,
) -> None:
    for sample in (before, after):
        round_trip = (
            sample.response_end_monotonic_ns
            - sample.request_start_monotonic_ns
        )
        if round_trip <= 0 or round_trip > 2_000_000_000:
            raise TerminalSourceFailure("server-time round trip is invalid")
        lower = sample.request_start_utc_ns // 1_000_000 - 1_000
        upper = sample.response_end_utc_ns // 1_000_000 + 1_000
        if not lower <= sample.server_ms <= upper:
            raise TerminalSourceFailure("server time is outside host bracket")
    if after.request_start_utc_ns < before.response_end_utc_ns:
        raise TerminalSourceFailure("host UTC reverses between clock samples")
    if (
        after.request_start_monotonic_ns
        <= before.response_end_monotonic_ns
    ):
        raise TerminalSourceFailure("monotonic clock does not advance")
    if after.server_ms < before.server_ms:
        raise TerminalSourceFailure("Deribit server time reverses")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def assert_protocol_committed() -> str:
    if sha256_file(BOUNDARY_PATH) != BOUNDARY_SHA256:
        raise LedgerError("boundary hash differs")
    for path in (BOUNDARY_PATH, SCRIPT_PATH, TEST_PATH):
        result = _git("ls-files", "--error-unmatch", str(path))
        if result.returncode != 0:
            raise LedgerError(f"{path} is not committed")
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        if _git(*args).returncode != 0:
            raise LedgerError("repository is not HEAD-clean")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    if untracked.returncode != 0 or untracked.stdout.strip():
        raise LedgerError("repository has untracked files")
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0 or not head.stdout.strip():
        raise LedgerError("cannot resolve verifier commit")
    return head.stdout.strip()


def used_gib(path: Path = REPOSITORY_ROOT) -> int:
    usage = shutil.disk_usage(path)
    return usage.used // (1024**3)


def assert_disk_guard() -> int:
    usage = shutil.disk_usage(REPOSITORY_ROOT)
    used_bytes = usage.used
    used = used_bytes // (1024**3)
    if used_bytes + DISK_HEADROOM_BYTES >= DISK_LIMIT_GIB * 1024**3:
        raise LedgerError(
            "filesystem lacks the frozen one-GiB headroom below "
            f"{DISK_LIMIT_GIB} GiB"
        )
    return used


def _default_fetch(request: PageRequest) -> HttpPayload:
    opener = urllib.request.build_opener(_NoRedirectHandler())
    retry_delays = (0.0, *RETRY_DELAYS_SECONDS)
    last_error: BaseException | None = None
    for attempt, delay in enumerate(retry_delays):
        if delay:
            time.sleep(delay)
        req = urllib.request.Request(
            request.url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        try:
            response = opener.open(req, timeout=HTTP_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == len(retry_delays) - 1:
                raise TerminalSourceFailure(
                    f"HTTP request failed with status {exc.code}"
                ) from exc
            exc.close()
            continue
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == len(retry_delays) - 1:
                raise TerminalSourceFailure(
                    "HTTP connection failed before response"
                ) from exc
            continue

        with response:
            try:
                raw = response.read(MAXIMUM_PAGE_BYTES + 1)
            except BaseException as exc:
                raise TerminalSourceFailure(
                    "HTTP response body failed after transfer began"
                ) from exc
            if len(raw) > MAXIMUM_PAGE_BYTES:
                raise TerminalSourceFailure(
                    "HTTP response exceeds page byte cap"
                )
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    expected_bytes = int(content_length)
                except ValueError as exc:
                    raise TerminalSourceFailure(
                        "HTTP Content-Length is invalid"
                    ) from exc
                if expected_bytes != len(raw):
                    raise TerminalSourceFailure(
                        "HTTP response body is truncated"
                    )
            return HttpPayload(
                final_url=response.geturl(),
                status=int(response.status),
                headers=tuple(sorted(response.headers.items())),
                raw=raw,
            )
    raise AssertionError(f"unreachable HTTP retry state: {last_error!r}")


def _fetch_time_sample() -> ClockSample:
    start_utc = time.time_ns()
    start_monotonic = _monotonic_raw_ns()
    request = PageRequest(endpoint=TIME_ENDPOINT, params=())
    payload = _default_fetch(request)
    end_monotonic = _monotonic_raw_ns()
    end_utc = time.time_ns()
    _validate_http_payload(payload, request)
    decoded = parse_json(payload.raw)
    if not isinstance(decoded, dict) or decoded.get("jsonrpc") != "2.0":
        raise TerminalSourceFailure("server-time envelope differs")
    server_ms = _integer(decoded.get("result"), "server time", positive=True)
    return ClockSample(
        request_start_utc_ns=start_utc,
        response_end_utc_ns=end_utc,
        request_start_monotonic_ns=start_monotonic,
        response_end_monotonic_ns=end_monotonic,
        server_ms=server_ms,
    )


async def _sleep_until_ms(target_ms: int) -> None:
    while True:
        remaining = target_ms / 1_000 - time.time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 1.0))


async def capture_live_websocket() -> tuple[WebSocketAudit, ClockSample, ClockSample]:
    from websockets.asyncio.client import connect

    if int(time.time() * 1_000) >= LIVE_START_MS:
        raise TerminalSourceFailure("verifier started after the frozen live start")
    await _sleep_until_ms(LIVE_START_MS - 30_000)
    before = await asyncio.to_thread(_fetch_time_sample)
    ssl_context = ssl.create_default_context()
    messages: list[bytes] = []
    async with connect(
        WS_ENDPOINT,
        ssl=ssl_context,
        max_size=MAXIMUM_WS_MESSAGE_BYTES,
        open_timeout=10,
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "public/subscribe",
                    "params": {"channels": [WS_CHANNEL]},
                },
                separators=(",", ":"),
            )
        )
        subscribed = False
        while not subscribed:
            timeout = max(0.001, (LIVE_START_MS - time.time() * 1_000) / 1_000)
            if timeout <= 0:
                raise TerminalSourceFailure(
                    "subscription ack missed the frozen start"
                )
            raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            raw = (
                raw_message.encode("utf-8")
                if isinstance(raw_message, str)
                else bytes(raw_message)
            )
            messages.append(raw)
            kind, _ = websocket_message(
                raw, start_ms=LIVE_START_MS, end_ms=LIVE_END_MS
            )
            subscribed = kind == "subscribe"

        while time.time() * 1_000 < LIVE_DRAIN_END_MS:
            timeout = max(
                0.001,
                min(
                    1.0,
                    (LIVE_DRAIN_END_MS - time.time() * 1_000) / 1_000,
                ),
            )
            try:
                raw_message = await asyncio.wait_for(
                    websocket.recv(), timeout=timeout
                )
            except asyncio.TimeoutError:
                continue
            raw = (
                raw_message.encode("utf-8")
                if isinstance(raw_message, str)
                else bytes(raw_message)
            )
            messages.append(raw)

        await websocket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "public/unsubscribe",
                    "params": {"channels": [WS_CHANNEL]},
                },
                separators=(",", ":"),
            )
        )
        unsubscribed = False
        while not unsubscribed:
            timeout = (
                LIVE_UNSUBSCRIBE_DEADLINE_MS - time.time() * 1_000
            ) / 1_000
            if timeout <= 0:
                raise TerminalSourceFailure(
                    "unsubscribe ack missed its frozen deadline"
                )
            raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            raw = (
                raw_message.encode("utf-8")
                if isinstance(raw_message, str)
                else bytes(raw_message)
            )
            messages.append(raw)
            kind, _ = websocket_message(
                raw, start_ms=LIVE_START_MS, end_ms=LIVE_END_MS
            )
            unsubscribed = kind == "unsubscribe"

    after = await asyncio.to_thread(_fetch_time_sample)
    validate_clock_samples(before, after)
    audit = audit_websocket_messages(messages)
    return audit, before, after


def deterministic_trades_gzip(trades: Sequence[Trade]) -> bytes:
    rows = [trade.hard_row() for trade in sorted(trades, key=lambda row: int(row.trade_id))]
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        for row in rows:
            zipped.write(canonical_json_bytes(row))
    return raw.getvalue()


def deterministic_messages_gzip(messages: Sequence[bytes]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        for ordinal, message in enumerate(messages, start=1):
            zipped.write(
                canonical_json_bytes(
                    {
                        "ordinal": ordinal,
                        "raw_sha256": sha256_bytes(message),
                        "raw_utf8": message.decode("utf-8"),
                    }
                )
            )
    return raw.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    absolute = repository_path(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{absolute.name}.",
        dir=absolute.parent,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, absolute)
        directory_fd = os.open(absolute.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def reserve_one_shot(
    *,
    sentinel_path: Path,
    report_path: Path,
    data_dir: Path,
    commit: str,
) -> str:
    for path, label in (
        (report_path, "report"),
        (data_dir, "data directory"),
    ):
        if repository_path(path).exists():
            raise LedgerError(f"one-shot {label} already exists")
    absolute = repository_path(sentinel_path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(
        {
            "protocol_version": PROTOCOL_VERSION,
            "boundary_sha256": BOUNDARY_SHA256,
            "verifier_commit": commit,
            "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    try:
        descriptor = os.open(
            absolute,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise LedgerError("one-shot sentinel already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(absolute.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Preserve a created sentinel on any failure. The protocol is one-shot,
        # so a partial reservation must fail closed rather than authorize retry.
        raise
    return sha256_bytes(payload)


def _report_base(commit: str, used_before: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "boundary_path": str(BOUNDARY_PATH),
            "boundary_sha256": BOUNDARY_SHA256,
            "script_path": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "test_path": str(TEST_PATH),
            "test_sha256": sha256_file(TEST_PATH),
            "verifier_commit": commit,
        },
        "disk": {
            "limit_gib": DISK_LIMIT_GIB,
            "used_gib_before": used_before,
            "raw_historical_responses_persisted": False,
        },
        "outcome_boundary": {
            "candidate_incidence_opened": False,
            "comparators_opened": False,
            "btc_market_opened": False,
            "funding_opened": False,
            "returns_or_pnl_opened": False,
            "model_or_reward_opened": False,
            "cagr_or_mdd_opened": False,
        },
    }


async def run(
    *,
    report_path: Path = DEFAULT_REPORT,
    data_dir: Path = DEFAULT_DATA_DIR,
    sentinel_path: Path = DEFAULT_SENTINEL,
    fetcher: Fetcher = _default_fetch,
    live_capture: Callable[
        [], Awaitable[tuple[WebSocketAudit, ClockSample, ClockSample]]
    ] = capture_live_websocket,
    sleep: Sleep = time.sleep,
) -> dict[str, Any]:
    commit = assert_protocol_committed()
    used_before = assert_disk_guard()
    sentinel_sha256 = reserve_one_shot(
        sentinel_path=sentinel_path,
        report_path=report_path,
        data_dir=data_dir,
        commit=commit,
    )
    report = _report_base(commit, used_before)
    report["bindings"]["run_sentinel_path"] = str(sentinel_path)
    report["bindings"]["run_sentinel_sha256"] = sentinel_sha256
    source_started = False
    try:
        source_started = True
        history = paginate(
            endpoint=HISTORY_ENDPOINT,
            start_ms=HISTORY_START_MS,
            end_ms=HISTORY_END_MS,
            fetcher=fetcher,
            sleep=sleep,
        )
        history_replay = replay_pages(
            history,
            start_ms=HISTORY_START_MS,
            end_ms=HISTORY_END_MS,
            fetcher=fetcher,
            sleep=sleep,
        )
        validate_instrument_sequences(history.accepted, "historical")
        history_instruments = {
            trade.instrument_name for trade in history.accepted
        }
        if len(history.accepted) < MINIMUM_HISTORY_TRADES:
            raise TerminalSourceFailure("historical operational trade floor fails")
        if len(history_instruments) < MINIMUM_HISTORY_INSTRUMENTS:
            raise TerminalSourceFailure(
                "historical operational instrument floor fails"
            )

        websocket, clock_before, clock_after = await live_capture()
        if len(websocket.trades) < MINIMUM_LIVE_TRADES:
            raise TerminalSourceFailure("live operational trade floor fails")
        live_instruments = {
            trade.instrument_name for trade in websocket.trades
        }
        if len(live_instruments) < MINIMUM_LIVE_INSTRUMENTS:
            raise TerminalSourceFailure(
                "live operational instrument floor fails"
            )
        recent = paginate(
            endpoint=RECENT_ENDPOINT,
            start_ms=LIVE_START_MS,
            end_ms=LIVE_END_MS,
            fetcher=fetcher,
            sleep=sleep,
        )
        validate_instrument_sequences(recent.accepted, "recent_rest")
        failures, parity = evaluate_hard_parity(
            websocket.trades, recent.accepted
        )
        if failures:
            raise TerminalSourceFailure(",".join(failures))

        history_payload = deterministic_trades_gzip(history.accepted)
        websocket_payload = deterministic_trades_gzip(websocket.trades)
        recent_payload = deterministic_trades_gzip(recent.accepted)
        raw_ws_payload = deterministic_messages_gzip(websocket.raw_messages)
        outputs = {
            "history_hard_ndjson_gz": data_dir / "history_hard.ndjson.gz",
            "websocket_hard_ndjson_gz": data_dir / "websocket_hard.ndjson.gz",
            "recent_rest_hard_ndjson_gz": (
                data_dir / "recent_rest_hard.ndjson.gz"
            ),
            "websocket_raw_ndjson_gz": (
                data_dir / "websocket_raw.ndjson.gz"
            ),
        }
        for name, payload in (
            ("history_hard_ndjson_gz", history_payload),
            ("websocket_hard_ndjson_gz", websocket_payload),
            ("recent_rest_hard_ndjson_gz", recent_payload),
            ("websocket_raw_ndjson_gz", raw_ws_payload),
        ):
            _atomic_write(outputs[name], payload)

        report.update(
            {
                "decision": "SOURCE_PARITY_PASS",
                "failures": [],
                "history": {
                    "start_ms": HISTORY_START_MS,
                    "end_ms": HISTORY_END_MS,
                    "trades": len(history.accepted),
                    "instruments": len(history_instruments),
                    "pages": [page.as_dict() for page in history.pages],
                    "replay_pages": [
                        page.as_dict() for page in history_replay
                    ],
                    "auxiliary_presence": dict(
                        history.auxiliary_presence
                    ),
                },
                "live": {
                    "start_ms": LIVE_START_MS,
                    "end_ms": LIVE_END_MS,
                    "trades": len(websocket.trades),
                    "instruments": len(live_instruments),
                    "messages": websocket.messages,
                    "raw_bytes": websocket.bytes_read,
                    "clock_before": clock_before.as_dict(),
                    "clock_after": clock_after.as_dict(),
                    "recent_pages": [
                        page.as_dict() for page in recent.pages
                    ],
                    "recent_auxiliary_presence": dict(
                        recent.auxiliary_presence
                    ),
                    "parity": parity,
                },
                "outputs": {
                    name: {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "bytes": repository_path(path).stat().st_size,
                    }
                    for name, path in outputs.items()
                },
            }
        )
    except BaseException as exc:
        report.update(
            {
                "decision": "SOURCE_PARITY_REJECT",
                "failures": [
                    f"source:{type(exc).__name__}:{exc}"
                    if source_started
                    else f"preflight:{type(exc).__name__}:{exc}"
                ],
            }
        )
        report_without_manifest = dict(report)
        report["manifest_hash_without_self"] = canonical_hash(
            report_without_manifest
        )
        _atomic_write(report_path, canonical_json_bytes(report))
        raise

    report_without_manifest = dict(report)
    report["manifest_hash_without_self"] = canonical_hash(
        report_without_manifest
    )
    _atomic_write(report_path, canonical_json_bytes(report))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = asyncio.run(
            run(report_path=args.report, data_dir=args.data_dir)
        )
    except BaseException as exc:
        print(f"DOTL source parity rejected: {exc}")
        return 1
    print(
        "DOTL source parity passed: "
        f"history={report['history']['trades']} "
        f"live={report['live']['trades']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
