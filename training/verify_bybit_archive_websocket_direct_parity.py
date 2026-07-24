"""Verify exact Bybit archive-to-WebSocket source parity without opening outcomes."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import http.client
import io
import json
import os
import ssl
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
from urllib.parse import urlparse


PROTOCOL_VERSION = "bybit_archive_websocket_direct_parity_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

BOUNDARY = Path(
    "docs/bybit-archive-websocket-direct-parity-source-boundary-2026-07-24.md"
)
BOUNDARY_SHA256 = (
    "b4a7f20d5f21cc79e1a0fbe298bb2189f8bdc4027d0f7030841d3f95cc7e93d3"
)
SOURCE_DECISION = Path(
    "docs/bybit-public-trade-sequence-source-axis-decision-2026-07-23.md"
)
SOURCE_DECISION_SHA256 = (
    "fb12c54b8a4a89cb446baa9014f89546bf6c99e46687be2471b51a2bf1989a21"
)
SOURCE_AUDIT = Path("docs/bybit-public-trade-sequence-source-audit-2026-07-23.md")
SOURCE_AUDIT_SHA256 = (
    "fe324cccfb0c3f66963c142b9a6c0237489420313750de873622cadb10e8c112"
)
BSEA_CONTRACT = Path(
    "docs/bybit-public-trade-live-parity-capture-contract-2026-07-23.md"
)
BSEA_CONTRACT_SHA256 = (
    "50cca9c3e103e8978bb260c65b103dd90361615f6e69443181350ae560622b6c"
)
BSEA_REJECTION = Path(
    "docs/bybit-public-trade-live-parity-v3-rejection-2026-07-23.md"
)
BSEA_REJECTION_SHA256 = (
    "f894abd2fc55c02a75e4e82c076b3ffc67ced0e63f1f774e1ce6be0671ed0ed6"
)
BSEA_REJECTION_RESULT = Path(
    "results/bybit_public_trade_live_parity_capture_v3_reject_2026-07-23.json"
)
BSEA_REJECTION_RESULT_SHA256 = (
    "493cde97193c3c837cda4ca2101c7d2068cab8972836fdb511a68b2b7b9fc5d5"
)

CAPTURE_ROOT = Path(
    "data/bybit_public_trade_parity_capture_2026-07-23T07-10-40-429968Z"
)
CAPTURE_MANIFEST = CAPTURE_ROOT / "manifest.json"
CAPTURE_MANIFEST_SHA256 = (
    "38027f767122c9f7d5a57ae5a5a0f1445525e63bb05fcd75981de42677f68e16"
)
WEBSOCKET_CAPTURE = CAPTURE_ROOT / "websocket_messages.ndjson.gz"
WEBSOCKET_CAPTURE_SHA256 = (
    "b63ec8ee4f8f2c260631e09ce02d8dec71b8f17827a8e3592d792e4b2f5b6937"
)

SCRIPT_PATH = Path("training/verify_bybit_archive_websocket_direct_parity.py")
TEST_PATH = Path("tests/test_verify_bybit_archive_websocket_direct_parity.py")

ARCHIVE_DAY = "2026-07-23"
ARCHIVE_URL = (
    "https://public.bybit.com/trading/BTCUSDT/"
    "BTCUSDT2026-07-23.csv.gz"
)
ARCHIVE_CONTENT_LENGTH = 51_913_751
ARCHIVE_ETAG = '"b4cd5a78805f5456092fe04e83913178-7"'
ARCHIVE_HEADER = (
    "timestamp",
    "symbol",
    "side",
    "size",
    "price",
    "tickDirection",
    "trdMatchID",
    "grossValue",
    "homeNotional",
    "foreignNotional",
    "RPI",
)
ARCHIVE_DAY_START_MS = int(
    datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp()
) * 1_000
ARCHIVE_DAY_END_MS = int(
    datetime(2026, 7, 24, tzinfo=timezone.utc).timestamp()
) * 1_000

EDGE_EXCLUSION_MS = 5_000
MINIMUM_INTERVAL_MS = 300_000
MINIMUM_INTERVAL_IDS = 1_000
DISK_LIMIT_GIB = 300
HTTP_TIMEOUT_SECONDS = 120
USER_AGENT = "rllm-bybit-direct-parity/1.0"

DEFAULT_REPORT = Path(
    "results/bybit_archive_websocket_direct_parity_2026-07-24.json"
)
DEFAULT_INTERVAL = Path(
    "data/bybit_archive_websocket_direct_parity_interval_2026-07-23.csv.gz"
)
INTERVAL_COLUMNS = (
    "time_ms",
    "trade_id",
    "symbol",
    "side",
    "price",
    "size",
)


class DirectParityError(RuntimeError):
    """Raised when the frozen source protocol cannot be evaluated."""


class TerminalSourceFailure(DirectParityError):
    """Raised for a no-repair failure after the source request begins."""


@dataclass(frozen=True, slots=True)
class Trade:
    source_id: str
    symbol: str
    side: str
    price: Decimal
    size: Decimal
    time_ms: int

    def canonical_row(self) -> dict[str, str]:
        return {
            "time_ms": str(self.time_ms),
            "trade_id": self.source_id,
            "symbol": self.symbol,
            "side": self.side,
            "price": canonical_decimal(self.price),
            "size": canonical_decimal(self.size),
        }


@dataclass(frozen=True, slots=True)
class WebSocketAudit:
    records: tuple[Trade, ...]
    total_frames: int
    trade_frames: int
    subscription_acks: int
    pongs: int
    first_ws_ms: int
    last_ws_ms: int
    start_ms: int
    end_ms: int
    interval_records: tuple[Trade, ...]


@dataclass(frozen=True, slots=True)
class ArchiveMetadata:
    status: int
    final_url: str
    content_type: str
    content_length: int
    etag: str
    last_modified: str
    response_date: str


@dataclass(frozen=True, slots=True)
class ArchiveAudit:
    records: tuple[Trade, ...]
    total_rows: int
    interval_rows: int
    compressed_bytes: int
    compressed_sha256: str
    metadata: ArchiveMetadata


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


class HashingRawReader(io.RawIOBase):
    """Read-only raw wrapper that hashes every consumed compressed byte."""

    def __init__(self, source: BinaryIO) -> None:
        super().__init__()
        self._source = source
        self._digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        raw = self._source.read(len(buffer))
        if not raw:
            return 0
        length = len(raw)
        buffer[:length] = raw
        self._digest.update(raw)
        self.bytes_read += length
        return length

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def repository_path(path: Path) -> Path:
    return REPOSITORY_ROOT / path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
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


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload).rstrip(b"\n"))


def canonical_decimal(value: Decimal) -> str:
    if not value.is_finite() or value <= 0:
        raise DirectParityError("canonical decimal must be finite and positive")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _positive_decimal(value: Any, field: str) -> Decimal:
    if not isinstance(value, (str, int)):
        raise TerminalSourceFailure(f"{field} must be a decimal literal")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise TerminalSourceFailure(f"{field} is not a decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise TerminalSourceFailure(f"{field} must be finite and positive")
    return parsed


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise TerminalSourceFailure(f"{field} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isdigit():
        parsed = int(value)
    else:
        raise TerminalSourceFailure(f"{field} must be an integer")
    if parsed < 0:
        raise TerminalSourceFailure(f"{field} must be nonnegative")
    return parsed


def _source_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TerminalSourceFailure(f"{field} must be a canonical nonempty string")
    if len(value) > 256:
        raise TerminalSourceFailure(f"{field} is too long")
    return value


def _side(value: Any, field: str) -> str:
    if value not in {"Buy", "Sell"}:
        raise TerminalSourceFailure(f"{field} must be Buy or Sell")
    return str(value)


def _archive_time_ms(value: Any) -> int:
    timestamp = _positive_decimal(value, "archive timestamp")
    milliseconds = timestamp * Decimal(1_000)
    integral = milliseconds.to_integral_value()
    if milliseconds != integral:
        raise TerminalSourceFailure(
            "archive timestamp does not map to an exact millisecond"
        )
    parsed = int(integral)
    if not ARCHIVE_DAY_START_MS <= parsed < ARCHIVE_DAY_END_MS:
        raise TerminalSourceFailure("archive timestamp is outside the frozen day")
    return parsed


def _trade(
    *,
    source_id: Any,
    symbol: Any,
    side: Any,
    price: Any,
    size: Any,
    time_ms: Any,
    prefix: str,
) -> Trade:
    if symbol != "BTCUSDT":
        raise TerminalSourceFailure(f"{prefix} symbol differs")
    return Trade(
        source_id=_source_id(source_id, f"{prefix} source id"),
        symbol="BTCUSDT",
        side=_side(side, f"{prefix} side"),
        price=_positive_decimal(price, f"{prefix} price"),
        size=_positive_decimal(size, f"{prefix} size"),
        time_ms=_nonnegative_integer(time_ms, f"{prefix} time"),
    )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def assert_protocol_committed() -> str:
    required = (BOUNDARY, SCRIPT_PATH, TEST_PATH)
    tracked = _git("ls-files", "--error-unmatch", "--", *required)
    if tracked.returncode != 0:
        raise DirectParityError("BAWDP protocol is not committed")
    if _git("diff", "--quiet", "HEAD", "--", *required).returncode:
        raise DirectParityError("BAWDP protocol differs from HEAD")
    if _git("diff", "--cached", "--quiet").returncode:
        raise DirectParityError("BAWDP index is not clean")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0 or status.stdout.strip():
        raise DirectParityError("BAWDP worktree is not HEAD-clean")
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0 or not head.stdout.strip():
        raise DirectParityError("BAWDP HEAD is unavailable")
    return head.stdout.strip()


def validate_frozen_bindings() -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        BOUNDARY: BOUNDARY_SHA256,
        SOURCE_DECISION: SOURCE_DECISION_SHA256,
        SOURCE_AUDIT: SOURCE_AUDIT_SHA256,
        BSEA_CONTRACT: BSEA_CONTRACT_SHA256,
        BSEA_REJECTION: BSEA_REJECTION_SHA256,
        BSEA_REJECTION_RESULT: BSEA_REJECTION_RESULT_SHA256,
        CAPTURE_MANIFEST: CAPTURE_MANIFEST_SHA256,
        WEBSOCKET_CAPTURE: WEBSOCKET_CAPTURE_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise DirectParityError(f"BAWDP frozen binding changed: {path}")

    capture = json.loads(repository_path(CAPTURE_MANIFEST).read_text("utf-8"))
    if not isinstance(capture, dict):
        raise DirectParityError("BAWDP capture manifest is not an object")
    capture_core = {
        key: value
        for key, value in capture.items()
        if key != "manifest_hash_without_self"
    }
    if capture.get("manifest_hash_without_self") != canonical_hash(capture_core):
        raise DirectParityError("BAWDP capture manifest self-hash differs")
    if (
        capture.get("protocol_version")
        != "bybit_public_trade_live_parity_capture_v3"
        or capture.get("decision") != "REJECT_NO_REPAIR"
        or capture.get("capture_day_utc") != ARCHIVE_DAY
    ):
        raise DirectParityError("BAWDP capture identity differs")
    transport = capture.get("transport")
    if not isinstance(transport, dict) or (
        transport.get("websocket_sessions") != 1
        or transport.get("reconnects") != 0
        or transport.get("websocket_topic") != "publicTrade.BTCUSDT"
        or transport.get("websocket_url")
        != "wss://stream.bybit.com/v5/public/linear"
    ):
        raise DirectParityError("BAWDP WebSocket transport differs")
    provider = capture.get("clock_provider")
    if not isinstance(provider, dict) or (
        provider.get("closed_cleanly") is not True
        or provider.get("fallback_used") is not False
        or provider.get("monotonic_source") != "CLOCK_MONOTONIC_RAW"
    ):
        raise DirectParityError("BAWDP capture clock contract differs")
    outcome = capture.get("outcome_boundary")
    if not isinstance(outcome, dict) or any(outcome.values()):
        raise DirectParityError("BAWDP capture opened a forbidden outcome")
    raw_artifacts = capture.get("raw_artifacts")
    websocket = (
        raw_artifacts.get("websocket")
        if isinstance(raw_artifacts, dict)
        else None
    )
    if not isinstance(websocket, dict) or (
        websocket.get("path") != str(WEBSOCKET_CAPTURE)
        or websocket.get("sha256") != WEBSOCKET_CAPTURE_SHA256
    ):
        raise DirectParityError("BAWDP WebSocket artifact binding differs")

    rejection = json.loads(
        repository_path(BSEA_REJECTION_RESULT).read_text("utf-8")
    )
    if not isinstance(rejection, dict) or (
        rejection.get("decision")
        != "SOURCE_PARITY_REJECT_REST_WINDOW_OVERFLOW"
    ):
        raise DirectParityError("BAWDP BSEA rejection identity differs")
    rejection_outcome = rejection.get("outcome_boundary")
    if not isinstance(rejection_outcome, dict) or any(
        rejection_outcome.values()
    ):
        raise DirectParityError("BAWDP BSEA rejection opened outcomes")
    return capture, rejection


def _decode_raw_frame(record: Mapping[str, Any], ordinal: int) -> bytes:
    if record.get("ordinal") != ordinal or record.get("frame_type") != "text":
        raise TerminalSourceFailure("WebSocket capture ledger differs")
    encoded = record.get("raw_frame_base64")
    digest = record.get("raw_frame_sha256")
    if not isinstance(encoded, str) or not isinstance(digest, str):
        raise TerminalSourceFailure("WebSocket capture frame binding is missing")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise TerminalSourceFailure("WebSocket capture frame is not base64") from exc
    if sha256_bytes(raw) != digest:
        raise TerminalSourceFailure("WebSocket capture frame hash differs")
    return raw


def _parse_ws_frame(raw: bytes) -> tuple[str, tuple[Trade, ...]]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalSourceFailure("WebSocket frame is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TerminalSourceFailure("WebSocket frame is not an object")
    if payload.get("op") == "subscribe":
        if payload.get("success") is not True:
            raise TerminalSourceFailure("WebSocket subscription failed")
        return "subscribe", ()
    if payload.get("op") == "pong" or payload.get("ret_msg") == "pong":
        return "pong", ()
    if (
        payload.get("topic") != "publicTrade.BTCUSDT"
        or payload.get("type") != "snapshot"
    ):
        raise TerminalSourceFailure("unexpected WebSocket frame type")
    _nonnegative_integer(payload.get("ts"), "WebSocket message ts")
    rows = payload.get("data")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1_024:
        raise TerminalSourceFailure("WebSocket trade batch size differs")
    parsed: list[Trade] = []
    prior_time: int | None = None
    for row in rows:
        if not isinstance(row, dict):
            raise TerminalSourceFailure("WebSocket trade is not an object")
        trade = _trade(
            source_id=row.get("i"),
            symbol=row.get("s"),
            side=row.get("S"),
            price=row.get("p"),
            size=row.get("v"),
            time_ms=row.get("T"),
            prefix="WebSocket",
        )
        if prior_time is not None and trade.time_ms < prior_time:
            raise TerminalSourceFailure(
                "WebSocket within-message time order differs"
            )
        prior_time = trade.time_ms
        parsed.append(trade)
    return "trade", tuple(parsed)


def load_websocket_capture(path: Path = WEBSOCKET_CAPTURE) -> WebSocketAudit:
    records: list[Trade] = []
    total_frames = 0
    trade_frames = 0
    subscription_acks = 0
    pongs = 0
    previous_global_time: int | None = None
    with gzip.open(repository_path(path), "rt", encoding="utf-8", newline="") as handle:
        for ordinal, line in enumerate(handle, start=1):
            total_frames += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TerminalSourceFailure(
                    "WebSocket capture NDJSON is malformed"
                ) from exc
            if not isinstance(record, dict):
                raise TerminalSourceFailure(
                    "WebSocket capture record is not an object"
                )
            raw = _decode_raw_frame(record, ordinal)
            kind, trades = _parse_ws_frame(raw)
            if kind == "subscribe":
                subscription_acks += 1
            elif kind == "pong":
                pongs += 1
            else:
                trade_frames += 1
                for trade in trades:
                    if (
                        previous_global_time is not None
                        and trade.time_ms < previous_global_time
                    ):
                        raise TerminalSourceFailure(
                            "WebSocket global time order differs"
                        )
                    previous_global_time = trade.time_ms
                    records.append(trade)
    if total_frames == 0 or not records:
        raise TerminalSourceFailure("WebSocket capture has no trade records")
    if subscription_acks != 1:
        raise TerminalSourceFailure(
            "WebSocket capture must have one subscription acknowledgement"
        )

    first_ws_ms = min(record.time_ms for record in records)
    last_ws_ms = max(record.time_ms for record in records)
    start_ms = first_ws_ms + EDGE_EXCLUSION_MS
    end_ms = last_ws_ms - EDGE_EXCLUSION_MS
    if end_ms - start_ms < MINIMUM_INTERVAL_MS:
        raise TerminalSourceFailure("BAWDP fixed interval is too short")
    interval = tuple(
        record for record in records if start_ms <= record.time_ms < end_ms
    )
    if len(interval) < MINIMUM_INTERVAL_IDS:
        raise TerminalSourceFailure("BAWDP fixed interval has too few rows")
    ids = [record.source_id for record in interval]
    if len(ids) != len(set(ids)):
        raise TerminalSourceFailure("WebSocket interval contains duplicate IDs")
    return WebSocketAudit(
        records=tuple(records),
        total_frames=total_frames,
        trade_frames=trade_frames,
        subscription_acks=subscription_acks,
        pongs=pongs,
        first_ws_ms=first_ws_ms,
        last_ws_ms=last_ws_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        interval_records=interval,
    )


def _metadata(response: Any) -> ArchiveMetadata:
    final_url = str(response.geturl())
    parsed = urlparse(final_url)
    if (
        int(response.status) != 200
        or final_url != ARCHIVE_URL
        or parsed.scheme != "https"
        or parsed.hostname != "public.bybit.com"
    ):
        raise TerminalSourceFailure("Bybit archive response identity differs")
    content_type = str(response.headers.get("Content-Type") or "")
    content_length_raw = response.headers.get("Content-Length")
    etag = str(response.headers.get("ETag") or "")
    if not content_length_raw or not str(content_length_raw).isdigit():
        raise TerminalSourceFailure("Bybit archive content length is missing")
    content_length = int(content_length_raw)
    if (
        content_type.split(";", 1)[0].strip().lower() != "text/csv"
        or content_length != ARCHIVE_CONTENT_LENGTH
        or etag != ARCHIVE_ETAG
    ):
        raise TerminalSourceFailure("Bybit archive frozen metadata differs")
    return ArchiveMetadata(
        status=200,
        final_url=final_url,
        content_type=content_type,
        content_length=content_length,
        etag=etag,
        last_modified=str(response.headers.get("Last-Modified") or ""),
        response_date=str(response.headers.get("Date") or ""),
    )


def _read_archive(
    source: BinaryIO,
    *,
    metadata: ArchiveMetadata,
    start_ms: int,
    end_ms: int,
) -> ArchiveAudit:
    hashing = HashingRawReader(source)
    buffered = io.BufferedReader(hashing, buffer_size=1 << 20)
    interval: list[Trade] = []
    total_rows = 0
    previous_time: int | None = None
    try:
        with gzip.GzipFile(fileobj=buffered, mode="rb") as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="",
            ) as text:
                reader = csv.DictReader(text)
                if tuple(reader.fieldnames or ()) != ARCHIVE_HEADER:
                    raise TerminalSourceFailure("Bybit archive header differs")
                for row in reader:
                    total_rows += 1
                    if None in row or any(row.get(column) is None for column in ARCHIVE_HEADER):
                        raise TerminalSourceFailure(
                            "Bybit archive row width differs"
                        )
                    time_ms = _archive_time_ms(row["timestamp"])
                    if previous_time is not None and time_ms < previous_time:
                        raise TerminalSourceFailure(
                            "Bybit archive time order differs"
                        )
                    previous_time = time_ms
                    if start_ms <= time_ms < end_ms:
                        interval.append(
                            _trade(
                                source_id=row["trdMatchID"],
                                symbol=row["symbol"],
                                side=row["side"],
                                price=row["price"],
                                size=row["size"],
                                time_ms=time_ms,
                                prefix="archive",
                            )
                        )
    except (
        EOFError,
        OSError,
        UnicodeDecodeError,
        csv.Error,
        http.client.IncompleteRead,
        urllib.error.URLError,
    ) as exc:
        raise TerminalSourceFailure("Bybit archive stream is malformed") from exc
    if hashing.bytes_read != metadata.content_length:
        raise TerminalSourceFailure("Bybit archive compressed length differs")
    ids = [record.source_id for record in interval]
    if len(ids) != len(set(ids)):
        raise TerminalSourceFailure("Bybit archive interval contains duplicate IDs")
    return ArchiveAudit(
        records=tuple(interval),
        total_rows=total_rows,
        interval_rows=len(interval),
        compressed_bytes=hashing.bytes_read,
        compressed_sha256=hashing.sha256,
        metadata=metadata,
    )


def _default_opener() -> Any:
    return urllib.request.build_opener(
        _NoRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )


def fetch_archive(
    *,
    start_ms: int,
    end_ms: int,
    opener: Any | None = None,
) -> ArchiveAudit:
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",
        },
        method="GET",
    )
    transport = opener or _default_opener()
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = transport.open(request, timeout=HTTP_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            raise TerminalSourceFailure(
                "Bybit archive returned an HTTP error"
            ) from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == 0:
                continue
            raise DirectParityError(
                "Bybit archive transport failed before response body"
            ) from exc
        with response:
            metadata = _metadata(response)
            return _read_archive(
                response,
                metadata=metadata,
                start_ms=start_ms,
                end_ms=end_ms,
            )
    raise DirectParityError("Bybit archive transport failed") from last_error


def evaluate_parity(
    websocket_records: Sequence[Trade],
    archive_records: Sequence[Trade],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    failures: list[str] = []
    ws_by_id = {record.source_id: record for record in websocket_records}
    archive_by_id = {record.source_id: record for record in archive_records}
    ws_ids = set(ws_by_id)
    archive_ids = set(archive_by_id)
    missing_from_archive = sorted(ws_ids - archive_ids)
    missing_from_websocket = sorted(archive_ids - ws_ids)
    if missing_from_archive:
        failures.append("parity:websocket_ids_missing_from_archive")
    if missing_from_websocket:
        failures.append("parity:archive_ids_missing_from_websocket")
    mismatches = sorted(
        source_id
        for source_id in ws_ids & archive_ids
        if ws_by_id[source_id] != archive_by_id[source_id]
    )
    if mismatches:
        failures.append("parity:canonical_field_mismatch")
    ws_times = [record.time_ms for record in websocket_records]
    archive_times = [record.time_ms for record in archive_records]
    if any(right < left for left, right in zip(ws_times, ws_times[1:])):
        failures.append("parity:websocket_time_order")
    if any(right < left for left, right in zip(archive_times, archive_times[1:])):
        failures.append("parity:archive_time_order")
    diagnostics = {
        "websocket_ids": len(ws_ids),
        "archive_ids": len(archive_ids),
        "common_ids": len(ws_ids & archive_ids),
        "websocket_ids_missing_from_archive": len(missing_from_archive),
        "archive_ids_missing_from_websocket": len(missing_from_websocket),
        "canonical_field_mismatches": len(mismatches),
        "missing_from_archive_sample_sha256": canonical_hash(
            {"ids": missing_from_archive[:100]}
        ),
        "missing_from_websocket_sample_sha256": canonical_hash(
            {"ids": missing_from_websocket[:100]}
        ),
        "mismatch_sample_sha256": canonical_hash({"ids": mismatches[:100]}),
    }
    return tuple(failures), diagnostics


def deterministic_interval_gzip(records: Sequence[Trade]) -> bytes:
    raw = io.StringIO(newline="")
    writer = csv.DictWriter(
        raw,
        fieldnames=INTERVAL_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for record in sorted(records, key=lambda item: (item.time_ms, item.source_id)):
        writer.writerow(record.canonical_row())
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as zipped:
        zipped.write(raw.getvalue().encode("utf-8"))
    return output.getvalue()


def _disk_used_gib(path: Path = REPOSITORY_ROOT) -> int:
    stats = os.statvfs(path)
    used = (stats.f_blocks - stats.f_bfree) * stats.f_frsize
    return used // (1 << 30)


def build_report(
    *,
    protocol_commit: str,
    capture_manifest: Mapping[str, Any],
    websocket: WebSocketAudit,
    archive: ArchiveAudit,
    failures: Sequence[str],
    diagnostics: Mapping[str, Any],
    interval_path: Path,
    interval_sha256: str | None,
    disk_used_gib: int,
) -> dict[str, Any]:
    passed = not failures
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "transport_id": "BAWDP-v1",
        "decision": (
            "PASS_AUTHORIZES_ORTHOGONAL_CANDIDATE_BOUNDARY"
            if passed
            else "RETIRE_BAWDP_V1_NO_REPAIR"
        ),
        "passed": passed,
        "failures": list(failures),
        "protocol_commit": protocol_commit,
        "capture_day_utc": ARCHIVE_DAY,
        "fixed_interval": {
            "edge_exclusion_ms": EDGE_EXCLUSION_MS,
            "minimum_interval_ms": MINIMUM_INTERVAL_MS,
            "minimum_interval_ids": MINIMUM_INTERVAL_IDS,
            "first_ws_ms": websocket.first_ws_ms,
            "last_ws_ms": websocket.last_ws_ms,
            "start_ms_inclusive": websocket.start_ms,
            "end_ms_exclusive": websocket.end_ms,
            "duration_ms": websocket.end_ms - websocket.start_ms,
        },
        "websocket": {
            "capture_path": str(WEBSOCKET_CAPTURE),
            "capture_sha256": WEBSOCKET_CAPTURE_SHA256,
            "total_frames": websocket.total_frames,
            "trade_frames": websocket.trade_frames,
            "subscription_acks": websocket.subscription_acks,
            "pongs": websocket.pongs,
            "total_trades": len(websocket.records),
            "interval_trades": len(websocket.interval_records),
        },
        "archive": {
            "url": ARCHIVE_URL,
            "status": archive.metadata.status,
            "final_url": archive.metadata.final_url,
            "content_type": archive.metadata.content_type,
            "content_length_header": archive.metadata.content_length,
            "etag": archive.metadata.etag,
            "last_modified": archive.metadata.last_modified,
            "response_date": archive.metadata.response_date,
            "compressed_bytes": archive.compressed_bytes,
            "compressed_sha256": archive.compressed_sha256,
            "total_rows": archive.total_rows,
            "interval_rows": archive.interval_rows,
            "raw_archive_persisted": False,
        },
        "parity": dict(diagnostics),
        "interval_artifact": {
            "written": interval_sha256 is not None,
            "path": str(interval_path),
            "sha256": interval_sha256,
            "rows": len(websocket.interval_records) if passed else 0,
        },
        "disk": {
            "used_gib_before_request": disk_used_gib,
            "limit_gib": DISK_LIMIT_GIB,
        },
        "bindings": {
            "boundary": {"path": str(BOUNDARY), "sha256": BOUNDARY_SHA256},
            "source_decision": {
                "path": str(SOURCE_DECISION),
                "sha256": SOURCE_DECISION_SHA256,
            },
            "source_audit": {
                "path": str(SOURCE_AUDIT),
                "sha256": SOURCE_AUDIT_SHA256,
            },
            "bsea_contract": {
                "path": str(BSEA_CONTRACT),
                "sha256": BSEA_CONTRACT_SHA256,
            },
            "bsea_rejection": {
                "path": str(BSEA_REJECTION),
                "sha256": BSEA_REJECTION_SHA256,
            },
            "bsea_rejection_result": {
                "path": str(BSEA_REJECTION_RESULT),
                "sha256": BSEA_REJECTION_RESULT_SHA256,
            },
            "capture_manifest": {
                "path": str(CAPTURE_MANIFEST),
                "sha256": CAPTURE_MANIFEST_SHA256,
                "manifest_hash": capture_manifest[
                    "manifest_hash_without_self"
                ],
            },
            "websocket_capture": {
                "path": str(WEBSOCKET_CAPTURE),
                "sha256": WEBSOCKET_CAPTURE_SHA256,
            },
            "verifier": {
                "path": str(SCRIPT_PATH),
                "sha256": sha256_file(SCRIPT_PATH),
            },
            "tests": {
                "path": str(TEST_PATH),
                "sha256": sha256_file(TEST_PATH),
            },
        },
        "outcome_boundary": {
            "bsea_reopened": False,
            "bsea_family_authorized": False,
            "candidate_defined": False,
            "candidate_incidence_opened": False,
            "binance_comparator_opened": False,
            "market_rows_opened": False,
            "funding_rows_opened": False,
            "future_returns_opened": False,
            "labels_opened": False,
            "model_opened": False,
            "pnl_cagr_mdd_opened": False,
        },
        "next_authorized_step": (
            "freeze_one_orthogonal_candidate_boundary_before_historical_reduction"
            if passed
            else "none_retired_no_repair"
        ),
        "manifest_hash_scope": "canonical_json_excluding_manifest_hash",
    }
    report["manifest_hash"] = canonical_hash(report)
    return report


def build_terminal_failure_report(
    *,
    protocol_commit: str,
    capture_manifest: Mapping[str, Any],
    stage: str,
    failure: TerminalSourceFailure,
    disk_used_gib: int | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "transport_id": "BAWDP-v1",
        "decision": "RETIRE_BAWDP_V1_NO_REPAIR",
        "passed": False,
        "failures": [f"source:{stage}:{failure}"],
        "protocol_commit": protocol_commit,
        "capture_day_utc": ARCHIVE_DAY,
        "archive": {
            "url": ARCHIVE_URL,
            "raw_archive_persisted": False,
            "complete_archive_hash_available": False,
        },
        "disk": {
            "used_gib_before_request": disk_used_gib,
            "limit_gib": DISK_LIMIT_GIB,
        },
        "bindings": {
            "boundary": {"path": str(BOUNDARY), "sha256": BOUNDARY_SHA256},
            "capture_manifest": {
                "path": str(CAPTURE_MANIFEST),
                "sha256": CAPTURE_MANIFEST_SHA256,
                "manifest_hash": capture_manifest[
                    "manifest_hash_without_self"
                ],
            },
            "websocket_capture": {
                "path": str(WEBSOCKET_CAPTURE),
                "sha256": WEBSOCKET_CAPTURE_SHA256,
            },
            "verifier": {
                "path": str(SCRIPT_PATH),
                "sha256": sha256_file(SCRIPT_PATH),
            },
            "tests": {
                "path": str(TEST_PATH),
                "sha256": sha256_file(TEST_PATH),
            },
        },
        "outcome_boundary": {
            "bsea_reopened": False,
            "bsea_family_authorized": False,
            "candidate_defined": False,
            "candidate_incidence_opened": False,
            "binance_comparator_opened": False,
            "market_rows_opened": False,
            "funding_rows_opened": False,
            "future_returns_opened": False,
            "labels_opened": False,
            "model_opened": False,
            "pnl_cagr_mdd_opened": False,
        },
        "next_authorized_step": "none_retired_no_repair",
        "manifest_hash_scope": "canonical_json_excluding_manifest_hash",
    }
    report["manifest_hash"] = canonical_hash(report)
    return report


def atomic_write(path: Path, payload: bytes) -> str:
    target = repository_path(path)
    if target.exists():
        if target.read_bytes() != payload:
            raise DirectParityError(f"existing BAWDP artifact differs: {path}")
        return "verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        if target.exists():
            if target.read_bytes() != payload:
                raise DirectParityError(
                    f"concurrent BAWDP artifact differs: {path}"
                )
        else:
            os.replace(temporary, target)
            if os.name == "posix":
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                directory_fd = os.open(target.parent, flags)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def run(
    *,
    report_path: Path = DEFAULT_REPORT,
    interval_path: Path = DEFAULT_INTERVAL,
    fetcher: Any = fetch_archive,
) -> tuple[dict[str, Any], dict[str, str]]:
    protocol_commit = assert_protocol_committed()
    capture_manifest, _ = validate_frozen_bindings()
    stage = "websocket_capture"
    disk_used_gib: int | None = None
    try:
        websocket = load_websocket_capture()
    except TerminalSourceFailure as exc:
        report = build_terminal_failure_report(
            protocol_commit=protocol_commit,
            capture_manifest=capture_manifest,
            stage=stage,
            failure=exc,
            disk_used_gib=disk_used_gib,
        )
        return report, {
            "report": atomic_write(report_path, canonical_json_bytes(report))
        }
    disk_used_gib = _disk_used_gib()
    if disk_used_gib >= DISK_LIMIT_GIB:
        raise DirectParityError("BAWDP disk guard rejected the archive request")
    stage = "archive_transport_or_stream"
    try:
        archive = fetcher(
            start_ms=websocket.start_ms,
            end_ms=websocket.end_ms,
        )
    except TerminalSourceFailure as exc:
        report = build_terminal_failure_report(
            protocol_commit=protocol_commit,
            capture_manifest=capture_manifest,
            stage=stage,
            failure=exc,
            disk_used_gib=disk_used_gib,
        )
        return report, {
            "report": atomic_write(report_path, canonical_json_bytes(report))
        }
    failures, diagnostics = evaluate_parity(
        websocket.interval_records,
        archive.records,
    )
    interval_bytes: bytes | None = None
    interval_sha256: str | None = None
    if not failures:
        interval_bytes = deterministic_interval_gzip(websocket.interval_records)
        interval_sha256 = sha256_bytes(interval_bytes)
    report = build_report(
        protocol_commit=protocol_commit,
        capture_manifest=capture_manifest,
        websocket=websocket,
        archive=archive,
        failures=failures,
        diagnostics=diagnostics,
        interval_path=interval_path,
        interval_sha256=interval_sha256,
        disk_used_gib=disk_used_gib,
    )
    statuses: dict[str, str] = {}
    if interval_bytes is not None:
        statuses["interval"] = atomic_write(interval_path, interval_bytes)
    statuses["report"] = atomic_write(
        report_path,
        canonical_json_bytes(report),
    )
    return report, statuses


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--interval", default=str(DEFAULT_INTERVAL))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, statuses = run(
        report_path=Path(args.report),
        interval_path=Path(args.interval),
    )
    print(
        json.dumps(
            {
                "transport_id": report["transport_id"],
                "decision": report["decision"],
                "passed": report["passed"],
                "manifest_hash": report["manifest_hash"],
                "archive_sha256": report["archive"].get("compressed_sha256"),
                "interval_rows": report.get("interval_artifact", {}).get(
                    "rows",
                    0,
                ),
                "statuses": statuses,
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
