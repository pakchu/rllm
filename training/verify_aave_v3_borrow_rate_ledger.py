"""Verify the frozen Aave V3 borrow-rate source identity without outcomes."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import http.client
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from Crypto.Hash import keccak


PROTOCOL_VERSION = "av3brl_source_parity_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_PATH = Path(
    "docs/aave-v3-borrow-rate-ledger-source-boundary-2026-07-24.md"
)
BOUNDARY_SHA256 = (
    "3e52dd1d6c92c585cc1d8437fe66e4ea16fa71bfc02d0c5764a45266ea4f7390"
)
SCRIPT_PATH = Path("training/verify_aave_v3_borrow_rate_ledger.py")
TEST_PATH = Path("tests/test_verify_aave_v3_borrow_rate_ledger.py")

PRIMARY_RPC = "https://eth-mainnet.public.blastapi.io"
PARITY_RPC = "https://ethereum-rpc.publicnode.com"
EXPECTED_CHAIN_ID = 1

POOL_ADDRESS = "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2"
RESERVE_ADDRESSES = (
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "0xdac17f958d2ee523a2206206994597c13d831ec7",
)
RESERVE_TOPICS = tuple(
    "0x" + "0" * 24 + address[2:] for address in RESERVE_ADDRESSES
)
RESERVE_BY_TOPIC = dict(zip(RESERVE_TOPICS, RESERVE_ADDRESSES, strict=True))

EVENT_SIGNATURE = (
    "ReserveDataUpdated(address,uint256,uint256,uint256,uint256,uint256)"
)
EVENT_TOPIC = (
    "0x804c9b842b2748a22bb64b345453a3de7ca54a6ca45ce00d415894979e22897a"
)
NORMALIZED_EVENT_DECLARATION = (
    "event ReserveDataUpdated( address indexed reserve, "
    "uint256 liquidityRate, uint256 stableBorrowRate, "
    "uint256 variableBorrowRate, uint256 liquidityIndex, "
    "uint256 variableBorrowIndex );"
)

CHUNK_SIZE = 512
SUBDIVISION_SIZE = 128
TOTAL_STAGE_A_CHUNKS = 29
FULL_HISTORY_FIRST_BLOCK = 17_382_266
FULL_HISTORY_LAST_BLOCK = 25_602_012
FULL_HISTORY_CHUNK_COUNT = 16_055

SCHEMA_SHA256 = (
    "33e62c18f3e221a646ac3f18e90fe1cf447c6c17bc1b14ca52cb72fcb0775590"
)
SCHEMA_PREIMAGE = (
    b"AV3BRL-v1\0schema\0"
    b"event_keys=address,block_hash,block_number,data_words,log_index,"
    b"reserve,topics,transaction_hash,transaction_index\n"
    b"data_words=liquidity_rate,stable_borrow_rate,variable_borrow_rate,"
    b"liquidity_index,variable_borrow_index\n"
    b"ledger_header=block_number,block_timestamp,transaction_index,"
    b"log_index,block_hash,transaction_hash,reserve,variable_borrow_rate\n"
)

DISK_LIMIT_BYTES = 300 * 1024**3
DISK_MINIMUM_FREE_BYTES = 1 * 1024**3
MAXIMUM_HTTP_BODY_BYTES = 64 * 1024**2
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 60
REQUEST_INTERVAL_SECONDS = 0.5
RETRY_DELAYS_SECONDS = (1.0, 2.0)
USER_AGENT = "rllm-av3brl-source-parity/1.0"

DEFAULT_SENTINEL = Path(
    "results/.aave_v3_borrow_rate_ledger_source_parity_2026-07-24.started"
)
DEFAULT_TEMP_REPORT = Path(
    "results/.aave_v3_borrow_rate_ledger_source_parity_2026-07-24.json.tmp"
)
DEFAULT_REPORT = Path(
    "results/aave_v3_borrow_rate_ledger_source_parity_2026-07-24.json"
)

PINNED_ADDRESS_BOOK_URL = (
    "https://raw.githubusercontent.com/aave-dao/aave-address-book/"
    "21004a8871ca774b8da27114cbbd74931f3a436f/src/AaveV3Ethereum.sol"
)
PINNED_ADDRESS_BOOK_LICENSE_URL = (
    "https://raw.githubusercontent.com/aave-dao/aave-address-book/"
    "21004a8871ca774b8da27114cbbd74931f3a436f/LICENSE"
)
PINNED_CURRENT_IPOOL_URL = (
    "https://raw.githubusercontent.com/aave-dao/aave-v3-origin/"
    "fd1fbd9150426ca8ace9cee45b4acf912ae84f5b/"
    "src/contracts/interfaces/IPool.sol"
)
PINNED_ARCHIVED_IPOOL_URL = (
    "https://raw.githubusercontent.com/aave/aave-v3-core/"
    "782f51917056a53a2c228701058a6c3fb233684a/"
    "contracts/interfaces/IPool.sol"
)
PINNED_AAVE_UTILITIES_README_URL = (
    "https://raw.githubusercontent.com/aave/aave-utilities/"
    "2d0db501009fdb34824cbc9a486c3e9fd7191ec7/README.md"
)


class Av3brlError(RuntimeError):
    """Base failure for the frozen source protocol."""


class ProtocolError(Av3brlError):
    """The committed one-shot protocol precondition failed."""


class TransportError(Av3brlError):
    """A frozen transport failed."""


class SchemaError(Av3brlError):
    """A source payload violated the frozen schema."""


class ParityError(Av3brlError):
    """Canonical source views disagreed."""


class SupportError(Av3brlError):
    """The fixed source-support gate failed."""


class HeaderError(Av3brlError):
    """An Ethereum block header violated the frozen identity."""


class DiskGuardError(Av3brlError):
    """The WSL filesystem guard failed."""


class PublicationError(Av3brlError):
    """A terminal artifact could not be published atomically."""


class InterruptedError(Av3brlError):
    """The one-shot source run was interrupted."""


@dataclass(frozen=True, slots=True)
class PinSpec:
    name: str
    url: str
    sha256: str


PIN_SPECS = (
    PinSpec(
        "address_book",
        PINNED_ADDRESS_BOOK_URL,
        "1a862b7389de3d59ee77680a8edb451a29e630a07813a0c5becdce65d730a22a",
    ),
    PinSpec(
        "address_book_license",
        PINNED_ADDRESS_BOOK_LICENSE_URL,
        "719dd01a9b4549c3f489ff04420d4e8f010d91a368f57894505fe691e19b6c48",
    ),
    PinSpec(
        "current_ipool",
        PINNED_CURRENT_IPOOL_URL,
        "ee58bacafc0b033adeaf673da3b08130d9f3a9a3eb563f7f19a03240e488eba7",
    ),
    PinSpec(
        "archived_ipool",
        PINNED_ARCHIVED_IPOOL_URL,
        "d708ba2c3cf29fe81083b7b8127ef61b7d750dc017399ebaa2951d6f61a93dda",
    ),
    PinSpec(
        "aave_utilities_readme",
        PINNED_AAVE_UTILITIES_README_URL,
        "42f2a95d8938619b074e143f31388c02c39c38bfde731dccefea594efd3963fe",
    ),
)


def _utc_timestamp(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


@dataclass(frozen=True, slots=True)
class BoundaryHeader:
    number: int
    timestamp: int
    block_hash: str


BOUNDARY_HEADERS = (
    BoundaryHeader(
        17_382_265,
        _utc_timestamp("2023-05-31T23:59:59Z"),
        "0x5b54c7cbb1816c36c6b5431e31ca12b1162b8d87a305318702d954e1f2b0b0cc",
    ),
    BoundaryHeader(
        17_382_266,
        _utc_timestamp("2023-06-01T00:00:11Z"),
        "0xe0ef11cab4909c80599087b4ffb0bf1e92b1affcc72abc3b802f20a9d5d21096",
    ),
    BoundaryHeader(
        17_389_364,
        _utc_timestamp("2023-06-01T23:59:59Z"),
        "0x34e4ed79dbd53a7fd5e8455b3d01b44ce186d8b79e7b186c8b153ca869d91cfc",
    ),
    BoundaryHeader(
        17_389_365,
        _utc_timestamp("2023-06-02T00:00:11Z"),
        "0x73be93661691df088828824d1bd7ff3cd555426678744f8ff7aa0a2dde15643b",
    ),
    BoundaryHeader(
        25_594_813,
        _utc_timestamp("2026-07-23T10:23:35Z"),
        "0x048f2cdcafc8b22de300cb6bf9b7c4d60cdba5b7d2ecb99871b152e586ce30a2",
    ),
    BoundaryHeader(
        25_602_012,
        _utc_timestamp("2026-07-24T10:30:47Z"),
        "0x53d736133c333e9751920c0a45398938533a04c0601599aafe22636c0d43ddad",
    ),
)


@dataclass(frozen=True, slots=True)
class WindowSpec:
    name: str
    first_block: int
    first_block_hash: str
    last_block: int
    last_block_hash: str
    chunk_count: int
    first_ordinal: int


HISTORICAL_WINDOW = WindowSpec(
    name="historical",
    first_block=17_382_266,
    first_block_hash=(
        "0xe0ef11cab4909c80599087b4ffb0bf1e92b1affcc72abc3b802f20a9d5d21096"
    ),
    last_block=17_389_364,
    last_block_hash=(
        "0x34e4ed79dbd53a7fd5e8455b3d01b44ce186d8b79e7b186c8b153ca869d91cfc"
    ),
    chunk_count=14,
    first_ordinal=0,
)
RECENT_WINDOW = WindowSpec(
    name="recent",
    first_block=25_594_813,
    first_block_hash=(
        "0x048f2cdcafc8b22de300cb6bf9b7c4d60cdba5b7d2ecb99871b152e586ce30a2"
    ),
    last_block=25_602_012,
    last_block_hash=(
        "0x53d736133c333e9751920c0a45398938533a04c0601599aafe22636c0d43ddad"
    ),
    chunk_count=15,
    first_ordinal=14,
)
WINDOWS = (HISTORICAL_WINDOW, RECENT_WINDOW)


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    start: int
    end: int

    @property
    def request_id(self) -> int:
        return 200_000_000 + self.ordinal

    def subdivisions(self) -> tuple["Subdivision", ...]:
        rows: list[Subdivision] = []
        start = self.start
        local = 0
        while start <= self.end:
            end = min(self.end, start + SUBDIVISION_SIZE - 1)
            rows.append(
                Subdivision(
                    parent_ordinal=self.ordinal,
                    ordinal=local,
                    start=start,
                    end=end,
                )
            )
            local += 1
            start = end + 1
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class Subdivision:
    parent_ordinal: int
    ordinal: int
    start: int
    end: int

    @property
    def request_id(self) -> int:
        return 300_000_000 + 4 * self.parent_ordinal + self.ordinal


def chunks_for_window(window: WindowSpec) -> tuple[Chunk, ...]:
    rows: list[Chunk] = []
    start = window.first_block
    ordinal = window.first_ordinal
    while start <= window.last_block:
        end = min(window.last_block, start + CHUNK_SIZE - 1)
        rows.append(Chunk(ordinal=ordinal, start=start, end=end))
        ordinal += 1
        start = end + 1
    if len(rows) != window.chunk_count:
        raise AssertionError("frozen window chunk count differs")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class BlockHeader:
    number: int
    block_hash: str
    parent_hash: str
    timestamp: int


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    address: str
    block_hash: str
    block_number: int
    data_words: tuple[int, int, int, int, int]
    log_index: int
    reserve: str
    topics: tuple[str, str]
    transaction_hash: str
    transaction_index: int

    @property
    def order_key(self) -> tuple[int, int, int]:
        return self.block_number, self.transaction_index, self.log_index

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.block_hash, self.transaction_hash, self.log_index

    def as_json_object(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "block_hash": self.block_hash,
            "block_number": self.block_number,
            "data_words": list(self.data_words),
            "log_index": self.log_index,
            "reserve": self.reserve,
            "topics": list(self.topics),
            "transaction_hash": self.transaction_hash,
            "transaction_index": self.transaction_index,
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_json_object())


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    historical_sha256: str
    recent_sha256: str


class PairRpc(Protocol):
    def call_pair(
        self,
        method: str,
        params: Sequence[Any],
        request_id: int,
        *,
        capability: object | None = None,
    ) -> tuple[Any, Any]:
        """Return primary and parity JSON-RPC results."""


Fetcher = Callable[[str], bytes]
DiskGuard = Callable[[], tuple[int, int]]
Progress = Callable[[int, int, float], None]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def repository_artifact_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPOSITORY_ROOT / path).resolve()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _deduplicating_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError("JSON object contains a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SchemaError(f"JSON constant {value} is forbidden")


def parse_json_object(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_deduplicating_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError("JSON body is malformed") from exc
    if not isinstance(value, dict):
        raise SchemaError("JSON-RPC body is not one object")
    return value


def validate_http_body(
    raw: bytes,
    headers: Mapping[str, str],
) -> None:
    if len(raw) > MAXIMUM_HTTP_BODY_BYTES:
        raise TransportError("HTTP body exceeds byte cap")
    content_length: str | None = None
    for key, value in headers.items():
        if key.lower() == "content-length":
            content_length = value
            break
    if content_length is None:
        return
    try:
        expected = int(content_length)
    except ValueError as exc:
        raise TransportError("HTTP content length is invalid") from exc
    if expected != len(raw):
        raise TransportError("HTTP body is partial")


def _source_capability_api() -> tuple[
    Callable[[Path, str], object],
    Callable[[object], None],
]:
    class SourceCapability:
        __slots__ = ("commit", "sentinel_path", "sentinel_sha256")

        def __init__(
            self,
            *,
            commit: str,
            sentinel_path: Path,
            sentinel_sha256: str,
        ) -> None:
            self.commit = commit
            self.sentinel_path = sentinel_path
            self.sentinel_sha256 = sentinel_sha256

    def mint(sentinel_path: Path, verifier_commit: str) -> object:
        expected_path = repository_artifact_path(DEFAULT_SENTINEL)
        if sentinel_path.resolve() != expected_path:
            raise ProtocolError("source capability sentinel path differs")
        committed = assert_protocol_committed()
        if committed != verifier_commit:
            raise ProtocolError("source capability verifier commit differs")
        assert_disk_guard()
        try:
            payload = sentinel_path.read_bytes()
        except OSError as exc:
            raise ProtocolError("source capability sentinel is unavailable") from exc
        decoded = parse_json_object(payload)
        if set(decoded) != {
            "protocol_version",
            "source_boundary_sha256",
            "started_at_utc",
            "verifier_commit",
        }:
            raise ProtocolError("source capability sentinel schema differs")
        if (
            decoded["protocol_version"] != PROTOCOL_VERSION
            or decoded["source_boundary_sha256"] != BOUNDARY_SHA256
            or decoded["verifier_commit"] != verifier_commit
        ):
            raise ProtocolError("source capability sentinel identity differs")
        return SourceCapability(
            commit=verifier_commit,
            sentinel_path=sentinel_path,
            sentinel_sha256=sha256_bytes(payload),
        )

    def require(capability: object) -> None:
        if not isinstance(capability, SourceCapability):
            raise ProtocolError("source capability is absent")
        try:
            current = sha256_file(capability.sentinel_path)
        except OSError as exc:
            raise ProtocolError("source capability sentinel is unavailable") from exc
        if current != capability.sentinel_sha256:
            raise ProtocolError("source capability sentinel changed")
        assert_disk_guard()

    return mint, require


_mint_source_capability, _require_source_capability = _source_capability_api()


_FIXED_HEX_RE = re.compile(r"^0x[0-9a-fA-F]+$")
_QUANTITY_RE = re.compile(r"^0x(?:0|[1-9a-fA-F][0-9a-fA-F]*)$")
_LOWER_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CANONICAL_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def canonical_fixed_hex(value: Any, byte_length: int, field: str) -> str:
    if not isinstance(value, str) or not _FIXED_HEX_RE.fullmatch(value):
        raise SchemaError(f"{field} is not hexadecimal bytes")
    if len(value) != 2 + 2 * byte_length:
        raise SchemaError(f"{field} has the wrong byte length")
    return value.lower()


def parse_quantity(value: Any, field: str) -> int:
    if not isinstance(value, str) or not _QUANTITY_RE.fullmatch(value):
        raise SchemaError(f"{field} is not a minimal hexadecimal quantity")
    parsed = int(value[2:], 16)
    if hex(parsed) != value.lower():
        raise SchemaError(f"{field} is not canonically minimal")
    return parsed


def canonical_quantity(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError("quantity input is invalid")
    return hex(value)


def parse_block_header(value: Any, expected_number: int | None = None) -> BlockHeader:
    if not isinstance(value, dict):
        raise HeaderError("block header is not an object")
    try:
        number = parse_quantity(value["number"], "block number")
        block_hash = canonical_fixed_hex(value["hash"], 32, "block hash")
        parent_hash = canonical_fixed_hex(
            value["parentHash"], 32, "parent block hash"
        )
        timestamp = parse_quantity(value["timestamp"], "block timestamp")
    except KeyError as exc:
        raise HeaderError("block header field is missing") from exc
    except SchemaError as exc:
        raise HeaderError("block header field is malformed") from exc
    if expected_number is not None and number != expected_number:
        raise HeaderError("exact block number differs")
    return BlockHeader(
        number=number,
        block_hash=block_hash,
        parent_hash=parent_hash,
        timestamp=timestamp,
    )


def parse_event(value: Any, *, first_block: int, last_block: int) -> CanonicalEvent:
    if not isinstance(value, dict):
        raise SchemaError("log row is not an object")
    try:
        address = canonical_fixed_hex(value["address"], 20, "log address")
        raw_topics = value["topics"]
        data = canonical_fixed_hex(value["data"], 160, "log data")
        block_number = parse_quantity(value["blockNumber"], "block number")
        block_hash = canonical_fixed_hex(value["blockHash"], 32, "block hash")
        transaction_hash = canonical_fixed_hex(
            value["transactionHash"], 32, "transaction hash"
        )
        transaction_index = parse_quantity(
            value["transactionIndex"], "transaction index"
        )
        log_index = parse_quantity(value["logIndex"], "log index")
        removed = value["removed"]
    except KeyError as exc:
        raise SchemaError("required log field is missing") from exc
    if address != POOL_ADDRESS:
        raise SchemaError("log address differs")
    if not isinstance(raw_topics, list) or len(raw_topics) != 2:
        raise SchemaError("log topics differ")
    topic0 = canonical_fixed_hex(raw_topics[0], 32, "event topic")
    topic1 = canonical_fixed_hex(raw_topics[1], 32, "reserve topic")
    if topic0 != EVENT_TOPIC or topic1 not in RESERVE_BY_TOPIC:
        raise SchemaError("log topic identity differs")
    if removed is not False:
        raise SchemaError("removed log is forbidden")
    if not first_block <= block_number <= last_block:
        raise SchemaError("log is outside requested range")
    words = tuple(
        int(data[2 + offset : 2 + offset + 64], 16)
        for offset in range(0, 320, 64)
    )
    if len(words) != 5:
        raise AssertionError("ABI word count differs")
    return CanonicalEvent(
        address=address,
        block_hash=block_hash,
        block_number=block_number,
        data_words=words,  # type: ignore[arg-type]
        log_index=log_index,
        reserve=RESERVE_BY_TOPIC[topic1],
        topics=(topic0, topic1),
        transaction_hash=transaction_hash,
        transaction_index=transaction_index,
    )


def canonicalize_events(
    value: Any,
    *,
    first_block: int,
    last_block: int,
) -> tuple[CanonicalEvent, ...]:
    if not isinstance(value, list):
        raise SchemaError("eth_getLogs result is not an array")
    events = tuple(
        sorted(
            (
                parse_event(
                    row,
                    first_block=first_block,
                    last_block=last_block,
                )
                for row in value
            ),
            key=lambda row: row.order_key,
        )
    )
    identities: set[tuple[str, str, int]] = set()
    order_keys: set[tuple[int, int, int]] = set()
    for event in events:
        if event.identity in identities:
            raise ParityError("canonical event identity is duplicated")
        if event.order_key in order_keys:
            raise ParityError("canonical event position is duplicated")
        identities.add(event.identity)
        order_keys.add(event.order_key)
    return events


def window_stream_sha256(name: str, events: Sequence[CanonicalEvent]) -> str:
    if name not in {"historical", "recent"}:
        raise ProtocolError("window hash name differs")
    digest = hashlib.sha256()
    digest.update(b"AV3BRL-v1\0window\0" + name.encode("ascii") + b"\0")
    for event in events:
        row = event.canonical_json_bytes()
        digest.update(len(row).to_bytes(4, "big"))
        digest.update(row)
    return digest.hexdigest()


def get_logs_filter(first_block: int, last_block: int) -> dict[str, Any]:
    return {
        "address": POOL_ADDRESS,
        "fromBlock": canonical_quantity(first_block),
        "toBlock": canonical_quantity(last_block),
        "topics": [EVENT_TOPIC, list(RESERVE_TOPICS)],
    }


class _RateLimiter:
    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_request: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = self._monotonic()
            if self._last_request is not None:
                delay = REQUEST_INTERVAL_SECONDS - (now - self._last_request)
                if delay > 0:
                    self._sleep(delay)
                    now = self._monotonic()
            self._last_request = now


class JsonRpcClient:
    """Strict one-object JSON-RPC HTTPS client for one frozen provider."""

    def __init__(
        self,
        base_url: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ProtocolError("RPC base URL differs")
        self.base_url = base_url
        self.host = parsed.hostname
        self.port = parsed.port
        self.path = parsed.path or "/"
        self._sleep = sleep
        self._limiter = _RateLimiter(monotonic=monotonic, sleep=sleep)
        self._ssl_context = ssl.create_default_context()

    def _request_once(self, body: bytes) -> tuple[int, Mapping[str, str], bytes]:
        self._limiter.wait()
        connection = http.client.HTTPSConnection(
            self.host,
            self.port,
            timeout=CONNECT_TIMEOUT_SECONDS,
            context=self._ssl_context,
        )
        response: http.client.HTTPResponse | None = None
        try:
            connection.request(
                "POST",
                self.path,
                body=body,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
            response = connection.getresponse()
            status = int(response.status)
            if status == 429 or 500 <= status <= 599:
                return status, dict(response.headers.items()), b""
            if status != 200:
                raise TransportError("HTTP status is not accepted")
            if connection.sock is not None:
                connection.sock.settimeout(READ_TIMEOUT_SECONDS)
            raw = response.read(MAXIMUM_HTTP_BODY_BYTES + 1)
            headers = dict(response.headers.items())
            validate_http_body(raw, headers)
            return status, headers, raw
        except TransportError:
            raise
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            if response is None:
                raise ConnectionError("pre-body connection failure") from exc
            raise TransportError("HTTP body failed") from exc
        finally:
            connection.close()

    def call(
        self,
        method: str,
        params: Sequence[Any],
        request_id: int,
        *,
        capability: object | None = None,
    ) -> Any:
        if method == "eth_getLogs":
            if capability is None:
                raise ProtocolError("eth_getLogs source capability is absent")
            _require_source_capability(capability)
        request = {
            "id": request_id,
            "jsonrpc": "2.0",
            "method": method,
            "params": list(params),
        }
        body = canonical_json_bytes(request)
        attempts = 1 + len(RETRY_DELAYS_SECONDS)
        for attempt in range(attempts):
            if attempt:
                self._sleep(RETRY_DELAYS_SECONDS[attempt - 1])
            try:
                status, _, raw = self._request_once(body)
            except ConnectionError as exc:
                if attempt + 1 == attempts:
                    raise TransportError("pre-body connection retries exhausted") from exc
                continue
            if status == 429 or 500 <= status <= 599:
                if attempt + 1 == attempts:
                    raise TransportError("retryable HTTP status exhausted")
                continue
            envelope = parse_json_object(raw)
            if envelope.get("jsonrpc") != "2.0":
                raise SchemaError("JSON-RPC version differs")
            response_id = envelope.get("id")
            if type(response_id) is not int or response_id != request_id:
                raise SchemaError("JSON-RPC response ID differs")
            if "error" in envelope:
                raise TransportError("JSON-RPC error returned")
            if "result" not in envelope:
                raise SchemaError("JSON-RPC result is missing")
            return envelope["result"]
        raise AssertionError("unreachable retry state")


class DualRpc:
    def __init__(
        self,
        primary: JsonRpcClient | None = None,
        parity: JsonRpcClient | None = None,
    ) -> None:
        self.primary = primary or JsonRpcClient(PRIMARY_RPC)
        self.parity = parity or JsonRpcClient(PARITY_RPC)

    def call_pair(
        self,
        method: str,
        params: Sequence[Any],
        request_id: int,
        *,
        capability: object | None = None,
    ) -> tuple[Any, Any]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            primary_future = executor.submit(
                self.primary.call,
                method,
                params,
                request_id,
                capability=capability,
            )
            parity_future = executor.submit(
                self.parity.call,
                method,
                params,
                request_id,
                capability=capability,
            )
            return primary_future.result(), parity_future.result()


class HeaderCachingPairRpc:
    """Deduplicate exact block-header pairs across one frozen stage."""

    def __init__(self, delegate: PairRpc) -> None:
        self.delegate = delegate
        self._headers: dict[
            int,
            tuple[tuple[Any, ...], tuple[Any, Any]],
        ] = {}

    def call_pair(
        self,
        method: str,
        params: Sequence[Any],
        request_id: int,
        *,
        capability: object | None = None,
    ) -> tuple[Any, Any]:
        frozen_params = tuple(params)
        is_exact_header = (
            method == "eth_getBlockByNumber"
            and len(frozen_params) == 2
            and frozen_params[0] != "finalized"
        )
        if not is_exact_header:
            return self.delegate.call_pair(
                method,
                frozen_params,
                request_id,
                capability=capability,
            )
        previous = self._headers.get(request_id)
        if previous is not None:
            previous_params, result = previous
            if previous_params != frozen_params:
                raise ProtocolError("block request ID was reused")
            return result
        result = self.delegate.call_pair(
            method,
            frozen_params,
            request_id,
            capability=capability,
        )
        self._headers[request_id] = (frozen_params, result)
        return result


def fetch_https_bytes(url: str) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ProtocolError("pinned URL differs")
    connection = http.client.HTTPSConnection(
        parsed.hostname,
        parsed.port,
        timeout=CONNECT_TIMEOUT_SECONDS,
        context=ssl.create_default_context(),
    )
    try:
        path = parsed.path or "/"
        connection.request(
            "GET",
            path,
            headers={
                "Accept": "text/plain,application/octet-stream",
                "Accept-Encoding": "identity",
                "User-Agent": USER_AGENT,
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            raise TransportError("pinned source HTTP status differs")
        if connection.sock is not None:
            connection.sock.settimeout(READ_TIMEOUT_SECONDS)
        raw = response.read(MAXIMUM_HTTP_BODY_BYTES + 1)
        validate_http_body(raw, dict(response.headers.items()))
        return raw
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        raise TransportError("pinned source transport failed") from exc
    finally:
        connection.close()


_SOLIDITY_LINE_COMMENT = re.compile(r"//[^\n]*")
_SOLIDITY_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_EVENT_DECLARATION = re.compile(
    r"\bevent\s+ReserveDataUpdated\s*\(.*?\)\s*;",
    re.DOTALL,
)


def normalized_event_declaration(source: bytes) -> str:
    try:
        text = source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SchemaError("pinned interface is not UTF-8") from exc
    without_comments = _SOLIDITY_BLOCK_COMMENT.sub(
        "", _SOLIDITY_LINE_COMMENT.sub("", text)
    )
    matches = _EVENT_DECLARATION.findall(without_comments)
    if len(matches) != 1:
        raise SchemaError("ReserveDataUpdated declaration count differs")
    return " ".join(matches[0].split())


def computed_event_topic() -> str:
    digest = keccak.new(digest_bits=256)
    digest.update(EVENT_SIGNATURE.encode("ascii"))
    return "0x" + digest.hexdigest()


def _verify_pins(fetcher: Fetcher = fetch_https_bytes) -> None:
    payloads: dict[str, bytes] = {}
    for pin in PIN_SPECS:
        payload = fetcher(pin.url)
        if sha256_bytes(payload) != pin.sha256:
            raise ProtocolError("pinned source hash differs")
        payloads[pin.name] = payload
    current = normalized_event_declaration(payloads["current_ipool"])
    archived = normalized_event_declaration(payloads["archived_ipool"])
    if (
        current != NORMALIZED_EVENT_DECLARATION
        or archived != NORMALIZED_EVENT_DECLARATION
        or current != archived
    ):
        raise SchemaError("pinned event declarations differ")
    if computed_event_topic() != EVENT_TOPIC:
        raise SchemaError("computed event topic differs")
    try:
        address_book = payloads["address_book"].decode("utf-8", errors="strict").lower()
        utilities = payloads["aave_utilities_readme"].decode(
            "utf-8", errors="strict"
        )
    except UnicodeDecodeError as exc:
        raise SchemaError("pinned text source is not UTF-8") from exc
    for address in (POOL_ADDRESS, *RESERVE_ADDRESSES):
        if address[2:] not in address_book:
            raise SchemaError("frozen address is absent from address book")
    if PRIMARY_RPC not in utilities:
        raise SchemaError("frozen primary RPC is absent from pinned example")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def assert_protocol_committed() -> str:
    if sha256_file(REPOSITORY_ROOT / BOUNDARY_PATH) != BOUNDARY_SHA256:
        raise ProtocolError("source boundary hash differs")
    for path in (BOUNDARY_PATH, SCRIPT_PATH, TEST_PATH):
        result = _git("ls-files", "--error-unmatch", str(path))
        if result.returncode != 0:
            raise ProtocolError("required protocol path is not committed")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0 or status.stdout:
        raise ProtocolError("repository is not HEAD-clean")
    head = _git("rev-parse", "HEAD")
    commit = head.stdout.strip()
    if head.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ProtocolError("verifier commit cannot be resolved")
    return commit


def assert_disk_guard(path: Path = REPOSITORY_ROOT) -> tuple[int, int]:
    usage = shutil.disk_usage(path)
    if usage.used >= DISK_LIMIT_BYTES or usage.free < DISK_MINIMUM_FREE_BYTES:
        raise DiskGuardError("filesystem guard failed")
    return usage.used, usage.free


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _renameat2_function() -> Any:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.renameat2
    except AttributeError as exc:
        raise PublicationError("renameat2 is unavailable") from exc
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    return function


def rename_noreplace(source: Path, destination: Path) -> None:
    function = _renameat2_function()
    result = function(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise PublicationError("renameat2 publication failed") from OSError(
            error, os.strerror(error)
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def assert_atomic_publication_supported(parent: Path) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    probe = Path(tempfile.mkdtemp(prefix=".av3brl-rename-probe-", dir=parent))
    destination = probe.with_name(probe.name + ".renamed")
    try:
        rename_noreplace(probe, destination)
        destination.rmdir()
    except BaseException:
        if probe.exists():
            probe.rmdir()
        if destination.exists():
            destination.rmdir()
        raise


def atomic_publish_report(
    *,
    temporary: Path,
    destination: Path,
    payload: bytes,
) -> None:
    if temporary.parent.resolve() != destination.parent.resolve():
        raise PublicationError("report staging parent differs")
    if temporary.exists() or destination.exists():
        raise PublicationError("terminal report path already exists")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        rename_noreplace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ProtocolError("timestamp is not timezone-aware")
    return value.astimezone(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def reserve_one_shot(
    *,
    sentinel_path: Path,
    report_path: Path,
    temp_report_path: Path,
    verifier_commit: str,
    started_at_utc: str,
) -> None:
    for path in (sentinel_path, report_path, temp_report_path):
        if path.exists():
            raise ProtocolError("one-shot artifact already exists")
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(
        {
            "protocol_version": PROTOCOL_VERSION,
            "source_boundary_sha256": BOUNDARY_SHA256,
            "started_at_utc": started_at_utc,
            "verifier_commit": verifier_commit,
        }
    ) + b"\n"
    descriptor = os.open(
        sentinel_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(sentinel_path.parent)
    except BaseException:
        raise


def _require_equal_headers(
    primary: Any,
    parity: Any,
    *,
    expected_number: int,
) -> BlockHeader:
    first = parse_block_header(primary, expected_number)
    second = parse_block_header(parity, expected_number)
    if first != second:
        raise HeaderError("provider block headers disagree")
    return first


def _verify_chain_and_boundaries(rpc: PairRpc) -> None:
    primary_chain, parity_chain = rpc.call_pair("eth_chainId", (), 1)
    try:
        chain_ids = (
            parse_quantity(primary_chain, "primary chain ID"),
            parse_quantity(parity_chain, "parity chain ID"),
        )
    except SchemaError as exc:
        raise HeaderError("chain identity is malformed") from exc
    if chain_ids != (EXPECTED_CHAIN_ID, EXPECTED_CHAIN_ID):
        raise HeaderError("chain identity differs")

    primary_finalized, parity_finalized = rpc.call_pair(
        "eth_getBlockByNumber", ("finalized", False), 2
    )
    finalized = (
        parse_block_header(primary_finalized),
        parse_block_header(parity_finalized),
    )
    if any(row.number < FULL_HISTORY_LAST_BLOCK for row in finalized):
        raise HeaderError("finalized head is behind frozen range")

    for expected in BOUNDARY_HEADERS:
        request_id = 100_000_000 + expected.number
        primary, parity = rpc.call_pair(
            "eth_getBlockByNumber",
            (canonical_quantity(expected.number), False),
            request_id,
        )
        actual = _require_equal_headers(
            primary,
            parity,
            expected_number=expected.number,
        )
        if (
            actual.block_hash != expected.block_hash
            or actual.timestamp != expected.timestamp
        ):
            raise HeaderError("frozen boundary header differs")


def _fetch_log_view(
    capability: object,
    rpc: PairRpc,
    *,
    start: int,
    end: int,
    request_id: int,
    disk_guard: DiskGuard,
) -> tuple[tuple[CanonicalEvent, ...], tuple[CanonicalEvent, ...]]:
    _require_source_capability(capability)
    disk_guard()
    primary, parity = rpc.call_pair(
        "eth_getLogs",
        (get_logs_filter(start, end),),
        request_id,
        capability=capability,
    )
    return (
        canonicalize_events(primary, first_block=start, last_block=end),
        canonicalize_events(parity, first_block=start, last_block=end),
    )


def _fetch_chunk_with_redundancy(
    capability: object,
    rpc: PairRpc,
    chunk: Chunk,
    *,
    disk_guard: DiskGuard = assert_disk_guard,
) -> tuple[CanonicalEvent, ...]:
    primary_parent, parity_parent = _fetch_log_view(
        capability,
        rpc,
        start=chunk.start,
        end=chunk.end,
        request_id=chunk.request_id,
        disk_guard=disk_guard,
    )
    if primary_parent != parity_parent:
        raise ParityError("parent provider streams disagree")

    primary_parts: list[CanonicalEvent] = []
    parity_parts: list[CanonicalEvent] = []
    for subdivision in chunk.subdivisions():
        primary, parity = _fetch_log_view(
            capability,
            rpc,
            start=subdivision.start,
            end=subdivision.end,
            request_id=subdivision.request_id,
            disk_guard=disk_guard,
        )
        if primary != parity:
            raise ParityError("subdivision provider streams disagree")
        primary_parts.extend(primary)
        parity_parts.extend(parity)

    primary_union = tuple(sorted(primary_parts, key=lambda row: row.order_key))
    parity_union = tuple(sorted(parity_parts, key=lambda row: row.order_key))
    if (
        primary_union != primary_parent
        or parity_union != parity_parent
        or primary_union != parity_union
    ):
        raise ParityError("parent and subdivision streams disagree")
    return primary_parent


def _fetch_windows(
    capability: object,
    rpc: PairRpc,
    *,
    disk_guard: DiskGuard = assert_disk_guard,
    progress: Progress | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, tuple[CanonicalEvent, ...]]:
    started = monotonic()
    streams: dict[str, tuple[CanonicalEvent, ...]] = {}
    global_identities: set[tuple[str, str, int]] = set()
    completed = 0
    for window in WINDOWS:
        rows: list[CanonicalEvent] = []
        for chunk in chunks_for_window(window):
            events = _fetch_chunk_with_redundancy(
                capability,
                rpc,
                chunk,
                disk_guard=disk_guard,
            )
            for event in events:
                if event.identity in global_identities:
                    raise ParityError("event identity repeats across chunks")
                global_identities.add(event.identity)
            rows.extend(events)
            completed += 1
            if progress is not None:
                progress(completed, TOTAL_STAGE_A_CHUNKS, monotonic() - started)
        streams[window.name] = tuple(rows)
    return streams


def require_window_support(
    streams: Mapping[str, Sequence[CanonicalEvent]],
) -> None:
    for window in WINDOWS:
        events = streams.get(window.name)
        if events is None:
            raise SupportError("fixed window stream is absent")
        support = {event.reserve for event in events}
        if support != set(RESERVE_ADDRESSES):
            raise SupportError("fixed reserve support failed")


def header_audit_block_hashes(
    streams: Mapping[str, Sequence[CanonicalEvent]],
) -> dict[int, str]:
    selected: list[CanonicalEvent] = []
    for window in WINDOWS:
        events = tuple(streams[window.name])
        selected.extend(events[255::256])
        for reserve in RESERVE_ADDRESSES:
            group = tuple(row for row in events if row.reserve == reserve)
            if not group:
                raise SupportError("fixed reserve support failed")
            selected.append(group[0])
            selected.append(group[-1])
    expected: dict[int, str] = {}
    for event in selected:
        previous = expected.get(event.block_number)
        if previous is not None and previous != event.block_hash:
            raise HeaderError("selected event block hashes conflict")
        expected[event.block_number] = event.block_hash
    return expected


def _verify_event_header_audit(
    capability: object,
    rpc: PairRpc,
    streams: Mapping[str, Sequence[CanonicalEvent]],
) -> None:
    _require_source_capability(capability)
    expected = header_audit_block_hashes(streams)
    previous_number: int | None = None
    previous_timestamp: int | None = None
    for block_number in sorted(expected):
        primary, parity = rpc.call_pair(
            "eth_getBlockByNumber",
            (canonical_quantity(block_number), False),
            100_000_000 + block_number,
        )
        header = _require_equal_headers(
            primary,
            parity,
            expected_number=block_number,
        )
        if header.block_hash != expected[block_number]:
            raise HeaderError("event block hash differs")
        if (
            previous_number is not None
            and block_number > previous_number
            and previous_timestamp is not None
            and header.timestamp <= previous_timestamp
        ):
            raise HeaderError("event header timestamps are non-monotone")
        previous_number = block_number
        previous_timestamp = header.timestamp


def _verify_source(
    capability: object,
    rpc: PairRpc,
    *,
    fetcher: Fetcher = fetch_https_bytes,
    disk_guard: DiskGuard = assert_disk_guard,
    progress: Progress | None = None,
) -> SourceEvidence:
    _require_source_capability(capability)
    rpc = HeaderCachingPairRpc(rpc)
    _verify_pins(fetcher)
    _verify_chain_and_boundaries(rpc)
    streams = _fetch_windows(
        capability,
        rpc,
        disk_guard=disk_guard,
        progress=progress,
    )
    require_window_support(streams)
    _verify_event_header_audit(capability, rpc, streams)
    return SourceEvidence(
        historical_sha256=window_stream_sha256(
            "historical", streams["historical"]
        ),
        recent_sha256=window_stream_sha256("recent", streams["recent"]),
    )


STAGE_A_GATE_NAMES = (
    "committed_protocol",
    "clean_head",
    "disk",
    "pinned_sources",
    "chain_identity",
    "boundary_headers",
    "subdivision_redundancy",
    "historical_window",
    "recent_window",
    "event_header_audit",
    "forbidden_access",
)
FORBIDDEN_ACCESS = {
    "market": False,
    "funding": False,
    "outcome": False,
    "return": False,
    "pnl": False,
    "reward": False,
    "model": False,
    "checkpoint": False,
    "cagr": False,
    "mdd": False,
}


def _range_payload() -> dict[str, Any]:
    def window_payload(window: WindowSpec) -> dict[str, Any]:
        return {
            "first_block": window.first_block,
            "first_block_hash": window.first_block_hash,
            "last_block": window.last_block,
            "last_block_hash": window.last_block_hash,
            "chunk_count": window.chunk_count,
        }

    return {
        "chunk_size": CHUNK_SIZE,
        "subdivision_size": SUBDIVISION_SIZE,
        "historical_window": window_payload(HISTORICAL_WINDOW),
        "recent_window": window_payload(RECENT_WINDOW),
        "full_history": {
            "first_block": FULL_HISTORY_FIRST_BLOCK,
            "first_block_hash": HISTORICAL_WINDOW.first_block_hash,
            "last_block": FULL_HISTORY_LAST_BLOCK,
            "last_block_hash": RECENT_WINDOW.last_block_hash,
            "chunk_count": FULL_HISTORY_CHUNK_COUNT,
        },
    }


def _pins_payload() -> dict[str, str]:
    pins = {pin.name: pin.sha256 for pin in PIN_SPECS}
    return {
        "address_book_sha256": pins["address_book"],
        "address_book_license_sha256": pins["address_book_license"],
        "aave_utilities_readme_sha256": pins["aave_utilities_readme"],
        "archived_ipool_sha256": pins["archived_ipool"],
        "current_ipool_sha256": pins["current_ipool"],
        "event_topic": EVENT_TOPIC,
    }


def _failure_metadata(
    error: BaseException,
) -> tuple[str, str, str]:
    if isinstance(error, DiskGuardError):
        return "preflight", "DiskGuardError", "disk guard failed"
    if isinstance(error, TransportError):
        return "source_read", "TransportError", "transport failed"
    if isinstance(error, SchemaError):
        return "canonicalization", "SchemaError", "schema validation failed"
    if isinstance(error, ParityError):
        return "canonicalization", "ParityError", "canonical parity failed"
    if isinstance(error, SupportError):
        return "support", "SupportError", "support gate failed"
    if isinstance(error, HeaderError):
        return "header_validation", "HeaderError", "header validation failed"
    if isinstance(error, PublicationError):
        return "publication", "PublicationError", "atomic publication failed"
    if isinstance(error, (KeyboardInterrupt, SystemExit, InterruptedError)):
        return "terminalization", "InterruptedError", "interrupted"
    if isinstance(error, ProtocolError):
        return "preflight", "ProtocolError", "protocol precondition failed"
    return "terminalization", "UnexpectedError", "unexpected failure"


def build_terminal_report(
    *,
    status: str,
    verifier_commit: str,
    started_at_utc: str,
    terminal_at_utc: str,
    gates: Mapping[str, bool],
    evidence: SourceEvidence | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if status not in {"PASS", "REJECT"}:
        raise ProtocolError("terminal status differs")
    if set(gates) != set(STAGE_A_GATE_NAMES):
        raise ProtocolError("terminal gate names differ")
    if any(type(value) is not bool for value in gates.values()):
        raise ProtocolError("terminal gate value differs")
    if not _COMMIT_RE.fullmatch(verifier_commit):
        raise ProtocolError("terminal verifier commit differs")
    if not _CANONICAL_UTC_RE.fullmatch(
        started_at_utc
    ) or not _CANONICAL_UTC_RE.fullmatch(terminal_at_utc):
        raise ProtocolError("terminal timestamp differs")
    if status == "PASS":
        if evidence is None or error is not None or not all(gates.values()):
            raise ProtocolError("pass report inputs differ")
        if not _LOWER_SHA256_RE.fullmatch(
            evidence.historical_sha256
        ) or not _LOWER_SHA256_RE.fullmatch(evidence.recent_sha256):
            raise ProtocolError("window stream hash differs")
        failure: dict[str, str] | None = None
        window_streams: dict[str, str] | None = {
            "historical_sha256": evidence.historical_sha256,
            "recent_sha256": evidence.recent_sha256,
        }
    else:
        if error is None:
            raise ProtocolError("reject report lacks failure")
        stage, exception_type, message = _failure_metadata(error)
        failure = {
            "stage": stage,
            "exception_type": exception_type,
            "message": message,
        }
        window_streams = None

    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "state": "TERMINAL_PASS" if status == "PASS" else "TERMINAL_REJECT",
        "source_boundary_sha256": BOUNDARY_SHA256,
        "verifier_commit": verifier_commit,
        "started_at_utc": started_at_utc,
        "terminal_at_utc": terminal_at_utc,
        "pins": _pins_payload(),
        "bindings": {
            "source_parity_report_path": None,
            "source_parity_report_sha256": None,
            "mechanism_document_path": None,
            "mechanism_document_sha256": None,
            "preregistration_document_path": None,
            "preregistration_document_sha256": None,
        },
        "range": _range_payload(),
        "providers": {
            "primary": PRIMARY_RPC,
            "parity": PARITY_RPC,
            "chain_id": EXPECTED_CHAIN_ID,
            "canonical_streams_equal": status == "PASS",
        },
        "schema_sha256": SCHEMA_SHA256,
        "window_streams": window_streams,
        "full_history_sha256": None,
        "ledger_gzip_sha256": None,
        "ledger_gzip_bytes": None,
        "gates": dict(gates),
        "failure": failure,
        "forbidden_access": dict(FORBIDDEN_ACCESS),
    }
    preimage = (
        b"AV3BRL-v1\0terminal-report\0" + canonical_json_bytes(report)
    )
    report["manifest_sha256"] = sha256_bytes(preimage)
    return report


def terminal_report_bytes(report: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(report)) + b"\n"


def run_stage_a() -> dict[str, Any]:
    sentinel_path = repository_artifact_path(DEFAULT_SENTINEL)
    temp_report_path = repository_artifact_path(DEFAULT_TEMP_REPORT)
    report_path = repository_artifact_path(DEFAULT_REPORT)
    verifier_commit = assert_protocol_committed()
    assert_atomic_publication_supported(report_path.parent)
    started_at_utc = canonical_utc(utc_now())
    reserve_one_shot(
        sentinel_path=sentinel_path,
        report_path=report_path,
        temp_report_path=temp_report_path,
        verifier_commit=verifier_commit,
        started_at_utc=started_at_utc,
    )
    gates = {name: False for name in STAGE_A_GATE_NAMES}
    gates["committed_protocol"] = True
    gates["clean_head"] = True
    gates["forbidden_access"] = True
    try:
        assert_disk_guard()
        gates["disk"] = True
        capability = _mint_source_capability(sentinel_path, verifier_commit)
        evidence = _verify_source(
            capability,
            DualRpc(),
            disk_guard=assert_disk_guard,
            progress=None,
        )
        for name in (
            "pinned_sources",
            "chain_identity",
            "boundary_headers",
            "subdivision_redundancy",
            "historical_window",
            "recent_window",
            "event_header_audit",
        ):
            gates[name] = True
        report = build_terminal_report(
            status="PASS",
            verifier_commit=verifier_commit,
            started_at_utc=started_at_utc,
            terminal_at_utc=canonical_utc(utc_now()),
            gates=gates,
            evidence=evidence,
        )
        atomic_publish_report(
            temporary=temp_report_path,
            destination=report_path,
            payload=terminal_report_bytes(report),
        )
        return report
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        for name in (
            "pinned_sources",
            "chain_identity",
            "boundary_headers",
            "subdivision_redundancy",
            "historical_window",
            "recent_window",
            "event_header_audit",
        ):
            gates[name] = False
        report = build_terminal_report(
            status="REJECT",
            verifier_commit=verifier_commit,
            started_at_utc=started_at_utc,
            terminal_at_utc=canonical_utc(utc_now()),
            gates=gates,
            error=error,
        )
        atomic_publish_report(
            temporary=temp_report_path,
            destination=report_path,
            payload=terminal_report_bytes(report),
        )
        return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-shot AV3BRL Stage A source-parity verifier."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    del args
    try:
        report = run_stage_a()
    except BaseException:
        print("terminal=NOT_STARTED_OR_ORPHANED", flush=True)
        return 2
    print(f"terminal={report['status']}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
