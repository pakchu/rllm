"""Build the frozen, source-only TRON mainnet USDT supply-event artifact.

The production replay is deliberately fail-closed and one-shot.  It reads only
TRON JSON-RPC chain data from two fixed independent transports; it never opens
market, policy, label, return, or performance data.  There is no retry,
fallback, checkpoint, resume, or response-dependent delay.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence, TypedDict, cast
import urllib.parse
import urllib.request


class FrozenHeaderPayload(TypedDict):
    number: int
    hash: str
    parentHash: str
    timestamp: int


class FrozenBoundary(TypedDict):
    utc: str
    number: int
    hash: str
    previous: FrozenHeaderPayload
    timestamp: int


PROTOCOL_VERSION = "tron_usdt_supply_events_source_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = Path("training/build_tron_usdt_supply_events.py")
TEST_PATH = Path("tests/test_build_tron_usdt_supply_events.py")
SOURCE_DECISION_PATH = Path(
    "docs/tron-usdt-supply-events-source-axis-decision-2026-07-30.md"
)
SOURCE_DECISION_SHA256 = (
    "ad742cc261b5dfa23bdc7cd730e3e6bd01c3d19687806fe9c65c995893ce3300"
)
MECHANISM_DECISION_PATH = Path(
    "docs/tron-usdt-supply-impulse-mechanism-decision-2026-07-30.md"
)
MECHANISM_DECISION_SHA256 = (
    "29fa76ca1fd1f86910d257e3d3bc05dcc76de0c5de2da88cd90a3930a5c65e98"
)
PREREGISTER_PATH = Path("training/preregister_tron_usdt_supply_impulse.py")
PREREGISTER_TEST_PATH = Path("tests/test_preregister_tron_usdt_supply_impulse.py")
PREREGISTRATION_ARTIFACT_PATH = Path(
    "results/tron_usdt_supply_impulse_preregistration_2026-07-30.json"
)
PREREGISTRATION_ARTIFACT_TEST_PATH = Path(
    "tests/test_tron_usdt_supply_impulse_preregistration_artifact.py"
)
PREREGISTRATION_ARTIFACT_SHA256 = (
    "54817044b8df76dc347ed64b6fe5f6f2dfdddcdb211bded4ba2b1af133d49067"
)
PREREGISTRATION_MANIFEST_HASH = (
    "d67cd1b67632ae92e9458395e729627a6f4c3b4b75ce97187653eac3a09e40c1"
)
SOURCE_SUPPORT_PATH = Path(
    "training/evaluate_tron_usdt_supply_impulse_source_support.py"
)
SOURCE_SUPPORT_TEST_PATH = Path(
    "tests/test_evaluate_tron_usdt_supply_impulse_source_support.py"
)
NOVELTY_PATH = Path("training/evaluate_tron_usdt_supply_impulse_novelty.py")
NOVELTY_TEST_PATH = Path(
    "tests/test_evaluate_tron_usdt_supply_impulse_novelty.py"
)
ECONOMICS_PATH = Path("training/evaluate_tron_usdt_supply_impulse_economics.py")
ECONOMICS_TEST_PATH = Path(
    "tests/test_evaluate_tron_usdt_supply_impulse_economics.py"
)
ESDI_PREREGISTER_HELPER_PATH = Path(
    "training/preregister_ethereum_settlement_demand_impulse.py"
)
ESDI_NOVELTY_HELPER_PATH = Path(
    "training/evaluate_ethereum_settlement_demand_impulse_novelty.py"
)
ESDI_SOURCE_BUILDER_HELPER_PATH = Path(
    "training/build_ethereum_settlement_demand_impulse_source.py"
)
ESDI_SOURCE_SUPPORT_HELPER_PATH = Path(
    "training/evaluate_ethereum_settlement_demand_impulse_source_support.py"
)
ESDI_ECONOMICS_HELPER_PATH = Path(
    "training/evaluate_ethereum_settlement_demand_impulse_economics.py"
)
ESDI_ECONOMICS_HELPER_SHA256 = (
    "fba7de6a26ede945edfe63c32dd4a0c88760c6459ac0d4f079dd12d546580235"
)
PROTOCOL_PATHS = (
    SOURCE_DECISION_PATH,
    MECHANISM_DECISION_PATH,
    PREREGISTER_PATH,
    PREREGISTER_TEST_PATH,
    PREREGISTRATION_ARTIFACT_PATH,
    PREREGISTRATION_ARTIFACT_TEST_PATH,
    BUILDER_PATH,
    TEST_PATH,
    SOURCE_SUPPORT_PATH,
    SOURCE_SUPPORT_TEST_PATH,
    NOVELTY_PATH,
    NOVELTY_TEST_PATH,
    ECONOMICS_PATH,
    ECONOMICS_TEST_PATH,
    ESDI_PREREGISTER_HELPER_PATH,
    ESDI_NOVELTY_HELPER_PATH,
    ESDI_SOURCE_BUILDER_HELPER_PATH,
    ESDI_SOURCE_SUPPORT_HELPER_PATH,
    ESDI_ECONOMICS_HELPER_PATH,
)
SYNTHETIC_PROTOCOL_PATHS = (BUILDER_PATH, TEST_PATH)
FROZEN_PROTOCOL_FILE_SHA256 = {
    SOURCE_DECISION_PATH.as_posix(): SOURCE_DECISION_SHA256,
    MECHANISM_DECISION_PATH.as_posix(): MECHANISM_DECISION_SHA256,
    PREREGISTRATION_ARTIFACT_PATH.as_posix(): (
        PREREGISTRATION_ARTIFACT_SHA256
    ),
    ESDI_ECONOMICS_HELPER_PATH.as_posix(): ESDI_ECONOMICS_HELPER_SHA256,
}

CHAIN_ID_HEX = "0x2b6653dc"
CHAIN_ID = int(CHAIN_ID_HEX, 16)
USDT_CONTRACT_BASE58 = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
USDT_CONTRACT = "0xa614f803b6fd780986a42c78ec9c7f77e6ded13c"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
ZERO_ADDRESS_TOPIC = "0x" + "0" * 64

ISSUE_TOPIC = "0xcb8241adb0c3fdb35b70c24ce35c5eb0c17af7431c99f827d44a445ca624176a"
REDEEM_TOPIC = "0x702d5967f45f6513a38ffc42d6ba9bf230bd40e8f53b16363c7eb4fd2deb9a44"
DESTROYED_BLACK_FUNDS_TOPIC = (
    "0x61e6e66b0d6339b2980aecc6ccc0039736791f0ccde9ed512e789a7fbdd698c6"
)
DEPRECATE_TOPIC = "0xcc358699805e9a8b7f77b522628c7cb9abd07d9efb86b6fb616af1609036a99e"
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
SEMANTIC_TOPICS = (
    ISSUE_TOPIC,
    REDEEM_TOPIC,
    DESTROYED_BLACK_FUNDS_TOPIC,
    DEPRECATE_TOPIC,
)

SOURCE_START_BLOCK = 47_313_358
SOURCE_START_UTC = "2023-01-01T00:00:00Z"
END_BOUNDARY_BLOCK = 83_201_056
END_BOUNDARY_UTC = "2026-06-01T00:00:00Z"
CONFIRMATION_BLOCKS = 64
LAST_EVENT_BLOCK = 83_200_991
SOURCE_END_BLOCK_EXCLUSIVE = LAST_EVENT_BLOCK + 1
LAST_SOURCE_BLOCK = LAST_EVENT_BLOCK
SOURCE_BLOCK_COUNT = SOURCE_END_BLOCK_EXCLUSIVE - SOURCE_START_BLOCK
LAST_CONFIRMATION_BLOCK = LAST_EVENT_BLOCK + CONFIRMATION_BLOCKS
CHUNK_SIZE = 5_000
EXPECTED_CHUNK_COUNT = 7_178
PRODUCTION_THROTTLE_SECONDS = 0.25
BOUNDARY_HEADER_SET_SHA256 = (
    "d2513baf86cab444b034ef19079e18515ee8da3d756f4dc041fb1e889707927b"
)
FROZEN_BOUNDARIES: tuple[FrozenBoundary, ...] = (
    {
        "utc": "2023-01-01T00:00:00Z",
        "number": 47_313_358,
        "hash": "0x0000000002d1f1ce5e430281e5308004cf19dd6e31afd4402b670fc05da5b340",
        "previous": {
            "number": 47_313_357,
            "hash": "0x0000000002d1f1cdd4da795414f43e7959c52951203ff7538d43a33db43675b6",
            "parentHash": "0x0000000002d1f1ccf20d5d3dc4f2dcac48678ac4483ddf83d02d7fd2f0097674",
            "timestamp": 1_672_531_197,
        },
        "timestamp": 1_672_531_200,
    },
    {
        "utc": "2023-06-01T00:00:00Z",
        "number": 51_652_374,
        "hash": "0x0000000003142716b7305d5d621414bc745837a849273ce4eab4b200c598af9d",
        "previous": {
            "number": 51_652_373,
            "hash": "0x0000000003142715803b39e9684b884bcd503fcf3065f3c397ad8462fca623b9",
            "parentHash": "0x00000000031427149d66ab8edef21e6199b21d1847e656f83778b45b564ad4b6",
            "timestamp": 1_685_577_597,
        },
        "timestamp": 1_685_577_600,
    },
    {
        "utc": "2024-01-01T00:00:00Z",
        "number": 57_811_194,
        "hash": "0x00000000037220fa937d59050fab5c3740ef10f5f7715b0f45035353878cd98f",
        "previous": {
            "number": 57_811_193,
            "hash": "0x00000000037220f977c1f549ddec3adc4da6799ad227c26896437d6ea915e1fb",
            "parentHash": "0x00000000037220f8b1f20c26996ba00d014c5c6dcefa3c5da076cbbcb985a5c4",
            "timestamp": 1_704_067_197,
        },
        "timestamp": 1_704_067_200,
    },
    {
        "utc": "2025-01-01T00:00:00Z",
        "number": 68_346_198,
        "hash": "0x000000000412e156401b47b5e85900fecd1744a7dd70e0ec6d7c9db7b5b7b8fd",
        "previous": {
            "number": 68_346_197,
            "hash": "0x000000000412e15527fe8e388e2f86934cc151befc2d07dce7155652e76f16d9",
            "parentHash": "0x000000000412e1541b1e1a88e1a652d74b9a17579774f615bb6b9f0184d7c10d",
            "timestamp": 1_735_689_597,
        },
        "timestamp": 1_735_689_600,
    },
    {
        "utc": "2026-01-01T00:00:00Z",
        "number": 78_854_231,
        "hash": "0x0000000004b338578474ba7a2a5fd3f2e19d303cb79f30d0b8e05ee361607b33",
        "previous": {
            "number": 78_854_230,
            "hash": "0x0000000004b338565ad850656f16a58a530c3b0aef16e3949e3b3c6c253473c3",
            "parentHash": "0x0000000004b338550d043da6919c3eb6bf6383b83a61434401312b9559e35ae4",
            "timestamp": 1_767_225_597,
        },
        "timestamp": 1_767_225_600,
    },
    {
        "utc": "2026-06-01T00:00:00Z",
        "number": 83_201_056,
        "hash": "0x0000000004f58c20deab323895309dd25eecc6bbbe4cd6c940713da2d78ca67a",
        "previous": {
            "number": 83_201_055,
            "hash": "0x0000000004f58c1f11b270096780f20b6de4b249ea8e3f8d3d95a5886ab9be22",
            "parentHash": "0x0000000004f58c1e556c1c20ee67eadf24b647af59248cb2cba57157a9f1fd71",
            "timestamp": 1_780_271_997,
        },
        "timestamp": 1_780_272_000,
    },
)

PRIMARY_RPC_ENV = "TRON_PRIMARY_RPC_URL"
VERIFY_RPC_ENV = "TRON_VERIFY_RPC_URL"
PRIMARY_HOST = "api.trongrid.io"
VERIFY_HOST = "tron-mainnet.core.chainstack.com"
TRANSPORT_ROLES = ("primary", "verification")
TRANSPORT_HOSTS = (PRIMARY_HOST, VERIFY_HOST)
TRANSPORT_MAX_BATCH = {"primary": 100, "verification": 30}
SANITIZED_TRANSPORTS = (
    {"role": "primary", "scheme": "https", "hostname": PRIMARY_HOST, "port": 443},
    {
        "role": "verification",
        "scheme": "https",
        "hostname": VERIFY_HOST,
        "port": 443,
    },
)
RPC_METHODS = frozenset(
    {
        "eth_chainId",
        "eth_getLogs",
        "eth_getBlockByNumber",
        "eth_getTransactionReceipt",
    }
)

SOURCE_OUTPUT_DIRECTORY = Path("data/tron_usdt_supply_events_2023_2026")
DEFAULT_CSV_OUTPUT = (
    SOURCE_OUTPUT_DIRECTORY / "tron_usdt_supply_events_2023_2026.csv.gz"
)
DEFAULT_MANIFEST_OUTPUT = Path(
    "results/tron_usdt_supply_events_source_manifest_2026-07-30.json"
)
REPLAY_CLAIM_PATH = Path(
    "results/tron_usdt_supply_events_source_replay_claim_2026-07-30.json"
)
GENERATION_STAGE_DIRECTORY = Path(
    "results/.tron_usdt_supply_events_source_generation_v1.stage"
)
STAGED_CSV_NAME = "source.csv.gz"
STAGED_MANIFEST_NAME = "manifest.json"
PRODUCTION_GENERATION_COMMIT = {
    "protocol": "manifest_last_v1",
    "mode": "production",
    "full_envelope_integrity": True,
    "canonical_publication_eligible": True,
    "manifest_is_commit_marker": True,
    "posix_multi_path_atomic": False,
}
SYNTHETIC_GENERATION_COMMIT = {
    "protocol": "computation_only_v1",
    "mode": "synthetic_nonproduction",
    "full_envelope_integrity": False,
    "canonical_publication_eligible": False,
    "manifest_is_commit_marker": False,
    "posix_multi_path_atomic": False,
}
_TRANSPORT_IDENTITY_KEYS = frozenset(
    {"role", "scheme", "hostname", "port"}
)

CATEGORY_SEMANTIC = "semantic"
CATEGORY_MINT = "mint_transfer"
CATEGORY_BURN = "burn_transfer"
CATEGORIES = (CATEGORY_SEMANTIC, CATEGORY_MINT, CATEGORY_BURN)

CSV_COLUMNS = (
    "event_type",
    "supply_direction",
    "actor_address",
    "amount_raw",
    "block_number",
    "block_hash",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "paired_transfer_log_index",
    "event_timestamp_utc",
    "confirmation_block",
    "confirmation_block_hash",
    "available_at_utc",
)

_QUANTITY_RE = re.compile(r"0x(?:0|[1-9a-f][0-9a-f]*)\Z")
_HASH_RE = re.compile(r"0x[0-9a-f]{64}\Z")
_ADDRESS_RE = re.compile(r"0x[0-9a-f]{40}\Z")
_DATA_RE = re.compile(r"0x(?:[0-9a-f]{2})*\Z")
_WORD_RE = re.compile(r"0x[0-9a-f]{64}\Z")
_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_LOG_FIELDS = frozenset(
    {
        "address",
        "topics",
        "data",
        "blockNumber",
        "transactionHash",
        "transactionIndex",
        "blockHash",
        "logIndex",
        "removed",
    }
)
ZERO_SOURCE_INTEGRITY = {
    "dual_log_differences": 0,
    "chunk_gaps": 0,
    "chunk_overlaps": 0,
    "missing_response_ids": 0,
    "receipt_differences": 0,
    "header_differences": 0,
    "Issue_pairing_differences": 0,
    "Issue_orphans": 0,
    "Redeem_pairing_differences": 0,
    "Redeem_orphans": 0,
    "Deprecate": 0,
    "failed_receipts": 0,
}


class TerminalSourceFailure(RuntimeError):
    """A terminal violation of the frozen source protocol."""


class RpcLike(Protocol):
    url: str
    role: str

    def call(self, method: str, params: Sequence[Any]) -> Any:
        """Issue exactly one JSON-RPC request."""
        ...

    def batch(
        self, requests: Sequence[tuple[str, Sequence[Any]]]
    ) -> Sequence[Any]:
        """Issue exactly one JSON-RPC batch request."""
        ...


@dataclass(frozen=True, order=True)
class CanonicalLog:
    block_number: int
    transaction_index: int
    log_index: int
    block_hash: str
    transaction_hash: str
    address: str
    topics: tuple[str, ...]
    data: str
    removed: bool

    @property
    def identity(self) -> tuple[str, int]:
        return (self.transaction_hash, self.log_index)


@dataclass(frozen=True)
class Header:
    number: int
    block_hash: str
    parent_hash: str
    timestamp: int


@dataclass(frozen=True)
class Receipt:
    transaction_hash: str
    transaction_index: int
    block_hash: str
    block_number: int
    status: int
    logs: tuple[CanonicalLog, ...]


@dataclass(frozen=True)
class BuildConfig:
    """Explicit opt-in for a computation-only injectable replay."""

    non_production: bool


@dataclass(frozen=True)
class ComputedBuild:
    """Unpublished bytes returned by the synthetic computation path."""

    rows: tuple[dict[str, Any], ...]
    csv_bytes: bytes
    manifest: dict[str, Any]
    manifest_bytes: bytes


@dataclass
class SourceIntegrityAudit:
    """Tracks executable validation stages required for zero-difference evidence."""

    completed: set[str] = field(default_factory=set)

    def mark(self, stage: str) -> None:
        self.completed.add(stage)

    def zero_evidence(self) -> dict[str, int]:
        required = {
            "chunk_geometry",
            "response_ids",
            "dual_logs",
            "semantic_pairing",
            "deprecate_absent",
            "headers",
            "receipts",
            "successful_receipts",
            "receipt_membership",
        }
        if self.completed != required:
            raise TerminalSourceFailure(
                "source integrity validation stages are incomplete"
            )
        return dict(ZERO_SOURCE_INTEGRITY)


def _canonical_json_bytes(payload: Any, *, trailing_lf: bool = False) -> bytes:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if trailing_lf else b"")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_transport_identities(
    transport_identities: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return the exact frozen, path-free transport identities."""

    try:
        identities = tuple(transport_identities)
    except TypeError:
        raise TerminalSourceFailure(
            "transport identities differ from the frozen schema"
        ) from None
    if len(identities) != len(SANITIZED_TRANSPORTS):
        raise TerminalSourceFailure(
            "transport identities differ from the frozen schema"
        )
    canonical: list[dict[str, Any]] = []
    for identity, expected in zip(
        identities, SANITIZED_TRANSPORTS, strict=True
    ):
        if (
            not isinstance(identity, Mapping)
            or set(identity) != _TRANSPORT_IDENTITY_KEYS
            or type(identity.get("role")) is not str
            or type(identity.get("scheme")) is not str
            or type(identity.get("hostname")) is not str
            or type(identity.get("port")) is not int
            or dict(identity) != expected
        ):
            raise TerminalSourceFailure(
                "transport identities differ from the frozen schema"
            )
        canonical.append(dict(expected))
    return tuple(canonical)


def _utc(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def frozen_boundary_header_payload() -> tuple[FrozenHeaderPayload, ...]:
    """Return the frozen previous/current headers in canonical boundary order."""

    payload: list[FrozenHeaderPayload] = []
    for boundary in FROZEN_BOUNDARIES:
        frozen_previous = boundary["previous"]
        previous: FrozenHeaderPayload = {
            "number": frozen_previous["number"],
            "hash": frozen_previous["hash"],
            "parentHash": frozen_previous["parentHash"],
            "timestamp": frozen_previous["timestamp"],
        }
        current: FrozenHeaderPayload = {
            "number": boundary["number"],
            "hash": boundary["hash"],
            "parentHash": previous["hash"],
            "timestamp": boundary["timestamp"],
        }
        payload.extend((previous, current))
    canonical = tuple(payload)
    if _sha256_bytes(_canonical_json_bytes(canonical)) != BOUNDARY_HEADER_SET_SHA256:
        raise AssertionError("frozen boundary header serialization hash differs")
    return canonical


def parse_quantity(value: Any, label: str = "quantity") -> int:
    if not isinstance(value, str) or _QUANTITY_RE.fullmatch(value) is None:
        raise TerminalSourceFailure(f"{label} is not a canonical quantity")
    return int(value, 16)


def _parse_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise TerminalSourceFailure(f"{label} is not a canonical 32-byte hash")
    return value


def _parse_address(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS_RE.fullmatch(value) is None:
        raise TerminalSourceFailure(f"{label} is not a canonical EVM address")
    return value


def _parse_data(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DATA_RE.fullmatch(value) is None:
        raise TerminalSourceFailure(f"{label} is not canonical hex bytes")
    return value


def iter_chunks(
    start_block: int = SOURCE_START_BLOCK,
    end_block_exclusive: int = SOURCE_END_BLOCK_EXCLUSIVE,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[tuple[int, int], ...]:
    """Return contiguous inclusive chunks covering [start, end_boundary)."""

    if (
        isinstance(start_block, bool)
        or isinstance(end_block_exclusive, bool)
        or isinstance(chunk_size, bool)
        or start_block < 0
        or end_block_exclusive <= start_block
        or chunk_size <= 0
    ):
        raise ValueError("invalid source range or chunk size")
    chunks = tuple(
        (first, min(first + chunk_size - 1, end_block_exclusive - 1))
        for first in range(start_block, end_block_exclusive, chunk_size)
    )
    expected = start_block
    for first, last in chunks:
        if first != expected or last < first or last - first + 1 > chunk_size:
            raise AssertionError("chunk construction is not contiguous")
        expected = last + 1
    if expected != end_block_exclusive:
        raise AssertionError("chunk construction does not cover the source range")
    return chunks


def frozen_chunks() -> tuple[tuple[int, int], ...]:
    chunks = iter_chunks()
    if (
        len(chunks) != EXPECTED_CHUNK_COUNT
        or any(last - first + 1 != CHUNK_SIZE for first, last in chunks[:-1])
        or chunks[-1] != (83_198_358, 83_200_991)
        or chunks[-1][1] - chunks[-1][0] + 1 != 2_634
    ):
        raise AssertionError("frozen chunk geometry differs")
    return chunks


def validate_replay_chunks(
    chunks: Sequence[tuple[int, int]],
    *,
    require_source_envelope: bool = False,
    require_frozen_envelope: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Validate contiguous inclusive chunk geometry without opening a source."""

    requested = tuple(chunks)
    if not requested:
        raise TerminalSourceFailure("source chunk list is empty")
    normalized: list[tuple[int, int]] = []
    expected_first: int | None = None
    for chunk in requested:
        if (
            not isinstance(chunk, tuple)
            or len(chunk) != 2
            or type(chunk[0]) is not int
            or type(chunk[1]) is not int
        ):
            raise TerminalSourceFailure("source chunk shape differs")
        first, last = chunk
        if (
            last < first
            or last - first + 1 > CHUNK_SIZE
            or (expected_first is not None and first != expected_first)
        ):
            raise TerminalSourceFailure(
                "source chunks have a gap, overlap, or invalid size"
            )
        normalized.append((first, last))
        expected_first = last + 1
    result = tuple(normalized)
    if require_source_envelope and (
        result[0][0] < SOURCE_START_BLOCK
        or result[-1][1] > LAST_EVENT_BLOCK
    ):
        raise TerminalSourceFailure("source chunks escape the frozen envelope")
    if require_frozen_envelope and result != iter_chunks(
        SOURCE_START_BLOCK, SOURCE_END_BLOCK_EXCLUSIVE, CHUNK_SIZE
    ):
        raise TerminalSourceFailure(
            "production replay chunks differ from the frozen envelope"
        )
    return result


def category_filter(category: str, first: int, last: int) -> dict[str, Any]:
    if category == CATEGORY_SEMANTIC:
        topics: list[Any] = [list(SEMANTIC_TOPICS)]
    elif category == CATEGORY_MINT:
        topics = [TRANSFER_TOPIC, ZERO_ADDRESS_TOPIC]
    elif category == CATEGORY_BURN:
        topics = [TRANSFER_TOPIC, None, ZERO_ADDRESS_TOPIC]
    else:
        raise ValueError(f"unknown source category {category!r}")
    return {
        "address": USDT_CONTRACT,
        "fromBlock": hex(first),
        "toBlock": hex(last),
        "topics": topics,
    }


def normalize_log(
    raw: Any,
    *,
    first_block: int | None = None,
    last_block: int | None = None,
    require_contract: bool = True,
) -> CanonicalLog:
    """Strictly canonicalize the exact raw JSON-RPC log fields."""

    if not isinstance(raw, dict) or set(raw) != _LOG_FIELDS:
        raise TerminalSourceFailure("raw log fields differ from the frozen schema")
    address = _parse_address(raw["address"], "log address")
    if require_contract and address != USDT_CONTRACT:
        raise TerminalSourceFailure("log address differs from frozen USDT contract")
    topics_raw = raw["topics"]
    if not isinstance(topics_raw, list) or not topics_raw:
        raise TerminalSourceFailure("log topics are absent or malformed")
    topics = tuple(_parse_hash(topic, "log topic") for topic in topics_raw)
    data = _parse_data(raw["data"], "log data")
    block_number = parse_quantity(raw["blockNumber"], "log block number")
    transaction_index = parse_quantity(
        raw["transactionIndex"], "log transaction index"
    )
    log_index = parse_quantity(raw["logIndex"], "log index")
    if first_block is not None and block_number < first_block:
        raise TerminalSourceFailure("log precedes requested range")
    if last_block is not None and block_number > last_block:
        raise TerminalSourceFailure("log follows requested range")
    if raw["removed"] is not False:
        raise TerminalSourceFailure("removed log is forbidden")
    return CanonicalLog(
        block_number=block_number,
        transaction_index=transaction_index,
        log_index=log_index,
        block_hash=_parse_hash(raw["blockHash"], "log block hash"),
        transaction_hash=_parse_hash(
            raw["transactionHash"], "log transaction hash"
        ),
        address=address,
        topics=topics,
        data=data,
        removed=False,
    )


def canonicalize_logs(
    raw_logs: Any,
    *,
    first_block: int | None = None,
    last_block: int | None = None,
    require_contract: bool = True,
) -> tuple[CanonicalLog, ...]:
    if not isinstance(raw_logs, list):
        raise TerminalSourceFailure("RPC log result is not a list")
    logs = tuple(
        normalize_log(
            raw,
            first_block=first_block,
            last_block=last_block,
            require_contract=require_contract,
        )
        for raw in raw_logs
    )
    identities = [log.identity for log in logs]
    if len(set(identities)) != len(identities):
        raise TerminalSourceFailure("duplicate canonical log identity")
    return tuple(sorted(logs))


def canonical_log_payload(log: CanonicalLog) -> dict[str, Any]:
    return {
        "address": log.address,
        "topics": list(log.topics),
        "data": log.data,
        "block_number": log.block_number,
        "transaction_hash": log.transaction_hash,
        "transaction_index": log.transaction_index,
        "block_hash": log.block_hash,
        "log_index": log.log_index,
        "removed": log.removed,
    }


def canonical_log_hash(logs: Sequence[CanonicalLog]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes([canonical_log_payload(log) for log in sorted(logs)])
    )


def compare_log_sets(
    left_raw: Any,
    right_raw: Any,
    *,
    first_block: int,
    last_block: int,
) -> tuple[CanonicalLog, ...]:
    left = canonicalize_logs(
        left_raw, first_block=first_block, last_block=last_block
    )
    right = canonicalize_logs(
        right_raw, first_block=first_block, last_block=last_block
    )
    if left != right:
        raise TerminalSourceFailure("provider canonical log sets disagree")
    return left


def _decode_word(data: str, label: str) -> int:
    if _WORD_RE.fullmatch(data) is None:
        raise TerminalSourceFailure(f"{label} is not one canonical ABI word")
    return int(data, 16)


def _decode_actor_topic(topic: str, label: str) -> str:
    if _WORD_RE.fullmatch(topic) is None or topic[2:26] != "0" * 24:
        raise TerminalSourceFailure(f"{label} is not an exact left-padded address")
    return "0x" + topic[-40:]


def _decode_actor_word(word_value: str, label: str) -> str:
    if len(word_value) != 64 or word_value[:24] != "0" * 24:
        raise TerminalSourceFailure(f"{label} is not an exact left-padded address")
    return "0x" + word_value[-40:]


def _positive_amount(value: int, label: str) -> int:
    if value <= 0:
        raise TerminalSourceFailure(f"{label} must be positive")
    return value


def _validate_semantic_log(log: CanonicalLog) -> tuple[str, int, str]:
    topic0 = log.topics[0]
    if topic0 == ISSUE_TOPIC:
        if len(log.topics) != 1:
            raise TerminalSourceFailure("Issue topic shape differs")
        return (
            "Issue",
            _positive_amount(_decode_word(log.data, "Issue amount"), "Issue amount"),
            "",
        )
    if topic0 == REDEEM_TOPIC:
        if len(log.topics) != 1:
            raise TerminalSourceFailure("Redeem topic shape differs")
        return (
            "Redeem",
            _positive_amount(
                _decode_word(log.data, "Redeem amount"), "Redeem amount"
            ),
            "",
        )
    if topic0 == DESTROYED_BLACK_FUNDS_TOPIC:
        if len(log.topics) != 1:
            raise TerminalSourceFailure("DestroyedBlackFunds topic shape differs")
        if not isinstance(log.data, str) or re.fullmatch(
            r"0x[0-9a-f]{128}", log.data
        ) is None:
            raise TerminalSourceFailure(
                "DestroyedBlackFunds data is not two ABI words"
            )
        first_word, second_word = log.data[2:66], log.data[66:130]
        return (
            "DestroyedBlackFunds",
            _positive_amount(
                int(second_word, 16), "DestroyedBlackFunds amount"
            ),
            _decode_actor_word(first_word, "DestroyedBlackFunds actor"),
        )
    if topic0 == DEPRECATE_TOPIC:
        if len(log.topics) != 1 or _WORD_RE.fullmatch(log.data) is None:
            raise TerminalSourceFailure("Deprecate shape differs")
        _decode_actor_word(log.data[2:], "Deprecate replacement")
        raise TerminalSourceFailure("Deprecate event is terminal")
    raise TerminalSourceFailure("unfrozen semantic event topic")


def _validate_transfer(log: CanonicalLog, *, mint: bool) -> tuple[int, str]:
    if len(log.topics) != 3 or log.topics[0] != TRANSFER_TOPIC:
        raise TerminalSourceFailure("Transfer topic shape differs")
    sender = _decode_actor_topic(log.topics[1], "Transfer sender")
    recipient = _decode_actor_topic(log.topics[2], "Transfer recipient")
    if mint:
        if log.topics[1] != ZERO_ADDRESS_TOPIC:
            raise TerminalSourceFailure("mint Transfer does not have zero sender")
        if log.topics[2] == ZERO_ADDRESS_TOPIC:
            raise TerminalSourceFailure("zero-to-zero Transfer is forbidden")
    else:
        if log.topics[2] != ZERO_ADDRESS_TOPIC:
            raise TerminalSourceFailure("burn Transfer does not have zero recipient")
        if log.topics[1] == ZERO_ADDRESS_TOPIC:
            raise TerminalSourceFailure("zero-to-zero Transfer is forbidden")
    amount = _positive_amount(
        _decode_word(log.data, "Transfer amount"), "Transfer amount"
    )
    return amount, recipient if mint else sender


def pair_supply_events(
    semantic_logs: Sequence[CanonicalLog],
    mint_logs: Sequence[CanonicalLog],
    burn_logs: Sequence[CanonicalLog],
    *,
    audit: SourceIntegrityAudit | None = None,
) -> tuple[dict[str, Any], ...]:
    """Pair Issue/Redeem exactly once and retain blacklist destruction separately."""

    all_logs = tuple(semantic_logs) + tuple(mint_logs) + tuple(burn_logs)
    identities = [log.identity for log in all_logs]
    if len(set(identities)) != len(identities):
        raise TerminalSourceFailure("source categories overlap or duplicate globally")

    mints: dict[tuple[int, str, int], tuple[CanonicalLog, str]] = {}
    burns: dict[tuple[int, str, int], tuple[CanonicalLog, str]] = {}
    for collection, logs, is_mint in (
        (mints, mint_logs, True),
        (burns, burn_logs, False),
    ):
        for log in logs:
            amount, actor = _validate_transfer(log, mint=is_mint)
            key = (log.block_number, log.transaction_hash, amount)
            if key in collection:
                raise TerminalSourceFailure("duplicate candidate Transfer pair")
            collection[key] = (log, actor)

    rows: list[dict[str, Any]] = []
    for log in sorted(semantic_logs):
        if log.block_number > LAST_EVENT_BLOCK:
            raise TerminalSourceFailure(
                "semantic event exceeds frozen last event block"
            )
        event_type, amount, actor_address = _validate_semantic_log(log)
        paired: CanonicalLog | None = None
        if event_type == "Issue":
            candidate = mints.pop(
                (log.block_number, log.transaction_hash, amount), None
            )
            if candidate is not None:
                paired, actor_address = candidate
            direction = 1
        elif event_type == "Redeem":
            candidate = burns.pop(
                (log.block_number, log.transaction_hash, amount), None
            )
            if candidate is not None:
                paired, actor_address = candidate
            direction = -1
        else:
            direction = -1
        if event_type in {"Issue", "Redeem"} and paired is None:
            raise TerminalSourceFailure(f"{event_type} has no unique Transfer pair")
        if paired is not None and (
            paired.block_hash != log.block_hash
            or paired.transaction_index != log.transaction_index
        ):
            raise TerminalSourceFailure("paired Transfer location differs")
        rows.append(
            {
                "event_type": event_type,
                "supply_direction": direction,
                "actor_address": actor_address,
                "amount_raw": str(amount),
                "block_number": log.block_number,
                "block_hash": log.block_hash,
                "transaction_hash": log.transaction_hash,
                "transaction_index": log.transaction_index,
                "log_index": log.log_index,
                "paired_transfer_log_index": (
                    paired.log_index if paired is not None else ""
                ),
                "_semantic_log": log,
                "_paired_log": paired,
            }
        )
    if mints or burns:
        raise TerminalSourceFailure("unmatched mint or burn Transfer")
    if audit is not None:
        audit.mark("semantic_pairing")
        audit.mark("deprecate_absent")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row["block_number"],
                row["transaction_index"],
                row["log_index"],
            ),
        )
    )


def normalize_header(raw: Any, *, expected_number: int | None = None) -> Header:
    if not isinstance(raw, dict):
        raise TerminalSourceFailure("block header is not an object")
    required = {"number", "hash", "parentHash", "timestamp"}
    if not required.issubset(raw):
        raise TerminalSourceFailure("block header fields are missing")
    number = parse_quantity(raw["number"], "block number")
    if expected_number is not None and number != expected_number:
        raise TerminalSourceFailure("block header number differs")
    return Header(
        number=number,
        block_hash=_parse_hash(raw["hash"], "block hash"),
        parent_hash=_parse_hash(raw["parentHash"], "parent hash"),
        timestamp=parse_quantity(raw["timestamp"], "block timestamp"),
    )


def normalize_receipt(raw: Any, *, expected_hash: str | None = None) -> Receipt:
    if not isinstance(raw, dict):
        raise TerminalSourceFailure("transaction receipt is missing or malformed")
    required = {
        "transactionHash",
        "transactionIndex",
        "blockHash",
        "blockNumber",
        "status",
        "logs",
    }
    if not required.issubset(raw):
        raise TerminalSourceFailure("transaction receipt fields are missing")
    transaction_hash = _parse_hash(
        raw["transactionHash"], "receipt transaction hash"
    )
    if expected_hash is not None and transaction_hash != expected_hash:
        raise TerminalSourceFailure("receipt transaction hash differs")
    logs = canonicalize_logs(raw["logs"], require_contract=False)
    receipt = Receipt(
        transaction_hash=transaction_hash,
        transaction_index=parse_quantity(
            raw["transactionIndex"], "receipt transaction index"
        ),
        block_hash=_parse_hash(raw["blockHash"], "receipt block hash"),
        block_number=parse_quantity(raw["blockNumber"], "receipt block number"),
        status=parse_quantity(raw["status"], "receipt status"),
        logs=logs,
    )
    if receipt.status != 1:
        raise TerminalSourceFailure("event transaction receipt is not successful")
    for log in logs:
        if (
            log.transaction_hash != receipt.transaction_hash
            or log.transaction_index != receipt.transaction_index
            or log.block_hash != receipt.block_hash
            or log.block_number != receipt.block_number
        ):
            raise TerminalSourceFailure("receipt log location differs from receipt")
    return receipt


def materialize_events(
    paired_rows: Sequence[Mapping[str, Any]],
    headers: Mapping[int, Header],
    receipts: Mapping[str, Receipt],
    *,
    audit: SourceIntegrityAudit | None = None,
) -> tuple[dict[str, Any], ...]:
    expected_by_transaction: dict[str, set[CanonicalLog]] = {}
    for source_row in paired_rows:
        expected = expected_by_transaction.setdefault(
            str(source_row["transaction_hash"]), set()
        )
        expected.add(source_row["_semantic_log"])
        paired_log = source_row["_paired_log"]
        if paired_log is not None:
            expected.add(paired_log)
    for transaction_hash, expected in expected_by_transaction.items():
        receipt = receipts.get(transaction_hash)
        if receipt is None:
            raise TerminalSourceFailure("event receipt is absent")
        relevant = {
            log
            for log in receipt.logs
            if log.address == USDT_CONTRACT
            and (
                log.topics[0] in SEMANTIC_TOPICS
                or log.topics[0] == TRANSFER_TOPIC
                and (
                    len(log.topics) != 3
                    or log.topics[1] == ZERO_ADDRESS_TOPIC
                    or log.topics[2] == ZERO_ADDRESS_TOPIC
                )
            )
        }
        if relevant != expected:
            raise TerminalSourceFailure(
                "receipt semantic/zero-Transfer set differs from exact pairing"
            )

    materialized: list[dict[str, Any]] = []
    for source_row in paired_rows:
        row = dict(source_row)
        event_number = row["block_number"]
        confirmation_number = event_number + CONFIRMATION_BLOCKS
        event_header = headers.get(event_number)
        confirmation_header = headers.get(confirmation_number)
        if event_header is None or confirmation_header is None:
            raise TerminalSourceFailure("event or confirmation header is absent")
        if event_header.block_hash != row["block_hash"]:
            raise TerminalSourceFailure("event block hash differs from header")
        if confirmation_header.timestamp <= event_header.timestamp:
            raise TerminalSourceFailure("confirmation timestamp is not later")
        if (
            event_header.timestamp < FROZEN_BOUNDARIES[0]["timestamp"]
            or confirmation_header.timestamp
            >= FROZEN_BOUNDARIES[-1]["timestamp"]
        ):
            raise TerminalSourceFailure("event causal timestamps escape UTC envelope")
        for number, header in (
            (event_number, event_header),
            (confirmation_number, confirmation_header),
        ):
            previous = headers.get(number - 1)
            if previous is None or header.parent_hash != previous.block_hash:
                raise TerminalSourceFailure("header parent relation is unproven")
            if previous.timestamp > header.timestamp:
                raise TerminalSourceFailure("header timestamps decrease")

        receipt = receipts.get(row["transaction_hash"])
        if receipt is None:
            raise TerminalSourceFailure("event receipt is absent")
        if (
            receipt.block_number != event_number
            or receipt.block_hash != row["block_hash"]
            or receipt.transaction_index != row["transaction_index"]
        ):
            raise TerminalSourceFailure("event receipt location differs")
        receipt_logs = set(receipt.logs)
        semantic_log = row.pop("_semantic_log")
        paired_log = row.pop("_paired_log")
        if semantic_log not in receipt_logs:
            raise TerminalSourceFailure("semantic log is absent from exact receipt")
        if paired_log is not None and paired_log not in receipt_logs:
            raise TerminalSourceFailure("paired Transfer is absent from exact receipt")
        row.update(
            {
                "event_timestamp_utc": _utc(event_header.timestamp),
                "confirmation_block": confirmation_number,
                "confirmation_block_hash": confirmation_header.block_hash,
                "available_at_utc": _utc(confirmation_header.timestamp),
            }
        )
        materialized.append(row)
    for previous, current in zip(materialized, materialized[1:]):
        if (
            current["event_timestamp_utc"] < previous["event_timestamp_utc"]
            or current["available_at_utc"] < previous["available_at_utc"]
        ):
            raise TerminalSourceFailure(
                "source event or causal availability timestamps decrease"
            )
    if audit is not None:
        audit.mark("receipt_membership")
    return tuple(materialized)


def _deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer, mode="wb", filename="", compresslevel=9, mtime=0
    ) as stream:
        stream.write(payload)
    return buffer.getvalue()


def serialize_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer: csv.DictWriter[str] = csv.DictWriter(
        text, fieldnames=CSV_COLUMNS, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return _deterministic_gzip(text.getvalue().encode("utf-8"))


def canonical_header_hash(headers: Mapping[int, Header]) -> str:
    payload = [
        {
            "number": header.number,
            "hash": header.block_hash,
            "parentHash": header.parent_hash,
            "timestamp": header.timestamp,
        }
        for _, header in sorted(headers.items())
    ]
    return _sha256_bytes(_canonical_json_bytes(payload))


def canonical_receipt_hash(receipts: Mapping[str, Receipt]) -> str:
    payload = [
        {
            "transactionHash": receipt.transaction_hash,
            "transactionIndex": receipt.transaction_index,
            "blockHash": receipt.block_hash,
            "blockNumber": receipt.block_number,
            "status": receipt.status,
            "logs": [
                canonical_log_payload(log) for log in sorted(receipt.logs)
            ],
        }
        for _, receipt in sorted(receipts.items())
    ]
    return _sha256_bytes(_canonical_json_bytes(payload))


def _source_range_manifest(
    replay_chunks: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    chunks = validate_replay_chunks(
        replay_chunks,
        require_source_envelope=True,
    )
    first_block = chunks[0][0]
    last_block = chunks[-1][1]
    final_first, final_last = chunks[-1]
    return {
        "start_block_inclusive": first_block,
        "start_utc": SOURCE_START_UTC if first_block == SOURCE_START_BLOCK else None,
        "end_block_exclusive": last_block + 1,
        "last_source_block": last_block,
        "block_count": last_block - first_block + 1,
        "chunk_size_inclusive": CHUNK_SIZE,
        "chunk_count": len(chunks),
        "full_chunk_count": sum(
            last - first + 1 == CHUNK_SIZE for first, last in chunks
        ),
        "final_chunk": {
            "first_block": final_first,
            "last_block": final_last,
            "block_count": final_last - final_first + 1,
        },
        "confirmation_blocks": CONFIRMATION_BLOCKS,
        "maximum_admissible_event_block": LAST_EVENT_BLOCK,
        "last_confirmation_block": LAST_CONFIRMATION_BLOCK,
        "causal_end_boundary": {
            "block": END_BOUNDARY_BLOCK,
            "utc": END_BOUNDARY_UTC,
        },
    }


def build_manifest(
    rows: Sequence[Mapping[str, Any]],
    csv_bytes: bytes,
    *,
    category_logs: Mapping[str, Sequence[CanonicalLog]],
    replay_chunks: Sequence[tuple[int, int]],
    finalized_head: Header,
    boundary_evidence: Mapping[str, Any],
    source_integrity: Mapping[str, int],
    headers: Mapping[int, Header],
    receipts: Mapping[str, Receipt],
    protocol_seal: Mapping[str, Any] | None = None,
    claim_binding: Mapping[str, Any] | None = None,
    transport_identities: Sequence[Mapping[str, Any]] = SANITIZED_TRANSPORTS,
    production: bool = False,
) -> dict[str, Any]:
    identities = validate_transport_identities(transport_identities)
    chunks = validate_replay_chunks(
        replay_chunks,
        require_source_envelope=True,
        require_frozen_envelope=production,
    )
    if dict(source_integrity) != ZERO_SOURCE_INTEGRITY:
        raise TerminalSourceFailure(
            "source integrity evidence is not exact zero-difference"
        )
    if set(category_logs) != set(CATEGORIES):
        raise TerminalSourceFailure("source log categories differ")
    if finalized_head.number < LAST_CONFIRMATION_BLOCK:
        raise TerminalSourceFailure(
            "common finalized head is below last confirmation"
        )
    if production and (
        protocol_seal is None
        or claim_binding is None
        or boundary_evidence.get("header_count") != 12
        or boundary_evidence.get("canonical_header_set_sha256")
        != BOUNDARY_HEADER_SET_SHA256
        or boundary_evidence.get("frozen_header_set_exact") is not True
        or boundary_evidence.get("outside_before_count") != 0
        or boundary_evidence.get(
            "outside_after_maximum_admissible_count"
        )
        != 0
    ):
        raise TerminalSourceFailure(
            "production manifest lacks frozen replay evidence"
        )
    if not production and claim_binding is not None:
        raise TerminalSourceFailure(
            "non-production computation may not bind a replay claim"
        )
    global_logs = tuple(
        log for category in CATEGORIES for log in category_logs[category]
    )
    protocol_parent_commit = (
        protocol_seal.get("git_head") if protocol_seal is not None else None
    )
    replay_claim_commit = (
        claim_binding.get("claim_commit") if claim_binding is not None else None
    )
    replay_claim_sha256 = (
        claim_binding.get("sha256") if claim_binding is not None else None
    )
    if claim_binding is not None and (
        protocol_parent_commit != claim_binding.get("protocol_parent_commit")
        or _GIT_OBJECT_RE.fullmatch(str(replay_claim_commit)) is None
        or _SHA256_RE.fullmatch(str(replay_claim_sha256)) is None
    ):
        raise TerminalSourceFailure("manifest replay claim binding differs")
    year_counts: dict[str, int] = {
        "2023": 0,
        "2024": 0,
        "2025": 0,
        "2026": 0,
    }
    for row in rows:
        timestamp = row.get("event_timestamp_utc")
        if (
            not isinstance(timestamp, str)
            or re.fullmatch(
                (
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
                    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
                ),
                timestamp,
            )
            is None
        ):
            raise TerminalSourceFailure("event timestamp is not canonical UTC")
        year = timestamp[:4]
        if year not in year_counts:
            raise TerminalSourceFailure("event timestamp year is outside source")
        year_counts[year] += 1
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "source_only": True,
        "protocol_parent_commit": protocol_parent_commit,
        "replay_claim_commit": replay_claim_commit,
        "replay_claim_sha256": replay_claim_sha256,
        "generation_commit": dict(
            PRODUCTION_GENERATION_COMMIT
            if production
            else SYNTHETIC_GENERATION_COMMIT
        ),
        "chain": {
            "name": "TRON mainnet",
            "chain_id": CHAIN_ID_HEX,
            "usdt_contract_base58": USDT_CONTRACT_BASE58,
            "usdt_contract_evm": USDT_CONTRACT,
        },
        "source_range": _source_range_manifest(chunks),
        "transports": [dict(identity) for identity in identities],
        "source_replay_schedule": {
            "inter_batch_throttle_seconds": PRODUCTION_THROTTLE_SECONDS,
            "maximum_batch_by_role": dict(TRANSPORT_MAX_BATCH),
            "rpc_methods": sorted(RPC_METHODS),
        },
        "transport_exact_set_equal": True,
        "category_counts": {
            category: len(category_logs[category]) for category in CATEGORIES
        },
        "category_canonical_sha256": {
            category: canonical_log_hash(category_logs[category])
            for category in CATEGORIES
        },
        "global_log_count": len(global_logs),
        "global_canonical_sha256": canonical_log_hash(global_logs),
        "event_counts": {
            event_type: sum(row["event_type"] == event_type for row in rows)
            for event_type in (
                "Issue",
                "Redeem",
                "DestroyedBlackFunds",
            )
        },
        "event_count": len(rows),
        "event_canonical_sha256": _sha256_bytes(
            _canonical_json_bytes(list(rows))
        ),
        "year_counts": year_counts,
        "source_csv_sha256": _sha256_bytes(csv_bytes),
        "receipt_count": len(receipts),
        "receipt_canonical_sha256": canonical_receipt_hash(receipts),
        "header_count": len(headers),
        "header_canonical_sha256": canonical_header_hash(headers),
        "common_finalized_head": {
            "number": finalized_head.number,
            "hash": finalized_head.block_hash,
            "timestamp_utc": _utc(finalized_head.timestamp),
            "covers_last_confirmation": True,
        },
        "boundary_evidence": dict(boundary_evidence),
        "protocol_guards": {
            "retry_backoff_fallback_resume": False,
            "response_dependent_sleep": False,
            "deprecate_terminal": True,
            "market_policy_performance_opened": False,
        },
        "outcome_access": {
            "btc_market_rows_opened": 0,
            "funding_rows_opened": 0,
            "returns_opened": 0,
            "pnl_opened": 0,
            "cagr_opened": 0,
            "strict_mdd_opened": 0,
            "outcomes_opened": 0,
        },
        "source_integrity": dict(source_integrity),
    }
    return {**core, "manifest_hash": _sha256_bytes(_canonical_json_bytes(core))}


def serialize_manifest(manifest: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _safe_output_path(path: str | Path, allowed_root: str | Path) -> Path:
    candidate = Path(path)
    root_candidate = Path(allowed_root)
    if not root_candidate.is_absolute():
        root_candidate = REPOSITORY_ROOT / root_candidate
    if root_candidate.is_symlink():
        raise TerminalSourceFailure("allowed output root may not be a symlink")
    root_candidate.mkdir(parents=True, exist_ok=True)
    root = root_candidate.resolve()
    absolute = candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise TerminalSourceFailure("output path escapes the allowed root") from exc
    if ".." in relative.parts:
        raise TerminalSourceFailure("output path escapes the allowed root")
    current = root
    for part in relative.parent.parts:
        current /= part
        if current.is_symlink():
            raise TerminalSourceFailure("output path contains a symlink")
        if current.exists():
            if not current.is_dir():
                raise TerminalSourceFailure(
                    "output path parent is not a directory"
                )
        else:
            current.mkdir()
    resolved = current / relative.name
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"write-once output already exists: {resolved}")
    return resolved


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_outputs(
    payloads: Mapping[str | Path, bytes], *, allowed_root: str | Path
) -> tuple[Path, ...]:
    """Publish write-once files; multi-path calls have rollback, not POSIX atomicity."""

    canonical_targets = {
        (REPOSITORY_ROOT / relative).absolute()
        for relative in (DEFAULT_CSV_OUTPUT, DEFAULT_MANIFEST_OUTPUT)
    }
    requested_targets = {
        (
            path.absolute()
            if path.is_absolute()
            else (REPOSITORY_ROOT / path).absolute()
        )
        for path in (Path(raw_path) for raw_path in payloads)
    }
    if requested_targets & canonical_targets:
        raise TerminalSourceFailure(
            "canonical generation publication requires production replay"
        )
    finals = tuple(_safe_output_path(path, allowed_root) for path in payloads)
    if len(set(finals)) != len(finals):
        raise TerminalSourceFailure("output paths must be distinct")
    temporaries: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for final, payload in zip(finals, payloads.values(), strict=True):
            handle = tempfile.NamedTemporaryFile(
                prefix=f".{final.name}.",
                suffix=".tmp",
                dir=final.parent,
                delete=False,
            )
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            temporaries[final] = temporary
        for final, temporary in temporaries.items():
            os.link(temporary, final)
            published.append(final)
        for directory in {path.parent for path in finals}:
            _fsync_directory(directory)
    except Exception:
        for final in reversed(published):
            final.unlink(missing_ok=True)
            _fsync_directory(final.parent)
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
    return finals


def _repository_path(
    relative: Path,
    *,
    repository_root: str | Path,
) -> tuple[Path, Path]:
    root_candidate = Path(repository_root)
    if root_candidate.is_symlink():
        raise TerminalSourceFailure("repository root may not be a symlink")
    if not root_candidate.is_dir():
        raise TerminalSourceFailure("repository root is absent")
    root = root_candidate.resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise TerminalSourceFailure("canonical generation path is unsafe")
    current = root
    for part in relative.parent.parts:
        current /= part
        if current.is_symlink():
            raise TerminalSourceFailure(
                "canonical generation path contains a symlink"
            )
        if current.exists() and not current.is_dir():
            raise TerminalSourceFailure(
                "canonical generation parent is not a directory"
            )
    candidate = root / relative
    if candidate.is_symlink():
        raise TerminalSourceFailure(
            "canonical generation path may not be a symlink"
        )
    return root, candidate


def _generation_paths(
    *, repository_root: str | Path
) -> tuple[Path, Path, Path, Path, Path, Path]:
    root, csv_path = _repository_path(
        DEFAULT_CSV_OUTPUT, repository_root=repository_root
    )
    _, manifest_path = _repository_path(
        DEFAULT_MANIFEST_OUTPUT, repository_root=root
    )
    _, stage_directory = _repository_path(
        GENERATION_STAGE_DIRECTORY, repository_root=root
    )
    return (
        root,
        csv_path,
        manifest_path,
        stage_directory,
        stage_directory / STAGED_CSV_NAME,
        stage_directory / STAGED_MANIFEST_NAME,
    )


def _ensure_safe_directory(
    directory: Path,
    *,
    root: Path,
    created: list[Path],
) -> None:
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise TerminalSourceFailure(
            "canonical generation directory escapes repository"
        ) from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise TerminalSourceFailure(
                "canonical generation directory contains a symlink"
            )
        if current.exists():
            if not current.is_dir():
                raise TerminalSourceFailure(
                    "canonical generation parent is not a directory"
                )
            continue
        current.mkdir()
        created.append(current)
        _fsync_directory(current.parent)


def _durable_create_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("durable staged write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _unlink_file_durable(path: Path) -> None:
    path.unlink(missing_ok=True)
    if path.parent.is_dir():
        _fsync_directory(path.parent)


def _cleanup_stage_directory(stage_directory: Path) -> None:
    if not stage_directory.exists():
        return
    if stage_directory.is_symlink() or not stage_directory.is_dir():
        raise TerminalSourceFailure("generation stage directory is unsafe")
    expected = {
        stage_directory / STAGED_CSV_NAME,
        stage_directory / STAGED_MANIFEST_NAME,
    }
    children = set(stage_directory.iterdir())
    if not children.issubset(expected):
        raise TerminalSourceFailure("generation stage contains unknown files")
    for child in sorted(children):
        if child.is_symlink() or not child.is_file():
            raise TerminalSourceFailure("generation stage file is unsafe")
        child.unlink()
    _fsync_directory(stage_directory)
    stage_directory.rmdir()
    _fsync_directory(stage_directory.parent)


def _rollback_created_directories(created: Sequence[Path]) -> None:
    for directory in reversed(created):
        try:
            directory.rmdir()
        except OSError:
            continue
        _fsync_directory(directory.parent)


def _decode_canonical_manifest(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalSourceFailure("source manifest is invalid") from exc
    if not isinstance(payload, dict) or raw != serialize_manifest(payload):
        raise TerminalSourceFailure("source manifest bytes are not canonical")
    manifest_hash = payload.get("manifest_hash")
    if not isinstance(manifest_hash, str):
        raise TerminalSourceFailure("source manifest hash is absent")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if (
        _SHA256_RE.fullmatch(manifest_hash) is None
        or manifest_hash != _sha256_bytes(_canonical_json_bytes(core))
    ):
        raise TerminalSourceFailure("source manifest hash differs")
    return payload


def _validate_production_manifest(
    manifest: Mapping[str, Any],
    csv_bytes: bytes,
    *,
    claim_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("source_only") is not True
    ):
        raise TerminalSourceFailure("source manifest protocol differs")
    if manifest.get("generation_commit") != PRODUCTION_GENERATION_COMMIT:
        raise TerminalSourceFailure(
            "source manifest lacks the production generation marker"
        )
    expected_range = _source_range_manifest(
        iter_chunks(
            SOURCE_START_BLOCK,
            SOURCE_END_BLOCK_EXCLUSIVE,
            CHUNK_SIZE,
        )
    )
    if manifest.get("source_range") != expected_range:
        raise TerminalSourceFailure("source manifest envelope differs")
    if manifest.get("source_integrity") != ZERO_SOURCE_INTEGRITY:
        raise TerminalSourceFailure("source manifest integrity differs")
    boundary = manifest.get("boundary_evidence")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("outside_before_count") != 0
        or boundary.get("outside_after_maximum_admissible_count") != 0
        or boundary.get("header_count") != 12
        or boundary.get("canonical_header_set_sha256")
        != BOUNDARY_HEADER_SET_SHA256
        or boundary.get("frozen_header_set_exact") is not True
    ):
        raise TerminalSourceFailure("source manifest boundary evidence differs")
    if manifest.get("source_replay_schedule") != {
        "inter_batch_throttle_seconds": PRODUCTION_THROTTLE_SECONDS,
        "maximum_batch_by_role": dict(TRANSPORT_MAX_BATCH),
        "rpc_methods": sorted(RPC_METHODS),
    }:
        raise TerminalSourceFailure("source manifest replay schedule differs")
    transports = manifest.get("transports")
    if (
        not isinstance(transports, list)
        or manifest.get("transport_exact_set_equal") is not True
    ):
        raise TerminalSourceFailure("source manifest transports differ")
    validate_transport_identities(transports)
    if (
        _GIT_OBJECT_RE.fullmatch(
            str(manifest.get("protocol_parent_commit"))
        )
        is None
        or _GIT_OBJECT_RE.fullmatch(
            str(manifest.get("replay_claim_commit"))
        )
        is None
        or _SHA256_RE.fullmatch(str(manifest.get("replay_claim_sha256")))
        is None
    ):
        raise TerminalSourceFailure("source manifest claim identity differs")
    finalized = manifest.get("common_finalized_head")
    if not isinstance(finalized, Mapping):
        raise TerminalSourceFailure("source manifest finalized head differs")
    finalized_number = finalized.get("number")
    if (
        type(finalized_number) is not int
        or finalized_number < LAST_CONFIRMATION_BLOCK
        or finalized.get("covers_last_confirmation") is not True
    ):
        raise TerminalSourceFailure("source manifest finalized head differs")
    csv_sha256 = manifest.get("source_csv_sha256")
    if (
        not isinstance(csv_sha256, str)
        or _SHA256_RE.fullmatch(csv_sha256) is None
        or csv_sha256 != _sha256_bytes(csv_bytes)
    ):
        raise TerminalSourceFailure("source CSV hash differs")
    if claim_binding is not None and (
        manifest.get("protocol_parent_commit")
        != claim_binding.get("protocol_parent_commit")
        or manifest.get("replay_claim_commit")
        != claim_binding.get("claim_commit")
        or manifest.get("replay_claim_sha256") != claim_binding.get("sha256")
    ):
        raise TerminalSourceFailure("source generation claim binding differs")
    return dict(manifest)


def read_committed_generation(
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any] | None:
    """Read only a manifest-marked generation whose CSV hash is exact."""

    _, csv_path, manifest_path, _, _, _ = _generation_paths(
        repository_root=repository_root
    )
    if manifest_path.is_symlink():
        raise TerminalSourceFailure("canonical source manifest is unsafe")
    if not manifest_path.exists():
        return None
    if not manifest_path.is_file():
        raise TerminalSourceFailure("canonical source manifest is unsafe")
    raw = manifest_path.read_bytes()
    try:
        provisional = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(provisional, dict)
        or provisional.get("generation_commit")
        != PRODUCTION_GENERATION_COMMIT
        or not isinstance(provisional.get("manifest_hash"), str)
        or not isinstance(provisional.get("source_csv_sha256"), str)
    ):
        return None
    if csv_path.is_symlink():
        raise TerminalSourceFailure("canonical source CSV is unsafe")
    if not csv_path.exists():
        return None
    if not csv_path.is_file():
        raise TerminalSourceFailure("canonical source CSV is unsafe")
    payload = _decode_canonical_manifest(raw)
    return _validate_production_manifest(payload, csv_path.read_bytes())


def _publish_production_generation(
    csv_bytes: bytes,
    manifest_bytes: bytes,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Durably stage both files and publish the manifest as the commit point.

    POSIX cannot atomically publish two paths.  The CSV link is deliberately
    not a generation until the hash-binding manifest link exists.
    """

    manifest = _validate_production_manifest(
        _decode_canonical_manifest(manifest_bytes),
        csv_bytes,
    )
    _, claim_binding = validate_production_claim(
        SANITIZED_TRANSPORTS,
        repository_root=repository_root,
    )
    manifest = _validate_production_manifest(
        manifest,
        csv_bytes,
        claim_binding=claim_binding,
    )
    (
        root,
        csv_path,
        manifest_path,
        stage_directory,
        stage_csv,
        stage_manifest,
    ) = _generation_paths(repository_root=repository_root)
    if any(
        path.exists() or path.is_symlink()
        for path in (csv_path, manifest_path, stage_directory)
    ):
        raise FileExistsError(
            "canonical generation state already exists; recovery is required"
        )

    created_directories: list[Path] = []
    csv_published = False
    manifest_published = False
    committed = False
    try:
        _ensure_safe_directory(
            csv_path.parent, root=root, created=created_directories
        )
        _ensure_safe_directory(
            manifest_path.parent, root=root, created=created_directories
        )
        stage_directory.mkdir()
        _fsync_directory(stage_directory.parent)
        _durable_create_file(stage_csv, csv_bytes)
        _durable_create_file(stage_manifest, manifest_bytes)
        _fsync_directory(stage_directory)

        os.link(stage_csv, csv_path)
        csv_published = True
        _fsync_directory(csv_path.parent)

        os.link(stage_manifest, manifest_path)
        manifest_published = True
        _fsync_directory(manifest_path.parent)
        committed = True
    except Exception:
        if manifest_published:
            _unlink_file_durable(manifest_path)
        if csv_published:
            _unlink_file_durable(csv_path)
        try:
            _cleanup_stage_directory(stage_directory)
        finally:
            _rollback_created_directories(created_directories)
        raise

    if committed:
        try:
            _cleanup_stage_directory(stage_directory)
        except Exception:
            # A staged hard link is not part of the committed artifact.  Once
            # the manifest directory is fsynced, cleanup can be retried.
            pass
    return manifest


def _recover_production_generation(
    *,
    action: str,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any] | None:
    """Finalize or clean a durable pre-marker stage without any RPC access."""

    if action not in {"finalize", "clean"}:
        raise ValueError("generation recovery action must be finalize or clean")
    _, claim_binding = validate_production_claim(
        SANITIZED_TRANSPORTS,
        repository_root=repository_root,
    )
    (
        _,
        csv_path,
        manifest_path,
        stage_directory,
        stage_csv,
        stage_manifest,
    ) = _generation_paths(repository_root=repository_root)

    if manifest_path.exists() or manifest_path.is_symlink():
        committed = read_committed_generation(
            repository_root=repository_root
        )
        if committed is None:
            raise TerminalSourceFailure(
                "canonical manifest path is not a committed generation"
            )
        _validate_production_manifest(
            committed,
            csv_path.read_bytes(),
            claim_binding=claim_binding,
        )
        try:
            _cleanup_stage_directory(stage_directory)
        except Exception:
            pass
        return committed

    if not stage_directory.exists():
        if stage_directory.is_symlink():
            raise TerminalSourceFailure("generation stage directory is unsafe")
        if csv_path.exists() or csv_path.is_symlink():
            raise TerminalSourceFailure(
                "uncommitted source CSV lacks a durable recovery stage"
            )
        return None

    if stage_directory.is_symlink() or not stage_directory.is_dir():
        raise TerminalSourceFailure("generation stage directory is unsafe")
    staged_children = set(stage_directory.iterdir())
    expected_stage_children = {stage_csv, stage_manifest}
    if not staged_children.issubset(expected_stage_children):
        raise TerminalSourceFailure("generation stage is incomplete or unknown")
    if any(
        path.is_symlink() or not path.is_file()
        for path in staged_children
    ):
        raise TerminalSourceFailure("generation stage file is unsafe")
    if action == "clean":
        if csv_path.exists() or csv_path.is_symlink():
            if csv_path.is_symlink() or not csv_path.is_file():
                raise TerminalSourceFailure("canonical source CSV is unsafe")
            if staged_children != expected_stage_children:
                raise TerminalSourceFailure(
                    "uncommitted source CSV lacks complete staged evidence"
                )
            staged_csv_bytes = stage_csv.read_bytes()
            _validate_production_manifest(
                _decode_canonical_manifest(stage_manifest.read_bytes()),
                staged_csv_bytes,
                claim_binding=claim_binding,
            )
            if (
                csv_path.read_bytes() != staged_csv_bytes
            ):
                raise TerminalSourceFailure(
                    "uncommitted canonical source CSV differs from stage"
                )
            _unlink_file_durable(csv_path)
        _cleanup_stage_directory(stage_directory)
        return None
    if staged_children != expected_stage_children:
        raise TerminalSourceFailure("generation stage is incomplete or unknown")
    csv_bytes = stage_csv.read_bytes()
    manifest_bytes = stage_manifest.read_bytes()
    manifest = _validate_production_manifest(
        _decode_canonical_manifest(manifest_bytes),
        csv_bytes,
        claim_binding=claim_binding,
    )

    csv_published_here = False
    manifest_published = False
    try:
        if csv_path.exists() or csv_path.is_symlink():
            if (
                csv_path.is_symlink()
                or not csv_path.is_file()
                or csv_path.read_bytes() != csv_bytes
            ):
                raise TerminalSourceFailure(
                    "uncommitted canonical source CSV differs from stage"
                )
        else:
            os.link(stage_csv, csv_path)
            csv_published_here = True
            _fsync_directory(csv_path.parent)
        os.link(stage_manifest, manifest_path)
        manifest_published = True
        _fsync_directory(manifest_path.parent)
    except Exception:
        if manifest_published:
            _unlink_file_durable(manifest_path)
        if csv_published_here:
            _unlink_file_durable(csv_path)
        raise

    try:
        _cleanup_stage_directory(stage_directory)
    except Exception:
        pass
    return manifest


def _git(
    *arguments: str,
    repository_root: str | Path = REPOSITORY_ROOT,
    allow_failure: bool = False,
) -> bytes:
    root = Path(repository_root).resolve()
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode and not allow_failure:
        raise TerminalSourceFailure(
            f"protocol git command failed: {' '.join(arguments)}"
        )
    return completed.stdout


def _protocol_seal_payload(
    *,
    git_head: str,
    protocol_paths: Sequence[Path],
    files: Mapping[str, Mapping[str, str | None]],
) -> dict[str, Any]:
    core = {
        "protocol_version": "tron_usdt_supply_events_protocol_seal_v2",
        "git_head": git_head,
        "protocol_paths": [path.as_posix() for path in protocol_paths],
        "files": {name: dict(binding) for name, binding in files.items()},
        "source_replay_schedule": {
            "inter_batch_throttle_seconds": PRODUCTION_THROTTLE_SECONDS,
            "maximum_batch_by_role": dict(TRANSPORT_MAX_BATCH),
            "rpc_methods": sorted(RPC_METHODS),
        },
    }
    return {**core, "seal_hash": _sha256_bytes(_canonical_json_bytes(core))}


def current_protocol_seal(
    *,
    require_committed: bool = True,
    repository_root: str | Path = REPOSITORY_ROOT,
    protocol_paths: Sequence[Path] = PROTOCOL_PATHS,
    synthetic_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind protocol files; production defaults to the complete frozen closure."""

    paths = tuple(Path(path) for path in protocol_paths)
    if len(set(paths)) != len(paths) or not paths:
        raise TerminalSourceFailure("protocol paths are empty or duplicate")
    if synthetic_identity is not None:
        if require_committed:
            raise TerminalSourceFailure(
                "synthetic protocol identity is forbidden in production"
            )
        git_head = synthetic_identity.get("git_head")
        synthetic_files = synthetic_identity.get("files")
        if (
            not isinstance(git_head, str)
            or _GIT_OBJECT_RE.fullmatch(git_head) is None
            or not isinstance(synthetic_files, Mapping)
            or set(synthetic_files) != {path.as_posix() for path in paths}
        ):
            raise TerminalSourceFailure("synthetic protocol identity differs")
        return _protocol_seal_payload(
            git_head=git_head,
            protocol_paths=paths,
            files=cast(
                Mapping[str, Mapping[str, str | None]],
                synthetic_files,
            ),
        )

    root = Path(repository_root).resolve()
    commit = _git(
        "rev-parse",
        "--verify",
        "HEAD",
        repository_root=root,
    ).decode("ascii").strip()
    if _GIT_OBJECT_RE.fullmatch(commit) is None:
        raise TerminalSourceFailure("invalid Git HEAD")
    files: dict[str, Any] = {}
    for relative in paths:
        name = relative.as_posix()
        candidate = root / relative
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or not candidate.resolve().is_relative_to(root)
        ):
            raise TerminalSourceFailure(f"protocol path is unsafe: {name}")
        tracked = bool(
            _git(
                "ls-files",
                "--error-unmatch",
                "--",
                name,
                repository_root=root,
                allow_failure=True,
            ).strip()
        )
        clean = (
            subprocess.run(
                ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", name],
                check=False,
            ).returncode
            == 0
        )
        if require_committed and (not tracked or not clean):
            raise TerminalSourceFailure(f"protocol path is not committed-clean: {name}")
        actual = candidate.read_bytes()
        expected_sha256 = FROZEN_PROTOCOL_FILE_SHA256.get(name)
        if (
            expected_sha256 is not None
            and _sha256_bytes(actual) != expected_sha256
        ):
            raise TerminalSourceFailure(f"frozen protocol hash differs: {name}")
        if relative == PREREGISTRATION_ARTIFACT_PATH:
            try:
                preregistration = json.loads(actual)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise TerminalSourceFailure(
                    "frozen preregistration artifact is not JSON"
                ) from error
            if (
                not isinstance(preregistration, dict)
                or preregistration.get("manifest_hash")
                != PREREGISTRATION_MANIFEST_HASH
            ):
                raise TerminalSourceFailure(
                    "frozen preregistration manifest hash differs"
                )
        if tracked:
            head = _git("show", f"HEAD:{name}", repository_root=root)
            if actual != head:
                raise TerminalSourceFailure(f"protocol path differs from HEAD: {name}")
            git_blob: str | None = (
                _git(
                    "rev-parse",
                    f"HEAD:{name}",
                    repository_root=root,
                )
                .decode("ascii")
                .strip()
            )
        else:
            git_blob = None
        files[name] = {
            "git_blob": git_blob,
            "sha256": _sha256_bytes(actual),
        }
    return _protocol_seal_payload(
        git_head=commit,
        protocol_paths=paths,
        files=files,
    )


def validate_protocol_seal(
    recorded: Mapping[str, Any],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    protocol_paths: Sequence[Path] = PROTOCOL_PATHS,
) -> dict[str, Any]:
    """Prove every sealed protocol file is tracked, clean, and blob-identical."""

    paths = tuple(Path(path) for path in protocol_paths)
    if not isinstance(recorded, Mapping) or set(recorded) != {
        "protocol_version",
        "git_head",
        "protocol_paths",
        "files",
        "source_replay_schedule",
        "seal_hash",
    }:
        raise TerminalSourceFailure("recorded protocol seal schema differs")
    core = {key: value for key, value in recorded.items() if key != "seal_hash"}
    if (
        recorded["protocol_version"]
        != "tron_usdt_supply_events_protocol_seal_v2"
        or recorded["protocol_paths"] != [path.as_posix() for path in paths]
        or recorded["source_replay_schedule"]
        != {
            "inter_batch_throttle_seconds": PRODUCTION_THROTTLE_SECONDS,
            "maximum_batch_by_role": dict(TRANSPORT_MAX_BATCH),
            "rpc_methods": sorted(RPC_METHODS),
        }
        or recorded["seal_hash"] != _sha256_bytes(_canonical_json_bytes(core))
        or _GIT_OBJECT_RE.fullmatch(str(recorded["git_head"])) is None
    ):
        raise TerminalSourceFailure("recorded protocol seal identity differs")
    bindings = recorded["files"]
    if not isinstance(bindings, Mapping) or set(bindings) != {
        path.as_posix() for path in paths
    }:
        raise TerminalSourceFailure("recorded protocol file bindings differ")

    root = Path(repository_root).resolve()
    for relative in paths:
        name = relative.as_posix()
        binding = bindings[name]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"git_blob", "sha256"}
            or _GIT_OBJECT_RE.fullmatch(str(binding["git_blob"])) is None
            or _SHA256_RE.fullmatch(str(binding["sha256"])) is None
        ):
            raise TerminalSourceFailure(f"protocol binding is invalid: {name}")
        candidate = root / relative
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or not candidate.resolve().is_relative_to(root)
        ):
            raise TerminalSourceFailure(f"protocol path is unsafe: {name}")
        tracked = _git(
            "ls-files",
            "--error-unmatch",
            "--",
            name,
            repository_root=root,
            allow_failure=True,
        )
        if not tracked.strip():
            raise TerminalSourceFailure(f"protocol path is not tracked: {name}")
        if subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", name],
            check=False,
        ).returncode:
            raise TerminalSourceFailure(f"protocol path is not HEAD-clean: {name}")
        actual = candidate.read_bytes()
        head = _git("show", f"HEAD:{name}", repository_root=root)
        current_blob = (
            _git("rev-parse", f"HEAD:{name}", repository_root=root)
            .decode("ascii")
            .strip()
        )
        if (
            actual != head
            or _sha256_bytes(actual) != binding["sha256"]
            or current_blob != binding["git_blob"]
            or (
                name in FROZEN_PROTOCOL_FILE_SHA256
                and binding["sha256"] != FROZEN_PROTOCOL_FILE_SHA256[name]
            )
        ):
            raise TerminalSourceFailure(f"protocol path binding differs: {name}")
    return dict(recorded)


def _claim_payload(
    protocol_seal: Mapping[str, Any],
    transport_identities: Sequence[Mapping[str, Any]] = SANITIZED_TRANSPORTS,
) -> dict[str, Any]:
    identities = validate_transport_identities(transport_identities)
    core = {
        "protocol_version": "tron_usdt_supply_events_source_replay_claim_v2",
        "status": "claim_created_for_committed_replay",
        "one_shot": True,
        "protocol_parent_commit": protocol_seal.get("git_head"),
        "transports": [dict(identity) for identity in identities],
        "rpc_methods": sorted(RPC_METHODS),
        "retry_backoff_fallback_resume": False,
        "source_replay_schedule": {
            "inter_batch_throttle_seconds": PRODUCTION_THROTTLE_SECONDS,
            "maximum_batch_by_role": dict(TRANSPORT_MAX_BATCH),
        },
        "source_envelope": {
            "start_block_inclusive": SOURCE_START_BLOCK,
            "end_block_exclusive": SOURCE_END_BLOCK_EXCLUSIVE,
            "last_admissible_event_block": LAST_EVENT_BLOCK,
            "block_count": SOURCE_BLOCK_COUNT,
            "chunk_size": CHUNK_SIZE,
            "chunk_count": EXPECTED_CHUNK_COUNT,
            "final_chunk": [83_198_358, 83_200_991],
            "confirmation_blocks": CONFIRMATION_BLOCKS,
            "last_confirmation_block": LAST_CONFIRMATION_BLOCK,
            "exclusive_utc_boundary_block": END_BOUNDARY_BLOCK,
            "boundary_header_set_sha256": BOUNDARY_HEADER_SET_SHA256,
        },
        "access_at_claim_creation": {
            "rpc_clients_constructed": 0,
            "rpc_calls_issued": 0,
            "pre_cutoff_events_opened": 0,
            "pre_cutoff_logs_opened": 0,
            "pre_cutoff_receipts_opened": 0,
            "source_incidence_opened": 0,
            "market_rows_opened": 0,
            "outcomes_opened": 0,
        },
        "canonical_outputs": [
            DEFAULT_CSV_OUTPUT.as_posix(),
            DEFAULT_MANIFEST_OUTPUT.as_posix(),
        ],
        "protocol_seal": dict(protocol_seal),
    }
    return {**core, "claim_hash": _sha256_bytes(_canonical_json_bytes(core))}


def create_production_claim(
    protocol_seal: Mapping[str, Any],
    transport_identities: Sequence[Mapping[str, Any]] = SANITIZED_TRANSPORTS,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    claim_path: Path = REPLAY_CLAIM_PATH,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    claim = root / claim_path
    if (
        claim.exists()
        or claim.is_symlink()
        or (root / DEFAULT_CSV_OUTPUT).exists()
        or (root / DEFAULT_MANIFEST_OUTPUT).exists()
    ):
        raise FileExistsError("canonical claim or output already exists")
    payload = _claim_payload(protocol_seal, transport_identities)
    raw = _canonical_json_bytes(payload, trailing_lf=True)
    atomic_write_outputs({claim: raw}, allowed_root=root)
    return {
        "path": claim_path.as_posix(),
        "sha256": _sha256_bytes(raw),
        "claim_hash": payload["claim_hash"],
    }


def validate_production_claim(
    transport_identities: Sequence[Mapping[str, Any]],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    protocol_paths: Sequence[Path] = PROTOCOL_PATHS,
    claim_path: Path = REPLAY_CLAIM_PATH,
    require_upstream: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate claim-only commit C and sealed parent P before any RPC exists."""

    root = Path(repository_root).resolve()
    claim = root / claim_path
    if (
        not claim.is_file()
        or claim.is_symlink()
        or not claim.resolve().is_relative_to(root)
    ):
        raise TerminalSourceFailure("production replay claim is absent or unsafe")
    raw = claim.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalSourceFailure("production replay claim is invalid") from exc
    if not isinstance(payload, dict):
        raise TerminalSourceFailure("production replay claim is not an object")
    seal = payload.get("protocol_seal")
    if (
        not isinstance(seal, Mapping)
        or payload != _claim_payload(seal, transport_identities)
        or raw != _canonical_json_bytes(payload, trailing_lf=True)
    ):
        raise TerminalSourceFailure("production replay claim bytes differ")

    head = (
        _git("rev-parse", "--verify", "HEAD", repository_root=root)
        .decode("ascii")
        .strip()
    )
    if require_upstream:
        upstream = (
            _git(
                "rev-parse",
                "--verify",
                "@{upstream}",
                repository_root=root,
            )
            .decode("ascii")
            .strip()
        )
        if head != upstream:
            raise TerminalSourceFailure("HEAD does not equal upstream claim commit")
    parent_line = (
        _git("rev-list", "--parents", "-n", "1", "HEAD", repository_root=root)
        .decode("ascii")
        .strip()
        .split()
    )
    if (
        len(parent_line) != 2
        or parent_line[0] != head
        or parent_line[1] != seal.get("git_head")
        or payload.get("protocol_parent_commit") != parent_line[1]
    ):
        raise TerminalSourceFailure("claim commit first-parent relation differs")
    changed = [
        line
        for line in _git(
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "HEAD",
            repository_root=root,
        )
        .decode("utf-8")
        .splitlines()
        if line
    ]
    if changed != [f"A\t{claim_path.as_posix()}"]:
        raise TerminalSourceFailure("HEAD is not a claim-only commit")
    if not _git(
        "ls-files",
        "--error-unmatch",
        "--",
        claim_path.as_posix(),
        repository_root=root,
        allow_failure=True,
    ).strip():
        raise TerminalSourceFailure("production replay claim is not tracked")
    if subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            claim_path.as_posix(),
        ],
        check=False,
    ).returncode:
        raise TerminalSourceFailure("production replay claim is not HEAD-clean")
    if _git(
        "show",
        f"HEAD:{claim_path.as_posix()}",
        repository_root=root,
    ) != raw:
        raise TerminalSourceFailure("production replay claim differs from HEAD")
    validate_protocol_seal(
        seal,
        repository_root=root,
        protocol_paths=protocol_paths,
    )
    binding = {
        "path": claim_path.as_posix(),
        "sha256": _sha256_bytes(raw),
        "claim_hash": payload["claim_hash"],
        "claim_commit": head,
        "protocol_parent_commit": parent_line[1],
    }
    return dict(seal), binding


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        raise TerminalSourceFailure("RPC redirect is forbidden")


def sanitize_transport(
    url: str, *, expected_role: str | None = None
) -> dict[str, Any]:
    """Validate a secret-bearing endpoint and return only safe identity fields."""

    if not isinstance(url, str) or not url:
        raise ValueError("TRON RPC URL must be non-empty")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port or 443
    except ValueError:
        raise ValueError("TRON RPC URL is malformed") from None
    if (
        parsed.scheme != "https"
        or parsed.hostname not in TRANSPORT_HOSTS
        or parsed.netloc
        not in {
            PRIMARY_HOST,
            f"{PRIMARY_HOST}:443",
            VERIFY_HOST,
            f"{VERIFY_HOST}:443",
        }
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path in {"", "/"}
        or port != 443
    ):
        raise ValueError("TRON RPC URL violates the frozen transport policy")
    role = (
        "primary" if parsed.hostname == PRIMARY_HOST else "verification"
    )
    if expected_role is not None and role != expected_role:
        raise TerminalSourceFailure("TRON RPC transport role differs")
    return {
        "role": role,
        "scheme": parsed.scheme,
        "hostname": parsed.hostname,
        "port": port,
    }


class HttpJsonRpcClient:
    """Strict one-attempt JSON-RPC client with frozen deterministic batching."""

    def __init__(
        self,
        url: str,
        *,
        throttle_seconds: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        identity = sanitize_transport(url)
        if throttle_seconds < 0 or timeout_seconds <= 0:
            raise ValueError("invalid deterministic transport timing")
        self.url = url
        self.role = identity["role"]
        self.identity = identity
        self.max_batch = TRANSPORT_MAX_BATCH[self.role]
        self.throttle_seconds = throttle_seconds
        self.timeout_seconds = timeout_seconds
        self._next_id = 1
        self._batch_count = 0
        context = ssl.create_default_context()
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _RejectRedirects(),
            urllib.request.HTTPSHandler(context=context),
        )

    def _request(self, entries: Sequence[tuple[int, str, Sequence[Any]]]) -> list[Any]:
        if self._batch_count and self.throttle_seconds:
            time.sleep(self.throttle_seconds)
        self._batch_count += 1
        requests = [
            {"jsonrpc": "2.0", "id": ident, "method": method, "params": list(params)}
            for ident, method, params in entries
        ]
        body = self._post(_canonical_json_bytes(requests))
        try:
            decoded = json.loads(
                body.decode("utf-8"), object_pairs_hook=_unique_object
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise TerminalSourceFailure("RPC response is not strict JSON") from None
        if not isinstance(decoded, list) or len(decoded) != len(entries):
            raise TerminalSourceFailure("RPC batch shape differs")
        by_id: dict[int, Any] = {}
        expected_ids = {entry[0] for entry in entries}
        for item in decoded:
            if (
                not isinstance(item, dict)
                or set(item) != {"jsonrpc", "id", "result"}
                or item.get("jsonrpc") != "2.0"
                or type(item.get("id")) is not int
                or item["id"] not in expected_ids
                or item["id"] in by_id
            ):
                raise TerminalSourceFailure("RPC response identity differs")
            by_id[item["id"]] = item["result"]
        return [by_id[ident] for ident, _, _ in entries]

    def _post(self, payload: bytes) -> bytes:
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "rllm-tron-usdt-source/1",
            },
            method="POST",
        )
        try:
            with self._opener.open(
                request, timeout=self.timeout_seconds
            ) as response:
                if response.status != 200:
                    raise TerminalSourceFailure(
                        f"{self.role} TRON RPC returned HTTP {response.status}"
                    )
                return response.read()
        except TerminalSourceFailure:
            raise
        except Exception:
            raise TerminalSourceFailure(
                f"single RPC attempt failed for {self.role} TRON RPC"
            ) from None

    def batch(
        self, requests: Sequence[tuple[str, Sequence[Any]]]
    ) -> Sequence[Any]:
        if not requests or len(requests) > self.max_batch:
            raise TerminalSourceFailure("RPC batch exceeds frozen transport limit")
        entries: list[tuple[int, str, Sequence[Any]]] = []
        for method, params in requests:
            if method not in RPC_METHODS:
                raise TerminalSourceFailure(f"unfrozen RPC method {method!r}")
            entries.append((self._next_id, method, params))
            self._next_id += 1
        return self._request(entries)

    def call(self, method: str, params: Sequence[Any]) -> Any:
        return self.batch(((method, params),))[0]


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TerminalSourceFailure("duplicate JSON object key")
        result[key] = value
    return result


def _check_clients(clients: Sequence[RpcLike]) -> None:
    if len(clients) != 2:
        raise TerminalSourceFailure("exactly the two frozen transports are required")
    identities = tuple(
        sanitize_transport(client.url, expected_role=role)
        for client, role in zip(clients, TRANSPORT_ROLES, strict=True)
    )
    if tuple(identity["hostname"] for identity in identities) != TRANSPORT_HOSTS:
        raise TerminalSourceFailure("frozen transport hostnames differ")


def _batched_calls(
    client: RpcLike,
    requests: Sequence[tuple[str, Sequence[Any]]],
) -> tuple[Any, ...]:
    identity = sanitize_transport(client.url)
    limit = TRANSPORT_MAX_BATCH[identity["role"]]
    output: list[Any] = []
    for offset in range(0, len(requests), limit):
        request_batch = requests[offset : offset + limit]
        response_batch = tuple(client.batch(request_batch))
        if len(response_batch) != len(request_batch):
            raise TerminalSourceFailure("RPC batch has missing response ids")
        output.extend(response_batch)
    return tuple(output)


def replay_log_categories(
    clients: Sequence[RpcLike],
    *,
    chunks: Sequence[tuple[int, int]] | None = None,
    audit: SourceIntegrityAudit | None = None,
) -> dict[str, tuple[CanonicalLog, ...]]:
    _check_clients(clients)
    requested_chunks = validate_replay_chunks(
        tuple(chunks) if chunks is not None else frozen_chunks()
    )

    requests = tuple(
        (
            "eth_getLogs",
            (category_filter(category, first, last),),
        )
        for first, last in requested_chunks
        for category in CATEGORIES
    )
    replays = [_batched_calls(client, requests) for client in clients]
    category_logs: dict[str, list[CanonicalLog]] = {
        category: [] for category in CATEGORIES
    }
    for index, (first, last) in enumerate(requested_chunks):
        for category_offset, category in enumerate(CATEGORIES):
            request_index = index * len(CATEGORIES) + category_offset
            agreed = compare_log_sets(
                replays[0][request_index],
                replays[1][request_index],
                first_block=first,
                last_block=last,
            )
            category_logs[category].extend(agreed)
    result = {
        category: tuple(sorted(logs)) for category, logs in category_logs.items()
    }
    global_logs = tuple(log for category in CATEGORIES for log in result[category])
    if len({log.identity for log in global_logs}) != len(global_logs):
        raise TerminalSourceFailure("global source log set contains duplicates")
    if audit is not None:
        audit.mark("chunk_geometry")
        audit.mark("response_ids")
        audit.mark("dual_logs")
    return result


def _paired_exact(
    clients: Sequence[RpcLike],
    requests: Sequence[tuple[str, Sequence[Any]]],
) -> tuple[Any, ...]:
    left = _batched_calls(clients[0], requests)
    right = _batched_calls(clients[1], requests)
    if len(left) != len(right):
        raise TerminalSourceFailure("provider response counts disagree")
    return tuple(zip(left, right, strict=True))


def retrieve_evidence(
    clients: Sequence[RpcLike],
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    enforce_frozen_boundaries: bool = False,
    audit: SourceIntegrityAudit | None = None,
) -> tuple[dict[int, Header], dict[str, Receipt], Header, dict[str, Any]]:
    _check_clients(clients)
    chain_pairs = _paired_exact(clients, (("eth_chainId", ()),))
    left_chain, right_chain = chain_pairs[0]
    if (
        left_chain != right_chain
        or parse_quantity(left_chain, "chain id") != CHAIN_ID
    ):
        raise TerminalSourceFailure("transports do not agree on TRON mainnet")

    needed = {
        number
        for row in paired_rows
        for number in (
            row["block_number"] - 1,
            row["block_number"],
            row["block_number"] + CONFIRMATION_BLOCKS - 1,
            row["block_number"] + CONFIRMATION_BLOCKS,
        )
    }
    needed.update(
        number
        for boundary in FROZEN_BOUNDARIES
        for number in (boundary["number"] - 1, boundary["number"])
    )
    numbers = tuple(sorted(needed))
    header_requests = tuple(
        ("eth_getBlockByNumber", (hex(number), False)) for number in numbers
    )
    header_pairs = _paired_exact(clients, header_requests)
    headers: dict[int, Header] = {}
    for number, (left_raw, right_raw) in zip(numbers, header_pairs, strict=True):
        left = normalize_header(left_raw, expected_number=number)
        right = normalize_header(right_raw, expected_number=number)
        if left != right:
            raise TerminalSourceFailure("provider block headers disagree")
        headers[number] = left
    boundary_payload: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    expected_payload = frozen_boundary_header_payload()
    for boundary_index, boundary in enumerate(FROZEN_BOUNDARIES):
        number = boundary["number"]
        previous = headers[number - 1]
        current = headers[number]
        if current.parent_hash != previous.block_hash:
            raise TerminalSourceFailure("boundary parent relation differs")
        threshold = int(
            datetime.fromisoformat(
                str(boundary["utc"]).replace("Z", "+00:00")
            ).timestamp()
        )
        if not previous.timestamp < threshold <= current.timestamp:
            raise TerminalSourceFailure("boundary timestamp relation differs")
        pair = (
            {
                "number": previous.number,
                "hash": previous.block_hash,
                "parentHash": previous.parent_hash,
                "timestamp": previous.timestamp,
            },
            {
                "number": current.number,
                "hash": current.block_hash,
                "parentHash": current.parent_hash,
                "timestamp": current.timestamp,
            },
        )
        boundary_payload.extend(pair)
        if enforce_frozen_boundaries:
            expected_pair = expected_payload[
                boundary_index * 2 : boundary_index * 2 + 2
            ]
            if tuple(pair) != expected_pair:
                raise TerminalSourceFailure("frozen boundary header differs")
        boundary_rows.append(
            {
                "utc": boundary["utc"],
                "previous_block": number - 1,
                "first_block_at_or_after": number,
                "parent_relation_exact": True,
                "timestamp_relation_exact": True,
                "frozen_hash_exact": enforce_frozen_boundaries,
            }
        )
    boundary_sha256 = _sha256_bytes(_canonical_json_bytes(boundary_payload))
    if (
        enforce_frozen_boundaries
        and boundary_sha256 != BOUNDARY_HEADER_SET_SHA256
    ):
        raise TerminalSourceFailure("frozen boundary header-set hash differs")

    hashes = tuple(sorted({row["transaction_hash"] for row in paired_rows}))
    receipt_requests = tuple(
        ("eth_getTransactionReceipt", (transaction_hash,))
        for transaction_hash in hashes
    )
    receipt_pairs = _paired_exact(clients, receipt_requests)
    receipts: dict[str, Receipt] = {}
    for transaction_hash, (left_raw, right_raw) in zip(
        hashes, receipt_pairs, strict=True
    ):
        left = normalize_receipt(left_raw, expected_hash=transaction_hash)
        right = normalize_receipt(right_raw, expected_hash=transaction_hash)
        if left != right:
            raise TerminalSourceFailure("provider transaction receipts disagree")
        receipts[transaction_hash] = left

    finalized_pair = _paired_exact(
        clients, (("eth_getBlockByNumber", ("finalized", False)),)
    )[0]
    left_finalized = normalize_header(finalized_pair[0])
    right_finalized = normalize_header(finalized_pair[1])
    if left_finalized != right_finalized:
        raise TerminalSourceFailure("provider finalized heads disagree")
    if left_finalized.number < LAST_CONFIRMATION_BLOCK:
        raise TerminalSourceFailure("common finalized head is below last confirmation")

    def fetch_explicit_finalized_height() -> Header:
        explicit_pair = _paired_exact(
            clients,
            (
                (
                    "eth_getBlockByNumber",
                    (hex(left_finalized.number), False),
                ),
            ),
        )[0]
        left_explicit = normalize_header(
            explicit_pair[0],
            expected_number=left_finalized.number,
        )
        right_explicit = normalize_header(
            explicit_pair[1],
            expected_number=left_finalized.number,
        )
        if left_explicit != right_explicit:
            raise TerminalSourceFailure(
                "provider explicit finalized-height headers disagree"
            )
        return left_explicit

    left_explicit_finalized = fetch_explicit_finalized_height()
    if left_explicit_finalized != left_finalized:
        raise TerminalSourceFailure(
            "finalized tag differs from its explicit-height header"
        )
    existing_finalized = headers.get(left_finalized.number)
    if (
        existing_finalized is not None
        and existing_finalized != left_explicit_finalized
    ):
        raise TerminalSourceFailure(
            "finalized header differs from an already fetched header"
        )
    headers[left_finalized.number] = left_explicit_finalized
    independently_refetched_finalized = fetch_explicit_finalized_height()
    if (
        independently_refetched_finalized != left_finalized
        or independently_refetched_finalized
        != headers[left_finalized.number]
    ):
        raise TerminalSourceFailure(
            "independent explicit finalized-height re-fetch differs"
        )
    boundary = {
        "outside_before_count": 0,
        "outside_after_maximum_admissible_count": 0,
        "header_count": len(boundary_payload),
        "canonical_header_set_sha256": boundary_sha256,
        "frozen_header_set_exact": enforce_frozen_boundaries,
        "boundaries": boundary_rows,
    }
    if audit is not None:
        audit.mark("headers")
        audit.mark("receipts")
        audit.mark("successful_receipts")
    return headers, receipts, left_finalized, boundary


def build_from_clients(
    clients: Sequence[RpcLike],
    *,
    config: BuildConfig,
    chunks: Sequence[tuple[int, int]] | None = None,
    production: bool = False,
    synthetic_protocol_paths: Sequence[Path] = SYNTHETIC_PROTOCOL_PATHS,
    synthetic_protocol_identity: Mapping[str, Any] | None = None,
) -> ComputedBuild:
    """Compute unpublished synthetic bytes using only injected RPC fixtures."""

    if type(config.non_production) is not bool or not config.non_production:
        raise TerminalSourceFailure(
            "injectable replay requires explicit non-production mode"
        )
    if production:
        raise TerminalSourceFailure(
            "production replay requires the committed-claim --replay path"
        )
    if any(isinstance(client, HttpJsonRpcClient) for client in clients):
        raise TerminalSourceFailure(
            "synthetic replay may not use network-capable RPC clients"
        )
    _check_clients(clients)
    identities = tuple(
        sanitize_transport(client.url, expected_role=role)
        for client, role in zip(clients, TRANSPORT_ROLES, strict=True)
    )
    identities = validate_transport_identities(identities)
    requested_chunks = validate_replay_chunks(
        tuple(chunks) if chunks is not None else frozen_chunks(),
        require_source_envelope=True,
    )
    protocol_seal = current_protocol_seal(
        require_committed=False,
        protocol_paths=synthetic_protocol_paths,
        synthetic_identity=synthetic_protocol_identity,
    )

    audit = SourceIntegrityAudit()
    category_logs = replay_log_categories(
        clients,
        chunks=requested_chunks,
        audit=audit,
    )
    paired = pair_supply_events(
        category_logs[CATEGORY_SEMANTIC],
        category_logs[CATEGORY_MINT],
        category_logs[CATEGORY_BURN],
        audit=audit,
    )
    headers, receipts, finalized, boundary = retrieve_evidence(
        clients, paired, audit=audit
    )
    rows = materialize_events(paired, headers, receipts, audit=audit)
    csv_bytes = serialize_csv(rows)
    manifest = build_manifest(
        rows,
        csv_bytes,
        category_logs=category_logs,
        replay_chunks=requested_chunks,
        finalized_head=finalized,
        boundary_evidence=boundary,
        source_integrity=audit.zero_evidence(),
        headers=headers,
        receipts=receipts,
        protocol_seal=protocol_seal,
        claim_binding=None,
        transport_identities=identities,
        production=False,
    )
    manifest_bytes = serialize_manifest(manifest)
    return ComputedBuild(
        rows=tuple(dict(row) for row in rows),
        csv_bytes=csv_bytes,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
    )


def create_claim_only() -> dict[str, Any]:
    """Create only the one-shot claim; never construct or call an RPC client."""

    primary_url = os.environ.get(PRIMARY_RPC_ENV, "")
    verification_url = os.environ.get(VERIFY_RPC_ENV, "")
    identities = (
        sanitize_transport(primary_url, expected_role="primary"),
        sanitize_transport(verification_url, expected_role="verification"),
    )
    protocol_seal = current_protocol_seal(require_committed=True)
    return create_production_claim(protocol_seal, identities)


def _assert_replay_outputs_absent(
    *, repository_root: str | Path = REPOSITORY_ROOT
) -> None:
    (
        _,
        csv_path,
        manifest_path,
        stage_directory,
        _,
        _,
    ) = _generation_paths(repository_root=repository_root)
    for relative, candidate in (
        (DEFAULT_CSV_OUTPUT, csv_path),
        (DEFAULT_MANIFEST_OUTPUT, manifest_path),
        (GENERATION_STAGE_DIRECTORY, stage_directory),
    ):
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(
                f"write-once production output already exists: {relative}"
            )


def replay_production() -> dict[str, Any]:
    """Validate committed claim C and parent P before constructing RPC clients."""

    primary_url = os.environ.get(PRIMARY_RPC_ENV, "")
    verification_url = os.environ.get(VERIFY_RPC_ENV, "")
    identities = validate_transport_identities(
        (
            sanitize_transport(primary_url, expected_role="primary"),
            sanitize_transport(verification_url, expected_role="verification"),
        )
    )
    protocol_seal, claim_binding = validate_production_claim(identities)
    replay_chunks = validate_replay_chunks(
        frozen_chunks(),
        require_source_envelope=True,
        require_frozen_envelope=True,
    )
    (
        _,
        _,
        manifest_path,
        _,
        _,
        _,
    ) = _generation_paths(repository_root=REPOSITORY_ROOT)
    manifest_was_present = manifest_path.exists() or manifest_path.is_symlink()
    recovered = _recover_production_generation(
        action="finalize",
        repository_root=REPOSITORY_ROOT,
    )
    if recovered is not None:
        if manifest_was_present:
            raise FileExistsError(
                "write-once production generation is already committed"
            )
        return recovered
    _assert_replay_outputs_absent(repository_root=REPOSITORY_ROOT)
    clients = tuple(
        HttpJsonRpcClient(
            url,
            throttle_seconds=PRODUCTION_THROTTLE_SECONDS,
        )
        for url in (primary_url, verification_url)
    )
    audit = SourceIntegrityAudit()
    category_logs = replay_log_categories(
        clients,
        chunks=replay_chunks,
        audit=audit,
    )
    paired = pair_supply_events(
        category_logs[CATEGORY_SEMANTIC],
        category_logs[CATEGORY_MINT],
        category_logs[CATEGORY_BURN],
        audit=audit,
    )
    headers, receipts, finalized, boundary = retrieve_evidence(
        clients,
        paired,
        enforce_frozen_boundaries=True,
        audit=audit,
    )
    rows = materialize_events(paired, headers, receipts, audit=audit)
    csv_bytes = serialize_csv(rows)
    manifest = build_manifest(
        rows,
        csv_bytes,
        category_logs=category_logs,
        replay_chunks=replay_chunks,
        finalized_head=finalized,
        boundary_evidence=boundary,
        source_integrity=audit.zero_evidence(),
        headers=headers,
        receipts=receipts,
        protocol_seal=protocol_seal,
        claim_binding=claim_binding,
        transport_identities=identities,
        production=True,
    )
    return _publish_production_generation(
        csv_bytes,
        serialize_manifest(manifest),
        repository_root=REPOSITORY_ROOT,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--create-claim",
        action="store_true",
        help="atomically create the claim and exit without RPC construction",
    )
    action.add_argument(
        "--replay",
        action="store_true",
        help="validate the committed claim and run the one-shot replay",
    )
    args = parser.parse_args(argv)
    if args.create_claim:
        create_claim_only()
    else:
        replay_production()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
