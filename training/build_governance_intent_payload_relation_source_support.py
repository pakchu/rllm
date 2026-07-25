"""Build the frozen, source-only GIPR-D1 Ethereum governance evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_ethereum_stablecoin_issuance_redemption as ethereum
from training import preregister_governance_intent_payload_relation as prereg


UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(
    "training/build_governance_intent_payload_relation_source_support.py"
)
TEST_PATH = Path(
    "tests/test_build_governance_intent_payload_relation_source_support.py"
)
CONTRACT_PATH = Path(
    "docs/gipr-d1-source-support-implementation-contract-2026-07-25.md"
)
CONTRACT_COMMIT = "ffd5c6396db57dff05b1cff273a966b12095be48"
CONTRACT_SHA256 = (
    "47c486b906e2442c92ac0ca1e44271397364d7b02e6b19b0a2eeec4401c32bed"
)
PREREGISTRATION_PATH = Path(
    "results/governance_intent_payload_relation_preregistration_2026-07-25.json"
)
PREREGISTRATION_COMMIT = "50269717b64bbef02b8da96bb2bdc010851b47d7"
PREREGISTRATION_SHA256 = (
    "319ac26108e936331f95d047a69f739ffecfd2bdc777573100a1ce83c771c197"
)
PREREGISTRATION_MANIFEST_HASH = (
    "daccd86ac2b218552f6312189fe4cd0ce6c775fe3b91f37ace31e8d12e1b1645"
)
PREREGISTRATION_PRODUCER_PATH = Path(
    "training/preregister_governance_intent_payload_relation.py"
)
PREREGISTRATION_PRODUCER_SHA256 = (
    "cb9273a98aebb21c6a11803eac106663235440c4cb7dcc1d7232b37859d1c24f"
)
ETHEREUM_HELPER_PATH = Path(
    "training/build_ethereum_stablecoin_issuance_redemption.py"
)
ETHEREUM_HELPER_SHA256 = (
    "af699d91404e44298573ca148c6cf1b90b68d9aac2972da9ad228a26b9e8a9a5"
)

EXECUTION_SEAL_PATH = Path(
    "results/gipr_d1_source_support_execution_seal_2026-07-25.json"
)
EVENT_OUTPUT = Path(
    "data/ethereum_governance_intent_payload_2020_2023/"
    "governance_events_2020_2023.csv.gz"
)
PROPOSAL_OUTPUT = Path(
    "data/ethereum_governance_intent_payload_2020_2023/"
    "governance_proposals_2020_2023.jsonl.gz"
)
CARD_OUTPUT = Path(
    "data/ethereum_governance_intent_payload_2020_2023/"
    "gipr_daily_source_cards_2020_2023.jsonl.gz"
)
CONTROL_OUTPUT = Path(
    "results/governance_intent_payload_relation_source_controls_2026-07-25.json"
)
PASS_REPORT = Path(
    "results/governance_intent_payload_relation_source_support_2026-07-25.json"
)
REJECTION_REPORT = Path(
    "results/governance_intent_payload_relation_source_rejection_2026-07-25.json"
)
DEFAULT_CHECKPOINT_DB = Path("data/.cache/gipr_d1_source_backfill.sqlite3")

SELF_CHECK_PROTOCOL = "gipr_d1_source_support_self_check_v1"
SEAL_PROTOCOL = "gipr_d1_source_support_execution_seal_v1"
RESULT_PROTOCOL = "gipr_d1_source_support_result_v1"
POLICY_ID = "GIPR-D1"
PASS_ACTION = "AUTHORIZE_GIPR_D1_ECONOMIC_RLLM_EVALUATOR_FREEZE_ONLY"
FAILURE_ACTION = (
    "REJECT_GIPR_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES"
)

GATE_NAMES = tuple(prereg.SOURCE_ONLY_GATES)
CONTROL_NAMES = tuple(prereg.RELATION_CONTROLS)
FORBIDDEN_COUNTER_NAMES = tuple(prereg.FORBIDDEN_COUNTERS)

FIRST_LOG_BLOCK = min(row.first_code_block for row in prereg.GOVERNORS)
LAST_LOG_BLOCK = prereg.LAST_SOURCE_BLOCK
FIRST_DECISION = datetime(2020, 1, 29, tzinfo=UTC)
LAST_DECISION = datetime(2023, 12, 31, tzinfo=UTC)
SOURCE_CUTOFF = datetime(2024, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)

ADDRESS_TO_GOVERNOR = {row.address: row for row in prereg.GOVERNORS}
TOPIC_TO_EVENT = {row.topic: row for row in prereg.EVENT_SPECS}
KNOWN_TARGET_ROLES = {
    row.address: f"{row.protocol}:{row.role}" for row in prereg.KNOWN_TARGET_ROLES
}

EVENT_COLUMNS = (
    "protocol",
    "generation",
    "governor_address",
    "event",
    "proposal_id",
    "block_number",
    "block_hash",
    "block_timestamp",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "confirmation_block_number",
    "confirmation_block_hash",
    "available_at",
)

HEX_BLOB = re.compile(r"0x(?:[0-9a-fA-F]{2})*\Z", re.ASCII)
HEX40 = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
MARKDOWN_HEADING = re.compile(r"(?m)^#{1,6} ", re.ASCII)


@dataclass(frozen=True)
class Config:
    primary_rpc_url: str = ""
    verification_rpc_url: str = ""
    checkpoint_db: str = str(DEFAULT_CHECKPOINT_DB)
    max_block_range: int = prereg.MAX_BLOCK_RANGE
    header_batch_size: int = prereg.HEADER_BATCH_SIZE
    timeout_sec: float = 60.0
    max_retries: int = ethereum.DEFAULT_MAX_RETRIES
    request_interval_sec: float = 0.0


@dataclass(frozen=True)
class CanonicalLog:
    address: str
    topics: tuple[str, ...]
    data: str
    removed: bool
    block_number: int
    block_hash: str
    transaction_hash: str
    transaction_index: int
    log_index: int

    @property
    def canonical_identity(self) -> tuple[str, str, int]:
        return self.block_hash, self.transaction_hash, self.log_index

    @property
    def order_key(self) -> tuple[int, int, int, str]:
        return (
            self.block_number,
            self.transaction_index,
            self.log_index,
            self.address,
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "topics": list(self.topics),
            "data": self.data,
            "removed": self.removed,
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "transaction_hash": self.transaction_hash,
            "transaction_index": self.transaction_index,
            "log_index": self.log_index,
        }


@dataclass(frozen=True)
class NormalizedEvent:
    protocol: str
    generation: str
    governor_address: str
    event: str
    proposal_id: int
    proposer: str
    targets: tuple[str, ...]
    values: tuple[int, ...]
    signatures: tuple[str, ...]
    calldatas: tuple[str, ...]
    start_block: int | None
    end_block: int | None
    description: str
    queue_eta: int | None
    block_number: int
    block_hash: str
    transaction_hash: str
    transaction_index: int
    log_index: int
    confirmation_block_number: int | None = None
    confirmation_block_hash: str = ""
    event_at: datetime | None = None
    available_at: datetime | None = None

    @property
    def proposal_key(self) -> tuple[str, int]:
        return self.governor_address, self.proposal_id

    @property
    def chain_order_key(self) -> tuple[int, int, int, str]:
        return (
            self.block_number,
            self.transaction_index,
            self.log_index,
            self.governor_address,
        )

    @property
    def availability_order_key(
        self,
    ) -> tuple[datetime, int, int, int, str]:
        if self.available_at is None:
            raise RuntimeError("GIPR event has no availability timestamp")
        return (
            self.available_at,
            self.block_number,
            self.transaction_index,
            self.log_index,
            self.governor_address,
        )

    def canonical_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol": self.protocol,
            "generation": self.generation,
            "governor_address": self.governor_address,
            "event": self.event,
            "proposal_id": str(self.proposal_id),
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "transaction_hash": self.transaction_hash,
            "transaction_index": self.transaction_index,
            "log_index": self.log_index,
            "confirmation_block_number": self.confirmation_block_number,
            "confirmation_block_hash": self.confirmation_block_hash,
            "event_at": format_time(self.event_at),
            "available_at": format_time(self.available_at),
        }
        if include_payload:
            payload.update(
                {
                    "proposer": self.proposer,
                    "targets": list(self.targets),
                    "values": [str(value) for value in self.values],
                    "signatures": list(self.signatures),
                    "calldatas": list(self.calldatas),
                    "start_block": self.start_block,
                    "end_block": self.end_block,
                    "description": self.description,
                    "queue_eta": (
                        None if self.queue_eta is None else str(self.queue_eta)
                    ),
                }
            )
        return payload


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    metrics: Mapping[str, Any]
    failure: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "failure": self.failure,
        }


@dataclass
class AccessLedger:
    counters: dict[str, int]

    @classmethod
    def zero(cls) -> "AccessLedger":
        return cls({name: 0 for name in FORBIDDEN_COUNTER_NAMES})

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _safe_destination(
    path: str | Path,
    *,
    create_parent: bool,
) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        destination = candidate
        current = Path(candidate.anchor)
        parts = candidate.parent.parts[1:]
    else:
        if any(part == ".." for part in candidate.parts):
            raise RuntimeError("GIPR artifact path escapes repository")
        destination = REPO_ROOT / candidate
        current = REPO_ROOT
        parts = candidate.parent.parts
    if current.is_symlink() or not current.is_dir():
        raise RuntimeError("GIPR artifact root is unsafe")
    for part in parts:
        if part in {"", "."}:
            continue
        current /= part
        if os.path.lexists(current):
            if current.is_symlink() or not current.is_dir():
                raise RuntimeError("GIPR artifact parent is unsafe")
        elif create_parent:
            os.mkdir(current)
        else:
            raise RuntimeError("GIPR artifact parent is absent")
    return destination


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(repository_path(path).read_bytes())


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload, pretty=False).rstrip(b"\n"))


def format_time(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: str) -> datetime:
    return ethereum._parse_datetime(value)


def _hex_blob(value: Any, field: str) -> str:
    if not isinstance(value, str) or HEX_BLOB.fullmatch(value) is None:
        raise ValueError(f"{field} must be even-length hexadecimal data")
    return value.lower()


def _sha_runtime_code(value: Any, field: str) -> tuple[int, str]:
    code = _hex_blob(value, field)
    raw = bytes.fromhex(code[2:])
    return len(raw), sha256_bytes(raw)


def _load_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise RuntimeError("GIPR preregistration artifact hash changed")
    if sha256_file(PREREGISTRATION_PRODUCER_PATH) != (
        PREREGISTRATION_PRODUCER_SHA256
    ):
        raise RuntimeError("GIPR preregistration producer hash changed")
    if sha256_file(CONTRACT_PATH) != CONTRACT_SHA256:
        raise RuntimeError("GIPR implementation contract hash changed")
    if sha256_file(ETHEREUM_HELPER_PATH) != ETHEREUM_HELPER_SHA256:
        raise RuntimeError("GIPR Ethereum helper hash changed")
    payload = json.loads(
        repository_path(PREREGISTRATION_PATH).read_text(encoding="utf-8")
    )
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("GIPR preregistration manifest changed")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if prereg.canonical_hash(core) != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("GIPR preregistration is not canonically bound")
    return payload


def _write_once_bytes(path: str | Path, raw: bytes) -> Path:
    destination = _safe_destination(path, create_parent=True)
    if os.path.lexists(destination):
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != raw
        ):
            raise RuntimeError(f"conflicting existing GIPR artifact: {path}")
        return destination
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(destination, flags, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except FileExistsError:
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != raw
        ):
            raise RuntimeError(
                f"conflicting existing GIPR artifact: {path}"
            ) from None
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        destination.unlink(missing_ok=True)
        raise
    return destination


def deterministic_gzip(raw: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as handle:
        handle.write(raw)
    return buffer.getvalue()


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            dict(row),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def csv_bytes(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return text.getvalue().encode("utf-8")


def _word(data: bytes, offset: int, field: str) -> bytes:
    if offset < 0 or offset % 32 != 0 or offset + 32 > len(data):
        raise ValueError(f"{field} word offset is invalid")
    return data[offset : offset + 32]


def _uint(data: bytes, offset: int, field: str) -> int:
    return int.from_bytes(_word(data, offset, field), byteorder="big")


def _word_address(data: bytes, offset: int, field: str) -> str:
    raw = _word(data, offset, field)
    if raw[:12] != b"\x00" * 12:
        raise ValueError(f"{field} is not a canonical left-padded address")
    return "0x" + raw[12:].hex()


def _dynamic_offset(data: bytes, head_offset: int, field: str) -> int:
    offset = _uint(data, head_offset, field)
    if offset % 32 != 0:
        raise ValueError(f"{field} dynamic offset is not 32-byte aligned")
    if offset < 9 * 32 or offset + 32 > len(data):
        raise ValueError(f"{field} dynamic offset is outside the ABI tail")
    return offset


def _decode_bytes_at(
    data: bytes,
    offset: int,
    *,
    maximum_bytes: int,
    field: str,
) -> tuple[bytes, tuple[int, int, str]]:
    length = _uint(data, offset, f"{field} length")
    if length > maximum_bytes:
        raise ValueError(f"{field} exceeds frozen byte bound")
    padded = ((length + 31) // 32) * 32
    end = offset + 32 + padded
    if end > len(data):
        raise ValueError(f"{field} tail exceeds event data")
    raw = data[offset + 32 : offset + 32 + length]
    padding = data[offset + 32 + length : end]
    if any(padding):
        raise ValueError(f"{field} has nonzero ABI padding")
    return raw, (offset, end, field)


def _decode_address_array(
    data: bytes,
    offset: int,
    field: str,
) -> tuple[tuple[str, ...], list[tuple[int, int, str]]]:
    length = _uint(data, offset, f"{field} length")
    end = offset + 32 + length * 32
    if end > len(data):
        raise ValueError(f"{field} array exceeds event data")
    values = tuple(
        _word_address(data, offset + 32 + index * 32, f"{field}[{index}]")
        for index in range(length)
    )
    return values, [(offset, end, field)]


def _decode_uint_array(
    data: bytes,
    offset: int,
    field: str,
) -> tuple[tuple[int, ...], list[tuple[int, int, str]]]:
    length = _uint(data, offset, f"{field} length")
    end = offset + 32 + length * 32
    if end > len(data):
        raise ValueError(f"{field} array exceeds event data")
    values = tuple(
        _uint(data, offset + 32 + index * 32, f"{field}[{index}]")
        for index in range(length)
    )
    return values, [(offset, end, field)]


def _decode_dynamic_bytes_array(
    data: bytes,
    offset: int,
    *,
    maximum_item_bytes: int,
    field: str,
) -> tuple[tuple[bytes, ...], list[tuple[int, int, str]]]:
    length = _uint(data, offset, f"{field} length")
    base = offset + 32
    head_end = base + length * 32
    if head_end > len(data):
        raise ValueError(f"{field} offset table exceeds event data")
    intervals: list[tuple[int, int, str]] = [(offset, head_end, f"{field} head")]
    values: list[bytes] = []
    for index in range(length):
        relative = _uint(
            data,
            base + index * 32,
            f"{field}[{index}] relative offset",
        )
        if relative % 32 != 0 or relative < length * 32:
            raise ValueError(f"{field}[{index}] dynamic offset is invalid")
        item_offset = base + relative
        raw, interval = _decode_bytes_at(
            data,
            item_offset,
            maximum_bytes=maximum_item_bytes,
            field=f"{field}[{index}]",
        )
        values.append(raw)
        intervals.append(interval)
    return tuple(values), intervals


def _validate_abi_coverage(
    data: bytes,
    intervals: Iterable[tuple[int, int, str]],
) -> None:
    ordered = sorted(intervals, key=lambda item: (item[0], item[1], item[2]))
    cursor = 0
    for start, end, field in ordered:
        if start != cursor:
            relation = "overlap" if start < cursor else "gap"
            raise ValueError(f"GIPR dynamic ABI has {relation} before {field}")
        if end <= start:
            raise ValueError(f"GIPR dynamic ABI interval is empty: {field}")
        cursor = end
    if cursor != len(data):
        raise ValueError("GIPR dynamic ABI has trailing unparsed bytes")


def decode_proposal_created(data_hex: str) -> dict[str, Any]:
    normalized = _hex_blob(data_hex, "ProposalCreated data")
    data = bytes.fromhex(normalized[2:])
    if len(data) < 9 * 32 or len(data) % 32 != 0:
        raise ValueError("ProposalCreated data has an invalid ABI length")
    proposal_id = _uint(data, 0, "proposal id")
    proposer = _word_address(data, 32, "proposer")
    targets_offset = _dynamic_offset(data, 64, "targets")
    values_offset = _dynamic_offset(data, 96, "values")
    signatures_offset = _dynamic_offset(data, 128, "signatures")
    calldatas_offset = _dynamic_offset(data, 160, "calldatas")
    start_block = _uint(data, 192, "start block")
    end_block = _uint(data, 224, "end block")
    description_offset = _dynamic_offset(data, 256, "description")
    if end_block <= start_block:
        raise ValueError("ProposalCreated end block must exceed start block")

    targets, target_intervals = _decode_address_array(
        data, targets_offset, "targets"
    )
    values, value_intervals = _decode_uint_array(data, values_offset, "values")
    signature_bytes, signature_intervals = _decode_dynamic_bytes_array(
        data,
        signatures_offset,
        maximum_item_bytes=prereg.MAX_SIGNATURE_BYTES,
        field="signatures",
    )
    calldata_bytes, calldata_intervals = _decode_dynamic_bytes_array(
        data,
        calldatas_offset,
        maximum_item_bytes=prereg.MAX_CALLDATA_BYTES_PER_ACTION,
        field="calldatas",
    )
    description_bytes, description_interval = _decode_bytes_at(
        data,
        description_offset,
        maximum_bytes=prereg.MAX_DESCRIPTION_BYTES,
        field="description",
    )
    lengths = {
        len(targets),
        len(values),
        len(signature_bytes),
        len(calldata_bytes),
    }
    if len(lengths) != 1:
        raise ValueError("ProposalCreated action arrays have unequal lengths")
    action_count = len(targets)
    if not 1 <= action_count <= prereg.MAX_ACTIONS_PER_PROPOSAL:
        raise ValueError("ProposalCreated action count is outside frozen bounds")
    if sum(len(raw) for raw in calldata_bytes) > (
        prereg.MAX_CALLDATA_BYTES_PER_PROPOSAL
    ):
        raise ValueError("ProposalCreated aggregate calldata exceeds frozen bound")
    try:
        signatures = tuple(raw.decode("utf-8", errors="strict") for raw in signature_bytes)
        description = description_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("ProposalCreated text is not strict UTF-8") from exc
    if "\x00" in description or any("\x00" in value for value in signatures):
        raise ValueError("ProposalCreated text contains NUL")
    intervals = [
        (0, 9 * 32, "top-level head"),
        *target_intervals,
        *value_intervals,
        *signature_intervals,
        *calldata_intervals,
        description_interval,
    ]
    _validate_abi_coverage(data, intervals)
    return {
        "proposal_id": proposal_id,
        "proposer": proposer,
        "targets": targets,
        "values": values,
        "signatures": signatures,
        "calldatas": tuple("0x" + raw.hex() for raw in calldata_bytes),
        "start_block": start_block,
        "end_block": end_block,
        "description": description,
    }


def decode_lifecycle_event(event: str, data_hex: str) -> dict[str, int | None]:
    normalized = _hex_blob(data_hex, f"{event} data")
    data = bytes.fromhex(normalized[2:])
    expected_words = 2 if event == "proposal_queued" else 1
    if len(data) != expected_words * 32:
        raise ValueError(f"{event} data has an invalid fixed ABI length")
    return {
        "proposal_id": _uint(data, 0, f"{event} proposal id"),
        "queue_eta": (
            _uint(data, 32, "proposal queue eta")
            if event == "proposal_queued"
            else None
        ),
    }


def normalize_log(raw: Mapping[str, Any]) -> CanonicalLog:
    address = ethereum._address(raw.get("address"))
    governor = ADDRESS_TO_GOVERNOR.get(address)
    if governor is None:
        raise ValueError("GIPR log address is outside the frozen governor roster")
    raw_topics = raw.get("topics")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ValueError("GIPR log has no event topics")
    topics = tuple(
        ethereum._hash(value, "GIPR event topic") for value in raw_topics
    )
    spec = TOPIC_TO_EVENT.get(topics[0])
    if spec is None:
        raise ValueError("GIPR log topic is outside the frozen event roster")
    if len(topics) != spec.topic_count:
        raise ValueError("GIPR log topic count differs from frozen ABI")
    if raw.get("removed") is not False:
        raise ValueError("GIPR log is removed or has ambiguous reorg state")
    block_number = ethereum._quantity(
        raw.get("blockNumber"), "GIPR event block number"
    )
    if not FIRST_LOG_BLOCK <= block_number <= LAST_LOG_BLOCK:
        raise ValueError("GIPR log is outside the frozen source envelope")
    if block_number < governor.first_code_block:
        raise ValueError("GIPR log precedes governor deployment")
    return CanonicalLog(
        address=address,
        topics=topics,
        data=_hex_blob(raw.get("data"), "GIPR event data"),
        removed=False,
        block_number=block_number,
        block_hash=ethereum._hash(raw.get("blockHash"), "GIPR event block hash"),
        transaction_hash=ethereum._hash(
            raw.get("transactionHash"), "GIPR transaction hash"
        ),
        transaction_index=ethereum._quantity(
            raw.get("transactionIndex"), "GIPR transaction index"
        ),
        log_index=ethereum._quantity(raw.get("logIndex"), "GIPR log index"),
    )


def normalize_logs(raw_logs: Sequence[Mapping[str, Any]]) -> list[CanonicalLog]:
    rows = [normalize_log(row) for row in raw_logs]
    identities: set[tuple[str, str, int]] = set()
    for row in rows:
        if row.canonical_identity in identities:
            raise ValueError("GIPR logs contain duplicate canonical identities")
        identities.add(row.canonical_identity)
    rows.sort(key=lambda row: row.order_key)
    return rows


def canonical_log_from_dict(raw: Mapping[str, Any]) -> CanonicalLog:
    expected = {
        "address",
        "topics",
        "data",
        "removed",
        "block_number",
        "block_hash",
        "transaction_hash",
        "transaction_index",
        "log_index",
    }
    if set(raw) != expected:
        raise RuntimeError("GIPR cached log schema changed")
    for field in ("block_number", "transaction_index", "log_index"):
        if type(raw[field]) is not int or raw[field] < 0:
            raise RuntimeError("GIPR cached log quantity is noncanonical")
    topics = raw["topics"]
    if not isinstance(topics, list):
        raise RuntimeError("GIPR cached log topics are noncanonical")
    reconstructed = {
        "address": raw["address"],
        "topics": topics,
        "data": raw["data"],
        "removed": raw["removed"],
        "blockNumber": hex(raw["block_number"]),
        "blockHash": raw["block_hash"],
        "transactionHash": raw["transaction_hash"],
        "transactionIndex": hex(raw["transaction_index"]),
        "logIndex": hex(raw["log_index"]),
    }
    canonical = normalize_log(reconstructed)
    if canonical.canonical_dict() != dict(raw):
        raise RuntimeError("GIPR cached log bytes are noncanonical")
    return canonical


def parse_log(row: CanonicalLog) -> NormalizedEvent:
    governor = ADDRESS_TO_GOVERNOR[row.address]
    event = TOPIC_TO_EVENT[row.topics[0]].event
    if event == "proposal_created":
        decoded = decode_proposal_created(row.data)
        return NormalizedEvent(
            protocol=governor.protocol,
            generation=governor.generation,
            governor_address=row.address,
            event=event,
            proposal_id=int(decoded["proposal_id"]),
            proposer=str(decoded["proposer"]),
            targets=tuple(decoded["targets"]),
            values=tuple(decoded["values"]),
            signatures=tuple(decoded["signatures"]),
            calldatas=tuple(decoded["calldatas"]),
            start_block=int(decoded["start_block"]),
            end_block=int(decoded["end_block"]),
            description=str(decoded["description"]),
            queue_eta=None,
            block_number=row.block_number,
            block_hash=row.block_hash,
            transaction_hash=row.transaction_hash,
            transaction_index=row.transaction_index,
            log_index=row.log_index,
        )
    decoded_lifecycle = decode_lifecycle_event(event, row.data)
    return NormalizedEvent(
        protocol=governor.protocol,
        generation=governor.generation,
        governor_address=row.address,
        event=event,
        proposal_id=int(decoded_lifecycle["proposal_id"]),
        proposer="",
        targets=(),
        values=(),
        signatures=(),
        calldatas=(),
        start_block=None,
        end_block=None,
        description="",
        queue_eta=(
            None
            if decoded_lifecycle["queue_eta"] is None
            else int(decoded_lifecycle["queue_eta"])
        ),
        block_number=row.block_number,
        block_hash=row.block_hash,
        transaction_hash=row.transaction_hash,
        transaction_index=row.transaction_index,
        log_index=row.log_index,
    )


def parse_logs(rows: Sequence[CanonicalLog]) -> list[NormalizedEvent]:
    events = [parse_log(row) for row in rows]
    events.sort(key=lambda row: row.chain_order_key)
    return events


class LogCheckpoint:
    """Protocol-bound cache for completed GIPR eth_getLogs ranges."""

    def __init__(self, path: str | Path) -> None:
        destination = _safe_destination(path, create_parent=True)
        if destination.is_symlink():
            raise RuntimeError("GIPR checkpoint may not be a symlink")
        self.path = destination
        self.connection = sqlite3.connect(destination)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS log_chunks (
                role TEXT NOT NULL,
                first_block INTEGER NOT NULL,
                last_block INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(role, first_block, last_block)
            )
            """
        )
        source_identity = canonical_hash(
            {
                "cache_protocol": "gipr_d1_canonical_log_cache_v1",
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
                "addresses": sorted(ADDRESS_TO_GOVERNOR),
                "topics": sorted(TOPIC_TO_EVENT),
                "first_log_block": FIRST_LOG_BLOCK,
                "last_log_block": LAST_LOG_BLOCK,
                "max_block_range": prereg.MAX_BLOCK_RANGE,
            }
        )
        self.source_identity = source_identity
        existing = self.connection.execute(
            "SELECT value FROM metadata WHERE key='source_identity'"
        ).fetchone()
        if existing is not None and existing[0] != source_identity:
            self.connection.close()
            raise RuntimeError("GIPR checkpoint belongs to another protocol")
        self.connection.execute(
            "INSERT OR IGNORE INTO metadata(key,value) "
            "VALUES('source_identity',?)",
            (source_identity,),
        )
        self.connection.commit()

    def get(
        self,
        role: str,
        first_block: int,
        last_block: int,
    ) -> list[CanonicalLog] | None:
        found = self.connection.execute(
            "SELECT payload_json FROM log_chunks "
            "WHERE role=? AND first_block=? AND last_block=?",
            (role, first_block, last_block),
        ).fetchone()
        if found is None:
            return None
        payload = json.loads(found[0])
        if (
            not isinstance(payload, dict)
            or set(payload) != {"request_hash", "row_hash", "rows"}
            or payload.get("request_hash")
            != self.request_hash(role, first_block, last_block)
            or not isinstance(payload.get("rows"), list)
            or any(not isinstance(row, dict) for row in payload["rows"])
            or payload.get("row_hash")
            != canonical_hash(payload["rows"])
        ):
            raise RuntimeError("GIPR checkpoint payload is malformed")
        rows = [
            canonical_log_from_dict(row) for row in payload["rows"]
        ]
        if any(
            not first_block <= row.block_number <= last_block
            for row in rows
        ):
            raise RuntimeError("GIPR cached log is outside its request chunk")
        if rows != sorted(rows, key=lambda row: row.order_key):
            raise RuntimeError("GIPR cached log order is noncanonical")
        return rows

    def request_hash(
        self,
        role: str,
        first_block: int,
        last_block: int,
    ) -> str:
        return canonical_hash(
            {
                "source_identity": self.source_identity,
                "role": role,
                "from_block": first_block,
                "to_block": last_block,
                "addresses": sorted(ADDRESS_TO_GOVERNOR),
                "topics": sorted(TOPIC_TO_EVENT),
            }
        )

    def put(
        self,
        role: str,
        first_block: int,
        last_block: int,
        rows: Sequence[CanonicalLog],
    ) -> None:
        if any(
            not first_block <= row.block_number <= last_block
            for row in rows
        ):
            raise RuntimeError("GIPR log is outside its requested cache chunk")
        ordered = sorted(rows, key=lambda row: row.order_key)
        if list(rows) != ordered:
            raise RuntimeError("GIPR cache input order is noncanonical")
        canonical_rows = [row.canonical_dict() for row in rows]
        payload = {
            "request_hash": self.request_hash(
                role,
                first_block,
                last_block,
            ),
            "row_hash": canonical_hash(canonical_rows),
            "rows": canonical_rows,
        }
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        with self.connection:
            existing = self.connection.execute(
                "SELECT payload_json FROM log_chunks "
                "WHERE role=? AND first_block=? AND last_block=?",
                (role, first_block, last_block),
            ).fetchone()
            if existing is not None:
                if existing[0] != raw:
                    raise RuntimeError(
                        "GIPR checkpoint chunk conflicts with frozen cache"
                    )
                return
            self.connection.execute(
                "INSERT INTO log_chunks"
                "(role,first_block,last_block,payload_json) VALUES(?,?,?,?)",
                (role, first_block, last_block, raw),
            )

    def close(self) -> None:
        self.connection.close()


def fetch_logs(
    rpc: ethereum.Rpc,
    *,
    role: str,
    checkpoint: LogCheckpoint | None,
    max_block_range: int,
) -> list[CanonicalLog]:
    if max_block_range <= 0 or max_block_range > prereg.MAX_BLOCK_RANGE:
        raise ValueError("GIPR max block range is outside frozen bounds")
    addresses = sorted(ADDRESS_TO_GOVERNOR)
    topics = sorted(TOPIC_TO_EVENT)
    rows: list[CanonicalLog] = []
    for first in range(FIRST_LOG_BLOCK, LAST_LOG_BLOCK + 1, max_block_range):
        last = min(first + max_block_range - 1, LAST_LOG_BLOCK)
        canonical = checkpoint.get(role, first, last) if checkpoint else None
        if canonical is None:
            result = rpc.call(
                "eth_getLogs",
                [
                    {
                        "fromBlock": hex(first),
                        "toBlock": hex(last),
                        "address": addresses,
                        "topics": [topics],
                    }
                ],
            )
            if not isinstance(result, list):
                raise RuntimeError("GIPR eth_getLogs result is not a list")
            if any(not isinstance(row, Mapping) for row in result):
                raise RuntimeError("GIPR eth_getLogs row is not an object")
            canonical = normalize_logs(result)
            if any(
                not first <= row.block_number <= last for row in canonical
            ):
                raise RuntimeError(
                    "GIPR provider returned a log outside its request chunk"
                )
            if checkpoint:
                checkpoint.put(role, first, last, canonical)
        rows.extend(canonical)
    identities: set[tuple[str, str, int]] = set()
    for row in rows:
        if row.canonical_identity in identities:
            raise ValueError(
                "GIPR logs contain duplicate cross-chunk identities"
            )
        identities.add(row.canonical_identity)
    rows.sort(key=lambda row: row.order_key)
    return rows


def _header_dict(header: ethereum.BlockHeader) -> dict[str, Any]:
    return {
        "number": header.number,
        "block_hash": header.block_hash,
        "parent_hash": header.parent_hash,
        "timestamp": header.timestamp,
    }


def _fetch_headers(
    rpc: ethereum.Rpc,
    block_numbers: Sequence[int],
    *,
    batch_size: int,
) -> dict[int, ethereum.BlockHeader]:
    if batch_size <= 0 or batch_size > prereg.HEADER_BATCH_SIZE:
        raise ValueError("GIPR header batch size is outside frozen bounds")
    unique = sorted(set(block_numbers))
    result: dict[int, ethereum.BlockHeader] = {}
    for offset in range(0, len(unique), batch_size):
        batch = unique[offset : offset + batch_size]
        payloads = rpc.batch(
            [
                ("eth_getBlockByNumber", [hex(block_number), False])
                for block_number in batch
            ]
        )
        for block_number, payload in zip(batch, payloads, strict=True):
            result[block_number] = ethereum.parse_header(
                payload, expected_number=block_number
            )
    return result


def materialize_events(
    events: Sequence[NormalizedEvent],
    primary_headers: Mapping[int, ethereum.BlockHeader],
    verification_headers: Mapping[int, ethereum.BlockHeader],
) -> list[NormalizedEvent]:
    required = {
        number
        for row in events
        for number in (
            row.block_number,
            row.block_number + prereg.CONFIRMATION_BLOCKS,
        )
    }
    if set(primary_headers) != required or set(verification_headers) != required:
        raise RuntimeError("GIPR header maps do not cover exact required blocks")
    rows: list[NormalizedEvent] = []
    for row in events:
        source_header = primary_headers[row.block_number]
        confirmation_number = row.block_number + prereg.CONFIRMATION_BLOCKS
        confirmation_header = primary_headers[confirmation_number]
        if _header_dict(source_header) != _header_dict(
            verification_headers[row.block_number]
        ):
            raise RuntimeError("GIPR event header differs between transports")
        if _header_dict(confirmation_header) != _header_dict(
            verification_headers[confirmation_number]
        ):
            raise RuntimeError(
                "GIPR confirmation header differs between transports"
            )
        if row.block_hash != source_header.block_hash:
            raise RuntimeError("GIPR event block hash differs from canonical header")
        if confirmation_header.timestamp < source_header.timestamp:
            raise RuntimeError("GIPR confirmation timestamp precedes event timestamp")
        rows.append(
            replace(
                row,
                confirmation_block_number=confirmation_number,
                confirmation_block_hash=confirmation_header.block_hash,
                event_at=datetime.fromtimestamp(source_header.timestamp, tz=UTC),
                available_at=datetime.fromtimestamp(
                    confirmation_header.timestamp, tz=UTC
                ),
            )
        )
    rows.sort(key=lambda row: row.chain_order_key)
    return rows


def audit_boundary(
    primary: ethereum.Rpc,
    verification: ethereum.Rpc,
) -> dict[str, Any]:
    transports = (primary, verification)
    chain_ids = [
        ethereum._quantity(rpc.call("eth_chainId", []), "Ethereum chain id")
        for rpc in transports
    ]
    if chain_ids != [prereg.CHAIN_ID, prereg.CHAIN_ID]:
        raise RuntimeError("GIPR transport chain ID differs from frozen chain")
    boundaries: list[dict[str, Any]] = []
    for boundary in prereg.YEAR_BOUNDARIES:
        headers = [
            ethereum.get_header(rpc, boundary.first_block) for rpc in transports
        ]
        expected_timestamp = int(parse_time(boundary.observed_block_timestamp).timestamp())
        expected = {
            "number": boundary.first_block,
            "block_hash": boundary.block_hash,
            "timestamp": expected_timestamp,
        }
        for header in headers:
            if (
                header.number != expected["number"]
                or header.block_hash != expected["block_hash"]
                or header.timestamp != expected["timestamp"]
            ):
                raise RuntimeError("GIPR frozen year boundary changed")
        if _header_dict(headers[0]) != _header_dict(headers[1]):
            raise RuntimeError("GIPR year boundary differs between transports")
        boundaries.append(expected)
    codes: list[dict[str, Any]] = []
    for governor in prereg.GOVERNORS:
        provider_rows: list[dict[str, Any]] = []
        for rpc in transports:
            prior = rpc.call(
                "eth_getCode",
                [governor.address, hex(governor.first_code_block - 1)],
            )
            if _hex_blob(prior, "GIPR prior runtime code") != "0x":
                raise RuntimeError("GIPR governor code exists before first-code block")
            first = _sha_runtime_code(
                rpc.call(
                    "eth_getCode",
                    [governor.address, hex(governor.first_code_block)],
                ),
                "GIPR first runtime code",
            )
            last = _sha_runtime_code(
                rpc.call(
                    "eth_getCode",
                    [governor.address, hex(prereg.LAST_SOURCE_BLOCK)],
                ),
                "GIPR last-source runtime code",
            )
            expected_first = (
                governor.runtime_code_bytes,
                governor.runtime_code_sha256,
            )
            expected_last = (
                governor.last_source_code_bytes,
                governor.last_source_code_sha256,
            )
            if first != expected_first or last != expected_last:
                raise RuntimeError("GIPR governor runtime code changed")
            provider_rows.append(
                {
                    "first_code_bytes": first[0],
                    "first_code_sha256": first[1],
                    "last_code_bytes": last[0],
                    "last_code_sha256": last[1],
                }
            )
        if provider_rows[0] != provider_rows[1]:
            raise RuntimeError("GIPR governor runtime code differs by transport")
        codes.append(
            {
                "protocol": governor.protocol,
                "generation": governor.generation,
                "address": governor.address,
                **provider_rows[0],
            }
        )
    heads = [
        ethereum._quantity(
            rpc.call("eth_blockNumber", []), "Ethereum finalized source head"
        )
        for rpc in transports
    ]
    minimum_head = prereg.LAST_SOURCE_BLOCK + prereg.CONFIRMATION_BLOCKS
    if any(head < minimum_head for head in heads):
        raise RuntimeError("GIPR transport head lacks final confirmation coverage")
    return {
        "chain_id": prereg.CHAIN_ID,
        "year_boundaries": boundaries,
        "governor_code": codes,
        "minimum_confirmation_head": minimum_head,
        "transport_heads_at_or_above_minimum": [head >= minimum_head for head in heads],
        "event_logs_opened": False,
    }


def validate_lifecycle(
    events: Sequence[NormalizedEvent],
) -> dict[tuple[str, int], tuple[NormalizedEvent, ...]]:
    by_proposal: dict[tuple[str, int], list[NormalizedEvent]] = defaultdict(list)
    states: dict[tuple[str, int], str] = {}
    seen_events: set[tuple[tuple[str, int], str]] = set()
    for row in sorted(events, key=lambda value: value.chain_order_key):
        key = row.proposal_key
        identity = (key, row.event)
        if identity in seen_events:
            raise ValueError("GIPR lifecycle contains a duplicate event")
        seen_events.add(identity)
        current = states.get(key)
        if row.event == "proposal_created":
            if current is not None:
                raise ValueError("GIPR proposal is created more than once")
            states[key] = "proposal_created"
        else:
            if current is None:
                raise ValueError("GIPR lifecycle event precedes proposal creation")
            allowed = prereg.LIFECYCLE_TRANSITIONS[current]
            if row.event not in allowed:
                raise ValueError(
                    f"GIPR invalid lifecycle transition {current}->{row.event}"
                )
            states[key] = row.event
        by_proposal[key].append(row)
    return {
        key: tuple(sorted(rows, key=lambda value: value.chain_order_key))
        for key, rows in by_proposal.items()
    }


def split_name(value: datetime | None) -> str | None:
    if value is None:
        return None
    for split in prereg.SPLITS:
        start = parse_time(split["start"])
        end = parse_time(split["end_exclusive"])
        if start <= value < end:
            return str(split["name"])
    return None


def first_daily_boundary(value: datetime) -> datetime:
    normalized = value.astimezone(UTC)
    midnight = normalized.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight if normalized == midnight else midnight + DAY


def decision_times() -> tuple[datetime, ...]:
    rows: list[datetime] = []
    current = FIRST_DECISION
    while current <= LAST_DECISION:
        rows.append(current)
        current += DAY
    return tuple(rows)


def action_count_bucket(count: int) -> str:
    if count == 1:
        return "ONE"
    if count == 2:
        return "TWO"
    if 3 <= count <= 5:
        return "THREE_TO_FIVE"
    if 6 <= count <= 10:
        return "SIX_TO_TEN"
    raise ValueError("GIPR action count is outside frozen buckets")


def activity_bucket(count: int) -> str:
    if count == 0:
        return "ZERO"
    if count == 1:
        return "ONE"
    if count > 1:
        return "TWO_PLUS"
    raise ValueError("GIPR activity count cannot be negative")


def age_bucket(age: timedelta | None) -> str:
    if age is None or age < timedelta(0):
        return "STALE_OR_NONE"
    if age < timedelta(days=2):
        return "D0_1"
    if age < timedelta(days=8):
        return "D2_7"
    if age < timedelta(days=29):
        return "D8_28"
    if age <= timedelta(days=90):
        return "D29_90"
    return "STALE_OR_NONE"


def selector_token(signature: str, calldata: str) -> str:
    if signature:
        return signature
    raw = bytes.fromhex(_hex_blob(calldata, "GIPR calldata")[2:])
    return "0x" + raw[:4].hex() if len(raw) >= 4 else "EMPTY_CALLDATA"


def proposal_features(proposal: NormalizedEvent) -> dict[str, Any]:
    if proposal.event != "proposal_created":
        raise ValueError("GIPR proposal feature input is not a creation event")
    roles = [KNOWN_TARGET_ROLES.get(target, "UNKNOWN_TARGET") for target in proposal.targets]
    known_count = sum(role != "UNKNOWN_TARGET" for role in roles)
    if known_count == 0:
        role_mix = "ALL_UNKNOWN"
    elif known_count == len(roles):
        role_mix = "KNOWN_ONLY"
    else:
        role_mix = "MIXED"
    selectors = tuple(
        selector_token(signature, calldata)
        for signature, calldata in zip(
            proposal.signatures, proposal.calldatas, strict=True
        )
    )
    selector_count = len(set(selectors))
    if selector_count == 1:
        selector_bucket = "ONE"
    elif selector_count == 2:
        selector_bucket = "TWO"
    elif selector_count >= 3:
        selector_bucket = "THREE_PLUS"
    else:
        raise ValueError("GIPR proposal has no selector")
    nonzero = sum(value != 0 for value in proposal.values)
    if nonzero == 0:
        value_bucket = "ZERO"
    elif nonzero == len(proposal.values):
        value_bucket = "ALL_NONZERO"
    else:
        value_bucket = "MIXED"
    description_raw = proposal.description.encode("utf-8")
    if not description_raw:
        length_bucket = "EMPTY"
    elif len(description_raw) <= 256:
        length_bucket = "SHORT"
    elif len(description_raw) <= 2_048:
        length_bucket = "MEDIUM"
    else:
        length_bucket = "LONG"
    lowered = proposal.description.lower()
    url_bucket = (
        "HAS_URL"
        if "http://" in lowered or "https://" in lowered
        else "NO_URL"
    )
    heading_bucket = (
        "HAS_MARKDOWN_HEADING"
        if MARKDOWN_HEADING.search(proposal.description)
        else "NO_MARKDOWN_HEADING"
    )
    payload_core = {
        "targets": list(proposal.targets),
        "values": [str(value) for value in proposal.values],
        "signatures": list(proposal.signatures),
        "calldatas": list(proposal.calldatas),
    }
    return {
        "action_count_bucket": action_count_bucket(len(proposal.targets)),
        "target_roles": roles,
        "target_role_mix": role_mix,
        "selectors": selectors,
        "selector_count_bucket": selector_bucket,
        "native_value_bucket": value_bucket,
        "description_structure": "|".join(
            (length_bucket, url_bucket, heading_bucket)
        ),
        "description_sha256": sha256_bytes(description_raw),
        "payload_sha256": canonical_hash(payload_core),
        "pair_sha256": canonical_hash(
            {
                "description_sha256": sha256_bytes(description_raw),
                "payload_sha256": canonical_hash(payload_core),
            }
        ),
    }


def _event_day(value: NormalizedEvent) -> datetime:
    if value.available_at is None:
        raise RuntimeError("GIPR event is missing available_at")
    return first_daily_boundary(value.available_at)


def _same_day_order_fingerprints(
    events: Sequence[NormalizedEvent],
    *,
    reverse_within_day: bool = False,
) -> dict[datetime, str]:
    by_day: dict[datetime, list[NormalizedEvent]] = defaultdict(list)
    for event in events:
        day = _event_day(event)
        if FIRST_DECISION <= day <= LAST_DECISION:
            by_day[day].append(event)
    fingerprints: dict[datetime, str] = {}
    for day, values in by_day.items():
        ordered = sorted(values, key=lambda row: row.availability_order_key)
        if reverse_within_day:
            ordered = list(reversed(ordered))
        fingerprints[day] = canonical_hash(
            [
                {
                    "protocol": row.protocol,
                    "generation": row.generation,
                    "event": row.event,
                    "proposal_identity_sha256": canonical_hash(
                        [row.governor_address, str(row.proposal_id)]
                    ),
                }
                for row in ordered
            ]
        )
    return fingerprints


def _latest_lifecycle_by_decision(
    proposal: NormalizedEvent,
    lifecycle: Mapping[tuple[str, int], Sequence[NormalizedEvent]],
    decision: datetime,
) -> NormalizedEvent:
    eligible = [
        row
        for row in lifecycle[proposal.proposal_key]
        if row.available_at is not None and row.available_at <= decision
    ]
    if not eligible:
        raise RuntimeError("GIPR proposal creation is absent from its lifecycle")
    return max(eligible, key=lambda row: row.availability_order_key)


def _protocol_card(
    protocol: str,
    decision: datetime,
    proposals: Sequence[NormalizedEvent],
    lifecycle: Mapping[tuple[str, int], Sequence[NormalizedEvent]],
    same_day_fingerprint: str,
) -> tuple[dict[str, str], dict[str, int]]:
    eligible = [
        row
        for row in proposals
        if row.protocol == protocol
        and row.available_at is not None
        and row.available_at <= decision
    ]
    counts: dict[str, int] = {}
    for days in (1, 7, 28, 90):
        lower = decision - timedelta(days=days)
        counts[f"activity_{days}d"] = sum(
            row.available_at is not None and lower < row.available_at <= decision
            for row in eligible
        )
    latest = (
        max(
            eligible,
            key=lambda row: (
                row.available_at,
                row.block_number,
                row.transaction_index,
                row.log_index,
                row.governor_address,
                row.proposal_id,
            ),
        )
        if eligible
        else None
    )
    proposal_age = (
        None
        if latest is None or latest.available_at is None
        else decision - latest.available_at
    )
    stale = latest is None or proposal_age is None or proposal_age > timedelta(days=90)
    if stale:
        return (
            {
                "protocol": protocol.upper(),
                "activity_1d": activity_bucket(counts["activity_1d"]),
                "activity_7d": activity_bucket(counts["activity_7d"]),
                "activity_28d": activity_bucket(counts["activity_28d"]),
                "activity_90d": activity_bucket(counts["activity_90d"]),
                "proposal_age_bucket": "STALE_OR_NONE",
                "lifecycle_state": "STALE_OR_NO_PROPOSAL",
                "lifecycle_age_bucket": "STALE_OR_NONE",
                "action_count_bucket": "UNAVAILABLE",
                "target_role_mix": "UNAVAILABLE",
                "selector_count_bucket": "UNAVAILABLE",
                "native_value_bucket": "UNAVAILABLE",
                "description_structure": "UNAVAILABLE",
                "same_day_event_order_fingerprint": same_day_fingerprint,
            },
            counts,
        )
    assert latest is not None and latest.available_at is not None
    latest_lifecycle = _latest_lifecycle_by_decision(
        latest, lifecycle, decision
    )
    if latest_lifecycle.available_at is None:
        raise RuntimeError("GIPR lifecycle row has no availability")
    features = proposal_features(latest)
    return (
        {
            "protocol": protocol.upper(),
            "activity_1d": activity_bucket(counts["activity_1d"]),
            "activity_7d": activity_bucket(counts["activity_7d"]),
            "activity_28d": activity_bucket(counts["activity_28d"]),
            "activity_90d": activity_bucket(counts["activity_90d"]),
            "proposal_age_bucket": age_bucket(proposal_age),
            "lifecycle_state": latest_lifecycle.event.upper(),
            "lifecycle_age_bucket": age_bucket(
                decision - latest_lifecycle.available_at
            ),
            "action_count_bucket": features["action_count_bucket"],
            "target_role_mix": features["target_role_mix"],
            "selector_count_bucket": features["selector_count_bucket"],
            "native_value_bucket": features["native_value_bucket"],
            "description_structure": features["description_structure"],
            "same_day_event_order_fingerprint": same_day_fingerprint,
        },
        counts,
    )


ACTION_COMPLEXITY = {
    "ONE": 1,
    "TWO": 2,
    "THREE_TO_FIVE": 3,
    "SIX_TO_TEN": 4,
}


def build_daily_cards(
    events: Sequence[NormalizedEvent],
    *,
    swap_protocol_labels: bool = False,
    reverse_within_day: bool = False,
    availability_shift: timedelta = timedelta(0),
) -> list[dict[str, Any]]:
    transformed = [
        replace(
            row,
            protocol=(
                ("uniswap" if row.protocol == "compound" else "compound")
                if swap_protocol_labels
                else row.protocol
            ),
            available_at=(
                None
                if row.available_at is None
                else row.available_at + availability_shift
            ),
        )
        for row in events
    ]
    lifecycle = validate_lifecycle(transformed)
    proposals = [row for row in transformed if row.event == "proposal_created"]
    order_fingerprints = _same_day_order_fingerprints(
        transformed, reverse_within_day=reverse_within_day
    )
    empty_order = canonical_hash([])
    cards: list[dict[str, Any]] = []
    previous_relations: tuple[str, str, str] | None = None
    for decision in decision_times():
        fingerprint = order_fingerprints.get(decision, empty_order)
        compound, compound_counts = _protocol_card(
            "compound", decision, proposals, lifecycle, fingerprint
        )
        uniswap, uniswap_counts = _protocol_card(
            "uniswap", decision, proposals, lifecycle, fingerprint
        )
        left_activity = compound_counts["activity_28d"]
        right_activity = uniswap_counts["activity_28d"]
        activity_relation = (
            "COMPOUND_GT"
            if left_activity > right_activity
            else "UNISWAP_GT"
            if right_activity > left_activity
            else "EQUAL"
        )
        compound_stale = (
            compound["lifecycle_state"] == "STALE_OR_NO_PROPOSAL"
        )
        uniswap_stale = (
            uniswap["lifecycle_state"] == "STALE_OR_NO_PROPOSAL"
        )
        if compound_stale and uniswap_stale:
            lifecycle_relation = "BOTH_STALE"
        elif compound_stale or uniswap_stale:
            lifecycle_relation = "ONE_STALE"
        else:
            lifecycle_relation = (
                "SAME"
                if compound["lifecycle_state"] == uniswap["lifecycle_state"]
                else "DIFFERENT"
            )
        if compound_stale or uniswap_stale:
            action_relation = "UNAVAILABLE"
        else:
            left = ACTION_COMPLEXITY[compound["action_count_bucket"]]
            right = ACTION_COMPLEXITY[uniswap["action_count_bucket"]]
            action_relation = (
                "COMPOUND_GT"
                if left > right
                else "UNISWAP_GT"
                if right > left
                else "EQUAL"
            )
        current_relations = (
            activity_relation,
            lifecycle_relation,
            action_relation,
        )
        transition = (
            "INITIAL"
            if previous_relations is None
            else "PERSIST"
            if current_relations == previous_relations
            else "FLIP"
        )
        relation = {
            "activity_28d_relation": activity_relation,
            "lifecycle_relation": lifecycle_relation,
            "action_complexity_relation": action_relation,
            "relation_transition": transition,
        }
        model_card = {
            "compound": compound,
            "uniswap": uniswap,
            "relation": relation,
        }
        cards.append(
            {
                "split": split_name(decision),
                "decision_at": format_time(decision),
                "model_card": model_card,
                "card_sha256": canonical_hash(model_card),
            }
        )
        previous_relations = current_relations
    return cards


def _split_events(
    events: Sequence[NormalizedEvent],
    *,
    event_type: str | None = None,
) -> dict[str, list[NormalizedEvent]]:
    rows = {str(split["name"]): [] for split in prereg.SPLITS}
    for event in events:
        if event_type is not None and event.event != event_type:
            continue
        name = split_name(event.available_at)
        if name is not None:
            rows[name].append(event)
    return rows


def _fraction_at_least(
    numerator: int,
    denominator: int,
    *,
    minimum_numerator: int,
    minimum_denominator: int,
) -> bool:
    return (
        denominator > 0
        and numerator * minimum_denominator
        >= denominator * minimum_numerator
    )


def _fraction_at_most(
    numerator: int,
    denominator: int,
    *,
    maximum_numerator: int,
    maximum_denominator: int,
) -> bool:
    return (
        denominator > 0
        and numerator * maximum_denominator
        <= denominator * maximum_numerator
    )


def split_support_metrics(
    events: Sequence[NormalizedEvent],
    cards: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    proposals_by_split = _split_events(events, event_type="proposal_created")
    events_by_split = _split_events(events)
    cards_by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for card in cards:
        if card.get("split") is not None:
            cards_by_split[str(card["split"])].append(card)
    metrics: dict[str, dict[str, Any]] = {}
    for split in prereg.SPLITS:
        name = str(split["name"])
        proposals = proposals_by_split[name]
        split_events = events_by_split[name]
        split_cards = cards_by_split[name]
        features = [proposal_features(row) for row in proposals]
        descriptions = [row.description for row in proposals]
        nonempty = sum(bool(value.encode("utf-8")) for value in descriptions)
        unique_descriptions = len(
            {sha256_bytes(value.encode("utf-8")) for value in descriptions}
        )
        targets = {target for row in proposals for target in row.targets}
        selectors = {
            selector
            for feature in features
            for selector in feature["selectors"]
        }
        active_months = {
            row.available_at.strftime("%Y-%m")
            for row in proposals
            if row.available_at is not None
        }
        protocol_counts = Counter(row.protocol for row in proposals)
        card_counts = Counter(str(row["card_sha256"]) for row in split_cards)
        complete_card_max = max(card_counts.values(), default=0)
        compound_stale = sum(
            row["model_card"]["compound"]["lifecycle_state"]
            == "STALE_OR_NO_PROPOSAL"
            for row in split_cards
        )
        uniswap_stale = sum(
            row["model_card"]["uniswap"]["lifecycle_state"]
            == "STALE_OR_NO_PROPOSAL"
            for row in split_cards
        )
        both_stale = sum(
            row["model_card"]["relation"]["lifecycle_relation"]
            == "BOTH_STALE"
            for row in split_cards
        )
        metrics[name] = {
            "proposals": len(proposals),
            "actions": sum(len(row.targets) for row in proposals),
            "active_calendar_months": len(active_months),
            "protocol_proposals": dict(sorted(protocol_counts.items())),
            "unique_targets": len(targets),
            "unique_selectors": len(selectors),
            "nonempty_descriptions": nonempty,
            "unique_descriptions": unique_descriptions,
            "action_count_buckets": sorted(
                {feature["action_count_bucket"] for feature in features}
            ),
            "target_role_mix_tokens": sorted(
                {feature["target_role_mix"] for feature in features}
            ),
            "selector_count_buckets": sorted(
                {feature["selector_count_bucket"] for feature in features}
            ),
            "lifecycle_event_types": sorted(
                {row.event for row in split_events}
            ),
            "daily_decisions": len(split_cards),
            "unique_complete_daily_cards": len(card_counts),
            "maximum_complete_daily_card_count": complete_card_max,
            "compound_stale_days": compound_stale,
            "uniswap_stale_days": uniswap_stale,
            "both_stale_days": both_stale,
        }
    return metrics


def gate_split_proposal_action_support(
    metrics: Mapping[str, Mapping[str, Any]],
) -> GateResult:
    failures: list[str] = []
    summary: dict[str, Any] = {}
    for split in prereg.SPLITS:
        name = str(split["name"])
        row = metrics[name]
        checks = {
            "proposals": int(row["proposals"]) >= split["minimum_proposals"],
            "actions": int(row["actions"]) >= split["minimum_actions"],
            "active_calendar_months": (
                int(row["active_calendar_months"])
                >= split["minimum_active_calendar_months"]
            ),
        }
        summary[name] = checks
        failures.extend(
            f"{name}:{field}" for field, passed in checks.items() if not passed
        )
    return GateResult(
        name="split_proposal_action_support",
        passed=not failures,
        metrics=summary,
        failure=";".join(failures),
    )


def gate_cross_protocol_support(
    metrics: Mapping[str, Mapping[str, Any]],
) -> GateResult:
    failures: list[str] = []
    summary: dict[str, Any] = {}
    for split in prereg.SPLITS:
        name = str(split["name"])
        counts = dict(metrics[name]["protocol_proposals"])
        minimum = int(split["minimum_proposals_per_protocol"])
        checks = {
            protocol: int(counts.get(protocol, 0)) >= minimum
            for protocol in ("compound", "uniswap")
        }
        summary[name] = checks
        failures.extend(
            f"{name}:{protocol}"
            for protocol, passed in checks.items()
            if not passed
        )
    return GateResult(
        name="cross_protocol_support",
        passed=not failures,
        metrics=summary,
        failure=";".join(failures),
    )


def gate_structural_vocabulary(
    metrics: Mapping[str, Mapping[str, Any]],
) -> GateResult:
    failures: list[str] = []
    summary: dict[str, Any] = {}
    for split in prereg.SPLITS:
        name = str(split["name"])
        row = metrics[name]
        proposal_count = int(row["proposals"])
        event_types = set(row["lifecycle_event_types"])
        checks = {
            "unique_targets": (
                int(row["unique_targets"]) >= split["minimum_unique_targets"]
            ),
            "unique_selectors": (
                int(row["unique_selectors"])
                >= split["minimum_unique_selectors"]
            ),
            "nonempty_description_fraction": _fraction_at_least(
                int(row["nonempty_descriptions"]),
                proposal_count,
                minimum_numerator=95,
                minimum_denominator=100,
            ),
            "unique_description_fraction": _fraction_at_least(
                int(row["unique_descriptions"]),
                proposal_count,
                minimum_numerator=90,
                minimum_denominator=100,
            ),
            "action_count_bucket_diversity": (
                len(row["action_count_buckets"]) >= 2
            ),
            "target_role_mix_diversity": (
                len(row["target_role_mix_tokens"]) >= 2
            ),
            "selector_count_bucket_diversity": (
                len(row["selector_count_buckets"]) >= 2
            ),
            "lifecycle_vocabulary": (
                "proposal_created" in event_types
                and len(event_types - {"proposal_created"}) >= 2
            ),
        }
        summary[name] = checks
        failures.extend(
            f"{name}:{field}" for field, passed in checks.items() if not passed
        )
    return GateResult(
        name="structural_vocabulary_diversity",
        passed=not failures,
        metrics=summary,
        failure=";".join(failures),
    )


def gate_daily_schedule(
    metrics: Mapping[str, Mapping[str, Any]],
) -> GateResult:
    failures: list[str] = []
    summary: dict[str, Any] = {}
    unique_minimum = {"train": 30, "test": 15, "eval": 15}
    for split in prereg.SPLITS:
        name = str(split["name"])
        row = metrics[name]
        days = int(row["daily_decisions"])
        checks = {
            "decision_count": (
                days >= split["minimum_daily_decisions_after_warmup"]
            ),
            "unique_complete_cards": (
                int(row["unique_complete_daily_cards"])
                >= unique_minimum[name]
            ),
            "maximum_complete_card_fraction": _fraction_at_most(
                int(row["maximum_complete_daily_card_count"]),
                days,
                maximum_numerator=35,
                maximum_denominator=100,
            ),
            "compound_stale_fraction": _fraction_at_most(
                int(row["compound_stale_days"]),
                days,
                maximum_numerator=75,
                maximum_denominator=100,
            ),
            "uniswap_stale_fraction": _fraction_at_most(
                int(row["uniswap_stale_days"]),
                days,
                maximum_numerator=75,
                maximum_denominator=100,
            ),
            "both_stale_fraction": _fraction_at_most(
                int(row["both_stale_days"]),
                days,
                maximum_numerator=60,
                maximum_denominator=100,
            ),
        }
        summary[name] = checks
        failures.extend(
            f"{name}:{field}" for field, passed in checks.items() if not passed
        )
    return GateResult(
        name="daily_schedule_coverage_and_staleness",
        passed=not failures,
        metrics=summary,
        failure=";".join(failures),
    )


def _card_hash_by_time(
    cards: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        str(row["decision_at"]): str(row["card_sha256"])
        for row in cards
    }


def _changed_by_split(
    baseline: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
    *,
    eligible_times: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    base = _card_hash_by_time(baseline)
    changed = _card_hash_by_time(control)
    if set(base) != set(changed):
        raise RuntimeError("GIPR control card schedules differ")
    metrics = {
        str(split["name"]): {"eligible": 0, "changed": 0}
        for split in prereg.SPLITS
    }
    for row in baseline:
        decision_at = str(row["decision_at"])
        if eligible_times is not None and decision_at not in eligible_times:
            continue
        name = str(row["split"])
        metrics[name]["eligible"] += 1
        metrics[name]["changed"] += int(base[decision_at] != changed[decision_at])
    return metrics


def _proposal_payload_fingerprint(
    proposal: NormalizedEvent,
    *,
    reverse_actions: bool = False,
) -> str:
    actions = list(
        zip(
            proposal.targets,
            proposal.values,
            proposal.signatures,
            proposal.calldatas,
            strict=True,
        )
    )
    if reverse_actions:
        actions.reverse()
    return canonical_hash(
        [
            {
                "target": target,
                "value": str(value),
                "signature": signature,
                "calldata": calldata,
            }
            for target, value, signature, calldata in actions
        ]
    )


def control_metrics(
    events: Sequence[NormalizedEvent],
    baseline_cards: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    results: dict[str, dict[str, dict[str, int]]] = {}
    results["protocol_label_swap"] = _changed_by_split(
        baseline_cards,
        build_daily_cards(events, swap_protocol_labels=True),
    )
    day_counts = Counter(format_time(_event_day(row)) for row in events)
    eligible_days = {day for day, count in day_counts.items() if count >= 2}
    results["within_day_event_order_reverse"] = _changed_by_split(
        baseline_cards,
        build_daily_cards(events, reverse_within_day=True),
        eligible_times=eligible_days,
    )

    proposals_by_split = _split_events(events, event_type="proposal_created")
    pair_metrics: dict[str, dict[str, int]] = {}
    action_metrics: dict[str, dict[str, int]] = {}
    for split in prereg.SPLITS:
        name = str(split["name"])
        proposals = sorted(
            proposals_by_split[name],
            key=lambda row: row.availability_order_key,
        )
        payloads = [_proposal_payload_fingerprint(row) for row in proposals]
        rotated = payloads[1:] + payloads[:1]
        descriptions = [
            sha256_bytes(row.description.encode("utf-8")) for row in proposals
        ]
        baseline_pairs = [
            canonical_hash([description, payload])
            for description, payload in zip(
                descriptions, payloads, strict=True
            )
        ]
        control_pairs = [
            canonical_hash([description, payload])
            for description, payload in zip(
                descriptions, rotated, strict=True
            )
        ]
        pair_metrics[name] = {
            "eligible": len(proposals),
            "changed": sum(
                left != right
                for left, right in zip(
                    baseline_pairs, control_pairs, strict=True
                )
            ),
        }
        eligible_actions = [row for row in proposals if len(row.targets) >= 2]
        action_metrics[name] = {
            "eligible": len(eligible_actions),
            "changed": sum(
                _proposal_payload_fingerprint(row)
                != _proposal_payload_fingerprint(row, reverse_actions=True)
                for row in eligible_actions
            ),
        }
    results["text_payload_pair_permutation"] = pair_metrics
    results["ordered_action_permutation"] = action_metrics

    lifecycle_by_split = _split_events(events)
    lifecycle_metrics: dict[str, dict[str, int]] = {}
    rotation = {
        "proposal_queued": "proposal_executed",
        "proposal_executed": "proposal_canceled",
        "proposal_canceled": "proposal_queued",
    }
    for split in prereg.SPLITS:
        name = str(split["name"])
        rows = [
            row
            for row in lifecycle_by_split[name]
            if row.event != "proposal_created"
        ]
        lifecycle_metrics[name] = {
            "eligible": len(rows),
            "changed": sum(
                canonical_hash([row.event, row.governor_address, str(row.proposal_id)])
                != canonical_hash(
                    [
                        rotation[row.event],
                        row.governor_address,
                        str(row.proposal_id),
                    ]
                )
                for row in rows
            ),
        }
    results["lifecycle_event_rotation"] = lifecycle_metrics
    results["availability_plus_seven_days"] = _changed_by_split(
        baseline_cards,
        build_daily_cards(events, availability_shift=timedelta(days=7)),
    )
    return results


def gate_controls(
    metrics: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> GateResult:
    failures: list[str] = []
    checks: dict[str, Any] = {}
    for control in CONTROL_NAMES:
        control_rows = metrics[control]
        checks[control] = {}
        for split in prereg.SPLITS:
            name = str(split["name"])
            eligible = int(control_rows[name]["eligible"])
            changed = int(control_rows[name]["changed"])
            minimum_eligible = (
                2
                if control
                in {
                    "within_day_event_order_reverse",
                    "lifecycle_event_rotation",
                }
                else 1
            )
            passed = (
                eligible >= minimum_eligible
                and _fraction_at_least(
                    changed,
                    eligible,
                    minimum_numerator=10,
                    minimum_denominator=100,
                )
            )
            checks[control][name] = {
                "eligible": eligible,
                "changed": changed,
                "passed": passed,
            }
            if not passed:
                failures.append(f"{control}:{name}")
    return GateResult(
        name="relation_control_sensitivity",
        passed=not failures,
        metrics=checks,
        failure=";".join(failures),
    )


def future_append_gate(
    events: Sequence[NormalizedEvent],
    baseline_cards: Sequence[Mapping[str, Any]],
) -> GateResult:
    source_events = [
        row for row in events if row.block_number <= prereg.LAST_SOURCE_BLOCK
    ]
    source_proposals = [
        row for row in source_events if row.event == "proposal_created"
    ]
    if not source_proposals:
        return GateResult(
            name="future_append_invariance",
            passed=False,
            metrics={
                "source_events": len(source_events),
                "source_proposals": 0,
            },
            failure="no source proposal available for synthetic append",
        )
    seed = source_proposals[0]
    sentinel_time = datetime(2024, 1, 2, 12, tzinfo=UTC)
    sentinel = replace(
        seed,
        proposal_id=2**255,
        block_number=prereg.END_BOUNDARY_BLOCK + 1,
        block_hash="0x" + "ab" * 32,
        transaction_hash="0x" + "cd" * 32,
        transaction_index=0,
        log_index=0,
        confirmation_block_number=prereg.END_BOUNDARY_BLOCK + 65,
        confirmation_block_hash="0x" + "ef" * 32,
        event_at=sentinel_time,
        available_at=sentinel_time + timedelta(minutes=15),
    )
    proposal_features(sentinel)
    appended = [*source_events, sentinel]
    validate_lifecycle(appended)
    filtered = [
        row for row in appended if row.block_number <= prereg.LAST_SOURCE_BLOCK
    ]
    filtered_cards = build_daily_cards(filtered)

    def event_rows(rows: Sequence[NormalizedEvent]) -> list[dict[str, Any]]:
        return [row.canonical_dict() for row in rows]

    def proposal_rows(
        rows: Sequence[NormalizedEvent],
    ) -> list[dict[str, Any]]:
        return [
            row.canonical_dict()
            for row in rows
            if row.event == "proposal_created"
        ]

    def lifecycle_rows(
        rows: Sequence[NormalizedEvent],
    ) -> list[dict[str, Any]]:
        lifecycle = validate_lifecycle(rows)
        return [
            {
                "governor_address": key[0],
                "proposal_id": str(key[1]),
                "events": [
                    {
                        "event": event.event,
                        "available_at": format_time(event.available_at),
                        "identity": canonical_hash(
                            event.canonical_dict(include_payload=False)
                        ),
                    }
                    for event in lifecycle[key]
                ],
            }
            for key in sorted(lifecycle)
        ]

    def relation_rows(
        rows: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        return [
            {
                "split": row["split"],
                "relation": row["model_card"]["relation"],
            }
            for row in rows
        ]

    comparisons = {
        "normalized_events": (
            canonical_json_bytes(event_rows(source_events))
            == canonical_json_bytes(event_rows(filtered))
        ),
        "proposals": (
            canonical_json_bytes(proposal_rows(source_events))
            == canonical_json_bytes(proposal_rows(filtered))
        ),
        "lifecycle": (
            canonical_json_bytes(lifecycle_rows(source_events))
            == canonical_json_bytes(lifecycle_rows(filtered))
        ),
        "daily_cards": (
            canonical_json_bytes(list(baseline_cards))
            == canonical_json_bytes(filtered_cards)
        ),
        "relation_cards": (
            canonical_json_bytes(relation_rows(baseline_cards))
            == canonical_json_bytes(relation_rows(filtered_cards))
        ),
    }
    canonical_hashes = {
        name: canonical_hash(
            event_rows(source_events)
            if name == "normalized_events"
            else proposal_rows(source_events)
            if name == "proposals"
            else lifecycle_rows(source_events)
            if name == "lifecycle"
            else list(baseline_cards)
            if name == "daily_cards"
            else relation_rows(baseline_cards)
        )
        for name in comparisons
    }
    passed = all(comparisons.values())
    return GateResult(
        name="future_append_invariance",
        passed=passed,
        metrics={
            "source_events": len(source_events),
            "source_proposals": len(source_proposals),
            "appended_events": len(appended),
            "filtered_events": len(filtered),
            "sentinel_schema_valid": True,
            "byte_identical": comparisons,
            "canonical_hashes": canonical_hashes,
        },
        failure="" if passed else "future append changed frozen source",
    )


def forbidden_gate(ledger: AccessLedger) -> GateResult:
    snapshot = ledger.snapshot()
    passed = all(value == 0 for value in snapshot.values())
    return GateResult(
        name="forbidden_access_zero",
        passed=passed,
        metrics=snapshot,
        failure="" if passed else "forbidden counter is nonzero",
    )


def _abi_uint(value: int) -> bytes:
    if not 0 <= value < 2**256:
        raise ValueError("GIPR synthetic uint256 is outside bounds")
    return value.to_bytes(32, byteorder="big")


def _abi_address(value: str) -> bytes:
    normalized = ethereum._address(value)
    return b"\x00" * 12 + bytes.fromhex(normalized[2:])


def _abi_bytes(value: bytes) -> bytes:
    padding = (-len(value)) % 32
    return _abi_uint(len(value)) + value + b"\x00" * padding


def _abi_address_array(values: Sequence[str]) -> bytes:
    return _abi_uint(len(values)) + b"".join(
        _abi_address(value) for value in values
    )


def _abi_uint_array(values: Sequence[int]) -> bytes:
    return _abi_uint(len(values)) + b"".join(
        _abi_uint(value) for value in values
    )


def _abi_dynamic_bytes_array(values: Sequence[bytes]) -> bytes:
    encoded = [_abi_bytes(value) for value in values]
    head_size = len(values) * 32
    offsets: list[bytes] = []
    cursor = head_size
    for item in encoded:
        offsets.append(_abi_uint(cursor))
        cursor += len(item)
    return _abi_uint(len(values)) + b"".join(offsets) + b"".join(encoded)


def _encode_proposal_created_unchecked(
    *,
    proposal_id: int,
    proposer: str,
    targets: Sequence[str],
    values: Sequence[int],
    signatures: Sequence[bytes],
    calldatas: Sequence[bytes],
    start_block: int,
    end_block: int,
    description: bytes,
) -> str:
    tails = (
        _abi_address_array(targets),
        _abi_uint_array(values),
        _abi_dynamic_bytes_array(signatures),
        _abi_dynamic_bytes_array(calldatas),
        _abi_bytes(description),
    )
    head_size = 9 * 32
    offsets: list[int] = []
    cursor = head_size
    for tail in tails:
        offsets.append(cursor)
        cursor += len(tail)
    head = b"".join(
        (
            _abi_uint(proposal_id),
            _abi_address(proposer),
            _abi_uint(offsets[0]),
            _abi_uint(offsets[1]),
            _abi_uint(offsets[2]),
            _abi_uint(offsets[3]),
            _abi_uint(start_block),
            _abi_uint(end_block),
            _abi_uint(offsets[4]),
        )
    )
    return "0x" + (head + b"".join(tails)).hex()


def encode_proposal_created(
    *,
    proposal_id: int,
    proposer: str,
    targets: Sequence[str],
    values: Sequence[int],
    signatures: Sequence[str],
    calldatas: Sequence[str],
    start_block: int,
    end_block: int,
    description: str,
) -> str:
    """Canonical synthetic encoder used only by tests and self-check."""

    if not (
        len(targets) == len(values) == len(signatures) == len(calldatas)
    ):
        raise ValueError("GIPR synthetic action arrays have unequal lengths")
    signature_bytes = [value.encode("utf-8") for value in signatures]
    calldata_bytes = [
        bytes.fromhex(_hex_blob(value, "GIPR synthetic calldata")[2:])
        for value in calldatas
    ]
    encoded = _encode_proposal_created_unchecked(
        proposal_id=proposal_id,
        proposer=proposer,
        targets=targets,
        values=values,
        signatures=signature_bytes,
        calldatas=calldata_bytes,
        start_block=start_block,
        end_block=end_block,
        description=description.encode("utf-8"),
    )
    decode_proposal_created(encoded)
    return encoded


def _synthetic_address(index: int) -> str:
    if not 0 < index < 2**160:
        raise ValueError("GIPR synthetic address index is outside bounds")
    return "0x" + index.to_bytes(20, byteorder="big").hex()


def _synthetic_hash(index: int) -> str:
    if not 0 <= index < 2**256:
        raise ValueError("GIPR synthetic hash index is outside bounds")
    return "0x" + index.to_bytes(32, byteorder="big").hex()


def _synthetic_event(
    *,
    protocol: str,
    event: str,
    proposal_id: int,
    available_at: datetime,
    block_number: int,
    log_index: int,
    targets: Sequence[str] = (),
    values: Sequence[int] = (),
    signatures: Sequence[str] = (),
    calldatas: Sequence[str] = (),
    description: str = "",
) -> NormalizedEvent:
    governor = next(
        row for row in prereg.GOVERNORS if row.protocol == protocol
    )
    created = event == "proposal_created"
    return NormalizedEvent(
        protocol=protocol,
        generation=governor.generation,
        governor_address=governor.address,
        event=event,
        proposal_id=proposal_id,
        proposer=_synthetic_address(10_000 + proposal_id) if created else "",
        targets=tuple(targets) if created else (),
        values=tuple(values) if created else (),
        signatures=tuple(signatures) if created else (),
        calldatas=tuple(calldatas) if created else (),
        start_block=block_number + 100 if created else None,
        end_block=block_number + 200 if created else None,
        description=description if created else "",
        queue_eta=(
            int(available_at.timestamp()) + 86_400
            if event == "proposal_queued"
            else None
        ),
        block_number=block_number,
        block_hash=_synthetic_hash(block_number),
        transaction_hash=_synthetic_hash(block_number * 10 + log_index),
        transaction_index=0,
        log_index=log_index,
        confirmation_block_number=block_number + prereg.CONFIRMATION_BLOCKS,
        confirmation_block_hash=_synthetic_hash(
            block_number + prereg.CONFIRMATION_BLOCKS
        ),
        event_at=available_at - timedelta(minutes=15),
        available_at=available_at,
    )


def _synthetic_source_events() -> list[NormalizedEvent]:
    split_days = {
        "train": (
            "2020-03-10",
            "2020-05-10",
            "2020-07-10",
            "2020-09-10",
            "2020-11-10",
            "2021-01-10",
            "2021-03-10",
            "2021-05-10",
            "2021-07-10",
            "2021-09-10",
            "2021-11-10",
            "2021-12-10",
        ),
        "test": (
            "2022-01-10",
            "2022-03-10",
            "2022-05-10",
            "2022-07-10",
            "2022-09-10",
            "2022-11-10",
        ),
        "eval": (
            "2023-01-10",
            "2023-03-10",
            "2023-05-10",
            "2023-07-10",
            "2023-09-10",
            "2023-11-10",
        ),
    }
    known_roles = {
        protocol: next(
            row.address
            for row in prereg.KNOWN_TARGET_ROLES
            if row.protocol == protocol
        )
        for protocol in ("compound", "uniswap")
    }
    events: list[NormalizedEvent] = []
    proposal_id = 1
    block_number = 10_900_000
    for days in split_days.values():
        for day_index, day_text in enumerate(days):
            day = datetime.fromisoformat(day_text + "T01:00:00+00:00")
            for protocol_index, protocol in enumerate(
                ("compound", "uniswap")
            ):
                pattern = (day_index + protocol_index) % 3
                action_count = pattern + 1
                if pattern == 0:
                    targets = [known_roles[protocol]]
                elif pattern == 1:
                    targets = [
                        known_roles[protocol],
                        _synthetic_address(100_000 + proposal_id),
                    ]
                else:
                    targets = [
                        _synthetic_address(
                            100_000 + proposal_id * 10 + action_index
                        )
                        for action_index in range(action_count)
                    ]
                signatures = [
                    f"syntheticAction{proposal_id}_{action_index}(uint256)"
                    for action_index in range(action_count)
                ]
                calldatas = [
                    "0x"
                    + (proposal_id * 10 + action_index)
                    .to_bytes(4, byteorder="big")
                    .hex()
                    + _abi_uint(action_index).hex()
                    for action_index in range(action_count)
                ]
                values = [
                    proposal_id if action_index == action_count - 1 else 0
                    for action_index in range(action_count)
                ]
                description = (
                    f"# Synthetic {protocol} proposal {proposal_id}\n"
                    f"https://example.invalid/{proposal_id}"
                )
                creation = _synthetic_event(
                    protocol=protocol,
                    event="proposal_created",
                    proposal_id=proposal_id,
                    available_at=day + timedelta(minutes=protocol_index * 5),
                    block_number=block_number,
                    log_index=0,
                    targets=targets,
                    values=values,
                    signatures=signatures,
                    calldatas=calldatas,
                    description=description,
                )
                events.append(creation)
                block_number += 1
                if proposal_id % 2 == 0:
                    events.append(
                        _synthetic_event(
                            protocol=protocol,
                            event="proposal_queued",
                            proposal_id=proposal_id,
                            available_at=day
                            + timedelta(
                                hours=1,
                                minutes=protocol_index * 5,
                            ),
                            block_number=block_number,
                            log_index=0,
                        )
                    )
                    block_number += 1
                    events.append(
                        _synthetic_event(
                            protocol=protocol,
                            event="proposal_executed",
                            proposal_id=proposal_id,
                            available_at=day
                            + timedelta(
                                hours=2,
                                minutes=protocol_index * 5,
                            ),
                            block_number=block_number,
                            log_index=0,
                        )
                    )
                else:
                    events.append(
                        _synthetic_event(
                            protocol=protocol,
                            event="proposal_canceled",
                            proposal_id=proposal_id,
                            available_at=day
                            + timedelta(
                                hours=1,
                                minutes=protocol_index * 5,
                            ),
                            block_number=block_number,
                            log_index=0,
                        )
                    )
                block_number += 1
                proposal_id += 1
    events.sort(key=lambda row: row.chain_order_key)
    validate_lifecycle(events)
    return events


def _synthetic_raw_log(
    *,
    uppercase: bool = False,
    padded_quantities: bool = False,
) -> dict[str, Any]:
    governor = prereg.GOVERNORS[0]
    data = encode_proposal_created(
        proposal_id=7,
        proposer=_synthetic_address(70),
        targets=(
            prereg.KNOWN_TARGET_ROLES[0].address,
            _synthetic_address(71),
        ),
        values=(0, 7),
        signatures=("setValue(uint256)", ""),
        calldatas=("0x" + "01" * 36, "0x01020304"),
        start_block=10_000_000,
        end_block=10_000_100,
        description="# Synthetic\nhttps://example.invalid",
    )

    def quantity(value: int) -> str:
        raw = f"{value:x}"
        return "0x" + (raw.rjust(8, "0") if padded_quantities else raw)

    def maybe_upper(value: str) -> str:
        return "0x" + value[2:].upper() if uppercase else value

    return {
        "address": maybe_upper(governor.address),
        "topics": [maybe_upper(prereg.EVENT_SPECS[0].topic)],
        "data": maybe_upper(data),
        "removed": False,
        "blockNumber": quantity(governor.first_code_block + 100),
        "blockHash": maybe_upper(_synthetic_hash(501)),
        "transactionHash": maybe_upper(_synthetic_hash(502)),
        "transactionIndex": quantity(3),
        "logIndex": quantity(4),
    }


RAW_MODEL_CARD_FORBIDDEN_KEYS = {
    "block_number",
    "block_hash",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "event_at",
    "available_at",
    "decision_at",
    "proposal_id",
    "governor_address",
    "targets",
    "values",
    "signatures",
    "calldatas",
    "price",
    "return",
    "rank",
    "pnl",
}


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return set(map(str, value)) | {
            nested
            for child in value.values()
            for nested in _nested_keys(child)
        }
    if isinstance(value, (list, tuple)):
        return {
            nested for child in value for nested in _nested_keys(child)
        }
    return set()


def _raises_value_error(builder: Any) -> bool:
    try:
        builder()
    except (ValueError, RuntimeError):
        return True
    return False


def _mutate_abi_word(data_hex: str, word_index: int, value: int) -> str:
    raw = bytearray.fromhex(data_hex[2:])
    raw[word_index * 32 : (word_index + 1) * 32] = _abi_uint(value)
    return "0x" + raw.hex()


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def build_self_check_manifest() -> dict[str, Any]:
    before = {
        str(path): repository_path(path).exists()
        for path in (
            EVENT_OUTPUT,
            PROPOSAL_OUTPUT,
            CARD_OUTPUT,
            CONTROL_OUTPUT,
            PASS_REPORT,
            REJECTION_REPORT,
        )
    }
    canonical_first = normalize_log(_synthetic_raw_log())
    canonical_second = normalize_log(
        _synthetic_raw_log(uppercase=True, padded_quantities=True)
    )
    decoded = parse_log(canonical_first)
    events = _synthetic_source_events()
    cards = build_daily_cards(events)
    output_cards = card_output_rows(cards)
    metrics = split_support_metrics(events, cards)
    controls = control_metrics(events, cards)
    gates = (
        gate_split_proposal_action_support(metrics),
        gate_cross_protocol_support(metrics),
        gate_structural_vocabulary(metrics),
        gate_daily_schedule(metrics),
        future_append_gate(events, cards),
        gate_controls(controls),
        forbidden_gate(AccessLedger.zero()),
    )
    removed_log = {**_synthetic_raw_log(), "removed": True}
    out_of_boundary_log = {
        **_synthetic_raw_log(),
        "blockNumber": hex(FIRST_LOG_BLOCK - 1),
    }
    wrong_topic_log = {
        **_synthetic_raw_log(),
        "topics": ["0x" + "ff" * 32],
    }
    transport_rejections = {
        "removed": _raises_value_error(lambda: normalize_log(removed_log)),
        "out_of_boundary": _raises_value_error(
            lambda: normalize_log(out_of_boundary_log)
        ),
        "wrong_topic": _raises_value_error(
            lambda: normalize_log(wrong_topic_log)
        ),
        "duplicate": _raises_value_error(
            lambda: normalize_logs(
                [_synthetic_raw_log(), _synthetic_raw_log()]
            )
        ),
    }

    valid_abi = canonical_first.data
    raw_abi = bytes.fromhex(valid_abi[2:])
    targets_offset = int.from_bytes(raw_abi[64:96], byteorder="big")
    values_offset = int.from_bytes(raw_abi[96:128], byteorder="big")
    bad_padding = bytearray(raw_abi)
    bad_padding[-1] = 1
    unchecked_common = {
        "proposal_id": 1,
        "proposer": _synthetic_address(2),
        "targets": (_synthetic_address(3),),
        "values": (0,),
        "signatures": (b"setValue(uint256)",),
        "calldatas": (b"\x01\x02\x03\x04",),
        "start_block": 100,
        "end_block": 200,
    }
    abi_rejections = {
        "length_mismatch": _raises_value_error(
            lambda: decode_proposal_created(
                _encode_proposal_created_unchecked(
                    **{
                        **unchecked_common,
                        "values": (0, 1),
                        "description": b"valid",
                    }
                )
            )
        ),
        "misalignment": _raises_value_error(
            lambda: decode_proposal_created(
                _mutate_abi_word(valid_abi, 2, targets_offset + 1)
            )
        ),
        "alias_overlap": _raises_value_error(
            lambda: decode_proposal_created(
                _mutate_abi_word(valid_abi, 3, targets_offset)
            )
        ),
        "gap": _raises_value_error(
            lambda: decode_proposal_created(
                _mutate_abi_word(valid_abi, 3, values_offset + 32)
            )
        ),
        "invalid_padding": _raises_value_error(
            lambda: decode_proposal_created("0x" + bad_padding.hex())
        ),
        "invalid_utf8": _raises_value_error(
            lambda: decode_proposal_created(
                _encode_proposal_created_unchecked(
                    **{**unchecked_common, "description": b"\xff"}
                )
            )
        ),
        "nul": _raises_value_error(
            lambda: decode_proposal_created(
                _encode_proposal_created_unchecked(
                    **{**unchecked_common, "description": b"a\x00b"}
                )
            )
        ),
        "parser_bound": _raises_value_error(
            lambda: decode_proposal_created(
                _encode_proposal_created_unchecked(
                    **{
                        **unchecked_common,
                        "description": b"x"
                        * (prereg.MAX_DESCRIPTION_BYTES + 1),
                    }
                )
            )
        ),
        "trailing_bytes": _raises_value_error(
            lambda: decode_proposal_created(valid_abi + "00" * 32)
        ),
    }

    source_header = ethereum.BlockHeader(
        decoded.block_number,
        decoded.block_hash,
        _synthetic_hash(900),
        int(datetime(2021, 12, 31, 23, 59, tzinfo=UTC).timestamp()),
    )
    confirmation_number = (
        decoded.block_number + prereg.CONFIRMATION_BLOCKS
    )
    confirmation_header = ethereum.BlockHeader(
        confirmation_number,
        _synthetic_hash(901),
        _synthetic_hash(902),
        int(datetime(2022, 1, 1, 0, 1, tzinfo=UTC).timestamp()),
    )
    headers = {
        decoded.block_number: source_header,
        confirmation_number: confirmation_header,
    }
    materialized = materialize_events([decoded], headers, dict(headers))[0]
    mismatched_headers = {
        **headers,
        decoded.block_number: replace(
            source_header,
            block_hash=_synthetic_hash(903),
        ),
    }
    header_mismatch_rejected = _raises_value_error(
        lambda: materialize_events(
            [decoded],
            headers,
            mismatched_headers,
        )
    )

    lifecycle_start = events[0]
    lifecycle_queue = replace(
        events[1],
        event="proposal_queued",
        queue_eta=1,
    )
    lifecycle_execute = replace(
        lifecycle_queue,
        event="proposal_executed",
        queue_eta=None,
        block_number=lifecycle_queue.block_number + 1,
        block_hash=_synthetic_hash(
            lifecycle_queue.block_number + 1
        ),
        transaction_hash=_synthetic_hash(
            lifecycle_queue.block_number + 10_000
        ),
        available_at=lifecycle_queue.available_at
        + timedelta(minutes=1),
    )
    lifecycle_cancel_after_queue = replace(
        lifecycle_execute,
        event="proposal_canceled",
    )
    valid_lifecycle_paths = all(
        bool(validate_lifecycle(path))
        for path in (
            [lifecycle_start, lifecycle_queue, lifecycle_execute],
            [lifecycle_start, events[1]],
            [
                lifecycle_start,
                lifecycle_queue,
                lifecycle_cancel_after_queue,
            ],
        )
    )
    invalid_lifecycle_checks = {
        "missing_creation": _raises_value_error(
            lambda: validate_lifecycle([events[1]])
        ),
        "execution_without_queue": _raises_value_error(
            lambda: validate_lifecycle(
                [
                    events[0],
                    replace(
                        events[1],
                        event="proposal_executed",
                    ),
                ]
            )
        ),
        "duplicate_creation": _raises_value_error(
            lambda: validate_lifecycle([events[0], events[0]])
        ),
        "duplicate_queue": _raises_value_error(
            lambda: validate_lifecycle(
                [
                    lifecycle_start,
                    lifecycle_queue,
                    lifecycle_queue,
                ]
            )
        ),
        "post_execution": _raises_value_error(
            lambda: validate_lifecycle(
                [
                    lifecycle_start,
                    lifecycle_queue,
                    lifecycle_execute,
                    lifecycle_cancel_after_queue,
                ]
            )
        ),
        "post_cancellation": _raises_value_error(
            lambda: validate_lifecycle(
                [lifecycle_start, events[1], lifecycle_queue]
            )
        ),
    }
    stale_lifecycle = validate_lifecycle([lifecycle_start])
    active_at_90, _ = _protocol_card(
        "compound",
        lifecycle_start.available_at + timedelta(days=90),
        [lifecycle_start],
        stale_lifecycle,
        canonical_hash([]),
    )
    stale_after_90, _ = _protocol_card(
        "compound",
        lifecycle_start.available_at
        + timedelta(days=90, seconds=1),
        [lifecycle_start],
        stale_lifecycle,
        canonical_hash([]),
    )

    support_minus_one = _json_copy(metrics)
    support_minus_one["train"]["proposals"] = (
        prereg.SPLITS[0]["minimum_proposals"] - 1
    )
    structural_exact = _json_copy(metrics)
    for split in prereg.SPLITS:
        name = str(split["name"])
        structural_exact[name]["proposals"] = 100
        structural_exact[name]["nonempty_descriptions"] = 95
        structural_exact[name]["unique_descriptions"] = 90
    structural_minus_one = _json_copy(structural_exact)
    structural_minus_one["test"]["nonempty_descriptions"] = 94
    collapsed = _json_copy(structural_exact)
    collapsed["eval"]["action_count_buckets"] = ["ONE"]
    daily_exact = _json_copy(metrics)
    for split in prereg.SPLITS:
        name = str(split["name"])
        days = int(daily_exact[name]["daily_decisions"])
        daily_exact[name]["maximum_complete_daily_card_count"] = (
            days * 35 // 100
        )
    daily_minus_one = _json_copy(daily_exact)
    daily_minus_one["test"]["maximum_complete_daily_card_count"] += 1
    control_exact = {
        control: {
            str(split["name"]): {"eligible": 10, "changed": 1}
            for split in prereg.SPLITS
        }
        for control in CONTROL_NAMES
    }
    control_minus_one = _json_copy(control_exact)
    control_minus_one["protocol_label_swap"]["eval"]["changed"] = 0

    ordered_failure_gates = [
        GateResult(GATE_NAMES[0], True, {}),
        GateResult(GATE_NAMES[1], False, {}, "synthetic"),
    ]
    ordered_failure_report = build_result_report(
        decision="reject",
        authority={},
        gates=ordered_failure_gates,
        source_audit={},
        event_count=0,
        proposal_count=0,
        card_count=0,
        artifacts=None,
        ledger=AccessLedger.zero(),
    )
    forbidden_ledger = AccessLedger.zero()
    forbidden_ledger.counters["pnl_rows_built"] = 1
    forbidden_rejection_gates = [
        *[GateResult(name, True, {}) for name in GATE_NAMES[:-1]],
        forbidden_gate(forbidden_ledger),
    ]
    forbidden_rejection = build_result_report(
        decision="reject",
        authority={},
        gates=forbidden_rejection_gates,
        source_audit={"event_logs_opened": True},
        event_count=1,
        proposal_count=1,
        card_count=1,
        artifacts=None,
        ledger=forbidden_ledger,
    )
    _validate_result_report(forbidden_rejection, decision="reject")

    with tempfile.TemporaryDirectory(prefix="gipr-self-check-") as directory:
        publication = Path(directory) / "artifact.json"
        _write_once_bytes(publication, b"canonical")
        _write_once_bytes(publication, b"canonical")
        publication_idempotent = publication.read_bytes() == b"canonical"
        publication_conflict_rejected = _raises_value_error(
            lambda: _write_once_bytes(publication, b"conflict")
        )

    event_bytes = deterministic_gzip(
        csv_bytes(
            (
                event_csv_row(row)
                for row in events
            ),
            EVENT_COLUMNS,
        )
    )
    card_bytes = deterministic_gzip(jsonl_bytes(output_cards))
    checks = {
        "canonical_transport_equivalence": (
            canonical_first == canonical_second
        ),
        "canonical_abi_round_trip": (
            decoded.proposal_id == 7
            and len(decoded.targets) == 2
            and decoded.signatures[-1] == ""
            and decoded.calldatas[-1] == "0x01020304"
        ),
        "transport_rejections": all(transport_rejections.values()),
        "abi_adversarial_rejections": all(abi_rejections.values()),
        "header_mismatch_rejected": header_mismatch_rejected,
        "n_plus_64_availability": (
            materialized.available_at
            == datetime(2022, 1, 1, 0, 1, tzinfo=UTC)
            and _event_day(materialized)
            == datetime(2022, 1, 2, 0, 0, tzinfo=UTC)
        ),
        "valid_lifecycle_paths": valid_lifecycle_paths,
        "invalid_lifecycle_rejections": all(
            invalid_lifecycle_checks.values()
        ),
        "exact_age_and_stale_boundaries": (
            age_bucket(timedelta(days=2) - timedelta(seconds=1))
            == "D0_1"
            and age_bucket(timedelta(days=2)) == "D2_7"
            and age_bucket(timedelta(days=8)) == "D8_28"
            and age_bucket(timedelta(days=29)) == "D29_90"
            and age_bucket(timedelta(days=90)) == "D29_90"
            and age_bucket(timedelta(days=90, seconds=1))
            == "STALE_OR_NONE"
            and active_at_90["lifecycle_state"]
            == "PROPOSAL_CREATED"
            and stale_after_90["lifecycle_state"]
            == "STALE_OR_NO_PROPOSAL"
        ),
        "valid_lifecycle": bool(validate_lifecycle(events)),
        "all_synthetic_support_gates": all(gate.passed for gate in gates),
        "split_threshold_minus_one_rejected": not (
            gate_split_proposal_action_support(
                support_minus_one
            ).passed
        ),
        "structural_exact_threshold_passed": (
            gate_structural_vocabulary(structural_exact).passed
        ),
        "structural_threshold_minus_one_rejected": not (
            gate_structural_vocabulary(structural_minus_one).passed
        ),
        "collapsed_vocabulary_rejected": not (
            gate_structural_vocabulary(collapsed).passed
        ),
        "daily_exact_threshold_passed": gate_daily_schedule(
            daily_exact
        ).passed,
        "daily_threshold_minus_one_rejected": not (
            gate_daily_schedule(daily_minus_one).passed
        ),
        "control_exact_threshold_passed": gate_controls(
            control_exact
        ).passed,
        "control_threshold_minus_one_rejected": not (
            gate_controls(control_minus_one).passed
        ),
        "future_append_invariant": gates[-3].passed,
        "all_six_relation_controls": (
            tuple(controls) == CONTROL_NAMES and gates[-2].passed
        ),
        "ordered_first_failure": (
            ordered_failure_report["first_failure"]
            == {"gate_id": 2, "name": GATE_NAMES[1]}
            and len(ordered_failure_report["gates"]) == 2
        ),
        "forbidden_success_and_rejection_paths": (
            gates[-1].passed
            and forbidden_rejection["first_failure"]
            == {"gate_id": 12, "name": GATE_NAMES[-1]}
            and forbidden_rejection["forbidden_access"][
                "pnl_rows_built"
            ]
            == 1
        ),
        "canonical_publication_idempotent": publication_idempotent,
        "publication_conflict_rejected": publication_conflict_rejected,
        "daily_card_forbidden_fields_absent": not (
            _nested_keys(output_cards) & RAW_MODEL_CARD_FORBIDDEN_KEYS
        ),
        "daily_schedule_exact": (
            len(cards) == len(decision_times())
            and cards[0]["decision_at"] == format_time(FIRST_DECISION)
            and cards[-1]["decision_at"] == format_time(LAST_DECISION)
        ),
        "event_gzip_deterministic": event_bytes
        == deterministic_gzip(
            csv_bytes(
                (event_csv_row(row) for row in events),
                EVENT_COLUMNS,
            )
        ),
        "card_gzip_deterministic": card_bytes
        == deterministic_gzip(jsonl_bytes(output_cards)),
        "forbidden_access_zero": AccessLedger.zero().snapshot()
        == {name: 0 for name in FORBIDDEN_COUNTER_NAMES},
    }
    after = {
        str(path): repository_path(path).exists()
        for path in (
            EVENT_OUTPUT,
            PROPOSAL_OUTPUT,
            CARD_OUTPUT,
            CONTROL_OUTPUT,
            PASS_REPORT,
            REJECTION_REPORT,
        )
    }
    checks["source_artifact_state_unchanged"] = before == after
    if not all(checks.values()):
        raise RuntimeError(f"GIPR-D1 synthetic self-check failed: {checks}")
    core = {
        "protocol_version": SELF_CHECK_PROTOCOL,
        "policy_id": POLICY_ID,
        "checks": checks,
        "synthetic": {
            "events": len(events),
            "proposals": sum(
                row.event == "proposal_created" for row in events
            ),
            "daily_cards": len(cards),
            "event_hash": canonical_hash(
                [row.canonical_dict() for row in events]
            ),
            "card_hash": canonical_hash(output_cards),
            "control_hash": canonical_hash(controls),
            "event_gzip_sha256": sha256_bytes(event_bytes),
            "card_gzip_sha256": sha256_bytes(card_bytes),
        },
        "network_calls": 0,
        "source_event_rows_opened": 0,
        "outcomes_opened": False,
        "forbidden_access": AccessLedger.zero().snapshot(),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def self_check_bytes() -> bytes:
    return canonical_json_bytes(build_self_check_manifest())


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_committed(
    path: str | Path,
    *,
    expected_commit: str | None = None,
) -> str:
    relative = Path(path).as_posix()
    commit = _git_output("log", "-1", "--format=%H", "--", relative)
    if not HEX40.fullmatch(commit):
        raise RuntimeError(f"GIPR path is not committed: {relative}")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(f"GIPR path commit changed: {relative}")
    return commit


def _git_blob_sha256(commit: str, path: str | Path) -> str:
    relative = Path(path).as_posix()
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return sha256_bytes(completed.stdout)


def _worktree_clean() -> bool:
    return not _git_output(
        "status", "--porcelain", "--untracked-files=all"
    )


def _binding(
    path: str | Path,
    commit: str,
    digest: str,
) -> dict[str, str]:
    return {
        "path": Path(path).as_posix(),
        "commit": commit,
        "sha256": digest,
    }


def _implementation_binding(path: str | Path) -> dict[str, str]:
    commit = _assert_committed(path)
    digest = sha256_file(path)
    if _git_blob_sha256(commit, path) != digest:
        raise RuntimeError(f"GIPR committed implementation differs: {path}")
    return _binding(path, commit, digest)


def _validate_binding(
    binding: Mapping[str, Any],
    *,
    path: str | Path,
    expected_commit: str | None = None,
    expected_sha256: str | None = None,
) -> None:
    relative = Path(path).as_posix()
    if set(binding) != {"path", "commit", "sha256"}:
        raise RuntimeError("GIPR binding schema changed")
    commit = binding.get("commit")
    digest = binding.get("sha256")
    if (
        binding.get("path") != relative
        or not isinstance(commit, str)
        or HEX40.fullmatch(commit) is None
        or not isinstance(digest, str)
        or HEX64.fullmatch(digest) is None
        or (expected_commit is not None and commit != expected_commit)
        or (expected_sha256 is not None and digest != expected_sha256)
        or _assert_committed(path, expected_commit=commit) != commit
        or sha256_file(path) != digest
        or _git_blob_sha256(commit, path) != digest
    ):
        raise RuntimeError(f"GIPR binding mismatch: {relative}")


def python_runtime() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    return {
        "python": {
            "path": str(executable),
            "sha256": sha256_file(executable),
            "version": sys.version,
        },
        "git_version": _git_output("--version"),
    }


def static_authority() -> dict[str, Any]:
    preregistration = _load_preregistration()
    bindings = {
        "contract": _binding(
            CONTRACT_PATH,
            CONTRACT_COMMIT,
            CONTRACT_SHA256,
        ),
        "preregistration": _binding(
            PREREGISTRATION_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_SHA256,
        ),
        "preregistration_producer": _binding(
            PREREGISTRATION_PRODUCER_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_PRODUCER_SHA256,
        ),
        "ethereum_helper": _implementation_binding(ETHEREUM_HELPER_PATH),
    }
    _validate_binding(
        bindings["contract"],
        path=CONTRACT_PATH,
        expected_commit=CONTRACT_COMMIT,
        expected_sha256=CONTRACT_SHA256,
    )
    _validate_binding(
        bindings["preregistration"],
        path=PREREGISTRATION_PATH,
        expected_commit=PREREGISTRATION_COMMIT,
        expected_sha256=PREREGISTRATION_SHA256,
    )
    _validate_binding(
        bindings["preregistration_producer"],
        path=PREREGISTRATION_PRODUCER_PATH,
        expected_commit=PREREGISTRATION_COMMIT,
        expected_sha256=PREREGISTRATION_PRODUCER_SHA256,
    )
    source_core = {
        "chain_id": prereg.CHAIN_ID,
        "first_log_block": FIRST_LOG_BLOCK,
        "last_log_block": LAST_LOG_BLOCK,
        "confirmation_blocks": prereg.CONFIRMATION_BLOCKS,
        "governors": [asdict(row) for row in prereg.GOVERNORS],
        "event_specs": [asdict(row) for row in prereg.EVENT_SPECS],
        "splits": list(prereg.SPLITS),
        "gates": list(GATE_NAMES),
        "controls": list(CONTROL_NAMES),
    }
    return {
        "runtime": python_runtime(),
        **bindings,
        "preregistration_manifest_hash": preregistration["manifest_hash"],
        "source_authority_hash": canonical_hash(source_core),
    }


def _run_self_check_subprocess() -> dict[str, Any]:
    argv = [sys.executable, RUNNER_PATH.as_posix(), "self-check"]
    completed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError(
            "GIPR self-check subprocess failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("GIPR self-check output is not JSON") from error
    if completed.stdout != canonical_json_bytes(payload):
        raise RuntimeError("GIPR self-check output is noncanonical")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if (
        payload.get("protocol_version") != SELF_CHECK_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("manifest_hash") != canonical_hash(core)
        or payload.get("network_calls") != 0
        or payload.get("source_event_rows_opened") != 0
        or payload.get("outcomes_opened") is not False
        or payload.get("forbidden_access")
        != AccessLedger.zero().snapshot()
    ):
        raise RuntimeError("GIPR self-check manifest changed")
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "manifest_hash": payload["manifest_hash"],
        "network_calls": 0,
        "source_event_rows_opened": 0,
        "outcomes_opened": False,
        "forbidden_access": AccessLedger.zero().snapshot(),
    }


def _pytest_counts(stdout: str, stderr: str) -> dict[str, int]:
    summary_lines = [
        line.strip()
        for line in (stdout + "\n" + stderr).splitlines()
        if re.search(
            r"\b(?:passed|failed|skipped|errors?|xfailed|xpassed)\b",
            line,
        )
    ]
    if not summary_lines:
        raise RuntimeError("GIPR pytest summary is absent")
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    aliases = {"error": "errors", "errors": "errors"}
    for count, label in re.findall(
        r"([0-9]+)\s+"
        r"(passed|failed|skipped|errors?|xfailed|xpassed)\b",
        summary_lines[-1],
    ):
        counts[aliases.get(label, label)] += int(count)
    if counts["passed"] <= 0:
        raise RuntimeError("GIPR pytest passed count is absent")
    return counts


def _run_pytest_verification() -> dict[str, Any]:
    argv = [".venv/bin/pytest", "-q", TEST_PATH.as_posix()]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."
    completed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    counts = _pytest_counts(completed.stdout, completed.stderr)
    if (
        completed.returncode != 0
        or counts["failed"]
        or counts["skipped"]
        or counts["errors"]
        or counts["xfailed"]
        or counts["xpassed"]
    ):
        raise RuntimeError(
            "GIPR exact pytest verification failed\n"
            + completed.stdout
            + completed.stderr
        )
    return {
        "argv": argv,
        "environment_override": {"PYTHONPATH": "."},
        "exit_code": completed.returncode,
        **counts,
    }


def build_execution_seal() -> dict[str, Any]:
    if not _worktree_clean():
        raise RuntimeError("GIPR seal creation requires a clean worktree")
    authority = static_authority()
    runner = _implementation_binding(RUNNER_PATH)
    tests = _implementation_binding(TEST_PATH)
    head = _git_output("rev-parse", "HEAD")
    if runner["commit"] != tests["commit"] or runner["commit"] != head:
        raise RuntimeError(
            "GIPR runner and tests must share current HEAD"
        )
    core = {
        "protocol_version": SEAL_PROTOCOL,
        "policy_id": POLICY_ID,
        "authority": authority,
        "runner": runner,
        "tests": tests,
        "shared_commit": head,
        "synthetic_verification": {
            "self_check": _run_self_check_subprocess(),
            "pytest": _run_pytest_verification(),
        },
        "forbidden_access": AccessLedger.zero().snapshot(),
    }
    return {**core, "seal_hash": canonical_hash(core)}


def create_execution_seal() -> dict[str, Any]:
    payload = build_execution_seal()
    _write_once_bytes(
        EXECUTION_SEAL_PATH,
        canonical_json_bytes(payload),
    )
    return payload


def validate_execution_seal() -> dict[str, Any]:
    path = repository_path(EXECUTION_SEAL_PATH)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("GIPR execution seal is absent or unsafe")
    _assert_committed(EXECUTION_SEAL_PATH)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("GIPR execution seal is unreadable") from error
    if raw != canonical_json_bytes(payload):
        raise RuntimeError("GIPR execution seal bytes are noncanonical")
    expected = {
        "protocol_version",
        "policy_id",
        "authority",
        "runner",
        "tests",
        "shared_commit",
        "synthetic_verification",
        "forbidden_access",
        "seal_hash",
    }
    core = {
        key: value for key, value in payload.items() if key != "seal_hash"
    }
    if (
        set(payload) != expected
        or payload.get("protocol_version") != SEAL_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("seal_hash") != canonical_hash(core)
        or payload.get("authority") != static_authority()
        or payload.get("forbidden_access")
        != AccessLedger.zero().snapshot()
    ):
        raise RuntimeError("GIPR execution seal core changed")
    shared_commit = payload.get("shared_commit")
    if not isinstance(shared_commit, str) or HEX40.fullmatch(shared_commit) is None:
        raise RuntimeError("GIPR sealed commit is malformed")
    _validate_binding(
        payload["runner"],
        path=RUNNER_PATH,
        expected_commit=shared_commit,
    )
    _validate_binding(
        payload["tests"],
        path=TEST_PATH,
        expected_commit=shared_commit,
    )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", shared_commit, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("GIPR sealed implementation is not an ancestor")
    expected_verification = {
        "self_check": _run_self_check_subprocess(),
        "pytest": _run_pytest_verification(),
    }
    if payload.get("synthetic_verification") != expected_verification:
        raise RuntimeError("GIPR sealed synthetic verification changed")
    return payload


def event_csv_row(event: NormalizedEvent) -> dict[str, Any]:
    if event.event_at is None or event.available_at is None:
        raise RuntimeError("GIPR output event is not materialized")
    if (
        event.confirmation_block_number is None
        or not event.confirmation_block_hash
    ):
        raise RuntimeError("GIPR output event has no confirmation identity")
    return {
        "protocol": event.protocol,
        "generation": event.generation,
        "governor_address": event.governor_address,
        "event": event.event,
        "proposal_id": str(event.proposal_id),
        "block_number": event.block_number,
        "block_hash": event.block_hash,
        "block_timestamp": format_time(event.event_at),
        "transaction_hash": event.transaction_hash,
        "transaction_index": event.transaction_index,
        "log_index": event.log_index,
        "confirmation_block_number": event.confirmation_block_number,
        "confirmation_block_hash": event.confirmation_block_hash,
        "available_at": format_time(event.available_at),
    }


def proposal_output_rows(
    events: Sequence[NormalizedEvent],
) -> list[dict[str, Any]]:
    return [
        event.canonical_dict()
        for event in events
        if event.event == "proposal_created"
    ]


def card_output_rows(
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "split": row["split"],
            "model_card": row["model_card"],
            "card_sha256": row["card_sha256"],
        }
        for row in cards
    ]
    forbidden = _nested_keys(rows) & RAW_MODEL_CARD_FORBIDDEN_KEYS
    if forbidden:
        raise RuntimeError(
            "GIPR output daily card contains forbidden fields: "
            + ",".join(sorted(forbidden))
        )
    return rows


def build_control_report(
    cards: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Mapping[str, int]]],
    gate: GateResult,
) -> dict[str, Any]:
    if tuple(metrics) != CONTROL_NAMES:
        raise RuntimeError("GIPR control order differs from preregistration")
    core = {
        "protocol_version": "gipr_d1_source_controls_v1",
        "policy_id": POLICY_ID,
        "baseline_card_hash": canonical_hash(card_output_rows(cards)),
        "control_order": list(CONTROL_NAMES),
        "metrics": {
            control: {
                split: {
                    "eligible": int(values["eligible"]),
                    "changed": int(values["changed"]),
                }
                for split, values in rows.items()
            }
            for control, rows in metrics.items()
        },
        "gate": gate.as_dict(),
        "outcomes_opened": False,
        "forbidden_access": AccessLedger.zero().snapshot(),
    }
    return {**core, "report_hash": canonical_hash(core)}


def _artifact_entry(
    path: str | Path,
    raw: bytes,
    *,
    rows: int,
    row_hash: str,
) -> dict[str, Any]:
    return {
        "path": Path(path).as_posix(),
        "sha256": sha256_bytes(raw),
        "rows": rows,
        "row_hash": row_hash,
    }


def build_pass_artifacts(
    events: Sequence[NormalizedEvent],
    cards: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Mapping[str, Mapping[str, int]]],
    control_gate: GateResult,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    event_rows = [event_csv_row(row) for row in events]
    proposal_rows = proposal_output_rows(events)
    output_cards = card_output_rows(cards)
    control_report = build_control_report(cards, controls, control_gate)
    raw_by_path = {
        EVENT_OUTPUT: deterministic_gzip(
            csv_bytes(event_rows, EVENT_COLUMNS)
        ),
        PROPOSAL_OUTPUT: deterministic_gzip(jsonl_bytes(proposal_rows)),
        CARD_OUTPUT: deterministic_gzip(jsonl_bytes(output_cards)),
        CONTROL_OUTPUT: canonical_json_bytes(control_report),
    }
    manifest = {
        "events": _artifact_entry(
            EVENT_OUTPUT,
            raw_by_path[EVENT_OUTPUT],
            rows=len(event_rows),
            row_hash=canonical_hash(event_rows),
        ),
        "proposals": _artifact_entry(
            PROPOSAL_OUTPUT,
            raw_by_path[PROPOSAL_OUTPUT],
            rows=len(proposal_rows),
            row_hash=canonical_hash(proposal_rows),
        ),
        "daily_cards": _artifact_entry(
            CARD_OUTPUT,
            raw_by_path[CARD_OUTPUT],
            rows=len(output_cards),
            row_hash=canonical_hash(output_cards),
        ),
        "controls": _artifact_entry(
            CONTROL_OUTPUT,
            raw_by_path[CONTROL_OUTPUT],
            rows=len(CONTROL_NAMES),
            row_hash=canonical_hash(control_report["metrics"]),
        ),
    }
    return raw_by_path, manifest


def _authority_report(
    seal: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "execution_seal": {
            "path": EXECUTION_SEAL_PATH.as_posix(),
            "sha256": sha256_file(EXECUTION_SEAL_PATH),
            "seal_hash": seal["seal_hash"],
            "shared_commit": seal["shared_commit"],
            "runner": seal["runner"],
            "tests": seal["tests"],
        },
        **dict(authority),
    }


def _first_failure(
    gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for index, gate in enumerate(gates, start=1):
        if not gate["passed"]:
            return {"gate_id": index, "name": gate["name"]}
    return None


def build_result_report(
    *,
    decision: str,
    authority: Mapping[str, Any],
    gates: Sequence[GateResult],
    source_audit: Mapping[str, Any],
    event_count: int,
    proposal_count: int,
    card_count: int,
    artifacts: Mapping[str, Any] | None,
    ledger: AccessLedger,
    error: BaseException | None = None,
) -> dict[str, Any]:
    gate_rows = [gate.as_dict() for gate in gates]
    first_failure = _first_failure(gate_rows)
    if decision == "pass":
        if (
            len(gate_rows) != len(GATE_NAMES)
            or first_failure is not None
            or artifacts is None
            or any(not row["passed"] for row in gate_rows)
        ):
            raise RuntimeError("GIPR pass report is incomplete")
    elif decision == "reject":
        if not gate_rows or first_failure is None or artifacts is not None:
            raise RuntimeError("GIPR rejection report is incomplete")
    else:
        raise RuntimeError("GIPR result decision changed")
    if [row["name"] for row in gate_rows] != list(
        GATE_NAMES[: len(gate_rows)]
    ):
        raise RuntimeError("GIPR gate order changed")
    core = {
        "protocol_version": RESULT_PROTOCOL,
        "policy_id": POLICY_ID,
        "decision": decision,
        "terminal_action": (
            PASS_ACTION if decision == "pass" else FAILURE_ACTION
        ),
        "profitability_result": False,
        "outcomes_opened": False,
        "source_incidence_opened": bool(
            source_audit.get("event_logs_opened", False)
        ),
        "authority": dict(authority),
        "gates": gate_rows,
        "first_failure": first_failure,
        "source_audit": dict(source_audit),
        "counts": {
            "events": event_count,
            "proposals": proposal_count,
            "daily_cards": card_count,
        },
        "artifacts": None if artifacts is None else dict(artifacts),
        "forbidden_access": ledger.snapshot(),
        "error": (
            None
            if error is None
            else {"type": type(error).__name__}
        ),
    }
    return {**core, "result_hash": canonical_hash(core)}


def _read_canonical_json(path: str | Path) -> dict[str, Any]:
    target = _safe_destination(path, create_parent=False)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"GIPR artifact path is unsafe: {path}")
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"GIPR artifact is unreadable: {path}") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError(f"GIPR artifact is noncanonical: {path}")
    return payload


def _entry_exists(path: str | Path) -> bool:
    return os.path.lexists(
        _safe_destination(path, create_parent=True)
    )


def _validate_result_report(
    payload: Mapping[str, Any],
    *,
    decision: str,
) -> None:
    core = {
        key: value for key, value in payload.items() if key != "result_hash"
    }
    gates = payload.get("gates")
    if (
        payload.get("protocol_version") != RESULT_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("decision") != decision
        or payload.get("terminal_action")
        != (PASS_ACTION if decision == "pass" else FAILURE_ACTION)
        or payload.get("profitability_result") is not False
        or payload.get("outcomes_opened") is not False
        or payload.get("result_hash") != canonical_hash(core)
        or not isinstance(gates, list)
        or [row.get("name") for row in gates]
        != list(GATE_NAMES[: len(gates)])
        or payload.get("first_failure") != _first_failure(gates)
    ):
        raise RuntimeError("GIPR terminal result report changed")
    forbidden = payload.get("forbidden_access")
    if (
        not isinstance(forbidden, dict)
        or set(forbidden) != set(FORBIDDEN_COUNTER_NAMES)
        or any(
            type(value) is not int or value < 0
            for value in forbidden.values()
        )
    ):
        raise RuntimeError("GIPR forbidden-access evidence is malformed")
    if decision == "pass":
        artifacts = payload.get("artifacts")
        if (
            len(gates) != len(GATE_NAMES)
            or any(not row.get("passed") for row in gates)
            or not isinstance(artifacts, dict)
            or set(artifacts)
            != {"events", "proposals", "daily_cards", "controls"}
            or payload.get("source_incidence_opened") is not True
            or forbidden != AccessLedger.zero().snapshot()
        ):
            raise RuntimeError("GIPR pass report is invalid")
        expected_paths = {
            "events": EVENT_OUTPUT.as_posix(),
            "proposals": PROPOSAL_OUTPUT.as_posix(),
            "daily_cards": CARD_OUTPUT.as_posix(),
            "controls": CONTROL_OUTPUT.as_posix(),
        }
        for name, entry in artifacts.items():
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "sha256", "rows", "row_hash"}
                or entry.get("path") != expected_paths[name]
                or not isinstance(entry.get("sha256"), str)
                or HEX64.fullmatch(entry["sha256"]) is None
                or type(entry.get("rows")) is not int
                or entry["rows"] < 0
                or not isinstance(entry.get("row_hash"), str)
                or HEX64.fullmatch(entry["row_hash"]) is None
            ):
                raise RuntimeError("GIPR pass artifact manifest is invalid")
    else:
        first_failure = payload.get("first_failure")
        if (
            not gates
            or first_failure is None
            or payload.get("artifacts") is not None
        ):
            raise RuntimeError("GIPR rejection report is invalid")
        if first_failure["name"] == GATE_NAMES[-1]:
            failed_gate = gates[first_failure["gate_id"] - 1]
            if (
                failed_gate.get("passed") is not False
                or failed_gate.get("metrics") != forbidden
                or not any(forbidden.values())
            ):
                raise RuntimeError(
                    "GIPR forbidden-access rejection evidence differs"
                )
        elif forbidden != AccessLedger.zero().snapshot():
            raise RuntimeError(
                "GIPR pre-forbidden-gate rejection has nonzero counters"
            )


def terminal_state() -> dict[str, Any] | None:
    pass_paths = (
        EVENT_OUTPUT,
        PROPOSAL_OUTPUT,
        CARD_OUTPUT,
        CONTROL_OUTPUT,
        PASS_REPORT,
    )
    rejection_exists = _entry_exists(REJECTION_REPORT)
    pass_exists = {path: _entry_exists(path) for path in pass_paths}
    if not rejection_exists and not any(pass_exists.values()):
        return None
    if rejection_exists and not any(pass_exists.values()):
        payload = _read_canonical_json(REJECTION_REPORT)
        _validate_result_report(payload, decision="reject")
        return payload
    if not rejection_exists and all(pass_exists.values()):
        payload = _read_canonical_json(PASS_REPORT)
        _validate_result_report(payload, decision="pass")
        artifacts = payload["artifacts"]
        for entry in artifacts.values():
            path = repository_path(entry["path"])
            if (
                path.is_symlink()
                or not path.is_file()
                or sha256_file(path) != entry["sha256"]
            ):
                raise RuntimeError("GIPR pass artifact hash changed")
        return payload
    raise RuntimeError("GIPR terminal state is partial or conflicting")


def _publish_rejection(payload: Mapping[str, Any]) -> None:
    _validate_result_report(payload, decision="reject")
    if any(
        _entry_exists(path)
        for path in (
            EVENT_OUTPUT,
            PROPOSAL_OUTPUT,
            CARD_OUTPUT,
            CONTROL_OUTPUT,
            PASS_REPORT,
        )
    ):
        raise RuntimeError("GIPR pass artifact exists before rejection")
    _write_once_bytes(REJECTION_REPORT, canonical_json_bytes(payload))


def _publish_pass_group(
    raw_by_path: Mapping[Path, bytes],
    report: Mapping[str, Any],
) -> None:
    _validate_result_report(report, decision="pass")
    if set(raw_by_path) != {
        EVENT_OUTPUT,
        PROPOSAL_OUTPUT,
        CARD_OUTPUT,
        CONTROL_OUTPUT,
    }:
        raise RuntimeError("GIPR pass artifact group changed")
    expected_entries = {
        **dict(raw_by_path),
        PASS_REPORT: canonical_json_bytes(report),
    }
    rejection_exists = _entry_exists(REJECTION_REPORT)
    existing = {path: _entry_exists(path) for path in expected_entries}
    if not rejection_exists and all(existing.values()):
        for relative, raw in expected_entries.items():
            target = _safe_destination(relative, create_parent=False)
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != raw
            ):
                raise RuntimeError(
                    "GIPR existing pass artifact conflicts with publication"
                )
        return
    if rejection_exists or any(existing.values()):
        raise RuntimeError("GIPR terminal target already exists")
    entries = list(expected_entries.items())
    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for relative, raw in entries:
            target = _safe_destination(relative, create_parent=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".gipr-stage",
                dir=target.parent,
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, target))
        for temporary, target in staged:
            os.link(temporary, target)
            published.append((temporary, target))
        for parent in {target.parent for _, target in staged}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for temporary, target in reversed(published):
            try:
                if target.exists() and os.path.samefile(temporary, target):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _gate_error(name: str, error: BaseException) -> GateResult:
    return GateResult(
        name=name,
        passed=False,
        metrics={
            "gate_evaluation_completed": False,
            "error_type": type(error).__name__,
        },
        failure=f"{name} raised {type(error).__name__}",
    )


def _evaluate_gate(
    name: str,
    builder: Any,
) -> tuple[GateResult, BaseException | None]:
    try:
        gate = builder()
    except Exception as error:
        return _gate_error(name, error), error
    if not isinstance(gate, GateResult) or gate.name != name:
        error = RuntimeError("GIPR gate builder returned wrong identity")
        return _gate_error(name, error), error
    return gate, None


def _canonical_log_rows(
    rows: Sequence[CanonicalLog],
) -> list[dict[str, Any]]:
    return [row.canonical_dict() for row in rows]


def _materialized_event_rows(
    rows: Sequence[NormalizedEvent],
) -> list[dict[str, Any]]:
    return [row.canonical_dict() for row in rows]


def _source_configuration(config: Config) -> tuple[str, str]:
    primary = config.primary_rpc_url.strip()
    verification = config.verification_rpc_url.strip()
    if not primary or not verification:
        raise ValueError("GIPR run requires two nonempty RPC URLs")
    if primary.rstrip("/") == verification.rstrip("/"):
        raise ValueError("GIPR run requires two distinct RPC URLs")
    if (
        config.max_block_range <= 0
        or config.max_block_range > prereg.MAX_BLOCK_RANGE
        or config.header_batch_size <= 0
        or config.header_batch_size > prereg.HEADER_BATCH_SIZE
        or config.timeout_sec <= 0
        or config.max_retries < 0
        or config.request_interval_sec < 0
    ):
        raise ValueError("GIPR run configuration exceeds frozen bounds")
    return primary, verification


def run_official(config: Config) -> dict[str, Any]:
    existing = terminal_state()
    if existing is not None:
        seal = validate_execution_seal()
        authority = static_authority()
        if existing.get("authority") != _authority_report(seal, authority):
            raise RuntimeError("GIPR terminal authority changed")
        return existing

    primary_url, verification_url = _source_configuration(config)
    authority = static_authority()
    seal = validate_execution_seal()
    authority_record = _authority_report(seal, authority)
    if not _worktree_clean():
        raise RuntimeError("GIPR official run requires a clean worktree")
    used_gib = shutil.disk_usage(REPO_ROOT).used / (1024**3)
    if used_gib > prereg.DISK_LIMIT_GIB:
        raise RuntimeError("GIPR WSL disk usage exceeds frozen 300 GiB limit")
    if terminal_state() is not None:
        raise RuntimeError("GIPR terminal state appeared concurrently")

    ledger = AccessLedger.zero()
    gates: list[GateResult] = []
    source_audit: dict[str, Any] = {
        "boundary": None,
        "canonical_log_rows": 0,
        "canonical_log_hash": None,
        "normalized_event_rows": 0,
        "normalized_event_hash": None,
        "event_logs_opened": False,
        "provider_identity_persisted": False,
        "disk_limit_gib": prereg.DISK_LIMIT_GIB,
        "disk_below_limit_at_start": True,
    }
    events: list[NormalizedEvent] = []
    cards: list[dict[str, Any]] = []
    controls: dict[str, dict[str, dict[str, int]]] | None = None

    def reject(error: BaseException | None = None) -> dict[str, Any]:
        report = build_result_report(
            decision="reject",
            authority=authority_record,
            gates=gates,
            source_audit=source_audit,
            event_count=len(events),
            proposal_count=sum(
                row.event == "proposal_created" for row in events
            ),
            card_count=len(cards),
            artifacts=None,
            ledger=ledger,
            error=error,
        )
        _publish_rejection(report)
        return report

    rpc_options = {
        "timeout_sec": config.timeout_sec,
        "max_retries": config.max_retries,
        "request_interval_sec": config.request_interval_sec,
    }
    primary = ethereum.JsonRpcClient(primary_url, **rpc_options)
    verification = ethereum.JsonRpcClient(verification_url, **rpc_options)

    try:
        boundary = audit_boundary(primary, verification)
        source_audit["boundary"] = boundary
        gate = GateResult(
            name=GATE_NAMES[0],
            passed=(
                boundary["chain_id"] == prereg.CHAIN_ID
                and not boundary["event_logs_opened"]
                and all(
                    boundary["transport_heads_at_or_above_minimum"]
                )
            ),
            metrics=boundary,
            failure="",
        )
        if not gate.passed:
            gate = replace(
                gate,
                failure="frozen boundary audit did not reconcile",
            )
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[0], error))
        return reject(error)
    gates.append(gate)
    if not gate.passed:
        return reject()

    checkpoint = LogCheckpoint(config.checkpoint_db)
    try:
        try:
            source_audit["event_logs_opened"] = True
            logs_primary = fetch_logs(
                primary,
                role="primary",
                checkpoint=checkpoint,
                max_block_range=config.max_block_range,
            )
            logs_verification = fetch_logs(
                verification,
                role="verification",
                checkpoint=checkpoint,
                max_block_range=config.max_block_range,
            )
            primary_rows = _canonical_log_rows(logs_primary)
            verification_rows = _canonical_log_rows(logs_verification)
            primary_hash = canonical_hash(primary_rows)
            verification_hash = canonical_hash(verification_rows)
            source_audit["canonical_log_rows"] = len(primary_rows)
            source_audit["canonical_log_hash"] = primary_hash
            gate = GateResult(
                name=GATE_NAMES[1],
                passed=(
                    primary_rows == verification_rows
                    and primary_hash == verification_hash
                ),
                metrics={
                    "primary_rows": len(primary_rows),
                    "verification_rows": len(verification_rows),
                    "primary_hash": primary_hash,
                    "verification_hash": verification_hash,
                    "canonical_lists_equal": (
                        primary_rows == verification_rows
                    ),
                },
                failure="",
            )
            if not gate.passed:
                gate = replace(
                    gate,
                    failure="dual transport canonical logs differ",
                )
        except Exception as error:
            gates.append(_gate_error(GATE_NAMES[1], error))
            return reject(error)
        gates.append(gate)
        if not gate.passed:
            return reject()

        try:
            parsed_primary = parse_logs(logs_primary)
            parsed_verification = parse_logs(logs_verification)
            parsed_primary_rows = [
                row.canonical_dict() for row in parsed_primary
            ]
            parsed_verification_rows = [
                row.canonical_dict() for row in parsed_verification
            ]
            proposal_rows = [
                row
                for row in parsed_primary
                if row.event == "proposal_created"
            ]
            gate = GateResult(
                name=GATE_NAMES[2],
                passed=parsed_primary_rows == parsed_verification_rows,
                metrics={
                    "decoded_events": len(parsed_primary),
                    "decoded_proposals": len(proposal_rows),
                    "decoded_actions": sum(
                        len(row.targets) for row in proposal_rows
                    ),
                    "normalized_hash": canonical_hash(
                        parsed_primary_rows
                    ),
                    "deterministic_decode": (
                        parsed_primary_rows == parsed_verification_rows
                    ),
                },
                failure="",
            )
            if not gate.passed:
                gate = replace(
                    gate,
                    failure="dual decode results differ",
                )
        except Exception as error:
            gates.append(_gate_error(GATE_NAMES[2], error))
            return reject(error)
        gates.append(gate)
        if not gate.passed:
            return reject()

        try:
            required_headers = sorted(
                {
                    number
                    for row in parsed_primary
                    for number in (
                        row.block_number,
                        row.block_number + prereg.CONFIRMATION_BLOCKS,
                    )
                }
            )
            primary_headers = _fetch_headers(
                primary,
                required_headers,
                batch_size=config.header_batch_size,
            )
            verification_headers = _fetch_headers(
                verification,
                required_headers,
                batch_size=config.header_batch_size,
            )
            events = materialize_events(
                parsed_primary,
                primary_headers,
                verification_headers,
            )
            duplicate_events = materialize_events(
                parsed_verification,
                verification_headers,
                primary_headers,
            )
            event_rows = _materialized_event_rows(events)
            duplicate_rows = _materialized_event_rows(duplicate_events)
            event_hash = canonical_hash(event_rows)
            source_audit["normalized_event_rows"] = len(events)
            source_audit["normalized_event_hash"] = event_hash
            gate = GateResult(
                name=GATE_NAMES[3],
                passed=event_rows == duplicate_rows,
                metrics={
                    "required_headers_per_transport": len(
                        required_headers
                    ),
                    "materialized_events": len(events),
                    "materialized_event_hash": event_hash,
                    "dual_header_materialization_equal": (
                        event_rows == duplicate_rows
                    ),
                },
                failure="",
            )
            if not gate.passed:
                gate = replace(
                    gate,
                    failure="canonical header materialization differs",
                )
        except Exception as error:
            gates.append(_gate_error(GATE_NAMES[3], error))
            return reject(error)
        gates.append(gate)
        if not gate.passed:
            return reject()
    finally:
        checkpoint.close()

    try:
        lifecycle = validate_lifecycle(events)
        gate = GateResult(
            name=GATE_NAMES[4],
            passed=True,
            metrics={
                "proposal_lifecycles": len(lifecycle),
                "events": len(events),
                "lifecycle_hash": canonical_hash(
                    [
                        {
                            "governor_address": key[0],
                            "proposal_id": str(key[1]),
                            "events": [
                                row.event for row in lifecycle[key]
                            ],
                        }
                        for key in sorted(lifecycle)
                    ]
                ),
            },
        )
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[4], error))
        return reject(error)
    gates.append(gate)

    try:
        cards = build_daily_cards(events)
        support = split_support_metrics(events, cards)
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[5], error))
        return reject(error)

    builders = (
        lambda: gate_split_proposal_action_support(support),
        lambda: gate_cross_protocol_support(support),
        lambda: gate_structural_vocabulary(support),
        lambda: gate_daily_schedule(support),
        lambda: future_append_gate(events, cards),
    )
    for expected_name, builder in zip(
        GATE_NAMES[5:10],
        builders,
        strict=True,
    ):
        gate, error = _evaluate_gate(expected_name, builder)
        gates.append(gate)
        if not gate.passed:
            return reject(error)

    try:
        controls = control_metrics(events, cards)
    except Exception as error:
        gates.append(_gate_error(GATE_NAMES[10], error))
        return reject(error)
    gate, error = _evaluate_gate(
        GATE_NAMES[10],
        lambda: gate_controls(controls),
    )
    gates.append(gate)
    if not gate.passed:
        return reject(error)
    control_gate = gate

    gate, error = _evaluate_gate(
        GATE_NAMES[11],
        lambda: forbidden_gate(ledger),
    )
    gates.append(gate)
    if not gate.passed:
        return reject(error)
    if controls is None:
        raise RuntimeError("GIPR control evidence disappeared")

    raw_by_path, artifacts = build_pass_artifacts(
        events,
        cards,
        controls,
        control_gate,
    )
    report = build_result_report(
        decision="pass",
        authority=authority_record,
        gates=gates,
        source_audit=source_audit,
        event_count=len(events),
        proposal_count=sum(
            row.event == "proposal_created" for row in events
        ),
        card_count=len(cards),
        artifacts=artifacts,
        ledger=ledger,
    )
    _publish_pass_group(raw_by_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-check")
    subparsers.add_parser("seal")
    subparsers.add_parser("validate-seal")
    run = subparsers.add_parser("run")
    run.add_argument(
        "--primary-rpc-url",
        default=os.environ.get("GIPR_PRIMARY_RPC_URL", ""),
    )
    run.add_argument(
        "--verification-rpc-url",
        default=os.environ.get("GIPR_VERIFICATION_RPC_URL", ""),
    )
    run.add_argument(
        "--checkpoint-db",
        default=str(DEFAULT_CHECKPOINT_DB),
    )
    run.add_argument(
        "--max-block-range",
        type=int,
        default=prereg.MAX_BLOCK_RANGE,
    )
    run.add_argument(
        "--header-batch-size",
        type=int,
        default=prereg.HEADER_BATCH_SIZE,
    )
    run.add_argument("--timeout-sec", type=float, default=60.0)
    run.add_argument(
        "--max-retries",
        type=int,
        default=ethereum.DEFAULT_MAX_RETRIES,
    )
    run.add_argument(
        "--request-interval-sec",
        type=float,
        default=0.0,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.command == "self-check":
        sys.stdout.buffer.write(self_check_bytes())
        return
    if arguments.command == "seal":
        payload = create_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH.as_posix(),
                    "seal_hash": payload["seal_hash"],
                    "shared_commit": payload["shared_commit"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return
    if arguments.command == "validate-seal":
        payload = validate_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH.as_posix(),
                    "seal_hash": payload["seal_hash"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return
    report = run_official(
        Config(
            primary_rpc_url=arguments.primary_rpc_url,
            verification_rpc_url=arguments.verification_rpc_url,
            checkpoint_db=arguments.checkpoint_db,
            max_block_range=arguments.max_block_range,
            header_batch_size=arguments.header_batch_size,
            timeout_sec=arguments.timeout_sec,
            max_retries=arguments.max_retries,
            request_interval_sec=arguments.request_interval_sec,
        )
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "terminal_action": report["terminal_action"],
                "first_failure": report["first_failure"],
                "result_hash": report["result_hash"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    if report["decision"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
