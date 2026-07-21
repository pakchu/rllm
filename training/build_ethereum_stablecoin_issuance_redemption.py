"""Build a dual-replayed Ethereum USDT/USDC issuance-redemption source panel.

The builder reads only canonical Ethereum contract logs and block headers.  It
does not read BTC market data, funding, labels, returns, or strategy outcomes.
Every event is delayed until the canonical timestamp of block ``N + 64`` and
the complete source is required to replay identically through two independent
JSON-RPC transports.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence


CHAIN_ID = 1
DEFAULT_START = "2020-01-01T00:00:00Z"
DEFAULT_END_EXCLUSIVE = "2024-01-01T00:00:00Z"
DEFAULT_CONFIRMATION_BLOCKS = 64
DEFAULT_BLOCK_RANGE = 10_000
DEFAULT_HEADER_BATCH_SIZE = 100
DEFAULT_MAX_RETRIES = 6
DEFAULT_REQUEST_INTERVAL_SEC = 0.0
USER_AGENT = "rllm-source-research/1.0"

USDT_ADDRESS = "0xdac17f958d2ee523a2206206994597c13d831ec7"
USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

USDT_ISSUE_TOPIC = "0xcb8241adb0c3fdb35b70c24ce35c5eb0c17af7431c99f827d44a445ca624176a"
USDT_REDEEM_TOPIC = "0x702d5967f45f6513a38ffc42d6ba9bf230bd40e8f53b16363c7eb4fd2deb9a44"
USDT_DESTROYED_BLACK_FUNDS_TOPIC = (
    "0x61e6e66b0d6339b2980aecc6ccc0039736791f0ccde9ed512e789a7fbdd698c6"
)
USDT_DEPRECATE_TOPIC = (
    "0xcc358699805e9a8b7f77b522628c7cb9abd07d9efb86b6fb616af1609036a99e"
)
USDC_MINT_TOPIC = "0xab8530f87dc9b59234c4623bf917212bb2536d647574c8e7e5da92c2ede0c9f8"
USDC_BURN_TOPIC = "0xcc16f5dbb4873280815c1ee09dbd06736cffcc184412cf7a71a0fdb75d397ca5"

OFFICIAL_SOURCES = (
    "https://ethereum.org/developers/docs/apis/json-rpc/",
    "https://tether.to/en/supported-protocols/",
    "https://developers.circle.com/stablecoins/usdc-contract-addresses",
    "https://github.com/circlefin/stablecoin-evm",
)

OUTPUT_COLUMNS = (
    "asset",
    "contract_address",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "indexed_address_1",
    "indexed_address_2",
    "data_address",
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


@dataclass(frozen=True)
class EventSpec:
    asset: str
    address: str
    topic: str
    event: str
    event_sign: int
    topic_count: int
    data_words: int
    amount_word: int | None
    data_address_word: int | None = None
    decimals: int = 6


EVENT_SPECS = (
    EventSpec("usdt_eth", USDT_ADDRESS, USDT_ISSUE_TOPIC, "issue", 1, 1, 1, 0),
    EventSpec("usdt_eth", USDT_ADDRESS, USDT_REDEEM_TOPIC, "redeem", -1, 1, 1, 0),
    EventSpec(
        "usdt_eth",
        USDT_ADDRESS,
        USDT_DESTROYED_BLACK_FUNDS_TOPIC,
        "destroyed_black_funds",
        -1,
        1,
        2,
        1,
        0,
    ),
    EventSpec(
        "usdt_eth",
        USDT_ADDRESS,
        USDT_DEPRECATE_TOPIC,
        "deprecate",
        0,
        1,
        1,
        None,
        0,
    ),
    EventSpec("usdc_eth", USDC_ADDRESS, USDC_MINT_TOPIC, "mint", 1, 3, 1, 0),
    EventSpec("usdc_eth", USDC_ADDRESS, USDC_BURN_TOPIC, "burn", -1, 2, 1, 0),
)


@dataclass(frozen=True)
class Config:
    output_csv: str = (
        "data/ethereum_stablecoin_issuance_redemption_2020_2023/"
        "ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz"
    )
    manifest_output: str = "results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json"
    primary_rpc_url: str = ""
    verification_rpc_url: str = ""
    header_rpc_url: str = ""
    start: str = DEFAULT_START
    end_exclusive: str = DEFAULT_END_EXCLUSIVE
    confirmation_blocks: int = DEFAULT_CONFIRMATION_BLOCKS
    max_block_range: int = DEFAULT_BLOCK_RANGE
    header_batch_size: int = DEFAULT_HEADER_BATCH_SIZE
    timeout_sec: float = 60.0
    max_retries: int = DEFAULT_MAX_RETRIES
    request_interval_sec: float = DEFAULT_REQUEST_INTERVAL_SEC


@dataclass(frozen=True)
class BlockHeader:
    number: int
    block_hash: str
    parent_hash: str
    timestamp: int


class EthereumRpcError(RuntimeError):
    def __init__(self, code: int | None, message: str) -> None:
        super().__init__(f"Ethereum RPC error {code}: {message}")
        self.code = code
        self.rpc_message = message

    @property
    def transient(self) -> bool:
        return self.code == -32603


class Rpc(Protocol):
    def call(self, method: str, params: list[Any]) -> Any: ...

    def batch(self, requests: Sequence[tuple[str, list[Any]]]) -> list[Any]: ...


class JsonRpcClient:
    """Minimal JSON-RPC client that preserves request ordering."""

    def __init__(
        self,
        url: str,
        *,
        timeout_sec: float,
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_interval_sec: float = DEFAULT_REQUEST_INTERVAL_SEC,
    ) -> None:
        if not url.strip():
            raise ValueError("Ethereum RPC URL must be non-empty")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if request_interval_sec < 0:
            raise ValueError("request_interval_sec must be non-negative")
        self._url = url
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._request_interval_sec = request_interval_sec
        self._last_request_at: float | None = None
        self._next_id = 1

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self._request_interval_sec - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _retry_delay(exc: BaseException, attempt: int) -> float:
        if isinstance(exc, urllib.error.HTTPError):
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after is not None:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        return min(2.0**attempt, 16.0)

    def _post(self, payload: Any) -> Any:
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        )
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                with urllib.request.urlopen(
                    request, timeout=self._timeout_sec
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise
                if attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_delay(exc, attempt))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_delay(exc, attempt))
        raise AssertionError("bounded Ethereum RPC retry loop exhausted unexpectedly")

    @staticmethod
    def _result(response: Any, expected_id: int) -> Any:
        if not isinstance(response, dict) or response.get("id") != expected_id:
            raise RuntimeError("Ethereum RPC returned a malformed response identity")
        if response.get("error") is not None:
            error = response["error"]
            code = error.get("code") if isinstance(error, dict) else None
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise EthereumRpcError(
                code if isinstance(code, int) else None,
                str(message),
            )
        if "result" not in response:
            raise RuntimeError("Ethereum RPC response has no result")
        return response["result"]

    def call(self, method: str, params: list[Any]) -> Any:
        request_id = self._next_id
        self._next_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        for attempt in range(self._max_retries + 1):
            response = self._post(payload)
            try:
                return self._result(response, request_id)
            except EthereumRpcError as exc:
                if not exc.transient or attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_delay(exc, attempt))
        raise AssertionError(
            "bounded Ethereum JSON-RPC retry loop exhausted unexpectedly"
        )

    def batch(self, requests: Sequence[tuple[str, list[Any]]]) -> list[Any]:
        if not requests:
            return []
        payload: list[dict[str, Any]] = []
        request_ids: list[int] = []
        for method, params in requests:
            request_id = self._next_id
            self._next_id += 1
            request_ids.append(request_id)
            payload.append(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )
        for attempt in range(self._max_retries + 1):
            response = self._post(payload)
            if not isinstance(response, list) or len(response) != len(payload):
                raise RuntimeError("Ethereum RPC returned a malformed batch")
            by_id: dict[int, Any] = {}
            for item in response:
                if not isinstance(item, dict) or not isinstance(item.get("id"), int):
                    raise RuntimeError("Ethereum RPC batch item has no integer id")
                item_id = int(item["id"])
                if item_id in by_id:
                    raise RuntimeError("Ethereum RPC batch contains a duplicate id")
                by_id[item_id] = item
            if set(by_id) != set(request_ids):
                raise RuntimeError(
                    "Ethereum RPC batch response ids do not match requests"
                )
            try:
                return [
                    self._result(by_id[request_id], request_id)
                    for request_id in request_ids
                ]
            except EthereumRpcError as exc:
                if not exc.transient or attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_delay(exc, attempt))
        raise AssertionError("bounded Ethereum batch retry loop exhausted unexpectedly")


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("source boundary must include a timezone")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _quantity(value: Any, field: str) -> int:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"{field} must be a hexadecimal quantity")
    try:
        parsed = int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} is not hexadecimal") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be nonnegative")
    return parsed


def _hex_data(value: Any, field: str, *, bytes_length: int) -> str:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"{field} must be hexadecimal data")
    payload = value[2:].lower()
    if len(payload) != bytes_length * 2:
        raise ValueError(f"{field} must contain exactly {bytes_length} bytes")
    try:
        bytes.fromhex(payload)
    except ValueError as exc:
        raise ValueError(f"{field} is not hexadecimal") from exc
    return "0x" + payload


def _address(value: Any, field: str = "address") -> str:
    return _hex_data(value, field, bytes_length=20)


def _hash(value: Any, field: str) -> str:
    return _hex_data(value, field, bytes_length=32)


def _word_address(word: str, field: str) -> str:
    normalized = _hex_data(word, field, bytes_length=32)
    if normalized[2:26] != "0" * 24:
        raise ValueError(f"{field} is not a left-padded Ethereum address")
    return "0x" + normalized[-40:]


def parse_header(raw: Any, *, expected_number: int | None = None) -> BlockHeader:
    if not isinstance(raw, dict):
        raise RuntimeError("Ethereum block header is missing")
    number = _quantity(raw.get("number"), "block number")
    if expected_number is not None and number != expected_number:
        raise RuntimeError("Ethereum block header number does not match request")
    return BlockHeader(
        number=number,
        block_hash=_hash(raw.get("hash"), "block hash"),
        parent_hash=_hash(raw.get("parentHash"), "parent hash"),
        timestamp=_quantity(raw.get("timestamp"), "block timestamp"),
    )


def get_header(rpc: Rpc, block: int | str) -> BlockHeader:
    parameter = hex(block) if isinstance(block, int) else block
    expected = block if isinstance(block, int) else None
    return parse_header(
        rpc.call("eth_getBlockByNumber", [parameter, False]),
        expected_number=expected,
    )


def find_first_block_at_or_after(rpc: Rpc, timestamp: int) -> BlockHeader:
    latest = get_header(rpc, "latest")
    if timestamp > latest.timestamp:
        raise RuntimeError("source boundary is later than the Ethereum head")
    low = 0
    high = latest.number
    while low < high:
        middle = (low + high) // 2
        header = get_header(rpc, middle)
        if header.timestamp < timestamp:
            low = middle + 1
        else:
            high = middle
    boundary = get_header(rpc, low)
    if boundary.timestamp < timestamp:
        raise RuntimeError("failed to locate Ethereum time boundary")
    if low:
        previous = get_header(rpc, low - 1)
        if previous.timestamp >= timestamp:
            raise RuntimeError("Ethereum time boundary is not the first matching block")
    return boundary


def _spec_maps() -> tuple[dict[str, EventSpec], dict[str, tuple[str, ...]]]:
    by_topic = {spec.topic: spec for spec in EVENT_SPECS}
    by_address: dict[str, list[str]] = {}
    for spec in EVENT_SPECS:
        by_address.setdefault(spec.address, []).append(spec.topic)
    return by_topic, {key: tuple(value) for key, value in by_address.items()}


def fetch_logs(
    rpc: Rpc, start_block: int, end_block: int, *, max_block_range: int
) -> list[dict[str, Any]]:
    if start_block > end_block:
        return []
    if max_block_range <= 0:
        raise ValueError("max_block_range must be positive")
    _, by_address = _spec_maps()
    rows: list[dict[str, Any]] = []
    for address, topics in sorted(by_address.items()):
        for first in range(start_block, end_block + 1, max_block_range):
            last = min(first + max_block_range - 1, end_block)
            result = rpc.call(
                "eth_getLogs",
                [
                    {
                        "fromBlock": hex(first),
                        "toBlock": hex(last),
                        "address": address,
                        "topics": [list(topics)],
                    }
                ],
            )
            if not isinstance(result, list):
                raise RuntimeError("eth_getLogs result is not a list")
            for raw in result:
                if not isinstance(raw, dict):
                    raise RuntimeError("eth_getLogs row is not an object")
                rows.append(raw)
    return rows


def normalize_log(
    raw: dict[str, Any], *, start_block: int, end_block: int
) -> dict[str, Any]:
    by_topic, _ = _spec_maps()
    address = _address(raw.get("address"))
    topics = raw.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError("Ethereum event has no topics")
    normalized_topics = [_hash(topic, "event topic") for topic in topics]
    spec = by_topic.get(normalized_topics[0])
    if spec is None or address != spec.address:
        raise ValueError("Ethereum event does not match a frozen contract/topic")
    if len(normalized_topics) != spec.topic_count:
        raise ValueError("Ethereum event topic count differs from frozen ABI")
    if raw.get("removed") is not False:
        raise RuntimeError("Ethereum event is removed or has ambiguous reorg state")
    block_number = _quantity(raw.get("blockNumber"), "event block number")
    if not start_block <= block_number <= end_block:
        raise RuntimeError("Ethereum RPC returned an event outside the requested range")
    data = _hex_data(raw.get("data"), "event data", bytes_length=32 * spec.data_words)
    words = [
        "0x" + data[2 + offset : 2 + offset + 64]
        for offset in range(0, 64 * spec.data_words, 64)
    ]
    amount_raw = ""
    if spec.amount_word is not None:
        amount = int(words[spec.amount_word], 16)
        if amount <= 0:
            raise ValueError("stablecoin issuance/redemption amount must be positive")
        amount_raw = str(amount)
    indexed_addresses = [
        _word_address(topic, "indexed event address") for topic in normalized_topics[1:]
    ]
    data_address = (
        _word_address(words[spec.data_address_word], "event data address")
        if spec.data_address_word is not None
        else ""
    )
    return {
        "asset": spec.asset,
        "contract_address": spec.address,
        "event": spec.event,
        "event_sign": spec.event_sign,
        "amount_raw": amount_raw,
        "decimals": spec.decimals,
        "indexed_address_1": indexed_addresses[0] if indexed_addresses else "",
        "indexed_address_2": indexed_addresses[1] if len(indexed_addresses) > 1 else "",
        "data_address": data_address,
        "block_number": block_number,
        "block_hash": _hash(raw.get("blockHash"), "event block hash"),
        "transaction_hash": _hash(raw.get("transactionHash"), "transaction hash"),
        "transaction_index": _quantity(
            raw.get("transactionIndex"), "transaction index"
        ),
        "log_index": _quantity(raw.get("logIndex"), "log index"),
    }


def fetch_headers(
    rpc: Rpc, block_numbers: Sequence[int], *, batch_size: int
) -> dict[int, BlockHeader]:
    if batch_size <= 0:
        raise ValueError("header_batch_size must be positive")
    unique = sorted(set(block_numbers))
    headers: dict[int, BlockHeader] = {}
    for offset in range(0, len(unique), batch_size):
        batch_numbers = unique[offset : offset + batch_size]
        raw_headers = rpc.batch(
            [("eth_getBlockByNumber", [hex(number), False]) for number in batch_numbers]
        )
        for number, raw in zip(batch_numbers, raw_headers, strict=True):
            headers[number] = parse_header(raw, expected_number=number)
    return headers


def materialize_rows(
    rpc: Rpc,
    raw_logs: Sequence[dict[str, Any]],
    *,
    start_block: int,
    end_block: int,
    confirmation_blocks: int,
    header_batch_size: int,
) -> list[dict[str, Any]]:
    if confirmation_blocks <= 0:
        raise ValueError("confirmation_blocks must be positive")
    normalized = normalize_logs(raw_logs, start_block=start_block, end_block=end_block)
    return materialize_normalized_rows(
        rpc,
        normalized,
        confirmation_blocks=confirmation_blocks,
        header_batch_size=header_batch_size,
    )


def normalize_logs(
    raw_logs: Sequence[dict[str, Any]], *, start_block: int, end_block: int
) -> list[dict[str, Any]]:
    normalized = [
        normalize_log(raw, start_block=start_block, end_block=end_block)
        for raw in raw_logs
    ]
    identities: set[tuple[str, str, int]] = set()
    for row in normalized:
        identity = (row["block_hash"], row["transaction_hash"], row["log_index"])
        if identity in identities:
            raise RuntimeError(
                "Ethereum source contains a duplicate canonical log identity"
            )
        identities.add(identity)
    normalized.sort(
        key=lambda row: (
            row["block_number"],
            row["transaction_index"],
            row["log_index"],
            row["asset"],
            row["event"],
        )
    )
    return normalized


def materialize_normalized_rows(
    rpc: Rpc,
    normalized: Sequence[dict[str, Any]],
    *,
    confirmation_blocks: int,
    header_batch_size: int,
) -> list[dict[str, Any]]:
    if confirmation_blocks <= 0:
        raise ValueError("confirmation_blocks must be positive")
    needed = [row["block_number"] for row in normalized]
    needed.extend(row["block_number"] + confirmation_blocks for row in normalized)
    headers = fetch_headers(rpc, needed, batch_size=header_batch_size)
    rows: list[dict[str, Any]] = []
    for row in normalized:
        event_header = headers[row["block_number"]]
        confirmation_number = row["block_number"] + confirmation_blocks
        confirmation_header = headers[confirmation_number]
        if row["block_hash"] != event_header.block_hash:
            raise RuntimeError("event block hash differs from canonical header")
        if confirmation_header.timestamp <= event_header.timestamp:
            raise RuntimeError("confirmation block does not follow the event in time")
        rows.append(
            {
                **row,
                "block_timestamp": _format_timestamp(event_header.timestamp),
                "confirmation_block_number": confirmation_number,
                "confirmation_block_hash": confirmation_header.block_hash,
                "available_at": _format_timestamp(confirmation_header.timestamp),
            }
        )
    rows.sort(
        key=lambda row: (
            row["block_number"],
            row["transaction_index"],
            row["log_index"],
            row["asset"],
            row["event"],
        )
    )
    return rows


def _contract_code_audit(rpc: Rpc, start_block: int, end_block: int) -> dict[str, str]:
    audit: dict[str, str] = {}
    for address in sorted({spec.address for spec in EVENT_SPECS}):
        for label, block in (("start", start_block), ("end", end_block)):
            code = rpc.call("eth_getCode", [address, hex(block)])
            if not isinstance(code, str) or code in {"0x", "0x0"}:
                raise RuntimeError(
                    f"stablecoin contract has no code at {label} boundary"
                )
            normalized = _hex_data(
                code, "contract bytecode", bytes_length=(len(code) - 2) // 2
            )
            audit[f"{address}:{label}"] = hashlib.sha256(
                bytes.fromhex(normalized[2:])
            ).hexdigest()
    return audit


def collect_log_source(
    rpc: Rpc, cfg: Config
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
    chain_id = _quantity(rpc.call("eth_chainId", []), "chain id")
    if chain_id != CHAIN_ID:
        raise RuntimeError(
            f"expected Ethereum mainnet chain id {CHAIN_ID}, got {chain_id}"
        )
    start_dt = _parse_datetime(cfg.start)
    end_dt = _parse_datetime(cfg.end_exclusive)
    if start_dt >= end_dt:
        raise ValueError("source interval must be non-empty")
    start_header = find_first_block_at_or_after(rpc, math.floor(start_dt.timestamp()))
    end_header = find_first_block_at_or_after(rpc, math.floor(end_dt.timestamp()))
    if start_header.number >= end_header.number:
        raise RuntimeError("Ethereum source interval contains no blocks")
    last_source_block = end_header.number - 1
    code_audit = _contract_code_audit(rpc, start_header.number, last_source_block)
    raw_logs = fetch_logs(
        rpc,
        start_header.number,
        last_source_block,
        max_block_range=cfg.max_block_range,
    )
    normalized = normalize_logs(
        raw_logs,
        start_block=start_header.number,
        end_block=last_source_block,
    )
    if not normalized:
        raise RuntimeError("Ethereum stablecoin source returned no events")
    if any(row["event"] == "deprecate" for row in normalized):
        raise RuntimeError(
            "USDT deprecation event requires an explicitly reviewed successor contract"
        )
    audit = {
        "chain_id": chain_id,
        "start_boundary": {
            "requested": cfg.start,
            "block_number": start_header.number,
            "block_hash": start_header.block_hash,
            "block_timestamp": _format_timestamp(start_header.timestamp),
        },
        "end_boundary_exclusive": {
            "requested": cfg.end_exclusive,
            "block_number": end_header.number,
            "block_hash": end_header.block_hash,
            "block_timestamp": _format_timestamp(end_header.timestamp),
        },
        "last_source_block": last_source_block,
        "contract_code_sha256": code_audit,
        "event_rows": len(normalized),
        "canonical_log_hash": _canonical_hash(normalized),
    }
    return normalized, audit, start_header.number, last_source_block


def collect_source(
    rpc: Rpc, cfg: Config, *, header_rpc: Rpc | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized, log_audit, _, _ = collect_log_source(rpc, cfg)
    header_rpc = header_rpc or rpc
    rows = materialize_normalized_rows(
        header_rpc,
        normalized,
        confirmation_blocks=cfg.confirmation_blocks,
        header_batch_size=cfg.header_batch_size,
    )
    required_finalized_block = max(row["confirmation_block_number"] for row in rows)
    finalized_header = get_header(header_rpc, "finalized")
    if finalized_header.number < required_finalized_block:
        raise RuntimeError(
            "Ethereum finalized head does not cover all confirmation blocks"
        )
    audit = {
        "log_source": log_audit,
        "materialized_rows": len(rows),
        "canonical_event_hash": _canonical_hash(rows),
        "finalized_coverage": {
            "required_through_block": required_finalized_block,
            "observed_finalized_block_at_least_required": True,
        },
    }
    return rows, audit


def _write_csv_gz(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(OUTPUT_COLUMNS)
                writer.writerows(
                    [[row[column] for column in OUTPUT_COLUMNS] for row in rows]
                )


def _transport_hosts(cfg: Config) -> tuple[str, str, str]:
    primary_host = urllib.parse.urlsplit(cfg.primary_rpc_url).hostname or ""
    verification_host = urllib.parse.urlsplit(cfg.verification_rpc_url).hostname or ""
    header_url = cfg.header_rpc_url or cfg.primary_rpc_url
    header_host = urllib.parse.urlsplit(header_url).hostname or ""
    if not primary_host or not verification_host:
        raise ValueError("both Ethereum RPC URLs must have hostnames")
    if not header_host:
        raise ValueError("Ethereum header RPC URL must have a hostname")
    if primary_host.lower() == verification_host.lower():
        raise ValueError("verification RPC must use an independent hostname")
    return primary_host, verification_host, header_host


def build_outputs(
    cfg: Config,
    *,
    primary_rpc: Rpc | None = None,
    verification_rpc: Rpc | None = None,
    header_rpc: Rpc | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_host, _, header_host = _transport_hosts(cfg)
    primary_rpc = primary_rpc or JsonRpcClient(
        cfg.primary_rpc_url,
        timeout_sec=cfg.timeout_sec,
        max_retries=cfg.max_retries,
        request_interval_sec=cfg.request_interval_sec,
    )
    verification_rpc = verification_rpc or JsonRpcClient(
        cfg.verification_rpc_url,
        timeout_sec=cfg.timeout_sec,
        max_retries=cfg.max_retries,
        request_interval_sec=cfg.request_interval_sec,
    )
    if header_rpc is None:
        if cfg.header_rpc_url:
            header_rpc = JsonRpcClient(
                cfg.header_rpc_url,
                timeout_sec=cfg.timeout_sec,
                max_retries=cfg.max_retries,
                request_interval_sec=cfg.request_interval_sec,
            )
        else:
            header_rpc = primary_rpc
    primary_rows, primary_audit = collect_source(
        primary_rpc, cfg, header_rpc=header_rpc
    )
    _, verification_audit, _, _ = collect_log_source(verification_rpc, cfg)
    if primary_audit["log_source"] != verification_audit:
        raise RuntimeError("independent Ethereum RPC replays disagree")
    counts = Counter(f"{row['asset']}:{row['event']}" for row in primary_rows)
    year_counts = Counter(row["block_timestamp"][:4] for row in primary_rows)
    core = {
        "protocol_version": "ethereum_stablecoin_issuance_redemption_source_v1",
        "official_sources": list(OFFICIAL_SOURCES),
        "source_contract": {
            "chain_id": CHAIN_ID,
            "start": cfg.start,
            "end_exclusive": cfg.end_exclusive,
            "confirmation_blocks": cfg.confirmation_blocks,
            "max_block_range": cfg.max_block_range,
            "contracts": [
                {
                    "asset": spec.asset,
                    "address": spec.address,
                    "event": spec.event,
                    "topic": spec.topic,
                    "event_sign": spec.event_sign,
                    "topic_count": spec.topic_count,
                    "data_words": spec.data_words,
                    "amount_word": spec.amount_word,
                    "data_address_word": spec.data_address_word,
                    "decimals": spec.decimals,
                }
                for spec in EVENT_SPECS
            ],
        },
        "source_audit": primary_audit,
        "dual_replay": {
            "independent_transport_count": 2,
            "canonical_replay_equal": True,
            "canonical_log_hashes": [
                primary_audit["log_source"]["canonical_log_hash"],
                verification_audit["canonical_log_hash"],
            ],
            "provider_urls_embedded": False,
        },
        "header_materialization": {
            "transport_independent_from_primary_logs": (
                header_host.lower() != primary_host.lower()
            ),
            "event_block_hash_cross_checked": True,
            "provider_url_embedded": False,
        },
        "event_counts": dict(sorted(counts.items())),
        "year_counts": dict(sorted(year_counts.items())),
        "output": {
            "path": cfg.output_csv,
            "columns": list(OUTPUT_COLUMNS),
            "rows": len(primary_rows),
        },
        "outcome_boundary": {
            "source_only": True,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
            "post_2023_contract_event_rows_read": 0,
            "post_2023_confirmation_headers_may_be_read": True,
        },
    }
    return primary_rows, core


def run(cfg: Config) -> dict[str, Any]:
    rows, core = build_outputs(cfg)
    output = Path(cfg.output_csv)
    _write_csv_gz(output, rows)
    core["output"]["sha256"] = _sha256(output)
    core["output"]["bytes"] = output.stat().st_size
    manifest = {**core, "manifest_hash": _canonical_hash(core)}
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return manifest


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", default=Config.output_csv)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument(
        "--primary-rpc-url",
        default=os.environ.get("ETHEREUM_PRIMARY_RPC_URL", ""),
    )
    parser.add_argument(
        "--verification-rpc-url",
        default=os.environ.get("ETHEREUM_VERIFICATION_RPC_URL", ""),
    )
    parser.add_argument(
        "--header-rpc-url",
        default=os.environ.get("ETHEREUM_HEADER_RPC_URL", ""),
    )
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end-exclusive", default=Config.end_exclusive)
    parser.add_argument(
        "--confirmation-blocks", type=int, default=Config.confirmation_blocks
    )
    parser.add_argument("--max-block-range", type=int, default=Config.max_block_range)
    parser.add_argument(
        "--header-batch-size", type=int, default=Config.header_batch_size
    )
    parser.add_argument("--timeout-sec", type=float, default=Config.timeout_sec)
    parser.add_argument("--max-retries", type=int, default=Config.max_retries)
    parser.add_argument(
        "--request-interval-sec", type=float, default=Config.request_interval_sec
    )
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
