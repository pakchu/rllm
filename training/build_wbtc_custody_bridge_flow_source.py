"""Build a dual-replayed, source-only WBTC mint/burn event panel.

The builder reads only canonical Ethereum logs, transaction receipts, block
headers, and contract bytecode.  It never opens BTC prices, funding, labels,
returns, strategy outcomes, or post-2023 WBTC events.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from training import build_ethereum_stablecoin_issuance_redemption as ethereum
from training import preregister_wbtc_custody_bridge_flow_source as protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_wbtc_custody_bridge_flow_source.py")
PROTOCOL_PATH = Path(
    "results/wbtc_custody_bridge_flow_source_protocol_2026-07-23.json"
)
PROTOCOL_SHA256 = "e85811511d23cc12309a73461c9bf8b70356ef5a6b96ff2df28f61face909c64"
PROTOCOL_MANIFEST_HASH = (
    "dbf263a3baae3c35833e4b2297880e7c2ec5ea23b6e2065f34ce14106a2a72a9"
)
ETHEREUM_HELPER_PATH = Path(
    "training/build_ethereum_stablecoin_issuance_redemption.py"
)
ETHEREUM_HELPER_SHA256 = (
    "af699d91404e44298573ca148c6cf1b90b68d9aac2972da9ad228a26b9e8a9a5"
)

DEFAULT_OUTPUT_CSV = (
    "data/wbtc_custody_bridge_flow_2020_2023/"
    "wbtc_mint_burn_2020_2023.csv.gz"
)
DEFAULT_MANIFEST_OUTPUT = (
    "results/wbtc_custody_bridge_flow_source_manifest_2026-07-23.json"
)
DEFAULT_CHECKPOINT_DB = "data/.cache/wcbf_source_backfill.sqlite3"

OUTPUT_COLUMNS = (
    "asset",
    "contract_address",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "actor_address",
    "block_number",
    "block_hash",
    "block_timestamp",
    "transaction_hash",
    "transaction_index",
    "semantic_log_index",
    "companion_transfer_log_index",
    "confirmation_block_number",
    "confirmation_block_hash",
    "available_at",
)


@dataclass(frozen=True)
class Config:
    output_csv: str = DEFAULT_OUTPUT_CSV
    manifest_output: str = DEFAULT_MANIFEST_OUTPUT
    primary_rpc_url: str = ""
    verification_rpc_url: str = ""
    header_rpc_url: str = ""
    receipt_rpc_url: str = ""
    checkpoint_db: str = DEFAULT_CHECKPOINT_DB
    max_block_range: int = protocol.MAX_BLOCK_RANGE
    header_batch_size: int = protocol.HEADER_BATCH_SIZE
    receipt_batch_size: int = protocol.RECEIPT_BATCH_SIZE
    timeout_sec: float = 60.0
    max_retries: int = ethereum.DEFAULT_MAX_RETRIES
    request_interval_sec: float = 0.0


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _sha256(path: str | Path) -> str:
    return ethereum._sha256(_repo_path(path))


def _load_protocol() -> dict[str, Any]:
    if _sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("WCBF source protocol file hash changed")
    if _sha256(ETHEREUM_HELPER_PATH) != ETHEREUM_HELPER_SHA256:
        raise RuntimeError("WCBF Ethereum helper implementation hash changed")
    payload = json.loads(_repo_path(PROTOCOL_PATH).read_text(encoding="utf-8"))
    if payload.get("manifest_hash") != PROTOCOL_MANIFEST_HASH:
        raise RuntimeError("WCBF source protocol manifest hash changed")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if ethereum._canonical_hash(core) != PROTOCOL_MANIFEST_HASH:
        raise RuntimeError("WCBF source protocol is not canonically hash-bound")
    if payload.get("outcome_boundary", {}).get("source_only") is not True:
        raise RuntimeError("WCBF source protocol no longer has a source-only boundary")
    return payload


def _check_disk_limit() -> None:
    used_bytes = shutil.disk_usage(REPO_ROOT).used
    limit_bytes = protocol.DISK_LIMIT_GIB * 1024**3
    if used_bytes >= limit_bytes:
        raise RuntimeError(
            f"WCBF source build requires used disk below {protocol.DISK_LIMIT_GIB} GiB"
        )


def _progress(message: str) -> None:
    print(f"[wcbf-source] {message}", file=sys.stderr, flush=True)


class LogCheckpoint:
    """Small protocol-bound cache for completed eth_getLogs block ranges."""

    def __init__(self, path: str | Path) -> None:
        candidate = _repo_path(path)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.is_symlink():
            raise RuntimeError("WCBF checkpoint path may not be a symlink")
        self.path = candidate
        self.connection = sqlite3.connect(candidate)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS log_chunks (
                role TEXT NOT NULL,
                first_block INTEGER NOT NULL,
                last_block INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (role, first_block, last_block)
            )
            """
        )
        identity = ethereum._canonical_hash(
            {
                "protocol_sha256": PROTOCOL_SHA256,
                "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
                "chain_id": protocol.CHAIN_ID,
                "contract": protocol.WBTC_ADDRESS,
                "topics": [spec.topic for spec in protocol.EVENT_SPECS],
                "start_block": protocol.START_BOUNDARY_BLOCK,
                "last_source_block": protocol.LAST_SOURCE_BLOCK,
            }
        )
        existing = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'source_identity'"
        ).fetchone()
        if existing is not None and existing[0] != identity:
            self.connection.close()
            raise RuntimeError("WCBF checkpoint belongs to a different source protocol")
        self.connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES('source_identity', ?)",
            (identity,),
        )
        self.connection.commit()

    def get(self, role: str, first_block: int, last_block: int) -> list[dict[str, Any]] | None:
        found = self.connection.execute(
            """
            SELECT payload_json FROM log_chunks
            WHERE role = ? AND first_block = ? AND last_block = ?
            """,
            (role, first_block, last_block),
        ).fetchone()
        if found is None:
            return None
        payload = json.loads(found[0])
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise RuntimeError("WCBF checkpoint log chunk is malformed")
        return payload

    def put(
        self,
        role: str,
        first_block: int,
        last_block: int,
        rows: Sequence[dict[str, Any]],
    ) -> None:
        payload = json.dumps(
            list(rows),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO log_chunks(role, first_block, last_block, payload_json)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(role, first_block, last_block)
                DO UPDATE SET payload_json = excluded.payload_json
                """,
                (role, first_block, last_block, payload),
            )

    def close(self) -> None:
        self.connection.close()


def _event_specs_by_topic() -> dict[str, protocol.EventSpec]:
    return {spec.topic: spec for spec in protocol.EVENT_SPECS}


def fetch_semantic_logs(
    rpc: ethereum.Rpc,
    start_block: int,
    end_block: int,
    *,
    max_block_range: int,
    checkpoint: LogCheckpoint | None = None,
    checkpoint_role: str = "",
) -> list[dict[str, Any]]:
    if start_block > end_block:
        return []
    if max_block_range <= 0 or max_block_range > protocol.MAX_BLOCK_RANGE:
        raise ValueError("WCBF max block range is outside the frozen guard")
    rows: list[dict[str, Any]] = []
    topics = [spec.topic for spec in protocol.EVENT_SPECS]
    for first in range(start_block, end_block + 1, max_block_range):
        last = min(first + max_block_range - 1, end_block)
        result = (
            checkpoint.get(checkpoint_role, first, last)
            if checkpoint is not None
            else None
        )
        if result is None:
            result = rpc.call(
                "eth_getLogs",
                [
                    {
                        "fromBlock": hex(first),
                        "toBlock": hex(last),
                        "address": protocol.WBTC_ADDRESS,
                        "topics": [topics],
                    }
                ],
            )
            if checkpoint is not None:
                if not isinstance(result, list):
                    raise RuntimeError("WCBF eth_getLogs result is not a list")
                checkpoint.put(checkpoint_role, first, last, result)
        if not isinstance(result, list):
            raise RuntimeError("WCBF eth_getLogs result is not a list")
        for raw in result:
            if not isinstance(raw, dict):
                raise RuntimeError("WCBF eth_getLogs row is not an object")
            rows.append(raw)
    return rows


def normalize_semantic_log(
    raw: dict[str, Any], *, start_block: int, end_block: int
) -> dict[str, Any]:
    address = ethereum._address(raw.get("address"))
    topics = raw.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError("WCBF event has no topics")
    normalized_topics = [ethereum._hash(topic, "event topic") for topic in topics]
    spec = _event_specs_by_topic().get(normalized_topics[0])
    if spec is None or address != protocol.WBTC_ADDRESS:
        raise ValueError("WCBF event does not match the frozen contract/topic")
    if len(normalized_topics) != 2:
        raise ValueError("WCBF event topic count differs from the frozen ABI")
    if raw.get("removed") is not False:
        raise RuntimeError("WCBF event is removed or has ambiguous reorg state")
    block_number = ethereum._quantity(raw.get("blockNumber"), "event block number")
    if not start_block <= block_number <= end_block:
        raise RuntimeError("WCBF event is outside the frozen block envelope")
    data = ethereum._hex_data(raw.get("data"), "event data", bytes_length=32)
    amount = int(data, 16)
    if amount <= 0:
        raise ValueError("WCBF mint/burn amount must be positive")
    actor = ethereum._word_address(normalized_topics[1], "event actor")
    return {
        "asset": "wbtc_eth",
        "contract_address": protocol.WBTC_ADDRESS,
        "event": spec.event,
        "event_sign": spec.event_sign,
        "amount_raw": str(amount),
        "decimals": protocol.WBTC_DECIMALS,
        "actor_address": actor,
        "actor_topic": normalized_topics[1],
        "event_data": data,
        "block_number": block_number,
        "block_hash": ethereum._hash(raw.get("blockHash"), "event block hash"),
        "transaction_hash": ethereum._hash(
            raw.get("transactionHash"), "transaction hash"
        ),
        "transaction_index": ethereum._quantity(
            raw.get("transactionIndex"), "transaction index"
        ),
        "log_index": ethereum._quantity(raw.get("logIndex"), "semantic log index"),
    }


def normalize_semantic_logs(
    raw_logs: Sequence[dict[str, Any]], *, start_block: int, end_block: int
) -> list[dict[str, Any]]:
    rows = [
        normalize_semantic_log(raw, start_block=start_block, end_block=end_block)
        for raw in raw_logs
    ]
    identities: set[tuple[str, str, int]] = set()
    for row in rows:
        identity = (row["block_hash"], row["transaction_hash"], row["log_index"])
        if identity in identities:
            raise RuntimeError("WCBF source contains a duplicate canonical log identity")
        identities.add(identity)
    rows.sort(
        key=lambda row: (
            row["block_number"],
            row["transaction_index"],
            row["log_index"],
            row["event"],
        )
    )
    return rows


def _frozen_header(
    rpc: ethereum.Rpc,
    *,
    block_number: int,
    block_hash: str,
    block_timestamp: str,
) -> ethereum.BlockHeader:
    header = ethereum.get_header(rpc, block_number)
    expected_timestamp = int(ethereum._parse_datetime(block_timestamp).timestamp())
    if header.block_hash != block_hash or header.timestamp != expected_timestamp:
        raise RuntimeError("WCBF frozen boundary header changed")
    return header


def _boundary_code_audit(rpc: ethereum.Rpc) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for label, block in (
        ("start", protocol.START_BOUNDARY_BLOCK),
        ("end", protocol.LAST_SOURCE_BLOCK),
    ):
        code = rpc.call("eth_getCode", [protocol.WBTC_ADDRESS, hex(block)])
        if not isinstance(code, str) or code in {"0x", "0x0"}:
            raise RuntimeError("WCBF contract has no code at a frozen boundary")
        normalized = ethereum._hex_data(
            code, "WBTC contract bytecode", bytes_length=(len(code) - 2) // 2
        )
        digest = hashlib.sha256(bytes.fromhex(normalized[2:])).hexdigest()
        if digest != protocol.BOUNDARY_CODE_SHA256:
            raise RuntimeError("WCBF contract bytecode differs from the frozen hash")
        rows[label] = {
            "block_number": block,
            "bytecode_bytes": (len(normalized) - 2) // 2,
            "sha256": digest,
        }
    if rows["start"]["sha256"] != rows["end"]["sha256"]:
        raise RuntimeError("WCBF boundary contract bytecode hashes disagree")
    return rows


def collect_log_source(
    rpc: ethereum.Rpc,
    cfg: Config,
    *,
    checkpoint: LogCheckpoint | None = None,
    checkpoint_role: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chain_id = ethereum._quantity(rpc.call("eth_chainId", []), "chain id")
    if chain_id != protocol.CHAIN_ID:
        raise RuntimeError(
            f"expected Ethereum mainnet chain id {protocol.CHAIN_ID}, got {chain_id}"
        )
    start = _frozen_header(
        rpc,
        block_number=protocol.START_BOUNDARY_BLOCK,
        block_hash=protocol.START_BOUNDARY_HASH,
        block_timestamp=protocol.START_BOUNDARY_TIMESTAMP,
    )
    end = _frozen_header(
        rpc,
        block_number=protocol.END_BOUNDARY_BLOCK,
        block_hash=protocol.END_BOUNDARY_HASH,
        block_timestamp=protocol.END_BOUNDARY_TIMESTAMP,
    )
    code = _boundary_code_audit(rpc)
    raw = fetch_semantic_logs(
        rpc,
        protocol.START_BOUNDARY_BLOCK,
        protocol.LAST_SOURCE_BLOCK,
        max_block_range=cfg.max_block_range,
        checkpoint=checkpoint,
        checkpoint_role=checkpoint_role,
    )
    rows = normalize_semantic_logs(
        raw,
        start_block=protocol.START_BOUNDARY_BLOCK,
        end_block=protocol.LAST_SOURCE_BLOCK,
    )
    if not rows:
        raise RuntimeError("WCBF source returned no semantic events")
    audit = {
        "chain_id": chain_id,
        "start_boundary": {
            "block_number": start.number,
            "block_hash": start.block_hash,
            "block_timestamp": ethereum._format_timestamp(start.timestamp),
        },
        "end_boundary_exclusive": {
            "block_number": end.number,
            "block_hash": end.block_hash,
            "block_timestamp": ethereum._format_timestamp(end.timestamp),
        },
        "last_source_block": protocol.LAST_SOURCE_BLOCK,
        "contract_code": code,
        "semantic_event_rows": len(rows),
        "canonical_log_hash": ethereum._canonical_hash(rows),
    }
    return rows, audit


def _receipt_requests(
    rpc: ethereum.Rpc, transaction_hashes: Sequence[str], *, batch_size: int
) -> dict[str, dict[str, Any]]:
    if batch_size <= 0 or batch_size > protocol.RECEIPT_BATCH_SIZE:
        raise ValueError("WCBF receipt batch size is outside the frozen guard")
    unique = sorted(set(transaction_hashes))
    receipts: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(unique), batch_size):
        hashes = unique[offset : offset + batch_size]
        raw_receipts = rpc.batch(
            [("eth_getTransactionReceipt", [tx_hash]) for tx_hash in hashes]
        )
        for expected_hash, raw in zip(hashes, raw_receipts, strict=True):
            if not isinstance(raw, dict):
                raise RuntimeError("WCBF transaction receipt is missing or malformed")
            observed_hash = ethereum._hash(
                raw.get("transactionHash"), "receipt transaction hash"
            )
            if observed_hash != expected_hash:
                raise RuntimeError("WCBF receipt transaction hash changed")
            receipts[expected_hash] = raw
    return receipts


def _transfer_companion_matches(
    raw: Any, row: dict[str, Any], *, expected_log_index: int
) -> bool:
    if not isinstance(raw, dict):
        return False
    try:
        if ethereum._address(raw.get("address")) != protocol.WBTC_ADDRESS:
            return False
        topics = raw.get("topics")
        if not isinstance(topics, list) or len(topics) != 3:
            return False
        normalized = [ethereum._hash(topic, "receipt event topic") for topic in topics]
        if normalized[0] != protocol.TRANSFER_TOPIC:
            return False
        if row["event"] == "mint":
            expected_topics = [
                protocol.TRANSFER_TOPIC,
                protocol.ZERO_TOPIC,
                row["actor_topic"],
            ]
        else:
            expected_topics = [
                protocol.TRANSFER_TOPIC,
                row["actor_topic"],
                protocol.ZERO_TOPIC,
            ]
        if normalized != expected_topics:
            return False
        if ethereum._hex_data(raw.get("data"), "transfer data", bytes_length=32) != row[
            "event_data"
        ]:
            return False
        if raw.get("removed") is not False:
            return False
        return (
            ethereum._quantity(raw.get("blockNumber"), "transfer block number")
            == row["block_number"]
            and ethereum._hash(raw.get("blockHash"), "transfer block hash")
            == row["block_hash"]
            and ethereum._hash(
                raw.get("transactionHash"), "transfer transaction hash"
            )
            == row["transaction_hash"]
            and ethereum._quantity(
                raw.get("transactionIndex"), "transfer transaction index"
            )
            == row["transaction_index"]
            and ethereum._quantity(raw.get("logIndex"), "transfer log index")
            == expected_log_index
        )
    except (TypeError, ValueError):
        return False


def validate_receipt_pairs(
    rpc: ethereum.Rpc,
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipts = _receipt_requests(
        rpc,
        [row["transaction_hash"] for row in rows],
        batch_size=batch_size,
    )
    paired: list[dict[str, Any]] = []
    pair_identities: list[dict[str, Any]] = []
    for row in rows:
        receipt = receipts[row["transaction_hash"]]
        if ethereum._quantity(receipt.get("status"), "receipt status") != 1:
            raise RuntimeError("WCBF semantic event belongs to a reverted transaction")
        if (
            ethereum._quantity(receipt.get("blockNumber"), "receipt block number")
            != row["block_number"]
            or ethereum._hash(receipt.get("blockHash"), "receipt block hash")
            != row["block_hash"]
            or ethereum._quantity(
                receipt.get("transactionIndex"), "receipt transaction index"
            )
            != row["transaction_index"]
        ):
            raise RuntimeError("WCBF receipt block identity differs from semantic log")
        logs = receipt.get("logs")
        if not isinstance(logs, list):
            raise RuntimeError("WCBF receipt logs are missing or malformed")
        semantic_matches: list[dict[str, Any]] = []
        for candidate in logs:
            if not isinstance(candidate, dict):
                continue
            try:
                candidate_index = ethereum._quantity(
                    candidate.get("logIndex"), "receipt semantic log index"
                )
            except (TypeError, ValueError):
                continue
            if candidate_index != row["log_index"]:
                continue
            try:
                normalized = normalize_semantic_log(
                    candidate,
                    start_block=protocol.START_BOUNDARY_BLOCK,
                    end_block=protocol.LAST_SOURCE_BLOCK,
                )
            except (TypeError, ValueError, RuntimeError):
                continue
            if normalized == row:
                semantic_matches.append(candidate)
        if len(semantic_matches) != 1:
            raise RuntimeError("WCBF receipt does not reproduce exactly one semantic log")
        companion_index = row["log_index"] + 1
        companions = [
            candidate
            for candidate in logs
            if _transfer_companion_matches(
                candidate, row, expected_log_index=companion_index
            )
        ]
        if len(companions) != 1:
            raise RuntimeError(
                "WCBF receipt does not contain exactly one adjacent zero-transfer pair"
            )
        materialized = {
            **row,
            "companion_transfer_log_index": companion_index,
        }
        paired.append(materialized)
        pair_identities.append(
            {
                "transaction_hash": row["transaction_hash"],
                "semantic_log_index": row["log_index"],
                "companion_transfer_log_index": companion_index,
                "event": row["event"],
                "actor_address": row["actor_address"],
                "amount_raw": row["amount_raw"],
            }
        )
    return paired, {
        "unique_receipts": len(receipts),
        "semantic_events_verified": len(paired),
        "zero_transfer_pairs_verified": len(paired),
        "canonical_pair_hash": ethereum._canonical_hash(pair_identities),
    }


def materialize_rows(
    header_rpc: ethereum.Rpc,
    rows: Sequence[dict[str, Any]],
    *,
    header_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if header_batch_size <= 0 or header_batch_size > protocol.HEADER_BATCH_SIZE:
        raise ValueError("WCBF header batch size is outside the frozen guard")
    materialized = ethereum.materialize_normalized_rows(
        header_rpc,
        rows,
        confirmation_blocks=protocol.CONFIRMATION_BLOCKS,
        header_batch_size=header_batch_size,
    )
    required_finalized_block = max(
        row["confirmation_block_number"] for row in materialized
    )
    finalized = ethereum.get_header(header_rpc, "finalized")
    if finalized.number < required_finalized_block:
        raise RuntimeError("WCBF finalized head does not cover confirmation blocks")
    return materialized, {
        "required_through_block": required_finalized_block,
        "finalized_tag_checked": True,
        "observed_finalized_block_at_least_required": True,
        "event_block_hash_cross_checked": True,
    }


def _source_support(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    event_counts = Counter(row["event"] for row in rows)
    year_event_counts = Counter(
        (row["block_timestamp"][:4], row["event"]) for row in rows
    )
    amount_sums = Counter()
    for row in rows:
        amount_sums[row["event"]] += int(row["amount_raw"])
    missing = [
        f"{year}:{event}"
        for year in ("2020", "2021", "2022", "2023")
        for event in ("mint", "burn")
        if year_event_counts[(year, event)] <= 0
    ]
    if missing:
        raise RuntimeError(
            "WCBF source lacks a frozen year/event cell: " + ", ".join(missing)
        )
    return {
        "decision": "PASS_SOURCE",
        "event_counts": dict(sorted(event_counts.items())),
        "year_event_counts": {
            year: {
                event: year_event_counts[(year, event)]
                for event in ("mint", "burn")
            }
            for year in ("2020", "2021", "2022", "2023")
        },
        "amount_sums_raw": dict(sorted(amount_sums.items())),
        "mint_and_burn_present_in_each_calendar_year": True,
    }


def _transport_hosts(cfg: Config) -> dict[str, str]:
    urls = {
        "primary": cfg.primary_rpc_url,
        "verification": cfg.verification_rpc_url,
        "header": cfg.header_rpc_url or cfg.primary_rpc_url,
        "receipt": cfg.receipt_rpc_url or cfg.verification_rpc_url,
    }
    hosts = {
        role: (urllib.parse.urlsplit(url).hostname or "").lower()
        for role, url in urls.items()
    }
    if any(not host for host in hosts.values()):
        raise ValueError("all WCBF RPC transports must have hostnames")
    if hosts["primary"] == hosts["verification"]:
        raise ValueError("WCBF verification RPC must use an independent hostname")
    return hosts


def _client(url: str, cfg: Config) -> ethereum.JsonRpcClient:
    return ethereum.JsonRpcClient(
        url,
        timeout_sec=cfg.timeout_sec,
        max_retries=cfg.max_retries,
        request_interval_sec=cfg.request_interval_sec,
    )


def build_outputs(
    cfg: Config,
    *,
    primary_rpc: ethereum.Rpc | None = None,
    verification_rpc: ethereum.Rpc | None = None,
    header_rpc: ethereum.Rpc | None = None,
    receipt_rpc: ethereum.Rpc | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _check_disk_limit()
    protocol_payload = _load_protocol()
    hosts = _transport_hosts(cfg)
    primary_rpc = primary_rpc or _client(cfg.primary_rpc_url, cfg)
    verification_rpc = verification_rpc or _client(cfg.verification_rpc_url, cfg)
    header_rpc = header_rpc or _client(cfg.header_rpc_url or cfg.primary_rpc_url, cfg)
    receipt_rpc = receipt_rpc or _client(
        cfg.receipt_rpc_url or cfg.verification_rpc_url, cfg
    )

    checkpoint = LogCheckpoint(cfg.checkpoint_db) if cfg.checkpoint_db else None
    try:
        _progress("primary semantic-log replay started")
        primary_rows, primary_audit = collect_log_source(
            primary_rpc,
            cfg,
            checkpoint=checkpoint,
            checkpoint_role=f"primary:{hosts['primary']}",
        )
        _progress(f"primary replay complete: {len(primary_rows)} semantic events")
        _progress("independent verification replay started")
        verification_rows, verification_audit = collect_log_source(
            verification_rpc,
            cfg,
            checkpoint=checkpoint,
            checkpoint_role=f"verification:{hosts['verification']}",
        )
        _progress(
            f"verification replay complete: {len(verification_rows)} semantic events"
        )
    finally:
        if checkpoint is not None:
            checkpoint.close()
    if primary_audit != verification_audit or primary_rows != verification_rows:
        raise RuntimeError("independent WCBF Ethereum log replays disagree")
    _progress("transaction-receipt pair audit started")
    receipt_rows, receipt_audit = validate_receipt_pairs(
        receipt_rpc,
        primary_rows,
        batch_size=cfg.receipt_batch_size,
    )
    _progress(
        "receipt audit complete: "
        f"{receipt_audit['zero_transfer_pairs_verified']} zero-transfer pairs"
    )
    _progress("canonical event and confirmation header materialization started")
    materialized, header_audit = materialize_rows(
        header_rpc,
        receipt_rows,
        header_batch_size=cfg.header_batch_size,
    )
    _progress(f"header materialization complete: {len(materialized)} source rows")
    support = _source_support(materialized)
    output_rows = [
        {
            **{key: row[key] for key in OUTPUT_COLUMNS if key != "semantic_log_index"},
            "semantic_log_index": row["log_index"],
        }
        for row in materialized
    ]
    output_rows.sort(
        key=lambda row: (
            row["block_number"],
            row["transaction_index"],
            row["semantic_log_index"],
        )
    )
    core: dict[str, Any] = {
        "protocol_version": "wbtc_custody_bridge_flow_source_v1",
        "protocol_binding": {
            "path": str(PROTOCOL_PATH),
            "sha256": PROTOCOL_SHA256,
            "manifest_hash": PROTOCOL_MANIFEST_HASH,
            "decision_binding": protocol_payload["decision_binding"],
        },
        "official_sources": list(protocol.OFFICIAL_SOURCES),
        "source_contract": protocol_payload["source_contract"],
        "source_audit": {
            "log_source": primary_audit,
            "receipt_pairing": receipt_audit,
            "header_materialization": header_audit,
        },
        "dual_replay": {
            "independent_transport_count": 2,
            "canonical_replay_equal": True,
            "canonical_log_hashes": [
                primary_audit["canonical_log_hash"],
                verification_audit["canonical_log_hash"],
            ],
            "provider_urls_embedded": False,
        },
        "transport_roles": {
            "primary_and_verification_hosts_differ": (
                hosts["primary"] != hosts["verification"]
            ),
            "receipt_role": "explicit or verification transport",
            "header_role": "explicit or primary transport",
            "provider_hosts_embedded": False,
        },
        "source_support": support,
        "output": {
            "path": cfg.output_csv,
            "columns": list(OUTPUT_COLUMNS),
            "rows": len(output_rows),
        },
        "implementation_dependencies": {
            "ethereum_helper_path": str(ETHEREUM_HELPER_PATH),
            "ethereum_helper_sha256": ETHEREUM_HELPER_SHA256,
        },
        "outcome_boundary": {
            "source_only": True,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "labels_opened": False,
            "pnl_cagr_mdd_opened": False,
            "mechanism_features_opened": False,
            "post_2023_contract_event_rows_read": 0,
            "post_2023_confirmation_headers_may_be_read": True,
        },
    }
    return output_rows, core


def _write_csv_gz(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(OUTPUT_COLUMNS)
                writer.writerows([[row[column] for column in OUTPUT_COLUMNS] for row in rows])


def run(cfg: Config) -> dict[str, Any]:
    rows, core = build_outputs(cfg)
    output_path = _repo_path(cfg.output_csv)
    _write_csv_gz(output_path, rows)
    core["output"]["sha256"] = ethereum._sha256(output_path)
    core["output"]["bytes"] = output_path.stat().st_size
    core["implementation_binding"] = {
        "path": str(SCRIPT_PATH),
        "sha256": _sha256(SCRIPT_PATH),
    }
    manifest = {**core, "manifest_hash": ethereum._canonical_hash(core)}
    manifest_path = _repo_path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
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
    parser.add_argument(
        "--receipt-rpc-url",
        default=os.environ.get("ETHEREUM_RECEIPT_RPC_URL", ""),
    )
    parser.add_argument("--checkpoint-db", default=Config.checkpoint_db)
    parser.add_argument("--max-block-range", type=int, default=Config.max_block_range)
    parser.add_argument(
        "--header-batch-size", type=int, default=Config.header_batch_size
    )
    parser.add_argument(
        "--receipt-batch-size", type=int, default=Config.receipt_batch_size
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
