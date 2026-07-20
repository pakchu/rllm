"""Download frozen Bitcoin UTXO/fee block stats from Mempool's block endpoint.

This is a source-only UFCP downloader.  It persists only deterministic confirmed
ledger fields needed for UTXO Fee-Clearing Polarity, uses Mempool's descending
15-block `/api/v1/blocks/{height}` endpoint, and cross-checks every basic block
field against the already frozen 2020-2023 block-summary artifact before writing
final CSV/manifest artifacts.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from decimal import Decimal
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Callable, Iterable, TypedDict
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_BASE_URL = "https://mempool.space/api"
OFFICIAL_API = "https://mempool.space/docs/api/rest"
SOURCE_DECISION = "docs/utxo-fee-clearing-polarity-mechanism-decision-2026-07-20.md"
SOURCE_DECISION_SHA256 = "95bf889fd053987e1717b182dc5da4f19ef51d75a1cbda427913089368c4852e"
REFERENCE_BLOCK_SUMMARIES = "data/bitcoin_block_summaries_2020_2023.csv.gz"
REFERENCE_BLOCK_SUMMARIES_SHA256 = "1f8d1c0153717f2c18d8fa6f09428c780a850b2aecc8fb42bc497e16e68e1833"
PROTOCOL_VERSION = "bitcoin_utxo_fee_block_stats_source_v1"
PAGE_SIZE = 15
MAX_BLOCK_WEIGHT = 4_000_000
CONFIRMATION_BLOCKS = 6
FROZEN_START_HEIGHT = 610_691
FROZEN_END_HEIGHT = 823_785
FIRST_2024_TIMESTAMP = 1_704_067_200
USER_AGENT = "rllm-private-research/1.0"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

BASIC_COLUMNS = (
    "height",
    "id",
    "previousblockhash",
    "timestamp",
    "mediantime",
    "tx_count",
    "size",
    "weight",
)
OUTPUT_COLUMNS = BASIC_COLUMNS + (
    "total_fees",
    "total_inputs",
    "total_outputs",
    "utxo_set_change",
)
REQUIRED_TOP_KEYS = frozenset(BASIC_COLUMNS + ("stale", "extras"))
REQUIRED_EXTRAS_KEYS = frozenset(
    {"totalFees", "totalInputs", "totalOutputs", "utxoSetChange"}
)


class BlockStatsRow(TypedDict):
    height: int
    id: str
    previousblockhash: str
    timestamp: int
    mediantime: int
    tx_count: int
    size: int
    weight: int
    total_fees: int
    total_inputs: int
    total_outputs: int
    utxo_set_change: int


@dataclass(frozen=True)
class Config:
    output_csv: str = "data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz"
    manifest_output: str = (
        "results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json"
    )
    checkpoint_db: str = (
        "data/.bitcoin_utxo_fee_block_stats_2020_2023.checkpoint.sqlite3"
    )
    reference_block_summaries: str = REFERENCE_BLOCK_SUMMARIES
    reference_block_summaries_sha256: str = REFERENCE_BLOCK_SUMMARIES_SHA256
    start_height: int = FROZEN_START_HEIGHT
    end_height: int = FROZEN_END_HEIGHT
    end_timestamp_exclusive: int = FIRST_2024_TIMESTAMP
    timeout_sec: float = 30.0
    request_pause_sec: float = 0.20
    request_workers: int = 4
    maximum_retries: int = 8
    base_url: str = DEFAULT_BASE_URL


Fetch = Callable[[str], Any]
Sleep = Callable[[float], None]


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character hex hash")
    return value


def _validate_config(cfg: Config) -> None:
    for name, value in (
        ("start_height", cfg.start_height),
        ("end_height", cfg.end_height),
        ("end_timestamp_exclusive", cfg.end_timestamp_exclusive),
    ):
        _validate_positive_int(value, name)
    if cfg.start_height > cfg.end_height:
        raise ValueError("start_height must not exceed end_height")
    if (
        cfg.start_height < FROZEN_START_HEIGHT
        or cfg.end_height > FROZEN_END_HEIGHT
        or cfg.end_timestamp_exclusive > FIRST_2024_TIMESTAMP
    ):
        raise ValueError("source interval exceeds the frozen 2020-2023 boundary")
    if cfg.end_height - cfg.start_height + 1 < CONFIRMATION_BLOCKS + 1:
        raise ValueError("height interval must contain at least seven blocks")
    if not math.isfinite(cfg.timeout_sec) or cfg.timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive and finite")
    if not math.isfinite(cfg.request_pause_sec) or cfg.request_pause_sec < 0:
        raise ValueError("request_pause_sec must be finite and non-negative")
    if (
        isinstance(cfg.request_workers, bool)
        or not isinstance(cfg.request_workers, int)
        or not 1 <= cfg.request_workers <= 8
    ):
        raise ValueError("request_workers must be an integer in [1, 8]")
    if (
        isinstance(cfg.maximum_retries, bool)
        or not isinstance(cfg.maximum_retries, int)
        or cfg.maximum_retries < 0
    ):
        raise ValueError("maximum_retries must be a non-negative integer")
    parsed = urllib.parse.urlparse(cfg.base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an absolute HTTP(S) URL without a query")
    artifact_paths = {
        Path(cfg.output_csv).resolve(),
        Path(cfg.manifest_output).resolve(),
        Path(cfg.checkpoint_db).resolve(),
    }
    if len(artifact_paths) != 3:
        raise ValueError("output, manifest, and checkpoint paths must be distinct")
    protected = {
        Path(SOURCE_DECISION).resolve(),
        Path(cfg.reference_block_summaries).resolve(),
    }
    if artifact_paths & protected:
        raise ValueError("artifact paths must not overwrite protected source inputs")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Mempool response used non-standard JSON value {value}")


def _decode_payload(raw: bytes) -> Any:
    return json.loads(
        raw.decode("utf-8"),
        parse_float=Decimal,
        parse_constant=_reject_json_constant,
    )


def _http_page(cfg: Config, url: str) -> Any:
    for attempt in range(cfg.maximum_retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_sec) as response:
                return _decode_payload(response.read())
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt >= cfg.maximum_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2.0**attempt
            except ValueError:
                delay = 2.0**attempt
            time.sleep(max(cfg.request_pause_sec, min(60.0, delay)))
        except (TimeoutError, urllib.error.URLError):
            if attempt >= cfg.maximum_retries:
                raise
            time.sleep(max(cfg.request_pause_sec, min(60.0, 2.0**attempt)))
    raise AssertionError("unreachable retry loop")


def _validate_stale(value: Any) -> None:
    if isinstance(value, bool):
        stale = int(value)
    elif isinstance(value, int):
        stale = value
    else:
        raise ValueError("stale must be an integer zero/false value")
    if stale != 0:
        raise ValueError("Mempool block row is marked stale")


def _normalise_row(raw: dict[str, Any], *, cutoff: int) -> BlockStatsRow:
    missing_top = sorted(REQUIRED_TOP_KEYS - frozenset(raw))
    if missing_top:
        raise ValueError(f"Mempool block row schema drift: missing={missing_top!r}")
    extras = raw["extras"]
    if not isinstance(extras, dict):
        raise ValueError("Mempool block row extras must be an object")
    missing_extras = sorted(REQUIRED_EXTRAS_KEYS - frozenset(extras))
    if missing_extras:
        raise ValueError(
            f"Mempool extras schema drift: missing={missing_extras!r}"
        )

    height = _validate_nonnegative_int(raw["height"], "height")
    block_hash = _validate_hash(raw["id"], "id")
    previous = _validate_hash(raw["previousblockhash"], "previousblockhash")
    timestamp = _validate_positive_int(raw["timestamp"], "timestamp")
    mediantime = _validate_positive_int(raw["mediantime"], "mediantime")
    if timestamp >= cutoff:
        raise ValueError("Mempool block timestamp is outside the frozen interval")
    tx_count = _validate_positive_int(raw["tx_count"], "tx_count")
    size = _validate_positive_int(raw["size"], "size")
    weight = _validate_positive_int(raw["weight"], "weight")
    if weight > MAX_BLOCK_WEIGHT:
        raise ValueError("weight exceeds the BIP 141 block-weight limit")
    if not size <= weight <= 4 * size:
        raise ValueError("size and weight violate the serialized block invariant")

    _validate_stale(raw["stale"])
    total_fees = _validate_nonnegative_int(extras["totalFees"], "extras.totalFees")
    total_inputs = _validate_nonnegative_int(
        extras["totalInputs"], "extras.totalInputs"
    )
    total_outputs = _validate_nonnegative_int(
        extras["totalOutputs"], "extras.totalOutputs"
    )
    utxo_set_change = _validate_int(
        extras["utxoSetChange"], "extras.utxoSetChange"
    )
    if utxo_set_change != total_outputs - total_inputs:
        raise ValueError("utxo_set_change must equal total_outputs - total_inputs")

    return {
        "height": height,
        "id": block_hash,
        "previousblockhash": previous,
        "timestamp": timestamp,
        "mediantime": mediantime,
        "tx_count": tx_count,
        "size": size,
        "weight": weight,
        "total_fees": total_fees,
        "total_inputs": total_inputs,
        "total_outputs": total_outputs,
        "utxo_set_change": utxo_set_change,
    }


def _normalise_page(
    payload: Any,
    *,
    cursor: int,
    start_height: int,
    cutoff: int,
) -> list[BlockStatsRow]:
    if not isinstance(payload, list):
        raise RuntimeError("Mempool blocks response is not a list")
    if not payload:
        raise RuntimeError("Mempool returned an empty page before source completion")
    if len(payload) > PAGE_SIZE:
        raise RuntimeError("Mempool blocks response exceeded the 15-block page cap")
    if any(not isinstance(row, dict) for row in payload):
        raise ValueError("Mempool block page item must be an object")
    rows = [_normalise_row(row, cutoff=cutoff) for row in payload]
    heights = [row["height"] for row in rows]
    if heights[0] != cursor:
        raise RuntimeError(
            "Mempool page did not start at the requested height: "
            f"requested={cursor} received={heights[0]}"
        )
    if any(left - right != 1 for left, right in zip(heights, heights[1:])):
        raise RuntimeError("Mempool page heights are not strictly contiguous descending")
    for high, low in zip(rows, rows[1:]):
        if high["previousblockhash"] != low["id"]:
            raise RuntimeError("Mempool page broke Bitcoin hash-chain linkage")
    target_count = min(PAGE_SIZE, cursor - start_height + 1)
    target = [row for row in rows if row["height"] >= start_height]
    expected = list(range(cursor, cursor - target_count, -1))
    if len(target) != target_count or [row["height"] for row in target] != expected:
        raise RuntimeError("Mempool page did not cover the exact requested height suffix")
    return target


def _contract(cfg: Config) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "base_url": cfg.base_url.rstrip("/"),
        "start_height": cfg.start_height,
        "end_height": cfg.end_height,
        "end_timestamp_exclusive": cfg.end_timestamp_exclusive,
        "page_size": PAGE_SIZE,
        "reference_block_summaries": cfg.reference_block_summaries,
        "reference_block_summaries_sha256": cfg.reference_block_summaries_sha256,
        "source_decision_sha256": SOURCE_DECISION_SHA256,
    }


def _open_checkpoint(cfg: Config) -> sqlite3.Connection:
    path = Path(cfg.checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA synchronous = FULL")
    result = connection.execute("PRAGMA quick_check").fetchone()
    if result is None or result[0] != "ok":
        connection.close()
        raise RuntimeError("Mempool checkpoint failed SQLite quick_check")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS blocks (
            height INTEGER PRIMARY KEY,
            id TEXT NOT NULL UNIQUE,
            previousblockhash TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            mediantime INTEGER NOT NULL,
            tx_count INTEGER NOT NULL,
            size INTEGER NOT NULL,
            weight INTEGER NOT NULL,
            total_fees INTEGER NOT NULL,
            total_inputs INTEGER NOT NULL,
            total_outputs INTEGER NOT NULL,
            utxo_set_change INTEGER NOT NULL
        );
        """
    )
    contract_json = json.dumps(_contract(cfg), sort_keys=True, separators=(",", ":"))
    stored = connection.execute(
        "SELECT value FROM metadata WHERE key = 'contract'"
    ).fetchone()
    if stored is None:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES('contract', ?)",
            (contract_json,),
        )
        connection.commit()
    elif stored[0] != contract_json:
        connection.close()
        raise RuntimeError("Mempool resume-state contract mismatch")
    return connection


def _rows_from_checkpoint(connection: sqlite3.Connection) -> list[BlockStatsRow]:
    selected = connection.execute(
        "SELECT height, id, previousblockhash, timestamp, mediantime, tx_count, "
        "size, weight, total_fees, total_inputs, total_outputs, utxo_set_change "
        "FROM blocks ORDER BY height"
    ).fetchall()
    return [
        {column: int(row[column]) if column not in {"id", "previousblockhash"} else str(row[column]) for column in OUTPUT_COLUMNS}  # type: ignore[misc]
        for row in selected
    ]


def _validate_checkpoint_row(row: BlockStatsRow, cfg: Config) -> None:
    height = _validate_nonnegative_int(row["height"], "checkpoint height")
    if not cfg.start_height <= height <= cfg.end_height:
        raise RuntimeError("Mempool checkpoint height is outside the frozen range")
    _validate_hash(row["id"], "checkpoint id")
    _validate_hash(row["previousblockhash"], "checkpoint previousblockhash")
    timestamp = _validate_positive_int(row["timestamp"], "checkpoint timestamp")
    if timestamp >= cfg.end_timestamp_exclusive:
        raise RuntimeError("Mempool checkpoint crossed the frozen time boundary")
    _validate_positive_int(row["mediantime"], "checkpoint mediantime")
    _validate_positive_int(row["tx_count"], "checkpoint tx_count")
    size = _validate_positive_int(row["size"], "checkpoint size")
    weight = _validate_positive_int(row["weight"], "checkpoint weight")
    if weight > MAX_BLOCK_WEIGHT:
        raise RuntimeError("Mempool checkpoint exceeds the block-weight limit")
    if not size <= weight <= 4 * size:
        raise RuntimeError("Mempool checkpoint violates the size/weight invariant")
    total_fees = _validate_nonnegative_int(row["total_fees"], "checkpoint total_fees")
    total_inputs = _validate_nonnegative_int(
        row["total_inputs"], "checkpoint total_inputs"
    )
    total_outputs = _validate_nonnegative_int(
        row["total_outputs"], "checkpoint total_outputs"
    )
    utxo_set_change = _validate_int(
        row["utxo_set_change"], "checkpoint utxo_set_change"
    )
    if total_fees < 0 or utxo_set_change != total_outputs - total_inputs:
        raise RuntimeError("Mempool checkpoint violates the UTXO fee identity")


def _audit_suffix(rows: list[BlockStatsRow], cfg: Config) -> int:
    if not rows:
        return cfg.end_height
    for row in rows:
        _validate_checkpoint_row(row, cfg)
    heights = [row["height"] for row in rows]
    if heights[-1] != cfg.end_height:
        raise RuntimeError("Mempool checkpoint does not end at end_height")
    expected = list(range(heights[0], cfg.end_height + 1))
    if heights != expected:
        raise RuntimeError("Mempool checkpoint is not a contiguous height suffix")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Mempool checkpoint contains duplicate block hashes")
    for low, high in zip(rows, rows[1:]):
        if high["previousblockhash"] != low["id"]:
            raise RuntimeError("Mempool checkpoint broke Bitcoin hash-chain linkage")
    return heights[0] - 1


def _insert_page(
    connection: sqlite3.Connection,
    rows_descending: list[BlockStatsRow],
    *,
    cursor: int,
    end_height: int,
) -> None:
    if cursor < end_height:
        boundary = connection.execute(
            "SELECT previousblockhash FROM blocks WHERE height = ?",
            (cursor + 1,),
        ).fetchone()
        if boundary is None or boundary[0] != rows_descending[0]["id"]:
            raise RuntimeError("Mempool page broke checkpoint boundary linkage")
    values = [tuple(row[column] for column in OUTPUT_COLUMNS) for row in rows_descending]
    with connection:
        connection.executemany(
            "INSERT INTO blocks(height, id, previousblockhash, timestamp, "
            "mediantime, tx_count, size, weight, total_fees, total_inputs, "
            "total_outputs, utxo_set_change) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )


def download(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Sleep = time.sleep,
) -> list[BlockStatsRow]:
    _validate_config(cfg)
    fetch = fetch or (lambda url: _http_page(cfg, url))
    connection = _open_checkpoint(cfg)
    try:
        existing = _rows_from_checkpoint(connection)
        cursor = _audit_suffix(existing, cfg)
        seen_cursors: set[int] = set()
        while cursor >= cfg.start_height:
            cursors = [
                page_cursor
                for offset in range(cfg.request_workers)
                if (page_cursor := cursor - offset * PAGE_SIZE) >= cfg.start_height
            ]
            if any(page_cursor in seen_cursors for page_cursor in cursors):
                raise RuntimeError("Mempool pagination loop detected")
            seen_cursors.update(cursors)

            def fetch_page(page_cursor: int) -> list[BlockStatsRow]:
                url = f"{cfg.base_url.rstrip('/')}/v1/blocks/{page_cursor}"
                return _normalise_page(
                    fetch(url),
                    cursor=page_cursor,
                    start_height=cfg.start_height,
                    cutoff=cfg.end_timestamp_exclusive,
                )

            if len(cursors) == 1:
                pages = {cursors[0]: fetch_page(cursors[0])}
            else:
                with ThreadPoolExecutor(max_workers=len(cursors)) as executor:
                    loaded = executor.map(fetch_page, cursors)
                    pages = dict(zip(cursors, loaded))
            for page_cursor in cursors:
                _insert_page(
                    connection,
                    pages[page_cursor],
                    cursor=page_cursor,
                    end_height=cfg.end_height,
                )
            cursor = min(row["height"] for row in pages[cursors[-1]]) - 1
            if cfg.request_pause_sec and cursor >= cfg.start_height:
                sleep(cfg.request_pause_sec)
        rows = _rows_from_checkpoint(connection)
        _audit_complete(rows, cfg)
        return rows
    finally:
        connection.close()


def _audit_complete(rows: list[BlockStatsRow], cfg: Config) -> dict[str, Any]:
    expected_rows = cfg.end_height - cfg.start_height + 1
    if len(rows) != expected_rows:
        raise RuntimeError(
            "Mempool source is incomplete: "
            f"expected={expected_rows} observed={len(rows)}"
        )
    for row in rows:
        _validate_checkpoint_row(row, cfg)
    heights = [row["height"] for row in rows]
    if heights != list(range(cfg.start_height, cfg.end_height + 1)):
        raise RuntimeError("Mempool source is not the exact inclusive height range")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Mempool source contains duplicate block hashes")
    for low, high in zip(rows, rows[1:]):
        if high["previousblockhash"] != low["id"]:
            raise RuntimeError("Mempool source broke Bitcoin hash-chain linkage")
    if any(row["timestamp"] >= cfg.end_timestamp_exclusive for row in rows):
        raise RuntimeError("Mempool source crossed the frozen time boundary")
    return {
        "endpoint": f"{cfg.base_url.rstrip('/')}/v1/blocks/:start_height",
        "official_api": OFFICIAL_API,
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "start_height": rows[0]["height"],
        "end_height": rows[-1]["height"],
        "latest_eligible_packet_end": cfg.end_height - CONFIRMATION_BLOCKS,
        "minimum_header_timestamp": min(row["timestamp"] for row in rows),
        "maximum_header_timestamp": max(row["timestamp"] for row in rows),
        "end_timestamp_exclusive": cfg.end_timestamp_exclusive,
        "height_links_checked": len(rows) - 1,
        "complete_inclusive_height_range": True,
        "unique_block_hashes": True,
        "all_rows_pre_cutoff": True,
        "utxo_identity_checked": True,
    }


def _read_reference_basic_rows(cfg: Config) -> dict[int, dict[str, str]]:
    reference = Path(cfg.reference_block_summaries)
    if not reference.is_file():
        raise RuntimeError("frozen block-summary reference is missing")
    observed_sha = sha256_file(reference)
    if observed_sha != cfg.reference_block_summaries_sha256:
        raise RuntimeError(
            "frozen block-summary reference SHA mismatch: "
            f"expected={cfg.reference_block_summaries_sha256} observed={observed_sha}"
        )
    rows: dict[int, dict[str, str]] = {}
    with gzip.open(reference, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(BASIC_COLUMNS).issubset(reader.fieldnames):
            raise RuntimeError("frozen block-summary reference schema mismatch")
        for raw in reader:
            height = int(raw["height"])
            if cfg.start_height <= height <= cfg.end_height:
                rows[height] = {column: raw[column] for column in BASIC_COLUMNS}
    return rows


def _cross_check_reference(rows: list[BlockStatsRow], cfg: Config) -> dict[str, Any]:
    reference_rows = _read_reference_basic_rows(cfg)
    expected_heights = set(range(cfg.start_height, cfg.end_height + 1))
    if set(reference_rows) != expected_heights:
        raise RuntimeError("frozen block-summary reference does not cover exact range")
    for row in rows:
        reference = reference_rows[row["height"]]
        for column in BASIC_COLUMNS:
            observed = str(row[column])
            if observed != reference[column]:
                raise RuntimeError(
                    "Mempool/reference basic field mismatch: "
                    f"height={row['height']} column={column} "
                    f"observed={observed!r} reference={reference[column]!r}"
                )
    return {
        "reference_path": cfg.reference_block_summaries,
        "reference_sha256": cfg.reference_block_summaries_sha256,
        "rows_cross_checked": len(rows),
        "columns_cross_checked": list(BASIC_COLUMNS),
        "all_basic_fields_match_reference": True,
    }


def _write_canonical_gzip(path: str | Path, rows: Iterable[BlockStatsRow]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _write_canonical_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Sleep = time.sleep,
) -> dict[str, Any]:
    decision_path = Path(SOURCE_DECISION)
    if not decision_path.is_file() or sha256_file(decision_path) != SOURCE_DECISION_SHA256:
        raise RuntimeError("UFCP mechanism decision is missing or has drifted")
    rows = download(cfg, fetch=fetch, sleep=sleep)
    source_audit = _audit_complete(rows, cfg)
    reference_audit = _cross_check_reference(rows, cfg)
    _write_canonical_gzip(cfg.output_csv, rows)
    output = Path(cfg.output_csv)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "source_decision": {
            "path": SOURCE_DECISION,
            "sha256": SOURCE_DECISION_SHA256,
        },
        "config": asdict(cfg),
        "source_audit": source_audit,
        "reference_audit": reference_audit,
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "columns": list(OUTPUT_COLUMNS),
        },
        "outcome_boundary": {
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "outcome_rows_loaded": 0,
            "return_or_pnl_fields": 0,
            "post_2023_source_rows_loaded": sum(
                row["timestamp"] >= FIRST_2024_TIMESTAMP for row in rows
            ),
            "raw_mempool_responses_persisted": False,
            "unrelated_mempool_metadata_persisted": False,
        },
        "data_use": (
            "private internal research cache of consensus-derived fields via "
            "public Mempool REST transport; the API output has no separately "
            "documented data licence, production must use an owned Bitcoin Core "
            "node, and no raw response or unrelated metadata is redistributed"
        ),
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    _write_canonical_json(cfg.manifest_output, manifest)
    return manifest


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", default=Config.output_csv)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--checkpoint-db", default=Config.checkpoint_db)
    parser.add_argument("--reference-block-summaries", default=Config.reference_block_summaries)
    parser.add_argument(
        "--reference-block-summaries-sha256",
        default=Config.reference_block_summaries_sha256,
    )
    parser.add_argument("--start-height", type=int, default=Config.start_height)
    parser.add_argument("--end-height", type=int, default=Config.end_height)
    parser.add_argument(
        "--end-timestamp-exclusive", type=int, default=Config.end_timestamp_exclusive
    )
    parser.add_argument("--timeout-sec", type=float, default=Config.timeout_sec)
    parser.add_argument("--request-pause-sec", type=float, default=Config.request_pause_sec)
    parser.add_argument("--request-workers", type=int, default=Config.request_workers)
    parser.add_argument("--maximum-retries", type=int, default=Config.maximum_retries)
    parser.add_argument("--base-url", default=Config.base_url)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
