"""Download a frozen, source-only Bitcoin block-summary prefix from Esplora.

The downloader persists only block metadata needed by BATE-288.  It never
requests transactions, addresses, market data, funding, future returns, or
PnL.  Esplora returns descending ten-block pages; each validated page is
committed atomically to a resumable SQLite checkpoint before the next request.

Official API:
https://github.com/Blockstream/esplora/blob/master/API.md
"""
from __future__ import annotations

import argparse
import csv
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


DEFAULT_BASE_URL = "https://blockstream.info/api"
OFFICIAL_API = "https://github.com/Blockstream/esplora/blob/master/API.md"
SOURCE_DECISION = (
    "docs/block-arrival-throughput-elasticity-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "d2643fa3f8f0ead2b705ee48fa83e59b16d73fc2e7df9dabde271f166d709647"
)
PROTOCOL_VERSION = "bitcoin_block_summaries_source_v1"
PAGE_SIZE = 10
MAX_BLOCK_WEIGHT = 4_000_000
CONFIRMATION_BLOCKS = 6
FROZEN_START_HEIGHT = 610_691
FROZEN_END_HEIGHT = 823_785
FIRST_2024_TIMESTAMP = 1_704_067_200
USER_AGENT = "rllm-private-research/1.0"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
BLOCK_KEYS = frozenset(
    {
        "id",
        "height",
        "version",
        "timestamp",
        "tx_count",
        "size",
        "weight",
        "merkle_root",
        "previousblockhash",
        "mediantime",
        "nonce",
        "bits",
        "difficulty",
    }
)
OUTPUT_COLUMNS = (
    "height",
    "id",
    "previousblockhash",
    "timestamp",
    "mediantime",
    "tx_count",
    "size",
    "weight",
)


class BlockRow(TypedDict):
    height: int
    id: str
    previousblockhash: str
    timestamp: int
    mediantime: int
    tx_count: int
    size: int
    weight: int


@dataclass(frozen=True)
class Config:
    output_csv: str = "data/bitcoin_block_summaries_2020_2023.csv.gz"
    manifest_output: str = (
        "results/bitcoin_block_summaries_source_manifest_2026-07-20.json"
    )
    checkpoint_db: str = (
        "data/.bitcoin_block_summaries_2020_2023.checkpoint.sqlite3"
    )
    start_height: int = FROZEN_START_HEIGHT
    end_height: int = FROZEN_END_HEIGHT
    end_timestamp_exclusive: int = FIRST_2024_TIMESTAMP
    timeout_sec: float = 30.0
    request_pause_sec: float = 0.20
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
        raise ValueError("source interval exceeds the frozen pre-2024 boundary")
    if cfg.end_height - cfg.start_height + 1 < CONFIRMATION_BLOCKS + 1:
        raise ValueError("height interval must contain at least seven blocks")
    if not math.isfinite(cfg.timeout_sec) or cfg.timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive and finite")
    if not math.isfinite(cfg.request_pause_sec) or cfg.request_pause_sec < 0:
        raise ValueError("request_pause_sec must be finite and non-negative")
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
    if Path(SOURCE_DECISION).resolve() in artifact_paths:
        raise ValueError("artifact paths must not overwrite the source decision")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Esplora response used non-standard JSON value {value}")


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
            with urllib.request.urlopen(
                request, timeout=cfg.timeout_sec
            ) as response:
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


def _normalise_row(raw: dict[str, Any], *, cutoff: int) -> BlockRow:
    keys = frozenset(raw)
    if keys != BLOCK_KEYS:
        missing = sorted(BLOCK_KEYS - keys)
        unexpected = sorted(keys - BLOCK_KEYS)
        raise ValueError(
            "Esplora block row schema drift: "
            f"missing={missing!r} unexpected={unexpected!r}"
        )
    height = _validate_nonnegative_int(raw["height"], "height")
    block_hash = _validate_hash(raw["id"], "id")
    previous = _validate_hash(raw["previousblockhash"], "previousblockhash")
    _validate_hash(raw["merkle_root"], "merkle_root")
    timestamp = _validate_positive_int(raw["timestamp"], "timestamp")
    mediantime = _validate_positive_int(raw["mediantime"], "mediantime")
    if timestamp >= cutoff:
        raise ValueError("Esplora block timestamp is outside the frozen interval")
    tx_count = _validate_positive_int(raw["tx_count"], "tx_count")
    size = _validate_positive_int(raw["size"], "size")
    weight = _validate_positive_int(raw["weight"], "weight")
    if weight > MAX_BLOCK_WEIGHT:
        raise ValueError("weight exceeds the BIP 141 block-weight limit")
    if not size <= weight <= 4 * size:
        raise ValueError("size and weight violate the serialized block invariant")
    _validate_nonnegative_int(raw["nonce"], "nonce")
    _validate_nonnegative_int(raw["bits"], "bits")
    if isinstance(raw["version"], bool) or not isinstance(raw["version"], int):
        raise ValueError("version must be an integer")
    difficulty = raw["difficulty"]
    if isinstance(difficulty, bool) or not isinstance(
        difficulty, (int, Decimal)
    ):
        raise ValueError("difficulty must be an exact JSON number")
    difficulty_decimal = Decimal(difficulty)
    if not difficulty_decimal.is_finite() or difficulty_decimal <= 0:
        raise ValueError("difficulty must be positive and finite")
    return {
        "height": height,
        "id": block_hash,
        "previousblockhash": previous,
        "timestamp": timestamp,
        "mediantime": mediantime,
        "tx_count": tx_count,
        "size": size,
        "weight": weight,
    }


def _normalise_page(
    payload: Any,
    *,
    cursor: int,
    start_height: int,
    cutoff: int,
) -> list[BlockRow]:
    if not isinstance(payload, list):
        raise RuntimeError("Esplora blocks response is not a list")
    if not payload:
        raise RuntimeError("Esplora returned an empty page before source completion")
    if len(payload) > PAGE_SIZE:
        raise RuntimeError("Esplora blocks response exceeded the ten-block page cap")
    if any(not isinstance(row, dict) for row in payload):
        raise ValueError("Esplora block page item must be an object")
    rows = [_normalise_row(row, cutoff=cutoff) for row in payload]
    heights = [row["height"] for row in rows]
    if heights[0] != cursor:
        raise RuntimeError(
            "Esplora page did not start at the requested height: "
            f"requested={cursor} received={heights[0]}"
        )
    if any(left - right != 1 for left, right in zip(heights, heights[1:])):
        raise RuntimeError("Esplora page heights are not strictly contiguous descending")
    for high, low in zip(rows, rows[1:]):
        if high["previousblockhash"] != low["id"]:
            raise RuntimeError("Esplora page broke Bitcoin hash-chain linkage")
    target_count = min(PAGE_SIZE, cursor - start_height + 1)
    target = [row for row in rows if row["height"] >= start_height]
    expected = list(range(cursor, cursor - target_count, -1))
    if len(target) != target_count or [row["height"] for row in target] != expected:
        raise RuntimeError("Esplora page did not cover the exact requested height suffix")
    return target


def _contract(cfg: Config) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "base_url": cfg.base_url.rstrip("/"),
        "start_height": cfg.start_height,
        "end_height": cfg.end_height,
        "end_timestamp_exclusive": cfg.end_timestamp_exclusive,
        "page_size": PAGE_SIZE,
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
        raise RuntimeError("Esplora checkpoint failed SQLite quick_check")
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
            weight INTEGER NOT NULL
        );
        """
    )
    contract_json = json.dumps(
        _contract(cfg), sort_keys=True, separators=(",", ":")
    )
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
        raise RuntimeError("Esplora resume-state contract mismatch")
    return connection


def _rows_from_checkpoint(connection: sqlite3.Connection) -> list[BlockRow]:
    selected = connection.execute(
        "SELECT height, id, previousblockhash, timestamp, mediantime, "
        "tx_count, size, weight FROM blocks ORDER BY height"
    ).fetchall()
    return [
        {
            "height": int(row["height"]),
            "id": str(row["id"]),
            "previousblockhash": str(row["previousblockhash"]),
            "timestamp": int(row["timestamp"]),
            "mediantime": int(row["mediantime"]),
            "tx_count": int(row["tx_count"]),
            "size": int(row["size"]),
            "weight": int(row["weight"]),
        }
        for row in selected
    ]


def _validate_checkpoint_row(row: BlockRow, cfg: Config) -> None:
    height = _validate_nonnegative_int(row["height"], "checkpoint height")
    if not cfg.start_height <= height <= cfg.end_height:
        raise RuntimeError("Esplora checkpoint height is outside the frozen range")
    _validate_hash(row["id"], "checkpoint id")
    _validate_hash(row["previousblockhash"], "checkpoint previousblockhash")
    timestamp = _validate_positive_int(row["timestamp"], "checkpoint timestamp")
    if timestamp >= cfg.end_timestamp_exclusive:
        raise RuntimeError("Esplora checkpoint crossed the frozen time boundary")
    _validate_positive_int(row["mediantime"], "checkpoint mediantime")
    _validate_positive_int(row["tx_count"], "checkpoint tx_count")
    size = _validate_positive_int(row["size"], "checkpoint size")
    weight = _validate_positive_int(row["weight"], "checkpoint weight")
    if weight > MAX_BLOCK_WEIGHT:
        raise RuntimeError("Esplora checkpoint exceeds the block-weight limit")
    if not size <= weight <= 4 * size:
        raise RuntimeError("Esplora checkpoint violates the size/weight invariant")


def _audit_suffix(rows: list[BlockRow], cfg: Config) -> int:
    if not rows:
        return cfg.end_height
    for row in rows:
        _validate_checkpoint_row(row, cfg)
    heights = [row["height"] for row in rows]
    if heights[-1] != cfg.end_height:
        raise RuntimeError("Esplora checkpoint does not end at end_height")
    expected = list(range(heights[0], cfg.end_height + 1))
    if heights != expected:
        raise RuntimeError("Esplora checkpoint is not a contiguous height suffix")
    if heights[0] < cfg.start_height:
        raise RuntimeError("Esplora checkpoint contains a row below start_height")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Esplora checkpoint contains duplicate block hashes")
    for low, high in zip(rows, rows[1:]):
        if high["previousblockhash"] != low["id"]:
            raise RuntimeError("Esplora checkpoint broke Bitcoin hash-chain linkage")
    return heights[0] - 1


def _insert_page(
    connection: sqlite3.Connection,
    rows_descending: list[BlockRow],
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
            raise RuntimeError("Esplora page broke checkpoint boundary linkage")
    values = [tuple(row[column] for column in OUTPUT_COLUMNS) for row in rows_descending]
    with connection:
        connection.executemany(
            "INSERT INTO blocks(height, id, previousblockhash, timestamp, "
            "mediantime, tx_count, size, weight) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )


def download(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Sleep = time.sleep,
) -> list[BlockRow]:
    _validate_config(cfg)
    fetch = fetch or (lambda url: _http_page(cfg, url))
    connection = _open_checkpoint(cfg)
    try:
        existing = _rows_from_checkpoint(connection)
        cursor = _audit_suffix(existing, cfg)
        seen_cursors: set[int] = set()
        while cursor >= cfg.start_height:
            if cursor in seen_cursors:
                raise RuntimeError("Esplora pagination loop detected")
            seen_cursors.add(cursor)
            url = f"{cfg.base_url.rstrip('/')}/blocks/{cursor}"
            page = _normalise_page(
                fetch(url),
                cursor=cursor,
                start_height=cfg.start_height,
                cutoff=cfg.end_timestamp_exclusive,
            )
            _insert_page(
                connection,
                page,
                cursor=cursor,
                end_height=cfg.end_height,
            )
            cursor = min(row["height"] for row in page) - 1
            if cfg.request_pause_sec and cursor >= cfg.start_height:
                sleep(cfg.request_pause_sec)
        rows = _rows_from_checkpoint(connection)
        _audit_complete(rows, cfg)
        return rows
    finally:
        connection.close()


def _audit_complete(rows: list[BlockRow], cfg: Config) -> dict[str, Any]:
    expected_rows = cfg.end_height - cfg.start_height + 1
    if len(rows) != expected_rows:
        raise RuntimeError(
            "Esplora source is incomplete: "
            f"expected={expected_rows} observed={len(rows)}"
        )
    for row in rows:
        _validate_checkpoint_row(row, cfg)
    heights = [row["height"] for row in rows]
    if heights != list(range(cfg.start_height, cfg.end_height + 1)):
        raise RuntimeError("Esplora source is not the exact inclusive height range")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Esplora source contains duplicate block hashes")
    for low, high in zip(rows, rows[1:]):
        if high["previousblockhash"] != low["id"]:
            raise RuntimeError("Esplora source broke Bitcoin hash-chain linkage")
    if any(row["timestamp"] >= cfg.end_timestamp_exclusive for row in rows):
        raise RuntimeError("Esplora source crossed the frozen time boundary")
    return {
        "endpoint": f"{cfg.base_url.rstrip('/')}/blocks/:start_height",
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
    }


def _write_canonical_gzip(path: str | Path, rows: Iterable[BlockRow]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with io.TextIOWrapper(
                    compressed, encoding="utf-8", newline=""
                ) as text:
                    writer = csv.DictWriter(
                        text,
                        fieldnames=OUTPUT_COLUMNS,
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _write_canonical_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
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
        raise RuntimeError("BATE mechanism decision is missing or has drifted")
    rows = download(cfg, fetch=fetch, sleep=sleep)
    audit = _audit_complete(rows, cfg)
    _write_canonical_gzip(cfg.output_csv, rows)
    output = Path(cfg.output_csv)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "source_decision": {
            "path": SOURCE_DECISION,
            "sha256": SOURCE_DECISION_SHA256,
        },
        "config": asdict(cfg),
        "source_audit": audit,
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "columns": list(OUTPUT_COLUMNS),
        },
        "outcome_boundary": {
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "return_or_pnl_fields": 0,
            "post_2023_source_rows_loaded": sum(
                row["timestamp"] >= FIRST_2024_TIMESTAMP for row in rows
            ),
            "raw_esplora_responses_persisted": False,
        },
        "data_use": (
            "private internal source research under the Esplora MIT software "
            "license; no raw HTTP response is redistributed"
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
    parser.add_argument("--start-height", type=int, default=Config.start_height)
    parser.add_argument("--end-height", type=int, default=Config.end_height)
    parser.add_argument(
        "--end-timestamp-exclusive",
        type=int,
        default=Config.end_timestamp_exclusive,
    )
    parser.add_argument("--timeout-sec", type=float, default=Config.timeout_sec)
    parser.add_argument(
        "--request-pause-sec", type=float, default=Config.request_pause_sec
    )
    parser.add_argument(
        "--maximum-retries", type=int, default=Config.maximum_retries
    )
    parser.add_argument("--base-url", default=Config.base_url)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
