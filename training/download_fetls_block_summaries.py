"""Download the frozen source-only canonical block prefix needed by FETLS-6."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import urllib.error
import urllib.request

from training.download_bitcoin_block_summaries import (
    BLOCK_KEYS,
    HASH_RE,
    MAX_BLOCK_WEIGHT,
)


BASE_URL = "https://mempool.space/api/v1"
START_HEIGHT = 795_000
END_HEIGHT = 961_681
OUTPUT = Path("data/fetls_block_summaries_2023_2026.csv.gz")
CHECKPOINT = Path("data/.fetls_block_summaries_2023_2026_mempool_v1.sqlite3")
MANIFEST = Path("results/fetls_block_summaries_source_manifest_2026-08-09.json")
PAGE_SIZE = 15
COLUMNS = (
    "height", "id", "previousblockhash", "timestamp", "mediantime",
    "tx_count", "size", "weight",
)


def _fetch(cursor: int, retries: int = 4) -> tuple[int, list[dict]]:
    url = f"{BASE_URL}/blocks/{cursor}"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rllm-private-research/1.0"})
            with urllib.request.urlopen(req, timeout=45) as response:
                payload = json.loads(response.read())
            return cursor, payload
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code == 429 or exc.code >= 500
            if not retryable or attempt == retries:
                raise
            time.sleep(min(8.0, 0.5 * 2**attempt))
    raise AssertionError("unreachable")


def _validate_page(cursor: int, payload: list[dict]) -> list[dict]:
    if not isinstance(payload, list) or not payload or len(payload) > PAGE_SIZE:
        raise RuntimeError("invalid Esplora page cardinality")
    rows = []
    for raw in payload:
        if not isinstance(raw, dict) or not BLOCK_KEYS.issubset(raw):
            raise RuntimeError("Esplora block schema drift")
        row = {key: raw[key] for key in COLUMNS}
        if (
            isinstance(row["height"], bool)
            or not isinstance(row["height"], int)
            or row["height"] < 0
            or HASH_RE.fullmatch(row["id"]) is None
            or HASH_RE.fullmatch(row["previousblockhash"]) is None
        ):
            raise RuntimeError("invalid canonical block identity")
        for key in ("timestamp", "mediantime", "tx_count", "size", "weight"):
            if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] <= 0:
                raise RuntimeError(f"invalid positive block field {key}")
        if row["weight"] > MAX_BLOCK_WEIGHT or not row["size"] <= row["weight"] <= 4 * row["size"]:
            raise RuntimeError("invalid block size/weight invariant")
        rows.append(row)
    expected = list(range(cursor, cursor - len(rows), -1))
    if [row["height"] for row in rows] != expected:
        raise RuntimeError("Esplora page height drift")
    return rows


def _db() -> sqlite3.Connection:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CHECKPOINT)
    db.execute("create table if not exists pages(cursor integer primary key, payload text not null)")
    db.commit()
    return db


def download(workers: int) -> None:
    cursors = list(range(END_HEIGHT, START_HEIGHT - 1, -PAGE_SIZE))
    with _db() as db:
        have = {row[0] for row in db.execute("select cursor from pages")}
        missing = [cursor for cursor in cursors if cursor not in have]
        for offset in range(0, len(missing), 64):
            batch = missing[offset : offset + 64]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_fetch, cursor) for cursor in batch]
                for future in as_completed(futures):
                    cursor, payload = future.result()
                    rows = _validate_page(cursor, payload)
                    db.execute(
                        "insert or replace into pages(cursor,payload) values(?,?)",
                        (cursor, json.dumps(rows, sort_keys=True, separators=(",", ":"))),
                    )
            db.commit()
            print(f"checkpoint {min(offset + len(batch), len(missing))}/{len(missing)}")


def assemble() -> dict:
    rows = []
    with _db() as db:
        for cursor in range(END_HEIGHT, START_HEIGHT - 1, -PAGE_SIZE):
            item = db.execute("select payload from pages where cursor=?", (cursor,)).fetchone()
            if item is None:
                raise RuntimeError(f"missing checkpoint page {cursor}")
            rows.extend(json.loads(item[0]))
    by_height = {int(row["height"]): row for row in rows if START_HEIGHT <= int(row["height"]) <= END_HEIGHT}
    ordered = [by_height[height] for height in range(START_HEIGHT, END_HEIGHT + 1)]
    if len(ordered) != END_HEIGHT - START_HEIGHT + 1:
        raise RuntimeError("assembled source is not height complete")
    for previous, current in zip(ordered, ordered[1:]):
        if current["previousblockhash"] != previous["id"]:
            raise RuntimeError("canonical previous-hash continuity failed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    import csv, io
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader(); writer.writerows(ordered)
    with OUTPUT.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
            zipped.write(text.getvalue().encode())
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    core = {
        "protocol_version": "fetls_block_summaries_source_v1",
        "official_api": "https://github.com/mempool/mempool/blob/master/docs/api/rest.md",
        "transport": "https://mempool.space/api/v1/blocks/{height}",
        "start_height": START_HEIGHT,
        "end_height": END_HEIGHT,
        "rows": len(ordered),
        "path": str(OUTPUT),
        "sha256": digest,
        "market_data_opened": False,
        "postentry_outcomes_opened": False,
    }
    core["manifest_hash"] = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    MANIFEST.write_text(json.dumps(core, indent=2) + "\n")
    return core


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be in [1,32]")
    download(args.workers)
    print(json.dumps(assemble(), indent=2))
