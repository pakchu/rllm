"""Privately stream BitMEX English Trollbox history into 5m attention bars.

Official REST source:
https://docs.bitmex.com/api-explorer/chat-get

Raw text is stored only in ignored private page files so a later outcome-blind
semantic stage need not redownload the source.  The sender-username field is
replaced by a stable pseudonymous hash before persistence; message text can
still contain user-authored identifiers.  The committed surface is a
hash-bound aggregate manifest with neither sender fields nor message text.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


ENDPOINT = "https://www.bitmex.com/api/v1/chat"
OFFICIAL_DOCS = "https://docs.bitmex.com/api-explorer/chat-get"
WEBSOCKET_DOCS = "https://www.bitmex.com/app/wsAPI"
API_CHANGELOG = "https://www.bitmex.com/static/md/en-US/apiChangelog"


@dataclass(frozen=True)
class Config:
    page_dir: str = "data/bitmex_trollbox_english_2020_2022_pages"
    aggregate_output: str = (
        "data/bitmex_trollbox_attention_5m_2020_2022.csv.gz"
    )
    state_output: str = (
        "data/bitmex_trollbox_english_2020_2022_download_state.json"
    )
    manifest_output: str = (
        "results/bitmex_trollbox_attention_source_manifest_2026-07-20.json"
    )
    start_cursor: int = 0
    end_exclusive: str = "2023-01-01"
    channel_id: int = 1
    page_size: int = 500
    request_pause_sec: float = 0.55
    timeout_sec: float = 30.0
    maximum_retries: int = 8
    participant_salt_label: str = "TBASR-24-private-participant-v1"


Fetch = Callable[[dict[str, Any]], list[dict[str, Any]]]


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _format_utc(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _contract(cfg: Config) -> dict[str, Any]:
    return {
        "endpoint": ENDPOINT,
        "start_cursor": cfg.start_cursor,
        "end_exclusive": cfg.end_exclusive,
        "channel_id": cfg.channel_id,
        "page_size": cfg.page_size,
        "participant_salt_label": cfg.participant_salt_label,
        "availability_clock": "cumulative_max_raw_date_in_increasing_id_order",
    }


def _state_template(cfg: Config) -> dict[str, Any]:
    return {
        "protocol_version": "bitmex_trollbox_private_download_state_v2",
        "contract": _contract(cfg),
        "contract_hash": canonical_hash(_contract(cfg)),
        "last_fetched_id": None,
        "timestamp_watermark": None,
        "maximum_raw_timestamp_regression_seconds": 0.0,
        "pages": 0,
        "messages": 0,
        "first_id": None,
        "last_id": None,
        "first_date": None,
        "last_date": None,
        "first_raw_date": None,
        "last_raw_date": None,
        "pagination": "repeat_existing_last_id_then_drop_overlap",
        "complete": False,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _load_state(cfg: Config) -> dict[str, Any]:
    path = Path(cfg.state_output)
    if not path.exists():
        return _state_template(cfg)
    state = json.loads(path.read_text())
    if state.get("contract_hash") != canonical_hash(_contract(cfg)):
        raise RuntimeError("Trollbox resume-state contract mismatch")
    if state.get("contract") != _contract(cfg):
        raise RuntimeError("Trollbox resume-state payload mismatch")
    return state


def _http_page(cfg: Config, params: dict[str, Any]) -> list[dict[str, Any]]:
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    for attempt in range(cfg.maximum_retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "rllm-private-research/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                raise RuntimeError("BitMEX chat response is not a list")
            return payload
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt >= cfg.maximum_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2.0**attempt
            except ValueError:
                delay = 2.0**attempt
            delay = min(60.0, delay)
            time.sleep(max(delay, cfg.request_pause_sec))
        except (TimeoutError, urllib.error.URLError):
            if attempt >= cfg.maximum_retries:
                raise
            time.sleep(max(cfg.request_pause_sec, min(60.0, 2.0**attempt)))
    raise AssertionError("unreachable retry loop")


def _participant_hash(user: str, salt_label: str) -> str:
    return hashlib.sha256(f"{salt_label}\0{user}".encode("utf-8")).hexdigest()


def _normalise_row(
    row: dict[str, Any],
    *,
    cfg: Config,
) -> dict[str, Any]:
    identifier = row.get("id")
    if isinstance(identifier, bool) or not isinstance(identifier, int):
        raise ValueError("BitMEX chat id must be an integer")
    channel = row.get("channelID")
    if int(channel) != cfg.channel_id:
        raise ValueError("BitMEX chat returned another channel")
    timestamp = _utc(str(row.get("date")))
    user = row.get("user")
    message = row.get("message")
    if not isinstance(user, str) or not user:
        raise ValueError("BitMEX chat user must be a non-empty string")
    if not isinstance(message, str):
        raise ValueError("BitMEX chat message must be a string")
    return {
        "id": identifier,
        "date": _format_utc(timestamp),
        "user_hash": _participant_hash(user, cfg.participant_salt_label),
        "message": message,
    }


def _write_page(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                for row in rows:
                    text.write(
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    os.replace(temporary, path)


def download_pages(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not 2 <= cfg.page_size <= 500:
        raise ValueError("BitMEX chat page_size must be in [2, 500]")
    end = _utc(cfg.end_exclusive)
    state_path = Path(cfg.state_output)
    state = _load_state(cfg)
    if state["complete"]:
        return state
    fetch = fetch or (lambda params: _http_page(cfg, params))
    page_dir = Path(cfg.page_dir)
    page_dir.mkdir(parents=True, exist_ok=True)

    while not state["complete"]:
        previous_fetched_id = state["last_fetched_id"]
        cursor = (
            cfg.start_cursor
            if previous_fetched_id is None
            else int(previous_fetched_id)
        )
        params = {
            "count": cfg.page_size,
            "start": cursor,
            "reverse": "false",
            "channelID": cfg.channel_id,
        }
        batch = fetch(params)
        if len(batch) > cfg.page_size:
            raise RuntimeError("BitMEX chat page exceeds frozen page size")
        if not batch:
            raise RuntimeError("BitMEX chat history ended before frozen cutoff")
        normalized = [_normalise_row(row, cfg=cfg) for row in batch]
        identifiers = [int(row["id"]) for row in normalized]
        timestamps = [_utc(str(row["date"])) for row in normalized]
        if identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
            raise RuntimeError("BitMEX chat page IDs are not strictly increasing")
        if previous_fetched_id is None:
            if identifiers[0] < cursor:
                raise RuntimeError("BitMEX chat page moved behind its initial cursor")
        else:
            # BitMEX interprets `start` as an ID only when that exact ID exists;
            # otherwise it falls back to a row offset.  Re-request the known
            # existing last ID and remove that one-row overlap.  Using last+1
            # is unsafe because chat IDs are not guaranteed contiguous.
            if identifiers[0] != int(previous_fetched_id):
                raise RuntimeError(
                    "BitMEX chat page did not restart at the committed ID"
                )
            normalized = normalized[1:]
            identifiers = identifiers[1:]
            timestamps = timestamps[1:]
            if not identifiers:
                raise RuntimeError("BitMEX chat page made no forward progress")

        watermark = (
            None
            if state["timestamp_watermark"] is None
            else _utc(state["timestamp_watermark"])
        )
        maximum_regression = float(
            state["maximum_raw_timestamp_regression_seconds"]
        )
        available_timestamps: list[pd.Timestamp] = []
        for row, timestamp in zip(normalized, timestamps):
            if watermark is None:
                watermark = timestamp
            else:
                maximum_regression = max(
                    maximum_regression,
                    max(0.0, (watermark - timestamp).total_seconds()),
                )
                watermark = max(watermark, timestamp)
            row["available_date"] = _format_utc(watermark)
            available_timestamps.append(watermark)

        crossed_cutoff = any(timestamp >= end for timestamp in available_timestamps)
        if not crossed_cutoff and len(batch) < cfg.page_size:
            raise RuntimeError("BitMEX chat returned a short page before cutoff")

        selected = [
            row
            for row, timestamp in zip(normalized, available_timestamps)
            if timestamp < end
        ]
        if (
            selected
            and previous_fetched_id is not None
            and selected[0]["id"] <= int(previous_fetched_id)
        ):
            raise RuntimeError("BitMEX chat page overlaps committed resume state")
        if selected:
            page_path = page_dir / f"page_{cursor:012d}.jsonl.gz"
            _write_page(page_path, selected)
            state["pages"] = int(state["pages"]) + 1
            state["messages"] = int(state["messages"]) + len(selected)
            state["first_id"] = (
                selected[0]["id"] if state["first_id"] is None else state["first_id"]
            )
            state["first_date"] = (
                selected[0]["available_date"]
                if state["first_date"] is None
                else state["first_date"]
            )
            state["first_raw_date"] = (
                selected[0]["date"]
                if state["first_raw_date"] is None
                else state["first_raw_date"]
            )
            state["last_id"] = selected[-1]["id"]
            state["last_date"] = selected[-1]["available_date"]
            state["last_raw_date"] = selected[-1]["date"]
        state["last_fetched_id"] = identifiers[-1]
        state["timestamp_watermark"] = _format_utc(watermark)
        state["maximum_raw_timestamp_regression_seconds"] = maximum_regression
        if crossed_cutoff:
            state["complete"] = True
        _atomic_json(state_path, state)
        if not state["complete"] and cfg.request_pause_sec > 0.0:
            sleep(cfg.request_pause_sec)
    return state


def _iter_private_rows(page_paths: Iterable[Path]) -> Iterable[tuple[bytes, dict[str, Any]]]:
    for page in page_paths:
        with gzip.open(page, "rt", encoding="utf-8") as handle:
            for line in handle:
                raw = line.encode("utf-8")
                yield raw, json.loads(line)


def _write_aggregate(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text,
                    fieldnames=[
                        "date",
                        "message_count",
                        "unique_participant_count",
                        "maximum_participant_share",
                        "character_count",
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)


def finalize(cfg: Config, state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("complete"):
        raise RuntimeError("cannot finalize an incomplete Trollbox download")
    page_paths = sorted(Path(cfg.page_dir).glob("page_*.jsonl.gz"))
    if len(page_paths) != int(state["pages"]):
        raise RuntimeError("Trollbox private page count does not match resume state")

    raw_hasher = hashlib.sha256()
    container_hasher = hashlib.sha256()
    buckets: dict[pd.Timestamp, dict[str, Any]] = defaultdict(
        lambda: {"messages": 0, "participants": Counter(), "characters": 0}
    )
    prior_id: int | None = None
    availability_watermark: pd.Timestamp | None = None
    maximum_raw_timestamp_regression = 0.0
    messages = 0
    for page in page_paths:
        container_hasher.update(page.name.encode("utf-8") + b"\0")
        container_hasher.update(bytes.fromhex(sha256_file(page)))
        for raw, row in _iter_private_rows([page]):
            raw_hasher.update(raw)
            identifier = int(row["id"])
            raw_timestamp = _utc(str(row["date"]))
            available_timestamp = _utc(str(row["available_date"]))
            if prior_id is not None and identifier <= prior_id:
                raise RuntimeError("Trollbox private stream IDs are not increasing")
            expected_available = (
                raw_timestamp
                if availability_watermark is None
                else max(availability_watermark, raw_timestamp)
            )
            if available_timestamp != expected_available:
                raise RuntimeError("Trollbox availability clock is not causal")
            if availability_watermark is not None:
                maximum_raw_timestamp_regression = max(
                    maximum_raw_timestamp_regression,
                    max(
                        0.0,
                        (availability_watermark - raw_timestamp).total_seconds(),
                    ),
                )
            if "user" in row or set(row) != {
                "id",
                "date",
                "available_date",
                "user_hash",
                "message",
            }:
                raise RuntimeError("Trollbox private stream privacy schema mismatch")
            participant = str(row["user_hash"])
            message = str(row["message"])
            bucket = available_timestamp.floor("5min")
            aggregate = buckets[bucket]
            aggregate["messages"] += 1
            aggregate["participants"][participant] += 1
            aggregate["characters"] += len(message)
            prior_id = identifier
            availability_watermark = available_timestamp
            messages += 1
    if messages != int(state["messages"]) or messages == 0:
        raise RuntimeError("Trollbox private stream message count mismatch")

    first_bucket = _utc(str(state["first_date"])).floor("5min")
    end = _utc(cfg.end_exclusive)
    grid = pd.date_range(first_bucket, end - pd.Timedelta(minutes=5), freq="5min")

    def aggregate_rows() -> Iterable[dict[str, Any]]:
        for timestamp in grid:
            aggregate = buckets.get(timestamp)
            if aggregate is None:
                yield {
                    "date": timestamp.tz_convert(None),
                    "message_count": 0,
                    "unique_participant_count": 0,
                    "maximum_participant_share": 0.0,
                    "character_count": 0,
                }
                continue
            count = int(aggregate["messages"])
            participants: Counter[str] = aggregate["participants"]
            yield {
                "date": timestamp.tz_convert(None),
                "message_count": count,
                "unique_participant_count": len(participants),
                "maximum_participant_share": max(participants.values()) / count,
                "character_count": int(aggregate["characters"]),
            }

    output = Path(cfg.aggregate_output)
    _write_aggregate(output, aggregate_rows())
    spool_bytes = sum(path.stat().st_size for path in page_paths)
    core = {
        "protocol_version": "bitmex_trollbox_attention_source_v1",
        "config": asdict(cfg),
        "official_sources": {
            "rest": OFFICIAL_DOCS,
            "websocket": WEBSOCKET_DOCS,
            "pagination_semantics": API_CHANGELOG,
            "endpoint": ENDPOINT,
        },
        "source_audit": {
            "pages": len(page_paths),
            "messages": messages,
            "first_id": int(state["first_id"]),
            "last_id": int(state["last_id"]),
            "first_date": str(state["first_date"]),
            "last_date": str(state["last_date"]),
            "first_raw_date": str(state["first_raw_date"]),
            "last_raw_date": str(state["last_raw_date"]),
            "last_fetched_id": int(state["last_fetched_id"]),
            "end_exclusive": cfg.end_exclusive,
            "channel_id": cfg.channel_id,
            "chronological_ids": True,
            "availability_timestamps_monotonic": True,
            "availability_clock": (
                "cumulative_max_raw_date_in_increasing_id_order"
            ),
            "selected_maximum_raw_timestamp_regression_seconds": (
                maximum_raw_timestamp_regression
            ),
            "fetched_maximum_raw_timestamp_regression_seconds": float(
                state["maximum_raw_timestamp_regression_seconds"]
            ),
            "raw_stream_sha256": raw_hasher.hexdigest(),
            "private_page_container_sha256": container_hasher.hexdigest(),
            "private_page_bytes": spool_bytes,
        },
        "aggregate": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "rows": len(grid),
            "start": str(grid[0]),
            "end": str(grid[-1]),
            "columns": [
                "date",
                "message_count",
                "unique_participant_count",
                "maximum_participant_share",
                "character_count",
            ],
        },
        "privacy": {
            "raw_responses_committed": False,
            "sender_username_field_persisted": False,
            "private_page_fields": [
                "id",
                "date",
                "available_date",
                "user_hash",
                "message",
            ],
            "message_text_committed": False,
            "private_message_text_may_contain_user_authored_identifiers": True,
            "participant_hash_is_anonymization": False,
            "participant_hash_is_pseudonymization": True,
            "participant_salt_sha256": hashlib.sha256(
                cfg.participant_salt_label.encode("utf-8")
            ).hexdigest(),
        },
    }
    manifest = {
        **core,
        "manifest_hash": canonical_hash(core),
    }
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def run(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    state = download_pages(cfg, fetch=fetch, sleep=sleep)
    return finalize(cfg, state)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-dir", default=Config.page_dir)
    parser.add_argument("--aggregate-output", default=Config.aggregate_output)
    parser.add_argument("--state-output", default=Config.state_output)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--start-cursor", type=int, default=Config.start_cursor)
    parser.add_argument("--end-exclusive", default=Config.end_exclusive)
    parser.add_argument("--channel-id", type=int, default=Config.channel_id)
    parser.add_argument("--page-size", type=int, default=Config.page_size)
    parser.add_argument(
        "--request-pause-sec", type=float, default=Config.request_pause_sec
    )
    parser.add_argument("--timeout-sec", type=float, default=Config.timeout_sec)
    parser.add_argument("--maximum-retries", type=int, default=Config.maximum_retries)
    parser.add_argument(
        "--participant-salt-label", default=Config.participant_salt_label
    )
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
