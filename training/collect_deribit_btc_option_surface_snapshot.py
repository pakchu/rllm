"""Collect an immutable forward-only Deribit BTC option-surface snapshot.

This collector deliberately does not backfill.  A row becomes causally
available only after both the active-instrument response and the current book
summary response have been received.

Official endpoints:
https://docs.deribit.com/api-reference/market-data/public-get_instruments
https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_ROOT = "https://www.deribit.com/api/v2/public"
INSTRUMENTS_METHOD = "get_instruments"
SUMMARIES_METHOD = "get_book_summary_by_currency"
DEFAULT_OUTPUT_DIR = Path("data/forward_deribit_btc_option_surface")
Clock = Callable[[], datetime]
Fetch = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("snapshot clocks must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def get_json(method: str, params: Mapping[str, Any], *, timeout_sec: float) -> Mapping[str, Any]:
    url = f"{API_ROOT}/{method}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "rllm-forward-source/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Deribit {method} response is not an object")
    return payload


def result_rows(payload: Mapping[str, Any], method: str) -> list[dict[str, Any]]:
    if payload.get("error") is not None:
        raise RuntimeError(f"Deribit {method} error: {payload['error']}")
    rows = payload.get("result")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Deribit {method} returned no rows")
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Deribit {method} contains a non-object row")
    return [dict(row) for row in rows]


def validate_and_join(
    instruments: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    required_instrument = {
        "instrument_name",
        "kind",
        "is_active",
        "option_type",
        "strike",
        "expiration_timestamp",
        "creation_timestamp",
    }
    required_summary = {
        "instrument_name",
        "creation_timestamp",
        "mark_iv",
        "mark_price",
        "open_interest",
        "underlying_price",
    }
    by_name: dict[str, dict[str, Any]] = {}
    for row in instruments:
        missing = required_instrument - row.keys()
        if missing:
            raise RuntimeError(f"instrument schema missing {sorted(missing)}")
        name = str(row["instrument_name"])
        if name in by_name:
            raise RuntimeError(f"duplicate instrument: {name}")
        if row["kind"] != "option" or row["is_active"] is not True:
            raise RuntimeError(f"non-active option in active option response: {name}")
        if row["option_type"] not in ("call", "put"):
            raise RuntimeError(f"invalid option type: {name}")
        by_name[name] = row

    summaries_by_name: dict[str, dict[str, Any]] = {}
    for row in summaries:
        missing = required_summary - row.keys()
        if missing:
            raise RuntimeError(f"summary schema missing {sorted(missing)}")
        name = str(row["instrument_name"])
        if name in summaries_by_name:
            raise RuntimeError(f"duplicate summary: {name}")
        summaries_by_name[name] = row

    common = sorted(by_name.keys() & summaries_by_name.keys())
    if not common:
        raise RuntimeError("instrument and summary responses have no common options")
    joined = [
        {
            "instrument": by_name[name],
            "summary": summaries_by_name[name],
        }
        for name in common
    ]
    coverage = {
        "active_instruments": len(by_name),
        "book_summaries": len(summaries_by_name),
        "joined_options": len(joined),
        "active_without_summary": len(by_name.keys() - summaries_by_name.keys()),
        "summary_without_active_instrument": len(summaries_by_name.keys() - by_name.keys()),
    }
    if coverage["joined_options"] / coverage["active_instruments"] < 0.95:
        raise RuntimeError(f"Deribit option snapshot coverage below 95%: {coverage}")
    return joined, coverage


def collect(*, fetch: Fetch, clock: Clock = utc_now) -> dict[str, Any]:
    request_started_at = clock()
    params = {"currency": "BTC", "kind": "option"}
    instruments_payload = fetch(INSTRUMENTS_METHOD, {**params, "expired": "false"})
    instruments_received_at = clock()
    summaries_payload = fetch(SUMMARIES_METHOD, params)
    summaries_received_at = clock()
    instruments = result_rows(instruments_payload, INSTRUMENTS_METHOD)
    summaries = result_rows(summaries_payload, SUMMARIES_METHOD)
    rows, coverage = validate_and_join(instruments, summaries)
    core = {
        "protocol_version": "deribit_btc_option_surface_forward_snapshot_v1",
        "source": "Deribit public API",
        "currency": "BTC",
        "kind": "option",
        "forward_only": True,
        "historical_backfill_authorized": False,
        "request_started_at": iso(request_started_at),
        "instruments_received_at": iso(instruments_received_at),
        "summaries_received_at": iso(summaries_received_at),
        "feature_available_time": iso(max(instruments_received_at, summaries_received_at)),
        "endpoints": {
            "instruments": f"{API_ROOT}/{INSTRUMENTS_METHOD}",
            "summaries": f"{API_ROOT}/{SUMMARIES_METHOD}",
        },
        "response_hashes": {
            "instruments": canonical_hash(instruments_payload),
            "summaries": canonical_hash(summaries_payload),
        },
        "coverage": coverage,
        "rows": rows,
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def snapshot_path(output_dir: Path, feature_available_time: str) -> Path:
    stamp = feature_available_time.replace("-", "").replace(":", "").replace(".", "_")
    day = feature_available_time[:10]
    return output_dir / day / f"{stamp}.json.gz"


def write_snapshot(snapshot: Mapping[str, Any], output_dir: Path) -> Path:
    core = {key: value for key, value in snapshot.items() if key != "manifest_hash"}
    if snapshot.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("snapshot manifest drift")
    destination = snapshot_path(output_dir, str(snapshot["feature_available_time"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    with destination.open("xb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
            compressed.write(raw)
    return destination


def load_snapshot(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if not isinstance(snapshot, dict):
        raise TypeError(f"snapshot is not an object: {path}")
    core = {key: value for key, value in snapshot.items() if key != "manifest_hash"}
    if snapshot.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"snapshot manifest drift: {path}")
    return snapshot


def archive_paths(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("????-??-??/*.json.gz"))


def seal_for_archive(snapshot: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    core = {key: value for key, value in snapshot.items() if key != "manifest_hash"}
    if snapshot.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("unsealed snapshot manifest drift")
    existing = archive_paths(output_dir)
    predecessor = None
    if existing:
        previous_path = existing[-1]
        previous = load_snapshot(previous_path)
        if str(snapshot["feature_available_time"]) <= str(previous["feature_available_time"]):
            raise RuntimeError("new snapshot is not later than archive predecessor")
        predecessor = {
            "path": str(previous_path),
            "sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
            "manifest_hash": previous["manifest_hash"],
            "feature_available_time": previous["feature_available_time"],
        }
    sealed_core = {**core, "archive_predecessor": predecessor}
    return {**sealed_core, "manifest_hash": canonical_hash(sealed_core)}


def verify_archive(output_dir: Path) -> dict[str, Any]:
    paths = archive_paths(output_dir)
    if not paths:
        raise RuntimeError(f"empty Deribit option surface archive: {output_dir}")
    previous_path: Path | None = None
    previous: dict[str, Any] | None = None
    total_rows = 0
    for path in paths:
        snapshot = load_snapshot(path)
        if snapshot.get("protocol_version") != "deribit_btc_option_surface_forward_snapshot_v1":
            raise RuntimeError(f"snapshot protocol drift: {path}")
        if snapshot.get("forward_only") is not True:
            raise RuntimeError(f"snapshot forward-only boundary drift: {path}")
        coverage = snapshot.get("coverage", {})
        if coverage.get("joined_options") != len(snapshot.get("rows", [])):
            raise RuntimeError(f"snapshot row-count drift: {path}")
        predecessor = snapshot.get("archive_predecessor")
        if previous is not None:
            expected = {
                "path": str(previous_path),
                "sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                "manifest_hash": previous["manifest_hash"],
                "feature_available_time": previous["feature_available_time"],
            }
            if predecessor != expected:
                raise RuntimeError(f"snapshot predecessor drift: {path}")
            if snapshot["feature_available_time"] <= previous["feature_available_time"]:
                raise RuntimeError(f"snapshot chronology drift: {path}")
        elif predecessor is not None:
            raise RuntimeError(f"first snapshot unexpectedly has predecessor: {path}")
        total_rows += len(snapshot["rows"])
        previous_path, previous = path, snapshot
    return {
        "verified": True,
        "snapshots": len(paths),
        "total_option_rows": total_rows,
        "first_feature_available_time": load_snapshot(paths[0])["feature_available_time"],
        "last_feature_available_time": previous["feature_available_time"],
        "tip_path": str(previous_path),
        "tip_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        "tip_manifest_hash": previous["manifest_hash"],
    }


def run(output_dir: Path, *, timeout_sec: float = 30.0) -> dict[str, Any]:
    snapshot = collect(fetch=lambda method, params: get_json(method, params, timeout_sec=timeout_sec))
    snapshot = seal_for_archive(snapshot, output_dir)
    path = write_snapshot(snapshot, output_dir)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_hash": snapshot["manifest_hash"],
        "feature_available_time": snapshot["feature_available_time"],
        "coverage": snapshot["coverage"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_archive(args.output_dir) if args.verify_only else run(args.output_dir, timeout_sec=args.timeout_sec)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
