"""Freeze Mempool's four-year block size/weight composition history.

This is a source-only WCTR-288 transport. It archives the exact HTTP response
bytes and emits a deterministic paired 12-hour bucket table. It deliberately
does not derive witness-transport features, signal incidence, market joins,
returns, or PnL.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, TypedDict
import urllib.error
import urllib.request


ENDPOINT = "https://mempool.space/api/v1/mining/blocks/sizes-weights/4y"
OFFICIAL_REST_DOCS = "https://mempool.space/docs/api/rest"
UPSTREAM_REPOSITORY = "https://github.com/mempool/mempool"
UPSTREAM_COMMIT = "e9d6cf8c042f946be53e372bb36530cd7b7851a4"
SOURCE_DECISION = (
    "docs/witness-composition-transport-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "101e84303efb3146dae587048c179b6688c93b50bb4c99edec8ba8daab72bc98"
)
SOURCE_BUILDER = "training/download_mempool_witness_composition_history.py"
PROTOCOL_VERSION = "mempool_witness_composition_history_source_v1"
USER_AGENT = "rllm-private-research/1.0"
BUCKET_SECONDS = 43_200
SOURCE_AVAILABILITY_LAG_SECONDS = 48 * 60 * 60
EXECUTION_LATENCY_BAR_SECONDS = 5 * 60
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

TOP_LEVEL_KEYS = frozenset({"sizes", "weights"})
SIZE_KEYS = frozenset({"avgHeight", "timestamp", "avgSize"})
WEIGHT_KEYS = frozenset({"avgHeight", "timestamp", "avgWeight"})
MAX_BLOCK_WEIGHT = 4_000_000
ROUNDING_TOLERANCE_BYTES = 4
OUTPUT_COLUMNS = (
    "bucket_start_utc",
    "bucket_end_utc",
    "available_at_utc",
    "avg_height",
    "avg_timestamp",
    "avg_size",
    "avg_weight",
)
RECORDED_HEADERS = ("date", "etag", "last-modified", "content-type")


class NormalizedRow(TypedDict):
    bucket_start_utc: str
    bucket_end_utc: str
    available_at_utc: str
    avg_height: int
    avg_timestamp: int
    avg_size: int
    avg_weight: int


@dataclass(frozen=True)
class Config:
    raw_output: str = (
        "data/mempool_witness_composition_4y_2026-07-20.raw.json.gz"
    )
    output_csv: str = (
        "data/mempool_witness_composition_4y_2026-07-20.csv.gz"
    )
    manifest_output: str = (
        "results/mempool_witness_composition_source_manifest_2026-07-20.json"
    )
    timeout_sec: float = 30.0
    request_pause_sec: float = 0.25
    maximum_retries: int = 8
    maximum_response_bytes: int = 3_000_000
    minimum_response_rows: int = 2_800


@dataclass(frozen=True)
class FetchResult:
    raw: bytes
    status: int
    final_url: str
    headers: Mapping[str, str]
    retrieved_at_utc: str


Fetch = Callable[[], FetchResult]
Sleep = Callable[[float], None]


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(logical_path: str) -> Path:
    return (REPOSITORY_ROOT / logical_path).resolve()


def _artifact_path(configured_path: str) -> Path:
    path = Path(configured_path)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _utc_text(unix_seconds: int) -> str:
    return (
        datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _validate_utc_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must use UTC")
    return value


def _validate_config(cfg: Config) -> None:
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
    for name, value in (
        ("maximum_response_bytes", cfg.maximum_response_bytes),
        ("minimum_response_rows", cfg.minimum_response_rows),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg.minimum_response_rows < 3:
        raise ValueError("minimum_response_rows must leave two edge buckets")

    paths = {
        "raw_output": Path(cfg.raw_output),
        "output_csv": Path(cfg.output_csv),
        "manifest_output": Path(cfg.manifest_output),
    }
    expected_suffixes = {
        "raw_output": ".raw.json.gz",
        "output_csv": ".csv.gz",
        "manifest_output": ".json",
    }
    for name, path in paths.items():
        if not str(path).endswith(expected_suffixes[name]):
            raise ValueError(f"{name} must end with {expected_suffixes[name]}")
        if path.exists() and path.is_dir():
            raise ValueError(f"{name} must not be a directory")
    resolved = {
        name: _artifact_path(str(path)) for name, path in paths.items()
    }
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("raw, normalized, and manifest paths must be distinct")
    protected = {
        _repository_path(SOURCE_DECISION),
        _repository_path(SOURCE_BUILDER),
    }
    if set(resolved.values()) & protected:
        raise ValueError("artifact paths must not overwrite source inputs")
    existing = [name for name, path in resolved.items() if path.exists()]
    if existing:
        raise FileExistsError(
            "frozen source artifacts are immutable; paths already exist: "
            f"{existing!r}"
        )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Mempool response used non-standard JSON value {value}")


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Mempool response contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_payload(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Mempool response is not valid UTF-8") from exc
    return json.loads(
        text,
        parse_float=Decimal,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_object,
    )


def _header_map(headers: Mapping[str, str]) -> dict[str, str | None]:
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    return {name: lowered.get(name) for name in RECORDED_HEADERS}


def _retry_delay(
    *, attempt: int, pause: float, retry_after: str | None
) -> float:
    try:
        requested = float(retry_after) if retry_after is not None else 2.0**attempt
    except ValueError:
        requested = 2.0**attempt
    if not math.isfinite(requested) or requested < 0:
        requested = 2.0**attempt
    return max(pause, min(60.0, requested))


def _http_fetch(cfg: Config, *, sleep: Sleep = time.sleep) -> FetchResult:
    for attempt in range(cfg.maximum_retries + 1):
        request = urllib.request.Request(
            ENDPOINT,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_sec) as response:
                raw = response.read(cfg.maximum_response_bytes + 1)
                if len(raw) > cfg.maximum_response_bytes:
                    raise RuntimeError(
                        "Mempool response exceeds maximum_response_bytes"
                    )
                status = int(getattr(response, "status", response.getcode()))
                final_url = str(response.geturl())
                headers = {
                    str(key): str(value)
                    for key, value in response.headers.items()
                }
            retrieved = (
                datetime.now(tz=timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
            return FetchResult(
                raw=raw,
                status=status,
                final_url=final_url,
                headers=headers,
                retrieved_at_utc=retrieved,
            )
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt >= cfg.maximum_retries:
                raise
            sleep(
                _retry_delay(
                    attempt=attempt,
                    pause=cfg.request_pause_sec,
                    retry_after=exc.headers.get("Retry-After"),
                )
            )
        except (TimeoutError, urllib.error.URLError):
            if attempt >= cfg.maximum_retries:
                raise
            sleep(
                _retry_delay(
                    attempt=attempt,
                    pause=cfg.request_pause_sec,
                    retry_after=None,
                )
            )
    raise AssertionError("unreachable retry loop")


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative JSON integer")
    return value


def _positive_integer(value: Any, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _normalise_rows(
    payload: Any, *, minimum_rows: int
) -> tuple[list[NormalizedRow], dict[str, Any]]:
    if not isinstance(payload, dict) or frozenset(payload) != TOP_LEVEL_KEYS:
        raise ValueError(
            "Mempool size/weight response must be an object with exact "
            "sizes and weights keys"
        )
    sizes = payload["sizes"]
    weights = payload["weights"]
    if not isinstance(sizes, list) or not isinstance(weights, list):
        raise ValueError("Mempool sizes and weights must both be JSON lists")
    if len(sizes) != len(weights):
        raise ValueError("Mempool size and weight arrays must have equal length")
    if len(sizes) < minimum_rows:
        raise ValueError(
            "Mempool size/weight response is shorter than minimum_response_rows"
        )

    parsed: list[tuple[int, int, int, int, int]] = []
    tolerance_rows = 0
    for index, (size_row, weight_row) in enumerate(zip(sizes, weights)):
        if not isinstance(size_row, dict) or not isinstance(weight_row, dict):
            raise ValueError(
                f"Mempool size/weight pair {index} must contain two objects"
            )
        if frozenset(size_row) != SIZE_KEYS:
            missing = sorted(SIZE_KEYS - frozenset(size_row))
            extra = sorted(frozenset(size_row) - SIZE_KEYS)
            raise ValueError(
                f"Mempool size row {index} schema drift: "
                f"missing={missing!r}, extra={extra!r}"
            )
        if frozenset(weight_row) != WEIGHT_KEYS:
            missing = sorted(WEIGHT_KEYS - frozenset(weight_row))
            extra = sorted(frozenset(weight_row) - WEIGHT_KEYS)
            raise ValueError(
                f"Mempool weight row {index} schema drift: "
                f"missing={missing!r}, extra={extra!r}"
            )
        size_height = _positive_integer(size_row["avgHeight"], "size.avgHeight")
        weight_height = _positive_integer(
            weight_row["avgHeight"], "weight.avgHeight"
        )
        size_timestamp = _positive_integer(
            size_row["timestamp"], "size.timestamp"
        )
        weight_timestamp = _positive_integer(
            weight_row["timestamp"], "weight.timestamp"
        )
        if size_height != weight_height or size_timestamp != weight_timestamp:
            raise ValueError(
                f"Mempool size/weight pair {index} has mismatched clock or height"
            )
        size = _positive_integer(size_row["avgSize"], "avgSize")
        weight = _positive_integer(weight_row["avgWeight"], "avgWeight")
        if size > MAX_BLOCK_WEIGHT:
            raise ValueError("avgSize exceeds the BIP 141 serialized-size bound")
        if weight > MAX_BLOCK_WEIGHT:
            raise ValueError("avgWeight exceeds the BIP 141 block-weight limit")
        witness_numerator = 4 * size - weight
        if (
            witness_numerator < -ROUNDING_TOLERANCE_BYTES
            or witness_numerator > 3 * size + ROUNDING_TOLERANCE_BYTES
        ):
            raise ValueError(
                "avgSize and avgWeight imply an invalid witness-byte share"
            )
        if not 0 <= witness_numerator <= 3 * size:
            tolerance_rows += 1
        bucket_start = (size_timestamp // BUCKET_SECONDS) * BUCKET_SECONDS
        parsed.append(
            (bucket_start, size_height, size_timestamp, size, weight)
        )

    heights = [row[1] for row in parsed]
    timestamps = [row[2] for row in parsed]
    buckets = [row[0] for row in parsed]
    if heights != sorted(heights) or len(set(heights)) != len(heights):
        raise ValueError("avgHeight values must be strictly increasing and unique")
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise ValueError("timestamp values must be strictly increasing and unique")
    if len(set(buckets)) != len(buckets):
        raise ValueError("derived 12-hour bucket IDs must be unique")

    deltas = [current - previous for previous, current in zip(buckets, buckets[1:])]
    if any(delta != BUCKET_SECONDS for delta in deltas):
        raise ValueError("derived bucket cadence must be a complete 12-hour grid")

    retained = parsed[1:-1]
    rows: list[NormalizedRow] = []
    for bucket_start, height, timestamp, size, weight in retained:
        bucket_end = bucket_start + BUCKET_SECONDS
        available = bucket_end + SOURCE_AVAILABILITY_LAG_SECONDS
        rows.append(
            {
                "bucket_start_utc": _utc_text(bucket_start),
                "bucket_end_utc": _utc_text(bucket_end),
                "available_at_utc": _utc_text(available),
                "avg_height": height,
                "avg_timestamp": timestamp,
                "avg_size": size,
                "avg_weight": weight,
            }
        )

    audit = {
        "response_rows": len(parsed),
        "retained_rows": len(rows),
        "edge_rows_dropped": 2,
        "first_dropped_bucket_start_utc": _utc_text(parsed[0][0]),
        "last_dropped_bucket_start_utc": _utc_text(parsed[-1][0]),
        "first_retained_bucket_start_utc": rows[0]["bucket_start_utc"],
        "last_retained_bucket_start_utc": rows[-1]["bucket_start_utc"],
        "first_retained_avg_height": rows[0]["avg_height"],
        "last_retained_avg_height": rows[-1]["avg_height"],
        "strictly_increasing_unique_timestamps": True,
        "strictly_increasing_unique_heights": True,
        "unique_bucket_ids": True,
        "paired_size_weight_rows": len(parsed),
        "size_weight_pairing_valid": True,
        "block_weight_limit_valid": True,
        "witness_share_bounds_valid": True,
        "rounding_tolerance_rows": tolerance_rows,
        "rounding_tolerance_bytes": ROUNDING_TOLERANCE_BYTES,
        "missing_12h_buckets": 0,
        "maximum_bucket_gap_seconds": BUCKET_SECONDS,
        "missing_bucket_policy": "reject any gap; never forward-fill or backdate",
    }
    return rows, audit


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def _write_raw_gzip(path: Path, raw: bytes) -> None:
    with path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
            zipped.write(raw)


def _write_csv_gzip(path: Path, rows: list[NormalizedRow]) -> None:
    with path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=OUTPUT_COLUMNS, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)


def _publish_new(temporary: Path, final: Path) -> None:
    """Publish one immutable artifact without replacing an existing inode."""
    try:
        os.link(temporary, final)
    except BaseException:
        try:
            temporary_stat = temporary.stat()
            final_stat = final.stat()
            created_by_this_call = (
                temporary_stat.st_dev == final_stat.st_dev
                and temporary_stat.st_ino == final_stat.st_ino
            )
            if created_by_this_call:
                final.unlink()
        except OSError:
            pass
        raise


def _validate_fetch(result: FetchResult, cfg: Config) -> dict[str, str | None]:
    if not isinstance(result.raw, bytes) or not result.raw:
        raise ValueError("Mempool response bytes must be non-empty")
    if len(result.raw) > cfg.maximum_response_bytes:
        raise ValueError("Mempool response exceeds maximum_response_bytes")
    if result.status != 200:
        raise ValueError("Mempool response status must be 200")
    if result.final_url != ENDPOINT:
        raise ValueError("Mempool response final URL differs from frozen endpoint")
    _validate_utc_text(result.retrieved_at_utc, "retrieved_at_utc")
    headers = _header_map(result.headers)
    content_type = headers["content-type"]
    media_type = (
        content_type.split(";", 1)[0].strip().lower()
        if content_type is not None
        else None
    )
    if media_type != "application/json":
        raise ValueError("Mempool response content-type must be application/json")
    return headers


def run(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Sleep = time.sleep,
) -> dict[str, Any]:
    _validate_config(cfg)
    decision_hash = sha256_file(_repository_path(SOURCE_DECISION))
    if decision_hash != SOURCE_DECISION_SHA256:
        raise RuntimeError("WCTR source decision SHA mismatch")

    result = fetch() if fetch is not None else _http_fetch(cfg, sleep=sleep)
    response_headers = _validate_fetch(result, cfg)
    payload = _decode_payload(result.raw)
    rows, bucket_audit = _normalise_rows(
        payload, minimum_rows=cfg.minimum_response_rows
    )

    raw_path = _artifact_path(cfg.raw_output)
    csv_path = _artifact_path(cfg.output_csv)
    manifest_path = _artifact_path(cfg.manifest_output)
    raw_tmp = _temporary_path(raw_path)
    csv_tmp = _temporary_path(csv_path)
    manifest_tmp = _temporary_path(manifest_path)
    try:
        _write_raw_gzip(raw_tmp, result.raw)
        _write_csv_gzip(csv_tmp, rows)
        core = {
            "protocol_version": PROTOCOL_VERSION,
            "config": asdict(cfg),
            "source_decision": {
                "path": SOURCE_DECISION,
                "sha256": decision_hash,
            },
            "source_builder": {
                "path": SOURCE_BUILDER,
                "sha256": sha256_file(_repository_path(SOURCE_BUILDER)),
            },
            "source_audit": {
                "endpoint": ENDPOINT,
                "official_rest_docs": OFFICIAL_REST_DOCS,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "http_status": result.status,
                "final_url": result.final_url,
                "retrieved_at_utc": result.retrieved_at_utc,
                "response_headers": response_headers,
                "raw_response_bytes": len(result.raw),
                "raw_response_sha256": sha256_bytes(result.raw),
                **bucket_audit,
            },
            "raw_artifact": {
                "path": cfg.raw_output,
                "sha256": sha256_file(raw_tmp),
                "bytes": raw_tmp.stat().st_size,
                "decompressed_sha256": sha256_bytes(result.raw),
                "encoding": "gzip of exact HTTP response bytes; mtime=0",
            },
            "normalized_artifact": {
                "path": cfg.output_csv,
                "sha256": sha256_file(csv_tmp),
                "bytes": csv_tmp.stat().st_size,
                "rows": len(rows),
                "columns": list(OUTPUT_COLUMNS),
            },
            "source_semantics": {
                "avg_size": (
                    "integer-cast average serialized block size among non-stale "
                    "blocks in the 12-hour source bucket"
                ),
                "avg_weight": (
                    "integer-cast average BIP 141 block weight among the same "
                    "non-stale blocks and source bucket"
                ),
                "pairing": (
                    "sizes and weights require exact avgHeight and timestamp "
                    "identity before normalization"
                ),
                "derived_features": "none in the frozen source artifact",
            },
            "causal_availability": {
                "source_bucket_seconds": BUCKET_SECONDS,
                "source_availability_lag_seconds": SOURCE_AVAILABILITY_LAG_SECONDS,
                "available_at_rule": "fixed bucket end + 48 hours",
                "earliest_entry_rule": (
                    "first complete 5-minute execution bar after available_at"
                ),
                "earliest_entry_additional_seconds": EXECUTION_LATENCY_BAR_SECONDS,
                "edge_policy": "drop first and last response buckets",
                "missing_bucket_policy": "reject any gap; never forward-fill or backdate",
            },
            "outcome_boundary": {
                "btc_market_rows_loaded": 0,
                "funding_rows_loaded": 0,
                "premium_or_oi_rows_loaded": 0,
                "return_or_pnl_fields": 0,
                "wctr_features_derived": 0,
                "signal_incidence_rows_derived": 0,
            },
            "data_use": (
                "private research snapshot; public-route operational and legal "
                "review remains required before live promotion"
            ),
        }
        manifest = {**core, "manifest_hash": canonical_hash(core)}
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        published: list[Path] = []
        try:
            _publish_new(raw_tmp, raw_path)
            published.append(raw_path)
            _publish_new(csv_tmp, csv_path)
            published.append(csv_path)
            _publish_new(manifest_tmp, manifest_path)
            published.append(manifest_path)
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            raise
        return manifest
    finally:
        for temporary in (raw_tmp, csv_tmp, manifest_tmp):
            temporary.unlink(missing_ok=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", default=Config.raw_output)
    parser.add_argument("--output-csv", default=Config.output_csv)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--timeout-sec", type=float, default=Config.timeout_sec)
    parser.add_argument(
        "--request-pause-sec", type=float, default=Config.request_pause_sec
    )
    parser.add_argument(
        "--maximum-retries", type=int, default=Config.maximum_retries
    )
    parser.add_argument(
        "--maximum-response-bytes",
        type=int,
        default=Config.maximum_response_bytes,
    )
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
