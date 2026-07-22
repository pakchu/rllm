"""Build and gate the outcome-blind BRCR 2020-2023 RIPE source."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sqlite3
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from training import preregister_bgp_routing_churn_relay as protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = Path("training/build_bgp_routing_churn_relay_source.py")
PROTOCOL_PATH = protocol.DEFAULT_OUTPUT
PROTOCOL_FILE_SHA256 = (
    "5f4204f6e630ce690bab3f961d2fce74849b8c48dd68b5ca56a2f9daac09aee1"
)
PROTOCOL_MANIFEST_HASH = (
    "97dd7fff76fee4e9cefd76a9e8af7e6e2df04a9c2e69e5dc68f6827748ef44e7"
)

DEFAULT_SOURCE = Path("data/bgp_routing_churn_relay_2020_2023.jsonl.gz")
DEFAULT_OBJECT_MANIFEST = Path(
    "results/bgp_routing_churn_relay_object_manifest_2026-07-22.json.gz"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "results/bgp_routing_churn_relay_source_manifest_2026-07-22.json"
)
DEFAULT_SUPPORT = Path(
    "results/bgp_routing_churn_relay_source_support_2026-07-22.json"
)
DEFAULT_CHECKPOINT = Path("checkpoints/brcr_source_checkpoint.sqlite3")

USER_AGENT = "rllm-brcr-source-freeze/1.0"
READ_CHUNK_BYTES = 1024 * 1024
REPLAY_AUDIT_LABELS = frozenset(protocol.replay_audit_labels())


class SourceContractError(RuntimeError):
    """A frozen source invariant failed and BRCR must not be repaired."""


class SourceFetchError(RuntimeError):
    """A transient or non-missing transport failure interrupted collection."""


@dataclass(frozen=True)
class FetchedObject:
    body: bytes
    last_modified: str
    etag: str
    declared_content_length: int


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = _repo_path(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(payload).rstrip(b"\n"))


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("BRCR timestamp must be UTC-aware")
    return value.isoformat().replace("+00:00", "Z")


def load_frozen_protocol() -> dict[str, Any]:
    path = _repo_path(PROTOCOL_PATH)
    if sha256_file(path) != PROTOCOL_FILE_SHA256:
        raise RuntimeError("BRCR source protocol file hash differs from builder binding")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("BRCR source protocol artifact is not an object")
    protocol.validate_manifest(payload)
    if payload.get("manifest_hash") != PROTOCOL_MANIFEST_HASH:
        raise RuntimeError("BRCR source protocol manifest hash differs from builder binding")
    return payload


def _used_gib(path: Path) -> int:
    stats = os.statvfs(path)
    used = (stats.f_blocks - stats.f_bfree) * stats.f_frsize
    return used // (1024**3)


def enforce_disk_guard(path: str | Path) -> int:
    candidate = _repo_path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    used_gib = _used_gib(candidate.parent)
    if used_gib >= protocol.DISK_LIMIT_GIB:
        raise SourceFetchError(
            f"BRCR disk guard rejected {used_gib} GiB used at "
            f"{protocol.DISK_LIMIT_GIB} GiB"
        )
    return used_gib


def validate_official_archive_url(label: datetime, url: str) -> None:
    expected = protocol.archive_url(label)
    parsed = urllib.parse.urlsplit(url)
    if (
        url != expected
        or parsed.scheme != "https"
        or parsed.hostname != "data.ris.ripe.net"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("BRCR archive URL differs from the frozen official route")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        raise SourceFetchError("BRCR archive transport attempted a redirect")


def _parse_content_length(value: str | None) -> int:
    if value is None:
        raise SourceFetchError("BRCR response omitted Content-Length")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SourceFetchError("BRCR response has malformed Content-Length") from exc
    if parsed <= 0 or parsed > protocol.MAXIMUM_COMPRESSED_BYTES:
        raise SourceFetchError("BRCR response Content-Length is outside the bound")
    return parsed


def _read_bounded_response(response: Any) -> FetchedObject:
    advertised = _parse_content_length(response.headers.get("Content-Length"))
    last_modified = response.headers.get("Last-Modified")
    etag = response.headers.get("ETag")
    if not last_modified or not etag:
        raise SourceFetchError("BRCR response omitted required validation headers")

    payload = bytearray()
    while True:
        chunk = response.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > protocol.MAXIMUM_COMPRESSED_BYTES:
            raise SourceFetchError("BRCR compressed object exceeds the memory bound")
    if len(payload) != advertised:
        raise SourceFetchError("BRCR response body differs from Content-Length")
    return FetchedObject(
        body=bytes(payload),
        last_modified=last_modified,
        etag=etag,
        declared_content_length=advertised,
    )


def fetch_archive_object(
    label: datetime,
    *,
    retries: int,
    timeout_seconds: float,
    retry_base_seconds: float,
) -> FetchedObject | None:
    url = protocol.archive_url(label)
    validate_official_archive_url(label, url)
    opener = urllib.request.build_opener(
        _NoRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        method="GET",
    )
    for attempt in range(retries + 1):
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                if response.status != 200 or response.geturl() != url:
                    raise SourceFetchError(
                        "BRCR archive returned an unexpected response"
                    )
                return _read_bounded_response(response)
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 410}:
                return None
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == retries:
                raise SourceFetchError(
                    f"BRCR archive HTTP failure {exc.code} for {url}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise SourceFetchError(f"BRCR archive fetch failed for {url}") from exc
        except SourceFetchError:
            if attempt == retries:
                raise
        time.sleep(retry_base_seconds * (2**attempt))
    raise AssertionError("BRCR retry loop terminated unexpectedly")


def decompress_gzip_bounded(compressed: bytes) -> bytes:
    if not compressed.startswith(protocol.GZIP_MAGIC):
        raise SourceContractError("BRCR object is not a gzip stream")
    output = bytearray()
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as handle:
            while True:
                chunk = handle.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > protocol.MAXIMUM_DECOMPRESSED_BYTES:
                    raise SourceContractError(
                        "BRCR decompressed object exceeds the memory bound"
                    )
    except (EOFError, OSError) as exc:
        raise SourceContractError("BRCR gzip CRC or stream validation failed") from exc
    return bytes(output)


def process_fetched_object(label: datetime, fetched: FetchedObject) -> dict[str, Any]:
    try:
        transport = protocol.validate_transport_headers(
            label=label,
            last_modified=fetched.last_modified,
            etag=fetched.etag,
            declared_content_length=fetched.declared_content_length,
            actual_content_length=len(fetched.body),
        )
        decompressed = decompress_gzip_bounded(fetched.body)
        first = protocol.parse_mrt_stream(decompressed, label=label)
        second = protocol.parse_mrt_stream(decompressed, label=label)
        if first != second:
            raise SourceContractError("BRCR two in-memory parse passes differ")
        parsed = {
            "compressed_bytes": len(fetched.body),
            "compressed_sha256": sha256_bytes(fetched.body),
            **first,
        }
        identity = protocol.canonical_object_identity(label=label, parsed=parsed)
    except SourceContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise SourceContractError("BRCR transport or MRT source contract failed") from exc

    source_row = {
        "archive_label_utc": _utc_iso(label),
        "source_availability_utc": _utc_iso(protocol.public_availability_time(label)),
        "archive_url": protocol.archive_url(label),
        **parsed,
        "object_identity_sha256": identity,
    }
    return {
        "object": {
            **source_row,
            "status": "available",
            "transport": transport,
            "replay_required": label in REPLAY_AUDIT_LABELS,
            "replay_verified": False,
        },
        "rows": [source_row],
    }


def missing_object_entry(label: datetime) -> dict[str, Any]:
    return {
        "object": {
            "archive_label_utc": _utc_iso(label),
            "source_availability_utc": _utc_iso(
                protocol.public_availability_time(label)
            ),
            "archive_url": protocol.archive_url(label),
            "status": "missing",
            "replay_required": label in REPLAY_AUDIT_LABELS,
            "replay_verified": False,
        },
        "rows": [],
    }


def _fetched_identity(fetched: FetchedObject) -> tuple[str, str, str, int]:
    return (
        sha256_bytes(fetched.body),
        fetched.last_modified,
        fetched.etag,
        fetched.declared_content_length,
    )


def verify_replay(
    label: datetime,
    original: FetchedObject,
    replay: FetchedObject | None,
) -> None:
    if replay is None:
        raise SourceContractError("BRCR required replay object is missing")
    if _fetched_identity(original) != _fetched_identity(replay):
        raise SourceContractError("BRCR replay bytes or validation headers differ")
    process_fetched_object(label, replay)


class SourceCheckpoint:
    def __init__(self, path: str | Path, binding: Mapping[str, Any]) -> None:
        self.path = _repo_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS objects ("
            "label TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL)"
        )
        existing = dict(self.connection.execute("SELECT key, value FROM metadata"))
        expected = {
            key: canonical_json(value).decode("utf-8").rstrip("\n")
            for key, value in binding.items()
        }
        if existing and existing != expected:
            self.close()
            raise RuntimeError("BRCR checkpoint binding differs from current builder")
        object_count = self.connection.execute(
            "SELECT COUNT(*) FROM objects"
        ).fetchone()[0]
        if not existing and object_count:
            self.close()
            raise RuntimeError("BRCR checkpoint has objects without a binding")
        if not existing:
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(expected.items()),
            )
            self.connection.commit()

    def labels(self) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT label FROM objects ORDER BY label"
            )
        ]

    def put(self, label: datetime, payload: Mapping[str, Any]) -> None:
        label_text = _utc_iso(label)
        payload_text = canonical_json(payload).decode("utf-8").rstrip("\n")
        payload_sha = sha256_bytes(payload_text.encode("utf-8"))
        existing = self.connection.execute(
            "SELECT payload, payload_sha256 FROM objects WHERE label = ?",
            (label_text,),
        ).fetchone()
        if existing is not None:
            if existing != (payload_text, payload_sha):
                raise RuntimeError("BRCR checkpoint label has conflicting payload")
            return
        self.connection.execute(
            "INSERT INTO objects(label, payload, payload_sha256) VALUES (?, ?, ?)",
            (label_text, payload_text, payload_sha),
        )
        self.connection.commit()

    def entries(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT payload, payload_sha256 FROM objects ORDER BY label"
        )
        entries: list[dict[str, Any]] = []
        for payload, expected_sha in rows:
            if sha256_bytes(payload.encode("utf-8")) != expected_sha:
                raise RuntimeError("BRCR checkpoint payload hash mismatch")
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise RuntimeError("BRCR checkpoint payload is not an object")
            entries.append(parsed)
        return entries

    def close(self) -> None:
        self.connection.close()

    def delete(self) -> None:
        self.close()
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{self.path}{suffix}")
            if candidate.exists():
                candidate.unlink()


def checkpoint_binding() -> dict[str, Any]:
    return {
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_sha256": sha256_file(BUILDER_PATH),
        "source_start": _utc_iso(protocol.SOURCE_START),
        "source_end_exclusive": _utc_iso(protocol.SOURCE_END_EXCLUSIVE),
        "expected_objects": len(protocol.archive_labels()),
    }


def collect_to_checkpoint(
    checkpoint: SourceCheckpoint,
    fetcher: Callable[[datetime], FetchedObject | None],
    *,
    max_new_objects: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    labels = protocol.archive_labels()
    processed = checkpoint.labels()
    expected_prefix = [_utc_iso(label) for label in labels[: len(processed)]]
    if processed != expected_prefix:
        raise RuntimeError("BRCR checkpoint labels are not a contiguous frozen prefix")
    added = 0
    for index, label in enumerate(labels[len(processed) :], start=len(processed)):
        fetched = fetcher(label)
        try:
            if fetched is None:
                entry = missing_object_entry(label)
            else:
                entry = process_fetched_object(label, fetched)
                if label in REPLAY_AUDIT_LABELS:
                    verify_replay(label, fetched, fetcher(label))
                    entry["object"]["replay_verified"] = True
        except SourceContractError as exc:
            if fetched is None:
                raise SourceContractError(f"{_utc_iso(label)}: {exc}") from exc
            confirmation = fetcher(label)
            if confirmation is None:
                raise SourceFetchError(
                    f"BRCR failed object disappeared during confirmation: {_utc_iso(label)}"
                ) from exc
            if _fetched_identity(confirmation) != _fetched_identity(fetched):
                raise SourceFetchError(
                    "BRCR source bytes or headers changed while confirming a failure at "
                    f"{_utc_iso(label)}"
                ) from exc
            raise SourceContractError(
                f"{_utc_iso(label)} [{sha256_bytes(fetched.body)}]: {exc}"
            ) from exc
        checkpoint.put(label, entry)
        added += 1
        if progress is not None:
            progress(index + 1, len(labels))
        if max_new_objects > 0 and added >= max_new_objects:
            break
    return len(checkpoint.labels()), len(labels)


def _gate(
    gate_id: str,
    observed: float | int,
    threshold: float | int,
    operator_name: str,
) -> dict[str, Any]:
    if operator_name == ">=":
        passed = observed >= threshold
    elif operator_name == "<=":
        passed = observed <= threshold
    elif operator_name == "==":
        passed = observed == threshold
    else:
        raise ValueError("BRCR gate operator is unsupported")
    return {
        "gate_id": gate_id,
        "observed": observed,
        "threshold": threshold,
        "operator": operator_name,
        "passed": passed,
    }


def evaluate_support(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = protocol.archive_labels()
    if len(entries) != len(labels):
        raise RuntimeError("BRCR support requires every expected archive label")
    availability: dict[datetime, bool] = {}
    source_rows = 0
    compressed_hashes: Counter[str] = Counter()
    replay_required = 0
    replay_verified = 0
    for label, entry in zip(labels, entries):
        object_row = entry["object"]
        if object_row["archive_label_utc"] != _utc_iso(label):
            raise RuntimeError("BRCR support entries are out of frozen order")
        if object_row["status"] == "missing":
            availability[label] = False
            if entry["rows"]:
                raise RuntimeError("BRCR missing object unexpectedly contains rows")
        elif object_row["status"] == "available":
            availability[label] = True
            if len(entry["rows"]) != 1:
                raise RuntimeError("BRCR available object must contain exactly one row")
            if entry["rows"][0]["object_identity_sha256"] != object_row[
                "object_identity_sha256"
            ]:
                raise RuntimeError("BRCR object summary differs from source row")
            source_rows += 1
            compressed_hashes[object_row["compressed_sha256"]] += 1
        else:
            raise RuntimeError("BRCR object has an unknown status")
        if object_row["replay_required"]:
            replay_required += 1
            replay_verified += int(object_row["replay_verified"])

    coverage = protocol.summarize_expected_coverage(availability)
    quality = protocol.build_manifest()["source_quality_gates"]
    gates: list[dict[str, Any]] = []
    for year, values in coverage["years"].items():
        gates.append(
            _gate(
                f"object_fraction_year_{year}",
                values["available_objects"] / values["expected_objects"],
                quality["transport"]["minimum_object_fraction_each_year"],
                ">=",
            )
        )
    for month, values in coverage["months"].items():
        gates.append(
            _gate(
                f"object_fraction_month_{month}",
                values["available_objects"] / values["expected_objects"],
                quality["transport"]["minimum_object_fraction_each_month"],
                ">=",
            )
        )
    gates.extend(
        (
            _gate(
                "maximum_consecutive_missing_objects",
                coverage["maximum_consecutive_missing_objects"],
                quality["transport"]["maximum_consecutive_missing_objects"],
                "<=",
            ),
            _gate(
                "duplicate_sha_across_labels",
                sum(count - 1 for count in compressed_hashes.values() if count > 1),
                quality["transport"]["duplicate_sha_across_labels_allowed"],
                "==",
            ),
            _gate("compressed_sha256_fraction", 1.0, 1.0, "=="),
            _gate("exact_cross_pass_equality_fraction", 1.0, 1.0, "=="),
            _gate("valid_mrt_fraction", 1.0, 1.0, "=="),
            _gate("imputed_or_forward_filled_objects", 0, 0, "=="),
            _gate(
                "replay_required_objects",
                replay_required,
                quality["replay"]["expected_objects"],
                "==",
            ),
            _gate(
                "replay_verified_objects",
                replay_verified,
                quality["replay"]["expected_objects"],
                "==",
            ),
        )
    )
    all_passed = all(gate["passed"] for gate in gates)
    return {
        "source_id": "BRCR",
        "status": "PASS_ADVANCE_TO_CHURN_FREEZE"
        if all_passed
        else "REJECT_NO_REPAIR",
        "all_gates_passed": all_passed,
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "mechanism_parser_opened": False,
        "expected_object_count": len(labels),
        "source_row_count": source_rows,
        "coverage": coverage,
        "replay_required_objects": replay_required,
        "replay_verified_objects": replay_verified,
        "gates": gates,
    }


def _gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return gzip.compress(
        b"".join(canonical_json(row) for row in rows),
        compresslevel=9,
        mtime=0,
    )


def _pretty_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_once(path: str | Path, raw: bytes) -> str:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != raw:
            raise RuntimeError(f"refusing to overwrite frozen BRCR artifact {output}")
        return "verified_existing"
    with output.open("xb") as handle:
        handle.write(raw)
    return "created"


def write_artifact_set(
    artifacts: Sequence[tuple[str, str | Path, bytes]],
) -> dict[str, str]:
    for _, path, raw in artifacts:
        output = _repo_path(path)
        if output.exists() and output.read_bytes() != raw:
            raise RuntimeError(f"refusing to overwrite frozen BRCR artifact {output}")
    return {name: write_once(path, raw) for name, path, raw in artifacts}


def finalize_artifacts(
    entries: Sequence[Mapping[str, Any]],
    *,
    source_path: str | Path = DEFAULT_SOURCE,
    object_manifest_path: str | Path = DEFAULT_OBJECT_MANIFEST,
    source_manifest_path: str | Path = DEFAULT_SOURCE_MANIFEST,
    support_path: str | Path = DEFAULT_SUPPORT,
) -> dict[str, Any]:
    labels = protocol.archive_labels()
    if len(entries) != len(labels):
        raise RuntimeError("BRCR cannot finalize an incomplete checkpoint")
    rows = [row for entry in entries for row in entry["rows"]]
    object_rows = [entry["object"] for entry in entries]
    source_raw = _gzip_jsonl(rows)
    object_raw = _gzip_jsonl(object_rows)
    support_core = {
        **evaluate_support(entries),
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_sha256": sha256_file(BUILDER_PATH),
    }
    support = {**support_core, "manifest_hash": canonical_hash(support_core)}
    support_raw = _pretty_json(support)

    source_manifest_core = {
        "source_id": "BRCR",
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_path": str(BUILDER_PATH),
        "builder_sha256": sha256_file(BUILDER_PATH),
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "mechanism_parser_opened": False,
        "source": {
            "path": str(source_path),
            "rows": len(rows),
            "bytes": len(source_raw),
            "sha256": sha256_bytes(source_raw),
        },
        "object_manifest": {
            "path": str(object_manifest_path),
            "objects": len(object_rows),
            "bytes": len(object_raw),
            "sha256": sha256_bytes(object_raw),
        },
        "support": {
            "path": str(support_path),
            "bytes": len(support_raw),
            "sha256": sha256_bytes(support_raw),
            "status": support["status"],
        },
    }
    source_manifest = {
        **source_manifest_core,
        "manifest_hash": canonical_hash(source_manifest_core),
    }
    source_manifest_raw = _pretty_json(source_manifest)
    statuses = write_artifact_set(
        (
            ("source", source_path, source_raw),
            ("object_manifest", object_manifest_path, object_raw),
            ("support", support_path, support_raw),
            ("source_manifest", source_manifest_path, source_manifest_raw),
        )
    )
    return {
        "status": support["status"],
        "all_gates_passed": support["all_gates_passed"],
        "source_rows": len(rows),
        "objects": len(object_rows),
        "artifact_statuses": statuses,
        "source_manifest_hash": source_manifest["manifest_hash"],
    }


def write_fatal_rejection(path: str | Path, reason: str) -> str:
    core = {
        "source_id": "BRCR",
        "status": "REJECT_NO_REPAIR",
        "all_gates_passed": False,
        "fatal_source_contract_failure": reason,
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_sha256": sha256_file(BUILDER_PATH),
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "mechanism_parser_opened": False,
    }
    return write_once(
        path, _pretty_json({**core, "manifest_hash": canonical_hash(core)})
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--object-manifest", type=Path, default=DEFAULT_OBJECT_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--max-new-objects", type=int, default=0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--request-pace-seconds", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_frozen_protocol()
    enforce_disk_guard(args.checkpoint)
    checkpoint = SourceCheckpoint(args.checkpoint, checkpoint_binding())

    def fetcher(label: datetime) -> FetchedObject | None:
        enforce_disk_guard(args.checkpoint)
        payload = fetch_archive_object(
            label,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            retry_base_seconds=args.retry_base_seconds,
        )
        if args.request_pace_seconds > 0:
            time.sleep(args.request_pace_seconds)
        return payload

    def progress(completed: int, total: int) -> None:
        if completed == total or completed % 25 == 0:
            print(json.dumps({"completed": completed, "total": total}), flush=True)

    try:
        completed, total = collect_to_checkpoint(
            checkpoint,
            fetcher,
            max_new_objects=args.max_new_objects,
            progress=progress,
        )
    except SourceContractError as exc:
        checkpoint.close()
        status = write_fatal_rejection(args.support, str(exc))
        raise SystemExit(f"BRCR REJECT_NO_REPAIR ({status}): {exc}") from exc

    if completed < total:
        checkpoint.close()
        print(
            json.dumps(
                {"status": "checkpointed", "completed": completed, "total": total},
                indent=2,
            )
        )
        return
    entries = checkpoint.entries()
    result = finalize_artifacts(
        entries,
        source_path=args.source,
        object_manifest_path=args.object_manifest,
        source_manifest_path=args.source_manifest,
        support_path=args.support,
    )
    checkpoint.delete()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
