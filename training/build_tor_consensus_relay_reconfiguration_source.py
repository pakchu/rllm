"""Build and gate the outcome-blind TCRR 2020-2023 Tor source."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import lzma
import os
import sqlite3
import ssl
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from training import preregister_tor_consensus_relay_reconfiguration as protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = Path("training/build_tor_consensus_relay_reconfiguration_source.py")
PROTOCOL_PATH = protocol.DEFAULT_OUTPUT
PROTOCOL_FILE_SHA256 = (
    "aa1d4670dcfcfeaf5e1d32b58c7e309b0745a05ca964cb0d2178b2ccf174db70"
)
PROTOCOL_MANIFEST_HASH = (
    "454afbd388651932de3c71bcfa5dcd0d76ce2633fda55174c81aee58c4603564"
)

DEFAULT_SOURCE = Path(
    "data/tor_consensus_relay_reconfiguration_source_2020_2023.jsonl.gz"
)
DEFAULT_ARCHIVE_MANIFEST = Path(
    "results/tor_consensus_relay_reconfiguration_archive_manifest_2026-07-23.json.gz"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "results/tor_consensus_relay_reconfiguration_source_manifest_2026-07-23.json"
)
DEFAULT_SUPPORT = Path(
    "results/tor_consensus_relay_reconfiguration_source_support_2026-07-23.json"
)
DEFAULT_CHECKPOINT = Path("checkpoints/tcrr_source_checkpoint.sqlite3")

USER_AGENT = "rllm-tcrr-source-freeze/1.0"
READ_CHUNK_BYTES = 1024 * 1024


class SourceContractError(RuntimeError):
    """A frozen source invariant failed and TCRR must not be repaired."""


class SourceFetchError(RuntimeError):
    """A transient transport failure interrupted collection."""


@dataclass(frozen=True)
class FetchedArchive:
    body: bytes
    declared_content_length: int
    content_type: str


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
        raise ValueError("TCRR timestamp must be UTC-aware")
    return value.isoformat().replace("+00:00", "Z")


def load_frozen_protocol() -> dict[str, Any]:
    path = _repo_path(PROTOCOL_PATH)
    if sha256_file(path) != PROTOCOL_FILE_SHA256:
        raise RuntimeError("TCRR source protocol file hash differs from builder binding")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("TCRR source protocol artifact is not an object")
    protocol.validate_protocol(payload)
    if payload.get("manifest_hash") != PROTOCOL_MANIFEST_HASH:
        raise RuntimeError("TCRR source protocol manifest hash differs from binding")
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
            f"TCRR disk guard rejected {used_gib} GiB used at "
            f"{protocol.DISK_LIMIT_GIB} GiB"
        )
    return used_gib


def validate_official_archive_url(manifest_row: Mapping[str, Any]) -> None:
    month = manifest_row.get("month")
    expected = f"{protocol.ARCHIVE_BASE}/consensuses-{month}.tar.xz"
    url = manifest_row.get("url")
    parsed = urllib.parse.urlsplit(str(url))
    if (
        url != expected
        or parsed.scheme != "https"
        or parsed.hostname != "collector.torproject.org"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("TCRR archive URL differs from the frozen official route")


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
        raise SourceFetchError("TCRR archive transport attempted a redirect")


def _read_bounded_response(response: Any, expected_bytes: int) -> FetchedArchive:
    header = response.headers.get("Content-Length")
    try:
        advertised = int(header) if header is not None else -1
    except ValueError as exc:
        raise SourceFetchError("TCRR response has malformed Content-Length") from exc
    if advertised != expected_bytes:
        raise SourceFetchError("TCRR response Content-Length differs from manifest")
    content_type = response.headers.get_content_type()
    if content_type not in {"application/x-xz", "application/octet-stream"}:
        raise SourceFetchError("TCRR response Content-Type differs from XZ transport")
    payload = bytearray()
    while True:
        chunk = response.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > expected_bytes:
            raise SourceFetchError("TCRR response exceeds frozen archive bytes")
    if len(payload) != expected_bytes:
        raise SourceFetchError("TCRR response body differs from Content-Length")
    return FetchedArchive(
        body=bytes(payload),
        declared_content_length=advertised,
        content_type=content_type,
    )


def fetch_archive(
    manifest_row: Mapping[str, Any],
    *,
    retries: int,
    timeout_seconds: float,
    retry_base_seconds: float,
) -> FetchedArchive:
    validate_official_archive_url(manifest_row)
    url = str(manifest_row["url"])
    expected_bytes = int(manifest_row["compressed_bytes"])
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
                        "TCRR archive returned an unexpected response"
                    )
                fetched = _read_bounded_response(response, expected_bytes)
                if sha256_bytes(fetched.body) != manifest_row["compressed_sha256"]:
                    raise SourceContractError(
                        f"TCRR archive SHA-256 differs for {manifest_row['month']}"
                    )
                return fetched
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == retries:
                raise SourceFetchError(
                    f"TCRR archive HTTP failure {exc.code} for {url}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise SourceFetchError(f"TCRR archive fetch failed for {url}") from exc
        except SourceFetchError:
            if attempt == retries:
                raise
        time.sleep(retry_base_seconds * (2**attempt))
    raise AssertionError("TCRR retry loop terminated unexpectedly")


def labels_for_month(month: str) -> list[datetime]:
    return [
        label
        for label in protocol.anchor_labels()
        if label.strftime("%Y-%m") == month
    ]


def _source_summary(
    *,
    label: datetime,
    archive_row: Mapping[str, Any],
    member_name: str,
    parsed: Mapping[str, Any],
) -> dict[str, Any]:
    relays = parsed["relays"]
    flag_counts: Counter[str] = Counter()
    relay_ids: list[str] = []
    guard_ids: list[str] = []
    exit_ids: list[str] = []
    bandwidth_state: list[tuple[str, int]] = []
    total_bandwidth = 0
    for relay in relays:
        identity = relay["relay_identity"]
        bandwidth = relay["consensus_bandwidth"]
        flags = relay["flags"]
        relay_ids.append(identity)
        bandwidth_state.append((identity, bandwidth))
        total_bandwidth += bandwidth
        flag_counts.update(flags)
        if "Guard" in flags:
            guard_ids.append(identity)
        if "Exit" in flags:
            exit_ids.append(identity)
    row = {
        "archive_month": archive_row["month"],
        "archive_url": archive_row["url"],
        "archive_compressed_sha256": archive_row["compressed_sha256"],
        "member_name": member_name,
        "member_bytes": parsed["member_bytes"],
        "member_sha256": parsed["member_sha256"],
        "summary_identity_sha256": parsed["summary_identity_sha256"],
        "source_label_utc": _utc_iso(label),
        "source_availability_utc": parsed["availability_time"],
        "fresh_until_utc": parsed["fresh_until"],
        "valid_until_utc": parsed["valid_until"],
        "consensus_method": parsed["consensus_method"],
        "voting_delay_seconds": parsed["voting_delay_seconds"],
        "relay_count": parsed["relay_count"],
        "guard_count": len(guard_ids),
        "exit_count": len(exit_ids),
        "total_consensus_bandwidth": total_bandwidth,
        "flag_counts": dict(sorted(flag_counts.items())),
        "authority_count": parsed["authority_count"],
        "signature_count": parsed["signature_count"],
        "signed_portion_sha1": parsed["signed_portion_sha1"],
        "relay_set_sha256": canonical_hash({"relay_ids": relay_ids}),
        "guard_set_sha256": canonical_hash({"guard_ids": guard_ids}),
        "exit_set_sha256": canonical_hash({"exit_ids": exit_ids}),
        "bandwidth_state_sha256": canonical_hash(
            {"bandwidth_state": bandwidth_state}
        ),
        "member_identity_sha256": protocol.canonical_member_identity(
            label=label, parsed=parsed
        ),
    }
    row["source_row_identity_sha256"] = canonical_hash(row)
    return row


def process_archive(
    manifest_row: Mapping[str, Any], fetched: FetchedArchive
) -> dict[str, Any]:
    month = str(manifest_row["month"])
    expected_labels = labels_for_month(month)
    expected_names = {protocol.expected_member_name(label): label for label in expected_labels}
    if len(fetched.body) != manifest_row["compressed_bytes"]:
        raise SourceContractError("TCRR archive bytes differ from frozen manifest")
    archive_sha = sha256_bytes(fetched.body)
    if archive_sha != manifest_row["compressed_sha256"]:
        raise SourceContractError("TCRR archive SHA-256 differs from frozen manifest")

    rows_by_name: dict[str, dict[str, Any]] = {}
    all_consensus_members = 0
    regular_members = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(fetched.body), mode="r:xz") as archive:
            for member in archive:
                protocol.validate_tar_member(
                    name=member.name,
                    size=member.size,
                    is_file=member.isfile(),
                    is_symlink=member.issym(),
                    is_hardlink=member.islnk(),
                )
                if not member.isfile():
                    continue
                regular_members += 1
                if not member.name.endswith("-consensus"):
                    raise SourceContractError(
                        "TCRR archive contains an unexpected regular member"
                    )
                all_consensus_members += 1
                if member.name not in expected_names:
                    continue
                if member.name in rows_by_name:
                    raise SourceContractError("TCRR archive duplicates a target member")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SourceContractError("TCRR target member cannot be extracted")
                raw = extracted.read(protocol.MAXIMUM_MEMBER_BYTES + 1)
                if len(raw) != member.size:
                    raise SourceContractError("TCRR target member bytes differ from tar")
                label = expected_names[member.name]
                first = protocol.parse_consensus_document(raw, expected_label=label)
                second = protocol.parse_consensus_document(raw, expected_label=label)
                if first != second:
                    raise SourceContractError("TCRR two in-memory parse passes differ")
                rows_by_name[member.name] = _source_summary(
                    label=label,
                    archive_row=manifest_row,
                    member_name=member.name,
                    parsed=first,
                )
    except SourceContractError:
        raise
    except (EOFError, lzma.LZMAError, tarfile.TarError, ValueError) as exc:
        raise SourceContractError("TCRR XZ, tar, or consensus contract failed") from exc

    if set(rows_by_name) != set(expected_names):
        missing = sorted(set(expected_names).difference(rows_by_name))
        extra = sorted(set(rows_by_name).difference(expected_names))
        raise SourceContractError(
            f"TCRR target membership differs: missing={missing[:3]} extra={extra[:3]}"
        )
    rows = [rows_by_name[protocol.expected_member_name(label)] for label in expected_labels]
    archive_core = {
        "month": month,
        "url": manifest_row["url"],
        "compressed_bytes": len(fetched.body),
        "compressed_sha256": archive_sha,
        "content_type": fetched.content_type,
        "regular_member_count": regular_members,
        "consensus_member_count": all_consensus_members,
        "target_member_count": len(rows),
        "first_target": rows[0]["source_label_utc"],
        "last_target": rows[-1]["source_label_utc"],
        "target_member_identities": [
            row["member_identity_sha256"] for row in rows
        ],
        "parse_twice_verified": True,
    }
    return {
        "archive": {
            **archive_core,
            "archive_identity_sha256": canonical_hash(archive_core),
        },
        "rows": rows,
    }


def _fetched_identity(fetched: FetchedArchive) -> tuple[str, int, str]:
    return (
        sha256_bytes(fetched.body),
        fetched.declared_content_length,
        fetched.content_type,
    )


class SourceCheckpoint:
    def __init__(self, path: str | Path, binding: Mapping[str, Any]) -> None:
        self.path = _repo_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS archives ("
            "month TEXT PRIMARY KEY, payload TEXT NOT NULL, "
            "payload_sha256 TEXT NOT NULL)"
        )
        existing = dict(self.connection.execute("SELECT key, value FROM metadata"))
        expected = {
            key: canonical_json(value).decode("utf-8").rstrip("\n")
            for key, value in binding.items()
        }
        if existing and existing != expected:
            self.close()
            raise RuntimeError("TCRR checkpoint binding differs from current builder")
        archive_count = self.connection.execute(
            "SELECT COUNT(*) FROM archives"
        ).fetchone()[0]
        if not existing and archive_count:
            self.close()
            raise RuntimeError("TCRR checkpoint has archives without a binding")
        if not existing:
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(expected.items()),
            )
            self.connection.commit()

    def months(self) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT month FROM archives ORDER BY month"
            )
        ]

    def put(self, month: str, payload: Mapping[str, Any]) -> None:
        payload_text = canonical_json(payload).decode("utf-8").rstrip("\n")
        payload_sha = sha256_bytes(payload_text.encode("utf-8"))
        existing = self.connection.execute(
            "SELECT payload, payload_sha256 FROM archives WHERE month = ?",
            (month,),
        ).fetchone()
        if existing is not None:
            if existing != (payload_text, payload_sha):
                raise RuntimeError("TCRR checkpoint month has conflicting payload")
            return
        self.connection.execute(
            "INSERT INTO archives(month, payload, payload_sha256) VALUES (?, ?, ?)",
            (month, payload_text, payload_sha),
        )
        self.connection.commit()

    def entries(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT payload, payload_sha256 FROM archives ORDER BY month"
        )
        entries: list[dict[str, Any]] = []
        for payload, expected_sha in rows:
            if sha256_bytes(payload.encode("utf-8")) != expected_sha:
                raise RuntimeError("TCRR checkpoint payload hash mismatch")
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise RuntimeError("TCRR checkpoint payload is not an object")
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
        "archive_manifest_sha256": canonical_hash(
            {"archives": protocol.archive_manifest()}
        ),
        "expected_archives": protocol.EXPECTED_ARCHIVES,
        "expected_target_documents": protocol.EXPECTED_TARGET_DOCUMENTS,
    }


def collect_to_checkpoint(
    checkpoint: SourceCheckpoint,
    fetcher: Callable[[Mapping[str, Any]], FetchedArchive],
    *,
    max_new_archives: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[int, int]:
    manifest = protocol.archive_manifest()
    processed = checkpoint.months()
    expected_prefix = [row["month"] for row in manifest[: len(processed)]]
    if processed != expected_prefix:
        raise RuntimeError("TCRR checkpoint months are not a contiguous frozen prefix")
    added = 0
    for index, row in enumerate(manifest[len(processed) :], start=len(processed)):
        fetched = fetcher(row)
        try:
            entry = process_archive(row, fetched)
        except SourceContractError as exc:
            confirmation = fetcher(row)
            if _fetched_identity(confirmation) != _fetched_identity(fetched):
                raise SourceFetchError(
                    "TCRR source bytes or headers changed while confirming a failure at "
                    f"{row['month']}"
                ) from exc
            raise SourceContractError(
                f"{row['month']} [{sha256_bytes(fetched.body)}]: {exc}"
            ) from exc
        checkpoint.put(str(row["month"]), entry)
        added += 1
        if progress is not None:
            progress(index + 1, len(manifest), str(row["month"]))
        if max_new_archives > 0 and added >= max_new_archives:
            break
    return len(checkpoint.months()), len(manifest)


def evaluate_support(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    manifest = protocol.archive_manifest()
    if len(entries) != len(manifest):
        raise RuntimeError("TCRR support requires all 48 frozen archives")
    rows: list[Mapping[str, Any]] = []
    archive_hashes: list[str] = []
    for expected, entry in zip(manifest, entries):
        archive = entry["archive"]
        if archive["month"] != expected["month"]:
            raise RuntimeError("TCRR support archives are out of frozen order")
        if archive["compressed_sha256"] != expected["compressed_sha256"]:
            raise RuntimeError("TCRR support archive hash differs")
        if archive["compressed_bytes"] != expected["compressed_bytes"]:
            raise RuntimeError("TCRR support archive bytes differ")
        if archive["target_member_count"] != len(labels_for_month(expected["month"])):
            raise RuntimeError("TCRR support monthly target count differs")
        archive_hashes.append(archive["compressed_sha256"])
        rows.extend(entry["rows"])
    labels = protocol.anchor_labels()
    if len(rows) != len(labels):
        raise RuntimeError("TCRR support target denominator differs")
    label_texts = [row["source_label_utc"] for row in rows]
    if label_texts != [_utc_iso(label) for label in labels]:
        raise RuntimeError("TCRR support source rows are out of frozen order")
    member_hashes = [row["member_sha256"] for row in rows]
    row_identities = [row["source_row_identity_sha256"] for row in rows]
    if len(set(member_hashes)) != len(member_hashes):
        raise RuntimeError("TCRR support has duplicate member bytes across labels")
    if len(set(row_identities)) != len(row_identities):
        raise RuntimeError("TCRR support has duplicate source row identities")
    yearly: dict[str, dict[str, Any]] = {}
    for year in range(2020, 2024):
        year_rows = [row for row in rows if row["source_label_utc"].startswith(str(year))]
        yearly[str(year)] = {
            "target_documents": len(year_rows),
            "minimum_relay_count": min(row["relay_count"] for row in year_rows),
            "maximum_relay_count": max(row["relay_count"] for row in year_rows),
            "minimum_authority_count": min(
                row["authority_count"] for row in year_rows
            ),
            "minimum_signature_count": min(
                row["signature_count"] for row in year_rows
            ),
        }
    gates = [
        {
            "gate_id": "frozen_archives",
            "observed": len(entries),
            "threshold": protocol.EXPECTED_ARCHIVES,
            "operator": "==",
            "passed": len(entries) == protocol.EXPECTED_ARCHIVES,
        },
        {
            "gate_id": "target_documents",
            "observed": len(rows),
            "threshold": protocol.EXPECTED_TARGET_DOCUMENTS,
            "operator": "==",
            "passed": len(rows) == protocol.EXPECTED_TARGET_DOCUMENTS,
        },
        {
            "gate_id": "unique_archive_hashes",
            "observed": len(set(archive_hashes)),
            "threshold": protocol.EXPECTED_ARCHIVES,
            "operator": "==",
            "passed": len(set(archive_hashes)) == protocol.EXPECTED_ARCHIVES,
        },
        {
            "gate_id": "unique_member_hashes",
            "observed": len(set(member_hashes)),
            "threshold": protocol.EXPECTED_TARGET_DOCUMENTS,
            "operator": "==",
            "passed": len(set(member_hashes)) == protocol.EXPECTED_TARGET_DOCUMENTS,
        },
        {
            "gate_id": "parse_twice_months",
            "observed": sum(
                int(entry["archive"]["parse_twice_verified"]) for entry in entries
            ),
            "threshold": protocol.EXPECTED_ARCHIVES,
            "operator": "==",
            "passed": all(entry["archive"]["parse_twice_verified"] for entry in entries),
        },
    ]
    all_passed = all(gate["passed"] for gate in gates)
    return {
        "source_id": "TCRR",
        "status": (
            "PASS_ADVANCE_TO_RECONFIGURATION_FREEZE"
            if all_passed
            else "REJECT_NO_REPAIR"
        ),
        "all_gates_passed": all_passed,
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "mechanism_features_opened": False,
        "expected_archive_count": protocol.EXPECTED_ARCHIVES,
        "expected_target_document_count": protocol.EXPECTED_TARGET_DOCUMENTS,
        "source_row_count": len(rows),
        "yearly_source_summary": yearly,
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
            raise RuntimeError(f"refusing to overwrite frozen TCRR artifact {output}")
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
            raise RuntimeError(f"refusing to overwrite frozen TCRR artifact {output}")
    return {name: write_once(path, raw) for name, path, raw in artifacts}


def finalize_artifacts(
    entries: Sequence[Mapping[str, Any]],
    *,
    source_path: str | Path = DEFAULT_SOURCE,
    archive_manifest_path: str | Path = DEFAULT_ARCHIVE_MANIFEST,
    source_manifest_path: str | Path = DEFAULT_SOURCE_MANIFEST,
    support_path: str | Path = DEFAULT_SUPPORT,
) -> dict[str, Any]:
    rows = [row for entry in entries for row in entry["rows"]]
    archive_rows = [entry["archive"] for entry in entries]
    support_core = {
        **evaluate_support(entries),
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_sha256": sha256_file(BUILDER_PATH),
    }
    support = {**support_core, "manifest_hash": canonical_hash(support_core)}
    source_raw = _gzip_jsonl(rows)
    archive_raw = _gzip_jsonl(archive_rows)
    support_raw = _pretty_json(support)
    source_manifest_core = {
        "source_id": "TCRR",
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_path": str(BUILDER_PATH),
        "builder_sha256": sha256_file(BUILDER_PATH),
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "mechanism_features_opened": False,
        "source": {
            "path": str(source_path),
            "rows": len(rows),
            "bytes": len(source_raw),
            "sha256": sha256_bytes(source_raw),
        },
        "archive_manifest": {
            "path": str(archive_manifest_path),
            "archives": len(archive_rows),
            "bytes": len(archive_raw),
            "sha256": sha256_bytes(archive_raw),
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
            ("archive_manifest", archive_manifest_path, archive_raw),
            ("support", support_path, support_raw),
            ("source_manifest", source_manifest_path, source_manifest_raw),
        )
    )
    return {
        "status": support["status"],
        "all_gates_passed": support["all_gates_passed"],
        "source_rows": len(rows),
        "archives": len(archive_rows),
        "artifact_statuses": statuses,
        "source_manifest_hash": source_manifest["manifest_hash"],
    }


def write_fatal_rejection(path: str | Path, reason: str) -> str:
    core = {
        "source_id": "TCRR",
        "status": "REJECT_NO_REPAIR",
        "all_gates_passed": False,
        "fatal_source_contract_failure": reason,
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_manifest_hash": PROTOCOL_MANIFEST_HASH,
        "builder_sha256": sha256_file(BUILDER_PATH),
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "mechanism_features_opened": False,
    }
    return write_once(
        path, _pretty_json({**core, "manifest_hash": canonical_hash(core)})
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--archive-manifest", type=Path, default=DEFAULT_ARCHIVE_MANIFEST
    )
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--max-new-archives", type=int, default=0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument(
        "--keep-checkpoint-on-success", action="store_true", default=False
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_frozen_protocol()
    enforce_disk_guard(args.checkpoint)
    checkpoint = SourceCheckpoint(args.checkpoint, checkpoint_binding())

    def fetcher(row: Mapping[str, Any]) -> FetchedArchive:
        enforce_disk_guard(args.checkpoint)
        return fetch_archive(
            row,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            retry_base_seconds=args.retry_base_seconds,
        )

    def progress(completed: int, total: int, month: str) -> None:
        print(f"TCRR source {completed}/{total} archives checkpointed through {month}", flush=True)

    try:
        completed, total = collect_to_checkpoint(
            checkpoint,
            fetcher,
            max_new_archives=args.max_new_archives,
            progress=progress,
        )
        if completed != total:
            checkpoint.close()
            print(
                json.dumps(
                    {
                        "status": "CHECKPOINT_INCOMPLETE",
                        "completed_archives": completed,
                        "expected_archives": total,
                        "outcomes_opened": False,
                    },
                    indent=2,
                )
            )
            return
        entries = checkpoint.entries()
        result = finalize_artifacts(
            entries,
            source_path=args.source,
            archive_manifest_path=args.archive_manifest,
            source_manifest_path=args.source_manifest,
            support_path=args.support,
        )
        if args.keep_checkpoint_on_success:
            checkpoint.close()
        else:
            checkpoint.delete()
        print(json.dumps(result, indent=2, sort_keys=True))
    except SourceContractError as exc:
        checkpoint.close()
        write_fatal_rejection(args.support, str(exc))
        print(str(exc), flush=True)
        raise SystemExit(2) from exc
    except BaseException:
        checkpoint.close()
        raise


if __name__ == "__main__":
    main()
