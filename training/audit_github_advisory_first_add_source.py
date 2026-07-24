"""Audit the frozen GitHub-reviewed advisory first-add source without outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = Path(
    "docs/github-advisory-first-add-source-axis-decision-2026-07-24.md"
)
SCRIPT_PATH = Path("training/audit_github_advisory_first_add_source.py")
TEST_PATH = Path("tests/test_audit_github_advisory_first_add_source.py")
BOUNDARY_SHA256 = (
    "b167da46a43308a5ce6be70563c455b1c4209499ae5a0423efbdad15080bb25f"
)
PROTOCOL_VERSION = "GHAD-GRFA-D1-source-audit-2026-07-24"
GIT_EXECUTABLE = Path("/usr/bin/git")
GIT_EXECUTABLE_SHA256 = (
    "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
)
GIT_VERSION = "git version 2.43.0"

OFFICIAL_REMOTE = "https://github.com/github/advisory-database.git"
FROZEN_COMMIT = "40e5791b176b832cb09323d3962abe2fe3249e34"
FROZEN_TREE = "283dcf468588e3f9fd4a1d7a671df11527788dfc"
FROZEN_PARENT = "0ab828c5a28c008f4c6f3344a8bb783484c41378"
FROZEN_DEFAULT_BRANCH = "refs/heads/main"
AUTHORIZED_ROOT = "advisories/github-reviewed"

SOURCE_START = datetime(2022, 2, 11, 22, 59, 38, tzinfo=timezone.utc)
SOURCE_AVAILABILITY_START = datetime(
    2022, 2, 12, 12, 0, 0, tzinfo=timezone.utc
)
SOURCE_END_EXCLUSIVE = datetime(2026, 1, 1, tzinfo=timezone.utc)

DISK_USED_LIMIT = 300 * 1024**3
DISK_FREE_FLOOR = 8 * 1024**3
GIT_OBJECT_STORE_CAP = 8 * 1024**3
CANDIDATE_MATERIAL_CAP = 2 * 1024**3
SINGLE_BLOB_CAP = 8 * 1024**2
CANDIDATE_FETCH_CHUNK_MAX = 128
MANIFEST_GROWTH_RESERVE = 64 * 1024**2

DEFAULT_SENTINEL = Path(
    "results/.github_advisory_first_add_source_2026-07-24.started"
)
DEFAULT_MANIFEST = Path(
    "results/.github_advisory_first_add_source_2026-07-24.manifest.ndjson"
)
DEFAULT_RAW_DIR = Path(
    "results/.github_advisory_first_add_source_2026-07-24.raw"
)
DEFAULT_REPORT = Path(
    "results/github_advisory_first_add_source_2026-07-24.json"
)

HEX40 = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
GHSA_ID = re.compile(
    r"GHSA-[23456789cfghjmpqrvwx]{4}-"
    r"[23456789cfghjmpqrvwx]{4}-"
    r"[23456789cfghjmpqrvwx]{4}\Z",
    re.ASCII,
)
AUTHORIZED_PATH = re.compile(
    r"advisories/github-reviewed/"
    r"(?:[A-Za-z0-9_-]+/)+"
    r"(?P<identity>GHSA-[23456789cfghjmpqrvwx]{4}-"
    r"[23456789cfghjmpqrvwx]{4}-"
    r"[23456789cfghjmpqrvwx]{4})\.json\Z",
    re.ASCII,
)
RAW_DIFF_HEADER = re.compile(
    rb":(?P<old_mode>[0-7]{6}) (?P<new_mode>[0-7]{6}) "
    rb"(?P<old_oid>[0-9a-f]{40}) (?P<new_oid>[0-9a-f]{40}) "
    rb"(?P<status>[ADMT])\Z"
)
RFC3339_UTC = re.compile(
    r"(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})T"
    r"(?P<clock>[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(?P<fraction>\.[0-9]{1,9})?Z\Z",
    re.ASCII,
)
REGULAR_MODES = frozenset({"100644", "100755"})
ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "modified",
        "published",
        "withdrawn",
        "aliases",
        "summary",
        "details",
        "severity",
        "affected",
        "references",
        "credits",
        "database_specific",
        "ecosystem_specific",
    }
)

_PRODUCTION_EXECUTION = object()
_FIXTURE_EXECUTION = object()


class GhadError(RuntimeError):
    """Base error for the frozen source audit."""


class ProtocolError(GhadError):
    """A local execution or preregistration invariant failed."""


class DiskGuardError(ProtocolError):
    """A frozen storage boundary failed."""


class TransportError(GhadError):
    """The exact Git transport failed."""


class HistoryError(GhadError):
    """The frozen first-parent history was malformed."""


class StructureError(GhadError):
    """An initial advisory blob violated the frozen structure."""


class SupportError(GhadError):
    """The aggregate support battery failed."""


class PublicationError(GhadError):
    """A write-once report could not be published."""


@dataclass(frozen=True, slots=True)
class FirstAddCandidate:
    identity: str
    path: str
    commit_hash: str
    blob_oid: str
    first_parent_position: int
    committer_time_utc: str
    ordered_committer_time_utc: str


@dataclass(frozen=True, slots=True)
class HistoryScan:
    candidates: tuple[FirstAddCandidate, ...]
    active_path_count: int
    active_tree_sha256: str
    first_parent_commit_count: int
    transition_count: int
    mutation_counts: dict[str, int]
    first_parent_chain_sha256: str
    raw_path_delta_sha256: str


@dataclass(frozen=True, slots=True)
class SourceEvent:
    identity_digest: str
    first_add_commit_digest: str
    initial_blob_sha1: str
    raw_sha256: str
    structural_sha256: str
    published_at_utc: str
    modified_at_utc: str
    availability_at_utc: str
    schema_version: str
    ecosystems: tuple[str, ...]
    withdrawn: bool
    summary_nonempty: bool
    details_nonempty: bool
    summary_utf8_bytes: int
    details_utf8_bytes: int
    severity_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceCorpus:
    selected_events: tuple[SourceEvent, ...]
    candidate_count: int
    candidate_raw_bytes: int
    candidate_raw_hashes_sha256: str
    opaque_postwindow_count: int
    postwindow_count: int
    prewindow_count: int


@dataclass(frozen=True, slots=True)
class AuditPaths:
    sentinel: Path
    manifest: Path
    raw_dir: Path
    report: Path


PRODUCTION_PATHS = AuditPaths(
    sentinel=DEFAULT_SENTINEL,
    manifest=DEFAULT_MANIFEST,
    raw_dir=DEFAULT_RAW_DIR,
    report=DEFAULT_REPORT,
)


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return encoded + (b"\n" if newline else b"")


def canonical_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("UTC timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def datetime_from_epoch(epoch: int) -> datetime:
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (OSError, OverflowError, ValueError) as exc:
        raise HistoryError("Git committer timestamp is outside datetime range") from exc


def parse_rfc3339_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise StructureError(f"{field} is not exact RFC3339 UTC")
    match = RFC3339_UTC.fullmatch(value)
    if match is None:
        raise StructureError(f"{field} is not exact RFC3339 UTC")
    fraction = match.group("fraction")
    microseconds = ""
    if fraction is not None:
        microseconds = fraction[1:][:6].ljust(6, "0")
    normalized = f"{match.group('date')}T{match.group('clock')}"
    if microseconds:
        normalized += f".{microseconds}"
    normalized += "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise StructureError(f"{field} is not a valid UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise StructureError(f"{field} is not UTC")
    return parsed


def _rfc3339_order_key(value: str, *, field: str) -> tuple[int, int]:
    parsed = parse_rfc3339_utc(value, field=field)
    match = RFC3339_UTC.fullmatch(value)
    if match is None:
        raise StructureError(f"{field} is not exact RFC3339 UTC")
    fraction = match.group("fraction")
    nanoseconds = int((fraction[1:] if fraction else "").ljust(9, "0"))
    whole_seconds = int(parsed.replace(microsecond=0).timestamp())
    return whole_seconds, nanoseconds


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def _duplicate_key_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructureError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise StructureError("JSON numbers must be finite")


def _assert_finite_numbers(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StructureError("JSON numbers must be finite")
    elif isinstance(value, list):
        for item in value:
            _assert_finite_numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_numbers(item)


def parse_json_object(raw: bytes) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise StructureError("UTF-8 BOM is forbidden")
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StructureError("advisory is not strict UTF-8") from exc
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_duplicate_key_object,
            parse_constant=_reject_nonfinite,
        )
    except StructureError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StructureError("advisory is not strict JSON") from exc
    if not isinstance(value, dict):
        raise StructureError("advisory JSON root is not an object")
    _assert_finite_numbers(value)
    return value


def _require_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructureError(f"{field} must be a nonempty string")
    if "\x00" in value:
        raise StructureError(f"{field} contains NUL")
    return value


def _optional_text(payload: Mapping[str, Any], field: str) -> tuple[bool, int]:
    if field not in payload:
        return False, 0
    value = payload[field]
    if not isinstance(value, str):
        raise StructureError(f"{field} must be a UTF-8 string")
    if "\x00" in value:
        raise StructureError(f"{field} contains NUL")
    encoded = value.encode("utf-8")
    return bool(value.strip()), len(encoded)


def _extract_ecosystems(affected: object) -> tuple[str, ...]:
    if not isinstance(affected, list) or not affected:
        raise StructureError("affected must be a nonempty list")
    ecosystems: set[str] = set()
    valid_package = False
    for item in affected:
        if not isinstance(item, dict):
            raise StructureError("affected entries must be objects")
        package = item.get("package")
        if package is None:
            continue
        if not isinstance(package, dict):
            raise StructureError("affected package must be an object")
        ecosystem = package.get("ecosystem")
        name = package.get("name")
        if (
            not isinstance(ecosystem, str)
            or not ecosystem.strip()
            or not isinstance(name, str)
            or not name.strip()
        ):
            raise StructureError("affected package ecosystem/name is missing")
        if "\x00" in ecosystem or "\x00" in name:
            raise StructureError("affected package contains NUL")
        purl = package.get("purl")
        if purl is not None and not isinstance(purl, str):
            raise StructureError("affected package purl has invalid type")
        for list_field in ("ranges", "versions"):
            if list_field in item and not isinstance(item[list_field], list):
                raise StructureError(f"affected {list_field} must be a list")
        ecosystems.add(ecosystem)
        valid_package = True
    if not valid_package:
        raise StructureError("affected contains no valid package")
    return tuple(sorted(ecosystems))


def _extract_severity_types(severity: object) -> tuple[str, ...]:
    if severity is None:
        return ()
    if not isinstance(severity, list):
        raise StructureError("severity must be a list")
    result: set[str] = set()
    for item in severity:
        if not isinstance(item, dict):
            raise StructureError("severity entries must be objects")
        severity_type = _require_nonempty_string(
            item.get("type"), field="severity type"
        )
        score = item.get("score")
        if score is not None and not isinstance(score, str):
            raise StructureError("severity score must be a string")
        result.add(severity_type)
    return tuple(sorted(result))


class _MinimalJsonClockScanner:
    """Validate JSON syntax while decoding only top-level identity/clock."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def _whitespace(self) -> None:
        while (
            self.position < len(self.text)
            and self.text[self.position] in " \t\r\n"
        ):
            self.position += 1

    def _expect(self, token: str) -> None:
        if not self.text.startswith(token, self.position):
            raise StructureError("minimal advisory JSON syntax is malformed")
        self.position += len(token)

    def _string(self, *, decode: bool) -> str | None:
        if self.position >= len(self.text) or self.text[self.position] != '"':
            raise StructureError("minimal advisory JSON string is malformed")
        start = self.position
        self.position += 1
        while self.position < len(self.text):
            character = self.text[self.position]
            codepoint = ord(character)
            if character == '"':
                self.position += 1
                raw = self.text[start : self.position]
                if not decode:
                    return None
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise StructureError(
                        "minimal advisory JSON string is malformed"
                    ) from exc
                if not isinstance(value, str):
                    raise StructureError(
                        "minimal advisory JSON string type changed"
                    )
                return value
            if codepoint < 0x20:
                raise StructureError(
                    "minimal advisory JSON string has a control character"
                )
            if character != "\\":
                self.position += 1
                continue
            self.position += 1
            if self.position >= len(self.text):
                raise StructureError("minimal advisory JSON escape is truncated")
            escape = self.text[self.position]
            if escape in '"\\/bfnrt':
                self.position += 1
                continue
            if escape != "u":
                raise StructureError("minimal advisory JSON escape is invalid")
            self.position += 1
            digits = self.text[self.position : self.position + 4]
            if len(digits) != 4 or re.fullmatch(r"[0-9A-Fa-f]{4}", digits) is None:
                raise StructureError("minimal advisory Unicode escape is invalid")
            value = int(digits, 16)
            self.position += 4
            if 0xD800 <= value <= 0xDBFF:
                if not self.text.startswith("\\u", self.position):
                    raise StructureError(
                        "minimal advisory Unicode surrogate is unpaired"
                    )
                low_digits = self.text[
                    self.position + 2 : self.position + 6
                ]
                if (
                    len(low_digits) != 4
                    or re.fullmatch(r"[0-9A-Fa-f]{4}", low_digits) is None
                    or not 0xDC00 <= int(low_digits, 16) <= 0xDFFF
                ):
                    raise StructureError(
                        "minimal advisory Unicode surrogate is unpaired"
                    )
                self.position += 6
            elif 0xDC00 <= value <= 0xDFFF:
                raise StructureError(
                    "minimal advisory Unicode surrogate is unpaired"
                )
        raise StructureError("minimal advisory JSON string is unterminated")

    def _number(self) -> None:
        match = re.match(
            r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
            r"(?:[eE][+-]?[0-9]+)?",
            self.text[self.position :],
            re.ASCII,
        )
        if match is None:
            raise StructureError("minimal advisory JSON number is malformed")
        self.position += len(match.group(0))

    def _value(self, depth: int) -> None:
        if depth > 256:
            raise StructureError("minimal advisory JSON nesting is excessive")
        self._whitespace()
        if self.position >= len(self.text):
            raise StructureError("minimal advisory JSON value is missing")
        character = self.text[self.position]
        if character == '"':
            self._string(decode=False)
        elif character == "{":
            self._object(depth + 1)
        elif character == "[":
            self._array(depth + 1)
        elif character in "-0123456789":
            self._number()
        elif self.text.startswith("true", self.position):
            self.position += 4
        elif self.text.startswith("false", self.position):
            self.position += 5
        elif self.text.startswith("null", self.position):
            self.position += 4
        else:
            raise StructureError("minimal advisory JSON value is malformed")

    def _array(self, depth: int) -> None:
        self._expect("[")
        self._whitespace()
        if self.position < len(self.text) and self.text[self.position] == "]":
            self.position += 1
            return
        while True:
            self._value(depth)
            self._whitespace()
            if self.position >= len(self.text):
                raise StructureError("minimal advisory JSON array is truncated")
            if self.text[self.position] == "]":
                self.position += 1
                return
            self._expect(",")
            self._whitespace()

    def _object(self, depth: int) -> None:
        self._expect("{")
        self._whitespace()
        if self.position < len(self.text) and self.text[self.position] == "}":
            self.position += 1
            return
        while True:
            self._string(decode=False)
            self._whitespace()
            self._expect(":")
            self._value(depth)
            self._whitespace()
            if self.position >= len(self.text):
                raise StructureError("minimal advisory JSON object is truncated")
            if self.text[self.position] == "}":
                self.position += 1
                return
            self._expect(",")
            self._whitespace()

    def root_identity_clock(self) -> tuple[str, str]:
        self._whitespace()
        self._expect("{")
        self._whitespace()
        found: dict[str, str] = {}
        seen_top_level: set[str] = set()
        if self.position < len(self.text) and self.text[self.position] == "}":
            self.position += 1
        else:
            while True:
                key = self._string(decode=True)
                if not isinstance(key, str):
                    raise StructureError("minimal advisory key is malformed")
                if key in seen_top_level:
                    raise StructureError("duplicate top-level JSON key")
                seen_top_level.add(key)
                self._whitespace()
                self._expect(":")
                self._whitespace()
                if key in {"id", "published"}:
                    value = self._string(decode=True)
                    if not isinstance(value, str):
                        raise StructureError(
                            f"minimal advisory {key} must be a string"
                        )
                    found[key] = value
                else:
                    self._value(1)
                self._whitespace()
                if self.position >= len(self.text):
                    raise StructureError(
                        "minimal advisory JSON object is truncated"
                    )
                if self.text[self.position] == "}":
                    self.position += 1
                    break
                self._expect(",")
                self._whitespace()
        self._whitespace()
        if self.position != len(self.text):
            raise StructureError("minimal advisory JSON has trailing bytes")
        if set(found) != {"id", "published"}:
            raise StructureError("minimal advisory id/published is missing")
        return found["id"], found["published"]


def parse_minimal_identity_clock(
    candidate: FirstAddCandidate,
    raw: bytes,
) -> tuple[str, datetime]:
    if not isinstance(raw, bytes) or len(raw) > SINGLE_BLOB_CAP:
        raise StructureError("minimal advisory blob violates byte cap")
    if git_blob_sha1(raw) != candidate.blob_oid:
        raise StructureError("minimal advisory blob Git identity changed")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise StructureError("minimal advisory UTF-8 BOM is forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StructureError("minimal advisory is not strict UTF-8") from exc
    identity, published_value = _MinimalJsonClockScanner(
        text
    ).root_identity_clock()
    if identity != candidate.identity:
        raise StructureError("minimal advisory path/id disagreement")
    published = parse_rfc3339_utc(
        published_value,
        field="minimal published",
    )
    return published_value, published


def parse_initial_blob(candidate: FirstAddCandidate, raw: bytes) -> SourceEvent:
    if not isinstance(raw, bytes):
        raise StructureError("initial advisory blob must be bytes")
    if len(raw) > SINGLE_BLOB_CAP:
        raise StructureError("initial advisory blob exceeds frozen cap")
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise StructureError("Git LFS pointer is forbidden")
    if git_blob_sha1(raw) != candidate.blob_oid:
        raise StructureError("initial advisory blob Git identity changed")
    payload = parse_json_object(raw)
    unknown = set(payload) - ALLOWED_TOP_LEVEL_FIELDS
    if unknown:
        raise StructureError("advisory contains a non-frozen top-level field")
    if payload.get("id") != candidate.identity:
        raise StructureError("advisory path/id disagreement")
    schema_version = _require_nonempty_string(
        payload.get("schema_version"), field="schema_version"
    )
    published_value = payload.get("published")
    modified_value = payload.get("modified")
    published = parse_rfc3339_utc(published_value, field="published")
    modified = parse_rfc3339_utc(modified_value, field="modified")
    if not isinstance(published_value, str) or not isinstance(modified_value, str):
        raise StructureError("published/modified timestamp type changed")
    if _rfc3339_order_key(
        modified_value, field="modified"
    ) < _rfc3339_order_key(published_value, field="published"):
        raise StructureError("modified precedes published")
    withdrawn_value = payload.get("withdrawn")
    withdrawn = withdrawn_value is not None
    if withdrawn:
        parse_rfc3339_utc(withdrawn_value, field="withdrawn")
    ecosystems = _extract_ecosystems(payload.get("affected"))
    summary_nonempty, summary_bytes = _optional_text(payload, "summary")
    details_nonempty, details_bytes = _optional_text(payload, "details")
    severity_types = _extract_severity_types(payload.get("severity"))
    ordered_commit = parse_rfc3339_utc(
        candidate.ordered_committer_time_utc,
        field="ordered committer time",
    )
    source_floor = max(ordered_commit, published)
    availability = datetime.combine(
        source_floor.date() + timedelta(days=1),
        time(12),
        tzinfo=timezone.utc,
    )
    structural = canonical_json_bytes(payload, newline=False)
    return SourceEvent(
        identity_digest=sha256_bytes(candidate.identity.encode("ascii")),
        first_add_commit_digest=sha256_bytes(
            candidate.commit_hash.encode("ascii")
        ),
        initial_blob_sha1=candidate.blob_oid,
        raw_sha256=sha256_bytes(raw),
        structural_sha256=sha256_bytes(structural),
        published_at_utc=published_value,
        modified_at_utc=modified_value,
        availability_at_utc=canonical_utc(availability),
        schema_version=schema_version,
        ecosystems=ecosystems,
        withdrawn=withdrawn,
        summary_nonempty=summary_nonempty,
        details_nonempty=details_nonempty,
        summary_utf8_bytes=summary_bytes,
        details_utf8_bytes=details_bytes,
        severity_types=severity_types,
    )


def _local_git_environment() -> dict[str, str]:
    environment = sealed_git_environment(
        REPOSITORY_ROOT / "results" / ".ghad-local-git-home"
    )
    environment["GIT_NO_LAZY_FETCH"] = "1"
    return environment


def _run_git(
    repo: Path,
    *args: str,
    input_bytes: bytes | None = None,
    environment: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        [str(GIT_EXECUTABLE), "-C", str(repo), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        close_fds=True,
        env=dict(environment or _local_git_environment()),
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise HistoryError(f"Git {' '.join(args)} failed: {detail}")
    return completed


def _parse_rev_list(raw: bytes) -> list[tuple[int, str, tuple[str, ...]]]:
    rows: list[tuple[int, str, tuple[str, ...]]] = []
    previous: str | None = None
    for line in raw.splitlines():
        parts = line.decode("ascii", errors="strict").split(" ")
        if len(parts) < 2 or not parts[0].isdigit() or HEX40.fullmatch(parts[1]) is None:
            raise HistoryError("first-parent rev-list row is malformed")
        parents = tuple(parts[2:])
        if any(HEX40.fullmatch(parent) is None for parent in parents):
            raise HistoryError("first-parent rev-list parent is malformed")
        commit = parts[1]
        if previous is None:
            if parents:
                raise HistoryError("first-parent traversal does not begin at root")
        elif not parents or parents[0] != previous:
            raise HistoryError("first-parent traversal is not contiguous")
        rows.append((int(parts[0]), commit, parents))
        previous = commit
    if not rows or len({row[1] for row in rows}) != len(rows):
        raise HistoryError("first-parent traversal is empty or duplicated")
    return rows


def _parse_diff_stream(
    raw: bytes,
    known_commits: set[str],
) -> list[tuple[str, dict[str, str]]]:
    if raw and not raw.endswith(b"\x00"):
        raise HistoryError("raw path-delta stream is not NUL-terminated")
    tokens = raw.split(b"\x00")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    result: list[tuple[str, dict[str, str]]] = []
    current_commit: str | None = None
    index = 0
    seen_paths_by_commit: dict[str, set[str]] = defaultdict(set)
    while index < len(tokens):
        token = tokens[index]
        try:
            decoded = token.decode("ascii", errors="strict")
        except UnicodeDecodeError:
            decoded = ""
        if HEX40.fullmatch(decoded) is not None:
            if decoded not in known_commits:
                raise HistoryError("diff stream contains a non-chain commit")
            current_commit = decoded
            index += 1
            continue
        if current_commit is None or index + 1 >= len(tokens):
            raise HistoryError("raw path-delta stream lost commit framing")
        match = RAW_DIFF_HEADER.fullmatch(token)
        if match is None:
            raise HistoryError("raw path-delta header is malformed")
        path_raw = tokens[index + 1]
        try:
            path = path_raw.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise StructureError("reviewed advisory path is not ASCII") from exc
        if path in seen_paths_by_commit[current_commit]:
            raise HistoryError("raw path-delta repeats a path")
        seen_paths_by_commit[current_commit].add(path)
        result.append(
            (
                current_commit,
                {
                    "old_mode": match.group("old_mode").decode("ascii"),
                    "new_mode": match.group("new_mode").decode("ascii"),
                    "old_oid": match.group("old_oid").decode("ascii"),
                    "new_oid": match.group("new_oid").decode("ascii"),
                    "status": match.group("status").decode("ascii"),
                    "path": path,
                },
            )
        )
        index += 2
    return result


def _validate_change(change: Mapping[str, str]) -> str:
    path = change["path"]
    path_match = AUTHORIZED_PATH.fullmatch(path)
    if path_match is None:
        raise StructureError("reviewed path is outside the authorized pattern")
    status = change["status"]
    old_mode = change["old_mode"]
    new_mode = change["new_mode"]
    if status == "A":
        if old_mode != "000000" or new_mode not in REGULAR_MODES:
            raise StructureError("advisory addition is not a regular blob")
    elif status == "D":
        if old_mode not in REGULAR_MODES or new_mode != "000000":
            raise StructureError("advisory deletion is not a regular blob")
    elif status == "M":
        if old_mode not in REGULAR_MODES or new_mode not in REGULAR_MODES:
            raise StructureError("advisory modification is not a regular blob")
    elif status == "T":
        raise StructureError("advisory path has a forbidden type change")
    else:
        raise HistoryError("unsupported raw path-delta status")
    return path_match.group("identity")


def collect_first_add_candidates(
    repo: Path,
    pinned_commit: str,
) -> HistoryScan:
    if HEX40.fullmatch(pinned_commit) is None:
        raise HistoryError("pinned commit is malformed")
    rev_list = _run_git(
        repo,
        "rev-list",
        "--first-parent",
        "--reverse",
        "--timestamp",
        "--parents",
        pinned_commit,
    ).stdout
    chain = _parse_rev_list(rev_list)
    if chain[-1][1] != pinned_commit:
        raise HistoryError("first-parent traversal does not end at pinned commit")

    commit_rows: dict[str, tuple[int, str, str]] = {}
    running_epoch: int | None = None
    chain_hasher = hashlib.sha256()
    for position, (epoch, commit, parents) in enumerate(chain):
        ordered_epoch = epoch if running_epoch is None else max(epoch, running_epoch)
        running_epoch = ordered_epoch
        committer_time = canonical_utc(datetime_from_epoch(epoch))
        ordered_time = canonical_utc(datetime_from_epoch(ordered_epoch))
        commit_rows[commit] = (position, committer_time, ordered_time)
        chain_hasher.update(
            canonical_json_bytes(
                {
                    "commit": commit,
                    "committer_epoch": epoch,
                    "ordered_committer_epoch": ordered_epoch,
                    "parents": list(parents),
                    "position": position,
                }
            )
        )

    commit_input = b"".join(
        commit.encode("ascii") + b"\n" for _, commit, _ in chain
    )
    raw_delta = _run_git(
        repo,
        "diff-tree",
        "--stdin",
        "--root",
        "--raw",
        "-r",
        "-z",
        "--no-renames",
        "--no-abbrev",
        "-m",
        "--first-parent",
        "--diff-filter=ADMT",
        "--",
        AUTHORIZED_ROOT,
        input_bytes=commit_input,
    ).stdout
    changes = _parse_diff_stream(raw_delta, set(commit_rows))

    mutation_counts: Counter[str] = Counter(
        {
            "addition": 0,
            "deletion": 0,
            "modification": 0,
            "readdition": 0,
            "type_change": 0,
        }
    )
    seen: dict[str, FirstAddCandidate] = {}
    active_paths: dict[str, set[str]] = defaultdict(set)
    active_entries: dict[str, dict[str, str]] = {}
    candidates: list[FirstAddCandidate] = []
    changes_by_commit: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(
        list
    )
    for commit, change in changes:
        identity = _validate_change(change)
        changes_by_commit[commit].append((identity, change))

    for _, commit, _ in chain:
        by_identity: dict[str, list[dict[str, str]]] = defaultdict(list)
        for identity, change in changes_by_commit[commit]:
            by_identity[identity].append(change)
            status = change["status"]
            mutation_key = {
                "A": "addition",
                "D": "deletion",
                "M": "modification",
                "T": "type_change",
            }[status]
            mutation_counts[mutation_key] += 1

        for identity, identity_changes in by_identity.items():
            additions = [
                change for change in identity_changes if change["status"] == "A"
            ]
            deletions = [
                change for change in identity_changes if change["status"] == "D"
            ]
            modifications = [
                change for change in identity_changes if change["status"] == "M"
            ]
            if len(additions) > 1:
                raise StructureError(
                    "duplicate identity in one first-parent transition"
                )
            for change in deletions:
                if change["path"] not in active_paths[identity]:
                    raise HistoryError("advisory deletion has no active identity path")
                current = active_entries.get(change["path"])
                if current != {
                    "mode": change["old_mode"],
                    "oid": change["old_oid"],
                    "path": change["path"],
                }:
                    raise HistoryError("advisory deletion does not replay active tree")
                active_paths[identity].remove(change["path"])
                del active_entries[change["path"]]
            for change in modifications:
                if change["path"] not in active_paths[identity]:
                    raise HistoryError(
                        "advisory modification has no active identity path"
                    )
                current = active_entries.get(change["path"])
                if current != {
                    "mode": change["old_mode"],
                    "oid": change["old_oid"],
                    "path": change["path"],
                }:
                    raise HistoryError(
                        "advisory modification does not replay active tree"
                    )
                active_entries[change["path"]] = {
                    "mode": change["new_mode"],
                    "oid": change["new_oid"],
                    "path": change["path"],
                }
            if not additions:
                continue
            addition = additions[0]
            if active_paths[identity]:
                raise StructureError("active duplicate identity was added")
            if addition["path"] in active_entries:
                raise HistoryError("advisory addition overwrites an active path")
            active_paths[identity].add(addition["path"])
            active_entries[addition["path"]] = {
                "mode": addition["new_mode"],
                "oid": addition["new_oid"],
                "path": addition["path"],
            }
            if identity in seen:
                mutation_counts["readdition"] += 1
                continue
            position, committer_time, ordered_time = commit_rows[commit]
            candidate = FirstAddCandidate(
                identity=identity,
                path=addition["path"],
                commit_hash=commit,
                blob_oid=addition["new_oid"],
                first_parent_position=position,
                committer_time_utc=committer_time,
                ordered_committer_time_utc=ordered_time,
            )
            seen[identity] = candidate
            candidates.append(candidate)

    candidates.sort(key=lambda row: (row.first_parent_position, row.path))
    active_tree_rows = [
        active_entries[path] for path in sorted(active_entries)
    ]
    return HistoryScan(
        candidates=tuple(candidates),
        active_path_count=len(active_tree_rows),
        active_tree_sha256=sha256_bytes(
            canonical_json_bytes(active_tree_rows, newline=False)
        ),
        first_parent_commit_count=len(chain),
        transition_count=len(changes),
        mutation_counts={
            key: mutation_counts[key]
            for key in (
                "addition",
                "deletion",
                "modification",
                "readdition",
                "type_change",
            )
        },
        first_parent_chain_sha256=chain_hasher.hexdigest(),
        raw_path_delta_sha256=sha256_bytes(raw_delta),
    )


def read_git_blobs(
    repo: Path,
    object_ids: Sequence[str],
) -> dict[str, bytes]:
    if len(set(object_ids)) != len(object_ids):
        raise StructureError("candidate blob request repeats an object")
    if any(HEX40.fullmatch(object_id) is None for object_id in object_ids):
        raise StructureError("candidate blob identity is malformed")
    process = subprocess.Popen(
        [str(GIT_EXECUTABLE), "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        env=_local_git_environment(),
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise HistoryError("Git cat-file pipes are unavailable")
    result: dict[str, bytes] = {}
    try:
        for object_id in object_ids:
            process.stdin.write(object_id.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().rstrip(b"\n").split(b" ")
            if (
                len(header) != 3
                or header[0].decode("ascii", errors="strict") != object_id
                or header[1] != b"blob"
                or not header[2].isdigit()
            ):
                raise StructureError("candidate object is not an exact Git blob")
            size = int(header[2])
            if size > SINGLE_BLOB_CAP:
                raise StructureError("candidate advisory blob exceeds frozen cap")
            raw = process.stdout.read(size)
            if len(raw) != size or process.stdout.read(1) != b"\n":
                raise StructureError("candidate advisory blob is truncated")
            result[object_id] = raw
        process.stdin.close()
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise HistoryError(f"Git cat-file failed: {detail}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return result


def read_git_blob(repo: Path, object_id: str) -> bytes:
    return read_git_blobs(repo, [object_id])[object_id]


def parse_candidates(
    repo: Path,
    candidates: Sequence[FirstAddCandidate],
) -> tuple[SourceEvent, ...]:
    object_ids = sorted({candidate.blob_oid for candidate in candidates})
    blobs = read_git_blobs(repo, object_ids)
    events = tuple(
        parse_initial_blob(candidate, blobs[candidate.blob_oid])
        for candidate in candidates
    )
    identities = [event.identity_digest for event in events]
    if len(identities) != len(set(identities)):
        raise StructureError("parsed source repeats an advisory identity")
    return events


def _candidate_commit_availability(candidate: FirstAddCandidate) -> datetime:
    ordered = parse_rfc3339_utc(
        candidate.ordered_committer_time_utc,
        field="ordered committer time",
    )
    return datetime.combine(
        ordered.date() + timedelta(days=1),
        time(12),
        tzinfo=timezone.utc,
    )


def parse_candidate_corpus(
    repo: Path,
    candidates: Sequence[FirstAddCandidate],
) -> SourceCorpus:
    object_ids = sorted({candidate.blob_oid for candidate in candidates})
    blobs = read_git_blobs(repo, object_ids)
    selected: list[SourceEvent] = []
    prewindow = 0
    postwindow = 0
    opaque_postwindow = 0
    raw_bytes = 0
    hash_ledger: list[dict[str, Any]] = []
    for candidate in candidates:
        raw = blobs[candidate.blob_oid]
        if len(raw) > SINGLE_BLOB_CAP:
            raise StructureError("candidate advisory blob exceeds frozen cap")
        if git_blob_sha1(raw) != candidate.blob_oid:
            raise StructureError("candidate advisory blob Git identity changed")
        raw_hash = sha256_bytes(raw)
        raw_bytes += len(raw)
        commit_availability = _candidate_commit_availability(candidate)
        if commit_availability >= SOURCE_END_EXCLUSIVE:
            postwindow += 1
            opaque_postwindow += 1
            classification = "opaque_postwindow"
        else:
            published_value, published = parse_minimal_identity_clock(
                candidate, raw
            )
            ordered_commit = parse_rfc3339_utc(
                candidate.ordered_committer_time_utc,
                field="ordered committer time",
            )
            minimal_availability = datetime.combine(
                max(ordered_commit, published).date() + timedelta(days=1),
                time(12),
                tzinfo=timezone.utc,
            )
            if (
                published >= SOURCE_END_EXCLUSIVE
                or minimal_availability >= SOURCE_END_EXCLUSIVE
            ):
                postwindow += 1
                opaque_postwindow += 1
                classification = "opaque_postwindow"
            elif (
                published < SOURCE_START
                or minimal_availability < SOURCE_AVAILABILITY_START
            ):
                prewindow += 1
                classification = "prewindow"
            else:
                event = parse_initial_blob(candidate, raw)
                if (
                    event.published_at_utc != published_value
                    or not in_source_window(event)
                ):
                    raise StructureError(
                        "minimal/full source-window classification disagrees"
                    )
                selected.append(event)
                classification = "source_window"
        hash_ledger.append(
            {
                "blob_oid": candidate.blob_oid,
                "classification": classification,
                "raw_bytes": len(raw),
                "raw_sha256": raw_hash,
            }
        )
    identities = [event.identity_digest for event in selected]
    if len(identities) != len(set(identities)):
        raise StructureError("selected source repeats an advisory identity")
    return SourceCorpus(
        selected_events=tuple(selected),
        candidate_count=len(candidates),
        candidate_raw_bytes=raw_bytes,
        candidate_raw_hashes_sha256=sha256_bytes(
            canonical_json_bytes(hash_ledger, newline=False)
        ),
        opaque_postwindow_count=opaque_postwindow,
        postwindow_count=postwindow,
        prewindow_count=prewindow,
    )


def in_source_window(event: SourceEvent) -> bool:
    published = parse_rfc3339_utc(
        event.published_at_utc, field="published event clock"
    )
    availability = parse_rfc3339_utc(
        event.availability_at_utc, field="availability event clock"
    )
    return (
        SOURCE_START <= published < SOURCE_END_EXCLUSIVE
        and SOURCE_AVAILABILITY_START
        <= availability
        < SOURCE_END_EXCLUSIVE
    )


def _gate(
    gate_id: str,
    observed: Any,
    requirement: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "observed": observed,
        "requirement": requirement,
        "passed": bool(passed),
    }


def _length_distribution(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "maximum": 0, "median": 0, "minimum": 0, "p95": 0}
    ordered = sorted(values)

    def nearest_rank(percent: int) -> int:
        index = max(0, math.ceil(len(ordered) * percent / 100) - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "maximum": ordered[-1],
        "median": nearest_rank(50),
        "minimum": ordered[0],
        "p95": nearest_rank(95),
    }


def _events_fingerprint(events: Sequence[SourceEvent]) -> str:
    hasher = hashlib.sha256()
    for event in sorted(
        events,
        key=lambda row: (
            row.availability_at_utc,
            row.identity_digest,
        ),
    ):
        hasher.update(canonical_json_bytes(asdict(event)))
    return hasher.hexdigest()


def evaluate_support(
    events: Sequence[SourceEvent],
    *,
    source_event_count: int | None = None,
    prewindow_count: int = 0,
    postwindow_count: int = 0,
    opaque_postwindow_count: int = 0,
    candidate_raw_hashes_sha256: str | None = None,
) -> dict[str, Any]:
    identities = [event.identity_digest for event in events]
    duplicates = len(identities) - len(set(identities))
    selected = [event for event in events if in_source_window(event)]
    year_counts: Counter[int] = Counter()
    month_counts: Counter[str] = Counter()
    day_counts: dict[int, Counter[str]] = defaultdict(Counter)
    ecosystem_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    withdrawal_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    nonwithdrawn = 0
    complete_text = 0
    summary_lengths: list[int] = []
    details_lengths: list[int] = []

    for event in selected:
        availability = parse_rfc3339_utc(
            event.availability_at_utc,
            field="availability event clock",
        )
        year = availability.year
        month_key = f"{year:04d}-{availability.month:02d}"
        day_key = availability.date().isoformat()
        year_counts[year] += 1
        month_counts[month_key] += 1
        day_counts[year][day_key] += 1
        for ecosystem in set(event.ecosystems):
            ecosystem_counts[ecosystem] += 1
        schema_counts[event.schema_version] += 1
        withdrawal_counts["withdrawn" if event.withdrawn else "not_withdrawn"] += 1
        severity_counts.update(set(event.severity_types) or {"__none__"})
        summary_lengths.append(event.summary_utf8_bytes)
        details_lengths.append(event.details_utf8_bytes)
        if not event.withdrawn:
            nonwithdrawn += 1
            if event.summary_nonempty and event.details_nonempty:
                complete_text += 1

    gates: list[dict[str, Any]] = [
        _gate(
            "duplicate_identity_zero",
            duplicates,
            "exactly 0",
            duplicates == 0,
        ),
        _gate(
            "events_2022",
            year_counts[2022],
            "at least 500",
            year_counts[2022] >= 500,
        ),
    ]
    for year in (2023, 2024, 2025):
        gates.append(
            _gate(
                f"events_{year}",
                year_counts[year],
                "at least 1000",
                year_counts[year] >= 1000,
            )
        )
        unique_days = len(day_counts[year])
        gates.append(
            _gate(
                f"unique_days_{year}",
                unique_days,
                "at least 200",
                unique_days >= 200,
            )
        )
        monthly = {
            f"{year:04d}-{month:02d}": month_counts[
                f"{year:04d}-{month:02d}"
            ]
            for month in range(1, 13)
        }
        gates.append(
            _gate(
                f"monthly_minimum_{year}",
                monthly,
                "every month at least 50",
                all(value >= 50 for value in monthly.values()),
            )
        )
        maximum_day = max(day_counts[year].values(), default=0)
        concentration = maximum_day / year_counts[year] if year_counts[year] else 1.0
        gates.append(
            _gate(
                f"daily_concentration_{year}",
                round(concentration, 12),
                "at most 0.10",
                bool(year_counts[year]) and concentration <= 0.10,
            )
        )

    gates.append(
        _gate(
            "ecosystem_count",
            len(ecosystem_counts),
            "at least 5",
            len(ecosystem_counts) >= 5,
        )
    )
    maximum_ecosystem = max(ecosystem_counts.values(), default=0)
    ecosystem_dominance = (
        maximum_ecosystem / len(selected) if selected else 1.0
    )
    gates.append(
        _gate(
            "ecosystem_dominance",
            round(ecosystem_dominance, 12),
            "at most 0.80 of selected events",
            bool(selected) and ecosystem_dominance <= 0.80,
        )
    )
    completeness = complete_text / nonwithdrawn if nonwithdrawn else 0.0
    gates.append(
        _gate(
            "text_completeness",
            round(completeness, 12),
            "at least 0.95 of non-withdrawn events",
            bool(nonwithdrawn) and completeness >= 0.95,
        )
    )
    all_passed = all(gate["passed"] for gate in gates)
    if source_event_count is None:
        source_event_count = len(events)
        prewindow_count = 0
        postwindow_count = 0
        for event in events:
            published = parse_rfc3339_utc(
                event.published_at_utc, field="published event clock"
            )
            availability = parse_rfc3339_utc(
                event.availability_at_utc, field="availability event clock"
            )
            if (
                published < SOURCE_START
                or availability < SOURCE_AVAILABILITY_START
            ):
                prewindow_count += 1
            if (
                published >= SOURCE_END_EXCLUSIVE
                or availability >= SOURCE_END_EXCLUSIVE
            ):
                postwindow_count += 1
    return {
        "aggregate": {
            "availability_unique_days_by_year": {
                str(year): len(day_counts[year])
                for year in sorted(day_counts)
            },
            "details_utf8_bytes": _length_distribution(details_lengths),
            "ecosystem_event_counts": dict(sorted(ecosystem_counts.items())),
            "month_counts": dict(sorted(month_counts.items())),
            "candidate_raw_hashes_sha256": candidate_raw_hashes_sha256,
            "not_selected_opaque_postwindow_events": opaque_postwindow_count,
            "not_selected_postwindow_events": postwindow_count,
            "not_selected_prewindow_events": prewindow_count,
            "schema_version_counts": dict(sorted(schema_counts.items())),
            "selected_event_count": len(selected),
            "selected_events_sha256": _events_fingerprint(selected),
            "severity_type_event_counts": dict(sorted(severity_counts.items())),
            "source_event_count": source_event_count,
            "summary_utf8_bytes": _length_distribution(summary_lengths),
            "withdrawal_counts": dict(sorted(withdrawal_counts.items())),
            "year_counts": {
                str(year): year_counts[year] for year in sorted(year_counts)
            },
        },
        "all_gates_passed": all_passed,
        "decision": "SOURCE_SUPPORT_PASS" if all_passed else "TERMINAL_REJECT",
        "gates": gates,
    }


def directory_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        if root_path.is_symlink():
            raise DiskGuardError("storage path traverses a symlink")
        for directory in directories:
            if (root_path / directory).is_symlink():
                raise DiskGuardError("storage path contains a symlink")
        for filename in files:
            candidate = root_path / filename
            if candidate.is_symlink():
                raise DiskGuardError("storage path contains a symlink")
            total += candidate.stat().st_size
    return total


def assert_disk_guard(path: Path = REPOSITORY_ROOT) -> tuple[int, int]:
    usage = shutil.disk_usage(path)
    if usage.used >= DISK_USED_LIMIT or usage.free < DISK_FREE_FLOOR:
        raise DiskGuardError("filesystem is outside the frozen disk guard")
    return usage.used, usage.free


def assert_object_store_guard(objects: Path) -> int:
    size = directory_bytes(objects)
    if size >= GIT_OBJECT_STORE_CAP:
        raise DiskGuardError("Git object store exceeds the frozen cap")
    return size


def plan_candidate_fetch_chunk(
    *,
    remaining_count: int,
    materialized_raw_bytes: int,
    candidate_manifest_bytes: int,
    retrieval_manifest_bytes: int,
) -> int:
    if remaining_count <= 0:
        return 0
    committed_bytes = (
        materialized_raw_bytes
        + candidate_manifest_bytes
        + retrieval_manifest_bytes
        + MANIFEST_GROWTH_RESERVE
    )
    headroom = CANDIDATE_MATERIAL_CAP - committed_bytes
    safe_by_worst_case = (headroom - 1) // SINGLE_BLOB_CAP
    if safe_by_worst_case < 1:
        raise DiskGuardError(
            "candidate material cap leaves no safe pre-fetch headroom"
        )
    return min(
        remaining_count,
        CANDIDATE_FETCH_CHUNK_MAX,
        safe_by_worst_case,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_no_symlink_components(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise ProtocolError("one-shot artifact path traverses a symlink")
        if current == current.parent:
            break
        current = current.parent


def _canonical_artifact_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise ProtocolError("one-shot artifact path must be a Path")
    lexical = repository_path(path)
    if ".." in lexical.parts:
        raise ProtocolError("one-shot artifact path contains a parent segment")
    _assert_no_symlink_components(lexical)
    canonical = lexical.resolve(strict=False)
    _assert_no_symlink_components(canonical)
    return canonical


def _absolute_paths(paths: AuditPaths) -> AuditPaths:
    return AuditPaths(
        sentinel=_canonical_artifact_path(paths.sentinel),
        manifest=_canonical_artifact_path(paths.manifest),
        raw_dir=_canonical_artifact_path(paths.raw_dir),
        report=_canonical_artifact_path(paths.report),
    )


def _validate_paths(paths: AuditPaths) -> AuditPaths:
    absolute = _absolute_paths(paths)
    values = (
        absolute.sentinel,
        absolute.manifest,
        absolute.raw_dir,
        absolute.report,
    )
    if len(set(values)) != len(values):
        raise ProtocolError("one-shot artifact paths overlap")
    return absolute


def validate_fixture_paths(paths: AuditPaths) -> AuditPaths:
    absolute = _validate_paths(paths)
    production_root = _canonical_artifact_path(Path("results"))
    for path in (
        absolute.sentinel,
        absolute.manifest,
        absolute.raw_dir,
        absolute.report,
    ):
        if (
            path == production_root
            or production_root in path.parents
            or path in production_root.parents
        ):
            raise ProtocolError("fixture artifacts must be disjoint from production")
    return absolute


def validate_fixture_source_repo(source_repo: Path) -> Path:
    canonical = _canonical_artifact_path(source_repo)
    production_raw = _canonical_artifact_path(DEFAULT_RAW_DIR)
    if (
        canonical == production_raw
        or production_raw in canonical.parents
        or canonical in production_raw.parents
    ):
        raise ProtocolError(
            "fixture source repository overlaps production source storage"
        )
    remotes = _run_git(
        canonical,
        "remote",
        "-v",
        check=False,
    )
    if remotes.returncode != 0:
        raise ProtocolError("fixture source repository is not readable Git")
    decoded = remotes.stdout.decode("utf-8", errors="strict")
    if OFFICIAL_REMOTE in decoded or "github.com/github/advisory-database" in decoded:
        raise ProtocolError("fixture source repository references official source")
    return canonical


def _exclusive_write(path: Path, payload: bytes, *, mode: int) -> None:
    _assert_no_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(path)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
    except FileExistsError as exc:
        raise ProtocolError("one-shot artifact already exists") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)
    _fsync_directory(path.parent)


def _atomic_publish(path: Path, payload: bytes) -> None:
    _assert_no_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _exclusive_write(temporary, payload, mode=0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PublicationError("one-shot report already exists") from exc
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(slots=True)
class AttemptGuard:
    sentinel_path: Path
    manifest_path: Path
    sentinel_bytes: bytes
    sentinel_sha256: str
    manifest_ordinal: int = 0
    manifest_last_hash: str = "0" * 64
    manifest_prefix_sha256: str = hashlib.sha256(b"").hexdigest()

    def _validate_sentinel(self) -> None:
        try:
            current = self.sentinel_path.read_bytes()
        except OSError as exc:
            raise ProtocolError("one-shot sentinel is unavailable") from exc
        if current != self.sentinel_bytes or sha256_bytes(current) != self.sentinel_sha256:
            raise ProtocolError("one-shot sentinel changed")

    def _validate_manifest(self) -> None:
        try:
            raw = self.manifest_path.read_bytes()
        except OSError as exc:
            raise ProtocolError("retrieval manifest is unavailable") from exc
        if sha256_bytes(raw) != self.manifest_prefix_sha256:
            raise ProtocolError("retrieval manifest exact prefix changed")
        if raw and not raw.endswith(b"\n"):
            raise ProtocolError("retrieval manifest has a partial record")
        ordinal = 0
        previous_hash = "0" * 64
        for line in raw.splitlines():
            try:
                record = json.loads(line.decode("ascii", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProtocolError("retrieval manifest is malformed") from exc
            if set(record) != {
                "event",
                "ordinal",
                "payload",
                "previous_sha256",
                "record_sha256",
            }:
                raise ProtocolError("retrieval manifest record shape changed")
            ordinal += 1
            if record["ordinal"] != ordinal or record["previous_sha256"] != previous_hash:
                raise ProtocolError("retrieval manifest chain changed")
            claimed = record["record_sha256"]
            unsigned = dict(record)
            unsigned.pop("record_sha256")
            computed = sha256_bytes(canonical_json_bytes(unsigned, newline=False))
            if claimed != computed or line != canonical_json_bytes(record, newline=False):
                raise ProtocolError("retrieval manifest record hash changed")
            previous_hash = computed
        if ordinal != self.manifest_ordinal or previous_hash != self.manifest_last_hash:
            raise ProtocolError("retrieval manifest prefix changed")

    def validate(self) -> None:
        self._validate_sentinel()
        self._validate_manifest()

    def append(self, event: str, payload: Mapping[str, Any]) -> str:
        self.validate()
        unsigned = {
            "event": event,
            "ordinal": self.manifest_ordinal + 1,
            "payload": dict(payload),
            "previous_sha256": self.manifest_last_hash,
        }
        record_hash = sha256_bytes(canonical_json_bytes(unsigned, newline=False))
        record = {**unsigned, "record_sha256": record_hash}
        encoded = canonical_json_bytes(record)
        descriptor = os.open(self.manifest_path, os.O_WRONLY | os.O_APPEND)
        with os.fdopen(descriptor, "ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self.manifest_ordinal += 1
        self.manifest_last_hash = record_hash
        self.manifest_prefix_sha256 = sha256_file(self.manifest_path)
        return record_hash


def reserve_attempt(
    *,
    paths: AuditPaths,
    verifier_commit: str,
    runner_blob: str,
    started_at_utc: datetime | None = None,
    run_id: str | None = None,
) -> AttemptGuard:
    absolute = _validate_paths(paths)
    for path in (
        absolute.sentinel,
        absolute.manifest,
        absolute.raw_dir,
        absolute.report,
    ):
        if path.exists():
            raise ProtocolError("one-shot path already exists")
    started = started_at_utc or datetime.now(timezone.utc)
    identifier = run_id or str(uuid.uuid4())
    try:
        uuid.UUID(identifier)
    except ValueError as exc:
        raise ProtocolError("run ID is not a UUID") from exc
    sentinel_bytes = canonical_json_bytes(
        {
            "boundary_sha256": BOUNDARY_SHA256,
            "protocol_version": PROTOCOL_VERSION,
            "reserved_at_utc": canonical_utc(started),
            "run_id": identifier,
            "runner_git_blob": runner_blob,
            "verifier_commit": verifier_commit,
        }
    )
    _exclusive_write(absolute.sentinel, sentinel_bytes, mode=0o400)
    try:
        _exclusive_write(absolute.manifest, b"", mode=0o600)
        absolute.raw_dir.mkdir(mode=0o700)
        _fsync_directory(absolute.raw_dir.parent)
    except BaseException:
        # The sentinel survives: an attempted source run is consumed.
        raise
    return AttemptGuard(
        sentinel_path=absolute.sentinel,
        manifest_path=absolute.manifest,
        sentinel_bytes=sentinel_bytes,
        sentinel_sha256=sha256_bytes(sentinel_bytes),
    )


def _repo_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(GIT_EXECUTABLE), *args],
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        close_fds=True,
        env=sealed_git_environment(
            REPOSITORY_ROOT / "results" / ".ghad-preflight-home"
        ),
    )


def assert_git_executable() -> dict[str, str]:
    if (
        not GIT_EXECUTABLE.is_file()
        or GIT_EXECUTABLE.is_symlink()
        or GIT_EXECUTABLE.resolve(strict=True) != GIT_EXECUTABLE
        or sha256_file(GIT_EXECUTABLE) != GIT_EXECUTABLE_SHA256
    ):
        raise ProtocolError("frozen Git executable changed")
    completed = subprocess.run(
        [str(GIT_EXECUTABLE), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        close_fds=True,
        env=sealed_git_environment(
            REPOSITORY_ROOT / "results" / ".ghad-preflight-home"
        ),
    )
    if completed.returncode != 0 or completed.stdout.strip() != GIT_VERSION:
        raise ProtocolError("frozen Git version changed")
    return {
        "path": str(GIT_EXECUTABLE),
        "sha256": GIT_EXECUTABLE_SHA256,
        "version": GIT_VERSION,
    }


def assert_protocol_committed() -> tuple[str, str]:
    assert_git_executable()
    if sha256_file(repository_path(BOUNDARY_PATH)) != BOUNDARY_SHA256:
        raise ProtocolError("boundary hash changed")
    for path in (BOUNDARY_PATH, SCRIPT_PATH, TEST_PATH):
        tracked = _repo_git("ls-files", "--error-unmatch", str(path))
        if tracked.returncode != 0:
            raise ProtocolError("required protocol path is not committed")
        clean = _repo_git("diff", "--quiet", "HEAD", "--", str(path))
        if clean.returncode != 0:
            raise ProtocolError("required protocol path differs from HEAD")
    status = _repo_git("status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0 or status.stdout:
        raise ProtocolError("repository is not HEAD-clean")
    head = _repo_git("rev-parse", "HEAD")
    blob = _repo_git("rev-parse", f"HEAD:{SCRIPT_PATH}")
    if head.returncode != 0 or HEX40.fullmatch(head.stdout.strip()) is None:
        raise ProtocolError("HEAD commit is unavailable")
    if blob.returncode != 0 or HEX40.fullmatch(blob.stdout.strip()) is None:
        raise ProtocolError("runner Git blob is unavailable")
    return head.stdout.strip(), blob.stdout.strip()


FORBIDDEN_ENVIRONMENT_NAMES = frozenset(
    {
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "DATABASE_URL",
        "DB_HOST",
        "DB_PASSWORD",
        "DB_PORT",
        "DB_USER",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "OPENAI_API_KEY",
        "WANDB_API_KEY",
    }
)
FORBIDDEN_MODULE_PREFIXES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "gymnasium",
    "numpy",
    "pandas",
    "peft",
    "psycopg",
    "sqlalchemy",
    "stable_baselines3",
    "torch",
    "transformers",
    "trl",
)


def assert_no_leak_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    names = set(environment or os.environ)
    forbidden = names & FORBIDDEN_ENVIRONMENT_NAMES
    if forbidden:
        raise ProtocolError("market/model/database environment is present")


def assert_no_leak_runtime() -> None:
    for module_name in sys.modules:
        if module_name.startswith(FORBIDDEN_MODULE_PREFIXES):
            raise ProtocolError("market/model/database module is imported")


def sealed_git_environment(home_root: Path) -> dict[str, str]:
    home = home_root.resolve(strict=False)
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / "xdg"),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_LFS_SKIP_SMUDGE": "1",
    }


def fetch_objects_command(
    repo: Path,
    *,
    allow_lazy_fetch: bool,
) -> list[str]:
    command = [
        str(GIT_EXECUTABLE),
        "-c",
        "protocol.version=2",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "credential.helper=",
        "-c",
        "http.proxy=",
        "-c",
        "https.proxy=",
        "-c",
        "http.followRedirects=false",
    ]
    if allow_lazy_fetch:
        command.extend(["-c", "fetch.negotiationAlgorithm=noop"])
    command.extend(
        [
            "-C",
            str(repo),
            "fetch",
            "origin",
            "--no-tags",
            "--no-write-fetch-head",
            "--recurse-submodules=no",
            "--filter=blob:none",
            "--stdin",
        ]
    )
    return command


def _run_sealed_command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        list(command),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        close_fds=True,
        env=dict(environment),
        cwd=REPOSITORY_ROOT,
    )
    if completed.returncode != 0:
        raise TransportError("sealed Git command failed")
    return completed


def _object_inventory(
    repo: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    result = _run_git(
        repo,
        "cat-file",
        "--batch-all-objects",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        environment=environment,
    )
    counts: Counter[str] = Counter()
    blobs: set[str] = set()
    blob_sizes: dict[str, int] = {}
    total_declared_bytes = 0
    for line in result.stdout.splitlines():
        parts = line.decode("ascii", errors="strict").split(" ")
        if (
            len(parts) != 3
            or HEX40.fullmatch(parts[0]) is None
            or parts[1] not in {"blob", "commit", "tag", "tree"}
            or not parts[2].isdigit()
        ):
            raise HistoryError("Git object inventory is malformed")
        counts[parts[1]] += 1
        total_declared_bytes += int(parts[2])
        if parts[1] == "blob":
            blobs.add(parts[0])
            blob_sizes[parts[0]] = int(parts[2])
    return {
        "blob_declared_bytes": sum(blob_sizes.values()),
        "blob_oids": blobs,
        "blob_sizes": blob_sizes,
        "counts": {
            kind: counts[kind] for kind in ("blob", "commit", "tag", "tree")
        },
        "declared_uncompressed_bytes": total_declared_bytes,
        "sha256": sha256_bytes(result.stdout),
    }


def _verify_current_tree(
    repo: Path,
    pinned_commit: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    raw = _run_git(
        repo,
        "ls-tree",
        "-r",
        "-z",
        pinned_commit,
        "--",
        AUTHORIZED_ROOT,
        environment=environment,
    ).stdout
    entries: list[dict[str, str]] = []
    for entry in raw.split(b"\x00"):
        if not entry:
            continue
        metadata, separator, path_raw = entry.partition(b"\t")
        fields = metadata.split(b" ")
        if (
            separator != b"\t"
            or len(fields) != 3
            or fields[0].decode("ascii", errors="strict") not in REGULAR_MODES
            or fields[1] != b"blob"
            or HEX40.fullmatch(fields[2].decode("ascii", errors="strict")) is None
        ):
            raise StructureError("current reviewed subtree is not regular blobs")
        try:
            path = path_raw.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise StructureError("current reviewed subtree path is not ASCII") from exc
        if AUTHORIZED_PATH.fullmatch(path) is None:
            raise StructureError("current reviewed subtree has unauthorized path")
        entries.append(
            {
                "mode": fields[0].decode("ascii", errors="strict"),
                "oid": fields[2].decode("ascii", errors="strict"),
                "path": path,
            }
        )
    entries.sort(key=lambda row: row["path"])
    return {
        "raw_tree_listing_sha256": sha256_bytes(raw),
        "regular_blob_count": len(entries),
        "tree_listing_sha256": sha256_bytes(
            canonical_json_bytes(entries, newline=False)
        ),
    }


def _verify_repository_boundary(
    repo: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    object_format = _run_git(
        repo, "rev-parse", "--show-object-format", environment=environment
    ).stdout.decode("ascii", errors="strict").strip()
    is_bare = _run_git(
        repo, "rev-parse", "--is-bare-repository", environment=environment
    ).stdout.decode("ascii", errors="strict").strip()
    commit = _run_git(
        repo, "rev-parse", f"{FROZEN_COMMIT}^{{commit}}", environment=environment
    ).stdout.decode("ascii", errors="strict").strip()
    tree = _run_git(
        repo, "rev-parse", f"{FROZEN_COMMIT}^{{tree}}", environment=environment
    ).stdout.decode("ascii", errors="strict").strip()
    parent = _run_git(
        repo, "rev-parse", f"{FROZEN_COMMIT}^1", environment=environment
    ).stdout.decode("ascii", errors="strict").strip()
    remote = _run_git(
        repo, "remote", "get-url", "origin", environment=environment
    ).stdout.decode("utf-8", errors="strict").strip()
    if (
        object_format != "sha1"
        or is_bare != "true"
        or commit != FROZEN_COMMIT
        or tree != FROZEN_TREE
        or parent != FROZEN_PARENT
        or remote != OFFICIAL_REMOTE
    ):
        raise HistoryError("frozen repository identity changed")
    for forbidden in (
        repo / "shallow",
        repo / "info" / "grafts",
        repo / "objects" / "info" / "alternates",
    ):
        if forbidden.exists():
            raise HistoryError("forbidden shallow/graft/alternate state exists")
    replacements = _run_git(
        repo,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace",
        environment=environment,
    ).stdout
    if replacements:
        raise HistoryError("replacement refs exist")
    gitmodules = _run_git(
        repo,
        "ls-tree",
        FROZEN_COMMIT,
        "--",
        ".gitmodules",
        environment=environment,
    ).stdout
    if gitmodules:
        raise HistoryError("frozen repository declares submodules")
    return {
        "commit": commit,
        "git_executable_sha256": GIT_EXECUTABLE_SHA256,
        "git_version": GIT_VERSION,
        "is_bare_repository": True,
        "object_format": object_format,
        "parent": parent,
        "remote": remote,
        "tree": tree,
    }


def _write_candidate_manifest(
    path: Path,
    scan: HistoryScan,
) -> tuple[int, str]:
    payload = {
        "candidates": [asdict(candidate) for candidate in scan.candidates],
        "first_parent_chain_sha256": scan.first_parent_chain_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "raw_path_delta_sha256": scan.raw_path_delta_sha256,
    }
    encoded = canonical_json_bytes(payload)
    _exclusive_write(path, encoded, mode=0o400)
    return len(encoded), sha256_bytes(encoded)


def _events_replay_fingerprint(
    scan: HistoryScan,
    events: Sequence[SourceEvent],
) -> dict[str, Any]:
    return {
        "active_path_count": scan.active_path_count,
        "active_tree_sha256": scan.active_tree_sha256,
        "candidate_count": len(scan.candidates),
        "candidate_manifest_sha256": sha256_bytes(
            canonical_json_bytes(
                [asdict(candidate) for candidate in scan.candidates],
                newline=False,
            )
        ),
        "events_sha256": _events_fingerprint(events),
        "first_parent_chain_sha256": scan.first_parent_chain_sha256,
        "raw_path_delta_sha256": scan.raw_path_delta_sha256,
    }


def _corpus_replay_fingerprint(
    scan: HistoryScan,
    corpus: SourceCorpus,
) -> dict[str, Any]:
    return {
        **_events_replay_fingerprint(scan, corpus.selected_events),
        "candidate_count": corpus.candidate_count,
        "candidate_raw_bytes": corpus.candidate_raw_bytes,
        "candidate_raw_hashes_sha256": corpus.candidate_raw_hashes_sha256,
        "opaque_postwindow_count": corpus.opaque_postwindow_count,
        "postwindow_count": corpus.postwindow_count,
        "prewindow_count": corpus.prewindow_count,
    }


def _outcome_boundary() -> dict[str, bool]:
    return {
        "advisory_semantics_or_labels_opened": False,
        "candidate_incidence_opened": False,
        "market_or_price_opened": False,
        "model_tokenizer_adapter_or_prompt_opened": False,
        "package_popularity_or_exploit_activity_opened": False,
        "portfolio_reward_checkpoint_or_performance_opened": False,
    }


def _bindings(
    *,
    verifier_commit: str,
    runner_blob: str,
    guard: AttemptGuard,
) -> dict[str, Any]:
    return {
        "boundary_path": str(BOUNDARY_PATH),
        "boundary_sha256": BOUNDARY_SHA256,
        "manifest_artifact": guard.manifest_path.name,
        "manifest_sha256": sha256_file(guard.manifest_path),
        "protocol_version": PROTOCOL_VERSION,
        "runner_git_blob": runner_blob,
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256_file(repository_path(SCRIPT_PATH)),
        "sentinel_artifact": guard.sentinel_path.name,
        "sentinel_sha256": guard.sentinel_sha256,
        "test_path": str(TEST_PATH),
        "test_sha256": sha256_file(repository_path(TEST_PATH)),
        "verifier_commit": verifier_commit,
    }


def _finalize_report_hash(report: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(report)
    unsigned.pop("manifest_hash_without_self", None)
    report["manifest_hash_without_self"] = sha256_bytes(
        canonical_json_bytes(unsigned, newline=False)
    )
    return report


def deterministic_source_fingerprint(
    *,
    repository_identity: Mapping[str, Any],
    current_tree: Mapping[str, Any],
    scan: HistoryScan,
    replay: Mapping[str, Any],
    support: Mapping[str, Any],
    transport: Mapping[str, Any],
) -> str:
    stable = {
        "boundary_sha256": BOUNDARY_SHA256,
        "current_tree": dict(current_tree),
        "history": {
            "active_path_count": scan.active_path_count,
            "active_tree_sha256": scan.active_tree_sha256,
            "candidate_first_add_count": len(scan.candidates),
            "first_parent_commit_count": scan.first_parent_commit_count,
            "first_parent_chain_sha256": scan.first_parent_chain_sha256,
            "mutation_counts": scan.mutation_counts,
            "raw_path_delta_sha256": scan.raw_path_delta_sha256,
            "transition_count": scan.transition_count,
        },
        "protocol_version": PROTOCOL_VERSION,
        "repository_identity": dict(repository_identity),
        "replay": dict(replay),
        "support": dict(support),
        "transport_source_hashes": {
            key: transport[key]
            for key in (
                "candidate_blob_count",
                "candidate_blob_oids_sha256",
                "candidate_manifest_sha256",
            )
        },
    }
    return sha256_bytes(canonical_json_bytes(stable, newline=False))


def _fixture_report(
    *,
    verifier_commit: str,
    runner_blob: str,
    guard: AttemptGuard,
    scan: HistoryScan,
    support: Mapping[str, Any],
) -> dict[str, Any]:
    return _finalize_report_hash(
        {
            "bindings": _bindings(
                verifier_commit=verifier_commit,
                runner_blob=runner_blob,
                guard=guard,
            ),
            "created_at_utc": canonical_utc(datetime.now(timezone.utc)),
            "decision": support["decision"],
            "execution_authority": "offline_fixture",
            "history": {
                "active_path_count": scan.active_path_count,
                "active_tree_sha256": scan.active_tree_sha256,
                "candidate_first_add_count": len(scan.candidates),
                "first_parent_commit_count": scan.first_parent_commit_count,
                "first_parent_chain_sha256": scan.first_parent_chain_sha256,
                "mutation_counts": scan.mutation_counts,
                "raw_path_delta_sha256": scan.raw_path_delta_sha256,
                "transition_count": scan.transition_count,
            },
            "mechanism_preregistration_authorized": False,
            "outcome_boundary": _outcome_boundary(),
            "retry_or_resume_authorized": False,
            "source_audit_authoritative": False,
            "support": dict(support),
        }
    )


def run_fixture_audit(
    *,
    source_repo: Path,
    pinned_commit: str,
    paths: AuditPaths,
    verifier_commit: str,
    runner_blob: str,
) -> dict[str, Any]:
    """Run an offline synthetic local-Git fixture with no source authority."""

    absolute = validate_fixture_paths(paths)
    source_repo = validate_fixture_source_repo(source_repo)
    guard = reserve_attempt(
        paths=absolute,
        verifier_commit=verifier_commit,
        runner_blob=runner_blob,
    )
    guard.append("fixture_reserved", {"network_authorized": False})
    try:
        scan = collect_first_add_candidates(source_repo, pinned_commit)
        corpus = parse_candidate_corpus(source_repo, scan.candidates)
        support = evaluate_support(
            corpus.selected_events,
            source_event_count=corpus.candidate_count,
            prewindow_count=corpus.prewindow_count,
            postwindow_count=corpus.postwindow_count,
            opaque_postwindow_count=corpus.opaque_postwindow_count,
            candidate_raw_hashes_sha256=corpus.candidate_raw_hashes_sha256,
        )
        report = _fixture_report(
            verifier_commit=verifier_commit,
            runner_blob=runner_blob,
            guard=guard,
            scan=scan,
            support=support,
        )
    except BaseException as exc:
        report = _finalize_report_hash(
            {
                "bindings": _bindings(
                    verifier_commit=verifier_commit,
                    runner_blob=runner_blob,
                    guard=guard,
                ),
                "created_at_utc": canonical_utc(datetime.now(timezone.utc)),
                "decision": "TERMINAL_REJECT",
                "execution_authority": "offline_fixture",
                "failure": {"exception_class": type(exc).__name__},
                "mechanism_preregistration_authorized": False,
                "outcome_boundary": _outcome_boundary(),
                "retry_or_resume_authorized": False,
                "source_audit_authoritative": False,
            }
        )
    _atomic_publish(absolute.report, canonical_json_bytes(report))
    return report


def _production_report(
    *,
    verifier_commit: str,
    runner_blob: str,
    guard: AttemptGuard,
    disk_used_before: int,
    disk_free_before: int,
    repository_identity: Mapping[str, Any],
    current_tree: Mapping[str, Any],
    scan: HistoryScan,
    replay: Mapping[str, Any],
    support: Mapping[str, Any],
    transport: Mapping[str, Any],
) -> dict[str, Any]:
    decision = str(support["decision"])
    return _finalize_report_hash(
        {
            "bindings": _bindings(
                verifier_commit=verifier_commit,
                runner_blob=runner_blob,
                guard=guard,
            ),
            "created_at_utc": canonical_utc(datetime.now(timezone.utc)),
            "decision": decision,
            "deterministic_source_fingerprint_sha256": (
                deterministic_source_fingerprint(
                    repository_identity=repository_identity,
                    current_tree=current_tree,
                    scan=scan,
                    replay=replay,
                    support=support,
                    transport=transport,
                )
            ),
            "disk": {
                "free_bytes_before": disk_free_before,
                "free_floor_bytes": DISK_FREE_FLOOR,
                "used_bytes_before": disk_used_before,
                "used_limit_bytes": DISK_USED_LIMIT,
            },
            "execution_authority": "production_one_shot",
            "history": {
                "active_path_count": scan.active_path_count,
                "active_tree_sha256": scan.active_tree_sha256,
                "candidate_first_add_count": len(scan.candidates),
                "first_parent_commit_count": scan.first_parent_commit_count,
                "first_parent_chain_sha256": scan.first_parent_chain_sha256,
                "mutation_counts": scan.mutation_counts,
                "raw_path_delta_sha256": scan.raw_path_delta_sha256,
                "transition_count": scan.transition_count,
            },
            "mechanism_preregistration_authorized": (
                decision == "SOURCE_SUPPORT_PASS"
            ),
            "outcome_boundary": _outcome_boundary(),
            "repository_identity": dict(repository_identity),
            "replay": dict(replay),
            "retry_or_resume_authorized": False,
            "source_audit_authoritative": True,
            "support": dict(support),
            "transport": {
                **dict(transport),
                "current_tree": dict(current_tree),
            },
        }
    )


def _reject_report(
    *,
    verifier_commit: str,
    runner_blob: str,
    guard: AttemptGuard,
    disk_used_before: int,
    disk_free_before: int,
    stage: str,
    exception: BaseException,
) -> dict[str, Any]:
    return _finalize_report_hash(
        {
            "bindings": _bindings(
                verifier_commit=verifier_commit,
                runner_blob=runner_blob,
                guard=guard,
            ),
            "created_at_utc": canonical_utc(datetime.now(timezone.utc)),
            "decision": "TERMINAL_REJECT",
            "disk": {
                "free_bytes_before": disk_free_before,
                "free_floor_bytes": DISK_FREE_FLOOR,
                "used_bytes_before": disk_used_before,
                "used_limit_bytes": DISK_USED_LIMIT,
            },
            "execution_authority": "production_one_shot",
            "failure": {
                "exception_class": type(exception).__name__,
                "stage": stage,
            },
            "mechanism_preregistration_authorized": False,
            "outcome_boundary": _outcome_boundary(),
            "retry_or_resume_authorized": False,
            "source_audit_authoritative": True,
        }
    )


def _run_bound_source_audit(
    execution_mode: object,
    isolated_authority: str | None,
) -> dict[str, Any]:
    if execution_mode is not _PRODUCTION_EXECUTION:
        raise ProtocolError("audit execution mode is not authorized")
    if (
        sys.flags.isolated != 1
        or __name__ != "__main__"
        or __spec__ is not None
        or not isinstance(isolated_authority, str)
        or HEX64.fullmatch(isolated_authority) is None
        or os.environ.get("GHAD_ISOLATED_CHILD_TOKEN") != isolated_authority
        or os.environ.get("GHAD_ISOLATED_PARENT_PID") != str(os.getppid())
    ):
        raise ProtocolError("production requires the isolated CLI child")

    literal_paths = AuditPaths(
        sentinel=Path(
            "results/.github_advisory_first_add_source_2026-07-24.started"
        ),
        manifest=Path(
            "results/.github_advisory_first_add_source_2026-07-24.manifest.ndjson"
        ),
        raw_dir=Path(
            "results/.github_advisory_first_add_source_2026-07-24.raw"
        ),
        report=Path(
            "results/github_advisory_first_add_source_2026-07-24.json"
        ),
    )
    absolute = _validate_paths(literal_paths)
    assert_no_leak_environment()
    assert_no_leak_runtime()
    verifier_commit, runner_blob = assert_protocol_committed()
    disk_used_before, disk_free_before = assert_disk_guard()
    guard = reserve_attempt(
        paths=absolute,
        verifier_commit=verifier_commit,
        runner_blob=runner_blob,
    )
    guard.append(
        "attempt_reserved",
        {
            "exact_remote": OFFICIAL_REMOTE,
            "frozen_commit": FROZEN_COMMIT,
            "network_requests_completed": 0,
        },
    )
    stage = "transport"
    try:
        home = absolute.raw_dir / "home"
        home.mkdir(mode=0o700)
        (home / "xdg").mkdir(mode=0o700)
        source_repo = absolute.raw_dir / "repository.git"
        git_environment = sealed_git_environment(home)
        git_environment["GIT_NO_LAZY_FETCH"] = "1"
        init_command = [
            str(GIT_EXECUTABLE),
            "-c",
            "init.templateDir=",
            "init",
            "--bare",
            str(source_repo),
        ]
        _run_sealed_command(init_command, environment=git_environment)
        _run_sealed_command(
            [
                str(GIT_EXECUTABLE),
                "-C",
                str(source_repo),
                "remote",
                "add",
                "origin",
                OFFICIAL_REMOTE,
            ],
            environment=git_environment,
        )
        for key, value in (
            ("remote.origin.promisor", "true"),
            ("remote.origin.partialclonefilter", "blob:none"),
            ("core.hooksPath", os.devnull),
        ):
            _run_sealed_command(
                [
                    str(GIT_EXECUTABLE),
                    "-C",
                    str(source_repo),
                    "config",
                    key,
                    value,
                ],
                environment=git_environment,
            )

        assert_disk_guard()
        assert_object_store_guard(source_repo / "objects")
        guard.append(
            "commit_tree_fetch_intent",
            {
                "filter": "blob:none",
                "object": FROZEN_COMMIT,
                "remote": OFFICIAL_REMOTE,
                "retry": False,
            },
        )
        commit_fetch = _run_sealed_command(
            fetch_objects_command(source_repo, allow_lazy_fetch=False),
            environment=git_environment,
            input_bytes=FROZEN_COMMIT.encode("ascii") + b"\n",
        )
        guard.append(
            "commit_tree_fetch_result",
            {
                "stderr_sha256": sha256_bytes(commit_fetch.stderr),
                "stdout_sha256": sha256_bytes(commit_fetch.stdout),
            },
        )
        _run_sealed_command(
            [
                str(GIT_EXECUTABLE),
                "-C",
                str(source_repo),
                "update-ref",
                "refs/source/frozen",
                FROZEN_COMMIT,
            ],
            environment=git_environment,
        )
        repository_identity = _verify_repository_boundary(
            source_repo, git_environment
        )
        inventory_before = _object_inventory(
            source_repo, environment=git_environment
        )
        if inventory_before["blob_oids"]:
            raise TransportError("commit/tree phase materialized advisory blobs")
        current_tree = _verify_current_tree(
            source_repo,
            FROZEN_COMMIT,
            environment=git_environment,
        )

        stage = "history"
        scan = collect_first_add_candidates(source_repo, FROZEN_COMMIT)
        if not scan.candidates:
            raise HistoryError("frozen history has no first-add candidates")
        if (
            scan.active_path_count != current_tree["regular_blob_count"]
            or scan.active_tree_sha256 != current_tree["tree_listing_sha256"]
        ):
            raise HistoryError("first-parent replay differs from frozen current tree")
        candidate_manifest_path = absolute.raw_dir / "candidate_manifest.json"
        candidate_manifest_bytes, candidate_manifest_sha = (
            _write_candidate_manifest(candidate_manifest_path, scan)
        )
        guard.append(
            "candidate_derivation",
            {
                "candidate_count": len(scan.candidates),
                "candidate_manifest_bytes": candidate_manifest_bytes,
                "candidate_manifest_sha256": candidate_manifest_sha,
                "first_parent_chain_sha256": scan.first_parent_chain_sha256,
                "raw_path_delta_sha256": scan.raw_path_delta_sha256,
            },
        )

        stage = "materialization"
        assert_disk_guard()
        object_store_before_blobs = assert_object_store_guard(
            source_repo / "objects"
        )
        unique_blob_oids = sorted(
            {candidate.blob_oid for candidate in scan.candidates}
        )
        candidate_oid_ledger = canonical_json_bytes(unique_blob_oids)
        blob_environment = dict(git_environment)
        blob_environment["GIT_NO_LAZY_FETCH"] = "0"
        guard.append(
            "candidate_blob_materialization_plan",
            {
                "candidate_blob_count": len(unique_blob_oids),
                "candidate_blob_oids_sha256": sha256_bytes(
                    candidate_oid_ledger
                ),
                "chunk_maximum": CANDIDATE_FETCH_CHUNK_MAX,
                "manifest_growth_reserve_bytes": MANIFEST_GROWTH_RESERVE,
                "single_blob_worst_case_bytes": SINGLE_BLOB_CAP,
            },
        )
        fetched_oids: set[str] = set()
        remaining_oids = list(unique_blob_oids)
        chunk_count = 0
        inventory_after = inventory_before
        while remaining_oids:
            assert_disk_guard()
            assert_object_store_guard(source_repo / "objects")
            chunk_size = plan_candidate_fetch_chunk(
                remaining_count=len(remaining_oids),
                materialized_raw_bytes=inventory_after[
                    "blob_declared_bytes"
                ],
                candidate_manifest_bytes=candidate_manifest_bytes,
                retrieval_manifest_bytes=absolute.manifest.stat().st_size,
            )
            chunk = remaining_oids[:chunk_size]
            chunk_ledger = canonical_json_bytes(chunk)
            chunk_count += 1
            guard.append(
                "candidate_blob_fetch_intent",
                {
                    "chunk_count": len(chunk),
                    "chunk_oids_sha256": sha256_bytes(chunk_ledger),
                    "chunk_ordinal": chunk_count,
                    "filter": "blob:none",
                    "retry": False,
                },
            )
            blob_fetch = _run_sealed_command(
                fetch_objects_command(source_repo, allow_lazy_fetch=True),
                environment=blob_environment,
                input_bytes=b"".join(
                    oid.encode("ascii") + b"\n" for oid in chunk
                ),
            )
            guard.append(
                "candidate_blob_fetch_result",
                {
                    "chunk_ordinal": chunk_count,
                    "stderr_sha256": sha256_bytes(blob_fetch.stderr),
                    "stdout_sha256": sha256_bytes(blob_fetch.stdout),
                },
            )
            fetched_oids.update(chunk)
            remaining_oids = remaining_oids[chunk_size:]
            assert_disk_guard()
            assert_object_store_guard(source_repo / "objects")
            inventory_after = _object_inventory(
                source_repo, environment=git_environment
            )
            if inventory_after["blob_oids"] != fetched_oids:
                raise TransportError(
                    "materialized blob inventory differs from authorized prefix"
                )
            if any(
                size > SINGLE_BLOB_CAP
                for size in inventory_after["blob_sizes"].values()
            ):
                raise DiskGuardError(
                    "materialized advisory blob exceeds single-blob cap"
                )
            materialized_bytes = (
                inventory_after["blob_declared_bytes"]
                + candidate_manifest_bytes
                + absolute.manifest.stat().st_size
            )
            if materialized_bytes >= CANDIDATE_MATERIAL_CAP:
                raise DiskGuardError(
                    "candidate material exceeds the frozen cap"
                )
        git_environment["GIT_NO_LAZY_FETCH"] = "1"
        object_store_after_blobs = assert_object_store_guard(
            source_repo / "objects"
        )

        stage = "structure"
        corpus = parse_candidate_corpus(source_repo, scan.candidates)
        events = corpus.selected_events
        candidate_raw_bytes = corpus.candidate_raw_bytes
        material_bytes = (
            candidate_raw_bytes
            + candidate_manifest_bytes
            + absolute.manifest.stat().st_size
        )
        if material_bytes >= CANDIDATE_MATERIAL_CAP:
            raise DiskGuardError("candidate material exceeds the frozen cap")
        if candidate_raw_bytes != inventory_after["blob_declared_bytes"]:
            raise StructureError("parsed candidate bytes differ from Git inventory")

        stage = "replay"
        replay_scan = collect_first_add_candidates(source_repo, FROZEN_COMMIT)
        replay_corpus = parse_candidate_corpus(
            source_repo, replay_scan.candidates
        )
        replay_a = _corpus_replay_fingerprint(scan, corpus)
        replay_b = _corpus_replay_fingerprint(replay_scan, replay_corpus)
        if replay_a != replay_b or corpus != replay_corpus:
            raise HistoryError("sealed source replay disagrees")
        replay = {"pass_a": replay_a, "pass_b": replay_b, "passed": True}

        stage = "support"
        support = evaluate_support(
            events,
            source_event_count=corpus.candidate_count,
            prewindow_count=corpus.prewindow_count,
            postwindow_count=corpus.postwindow_count,
            opaque_postwindow_count=corpus.opaque_postwindow_count,
            candidate_raw_hashes_sha256=corpus.candidate_raw_hashes_sha256,
        )
        guard.append(
            "source_support_result",
            {
                "all_gates_passed": support["all_gates_passed"],
                "decision": support["decision"],
                "selected_event_count": support["aggregate"][
                    "selected_event_count"
                ],
                "support_sha256": sha256_bytes(
                    canonical_json_bytes(support, newline=False)
                ),
            },
        )
        material_bytes = (
            candidate_raw_bytes
            + candidate_manifest_bytes
            + absolute.manifest.stat().st_size
        )
        if material_bytes >= CANDIDATE_MATERIAL_CAP:
            raise DiskGuardError(
                "final candidate material exceeds the frozen cap"
            )
        transport = {
            "candidate_blob_count": len(unique_blob_oids),
            "candidate_blob_oids_sha256": sha256_bytes(candidate_oid_ledger),
            "candidate_manifest_bytes": candidate_manifest_bytes,
            "candidate_manifest_sha256": candidate_manifest_sha,
            "candidate_material_bytes": material_bytes,
            "candidate_raw_blob_bytes": candidate_raw_bytes,
            "commit_tree_inventory": {
                **inventory_before,
                "blob_oids": [],
                "blob_sizes": None,
            },
            "git_object_store_bytes_after_blobs": object_store_after_blobs,
            "git_object_store_bytes_before_blobs": object_store_before_blobs,
            "materialized_inventory": {
                **inventory_after,
                "blob_oids": None,
                "blob_sizes": None,
                "blob_oids_sha256": sha256_bytes(candidate_oid_ledger),
            },
            "candidate_blob_fetch_chunks": chunk_count,
            "network_fetch_count": 1 + chunk_count,
            "retry_or_resume_used": False,
        }
        report = _production_report(
            verifier_commit=verifier_commit,
            runner_blob=runner_blob,
            guard=guard,
            disk_used_before=disk_used_before,
            disk_free_before=disk_free_before,
            repository_identity=repository_identity,
            current_tree=current_tree,
            scan=scan,
            replay=replay,
            support=support,
            transport=transport,
        )
    except BaseException as exc:
        try:
            guard.append(
                "terminal_reject",
                {
                    "exception_class": type(exc).__name__,
                    "stage": stage,
                },
            )
        except BaseException:
            pass
        report = _reject_report(
            verifier_commit=verifier_commit,
            runner_blob=runner_blob,
            guard=guard,
            disk_used_before=disk_used_before,
            disk_free_before=disk_free_before,
            stage=stage,
            exception=exc,
        )
    _atomic_publish(absolute.report, canonical_json_bytes(report))
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--_isolated-child-token",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.report != DEFAULT_REPORT:
        print("GHAD-GRFA source audit: PRECHECK_REJECT")
        return 2
    if args._isolated_child_token is None:
        if __name__ != "__main__" or __spec__ is not None:
            print("GHAD-GRFA source audit: PRECHECK_REJECT")
            return 2
        try:
            assert_no_leak_environment()
        except ProtocolError:
            print("GHAD-GRFA source audit: PRECHECK_REJECT")
            return 2
        token = secrets.token_hex(32)
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "XDG_CONFIG_HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "GHAD_ISOLATED_CHILD_TOKEN": token,
            "GHAD_ISOLATED_PARENT_PID": str(os.getpid()),
        }
        command = [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--_isolated-child-token",
            token,
        ]
        try:
            completed = subprocess.run(
                command,
                env=environment,
                cwd=REPOSITORY_ROOT,
                check=False,
                close_fds=True,
            )
        except OSError:
            print("GHAD-GRFA source audit: PRECHECK_REJECT")
            return 2
        return completed.returncode
    try:
        report = _run_bound_source_audit(
            _PRODUCTION_EXECUTION,
            args._isolated_child_token,
        )
    except BaseException:
        consumed = repository_path(DEFAULT_SENTINEL).exists()
        print(
            "GHAD-GRFA source audit: "
            + ("TERMINAL_REJECT" if consumed else "PRECHECK_REJECT")
        )
        return 2
    if (
        report["decision"] != "SOURCE_SUPPORT_PASS"
        or report["execution_authority"] != "production_one_shot"
        or report["source_audit_authoritative"] is not True
        or report["mechanism_preregistration_authorized"] is not True
    ):
        print("GHAD-GRFA source audit: TERMINAL_REJECT")
        return 1
    print("GHAD-GRFA source audit: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
