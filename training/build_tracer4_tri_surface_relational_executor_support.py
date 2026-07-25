"""Build TRACER-4H source/token support without opening outcomes.

This runner is intentionally source-only.  It validates the committed TRACER
boundary and preregistration artifact, projects only frozen physical columns
into deterministic pre-2024 gzip cuts, builds four-hour relation tokens from
those cuts, and writes support-only manifests/reports.  It must not import or
open execution OHLC, funding, future return, reward, model, or outcome data.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from training import preregister_tracer4_tri_surface_relational_executor as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = "training/build_tracer4_tri_surface_relational_executor_support.py"
TEST_PATH = "tests/test_build_tracer4_tri_surface_relational_executor_support.py"
CONTRACT_PATH = "docs/tracer4-source-support-implementation-contract-2026-07-25.md"
CONTRACT_SHA256 = "b39f88197e74e6e95404b4468315be732e8af53e2e37007d6cdb0dc9b40d869d"
PREREGISTRATION_SOURCE_PATH = (
    "training/preregister_tracer4_tri_surface_relational_executor.py"
)
PREREGISTRATION_TEST_PATH = (
    "tests/test_preregister_tracer4_tri_surface_relational_executor.py"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = "5299fc6f803b4faf7cf10655050d71d0f918b0899c6ef8f1a18c5028579a1c8b"
PREREGISTRATION_MANIFEST_HASH = "67f06047661336ebad4da19f0dd65578520a28f1d1b4b468c6300b4aa4d54318"
PROTOCOL_VERSION = "tracer4_tri_surface_relational_executor_support_v1"
DEFAULT_CUT_MANIFEST = Path(prereg.SOURCE_CUT_MANIFEST)
DEFAULT_TOKEN_OUTPUT = Path(prereg.TOKEN_OUTPUT)
DEFAULT_REPORT = Path(prereg.SUPPORT_OUTPUT)
SOURCE_END = pd.Timestamp(prereg.SOURCE_END_EXCLUSIVE)
SOURCE_START = pd.Timestamp(prereg.SOURCE_START)
ZERO_OUTCOME_COUNTERS = {
    "execution_market_rows_opened": 0,
    "funding_rows_opened": 0,
    "future_return_rows_opened": 0,
    "reward_rows_built": 0,
    "model_rows_built": 0,
    "pnl_values_computed": 0,
    "cagr_values_computed": 0,
    "mdd_values_computed": 0,
    "post_2023_numeric_source_rows_opened": 0,
}
NUMERIC_COLUMNS = {
    "leadership": tuple(
        col for col in prereg.LEADERSHIP_ALLOWLIST
        if col not in {"date", "feature_available_time_utc", "source_complete", "cross_venue_feature_valid"}
    ),
    "aggtrade": tuple(
        col for col in prereg.AGGTRADE_ALLOWLIST
        if col not in {"date"}
    ),
    "premium": tuple(
        col for col in prereg.PREMIUM_ALLOWLIST
        if col not in {"date", "source_close_time", "feature_available_time", "source_valid"}
    ),
}
NONNEGATIVE_COLUMNS = {
    "leadership": ("spot_quote_notional", "um_quote_notional"),
    "aggtrade": (
        "agg_trade_count", "quote_notional", "event_notional_hhi",
        "normalized_effective_event_count", "sign_flip_rate",
        "max_same_sign_run_share", "interarrival_mean_ms",
    ),
    "premium": (),
}
TOKEN_FILE_COLUMNS = (
    "boundary",
    "window_start",
    "window_end",
    "premium_cutoff",
    "decision_time",
    "execution_time",
    "core_source_ready",
    "line_ready",
    "sequence_ready",
    *prereg.TOKEN_COLUMNS,
    "canonical_line",
    "sequence_signature",
)
RANKED_PRIMITIVES = (
    "sponsor_score",
    "participation_hhi",
    "effective_participation",
    "flow_flip",
    "flow_run",
    "arrival_burst",
    "arrival_wait",
    "premium_range",
)
RANK_ORDER = {"LOW": 0, "MID": 1, "HIGH": 2}
PROTOCOL_PATHS = (
    prereg.BOUNDARY_DOCUMENT,
    PREREGISTRATION_SOURCE_PATH,
    PREREGISTRATION_TEST_PATH,
    str(PREREGISTRATION),
    CONTRACT_PATH,
    SCRIPT_PATH,
    TEST_PATH,
)


@dataclass(frozen=True)
class PrimitiveState:
    boundary: pd.Timestamp
    valid: bool
    invalid_reasons: tuple[str, ...]
    primitives: Mapping[str, float]
    ranks: Mapping[str, str]
    tokens: OrderedDict[str, str] | None
    line: str
    rank_ready: bool
    sequence_ready: bool


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("TRACER support paths must be repository-relative")
    return REPOSITORY_ROOT / candidate


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _sha256_path(repository_path(path))


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def canonical_ts(value: pd.Timestamp) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(("git", *args), cwd=REPOSITORY_ROOT, check=False, capture_output=True, text=True)


def assert_protocol_committed() -> str:
    """Guard artifact generation behind a committed, clean protocol boundary."""
    tracked = _git("ls-files", "--error-unmatch", "--", *PROTOCOL_PATHS)
    if tracked.returncode != 0:
        raise RuntimeError("TRACER support protocol/preregistration/boundary is not committed")
    dirty = _git("diff", "--quiet", "HEAD", "--", *PROTOCOL_PATHS)
    if dirty.returncode != 0:
        raise RuntimeError("TRACER support protocol/preregistration/boundary differs from HEAD")
    staged = _git("diff", "--cached", "--quiet")
    if staged.returncode != 0:
        raise RuntimeError("TRACER support index is not clean")
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0 or len(head.stdout.strip()) != 40:
        raise RuntimeError("TRACER support HEAD is unavailable")
    prereg.assert_boundary_committed()
    if sha256_file(CONTRACT_PATH) != CONTRACT_SHA256:
        raise RuntimeError("TRACER source-support implementation contract hash drift")
    return head.stdout.strip()


def load_preregistration(path: str | Path = PREREGISTRATION) -> dict[str, Any]:
    if sha256_file(path) != PREREGISTRATION_SHA256:
        raise RuntimeError("TRACER preregistration artifact hash mismatch")
    payload = json.loads(repository_path(path).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("TRACER preregistration manifest hash mismatch")
    if payload.get("outcome_boundary") != ZERO_OUTCOME_COUNTERS:
        raise RuntimeError("TRACER preregistration opened an outcome boundary")
    return payload


def read_header(path: str | Path) -> tuple[str, ...]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(repository_path(path), "rt", encoding="utf-8", newline="") as handle:  # type: ignore[arg-type]
        line = handle.readline()
    return tuple(next(csv.reader([line.rstrip("\n")])))


def header_sha256(path: str | Path) -> str:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(repository_path(path), "rb") as handle:  # type: ignore[arg-type]
        return sha256_bytes(handle.readline())


def gzip_mtime(path: str | Path) -> int:
    with repository_path(path).open("rb") as handle:
        header = handle.read(10)
    if len(header) != 10 or header[:2] != b"\x1f\x8b":
        raise RuntimeError(f"TRACER artifact is not gzip: {path}")
    return int.from_bytes(header[4:8], byteorder="little", signed=False)


def validate_source_contracts() -> dict[str, Any]:
    specs = {
        "leadership": (
            prereg.LEADERSHIP_SOURCE, prereg.LEADERSHIP_SOURCE_SHA256,
            prereg.LEADERSHIP_PHYSICAL_HEADER, prereg.LEADERSHIP_HEADER_SHA256,
            prereg.LEADERSHIP_MANIFEST, prereg.LEADERSHIP_MANIFEST_SHA256,
        ),
        "aggtrade": (
            prereg.AGGTRADE_SOURCE, prereg.AGGTRADE_SOURCE_SHA256,
            prereg.AGGTRADE_PHYSICAL_HEADER, prereg.AGGTRADE_HEADER_SHA256,
            prereg.AGGTRADE_MANIFEST, prereg.AGGTRADE_MANIFEST_SHA256,
        ),
        "premium": (
            prereg.PREMIUM_SOURCE, prereg.PREMIUM_SOURCE_SHA256,
            prereg.PREMIUM_PHYSICAL_HEADER, prereg.PREMIUM_HEADER_SHA256,
            None, None,
        ),
    }
    audit: dict[str, Any] = {}
    for name, (path, file_hash, header, hdr_hash, manifest, manifest_hash) in specs.items():
        if sha256_file(path) != file_hash:
            raise RuntimeError(f"TRACER {name} source file hash mismatch")
        actual_header = read_header(path)
        if actual_header != tuple(header):
            raise RuntimeError(f"TRACER {name} physical header mismatch")
        if header_sha256(path) != hdr_hash:
            raise RuntimeError(f"TRACER {name} physical header hash mismatch")
        entry = {"path": path, "sha256": file_hash, "header_sha256": hdr_hash}
        if manifest is not None:
            if sha256_file(manifest) != manifest_hash:
                raise RuntimeError(f"TRACER {name} source manifest hash mismatch")
            entry["manifest"] = manifest
            entry["manifest_sha256"] = manifest_hash
        audit[name] = entry
    return audit


def deterministic_gzip_bytes(text: str) -> bytes:
    """Pure deterministic gzip encoder: UTF-8, empty filename, mtime=0."""
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    return raw.getvalue()


def write_once_bytes(path: str | Path, payload: bytes) -> str:
    candidate = Path(path)
    target = candidate if candidate.is_absolute() else repository_path(candidate)
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeError(f"TRACER write-once artifact drift: {path}")
        return sha256_bytes(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return sha256_bytes(payload)


def write_once_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write canonical JSON once; existing byte-identical files are reused."""
    return write_once_bytes(path, canonical_json_bytes(payload))


def write_once_gzip_csv(path: str | Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Write deterministic gzip CSV once from supplied rows."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(list(header))
    writer.writerows(rows)
    return write_once_bytes(path, deterministic_gzip_bytes(output.getvalue()))


def install_write_once_file(path: str | Path, temporary: Path) -> tuple[str, int]:
    """Install an already-fsync'd temporary file without permitting drift."""
    candidate = Path(path)
    target = candidate if candidate.is_absolute() else repository_path(candidate)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_hash = hashlib.sha256()
    size = 0
    with temporary.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            temporary_hash.update(chunk)
            size += len(chunk)
    digest = temporary_hash.hexdigest()
    if target.exists():
        if target.stat().st_size != size or _sha256_path(target) != digest:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"TRACER write-once artifact drift: {path}")
        temporary.unlink()
        return digest, size
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return digest, size


def _finite_cell(raw: str, *, name: str, field: str) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"TRACER {name} non-numeric cell: {field}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"TRACER {name} non-finite numeric cell: {field}")
    return value


def _binary_cell(raw: str, *, name: str, field: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true"}:
        return True
    if value in {"0", "false"}:
        return False
    raise RuntimeError(f"TRACER {name} non-binary flag: {field}")


def _integer_cell(raw: str, *, name: str, field: str) -> int:
    value = str(raw).strip()
    try:
        integer = int(value)
    except ValueError as exc:
        raise RuntimeError(f"TRACER {name} non-integer cell: {field}") from exc
    if str(integer) != value and value not in {f"+{integer}", f"-{abs(integer)}"}:
        raise RuntimeError(f"TRACER {name} non-canonical integer cell: {field}")
    return integer


def validate_projected_row(
    name: str,
    row: Mapping[str, str],
    *,
    date_value: pd.Timestamp,
) -> None:
    """Validate only the frozen projected cells after the cutoff decision."""

    if name == "leadership":
        available = parse_utc(row["feature_available_time_utc"])
        if available != date_value + pd.Timedelta(minutes=5):
            raise RuntimeError("TRACER leadership availability clock mismatch")
        source_complete = _binary_cell(
            row["source_complete"], name=name, field="source_complete"
        )
        feature_valid = _binary_cell(
            row["cross_venue_feature_valid"],
            name=name,
            field="cross_venue_feature_valid",
        )
        if not (source_complete and feature_valid):
            for field in NUMERIC_COLUMNS[name]:
                raw = str(row[field]).strip()
                if not raw or raw.lower() in {"nan", "na", "null"}:
                    continue
                _finite_cell(raw, name=name, field=field)
            return
        values = {
            field: _finite_cell(row[field], name=name, field=field)
            for field in NUMERIC_COLUMNS[name]
        }
        for field in NONNEGATIVE_COLUMNS[name]:
            if values[field] < 0.0:
                raise RuntimeError(
                    f"TRACER leadership negative nonnegative cell: {field}"
                )
        for prefix in ("spot", "um"):
            if abs(values[f"{prefix}_signed_quote_notional"]) > values[
                f"{prefix}_quote_notional"
            ]:
                raise RuntimeError(
                    f"TRACER leadership {prefix} signed notional exceeds quote"
                )
        return

    if name == "aggtrade":
        first_ms = _integer_cell(
            row["first_transact_time_ms"],
            name=name,
            field="first_transact_time_ms",
        )
        last_ms = _integer_cell(
            row["last_transact_time_ms"],
            name=name,
            field="last_transact_time_ms",
        )
        first = pd.Timestamp(first_ms, unit="ms", tz="UTC")
        last = pd.Timestamp(last_ms, unit="ms", tz="UTC")
        if not (
            date_value
            <= first
            <= last
            < date_value + pd.Timedelta(minutes=5)
        ):
            raise RuntimeError("TRACER aggtrade transaction clock mismatch")
        count = _integer_cell(
            row["agg_trade_count"], name=name, field="agg_trade_count"
        )
        if count <= 0:
            raise RuntimeError("TRACER aggtrade count must be positive")
        values = {
            field: _finite_cell(row[field], name=name, field=field)
            for field in NUMERIC_COLUMNS[name]
            if field
            not in {
                "first_transact_time_ms",
                "last_transact_time_ms",
                "agg_trade_count",
            }
        }
        for field in NONNEGATIVE_COLUMNS[name]:
            if field == "agg_trade_count":
                continue
            if values[field] < 0.0:
                raise RuntimeError(
                    f"TRACER aggtrade negative nonnegative cell: {field}"
                )
        if abs(values["signed_quote_notional"]) > values["quote_notional"]:
            raise RuntimeError(
                "TRACER aggtrade signed notional exceeds quote"
            )
        for field in (
            "event_notional_hhi",
            "sign_flip_rate",
            "max_same_sign_run_share",
        ):
            if not 0.0 <= values[field] <= 1.0:
                raise RuntimeError(
                    f"TRACER aggtrade bounded ratio out of range: {field}"
                )
        if values["normalized_effective_event_count"] <= 0.0:
            raise RuntimeError(
                "TRACER aggtrade effective participation must be positive"
            )
        return

    if name == "premium":
        if parse_utc(row["source_close_time"]) != (
            date_value + pd.Timedelta(seconds=59, milliseconds=999)
        ):
            raise RuntimeError("TRACER premium close clock mismatch")
        if parse_utc(row["feature_available_time"]) != (
            date_value + pd.Timedelta(seconds=61)
        ):
            raise RuntimeError("TRACER premium availability clock mismatch")
        valid = _binary_cell(
            row["source_valid"], name=name, field="source_valid"
        )
        ohlc_fields = (
            "premium_open",
            "premium_high",
            "premium_low",
            "premium_close",
        )
        if not valid:
            for field in ohlc_fields:
                raw = str(row[field]).strip()
                if not raw or raw.lower() in {"nan", "na", "null"}:
                    continue
                raise RuntimeError(
                    "TRACER invalid premium row exposes usable or malformed OHLC"
                )
            return
        values = {
            field: _finite_cell(row[field], name=name, field=field)
            for field in ohlc_fields
        }
        if values["premium_high"] < max(
            values["premium_open"], values["premium_close"]
        ):
            raise RuntimeError("TRACER premium high envelope mismatch")
        if values["premium_low"] > min(
            values["premium_open"], values["premium_close"]
        ):
            raise RuntimeError("TRACER premium low envelope mismatch")
        return

    raise ValueError(f"unknown TRACER source: {name}")


def _project_rows_to_writer(
    *,
    name: str,
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    allowlist: Sequence[str],
    writer: Any,
    stop_at: pd.Timestamp = SOURCE_END,
) -> dict[str, Any]:
    """Project rows to a CSV writer without converting a cutoff row."""
    header = tuple(header)
    allowlist = tuple(allowlist)
    if len(set(header)) != len(header):
        raise RuntimeError(f"TRACER {name} physical header has duplicate fields")
    if len(set(allowlist)) != len(allowlist):
        raise RuntimeError(f"TRACER {name} allowlist has duplicate fields")
    missing = [col for col in allowlist if col not in header]
    if missing:
        raise RuntimeError(f"TRACER {name} allowlist fields missing: {missing}")
    positions = [header.index(col) for col in allowlist]
    date_index = header.index("date")
    writer.writerow(allowlist)
    rows_read = rows_written = post_cut_rows_seen = 0
    first_date = last_date = stopped_at = None
    previous_date: pd.Timestamp | None = None
    for raw_row in rows:
        row = tuple(raw_row)
        rows_read += 1
        if len(row) != len(header):
            raise RuntimeError(f"TRACER {name} physical field count mismatch")
        date_value = parse_utc(row[date_index])
        if previous_date is not None and date_value <= previous_date:
            kind = "duplicate" if date_value == previous_date else "non-monotone"
            raise RuntimeError(f"TRACER {name} {kind} source date")
        if date_value >= stop_at:
            stopped_at = canonical_ts(date_value)
            post_cut_rows_seen += 1
            break
        if date_value < SOURCE_START:
            raise RuntimeError(f"TRACER {name} source date precedes frozen start")
        projected = {col: row[pos] for col, pos in zip(allowlist, positions, strict=True)}
        validate_projected_row(name, projected, date_value=date_value)
        writer.writerow([projected[col] for col in allowlist])
        rows_written += 1
        first_date = first_date or canonical_ts(date_value)
        last_date = canonical_ts(date_value)
        previous_date = date_value
    return {
        "source_rows_read": rows_read,
        "cut_rows_written": rows_written,
        "first_date": first_date,
        "last_date": last_date,
        "stopped_at": stopped_at,
        "post_cut_rows_seen_before_stop": post_cut_rows_seen,
        "post_2023_numeric_source_rows_opened": 0,
        "forbidden_columns_projected": 0,
    }


def deterministic_project_rows(
    *,
    name: str,
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    allowlist: Sequence[str],
    stop_at: pd.Timestamp = SOURCE_END,
) -> tuple[str, dict[str, Any]]:
    """Pure in-memory projection used by adversarial protocol tests."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    audit = _project_rows_to_writer(
        name=name,
        header=header,
        rows=rows,
        allowlist=allowlist,
        writer=writer,
        stop_at=stop_at,
    )
    return output.getvalue(), audit


def project_source_cut(name: str, source: str, allowlist: Sequence[str], output: str) -> dict[str, Any]:
    """Stream exact physical projection into a write-once deterministic gzip cut."""
    source_header = read_header(source)
    target = repository_path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, delete=False
        ) as raw_output:
            temporary_path = Path(raw_output.name)
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_output, mtime=0
            ) as compressed:
                text_output = io.TextIOWrapper(
                    compressed,
                    encoding="utf-8",
                    newline="",
                    write_through=True,
                )
                try:
                    writer = csv.writer(text_output, lineterminator="\n")
                    with gzip.open(
                        repository_path(source),
                        "rt",
                        encoding="utf-8",
                        newline="",
                    ) as src:
                        reader = csv.reader(src)
                        actual_header = tuple(next(reader))
                        if actual_header != source_header:
                            raise RuntimeError(
                                f"TRACER {name} physical header mismatch"
                            )
                        audit = _project_rows_to_writer(
                            name=name,
                            header=actual_header,
                            rows=reader,
                            allowlist=allowlist,
                            writer=writer,
                            stop_at=SOURCE_END,
                        )
                    text_output.flush()
                finally:
                    text_output.detach()
            raw_output.flush()
            os.fsync(raw_output.fileno())
        cut_hash, compressed_bytes = install_write_once_file(
            output, temporary_path
        )
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if name in {"leadership", "aggtrade"} and audit["stopped_at"] is not None:
        audit["source_contract_failure"] = "declared_pre2024_container_had_later_row"
    if name == "premium" and audit["stopped_at"] is None:
        audit["source_contract_failure"] = (
            "premium_container_did_not_reach_frozen_cutoff"
        )
    audit.update({
        "source": source,
        "output": output,
        "parent_sha256": sha256_file(source),
        "physical_header_sha256": header_sha256(source),
        "cut_header": list(allowlist),
        "cut_header_sha256": sha256_bytes((",".join(allowlist) + "\n").encode("utf-8")),
        "cut_sha256": cut_hash,
        "compressed_bytes": compressed_bytes,
        "gzip_mtime": 0,
    })
    return audit


def build_source_cuts(*, source_contracts_validated: bool = False) -> dict[str, Any]:
    if not source_contracts_validated:
        validate_source_contracts()
    cuts = {
        "leadership": project_source_cut("leadership", prereg.LEADERSHIP_SOURCE, prereg.LEADERSHIP_ALLOWLIST, prereg.PRE2024_CUTS["leadership"]),
        "aggtrade": project_source_cut("aggtrade", prereg.AGGTRADE_SOURCE, prereg.AGGTRADE_ALLOWLIST, prereg.PRE2024_CUTS["aggtrade"]),
        "premium": project_source_cut("premium", prereg.PREMIUM_SOURCE, prereg.PREMIUM_ALLOWLIST, prereg.PRE2024_CUTS["premium"]),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "source_end_exclusive": prereg.SOURCE_END_EXCLUSIVE,
        "projection": "streaming physical allowlist before support load",
        "gzip_mtime": 0,
        "cuts": cuts,
        "outcome_boundary": ZERO_OUTCOME_COUNTERS,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_cut_manifest(payload: Mapping[str, Any]) -> None:
    claimed = payload.get("manifest_hash")
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if not isinstance(claimed, str) or canonical_hash(body) != claimed:
        raise RuntimeError("TRACER cut manifest hash mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("TRACER cut manifest protocol mismatch")
    if payload.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("TRACER cut manifest preregistration mismatch")
    if (
        payload.get("preregistration_manifest_hash")
        != PREREGISTRATION_MANIFEST_HASH
    ):
        raise RuntimeError("TRACER cut manifest preregistration hash mismatch")
    if payload.get("outcome_boundary") != ZERO_OUTCOME_COUNTERS:
        raise RuntimeError("TRACER cut manifest opened an outcome boundary")
    cuts = payload.get("cuts")
    if not isinstance(cuts, Mapping) or set(cuts) != set(
        prereg.PRE2024_CUTS
    ):
        raise RuntimeError("TRACER cut manifest source set mismatch")
    specifications = {
        "leadership": {
            "source": prereg.LEADERSHIP_SOURCE,
            "parent_sha256": prereg.LEADERSHIP_SOURCE_SHA256,
            "physical_header_sha256": prereg.LEADERSHIP_HEADER_SHA256,
            "allowlist": prereg.LEADERSHIP_ALLOWLIST,
        },
        "aggtrade": {
            "source": prereg.AGGTRADE_SOURCE,
            "parent_sha256": prereg.AGGTRADE_SOURCE_SHA256,
            "physical_header_sha256": prereg.AGGTRADE_HEADER_SHA256,
            "allowlist": prereg.AGGTRADE_ALLOWLIST,
        },
        "premium": {
            "source": prereg.PREMIUM_SOURCE,
            "parent_sha256": prereg.PREMIUM_SOURCE_SHA256,
            "physical_header_sha256": prereg.PREMIUM_HEADER_SHA256,
            "allowlist": prereg.PREMIUM_ALLOWLIST,
        },
    }
    for name, expected_output in prereg.PRE2024_CUTS.items():
        cut = cuts[name]
        if not isinstance(cut, Mapping):
            raise RuntimeError(f"TRACER {name} cut manifest entry malformed")
        specification = specifications[name]
        if cut.get("output") != expected_output:
            raise RuntimeError(f"TRACER {name} cut output mismatch")
        if cut.get("source") != specification["source"]:
            raise RuntimeError(f"TRACER {name} cut source mismatch")
        if cut.get("parent_sha256") != specification["parent_sha256"]:
            raise RuntimeError(f"TRACER {name} cut parent hash mismatch")
        if (
            cut.get("physical_header_sha256")
            != specification["physical_header_sha256"]
        ):
            raise RuntimeError(
                f"TRACER {name} cut parent header hash mismatch"
            )
        expected_header = tuple(specification["allowlist"])
        expected_header_hash = sha256_bytes(
            (",".join(expected_header) + "\n").encode("utf-8")
        )
        if tuple(cut.get("cut_header", ())) != expected_header:
            raise RuntimeError(f"TRACER {name} cut header mismatch")
        if cut.get("cut_header_sha256") != expected_header_hash:
            raise RuntimeError(f"TRACER {name} cut header hash mismatch")
        if read_header(expected_output) != expected_header:
            raise RuntimeError(
                f"TRACER {name} installed cut physical header mismatch"
            )
        if header_sha256(expected_output) != expected_header_hash:
            raise RuntimeError(
                f"TRACER {name} installed cut header hash mismatch"
            )
        if sha256_file(expected_output) != cut.get("cut_sha256"):
            raise RuntimeError(f"TRACER {name} installed cut hash mismatch")
        if cut.get("compressed_bytes") != repository_path(
            expected_output
        ).stat().st_size:
            raise RuntimeError(f"TRACER {name} cut byte count mismatch")
        if cut.get("gzip_mtime") != 0 or gzip_mtime(expected_output) != 0:
            raise RuntimeError(f"TRACER {name} cut gzip mtime mismatch")
        if (
            not isinstance(cut.get("source_rows_read"), int)
            or not isinstance(cut.get("cut_rows_written"), int)
            or cut["source_rows_read"] < cut["cut_rows_written"]
            or cut["cut_rows_written"] < 0
        ):
            raise RuntimeError(f"TRACER {name} cut row audit mismatch")
        if cut.get("post_2023_numeric_source_rows_opened") != 0:
            raise RuntimeError(
                f"TRACER {name} cut crossed post-2023 numeric boundary"
            )
        if cut.get("forbidden_columns_projected") != 0:
            raise RuntimeError(
                f"TRACER {name} cut projected a forbidden column"
            )
        if cut["cut_rows_written"]:
            first = parse_utc(cut.get("first_date"))
            last = parse_utc(cut.get("last_date"))
            if not (SOURCE_START <= first <= last < SOURCE_END):
                raise RuntimeError(
                    f"TRACER {name} cut timestamp audit mismatch"
                )
        elif cut.get("first_date") is not None or cut.get("last_date") is not None:
            raise RuntimeError(f"TRACER {name} empty cut timestamp mismatch")
        if name == "premium" and cut.get("stopped_at") != canonical_ts(
            SOURCE_END
        ):
            raise RuntimeError("TRACER premium cut did not stop at cutoff")
        if (
            name in {"leadership", "aggtrade"}
            and cut.get("stopped_at") is not None
            and cut.get("source_contract_failure")
            != "declared_pre2024_container_had_later_row"
        ):
            raise RuntimeError(
                f"TRACER {name} unexpected later-row audit mismatch"
            )


def validate_loaded_cut_frames(
    cut_manifest: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
) -> None:
    cuts = cut_manifest["cuts"]
    for name, frame in frames.items():
        cut = cuts[name]
        if len(frame) != cut["cut_rows_written"]:
            raise RuntimeError(f"TRACER {name} loaded cut row count mismatch")
        if len(frame):
            if canonical_ts(frame["date"].iloc[0]) != cut["first_date"]:
                raise RuntimeError(
                    f"TRACER {name} loaded cut first timestamp mismatch"
                )
            if canonical_ts(frame["date"].iloc[-1]) != cut["last_date"]:
                raise RuntimeError(
                    f"TRACER {name} loaded cut last timestamp mismatch"
                )


def load_cut_manifest(path: str | Path = DEFAULT_CUT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(repository_path(path).read_text(encoding="utf-8"))
    validate_cut_manifest(payload)
    return payload


def source_contracts_ready(
    source_audit: Mapping[str, Any],
    cut_manifest: Mapping[str, Any],
) -> bool:
    if set(source_audit) != set(prereg.PRE2024_CUTS):
        return False
    cuts = cut_manifest.get("cuts", {})
    if not isinstance(cuts, Mapping) or set(cuts) != set(
        prereg.PRE2024_CUTS
    ):
        return False
    return all(
        isinstance(cuts[name], Mapping)
        and not cuts[name].get("source_contract_failure")
        and cuts[name].get("parent_sha256")
        == source_audit[name].get("sha256")
        and cuts[name].get("output") == prereg.PRE2024_CUTS[name]
        and cuts[name].get("cut_header")
        == list(
            {
                "leadership": prereg.LEADERSHIP_ALLOWLIST,
                "aggtrade": prereg.AGGTRADE_ALLOWLIST,
                "premium": prereg.PREMIUM_ALLOWLIST,
            }[name]
        )
        and cuts[name].get("post_2023_numeric_source_rows_opened") == 0
        and cuts[name].get("forbidden_columns_projected") == 0
        for name in prereg.PRE2024_CUTS
    )


def _strict_binary_series(
    raw: pd.Series,
    *,
    name: str,
    field: str,
) -> pd.Series:
    normalized = raw.astype(str).str.strip().str.lower()
    allowed = {"0", "1", "false", "true"}
    invalid = ~normalized.isin(allowed)
    if invalid.any():
        raise RuntimeError(f"TRACER {name} non-binary flag in cut: {field}")
    return normalized.isin({"1", "true"})


def _numeric_series(
    raw: pd.Series,
    *,
    name: str,
    field: str,
    allow_missing: bool,
) -> tuple[pd.Series, pd.Series]:
    normalized = raw.astype(str).str.strip()
    missing = normalized.str.lower().isin({"", "nan", "na", "null"})
    if missing.any() and not allow_missing:
        raise RuntimeError(f"TRACER {name} missing numeric cell in cut: {field}")
    parsed = pd.to_numeric(normalized.mask(missing), errors="coerce")
    malformed = (~missing) & parsed.isna()
    if malformed.any():
        raise RuntimeError(f"TRACER {name} malformed numeric cell in cut: {field}")
    finite = np.isfinite(parsed.fillna(0.0).to_numpy(dtype=float))
    if not finite.all():
        raise RuntimeError(f"TRACER {name} non-finite numeric cell in cut: {field}")
    return parsed.astype(float), missing


def _validate_time_axis(
    frame: pd.DataFrame,
    *,
    name: str,
    field: str,
) -> None:
    series = frame[field]
    if not series.is_unique:
        raise RuntimeError(f"TRACER {name} duplicate {field} in cut")
    if not series.is_monotonic_increasing:
        raise RuntimeError(f"TRACER {name} non-monotone {field} in cut")


def load_cut_frame(name: str, path: str, columns: Sequence[str]) -> pd.DataFrame:
    if read_header(path) != tuple(columns):
        raise RuntimeError(f"TRACER {name} cut physical header mismatch")
    frame = pd.read_csv(
        repository_path(path),
        usecols=list(columns),
        dtype=str,
        keep_default_na=False,
    )
    if tuple(frame.columns) != tuple(columns):
        raise RuntimeError(f"TRACER {name} cut projection order mismatch")
    frame["date"] = pd.to_datetime(
        frame["date"], utc=True, errors="raise"
    )
    for col in ("feature_available_time_utc", "source_close_time", "feature_available_time"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(
                frame[col], utc=True, errors="raise"
            )
    _validate_time_axis(frame, name=name, field="date")
    if (
        (frame["date"] < SOURCE_START).any()
        or (frame["date"] >= SOURCE_END).any()
    ):
        raise RuntimeError(f"TRACER {name} cut date outside frozen interval")
    for flag in ("source_complete", "cross_venue_feature_valid", "source_valid"):
        if flag in frame.columns:
            frame[flag] = _strict_binary_series(
                frame[flag], name=name, field=flag
            )

    if name == "leadership":
        _validate_time_axis(
            frame,
            name=name,
            field="feature_available_time_utc",
        )
        if not frame["feature_available_time_utc"].equals(
            frame["date"] + pd.Timedelta(minutes=5)
        ):
            raise RuntimeError("TRACER leadership availability clock mismatch")
        valid = (
            frame["source_complete"]
            & frame["cross_venue_feature_valid"]
        )
        for col in NUMERIC_COLUMNS[name]:
            numeric, missing = _numeric_series(
                frame[col],
                name=name,
                field=col,
                allow_missing=True,
            )
            if (valid & missing).any():
                raise RuntimeError(
                    f"TRACER leadership valid row missing numeric cell: {col}"
                )
            frame[col] = numeric
        for col in NONNEGATIVE_COLUMNS[name]:
            if (frame.loc[valid, col] < 0.0).any():
                raise RuntimeError(
                    f"TRACER leadership negative nonnegative cell: {col}"
                )
        for prefix in ("spot", "um"):
            if (
                frame.loc[valid, f"{prefix}_signed_quote_notional"].abs()
                > frame.loc[valid, f"{prefix}_quote_notional"]
            ).any():
                raise RuntimeError(
                    f"TRACER leadership {prefix} signed notional exceeds quote"
                )
        return frame

    if name == "aggtrade":
        integer_fields = (
            "first_transact_time_ms",
            "last_transact_time_ms",
            "agg_trade_count",
        )
        for col in integer_fields:
            raw = frame[col].astype(str).str.strip()
            if not raw.str.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)").all():
                raise RuntimeError(
                    f"TRACER aggtrade non-canonical integer cell in cut: {col}"
                )
        for col in NUMERIC_COLUMNS[name]:
            numeric, _ = _numeric_series(
                frame[col],
                name=name,
                field=col,
                allow_missing=False,
            )
            frame[col] = numeric
        for col in integer_fields:
            frame[col] = frame[col].astype("int64")
        first = pd.to_datetime(
            frame["first_transact_time_ms"], unit="ms", utc=True
        )
        last = pd.to_datetime(
            frame["last_transact_time_ms"], unit="ms", utc=True
        )
        if not (
            (frame["date"] <= first)
            & (first <= last)
            & (last < frame["date"] + pd.Timedelta(minutes=5))
        ).all():
            raise RuntimeError("TRACER aggtrade transaction clock mismatch")
        if (frame["agg_trade_count"] <= 0).any():
            raise RuntimeError("TRACER aggtrade count must be positive")
        for col in NONNEGATIVE_COLUMNS[name]:
            if (frame[col] < 0.0).any():
                raise RuntimeError(
                    f"TRACER aggtrade negative nonnegative cell: {col}"
                )
        if (
            frame["signed_quote_notional"].abs()
            > frame["quote_notional"]
        ).any():
            raise RuntimeError(
                "TRACER aggtrade signed notional exceeds quote"
            )
        for col in (
            "event_notional_hhi",
            "sign_flip_rate",
            "max_same_sign_run_share",
        ):
            if ((frame[col] < 0.0) | (frame[col] > 1.0)).any():
                raise RuntimeError(
                    f"TRACER aggtrade bounded ratio out of range: {col}"
                )
        if (frame["normalized_effective_event_count"] <= 0.0).any():
            raise RuntimeError(
                "TRACER aggtrade effective participation must be positive"
            )
        return frame

    if name == "premium":
        for field in ("source_close_time", "feature_available_time"):
            _validate_time_axis(frame, name=name, field=field)
        if not frame["source_close_time"].equals(
            frame["date"] + pd.Timedelta(seconds=59, milliseconds=999)
        ):
            raise RuntimeError("TRACER premium close clock mismatch")
        if not frame["feature_available_time"].equals(
            frame["date"] + pd.Timedelta(seconds=61)
        ):
            raise RuntimeError("TRACER premium availability clock mismatch")
        valid = frame["source_valid"]
        for col in NUMERIC_COLUMNS[name]:
            numeric, missing = _numeric_series(
                frame[col],
                name=name,
                field=col,
                allow_missing=True,
            )
            if (valid & missing).any():
                raise RuntimeError(
                    f"TRACER premium valid row missing numeric cell: {col}"
                )
            if ((~valid) & (~missing)).any():
                raise RuntimeError(
                    "TRACER invalid premium row exposes usable OHLC"
                )
            frame[col] = numeric
        if (
            frame.loc[valid, "premium_high"]
            < np.maximum(
                frame.loc[valid, "premium_open"],
                frame.loc[valid, "premium_close"],
            )
        ).any():
            raise RuntimeError("TRACER premium high envelope mismatch")
        if (
            frame.loc[valid, "premium_low"]
            > np.minimum(
                frame.loc[valid, "premium_open"],
                frame.loc[valid, "premium_close"],
            )
        ).any():
            raise RuntimeError("TRACER premium low envelope mismatch")
        return frame

    raise ValueError(f"unknown TRACER source: {name}")


def _indexed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise RuntimeError("TRACER frame is missing date")
    if not frame["date"].is_unique or not frame["date"].is_monotonic_increasing:
        raise RuntimeError("TRACER frame date must be unique and monotone")
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.equals(
        pd.DatetimeIndex(frame["date"])
    ):
        return frame
    return frame.set_index("date", drop=False, verify_integrity=True)
    return frame


def _exact_grid(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, freq: str) -> bool:
    expected = pd.date_range(start, end, freq=freq, inclusive="left")
    return len(frame) == len(expected) and frame["date"].is_unique and frame["date"].equals(pd.Series(expected))


def _window(frame: pd.DataFrame, boundary: pd.Timestamp) -> pd.DataFrame:
    times = prereg.state_times(boundary)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise RuntimeError("TRACER window frame is not date-indexed")
    return frame.loc[
        times["window_start"] : times["window_end"] - pd.Timedelta(nanoseconds=1)
    ]


def sign_value(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def validate_window(name: str, window: pd.DataFrame, boundary: pd.Timestamp) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected_rows = prereg.Policy().premium_rows if name == "premium" else prereg.Policy().five_minute_rows
    freq = "1min" if name == "premium" else "5min"
    times = prereg.state_times(boundary)
    if not _exact_grid(window.reset_index(drop=True), times["window_start"], times["window_end"], freq):
        reasons.append(f"{name}_grid")
    if len(window) != expected_rows:
        reasons.append(f"{name}_row_count")
    numeric = list(NUMERIC_COLUMNS[name])
    if numeric and not np.isfinite(window[numeric].to_numpy(dtype=float)).all():
        reasons.append(f"{name}_finite")
    for col in NONNEGATIVE_COLUMNS[name]:
        if (window[col].to_numpy(dtype=float) < 0.0).any():
            reasons.append(f"{name}_{col}_negative")
    if name == "leadership":
        if not window["feature_available_time_utc"].equals(window["date"] + pd.Timedelta(minutes=5)):
            reasons.append("leadership_availability_clock")
        if not (window["source_complete"].all() and window["cross_venue_feature_valid"].all()):
            reasons.append("leadership_flags")
        if (window["spot_signed_quote_notional"].abs() > window["spot_quote_notional"]).any():
            reasons.append("leadership_spot_signed_exceeds_quote")
        if (window["um_signed_quote_notional"].abs() > window["um_quote_notional"]).any():
            reasons.append("leadership_um_signed_exceeds_quote")
    elif name == "aggtrade":
        first = pd.to_datetime(window["first_transact_time_ms"], unit="ms", utc=True)
        last = pd.to_datetime(window["last_transact_time_ms"], unit="ms", utc=True)
        if not ((window["date"] <= first) & (first <= last) & (last < window["date"] + pd.Timedelta(minutes=5))).all():
            reasons.append("aggtrade_transaction_clock")
        if not np.equal(window["agg_trade_count"].to_numpy(float), np.floor(window["agg_trade_count"].to_numpy(float))).all():
            reasons.append("aggtrade_count_integer")
        if (window["signed_quote_notional"].abs() > window["quote_notional"]).any():
            reasons.append("aggtrade_signed_exceeds_quote")
        if (window["agg_trade_count"] <= 0.0).any():
            reasons.append("aggtrade_count_nonpositive")
        if ((window["event_notional_hhi"] < 0.0) | (window["event_notional_hhi"] > 1.0)).any():
            reasons.append("aggtrade_hhi_range")
        if (window["normalized_effective_event_count"] <= 0.0).any():
            reasons.append("aggtrade_effective_nonpositive")
        for col in ("sign_flip_rate", "max_same_sign_run_share"):
            if ((window[col] < 0.0) | (window[col] > 1.0)).any():
                reasons.append(f"aggtrade_{col}_range")
    elif name == "premium":
        if not window["source_close_time"].equals(window["date"] + pd.Timedelta(seconds=59, milliseconds=999)):
            reasons.append("premium_close_clock")
        if not window["feature_available_time"].equals(window["date"] + pd.Timedelta(seconds=61)):
            reasons.append("premium_availability_clock")
        if not window["source_valid"].all():
            reasons.append("premium_valid_flag")
        if (window["premium_high"] < np.maximum(window["premium_open"], window["premium_close"])).any():
            reasons.append("premium_high_range")
        if (window["premium_low"] > np.minimum(window["premium_open"], window["premium_close"])).any():
            reasons.append("premium_low_range")
    return (not reasons, reasons)


def aggregate_primitives(boundary: pd.Timestamp, leadership: pd.DataFrame, aggtrade: pd.DataFrame, premium: pd.DataFrame) -> tuple[bool, tuple[str, ...], dict[str, float]]:
    lw, aw, pw = _window(leadership, boundary), _window(aggtrade, boundary), _window(premium, boundary)
    reasons: list[str] = []
    for name, window in (("leadership", lw), ("aggtrade", aw), ("premium", pw)):
        ok, why = validate_window(name, window, boundary)
        if not ok:
            reasons.extend(why)
    if reasons:
        return False, tuple(reasons), {}
    primitives = {
        "cash_flow": float(lw["spot_signed_quote_notional"].sum()),
        "leverage_flow": float(lw["um_signed_quote_notional"].sum()),
        "auction_flow": float(aw["signed_quote_notional"].sum()),
        "auction_return": float(aw["micro_log_return"].sum()),
        "sponsor_score": float((lw["spot_to_um_lagged_flow_response_bp"] - lw["um_to_spot_lagged_flow_response_bp"]).mean()),
        "participation_hhi": float(aw["event_notional_hhi"].median()),
        "effective_participation": float(aw["normalized_effective_event_count"].median()),
        "flow_flip": float(aw["sign_flip_rate"].median()),
        "flow_run": float(aw["max_same_sign_run_share"].median()),
        "arrival_burst": float(aw["interarrival_burstiness"].median()),
        "arrival_wait": float(aw["interarrival_mean_ms"].median()),
        "basis_change": float(lw["close_basis_bp"].iloc[-1] - lw["open_basis_bp"].iloc[0]),
        "premium_change": float(pw["premium_close"].iloc[-1] - pw["premium_open"].iloc[0]),
        "premium_range": float(pw["premium_high"].max() - pw["premium_low"].min()),
    }
    if not all(math.isfinite(v) for v in primitives.values()):
        return False, ("primitive_finite",), {}
    return True, tuple(), primitives


def current_tokens(primitives: Mapping[str, float], ranks: Mapping[str, str], prior: PrimitiveState | None) -> OrderedDict[str, str]:
    tokens: OrderedDict[str, str] = OrderedDict()
    tokens["sponsor"] = "CASH_LEADS" if ranks["sponsor_score"] == "HIGH" else "LEVERAGE_LEADS" if ranks["sponsor_score"] == "LOW" else "BALANCED"
    cash_sign = sign_value(primitives["cash_flow"])
    lev_sign = sign_value(primitives["leverage_flow"])
    if cash_sign > 0 and lev_sign > 0:
        tokens["flow_consensus"] = "CONSENSUS_BUY"
    elif cash_sign < 0 and lev_sign < 0:
        tokens["flow_consensus"] = "CONSENSUS_SELL"
    elif cash_sign > 0 and lev_sign < 0:
        tokens["flow_consensus"] = "CASH_BUY_LEVERAGE_SELL"
    elif cash_sign < 0 and lev_sign > 0:
        tokens["flow_consensus"] = "CASH_SELL_LEVERAGE_BUY"
    else:
        tokens["flow_consensus"] = "FLOW_NEUTRAL"
    auction_flow = primitives["auction_flow"]
    auction_return = primitives["auction_return"]
    if auction_flow > 0.0 and auction_return > 0.0:
        tokens["impact_relation"] = "BUY_FOLLOWTHROUGH"
    elif auction_flow > 0.0 and auction_return <= 0.0:
        tokens["impact_relation"] = "BUY_ABSORBED"
    elif auction_flow < 0.0 and auction_return < 0.0:
        tokens["impact_relation"] = "SELL_FOLLOWTHROUGH"
    elif auction_flow < 0.0 and auction_return >= 0.0:
        tokens["impact_relation"] = "SELL_ABSORBED"
    else:
        tokens["impact_relation"] = "RESPONSE_NEUTRAL"
    tokens["participation"] = "CONCENTRATED" if ranks["participation_hhi"] == "HIGH" and ranks["effective_participation"] == "LOW" else "BROAD" if ranks["participation_hhi"] == "LOW" and ranks["effective_participation"] == "HIGH" else "MIXED"
    tokens["flow_persistence"] = "PERSISTENT" if ranks["flow_flip"] == "LOW" and ranks["flow_run"] == "HIGH" else "ROTATING" if ranks["flow_flip"] == "HIGH" and ranks["flow_run"] == "LOW" else "MIXED"
    tokens["auction_tempo"] = "BURST" if ranks["arrival_burst"] == "HIGH" else "SLOW" if ranks["arrival_wait"] == "HIGH" else "STEADY"
    premium_change = primitives["premium_change"]
    if auction_return > 0.0 and premium_change > 0.0:
        tokens["premium_price_relation"] = "CROWDING_CONFIRMS_UP"
    elif auction_return < 0.0 and premium_change < 0.0:
        tokens["premium_price_relation"] = "CROWDING_CONFIRMS_DOWN"
    elif auction_return > 0.0 and premium_change < 0.0:
        tokens["premium_price_relation"] = "PREMIUM_DIVERGES_FROM_UP"
    elif auction_return < 0.0 and premium_change > 0.0:
        tokens["premium_price_relation"] = "PREMIUM_DIVERGES_FROM_DOWN"
    else:
        tokens["premium_price_relation"] = "PREMIUM_NEUTRAL"
    basis_change = primitives["basis_change"]
    if basis_change > 0.0 and premium_change > 0.0:
        tokens["basis_premium_relation"] = "BOTH_EXPAND"
    elif basis_change < 0.0 and premium_change < 0.0:
        tokens["basis_premium_relation"] = "BOTH_COMPRESS"
    elif basis_change != 0.0 and premium_change == 0.0:
        tokens["basis_premium_relation"] = "BASIS_ONLY"
    elif basis_change == 0.0 and premium_change != 0.0:
        tokens["basis_premium_relation"] = "PREMIUM_ONLY"
    elif basis_change * premium_change < 0.0:
        tokens["basis_premium_relation"] = "CROSS_DISAGREE"
    else:
        tokens["basis_premium_relation"] = "BOTH_NEUTRAL"
    if prior is None or not prior.valid or not prior.rank_ready or prior.tokens is None:
        tokens["sponsor_transition"] = "SPONSOR_MIXED"
        tokens["impact_transition"] = "IMPACT_MIXED"
        tokens["crowding_transition"] = "CROWDING_STABLE"
    else:
        prior_sponsor = prior.tokens["sponsor"]
        current_sponsor = tokens["sponsor"]
        if prior_sponsor == current_sponsor == "CASH_LEADS":
            tokens["sponsor_transition"] = "STABLE_CASH"
        elif prior_sponsor == current_sponsor == "LEVERAGE_LEADS":
            tokens["sponsor_transition"] = "STABLE_LEVERAGE"
        elif current_sponsor == "CASH_LEADS" and prior_sponsor != current_sponsor:
            tokens["sponsor_transition"] = "ROTATED_TO_CASH"
        elif current_sponsor == "LEVERAGE_LEADS" and prior_sponsor != current_sponsor:
            tokens["sponsor_transition"] = "ROTATED_TO_LEVERAGE"
        else:
            tokens["sponsor_transition"] = "SPONSOR_MIXED"
        prior_follow = "FOLLOWTHROUGH" in prior.tokens["impact_relation"]
        prior_absorb = "ABSORBED" in prior.tokens["impact_relation"]
        current_follow = "FOLLOWTHROUGH" in tokens["impact_relation"]
        current_absorb = "ABSORBED" in tokens["impact_relation"]
        if prior_follow and current_follow:
            tokens["impact_transition"] = "FOLLOWTHROUGH_PERSISTS"
        elif prior_absorb and current_absorb:
            tokens["impact_transition"] = "ABSORPTION_PERSISTS"
        elif prior_follow and current_absorb:
            tokens["impact_transition"] = "FOLLOWTHROUGH_TO_ABSORPTION"
        elif prior_absorb and current_follow:
            tokens["impact_transition"] = "ABSORPTION_TO_FOLLOWTHROUGH"
        else:
            tokens["impact_transition"] = "IMPACT_MIXED"
        prior_pc = sign_value(prior.primitives["premium_change"])
        current_pc = sign_value(premium_change)
        if prior_pc and current_pc and prior_pc == -current_pc:
            tokens["crowding_transition"] = "CROWDING_FLIPS"
        elif prior_pc and current_pc and prior_pc == current_pc and RANK_ORDER[ranks["premium_range"]] > RANK_ORDER[prior.ranks["premium_range"]]:
            tokens["crowding_transition"] = "CROWDING_BUILDS"
        elif prior_pc and current_pc and prior_pc == current_pc and RANK_ORDER[ranks["premium_range"]] < RANK_ORDER[prior.ranks["premium_range"]]:
            tokens["crowding_transition"] = "CROWDING_RELEASES"
        else:
            tokens["crowding_transition"] = "CROWDING_STABLE"
    prereg.validate_tokens(tokens)
    return tokens


def build_states(
    leadership: pd.DataFrame,
    aggtrade: pd.DataFrame,
    premium: pd.DataFrame,
    *,
    start: pd.Timestamp = SOURCE_START + pd.Timedelta(hours=4),
    end: pd.Timestamp = SOURCE_END,
) -> list[PrimitiveState]:
    leadership = _indexed_frame(leadership)
    aggtrade = _indexed_frame(aggtrade)
    premium = _indexed_frame(premium)
    boundaries = pd.date_range(
        parse_utc(start),
        parse_utc(end),
        freq="4h",
        inclusive="left",
    )
    histories = {name: deque(maxlen=prereg.Policy().rank_history_max) for name in RANKED_PRIMITIVES}
    states: list[PrimitiveState] = []
    prior: PrimitiveState | None = None
    for boundary in boundaries:
        valid, reasons, primitives = aggregate_primitives(boundary, leadership, aggtrade, premium)
        ranks: dict[str, str] = {}
        rank_ready = False
        tokens = None
        line = prereg.safety_line()
        sequence_ready = False
        if valid:
            rank_ready = all(len(histories[name]) >= prereg.Policy().rank_history_min for name in RANKED_PRIMITIVES)
            if rank_ready:
                policy = prereg.Policy()
                ranks = {
                    name: prereg.strict_prior_band(
                        primitives[name],
                        tuple(histories[name]),
                        minimum=policy.rank_history_min,
                        maximum=policy.rank_history_max,
                    )
                    for name in RANKED_PRIMITIVES
                }
                tokens = current_tokens(primitives, ranks, prior)
                line = prereg.canonical_line(tokens)
            for name in RANKED_PRIMITIVES:
                histories[name].append(float(primitives[name]))
        state = PrimitiveState(boundary=boundary, valid=valid, invalid_reasons=reasons, primitives=primitives, ranks=ranks, tokens=tokens, line=line, rank_ready=rank_ready, sequence_ready=False)
        states.append(state)
        prior = state
    # Sequence-ready means t-2, t-1, t are core-valid/rank-ready (safety lines excluded from support stats).
    ready: list[PrimitiveState] = []
    for i, state in enumerate(states):
        sequence_ready = (
            i >= 2
            and states[i - 1].boundary - states[i - 2].boundary
            == pd.Timedelta(hours=4)
            and state.boundary - states[i - 1].boundary
            == pd.Timedelta(hours=4)
            and all(
                states[j].valid and states[j].rank_ready
                for j in (i - 2, i - 1, i)
            )
        )
        ready.append(PrimitiveState(state.boundary, state.valid, state.invalid_reasons, state.primitives, state.ranks, state.tokens, state.line, state.rank_ready, sequence_ready))
    return ready


def make_token_rows(states: Sequence[PrimitiveState]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, state in enumerate(states):
        times = prereg.state_times(state.boundary)
        row: dict[str, Any] = {
            "boundary": canonical_ts(state.boundary),
            "window_start": canonical_ts(times["window_start"]),
            "window_end": canonical_ts(times["window_end"]),
            "premium_cutoff": canonical_ts(times["premium_cutoff"]),
            "decision_time": canonical_ts(times["decision_time"]),
            "execution_time": canonical_ts(times["execution_time"]),
            "core_source_ready": int(state.valid),
            "line_ready": int(state.valid and state.rank_ready),
            "sequence_ready": int(state.sequence_ready),
        }
        if state.tokens is None:
            safety = prereg.safety_line().split("|")
            row.update({name: safety[j] for j, name in enumerate(prereg.TOKEN_COLUMNS)})
        else:
            row.update(state.tokens)
        row["canonical_line"] = state.line
        if state.sequence_ready:
            sequence_text = f"{states[i - 2].line}\n{states[i - 1].line}\n{state.line}\n"
            row["sequence_signature"] = sha256_bytes(sequence_text.encode("utf-8"))
        else:
            row["sequence_signature"] = ""
        rows.append(row)
    return rows


def write_token_support(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(TOKEN_FILE_COLUMNS), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row[col] for col in TOKEN_FILE_COLUMNS})
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(output.getvalue().encode("utf-8"))
    return write_once_bytes(path, raw.getvalue())


def yearly_support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_year: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_year[str(pd.Timestamp(row["boundary"]).year)].append(row)
    reports: dict[str, Any] = {}
    for year in ("2020", "2021", "2022", "2023"):
        annual = by_year.get(year, [])
        seq = [r for r in annual if int(r["sequence_ready"]) == 1 and int(r["core_source_ready"]) == 1]
        field_counts = {name: Counter(str(r[name]) for r in seq) for name in prereg.TOKEN_COLUMNS}
        signatures = Counter("|".join(str(r[name]) for name in prereg.TOKEN_COLUMNS) for r in seq)
        total_seq = len(seq)
        reports[year] = {
            "nominal_boundaries": len(annual),
            "core_valid_boundaries": sum(int(r["core_source_ready"]) == 1 for r in annual),
            "source_invalid_boundaries": sum(int(r["core_source_ready"]) == 0 for r in annual),
            "sequence_ready_core_valid_boundaries": total_seq,
            "source_invalid_share": (sum(int(r["core_source_ready"]) == 0 for r in annual) / len(annual)) if annual else None,
            "core_valid_share": (sum(int(r["core_source_ready"]) == 1 for r in annual) / len(annual)) if annual else None,
            "category_counts": {name: dict(counts) for name, counts in field_counts.items()},
            "category_shares": {name: {k: v / total_seq for k, v in counts.items()} if total_seq else {} for name, counts in field_counts.items()},
            "distinct_signatures": len(signatures),
            "max_signature_share": (max(signatures.values()) / total_seq) if total_seq and signatures else None,
            "flow_buy_share": (field_counts["flow_consensus"].get("CONSENSUS_BUY", 0) / total_seq) if total_seq else None,
            "flow_sell_share": (field_counts["flow_consensus"].get("CONSENSUS_SELL", 0) / total_seq) if total_seq else None,
            "impact_followthrough_share": (sum(v for k, v in field_counts["impact_relation"].items() if "FOLLOWTHROUGH" in k) / total_seq) if total_seq else None,
            "impact_absorption_share": (sum(v for k, v in field_counts["impact_relation"].items() if "ABSORBED" in k) / total_seq) if total_seq else None,
            "sponsor_cash_share": (field_counts["sponsor"].get("CASH_LEADS", 0) / total_seq) if total_seq else None,
            "sponsor_leverage_share": (field_counts["sponsor"].get("LEVERAGE_LEADS", 0) / total_seq) if total_seq else None,
        }
    for left, right in (("2020", "2021"), ("2021", "2022"), ("2022", "2023")):
        reports[f"jsd_{left}_{right}"] = {
            name: prereg.jensen_shannon_divergence(
                reports[left]["category_counts"][name], reports[right]["category_counts"][name], prereg.TOKEN_VOCABULARY[name]
            ) if reports[left]["sequence_ready_core_valid_boundaries"] and reports[right]["sequence_ready_core_valid_boundaries"] else None
            for name in prereg.TOKEN_COLUMNS
        }
    return reports


def evaluate_support_gates(
    *,
    source_contracts_ok: bool,
    yearly: Mapping[str, Any],
    join: Mapping[str, Any],
    replay: Mapping[str, Any],
    controls: Mapping[str, Any],
    gates: Mapping[str, Any],
    outcome_boundary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] = (),
) -> OrderedDict[str, bool]:
    """Pure ordered evaluation of support gates from already-built evidence."""
    checks: OrderedDict[str, bool] = OrderedDict()
    q_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if int(row["sequence_ready"]) == 1:
            stamp = pd.Timestamp(row["boundary"])
            year = stamp.year
            quarter = (stamp.month - 1) // 3 + 1
            if (year, quarter) != (2020, 1):
                q_counts[f"{year}Q{quarter}"] += 1
    checks["gate_01_source_contracts_available"] = bool(
        source_contracts_ok
    )

    # Gate 2: all annual physical joins.
    for year in ("2020", "2021", "2022", "2023"):
        info = join.get(year, {})
        checks[f"gate_02_{year}_leadership_join_min"] = info.get("leadership_join_share_5m", -1.0) >= gates["source_join_min_each_year"]
        checks[f"gate_02_{year}_aggtrade_join_min"] = info.get("aggtrade_join_share_5m", -1.0) >= gates["source_join_min_each_year"]
        checks[f"gate_02_{year}_premium_join_min"] = info.get("premium_join_share_1m", -1.0) >= gates["source_join_min_each_year"]

    # Gate 3: all annual core-valid shares.
    for year in ("2020", "2021", "2022", "2023"):
        report = yearly[year]
        checks[f"gate_03_{year}_core_valid_min"] = bool(report["core_valid_share"] is not None and report["core_valid_share"] >= gates["core_valid_min_each_year"])

    # Gate 4: all annual and quarterly sequence-ready counts.
    for year in ("2020", "2021", "2022", "2023"):
        report = yearly[year]
        checks[f"gate_04_{year}_sequence_ready_min"] = bool(report["sequence_ready_core_valid_boundaries"] >= gates["sequence_ready_min"][year])
        for quarter in (1, 2, 3, 4):
            if (int(year), quarter) != (2020, 1):
                checks[f"gate_04_{year}Q{quarter}_quarter_ready_min"] = q_counts[f"{year}Q{quarter}"] >= gates["quarter_ready_min_after_warmup"]

    # Gate 5: all annual source-invalid shares.
    for year in ("2020", "2021", "2022", "2023"):
        report = yearly[year]
        checks[f"gate_05_{year}_source_invalid_max"] = bool(report["source_invalid_share"] is not None and report["source_invalid_share"] <= gates["source_invalid_max_each_year"])

    # Gate 6: annual category support for all fields.
    for year in ("2020", "2021", "2022", "2023"):
        report = yearly[year]
        for name, shares in report["category_shares"].items():
            values = [share for token, share in shares.items() if token in prereg.TOKEN_VOCABULARY[name]]
            checks[f"gate_06_{year}_{name}_two_categories_min"] = bool(sum(v >= gates["category_support_min"] for v in values) >= 2)
            checks[f"gate_06_{year}_{name}_category_share_max"] = bool(values and max(values) <= gates["category_share_max"])

    # Gates 7-10: preserve global gate-number ordering across years.
    for year in ("2020", "2021", "2022", "2023"):
        report = yearly[year]
        checks[f"gate_07_{year}_flow_buy_support"] = bool(report["flow_buy_share"] is not None and report["flow_buy_share"] >= gates["flow_buy_and_sell_min"])
        checks[f"gate_07_{year}_flow_sell_support"] = bool(report["flow_sell_share"] is not None and report["flow_sell_share"] >= gates["flow_buy_and_sell_min"])

    for year in ("2020", "2021", "2022", "2023"):
        report = yearly[year]
        checks[f"gate_08_{year}_impact_follow_support"] = bool(report["impact_followthrough_share"] is not None and report["impact_followthrough_share"] >= gates["impact_follow_and_absorb_min"])
        checks[f"gate_08_{year}_impact_absorb_support"] = bool(report["impact_absorption_share"] is not None and report["impact_absorption_share"] >= gates["impact_follow_and_absorb_min"])

    for year in ("2020", "2021", "2022", "2023"):
        report = yearly[year]
        checks[f"gate_09_{year}_sponsor_cash_support"] = bool(report["sponsor_cash_share"] is not None and report["sponsor_cash_share"] >= gates["sponsor_cash_and_leverage_min"])
        checks[f"gate_09_{year}_sponsor_leverage_support"] = bool(report["sponsor_leverage_share"] is not None and report["sponsor_leverage_share"] >= gates["sponsor_cash_and_leverage_min"])

    for year in ("2020", "2021", "2022", "2023"):
        report = yearly[year]
        checks[f"gate_10_{year}_distinct_signatures_min"] = bool(report["distinct_signatures"] >= gates["distinct_signatures_min"])
        checks[f"gate_10_{year}_signature_share_max"] = bool(report["max_signature_share"] is not None and report["max_signature_share"] <= gates["signature_share_max"])

    # Gates 11-14.
    for pair in ("jsd_2020_2021", "jsd_2021_2022", "jsd_2022_2023"):
        for name, value in yearly[pair].items():
            checks[f"gate_11_{pair}_{name}_max"] = bool(value is not None and value <= gates["adjacent_year_jsd_max"])
    for cid in prereg.CONTROL_IDS:
        result = controls.get(cid, {})
        checks[f"gate_12_control_{cid}_differs"] = result.get("canonical_line_stream_hash_differs") is True
    checks["gate_13_append_replay_byte_identical"] = bool(replay.get("byte_identical"))
    checks["gate_14_outcome_boundary_zero"] = bool(
        set(outcome_boundary) == set(ZERO_OUTCOME_COUNTERS)
        and all(value == 0 for value in outcome_boundary.values())
    )
    return checks


def source_join_audit(leadership: pd.DataFrame, aggtrade: pd.DataFrame, premium: pd.DataFrame) -> dict[str, Any]:
    for name, frame in (
        ("leadership", leadership),
        ("aggtrade", aggtrade),
        ("premium", premium),
    ):
        if not frame["date"].is_unique:
            raise RuntimeError(f"TRACER {name} duplicate date before join audit")
        if not frame["date"].is_monotonic_increasing:
            raise RuntimeError(
                f"TRACER {name} non-monotone date before join audit"
            )
    audit: dict[str, Any] = {}
    for year in range(2020, 2024):
        start, end = pd.Timestamp(f"{year}-01-01T00:00:00Z"), pd.Timestamp(f"{year+1}-01-01T00:00:00Z")
        grid5 = pd.date_range(start, end, freq="5min", inclusive="left")
        grid1 = pd.date_range(start, end, freq="1min", inclusive="left")
        leadership_dates = pd.DatetimeIndex(
            leadership.loc[
                leadership["date"].between(start, end, inclusive="left"),
                "date",
            ]
        )
        aggtrade_dates = pd.DatetimeIndex(
            aggtrade.loc[
                aggtrade["date"].between(start, end, inclusive="left"),
                "date",
            ]
        )
        premium_dates = pd.DatetimeIndex(
            premium.loc[
                premium["date"].between(start, end, inclusive="left"),
                "date",
            ]
        )
        audit[str(year)] = {
            "leadership_join_share_5m": float(
                len(leadership_dates.intersection(grid5)) / len(grid5)
            ),
            "aggtrade_join_share_5m": float(
                len(aggtrade_dates.intersection(grid5)) / len(grid5)
            ),
            "premium_join_share_1m": float(
                len(premium_dates.intersection(grid1)) / len(grid1)
            ),
        }
    return audit


def rotate_aggtrade_monthly(frame: pd.DataFrame, rows: int = prereg.Policy().control_aggtrade_rotate_rows) -> pd.DataFrame:
    pieces = []
    relation_cols = [
        "quote_notional",
        "signed_quote_notional",
        "micro_log_return",
        "event_notional_hhi",
        "normalized_effective_event_count",
        "sign_flip_rate",
        "max_same_sign_run_share",
        "interarrival_mean_ms",
        "interarrival_burstiness",
    ]
    grouped = frame.groupby([frame["date"].dt.year, frame["date"].dt.month], sort=True)
    for _, group in grouped:
        values = group.reset_index(drop=True).copy()
        values.loc[:, relation_cols] = np.roll(values[relation_cols].to_numpy(), rows, axis=0)
        pieces.append(values)
    return pd.concat(pieces, ignore_index=True)


def swap_cash_perp(frame: pd.DataFrame) -> pd.DataFrame:
    swapped = frame.copy()
    pairs = (
        ("spot_quote_notional", "um_quote_notional"),
        ("spot_signed_quote_notional", "um_signed_quote_notional"),
        ("spot_to_um_lagged_flow_response_bp", "um_to_spot_lagged_flow_response_bp"),
    )
    for left, right in pairs:
        swapped[left], swapped[right] = frame[right].to_numpy(), frame[left].to_numpy()
    swapped["open_basis_bp"] = -frame["open_basis_bp"].to_numpy()
    swapped["close_basis_bp"] = -frame["close_basis_bp"].to_numpy()
    return swapped


def stale_premium(frame: pd.DataFrame, minutes: int = prereg.Policy().control_premium_shift_minutes) -> pd.DataFrame:
    if minutes <= 0:
        raise ValueError("TRACER stale-premium shift must be positive")
    shifted = frame.copy()
    valid = frame["source_valid"].to_numpy(dtype=bool)
    stale_valid = np.zeros(len(frame), dtype=bool)
    if minutes < len(frame):
        stale_valid[minutes:] = valid[:-minutes]
    shifted["source_valid"] = stale_valid
    for column in (
        "premium_open",
        "premium_high",
        "premium_low",
        "premium_close",
    ):
        values = frame[column].to_numpy(dtype=float)
        stale_values = np.full(len(frame), np.nan, dtype=float)
        if minutes < len(frame):
            stale_values[minutes:] = values[:-minutes]
        shifted[column] = stale_values
    return shifted.reset_index(drop=True)


def controls_differ(primary_rows: Sequence[Mapping[str, Any]], leadership: pd.DataFrame, aggtrade: pd.DataFrame, premium: pd.DataFrame) -> dict[str, Any]:
    primary = [r["canonical_line"] for r in primary_rows]
    primary_hash = canonical_hash(primary)
    results: dict[str, Any] = {}
    for control_id, frames in {
        "premium_stale_1440m": (leadership, aggtrade, stale_premium(premium)),
        "cash_perpetual_swap": (swap_cash_perp(leadership), aggtrade, premium),
        "aggtrade_monthly_rotate_37_rows": (leadership, rotate_aggtrade_monthly(aggtrade), premium),
    }.items():
        try:
            rows = make_token_rows(build_states(*frames))
            sequence = [r["canonical_line"] for r in rows]
            control_hash = canonical_hash(sequence)
            results[control_id] = {
                "canonical_line_stream_hash": control_hash,
                "primary_canonical_line_stream_hash": primary_hash,
                "canonical_line_stream_hash_differs": control_hash != primary_hash,
                "rows": len(rows),
            }
        except Exception as exc:  # control failure is reportable, not outcome-bearing
            results[control_id] = {"canonical_line_stream_hash_differs": False, "error": str(exc)}
    return results


def _append_replay_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    columns = (
        "boundary",
        "core_source_ready",
        "line_ready",
        "sequence_ready",
        *prereg.TOKEN_COLUMNS,
        "canonical_line",
        "sequence_signature",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(columns),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in columns})
    return output.getvalue().encode("utf-8")


def source_prefix_frames(
    leadership: pd.DataFrame,
    aggtrade: pd.DataFrame,
    premium: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Physically hide rows unavailable at a frozen append-prefix cutoff."""
    cutoff = parse_utc(cutoff)
    leadership_prefix = leadership.loc[
        leadership["feature_available_time_utc"] <= cutoff
    ].copy()
    aggtrade_prefix = aggtrade.loc[
        aggtrade["date"] + pd.Timedelta(minutes=5) <= cutoff
    ].copy()
    premium_prefix = premium.loc[
        premium["feature_available_time"] <= cutoff
    ].copy()
    return leadership_prefix, aggtrade_prefix, premium_prefix


def append_replay_check(
    states: Sequence[PrimitiveState],
    leadership: pd.DataFrame,
    aggtrade: pd.DataFrame,
    premium: pd.DataFrame,
) -> dict[str, Any]:
    rows = make_token_rows(states)
    prefixes: dict[str, Any] = {}
    for cutoff in (
        "2021-01-01T00:00:00Z",
        "2022-01-01T00:00:00Z",
        "2023-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ):
        cutoff_ts = pd.Timestamp(cutoff)
        pivot = sum(state.boundary < cutoff_ts for state in states)
        prefix_frames = source_prefix_frames(
            leadership,
            aggtrade,
            premium,
            cutoff_ts,
        )
        rebuilt_states = build_states(
            *prefix_frames,
            end=cutoff_ts,
        )
        prefix_rows = make_token_rows(rebuilt_states)
        full_bytes = _append_replay_bytes(rows[:pivot])
        prefix_bytes = _append_replay_bytes(prefix_rows)
        prefixes[cutoff] = {
            "expected_rows": pivot,
            "rebuilt_rows": len(prefix_rows),
            "full_prefix_sha256": sha256_bytes(full_bytes),
            "rebuilt_prefix_sha256": sha256_bytes(prefix_bytes),
            "byte_identical": (
                len(prefix_rows) == pivot and prefix_bytes == full_bytes
            ),
        }
    return {
        "prefixes": prefixes,
        "byte_identical": all(
            result["byte_identical"] for result in prefixes.values()
        ),
        "full_rows": len(rows),
    }


def protocol_file_hashes() -> dict[str, str]:
    return {path: sha256_file(path) for path in PROTOCOL_PATHS}


def build_support_report(
    *,
    head_commit: str,
    preregistration: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    cut_manifest: Mapping[str, Any],
    cut_manifest_sha256: str,
    token_output: str,
    token_output_sha256: str,
    leadership: pd.DataFrame,
    aggtrade: pd.DataFrame,
    premium: pd.DataFrame,
    states: Sequence[PrimitiveState],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    yearly = yearly_support(rows)
    gates = preregistration["support_gates"]
    join = source_join_audit(leadership, aggtrade, premium)
    replay = append_replay_check(
        states,
        leadership,
        aggtrade,
        premium,
    )
    controls = controls_differ(
        rows, leadership, aggtrade, premium
    )
    outcome_boundary = dict(ZERO_OUTCOME_COUNTERS)
    contracts_ok = source_contracts_ready(source_audit, cut_manifest)
    checks = evaluate_support_gates(
        source_contracts_ok=contracts_ok,
        yearly=yearly,
        join=join,
        replay=replay,
        controls=controls,
        gates=gates,
        outcome_boundary=outcome_boundary,
        rows=rows,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "head_commit": head_commit,
        "protocol_file_sha256": protocol_file_hashes(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "cut_manifest": str(DEFAULT_CUT_MANIFEST),
        "cut_manifest_sha256": cut_manifest_sha256,
        "cut_manifest_manifest_hash": cut_manifest["manifest_hash"],
        "source_audit": source_audit,
        "source_contracts_ready": contracts_ok,
        "source_join": join,
        "state_rows": len(rows),
        "token_output": token_output,
        "token_output_sha256": token_output_sha256,
        "token_columns": list(TOKEN_FILE_COLUMNS),
        "yearly_support": yearly,
        "controls": controls,
        "append_replay": replay,
        "gate_checks": dict(checks),
        "first_failure": next((name for name, ok in checks.items() if not ok), None),
        "support_pass": all(checks.values()),
        "outcome_boundary": outcome_boundary,
        "decision": "authorize_stage_0_5_evaluator_freeze_only" if all(checks.values()) else "retire_tracer4_unchanged_before_outcomes",
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def _require_frozen_output(actual: str, expected: Path, label: str) -> None:
    if Path(actual).as_posix() != expected.as_posix():
        raise RuntimeError(
            f"TRACER {label} must use frozen path {expected.as_posix()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut-manifest", default=str(DEFAULT_CUT_MANIFEST))
    parser.add_argument("--token-output", default=str(DEFAULT_TOKEN_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--skip-cut-build", action="store_true", help="use already materialized cuts")
    args = parser.parse_args()
    _require_frozen_output(
        args.cut_manifest, DEFAULT_CUT_MANIFEST, "cut manifest"
    )
    _require_frozen_output(
        args.token_output, DEFAULT_TOKEN_OUTPUT, "token output"
    )
    _require_frozen_output(args.report, DEFAULT_REPORT, "support report")
    head = assert_protocol_committed()
    preregistration = load_preregistration()
    source_audit = validate_source_contracts()
    if not args.skip_cut_build:
        cut_manifest = build_source_cuts(source_contracts_validated=True)
        write_once_json(args.cut_manifest, cut_manifest)
        validate_cut_manifest(cut_manifest)
    else:
        cut_manifest = load_cut_manifest(args.cut_manifest)
    cut_manifest_sha256 = sha256_file(args.cut_manifest)
    leadership = _indexed_frame(
        load_cut_frame(
            "leadership",
            prereg.PRE2024_CUTS["leadership"],
            prereg.LEADERSHIP_ALLOWLIST,
        )
    )
    aggtrade = _indexed_frame(
        load_cut_frame(
            "aggtrade",
            prereg.PRE2024_CUTS["aggtrade"],
            prereg.AGGTRADE_ALLOWLIST,
        )
    )
    premium = _indexed_frame(
        load_cut_frame(
            "premium",
            prereg.PRE2024_CUTS["premium"],
            prereg.PREMIUM_ALLOWLIST,
        )
    )
    validate_loaded_cut_frames(
        cut_manifest,
        {
            "leadership": leadership,
            "aggtrade": aggtrade,
            "premium": premium,
        },
    )
    states = build_states(leadership, aggtrade, premium)
    rows = make_token_rows(states)
    token_sha = write_token_support(args.token_output, rows)
    report = build_support_report(
        head_commit=head,
        preregistration=preregistration,
        source_audit=source_audit,
        cut_manifest=cut_manifest,
        cut_manifest_sha256=cut_manifest_sha256,
        token_output=args.token_output,
        token_output_sha256=token_sha,
        leadership=leadership,
        aggtrade=aggtrade,
        premium=premium,
        states=states,
        rows=rows,
    )
    report_sha = write_once_json(args.report, report)
    print(json.dumps({"report": args.report, "report_sha256": report_sha, "token_output": args.token_output, "token_sha256": token_sha, "support_pass": report["support_pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
