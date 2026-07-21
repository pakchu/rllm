"""Freeze pre-2024 comparator timestamps for UGCI-288 novelty checks.

This sanitizer opens only previously frozen comparator clocks.  It retains the
preregistered candidate/control identities and timestamps inside each fixed
comparison interval.  The later UGCI support evaluator consumes this sealed
bundle instead of reopening comparator files that extend beyond 2023.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import preregister_usdc_gross_clearing_imbalance as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "ugci_prior_comparator_views_v1"
BUILDER = Path("training/freeze_ugci_prior_comparator_views.py")
PREREGISTRATION_ARTIFACT = Path(
    "results/usdc_gross_clearing_imbalance_preregistration_2026-07-22.json"
)
PREREGISTRATION_ARTIFACT_SHA256 = (
    "7056eadfd5b347b8b9afbe06cbc2a33f832a2913dc3227891a2a8d211aaa454a"
)
PREREGISTRATION_MANIFEST_HASH = (
    "61b6d60f8c2ef21b94b3343bc3cf2a5fd82366679ae9d768d846831b12829722"
)
DEFAULT_CLOCK = Path("results/ugci_prior_comparator_views_pre2024_2026-07-22.csv.gz")
DEFAULT_MANIFEST = Path(
    "results/ugci_prior_comparator_views_pre2024_manifest_2026-07-22.json"
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "entry_time",
    "comparison_start",
    "comparison_end_exclusive",
)
UTC = timezone.utc


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("UGCI comparator timestamp must be UTC")
    return parsed.astimezone(UTC)


def format_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_ARTIFACT) != PREREGISTRATION_ARTIFACT_SHA256:
        raise ValueError("UGCI preregistration artifact hash mismatch")
    payload = json.loads(
        repository_path(PREREGISTRATION_ARTIFACT).read_text(encoding="utf-8")
    )
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise ValueError("UGCI preregistration manifest hash mismatch")
    if payload.get("novelty_comparators") != list(prereg.COMPARATORS):
        raise ValueError("UGCI comparator contract changed")
    return payload


def sanitize_comparator(
    comparator: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    path = repository_path(comparator["path"])
    if sha256_file(path) != comparator["sha256"]:
        raise ValueError(f"UGCI comparator hash mismatch: {comparator['candidate']}")
    start = parse_time(comparator["comparison_start"])
    end = parse_time(comparator["comparison_end_exclusive"])
    controls = set(comparator["controls"])
    retained: list[dict[str, str]] = []
    physical_rows = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        entry_column = comparator["entry_column"]
        required = {"candidate", "control", entry_column}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"UGCI comparator schema changed: {path}")
        for row in reader:
            physical_rows += 1
            if row["candidate"] != comparator["candidate"]:
                continue
            if row["control"] not in controls:
                continue
            entry = parse_time(row[entry_column])
            if not start <= entry < end:
                continue
            retained.append(
                {
                    "candidate": comparator["candidate"],
                    "control": row["control"],
                    "entry_time": format_time(entry),
                    "comparison_start": comparator["comparison_start"],
                    "comparison_end_exclusive": comparator["comparison_end_exclusive"],
                }
            )
    retained.sort(key=lambda row: (row["control"], row["entry_time"]))
    identities = {(row["control"], row["entry_time"]) for row in retained}
    if len(identities) != len(retained):
        raise ValueError(f"UGCI comparator contains duplicate clocks: {path}")
    counts: dict[str, int] = {
        control: sum(row["control"] == control for row in retained)
        for control in comparator["controls"]
    }
    return retained, {
        "candidate": comparator["candidate"],
        "source_path": comparator["path"],
        "source_sha256": comparator["sha256"],
        "source_physical_rows_read_for_sanitization": physical_rows,
        "comparison_start": comparator["comparison_start"],
        "comparison_end_exclusive": comparator["comparison_end_exclusive"],
        "retained_counts": counts,
        "retained_rows": len(retained),
    }


def deterministic_gzip_csv(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=CLOCK_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in CLOCK_COLUMNS})
    return buffer.getvalue()


def write_once(path: str | Path, content: bytes) -> None:
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise FileExistsError(f"UGCI artifact is write-once: {destination}") from error


def build_payload(
    clock_output: str | Path = DEFAULT_CLOCK,
) -> tuple[dict[str, Any], bytes]:
    preregistration = validate_preregistration()
    rows: list[dict[str, str]] = []
    audits: list[dict[str, Any]] = []
    for comparator in prereg.COMPARATORS:
        retained, audit = sanitize_comparator(comparator)
        rows.extend(retained)
        audits.append(audit)
    rows.sort(key=lambda row: (row["candidate"], row["control"], row["entry_time"]))
    clock_bytes = deterministic_gzip_csv(rows)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": prereg.CANDIDATE,
        "purpose": "sealed_pre2024_timestamp_only_novelty_comparators",
        "preregistration": {
            "path": str(PREREGISTRATION_ARTIFACT),
            "sha256": PREREGISTRATION_ARTIFACT_SHA256,
            "manifest_hash": preregistration["manifest_hash"],
        },
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "inputs": audits,
        "output": {
            "path": str(clock_output),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "columns": list(CLOCK_COLUMNS),
            "rows": len(rows),
        },
        "outcome_boundary": {
            "ugci_source_rows_read": 0,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "comparator_timestamp_rows_read_for_sanitization": sum(
                audit["source_physical_rows_read_for_sanitization"] for audit in audits
            ),
            "post_2023_comparator_rows_retained": 0,
        },
        "next_action": "support evaluator may read only the sealed output clock",
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload, clock_bytes


def run(
    clock_output: str | Path = DEFAULT_CLOCK,
    manifest_output: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    if (
        repository_path(clock_output).exists()
        or repository_path(manifest_output).exists()
    ):
        raise FileExistsError("UGCI comparator outputs are write-once")
    payload, clock_bytes = build_payload(clock_output)
    manifest_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    write_once(clock_output, clock_bytes)
    write_once(manifest_output, manifest_bytes)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    payload = run(args.clock_output, args.manifest_output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
