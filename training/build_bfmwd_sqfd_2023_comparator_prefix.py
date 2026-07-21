"""Materialize the pre-2024 primary SQFD comparator prefix for BFMWD."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "bfmwd_sqfd_2023_comparator_prefix_v1"
SOURCE = Path("data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz")
SOURCE_SHA256 = "a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b"
TRANSPORT_FREEZE = Path(
    "results/bfmwd_sqfd_2023_comparator_prefix_transport_freeze_2026-07-20.json"
)
OUTPUT = Path("data/bfmwd_sqfd_primary_clocks_2023_prefix.csv.gz")
MANIFEST = Path(
    "results/bfmwd_sqfd_2023_comparator_prefix_manifest_2026-07-20.json"
)
BUILDER = Path("training/build_bfmwd_sqfd_2023_comparator_prefix.py")
START = cast(pd.Timestamp, pd.Timestamp("2023-01-01T00:00:00Z"))
END = cast(pd.Timestamp, pd.Timestamp("2024-01-01T00:00:00Z"))
OUTPUT_COLUMNS = ("control", "entry_time")


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer,
        mode="wb",
        filename="",
        compresslevel=9,
        mtime=0,
    ) as handle:
        handle.write(payload)
    return buffer.getvalue()


def extract_primary_prefix() -> tuple[bytes, dict[str, Any]]:
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise ValueError("SQFD comparator source hash changed")
    entries: set[pd.Timestamp] = set()
    rows_read = 0
    non_primary_rows = 0
    post_2023_rows = 0
    with gzip.open(repository_path(SOURCE), mode="rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if "control" not in header or "entry_time" not in header:
            raise ValueError("SQFD comparator source schema changed")
        control_index = header.index("control")
        entry_index = header.index("entry_time")
        for row in reader:
            rows_read += 1
            if len(row) != len(header):
                raise ValueError("SQFD comparator row width changed")
            entry = pd.Timestamp(row[entry_index])
            if entry is pd.NaT or entry.tzinfo is None:
                raise ValueError("SQFD comparator entry clock is invalid")
            entry = cast(pd.Timestamp, entry)
            if entry >= END:
                post_2023_rows += 1
                continue
            if entry < START:
                continue
            if row[control_index] != "primary":
                non_primary_rows += 1
                continue
            entries.add(entry)
    ordered = sorted(entries)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(OUTPUT_COLUMNS)
    for entry in ordered:
        writer.writerow(("primary", entry.isoformat()))
    payload = deterministic_gzip(output.getvalue().encode())
    diagnostics = {
        "source_rows_streamed": rows_read,
        "post_2023_rows_discarded": post_2023_rows,
        "pre_2024_non_primary_rows_discarded": non_primary_rows,
        "prefix_rows": len(ordered),
        "first_entry": ordered[0].isoformat() if ordered else None,
        "last_entry": ordered[-1].isoformat() if ordered else None,
    }
    return payload, diagnostics


def write_once_bytes(path: str | Path, payload: bytes) -> None:
    target = repository_path(path)
    if target.exists() and target.read_bytes() != payload:
        raise FileExistsError(f"refusing to overwrite frozen SQFD prefix: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def write_once_json(path: str | Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_once_bytes(path, encoded)


def build(
    *, output: str | Path = OUTPUT, manifest: str | Path = MANIFEST
) -> dict[str, Any]:
    transport = json.loads(repository_path(TRANSPORT_FREEZE).read_text())
    if transport.get("protocol_version") != (
        "bfmwd_sqfd_2023_comparator_prefix_transport_freeze_v1"
    ):
        raise ValueError("SQFD comparator prefix transport freeze changed")
    unhashed = dict(transport)
    manifest_hash = unhashed.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(unhashed):
        raise ValueError("SQFD comparator prefix transport freeze hash changed")
    if transport.get("bindings", {}).get("source", {}).get("sha256") != SOURCE_SHA256:
        raise ValueError("SQFD comparator prefix source binding changed")

    payload, diagnostics = extract_primary_prefix()
    write_once_bytes(output, payload)
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "transport_freeze": {
            "path": str(TRANSPORT_FREEZE),
            "sha256": sha256_file(TRANSPORT_FREEZE),
        },
        "source": {"path": str(SOURCE), "sha256": sha256_file(SOURCE)},
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "output": {
            "path": str(output),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "columns": list(OUTPUT_COLUMNS),
            "rows": diagnostics["prefix_rows"],
        },
        "filter": {
            "control": "primary",
            "start_inclusive": START.isoformat(),
            "end_exclusive": END.isoformat(),
        },
        "diagnostics": diagnostics,
        "outcome_boundary": {
            "btc_market_rows_read": 0,
            "funding_paid_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "outcomes_opened": False,
        },
    }
    result["manifest_hash"] = canonical_hash(result)
    write_once_json(manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args()
    print(
        json.dumps(
            build(output=args.output, manifest=args.manifest),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
