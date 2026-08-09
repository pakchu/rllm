"""Freeze one causally available NFCI observation from each next-Friday ALFRED vintage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=NFCI&vintage_date={vintage}"
DEFAULT_OUTPUT = Path("data/chicago_fed_nfci_alfred_causal_vintages_2020_2026.csv")
DEFAULT_MANIFEST = Path("data/chicago_fed_nfci_alfred_causal_vintages_2020_2026_manifest.json")
FIRST_REFERENCE = date(2020, 1, 3)
LAST_REFERENCE = date(2026, 7, 24)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fridays(start: date, end: date) -> list[date]:
    if start.weekday() != 4 or end.weekday() != 4:
        raise ValueError("reference endpoints must be Fridays")
    return [start + timedelta(days=7 * index) for index in range((end - start).days // 7 + 1)]


def parse_exact_observation(payload: bytes, reference: date, vintage: date) -> float | None:
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    expected_column = f"NFCI_{vintage:%Y%m%d}"
    if not rows or set(rows[0]) != {"observation_date", expected_column}:
        raise ValueError(f"unexpected ALFRED schema for {vintage}")
    matches = [row for row in rows if row["observation_date"] == reference.isoformat()]
    if not matches or (len(matches) == 1 and matches[0][expected_column] in {"", "."}):
        return None
    if len(matches) != 1:
        raise ValueError(f"duplicate NFCI observation {reference} in vintage {vintage}")
    return float(matches[0][expected_column])


def fetch(reference: date, attempts: int = 5) -> dict[str, object]:
    vintage = reference + timedelta(days=7)
    url = BASE_URL.format(vintage=vintage.isoformat())
    request = urllib.request.Request(url, headers={"User-Agent": "rllm-source-freezer/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
            value = parse_exact_observation(payload, reference, vintage)
            return {
                "reference_date": reference.isoformat(),
                "vintage_date": vintage.isoformat(),
                "available": value is not None,
                "nfci": "" if value is None else value,
                "source_url": url,
                "response_sha256": sha256(payload),
            }
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def bulk_url(references: list[date]) -> str:
    vintages = ",".join((reference + timedelta(days=7)).isoformat() for reference in references)
    series = ",".join("NFCI" for _ in references)
    return (
        f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={series}"
        f"&cosd={references[0].isoformat()}&coed={references[-1].isoformat()}"
        f"&vintage_date={vintages}"
    )


def parse_bulk(payload: bytes, references: list[date]) -> list[dict[str, object]]:
    records = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    by_date = {record["observation_date"]: record for record in records}
    rows: list[dict[str, object]] = []
    for reference in references:
        vintage = reference + timedelta(days=7)
        column = f"NFCI_{vintage:%Y%m%d}"
        if records and column not in records[0]:
            raise ValueError(f"missing ALFRED vintage column {column}")
        value = by_date.get(reference.isoformat(), {}).get(column, ".")
        available = value not in {"", ".", None}
        rows.append(
            {
                "reference_date": reference.isoformat(),
                "vintage_date": vintage.isoformat(),
                "available": available,
                "nfci": float(value) if available else "",
            }
        )
    return rows


def fetch_bulk(references: list[date], attempts: int = 5) -> tuple[list[dict[str, object]], str, str]:
    url = bulk_url(references)
    request = urllib.request.Request(url, headers={"User-Agent": "rllm-source-freezer/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            return parse_bulk(payload, references), url, sha256(payload)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(10 * 2**attempt)
    raise AssertionError("unreachable")


def canonical_hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def write(
    rows: list[dict[str, object]],
    output: Path,
    manifest_path: Path,
    source_requests: list[dict[str, str]],
) -> dict[str, object]:
    buffer = io.StringIO(newline="")
    columns = ["reference_date", "vintage_date", "available", "nfci"]
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    payload = buffer.getvalue().encode()
    output.write_bytes(payload)
    core = {
        "protocol_version": "chicago_fed_nfci_alfred_causal_vintages_v1",
        "series": "NFCI",
        "availability_rule": "observation Friday W read only from ALFRED vintage W+7 Friday",
        "latest_vintage_backfill": False,
        "raw_responses_persisted": False,
        "bulk_responses": source_requests,
        "rows": len(rows),
        "available_rows": sum(bool(row["available"]) for row in rows),
        "unavailable_reference_dates": [row["reference_date"] for row in rows if not row["available"]],
        "first_reference_date": rows[0]["reference_date"],
        "last_reference_date": rows[-1]["reference_date"],
        "first_vintage_date": rows[0]["vintage_date"],
        "last_vintage_date": rows[-1]["vintage_date"],
        "output": {"path": str(output), "sha256": sha256(payload), "columns": columns},
        "source_url_template": BASE_URL,
        "outcomes_opened": False,
        "btc_price_return_funding_or_pnl_opened": False,
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    references = fridays(FIRST_REFERENCE, LAST_REFERENCE)
    rows: list[dict[str, object]] = []
    source_requests: list[dict[str, str]] = []
    for start in range(0, len(references), args.chunk_size):
        chunk = references[start : start + args.chunk_size]
        chunk_rows, source_url, response_sha256 = fetch_bulk(chunk)
        rows.extend(chunk_rows)
        source_requests.append({"url": source_url, "response_sha256": response_sha256})
        print(f"downloaded {len(rows)}/{len(references)}", flush=True)
    manifest = write(rows, args.output, args.manifest, source_requests)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
