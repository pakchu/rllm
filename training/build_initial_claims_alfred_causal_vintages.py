"""Freeze first-print and first-revised ICSA values from decision-date ALFRED vintages."""
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

BASE_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
DEFAULT_OUTPUT = Path("data/initial_claims_alfred_causal_vintages_2020_2026.csv")
DEFAULT_MANIFEST = Path("data/initial_claims_alfred_causal_vintages_2020_2026_manifest.json")
FIRST_REFERENCE = date(2020, 1, 4)
LAST_REFERENCE = date(2026, 7, 18)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def saturdays(start: date, end: date) -> list[date]:
    if start.weekday() != 5 or end.weekday() != 5:
        raise ValueError("reference endpoints must be Saturdays")
    return [start + timedelta(days=7 * i) for i in range((end - start).days // 7 + 1)]


def bulk_url(references: list[date]) -> str:
    vintages = [v for reference in references for v in (reference + timedelta(days=5), reference + timedelta(days=12))]
    series = ",".join("ICSA" for _ in vintages)
    vintage_dates = ",".join(v.isoformat() for v in vintages)
    return (
        f"{BASE_URL}?id={series}&cosd={references[0].isoformat()}"
        f"&coed={references[-1].isoformat()}&vintage_date={vintage_dates}"
    )


def parse_bulk(payload: bytes, references: list[date]) -> list[dict[str, object]]:
    records = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    by_date = {record["observation_date"]: record for record in records}
    rows: list[dict[str, object]] = []
    for reference in references:
        first_vintage = reference + timedelta(days=5)
        revised_vintage = reference + timedelta(days=12)
        first_column = f"ICSA_{first_vintage:%Y%m%d}"
        revised_column = f"ICSA_{revised_vintage:%Y%m%d}"
        if records and (first_column not in records[0] or revised_column not in records[0]):
            raise ValueError(f"missing ALFRED vintage columns for {reference}")
        record = by_date.get(reference.isoformat(), {})
        first = record.get(first_column, ".")
        revised = record.get(revised_column, ".")
        rows.append(
            {
                "reference_date": reference.isoformat(),
                "first_vintage_date": first_vintage.isoformat(),
                "revised_vintage_date": revised_vintage.isoformat(),
                "first_available": first not in {"", ".", None},
                "revised_available": revised not in {"", ".", None},
                "icsa_first": float(first) if first not in {"", ".", None} else "",
                "icsa_revised": float(revised) if revised not in {"", ".", None} else "",
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


def write(rows: list[dict[str, object]], output: Path, manifest_path: Path, requests: list[dict[str, str]]) -> dict[str, object]:
    columns = ["reference_date", "first_vintage_date", "revised_vintage_date", "first_available", "revised_available", "icsa_first", "icsa_revised"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode()
    output.write_bytes(payload)
    core = {
        "protocol_version": "initial_claims_alfred_causal_vintages_v1",
        "series": "ICSA",
        "availability_rule": "Saturday D first print from D+5 Thursday vintage and first revision from D+12 Thursday decision vintage",
        "latest_vintage_backfill": False,
        "raw_responses_persisted": False,
        "bulk_responses": requests,
        "rows": len(rows),
        "first_reference_date": rows[0]["reference_date"],
        "last_reference_date": rows[-1]["reference_date"],
        "unavailable_revised_dates": [row["reference_date"] for row in rows if not row["revised_available"]],
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
    parser.add_argument("--chunk-size", type=int, default=20)
    args = parser.parse_args()
    references = saturdays(FIRST_REFERENCE, LAST_REFERENCE)
    rows: list[dict[str, object]] = []
    requests: list[dict[str, str]] = []
    for start in range(0, len(references), args.chunk_size):
        chunk = references[start : start + args.chunk_size]
        chunk_rows, url, response_sha = fetch_bulk(chunk)
        rows.extend(chunk_rows)
        requests.append({"url": url, "response_sha256": response_sha})
        print(f"downloaded {len(rows)}/{len(references)}", flush=True)
    print(json.dumps(write(rows, args.output, args.manifest, requests), indent=2))


if __name__ == "__main__":
    main()
