from __future__ import annotations

import hashlib
import json
from pathlib import Path


MANIFEST = Path(
    "results/bitmex_xbt_insurance_fund_source_manifest_2026-07-20.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bitmex_xbt_insurance_source_manifest_is_hash_bound() -> None:
    assert _sha256(MANIFEST) == (
        "c9b8df43a07a5f6887cc43dea300698af7d455c70bfc582899504fa3eb6dda6e"
    )
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["manifest_hash"] == (
        "4c751b96a4d877bc558bf37e693396fc529326feb050862ea5ddb9100cde8612"
    )
    assert manifest["config"] == {
        "output_csv": "data/bitmex_xbt_insurance_fund_2018_2022.csv.gz",
        "manifest_output": (
            "results/bitmex_xbt_insurance_fund_source_manifest_2026-07-20.json"
        ),
        "start": "2018-01-01",
        "end_exclusive": "2023-01-01",
        "currency": "XBt",
        "page_size": 500,
        "timeout_sec": 30.0,
    }
    assert manifest["output"]["sha256"] == (
        "523d179d4a4ac51e3ebf5ce24f188f23cda02f31d8f879e0d256361af333c6dc"
    )


def test_bitmex_xbt_insurance_source_grid_is_complete_and_private() -> None:
    manifest = json.loads(MANIFEST.read_text())
    audit = manifest["source_audit"]
    assert audit["page_lengths"] == [500, 500, 500, 326]
    assert audit["rows_received"] == 1826
    assert audit["rows_selected"] == 1826
    assert audit["expected_days"] == 1826
    assert audit["complete_daily_noon_utc_grid"] is True
    assert audit["start"] == "2018-01-01 12:00:00+00:00"
    assert audit["end"] == "2022-12-31 12:00:00+00:00"
    assert audit["response_currency"] == "XBt"
    assert manifest["data_use"].startswith("private internal research")
