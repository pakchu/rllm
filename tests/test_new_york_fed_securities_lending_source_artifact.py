from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from training import build_new_york_fed_securities_lending as source


ROOT = Path("data/new_york_fed_securities_lending_2019_2023")
MANIFEST = ROOT / "build_manifest.json"
OPERATIONS = ROOT / "new_york_fed_securities_lending_operations_2019_2023.csv.gz"
DETAILS = ROOT / "new_york_fed_securities_lending_details_2019_2023.csv.gz"
EXPECTED_MANIFEST_FILE_SHA = (
    "58b9eb56728065d919978b8969e9bbb4bcb291f723a290d22045fe2ca3da2019"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_manifest_and_panels_match_hashes() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert sha(MANIFEST) == EXPECTED_MANIFEST_FILE_SHA
    assert manifest["manifest_hash"] == source.canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    assert manifest["builder"]["sha256"] == sha(
        Path(manifest["builder"]["path"])
    )
    assert manifest["source_decision"]["sha256"] == sha(
        Path(manifest["source_decision"]["path"])
    )
    assert manifest["operations"]["sha256"] == sha(OPERATIONS)
    assert manifest["details"]["sha256"] == sha(DETAILS)
    assert manifest["operations"]["rows"] == 1259
    assert manifest["details"]["rows"] == 182616
    assert manifest["details"]["unique_operation_cusip_rows"] == 182616
    assert all(manifest["source_checks"].values())


def test_frozen_raw_cache_replays_exact_ledger() -> None:
    ledger = json.loads((ROOT / "raw/fetch_ledger.json").read_text())
    assert [row["name"] for row in ledger] == [
        "openapi",
        "operations_2019",
        "operations_2020",
        "operations_2021",
        "operations_2022",
        "operations_2023",
    ]
    raw_paths = source._raw_paths(ROOT)
    for row in ledger:
        payload = gzip.decompress(raw_paths[row["name"]].read_bytes())
        assert row["bytes"] == len(payload)
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()
    payloads = {
        name: gzip.decompress(path.read_bytes()) for name, path in raw_paths.items()
    }
    assert source._validate_cached_ledger(ledger, payloads) == ledger


def test_operation_calendar_and_headers_are_frozen() -> None:
    with gzip.open(OPERATIONS, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == source.OPERATION_COLUMNS
    assert rows[0]["operation_date"] == "2019-01-02"
    assert rows[-1]["operation_date"] == "2023-12-29"
    assert Counter(row["operation_date"][:4] for row in rows) == {
        "2019": 252,
        "2020": 253,
        "2021": 252,
        "2022": 251,
        "2023": 251,
    }
    assert len({row["operation_id"] for row in rows}) == len(rows)


def test_detail_nulls_preserve_only_zero_award_na_rates() -> None:
    rows = 0
    missing_rates = 0
    identities: set[tuple[str, str]] = set()
    with gzip.open(DETAILS, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == source.DETAIL_COLUMNS
        for row in reader:
            rows += 1
            identity = (row["operation_id"], row["cusip"])
            assert identity not in identities
            identities.add(identity)
            if not row["weighted_average_rate"]:
                missing_rates += 1
                assert row["par_accepted"] == "0"
            for field in (
                "par_submitted",
                "par_accepted",
                "soma_holdings",
                "theoretical_available_to_borrow",
                "actual_available_to_borrow",
                "outstanding_loans",
            ):
                assert row[field] != ""
    assert rows == 182616
    assert missing_rates == 744


def test_manifest_keeps_candidate_and_outcome_boundary_closed() -> None:
    boundary = json.loads(MANIFEST.read_text())["research_boundary"]
    assert boundary == {
        "candidate_features_computed": [],
        "candidate_incidence_opened": False,
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "return_rows_read": 0,
        "pnl_cagr_mdd_opened": False,
    }
