from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from training import freeze_ugci_prior_comparator_views as freeze


CLOCK = Path("results/ugci_prior_comparator_views_pre2024_2026-07-22.csv.gz")
MANIFEST = Path("results/ugci_prior_comparator_views_pre2024_manifest_2026-07-22.json")
EXPECTED_CLOCK_SHA256 = (
    "dfbf4808813c1b0db4c5a4f05af324473d3a92dfa5cdfc6581e1b07bc17271bd"
)
EXPECTED_MANIFEST_SHA256 = (
    "38abf60a8c9aa44c7fb53a5435f22cb650151b58e33a6fca1ffae1aeb36ed5c2"
)
EXPECTED_MANIFEST_HASH = (
    "a00301a229bc1c620f355cb42adc05b760d8734d7a903490ab2c1d3a0fd92d33"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_and_manifest_are_exactly_hash_bound() -> None:
    assert _sha256(CLOCK) == EXPECTED_CLOCK_SHA256
    assert _sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    unhashed = dict(payload)
    unhashed.pop("manifest_hash")
    assert freeze.canonical_hash(unhashed) == EXPECTED_MANIFEST_HASH
    assert payload["output"]["sha256"] == EXPECTED_CLOCK_SHA256
    assert payload["builder"]["sha256"] == _sha256(Path(payload["builder"]["path"]))


def test_bundle_is_timestamp_only_and_strictly_pre2024() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == freeze.CLOCK_COLUMNS
        rows = list(reader)
    assert len(rows) == payload["output"]["rows"] == 780
    assert all(
        row["comparison_start"]
        <= row["entry_time"]
        < row["comparison_end_exclusive"]
        <= "2024-01-01T00:00:00Z"
        for row in rows
    )
    assert len(
        {(row["candidate"], row["control"], row["entry_time"]) for row in rows}
    ) == len(rows)
    expected = Counter(
        {
            (audit["candidate"], control): count
            for audit in payload["inputs"]
            for control, count in audit["retained_counts"].items()
        }
    )
    actual = Counter((row["candidate"], row["control"]) for row in rows)
    assert actual == expected


def test_bundle_retains_no_current_candidate_or_outcome_field() -> None:
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["candidate"] != "UGCI-288" for row in rows)
    assert not {
        "price",
        "return",
        "pnl",
        "equity",
        "cagr",
        "mdd",
        "funding",
    } & set(freeze.CLOCK_COLUMNS)
