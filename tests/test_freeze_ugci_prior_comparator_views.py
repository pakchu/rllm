from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path

import pytest

from training import freeze_ugci_prior_comparator_views as freeze


def _write_comparator(path: Path) -> str:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("candidate", "control", "entry_time")
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "candidate": "X-1",
                    "control": "primary",
                    "entry_time": "2023-08-31T23:55:00Z",
                },
                {
                    "candidate": "X-1",
                    "control": "primary",
                    "entry_time": "2023-09-01T00:00:00Z",
                },
                {
                    "candidate": "X-1",
                    "control": "primary",
                    "entry_time": "2023-12-31T23:55:00Z",
                },
                {
                    "candidate": "X-1",
                    "control": "primary",
                    "entry_time": "2024-01-01T00:00:00Z",
                },
            ]
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sanitizer_retains_only_the_frozen_half_open_interval(tmp_path: Path) -> None:
    source = tmp_path / "comparator.csv.gz"
    digest = _write_comparator(source)
    rows, audit = freeze.sanitize_comparator(
        {
            "candidate": "X-1",
            "path": str(source),
            "sha256": digest,
            "controls": ["primary"],
            "entry_column": "entry_time",
            "comparison_start": "2023-09-01T00:00:00Z",
            "comparison_end_exclusive": "2024-01-01T00:00:00Z",
        }
    )
    assert [row["entry_time"] for row in rows] == [
        "2023-09-01T00:00:00Z",
        "2023-12-31T23:55:00Z",
    ]
    assert audit["source_physical_rows_read_for_sanitization"] == 4
    assert audit["retained_rows"] == 2


def test_sanitizer_rejects_duplicate_retained_clocks(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("candidate", "control", "entry_time")
        )
        writer.writeheader()
        for _ in range(2):
            writer.writerow(
                {
                    "candidate": "X-1",
                    "control": "primary",
                    "entry_time": "2023-09-01T00:00:00Z",
                }
            )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate clocks"):
        freeze.sanitize_comparator(
            {
                "candidate": "X-1",
                "path": str(source),
                "sha256": digest,
                "controls": ["primary"],
                "entry_column": "entry_time",
                "comparison_start": "2023-09-01T00:00:00Z",
                "comparison_end_exclusive": "2024-01-01T00:00:00Z",
            }
        )


def test_preregistration_comparator_contract_remains_hash_bound() -> None:
    payload = freeze.validate_preregistration()
    assert payload["candidate"] == "UGCI-288"
    assert payload["novelty_comparators"]


def test_deterministic_gzip_has_stable_bytes() -> None:
    rows = [
        {
            "candidate": "X-1",
            "control": "primary",
            "entry_time": "2023-09-01T00:00:00Z",
            "comparison_start": "2023-09-01T00:00:00Z",
            "comparison_end_exclusive": "2024-01-01T00:00:00Z",
        }
    ]
    assert freeze.deterministic_gzip_csv(rows) == freeze.deterministic_gzip_csv(rows)
