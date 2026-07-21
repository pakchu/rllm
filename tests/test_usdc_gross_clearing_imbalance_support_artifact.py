from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from training import build_usdc_gross_clearing_imbalance_support as support


CLOCK = Path("data/usdc_gross_clearing_imbalance_clocks_2021_2023.csv.gz")
REPORT = Path("results/usdc_gross_clearing_imbalance_support_2026-07-22.json")
EXPECTED_CLOCK_SHA256 = (
    "a0f861c69ac171e1efa665dc90a916d0351413ca07e5e46783bb8abd662175fd"
)
EXPECTED_REPORT_SHA256 = (
    "b61fc80bc879f15e9f1d15ac135ecbbda9384301cb8889def5b5a502af6068fa"
)
EXPECTED_MANIFEST_HASH = (
    "bc6b79c83e176ebd110a39fdf126b86504c5c9ce9411df65a5fb46670170a8f4"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_artifacts_are_exactly_hash_bound() -> None:
    assert _sha256(CLOCK) == EXPECTED_CLOCK_SHA256
    assert _sha256(REPORT) == EXPECTED_REPORT_SHA256
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    unhashed = dict(payload)
    unhashed.pop("manifest_hash")
    assert support.canonical_hash(unhashed) == EXPECTED_MANIFEST_HASH
    assert payload["clock"]["sha256"] == EXPECTED_CLOCK_SHA256
    assert payload["evaluator_source_sha256"] == _sha256(
        Path(payload["evaluator_source"])
    )


def test_support_verdict_retires_without_opening_outcomes_or_novelty() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["decision"] == "retire_UGCI_288_without_repair"
    assert payload["source_support_passed"] is False
    assert payload["support_passed_before_novelty"] is False
    assert payload["advance_to_strict_evaluator_freeze"] is False
    assert payload["novelty"] == {}
    assert payload["novelty_checks"] == {}
    assert payload["outcome_boundary"] == {
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "network_calls": 0,
        "original_comparator_files_opened": 0,
        "outcomes_opened": False,
        "post_2023_comparator_rows_parsed": 0,
        "post_2023_contract_event_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "sealed_comparator_bundle_files_opened": 0,
        "sealed_comparator_rows_parsed": 0,
        "subprocess_calls": 0,
    }


def test_clock_is_pre2024_and_matches_frozen_counts() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == support.CLOCK_COLUMNS
        rows = list(reader)
    assert len(rows) == payload["clock"]["rows"] == 1_245
    assert all(row["exit_time"] <= "2024-01-01T00:00:00Z" for row in rows)
    assert payload["clock"]["control_counts"] == {
        "primary": 98,
        "no_gross_tail": 784,
        "no_imbalance_floor": 265,
        "stale_6h": 98,
    }
    assert payload["primary_support"]["train_2021_2022"]["events"] == 87
    assert payload["primary_support"]["selection_2023"]["events"] == 11
    assert not all(payload["support_checks"].values())
