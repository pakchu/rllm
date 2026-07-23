from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from training import build_usdc_recipient_concentration_dislocation_support as support


CLOCK = Path(
    "data/usdc_recipient_concentration_dislocation_2021_2023/"
    "urcd72_support_clocks_2021_2023.csv.gz"
)
REPORT = Path(
    "results/usdc_recipient_concentration_dislocation_support_2026-07-23.json"
)
CLOCK_SHA256 = (
    "ad9617ec5af0c0189aa384a49ab9244e957758f7c8abe71b6b61e911b7663ea1"
)
REPORT_SHA256 = (
    "648825052812a8f436b8e7743973f3d6edcf4e013a767082103c98707e66f998"
)
MANIFEST_HASH = (
    "1ad500d2c60a18ce12345804beb8465bfdf00febd60e86d21f1eeedc94d2d685"
)
FAILED_CHECKS = {
    "amount_year_permutation_train_exact_entry_jaccard",
    "amount_year_permutation_train_same_side_reproduction",
    "each_selection_half_minimum",
    "each_train_half_minimum",
    "each_train_year_minimum",
    "maximum_gap_days",
    "maximum_month_share",
    "maximum_quarter_share",
    "recipient_year_permutation_train_exact_entry_jaccard",
    "recipient_year_permutation_train_same_side_reproduction",
    "selection_each_side_minimum",
    "selection_each_side_share",
    "selection_total_minimum",
    "train_each_side_minimum",
    "train_each_side_share",
    "train_total_minimum",
}


def load_report() -> dict[str, object]:
    return json.loads(REPORT.read_text("utf-8"))


def test_urcd_source_support_artifacts_are_exact() -> None:
    clock_raw = CLOCK.read_bytes()
    report_raw = REPORT.read_bytes()
    assert hashlib.sha256(clock_raw).hexdigest() == CLOCK_SHA256
    assert hashlib.sha256(report_raw).hexdigest() == REPORT_SHA256

    payload = json.loads(report_raw)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert support.canonical_hash(core) == MANIFEST_HASH
    assert payload["clock"]["sha256"] == CLOCK_SHA256
    assert payload["clock"]["rows"] == 186
    assert payload["clock"]["control_counts"] == {
        "amount_year_permutation": 31,
        "direction_flip": 14,
        "equal_recipient_breadth": 26,
        "event_count_hhi": 19,
        "no_materiality": 27,
        "primary": 14,
        "recipient_year_permutation": 42,
        "stale_24h": 13,
    }

    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        assert tuple(next(reader)) == support.CLOCK_COLUMNS
        assert sum(1 for _ in reader) == 186


def test_urcd_is_terminally_rejected_before_comparators_and_outcomes() -> None:
    payload = load_report()
    failed = {
        name for name, passed in payload["support_checks"].items() if not passed
    }
    assert payload["candidate"] == "URCD-72"
    assert payload["artifact_eligible"] is True
    assert payload["source_support_passed"] is False
    assert payload["novelty_status"] == "not_opened_source_support_failed"
    assert payload["novelty_passed"] is False
    assert payload["decision"] == (
        "retire_URCD_72_unchanged_before_comparators_and_outcomes"
    )
    assert payload["advance_to_strict_outcome_evaluator_freeze"] is False
    assert failed == FAILED_CHECKS
    assert sum(payload["support_checks"].values()) == 5

    train = payload["primary_support"]["train"]
    selection = payload["primary_support"]["selection"]
    assert train["trades"] == 0
    assert train["side_counts"] == {}
    assert selection["trades"] == 14
    assert selection["side_counts"] == {"LONG": 2, "SHORT": 12}
    assert selection["half_year_counts"] == {"2023H2": 14}

    comparator = payload["comparator_audit"]
    assert comparator["physical_rows_scanned"] == 0
    assert comparator["relevant_four_field_rows_decoded"] == 0
    assert comparator["out_of_overlap_timestamp_sentinels_scanned"] == 0

    boundary = payload["outcome_boundary"]
    assert boundary["post_2023_source_value_rows_decoded"] == 0
    assert boundary["comparator_four_field_rows_decoded"] == 0
    assert boundary["btc_market_rows_decoded"] == 0
    assert boundary["funding_rows_decoded"] == 0
    assert boundary["future_return_rows_decoded"] == 0
    assert boundary["return_or_pnl_fields_decoded"] == 0
    assert boundary["pnl_cagr_mdd_values_decoded"] == 0
    assert boundary["network_calls"] == 0
    assert payload["source_audit"]["eligible_mint_value_rows_decoded"] == 99_033
