from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from training import (
    build_funding_currency_custody_mobility_consensus_support as support,
)


CLOCK = Path(
    "data/funding_currency_custody_mobility_consensus_2021_2023/"
    "fccm72_support_clocks_2021_2023.csv.gz"
)
REPORT = Path(
    "results/funding_currency_custody_mobility_consensus_support_2026-07-23.json"
)
CLOCK_SHA256 = (
    "71180862d9dcc4d76e055c52fd72a2424ee12387a6b8062af8a9382675af3810"
)
REPORT_SHA256 = (
    "0cc2f741a9c174f13a050d73df7c668bdb9776c9c431b89bfb88cd814f899266"
)
MANIFEST_HASH = (
    "f88cae078ce090702b3fb874695f3ec50eafef483d076784e882b3dfe56f09c6"
)
FAILED_CHECKS = {
    "each_selection_half_minimum",
    "each_train_half_minimum",
    "each_train_year_minimum",
    "maximum_consecutive_same_side",
    "selection_each_side_minimum",
    "selection_every_quarter_active",
    "selection_maximum_entry_gap",
    "selection_maximum_month_share",
    "selection_total_minimum",
    "selection_wbtc_raw_transition_active_share",
    "train_every_quarter_active",
    "train_maximum_entry_gap",
    "train_maximum_month_share",
    "train_total_minimum",
}


def test_fccm_source_support_artifacts_are_exact_and_reproducible() -> None:
    clock_raw = CLOCK.read_bytes()
    report_raw = REPORT.read_bytes()
    assert hashlib.sha256(clock_raw).hexdigest() == CLOCK_SHA256
    assert hashlib.sha256(report_raw).hexdigest() == REPORT_SHA256
    payload = json.loads(report_raw)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert support.canonical_hash(core) == MANIFEST_HASH
    assert payload["clock"]["sha256"] == CLOCK_SHA256
    assert payload["clock"]["rows"] == 760

    reproduced, status = support.write_support()
    assert status == "verified_existing"
    assert reproduced == payload

    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        assert tuple(next(reader)) == support.CLOCK_COLUMNS
        assert sum(1 for _ in reader) == 760


def test_fccm_source_support_is_terminally_rejected_without_outcomes() -> None:
    payload = json.loads(REPORT.read_bytes())
    failed = {name for name, passed in payload["support_checks"].items() if not passed}

    assert payload["candidate"] == "FCCM-72"
    assert payload["artifact_eligible"] is True
    assert payload["source_support_passed"] is False
    assert payload["advance_to_novelty_evaluator"] is False
    assert payload["decision"] == (
        "retire_FCCM_72_unchanged_before_comparators_and_outcomes"
    )
    assert failed == FAILED_CHECKS
    assert sum(payload["support_checks"].values()) == 14

    train = payload["primary_support"]["train"]
    selection = payload["primary_support"]["selection"]
    assert train["accepted_entries"] == 46
    assert train["side_counts"] == {"long": 21, "short": 25}
    assert selection["accepted_entries"] == 11
    assert selection["side_counts"] == {"long": 4, "short": 7}
    assert payload["primary_support"]["raw_sponsorship"]["train"][
        "wbtc_active_share"
    ] == "145/618"
    assert payload["primary_support"]["raw_sponsorship"]["selection"][
        "wbtc_active_share"
    ] == "53/439"

    boundary = payload["outcome_boundary"]
    assert boundary["post_2023_source_value_rows_read"] == 0
    assert boundary["comparator_value_rows_read"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["realized_funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_values_opened"] == 0
