from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_dollar_collateral_liquidity_bank_relay_support as s


REPORT = Path(
    "results/dollar_collateral_liquidity_bank_relay_support_2026-07-24.json"
)
CLOCK = Path(
    "data/dollar_collateral_liquidity_bank_relay_clocks_2020_2023.csv.gz"
)
REPORT_SHA256 = (
    "05db079878d8ad218ab4350e79ec7899bb66d36de4ae4ecc8b0c0b884cd988c5"
)
CLOCK_SHA256 = (
    "1973a3a79c574b6cc53f93e36f2b6c1550f7d8050e3ac34fe903f28a0253cb37"
)


def test_rejected_dclb_support_artifacts_are_canonical_and_outcome_blind() -> None:
    assert s.sha256_file(REPORT) == REPORT_SHA256
    assert s.sha256_file(CLOCK) == CLOCK_SHA256
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    assert payload["manifest_hash"] == s.canonical_hash(core)
    assert payload["source_support_passed"] is False
    assert payload["relational_composition_passed"] is False
    assert payload["novelty_passed"] is False
    assert payload["first_failing_stage"] == "source_support"
    assert payload["first_failing_check"] == "train_maximum_entry_gap"
    assert payload["comparator_rows_decoded"] == 0
    assert payload["comparator_status"] == (
        "not_opened_source_support_or_composition_failed"
    )
    assert payload["decision"] == (
        "retire_DCLB_864_unchanged_before_comparators_and_outcomes"
    )
    boundary = payload["outcome_boundary"]
    assert boundary["btc_market_rows_loaded"] == 0
    assert boundary["funding_rows_loaded"] == 0
    assert boundary["future_return_rows_computed"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert boundary["post_2023_source_rows_loaded"] == 0
    assert boundary["network_calls"] == 0
    assert payload["clock"]["sha256"] == hashlib.sha256(
        CLOCK.read_bytes()
    ).hexdigest()


def test_dclb_failure_diagnostics_are_frozen() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    funnel = payload["feature_funnel"]
    assert funnel["common_causal_rows"] == 133
    assert funnel["raw_primary_eligible_rows"] == 107
    assert payload["clock"]["control_counts"]["primary"] == 105
    assert payload["source_statistics"]["train"]["events"] == 77
    assert payload["source_statistics"]["train"]["maximum_gap_days"] == 196.0
    assert payload["source_statistics"]["selection"]["events"] == 28
    assert payload["source_statistics"]["2023_q1"]["events"] == 0
    failed_source = {
        key
        for key, passed in payload["source_support_checks"].items()
        if not passed
    }
    assert failed_source == {
        "selection_each_quarter_events_min",
        "train_maximum_entry_gap",
    }
    failed_composition = {
        key
        for key, passed in payload["relational_composition_checks"].items()
        if not passed
    }
    assert failed_composition == {
        "selection:stale_rrp_one_interval_same_side_reproduction",
        "train:rrp_interval_only_same_side_reproduction",
    }
