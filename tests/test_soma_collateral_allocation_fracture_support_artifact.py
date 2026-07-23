from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_soma_collateral_allocation_fracture_support as s


REPORT = Path(
    "results/soma_collateral_allocation_fracture_support_2026-07-24.json"
)
CLOCK = Path(
    "data/soma_collateral_allocation_fracture_clocks_2020_2023.csv.gz"
)
REPORT_SHA256 = (
    "714263911817a8d4d7a820ff80a4315e6e7710223944e5338f11835cb7b976bd"
)
CLOCK_SHA256 = (
    "64e07005d70442bfa7a110b1e6bea9802ee94be16d95f6e7db9228f4790a28e6"
)


def test_rejected_scaf_support_artifacts_are_canonical_and_outcome_blind() -> None:
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
    assert payload["first_failing_check"] == (
        "selection_raw_consensus_share_max"
    )
    assert payload["comparator_rows_decoded"] == 0
    assert payload["comparator_status"] == (
        "not_opened_source_support_or_composition_failed"
    )
    assert payload["decision"] == (
        "retire_SCAF_48_unchanged_before_comparators_and_outcomes"
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


def test_scaf_failure_diagnostics_are_frozen() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    funnel = payload["feature_funnel"]
    assert funnel["causal_batches"] == 1249
    assert funnel["valid_batches"] == 1249
    assert funnel["valid_transitions"] == 1248
    assert funnel["raw_primary_opportunities"] == 660
    assert funnel["accepted_primary_rows"] == 368
    support = payload["source_support"]
    assert support["primary_clock"]["train"]["events"] == 259
    assert support["primary_clock"]["selection"]["events"] == 109
    assert support["raw_consensus_share"]["selection"] == (
        169 / 248
    )
    failed_source = {
        key
        for key, passed in payload["source_support_checks"].items()
        if not passed
    }
    assert failed_source == {"selection_raw_consensus_share_max"}
    failed_composition = {
        key
        for key, passed in payload["relational_composition_checks"].items()
        if not passed
    }
    assert failed_composition == {
        "train:award_distortion:agreement_max",
        "train:unmet_demand_mass:agreement_min",
    }
