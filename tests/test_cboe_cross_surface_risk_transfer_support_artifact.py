from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_cboe_cross_surface_risk_transfer_support as s


REPORT = Path(
    "results/cboe_cross_surface_risk_transfer_support_2026-07-24.json"
)
CLOCK = Path(
    "data/cboe_cross_surface_risk_transfer_clocks_2020_2023.csv.gz"
)
REPORT_SHA256 = (
    "58b33695b861a213af1f177adb986b2025c4276116f2ebbc6a461278456e678d"
)
CLOCK_SHA256 = (
    "b3cc6f3d6a19cb39ef63ec0ba9908c983ce03c56a0c7dd8786e51c2ef1c0885f"
)


def test_rejected_cxrt_support_artifacts_are_canonical_and_outcome_blind() -> None:
    assert s.sha256_file(REPORT) == REPORT_SHA256
    assert s.sha256_file(CLOCK) == CLOCK_SHA256
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    assert payload["manifest_hash"] == s.canonical_hash(core)
    assert payload["source_support_passed"] is False
    assert payload["composition_passed"] is False
    assert payload["novelty_passed"] is False
    assert payload["first_failing_stage"] == "source_support"
    assert payload["first_failing_check"] == "selection_max_same_side_run"
    assert payload["comparator_rows_decoded"] == 0
    assert payload["comparator_status"] == (
        "not_opened_source_support_or_composition_failed"
    )
    assert payload["decision"] == (
        "retire_CXRT_288_unchanged_before_comparators_and_outcomes"
    )
    boundary = payload["outcome_boundary"]
    assert boundary["comparator_rows_decoded"] == 0
    assert boundary["BTC_market_rows_decoded"] == 0
    assert boundary["funding_rows_decoded"] == 0
    assert boundary["future_return_rows_decoded"] == 0
    assert boundary["return_or_PnL_fields_decoded"] == 0
    assert boundary["PnL_CAGR_MDD_values_decoded"] == 0
    assert boundary["network_calls"] == 0
    assert payload["clock"]["sha256"] == hashlib.sha256(CLOCK.read_bytes()).hexdigest()


def test_cxrt_failure_diagnostics_are_frozen() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["feature_funnel"]["rank_complete_common_dates"] == 879
    assert payload["primary_statistics"]["train"]["events"] == 498
    assert payload["primary_statistics"]["selection"]["events"] == 246
    assert (
        payload["primary_statistics"]["selection"]["maximum_same_side_run"]
        == 30
    )
    failed_composition = {
        key
        for key, passed in payload["composition_checks"].items()
        if not passed
    }
    assert failed_composition == {
        "train:unanimous_share",
        "train:option_only_same_side_reproduction",
        "selection:tail_relief_share",
        "selection:option_only_same_side_reproduction",
    }
