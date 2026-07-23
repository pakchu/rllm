from __future__ import annotations

import json
from pathlib import Path

from training import build_cross_venue_intrinsic_clock_resolution_support as s


REPORT = Path(
    "results/cross_venue_intrinsic_clock_resolution_support_2026-07-24.json"
)
CLOCK = Path(
    "data/cross_venue_intrinsic_clock_resolution_clocks_2020_2023.csv.gz"
)
REPORT_SHA256 = (
    "d330cebf95a1af16162ac67847f81c2828a898d9a2e7cfe2a3efb12835523886"
)
CLOCK_SHA256 = (
    "9f05b372686805539dbf56fb9b7ea7a8f90f8887d6731e1a8e1b1c1db14d8c0e"
)


def test_frozen_cvicr_support_rejection_stays_outcome_blind() -> None:
    assert s.sha256_file(REPORT) == REPORT_SHA256
    assert s.sha256_file(CLOCK) == CLOCK_SHA256
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["manifest_hash"] == (
        "90c310db34b45b057140fcdd44e26536ee77260ec09a244c445fd9eb4cde454e"
    )
    assert payload["source_support_passed"] is False
    assert payload["first_failing_stage"] == "source_support"
    assert payload["first_failing_check"] == "train_events_min"
    assert payload["primary_statistics"]["train"]["events"] == 9
    assert payload["primary_statistics"]["selection"]["events"] == 0
    assert payload["comparator_status"] == (
        "not_opened_source_support_failed"
    )
    assert payload["comparator_rows_decoded"] == 0
    assert payload["outcomes_opened"] is False
    assert payload["post_entry_return_computed"] is False
    assert payload["funding_loaded"] is False
    assert payload["outcome_boundary"]["future_return_rows_decoded"] == 0
    assert payload["decision"] == (
        "retire_CVICR_72_unchanged_before_comparators_and_outcomes"
    )
