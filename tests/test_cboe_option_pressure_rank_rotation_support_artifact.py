from __future__ import annotations

import json
from pathlib import Path

from training import build_cboe_option_pressure_rank_rotation_support as s


REPORT = Path(
    "results/cboe_option_pressure_rank_rotation_support_2026-07-24.json"
)
CLOCK = Path(
    "data/cboe_option_pressure_rank_rotation_clocks_2020_2023.csv.gz"
)
REPORT_SHA256 = (
    "bf0e15622a52ae6f748fea87abde3728ac550b3d69e52187d8a500c29c8d8968"
)
CLOCK_SHA256 = (
    "a5c15e0d6444f79239276fb9c3da0555dea27a52eda254e7425d9b223d30d46c"
)
MANIFEST_HASH = (
    "190c2447bfe79d190ff14f4de988e1ef5f934367b50b6b54d3bafa99164cf7d4"
)


def test_frozen_oprr_support_artifacts_record_outcome_blind_rejection() -> None:
    assert s.sha256_file(REPORT) == REPORT_SHA256
    assert s.sha256_file(CLOCK) == CLOCK_SHA256
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["policy_id"] == "OPRR-288"
    assert payload["source_support_passed"] is False
    assert payload["composition_passed"] is False
    assert payload["novelty_passed"] is False
    assert payload["first_failing_stage"] == "source_support"
    assert payload["first_failing_check"] == "train_events_min"
    assert payload["comparator_status"] == (
        "not_opened_source_support_or_composition_failed"
    )
    assert payload["comparator_rows_decoded"] == 0
    assert payload["outcomes_opened"] is False
    assert payload["funding_loaded"] is False
    boundary = payload["outcome_boundary"]
    assert boundary["BTC_market_rows_decoded"] == 0
    assert boundary["funding_rows_decoded"] == 0
    assert boundary["future_return_rows_decoded"] == 0
    assert boundary["return_or_PnL_fields_decoded"] == 0
    assert boundary["PnL_CAGR_MDD_values_decoded"] == 0
    assert payload["primary_statistics"]["train"]["events"] == 11
    assert payload["primary_statistics"]["selection"]["events"] == 13
    assert payload["feature_funnel"]["adjacent_rank_complete_transitions"] == 878
    assert payload["clock"]["rows"] > 0
    assert payload["clock"]["sha256"] == CLOCK_SHA256
    assert payload["implementation"]["source_sha256"] == s.sha256_file(
        s.SCRIPT_PATH
    )
    assert payload["implementation"]["tests_sha256"] == s.sha256_file(
        s.TEST_PATH
    )
    assert payload["implementation"]["contract_sha256"] == (
        s.IMPLEMENTATION_CONTRACT_SHA256
    )
