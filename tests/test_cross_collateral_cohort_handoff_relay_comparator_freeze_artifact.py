from __future__ import annotations

import json
from pathlib import Path

from training import cchr_comparator_clock_common as common
from training import (
    freeze_cross_collateral_cohort_handoff_relay_comparators as freeze,
)


ARTIFACT = Path(
    "results/cross_collateral_cohort_handoff_relay_comparator_freeze_2026-07-21.json"
)
ARTIFACT_SHA256 = "84e84efe1c5a86b8b5c0f03515ce72f3dbc00a57c332b233d33ba297080abd0d"
MANIFEST_HASH = "6f7e64f2b0038de67da40a0cd2256983d3b969c2de7ad2900c14bca60154e147"


def test_frozen_cchr_comparator_artifact_is_exact_and_outcome_blind() -> None:
    assert common.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    freeze.validate_freeze(payload, verify_files=True)
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["comparator_member_count"] == 62
    assert payload["outcomes_opened"] is False
    assert payload["authorization"]["outcome_evaluator"] is False
    assert payload["outcome_boundary"]["pure_clock_rows_read"] == 0
    assert payload["outcome_boundary"]["raw_input_rows_read"] == 0
    assert payload["outcome_boundary"]["return_or_pnl_fields_read"] == 0


def test_frozen_cchr_comparator_artifact_retains_exact_family_coverage() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert {
        family: (
            binding["required_bindings"]["member_count"],
            binding["clock_metadata"]["rows"],
        )
        for family, binding in payload["generated_families"].items()
    } == {
        "pdlh": (16, 1_013),
        "dtv": (24, 1_318),
        "far": (12, 0),
        "live": (3, 440),
    }
    assert {
        family: binding["member_count"]
        for family, binding in payload["legacy_comparators"].items()
    } == {"ccpr": 6, "dlpd": 1}
