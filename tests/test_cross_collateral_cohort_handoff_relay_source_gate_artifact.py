from __future__ import annotations

import json
from pathlib import Path

from training import cchr_comparator_clock_common as common
from training import (
    evaluate_cross_collateral_cohort_handoff_relay_source_gate as gate,
)


ARTIFACT = Path(
    "results/cross_collateral_cohort_handoff_relay_source_gate_2026-07-21.json"
)
ARTIFACT_SHA256 = "a3da385c74df7334e984d6e6300050d6fe65515ce63b8a8f3c257c02dc88eb47"
MANIFEST_HASH = "e08e6da566cc665d0141b878536d06d3f7fa1f690573fa179148996d8562a618"


def test_frozen_cchr_source_gate_artifact_retires_without_outcomes() -> None:
    assert common.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    gate.validate_report(payload, verify_files=True)
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["decision"]["status"] == "retired_before_real_incidence"
    assert payload["decision"]["failed_family"] == "far"
    assert payload["decision"]["failed_member_count"] == 12
    assert payload["decision"]["cchr_source_incidence_opened"] is False
    assert payload["decision"]["economic_outcomes_opened"] is False
    assert payload["outcomes_opened"] is False


def test_frozen_cchr_source_gate_artifact_names_exact_empty_members() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    failed = payload["failed_required_members"]
    assert len(failed) == 12
    assert failed == payload["family_precheck"]["far"]["zero_row_members"]
    assert payload["family_precheck"]["far"]["coverage_rows"] == {
        "train": 0,
        "selection": 0,
    }
    assert payload["authorization"]["outcome_evaluator"] is False
    assert payload["authorization"]["next_action"] == (
        "new independently preregistered alpha only"
    )
