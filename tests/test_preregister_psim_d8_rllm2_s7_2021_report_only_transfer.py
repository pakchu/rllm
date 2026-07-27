from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    preregister_psim_d8_rllm2_s7_2021_report_only_transfer as prereg,
)


def test_predecessor_pass_is_exact_and_outcome_closed() -> None:
    evidence = prereg.validate_s6r1_schedule_pass()
    result = evidence["result"]

    assert result["decision"] == "pass"
    assert result["schedule_readiness"]["passed"] is True
    assert result["authorize_separate_2021_transfer_preregistration"]
    boundary = result["access_boundary"]
    assert boundary["2021_market_rows_parsed"] == 0
    assert boundary["2021_funding_rows_parsed"] == 0
    assert boundary["2021_reward_rows_created"] == 0
    assert boundary["2021_economic_metrics_computed"] == 0


def test_stage_source_is_bound_without_payload_hash_or_parse() -> None:
    source = prereg.validate_stage_source_metadata()
    payload = prereg.build_preregistration()
    frozen = payload["frozen_2021_outcome_source"]

    assert source["strategy_outcomes_calculated"] is False
    assert frozen["payload_bytes_hashed_at_preregistration"] is False
    assert frozen["numeric_rows_parsed_at_preregistration"] == 0
    assert frozen["market"]["expected_rows"] == 105_120
    assert frozen["funding"]["expected_rows"] == 1_095
    assert prereg.sha256_file(prereg.STAGE_SOURCE_MANIFEST) == (
        prereg.STAGE_SOURCE_MANIFEST_SHA256
    )


def test_family_gate_calendar_cost_and_inference_are_fixed() -> None:
    payload = prereg.build_preregistration()

    family = payload["frozen_schedule_family"]
    assert family["family_ids"] == list(prereg.FAMILY_IDS)
    assert family["family_count"] == 41
    assert len(set(family["family_ids"])) == 41
    assert prereg.PRIMARY_POLICY_ID in family["family_ids"]
    assert set(prereg.NONSEMANTIC_CONTROL_IDS) < set(prereg.FAMILY_IDS)
    economic = payload["economic_contract"]
    assert economic["calendar_start"] == prereg.CALENDAR_START
    assert economic["calendar_end"] == prereg.CALENDAR_END
    assert economic["full_calendar_cagr_includes_flat_periods"] is True
    assert economic["base_cost_rate"] == 0.0006
    assert economic["stress_cost_rate"] == 0.0010
    gate = payload["pass_gate"]
    assert gate["base_cagr_to_strict_mdd_minimum"] == 1.0
    assert gate["minimum_nonflat_intervals"] == 80
    assert gate["minimum_long_share"] == 0.20
    assert gate["minimum_short_share"] == 0.20
    assert gate["familywise_p_max_strictly_below"] == 0.25
    inference = payload["familywise_inference"]
    assert inference["draws"] == 100_000
    assert inference["seed"] == 20_260_725


def test_candidate_is_report_only_and_cannot_repair_from_result() -> None:
    payload = prereg.build_preregistration()

    candidate = payload["candidate"]
    assert candidate["primary_policy_id"] == prereg.PRIMARY_POLICY_ID
    assert candidate["single_promotable_primary"] is True
    assert candidate["selection_or_repair_from_2021_metrics"] is False
    assert candidate["globally_pristine_2021_claim"] is False
    assert candidate["success_is_live_promotion"] is False
    execution = payload["execution_contract"]
    assert execution[
        "attempt_written_before_market_or_funding_open_or_read"
    ]
    assert execution[
        "attempt_written_before_market_or_funding_payload_hash_or_parse"
    ]
    assert execution["no_2022_or_later_outcome_access"] is True
    assert execution["no_selection_or_repair_after_result"] is True
    assert payload["access_boundary"][
        "raw_market_or_funding_paths_opened_or_read_before_attempt"
    ] == []


def test_preregistration_is_deterministic_self_hashed_and_write_once(
    tmp_path: Path,
) -> None:
    first = prereg.build_preregistration()
    second = prereg.build_preregistration()
    core = {
        key: value
        for key, value in first.items()
        if key != "manifest_hash"
    }

    assert first == second
    assert first["manifest_hash"] == prereg.canonical_hash(core)
    output = tmp_path / "registration.json"
    assert prereg.write_preregistration(output) == first
    assert json.loads(output.read_text(encoding="utf-8")) == first
    assert prereg.write_preregistration(output) == first
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="drift"):
        prereg.write_preregistration(output)
