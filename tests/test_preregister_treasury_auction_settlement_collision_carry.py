from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_treasury_auction_settlement_collision_carry as prereg


def test_preregistration_discloses_heavily_seen_source_and_unopened_candidate() -> None:
    payload = prereg.build_preregistration()
    prereg.validate_preregistration(payload)
    assert payload["candidate"] == "TASCC-72-SOURCE-FAMILY-SEEN"
    assert payload["source_family_values_previously_opened"] is True
    assert payload["source_family_market_outcomes_previously_opened"] is True
    assert payload["exact_source_incidence_opened"] is False
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    disclosure = payload["prior_research_disclosure"]
    assert disclosure["tadi_2021_2022_btc_outcomes_opened_and_failed"] is True
    assert disclosure["tascc_market_outcomes_opened"] is False
    assert disclosure["pristine_source_family_claim"] is False


def test_policy_freezes_collision_geometry_and_causal_execution() -> None:
    policy = prereg.policy_payload()
    primary = policy["primary_clock"]
    execution = policy["execution"]
    assert policy["source_rows"]["belly_terms"] == ["5-Year", "7-Year"]
    assert policy["source_rows"]["long_terms"] == [
        "10-Year",
        "20-Year",
        "30-Year",
    ]
    assert primary["grouping_key"] == "exact issueDate"
    assert primary["component_result_availability"] == (
        "each result_available_at_utc <= settlement marker"
    )
    assert primary["side"] == "SHORT"
    assert execution["entry_time"] == "ceil_to_5m(signal_time) + 5 elapsed minutes"
    assert execution["exact_grid_signal_still_waits_one_bar"] is True
    assert execution["hold_elapsed_hours"] == 72
    assert execution["hold_bars_5m"] == 864
    assert execution["global_nonoverlap"] is True
    assert execution["side_override"] is False


def test_controls_support_novelty_economics_and_rllm_are_fixed() -> None:
    policy = prereg.policy_payload()
    assert set(policy["source_controls"]) == {
        "belly_settlement_calendar",
        "long_settlement_calendar",
        "any_multitenor_settlement",
        "single_tenor_settlement",
        "auction_date_collision",
        "term_year_permutation",
        "result_time_clock",
        "settlement_plus_7d",
    }
    support = policy["source_support_gates"]
    assert support["train_total_minimum"] == 18
    assert support["selection_total_minimum"] == 8
    assert support["maximum_calendar_gap_days"] == 90
    specificity = policy["mechanism_specificity_gates"]
    assert "result_time_clock" in specificity[
        "component_and_superset_controls_are_report_only"
    ]
    novelty = policy["novelty"]
    assert novelty["maximum_exact_entry_jaccard"] == 0.10
    assert novelty["maximum_tascc_to_comparator_near_containment"] == 0.35
    assert novelty["comparators"] == [spec["name"] for spec in prereg.COMPARATOR_SPECS]
    strict = policy["strict_economic_gates"]
    assert strict["cagr_to_strict_mdd_minimum"] == 3.0
    assert strict["strict_mdd_pct_maximum"] == 15.0
    assert strict["full_calendar_cagr"] is True
    assert policy["rllm_boundary"]["later_actions"] == [
        "TRADE_FIXED_SHORT",
        "ABSTAIN",
    ]


def test_bindings_are_hash_only_during_preregistration() -> None:
    payload = prereg.build_preregistration()
    source = payload["source_binding"]
    assert source["panel_value_rows_read_during_preregistration"] == 0
    assert source["raw_value_rows_read_during_preregistration"] == 0
    assert all(row["value_rows_read_during_preregistration"] == 0 for row in source["raw_pages"])
    assert all(
        row["value_rows_read_during_preregistration"] == 0
        for row in payload["comparator_bindings"]
    )
    assert all(
        row["values_read_during_tascc_preregistration"] == 0
        for row in payload["history_bindings"]
    )


def test_tampering_fails_frozen_policy_validation() -> None:
    payload = prereg.build_preregistration()
    payload["policy"]["execution"]["hold_elapsed_hours"] = 24
    with pytest.raises(RuntimeError, match="frozen policy drift"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_false_pristine_claim_is_rejected() -> None:
    payload = prereg.build_preregistration()
    payload["prior_research_disclosure"]["pristine_source_family_claim"] = True
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = prereg.canonical_hash(core)
    with pytest.raises(RuntimeError, match="prior-research disclosure drift"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_write_once_verifies_identical_and_refuses_drift(tmp_path: Path) -> None:
    relative = Path(".pytest_cache") / f"{tmp_path.name}-tascc-preregistration.json"
    output = prereg.REPOSITORY_ROOT / relative
    output.unlink(missing_ok=True)
    cfg = prereg.Config(output=str(relative))
    try:
        payload, status = prereg.write_preregistration(cfg)
        assert status == "created"
        second, status = prereg.write_preregistration(cfg)
        assert status == "verified_existing"
        assert second == payload
        stored = json.loads(output.read_text(encoding="utf-8"))
        stored["manifest_hash"] = "0" * 64
        output.write_text(json.dumps(stored), encoding="utf-8")
        with pytest.raises(RuntimeError, match="canonical hash mismatch"):
            prereg.write_preregistration(cfg)
    finally:
        output.unlink(missing_ok=True)


def test_repository_path_rejects_escape() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        prereg._repository_path("../outside.json")
