from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_wbtc_stablecoin_finalized_confirmation_relay as prereg


def test_preregistration_discloses_source_family_seen_outcome_blind_status() -> None:
    payload = prereg.build_preregistration()
    prereg.validate_preregistration(payload)
    assert payload["candidate"] == "WSCF-72-SOURCE-FAMILY-SEEN"
    assert payload["policy"]["research_status"] == (
        "source-family-seen_candidate-outcome-blind"
    )
    assert payload["policy"]["source_family_hypothesis_number"] == 3
    assert payload["source_family_values_previously_opened"] is True
    assert payload["exact_source_incidence_opened"] is False
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    assert payload["prior_research_disclosure"] == prereg.PRIOR_RESEARCH_DISCLOSURE
    assert payload["outcome_boundary"] == prereg.EXPECTED_OUTCOME_BOUNDARY
    assert all(
        source["value_rows_read_during_preregistration"] == 0
        for source in payload["source_bindings"].values()
    )
    assert all(
        view["value_rows_read_during_preregistration"] == 0
        for view in payload["comparator_bindings"]
    )


def test_policy_freezes_atomic_first_passage_and_causal_execution() -> None:
    policy = prereg.policy_payload()
    batches = policy["atomic_batches"]
    primary = policy["primary_clock"]
    execution = policy["execution"]
    assert batches["grouping_key"] == "exact available_at"
    assert batches["same_available_at_rows_are_simultaneous"] is True
    assert batches["intra_batch_transaction_or_log_order_forbidden"] is True
    assert primary["confirmation_interval"] == (
        "wbtc_available_at < stablecoin_available_at <= "
        "wbtc_available_at + 12 elapsed hours"
    )
    assert primary["cumulative_initial_value"] == 0
    assert primary["signal_time"] == "confirming stablecoin batch available_at"
    assert primary["amount_or_ratio_threshold"] is None
    assert primary["block_timestamp_forbidden"] is True
    assert execution["entry_time"] == (
        "ceil_to_5m(signal_time) + 5 elapsed minutes"
    )
    assert execution["exact_grid_signal_still_waits_one_bar"] is True
    assert execution["hold_elapsed_hours"] == 72
    assert execution["hold_bars_5m"] == 864
    assert execution["global_nonoverlap"] is True
    assert execution["accepted_confirmation_identity_reuse"] is False
    assert execution["split_crossing_action"] == "skip"


def test_controls_support_novelty_and_economic_gates_are_fixed() -> None:
    policy = prereg.policy_payload()
    assert set(policy["controls"]) == {
        "direction_flip",
        "deterministic_random_side",
        "wbtc_only_direct",
        "stablecoin_only_12h_grid",
        "anchored_first_nonzero",
        "opposite_confirmation",
        "lead_lag_reverse",
        "stale_wbtc_24h",
        "stale_wbtc_72h",
        "stablecoin_year_amount_permutation",
        "black_funds_veto",
        "usdc_only_confirmation",
        "usdt_only_confirmation",
    }
    support = policy["source_support_gates"]
    assert support["train_total_minimum"] == 50
    assert support["selection_total_minimum"] == 20
    assert support["maximum_month_share"] == 0.20
    assert support["maximum_quarter_share"] == 0.40
    assert support["maximum_consecutive_same_side"] == 10
    assert support["maximum_calendar_gap_days"] == 90
    novelty = policy["novelty"]
    assert novelty["near_window_elapsed_hours"] == 12
    assert novelty["maximum_exact_entry_jaccard"] == 0.10
    assert novelty["maximum_wscf_to_comparator_near_containment"] == 0.30
    assert novelty["comparators"] == [
        spec["name"] for spec in prereg.COMPARATOR_SPECS
    ]
    strict = policy["strict_economic_gates"]
    assert strict["cagr_to_strict_mdd_minimum"] == 3.0
    assert strict["strict_mdd_pct_maximum"] == 15.0
    assert strict["full_calendar_cagr"] is True
    assert strict["strict_intratrade_high_water_mdd"] is True
    assert policy["rllm_boundary"][
        "authorized_before_deterministic_train_and_selection_pass"
    ] is False


def test_comparator_bindings_are_hash_bound_without_reading_values() -> None:
    bindings = prereg.comparator_bindings()
    assert [view["name"] for view in bindings] == [
        "wcdr_primary",
        "wtsl_primary",
        "ugci_primary",
        "sealed_prior_stablecoin_bundle",
        "live_portfolio_pure_clocks",
    ]
    assert all(view["value_rows_read_during_preregistration"] == 0 for view in bindings)
    assert bindings[0]["filters"] == {
        "candidate": "WCDR-2016",
        "control": "primary",
    }
    assert bindings[3]["side_field"] is None
    assert bindings[4]["group_field"] == "candidate_id"


def test_tampering_fails_canonical_validation() -> None:
    payload = prereg.build_preregistration()
    payload["policy"]["execution"]["hold_elapsed_hours"] = 24
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_false_clean_research_claim_is_rejected() -> None:
    payload = prereg.build_preregistration()
    payload["source_family_values_previously_opened"] = False
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = prereg.canonical_hash(core)
    with pytest.raises(RuntimeError, match="source-family disclosure drift"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_write_once_verifies_identical_and_refuses_drift(tmp_path: Path) -> None:
    relative = Path(".pytest_cache") / f"{tmp_path.name}-wscf-preregistration.json"
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
