from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_sec_bitcoin_issuer_reactivation_breadth as prereg


def test_preregistration_discloses_seen_source_but_unopened_candidate() -> None:
    payload = prereg.build_preregistration()
    prereg.validate_preregistration(payload)
    assert payload["candidate"] == "BIRB-120-SOURCE-FAMILY-SEEN"
    assert payload["source_family_values_previously_opened"] is True
    assert payload["exact_source_incidence_opened"] is False
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    assert payload["prior_research_disclosure"]["source_family_hypothesis_number"] == 3
    assert payload["prior_research_disclosure"]["pristine_source_family_claim"] is False
    assert payload["outcome_boundary"] == prereg.EXPECTED_OUTCOME_BOUNDARY


def test_policy_freezes_reactivation_breadth_and_execution() -> None:
    policy = prereg.policy_payload()
    primary = policy["primary_clock"]
    execution = policy["execution"]
    assert primary["reactivation_gap_elapsed_days_minimum"] == 365
    assert primary["breadth_window_elapsed_days"] == 7
    assert primary["distinct_issuer_threshold"] == 3
    assert primary["first_ever_issuer_hit"] == "not a primary reactivation"
    assert primary["expiry_only_signal"] is False
    assert primary["side"] == "LONG"
    assert execution["entry_time"] == "ceil_to_5m(signal_ready) + 5 elapsed minutes"
    assert execution["exact_grid_signal_still_waits_one_bar"] is True
    assert execution["hold_elapsed_hours"] == 120
    assert execution["hold_bars_5m"] == 1440
    assert execution["global_nonoverlap"] is True
    assert execution["side_override"] is False


def test_controls_support_novelty_economics_and_rllm_are_fixed() -> None:
    policy = prereg.policy_payload()
    assert set(policy["source_controls"]) == {
        "first_ever_birth_breadth",
        "any_mention_breadth",
        "repeat_filer_breadth",
        "single_reactivation",
        "stale_30d",
        "year_cik_permutation",
        "threshold_two",
        "threshold_four",
    }
    support = policy["source_support_gates"]
    assert support["train_total_minimum"] == 24
    assert support["selection_total_minimum"] == 8
    assert support["maximum_month_share"] == 0.20
    novelty = policy["novelty"]
    assert novelty["maximum_exact_entry_jaccard"] == 0.10
    assert novelty["maximum_birb_to_comparator_near_containment"] == 0.35
    assert novelty["comparators"] == [
        spec["name"] for spec in prereg.COMPARATOR_SPECS
    ]
    specificity = policy["mechanism_specificity_gates"]
    assert specificity["single_reactivation_proximity_is_report_only"] is True
    strict = policy["strict_economic_gates"]
    assert strict["cagr_to_strict_mdd_minimum"] == 3.0
    assert strict["strict_mdd_pct_maximum"] == 15.0
    assert strict["full_calendar_cagr"] is True
    assert strict["strict_intratrade_high_water_mdd"] is True
    assert policy["rllm_boundary"][
        "authorized_before_deterministic_train_and_selection_pass"
    ] is False
    assert policy["rllm_boundary"]["later_actions"] == [
        "TRADE_FIXED_LONG",
        "ABSTAIN",
    ]


def test_bindings_are_hash_only_during_preregistration() -> None:
    payload = prereg.build_preregistration()
    assert payload["source_binding"]["value_rows_read_during_preregistration"] == 0
    assert [row["name"] for row in payload["comparator_bindings"]] == [
        "prior_microstructure_bundle",
        "bitmex_trollbox_semantic_clock",
        "live_portfolio_pure_clocks",
    ]
    assert all(
        row["value_rows_read_during_preregistration"] == 0
        for row in payload["comparator_bindings"]
    )


def test_tampering_fails_canonical_validation() -> None:
    payload = prereg.build_preregistration()
    payload["policy"]["primary_clock"]["distinct_issuer_threshold"] = 2
    with pytest.raises(RuntimeError, match="frozen policy drift"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_false_clean_research_claim_is_rejected() -> None:
    payload = prereg.build_preregistration()
    payload["prior_research_disclosure"]["pristine_source_family_claim"] = True
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = prereg.canonical_hash(core)
    with pytest.raises(RuntimeError, match="prior-research disclosure drift"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_write_once_verifies_identical_and_refuses_drift(tmp_path: Path) -> None:
    relative = Path(".pytest_cache") / f"{tmp_path.name}-birb-preregistration.json"
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
