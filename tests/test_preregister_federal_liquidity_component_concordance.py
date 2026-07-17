from __future__ import annotations

import copy

import pytest

from training import federal_liquidity_component_concordance_clock as clock
from training import preregister_federal_liquidity_component_concordance as prereg


def test_manifest_freezes_exact_family_and_opens_no_outcomes() -> None:
    payload = prereg.build_manifest()
    prereg.validate_manifest(payload, verify_sources=True)
    assert payload["outcomes_opened"] is False
    assert payload["support_passed"] is True
    assert payload["selection_protocol"]["family_size"] == 4
    assert payload["feature_contract"]["family"] == [
        {
            "candidate_id": "FLCC-H4-Q60",
            "horizon_releases": 4,
            "lower_rank_numerator": 83,
            "upper_rank_numerator": 125,
        },
        {
            "candidate_id": "FLCC-H4-Q65",
            "horizon_releases": 4,
            "lower_rank_numerator": 72,
            "upper_rank_numerator": 136,
        },
        {
            "candidate_id": "FLCC-H8-Q60",
            "horizon_releases": 8,
            "lower_rank_numerator": 83,
            "upper_rank_numerator": 125,
        },
        {
            "candidate_id": "FLCC-H8-Q65",
            "horizon_releases": 8,
            "lower_rank_numerator": 72,
            "upper_rank_numerator": 136,
        },
    ]
    excluded = payload["novelty_boundary"]["excluded_inputs"]
    assert "open_interest_or_long_short_ratio" in excluded
    assert "perpetual_funding_premium_or_basis" in excluded


def test_source_support_exactly_replays_preregistered_counts() -> None:
    support = prereg.source_only_support()
    assert set(support) == {spec.candidate_id for spec in clock.CANDIDATE_SPECS}
    for candidate_id, expected in prereg.EXPECTED_PRIMARY_SUPPORT.items():
        for window, (count, long_count, short_count) in expected.items():
            actual = support[candidate_id]["windows"][window]
            assert (actual["count"], actual["long"], actual["short"]) == (
                count,
                long_count,
                short_count,
            )


def test_stage_gates_and_no_fallback_are_strict() -> None:
    selection = prereg.build_manifest()["selection_protocol"]
    assert selection["stage1_gates"]["CAGR_to_strict_MDD_min"] == 3.0
    assert selection["stage1_gates"]["weekly_cluster_signflip_p_max_bonferroni"] == 0.025
    assert selection["stage2_gates"]["CAGR_to_strict_MDD_min"] == 3.0
    assert selection["stage2_candidate"] == "exact Stage1 winner only; no fallback or repair"
    assert selection["failure_action"] == "reject FLCC-1 without opening 2023 outcomes"


def test_manifest_hash_rejects_tampering() -> None:
    payload = prereg.build_manifest()
    tampered = copy.deepcopy(payload)
    tampered["execution_policy"]["hold_bars"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        prereg.validate_manifest(tampered)


def test_validate_rejects_outcome_or_support_flags() -> None:
    payload = prereg.build_manifest()
    opened = copy.deepcopy(payload)
    opened["outcomes_opened"] = True
    opened["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in opened.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="opened outcomes"):
        prereg.validate_manifest(opened)

    failed = copy.deepcopy(payload)
    failed["support_passed"] = False
    failed["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in failed.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="did not pass"):
        prereg.validate_manifest(failed)
