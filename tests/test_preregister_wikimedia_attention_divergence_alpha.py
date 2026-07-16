from __future__ import annotations

from training import preregister_wikimedia_attention_divergence_alpha as wad


def test_policy_grid_is_bounded_unique_and_deterministic() -> None:
    policies = wad.policy_grid()
    assert len(policies) == 14
    assert policies == sorted(policies)
    assert len(set(policies)) == len(policies)
    assert {policy.family for policy in policies} == {
        "broad_attention_reversal",
        "bitcoin_share_reversal",
        "silent_impulse_continuation",
    }


def test_manifest_keeps_holdouts_sealed_and_hash_valid() -> None:
    manifest = wad.build_manifest()
    wad.validate_manifest(manifest)
    assert manifest["outcomes_opened"] is False
    protocol = manifest["selection_protocol"]
    assert protocol["selection_data_must_be_physically_truncated_before"] == "2023-01-01"
    assert protocol["sealed_pre2024_holdout"] == ["2023-01-01", "2024-01-01"]
    assert protocol["future_seal_start"] == "2024-01-01"
    assert protocol["multiple_testing_hypotheses"] == 14


def test_availability_and_execution_are_delayed_and_fail_closed() -> None:
    manifest = wad.build_manifest()
    assert manifest["availability_contract"]["minimum_delay_after_observation_end_hours"] >= 36.0
    assert manifest["availability_contract"]["execution"] == "next 5m open at D+2 12:10 UTC"
    assert manifest["source_contract"]["missing_day_policy"] == "fail_closed_no_imputation"
    assert manifest["source_contract"]["historical_snapshot_is_point_in_time"] is False
    assert manifest["source_contract"]["promotion_requires_retrieval_timestamped_forward_shadow"] is True


def test_selection_does_not_use_2023_or_future() -> None:
    manifest = wad.build_manifest()
    protocol = manifest["selection_protocol"]
    assert protocol["fit"][1] <= protocol["selection"][0]
    assert protocol["selection"][1] <= protocol["sealed_pre2024_holdout"][0]
    assert protocol["sealed_pre2024_holdout"][1] <= protocol["future_seal_start"]
