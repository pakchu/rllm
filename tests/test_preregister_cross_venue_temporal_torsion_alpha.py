from __future__ import annotations

import json

import pytest

from training import preregister_cross_venue_temporal_torsion_alpha as prereg


def test_policy_grid_is_fixed_and_small() -> None:
    assert [policy.policy_id for policy in prereg.policy_grid()] == [
        "V01",
        "V02",
        "V03",
        "V04",
    ]
    assert {policy.route for policy in prereg.policy_grid()} == {
        "spot_preload_um_echo",
        "um_preload_spot_echo",
    }
    assert {policy.hold_bars for policy in prereg.policy_grid()} == {6, 18}


def test_manifest_preserves_sealed_holdout_and_no_outcomes() -> None:
    payload = prereg.build_manifest()
    prereg.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["selection_protocol"]["sealed_holdout"] == [
        "2023-01-01",
        "2024-01-01",
    ]
    assert payload["selection_protocol"]["multiple_testing_hypotheses"] == 4


def test_feature_contract_is_crossed_within_venue_clock() -> None:
    feature = prereg.build_manifest()["feature_contract"]
    assert feature["spot_flow_to_return_delay"].startswith(
        "spot_return_time_centroid"
    )
    assert feature["um_flow_to_return_delay"].startswith(
        "um_return_time_centroid"
    )
    assert "spot_preload * um_echo" in feature["spot_to_um_score"]
    assert "um_preload * spot_echo" in feature["um_to_spot_score"]
    assert feature["all_thresholds_shifted_bars"] == 1


def test_execution_is_delayed_and_costed() -> None:
    execution = prereg.build_manifest()["execution_contract"]
    assert execution["entry_delay_bars_from_bucket_open"] == 2
    assert execution["base_cost_notional_per_side"] == 0.0006
    assert execution["stress_cost_notional_per_side"] == 0.0008
    assert execution["realized_funding"] is True
    assert "global/pre-entry" in execution["strict_mdd"]


def test_tampered_manifest_fails_validation() -> None:
    payload = prereg.build_manifest()
    payload["execution_contract"]["entry_delay_bars_from_bucket_open"] = 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        prereg.validate_manifest(payload)


def test_write_once_refuses_different_preregistration(tmp_path) -> None:
    output = tmp_path / "prereg.json"
    payload = prereg.build_manifest()
    assert prereg.write_manifest_once(output, payload) == "created"
    assert prereg.write_manifest_once(output, payload) == "verified_existing"
    stored = json.loads(output.read_text())
    stored["manifest_hash"] = "0" * 64
    output.write_text(json.dumps(stored))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        prereg.write_manifest_once(output, payload)
