import json

from training import preregister_high_volatility_ethereum_missed_slot_delta_relay as p


def test_singleton_outcome_blind_canonical() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVEMSD-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["manifest_hash"] == p.canonical_hash(
        {key: item for key, item in value.items() if key != "manifest_hash"}
    )


def test_frozen_liveness_clock_and_gates() -> None:
    value = p.build()
    policy = value["policy"]
    assert (
        policy["missed_change_prior_observations"],
        policy["missed_change_prior_minimum"],
        policy["missed_change_midrank_min"],
    ) == (365, 180, 0.70)
    assert policy["confirmation_slots"] == 64
    assert value["features"]["scheduled_slots"] == 7200
    assert value["clock"]["entry"] == "exact BTCUSDT 00:25 UTC five-minute open"
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert len(value["diagnostic_controls"]["names"]) == 6


def test_artifact_matches() -> None:
    value = json.loads(p.DEFAULT_OUTPUT.read_text())
    p.validate(value)
    assert value == p.build()
