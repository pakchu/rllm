import json

from training import preregister_high_volatility_cross_alt_flow_price_transmission_relay as subject


def test_manifest_is_canonical_and_outcome_blind():
    value = subject.build()
    subject.validate(value)
    assert value["manifest_hash"] == subject.canonical_hash(
        {key: item for key, item in value.items() if key != "manifest_hash"}
    )
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False


def test_fixed_policy_and_gates():
    value = subject.build()
    assert value["policy"] == {
        "block_minutes": 480,
        "minimum_directional_breadth": 4,
        "history_decisions": 270,
        "minimum_history_decisions": 180,
        "transmission_rank_min": 0.70,
        "variation_rank_min": 0.65,
        "onset_required": True,
        "entry_delay_minutes": 5,
        "hold_hours": 8,
        "leverage": 0.5,
        "base_cost_per_notional_side": 0.0006,
        "stress_cost_per_notional_side": 0.001,
    }
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8
    }
    assert value["research_boundary"]["candidate_count"] == 1
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_json_roundtrip_is_unicode_safe():
    value = subject.build()
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    assert json.loads(encoded) == value
