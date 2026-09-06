import json

from training import preregister_high_volatility_chaikin_volatility_expansion_trend_relay as p


def test_manifest_is_singleton_outcome_blind_and_canonical():
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVCHV-12"
    assert value["singleton"] is True
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["research_boundary"]["candidate_count"] == 1
    assert value["research_boundary"]["grid"] is False
    assert value["manifest_hash"] == p.canonical_hash(
        {key: item for key, item in value.items() if key != "manifest_hash"}
    )


def test_frozen_policy_and_gates():
    value = p.build()
    assert value["policy"] == {
        "source_bar_minutes": 240,
        "range_ema_periods": 10,
        "range_roc_periods": 10,
        "history_decisions": 180,
        "minimum_history_decisions": 120,
        "chaikin_rank_min": 0.70,
        "variation_hours": 24,
        "variation_rank_min": 0.65,
        "entry_delay_minutes": 5,
        "hold_hours": 12,
        "leverage": 0.5,
        "base_cost_per_notional_side": 0.0006,
        "stress_cost_per_notional_side": 0.001,
    }
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert value["novelty_gates"]["must_pass_before_economics"] is True
    assert value["economic_gates"]["stop_on_first_failure"] is True
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    assert "postentry_return_or_pnl_opened" in encoded
