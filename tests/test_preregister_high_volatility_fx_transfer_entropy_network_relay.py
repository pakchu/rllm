import json

from training import preregister_high_volatility_fx_transfer_entropy_network_relay as prereg


def test_preregistration_is_frozen_blind_and_causal() -> None:
    value = prereg.build()
    prereg.validate(value)
    assert value["policy_id"] == "HVFXTE-12"
    assert value["singleton"]
    assert not value["outcomes_opened"]
    assert not value["source_incidence_opened"]
    assert not value["gross9_rows_opened"]
    assert value["features"]["no_imputation"]
    assert value["clock"]["entry"] == "exact BTCUSDT perpetual 21:05 UTC open"
    assert not value["research_boundary"]["grid"]
    assert not value["research_boundary"]["repair_of_prior_candidate"]
    assert not value["research_boundary"]["promoted_prior_control"]


def test_hash_formula_and_gates() -> None:
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
    assert "NaN" not in json.dumps(value, ensure_ascii=False, allow_nan=False)
    assert len(value["features"]["fx_universe"]) == 6
    assert "I(sign_i[t-1];sign_j[t]|sign_j[t-1])" in value["features"]["directed_transfer_entropy"]
    assert value["policy"]["minimum_positive_outgoing_edges"] == 4
    assert value["policy"]["rank_history_sessions"] == 90
    assert value["policy"]["rank_minimum_sessions"] == 60
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert value["economic_gates"]["cagr_to_strict_mdd_min"] == 3
    assert value["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
