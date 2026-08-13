import json

from training import preregister_high_volatility_crypto_equity_leadership_spread_relay as prereg


def test_preregistration_is_frozen_blind_and_causal() -> None:
    value = prereg.build()
    prereg.validate(value)
    assert value["policy_id"] == "HVCELSR-24" and value["singleton"]
    assert not value["outcomes_opened"] and not value["source_incidence_opened"]
    assert not value["gross9_rows_opened"]
    assert value["clock"]["entry"] == "exact BTCUSDT five-minute open ten elapsed minutes after official close"
    assert value["clock"]["no_imputation"]
    assert not value["research_boundary"]["grid"]
    assert not value["research_boundary"]["repair_of_prior_candidate"]


def test_hash_formula_and_gates() -> None:
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
    assert "NaN" not in json.dumps(value, ensure_ascii=False, allow_nan=False)
    assert value["policy"]["symbols"] == ["MSTR", "COIN"]
    assert value["policy"]["scale_prior_sessions"] == 63
    assert value["policy"]["scale_prior_minimum"] == 42
    assert value["policy"]["magnitude_midrank_min"] == 0.75
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert value["economic_gates"]["cagr_to_strict_mdd_min"] == 3
    assert value["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
