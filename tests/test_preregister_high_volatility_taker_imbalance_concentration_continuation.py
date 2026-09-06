import json

from training import preregister_high_volatility_taker_imbalance_concentration_continuation as prereg


def test_preregistration_is_frozen_blind_and_causal() -> None:
    value = prereg.build()
    prereg.validate(value)
    assert value["policy_id"] == "HVTICC-8"
    assert value["singleton"]
    assert not value["outcomes_opened"]
    assert not value["source_incidence_opened"]
    assert not value["gross9_rows_opened"]
    assert value["features"]["no_imputation"]
    assert value["clock"]["entry"] == "exact BTCUSDT perpetual D+5m open"
    assert not value["research_boundary"]["grid"]
    assert not value["research_boundary"]["repair_of_prior_candidate"]


def test_hash_formula_and_gates() -> None:
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
    assert "NaN" not in json.dumps(value, ensure_ascii=False, allow_nan=False)
    assert value["features"]["signed_concentration"].startswith("sum(f_i*abs(f_i))")
    assert value["policy"]["rank_history"] == 270
    assert value["policy"]["rank_minimum"] == 180
    assert value["policy"]["concentration_rank_min"] == 0.75
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert value["economic_gates"]["cagr_to_strict_mdd_min"] == 3
    assert value["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
