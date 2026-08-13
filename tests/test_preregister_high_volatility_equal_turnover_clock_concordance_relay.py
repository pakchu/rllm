import json

from training import preregister_high_volatility_equal_turnover_clock_concordance_relay as prereg


def test_preregistration_is_frozen_blind_and_causal() -> None:
    value = prereg.build()
    prereg.validate(value)
    assert value["policy_id"] == "HVETCC-8"
    assert value["singleton"]
    assert not value["outcomes_opened"]
    assert not value["source_incidence_opened"]
    assert not value["gross9_rows_opened"]
    assert value["features"]["no_imputation"]
    assert value["clock"]["entry"] == "exact BTCUSDT perpetual D+5m open"
    assert not value["research_boundary"]["grid"]
    assert not value["research_boundary"]["repair_of_prior_candidate"]
    assert not value["research_boundary"]["promoted_prior_control"]


def test_hash_formula_and_gates() -> None:
    value = prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
    assert "NaN" not in json.dumps(value, ensure_ascii=False, allow_nan=False)
    assert value["policy"]["turnover_segments"] == 4
    assert value["policy"]["decision_hours"] == [2, 10, 18]
    assert value["policy"]["variation_history_decisions"] == 270
    assert value["policy"]["minimum_history_decisions"] == 180
    assert value["policy"]["variation_rank_min"] == 0.65
    assert value["source_plan"]["bars"]["window"][0] == "2022-12-31T18:00:00Z"
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert value["economic_gates"]["cagr_to_strict_mdd_min"] == 3
    assert value["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5


def test_intrinsic_clock_is_indivisible_and_outcome_blind() -> None:
    value = prereg.build()
    assert "cumulative_quote_turnover_before_i" in value["features"]["turnover_clock_assignment"]
    assert "minutes are indivisible" in value["features"]["turnover_clock_assignment"]
    assert "log(close/open)" in value["features"]["minute_return"]
    assert "postentry" not in value["features"]["eligibility"]
    assert value["diagnostic_controls"]["cannot_be_promoted"]
