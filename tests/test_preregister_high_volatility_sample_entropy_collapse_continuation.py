import json

from training import preregister_high_volatility_sample_entropy_collapse_continuation as prereg


def test_preregistration_is_singleton_causal_and_outcome_blind() -> None:
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "HVSENC-8"
    assert payload["singleton"] is True
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["features"]["no_imputation"] is True
    assert payload["clock"]["entry"] == "exact BTCUSDT perpetual D+5m open"
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False


def test_hash_and_frozen_protocol_gates() -> None:
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert "NaN" not in json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert payload["policy"]["template_dimension"] == 2
    assert payload["policy"]["tolerance_standard_deviations"] == 0.2
    assert payload["source_support_gates"] == {
        "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
        "minority_side_share_min": 0.2,
        "max_month_share": 0.45,
    }
    assert payload["novelty_gates"]["absolute_signed_exposure_pearson_max"] == 0.35
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
