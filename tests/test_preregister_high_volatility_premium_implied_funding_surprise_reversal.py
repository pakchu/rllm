import json

from training import preregister_high_volatility_premium_implied_funding_surprise_reversal as prereg


def test_preregistration_is_singleton_causal_and_outcome_blind() -> None:
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "HVPIFSR-8"
    assert payload["singleton"] is True
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["clock"]["entry"] == "exact BTCUSDT perpetual S+5m open"
    assert payload["features"]["no_imputation"] is True
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False


def test_preregistration_hash_is_canonical_unicode_safe() -> None:
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert "NaN" not in json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_frozen_gates_match_research_protocol() -> None:
    payload = prereg.build()
    assert payload["source_support_gates"] == {
        "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
        "minority_side_share_min": 0.2,
        "max_month_share": 0.45,
    }
    assert payload["novelty_gates"]["exact_entry_jaccard_max"] == 0.1
    assert payload["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert payload["novelty_gates"]["occupied_5m_bar_jaccard_max"] == 0.25
    assert payload["novelty_gates"]["absolute_signed_exposure_pearson_max"] == 0.35
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
