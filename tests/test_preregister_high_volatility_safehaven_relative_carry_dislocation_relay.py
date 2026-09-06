from training import preregister_high_volatility_safehaven_relative_carry_dislocation_relay as prereg


def test_blind_canonical_registration() -> None:
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["manifest_hash"] == prereg.canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_fixed_relative_safehaven_policy() -> None:
    payload = prereg.build()
    assert prereg.FX == ("USDJPY", "USDCHF")
    assert payload["policy"]["dislocation_rank_min"] == 0.75
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["clock"]["side"] == "negative relative_dislocation sign"
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
