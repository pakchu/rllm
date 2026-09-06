from training import preregister_high_volatility_safehaven_variation_leadership_relay as prereg


def test_blind_canonical_registration() -> None:
    payload = prereg.build(); prereg.validate(payload)
    assert payload["manifest_hash"] == prereg.canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    assert not payload["outcomes_opened"] and not payload["source_incidence_opened"] and not payload["gross9_rows_opened"]


def test_fixed_variation_leadership_policy() -> None:
    payload = prereg.build()
    assert prereg.FX == ("USDJPY", "USDCHF")
    assert payload["policy"]["leadership_rank_min"] == 0.65
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["features"]["dominant_pair"].startswith("USDJPY iff")
    assert not payload["research_boundary"]["repair_of_prior_candidate"]
