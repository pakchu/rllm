from training import preregister_high_volatility_cross_alt_closing_impulse_dominance_relay as s


def test_blind_canonical_registration():
    value = s.build()
    s.validate(value)
    assert value["manifest_hash"] == s.canonical_hash({k: v for k, v in value.items() if k != "manifest_hash"})
    assert not value["outcomes_opened"] and not value["source_incidence_opened"] and not value["gross9_rows_opened"]


def test_fixed_impulse_policy():
    value = s.build()
    assert value["policy"]["impulse_share_min"] == 0.90
    assert value["policy"]["impulse_rank_min"] == 0.85
    assert value["policy"]["minimum_impulse_breadth"] == 4
    assert value["clock"]["side"].startswith("same")
    assert not value["research_boundary"]["repair_of_prior_candidate"]
