from training import preregister_high_volatility_macro_polarity_concordance_relay as prereg

def test_single_outcome_blind_cross_provider_policy_is_frozen():
    p=prereg.build();prereg.validate(p)
    assert p['policy_id']=='HVMPC-24' and p['singleton']
    assert not p['outcomes_opened'] and not p['source_incidence_opened'] and not p['gross9_rows_opened']
    assert p['policy']['change_days']==7 and p['policy']['publication_delay_days']==8
    assert p['policy']['variation_rank_min']==.65 and p['clock']['hold']=='24 elapsed hours'
    assert not p['research_boundary']['promoted_prior_control']
