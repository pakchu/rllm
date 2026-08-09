from training import preregister_high_volatility_meme_attention_saturation_reversal as prereg
def test_preregistration_is_outcome_blind():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="HVMASR-8";assert p["singleton"];assert not p["outcomes_opened"];assert not p["source_incidence_opened"];assert not p["gross9_rows_opened"];assert p["clock"]["side"].startswith("negative DOGE")
