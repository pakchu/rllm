from training import preregister_high_volatility_bls_fx_reaction_transmission_relay as p

def test_manifest_is_deterministic_blind_and_valid():
 x=p.build();assert x==p.build();p.validate(x);assert not x["outcomes_opened"] and not x["source_incidence_opened"] and not x["gross9_rows_opened"]
def test_frozen_policy_and_gates():
 x=p.build();assert x["policy_id"]=="HVBLSFX-12";assert x["policy"]["reaction_minutes"]==10;assert x["policy"]["hold_hours"]==12;assert x["clock"]["side"]=="USDJPY reaction sign";assert x["novelty_gates"]["must_pass_before_economics"] is True;assert x["economic_gates"]["stop_on_first_failure"] is True;assert x["diagnostic_controls"]["cannot_be_promoted"] is True
