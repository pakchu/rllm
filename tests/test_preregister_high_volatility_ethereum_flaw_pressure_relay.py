from training import preregister_high_volatility_ethereum_flaw_pressure_relay as p
def test_frozen_preregistration():
 v=p.build();c=dict(v);h=c.pop("manifest_hash");assert h==p.canonical_hash(c);assert v["policy_id"]=="HVEFPR-24";assert v["outcomes_opened"] is False;assert v["source_incidence_opened"] is False;assert v["policy"]["same_weekday_lag_days"]==7;assert v["policy"]["variation_midrank_min"]==.65;assert v["clock"]["hold"]=="24 elapsed hours";assert v["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
def test_git_source_and_grammar_are_fixed():
 v=p.build();g=v["source_plan"]["git"];assert g["remote"]=="https://github.com/ethereum/go-ethereum.git";assert g["branch"]=="master";assert g["sealed_tip"]=="87ab9435f542d48e70f65652242118e84795b83e";assert v["policy"]["flaw_terms"]==list(p.TERMS);assert v["research_boundary"]["repair_of_prior_candidate"] is False
