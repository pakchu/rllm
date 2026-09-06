import json
from training import preregister_high_volatility_same_sign_jump_aftershock_continuation as prereg
def test_preregistration_is_frozen_and_blind():
 x=prereg.build();prereg.validate(x);assert x["policy_id"]=="HVSJAC-8" and x["singleton"];assert not x["outcomes_opened"] and not x["source_incidence_opened"] and not x["gross9_rows_opened"];assert x["features"]["no_imputation"] and not x["research_boundary"]["grid"] and not x["research_boundary"]["repair_of_prior_candidate"]
def test_hash_and_gates():
 x=prereg.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==prereg.canonical_hash(core);assert "NaN" not in json.dumps(x,ensure_ascii=False,allow_nan=False);assert x["policy"]["jump_scale_multiple"]==3 and x["policy"]["minimum_jumps"]==4 and x["policy"]["aftershock_gap_bars_max"]==3;assert x["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8};assert x["economic_gates"]["cagr_to_strict_mdd_min"]==3 and x["economic_gates"]["stress_cagr_to_strict_mdd_min"]==2.5
