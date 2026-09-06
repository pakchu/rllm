import hashlib,json
from training import evaluate_high_volatility_rogers_satchell_excursion_asymmetry_reversal_relay_economics as e
def test_hvrsar_train_rejection_is_terminal_and_reproduced():
 p=e.OUTPUTS["train"];x=json.loads(p.read_text());core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==e.canonical_hash(core)
 assert x["policy_id"]=="HVRSAR-8" and x["stage"]=="train" and x["passed"] is False and x["decision"]=="terminal_reject_no_repair" and x["advance_to_next_stage"] is False and x["later_stage_outcomes_opened"] is False
 assert x["primary"]["base"]["absolute_return_pct"]<0 and x["primary"]["base"]["mean_gross_underlying_bp"]<0 and x["primary"]["stress"]["absolute_return_pct"]<0 and not x["checks"]["each_calendar_half_positive"]
 assert not any(e.OUTPUTS[k].exists() for k in ("test","eval","final")) and hashlib.sha256(p.read_bytes()).hexdigest()=="97d840a70e2270f8c3acd790685b0207ec09931c556204180d400b9ff018601b"
