import hashlib,json
from training import evaluate_high_volatility_drawup_drawdown_asymmetry_relay_economics as economics
def test_hvdudar_train_rejection_is_terminal_and_later_stages_are_sealed():
 p=economics.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="87345ae32d2aebddabb303cf9847431f8b7f9aefcb3ac5ff75a2923a8b570fc9"
 x=json.loads(p.read_text());core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==economics.canonical_hash(core)
 assert x["passed"] is False and x["decision"]=="terminal_reject_no_repair" and x["later_stage_outcomes_opened"] is False
 assert x["primary"]["base"]["absolute_return_pct"]<0 and x["primary"]["stress"]["absolute_return_pct"]< -2.5
 assert x["primary"]["calendar_halves"]["first"]["absolute_return_pct"]< -3.8 and x["primary"]["calendar_halves"]["second"]["absolute_return_pct"]>3.9
 assert not economics.OUTPUTS["test"].exists() and not economics.OUTPUTS["eval"].exists() and not economics.OUTPUTS["final"].exists()
