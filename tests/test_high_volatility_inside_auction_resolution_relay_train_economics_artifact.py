import hashlib,json
from training import evaluate_high_volatility_inside_auction_resolution_relay_economics as economics
def test_hviarr_train_rejection_is_terminal_and_later_stages_are_sealed():
 p=economics.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="38f8ff31456022df3bf21763aaf21ef0158c66fe90b1dd7d7d959441b411d593"
 x=json.loads(p.read_text());core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==economics.canonical_hash(core)
 assert x["passed"] is False and x["decision"]=="terminal_reject_no_repair" and x["later_stage_outcomes_opened"] is False
 assert x["primary"]["base"]["absolute_return_pct"]<0 and x["primary"]["stress"]["absolute_return_pct"]<0 and x["primary"]["base"]["mean_gross_underlying_bp"]<0
 assert x["checks"]["each_calendar_half_positive"] is False and x["checks"]["cluster_signflip_p_max_0_1"] is False
 assert not economics.OUTPUTS["test"].exists() and not economics.OUTPUTS["eval"].exists() and not economics.OUTPUTS["final"].exists()
