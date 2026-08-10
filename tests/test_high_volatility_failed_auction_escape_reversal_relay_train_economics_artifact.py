import hashlib,json
from training import evaluate_high_volatility_failed_auction_escape_reversal_relay_economics as economics
def test_hvfaer_train_rejection_is_terminal_and_later_stages_are_sealed():
 p=economics.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="6412a1db6d88b2ed4aad2c26c2cc91919cf4e499faf6992b286e270fe7026b1a"
 x=json.loads(p.read_text());core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==economics.canonical_hash(core)
 assert x["passed"] is False and x["decision"]=="terminal_reject_no_repair" and x["later_stage_outcomes_opened"] is False
 assert x["primary"]["base"]["absolute_return_pct"]<-3.7 and x["primary"]["stress"]["absolute_return_pct"]<-4.4 and x["primary"]["base"]["mean_gross_underlying_bp"]<-30
 assert x["checks"]["each_calendar_half_positive"] is False and x["checks"]["cluster_signflip_p_max_0_1"] is False
 assert not economics.OUTPUTS["test"].exists() and not economics.OUTPUTS["eval"].exists() and not economics.OUTPUTS["final"].exists()
