import hashlib,json
from training import evaluate_cboe_crypto_vix_transmission_relay_economics as economics
def test_ccvtr_train_economics_frozen_terminal():
 p=economics.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="9c0569edec6a6fd569ec7a993fdb2dbb71d34354ff77f08f13f06e7a82bec750"
 d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==economics.canonical_hash(core)
 assert d["passed"] is False and d["decision"]=="terminal_reject_no_repair" and d["later_stage_outcomes_opened"] is False
 assert d["primary"]["base"]["absolute_return_pct"]>2 and d["primary"]["base"]["mean_gross_underlying_bp"]>20 and d["primary"]["stress"]["absolute_return_pct"]>0
 assert d["checks"]["cagr_to_strict_mdd_min_3"] is False and d["checks"]["cluster_signflip_p_max_0_1"] is False and d["checks"]["each_calendar_half_positive"] is False
