import hashlib,json
from training import evaluate_cboe_vix_overnight_btc_confirmation_relay_gross9_novelty as n
def test_cvobr_novelty_frozen_pass():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=="ccf0a32b2b86ad8c6f93c459ca0cdf323a4801df5e8e1f13c211a57f0385730f";p=json.loads(n.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==n.chash(core) and p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True;assert p["evidence_boundary"]["outcomes_opened"] is False;assert max(x["metrics"]["one_to_one_6h_max_matched_share"] for x in p["gross9_sleeves"].values())==.12
