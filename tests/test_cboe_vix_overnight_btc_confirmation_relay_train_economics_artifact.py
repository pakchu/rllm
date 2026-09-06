import hashlib,json
from training import evaluate_cboe_vix_overnight_btc_confirmation_relay_economics as e
def test_cvobr_train_economics_frozen_terminal():
 p=e.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="5eef0712a0ffc69f6177332435d42fbb3bea14e7200a2a1d1bf55e2e7e8b5986";d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==e.canonical_hash(core) and d["passed"] is False and d["decision"]=="terminal_reject_no_repair";assert d["advance_to_next_stage"] is False and d["later_stage_outcomes_opened"] is False;assert d["primary"]["base"]["absolute_return_pct"]<0 and 0<d["primary"]["base"]["mean_gross_underlying_bp"]<20
