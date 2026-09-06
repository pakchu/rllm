import hashlib,json
from training import evaluate_cboe_surface_dislocation_overnight_btc_relay_economics as e
def test_cvsdr_train_economics_frozen_terminal():
 p=e.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="68b58cb883d36d41f59504dba0f60ded7d8ccffbc0009be62b2127516aa84ff7";d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==e.canonical_hash(core) and d["passed"] is False and d["decision"]=="terminal_reject_no_repair";assert d["advance_to_next_stage"] is False and d["later_stage_outcomes_opened"] is False;assert d["primary"]["base"]["absolute_return_pct"]>0 and d["primary"]["stress"]["absolute_return_pct"]<0;assert d["checks"]["each_calendar_half_positive"] is False
