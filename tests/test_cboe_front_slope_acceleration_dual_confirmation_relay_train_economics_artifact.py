import hashlib,json
from training import evaluate_cboe_front_slope_acceleration_dual_confirmation_relay_economics as e
def test_cvfdmr_train_economics_frozen_terminal():
 p=e.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="a9bf987fdfadcc3ad2a0a2c6db782ccf8e7acf979ddd3a2cb4d0f657f178a581";d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==e.canonical_hash(core) and d["passed"] is False and d["decision"]=="terminal_reject_no_repair";assert d["primary"]["base"]["absolute_return_pct"]>1 and d["primary"]["stress"]["absolute_return_pct"]>0;assert d["checks"]["each_calendar_half_positive"] is False
