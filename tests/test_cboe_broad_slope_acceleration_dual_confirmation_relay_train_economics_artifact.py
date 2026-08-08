import hashlib,json
from training import evaluate_cboe_broad_slope_acceleration_dual_confirmation_relay_economics as e
def test_cvbdmr_train_economics_frozen_terminal():
 p=e.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="90d2057d087a795a21791500e5591bae383d4f9b6f56b8847fd49af258f194c8";d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==e.canonical_hash(core) and d["passed"] is False and d["decision"]=="terminal_reject_no_repair";assert d["primary"]["base"]["absolute_return_pct"]<0 and d["primary"]["base"]["mean_gross_underlying_bp"]<0
