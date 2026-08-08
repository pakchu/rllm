import hashlib,json
from training import evaluate_cboe_vix_acceleration_dual_confirmation_relay_economics as e
def test_cvvdmr_train_economics_frozen_terminal():
 p=e.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="e129c9b139fc6f9d5f2899a6b21965d7334b36f5565f125513b14c914c21620c";d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==e.canonical_hash(core) and d["passed"] is False and d["decision"]=="terminal_reject_no_repair";assert d["primary"]["base"]["absolute_return_pct"]<0
