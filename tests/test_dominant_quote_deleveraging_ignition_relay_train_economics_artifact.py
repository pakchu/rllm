import hashlib,json
from training import evaluate_dominant_quote_deleveraging_ignition_relay_economics as economics
def test_dqdir_train_economics_frozen_terminal():
 p=economics.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="125b8355409758f8fe063b85657969643dd89def58bd82d2a3374a25d8e53ff0"
 d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==economics.canonical_hash(core)
 assert d["passed"] is False and d["decision"]=="terminal_reject_no_repair" and d["later_stage_outcomes_opened"] is False
 assert d["primary"]["base"]["absolute_return_pct"]<0 and d["primary"]["base"]["mean_gross_underlying_bp"]<1
 assert d["checks"]["each_calendar_half_positive"] is False
