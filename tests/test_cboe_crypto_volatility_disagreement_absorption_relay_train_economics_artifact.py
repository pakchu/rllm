import hashlib,json
from training import evaluate_cboe_crypto_volatility_disagreement_absorption_relay_economics as economics
def test_ccvdar_train_economics_frozen_terminal():
 p=economics.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="69b9ac442d99cdede573d989fbc7660949c963f6cc1e9e9b111d0540854ece9a"
 d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==economics.canonical_hash(core)
 assert d["passed"] is False and d["decision"]=="terminal_reject_no_repair" and d["later_stage_outcomes_opened"] is False
 assert d["primary"]["base"]["absolute_return_pct"]<0 and d["primary"]["base"]["mean_gross_underlying_bp"]<20
 assert d["checks"]["each_calendar_half_positive"] is False
