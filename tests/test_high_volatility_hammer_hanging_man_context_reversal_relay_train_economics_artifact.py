import hashlib,json
from training import evaluate_high_volatility_hammer_hanging_man_context_reversal_relay_economics as e
def test_train_reject_is_terminal_and_hash_bound():
 r=json.loads(e.OUTPUTS["train"].read_text());assert r["passed"] is False and r["decision"]=="terminal_reject_no_repair" and r["later_stage_outcomes_opened"] is False
 assert r["primary"]["base"]["absolute_return_pct"]==-1.999965769278822 and r["primary"]["base"]["mean_gross_underlying_bp"]==5.844184936129112 and r["primary"]["stress"]["absolute_return_pct"]==-4.287103079896237
 assert hashlib.sha256(e.OUTPUTS["train"].read_bytes()).hexdigest()=="a1c1bc1bc9427fb9abe039ee24526c5e19fa863b8730b8257fcff83b0d25a012"
 assert r["manifest_hash"]==e.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
 assert not e.OUTPUTS["test"].exists() and not e.OUTPUTS["eval"].exists() and not e.OUTPUTS["final"].exists()
