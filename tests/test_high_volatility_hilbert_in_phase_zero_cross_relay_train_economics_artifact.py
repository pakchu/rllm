import hashlib,json
from training import evaluate_high_volatility_hilbert_in_phase_zero_cross_relay_economics as e
def test_train_reject_is_terminal_and_hash_bound():
 r=json.loads(e.OUTPUTS["train"].read_text());assert r["passed"] is False and r["decision"]=="terminal_reject_no_repair" and r["later_stage_outcomes_opened"] is False
 b=r["primary"]["base"];assert b["absolute_return_pct"]==-6.963116954478643 and b["mean_gross_underlying_bp"]==-13.171050627507949 and b["strict_mdd_pct"]==15.034716419642159
 assert hashlib.sha256(e.OUTPUTS["train"].read_bytes()).hexdigest()=="43bbaa50dbe4f8039ded8d718e2dde307d105ea14d1b538cb2352f18acff6f46"
 assert r["manifest_hash"]==e.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"}) and not e.OUTPUTS["test"].exists() and not e.OUTPUTS["eval"].exists() and not e.OUTPUTS["final"].exists()
