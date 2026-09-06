import hashlib,json
from training import evaluate_high_volatility_acceleration_bands_breakout_relay_economics as e
def test_train_reject_is_terminal_and_hash_bound():
 r=json.loads(e.OUTPUTS["train"].read_text());assert r["passed"] is False and r["decision"]=="terminal_reject_no_repair" and r["later_stage_outcomes_opened"] is False
 b=r["primary"]["base"];assert b["absolute_return_pct"]==2.462046285371633 and b["mean_gross_underlying_bp"]==68.65932120827891 and b["cagr_to_strict_mdd"]==.9017981348421314
 assert r["primary"]["stress"]["absolute_return_pct"]==2.094998985244678 and r["primary"]["cluster_signflip"]["pvalue"]==.3039369606303937
 assert hashlib.sha256(e.OUTPUTS["train"].read_bytes()).hexdigest()=="5e4d8755802048d9099520956fdbd9d3a26e9b6b558bf816fa02d683bf1e88b3"
 assert r["manifest_hash"]==e.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"}) and not e.OUTPUTS["test"].exists() and not e.OUTPUTS["eval"].exists() and not e.OUTPUTS["final"].exists()
