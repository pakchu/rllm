import hashlib,json
from training import evaluate_high_volatility_quartile_median_staircase_relay_economics as economics
def test_hvqmsr_train_rejection_is_terminal_and_later_stages_are_sealed():
 p=economics.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="5e491ced73a0a42d3aa5a41b56c3fca6a005c86238a6a7b2c16bc8912842cd65"
 x=json.loads(p.read_text());core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==economics.canonical_hash(core)
 assert x["passed"] is False and x["decision"]=="terminal_reject_no_repair" and x["later_stage_outcomes_opened"] is False
 assert x["primary"]["base"]["absolute_return_pct"]>6 and x["primary"]["stress"]["absolute_return_pct"]>4 and x["primary"]["base"]["mean_gross_underlying_bp"]>36
 assert x["primary"]["calendar_halves"]["first"]["absolute_return_pct"]< -2 and x["primary"]["calendar_halves"]["second"]["absolute_return_pct"]>9
 assert not economics.OUTPUTS["test"].exists() and not economics.OUTPUTS["eval"].exists() and not economics.OUTPUTS["final"].exists()
