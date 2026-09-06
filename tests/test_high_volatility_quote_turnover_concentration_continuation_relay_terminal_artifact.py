import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_quote_turnover_concentration_continuation_relay_train_economics_2026-08-10.json")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvtccr_train_rejection_is_terminal_and_exact():
 x=json.loads(R.read_text());assert x["stage"]=="train" and x["passed"] is False and x["decision"]=="terminal_reject_no_repair"
 b=x["primary"]["base"];s=x["primary"]["stress"];assert b["absolute_return_pct"]>0 and b["mean_gross_underlying_bp"]>=20 and b["strict_mdd_pct"]<=15
 assert b["cagr_to_strict_mdd"]==1.7944148354500984 and s["cagr_to_strict_mdd"]==1.1215930963665857
 assert x["primary"]["cluster_signflip"]["pvalue"]==0.10400895991040089
 assert x["checks"]["cagr_to_strict_mdd_min_3"] is False and x["checks"]["cluster_signflip_p_max_0_1"] is False and x["checks"]["stress_cagr_to_strict_mdd_min_2_5"] is False
 assert x["checks"]["each_calendar_half_positive"] is True and x["later_stage_outcomes_opened"] is False
 assert sha(R)=="ea944ea942fa485e065e6f275a5d5ee5700f92ce7d348777805a058d1e597af5"
def test_hvtccr_later_stages_remain_sealed():
 for stage in ("test","eval","final"):
  assert not Path(f"results/high_volatility_quote_turnover_concentration_continuation_relay_{stage}_economics_2026-08-10.json").exists()
