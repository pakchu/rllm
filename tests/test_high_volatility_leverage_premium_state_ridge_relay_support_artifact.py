import hashlib,json
from pathlib import Path
from training import build_high_volatility_leverage_premium_state_ridge_relay_support as support
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvlpsr_source_support_failure_is_terminal_and_outcome_sealed():
 p=json.loads(support.RESULT.read_text());assert sha(support.RESULT)=="46866aeb831d927deba714758f057f3646740e9a7ebaea6d9cfa375f0bd7729b";assert p["support_passed"] is False and p["decision"]=="terminal_source_support_reject";assert p["advance_to_gross9_novelty"] is False and p["advance_to_economic_outcomes"] is False;assert p["oos_postentry_return_pnl_execution_price_opened"] is False and p["gross9_rows_opened"] is False;assert p["support_checks"]["test_month_concentration"] is False and p["support_checks"]["final_month_concentration"] is False
def test_hvlpsr_terminal_artifacts_bind_frozen_inputs():
 p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==support.canonical_hash(core)
 for k in ("preregistration","source_manifest","model_freeze"):assert p[k]["sha256"]==sha(Path(p[k]["path"]))
 assert p["clock"]["sha256"]==sha(Path(p["clock"]["path"]))
 for item in p["controls"].values():assert item["promotion_authorized"] is False and item["sha256"]==sha(Path(item["path"]))
