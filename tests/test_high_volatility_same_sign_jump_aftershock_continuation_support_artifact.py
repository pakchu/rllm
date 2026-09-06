import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_same_sign_jump_aftershock_continuation_support_2026-08-13.json");CLOCK=Path("data/high_volatility_same_sign_jump_aftershock_continuation_clocks_2023_2026.csv.gz")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_terminal_source_rejection_is_sealed_and_outcome_blind():
 x=json.loads(RESULT.read_text());assert sha(RESULT)=="9c0bc855a5d049d73657f1d1dca8d27d0330066b67dd1191d7687e294313dba5";assert x["policy_id"]=="HVSJAC-8" and not x["support_passed"] and x["decision"]=="terminal_source_support_reject";assert not x["advance_to_gross9_novelty"] and not x["advance_to_economic_outcomes"] and not x["postentry_return_pnl_execution_price_opened"] and not x["gross9_rows_opened"]
def test_frozen_failure_pattern_and_hashes():
 x=json.loads(RESULT.read_text());assert {k:v["events"] for k,v in x["support"].items()}=={"train":3,"test":8,"eval":6,"final":13};assert not x["support_checks"]["train_minimum_events"] and not x["support_checks"]["train_side_balance"] and not x["support_checks"]["test_minimum_events"] and not x["support_checks"]["eval_minimum_events"] and not x["support_checks"]["eval_side_balance"];assert x["clock"]=={"path":str(CLOCK),"sha256":"4a772b3f0e45cc6ba0d4b09f654521249fa8cd26fbe9072d949671180c2d8cbd","rows":30};assert x["source_manifest"]["sha256"]=="df58cf1150e4d0e1a35731109652e147f73bd4a82ba9462fb7d1edb0417ddd05"
