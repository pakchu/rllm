import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_taker_imbalance_flow_persistence_support_2026-08-13.json");CLOCK=Path("data/high_volatility_taker_imbalance_flow_persistence_clocks_2023_2026.csv.gz")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_source_pass_is_sealed_and_outcome_blind():
 x=json.loads(RESULT.read_text());assert sha(RESULT)=="9942814f49d32d548c3933c2ea21946a33bca8ab327f7a0466fa31b7f54e50a2";assert x["policy_id"]=="HVTIFP-8" and x["support_passed"] and x["decision"]=="pass_to_novelty";assert x["advance_to_gross9_novelty"] and not x["advance_to_economic_outcomes"] and not x["postentry_return_pnl_execution_price_opened"] and not x["gross9_rows_opened"]
def test_all_source_gates_and_hashes_pass():
 x=json.loads(RESULT.read_text());assert all(x["support_checks"].values());assert {k:v["events"] for k,v in x["support"].items()}=={"train":20,"test":38,"eval":55,"final":28};assert x["clock"]=={"path":str(CLOCK),"sha256":"d09e1bbd40238d3d75b3d487b08901fc8ea85a1990617ec607e75d8b338cab91","rows":141};assert x["source_manifest"]["sha256"]=="aef37a1c11001a774db6bfa2af18ab67f9dc10946c6f7f834300f8a2072472ab"
