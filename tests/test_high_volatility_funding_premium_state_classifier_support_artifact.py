import hashlib,json
from pathlib import Path
from training import build_high_volatility_funding_premium_state_classifier_support as s
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvfpsc_support_failure_is_terminal_and_outcome_sealed():
 p=json.loads(s.RESULT.read_text());assert sha(s.RESULT)=="d553ed60be57fc3baacd09c514e30d7a68ca8f8bb1aae8c6354bf74bc76583a1";assert p["support_passed"] is False and p["decision"]=="terminal_source_support_reject";assert p["advance_to_gross9_novelty"] is False and p["advance_to_economic_outcomes"] is False;assert p["oos_postentry_return_pnl_execution_price_opened"] is False;assert all(p["support"][x]["shorts"]==0 for x in ("train","test","eval","final"))
def test_hvfpsc_artifacts_bind():
 p=json.loads(s.RESULT.read_text());assert p["manifest_hash"]==s.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"})
 for k in ("preregistration","source_manifest","model_freeze"):assert p[k]["sha256"]==sha(Path(p[k]["path"]))
 assert p["clock"]["sha256"]==sha(Path(p["clock"]["path"]))
