import hashlib,json
from pathlib import Path
from training import build_high_volatility_ordinal_pattern_classifier_relay_support as s
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvocpr_support_pass_is_outcome_sealed():
 p=json.loads(s.RESULT.read_text());assert sha(s.RESULT)=="3e1da99925f558654ba51cba341df0d3b359a6073d2f9fbfa5de906a4a92983a";assert p["policy_id"]=="HVOCPR-8" and p["support_passed"] is True;assert p["advance_to_gross9_novelty"] is True and p["advance_to_economic_outcomes"] is False;assert p["oos_postentry_return_pnl_execution_price_opened"] is False and p["gross9_rows_opened"] is False;assert [p["support"][x]["events"] for x in ("train","test","eval","final")]==[91,183,163,111]
def test_hvocpr_support_hashes_bind():
 p=json.loads(s.RESULT.read_text());assert p["manifest_hash"]==s.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"})
 for k in ("preregistration","source_manifest","model_freeze"):assert p[k]["sha256"]==sha(Path(p[k]["path"]))
 assert p["clock"]["sha256"]==sha(Path(p["clock"]["path"]))
 for item in p["controls"].values():assert item["promotion_authorized"] is False and item["sha256"]==sha(Path(item["path"]))
