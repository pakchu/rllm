import hashlib,json
from pathlib import Path
from training import build_cross_asset_volatility_breadth_relay_support as support
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_cavbr_source_support_pass_is_outcome_sealed():
 assert sha(support.RESULT)=="2fd2f5afa866bc72287fc8c5deaf8b6511552f1dc49c3a216583ba21a6376005";r=json.loads(support.RESULT.read_text());assert r["policy_id"]=="CAVBR-12";assert r["support_passed"] is True;assert r["decision"]=="pass_to_novelty";assert r["advance_to_gross9_novelty"] is True;assert r["advance_to_economic_outcomes"] is False;assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False;assert [r["support"][s]["events"] for s in ("train","test","eval","final")]==[26,100,53,48]
def test_cavbr_support_hashes_bind_frozen_files():
 r=json.loads(support.RESULT.read_text());assert r["manifest_hash"]==support.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"});assert r["clock"]["sha256"]==sha(Path(r["clock"]["path"]));assert r["source_manifest"]["sha256"]==sha(Path(r["source_manifest"]["path"]));m=json.loads(support.SOURCE_MANIFEST.read_text());assert m["manifest_hash"]==support.canonical_hash({k:v for k,v in m.items() if k!="manifest_hash"})
 for x in r["controls"].values():assert x["promotion_authorized"] is False and x["sha256"]==sha(Path(x["path"]))
