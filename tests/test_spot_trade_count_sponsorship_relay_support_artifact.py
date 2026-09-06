import hashlib,json
from pathlib import Path
import pandas as pd
from training import build_spot_trade_count_sponsorship_relay_support as support
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_stcsr_source_rejection_is_outcome_sealed():
 assert sha(support.RESULT)=="631fd97d58a43bf4f56b7a551fd83e18d7f39ec029940e3b329a95ba42456a1b";r=json.loads(support.RESULT.read_text());assert r["policy_id"]=="STCSR-12";assert r["support_passed"] is False;assert r["decision"]=="terminal_source_support_reject";assert r["advance_to_gross9_novelty"] is False;assert r["advance_to_economic_outcomes"] is False;assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False;assert [r["support"][s]["events"] for s in ("train","test","eval","final")]==[0,0,0,0]
def test_stcsr_support_hashes_bind_frozen_files_and_sparse_source():
 r=json.loads(support.RESULT.read_text());assert r["manifest_hash"]==support.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"});assert r["clock"]["sha256"]==sha(Path(r["clock"]["path"]));m=json.loads(support.SOURCE_MANIFEST.read_text());assert m["output"]["valid_rows"]==26;assert m["output"]["sha256"]==sha(Path(m["output"]["path"]));f=pd.read_csv(support.PANEL,compression="gzip");assert f.spot_count_share_rank.notna().sum()==0;assert f.variation_rank.notna().sum()==0
 for x in r["controls"].values():assert x["promotion_authorized"] is False and x["sha256"]==sha(Path(x["path"]))
