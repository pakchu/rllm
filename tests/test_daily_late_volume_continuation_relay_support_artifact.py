import hashlib,json
from pathlib import Path
from training import build_daily_late_volume_continuation_relay_support as support
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_dlvcr_source_failure_is_terminal_and_outcome_sealed():
 r=json.loads(support.RESULT.read_text());assert r["policy_id"]=="DLVCR-12";assert r["support_passed"] is False;assert r["decision"]=="terminal_source_support_reject";assert r["advance_to_gross9_novelty"] is False;assert r["advance_to_economic_outcomes"] is False;assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False;assert [r["support"][s]["events"] for s in ("train","test","eval","final")]==[11,11,10,2]
def test_dlvcr_artifact_hashes_bind_frozen_files():
 r=json.loads(support.RESULT.read_text());assert r["manifest_hash"]==support.chash({k:v for k,v in r.items() if k!="manifest_hash"})
 for key in ("preregistration","source_manifest"):
  x=r[key];assert x["sha256"]==sha(Path(x["path"]))
 assert r["clock"]["sha256"]==sha(Path(r["clock"]["path"]))
 for x in r["controls"].values():assert x["promotion_authorized"] is False and x["sha256"]==sha(Path(x["path"]))
