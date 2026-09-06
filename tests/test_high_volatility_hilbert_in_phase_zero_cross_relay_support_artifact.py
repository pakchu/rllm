import hashlib,json
from pathlib import Path
from training import build_high_volatility_hilbert_in_phase_zero_cross_relay_support as support
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_support_pass_is_outcome_sealed():
 r=json.loads(support.RESULT.read_text());assert r["support_passed"] is True and r["decision"]=="pass_to_novelty" and r["advance_to_economic_outcomes"] is False
 assert r["postentry_return_pnl_execution_price_opened"] is False and r["funding_values_opened"] is False and r["gross9_rows_opened"] is False
 assert [r["support"][s]["events"] for s in ("train","test","eval","final")]==[53,92,93,46]
def test_hashes_bind_frozen_artifacts():
 r=json.loads(support.RESULT.read_text());assert r["manifest_hash"]==support.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
 assert sha(support.RESULT)=="47d93b0c4fd393f91f94b2424ad9a1bd7af975b9f4f8f456e3cc5c57dbb0e1f5" and r["clock"]["sha256"]==sha(Path(r["clock"]["path"]))
 for v in r["controls"].values():assert v["promotion_authorized"] is False and v["sha256"]==sha(Path(v["path"]))
