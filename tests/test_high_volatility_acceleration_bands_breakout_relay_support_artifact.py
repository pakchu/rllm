import hashlib,json
from pathlib import Path
from training import build_high_volatility_acceleration_bands_breakout_relay_support as support
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def test_support_pass_is_blind_and_frozen():
 r=json.loads(support.RESULT.read_text());assert r["support_passed"] is True and r["decision"]=="pass_to_novelty" and r["advance_to_economic_outcomes"] is False
 assert r["postentry_return_pnl_execution_price_opened"] is False and r["funding_values_opened"] is False and r["gross9_rows_opened"] is False
 assert [r["support"][s]["events"] for s in ("train","test","eval","final")]==[9,21,14,10]
def test_support_hashes_bind_artifacts():
 r=json.loads(support.RESULT.read_text());assert r["manifest_hash"]==support.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
 assert sha(support.RESULT)=="2ba3e9772d95e9a24d2645795c221645c30aa092648b9027aa0d2bf1ac9b3f13" and r["clock"]["sha256"]==sha(Path(r["clock"]["path"]))
 for v in r["controls"].values():assert v["promotion_authorized"] is False and v["sha256"]==sha(Path(v["path"]))
