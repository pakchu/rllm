import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_speculative_alt_rotation_relay_support_2026-08-12.json")
def test_hvsarr_source_artifact_passes_and_is_hash_bound():
 r=json.loads(RESULT.read_text());assert r["policy_id"]=="HVSARR-8";assert r["support_passed"] is True;assert r["advance_to_gross9_novelty"] is True;assert r["advance_to_economic_outcomes"] is False;assert r["decision"]=="pass_to_novelty";assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False;assert {s:r["support"][s]["events"] for s in ("train","test","eval","final")}=={"train":50,"test":88,"eval":99,"final":43};assert all(r["support_checks"].values());p=Path(r["source_manifest"]["path"]);assert hashlib.sha256(p.read_bytes()).hexdigest()==r["source_manifest"]["sha256"]
