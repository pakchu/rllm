import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_premium_occupation_crowding_reversal_support_2026-08-12.json")
def test_hvpocr_source_artifact_is_terminal_and_hash_bound():
 r=json.loads(RESULT.read_text());assert r["policy_id"]=="HVPOCR-8";assert r["support_passed"] is False;assert r["advance_to_gross9_novelty"] is False;assert r["decision"]=="terminal_source_support_reject";assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["support"]["final"]["events"]==15;assert r["support"]["final"]["max_month_share"]==8/15;assert r["support_checks"]["final_month_concentration"] is False;p=Path(r["source_manifest"]["path"]);assert hashlib.sha256(p.read_bytes()).hexdigest()==r["source_manifest"]["sha256"]
