import hashlib,json
from pathlib import Path
P=Path("results/high_volatility_alt_leadership_rotation_relay_support_2026-08-13.json");SHA="b1dd428521233166f3ad2bfe2ce0a4c4a2ad713ce8bf7f49c2a35b70d1e8b451"
def ch(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def test_source_failure_is_immutable_and_terminal():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==SHA;x=json.loads(P.read_text());h=x.pop("manifest_hash");assert ch(x)==h;assert x["policy_id"]=="HVALRR-8";assert x["support_passed"] is False;assert x["support_checks"]["final_month_concentration"] is False;assert x["support"]["final"]["max_month_share"]==.5;assert x["decision"]=="terminal_source_support_reject";assert x["advance_to_gross9_novelty"] is False;assert x["postentry_return_pnl_execution_price_opened"] is False;assert x["gross9_rows_opened"] is False
