import hashlib,json
from pathlib import Path
from training import build_cboe_volatility_surface_extended as builder
ROOT=Path("data/cboe_volatility_surface_2021_2026");PANEL=ROOT/"cboe_volatility_surface_2021-01-01_2026-08-07.csv.gz";MANIFEST=ROOT/"manifest.json"
def test_extended_cboe_surface_is_outcome_blind_and_hash_bound():
 assert hashlib.sha256(PANEL.read_bytes()).hexdigest()=="42eb1093f5167aec9c71a4733ab3451e40807c81dc7cb49568a6a0c634267ba0";assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest()=="ec1dd33efcee29b75c80294fb594969cd1b12a9343fc40f888525db4400bc936";p=json.loads(MANIFEST.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==builder.canonical_hash(core)=="1eb07fc5018fe55aca5748ff9c676988d60fdc34cecefa4c85c074791c80d8f3";assert p["common_rows"]==1404 and p["common_last"]=="2026-08-07";assert p["outcomes_opened"] is False and p["btc_price_return_funding_or_pnl_opened"] is False and p["raw_responses_persisted"] is False
