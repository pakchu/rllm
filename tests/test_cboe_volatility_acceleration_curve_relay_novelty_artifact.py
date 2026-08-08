import hashlib,json
from training import evaluate_cboe_volatility_acceleration_curve_relay_gross9_novelty as novelty
def test_cvacr_novelty_is_frozen_pass():
 assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest()=="1953e2e48e754d14d78865172a476db6326d339cdde95703c1fd88dfac37ca28";p=json.loads(novelty.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==novelty.chash(core);assert p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True;assert p["evidence_boundary"]["outcomes_opened"] is False and p["evidence_boundary"]["btc_price_or_return_rows_opened"]==0;assert max(v["metrics"]["one_to_one_6h_max_matched_share"] for v in p["gross9_sleeves"].values())==.17
