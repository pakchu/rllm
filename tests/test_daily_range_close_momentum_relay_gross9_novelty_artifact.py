import hashlib,json
from training import evaluate_daily_range_close_momentum_relay_gross9_novelty as novelty

def test_drcmr_novelty_artifact_passes_without_outcomes():
 assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest()=="1a2fae500b344c8d1bb896cf15af99dcc88d07e9c545e78cd297ffb8141baad2";p=json.loads(novelty.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==novelty.chash(core);assert p["gross9_novelty_status"]=="passed" and p["every_gross9_sleeve_passed"] is True and p["advance_to_economic_outcomes"] is True;assert p["evidence_boundary"]["outcomes_opened"] is False and p["evidence_boundary"]["btc_price_or_return_rows_opened"]==0
