import hashlib,json
from training import preregister_high_volatility_last_zero_crossing_trend_relay as p
def test_frozen_singleton_blind():
 v=p.build();p.validate(v);assert v["policy_id"]=="HVLZCTR-6" and v["policy"]["block_minutes"]==720 and v["policy"]["last_passage_rank_max"]==.3 and v["clock"]["hold"]=="6 elapsed hours";b=v["research_boundary"];assert b["candidate_count"]==1 and b["grid"] is False and b["candidate_incidence_opened"] is False and b["postentry_return_or_pnl_opened"] is False and b["gross9_rows_opened"] is False
def test_written_matches_builder_and_utf8_hash():
 v=json.loads(p.DEFAULT_OUTPUT.read_text());assert v==p.build();core={k:x for k,x in v.items() if k!="manifest_hash"};assert v["manifest_hash"]==p.canonical_hash(core);expected=hashlib.sha256(json.dumps({"한글":"zero"},sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert p.canonical_hash({"한글":"zero"})==expected
