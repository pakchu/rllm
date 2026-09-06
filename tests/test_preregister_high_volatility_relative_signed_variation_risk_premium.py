import hashlib,json
from training import preregister_high_volatility_relative_signed_variation_risk_premium as p
def test_frozen_singleton_blind():
 v=p.build();p.validate(v);assert v["policy_id"]=="HVRSVRP-12" and v["policy"]["five_minute_bars"]==288 and v["policy"]["lower_signed_variation_rank_max"]==.2 and v["policy"]["upper_signed_variation_rank_min"]==.8 and v["clock"]["hold"]=="12 elapsed hours";b=v["research_boundary"];assert b["candidate_count"]==1 and b["grid"] is False and b["candidate_incidence_opened"] is False and b["postentry_return_or_pnl_opened"] is False and b["gross9_rows_opened"] is False
def test_written_matches_builder_and_utf8_hash():
 v=json.loads(p.DEFAULT_OUTPUT.read_text());assert v==p.build();core={k:x for k,x in v.items() if k!="manifest_hash"};assert v["manifest_hash"]==p.canonical_hash(core);expected=hashlib.sha256(json.dumps({"한글":"signed"},sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert p.canonical_hash({"한글":"signed"})==expected
