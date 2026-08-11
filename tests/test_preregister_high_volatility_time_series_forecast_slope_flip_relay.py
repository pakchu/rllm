import hashlib,json
from training import preregister_high_volatility_time_series_forecast_slope_flip_relay as p
def test_frozen_singleton_blind():
 v=p.build();p.validate(v);assert v["policy_id"]=="HVTSF3-24" and v["policy"]["forecast_periods"]==3 and v["clock"]["hold"]=="24 elapsed hours" and v["research_boundary"]["candidate_count"]==1 and v["research_boundary"]["grid"] is False
 assert v["research_boundary"]["candidate_incidence_opened"] is False and v["research_boundary"]["postentry_return_or_pnl_opened"] is False and v["research_boundary"]["gross9_rows_opened"] is False
def test_written_matches_builder():
 v=json.loads(p.DEFAULT_OUTPUT.read_text());assert v==p.build();core={k:x for k,x in v.items() if k!="manifest_hash"};assert v["manifest_hash"]==p.canonical_hash(core);assert hashlib.sha256(p.DEFAULT_OUTPUT.read_bytes()).hexdigest()
