import hashlib,json
from training import preregister_high_volatility_rogers_satchell_excursion_asymmetry_reversal_relay as p
def test_singleton_blind_frozen_contract():
 v=p.build();p.validate(v);assert v["policy_id"]=="HVRSAR-8" and v["research_boundary"]["candidate_count"]==1 and v["research_boundary"]["grid"] is False
 assert v["research_boundary"]["candidate_incidence_opened"] is False and v["research_boundary"]["postentry_return_or_pnl_opened"] is False and v["research_boundary"]["gross9_rows_opened"] is False
 assert v["policy"]["bars"]==6 and v["policy"]["asymmetry_rank_min"]==.75 and v["clock"]["hold"]=="8 elapsed hours"
def test_written_contract_matches_builder():
 v=json.loads(p.DEFAULT_OUTPUT.read_text());assert v==p.build();core={k:x for k,x in v.items() if k!="manifest_hash"};assert v["manifest_hash"]==p.canonical_hash(core);assert hashlib.sha256(p.DEFAULT_OUTPUT.read_bytes()).hexdigest()
