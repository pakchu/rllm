import hashlib,json
from training import preregister_high_volatility_variance_signature_reversal_relay as p

def test_frozen_singleton_blind():
 v=p.build();p.validate(v);assert v['policy_id']=='HVVSR-24' and v['policy']['signature_rank_min']==.8 and v['policy']['variation_rank_min']==.65 and v['clock']['hold']=='24 elapsed hours';assert v['research_boundary']['candidate_count']==1 and v['research_boundary']['grid'] is False and v['research_boundary']['candidate_incidence_opened'] is False and v['research_boundary']['postentry_return_or_pnl_opened'] is False and v['research_boundary']['gross9_rows_opened'] is False

def test_written_matches_builder_and_utf8_hash():
 v=json.loads(p.DEFAULT_OUTPUT.read_text());assert v==p.build();core={k:x for k,x in v.items() if k!='manifest_hash'};assert v['manifest_hash']==p.canonical_hash(core);expected=hashlib.sha256(json.dumps({'한글':'signature'},sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert p.canonical_hash({'한글':'signature'})==expected
