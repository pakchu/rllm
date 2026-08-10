import json
from training import preregister_high_volatility_corwin_schultz_spread_expansion_reversal as p
def test_manifest_and_singleton():
 x=p.build();p.validate(x);assert x['policy_id']=='HVCSER-12';assert x['singleton'] is True
def test_boundaries_and_gates_are_frozen():
 x=p.build();assert x['clock']['entry']=='exact BTCUSDT decision+5m open';assert x['clock']['hold']=='12 elapsed hours';assert x['source_support_gates']['minimum_events']=={'train':8,'test':12,'eval':12,'final':8};assert x['novelty_gates']['candidate_near_6h_share_max']==.35
def test_outcomes_remain_sealed():
 x=p.build();assert x['outcomes_opened'] is False;assert x['source_incidence_opened'] is False;assert x['gross9_rows_opened'] is False;assert x['research_boundary']['postentry_return_or_pnl_opened'] is False
def test_output_roundtrip(tmp_path):
 x=p.build();q=tmp_path/'x.json';q.write_text(json.dumps(x));p.validate(json.loads(q.read_text()))
