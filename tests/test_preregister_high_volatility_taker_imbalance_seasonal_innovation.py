import json
from training import preregister_high_volatility_taker_imbalance_seasonal_innovation as prereg
def test_preregistration_is_frozen_blind_and_causal():
 x=prereg.build();prereg.validate(x);assert x['policy_id']=='HVTISI-8' and x['singleton'] and not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened'];assert x['features']['no_imputation'] and x['clock']['entry']=='exact BTCUSDT perpetual D+5m open' and not x['research_boundary']['grid'] and not x['research_boundary']['repair_of_prior_candidate']
def test_hash_and_gates():
 x=prereg.build();core={k:v for k,v in x.items() if k!='manifest_hash'};assert x['manifest_hash']==prereg.canonical_hash(core) and 'NaN' not in json.dumps(x,ensure_ascii=False,allow_nan=False);assert x['policy']['same_slot_history']==90 and x['policy']['same_slot_minimum']==60 and x['policy']['innovation_rank_min']==.75;assert x['source_support_gates']['minimum_events']=={'train':8,'test':12,'eval':12,'final':8} and x['economic_gates']['cagr_to_strict_mdd_min']==3 and x['economic_gates']['stress_cagr_to_strict_mdd_min']==2.5
