from training import preregister_high_volatility_execution_count_elasticity_relay as p
def test_singleton_boundary():
 x=p.build();p.validate(x);assert x['policy_id']=='HVECE-8';assert x['source_incidence_opened'] is False;assert x['research_boundary']['grid'] is False
def test_gates():
 x=p.build();assert x['policy']['elasticity_rank_min']==.8;assert x['source_support_gates']['minimum_events']=={'train':8,'test':12,'eval':12,'final':8};assert x['economic_gates']['cagr_to_strict_mdd_min']==3.
