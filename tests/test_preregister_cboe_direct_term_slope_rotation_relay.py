from training import preregister_cboe_direct_term_slope_rotation_relay as p
def test_boundary_and_sources():
 x=p.build();p.validate(x);assert x['policy_id']=='CVDTSR-24';assert x['source_incidence_opened'] is False;assert x['research_boundary']['grid'] is False;assert x['source_plan']['surface']['sha256']==p.SURFACE_SHA
def test_gates():
 x=p.build();assert x['policy']['rotation_rank_min']==.65;assert x['source_support_gates']['minimum_events']=={'train':8,'test':12,'eval':12,'final':8};assert x['economic_gates']['cagr_to_strict_mdd_min']==3.
