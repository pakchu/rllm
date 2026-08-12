from training import preregister_high_volatility_india_opening_reversal_relay as subject

def test_manifest_is_blind_and_bound():
 x=subject.build();subject.validate(x);assert x['policy_id']=='HVINOR-8';assert not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened'];assert x['source_plan']['fx']['symbol']=='USDINR';assert x['policy']['opening_boundary_utc']=='03:30';assert x['policy']['reversal_rank_min']==.70;assert x['policy']['hold_hours']==8

def test_required_gates():
 x=subject.build();assert x['source_support_gates']['minimum_events']=={'train':8,'test':12,'eval':12,'final':8};assert x['novelty_gates']['candidate_near_6h_share_max']==.35;assert x['economic_gates']['cagr_to_strict_mdd_min']==3.;assert x['economic_gates']['stress_cagr_to_strict_mdd_min']==2.5;assert x['diagnostic_controls']['cannot_be_promoted']
