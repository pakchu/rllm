from training import preregister_cross_quote_signed_ticket_leadership as prereg

def test_cqstl_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r['policy_id']=='CQSTL-8';assert r['outcomes_opened'] is False;assert r['source_incidence_opened'] is False;assert r['singleton'] is True;assert r['research_boundary']['candidate_count']==1;assert r['research_boundary']['grid'] is False;assert r['research_boundary']['repair_of_prior_candidate'] is False

def test_cqstl_freezes_signed_ticket_leadership_and_volatility():
 r=prereg.build();p=r['policy'];assert p['aggregate_trade_count_min']==80;assert p['sponsor_rank_min']==.65;assert p['realized_variation_rank_min']==.65;assert p['hold_hours']==8;assert 'sum(signed_taker_flow_btc)/sum(trade_count)' in r['features']['signed_ticket'];assert r['clock']['side']=='common strict USDC/FDUSD signed-ticket sign'

def test_cqstl_is_not_prior_stablecoin_control_promotion():
 r=prereg.build();assert r['research_boundary']['prior_stablecoin_event_sets_reused'] is False;assert r['research_boundary']['prior_candidate_outcomes_used_to_set_cqstl_rule'] is False;assert r['research_boundary']['promoted_prior_control'] is False;assert r['diagnostic_controls']['diagnostic_controls_cannot_be_promoted'] is True

def test_cqstl_hash_binds_core():
 r=prereg.build();assert r['manifest_hash']==prereg.canonical_hash({k:v for k,v in r.items() if k!='manifest_hash'})
