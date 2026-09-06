from training import preregister_high_volatility_pre_funding_pressure_release_reversal as p
def test_manifest_and_boundaries():
 x=p.build();p.validate(x);assert x['policy_id']=='HVPFPR-6';assert x['singleton'];assert not x['outcomes_opened'];assert not x['source_incidence_opened'];assert not x['gross9_rows_opened']
def test_signal_and_accounting_are_frozen():
 x=p.build();assert x['features']['pressure_alignment']=='strict sign(premium_pressure)=strict sign(btc_return)';assert x['clock']['hold']=='6 elapsed hours';assert x['policy']['return_tail_rank_min']==.75;assert x['economic_gates']['cagr_to_strict_mdd_min']==3.
def test_no_funding_value_or_repair():
 x=p.build();assert x['features']['actual_funding_value']=='not read or used';assert x['source_support_gates']['minimum_events']=={'train':8,'test':12,'eval':12,'final':8};assert 'no window' in x['stopping_rule']
