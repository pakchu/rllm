from training import preregister_high_volatility_haar_coarse_energy_migration_continuation as p
def test_manifest_singleton_and_boundary():
 x=p.build();p.validate(x);assert x['policy_id']=='HVHCEM-12';assert x['singleton'];assert not x['outcomes_opened'];assert not x['source_incidence_opened'];assert not x['gross9_rows_opened']
def test_frozen_haar_and_clock():
 x=p.build();assert x['policy']['bars']==128;assert x['policy']['coarse_levels']==[5,6,7];assert x['clock']['entry']=='exact BTCUSDT decision+5m open';assert x['clock']['hold']=='12 elapsed hours'
def test_gates_and_no_repair():
 x=p.build();assert x['source_support_gates']['minimum_events']=={'train':8,'test':12,'eval':12,'final':8};assert x['novelty_gates']['absolute_signed_exposure_pearson_max']==.35;assert x['economic_gates']['cagr_to_strict_mdd_min']==3.;assert 'no scale' in x['stopping_rule']
