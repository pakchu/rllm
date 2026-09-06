import json
from training import distill_macro_flow_alpha_candidate as d

def test_distilled_candidate_is_disabled_and_exact():
 c=d.build();assert not c['enabled'] and not c['live_authorized']
 assert c['components']['inverse_dollar_aggressive_flow']['weight']==.75
 assert c['components']['long_regime_flow_switch']['weight']==.25
 assert c['long_short_offset_before_risk_and_cost']
 assert c['evidence']['fresh_2026_06_01_to_09_05']['return_pct']>0
 assert c['evidence']['fresh_stress']['return_pct']>0
 assert c['evidence']['historical_reports']['report2026']['return_pct']<0
