import json
from training import preregister_high_volatility_funding_acceleration_divergence_relay as p
def test_registration_is_canonical_blind_and_frozen():
 x=p.build();p.validate(x);h=x.pop('manifest_hash');assert p.canonical_hash(x)==h;assert x['policy_id']=='HVFADR-8' and not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened'];assert x['policy']['change_extremity_rank_min']==.75 and x['policy']['variation_rank_min']==.65;assert x['stopping_rule'].startswith('terminal first failure');json.dumps(x,allow_nan=False)
