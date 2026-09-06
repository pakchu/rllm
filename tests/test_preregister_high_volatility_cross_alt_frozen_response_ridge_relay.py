import json
from training import preregister_high_volatility_cross_alt_frozen_response_ridge_relay as p
def test_registration_is_canonical_oos_blind_and_frozen():
 x=p.build();p.validate(x);h=x.pop('manifest_hash');assert p.canonical_hash(x)==h;assert x['policy_id']=='HVCAFRR-8' and not x['oos_outcomes_opened'] and not x['oos_source_incidence_opened'] and not x['gross9_rows_opened'];assert x['model']['ridge_lambda']==1. and x['model']['oos_refit'] is False and x['model']['grid'] is False;assert x['policy']['prediction_rank_min']==.75 and x['policy']['variation_rank_min']==.65;assert x['stopping_rule'].startswith('terminal first failure');json.dumps(x,allow_nan=False)
