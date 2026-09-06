import math
from training import preregister_high_volatility_ehlers_deviation_scaled_fisher_reentry_relay as p
def test_frozen():
 x=p.build();p.validate(x);q=x["policy"];assert x["policy_id"]=="HVEFD-24" and q["periods"]==40 and q["smoother_periods"]==20 and q["overbought"]==2 and q["oversold"]==-2
 assert math.isclose(q["raw_scaled_equivalent"],2*math.tanh(2)) and not x["research_boundary"]["candidate_incidence_opened"]
