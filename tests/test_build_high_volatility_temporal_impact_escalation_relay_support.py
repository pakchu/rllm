import numpy as np,pandas as pd
from training import build_high_volatility_temporal_impact_escalation_relay_support as support

def test_half_impacts_use_exact_halves_and_quote_turnover():
 opens=np.ones(96);closes=np.ones(96);quote=np.ones(96);closes[47]=1.1;closes[95]=1.2
 first,second,q1,q2,i1,i2,ratio=support.half_impacts(opens,closes,quote)
 assert np.isclose(first,np.log(1.1)) and np.isclose(second,np.log(1.2));assert q1==48 and q2==48;assert np.isclose(ratio,i2/i1)
def test_rank_excludes_current():
 x=pd.Series(np.arange(253,dtype=float));r=support.prior_midrank(x);assert np.isnan(r.iloc[:252]).all();assert r.iloc[252]==1.
def frame():return pd.DataFrame({"source_valid":[True]*3,"half_direction":[1.,-1.,1.],"block_return":[.1,.1,-.1],"impact_escalation_rank":[.85,.85,.7],"variation_rank":[.7,.6,.7]})
def test_primary_and_controls():
 f=frame();a,s,_=support.conditions(f);assert a.tolist()==[True,False,False];assert s.tolist()==[1,-1,1];assert support.conditions(f,"no_variation_gate")[0].tolist()==[True,True,False];assert support.conditions(f,"no_impact_escalation_gate")[0].tolist()==[True,False,True];assert support.conditions(f,"net_block_return_side")[1].tolist()==[1,1,-1];assert support.conditions(f,"direction_flip")[1].tolist()==[-1,1,-1]
def test_queries_and_prereg_hash():
 assert "bars_binance" in support.BAR_QUERY;assert "funding_rates_binance" not in support.BAR_QUERY;assert support.PREREG_SHA=="38d52446670b9d9c99abfef6592a39c7f1fa88c09a886c6dccc34a9f6523f123"
