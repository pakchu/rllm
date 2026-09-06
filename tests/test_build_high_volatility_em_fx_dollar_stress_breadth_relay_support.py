import numpy as np,pandas as pd
from training import build_high_volatility_em_fx_dollar_stress_breadth_relay_support as s

def test_causal_z_and_midrank_exclude_current():
 x=pd.Series(np.arange(61,dtype=float));assert s.causal_z(x,90,60).iloc[60]>0;assert s.midrank(x,90,60).iloc[60]==1.

def test_primary_side_is_opposite_dollar_direction():
 f=pd.DataFrame({'source_valid':[True],'common_dollar_direction':[1],'agreeing_pairs':[4],'median_absolute_pair_z':[2.],'variation_rank':[1.]});active,side=s.conditions(f,'primary');assert active.iloc[0] and side.iloc[0]==-1

def test_frozen_contract():
 assert s.SYMBOLS==('USDMXN','USDKRW','USDINR','USDCNY');assert s.P['minimum_agreeing_pairs']==3 and s.P['decision_hour_utc']==22 and s.P['decision_minute']==30;assert s.PREREG_SHA=='09be9ec220761d2a0956e81d48ec5740808ce3895d34ea13077d02e7e2c60cc8'
