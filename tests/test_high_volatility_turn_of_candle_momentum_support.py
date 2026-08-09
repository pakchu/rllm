import numpy as np,pandas as pd
from training import build_high_volatility_turn_of_candle_momentum_support as support
def test_rank_is_strict_prior():
 d=pd.date_range("2021-01-01","2021-04-15",freq="5min",tz="UTC",inclusive="left");p=np.exp(np.arange(len(d))*1e-7)*100;m=pd.DataFrame({"date":d,"open":p,"high":p*1.001,"low":p*.999,"close":p});s=support.score_states(m);assert s.variation_rank.iloc[:60].isna().all();assert np.isfinite(s.variation_rank.iloc[60])
def test_primary_clock_waits_and_uses_opening_sign():
 s=pd.DataFrame([{"day":pd.Timestamp("2023-07-02T00:00:00Z"),"decision_time":pd.Timestamp("2023-07-02T00:30:00Z"),"opening_return":.01,"second_half_hour_return":-.02,"pre_day_variation":.1,"variation_rank":.8}]);c=support.build_clock(s);assert c.side.tolist()==[1];assert c.entry_time.iloc[0]==pd.Timestamp("2023-07-02T23:30:00Z");assert c.exit_time.iloc[0]==pd.Timestamp("2023-07-03T00:00:00Z");assert support.build_clock(s,"opening_half_hour_fade").side.tolist()==[-1];assert support.build_clock(s,"second_half_hour_momentum").side.tolist()==[-1]
