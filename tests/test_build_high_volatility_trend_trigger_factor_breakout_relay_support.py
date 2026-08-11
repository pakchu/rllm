import hashlib
import numpy as np
import pandas as pd
from training import build_high_volatility_trend_trigger_factor_breakout_relay_support as s

def test_ttf_adjacent_windows(monkeypatch):
 monkeypatch.setitem(s.P,"trend_trigger_periods",2)
 high=pd.Series([5.,6.,8.,10.]);low=pd.Series([1.,2.,4.,7.]);valid=pd.Series([True]*4)
 x=s.trend_trigger_factor(high,low,valid)
 assert x.current_high.iloc[3]==10 and x.current_low.iloc[3]==4
 assert x.prior_high.iloc[3]==6 and x.prior_low.iloc[3]==1
 assert x.buy_power.iloc[3]==9 and x.sell_power.iloc[3]==2
 assert np.isclose(x.ttf.iloc[3],200*7/11)

def test_trigger_sides_strict_crossings():
 ttf=pd.Series([90.,100.,101.,120.,100.,-100.,-101.,-120.])
 assert s.trigger_sides(ttf,100.,-100.).tolist()==[0,0,1,0,0,0,-1,0]

def test_prior_rank_excludes_current(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_variation_history_decisions",2);monkeypatch.setitem(s.P,"variation_history_decisions",3)
 r=s.prior_rank(pd.Series([1.,2.,np.nan,3.]));assert np.isnan(r.iloc[1]) and r.iloc[3]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"outer_side":[0,0,1,0,-1,0,0],"zero_side":[0,1,0,0,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 zero,side,_=s.active(panel(),"zero_line_crossover");assert zero.iloc[1] and zero.iloc[5] and side[zero].tolist()==[1,-1]
 stale,side,_=s.active(panel(),"one_bar_stale_trigger");assert stale.iloc[3] and side.iloc[3]==1
 forced,side,_=s.active(panel(),"forced_long");assert side[forced].eq(1).all()

def test_source_is_blind_and_hash_bound():
 assert "funding" not in s.QUERY.lower() and "gross9" not in s.QUERY.lower()
 assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
