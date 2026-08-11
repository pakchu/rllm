import hashlib
import numpy as np
import pandas as pd
from training import build_high_volatility_volume_price_confirmation_zero_cross_relay_support as s

def test_vpci_formula(monkeypatch):
 monkeypatch.setitem(s.P,"short_periods",2);monkeypatch.setitem(s.P,"long_periods",3)
 close=pd.Series([1.,2.,4.]);volume=pd.Series([1.,1.,2.]);x=s.vpci_values(close,volume,pd.Series([True]*3))
 assert np.isclose(x.vwma_short.iloc[2],10/3) and np.isclose(x.vwma_long.iloc[2],11/4)
 assert np.isclose(x.vpc.iloc[2],11/4-7/3) and np.isclose(x.vpr.iloc[2],(10/3)/3)
 assert np.isclose(x.volume_multiplier.iloc[2],1.5/(4/3))
 assert np.isclose(x.vpci.iloc[2],x.vpc.iloc[2]*x.vpr.iloc[2]*x.volume_multiplier.iloc[2])

def test_zero_cross_side():
 assert s.zero_cross_side(pd.Series([-1.,0.,1.,2.,0.,-1.])).tolist()==[0,0,1,0,0,-1]

def test_prior_rank_excludes_current(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_variation_history_decisions",2);monkeypatch.setitem(s.P,"variation_history_decisions",3)
 r=s.prior_rank(pd.Series([1.,2.,np.nan,3.]));assert np.isnan(r.iloc[1]) and r.iloc[3]==1

def panel():return pd.DataFrame({"source_valid":[True]*7,"vpci_side":[0,0,1,0,-1,0,0],"vpc_side":[0,1,0,0,0,-1,0],"variation_rank":[.8,.8,.8,.8,.4,.8,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 vpc,side,_=s.active(panel(),"vpc_only_zero_cross");assert vpc.iloc[1] and vpc.iloc[5] and side[vpc].tolist()==[1,-1]
 stale,side,_=s.active(panel(),"one_bar_stale_cross");assert stale.iloc[3] and side.iloc[3]==1
 forced,side,_=s.active(panel(),"forced_long");assert side[forced].eq(1).all()

def test_source_is_blind_and_hash_bound():
 assert "funding" not in s.QUERY.lower() and "gross9" not in s.QUERY.lower()
 assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
