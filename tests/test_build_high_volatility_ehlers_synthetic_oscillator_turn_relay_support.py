import numpy as np
import pandas as pd
from training import build_high_volatility_ehlers_synthetic_oscillator_turn_relay_support as s

def test_hann_high_pass_and_synthetic_warmup_reset():
 assert s.cross_side(pd.Series([-1.,1.,2.,-1.])).tolist()==[0,1,0,-1]
 prices=pd.Series(100+np.sin(np.arange(500)/4)+np.arange(500)*.01);valid=pd.Series([True]*250+[False]+[True]*249);h=s.hann(prices,valid,12);hp=s.high_pass(h,np.isfinite(h),25);x=s.synthetic_oscillator(prices,valid,15,25,4)
 assert h.iloc[10]!=h.iloc[10] and np.isfinite(h.iloc[11]) and hp.iloc[11:14].tolist()==[0.,0.,0.] and np.isnan(hp.iloc[250])
 assert np.isfinite(x.rate.iloc[240]) and x.rate.iloc[251:470].isna().any() and np.isfinite(x.rate.iloc[-1])

def test_published_filter_coefficients():
 values=pd.Series(np.arange(1.,14.));valid=pd.Series([True]*13);weights=1-np.cos(2*np.pi*np.arange(1,13)/13)
 assert np.isclose(s.hann(values,valid,12).iloc[11],np.dot(weights,values.iloc[:12].to_numpy()[::-1])/weights.sum())
 hp=s.high_pass(values,valid,20);q=np.exp(-1.414*np.pi/20);c2=2*q*np.cos(1.414*np.pi/20);c3=-q*q;c1=(1+c2-c3)/4
 assert hp.iloc[:3].tolist()==[0.,0.,0.] and np.isclose(hp.iloc[3],c1*(4-2*3+2))

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series(np.arange(121,dtype=float)));assert np.isnan(r.iloc[119]) and r.iloc[120]==1

def panel():
 return pd.DataFrame({"source_valid":[True]*6,"entry_side":[0,0,1,0,-1,0],"direct_entry_side":[0,-1,0,1,0,0],"wealthlab_entry_side":[0,0,-1,0,1,0],"variation_rank":[.8,.8,.8,.8,.4,.8]})
def test_controls():
 a,z,_=s.active(panel());assert a.tolist()==[False,False,True,False,False,False] and z[a].tolist()==[1]
 assert s.active(panel(),"no_variation_gate")[0].iloc[4]
 raw,side,_=s.active(panel(),"direct_synth_rate");assert raw.iloc[1] and side.iloc[1]==-1 and raw.iloc[3] and side.iloc[3]==1
 osc,side,_=s.active(panel(),"wealthlab_12_25_12");assert osc.iloc[2] and side.iloc[2]==-1
