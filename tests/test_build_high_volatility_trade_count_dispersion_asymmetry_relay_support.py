import numpy as np,pandas as pd
from training import build_high_volatility_trade_count_dispersion_asymmetry_relay_support as s
def test_dispersion_asymmetry_detects_clustered_up_counts():
 r=np.r_[np.ones(240)*.001,np.ones(240)*-.001];c=np.r_[np.tile([1,20],120),np.ones(240)*5];up,dn,fu,fd,a,m,v=s.dispersion_statistics(r,c);assert up==dn==240 and fu>fd and a>0 and v>0 and np.isfinite(m)
def test_prepare_requires_integer_counts():
 f=pd.DataFrame({"ts":["2023-01-01T00:00:00Z"],"open":[100.],"high":[101.],"low":[99.],"close":[100.5],"number_of_trades":[3.]});assert bool(s.prepare(f).row_valid.iloc[0]);f.loc[0,"number_of_trades"]=3.5;assert not bool(s.prepare(f).row_valid.iloc[0])
def test_prior_rank_excludes_current(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_prior_blocks",2);monkeypatch.setitem(s.P,"prior_blocks",3);x=s.prior_rank(pd.Series([1.,1.,2.,0.]));assert np.isnan(x.iloc[0]) and x.iloc[2]==1 and x.iloc[3]==0
def test_onset_uses_previous_valid():assert s.previous_valid_onset(pd.Series([False,True,False,True]),pd.Series([True,True,False,True])).tolist()==[False,True,False,False]
def test_blind_bound():
 assert not {x.lower() for x in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS)}.intersection({"pnl","funding","execution_price","gross9"});assert s.sha(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA and s.P["asymmetry_magnitude_rank_min"]==.75
