import numpy as np,pandas as pd
from training import build_high_volatility_self_normalized_displacement_reversal_support as s
def test_displacement_statistics():
 x=np.linspace(100,110,480);b=pd.DataFrame({"open":x,"close":x+0.01});d,v,z,l=s.displacement_statistics(b);assert d>0 and v>0 and z>0 and l>0
def test_prepare_coherence():
 f=pd.DataFrame({"ts":["2023-01-01T00:00:00Z"],"open":[100.],"high":[101.],"low":[99.],"close":[100.5]});assert bool(s.prepare(f).row_valid.iloc[0]);f.loc[0,"high"]=99;assert not bool(s.prepare(f).row_valid.iloc[0])
def test_prior_rank(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_prior_blocks",2);monkeypatch.setitem(s.P,"prior_blocks",3);x=s.prior_rank(pd.Series([1.,1.,2.,0.]));assert np.isnan(x.iloc[0]) and x.iloc[2]==1 and x.iloc[3]==0
def test_onset():assert s.previous_valid_onset(pd.Series([False,True,False,True]),pd.Series([True,True,False,True])).tolist()==[False,True,False,False]
def test_blind_bound():assert not {x.lower() for x in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS)}.intersection({"pnl","funding","execution_price","gross9"}) and s.sha(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA
