import hashlib
import numpy as np
import pandas as pd
from training import build_high_volatility_range_participation_polarity_relay_support as s

def test_polarity_weights_range_and_turnover_direction():
 rows=[]
 for i in range(480):
  up=i<5;rows.append({"open":100.,"high":110. if up else 100.2,"low":99. if up else 99.8,"close":105. if up else 99.9,"quote_asset_volume":1000. if up else 1.,"minute_return":.001 if up else -.0001})
 polarity,magnitude,unweighted,variation=s.polarity_statistics(pd.DataFrame(rows));assert polarity>0 and magnitude==abs(polarity) and variation>0 and np.isfinite(unweighted)
def test_prepare_rejects_incoherent_or_negative_turnover():
 frame=pd.DataFrame({"ts":["2023-01-01T00:00:00Z"],"open":[100.],"high":[101.],"low":[99.],"close":[100.5],"quote_asset_volume":[10.]});assert bool(s.prepare(frame).row_valid.iloc[0]);frame.loc[0,"quote_asset_volume"]=-1;assert not bool(s.prepare(frame).row_valid.iloc[0])
def test_prior_rank_excludes_current_and_midranks(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_prior_blocks",2);monkeypatch.setitem(s.P,"prior_blocks",3);x=s.prior_rank(pd.Series([1.,1.,2.,0.]));assert np.isnan(x.iloc[0]) and np.isnan(x.iloc[1]) and x.iloc[2]==1 and x.iloc[3]==0
def test_onset_uses_previous_valid():
 assert s.previous_valid_onset(pd.Series([False,True,False,True]),pd.Series([True,True,False,True])).tolist()==[False,True,False,False]
def test_schema_blind_and_gzip_deterministic():
 assert not {x.lower() for x in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS)}.intersection({"pnl","funding","execution_price","gross9"});x=s.csv_gz(pd.DataFrame({"x":[1.,2.]}));assert x==s.csv_gz(pd.DataFrame({"x":[1.,2.]})) and hashlib.sha256(x).digest()==hashlib.sha256(x).digest()
def test_evaluator_bound_to_prereg():
 assert s.sha(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA and s.P["polarity_magnitude_rank_min"]==.75 and s.CONTROLS==tuple(s.REG["diagnostic_controls"]["names"])
