import hashlib
import numpy as np
import pandas as pd
from training import build_high_volatility_pastor_stambaugh_temporary_impact_reversal_support as s

def test_liquidity_gamma_recovers_temporary_impact(monkeypatch):
 monkeypatch.setitem(s.P,"bars_per_source",288);turnover=np.linspace(.5,1.5,288);returns=np.empty(288);returns[0]=.01
 mean=turnover.mean()
 for i in range(287):returns[i+1]=.2*returns[i]-.001*np.sign(returns[i])*turnover[i]/mean
 assert np.isclose(s.liquidity_gamma(returns,turnover,True),-.001,atol=1e-10)
 assert s.liquidity_gamma(np.ones(287),np.ones(287),True)!=s.liquidity_gamma(np.ones(287),np.ones(287),True)

def test_prepare_coherent_prices_and_turnover():
 frame=pd.DataFrame({"ts":["2023-01-01T00:00:00Z"],"open":[100.],"high":[101.],"low":[99.],"close":[100.5],"quote_asset_volume":[10.]})
 assert bool(s.prepare(frame).row_valid.iloc[0]);frame.loc[0,"quote_asset_volume"]=-1;assert not bool(s.prepare(frame).row_valid.iloc[0])

def test_prior_rank_excludes_current():
 r=s.prior_rank(pd.Series([1.,1.,2.,0.]),3,2);assert np.isnan(r.iloc[1]) and r.iloc[2]==1 and r.iloc[3]==0

def test_onset_skips_invalid_rows():
 assert s.previous_valid_onset(pd.Series([False,True,False,True]),pd.Series([True,True,False,True])).tolist()==[False,True,False,False]

def panel():
 times=pd.date_range("2023-07-01",periods=5,freq="4h",tz="UTC")
 return pd.DataFrame({"decision_time":times,"feature_available_time":times,"source_valid":[True]*5,"five_minute_count":[288]*5,"gamma":[-.1]*5,"gamma_omit_return":[-.1]*5,"illiquidity":[.1]*5,"illiquidity_rank":[.1,.9,.9,.1,.1],"omit_illiquidity":[.1]*5,"omit_rank":[.9,.1,.1,.1,.9],"realized_variation":[1.]*5,"variation_rank":[.8]*5,"completed_return":[-.01,.02,.03,-.02,.01],"eligible":[False,True,True,False,False],"onset":[False,True,False,False,False]})

def test_primary_and_omit_control_onsets_and_fade_side():
 p=panel();primary,side,_=s.active(p);omit,_,_=s.active(p,"omit_return_control")
 assert primary.tolist()==[False,True,False,False,False] and side.tolist()==[1,-1,-1,1,-1]
 assert omit.tolist()==[False,False,False,False,True]

def test_blind_deterministic_and_bound():
 forbidden={"pnl","funding","execution_price","gross9"};assert not {x.lower() for x in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS)}.intersection(forbidden)
 frame=pd.DataFrame({"x":[1.,2.]});assert s.csv_gz(frame)==s.csv_gz(frame)
 assert s.sha(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA and s.CONTROLS==tuple(s.REG["diagnostic_controls"]["names"])
