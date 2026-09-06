import hashlib
import numpy as np
import pandas as pd
from training import build_high_volatility_multifractal_coherence_relay_support as s

def test_multifractal_statistics_are_finite_and_deterministic():
 returns=np.random.default_rng(20260810).normal(0,.001,480);a=s.multifractal_statistics(returns);b=s.multifractal_statistics(returns);assert a==b and all(np.isfinite(a)) and a[2]==abs(a[0]-a[1]) and a[3]>0
def test_multifractal_statistics_reject_wrong_shape():
 assert all(np.isnan(x) for x in s.multifractal_statistics(np.ones(479)))
def test_prepare_requires_coherent_prices():
 f=pd.DataFrame({"ts":["2023-01-01T00:00:00Z"],"open":[100.],"high":[101.],"low":[99.],"close":[100.5]});assert bool(s.prepare(f).row_valid.iloc[0]);f.loc[0,"high"]=99.;assert not bool(s.prepare(f).row_valid.iloc[0])
def test_prior_rank_excludes_current_and_midranks(monkeypatch):
 monkeypatch.setitem(s.P,"minimum_prior_blocks",2);monkeypatch.setitem(s.P,"prior_blocks",3);x=s.prior_rank(pd.Series([1.,1.,2.,0.]));assert np.isnan(x.iloc[0]) and np.isnan(x.iloc[1]) and x.iloc[2]==1 and x.iloc[3]==0
def test_onset_uses_previous_valid():
 assert s.previous_valid_onset(pd.Series([False,True,False,True]),pd.Series([True,True,False,True])).tolist()==[False,True,False,False]
def test_schema_is_blind_and_gzip_deterministic():
 assert not {x.lower() for x in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS)}.intersection({"pnl","funding","execution_price","gross9"});a=s.csv_gz(pd.DataFrame({"x":[1.,2.]}));assert a==s.csv_gz(pd.DataFrame({"x":[1.,2.]})) and hashlib.sha256(a).digest()==hashlib.sha256(a).digest()
def test_evaluator_is_bound_to_preregistration():
 assert s.sha(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA and s.P["gap_rank_max"]==.25 and s.CONTROLS==tuple(s.REG["diagnostic_controls"]["names"])
