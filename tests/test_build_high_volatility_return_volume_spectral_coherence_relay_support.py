import numpy as np,pandas as pd,pytest
from training import build_high_volatility_return_volume_spectral_coherence_relay_support as s
def test_coherence_is_one_for_identical_wave_and_bounded():
 t=np.arange(96);v=np.exp(10+np.sin(2*np.pi*t/24))-1;r=np.sin(2*np.pi*t/24);assert s.spectral_coherence(v,r,np.arange(1,9))==pytest.approx(1.);assert 0<=s.spectral_coherence(v,r+0.2*np.sin(2*np.pi*t/7),np.arange(1,9))<=1
def test_coherence_fails_closed():assert np.isnan(s.spectral_coherence(np.ones(95),np.ones(95),np.arange(1,9)))
def test_rank_excludes_current():
 r=s.prior_rank(pd.Series(range(181),dtype=float));assert r.iloc[:180].isna().all();assert r.iloc[180]==1.
def test_onset_side_controls_and_hash():
 d=pd.date_range("2024-07-01T03:00:00Z",periods=4,freq="8h");x=pd.DataFrame({"decision_time":d,"feature_available_time":d,"source_valid":True,"spectral_coherence":[.8]*4,"all_frequency_coherence":[.8]*4,"coherence_rank":[.7,.8,.9,.7],"all_frequency_rank":[.8]*4,"realized_variation":[.01]*4,"variation_rank":[.7]*4,"block_return":[.01]*4,"late_return":[.01]*4,"side":[1]*4,"eligible":[False,True,True,False]},columns=s.PANEL_COLS);onset,side=s.active(x);assert onset.tolist()==[False,True,False,False];assert side.tolist()==[1]*4;assert s.active(x,"direction_flip")[1].tolist()==[-1]*4;assert s.PREREG_SHA=="1f90da4731be308d9859c87be56f5158693768a7fdc0808d6cb1d1fe5c52725b"
