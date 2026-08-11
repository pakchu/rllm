import hashlib,json,math
import numpy as np
import pandas as pd
from training import build_high_volatility_relative_signed_variation_risk_premium_support as s

def test_signed_variation_decomposition():
 r=np.resize(np.array([.01,-.02]),288);positive,negative,total,relative=s.signed_variation(r)
 assert positive>0 and negative>positive and total==positive+negative and relative<0
 assert math.isnan(s.signed_variation(np.ones(287))[0]) and math.isnan(s.signed_variation(np.zeros(288))[0])

def test_prior_rank_excludes_current_and_skips_invalid(monkeypatch):
 monkeypatch.setitem(s.P,'minimum_history_days',2);monkeypatch.setitem(s.P,'history_days',3)
 r=s.prior_rank(pd.Series([1.,2.,np.nan,3.,4.]),pd.Series([True,True,False,True,True]));assert np.isnan(r.iloc[1]) and r.iloc[3]==1 and r.iloc[4]==1

def test_frozen_controls():
 p=pd.DataFrame({'source_valid':[True]*5,'variation_tail':[True,True,False,True,True],'signed_tail':[False,True,True,False,True],'relative_signed_variation':[.1,.2,-.3,-.2,.4],'day_return':[.1,.2,-.3,-.2,.4],'eligible':[False,True,False,False,True],'entry_side':[0,-1,1,0,-1]})
 a,z,_=s.active(p);assert a.tolist()==[False,True,False,False,True] and z[a].tolist()==[-1,-1]
 a,z,_=s.active(p,'no_signed_variation_tail');assert a.tolist()==[True,True,False,True,True] and z[a].tolist()==[-1,-1,1,-1]
 a,_,_=s.active(p,'no_variation_gate');assert a.tolist()==[False,True,True,False,True]
 a,z,_=s.active(p,'raw_day_return_reversal');assert z[a].tolist()==[-1,-1]
 a,z,_=s.active(p,'one_day_stale_state');assert a.tolist()==[False,False,True,False,False] and z[a].tolist()==[-1]
 a,z,_=s.active(p,'direction_flip');assert z[a].tolist()==[1,1]
 a,z,_=s.active(p,'forced_long');assert z[a].eq(1).all()

def test_source_blind_and_hash_bound():
 assert 'bars_binance ' in s.PERP_QUERY and 'funding' not in s.PERP_QUERY.lower() and 'gross9' not in s.PERP_QUERY.lower()
 assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest();v={'한글':'signed'};expected=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert s.canonical_hash(v)==expected
