import hashlib,json,math
import numpy as np
import pandas as pd
from training import build_high_volatility_directional_change_fresh_overshoot_relay_support as s


def test_directional_change_confirms_new_overshoot_direction():
 closes=np.exp(np.array([0.,.11,.20,.09,0.,.11]+[.11]*90));count,side,index=s.directional_change_state(closes,.1);assert count==2 and side==1 and index==5
 monotone=np.exp(np.linspace(0,.3,96));assert s.directional_change_state(monotone,.1)==(0,0,-1)


def test_directional_change_invalid_inputs():
 assert s.directional_change_state(np.ones(95),.1)==(0,0,-1)
 assert s.directional_change_state(np.ones(96),0)==(0,0,-1)


def test_prior_rank_excludes_current(monkeypatch):
 monkeypatch.setitem(s.P,'minimum_variation_history_decisions',2);monkeypatch.setitem(s.P,'variation_history_decisions',3)
 r=s.prior_rank(pd.Series([1.,2.,3.,4.]));assert np.isnan(r.iloc[1]) and r.iloc[2]==1 and r.iloc[3]==1


def panel():
 return pd.DataFrame({'source_valid':[True]*5,'threshold':[.1]*5,'variation_rank':[.8,.8,.4,.8,.8],'current_variation':[.2]*5,'confirmation_count':[1,1,1,1,0],'latest_confirmation_index':[90,70,90,90,-1],'latest_side':[-1,1,-1,1,0],'fresh_confirmation':[True,False,True,True,False],'current_block_return':[1.,-1.,1.,-1.,1.],'eligible':[True,False,False,True,False],'feature_available_time':pd.date_range('2024-01-01',periods=5,freq='8h',tz='UTC')})


def test_primary_and_frozen_controls():
 p=panel();a,z,_=s.active(p);assert a.tolist()==[True,False,False,True,False] and z[a].tolist()==[-1,1]
 a,z,_=s.active(p,'no_variation_gate');assert a.tolist()==[True,False,True,True,False]
 a,z,_=s.active(p,'any_confirmation_in_block');assert a.tolist()==[True,True,False,True,False]
 a,z,_=s.active(p,'current_block_return_side');assert a.tolist()==[True,False,False,True,False] and z[a].tolist()==[1,-1]
 a,z,_=s.active(p,'one_block_stale_confirmation');assert a.tolist()==[False,True,False,False,True] and z[a].tolist()==[-1,1]
 a,z,_=s.active(p,'direction_flip');assert z[a].tolist()==[1,-1]
 a,z,_=s.active(p,'forced_long');assert z[a].eq(1).all()


def test_source_blind_and_hash_bound():
 q=s.QUERY.lower();assert 'open,high,low,close' in q and 'funding' not in q and 'gross9' not in q
 assert s.PREREG_SHA==hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
 value={'한글':'DC'};expected=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert s.chash(value)==expected
