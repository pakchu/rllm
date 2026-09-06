import numpy as np,pandas as pd
from training import build_high_volatility_premium_price_phase_loop_relay_support as s
def test_premium_first_same_direction_is_clockwise_negative():
 dp=np.r_[np.full(239,.001),np.zeros(240)];r=np.r_[np.zeros(239),np.full(240,.0001)];premium=np.r_[0.,np.cumsum(dp)];btc=100*np.exp(np.r_[0.,np.cumsum(r)]);area,pd_,bd,rv=s.phase_loop(btc,premium);assert area<0 and pd_>0 and bd>0 and rv>0
def test_price_first_is_counterclockwise_positive():
 r=np.r_[np.full(239,.0001),np.zeros(240)];dp=np.r_[np.zeros(239),np.full(240,.001)];premium=np.r_[0.,np.cumsum(dp)];btc=100*np.exp(np.r_[0.,np.cumsum(r)]);assert s.phase_loop(btc,premium)[0]>0
def test_rank_and_onset_contract():
 rank=s.prior_rank(pd.Series(range(181),dtype=float));assert rank.iloc[:180].isna().all() and rank.iloc[180]==1.;d=pd.date_range("2024-07-01T03:30:00Z",periods=4,freq="8h");x=pd.DataFrame({"decision_time":d,"feature_available_time":d,"source_valid":True,"phase_loop_area":[-.2]*4,"area_magnitude_rank":[.7,.8,.9,.7],"premium_displacement":[.01]*4,"btc_displacement":[.01]*4,"btc_realized_variation":[.01]*4,"variation_rank":[.7]*4,"side":[1]*4,"eligible":[False,True,True,False]},columns=s.PANEL_COLS);onset,side,_=s.active(x);assert onset.tolist()==[False,True,False,False] and side.tolist()==[1]*4;assert s.PREREG_SHA=="9af958f3414f44cc540f6d520cb0b3de0ebae268fabb5d354b67f66d3b0619e8"
