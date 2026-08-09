import numpy as np
import pandas as pd
from training import build_high_volatility_median_breadth_contradiction_relay_support as support

def test_prior_rank_excludes_current():
 r=support.prior_midrank(pd.Series(np.arange(253,dtype=float)));assert r.iloc[:252].isna().all();assert r.iloc[252]==1.0

def test_breadth_features_detect_jump_against_median():
 opens=np.full(96,100.0);closes=np.full(96,99.99);closes[-1]=101.0
 median,mean,block,variation,contradiction=support.breadth_features(opens,closes)
 assert median<0 and block>0 and contradiction and variation>0

def features(**changes):
 row={"decision_time":pd.Timestamp("2023-07-01T08:00Z"),"feature_available_time":pd.Timestamp("2023-07-01T08:00Z"),"source_valid":True,"median_bar_return":-0.0001,"mean_bar_return":0.0001,"breadth_direction":-1.0,"contradiction":True,"realized_variation":0.02,"block_return":0.01,"variation_rank":0.8};row.update(changes);return pd.DataFrame([row])

def test_primary_uses_median_direction_and_controls_remain_diagnostic():
 f=features();assert support.clock(f).side.tolist()==[-1];assert support.clock(f).entry_time.iloc[0]==pd.Timestamp("2023-07-01T08:05Z");assert support.clock(f,"block_return_side").side.tolist()==[1];assert support.clock(f,"mean_bar_return_side").side.tolist()==[1];assert support.clock(f,"direction_flip").side.tolist()==[1]

def test_gates_are_removed_only_by_named_controls():
 assert support.clock(features(contradiction=False)).empty;assert support.clock(features(contradiction=False),"no_contradiction_gate").side.tolist()==[-1];assert support.clock(features(variation_rank=.2)).empty;assert support.clock(features(variation_rank=.2),"no_variation_gate").side.tolist()==[-1]
