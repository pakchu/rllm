import pandas as pd
from training import build_high_volatility_premium_catchup_relay_support as support

def frame(**changes):
 row={"signal_valid":True,"premium_displacement":.001,"btc_return":-.01,"disagreement":True,"full_variation":.02,"variation_rank":.8};row.update(changes);return pd.DataFrame([row])

def test_primary_follows_premium_and_diagnostics_are_frozen():
 active,side=support.conditions(frame(),"primary");assert active.tolist()==[True] and side.tolist()==[1]
 assert support.conditions(frame(),"direction_flip")[1].tolist()==[-1]
 assert support.conditions(frame(),"same_clock_forced_long")[1].tolist()==[1]

def test_relation_and_volatility_controls_remove_only_named_gate():
 assert not support.conditions(frame(disagreement=False),"primary")[0].iloc[0]
 assert support.conditions(frame(disagreement=False),"contemporaneous_agreement")[0].iloc[0]
 assert not support.conditions(frame(variation_rank=.2),"primary")[0].iloc[0]
 assert support.conditions(frame(variation_rank=.2),"no_volatility_gate")[0].iloc[0]
