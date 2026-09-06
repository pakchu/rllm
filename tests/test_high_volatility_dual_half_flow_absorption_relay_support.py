import pandas as pd
from training import build_high_volatility_dual_half_flow_absorption_relay_support as support

def frame(**changes):
 row={"signal_valid":True,"block_return":.01,"first_half_imbalance":-.02,"second_half_imbalance":-.03,"flow_direction":-1.,"flow_persistent":True,"price_flow_contradiction":True,"full_variation":.02,"variation_rank":.8};row.update(changes);return pd.DataFrame([row])

def test_primary_follows_absorbing_price_and_controls_are_diagnostic():
 active,side=support.conditions(frame(),"primary");assert active.tolist()==[True] and side.tolist()==[1]
 assert support.conditions(frame(),"direction_flip")[1].tolist()==[-1]
 assert support.conditions(frame(),"same_clock_forced_long")[1].tolist()==[1]

def test_named_controls_remove_only_named_gate():
 assert not support.conditions(frame(flow_persistent=False),"primary")[0].iloc[0]
 assert support.conditions(frame(flow_persistent=False),"no_dual_half_persistence")[0].iloc[0]
 assert not support.conditions(frame(price_flow_contradiction=False),"primary")[0].iloc[0]
 assert support.conditions(frame(price_flow_contradiction=False),"no_price_flow_contradiction")[0].iloc[0]
 assert not support.conditions(frame(variation_rank=.2),"primary")[0].iloc[0]
 assert support.conditions(frame(variation_rank=.2),"no_volatility_gate")[0].iloc[0]
