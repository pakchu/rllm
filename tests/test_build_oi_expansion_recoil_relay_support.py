import numpy as np
import pandas as pd
from training import build_oi_expansion_recoil_relay_support as support

def _frame():
 return pd.DataFrame({"date":pd.date_range("2024-01-01",periods=12,freq="5min",tz="UTC"),"oi_ret_4h_z":1.0,"px_ret_4h_z":[-1.0]*6+[1.0]*6,"range_vol":.05,"rsi_norm":[-.1]*6+[.1]*6})

def test_symmetric_state_signals(monkeypatch):
 monkeypatch.setattr(support,"interval_slots",lambda dates,stride=6: np.ones(len(dates),dtype=bool))
 long_signal,short_signal=support.state_signals(_frame())
 assert long_signal.tolist()==[True]*6+[False]*6
 assert short_signal.tolist()==[False]*6+[True]*6

def test_range_and_rsi_controls(monkeypatch):
 monkeypatch.setattr(support,"interval_slots",lambda dates,stride=6: np.ones(len(dates),dtype=bool))
 frame=_frame(); frame["range_vol"]=.01; frame["rsi_norm"]=0.0
 assert not support.state_signals(frame)[0].any()
 assert support.state_signals(frame,"no_range_vol_gate")[0].any()==False
 frame["rsi_norm"]=[-.1]*6+[.1]*6
 assert support.state_signals(frame,"no_range_vol_gate")[0].any()

def test_outcomes_closed():
 source=open(support.__file__).read(); assert "ECONOMIC_OUTCOMES_AUTHORIZED = False" in source and '"btc_postentry_return_or_pnl_opened": False' in source
