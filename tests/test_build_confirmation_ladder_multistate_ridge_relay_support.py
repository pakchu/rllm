import pandas as pd
from training import build_confirmation_ladder_multistate_ridge_relay_support as support
def test_primary_gate_uses_frozen_variation_and_strength(monkeypatch):
 monkeypatch.setattr(support,"MODEL",support.MODEL);x=pd.DataFrame({"source_valid":[True,True],"prediction":[0.,1.],"variation":[0.,1.],"feature_available_time":pd.to_datetime(["2025-01-01T00:00Z","2025-01-02T00:00Z"],utc=True)});active,side,_=support.active_and_side(x);assert active.tolist()==[False,True];assert side.tolist()==[0,1]
