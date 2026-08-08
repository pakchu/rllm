import pandas as pd
from training import build_cboe_broad_slope_crypto_volatility_transmission_relay_support as support
def frame():return pd.DataFrame({"valid":[True,True],"delta_broad_slope":[.02,-.02],"previous_delta_broad_slope":[-.01,.02],"bvol_body":[.01,-.01],"dvol_body":[.02,-.02]})
def test_cbstr_requires_same_direction_crypto_vol_confirmation():assert support.sides(frame(),"primary").tolist()==[-1,1]
def test_cbstr_rejects_unconfirmed_and_flip_is_identical():
 f=frame();f.loc[0,"dvol_body"]=-.02;assert support.sides(f,"primary").iloc[0]==0
 a=support.sides(frame(),"primary");b=support.sides(frame(),"direction_flip");assert (a==-b).all()
