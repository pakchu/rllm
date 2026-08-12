import pandas as pd
from training import build_high_volatility_cross_alt_volume_shock_consensus_relay_support as s
def test_four_positive_shocked_returns_define_long():
 returns=pd.DataFrame([[1,1,1,1,-1,-1]]);shocks=pd.DataFrame([[1,1,1,1,0,0]],dtype=bool);side,pos,neg=s.shock_side(returns,shocks,4);assert side.iloc[0]==1 and pos.iloc[0]==4 and neg.iloc[0]==0
def test_direction_requires_strict_majority():
 returns=pd.DataFrame([[1,1,1,-1,-1,-1]]);shocks=pd.DataFrame(True,index=[0],columns=range(6));assert s.shock_side(returns,shocks,3)[0].iloc[0]==0
def test_pinned_registration():assert s.PREREG_SHA=="7996662fa06ad47e5aa188a73ec7e1be0c014ed0b695c31d3d493b78a41bffed"
