import pandas as pd
from training import build_high_volatility_cross_alt_execution_swarm_exhaustion_reversal_support as s
def test_common_execution_shock_direction_is_detected_before_reversal():
 returns=pd.DataFrame([[1,1,1,1,-1,-1]]);shocks=pd.DataFrame([[1,1,1,1,0,0]],dtype=bool);side,pos,neg=s.shock_side(returns,shocks,4);assert side.iloc[0]==1 and pos.iloc[0]==4 and neg.iloc[0]==0
def test_direction_requires_strict_majority():
 returns=pd.DataFrame([[1,1,1,-1,-1,-1]]);shocks=pd.DataFrame(True,index=[0],columns=range(6));assert s.shock_side(returns,shocks,3)[0].iloc[0]==0
def test_pinned_registration():assert s.PREREG_SHA=="ad2eb74800fd3d8b03b0fc391787dc7efcac5c9c9d60d794cc14deb1636e79f4"
