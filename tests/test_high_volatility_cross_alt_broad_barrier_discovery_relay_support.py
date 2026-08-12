import pandas as pd
from training import build_high_volatility_cross_alt_broad_barrier_discovery_relay_support as s
def test_four_clean_up_discoveries_define_long_side():
 up=pd.DataFrame([[1,1,1,1,0,0]],dtype=bool);down=pd.DataFrame([[0,0,0,0,0,0]],dtype=bool);side,pos,neg=s.discovery_side(up,down,4);assert side.iloc[0]==1 and pos.iloc[0]==4 and neg.iloc[0]==0
def test_opposite_discovery_invalidates_direction():
 up=pd.DataFrame([[1,1,1,1,0,0]],dtype=bool);down=pd.DataFrame([[0,0,0,0,1,0]],dtype=bool);assert s.discovery_side(up,down,4)[0].iloc[0]==0
def test_pinned_registration():assert s.PREREG_SHA=="1535319ea5dd66239e814649f82901b0d87b84cb923f8a189e4550c02bd3902c"
