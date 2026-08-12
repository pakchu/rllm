import numpy as np
import pandas as pd
from training import build_high_volatility_funding_streak_exhaustion_reversal_support as subject

def test_rank_excludes_current():
 values=pd.Series(np.arange(181,dtype=float));assert subject.rank(values).iloc[180]==1.

def test_primary_onset_and_side():
 x=pd.DataFrame({'decision_time':pd.date_range('2024-01-01',periods=4,freq='8h',tz='UTC'),'source_valid':[True]*4,'streak_length':[3,4,5,6],'streak_sign':[1,1,-1,-1],'funding_rate':[.1,.1,-.1,-.1],'magnitude_rank':[.7,.8,.9,.7],'variation_rank':[.7]*4})
 active,side=subject.conditions(x,'primary');assert active.tolist()==[False,True,False,False];assert side[active].tolist()==[-1]

def test_pinned_preregistration():
 assert subject.PREREG_SHA=='f412a4a592c59721807f50309022436f146c5be627dc8f65160acd3f62ba830c'
