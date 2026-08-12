import numpy as np
import pandas as pd
from training import build_high_volatility_india_opening_reversal_relay_support as subject

def test_rank_excludes_current():
 assert subject.rank(pd.Series(np.arange(61,dtype=float))).iloc[60]==1.

def test_primary_uses_reversal_dominance_and_side():
 x=pd.DataFrame({'session_valid':[True]*4,'btc_valid':[True]*4,'reversal':[True,True,False,True],'opening_dominance':[True,False,True,True],'reversal_rank':[.8,.9,.9,.6],'variation_rank':[.7,.7,.7,.7],'opening_return':[.1,-.1,.1,-.1]})
 active,side=subject.conditions(x,'primary');assert active.tolist()==[True,False,False,False];assert side[active].tolist()==[-1]
 assert subject.conditions(x,'no_opening_dominance')[0].tolist()==[True,True,False,False]

def test_pinned_preregistration():
 assert subject.PREREG_SHA=='69941911f22565483141a87814d5f89202d4008d5a8570a8a9ad091da6eb3f78'
