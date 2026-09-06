import numpy as np
import pandas as pd
from training import evaluate_macro_flow_fixed_fresh as s


def test_candidate_weights_are_net_and_fixed():
 n=48;x=pd.DataFrame(index=pd.date_range('2026-06-01',periods=n,freq='1h'))
 for col,value in [('vol24',.01),('flow6',.03),('dxy_change6',-.1),('mom720',1.),('z24',0.),('kimchi_premium_change24',.1)]:x[col]=value
 p,c=s.fixed_positions(x)
 assert set(p)==set(s.CANDIDATES)
 assert np.max(np.abs(np.column_stack(list(p.values()))))<=1
 assert np.allclose(p['dollar_flow_plus_regime_switch'],.75*c['dollar']+.25*c['flow_switch720_long'])


def test_design_is_one_shot_research_only():
 assert s.DESIGN['candidate_origin'].endswith('before this fresh query')
 assert s.DESIGN['decision'].startswith('report all')
 assert s.DESIGN['execution'].startswith('completed hour')
