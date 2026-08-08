import pandas as pd
from training import build_options_oi_chase_exhaustion_support as s

def rows():
 t=pd.date_range('2023-08-01T00:00:00Z',periods=3,freq='1h');return pd.DataFrame({'decision_time':t,'base_valid':[True]*3,'bvol_body':[.01]*3,'dvol_body':[.02]*3,'oi_change':[0,.02,0],'oi_tail':[.01]*3,'hour_return':[.001,.03,.001],'return_tail':[.02]*3,'funding_rate':[.001,.001,.001]})
def test_clock_fades_funding_confirmed_large_chase_onset():
 c=s.clock(rows());assert len(c)==1;assert c.iloc[0].entry_time==pd.Timestamp('2023-08-01T01:05:00Z');assert c.iloc[0].side==-1
def test_funding_sign_disagreement_blocks_primary():
 d=rows();d.loc[1,'funding_rate']=-.001;assert s.clock(d).empty
def test_support_cannot_open_postentry_outcomes():
 assert s.ECONOMIC_OUTCOMES_AUTHORIZED is False

def test_empty_clock_preserves_artifact_schema():
 d=rows();d['base_valid']=False;assert list(s.clock(d).columns)==list(s.CLOCK_COLUMNS)
