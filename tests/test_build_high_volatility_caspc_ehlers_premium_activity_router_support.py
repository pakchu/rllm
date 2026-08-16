from __future__ import annotations
import numpy as np,pandas as pd
from training import build_high_volatility_caspc_ehlers_premium_activity_router_support as s
def test_rank_excludes_current():
 r=s.rank(pd.Series([1.,2.,3.,2.]),3,2);assert np.isnan(r.iloc[:2]).all() and r.iloc[2]==1 and r.iloc[3]==.5
def test_clock_routes_base_side_by_median(tmp_path,monkeypatch):
 d=pd.to_datetime(['2023-07-01T03:00Z','2023-07-01T11:00Z']);base=pd.DataFrame({'candidate':['b','b'],'control':['primary','primary'],'split':['train','train'],'decision_time':d,'feature_available_time':d,'entry_time':d+pd.Timedelta(minutes=5),'exit_time':d+pd.Timedelta(hours=8,minutes=5),'side':[1,1]});path=tmp_path/'b.csv';base.to_csv(path,index=False);monkeypatch.setattr(s,'BASE_CLOCK',path)
 p=pd.DataFrame({'decision_time':d,'feature_available_time':d,'source_valid':[True,True],'premium_total_variation':[1.,2.],'btc_return':[.1,-.1],'btc_realized_variation':[.2,.3],'relative_premium_activity':[.1,.9],'relative_activity_rank':[.2,.8],'btc_variation_rank':[.8,.8],'eligible':[True,True],'side':[0,0]});assert s.build_clock(p).side.tolist()==[1,-1]
def test_prereg_hash_bound():assert s.sha(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA;s.prereg.validate(s.REG)
