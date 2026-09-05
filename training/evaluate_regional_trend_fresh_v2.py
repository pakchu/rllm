"""Receipt-only correction of the fixed regional-trend fresh replay."""
from __future__ import annotations
import argparse,json
import numpy as np,pandas as pd
from training import evaluate_regional_trend_fresh as v1
from training import evaluate_macro_flow_fixed_fresh as mf
from training import search_meaningful_alpha_combinations as base
from training import search_macro_flow_alpha_combinations as macro
from training import build_pposm_fresh_forward_signal_inventory_v2 as pposm_db
OUT=base.ROOT/'research/regional_trend_fresh_v2';DESIGN={**v1.DESIGN,'version':2,'correction':'availability receipt kimchi_premium_valid; strategy/window/costs unchanged'}
def register():
 p=OUT/'design.json';failure=base.ROOT/'research/regional_trend_fresh/runtime_failure.json';d={'design':DESIGN,'code_sha256':base.sha(__file__),'v1_sha256':base.sha(v1.__file__),'failure_sha256':base.sha(failure),'family_report_sha256':base.sha(base.ROOT/'research/family_champion_alphas/report.json'),'cache_sha256':base.sha(base.MARKET)}
 if p.exists() and json.loads(p.read_text())!=d:raise RuntimeError('V2 design drift')
 base.write_json(p,d);return d
def run():
 r=json.loads((OUT/'design.json').read_text());
 if r!=register():raise RuntimeError('Registration changed')
 db,funding,receipt=mf.load_extension();old=pd.read_csv(base.MARKET);old.date=pd.to_datetime(old.date,utc=True).dt.tz_convert(None);market=pposm_db.merge_cache_db_markets(old,db,cutoff=v1.SEAM)
 hf=pd.read_csv(base.FUNDING);hf['date']=pd.to_datetime(hf.funding_time,unit='ms',utc=True).dt.tz_convert(None);fund=pd.concat([hf[['date','funding_rate','mark_price']][hf.date<pd.Timestamp(v1.SEAM).tz_localize(None)],funding[funding.date>=pd.Timestamp(v1.SEAM).tz_localize(None)]],ignore_index=True).sort_values('date').drop_duplicates('date',keep='last')
 x=base.features(market,fund);x,data,engine=base.execution_blocks(market,fund,x);raw=pd.DataFrame({'date':market.date,'dxy':market.dxy,'usdkrw':market.usdkrw,'kimchi_premium':market.kimchi_premium,'dxy_available':market.dxy_available,'usdkrw_available':market.usdkrw_available,'kimchi_available':market.kimchi_available});x=pd.concat([x,macro.macro_features(raw,x.index)],axis=1)
 p,rawp=v1.position(x);mask=base.window_mask(data,v1.START,v1.END);reports={str(cost):base.stats_row(base.simulate(base.subset(data,mask),p[mask],cost=cost,fine=True)) for cost in v1.DESIGN['costs']}
 result={'registration':r,'source_receipt':receipt,'engine_receipt':engine,'kimchi_available':float(x.loc[x.index>=pd.Timestamp(v1.START).tz_localize(None),'kimchi_premium_valid'].mean()),'nonzero_decision_hours':int(np.count_nonzero(rawp[mask])),'position_hash':__import__('hashlib').sha256(p[mask].tobytes()).hexdigest(),'reports':reports,'live_enabled':False};base.write_json(OUT/'report.json',result);print(json.dumps(reports,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();register() if a.freeze else run()
