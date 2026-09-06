"""One-shot fresh DB replay for three fixed formulaic macro-flow candidates."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from preprocessing.live_db_features import LiveDbFeatureConfig, build_live_feature_frame_from_frames, query_live_source_frames, sqlalchemy_engine_from_env
from training import build_pposm_fresh_forward_signal_inventory_v2 as pposm_db
from training import search_meaningful_alpha_combinations as base
from training import search_macro_flow_alpha_combinations as macro

OUT=base.ROOT/'research/macro_flow_fresh'
SELECTION=base.ROOT/'research/macro_flow_combinations/selection_freeze.json'
SEAM='2026-05-31T15:05:00Z'; START='2026-06-01T00:00:00Z'; ASOF='2026-09-05T10:04:00Z'; END='2026-09-05T10:05:00Z'
CANDIDATES={
 'dollar_flow_plus_regime_switch':{'dollar':.75,'second':.25,'second_name':'flow_switch720_long'},
 'lowturn_dollar_plus_exhaustion':{'dollar':.25,'second':.75,'second_name':'exhaustion24_both'},
 'lowturn_dollar_plus_short_exhaustion':{'dollar':.25,'second':.75,'second_name':'exhaustion24_short'},
}
DESIGN={'version':1,'source':'authoritative PostgreSQL market/funding/FX/USDKRW/Upbit frames after cache tail','query_start':'2026-05-01T00:00:00Z','asof':ASOF,'fresh_window':[START,END],
 'candidates':CANDIDATES,'candidate_origin':'three formulaic finalists already frozen; primary picked from exposed historical reports before this fresh query','execution':'completed hour T, next 5m open T+5m, net same-symbol positions before costs','costs':[0.,.0006,.001],
 'funding':'canonical exact-value aliases within100ms; backward realized transfer; missing mark proxy inherited','risk':'net cap1x and conservative five-minute MDD','decision':'report all; no fresh-period substitution, no automatic live authorization'}


def register():
 p=OUT/'design.json';d={'design':DESIGN,'code_sha256':base.sha(__file__),'base_sha256':base.sha(base.__file__),'macro_sha256':base.sha(macro.__file__),'selection_sha256':base.sha(SELECTION),'cache_sha256':base.sha(base.MARKET)}
 if p.exists() and json.loads(p.read_text())!=d:raise RuntimeError('Fresh design drift')
 base.write_json(p,d);return d


def fixed_positions(x):
 size=np.clip(np.divide(.2,x.vol24.to_numpy()*np.sqrt(8766),out=np.ones(len(x)),where=x.vol24.to_numpy()>0),.1,1)
 flow=x.flow6.to_numpy();dollar=x.dxy_change6.to_numpy();mom=x.mom720.to_numpy();z=x.z24.to_numpy();premium=x.kimchi_premium_change24.to_numpy()
 dollar_raw=np.where((np.abs(flow)>.02)&(np.sign(flow)*dollar<0),np.sign(flow),0)*size
 dollar_signal=base.hold_signal(dollar_raw,24,x.index)
 trend=np.abs(mom)>.75;direction=np.sign(mom);reverse=np.where(np.abs(z)>1.5,-np.sign(z),0)
 switch=np.maximum(np.where(trend,np.where(direction*flow>0,direction,0),reverse),0)
 switch=base.hold_signal(switch,24,x.index)
 exhaustion=np.where((np.abs(z)>1.5)&(np.sign(z)*premium<0)&(np.sign(z)*flow<0),-np.sign(z),0)*size
 exhaustion_both=base.hold_signal(exhaustion,24,x.index); exhaustion_short=np.minimum(exhaustion_both,0)
 components={'dollar':dollar_signal,'flow_switch720_long':switch,'exhaustion24_both':exhaustion_both,'exhaustion24_short':exhaustion_short}
 return {name:spec['dollar']*components['dollar']+spec['second']*components[spec['second_name']] for name,spec in CANDIDATES.items()},components


def load_extension():
 cfg=LiveDbFeatureConfig(lookback_minutes=190_000,feature_window_size=144,include_spot_source=False)
 engine=sqlalchemy_engine_from_env('/home/pakchu/rllm/.env')
 try:
  frames=query_live_source_frames(engine,asof=ASOF,cfg=cfg,start_by_key={key:pd.Timestamp(DESIGN['query_start']) for key in ['btcusdt_1m','btckrw_1m','usdkrw_1m','forex_1m','premium_1m']})
 finally:engine.dispose()
 canonical,diag=pposm_db.canonicalize_funding_aliases(frames['funding']);
 if not diag['passed']:raise RuntimeError('Funding aliases ambiguous')
 frames['funding']=canonical
 enriched,_=build_live_feature_frame_from_frames(cfg=cfg,**frames)
 return enriched,canonical,{'row_counts':{k:len(v) for k,v in frames.items()},'funding_aliases':diag}


def run():
 registered=json.loads((OUT/'design.json').read_text())
 if registered!=register():raise RuntimeError('Registration changed')
 selection=json.loads(SELECTION.read_text());names={row['name'] for row in selection['top']}
 required={'mix_223_162_0.75','mix_223_289_0.25','mix_223_293_0.25'}
 if not required.issubset(names):raise RuntimeError('Frozen finalist identities changed')
 db,funding,receipt=load_extension();old=pd.read_csv(base.MARKET);old['date']=pd.to_datetime(old.date,utc=True).dt.tz_convert(None)
 merged=pposm_db.merge_cache_db_markets(old,db,cutoff=SEAM)
 dates=pd.to_datetime(merged.date);expected=pd.date_range(dates.min(),pd.Timestamp(END).tz_localize(None),freq='5min',inclusive='left')
 if not dates.reset_index(drop=True).equals(pd.Series(expected)):raise RuntimeError('Merged market not exact continuous grid')
 hist=pd.read_csv(base.FUNDING);hist['date']=pd.to_datetime(hist.funding_time,unit='ms',utc=True).dt.tz_convert(None)
 fund=pd.concat([hist[['date','funding_rate','mark_price']][hist.date<pd.Timestamp(SEAM).tz_localize(None)],funding[funding.date>=pd.Timestamp(SEAM).tz_localize(None)]],ignore_index=True).sort_values('date').drop_duplicates('date',keep='last')
 x=base.features(merged,fund);x,data,engine_receipt=base.execution_blocks(merged,fund,x)
 raw=pd.DataFrame({'date':merged.date,'dxy':merged.dxy,'usdkrw':merged.usdkrw,'kimchi_premium':merged.kimchi_premium,'dxy_available':merged.dxy_available,'usdkrw_available':merged.usdkrw_available,'kimchi_available':merged.kimchi_available})
 x=pd.concat([x,macro.macro_features(raw,x.index)],axis=1)
 positions,components=fixed_positions(x);names=list(positions);p=np.column_stack(list(positions.values()))
 mask=base.window_mask(data,START,END);reports={}
 for cost in DESIGN['costs']:
  st=base.simulate(base.subset(data,mask),p[mask],cost=cost,fine=True);reports[str(cost)]={n:base.stats_row(st,i) for i,n in enumerate(names)}
 result={'registration':registered,'source_receipt':receipt,'engine_receipt':engine_receipt,'merged':{'rows':len(merged),'first':str(dates.min()),'last':str(dates.max()),'sha256':hashlib_frame(merged)},
  'availability':{c:float(pd.to_numeric(merged.loc[(dates>=pd.Timestamp(START).tz_localize(None)),c],errors='coerce').mean()) for c in ['dxy_available','usdkrw_available','kimchi_available']},
  'window':DESIGN['fresh_window'],'candidate_positions_sha256':{n:__import__('hashlib').sha256(v[mask].tobytes()).hexdigest() for n,v in positions.items()},'component_nonzero_hours':{n:int(np.count_nonzero(v[mask])) for n,v in components.items()},'reports':reports,'live_enabled':False,'limitations':['One-shot recent research report, then exposed.','External macro publication-time parity not independently proven.','No capacity/liquidation model.']}
 base.write_json(OUT/'report.json',result);base.write_json(OUT/'research_config.json',{'research_only':True,'live_enabled':False,'candidates':CANDIDATES,'overlap_allowed':True,'long_short_offset':True})
 print(json.dumps(reports['0.0006'],indent=2),flush=True)


def hashlib_frame(frame):
 import hashlib
 h=hashlib.sha256();h.update(frame.to_csv(index=False,lineterminator='\n').encode());return h.hexdigest()


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
 if args.freeze:register()
 else:run()
