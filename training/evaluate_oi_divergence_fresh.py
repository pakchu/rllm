"""One-shot fresh replay of the frozen four-gate OI divergence long."""
from __future__ import annotations
import argparse,json,hashlib
import numpy as np,pandas as pd
from training import evaluate_macro_flow_fixed_fresh as mf
from training import build_oi_enriched_cache as enrich
from training import evaluate_oi_llm_selector as oi_eval
from training import search_inventory_purge_reclaim_alpha as execmod
from training import search_meaningful_alpha_combinations as base
from training import build_pposm_fresh_forward_signal_inventory_v2 as pposm_db
OUT=base.ROOT/'research/oi_divergence_fresh';CONFIG=base.ROOT/'configs/live/oi_divergence_pullback_range_rsi_h96_s6_candidate.json';OLD=base.DATA/'cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz';START='2026-06-01T00:00:00Z';SIGNAL_END='2026-08-03T13:40:00Z';EVAL_END='2026-08-03T21:45:00Z';SEAM=mf.SEAM
DESIGN={'version':1,'candidate_config':str(CONFIG.relative_to(base.ROOT)),'candidate_config_sha256':base.sha(CONFIG),'old_oi_cache_sha256':base.sha(OLD),'source':'authoritative open_interest_binance BTCUSDT5m plus DB market/funding seam','signal_window':[START,SIGNAL_END],'evaluation_window':[START,EVAL_END],'candidate_changes':'none; additionally require current OI availability','costs':[0.,.0006,.001],'execution':'frozen stride6, next5m open, nonoverlap, fixed8h hold, realized funding','decision':'one-shot report, no live promotion'}
def register():
 d={'design':DESIGN,'code_sha256':base.sha(__file__),'feature_module_sha256':base.sha(oi_eval.__file__),'engine_sha256':base.sha(execmod.__file__),'cache_builder_sha256':base.sha(enrich.__file__)};p=OUT/'design.json'
 if p.exists() and json.loads(p.read_text())!=d:raise RuntimeError('OI design drift')
 base.write_json(p,d);return d
def load_context():
 db,funding,receipt=mf.load_extension();cfg=enrich.OiEnrichConfig(input_csv='',output_csv='',env_file='/home/pakchu/rllm/.env',period='5m',tolerance='10min');oi=enrich._load_oi(cfg,pd.Timestamp('2026-05-01'),pd.Timestamp(SIGNAL_END))
 joined=pd.merge_asof(db.sort_values('date'),oi.sort_values('date'),on='date',direction='backward',tolerance=pd.Timedelta('10min'));joined['open_interest_available']=joined.open_interest.notna().astype(float);joined['open_interest']=pd.to_numeric(joined.open_interest,errors='coerce').ffill()
 old=pd.read_csv(OLD);old.date=pd.to_datetime(old.date,utc=True).dt.tz_convert(None);market=pposm_db.merge_cache_db_markets(old,joined,cutoff=SEAM)
 hf=pd.read_csv(base.FUNDING);hf['date']=pd.to_datetime(hf.funding_time,unit='ms',utc=True).dt.tz_convert(None);fund=pd.concat([hf[['date','funding_rate','mark_price']][hf.date<pd.Timestamp(SEAM).tz_localize(None)],funding[funding.date>=pd.Timestamp(SEAM).tz_localize(None)]],ignore_index=True).sort_values('date').drop_duplicates('date',keep='last')
 return market,fund,oi,receipt
def schedule(market,fund,candidate):
 feat=oi_eval._feature_frame(market,window_size=144);active=oi_eval._candidate_active(feat,candidate)&(pd.to_numeric(market.open_interest_available,errors='coerce').fillna(0).to_numpy()>.5);dates=pd.to_datetime(market.date);period=(dates>=pd.Timestamp(START).tz_localize(None))&(dates<pd.Timestamp(SIGNAL_END).tz_localize(None));slots=np.arange(143,len(market)-int(candidate['hold_bars'])-2,int(candidate['stride_bars']),dtype=np.int64);signals=slots[active[slots]&period.to_numpy()[slots]]
 cfg=execmod.Config(input_csv='',metrics_csv='',funding_csv='',output='',manifest_output='',leverage=1.,fee_rate=.0006,slippage_rate=0);engine=execmod.ExecutionEngine(market,fund,cfg);trades=[];nxt=0
 for signal in signals:
  if signal<nxt:continue
  trade=engine.trade_at(int(signal),1,int(candidate['hold_bars']),1_000_000,1_000_000)
  if trade is None or dates.iloc[trade.exit_position]>=pd.Timestamp(EVAL_END).tz_localize(None):continue
  trades.append(trade);nxt=trade.exit_position+1
 return trades,feat,cfg,signals
def run():
 reg=json.loads((OUT/'design.json').read_text());
 if reg!=register():raise RuntimeError('Registration changed')
 payload=json.loads(CONFIG.read_text());candidate=payload['signal'];market,fund,source,receipt=load_context();trades,feat,cfg,rawsignals=schedule(market,fund,candidate);reports={str(cost):execmod.equity_stats(trades,start=START,end=EVAL_END,cfg=cfg,cost_rate=cost) for cost in DESIGN['costs']}
 result={'registration':reg,'source_receipt':receipt,'oi':{'rows':len(source),'first':str(source.date.min()),'last':str(source.date.max()),'sha256':hashlib.sha256(source.to_csv(index=False).encode()).hexdigest()},'signal_candidates':len(rawsignals),'scheduled_nonoverlap_trades':len(trades),'schedule_hash':execmod._schedule_hash(trades),'reports':reports,'live_enabled':False};base.write_json(OUT/'report.json',result);print(json.dumps(reports,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();register() if a.freeze else run()
