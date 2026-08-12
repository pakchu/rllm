"""Fit pre-OOS HVCAFRR-8 once and materialize outcome-blind OOS clocks."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_cross_alt_frozen_response_ridge_relay as prereg
ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-01-01T00:00:00Z');END=pd.Timestamp('2026-08-01T00:00:00Z');FIT_START=pd.Timestamp('2023-01-09T06:00:00Z');OOS_START=pd.Timestamp('2023-07-01T00:00:00Z');PREREG_SHA='26303218ef0fc65f0b56610d94a2c967550ef3b0ead11c7f4e3d442e99dc88ff';REG=prereg.build();P=REG['policy'];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG['stages'].items()};GATES=REG['source_support_gates'];CONTROLS=tuple(REG['diagnostic_controls']['names']);SYMBOLS=('BTCUSDT',*prereg.ALTS)
QUERY="""WITH five AS (SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_start,symbol,(array_agg(open ORDER BY ts))[1] AS first_open,(array_agg(close ORDER BY ts DESC))[1] AS last_close,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS minute_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1,2) SELECT date_bin('8 hours',bar_start,TIMESTAMPTZ '1970-01-01 06:00:00+00') AS block_start,symbol,sum(ln(last_close/first_open)) AS displacement,sum(abs(ln(last_close/first_open))) AS path_variation,sum(minute_squared_return) AS minute_squared_return,sum(minute_rows) AS source_rows,count(*) AS five_minute_bars,min(bar_start) AS first_bar,max(bar_start) AS last_bar,bool_and(minute_rows=5 AND distinct_rows=5 AND first_ts=bar_start AND last_ts=bar_start+INTERVAL '4 minutes' AND coherent) AS coherent FROM five GROUP BY 1,2 ORDER BY 1,2""";OPEN_QUERY="""SELECT ts,open FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end AND extract(minute FROM ts)=5 AND extract(hour FROM ts)::int=ANY(:hours) ORDER BY ts"""
ROOT=Path('data/high_volatility_cross_alt_frozen_response_ridge_relay_sources_2023_2026');PANEL=ROOT/'decision_states.csv.gz';MODEL=ROOT/'frozen_model.json';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/high_volatility_cross_alt_frozen_response_ridge_relay_clocks_2023_2026.csv.gz');SPLIT_DIR=Path('data/high_volatility_cross_alt_frozen_response_ridge_relay_split_clocks_2023_2026');CONTROL_DIR=Path('data/high_volatility_cross_alt_frozen_response_ridge_relay_controls_2023_2026');RESULT=Path('results/high_volatility_cross_alt_frozen_response_ridge_relay_support_2026-08-13.json');BUILDER=Path(__file__).relative_to(Path.cwd())
FEATURES=('ETH_return','SOL_return','BNB_return','XRP_return','DOGE_return','ADA_return','ETH_signed_efficiency','SOL_signed_efficiency','BNB_signed_efficiency','XRP_signed_efficiency','DOGE_signed_efficiency','ADA_signed_efficiency','BTC_normalized_displacement');PANEL_COLUMNS=('decision_time','feature_available_time','source_valid','prediction','prediction_rank','returns_only_prediction','returns_only_rank','btc_realized_variation','variation_rank','eligible');CLOCK_COLUMNS=('candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','prediction','prediction_rank','btc_realized_variation','variation_rank')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def causal(series):
 vals=pd.to_numeric(series,errors='coerce').to_numpy(float);out=np.full(len(vals),np.nan);hist=[]
 for i,v in enumerate(vals):
  prior=np.asarray(hist[-P['history_decisions']:],float)
  if math.isfinite(v) and len(prior)>=P['minimum_history_decisions']:out[i]=float((np.sum(prior<v)+.5*np.sum(prior==v))/len(prior))
  if math.isfinite(v):hist.append(float(v))
 return pd.Series(out,index=series.index)
def fit_ridge(frame,features,labels):
 x=frame.loc[:,features].to_numpy(float);median=np.median(x,axis=0);q75=np.quantile(x,.75,axis=0);q25=np.quantile(x,.25,axis=0);scale=np.maximum(q75-q25,1e-12);z=(x-median)/scale;design=np.column_stack([np.ones(len(z)),z]);penalty=np.eye(design.shape[1]);penalty[0,0]=0.;beta=np.linalg.solve(design.T@design+P['ridge_lambda']*penalty,design.T@np.asarray(labels,float));return {'features':list(features),'median':median.tolist(),'scale':scale.tolist(),'beta':beta.tolist()}
def predict(frame,model):
 x=frame.loc[:,model['features']].to_numpy(float);z=(x-np.asarray(model['median']))/np.asarray(model['scale']);return np.column_stack([np.ones(len(z)),z])@np.asarray(model['beta'])
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def load_source():
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:
   paths=pd.read_sql_query(text(QUERY),c,params={'symbols':list(SYMBOLS),'start':START,'end':END});opens=pd.read_sql_query(text(OPEN_QUERY),c,params={'start':FIT_START+pd.Timedelta('5m'),'end':OOS_START,'hours':[6,14,22]})
 finally:db.dispose()
 return paths,opens
def prepare_paths(raw):
 required=['block_start','symbol','displacement','path_variation','minute_squared_return','source_rows','five_minute_bars','first_bar','last_bar','coherent']
 if raw.columns.tolist()!=required:raise RuntimeError('HVCAFRR path schema drift')
 x=raw.copy()
 for c in ('block_start','first_bar','last_bar'):x[c]=pd.to_datetime(x[c],utc=True,errors='coerce')
 for c in ('displacement','path_variation','minute_squared_return','source_rows','five_minute_bars'):x[c]=pd.to_numeric(x[c],errors='coerce')
 if x[['block_start','symbol']].isna().any().any() or x.duplicated(['block_start','symbol']).any():raise RuntimeError('HVCAFRR invalid path key')
 x['row_valid']=np.isfinite(x[['displacement','path_variation','minute_squared_return','source_rows','five_minute_bars']]).all(axis=1)&x.path_variation.gt(0)&x.minute_squared_return.gt(0)&x.source_rows.eq(480)&x.five_minute_bars.eq(96)&x.first_bar.eq(x.block_start)&x.last_bar.eq(x.block_start+pd.Timedelta('475m'))&x.coherent.eq(True);x['efficiency']=x.displacement.abs()/x.path_variation;x['decision_time']=x.block_start+pd.Timedelta('8h');return x.set_index(['decision_time','symbol']).sort_index()
def prepare_opens(raw):
 if raw.columns.tolist()!=['ts','open']:raise RuntimeError('HVCAFRR calibration open schema drift')
 x=raw.copy();x.ts=pd.to_datetime(x.ts,utc=True,errors='coerce');x.open=pd.to_numeric(x.open,errors='coerce')
 if x.ts.isna().any() or x.ts.duplicated().any() or not np.isfinite(x.open).all() or not x.open.gt(0).all():raise RuntimeError('HVCAFRR invalid calibration open')
 return x.set_index('ts').open.sort_index()
def build_panel(raw):
 x=prepare_paths(raw[0]);opens=prepare_opens(raw[1]);decisions=pd.date_range(START+pd.Timedelta('6h'),END,freq='8h',inclusive='left');full=x.reindex(pd.MultiIndex.from_product([decisions,SYMBOLS],names=['decision_time','symbol']));valid=full.row_valid.unstack('symbol').reindex(columns=SYMBOLS);disp=full.displacement.unstack('symbol').reindex(columns=SYMBOLS);eff=full.efficiency.unstack('symbol').reindex(columns=SYMBOLS);sq=full.minute_squared_return.unstack('symbol').reindex(columns=SYMBOLS);frame=pd.DataFrame(index=decisions)
 for alt,prefix in zip(prereg.ALTS,('ETH','SOL','BNB','XRP','DOGE','ADA')):frame[f'{prefix}_return']=disp[alt];frame[f'{prefix}_signed_efficiency']=np.sign(disp[alt])*eff[alt]
 frame['BTC_normalized_displacement']=disp.BTCUSDT/np.sqrt(sq.BTCUSDT);frame['btc_realized_variation']=np.sqrt(sq.BTCUSDT.where(valid.BTCUSDT).rolling(3,min_periods=3).sum());frame['source_valid']=valid.eq(True).all(axis=1)&np.isfinite(frame.loc[:,FEATURES]).all(axis=1)&np.isfinite(frame.btc_realized_variation)&frame.btc_realized_variation.gt(0)
 fit_mask=frame.index.to_series().between(FIT_START,OOS_START,inclusive='left')&frame.source_valid;fit=frame.loc[fit_mask].copy();entry_times=fit.index+pd.Timedelta(minutes=P['entry_delay_minutes']);exit_times=fit.index+pd.Timedelta(hours=P['hold_hours'],minutes=P['entry_delay_minutes']);entries=opens.reindex(entry_times).to_numpy(float);exits=opens.reindex(exit_times).to_numpy(float);admitted=np.isfinite(entries)&np.isfinite(exits)&(exit_times<OOS_START);fit=fit.iloc[admitted];labels=np.log(exits[admitted]/entries[admitted])
 if len(fit)<P['minimum_calibration_rows']:raise RuntimeError(f'HVCAFRR calibration floor {len(fit)}')
 model=fit_ridge(fit,FEATURES,labels);returns_model=fit_ridge(fit,FEATURES[:6],labels);valid_rows=frame.source_valid.eq(True);frame['prediction']=np.nan;frame['returns_only_prediction']=np.nan;frame.loc[valid_rows,'prediction']=predict(frame.loc[valid_rows],model);frame.loc[valid_rows,'returns_only_prediction']=predict(frame.loc[valid_rows],returns_model);frame['prediction_rank']=causal(frame.prediction.abs().where(valid_rows));frame['returns_only_rank']=causal(frame.returns_only_prediction.abs().where(valid_rows));frame['variation_rank']=causal(frame.btc_realized_variation.where(valid_rows));frame['eligible']=valid_rows&frame.prediction.ne(0)&frame.prediction_rank.ge(P['prediction_rank_min'])&frame.variation_rank.ge(P['variation_rank_min']);panel=frame.reset_index(names='decision_time');panel['feature_available_time']=panel.decision_time;model_core={'protocol_version':'hvcafrr_8_frozen_model_v1','calibration_window':[FIT_START.isoformat(),OOS_START.isoformat()],'calibration_rows':len(fit),'label_definition':REG['model']['label'],'feature_model':model,'returns_only_diagnostic_model':returns_model,'oos_refit':False,'grid':False,'oos_outcomes_opened':False};model_artifact={**model_core,'manifest_hash':chash(model_core)};return panel.loc[:,PANEL_COLUMNS],model_artifact
def onset(state,valid):
 out=pd.Series(False,index=state.index);prior=None
 for i in state.index:
  if not bool(valid.at[i]):continue
  if bool(state.at[i]) and prior is not None:out.at[i]=not bool(state.at[prior])
  prior=i
 return out
def active(panel,control='primary'):
 if control not in ('primary',*CONTROLS):raise ValueError(control)
 used=panel.copy();prediction=used.prediction;rank=used.prediction_rank
 if control=='alt_returns_only_ridge':prediction=used.returns_only_prediction;rank=used.returns_only_rank
 if control=='one_block_stale_prediction':prediction=prediction.shift(1);rank=rank.shift(1)
 valid=used.source_valid.eq(True)&np.isfinite(prediction)&prediction.ne(0)&np.isfinite(rank);state=valid&rank.ge(P['prediction_rank_min'])&used.variation_rank.ge(P['variation_rank_min'])
 if control=='no_prediction_tail':state=valid&used.variation_rank.ge(P['variation_rank_min'])
 elif control=='no_variation_gate':state=valid&rank.ge(P['prediction_rank_min'])
 selected=onset(state,used.source_valid.eq(True));side=np.sign(prediction).fillna(0).astype(int)
 if control=='direction_flip':side=-side
 elif control=='forced_long':side=side.where(side.eq(0),1)
 return selected&side.ne(0),side,used
def build_clock(panel,control='primary'):
 selected,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[selected]:
  decision=pd.Timestamp(panel.at[i,'decision_time']);entry=decision+pd.Timedelta(minutes=P['entry_delay_minutes']);exit_=entry+pd.Timedelta(hours=P['hold_hours'])
  if entry<OOS_START or (reserved is not None and entry<reserved):continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({'candidate':prereg.POLICY_ID,'control':control,'split':split,'decision_time':decision,'feature_available_time':used.at[i,'feature_available_time'],'entry_time':entry,'exit_time':exit_,'side':int(side.at[i]),'prediction':float(used.at[i,'prediction']),'prediction_rank':float(used.at[i,'prediction_rank']),'btc_realized_variation':float(used.at[i,'btc_realized_variation']),'variation_rank':float(used.at[i,'variation_rank'])})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':l,'shorts':s,'minority_side_share':min(l,s)/len(x),'max_month_share':int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('HVCAFRR prereg drift')
 raw=load_source();panel,model=build_panel(raw);common.immutable(MODEL,common.json_bytes(model));primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};common.immutable(PANEL,common.csv_gz(panel));common.immutable(CLOCK,common.csv_gz(primary))
 for n,x in controls.items():common.immutable(CONTROL_DIR/f'{n}.csv.gz',common.csv_gz(x))
 for n,x in splits.items():common.immutable(SPLIT_DIR/f'{n}.csv.gz',common.csv_gz(x))
 source_core={'protocol_version':'hvcafrr_8_sources_v1','queries':{'paths':QUERY,'calibration_opens':OPEN_QUERY},'query_sha256':{'paths':hashlib.sha256(QUERY.encode()).hexdigest(),'calibration_opens':hashlib.sha256(OPEN_QUERY.encode()).hexdigest()},'table':'bars_binance','symbols':list(SYMBOLS),'window':[START.isoformat(),END.isoformat()],'physical_rows':{'paths':len(raw[0]),'calibration_opens':len(raw[1])},'builder':{'path':str(BUILDER),'sha256':sha(BUILDER)},'model':{'path':str(MODEL),'sha256':sha(MODEL),'manifest_hash':model['manifest_hash'],'calibration_rows':model['calibration_rows']},'panel':{'path':str(PANEL),'sha256':sha(PANEL),'rows':len(panel),'valid_rows':int(panel.source_valid.sum())},'calibration_labels_opened':True,'oos_outcomes_opened':False,'gross9_rows_opened':False,'no_imputation':True};manifest={**source_core,'manifest_hash':chash(source_core)};common.immutable(MANIFEST,common.json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f'{n}_minimum_events',x['events']>=GATES['minimum_events'][n]),(f'{n}_side_balance',x['minority_side_share']>=GATES['minority_side_share_min']),(f'{n}_month_concentration',x['max_month_share']<=GATES['max_month_share']))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={'protocol_version':'hvcafrr_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST),'manifest_hash':manifest['manifest_hash']},'calibration_labels_opened':True,'oos_candidate_incidence_opened':True,'oos_postentry_return_pnl_execution_price_opened':False,'oos_funding_values_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'split_artifacts':{n:{'path':str(SPLIT_DIR/f'{n}.csv.gz'),'sha256':sha(SPLIT_DIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in splits.items()},'controls':{n:{'path':str(CONTROL_DIR/f'{n}.csv.gz'),'sha256':sha(CONTROL_DIR/f'{n}.csv.gz'),'rows':len(x),'promotion_authorized':False} for n,x in controls.items()},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};common.immutable(RESULT,common.json_bytes(r));return r
if __name__=='__main__':print(json.dumps({'passed':(r:=run())['support_passed'],'support':r['support']}))
