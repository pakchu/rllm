"""Build outcome-blind source support for frozen HVPOCR-8."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_premium_occupation_crowding_reversal as prereg
ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="fea5f22e0d7c9e3f52423f1e3458bfd8e9d2a41b7ccd1e4428be7c6cd2e22a13";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"])
BTC_QUERY="""SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts""";PREMIUM_QUERY="""SELECT ts,close FROM bars_binance_premium WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path("data/high_volatility_premium_occupation_crowding_reversal_sources_2023_2026");PANEL=ROOT/"block_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_premium_occupation_crowding_reversal_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_premium_occupation_crowding_reversal_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_premium_occupation_crowding_reversal_controls_2023_2026");RESULT=Path("results/high_volatility_premium_occupation_crowding_reversal_support_2026-08-12.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","btc_minute_count","premium_minute_count","block_mean","causal_baseline","deviation","absolute_deviation","deviation_rank","occupation_share","persistent_side","premium_displacement","realized_variation","variation_rank","eligible","onset");CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side",*PANEL_COLUMNS[5:15])
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def midrank(value:float,history:list[float])->float:
 prior=np.asarray(history[-P["history_decisions"]:],float)
 if not math.isfinite(value) or len(prior)<P["minimum_history_decisions"]:return math.nan
 return float((np.sum(prior<value)+.5*np.sum(prior==value))/len(prior))
def previous_valid_onset(state:pd.Series,valid:pd.Series)->pd.Series:
 out=pd.Series(False,index=state.index);previous=None
 for i in state.index:
  if not bool(valid.at[i]):continue
  if bool(state.at[i]) and previous is not None:out.at[i]=not bool(state.at[previous])
  previous=i
 return out
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_sources()->tuple[pd.DataFrame,pd.DataFrame]:
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:
   btc=pd.read_sql_query(text(BTC_QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()});premium=pd.read_sql_query(text(PREMIUM_QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
  return btc,premium
 finally:db.dispose()
def prepare_btc(x:pd.DataFrame)->pd.DataFrame:
 if x.columns.tolist()!=["ts","open","high","low","close"]:raise RuntimeError("HVPOCR BTC schema drift")
 y=x.copy();y["ts"]=pd.to_datetime(y.ts,utc=True,errors="coerce")
 for c in ("open","high","low","close"):y[c]=pd.to_numeric(y[c],errors="coerce")
 if y.ts.isna().any() or y.ts.duplicated().any():raise RuntimeError("HVPOCR BTC key drift")
 prices=y[["open","high","low","close"]];y["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&y.high.ge(prices[["open","close"]].max(axis=1))&y.low.le(prices[["open","close"]].min(axis=1))&y.high.ge(y.low);return y.set_index("ts").sort_index()
def prepare_premium(x:pd.DataFrame)->pd.DataFrame:
 if x.columns.tolist()!=["ts","close"]:raise RuntimeError("HVPOCR premium schema drift")
 y=x.copy();y["ts"]=pd.to_datetime(y.ts,utc=True,errors="coerce");y["close"]=pd.to_numeric(y.close,errors="coerce")
 if y.ts.isna().any() or y.ts.duplicated().any():raise RuntimeError("HVPOCR premium key drift")
 y["row_valid"]=np.isfinite(y.close);return y.set_index("ts").sort_index()
def build_panel(btc_raw:pd.DataFrame,premium_raw:pd.DataFrame)->pd.DataFrame:
 btc=prepare_btc(btc_raw);premium=prepare_premium(premium_raw);rows=[];means=[];deviations=[];variations=[]
 for decision in pd.date_range(START+pd.Timedelta("29h"),END,freq="8h",inclusive="left"):
  btc_idx=pd.date_range(decision-pd.Timedelta("24h"),decision,freq="1min",inclusive="left");prem_idx=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="1min",inclusive="left");b=btc.reindex(btc_idx);q=premium.reindex(prem_idx);btc_count=int(b.row_valid.eq(True).sum());prem_count=int(q.row_valid.eq(True).sum());valid=len(b)==1440 and len(q)==480 and bool(b.row_valid.eq(True).all()) and bool(q.row_valid.eq(True).all())
  mean=baseline=deviation=absolute=dev_rank=occupation=persistent=displacement=variation=var_rank=math.nan
  if valid:
   values=q.close.to_numpy(float);mean=float(values.mean());variation=float(np.square(np.log(b.close.to_numpy(float)/b.open.to_numpy(float))).sum());baseline=float(np.median(np.asarray(means[-P["history_decisions"]:],float))) if len(means)>=P["minimum_history_decisions"] else math.nan;deviation=mean-baseline if math.isfinite(baseline) else math.nan;absolute=abs(deviation) if math.isfinite(deviation) else math.nan;dev_rank=midrank(absolute,[abs(v) for v in deviations]);occupation=float(np.mean(values>baseline)) if math.isfinite(baseline) else math.nan;persistent=1. if occupation>=P["upper_occupation_min"] and deviation>0 else -1. if occupation<=P["lower_occupation_max"] and deviation<0 else 0.;displacement=float(values[-1]-values[0]);var_rank=midrank(variation,variations);valid=bool(math.isfinite(mean) and variation>0)
  rows.append({"decision_time":decision,"feature_available_time":decision,"source_valid":valid,"btc_minute_count":btc_count,"premium_minute_count":prem_count,"block_mean":mean,"causal_baseline":baseline,"deviation":deviation,"absolute_deviation":absolute,"deviation_rank":dev_rank,"occupation_share":occupation,"persistent_side":persistent,"premium_displacement":displacement,"realized_variation":variation,"variation_rank":var_rank})
  if valid:
   means.append(mean);variations.append(variation)
   if math.isfinite(deviation):deviations.append(deviation)
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["eligible"]=valid&panel.persistent_side.ne(0)&panel.deviation_rank.ge(P["deviation_magnitude_rank_min"])&panel.variation_rank.ge(P["variation_rank_min"]);panel["onset"]=previous_valid_onset(panel.eligible,valid);return panel.loc[:,PANEL_COLUMNS]
def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_block_stale_features":used[list(PANEL_COLUMNS[2:15])]=panel[list(PANEL_COLUMNS[2:15])].shift(1);used["feature_available_time"]=panel.feature_available_time.shift(1)
 valid=used.source_valid.eq(True);tail=used.deviation_rank.ge(P["deviation_magnitude_rank_min"]);variation=used.variation_rank.ge(P["variation_rank_min"]);occupied=used.persistent_side.ne(0);state=valid&tail&variation&occupied;side=-np.sign(pd.to_numeric(used.deviation,errors="coerce").fillna(0)).astype(int)
 if control=="no_occupation_requirement":state=valid&tail&variation&used.deviation.ne(0)
 elif control=="no_variation_gate":state=valid&tail&occupied
 elif control=="premium_displacement_side":state=valid&tail&variation&occupied&used.premium_displacement.ne(0);side=np.sign(pd.to_numeric(used.premium_displacement,errors="coerce").fillna(0)).astype(int)
 onset=previous_valid_onset(state,valid)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=side.where(side.eq(0),1)
 return onset&side.ne(0),side,used
def build_clock(panel:pd.DataFrame,control:str="primary")->pd.DataFrame:
 act,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[act]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta("5m");exit_=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":pd.Timestamp(used.at[i,"feature_available_time"]),"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(used.at[i,c]) for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict[str,float|int]:
 x=c[c.split.eq(s)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());q=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(x),"max_month_share":int(m.max())/len(x)}
def csv_gz(x:pd.DataFrame)->bytes:
 b=io.BytesIO();raw=x.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode()
 with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
 return b.getvalue()
def immutable(p:Path,b:bytes)->None:
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=b:raise RuntimeError(f"refusing overwrite {p}")
 p.write_bytes(b)
def json_bytes(x:Any)->bytes:return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVPOCR prereg drift")
 btc,premium=load_sources();panel=build_panel(btc,premium);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvpocr_8_source_v1","queries":{"btc_sha256":hashlib.sha256(BTC_QUERY.encode()).hexdigest(),"premium_sha256":hashlib.sha256(PREMIUM_QUERY.encode()).hexdigest()},"window":[START.isoformat(),END.isoformat()],"physical_rows":{"btc":len(btc),"premium":len(premium)},"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvpocr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
