"""Build outcome-blind source support for frozen HVCARER-12."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_cross_alt_range_expansion_confirmation_relay as prereg
ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-04-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="b2ac6ac2092a9f23c69add331d6a49fec4ca2641f9adc9d220f51c3f4dcf9c7f";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"])
SYMBOLS=("BTCUSDT",*prereg.ALTS);ALTS=prereg.ALTS;RANGE_COLS=tuple(f"{s}_range" for s in ALTS);RANK_COLS=tuple(f"{s}_range_rank" for s in ALTS)
QUERY="""SELECT ts,symbol,open,high,low,close FROM bars_binance WHERE symbol IN ('BTCUSDT','ADAUSDT','BNBUSDT','DOGEUSDT','ETHUSDT','SOLUSDT','XRPUSDT') AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts,symbol"""
ROOT=Path("data/high_volatility_cross_alt_range_expansion_confirmation_relay_sources_2023_2026");PANEL=ROOT/"daily_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_cross_alt_range_expansion_confirmation_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_cross_alt_range_expansion_confirmation_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_cross_alt_range_expansion_confirmation_relay_controls_2023_2026");RESULT=Path("results/high_volatility_cross_alt_range_expansion_confirmation_relay_support_2026-08-11.json");BUILDER=Path(__file__).relative_to(Path.cwd())
BASE_COLS=("decision_time","feature_available_time","source_valid","minute_count",*RANGE_COLS,*RANK_COLS,"expansion_breadth","median_alt_range_rank","btc_return","btc_realized_variation","btc_variation_rank","eligible")
CLOCK_COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","expansion_breadth","median_alt_range_rank","btc_return","btc_realized_variation","btc_variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def prior_rank(s:pd.Series)->pd.Series:
 v=pd.to_numeric(s,errors="coerce").to_numpy(float);o=np.full(len(v),np.nan);h=[]
 for i,x in enumerate(v):
  q=np.asarray(h[-P["history_days"]:],float)
  if math.isfinite(x) and len(q)>=P["minimum_history_days"]:o[i]=(np.sum(q<x)+.5*np.sum(q==x))/len(q)
  if math.isfinite(x):h.append(float(x))
 return pd.Series(o,index=s.index)
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source()->pd.DataFrame:
 from sqlalchemy import text
 e=postgres_engine()
 try:
  with e.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:e.dispose()
def prepare(f:pd.DataFrame)->pd.DataFrame:
 if f.columns.tolist()!=["ts","symbol","open","high","low","close"]:raise RuntimeError("HVCARER source schema drift")
 x=f.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce");x["symbol"]=x.symbol.astype(str)
 for c in ("open","high","low","close"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.duplicated(["ts","symbol"]).any() or not x.symbol.isin(SYMBOLS).all():raise RuntimeError("HVCARER invalid source key")
 z=x[["open","high","low","close"]];x["row_valid"]=np.isfinite(z).all(axis=1)&z.gt(0).all(axis=1)&x.high.ge(z[["open","close"]].max(axis=1))&x.low.le(z[["open","close"]].min(axis=1))&x.high.ge(x.low);return x.set_index(["ts","symbol"]).sort_index()
def block_stats(b:pd.DataFrame):
 ranges={}
 for s in ALTS:
  x=b.xs(s,level="symbol");ranges[f"{s}_range"]=float(np.log(float(x.high.max())/float(x.low.min())))
 btc=b.xs("BTCUSDT",level="symbol");ret=float(np.log(float(btc.close.iloc[-1])/float(btc.open.iloc[0])));variation=float(np.square(np.log(btc.close.to_numpy(float)/btc.open.to_numpy(float))).sum());return ranges,ret,variation
def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw);rows=[]
 for d in pd.date_range(START+pd.Timedelta("27h"),END,freq="1D",inclusive="left"):
  minutes=pd.date_range(d-pd.Timedelta("24h"),d,freq="1min",inclusive="left");expected=pd.MultiIndex.from_product([minutes,SYMBOLS],names=["ts","symbol"]);b=source.reindex(expected);count=int(b.row_valid.eq(True).sum());valid=len(b)==1440*len(SYMBOLS) and bool(b.row_valid.eq(True).all())
  if valid:
   ranges,ret,var=block_stats(b);valid=all(math.isfinite(v) and v>0 for v in ranges.values()) and math.isfinite(ret) and ret!=0 and math.isfinite(var) and var>0
  else:ranges={c:math.nan for c in RANGE_COLS};ret=var=math.nan
  rows.append({"decision_time":d,"feature_available_time":d,"source_valid":valid,"minute_count":count,**ranges,"btc_return":ret,"btc_realized_variation":var})
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True)
 for c,r in zip(RANGE_COLS,RANK_COLS):panel[r]=prior_rank(panel[c].where(valid))
 panel["expansion_breadth"]=panel[list(RANK_COLS)].ge(P["alt_range_rank_min"]).sum(axis=1);panel["median_alt_range_rank"]=panel[list(RANK_COLS)].median(axis=1,skipna=False);panel["btc_variation_rank"]=prior_rank(panel.btc_realized_variation.where(valid));panel["eligible"]=valid&panel.expansion_breadth.ge(P["minimum_expansion_breadth"])&panel.btc_variation_rank.ge(P["btc_variation_rank_min"]);return panel.loc[:,BASE_COLS]
def active(panel:pd.DataFrame,control="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=panel.copy()
 if control=="one_day_stale_breadth":u[[*RANK_COLS,"expansion_breadth","median_alt_range_rank","feature_available_time"]]=panel[[*RANK_COLS,"expansion_breadth","median_alt_range_rank","feature_available_time"]].shift(1)
 valid=u.source_valid.eq(True);breadth=u.expansion_breadth.ge(P["minimum_expansion_breadth"]);variation=u.btc_variation_rank.ge(P["btc_variation_rank_min"]);state=valid&breadth&variation
 if control=="no_btc_variation_gate":state=valid&breadth
 elif control=="three_of_six_breadth":state=valid&u.expansion_breadth.ge(3)&variation
 elif control=="median_alt_range_level":state=valid&u.median_alt_range_rank.ge(P["alt_range_rank_min"])&variation
 side=np.sign(pd.to_numeric(u.btc_return,errors="coerce").fillna(0)).astype(int)
 if control=="direction_flip":side=-side
 if control=="forced_long":side=pd.Series(1,index=u.index,dtype=int)
 return state&side.ne(0),side,u
def build_clock(panel,control="primary"):
 act,side,u=active(panel,control);rows=[]
 for i in panel.index[act]:
  d=pd.Timestamp(panel.at[i,"decision_time"]);entry=d+pd.Timedelta("5m");exit_time=entry+pd.Timedelta("12h");split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_time<=b),None)
  if split is None:continue
  rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":pd.Timestamp(u.at[i,"feature_available_time"]),"entry_time":entry,"exit_time":exit_time,"side":int(side.at[i]),**{c:float(u.at[i,c]) for c in CLOCK_COLS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLS)
def stats(c,split):
 x=c[c.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(m.max())/len(x)}
def csv_gz(x):
 b=io.BytesIO();raw=x.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode()
 with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
 return b.getvalue()
def immutable(p:Path,b:bytes):p.parent.mkdir(parents=True,exist_ok=True);(_ for _ in ()).throw(RuntimeError(f"refusing overwrite {p}")) if p.exists() and p.read_bytes()!=b else p.write_bytes(b)
def jb(x):return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCARER prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvcarer_12_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,jb(manifest));support={n:stats(primary,n) for n in SPLITS};checks={key:ok for n,x in support.items() for key,ok in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvcarer_12_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,jb(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
