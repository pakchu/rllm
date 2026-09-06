"""Build outcome-blind source support for frozen HVDTCAR-8."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_directional_turnover_concentration_asymmetry_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env"; START=pd.Timestamp("2023-04-01T00:00:00Z"); END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="9e312f1c5e332ecc85011daeb028a862cc3be29ada37f7f0cb5d685934a00cbe"; REG=prereg.build(); P=REG["policy"]
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()}; GATES=REG["source_support_gates"]; CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="""SELECT ts,open,high,low,close,quote_asset_volume FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path("data/high_volatility_directional_turnover_concentration_asymmetry_relay_sources_2023_2026"); PANEL=ROOT/"block_states.csv.gz"; MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_directional_turnover_concentration_asymmetry_relay_clocks_2023_2026.csv.gz"); SPLIT_DIR=Path("data/high_volatility_directional_turnover_concentration_asymmetry_relay_split_clocks_2023_2026"); CONTROL_DIR=Path("data/high_volatility_directional_turnover_concentration_asymmetry_relay_controls_2023_2026")
RESULT=Path("results/high_volatility_directional_turnover_concentration_asymmetry_relay_support_2026-08-10.json"); BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","minute_count","up_concentration","down_concentration","directional_concentration_contrast","concentration_rank","directional_turnover_mass_asymmetry","mass_asymmetry_rank","realized_variation","variation_rank","eligible","onset")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","up_concentration","down_concentration","directional_concentration_contrast","concentration_rank","directional_turnover_mass_asymmetry","mass_asymmetry_rank","realized_variation","variation_rank")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors="coerce").to_numpy(float); out=np.full(len(values),np.nan); history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-P["history_blocks"]:],float)
  if math.isfinite(value) and len(prior)>=P["minimum_history_blocks"]: out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value): history.append(float(value))
 return pd.Series(out,index=series.index)

def concentration_statistics(block:pd.DataFrame)->tuple[float,float,float,float,float]:
 groups=np.arange(480)//5
 grouped=block.assign(group=groups).groupby("group",sort=True).agg(open=("open","first"),close=("close","last"),turnover=("quote_asset_volume","sum"))
 returns=np.log(grouped.close.to_numpy(float)/grouped.open.to_numpy(float));turnover=grouped.turnover.to_numpy(float)
 def normalized(mask:np.ndarray)->tuple[float,float]:
  values=turnover[mask];n=len(values);total=float(values.sum())
  if n<2 or not math.isfinite(total) or total<=0:return math.nan,total
  hhi=float(np.square(values/total).sum());return float((n*hhi-1)/(n-1)),total
 up,up_mass=normalized(returns>0);down,down_mass=normalized(returns<0);contrast=up-down
 mass_asymmetry=(up_mass-down_mass)/(up_mass+down_mass)
 minute_returns=np.log(block.close.to_numpy(float)/block.open.to_numpy(float));variation=float(np.sqrt(np.square(minute_returns).sum()))
 return up,down,contrast,mass_asymmetry,variation

def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})

def load_source()->pd.DataFrame:
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c: frame=pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally: db.dispose()
 return frame

def prepare(frame:pd.DataFrame)->pd.DataFrame:
 if frame.columns.tolist()!=["ts","open","high","low","close","quote_asset_volume"]:raise RuntimeError("HVDTCAR source schema drift")
 x=frame.copy();x["ts"]=pd.to_datetime(x.ts,utc=True,errors="coerce")
 for c in ("open","high","low","close","quote_asset_volume"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if x.ts.isna().any() or x.ts.duplicated().any():raise RuntimeError("HVDTCAR invalid source key")
 prices=x[["open","high","low","close"]]
 x["row_valid"]=np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&np.isfinite(x.quote_asset_volume)&x.quote_asset_volume.ge(0)&x.high.ge(prices[["open","close"]].max(axis=1))&x.low.le(prices[["open","close"]].min(axis=1))&x.high.ge(x.low)
 return x.set_index("ts").sort_index()

def previous_valid_onset(eligible:pd.Series,valid:pd.Series)->pd.Series:
 out=pd.Series(False,index=eligible.index);previous=None
 for i in eligible.index:
  if not bool(valid.at[i]):continue
  if bool(eligible.at[i]) and previous is not None:out.at[i]=not bool(eligible.at[previous])
  previous=i
 return out

def build_panel(raw:pd.DataFrame)->pd.DataFrame:
 source=prepare(raw);rows=[]
 for decision in pd.date_range(START+pd.Timedelta("2h"),END,freq="8h",inclusive="left"):
  minutes=pd.date_range(decision-pd.Timedelta("8h"),decision,freq="1min",inclusive="left");block=source.reindex(minutes);count=int(block.row_valid.eq(True).sum())
  path_valid=len(block)==480 and bool(block.row_valid.eq(True).all())
  if path_valid:
   up,down,contrast,mass_asymmetry,var=concentration_statistics(block);valid=all(math.isfinite(v) for v in (up,down,contrast,mass_asymmetry,var)) and var>0 and contrast!=0
  else:up=down=contrast=mass_asymmetry=var=math.nan;valid=False
  rows.append({"decision_time":decision,"feature_available_time":decision,"source_valid":valid,"minute_count":count,"up_concentration":up,"down_concentration":down,"directional_concentration_contrast":contrast,"directional_turnover_mass_asymmetry":mass_asymmetry,"realized_variation":var})
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["concentration_rank"]=prior_rank(panel.directional_concentration_contrast.abs().where(valid));panel["mass_asymmetry_rank"]=prior_rank(panel.directional_turnover_mass_asymmetry.abs().where(valid));panel["variation_rank"]=prior_rank(panel.realized_variation.where(valid));panel["eligible"]=valid&panel.concentration_rank.ge(P["concentration_rank_min"])&panel.variation_rank.ge(P["variation_rank_min"]);panel["onset"]=previous_valid_onset(panel.eligible,valid);return panel.loc[:,PANEL_COLUMNS]

def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_block_stale_concentration":
  current_valid=panel.source_valid.eq(True);cols=["up_concentration","down_concentration","directional_concentration_contrast","concentration_rank","directional_turnover_mass_asymmetry","mass_asymmetry_rank","feature_available_time"];used[cols]=panel[cols].shift(1);used["source_valid"]=current_valid&panel.source_valid.shift(1).fillna(False)
 valid=used.source_valid.eq(True);strong=used.concentration_rank.ge(P["concentration_rank_min"]);variation=used.variation_rank.ge(P["variation_rank_min"]);state=valid&strong&variation
 if control=="no_concentration_tail":state=valid&variation
 elif control=="no_variation_gate":state=valid&strong
 elif control=="directional_turnover_mass_asymmetry":state=valid&used.directional_turnover_mass_asymmetry.ne(0)&used.mass_asymmetry_rank.ge(P["concentration_rank_min"])&variation
 onset=previous_valid_onset(state,valid);side=np.sign(pd.to_numeric(used.directional_concentration_contrast,errors="coerce").fillna(0)).astype(int)
 if control=="directional_turnover_mass_asymmetry":side=np.sign(pd.to_numeric(used.directional_turnover_mass_asymmetry,errors="coerce").fillna(0)).astype(int)
 if control=="direction_flip":side=-side
 if control=="forced_long":side=pd.Series(1,index=side.index,dtype=int)
 return onset&side.ne(0),side,used

def build_clock(panel:pd.DataFrame,control:str="primary")->pd.DataFrame:
 act,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[act]:
  decision=pd.Timestamp(panel.at[i,"decision_time"]);entry=decision+pd.Timedelta("5m");exit_time=entry+pd.Timedelta("8h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_time<=b),None)
  if split is None:continue
  reserved=exit_time;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":pd.Timestamp(used.at[i,"feature_available_time"]),"entry_time":entry,"exit_time":exit_time,"side":int(side.at[i]),**{c:float(used.at[i,c]) for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)

def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());months=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(months.max())/len(x)}

def csv_gz(frame):
 b=io.BytesIO();raw=frame.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode()
 with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
 return b.getvalue()
def immutable(path,content):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.read_bytes()!=content:raise RuntimeError(f"refusing overwrite {path}")
 path.write_bytes(content)
def json_bytes(x):return (json.dumps(x,indent=2,allow_nan=False)+"\n").encode()

def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVDTCAR prereg drift")
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS}
 immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvdtcar_8_source_v1","query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":len(raw),"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest))
 support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());registration=json.loads(prereg.DEFAULT_OUTPUT.read_text())
 core={"protocol_version":"hvdtcar_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
