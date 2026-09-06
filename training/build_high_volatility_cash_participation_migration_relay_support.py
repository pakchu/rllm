"""Build outcome-blind source support for frozen HVCPMR-8."""
from __future__ import annotations
import gzip,hashlib,io,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_cash_participation_migration_relay as prereg
ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="b4c36610d0d15b624fa87c83f95ca0c7e4319c557cdce2e59e9600e588dd6d34";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];CONTROLS=tuple(REG["diagnostic_controls"]["names"])
QUERY="SELECT ts,open,high,low,close,quote_asset_volume,number_of_trades FROM {table} WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
ROOT=Path("data/high_volatility_cash_participation_migration_relay_sources_2023_2026");PANEL=ROOT/"block_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_cash_participation_migration_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_cash_participation_migration_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_cash_participation_migration_relay_controls_2023_2026");RESULT=Path("results/high_volatility_cash_participation_migration_relay_support_2026-08-12.json");BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","spot_minute_count","perpetual_minute_count","first_spot_share","second_spot_share","migration","migration_rank","first_count_share","second_count_share","count_migration","spot_return","perpetual_return","final_two_hour_return","direction_side","realized_variation","variation_rank","eligible","onset");CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side",*PANEL_COLUMNS[5:18])
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def rank(value:float,history:list[float])->float:
 prior=np.asarray(history[-P["history_decisions"]:],float)
 if not math.isfinite(value) or len(prior)<P["minimum_history_decisions"]:return math.nan
 return float((np.sum(prior<value)+.5*np.sum(prior==value))/len(prior))
def onset(state:pd.Series,valid:pd.Series)->pd.Series:
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
 db=engine();params={"start":START.to_pydatetime(),"end":END.to_pydatetime()}
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY.format(table="bars_binance_spot")),c,params=params),pd.read_sql_query(text(QUERY.format(table="bars_binance")),c,params=params)
 finally:db.dispose()
def prepare(x:pd.DataFrame,label:str)->pd.DataFrame:
 expected=["ts","open","high","low","close","quote_asset_volume","number_of_trades"]
 if x.columns.tolist()!=expected:raise RuntimeError(f"HVCPMR {label} schema drift")
 y=x.copy();y["ts"]=pd.to_datetime(y.ts,utc=True,errors="coerce")
 for c in expected[1:]:y[c]=pd.to_numeric(y[c],errors="coerce")
 if y.ts.isna().any() or y.ts.duplicated().any():raise RuntimeError(f"HVCPMR {label} key drift")
 prices=y[["open","high","low","close"]];y["row_valid"]=np.isfinite(y[expected[1:]]).all(axis=1)&prices.gt(0).all(axis=1)&y.high.ge(prices[["open","close"]].max(axis=1))&y.low.le(prices[["open","close"]].min(axis=1))&y.high.ge(y.low)&y.quote_asset_volume.ge(0)&y.number_of_trades.ge(0);return y.set_index("ts").sort_index()
def share(spot:pd.DataFrame,perp:pd.DataFrame,column:str)->float:
 a=float(spot[column].sum());b=float(perp[column].sum());return a/(a+b) if a+b>0 else math.nan
def build_panel(spot_raw:pd.DataFrame,perp_raw:pd.DataFrame)->pd.DataFrame:
 spot=prepare(spot_raw,"spot");perp=prepare(perp_raw,"perpetual");rows=[];migrations=[];variations=[]
 for d in pd.date_range(START+pd.Timedelta("30h"),END,freq="8h",inclusive="left"):
  idx=pd.date_range(d-pd.Timedelta("24h"),d,freq="1min",inclusive="left");s=spot.reindex(idx);p=perp.reindex(idx);sc=int(s.row_valid.eq(True).sum());pc=int(p.row_valid.eq(True).sum());valid=len(s)==1440 and len(p)==1440 and bool(s.row_valid.eq(True).all()) and bool(p.row_valid.eq(True).all());first_share=second_share=migration=migration_rank=first_count=second_count=count_migration=spot_return=perp_return=final_return=direction=variation=variation_rank=math.nan
  if valid:
   sb=s.iloc[-480:];pb=p.iloc[-480:];first_share=share(sb.iloc[:240],pb.iloc[:240],"quote_asset_volume");second_share=share(sb.iloc[240:],pb.iloc[240:],"quote_asset_volume");migration=second_share-first_share;first_count=share(sb.iloc[:240],pb.iloc[:240],"number_of_trades");second_count=share(sb.iloc[240:],pb.iloc[240:],"number_of_trades");count_migration=second_count-first_count;spot_return=float(np.log(sb.close.iloc[-1]/sb.open.iloc[0]));perp_return=float(np.log(pb.close.iloc[-1]/pb.open.iloc[0]));final_return=float(np.log(pb.close.iloc[-1]/pb.open.iloc[-120]));direction=float(np.sign(spot_return)) if spot_return!=0 and np.sign(spot_return)==np.sign(perp_return)==np.sign(final_return) else 0.;variation=float(np.square(np.log(p.close.to_numpy(float)/p.open.to_numpy(float))).sum());migration_rank=rank(migration,migrations) if migration>0 else math.nan;variation_rank=rank(variation,variations);valid=bool(np.isfinite([first_share,second_share,migration,first_count,second_count,count_migration,spot_return,perp_return,final_return,variation]).all() and variation>0)
  rows.append({"decision_time":d,"feature_available_time":d,"source_valid":valid,"spot_minute_count":sc,"perpetual_minute_count":pc,"first_spot_share":first_share,"second_spot_share":second_share,"migration":migration,"migration_rank":migration_rank,"first_count_share":first_count,"second_count_share":second_count,"count_migration":count_migration,"spot_return":spot_return,"perpetual_return":perp_return,"final_two_hour_return":final_return,"direction_side":direction,"realized_variation":variation,"variation_rank":variation_rank})
  if valid:
   variations.append(variation)
   if migration>0:migrations.append(migration)
 panel=pd.DataFrame(rows);valid=panel.source_valid.eq(True);panel["eligible"]=valid&panel.direction_side.ne(0)&panel.migration.gt(0)&panel.migration_rank.ge(P["migration_rank_min"])&panel.variation_rank.ge(P["variation_rank_min"]);panel["onset"]=onset(panel.eligible,valid);return panel.loc[:,PANEL_COLUMNS]
def active(panel:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=panel.copy()
 if control=="one_block_stale_features":used[list(PANEL_COLUMNS[2:18])]=panel[list(PANEL_COLUMNS[2:18])].shift(1);used["feature_available_time"]=panel.feature_available_time.shift(1)
 valid=used.source_valid.eq(True);direction=used.direction_side.ne(0);tail=used.migration.gt(0)&used.migration_rank.ge(P["migration_rank_min"]);variation=used.variation_rank.ge(P["variation_rank_min"]);state=valid&direction&tail&variation;side=pd.to_numeric(used.direction_side,errors="coerce").fillna(0).astype(int)
 if control=="no_migration_tail":state=valid&direction&used.migration.gt(0)&variation
 elif control=="no_variation_gate":state=valid&direction&tail
 elif control=="trade_count_share_migration":state=valid&direction&used.count_migration.gt(0)&variation
 act=onset(state,valid)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=side.where(side.eq(0),1)
 return act&side.ne(0),side,used
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCPMR prereg drift")
 spot,perp=load_sources();panel=build_panel(spot,perp);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};immutable(PANEL,csv_gz(panel));immutable(CLOCK,csv_gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",csv_gz(x))
 for n,x in splits.items():immutable(SPLIT_DIR/f"{n}.csv.gz",csv_gz(x))
 source_core={"protocol_version":"hvcpmr_8_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":{"spot":len(spot),"perpetual":len(perp)},"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={key:passed for n,x in support.items() for key,passed in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvcpmr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in splits.items()},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,json_bytes(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
