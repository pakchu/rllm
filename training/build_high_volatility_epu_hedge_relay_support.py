"""Build source-only HVEPUH-24 clocks before Gross9 or economics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_high_volatility_epu_hedge_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market
PREREG_SHA="12292e59ef6643d2835da872ff733526bce2e14661c936dbc38f82226e3b48e6";SOURCE_SHA="63c8f211299d9fda371317eb899180f422272408ee3410b5689809958849b793";SOURCE_MANIFEST=Path("data/us_daily_epu_1985_2026_aug_manifest.json");SOURCE_MANIFEST_SHA="fc0001dd616222aa1031ed2c4eac1a1b6843e7785f8123af21c71f7fea99a1d1";HELPER=Path("training/build_scheduled_trend_concordance_relay_support.py");HELPER_SHA="8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f";END=pd.Timestamp("2026-08-01T00:00:00Z")
STATE=Path("data/high_volatility_epu_hedge_relay_sources_2023_2026/daily_states.csv.gz");CLOCK=Path("data/high_volatility_epu_hedge_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_epu_hedge_relay_controls_2023_2026");RESULT=Path("results/high_volatility_epu_hedge_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_epu_tail","no_btc_variation_gate","one_day_stale_epu","direction_flip","same_clock_forced_long")
COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","epu_change","epu_shock_rank","btc_variation","btc_variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(v:pd.Series)->pd.Series:
 x=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=x.index,dtype=float);h=[]
 for i,c in x.items():
  q=np.asarray(h[-270:],float)
  if np.isfinite(c) and len(q)>=180:o.at[i]=((q<c).sum()+.5*(q==c).sum())/len(q)
  if np.isfinite(c):h.append(float(c))
 return o
def score_states(epu:pd.DataFrame,market:pd.DataFrame)->pd.DataFrame:
 e=epu.copy();e["source_day"]=pd.to_datetime(dict(year=e.year,month=e.month,day=e.day),utc=True);e["epu"]=pd.to_numeric(e.daily_policy_index,errors="coerce");e=e.sort_values("source_day").reset_index(drop=True)
 if e.source_day.duplicated().any():raise RuntimeError("HVEPUH duplicate source day")
 e["epu_valid"]=np.isfinite(e.epu)&e.epu.gt(0)&e.source_day.diff().eq(pd.Timedelta(days=1));e["epu_change"]=np.log(e.epu/e.epu.shift(1)).where(e.epu_valid);e["epu_shock_rank"]=rank(e.epu_change.abs().where(e.epu_valid));e["decision_time"]=e.source_day+pd.Timedelta(days=2)
 m=market.copy();m["date"]=pd.to_datetime(m.date,utc=True);m=m.sort_values("date").set_index("date");close=pd.to_numeric(m.close,errors="coerce");valid=np.isfinite(close)&close.gt(0);step=m.index.to_series().diff().eq(pd.Timedelta(minutes=5));variation=np.sqrt(np.log(close/close.shift(1)).pow(2).rolling(288,min_periods=288).sum());ok=valid.rolling(289,min_periods=289).sum().eq(289)&step.rolling(288,min_periods=288).sum().eq(288);lookup=pd.DataFrame({"decision_time":m.index+pd.Timedelta(minutes=5),"btc_variation":variation.where(ok).to_numpy()});e=e.merge(lookup,on="decision_time",how="left",validate="one_to_one");e["state_valid"]=e.epu_valid&np.isfinite(e[["epu_change","epu_shock_rank","btc_variation"]]).all(axis=1)&e.epu_change.ne(0);e["btc_variation_rank"]=rank(e.btc_variation.where(e.state_valid));return e
def build_clock(states:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=states.shift(1) if control=="one_day_stale_epu" else states;valid=used.state_valid.eq(True)&used.epu_change.ne(0)
 if control=="one_day_stale_epu":valid&=states.source_day.sub(used.source_day).eq(pd.Timedelta(days=1))
 epu_gate=pd.Series(True,index=states.index) if control=="no_epu_tail" else used.epu_shock_rank.ge(.75);btc_gate=pd.Series(True,index=states.index) if control=="no_btc_variation_gate" else states.btc_variation_rank.ge(.65);active=valid&epu_gate&btc_gate;side=np.sign(used.epu_change).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=states.index)
 rows=[]
 for i in states.index[active]:
  decision=pd.Timestamp(states.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=24);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  source=used.loc[i];rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"source_day":source.source_day,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"epu_change":float(source.epu_change),"epu_shock_rank":float(source.epu_shock_rank),"btc_variation":float(states.at[i,"btc_variation"]),"btc_variation_rank":float(states.at[i,"btc_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 bindings={prereg.DEFAULT_OUTPUT:PREREG_SHA,prereg.SOURCE:SOURCE_SHA,SOURCE_MANIFEST:SOURCE_MANIFEST_SHA,HELPER:HELPER_SHA,prereg.MARKET:prereg.MARKET_SHA}
 for p,h in bindings.items():
  if sha(p)!=h:raise RuntimeError(f"HVEPUH binding drift: {p}")
 epu=pd.read_csv(prereg.SOURCE);market,market_source=load_market();states=score_states(epu,market);primary=build_clock(states);controls={n:build_clock(states,n) for n in CONTROLS};STATE.parent.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(states,STATE);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvepuh_24_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"bindings":{str(p):h for p,h in bindings.items()},"market_source":market_source,"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"source_state":{"path":str(STATE),"sha256":sha(STATE),"rows":len(states)},"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
