"""Build source-support clocks for CCXTR-6 without BTC outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
from training import preregister_cboe_convexity_crypto_volatility_transmission_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

CLOCK=Path("data/cboe_convexity_crypto_volatility_transmission_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/cboe_convexity_crypto_volatility_transmission_relay_controls_2023_2026");RESULT=Path("results/cboe_convexity_crypto_volatility_transmission_relay_support_2026-08-08.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MIN={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_crypto_confirmation","bvol_only_confirmation","dvol_only_confirmation","one_session_stale_convexity_change","direction_flip");NY=ZoneInfo("America/New_York");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","cboe_observation_date","next_cboe_source_date","decision_time","feature_available_time","entry_time","exit_time","side","delta_convexity","previous_delta_convexity","bvol_body","dvol_body")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 for raw,want in prereg.HASHES.items():
  if sha(Path(raw))!=want:raise RuntimeError(f"CCXTR source drift: {raw}")
 c=pd.read_csv(prereg.CBOE,compression="gzip",usecols=["observation_date","VVIX_close","VIX_close"]);c.observation_date=pd.to_datetime(c.observation_date,format="%Y-%m-%d");c.VVIX_close=pd.to_numeric(c.VVIX_close,errors="coerce");c.VIX_close=pd.to_numeric(c.VIX_close,errors="coerce")
 if c.observation_date.duplicated().any() or not c.observation_date.is_monotonic_increasing or not np.isfinite(c[["VVIX_close","VIX_close"]]).all(axis=1).all() or not c[["VVIX_close","VIX_close"]].gt(0).all(axis=1).all():raise RuntimeError("CCXTR Cboe source invalid")
 c["delta_convexity"]=np.log(c.VVIX_close/c.VIX_close).diff();c["previous_delta_convexity"]=c.delta_convexity.shift(1)
 b=pd.read_csv(prereg.BVOL,compression="gzip");b["decision_time"]=pd.to_datetime(b.feature_available_time_utc,utc=True);b["bvol_body"]=(pd.to_numeric(b.close,errors="coerce")-pd.to_numeric(b.open,errors="coerce"))/pd.to_numeric(b.open,errors="coerce");b["bvol_valid"]=b.feature_valid.astype(str).str.lower().eq("true")&np.isfinite(b.bvol_body)
 d=pd.read_csv(prereg.DVOL,compression="gzip");d["decision_time"]=pd.to_datetime(d.close_time,utc=True);d["dvol_body"]=(pd.to_numeric(d.close,errors="coerce")-pd.to_numeric(d.open,errors="coerce"))/pd.to_numeric(d.open,errors="coerce");d["dvol_valid"]=np.isfinite(d.dvol_body)&pd.to_numeric(d.open,errors="coerce").gt(0)&pd.to_numeric(d.close,errors="coerce").gt(0)
 v=b[["decision_time","bvol_body","bvol_valid"]].merge(d[["decision_time","dvol_body","dvol_valid"]],on="decision_time",validate="one_to_one").set_index("decision_time");rows=[]
 for i in range(1,len(c)-1):
  nd=c.at[i+1,"observation_date"];hour=pd.Timestamp(nd.date()).tz_localize(NY)+pd.Timedelta(hours=9);decision=hour.tz_convert("UTC")+pd.Timedelta(hours=1)
  if decision not in v.index:continue
  x=v.loc[decision];rows.append({"observation_date":c.at[i,"observation_date"].date(),"next_source_date":nd.date(),"decision_time":decision,"delta_convexity":float(c.at[i,"delta_convexity"]),"previous_delta_convexity":float(c.at[i,"previous_delta_convexity"]),"bvol_body":float(x.bvol_body),"dvol_body":float(x.dvol_body),"valid":bool(x.bvol_valid and x.dvol_valid)})
 return pd.DataFrame(rows)
def sides(f:pd.DataFrame,control:str)->pd.Series:
 delta=f.previous_delta_convexity if control=="one_session_stale_convexity_change" else f.delta_convexity;valid=f.valid&delta.notna()&delta.ne(0)&f.bvol_body.ne(0)&f.dvol_body.ne(0);same_b=np.sign(f.bvol_body).eq(np.sign(delta));same_d=np.sign(f.dvol_body).eq(np.sign(delta))
 if control=="no_crypto_confirmation":eligible=valid
 elif control=="bvol_only_confirmation":eligible=valid&same_b
 elif control=="dvol_only_confirmation":eligible=valid&same_d
 else:eligible=valid&same_b&same_d
 s=(-np.sign(delta)).astype("Int64").fillna(0).astype(int).where(eligible,0);return -s if control=="direction_flip" else s
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 s=sides(f,control);rows=[];next_allowed=None
 for i in f.index[s.ne(0)]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  next_allowed=exit_;rows.append({"candidate":"CCXTR-6","control":control,"split":split,"cboe_observation_date":f.at[i,"observation_date"].isoformat(),"next_cboe_source_date":f.at[i,"next_source_date"].isoformat(),"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(s.at[i]),"delta_convexity":float(f.at[i,"delta_convexity"]),"previous_delta_convexity":float(f.at[i,"previous_delta_convexity"]),"bvol_body":float(f.at[i,"bvol_body"]),"dvol_body":float(f.at[i,"dvol_body"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MIN[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"ccxtr_6_source_support_v1","policy_id":"CCXTR-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_bindings":prereg.HASHES,"completed_preentry_sources_opened":True,"btc_price_postentry_return_pnl_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
