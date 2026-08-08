"""Build source-support clocks for BSAR-24 without BTC outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_blockspace_fee_witness_concordance_support as source
from training import preregister_blockspace_fee_witness_concordance as old_prereg
from training import preregister_blockspace_stress_absorption_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
CLOCK=Path("data/blockspace_stress_absorption_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/blockspace_stress_absorption_relay_controls_2023_2026");RESULT=Path("results/blockspace_stress_absorption_relay_support_2026-08-08.json");SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MIN={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_stress_tail","no_stress_alignment","no_absorption_tail","no_witness_alignment","no_rotation_deceleration","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","stress_bucket_start","absorption_bucket_start","source_available_time","entry_time","exit_time","side","stress_R","stress_U","stress_rank","stress_q_rank","absorption_R","absorption_W","absorption_U","absorption_q_rank","signed_rotation_ratio")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 bfrt,wctr=source.load_sources();f,_=source.build_joint_features(bfrt,wctr);return f
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 p=f.shift(1);consecutive=f.bucket_start_utc.diff().eq(pd.Timedelta(hours=12));ready=f.base_valid&p.base_valid&consecutive&f.rank_n.ge(120)&p.rank_n.ge(120)&f.q_rank_n.ge(120)&p.q_rank_n.ge(120);side=np.sign(p.R);stress=p.R.ne(0)&p.U.ne(0)
 if control!="no_stress_tail":stress&=p["rank"].ge(.75)
 if control!="no_stress_alignment":stress&=np.sign(p.R).eq(np.sign(p.U))
 stress&=p.q_rank.ge(.50);absorb=f.W.ne(0)
 if control!="no_absorption_tail":absorb&=f.q_rank.ge(.75)
 if control!="no_witness_alignment":absorb&=np.sign(f.W).eq(side)
 signed=side*f.R;ratio=signed/p.R.abs();decelerate=signed.ge(0)
 if control!="no_rotation_deceleration":decelerate&=ratio.le(.50)
 active=ready&stress&absorb&decelerate;on=active&~active.shift(1,fill_value=False);rows=[];next_allowed=None
 for i in f.index[on]:
  available=pd.Timestamp(f.at[i,"joint_available_at_utc"]);entry=source._entry_time(available);exit_=entry+pd.Timedelta(hours=24)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  s=int(side.at[i]);s=-s if control=="direction_flip" else s;next_allowed=exit_;rows.append({"candidate":"BSAR-24","control":control,"split":split,"stress_bucket_start":f.at[i-1,"bucket_start_utc"],"absorption_bucket_start":f.at[i,"bucket_start_utc"],"source_available_time":available,"entry_time":entry,"exit_time":exit_,"side":s,"stress_R":float(p.at[i,"R"]),"stress_U":float(p.at[i,"U"]),"stress_rank":float(p.at[i,"rank"]),"stress_q_rank":float(p.at[i,"q_rank"]),"absorption_R":float(f.at[i,"R"]),"absorption_W":float(f.at[i,"W"]),"absorption_U":float(f.at[i,"U"]),"absorption_q_rank":float(f.at[i,"q_rank"]),"signed_rotation_ratio":float(ratio.at[i])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MIN[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={"protocol_version":"bsar_24_source_support_v1","policy_id":"BSAR-24","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_artifacts":{"fee_rotation":{"path":str(old_prereg.BFRT_NORMALIZED),"sha256":sha(old_prereg.BFRT_NORMALIZED)},"witness_composition":{"path":str(old_prereg.WCTR_NORMALIZED),"sha256":sha(old_prereg.WCTR_NORMALIZED)}},"completed_delayed_source_features_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
