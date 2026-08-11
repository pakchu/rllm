"""Build outcome-blind source support for frozen CFTCMSPS-168."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_cftc_micro_standard_positioning_segmentation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_cftc_asset_manager_option_position_second_week_relay_support import daily_variation_states
from training.build_scheduled_trend_concordance_relay_support import load_market

PREREG_SHA="8a433945bb33e05d282e32778192f75cd5b80886e4fde4c4381bfa1938bae30c"
HELPER=Path("training/build_cftc_asset_manager_option_position_second_week_relay_support.py");HELPER_SHA="c86702649584eed407e79e6d34d8be8e4504813094a5d29e66a5681dbf31a19e"
RAW=Path("data/cftc_camop2w_raw");YEARS=tuple(range(2021,2027));END=pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR=Path("data/cftc_micro_standard_positioning_segmentation_relay_sources_2021_2026");STATES=SOURCE_DIR/"weekly_preentry_states.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json"
CLOCK=Path("data/cftc_micro_standard_positioning_segmentation_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/cftc_micro_standard_positioning_segmentation_relay_controls_2023_2026");RESULT=Path("results/cftc_micro_standard_positioning_segmentation_relay_support_2026-08-11.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8}
CONTRACTS={"standard":("133741","BITCOIN - CHICAGO MERCANTILE EXCHANGE"),"micro":("133742","MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE")}
CONTROLS=("no_variation_gate","standard_asset_manager_only","micro_leveraged_only","one_report_stale_segmentation","direction_flip","forced_long")
COLUMNS=("candidate","control","split","report_date","decision_time","feature_available_time","entry_time","exit_time","side","standard_asset_manager_share","micro_leveraged_share","standard_change","micro_change","segmentation","btc_variation","btc_variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _rows(year:int,code:str,name:str)->list[dict[str,str]]:
 p=RAW/f"fut_fin_txt_{year}.zip"
 with zipfile.ZipFile(p) as z:
  with z.open(z.namelist()[0]) as raw:
   return [r for r in csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8-sig",newline="")) if r["CFTC_Contract_Market_Code"].strip()==code and r["Market_and_Exchange_Names"].strip()==name]
def load_cftc()->tuple[pd.DataFrame,dict[str,Any]]:
 frames=[];bindings={}
 for role,(code,name) in CONTRACTS.items():
  rows=[]
  for year in YEARS:
   p=RAW/f"fut_fin_txt_{year}.zip";bindings[str(p)]=sha(p);rows+=_rows(year,code,name)
  f=pd.DataFrame(rows);f["report_date"]=pd.to_datetime(f.Report_Date_as_YYYY_MM_DD if "Report_Date_as_YYYY_MM_DD" in f else f["Report_Date_as_YYYY-MM-DD"],utc=True)
  oi=pd.to_numeric(f.Open_Interest_All,errors="coerce")
  if role=="standard":net=pd.to_numeric(f.Asset_Mgr_Positions_Long_All,errors="coerce")-pd.to_numeric(f.Asset_Mgr_Positions_Short_All,errors="coerce")
  else:net=pd.to_numeric(f.Lev_Money_Positions_Long_All,errors="coerce")-pd.to_numeric(f.Lev_Money_Positions_Short_All,errors="coerce")
  f[f"{role}_share"]=(net/oi).where(oi.gt(0));f=f[["report_date",f"{role}_share"]]
  if f.report_date.duplicated().any():raise RuntimeError(f"duplicate CFTC {role} date")
  frames.append(f)
 out=frames[0].merge(frames[1],on="report_date",how="inner",validate="one_to_one").sort_values("report_date").reset_index(drop=True)
 return out,{"archive_sha256":bindings,"standard_rows":len(frames[0]),"micro_rows":len(frames[1]),"common_report_dates":len(out),"contracts":{k:{"code":v[0],"name":v[1]} for k,v in CONTRACTS.items()}}
def score_states(cftc:pd.DataFrame,variation:pd.DataFrame)->pd.DataFrame:
 s=cftc.copy();consecutive=s.report_date.diff().eq(pd.Timedelta(days=7));s["standard_change"]=(s.standard_share-s.standard_share.shift(1)).where(consecutive);s["micro_change"]=(s.micro_share-s.micro_share.shift(1)).where(consecutive);s["segmentation"]=s.standard_change-s.micro_change;s["stale_segmentation"]=(s.standard_change.shift(1)-s.micro_change.shift(1)).where(consecutive&consecutive.shift(1,fill_value=False));s["decision_time"]=s.report_date+pd.Timedelta(days=7);s["feature_available_time"]=s.decision_time+pd.Timedelta(minutes=5);return s.merge(variation,on="feature_available_time",how="left",validate="many_to_one")
def build_clock(s:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 if control=="standard_asset_manager_only":signal=s.standard_change
 elif control=="micro_leveraged_only":signal=-s.micro_change
 elif control=="one_report_stale_segmentation":signal=s.stale_segmentation
 else:signal=s.segmentation
 active=np.isfinite(signal)&signal.ne(0)&np.isfinite(s.btc_variation)
 if control!="no_variation_gate":active&=s.btc_variation_rank.ge(.5)
 side=np.sign(signal).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="forced_long":side=pd.Series(1,index=s.index)
 rows=[]
 for i in s.index[active]:
  x=s.loc[i];entry=pd.Timestamp(x.feature_available_time);exit_=entry+pd.Timedelta(hours=168);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"report_date":x.report_date,"decision_time":x.decision_time,"feature_available_time":x.feature_available_time,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"standard_asset_manager_share":float(x.standard_share),"micro_leveraged_share":float(x.micro_share),"standard_change":float(x.standard_change),"micro_change":float(x.micro_change),"segmentation":float(signal.at[i]),"btc_variation":float(x.btc_variation),"btc_variation_rank":float(x.btc_variation_rank)})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,split:str)->dict[str,Any]:
 r=c[c.split.eq(split)]
 if r.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(r.side.eq(1).sum());q=int(r.side.eq(-1).sum());m=pd.to_datetime(r.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(r),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(r),"max_month_share":int(m.max())/len(r)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA or sha(HELPER)!=HELPER_SHA:raise RuntimeError("CFTCMSPS frozen binding drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);cftc,csource=load_cftc();market,msource=load_market();states=score_states(cftc,daily_variation_states(market));primary=build_clock(states);controls={n:build_clock(states,n) for n in CONTROLS}
 SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(states,STATES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"cftcmsps_168_sources_v1","cftc":csource,"market_source":msource,"helper":{"path":str(HELPER),"sha256":HELPER_SHA},"states":{"path":str(STATES),"sha256":sha(STATES),"rows":len(states)},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**sc,"manifest_hash":prereg.canonical_hash(sc)};MANIFEST.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values())
 core={"protocol_version":"cftcmsps_168_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
