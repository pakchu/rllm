"""Materialize source-only CQSTL-8 support clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_cross_quote_signed_ticket_leadership as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA="691ffb9f1809d385029fe69c1a28c8d19d0420ba137909e3267d9d2ca01bc504"
PANEL=Path("data/binance_stablecoin_quote_flow_btc_2023_2026_aug/BTC_stablecoin_quote_flow_1h_2023-07-01_2026-07-31T23.csv.gz");PANEL_SHA="44374b9a2298ae4b64f0c1e7208665b1c08c8221045308694311123deae1c805";PANEL_MANIFEST=PANEL.parent/"build_manifest.json";PANEL_MANIFEST_SHA="b9c64c3ce651934d9761a6d0731e814b2a92f5237b3040e11f794d7eb024a898"
PRICE=Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz");PRICE_SHA="f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496";PRICE_MANIFEST=PRICE.parent/"manifest.json";PRICE_MANIFEST_SHA="3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"
FEATURE_DIR=Path("data/cross_quote_signed_ticket_leadership_sources_2023_2026");FEATURES=FEATURE_DIR/"preentry_features.csv.gz";SOURCE_MANIFEST=FEATURE_DIR/"manifest.json";CLOCK=Path("data/cross_quote_signed_ticket_leadership_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/cross_quote_signed_ticket_leadership_controls_2023_2026");RESULT=Path("results/cross_quote_signed_ticket_leadership_support_2026-08-09.json")
SYMBOLS=("BTCUSDT","BTCUSDC","BTCFDUSD");SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","no_sponsor_tail","no_usdt_subordination","one_block_stale_alternative_ticket","direction_flip")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","ticket_usdt","ticket_usdc","ticket_fdusd","alternative_sponsor_magnitude","sponsor_rank","btc_realized_variation","btc_realized_variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def strict_prior_midrank(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 n=pd.to_numeric(v,errors="coerce").astype(float);out=pd.Series(np.nan,index=n.index,dtype=float);history=[]
 for i,current in n.items():
  prior=history[-lookback:]
  if math.isfinite(current) and len(prior)>=minimum:
   a=np.asarray(prior);out.at[i]=(np.sum(a<current)+.5*np.sum(a==current))/len(a)
  if math.isfinite(current):history.append(current)
 return out
def build_features()->pd.DataFrame:
 if sha(PANEL)!=PANEL_SHA or sha(PANEL_MANIFEST)!=PANEL_MANIFEST_SHA or sha(PRICE)!=PRICE_SHA or sha(PRICE_MANIFEST)!=PRICE_MANIFEST_SHA:raise RuntimeError("CQSTL source drift")
 x=pd.read_csv(PANEL,compression="gzip");x["date"]=pd.to_datetime(x.date,utc=True,errors="raise");x=x[x.symbol.isin(SYMBOLS)].copy()
 if x[["date","symbol"]].duplicated().any():raise RuntimeError("CQSTL duplicate quote hour")
 x["trade_count"]=pd.to_numeric(x.trade_count,errors="coerce");x["signed_taker_flow_btc"]=pd.to_numeric(x.signed_taker_flow_btc,errors="coerce");x["decision_time"]=x.date.dt.floor("8h")+pd.Timedelta(hours=8)
 g=x.groupby(["decision_time","symbol"],as_index=False).agg(hours=("date","size"),first=("date","min"),last=("date","max"),complete=("source_complete","all"),trade_count=("trade_count","sum"),signed_flow=("signed_taker_flow_btc","sum"))
 g["valid"]=g.hours.eq(8)&g["first"].eq(g.decision_time-pd.Timedelta(hours=8))&g["last"].eq(g.decision_time-pd.Timedelta(hours=1))&g.complete&np.isfinite(g[["trade_count","signed_flow"]]).all(axis=1)&g.trade_count.ge(80)
 g["ticket"]=g.signed_flow/g.trade_count;w=g.pivot(index="decision_time",columns="symbol",values=["valid","trade_count","ticket"]);w.columns=[f"{a}_{b}" for a,b in w.columns];w=w.reset_index().sort_values("decision_time").reset_index(drop=True)
 for s in SYMBOLS:
  for a in ("valid","trade_count","ticket"):
   c=f"{a}_{s}"
   if c not in w:w[c]=np.nan
 w["block_valid"]=w[[f"valid_{s}" for s in SYMBOLS]].eq(True).all(axis=1)&np.isfinite(w[[f"ticket_{s}" for s in SYMBOLS]]).all(axis=1)
 w["alternative_sponsor_magnitude"]=(w.ticket_BTCUSDC.abs()+w.ticket_BTCFDUSD.abs())/2;w["sponsor_rank"]=strict_prior_midrank(w.alternative_sponsor_magnitude.where(w.block_valid))
 p=pd.read_csv(PRICE,compression="gzip");p["decision_time"]=pd.to_datetime(p.decision_time,utc=True,format="mixed");p["open"]=pd.to_numeric(p.open,errors="coerce");p["close"]=pd.to_numeric(p.close,errors="coerce");p["valid"]=p.source_valid.astype(str).str.lower().eq("true")&np.isfinite(p[["open","close"]]).all(axis=1)&p[["open","close"]].gt(0).all(axis=1);p=p.sort_values("decision_time").reset_index(drop=True);p["hour_return"]=np.log(p.close/p.open);consecutive=p.decision_time.diff().eq(pd.Timedelta(hours=1));p["btc_realized_variation"]=np.sqrt(p.hour_return.pow(2).rolling(24,min_periods=24).sum());p["btc_valid"]=p.valid.rolling(24,min_periods=24).sum().eq(24)&consecutive.rolling(23,min_periods=23).sum().eq(23)&np.isfinite(p.btc_realized_variation);p=p[p.decision_time.dt.hour.isin([0,8,16])][["decision_time","btc_realized_variation","btc_valid"]]
 w=w.merge(p,on="decision_time",how="left",validate="one_to_one");w["btc_realized_variation_rank"]=strict_prior_midrank(w.btc_realized_variation.where(w.btc_valid));w["signal_valid"]=w.block_valid&w.btc_valid.fillna(False)&np.isfinite(w[["sponsor_rank","btc_realized_variation","btc_realized_variation_rank"]]).all(axis=1);return w
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 usdc=f.ticket_BTCUSDC;fdusd=f.ticket_BTCFDUSD;sponsor=f.alternative_sponsor_magnitude;rank=f.sponsor_rank
 if control=="one_block_stale_alternative_ticket":usdc=usdc.shift(1);fdusd=fdusd.shift(1);sponsor=sponsor.shift(1);rank=rank.shift(1)
 consensus=usdc.ne(0)&fdusd.ne(0)&np.sign(usdc).eq(np.sign(fdusd));tail=pd.Series(True,index=f.index) if control=="no_sponsor_tail" else rank.ge(.65);subordinate=pd.Series(True,index=f.index) if control=="no_usdt_subordination" else f.ticket_BTCUSDT.abs().lt(sponsor);volatile=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.btc_realized_variation_rank.ge(.65);active=f.signal_valid&np.isfinite(usdc)&np.isfinite(fdusd)&np.isfinite(sponsor)&consensus&tail&subordinate&volatile;side=np.sign(usdc);side=-side if control=="direction_flip" else side;return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  next_allowed=exit_;rows.append({"candidate":"CQSTL-8","control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"ticket_usdt":float(f.at[i,"ticket_BTCUSDT"]),"ticket_usdc":float(f.at[i,"ticket_BTCUSDC"]),"ticket_fdusd":float(f.at[i,"ticket_BTCFDUSD"]),"alternative_sponsor_magnitude":float(f.at[i,"alternative_sponsor_magnitude"]),"sponsor_rank":float(f.at[i,"sponsor_rank"]),"btc_realized_variation":float(f.at[i,"btc_realized_variation"]),"btc_realized_variation_rank":float(f.at[i,"btc_realized_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("CQSTL preregistration hash drift")
 f=build_features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};FEATURE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 source_core={"protocol_version":"cqstl_8_preentry_sources_v1","source_panel":{"path":str(PANEL),"sha256":PANEL_SHA,"manifest_path":str(PANEL_MANIFEST),"manifest_sha256":PANEL_MANIFEST_SHA},"completed_btc":{"path":str(PRICE),"sha256":PRICE_SHA,"manifest_path":str(PRICE_MANIFEST),"manifest_sha256":PRICE_MANIFEST_SHA},"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(f)},"candidate_incidence_opened":False,"postentry_outcomes_opened":False,"no_imputation":True};source={**source_core,"manifest_hash":chash(source_core)};SOURCE_MANIFEST.write_text(json.dumps(source,indent=2,allow_nan=False)+"\n")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={"protocol_version":"cqstl_8_source_support_v1","policy_id":"CQSTL-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST),"manifest_hash":source["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
