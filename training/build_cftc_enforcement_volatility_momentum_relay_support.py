"""Build source-only CEVMR-24 clocks from the complete CFTC action archive."""
from __future__ import annotations
import argparse,gzip,hashlib,html,json,math,re,time,urllib.parse,urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_cftc_enforcement_volatility_momentum_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE="/home/pakchu/rllm/.env";BASE="https://www.cftc.gov/LawRegulation/EnforcementActions/index.htm";WORKERS=4;BTC_START=pd.Timestamp("2019-12-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR=Path("data/cftc_enforcement_volatility_momentum_relay_sources_1995_2026");CACHE=SOURCE_DIR/"raw_page_cache";RAW=SOURCE_DIR/"cftc_enforcement_archive_raw_pages.jsonl.gz";DAYS=SOURCE_DIR/"action_day_features.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/cftc_enforcement_volatility_momentum_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/cftc_enforcement_volatility_momentum_relay_controls_2023_2026");RESULT=Path("results/cftc_enforcement_volatility_momentum_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","one_action_stale_direction","direction_flip","weekday_same_time_without_action")
QUERY="SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def norm(x:Any)->str:return re.sub(r"\s+"," ",html.unescape(str(x or ""))).strip()
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.row=None;self.anchor=None;self.rows=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=="tr":self.row={"datetime":None,"links":[]}
  elif self.row is not None and tag=="time":self.row["datetime"]=a.get("datetime")
  elif self.row is not None and tag=="a":self.anchor={"href":a.get("href",""),"text":[]}
 def handle_data(self,data):
  if self.anchor is not None:self.anchor["text"].append(data)
 def handle_endtag(self,tag):
  if tag=="a" and self.anchor is not None and self.row is not None:self.anchor["text"]=norm(" ".join(self.anchor["text"]));self.row["links"].append(self.anchor);self.anchor=None
  elif tag=="tr" and self.row is not None:
   if self.row["datetime"] and self.row["links"]:
    first=self.row["links"][0];self.rows.append({"timestamp":self.row["datetime"],"title":first["text"],"action_url":urllib.parse.urljoin(BASE,first["href"]),"document_urls":sorted({urllib.parse.urljoin(BASE,x["href"]) for x in self.row["links"][1:] if x["href"]})})
   self.row=None
def parse(raw:bytes)->list[dict[str,Any]]:p=Parser();p.feed(raw.decode("utf-8"));return p.rows
def url(page:int)->str:return BASE+"?"+urllib.parse.urlencode({"page":page})
def fetch(page:int)->bytes:
 CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/f"page-{page:03d}.html"
 if path.exists():return path.read_bytes()
 req=urllib.request.Request(url(page),headers={"User-Agent":"rllm-research/1.0"});last=None
 for i in range(5):
  try:raw=urllib.request.urlopen(req,timeout=60).read();path.write_bytes(raw);time.sleep(.25);return raw
  except Exception as e:last=e;time.sleep(2**i)
 raise RuntimeError(f"CFTC archive page {page} failed") from last
def download()->tuple[list[dict[str,Any]],int]:
 first=fetch(0);m=re.search(rb'href="\?page=(\d+)" title="Go to last page"',first)
 if not m:raise RuntimeError("CFTC archive last page missing")
 pages=int(m.group(1))+1;raws=[first]
 for start in range(1,pages,WORKERS):
  nums=tuple(range(start,min(start+WORKERS,pages)))
  with ThreadPoolExecutor(max_workers=WORKERS) as ex:raws.extend(ex.map(fetch,nums))
 records=[]
 for raw in raws:records.extend(parse(raw))
 identities=[x["action_url"] for x in records]
 if len(identities)!=len(set(identities)):raise RuntimeError("CFTC duplicate action identity")
 SOURCE_DIR.mkdir(parents=True,exist_ok=True)
 with RAW.open("wb") as f:
  with gzip.GzipFile(filename="",mode="wb",fileobj=f,mtime=0,compresslevel=9) as z:
   for raw in raws:z.write(raw+b"\n")
 return records,pages
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def rank(values:pd.Series)->pd.Series:
 out=pd.Series(np.nan,index=values.index,dtype=float);hist=[]
 for i,v in pd.to_numeric(values,errors="coerce").items():
  prior=hist[-252:]
  if math.isfinite(v) and len(prior)>=126:a=np.asarray(prior);out.at[i]=(np.sum(a<v)+.5*np.sum(a==v))/len(a)
  if math.isfinite(v):hist.append(float(v))
 return out
def features_for_dates(dates,bars,rank_values=True):
 f=bars;rows=[]
 for date in sorted(set(dates)):
  decision=pd.Timestamp(date).floor("D")+pd.Timedelta(hours=22);expected=pd.date_range(decision-pd.Timedelta(hours=24),decision,freq="1min",inclusive="left");w=f.reindex(expected);valid=len(w)==1440 and np.isfinite(w[["open","high","low","close"]]).all().all() and w[["open","high","low","close"]].gt(0).all().all();direction=float(np.log(w.close.iloc[-1]/w.open.iloc[0])) if valid else np.nan;variation=float(np.log(w.close).diff().dropna().pow(2).sum()) if valid else np.nan;rows.append({"date":pd.Timestamp(date).floor("D"),"decision_time":decision,"direction":direction,"variation_24h":variation})
 out=pd.DataFrame(rows);out["variation_rank"]=rank(out.variation_24h) if rank_values else np.nan;return out
def action_days(records,bars):
 rows=[]
 for x in records:
  date=pd.Timestamp(x["timestamp"]).floor("D")
  if date>=END:continue
  rows.append({"date":date,"url":x["action_url"]})
 grouped=[]
 for date,g in pd.DataFrame(rows).groupby("date",sort=True):
  urls=sorted(g.url);grouped.append({"date":date,"action_count":len(urls),"action_hash":hashlib.sha256("\n".join(urls).encode()).hexdigest()})
 return pd.DataFrame(grouped).merge(features_for_dates([x["date"] for x in grouped],bars),on="date",validate="one_to_one")
def placebo_days(actions,bars):
 occupied=set(actions.date);dates=[]
 for date in actions.date:
  candidate=date+pd.Timedelta(days=1)
  while candidate.weekday()>=5 or candidate in occupied:candidate+=pd.Timedelta(days=1)
  dates.append(candidate)
 f=features_for_dates(dates,bars);f["action_count"]=0;f["action_hash"]="0"*64;return f
def clock(frame,control="primary"):
 f=frame.sort_values("decision_time").reset_index(drop=True);direction=f.direction.copy()
 if control=="one_action_stale_direction":direction=direction.shift(1)
 side=pd.Series(np.where(direction>0,1,-1),index=f.index);side[~np.isfinite(direction)|direction.eq(0)]=np.nan
 if control=="direction_flip":side=-side
 active=np.isfinite(side)&np.isfinite(f.variation_rank)&(f.variation_rank.ge(.65) if control!="no_volatility_gate" else True);rows=[];reserved=None
 for i in f.index[active]:
  decision=f.at[i,"decision_time"];entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=24)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":"CEVMR-24","control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"pre_return":float(f.at[i,"direction"]),"variation_24h":float(f.at[i,"variation_24h"]),"variation_rank":float(f.at[i,"variation_rank"]),"action_count":int(f.at[i,"action_count"]),"action_hash":f.at[i,"action_hash"]})
 return pd.DataFrame(rows,columns=["candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","pre_return","variation_24h","variation_rank","action_count","action_hash"])
def stats(c,split):
 s=c[c.split.eq(split)]
 if s.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(s.side.eq(1).sum());q=int(s.side.eq(-1).sum());m=pd.to_datetime(s.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(s),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(s),"max_month_share":int(m.max())/len(s)}
def run():
 from sqlalchemy import text
 records,pages=download();engine=postgres_engine()
 with engine.connect() as conn:bars=pd.read_sql_query(text(QUERY),conn,params={"start":BTC_START.to_pydatetime(),"end":END.to_pydatetime()})
 engine.dispose();bars.ts=pd.to_datetime(bars.ts,utc=True)
 for c in ("open","high","low","close"):bars[c]=pd.to_numeric(bars[c],errors="coerce")
 bars=bars.drop_duplicates("ts",keep=False).set_index("ts").sort_index();actions=action_days(records,bars);placebo=placebo_days(actions,bars);SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(actions,DAYS);primary=clock(actions);controls={n:clock(placebo if n=="weekday_same_time_without_action" else actions,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,c in controls.items():_write_gzip_csv(c,CONTROL_DIR/f"{n}.csv.gz")
 source_core={"protocol_version":"cevmr_24_source_v1","pages":pages,"records":len(records),"raw":{"path":str(RAW),"sha256":sha(RAW)},"days":{"path":str(DAYS),"sha256":sha(DAYS),"rows":len(actions)},"outcomes_opened":False,"gross9_rows_opened":False};source={**source_core,"manifest_hash":canonical_hash(source_core)};MANIFEST.write_text(json.dumps(source,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"cevmr_24_source_support_v1","policy_id":"CEVMR-24","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":source["manifest_hash"]},"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f'{n}.csv.gz'),"sha256":sha(CONTROL_DIR/f'{n}.csv.gz'),"rows":len(c),"promotion_authorized":False} for n,c in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
